import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from training.continuous_runtime import evaluate_clock, evaluate_path, expand_like
from training.solver_aware.clock import monotone_inverse_lookup
from training.solver_aware.monitors import (
    FD_DELTA_FLOOR,
    _cycle_loader,
    _fd_step,
    _jvp,
    _make_generator,
    _material_derivative_fd,
    _material_derivative_jvp,
    _prepare_reference_batch,
    _resolve_monitor_microbatch_size,
    _velocity_fn,
    _velocity_fn_fd,
)


logger = logging.getLogger(__name__)
PROFILE_FILENAME = "tack_profile.pt"
PROFILE_CACHE_FILENAME = "tack_profile_cache.pt"
PROFILE_JSON_FILENAME = "tack_profile.json"
PROFILE_CSV_FILENAME = "tack_profile.csv"
PROFILE_PLOT_FILENAME = "tack_profile_debug.png"
ONLINE_SUMMARY_FILENAME = "tack_online_summary.json"
ONLINE_STEPS_FILENAME = "tack_online_steps.csv"
ONLINE_PLOT_FILENAME = "tack_online_debug.png"
DEFAULT_PLOT_BINS = 12
EPS = 1e-12


def _round_optional(value: Optional[float], digits: int = 12) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _bool_histogram_key(value: float) -> str:
    rounded = int(round(float(value)))
    return str(rounded)


def _batch_l2_norm_mean(values: Tensor) -> Tensor:
    return values.flatten(start_dim=1).norm(dim=1).mean()


def _batch_ratio_mean(numerator: Tensor, denominator: Tensor, eps: float) -> float:
    numerator_norm = numerator.flatten(start_dim=1).norm(dim=1)
    denominator_norm = denominator.flatten(start_dim=1).norm(dim=1)
    ratio = numerator_norm / denominator_norm.clamp(min=float(eps))
    return float(ratio.mean().item())


def _moving_average(values: Tensor, window: int) -> Tensor:
    if window <= 1 or values.numel() <= 2:
        return values
    window = min(int(window), int(values.numel()))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = torch.nn.functional.pad(
        values.view(1, 1, -1),
        (radius, radius),
        mode="replicate",
    )
    kernel = torch.ones(1, 1, window, device=values.device, dtype=values.dtype) / float(window)
    smoothed = torch.nn.functional.conv1d(padded, kernel)
    return smoothed.view(-1)


def _compute_chi(g_history: list[Tensor], eps: float) -> Optional[float]:
    if len(g_history) < 3:
        return None
    numerator = _batch_l2_norm_mean(g_history[-1] - 2.0 * g_history[-2] + g_history[-3])
    denominator = _batch_l2_norm_mean(g_history[-1] - g_history[-2])
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= float(eps):
        return None
    chi_value = float((numerator / denominator).item())
    if not math.isfinite(chi_value):
        return None
    return chi_value


def _ab2_predict_uniform(
    z_n: Tensor,
    g_n: Tensor,
    g_nm1: Tensor,
    h: float,
) -> Tensor:
    return z_n + float(h) * (1.5 * g_n - 0.5 * g_nm1)


def _ab3_predict_uniform(
    z_n: Tensor,
    g_n: Tensor,
    g_nm1: Tensor,
    g_nm2: Tensor,
    h: float,
) -> Tensor:
    return z_n + float(h) * (
        (23.0 / 12.0) * g_n
        - (16.0 / 12.0) * g_nm1
        + (5.0 / 12.0) * g_nm2
    )


def _ab2_predict_nonuniform(
    z_n: Tensor,
    g_n: Tensor,
    g_nm1: Tensor,
    h: float,
    beta1: float,
    eps: float,
) -> Optional[Tensor]:
    if (
        not math.isfinite(float(h))
        or not math.isfinite(float(beta1))
        or float(h) <= 0.0
        or float(beta1) <= float(eps)
    ):
        return None
    h_over_two_beta1 = float(h) / (2.0 * float(beta1))
    return z_n + float(h) * ((1.0 + h_over_two_beta1) * g_n - h_over_two_beta1 * g_nm1)


def _ab3_predict_nonuniform(
    z_n: Tensor,
    g_n: Tensor,
    g_nm1: Tensor,
    g_nm2: Tensor,
    h: float,
    beta1: float,
    beta0: float,
    eps: float,
) -> Optional[Tensor]:
    if (
        not math.isfinite(float(h))
        or not math.isfinite(float(beta1))
        or not math.isfinite(float(beta0))
        or float(h) <= 0.0
        or float(beta1) <= float(eps)
        or float(beta0) <= float(eps)
        or float(beta0 + beta1) <= float(eps)
    ):
        return None
    h_sq = float(h) * float(h)
    beta_sum = float(beta0 + beta1)
    coeff_nm2 = h_sq * (2.0 * float(h) + 3.0 * float(beta1)) / (6.0 * float(beta0) * beta_sum)
    coeff_nm1 = -h_sq * (2.0 * float(h) + 3.0 * beta_sum) / (6.0 * float(beta0) * float(beta1))
    coeff_n = float(h) * (
        h_sq / 3.0
        + (float(beta0) + 2.0 * float(beta1)) * float(h) / 2.0
        + float(beta1) * beta_sum
    ) / (beta_sum * float(beta1))
    return z_n + coeff_nm2 * g_nm2 + coeff_nm1 * g_nm1 + coeff_n * g_n


def _strictly_monotone(values: Tensor) -> Tensor:
    monotone = values.clone()
    min_step = max(torch.finfo(monotone.dtype).eps, EPS)
    for index in range(1, monotone.numel()):
        monotone[index] = torch.maximum(monotone[index], monotone[index - 1] + min_step)
    monotone = monotone - monotone[0]
    monotone = monotone / monotone[-1].clamp(min=min_step)
    monotone[0] = 0.0
    monotone[-1] = 1.0
    return monotone


def _interp_lookup(query: Tensor, x_grid: Tensor, y_grid: Tensor) -> Tensor:
    flat_query = query.reshape(-1).clamp(float(x_grid[0].item()), float(x_grid[-1].item()))
    right_indices = torch.searchsorted(x_grid, flat_query, right=True)
    right_indices = right_indices.clamp(min=1, max=x_grid.numel() - 1)
    left_indices = right_indices - 1

    x_left = x_grid[left_indices]
    x_right = x_grid[right_indices]
    y_left = y_grid[left_indices]
    y_right = y_grid[right_indices]
    weight = (flat_query - x_left) / (x_right - x_left).clamp(min=EPS)
    values = y_left + weight * (y_right - y_left)
    return values.reshape_as(query)


def _stabilize_rho_star(
    rho_raw: Tensor,
    rho_floor_raw: Tensor,
    *,
    smoothing_window: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    floor_raw = rho_floor_raw.to(dtype=torch.float64).clamp(min=float(eps))
    rho_smoothed = _moving_average(
        rho_raw.to(dtype=torch.float64).clamp(min=float(eps)),
        smoothing_window,
    ).clamp(min=float(eps))
    floor_smoothed = _moving_average(floor_raw, smoothing_window).clamp(min=float(eps))
    rho_star = torch.maximum(rho_smoothed, floor_raw).clamp(min=float(eps))
    return floor_smoothed, rho_smoothed, rho_star


def _stabilize_psi_prime(
    *,
    psi_values: Tensor,
    r_grid: Tensor,
    rho_star: Tensor,
    rho_floor_raw: Tensor,
    total_mass: float,
    eps: float,
) -> tuple[Tensor, float, float]:
    rho_lookup = _interp_lookup(psi_values, r_grid, rho_star)
    rho_floor_lookup = _interp_lookup(psi_values, r_grid, rho_floor_raw)
    floor_scale = max(float(torch.median(rho_floor_raw).item()), float(eps))
    psi_prime_regularizer = max(float(eps), 0.05 * floor_scale)
    stable_denominator = torch.maximum(rho_lookup, rho_floor_lookup).clamp(min=float(eps))
    stable_denominator = stable_denominator + float(psi_prime_regularizer)
    raw_psi_prime = float(total_mass) / stable_denominator
    median_prime = max(float(torch.median(raw_psi_prime).item()), 1.0)
    quantile_prime = max(float(torch.quantile(raw_psi_prime, 0.95).item()), median_prime)
    psi_prime_cap = max(quantile_prime, 4.0 * median_prime)
    psi_prime_values = raw_psi_prime.clamp(max=float(psi_prime_cap)).to(dtype=torch.float64)
    return psi_prime_values, float(psi_prime_regularizer), float(psi_prime_cap)


def _resolve_tack_estimator(estimator: str) -> tuple[str, str]:
    requested = str(estimator or "auto")
    if requested == "jvp":
        return "jvp", "jvp"
    return "finite_diff", "fd"


def _resolve_smoothing_window(grid_size: int) -> int:
    return max(3, int(grid_size // 16) * 2 + 1)


def _power_of_two_level(scale: float, *, direction: str) -> int:
    log2_scale = math.log2(float(scale))
    if direction == "min":
        return int(math.ceil(log2_scale))
    if direction == "max":
        return int(math.floor(log2_scale))
    raise ValueError(f"Unsupported direction={direction}.")


def _time_from_solver_clock(
    samples: Tensor,
    noise: Tensor,
    t: Tensor,
    *,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
) -> Tensor:
    clock = evaluate_clock(
        r=t,
        clock_family=clock_family,
        clock_beta=clock_beta,
        path_family=path_family,
        signal_scale_sq=signal_scale_sq,
    )
    path = evaluate_path(s=clock.s, path_family=path_family)
    alpha = expand_like(path.alpha, samples)
    sigma = expand_like(path.sigma, samples)
    return sigma * noise + alpha * samples


@dataclass
class TACKConfig:
    requested_nfe: int
    path_family: str
    clock_family: str
    clock_beta: Optional[float]
    signal_scale_sq: Optional[float]
    checkpoint_source: str = ""
    profile_grid_size: int = 64
    profile_batch_size: int = 256
    profile_num_batches: int = 8
    profile_eps: float = 1.0e-8
    lambda_value: float = 1.0
    eta: float = 0.25
    profile_cache: bool = True
    force_recompute_profile: bool = False
    chi_lo: float = 0.10
    chi_hi: float = 0.50
    tau: float = 0.05
    startup_steps: int = 2
    enable_dyadic: bool = True
    batch_shared_adapt: bool = True
    min_dr_scale: float = 0.25
    max_dr_scale: float = 4.0
    monitor_estimator: str = "auto"
    mode: str = "full"
    cfg_scale: float = 0.0
    seed: int = 0


@dataclass
class TACKProfile:
    # Legacy field name kept for artifact/cache compatibility.
    # This tensor is the uniform FT-clock input r-grid, not the original physical time s.
    s_grid: Tensor
    q1_values: Tensor
    q2_values: Tensor
    q1_smoothed: Tensor
    q2_smoothed: Tensor
    rho_floor_raw: Tensor
    rho_floor_smoothed: Tensor
    rho_raw: Tensor
    rho_smoothed: Tensor
    rho_star: Tensor
    phi_values: Tensor
    psi_query_grid: Tensor
    psi_values: Tensor
    psi_prime_values: Tensor
    estimator_type: str
    lambda_value: float
    eta: float
    eps: float
    requested_eval_nfe: int
    profile_grid_size: int
    profile_batch_size: int
    profile_num_batches: int
    path_family: str
    clock_family: str
    clock_beta: Optional[float]
    signal_scale_sq: Optional[float]
    checkpoint_source: str
    smoothing_window: int
    psi_prime_regularizer: float = 0.0
    psi_prime_cap: float = 0.0
    distribution_info: Dict[str, object] = field(default_factory=dict)

    @property
    def r_grid(self) -> Tensor:
        return self.s_grid

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in (
            "s_grid",
            "q1_values",
            "q2_values",
            "q1_smoothed",
            "q2_smoothed",
            "rho_floor_raw",
            "rho_floor_smoothed",
            "rho_raw",
            "rho_smoothed",
            "rho_star",
            "phi_values",
            "psi_query_grid",
            "psi_values",
            "psi_prime_values",
        ):
            payload[key] = payload[key].detach().cpu()
        return payload

    def map_query_to_time(self, query: Tensor) -> Tensor:
        return _interp_lookup(
            query.to(device=self.psi_query_grid.device, dtype=self.psi_query_grid.dtype),
            self.psi_query_grid,
            self.psi_values,
        ).to(device=query.device, dtype=query.dtype)

    def map_query_to_prime(self, query: Tensor) -> Tensor:
        return _interp_lookup(
            query.to(device=self.psi_query_grid.device, dtype=self.psi_query_grid.dtype),
            self.psi_query_grid,
            self.psi_prime_values,
        ).to(device=query.device, dtype=query.dtype)


@dataclass
class TACKSolveOutput:
    sample: Tensor
    nfe: int
    step_count: int
    time_grid: Tensor
    step_methods: tuple[str, ...]
    trajectory: Optional[Tensor]
    deltas: Optional[Tensor]
    solver_stats: Dict[str, object]


def build_tack_config_from_namespace(
    args,
    *,
    requested_nfe: Optional[int] = None,
    checkpoint_source: Optional[str] = None,
) -> TACKConfig:
    return TACKConfig(
        requested_nfe=int(requested_nfe if requested_nfe is not None else getattr(args, "eval_nfe", 0)),
        path_family=str(getattr(args, "path_family", "linear")),
        clock_family=str(getattr(args, "clock_family", "uniform")),
        clock_beta=getattr(args, "clock_beta", None),
        signal_scale_sq=getattr(args, "signal_scale_sq", None),
        checkpoint_source=str(
            checkpoint_source
            if checkpoint_source is not None
            else getattr(args, "resume", "")
        ),
        profile_grid_size=int(getattr(args, "tack_profile_grid_size", 64)),
        profile_batch_size=int(getattr(args, "tack_profile_batch_size", 256)),
        profile_num_batches=int(getattr(args, "tack_profile_num_batches", 8)),
        profile_eps=float(getattr(args, "tack_profile_eps", 1.0e-8)),
        lambda_value=float(getattr(args, "tack_lambda", 1.0)),
        eta=float(getattr(args, "tack_eta", 0.25)),
        profile_cache=bool(getattr(args, "tack_profile_cache", True)),
        force_recompute_profile=bool(getattr(args, "tack_force_recompute_profile", False)),
        chi_lo=float(getattr(args, "tack_chi_lo", 0.10)),
        chi_hi=float(getattr(args, "tack_chi_hi", 0.50)),
        tau=float(getattr(args, "tack_tau", 0.05)),
        startup_steps=int(getattr(args, "tack_startup_steps", 2)),
        enable_dyadic=bool(getattr(args, "tack_enable_dyadic", True)),
        batch_shared_adapt=bool(getattr(args, "tack_batch_shared_adapt", True)),
        min_dr_scale=float(getattr(args, "tack_min_dr_scale", 0.25)),
        max_dr_scale=float(getattr(args, "tack_max_dr_scale", 4.0)),
        monitor_estimator=str(getattr(args, "tack_monitor_estimator", "auto")),
        mode=str(getattr(args, "tack_mode", "full")),
        cfg_scale=float(getattr(args, "cfg_scale", 0.0)),
        seed=int(getattr(args, "seed", 0)),
    )


def _profile_signature(config: TACKConfig, estimator_type: str) -> Dict[str, object]:
    return {
        "requested_nfe": int(config.requested_nfe),
        "path_family": str(config.path_family),
        "clock_family": str(config.clock_family),
        "clock_beta": _round_optional(config.clock_beta),
        "signal_scale_sq": _round_optional(config.signal_scale_sq),
        "checkpoint_source": str(config.checkpoint_source),
        "profile_grid_size": int(config.profile_grid_size),
        "profile_batch_size": int(config.profile_batch_size),
        "profile_num_batches": int(config.profile_num_batches),
        "profile_eps": float(config.profile_eps),
        "lambda_value": float(config.lambda_value),
        "eta": float(config.eta),
        "estimator_type": str(estimator_type),
        "seed": int(config.seed),
    }


def _load_profile_from_cache(
    cache_path: Path,
    signature: Dict[str, object],
) -> Optional[TACKProfile]:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("signature") != signature:
        logger.info("Ignoring TACK profile cache %s because its signature changed.", cache_path)
        return None
    profile_payload = payload.get("profile", {})
    expected_fields = {
        "s_grid",
        "q1_values",
        "q2_values",
        "q1_smoothed",
        "q2_smoothed",
        "rho_floor_raw",
        "rho_floor_smoothed",
        "rho_raw",
        "rho_smoothed",
        "rho_star",
        "phi_values",
        "psi_query_grid",
        "psi_values",
        "psi_prime_values",
        "estimator_type",
        "lambda_value",
        "eta",
        "eps",
        "requested_eval_nfe",
        "profile_grid_size",
        "profile_batch_size",
        "profile_num_batches",
        "path_family",
        "clock_family",
        "clock_beta",
        "signal_scale_sq",
        "checkpoint_source",
        "smoothing_window",
        "psi_prime_regularizer",
        "psi_prime_cap",
        "distribution_info",
    }
    if not expected_fields.issubset(profile_payload.keys()):
        logger.info("Ignoring TACK profile cache %s because required fields are missing.", cache_path)
        return None
    return TACKProfile(**{key: profile_payload[key] for key in expected_fields})


def _save_profile_cache(
    cache_path: Path,
    signature: Dict[str, object],
    profile: TACKProfile,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "profile": profile.to_dict()}, cache_path)


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_profile_plot(profile: TACKProfile, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_grid = profile.r_grid.detach().cpu()
    psi_query = profile.psi_query_grid.detach().cpu()
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    plots = [
        (axes[0, 0], r_grid, profile.q1_values.detach().cpu(), profile.q1_smoothed.detach().cpu(), "Q1(r)"),
        (axes[0, 1], r_grid, profile.q2_values.detach().cpu(), profile.q2_smoothed.detach().cpu(), "Q2(r)"),
        (axes[1, 0], r_grid, profile.rho_raw.detach().cpu(), profile.rho_star.detach().cpu(), "rho(r)"),
        (axes[1, 1], r_grid, profile.phi_values.detach().cpu(), None, "phi(r)"),
        (axes[2, 0], psi_query, profile.psi_values.detach().cpu(), None, "psi"),
        (axes[2, 1], psi_query, profile.psi_prime_values.detach().cpu(), None, "psi_prime"),
    ]
    for axis, x_values, y_primary, y_secondary, label in plots:
        axis.plot(x_values, y_primary, linewidth=2.0, label=f"{label}_raw")
        if y_secondary is not None:
            axis.plot(x_values, y_secondary, linewidth=2.0, linestyle="--", label=f"{label}_smoothed")
        axis.set_title(f"TACK {label}")
        axis.grid(alpha=0.22, linestyle="--", linewidth=0.8)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / PROFILE_PLOT_FILENAME, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_profile_artifacts(
    profile: TACKProfile,
    *,
    output_dir: Path,
    cache_path: Optional[Path],
    loaded_from_cache: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(profile.to_dict(), output_dir / PROFILE_FILENAME)
    profile_rows = []
    for index in range(profile.r_grid.numel()):
        profile_rows.append(
            {
                "grid_index": int(index),
                "r_grid": float(profile.r_grid[index].item()),
                "grid_domain": "uniform_r_input_domain",
                "q1_values": float(profile.q1_values[index].item()),
                "q2_values": float(profile.q2_values[index].item()),
                "q1_smoothed": float(profile.q1_smoothed[index].item()),
                "q2_smoothed": float(profile.q2_smoothed[index].item()),
                "rho_floor_raw": float(profile.rho_floor_raw[index].item()),
                "rho_floor_smoothed": float(profile.rho_floor_smoothed[index].item()),
                "rho_raw": float(profile.rho_raw[index].item()),
                "rho_smoothed": float(profile.rho_smoothed[index].item()),
                "rho_star": float(profile.rho_star[index].item()),
                "phi_values": float(profile.phi_values[index].item()),
            }
        )
    _write_csv(output_dir / PROFILE_CSV_FILENAME, profile_rows)
    payload = {
        "estimator_type": profile.estimator_type,
        "lambda": float(profile.lambda_value),
        "eta": float(profile.eta),
        "eps": float(profile.eps),
        "requested_eval_nfe": int(profile.requested_eval_nfe),
        "profile_grid_size": int(profile.profile_grid_size),
        "profile_batch_size": int(profile.profile_batch_size),
        "profile_num_batches": int(profile.profile_num_batches),
        "path_family": profile.path_family,
        "clock_family": profile.clock_family,
        "clock_beta": profile.clock_beta,
        "signal_scale_sq": profile.signal_scale_sq,
        "checkpoint_source": profile.checkpoint_source,
        "smoothing_window": int(profile.smoothing_window),
        "grid_domain": "uniform_r_input_domain",
        "grid_note": "profile.s_grid is a legacy field name for the uniform FT-clock input r-grid; it is not the original physical time s.",
        "cache_loaded": bool(loaded_from_cache),
        "cache_path": str(cache_path) if cache_path is not None else "",
        "r_grid": [float(value) for value in profile.r_grid.detach().cpu().tolist()],
        "q1_values": [float(value) for value in profile.q1_values.detach().cpu().tolist()],
        "q2_values": [float(value) for value in profile.q2_values.detach().cpu().tolist()],
        "q1_smoothed": [float(value) for value in profile.q1_smoothed.detach().cpu().tolist()],
        "q2_smoothed": [float(value) for value in profile.q2_smoothed.detach().cpu().tolist()],
        "rho_floor_raw": [float(value) for value in profile.rho_floor_raw.detach().cpu().tolist()],
        "rho_floor_smoothed": [
            float(value) for value in profile.rho_floor_smoothed.detach().cpu().tolist()
        ],
        "rho_raw": [float(value) for value in profile.rho_raw.detach().cpu().tolist()],
        "rho_smoothed": [float(value) for value in profile.rho_smoothed.detach().cpu().tolist()],
        "rho_star": [float(value) for value in profile.rho_star.detach().cpu().tolist()],
        "phi_values": [float(value) for value in profile.phi_values.detach().cpu().tolist()],
        "psi_query_grid": [float(value) for value in profile.psi_query_grid.detach().cpu().tolist()],
        "psi_values": [float(value) for value in profile.psi_values.detach().cpu().tolist()],
        "psi_prime_values": [
            float(value) for value in profile.psi_prime_values.detach().cpu().tolist()
        ],
        "psi_prime_regularizer": float(profile.psi_prime_regularizer),
        "psi_prime_cap": float(profile.psi_prime_cap),
        "distribution_info": dict(profile.distribution_info),
    }
    (output_dir / PROFILE_JSON_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_profile_plot(profile, output_dir)


def maybe_build_tack_profile(
    *,
    config: TACKConfig,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> Optional[TACKProfile]:
    if str(config.mode) == "online_only":
        return None

    resolved_estimator_type, internal_estimator = _resolve_tack_estimator(config.monitor_estimator)
    signature = _profile_signature(config=config, estimator_type=resolved_estimator_type)
    cache_path = output_dir / PROFILE_CACHE_FILENAME if output_dir is not None else None
    profile = None
    loaded_from_cache = False

    if (
        cache_path is not None
        and bool(config.profile_cache)
        and not bool(config.force_recompute_profile)
    ):
        profile = _load_profile_from_cache(cache_path=cache_path, signature=signature)
        if profile is not None:
            loaded_from_cache = True
            logger.info("Loaded TACK profile from cache %s.", cache_path)

    if profile is None:
        logger.info(
            "Recomputing TACK profile for requested_eval_nfe=%d with estimator=%s.",
            int(config.requested_nfe),
            resolved_estimator_type,
        )
        loader_iter = _cycle_loader(data_loader)
        noise_generator = _make_generator(device=device, seed=int(config.seed) + 24017)
        r_grid = torch.linspace(
            0.0,
            1.0,
            int(config.profile_grid_size),
            device=device,
            dtype=torch.float32,
        )
        q1_values = torch.zeros_like(r_grid)
        q2_values = torch.zeros_like(r_grid)
        microbatch_size = _resolve_monitor_microbatch_size(
            batch_size=int(config.profile_batch_size),
            estimator=internal_estimator,
        )

        for index, r_value in enumerate(r_grid):
            q1_sum = torch.zeros((), device=device, dtype=torch.float32)
            q2_sum = torch.zeros((), device=device, dtype=torch.float32)
            sample_count = 0
            for _ in range(int(config.profile_num_batches)):
                samples, labels, noise = _prepare_reference_batch(
                    loader_iter=loader_iter,
                    batch_size=int(config.profile_batch_size),
                    device=device,
                    noise_generator=noise_generator,
                )
                for sample_chunk, label_chunk, noise_chunk in zip(
                    samples.split(microbatch_size),
                    labels.split(microbatch_size),
                    noise.split(microbatch_size),
                ):
                    s_batch = torch.full(
                        (sample_chunk.shape[0],),
                        float(r_value.item()),
                        device=device,
                        dtype=sample_chunk.dtype,
                    )
                    x_s = _time_from_solver_clock(
                        samples=sample_chunk,
                        noise=noise_chunk,
                        t=s_batch,
                        path_family=config.path_family,
                        clock_family=config.clock_family,
                        clock_beta=config.clock_beta,
                        signal_scale_sq=config.signal_scale_sq,
                    )

                    if internal_estimator == "jvp":
                        first_derivative = _material_derivative_jvp(
                            velocity_model=velocity_model,
                            x=x_s,
                            s=s_batch,
                            labels=label_chunk,
                            cfg_scale=float(config.cfg_scale),
                        )

                        def a_fn(x_input: Tensor, s_input: Tensor) -> Tensor:
                            return _material_derivative_jvp(
                                velocity_model=velocity_model,
                                x=x_input,
                                s=s_input,
                                labels=label_chunk,
                                cfg_scale=float(config.cfg_scale),
                            )

                        velocity = _velocity_fn(
                            velocity_model,
                            x_s,
                            s_batch,
                            label_chunk,
                            float(config.cfg_scale),
                        )
                        _, second_derivative = _jvp(
                            a_fn,
                            (x_s, s_batch),
                            (velocity, torch.ones_like(s_batch)),
                        )
                    else:
                        velocity = _velocity_fn_fd(
                            velocity_model,
                            x_s,
                            s_batch,
                            label_chunk,
                            float(config.cfg_scale),
                        )
                        first_derivative = _material_derivative_fd(
                            velocity_model=velocity_model,
                            x=x_s,
                            s=s_batch,
                            labels=label_chunk,
                            cfg_scale=float(config.cfg_scale),
                            grid_size=int(config.profile_grid_size),
                        )
                        delta = _fd_step(s=s_batch, grid_size=int(config.profile_grid_size))
                        s_shift = (s_batch + delta).clamp(0.0, 1.0)
                        x_shift = x_s + expand_like(delta, x_s) * velocity
                        shifted_derivative = _material_derivative_fd(
                            velocity_model=velocity_model,
                            x=x_shift,
                            s=s_shift,
                            labels=label_chunk,
                            cfg_scale=float(config.cfg_scale),
                            grid_size=int(config.profile_grid_size),
                        )
                        delta_expand = expand_like(s_shift - s_batch, x_s)
                        delta_safe = torch.where(
                            delta_expand >= 0.0,
                            delta_expand.clamp(min=FD_DELTA_FLOOR),
                            delta_expand.clamp(max=-FD_DELTA_FLOOR),
                        )
                        second_derivative = (shifted_derivative - first_derivative) / delta_safe

                    q1_sum = q1_sum + first_derivative.flatten(start_dim=1).pow(2).sum(dim=1).detach().sum()
                    q2_sum = q2_sum + second_derivative.flatten(start_dim=1).pow(2).sum(dim=1).detach().sum()
                    sample_count += int(sample_chunk.shape[0])

                    del x_s, s_batch, first_derivative, second_derivative
                    if internal_estimator == "jvp":
                        del velocity
                    else:
                        del velocity, delta, s_shift, x_shift, shifted_derivative, delta_expand, delta_safe

            q1_values[index] = q1_sum / max(1, sample_count)
            q2_values[index] = q2_sum / max(1, sample_count)

        smoothing_window = _resolve_smoothing_window(int(config.profile_grid_size))
        q1_smoothed = _moving_average(q1_values.to(dtype=torch.float64).clamp(min=0.0), smoothing_window)
        q2_smoothed = _moving_average(q2_values.to(dtype=torch.float64).clamp(min=0.0), smoothing_window)
        profile_eps = float(config.profile_eps)
        target_steps = max(1, int(config.requested_nfe) - 1)
        rho_error = float(config.lambda_value) * torch.pow(q1_smoothed + profile_eps, 0.25)
        rho_floor_raw = (
            1.0
            / (3.0 * float(config.eta) * float(target_steps))
        ) * torch.sqrt((q2_smoothed + profile_eps) / (q1_smoothed + profile_eps))
        rho_raw = torch.maximum(rho_error, rho_floor_raw)
        rho_floor_smoothed, rho_smoothed, rho_star = _stabilize_rho_star(
            rho_raw=rho_raw,
            rho_floor_raw=rho_floor_raw,
            smoothing_window=smoothing_window,
            eps=profile_eps,
        )

        dr = r_grid[1:] - r_grid[:-1]
        phi_values = torch.zeros_like(r_grid, dtype=torch.float64)
        trapezoids = 0.5 * (rho_star[1:] + rho_star[:-1]) * dr.to(dtype=torch.float64)
        phi_values[1:] = torch.cumsum(trapezoids, dim=0)
        total_mass = float(phi_values[-1].item())
        phi_values = phi_values / max(total_mass, profile_eps)
        phi_values = _strictly_monotone(phi_values)
        psi_query_grid = torch.linspace(0.0, 1.0, int(config.profile_grid_size), device=device, dtype=torch.float64)
        psi_values = monotone_inverse_lookup(
            x_grid=r_grid.to(dtype=torch.float64),
            y_grid=phi_values,
            query=psi_query_grid,
        )
        psi_prime_values, psi_prime_regularizer, psi_prime_cap = _stabilize_psi_prime(
            psi_values=psi_values,
            r_grid=r_grid.to(dtype=torch.float64),
            rho_star=rho_star,
            rho_floor_raw=rho_floor_raw.to(dtype=torch.float64),
            total_mass=total_mass,
            eps=profile_eps,
        )

        profile = TACKProfile(
            s_grid=r_grid.detach(),
            q1_values=q1_values.detach(),
            q2_values=q2_values.detach(),
            q1_smoothed=q1_smoothed.to(dtype=torch.float32).detach(),
            q2_smoothed=q2_smoothed.to(dtype=torch.float32).detach(),
            rho_floor_raw=rho_floor_raw.to(dtype=torch.float32).detach(),
            rho_floor_smoothed=rho_floor_smoothed.to(dtype=torch.float32).detach(),
            rho_raw=rho_raw.to(dtype=torch.float32).detach(),
            rho_smoothed=rho_smoothed.to(dtype=torch.float32).detach(),
            rho_star=rho_star.to(dtype=torch.float32).detach(),
            phi_values=phi_values.to(dtype=torch.float32).detach(),
            psi_query_grid=psi_query_grid.to(dtype=torch.float32).detach(),
            psi_values=psi_values.to(dtype=torch.float32).detach(),
            psi_prime_values=psi_prime_values.to(dtype=torch.float32).detach(),
            estimator_type=resolved_estimator_type,
            lambda_value=float(config.lambda_value),
            eta=float(config.eta),
            eps=float(config.profile_eps),
            requested_eval_nfe=int(config.requested_nfe),
            profile_grid_size=int(config.profile_grid_size),
            profile_batch_size=int(config.profile_batch_size),
            profile_num_batches=int(config.profile_num_batches),
            path_family=str(config.path_family),
            clock_family=str(config.clock_family),
            clock_beta=config.clock_beta,
            signal_scale_sq=config.signal_scale_sq,
            checkpoint_source=str(config.checkpoint_source),
            smoothing_window=int(smoothing_window),
            psi_prime_regularizer=float(psi_prime_regularizer),
            psi_prime_cap=float(psi_prime_cap),
            distribution_info={
                "loaded_from_cache": False,
                "cache_path": str(cache_path) if cache_path is not None else "",
            },
        )

        if cache_path is not None and bool(config.profile_cache):
            _save_profile_cache(cache_path=cache_path, signature=signature, profile=profile)
            logger.info("Saved TACK profile cache to %s.", cache_path)

    profile.distribution_info = dict(profile.distribution_info)
    profile.distribution_info.update(
        {
            "loaded_from_cache": bool(loaded_from_cache),
            "cache_path": str(cache_path) if cache_path is not None else "",
        }
    )
    if output_dir is not None:
        _save_profile_artifacts(
            profile,
            output_dir=output_dir,
            cache_path=cache_path,
            loaded_from_cache=loaded_from_cache,
        )
    if cache_path is not None and loaded_from_cache:
        logger.info("TACK profile cache hit path: %s", cache_path)
    return profile


def _build_identity_profile(
    config: TACKConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> TACKProfile:
    query_grid = torch.linspace(0.0, 1.0, int(config.profile_grid_size), device=device, dtype=dtype)
    zeros = torch.zeros_like(query_grid)
    ones = torch.ones_like(query_grid)
    return TACKProfile(
        s_grid=query_grid,
        q1_values=zeros,
        q2_values=zeros,
        q1_smoothed=zeros,
        q2_smoothed=zeros,
        rho_floor_raw=ones,
        rho_floor_smoothed=ones,
        rho_raw=ones,
        rho_smoothed=ones,
        rho_star=ones,
        phi_values=query_grid,
        psi_query_grid=query_grid,
        psi_values=query_grid,
        psi_prime_values=ones,
        estimator_type="identity",
        lambda_value=float(config.lambda_value),
        eta=float(config.eta),
        eps=float(config.profile_eps),
        requested_eval_nfe=int(config.requested_nfe),
        profile_grid_size=int(config.profile_grid_size),
        profile_batch_size=int(config.profile_batch_size),
        profile_num_batches=int(config.profile_num_batches),
        path_family=str(config.path_family),
        clock_family=str(config.clock_family),
        clock_beta=config.clock_beta,
        signal_scale_sq=config.signal_scale_sq,
        checkpoint_source=str(config.checkpoint_source),
        smoothing_window=1,
        distribution_info={"identity_profile": True},
    )


def _eval_tack_field(
    profile: TACKProfile,
    q_value: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    q_tensor = torch.full((1,), float(q_value), device=device, dtype=dtype)
    mapped_time = profile.map_query_to_time(q_tensor)
    mapped_prime = profile.map_query_to_prime(q_tensor)
    return mapped_time, mapped_prime


def _save_online_artifacts(summary: Dict[str, object], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = list(summary.get("step_records", []))
    (output_dir / ONLINE_SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if steps:
        _write_csv(output_dir / ONLINE_STEPS_FILENAME, steps)
        dq_values = [float(row["dq"]) for row in steps]
        mode_to_color = {"heun": "#1f77b4", "ab2": "#ff7f0e", "ab3": "#2ca02c"}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(dq_values, bins=min(DEFAULT_PLOT_BINS, max(4, len(dq_values))), color="#1f77b4", alpha=0.85)
        axes[0].set_title("Accepted dq Histogram")
        axes[0].grid(alpha=0.22, linestyle="--", linewidth=0.8)

        time_centers = [0.5 * (float(row["q_start"]) + float(row["q_end"])) for row in steps]
        indices = list(range(len(steps)))
        for mode in ("heun", "ab2", "ab3"):
            xs = [time_centers[index] for index, row in enumerate(steps) if row["mode"] == mode]
            ys = [indices[index] for index, row in enumerate(steps) if row["mode"] == mode]
            if xs:
                axes[1].scatter(xs, ys, s=28, color=mode_to_color[mode], label=mode)
        axes[1].set_title("Mode vs Time")
        axes[1].set_xlabel("q")
        axes[1].set_ylabel("step_index")
        axes[1].grid(alpha=0.22, linestyle="--", linewidth=0.8)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / ONLINE_PLOT_FILENAME, dpi=220, bbox_inches="tight")
        plt.close(fig)


def solve_tack(
    *,
    velocity_model,
    x_init: Tensor,
    config: TACKConfig,
    profile: Optional[TACKProfile],
    return_trajectory: bool,
    artifact_dir: Optional[Path],
    **model_extras,
) -> TACKSolveOutput:
    if int(config.requested_nfe) < 2:
        raise ValueError("sampling_solver=tack requires requested_eval_nfe >= 2.")
    if not bool(config.batch_shared_adapt):
        raise NotImplementedError(
            "The current TACK implementation only supports batch-shared adaptive control."
        )
    if float(config.min_dr_scale) > 1.0 or float(config.max_dr_scale) < 1.0:
        raise ValueError("TACK requires min_dr_scale <= 1 <= max_dr_scale.")

    if hasattr(velocity_model, "reset_nfe_counter"):
        velocity_model.reset_nfe_counter()

    effective_profile = profile
    if effective_profile is None:
        effective_profile = _build_identity_profile(
            config=config,
            device=x_init.device,
            dtype=x_init.dtype,
        )

    target_steps = max(1, int(config.requested_nfe) - 1)
    min_level = _power_of_two_level(float(config.min_dr_scale), direction="min")
    max_level = _power_of_two_level(float(config.max_dr_scale), direction="max")
    if min_level > max_level:
        raise ValueError("Invalid dyadic scale range for TACK.")
    base_divisor = 2 ** max(0, -min_level)
    total_units = target_steps * base_divisor
    current_level = 0
    current_units = 0
    q_current = 0.0
    use_clock_only = str(config.mode) == "clock_only"
    use_dyadic_step_control = bool(config.enable_dyadic) and not use_clock_only
    use_nonuniform_ab = not use_clock_only
    startup_steps = max(1, int(config.startup_steps))
    trajectory_states = [x_init.clone()] if return_trajectory else None
    time_history = [0.0]
    step_methods: list[str] = []
    step_records: list[Dict[str, object]] = []
    defect_values: list[float] = []
    chi_values: list[float] = []
    mode_histogram = {"heun": 0, "ab2": 0, "ab3": 0}
    dyadic_step_histogram: Dict[str, int] = {}
    dq_history: list[float] = []
    defect_forced_heun_next = False
    z_t = x_init

    def eval_g(
        state: Tensor,
        q_value: float,
        physical_step: float,
    ) -> Tensor:
        mapped_time, mapped_prime = _eval_tack_field(
            effective_profile,
            q_value,
            device=state.device,
            dtype=state.dtype,
        )
        model_time = torch.full(
            (state.shape[0],),
            float(mapped_time.item()),
            device=state.device,
            dtype=state.dtype,
        )
        adapted_time = model_time
        adapt = getattr(velocity_model, "adapt_solver_time", None)
        if callable(adapt):
            adapted_time = adapt(
                t=model_time,
                step_size=float(physical_step),
                step_count=target_steps,
            )
        velocity = velocity_model(state, adapted_time, **model_extras)
        return expand_like(
            torch.full(
                (state.shape[0],),
                float(mapped_prime.item()),
                device=state.device,
                dtype=state.dtype,
            ),
            velocity,
        ) * velocity

    g_history = [eval_g(z_t, q_value=q_current, physical_step=1.0 / float(target_steps))]

    while current_units < total_units:
        if use_dyadic_step_control:
            unit_count = 2 ** (current_level - min_level)
        else:
            unit_count = base_divisor
        remaining_units = total_units - current_units
        while unit_count > remaining_units:
            unit_count //= 2
        dq = float(unit_count) / float(total_units)
        q_next = float(current_units + unit_count) / float(total_units)

        mapped_time_current, _ = _eval_tack_field(
            effective_profile,
            q_current,
            device=x_init.device,
            dtype=x_init.dtype,
        )
        mapped_time_next, _ = _eval_tack_field(
            effective_profile,
            q_next,
            device=x_init.device,
            dtype=x_init.dtype,
        )
        dt = float(mapped_time_next.item() - mapped_time_current.item())
        g_n = g_history[-1]

        chi_value = _compute_chi(g_history, eps=float(config.profile_eps))
        chi_valid = chi_value is not None
        if chi_valid:
            chi_values.append(float(chi_value))

        if len(step_methods) < startup_steps:
            mode = "heun"
        elif use_clock_only and defect_forced_heun_next:
            mode = "heun"
        elif not chi_valid:
            mode = "heun"
        elif float(chi_value) <= float(config.chi_lo):
            mode = "ab3"
        elif float(chi_value) <= float(config.chi_hi):
            mode = "ab2"
        else:
            mode = "heun"

        z_predict: Optional[Tensor] = None
        if mode == "ab3" and len(g_history) >= 3:
            if use_nonuniform_ab:
                if len(dq_history) >= 2:
                    z_predict = _ab3_predict_nonuniform(
                        z_n=z_t,
                        g_n=g_history[-1],
                        g_nm1=g_history[-2],
                        g_nm2=g_history[-3],
                        h=dq,
                        beta1=dq_history[-1],
                        beta0=dq_history[-2],
                        eps=float(config.profile_eps),
                    )
            else:
                z_predict = _ab3_predict_uniform(
                    z_n=z_t,
                    g_n=g_history[-1],
                    g_nm1=g_history[-2],
                    g_nm2=g_history[-3],
                    h=dq,
                )
        elif mode == "ab2" and len(g_history) >= 2:
            if use_nonuniform_ab:
                if len(dq_history) >= 1:
                    z_predict = _ab2_predict_nonuniform(
                        z_n=z_t,
                        g_n=g_history[-1],
                        g_nm1=g_history[-2],
                        h=dq,
                        beta1=dq_history[-1],
                        eps=float(config.profile_eps),
                    )
            else:
                z_predict = _ab2_predict_uniform(
                    z_n=z_t,
                    g_n=g_history[-1],
                    g_nm1=g_history[-2],
                    h=dq,
                )
        if z_predict is None:
            mode = "heun"
            z_predict = z_t + dq * g_n

        g_next = eval_g(z_predict, q_value=q_next, physical_step=dt)
        z_next = z_t + 0.5 * dq * (g_n + g_next)
        defect_value = _batch_ratio_mean(
            z_next - z_predict,
            z_next,
            float(config.profile_eps),
        )
        defect_values.append(defect_value)

        alpha = 1.0
        next_level = current_level
        next_defect_forced_heun = False
        if use_dyadic_step_control:
            alpha = float(
                math.pow(
                    float(config.tau) / max(defect_value + float(config.profile_eps), float(config.profile_eps)),
                    1.0 / 3.0,
                )
            )
            if alpha > 1.5:
                next_level = min(current_level + 1, max_level)
            elif alpha < 0.7:
                next_level = max(current_level - 1, min_level)
        elif use_clock_only and mode in {"ab2", "ab3"} and defect_value > float(config.tau):
            # Keep the one-query-per-step contract: a large defect only downgrades the next step to Heun.
            next_defect_forced_heun = True

        if next_level > current_level:
            doubling = 1
            halving = 0
        elif next_level < current_level:
            doubling = 0
            halving = 1
        else:
            doubling = 0
            halving = 0

        mode_histogram[mode] += 1
        dyadic_step_histogram[_bool_histogram_key(unit_count)] = (
            dyadic_step_histogram.get(_bool_histogram_key(unit_count), 0) + 1
        )
        step_methods.append(mode)
        step_records.append(
            {
                "step_index": int(len(step_methods) - 1),
                "mode": mode,
                "q_start": float(q_current),
                "q_end": float(q_next),
                "t_start": float(mapped_time_current.item()),
                "t_end": float(mapped_time_next.item()),
                "dq": float(dq),
                "dt": float(dt),
                "dq_history_depth": int(len(dq_history)),
                "chi": None if chi_value is None else float(chi_value),
                "chi_valid": bool(chi_valid),
                "defect": float(defect_value),
                "step_scale": float(2 ** current_level),
                "unit_count": int(unit_count),
                "alpha": float(alpha),
                "next_step_scale": float(2 ** next_level),
                "triggered_halving": int(halving),
                "triggered_doubling": int(doubling),
                "defect_forced_heun": int(defect_forced_heun_next),
                "defect_will_force_heun_next": int(next_defect_forced_heun),
            }
        )

        z_t = z_next
        q_current = q_next
        current_units += unit_count
        current_level = next_level
        defect_forced_heun_next = bool(next_defect_forced_heun)
        dq_history.append(float(dq))
        if len(dq_history) > 2:
            dq_history = dq_history[-2:]
        g_history.append(g_next)
        if len(g_history) > 3:
            g_history = g_history[-3:]
        time_history.append(float(mapped_time_next.item()))
        if trajectory_states is not None:
            trajectory_states.append(z_t.clone())

    if hasattr(velocity_model, "get_nfe"):
        realized_nfe = int(velocity_model.get_nfe())
    else:
        realized_nfe = 1 + len(step_methods)

    trajectory = None
    deltas = None
    if trajectory_states is not None:
        trajectory = torch.stack(trajectory_states, dim=0)
        deltas = trajectory[1:] - trajectory[:-1]

    mean_defect = float(sum(defect_values) / max(1, len(defect_values)))
    mean_chi = float(sum(chi_values) / max(1, len(chi_values))) if chi_values else 0.0
    solver_stats = {
        "solver": "tack",
        "requested_nfe_budget": int(config.requested_nfe),
        "actual_network_calls": int(realized_nfe),
        "requested_eval_nfe": int(config.requested_nfe),
        "realized_nfe": int(realized_nfe),
        "step_count": int(len(step_methods)),
        "virtual_stage_count": 0,
        "used_tail_step": False,
        "tail_step_methods": (),
        "is_exact_budget": bool(realized_nfe == int(config.requested_nfe)),
        "is_shared_budget": bool(
            realized_nfe == int(config.requested_nfe) and int(config.requested_nfe) in {6, 12, 18, 24, 30, 48, 96}
        ),
        "tack_mode": str(config.mode),
        "tack_profile_loaded_from_cache": bool(
            effective_profile.distribution_info.get("loaded_from_cache", False)
        ),
        "tack_profile_path": str(
            effective_profile.distribution_info.get("cache_path", "")
        ),
        "tack_num_accepted_steps": int(len(step_methods)),
        "tack_num_heun_steps": int(mode_histogram["heun"]),
        "tack_num_ab2_steps": int(mode_histogram["ab2"]),
        "tack_num_ab3_steps": int(mode_histogram["ab3"]),
        "tack_num_valid_chi_steps": int(len(chi_values)),
        "tack_num_halvings": int(sum(int(row["triggered_halving"]) for row in step_records)),
        "tack_num_doublings": int(sum(int(row["triggered_doubling"]) for row in step_records)),
        "tack_num_defect_heun_fallbacks": int(
            sum(int(row["defect_will_force_heun_next"]) for row in step_records)
        ),
        "tack_mean_defect": float(mean_defect),
        "tack_mean_chi": float(mean_chi),
        "mode_histogram": dict(mode_histogram),
        "dyadic_step_histogram": dict(dyadic_step_histogram),
        "endpoint_slope_source": "predictor_corrector_reuse",
        "step_records": step_records,
    }
    if artifact_dir is not None:
        _save_online_artifacts(summary=solver_stats, output_dir=artifact_dir)
    logger.info(
        "TACK run finished with requested_eval_nfe=%d, realized_nfe=%d, mode_counts=%s.",
        int(config.requested_nfe),
        int(realized_nfe),
        mode_histogram,
    )
    return TACKSolveOutput(
        sample=z_t,
        nfe=realized_nfe,
        step_count=len(step_methods),
        time_grid=torch.tensor(time_history, device=x_init.device, dtype=x_init.dtype),
        step_methods=tuple(step_methods),
        trajectory=trajectory,
        deltas=deltas,
        solver_stats=solver_stats,
    )
