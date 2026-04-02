from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXAMPLES_IMAGE_ROOT = REPO_ROOT / "examples" / "image"
DEFAULT_NFE_LIST = (6, 12, 18, 24, 30, 48, 96)


def bootstrap_repo_paths() -> None:
    for candidate in (REPO_ROOT, EXAMPLES_IMAGE_ROOT, SCRIPT_DIR):
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def _require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:  # pragma: no cover - depends on runtime.
        raise RuntimeError(
            f"Missing runtime dependency '{module_name}'. "
            "Install the project runtime environment before running the Euler debug toolchain."
        ) from error


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = _require_module("yaml")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return {} if payload is None else dict(payload)


def load_debug_config(config_path: Path) -> Dict[str, Any]:
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as handle:
            return dict(json.load(handle))
    return _load_yaml(config_path)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _float_tag(value: float) -> str:
    return format(float(value), "g").replace(".", "_")


def _sanitize_name(name: str) -> str:
    allowed = []
    for char in str(name):
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_dump(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _tensor_to_list(value) -> List[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class CheckpointMetadata:
    checkpoint_path: str
    artifact_group: str
    source_exp_name: str
    checkpoint_epoch: int
    dataset: str
    checkpoint_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "artifact_group": self.artifact_group,
            "source_exp_name": self.source_exp_name,
            "checkpoint_epoch": self.checkpoint_epoch,
            "dataset": self.dataset,
            "checkpoint_args": dict(self.checkpoint_args),
        }


@dataclass
class VariantProfile:
    variant_name: str
    variant_group: str
    monitor_grid_size: int
    monitor_batch_size: int
    estimator: str
    eps: float
    density_exponent: float
    smoothing_mode: str
    smoothing_window: Optional[int]
    gaussian_sigma: Optional[float]
    gaussian_radius: Optional[int]
    q_clip_quantile: Optional[float]
    density_cap_quantile: Optional[float]
    lambda_mix: float
    constraint_name: str
    min_step_ratio: Optional[float]
    max_step_ratio: Optional[float]
    max_step_over_uniform_cap: Optional[float]
    s_grid: Any
    q_raw: Any
    q_smoothed: Any
    q_clipped: Any
    density_pre_cap: Any
    density: Any
    phi: Any
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "variant_group": self.variant_group,
            "monitor_grid_size": self.monitor_grid_size,
            "monitor_batch_size": self.monitor_batch_size,
            "estimator": self.estimator,
            "eps": self.eps,
            "density_exponent": self.density_exponent,
            "smoothing_mode": self.smoothing_mode,
            "smoothing_window": self.smoothing_window,
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_radius": self.gaussian_radius,
            "q_clip_quantile": self.q_clip_quantile,
            "density_cap_quantile": self.density_cap_quantile,
            "lambda_mix": self.lambda_mix,
            "constraint_name": self.constraint_name,
            "min_step_ratio": self.min_step_ratio,
            "max_step_ratio": self.max_step_ratio,
            "max_step_over_uniform_cap": self.max_step_over_uniform_cap,
            "notes": list(self.notes),
            "s_grid": _tensor_to_list(self.s_grid),
            "q_raw": _tensor_to_list(self.q_raw),
            "q_smoothed": _tensor_to_list(self.q_smoothed),
            "q_clipped": _tensor_to_list(self.q_clipped),
            "density_pre_cap": _tensor_to_list(self.density_pre_cap),
            "density": _tensor_to_list(self.density),
            "phi": _tensor_to_list(self.phi),
        }


@dataclass
class NodeDiagnostics:
    variant_name: str
    nfe: int
    step_count: int
    smoothing_mode: str
    clipping_mode: str
    lambda_mix: float
    monitor_grid_size: int
    monitor_batch_size: int
    constraint_name: str
    uniform_nodes: Any
    nodes_unconstrained: Any
    nodes: Any
    r_grid: Any
    step_sizes: Any
    qe_non_negative: bool
    q_spike_ratio_max_over_p95: float
    phi_strictly_monotone: bool
    psi_roundtrip_max_abs_error: float
    nodes_strictly_increasing: bool
    step_sizes_positive: bool
    nodes_in_unit_interval: bool
    step_count_matches_requested: bool
    max_step: float
    min_positive_step: float
    max_step_over_uniform: float
    max_step_over_min_positive: float
    q_peak_interval: Tuple[float, float]
    density_peak_interval: Tuple[float, float]
    min_step_interval: Tuple[float, float]
    max_step_interval: Tuple[float, float]
    summary_sentence: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "nfe": self.nfe,
            "step_count": self.step_count,
            "smoothing_mode": self.smoothing_mode,
            "clipping_mode": self.clipping_mode,
            "lambda_mix": self.lambda_mix,
            "monitor_grid_size": self.monitor_grid_size,
            "monitor_batch_size": self.monitor_batch_size,
            "constraint_name": self.constraint_name,
            "qe_non_negative": self.qe_non_negative,
            "q_spike_ratio_max_over_p95": self.q_spike_ratio_max_over_p95,
            "phi_strictly_monotone": self.phi_strictly_monotone,
            "psi_roundtrip_max_abs_error": self.psi_roundtrip_max_abs_error,
            "nodes_strictly_increasing": self.nodes_strictly_increasing,
            "step_sizes_positive": self.step_sizes_positive,
            "nodes_in_unit_interval": self.nodes_in_unit_interval,
            "step_count_matches_requested": self.step_count_matches_requested,
            "max_step": self.max_step,
            "min_positive_step": self.min_positive_step,
            "max_step_over_uniform": self.max_step_over_uniform,
            "max_step_over_min_positive": self.max_step_over_min_positive,
            "q_peak_interval_start": self.q_peak_interval[0],
            "q_peak_interval_end": self.q_peak_interval[1],
            "density_peak_interval_start": self.density_peak_interval[0],
            "density_peak_interval_end": self.density_peak_interval[1],
            "min_step_interval_start": self.min_step_interval[0],
            "min_step_interval_end": self.min_step_interval[1],
            "max_step_interval_start": self.max_step_interval[0],
            "max_step_interval_end": self.max_step_interval[1],
            "summary_sentence": self.summary_sentence,
        }

    def to_json(self) -> Dict[str, Any]:
        payload = self.to_row()
        payload.update(
            {
                "uniform_nodes": _tensor_to_list(self.uniform_nodes),
                "nodes_unconstrained": _tensor_to_list(self.nodes_unconstrained),
                "nodes": _tensor_to_list(self.nodes),
                "r_grid": _tensor_to_list(self.r_grid),
                "step_sizes": _tensor_to_list(self.step_sizes),
            }
        )
        return payload


@dataclass
class StabilitySummary:
    batch_size: int
    s_grid: Any
    mean_curve: Any
    std_curve: Any
    cv_curve: Any
    key_s_rows: List[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "s_grid": _tensor_to_list(self.s_grid),
            "mean_curve": _tensor_to_list(self.mean_curve),
            "std_curve": _tensor_to_list(self.std_curve),
            "cv_curve": _tensor_to_list(self.cv_curve),
            "key_s_rows": list(self.key_s_rows),
        }


@dataclass
class RuntimeContext:
    torch: Any
    device: Any
    model: Any
    velocity_model: Any
    data_loader: Any
    dataset: Any
    dataset_name: str
    data_path: str
    checkpoint: CheckpointMetadata
    checkpoint_args: Dict[str, Any]
    path_family: str
    clock_family: str
    clock_beta: Optional[float]
    model_output_type: str
    cfg_scale: float
    seed: int
    eval_batch_size: int


@dataclass
class MonitorDebugBundle:
    context: RuntimeContext
    profiles: Dict[str, VariantProfile]
    node_diagnostics: Dict[str, Dict[int, NodeDiagnostics]]
    numerical_rows: List[Dict[str, Any]]
    stability: Dict[int, StabilitySummary]
    grid_sweep_rows: List[Dict[str, Any]]


def resolve_checkpoint_metadata(
    *,
    dataset: str,
    checkpoint_path: Optional[str],
    artifact_group: Optional[str],
    source_exp_name: Optional[str],
    checkpoint_epoch: Optional[int],
) -> CheckpointMetadata:
    bootstrap_repo_paths()
    from experiments.checkpoint_utils import find_checkpoint, load_checkpoint_args

    explicit_path = str(checkpoint_path or "").strip()
    resolved_epoch = -1 if checkpoint_epoch in {None, ""} else int(checkpoint_epoch)
    if explicit_path:
        resolved_path = Path(explicit_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = (REPO_ROOT / resolved_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {resolved_path}")
    else:
        if not artifact_group or not source_exp_name:
            raise ValueError(
                "Either checkpoint_path or artifact_group + source_exp_name must be provided."
            )
        exp_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / str(artifact_group)
            / str(dataset)
            / str(source_exp_name)
        )
        resolved_path = find_checkpoint(
            exp_dir=exp_dir,
            epoch=None if resolved_epoch < 0 else resolved_epoch,
        )
        if resolved_path is None:
            candidate = (
                exp_dir / "checkpoint.pth"
                if resolved_epoch < 0
                else exp_dir / f"checkpoint-{resolved_epoch}.pth"
            )
            raise FileNotFoundError(
                "Failed to resolve checkpoint from artifact reference. "
                f"Looked under {exp_dir} and expected {candidate.name}."
            )
    checkpoint_args = load_checkpoint_args(Path(resolved_path))
    return CheckpointMetadata(
        checkpoint_path=str(Path(resolved_path).resolve()),
        artifact_group=str(artifact_group or ""),
        source_exp_name=str(source_exp_name or ""),
        checkpoint_epoch=int(resolved_epoch),
        dataset=str(dataset),
        checkpoint_args=checkpoint_args,
    )


def _build_eval_dataset(dataset_name: str, data_path: str):
    torchvision = _require_module("torchvision")
    from training.data_transform import get_eval_transform

    transform = get_eval_transform()
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transform,
        )
    if dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=data_path,
            train=True,
            download=True,
            transform=transform,
        )
    if dataset_name == "imagenet":
        return torchvision.datasets.ImageFolder(data_path, transform=transform)
    raise ValueError(f"Unsupported dataset for Euler debug: {dataset_name}")


def prepare_runtime_context(config: Mapping[str, Any], output_root: Path) -> RuntimeContext:
    bootstrap_repo_paths()
    torch = _require_module("torch")
    _require_module("torchvision")
    checkpoint_cfg = dict(config.get("checkpoint", {}))
    dataset_name = str(config.get("dataset", checkpoint_cfg.get("dataset", "cifar10")))
    checkpoint = resolve_checkpoint_metadata(
        dataset=dataset_name,
        checkpoint_path=checkpoint_cfg.get("checkpoint_path"),
        artifact_group=checkpoint_cfg.get("artifact_group"),
        source_exp_name=checkpoint_cfg.get("source_exp_name"),
        checkpoint_epoch=checkpoint_cfg.get("checkpoint_epoch"),
    )
    checkpoint_args = dict(checkpoint.checkpoint_args)

    from models.model_configs import instantiate_model
    from training.continuous_runtime import estimate_signal_scale_sq_from_dataset
    from training.eval_loop import CFGScaledModel
    from training.load_and_save import load_model

    data_path = str(config.get("data_path") or checkpoint_args.get("data_path") or "./data/cifar10")
    dataset = _build_eval_dataset(dataset_name=dataset_name, data_path=data_path)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device for Euler debug, but CUDA is not available.")
    if device.type == "cpu" and not _as_bool(config.get("allow_cpu"), default=False):
        raise RuntimeError(
            "Euler debug defaults to GPU execution because monitor JVP and FID are heavy. "
            "Pass allow_cpu=true only if you really want a very slow CPU run."
        )

    model = instantiate_model(
        architechture=dataset_name,
        is_discrete=bool(checkpoint_args.get("discrete_flow_matching", False)),
        use_ema=_as_bool(checkpoint_args.get("use_ema"), default=False),
    )
    model.to(device=device)

    args = Namespace(
        resume=checkpoint.checkpoint_path,
        eval_only=True,
        dataset=dataset_name,
        path_family=str(checkpoint_args.get("path_family", "linear")),
        clock_family=str(checkpoint_args.get("clock_family", "uniform")),
        clock_beta=checkpoint_args.get("clock_beta"),
        model_output_type=str(checkpoint_args.get("model_output_type", "velocity")),
        time_sampling_strategy=str(checkpoint_args.get("time_sampling_strategy", "uniform")),
        mixed_lambda=float(checkpoint_args.get("mixed_lambda", 0.5)),
        stratified_bins=int(checkpoint_args.get("stratified_bins", 16)),
        signal_scale_sq=checkpoint_args.get("signal_scale_sq"),
        clock_semantics_tag=checkpoint_args.get("clock_semantics_tag"),
        start_epoch=0,
    )
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=None,
        loss_scaler=None,
        lr_schedule=None,
    )
    model.eval()

    if args.signal_scale_sq is None:
        args.signal_scale_sq = float(estimate_signal_scale_sq_from_dataset(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=_as_bool(config.get("pin_mem"), default=True),
        drop_last=False,
    )
    velocity_model = CFGScaledModel(
        model=model,
        path_family=args.path_family,
        clock_family=args.clock_family,
        clock_beta=args.clock_beta,
        signal_scale_sq=getattr(args, "signal_scale_sq", None),
        model_output_type=args.model_output_type,
    )
    velocity_model.train(False)

    checkpoint_resolution_path = output_root / "checkpoint_resolution.json"
    _json_dump(
        checkpoint_resolution_path,
        {
            **checkpoint.to_dict(),
            "resolved_data_path": data_path,
            "path_family": args.path_family,
            "clock_family": args.clock_family,
            "clock_beta": args.clock_beta,
            "model_output_type": args.model_output_type,
            "signal_scale_sq": args.signal_scale_sq,
        },
    )
    return RuntimeContext(
        torch=torch,
        device=device,
        model=model,
        velocity_model=velocity_model,
        data_loader=data_loader,
        dataset=dataset,
        dataset_name=dataset_name,
        data_path=data_path,
        checkpoint=checkpoint,
        checkpoint_args=checkpoint_args,
        path_family=str(args.path_family),
        clock_family=str(args.clock_family),
        clock_beta=args.clock_beta,
        model_output_type=str(args.model_output_type),
        cfg_scale=float(config.get("cfg_scale", checkpoint_args.get("cfg_scale", 0.0))),
        seed=int(config.get("seed", 0)),
        eval_batch_size=int(config.get("eval_batch_size", 256)),
    )


def _auto_moving_average_window(grid_size: int) -> int:
    return max(3, int(grid_size // 16) * 2 + 1)


def _moving_average(torch_mod, values, window: int):
    if values.numel() <= 2 or int(window) <= 1:
        return values.clone()
    window = min(int(window), int(values.numel()))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = torch_mod.nn.functional.pad(
        values.view(1, 1, -1),
        (radius, radius),
        mode="replicate",
    )
    kernel = torch_mod.ones(
        1,
        1,
        window,
        device=values.device,
        dtype=values.dtype,
    ) / float(window)
    return torch_mod.nn.functional.conv1d(padded, kernel).view(-1)


def _gaussian_smoothing(torch_mod, values, sigma: float, radius: int):
    if values.numel() <= 2:
        return values.clone()
    sigma = max(float(sigma), 1.0e-6)
    radius = max(1, int(radius))
    offsets = torch_mod.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch_mod.exp(-0.5 * torch_mod.square(offsets / sigma))
    kernel = kernel / kernel.sum().clamp(min=1.0e-12)
    padded = torch_mod.nn.functional.pad(
        values.view(1, 1, -1),
        (radius, radius),
        mode="replicate",
    )
    return torch_mod.nn.functional.conv1d(
        padded,
        kernel.view(1, 1, -1),
    ).view(-1)


def _normalize_density(torch_mod, density, s_grid):
    trapezoids = 0.5 * (density[1:] + density[:-1]) * (s_grid[1:] - s_grid[:-1]).to(dtype=density.dtype)
    integral = trapezoids.sum().clamp(min=1.0e-12)
    return density / integral


def _strictly_monotone_cdf(torch_mod, values):
    monotone = values.clone()
    min_step = max(float(torch_mod.finfo(monotone.dtype).eps), 1.0e-12)
    for index in range(1, monotone.numel()):
        monotone[index] = torch_mod.maximum(
            monotone[index],
            monotone[index - 1] + min_step,
        )
    monotone = monotone - monotone[0]
    monotone = monotone / monotone[-1].clamp(min=min_step)
    monotone[0] = 0.0
    monotone[-1] = 1.0
    return monotone


def _build_phi_from_density(torch_mod, s_grid, density):
    trapezoids = 0.5 * (density[1:] + density[:-1]) * (s_grid[1:] - s_grid[:-1]).to(dtype=density.dtype)
    phi = torch_mod.zeros_like(s_grid, dtype=density.dtype)
    phi[1:] = torch_mod.cumsum(trapezoids, dim=0)
    phi = phi / phi[-1].clamp(min=1.0e-12)
    return _strictly_monotone_cdf(torch_mod, phi)


def _linear_lookup(torch_mod, x_grid, y_grid, query):
    flat_query = query.reshape(-1).clamp(float(x_grid[0].item()), float(x_grid[-1].item()))
    right_indices = torch_mod.searchsorted(x_grid, flat_query, right=True)
    right_indices = right_indices.clamp(min=1, max=x_grid.numel() - 1)
    left_indices = right_indices - 1
    x_left = x_grid[left_indices]
    x_right = x_grid[right_indices]
    y_left = y_grid[left_indices]
    y_right = y_grid[right_indices]
    weight = (flat_query - x_left) / (x_right - x_left).clamp(min=1.0e-12)
    values = y_left + weight * (y_right - y_left)
    return values.reshape_as(query)


def _project_steps_with_bounds(torch_mod, steps, lower: float, upper: float):
    if lower < 0.0:
        raise ValueError("lower bound for step sizes must be non-negative.")
    if upper <= 0.0:
        raise ValueError("upper bound for step sizes must be positive.")
    step_count = int(steps.numel())
    if lower * step_count > 1.0 + 1.0e-8:
        raise ValueError("Infeasible min-step constraint: lower bound is too large.")
    if upper * step_count < 1.0 - 1.0e-8:
        raise ValueError("Infeasible max-step constraint: upper bound is too small.")

    lower_vec = torch_mod.full_like(steps, lower)
    upper_vec = torch_mod.full_like(steps, upper)
    left = float((steps - upper_vec).min().item())
    right = float((steps - lower_vec).max().item())
    projected = steps.clone()
    for _ in range(96):
        mid = 0.5 * (left + right)
        projected = torch_mod.clamp(steps - mid, min=lower_vec, max=upper_vec)
        total = float(projected.sum().item())
        if total > 1.0:
            left = mid
        else:
            right = mid
    return torch_mod.clamp(steps - right, min=lower_vec, max=upper_vec)


def _apply_step_constraints(torch_mod, nodes, min_step_ratio, max_step_ratio, max_step_over_uniform_cap):
    if all(
        value in {None, "", 0.0}
        for value in (min_step_ratio, max_step_ratio, max_step_over_uniform_cap)
    ):
        return nodes.clone()
    step_sizes = nodes[1:] - nodes[:-1]
    step_count = int(step_sizes.numel())
    uniform_step = 1.0 / float(step_count)
    lower = 0.0 if min_step_ratio in {None, ""} else float(min_step_ratio) * uniform_step
    upper = float("inf")
    if max_step_ratio not in {None, ""}:
        upper = min(upper, float(max_step_ratio) * uniform_step)
    if max_step_over_uniform_cap not in {None, ""}:
        upper = min(upper, float(max_step_over_uniform_cap) * uniform_step)
    if not math.isfinite(upper):
        upper = 1.0
    constrained_steps = _project_steps_with_bounds(
        torch_mod,
        steps=step_sizes,
        lower=lower,
        upper=upper,
    )
    constrained_nodes = torch_mod.zeros_like(nodes)
    constrained_nodes[1:] = torch_mod.cumsum(constrained_steps, dim=0)
    constrained_nodes[-1] = 1.0
    return constrained_nodes


def _interval_around_index(values, index: int) -> Tuple[float, float]:
    if int(values.numel()) == 1:
        scalar = float(values[0].item())
        return scalar, scalar
    left = max(0, int(index) - 1)
    right = min(int(values.numel()) - 1, int(index) + 1)
    return float(values[left].item()), float(values[right].item())


def _step_interval(nodes, step_index: int) -> Tuple[float, float]:
    left = int(step_index)
    right = min(int(nodes.numel()) - 1, left + 1)
    return float(nodes[left].item()), float(nodes[right].item())


def _build_interval_sentence(
    q_interval: Tuple[float, float],
    density_interval: Tuple[float, float],
    min_step_interval: Tuple[float, float],
    max_step_interval: Tuple[float, float],
) -> str:
    max_step_mid = 0.5 * (max_step_interval[0] + max_step_interval[1])
    min_step_mid = 0.5 * (min_step_interval[0] + min_step_interval[1])
    if 0.30 <= max_step_mid <= 0.70:
        coarse_region = "中段"
    elif max_step_mid > 0.70:
        coarse_region = "终段"
    else:
        coarse_region = "前段"
    if min_step_mid > 0.75:
        focused_region = "终端"
    elif min_step_mid < 0.25:
        focused_region = "前段"
    else:
        focused_region = "中段"
    return (
        f"Q_E 峰值主要落在 [{q_interval[0]:.4f}, {q_interval[1]:.4f}]，"
        f"density 峰值主要落在 [{density_interval[0]:.4f}, {density_interval[1]:.4f}]。"
        f"当前 nodes 在 {coarse_region} [{max_step_interval[0]:.4f}, {max_step_interval[1]:.4f}] 给出最大步长，"
        f"同时在 {focused_region} [{min_step_interval[0]:.4f}, {min_step_interval[1]:.4f}] 过度聚焦。"
    )


def _variant_output_dir(output_root: Path, variant_name: str) -> Path:
    return output_root / "profiles" / _sanitize_name(variant_name)


def _save_profile(profile: VariantProfile, output_root: Path) -> None:
    variant_dir = _variant_output_dir(output_root, profile.variant_name)
    curve_rows = []
    for index, values in enumerate(
        zip(
            _tensor_to_list(profile.s_grid),
            _tensor_to_list(profile.q_raw),
            _tensor_to_list(profile.q_smoothed),
            _tensor_to_list(profile.q_clipped),
            _tensor_to_list(profile.density_pre_cap),
            _tensor_to_list(profile.density),
            _tensor_to_list(profile.phi),
        )
    ):
        s_value, q_raw, q_smoothed, q_clipped, density_pre_cap, density, phi = values
        curve_rows.append(
            {
                "grid_index": index,
                "s_value": s_value,
                "q_raw": q_raw,
                "q_smoothed": q_smoothed,
                "q_clipped": q_clipped,
                "density_pre_cap": density_pre_cap,
                "density": density,
                "phi": phi,
            }
        )
    _json_dump(variant_dir / "profile.json", profile.to_json())
    _csv_dump(
        variant_dir / "profile.csv",
        (
            "grid_index",
            "s_value",
            "q_raw",
            "q_smoothed",
            "q_clipped",
            "density_pre_cap",
            "density",
            "phi",
        ),
        curve_rows,
    )


def _save_node_diagnostics(output_root: Path, diagnostics: NodeDiagnostics) -> None:
    variant_dir = _variant_output_dir(output_root, diagnostics.variant_name) / f"nfe_{int(diagnostics.nfe):03d}"
    _json_dump(variant_dir / "nodes.json", diagnostics.to_json())
    rows = []
    nodes = _tensor_to_list(diagnostics.nodes)
    nodes_unconstrained = _tensor_to_list(diagnostics.nodes_unconstrained)
    r_grid = _tensor_to_list(diagnostics.r_grid)
    uniform_nodes = _tensor_to_list(diagnostics.uniform_nodes)
    step_sizes = _tensor_to_list(diagnostics.step_sizes)
    for index in range(len(nodes)):
        rows.append(
            {
                "node_index": index,
                "r_value": r_grid[index],
                "uniform_s_value": uniform_nodes[index],
                "solver_aware_s_value": nodes[index],
                "solver_aware_unconstrained_s_value": nodes_unconstrained[index],
                "step_size_from_prev": 0.0 if index == 0 else step_sizes[index - 1],
            }
        )
    _csv_dump(
        variant_dir / "nodes.csv",
        (
            "node_index",
            "r_value",
            "uniform_s_value",
            "solver_aware_s_value",
            "solver_aware_unconstrained_s_value",
            "step_size_from_prev",
        ),
        rows,
    )


def _compute_raw_monitor(
    context: RuntimeContext,
    *,
    grid_size: int,
    batch_size: int,
    estimator: str,
    seed: int,
    cache_root: Path,
):
    torch = context.torch
    cache_dir = cache_root / "cache"
    cache_path = cache_dir / (
        f"raw_monitor_grid{int(grid_size)}_batch{int(batch_size)}"
        f"_seed{int(seed)}_est{_sanitize_name(estimator)}.pt"
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location=context.device)

    from training.solver_aware.monitors import compute_euler_monitor

    with torch.enable_grad():
        artifact = compute_euler_monitor(
            velocity_model=context.velocity_model,
            data_loader=context.data_loader,
            device=context.device,
            path_family=context.path_family,
            grid_size=int(grid_size),
            batch_size=int(batch_size),
            estimator=str(estimator),
            cfg_scale=float(context.cfg_scale),
            seed=int(seed),
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "s_grid": artifact.s_grid.detach().cpu(),
            "q_values": artifact.q_values.detach().cpu(),
            "resolved_estimator": artifact.resolved_estimator,
            "monitor_name": artifact.monitor_name,
            "density_exponent": artifact.density_exponent,
            "theorem_backed": artifact.theorem_backed,
            "notes": artifact.notes,
            "grid_size": int(grid_size),
            "batch_size": int(batch_size),
            "seed": int(seed),
        },
        cache_path,
    )
    return torch.load(cache_path, map_location=context.device)


def _build_profile_from_monitor(
    context: RuntimeContext,
    *,
    monitor_payload: Mapping[str, Any],
    output_root: Path,
    variant_name: str,
    variant_group: str,
    smoothing_mode: str,
    smoothing_window: Optional[int],
    gaussian_sigma: Optional[float],
    gaussian_radius: Optional[int],
    q_clip_quantile: Optional[float],
    density_cap_quantile: Optional[float],
    lambda_mix: float,
    constraint_name: str,
    min_step_ratio: Optional[float],
    max_step_ratio: Optional[float],
    max_step_over_uniform_cap: Optional[float],
    eps: float,
) -> VariantProfile:
    torch = context.torch
    s_grid = monitor_payload["s_grid"].to(device=context.device, dtype=torch.float64)
    q_raw = monitor_payload["q_values"].to(device=context.device, dtype=torch.float64).clamp(min=0.0)
    if smoothing_mode == "none":
        q_smoothed = q_raw.clone()
        resolved_window = 1
    elif smoothing_mode == "moving_average":
        resolved_window = (
            _auto_moving_average_window(int(s_grid.numel()))
            if smoothing_window in {None, "auto"}
            else max(1, int(smoothing_window))
        )
        q_smoothed = _moving_average(torch, q_raw, window=resolved_window)
    elif smoothing_mode == "gaussian":
        resolved_window = None
        q_smoothed = _gaussian_smoothing(
            torch,
            q_raw,
            sigma=float(gaussian_sigma or 1.5),
            radius=int(gaussian_radius or 4),
        )
    else:
        raise ValueError(f"Unsupported smoothing_mode={smoothing_mode}")

    q_clipped = q_smoothed.clone()
    notes = []
    if q_clip_quantile not in {None, ""}:
        q_cap = torch.quantile(q_clipped, float(q_clip_quantile))
        q_clipped = q_clipped.clamp(max=q_cap)
        notes.append(f"Applied Q quantile clipping at {float(q_clip_quantile):.2f}.")

    density_pre_cap = torch.pow(q_clipped + float(eps), float(monitor_payload["density_exponent"]))
    density_pre_cap = _normalize_density(torch, density_pre_cap, s_grid)
    density = density_pre_cap.clone()

    if density_cap_quantile not in {None, ""}:
        density_cap = torch.quantile(density, float(density_cap_quantile))
        density = density.clamp(max=density_cap)
        density = _normalize_density(torch, density, s_grid)
        notes.append(f"Applied density quantile cap at {float(density_cap_quantile):.2f}.")

    lambda_mix = float(lambda_mix)
    if lambda_mix < 1.0 - 1.0e-12:
        density = (1.0 - lambda_mix) * torch.ones_like(density) + lambda_mix * density
        density = _normalize_density(torch, density, s_grid)
        notes.append(f"Mixed density with uniform using lambda={lambda_mix:.2f}.")

    phi = _build_phi_from_density(torch, s_grid, density)
    profile = VariantProfile(
        variant_name=variant_name,
        variant_group=variant_group,
        monitor_grid_size=int(monitor_payload["grid_size"]),
        monitor_batch_size=int(monitor_payload["batch_size"]),
        estimator=str(monitor_payload["resolved_estimator"]),
        eps=float(eps),
        density_exponent=float(monitor_payload["density_exponent"]),
        smoothing_mode=smoothing_mode,
        smoothing_window=resolved_window if smoothing_mode == "moving_average" else None,
        gaussian_sigma=None if smoothing_mode != "gaussian" else float(gaussian_sigma or 1.5),
        gaussian_radius=None if smoothing_mode != "gaussian" else int(gaussian_radius or 4),
        q_clip_quantile=None if q_clip_quantile in {None, ""} else float(q_clip_quantile),
        density_cap_quantile=(
            None if density_cap_quantile in {None, ""} else float(density_cap_quantile)
        ),
        lambda_mix=lambda_mix,
        constraint_name=str(constraint_name),
        min_step_ratio=(
            None if min_step_ratio in {None, ""} else float(min_step_ratio)
        ),
        max_step_ratio=(
            None if max_step_ratio in {None, ""} else float(max_step_ratio)
        ),
        max_step_over_uniform_cap=(
            None
            if max_step_over_uniform_cap in {None, ""}
            else float(max_step_over_uniform_cap)
        ),
        s_grid=s_grid.to(dtype=torch.float32),
        q_raw=q_raw.to(dtype=torch.float32),
        q_smoothed=q_smoothed.to(dtype=torch.float32),
        q_clipped=q_clipped.to(dtype=torch.float32),
        density_pre_cap=density_pre_cap.to(dtype=torch.float32),
        density=density.to(dtype=torch.float32),
        phi=phi.to(dtype=torch.float32),
        notes=notes,
    )
    _save_profile(profile=profile, output_root=output_root)
    return profile


def materialize_nodes_for_profile(
    profile: VariantProfile,
    *,
    nfe: int,
    output_root: Path,
) -> NodeDiagnostics:
    bootstrap_repo_paths()
    from training.solver_aware.clock import build_solver_aware_nodes

    torch = _require_module("torch")
    r_grid, nodes_unconstrained = build_solver_aware_nodes(
        s_grid=profile.s_grid.to(dtype=torch.float32),
        phi=profile.phi.to(dtype=torch.float32),
        node_count=int(nfe) + 1,
    )
    uniform_nodes = torch.linspace(
        0.0,
        1.0,
        int(nfe) + 1,
        device=nodes_unconstrained.device,
        dtype=nodes_unconstrained.dtype,
    )
    nodes = _apply_step_constraints(
        torch,
        nodes_unconstrained,
        min_step_ratio=profile.min_step_ratio,
        max_step_ratio=profile.max_step_ratio,
        max_step_over_uniform_cap=profile.max_step_over_uniform_cap,
    )
    step_sizes = nodes[1:] - nodes[:-1]
    uniform_step = 1.0 / float(max(1, int(nfe)))
    positive_steps = step_sizes.clamp(min=float(torch.finfo(step_sizes.dtype).eps))
    q_spike_ratio = float(
        profile.q_raw.max().item()
        / torch.quantile(profile.q_raw, 0.95).clamp(min=1.0e-12).item()
    )
    phi_at_nodes = _linear_lookup(
        torch,
        profile.s_grid.to(dtype=torch.float64),
        profile.phi.to(dtype=torch.float64),
        nodes.to(dtype=torch.float64),
    )
    q_peak_index = int(torch.argmax(profile.q_raw).item())
    density_peak_index = int(torch.argmax(profile.density).item())
    min_step_index = int(torch.argmin(step_sizes).item())
    max_step_index = int(torch.argmax(step_sizes).item())
    diagnostics = NodeDiagnostics(
        variant_name=profile.variant_name,
        nfe=int(nfe),
        step_count=int(nfe),
        smoothing_mode=profile.smoothing_mode,
        clipping_mode=(
            "none"
            if profile.q_clip_quantile is None and profile.density_cap_quantile is None
            else (
                f"q_clip_{_float_tag(profile.q_clip_quantile)}"
                if profile.q_clip_quantile is not None
                else f"density_cap_{_float_tag(profile.density_cap_quantile)}"
            )
        ),
        lambda_mix=float(profile.lambda_mix),
        monitor_grid_size=int(profile.monitor_grid_size),
        monitor_batch_size=int(profile.monitor_batch_size),
        constraint_name=str(profile.constraint_name),
        uniform_nodes=uniform_nodes.detach().cpu(),
        nodes_unconstrained=nodes_unconstrained.detach().cpu(),
        nodes=nodes.detach().cpu(),
        r_grid=r_grid.detach().cpu(),
        step_sizes=step_sizes.detach().cpu(),
        qe_non_negative=bool(torch.all(profile.q_raw >= -1.0e-12)),
        q_spike_ratio_max_over_p95=q_spike_ratio,
        phi_strictly_monotone=bool(torch.all(profile.phi[1:] > profile.phi[:-1])),
        psi_roundtrip_max_abs_error=float(torch.max(torch.abs(phi_at_nodes - r_grid.to(dtype=phi_at_nodes.dtype))).item()),
        nodes_strictly_increasing=bool(torch.all(nodes[1:] > nodes[:-1])),
        step_sizes_positive=bool(torch.all(step_sizes > 0.0)),
        nodes_in_unit_interval=bool(
            torch.all(nodes >= -1.0e-8) and torch.all(nodes <= 1.0 + 1.0e-8)
        ),
        step_count_matches_requested=bool(int(nodes.numel()) == int(nfe) + 1),
        max_step=float(step_sizes.max().item()),
        min_positive_step=float(positive_steps.min().item()),
        max_step_over_uniform=float(step_sizes.max().item() / uniform_step),
        max_step_over_min_positive=float(step_sizes.max().item() / positive_steps.min().item()),
        q_peak_interval=_interval_around_index(profile.s_grid, q_peak_index),
        density_peak_interval=_interval_around_index(profile.s_grid, density_peak_index),
        min_step_interval=_step_interval(nodes, min_step_index),
        max_step_interval=_step_interval(nodes, max_step_index),
        summary_sentence="",
    )
    diagnostics.summary_sentence = _build_interval_sentence(
        q_interval=diagnostics.q_peak_interval,
        density_interval=diagnostics.density_peak_interval,
        min_step_interval=diagnostics.min_step_interval,
        max_step_interval=diagnostics.max_step_interval,
    )
    _save_node_diagnostics(output_root=output_root, diagnostics=diagnostics)
    return diagnostics


def _build_variants(
    context: RuntimeContext,
    config: Mapping[str, Any],
    raw_monitors: Mapping[int, Mapping[str, Any]],
    output_root: Path,
) -> Dict[str, VariantProfile]:
    monitor_cfg = dict(config.get("monitor", {}))
    smoothing_cfg = dict(config.get("smoothing", {}))
    clipping_cfg = dict(config.get("clipping", {}))
    mixture_cfg = dict(config.get("mixture", {}))
    constraints_cfg = dict(config.get("constraints", {}))
    default_grid_size = int(monitor_cfg.get("default_grid_size", 65))
    gaussian_sigma = float(smoothing_cfg.get("gaussian_sigma", 1.5))
    gaussian_radius = int(smoothing_cfg.get("gaussian_radius", 4))
    moving_average_window = smoothing_cfg.get("moving_average_window", "auto")
    eps = float(monitor_cfg.get("eps", 1.0e-6))

    default_monitor = raw_monitors[default_grid_size]
    variants: Dict[str, VariantProfile] = {}
    variants["solver_aware_no_smoothing"] = _build_profile_from_monitor(
        context,
        monitor_payload=default_monitor,
        output_root=output_root,
        variant_name="solver_aware_no_smoothing",
        variant_group="smoothing",
        smoothing_mode="none",
        smoothing_window=None,
        gaussian_sigma=None,
        gaussian_radius=None,
        q_clip_quantile=None,
        density_cap_quantile=None,
        lambda_mix=1.0,
        constraint_name="unconstrained",
        min_step_ratio=None,
        max_step_ratio=None,
        max_step_over_uniform_cap=None,
        eps=eps,
    )
    variants["solver_aware_current_impl"] = _build_profile_from_monitor(
        context,
        monitor_payload=default_monitor,
        output_root=output_root,
        variant_name="solver_aware_current_impl",
        variant_group="smoothing",
        smoothing_mode="moving_average",
        smoothing_window=moving_average_window,
        gaussian_sigma=None,
        gaussian_radius=None,
        q_clip_quantile=None,
        density_cap_quantile=None,
        lambda_mix=1.0,
        constraint_name="unconstrained",
        min_step_ratio=None,
        max_step_ratio=None,
        max_step_over_uniform_cap=None,
        eps=eps,
    )
    variants["solver_aware_gaussian"] = _build_profile_from_monitor(
        context,
        monitor_payload=default_monitor,
        output_root=output_root,
        variant_name="solver_aware_gaussian",
        variant_group="smoothing",
        smoothing_mode="gaussian",
        smoothing_window=None,
        gaussian_sigma=gaussian_sigma,
        gaussian_radius=gaussian_radius,
        q_clip_quantile=None,
        density_cap_quantile=None,
        lambda_mix=1.0,
        constraint_name="unconstrained",
        min_step_ratio=None,
        max_step_ratio=None,
        max_step_over_uniform_cap=None,
        eps=eps,
    )

    for quantile in clipping_cfg.get("q_quantiles", []):
        variant_name = f"solver_aware_qclip_{_float_tag(float(quantile))}"
        variants[variant_name] = _build_profile_from_monitor(
            context,
            monitor_payload=default_monitor,
            output_root=output_root,
            variant_name=variant_name,
            variant_group="clipping",
            smoothing_mode="moving_average",
            smoothing_window=moving_average_window,
            gaussian_sigma=None,
            gaussian_radius=None,
            q_clip_quantile=float(quantile),
            density_cap_quantile=None,
            lambda_mix=1.0,
            constraint_name="unconstrained",
            min_step_ratio=None,
            max_step_ratio=None,
            max_step_over_uniform_cap=None,
            eps=eps,
        )

    for quantile in clipping_cfg.get("density_cap_quantiles", []):
        variant_name = f"solver_aware_densitycap_{_float_tag(float(quantile))}"
        variants[variant_name] = _build_profile_from_monitor(
            context,
            monitor_payload=default_monitor,
            output_root=output_root,
            variant_name=variant_name,
            variant_group="clipping",
            smoothing_mode="moving_average",
            smoothing_window=moving_average_window,
            gaussian_sigma=None,
            gaussian_radius=None,
            q_clip_quantile=None,
            density_cap_quantile=float(quantile),
            lambda_mix=1.0,
            constraint_name="unconstrained",
            min_step_ratio=None,
            max_step_ratio=None,
            max_step_over_uniform_cap=None,
            eps=eps,
        )

    for lambda_mix in mixture_cfg.get("lambdas", []):
        variant_name = f"solver_aware_mix_lambda_{_float_tag(float(lambda_mix))}"
        variants[variant_name] = _build_profile_from_monitor(
            context,
            monitor_payload=default_monitor,
            output_root=output_root,
            variant_name=variant_name,
            variant_group="mixture",
            smoothing_mode="moving_average",
            smoothing_window=moving_average_window,
            gaussian_sigma=None,
            gaussian_radius=None,
            q_clip_quantile=None,
            density_cap_quantile=None,
            lambda_mix=float(lambda_mix),
            constraint_name="unconstrained",
            min_step_ratio=None,
            max_step_ratio=None,
            max_step_over_uniform_cap=None,
            eps=eps,
        )

    for constraint_name in ("mild", "strong"):
        constraint_payload = constraints_cfg.get(constraint_name, {})
        if not constraint_payload:
            continue
        variant_name = f"solver_aware_constraint_{constraint_name}"
        variants[variant_name] = _build_profile_from_monitor(
            context,
            monitor_payload=default_monitor,
            output_root=output_root,
            variant_name=variant_name,
            variant_group="constraints",
            smoothing_mode="moving_average",
            smoothing_window=moving_average_window,
            gaussian_sigma=None,
            gaussian_radius=None,
            q_clip_quantile=None,
            density_cap_quantile=None,
            lambda_mix=1.0,
            constraint_name=constraint_name,
            min_step_ratio=constraint_payload.get("min_step_ratio"),
            max_step_ratio=constraint_payload.get("max_step_ratio"),
            max_step_over_uniform_cap=constraint_payload.get("max_step_over_uniform"),
            eps=eps,
        )

    for grid_size, monitor_payload in raw_monitors.items():
        if int(grid_size) == int(default_grid_size):
            continue
        variant_name = f"solver_aware_grid_{int(grid_size)}"
        variants[variant_name] = _build_profile_from_monitor(
            context,
            monitor_payload=monitor_payload,
            output_root=output_root,
            variant_name=variant_name,
            variant_group="grid_sweep",
            smoothing_mode="moving_average",
            smoothing_window=moving_average_window,
            gaussian_sigma=None,
            gaussian_radius=None,
            q_clip_quantile=None,
            density_cap_quantile=None,
            lambda_mix=1.0,
            constraint_name="unconstrained",
            min_step_ratio=None,
            max_step_ratio=None,
            max_step_over_uniform_cap=None,
            eps=eps,
        )
    return variants


def _compute_stability(
    context: RuntimeContext,
    config: Mapping[str, Any],
    output_root: Path,
) -> Dict[int, StabilitySummary]:
    torch = context.torch
    monitor_cfg = dict(config.get("monitor", {}))
    if not _as_bool(dict(config.get("execution", {})).get("run_stability_check"), default=True):
        return {}
    grid_size = int(monitor_cfg.get("default_grid_size", 65))
    estimator = str(monitor_cfg.get("estimator", "auto"))
    seeds = [int(seed) for seed in monitor_cfg.get("stability_seeds", [0, 1, 2])]
    batch_sizes = [int(size) for size in monitor_cfg.get("stability_batch_sizes", [32, 64, 128])]
    key_s_values = [float(value) for value in monitor_cfg.get("key_s_values", [0.1, 0.3, 0.5, 0.7, 0.9])]
    summaries: Dict[int, StabilitySummary] = {}
    all_rows = []

    for batch_size in batch_sizes:
        curves = []
        for seed in seeds:
            payload = _compute_raw_monitor(
                context,
                grid_size=grid_size,
                batch_size=batch_size,
                estimator=estimator,
                seed=seed,
                cache_root=output_root,
            )
            curves.append(payload["q_values"].to(device=context.device, dtype=torch.float64))
        stacked = torch.stack(curves, dim=0)
        mean_curve = stacked.mean(dim=0)
        std_curve = stacked.std(dim=0, unbiased=False)
        cv_curve = std_curve / mean_curve.clamp(min=1.0e-12)
        s_grid = payload["s_grid"].to(device=context.device, dtype=torch.float64)
        key_rows = []
        for target_s in key_s_values:
            index = int(torch.argmin(torch.abs(s_grid - float(target_s))).item())
            key_row = {
                "batch_size": batch_size,
                "s_value": float(s_grid[index].item()),
                "mean_q": float(mean_curve[index].item()),
                "std_q": float(std_curve[index].item()),
                "cv_q": float(cv_curve[index].item()),
            }
            key_rows.append(key_row)
            all_rows.append(key_row)
        summary = StabilitySummary(
            batch_size=batch_size,
            s_grid=s_grid.detach().cpu(),
            mean_curve=mean_curve.detach().cpu(),
            std_curve=std_curve.detach().cpu(),
            cv_curve=cv_curve.detach().cpu(),
            key_s_rows=key_rows,
        )
        summaries[batch_size] = summary
        _json_dump(
            output_root / "stability" / f"batch_{batch_size:03d}" / "summary.json",
            summary.to_json(),
        )
        _csv_dump(
            output_root / "stability" / f"batch_{batch_size:03d}" / "key_s.csv",
            ("batch_size", "s_value", "mean_q", "std_q", "cv_q"),
            key_rows,
        )
    _csv_dump(
        output_root / "stability" / "key_s_summary.csv",
        ("batch_size", "s_value", "mean_q", "std_q", "cv_q"),
        all_rows,
    )
    return summaries


def run_monitor_debug(
    *,
    context: RuntimeContext,
    config: Mapping[str, Any],
    output_root: Path,
) -> MonitorDebugBundle:
    monitor_cfg = dict(config.get("monitor", {}))
    nfe_list = [int(value) for value in config.get("nfe_list", DEFAULT_NFE_LIST)]
    grid_size_sweep = [
        int(value)
        for value in monitor_cfg.get("grid_size_sweep", [monitor_cfg.get("default_grid_size", 65)])
    ]
    default_batch_size = int(monitor_cfg.get("default_batch_size", 64))
    estimator = str(monitor_cfg.get("estimator", "auto"))
    raw_monitors: Dict[int, Mapping[str, Any]] = {}
    for grid_size in grid_size_sweep:
        raw_monitors[int(grid_size)] = _compute_raw_monitor(
            context,
            grid_size=int(grid_size),
            batch_size=default_batch_size,
            estimator=estimator,
            seed=int(context.seed),
            cache_root=output_root,
        )

    profiles = _build_variants(
        context=context,
        config=config,
        raw_monitors=raw_monitors,
        output_root=output_root,
    )
    node_diagnostics: Dict[str, Dict[int, NodeDiagnostics]] = {}
    numerical_rows: List[Dict[str, Any]] = []
    for variant_name, profile in profiles.items():
        node_diagnostics[variant_name] = {}
        for nfe in nfe_list:
            diagnostics = materialize_nodes_for_profile(
                profile=profile,
                nfe=int(nfe),
                output_root=output_root,
            )
            node_diagnostics[variant_name][int(nfe)] = diagnostics
            numerical_rows.append(diagnostics.to_row())

    stability = _compute_stability(context=context, config=config, output_root=output_root)
    grid_sweep_rows = []
    reference_name = "solver_aware_current_impl"
    reference_nfes = (6, 12, 24, 96)
    if reference_name in node_diagnostics:
        reference_profile = profiles[reference_name]
        for variant_name, profile in profiles.items():
            if profile.variant_group != "grid_sweep":
                continue
            for nfe in reference_nfes:
                if int(nfe) not in node_diagnostics[reference_name]:
                    continue
                candidate = node_diagnostics[variant_name][int(nfe)]
                reference = node_diagnostics[reference_name][int(nfe)]
                node_diff = (
                    context.torch.tensor(_tensor_to_list(candidate.nodes))
                    - context.torch.tensor(_tensor_to_list(reference.nodes))
                )
                reference_s_grid = reference_profile.s_grid.to(
                    device=context.device,
                    dtype=context.torch.float64,
                )
                reference_phi = reference_profile.phi.to(
                    device=context.device,
                    dtype=context.torch.float64,
                )
                candidate_phi_on_reference = _linear_lookup(
                    context.torch,
                    profile.s_grid.to(device=context.device, dtype=context.torch.float64),
                    profile.phi.to(device=context.device, dtype=context.torch.float64),
                    reference_s_grid,
                )
                phi_diff = context.torch.max(
                    context.torch.abs(candidate_phi_on_reference - reference_phi)
                )
                grid_sweep_rows.append(
                    {
                        "variant_name": variant_name,
                        "reference_variant": reference_name,
                        "nfe": int(nfe),
                        "monitor_grid_size": int(profile.monitor_grid_size),
                        "reference_monitor_grid_size": int(reference_profile.monitor_grid_size),
                        "phi_linf_diff": float(phi_diff.item()),
                        "node_linf_diff": float(context.torch.max(context.torch.abs(node_diff)).item()),
                        "node_l2_diff": float(context.torch.linalg.norm(node_diff).item()),
                        "candidate_max_step_over_uniform": candidate.max_step_over_uniform,
                        "reference_max_step_over_uniform": reference.max_step_over_uniform,
                    }
                )
    _csv_dump(
        output_root / "numerical_checks.csv",
        tuple(numerical_rows[0].keys()) if numerical_rows else (),
        numerical_rows,
    )
    if grid_sweep_rows:
        _csv_dump(
            output_root / "grid_sweep" / "grid_sweep_nodes.csv",
            tuple(grid_sweep_rows[0].keys()),
            grid_sweep_rows,
        )
    return MonitorDebugBundle(
        context=context,
        profiles=profiles,
        node_diagnostics=node_diagnostics,
        numerical_rows=numerical_rows,
        stability=stability,
        grid_sweep_rows=grid_sweep_rows,
    )
