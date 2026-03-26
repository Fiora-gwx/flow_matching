from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

EPS = 1e-8
TIME_EPS = 1e-5
FT_CLOCK_GRID_SIZE = 4097
FT_TRIG_VP_UNIT_SCALE_TOL = 1e-6
PATH_FAMILIES = ("linear", "trig_vp")
FT_CLOCK_FAMILIES = ("ft_beta",)
MODEL_OUTPUT_TYPES = ("velocity", "base_velocity")
CLOCK_FAMILIES = (
    "uniform",
    "ft_beta",
    "poly_a0.5",
    "poly_a2.0",
    "cosine",
    "sigmoid_k8",
    "exp_l3",
)
_FT_CLOCK_GRID_CACHE: Dict[Tuple[str, float, float, str, str], Tuple[Tensor, Tensor]] = {}
_CLOCK_IMPORTANCE_CDF_CACHE: Dict[
    Tuple[str, str, float, float, str, str],
    Tuple[Tensor, Tensor],
] = {}


@dataclass
class ClockOutput:
    s: Tensor
    ds_dr: Tensor


@dataclass
class PathOutput:
    alpha: Tensor
    sigma: Tensor
    d_alpha: Tensor
    d_sigma: Tensor


@dataclass
class ContinuousPathBatch:
    r: Tensor
    s: Tensor
    ds_dr: Tensor
    x_t: Tensor
    base_velocity: Tensor
    target_velocity: Tensor


def is_ft_clock_family(clock_family: str) -> bool:
    return clock_family in FT_CLOCK_FAMILIES


def _validate_beta(clock_family: str, clock_beta: Optional[float]) -> float:
    if is_ft_clock_family(clock_family):
        if clock_beta is None:
            raise ValueError(f"clock_family={clock_family} requires --clock_beta.")
        if not (0.0 < clock_beta < 1.0):
            raise ValueError(f"clock_beta must be in (0, 1). Got {clock_beta}.")
        return clock_beta
    return 0.5 if clock_beta is None else clock_beta


def _safe_one_minus(x: Tensor) -> Tensor:
    return (1.0 - x).clamp(min=EPS)


def _with_exact_endpoints(r: Tensor, s: Tensor, ds_dr: Tensor) -> ClockOutput:
    s = torch.where(r <= 0.0, torch.zeros_like(s), s)
    s = torch.where(r >= 1.0, torch.ones_like(s), s)
    ds_dr = torch.where(r <= 0.0, torch.zeros_like(ds_dr), ds_dr)
    ds_dr = torch.where(r >= 1.0, torch.zeros_like(ds_dr), ds_dr)
    return ClockOutput(s=s.clamp(0.0, 1.0), ds_dr=ds_dr)


def _normalized_sigmoid(r: Tensor, k: float = 8.0) -> ClockOutput:
    z = k * (r - 0.5)
    lower = torch.sigmoid(torch.tensor(-0.5 * k, dtype=r.dtype, device=r.device))
    upper = torch.sigmoid(torch.tensor(0.5 * k, dtype=r.dtype, device=r.device))
    denom = (upper - lower).clamp(min=EPS)
    base = torch.sigmoid(z)
    s = (base - lower) / denom
    ds_dr = k * base * (1.0 - base) / denom
    return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)


def _normalized_exponential(r: Tensor, lamb: float = 3.0) -> ClockOutput:
    denom = torch.expm1(torch.tensor(lamb, dtype=r.dtype, device=r.device)).clamp(min=EPS)
    exp_term = torch.exp(lamb * r)
    s = torch.expm1(lamb * r) / denom
    ds_dr = lamb * exp_term / denom
    return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)


def _signal_scale_tensor(signal_scale_sq: float, reference: Tensor) -> Tensor:
    return torch.tensor(signal_scale_sq, dtype=reference.dtype, device=reference.device)


def evaluate_mean_terminal_error(path: PathOutput, signal_scale_sq: float) -> Tensor:
    signal_scale = _signal_scale_tensor(signal_scale_sq=signal_scale_sq, reference=path.alpha)
    return signal_scale * torch.pow(1.0 - path.alpha, 2.0) + torch.pow(path.sigma, 2.0)


def evaluate_mean_terminal_error_derivative(
    path: PathOutput,
    signal_scale_sq: float,
) -> Tensor:
    signal_scale = _signal_scale_tensor(signal_scale_sq=signal_scale_sq, reference=path.alpha)
    return (
        -2.0 * signal_scale * (1.0 - path.alpha) * path.d_alpha
        + 2.0 * path.sigma * path.d_sigma
    )


def resolve_clock_semantics_tag(
    path_family: str,
    clock_family: str,
    signal_scale_sq: Optional[float] = None,
) -> str:
    if not is_ft_clock_family(clock_family):
        return f"{path_family}:{clock_family}:v1"

    if path_family == "linear":
        return "ft_global_v2_linear_closed_form"

    if path_family == "trig_vp":
        if signal_scale_sq is not None and abs(float(signal_scale_sq) - 1.0) <= FT_TRIG_VP_UNIT_SCALE_TOL:
            return "ft_global_v2_trig_vp_rho1_closed_form"
        signal_scale_tag = "unknown" if signal_scale_sq is None else format(float(signal_scale_sq), ".6g")
        return f"ft_global_v2_trig_vp_numeric_inverse_ssq_{signal_scale_tag}"

    return f"ft_global_v2_{path_family}"


def _ft_clock_cache_key(
    path_family: str,
    beta: float,
    signal_scale_sq: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[str, float, float, str, str]:
    return (
        path_family,
        round(float(beta), 10),
        round(float(signal_scale_sq), 10),
        str(device),
        str(dtype),
    )


def _build_ft_clock_grid(
    path_family: str,
    beta: float,
    signal_scale_sq: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    cache_key = _ft_clock_cache_key(
        path_family=path_family,
        beta=beta,
        signal_scale_sq=signal_scale_sq,
        device=device,
        dtype=dtype,
    )
    cached = _FT_CLOCK_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    s_grid = torch.linspace(
        0.0,
        1.0,
        FT_CLOCK_GRID_SIZE,
        device=device,
        dtype=dtype,
    )
    path = evaluate_path(s=s_grid, path_family=path_family)
    mean_error = evaluate_mean_terminal_error(path=path, signal_scale_sq=signal_scale_sq)
    mean_error = mean_error.clamp(min=0.0)
    mean_error_0 = mean_error[0].clamp(min=EPS)
    r_grid = 1.0 - torch.pow((mean_error / mean_error_0).clamp(min=0.0, max=1.0), 1.0 - beta)
    r_grid = torch.cummax(r_grid, dim=0).values
    r_grid[0] = 0.0
    r_grid[-1] = 1.0

    _FT_CLOCK_GRID_CACHE[cache_key] = (s_grid, r_grid)
    return s_grid, r_grid


def _interpolate_monotone_inverse(
    r: Tensor,
    s_grid: Tensor,
    r_grid: Tensor,
) -> Tensor:
    flat_r = r.reshape(-1).clamp(0.0, 1.0)
    right_indices = torch.searchsorted(r_grid, flat_r, right=True)
    right_indices = right_indices.clamp(min=1, max=r_grid.numel() - 1)
    left_indices = right_indices - 1

    r_left = r_grid[left_indices]
    r_right = r_grid[right_indices]
    s_left = s_grid[left_indices]
    s_right = s_grid[right_indices]

    weight = (flat_r - r_left) / (r_right - r_left).clamp(min=EPS)
    interpolated = s_left + weight * (s_right - s_left)
    return interpolated.reshape_as(r)


def _clock_importance_cache_key(
    clock_family: str,
    path_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[str, str, float, float, str, str]:
    return (
        clock_family,
        path_family,
        -1.0 if clock_beta is None else round(float(clock_beta), 10),
        -1.0 if signal_scale_sq is None else round(float(signal_scale_sq), 10),
        str(device),
        str(dtype),
    )


def _build_clock_importance_cdf(
    clock_family: str,
    path_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    cache_key = _clock_importance_cache_key(
        clock_family=clock_family,
        path_family=path_family,
        clock_beta=clock_beta,
        signal_scale_sq=signal_scale_sq,
        device=device,
        dtype=dtype,
    )
    cached = _CLOCK_IMPORTANCE_CDF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    r_grid = torch.linspace(
        TIME_EPS,
        1.0 - TIME_EPS,
        FT_CLOCK_GRID_SIZE,
        device=device,
        dtype=dtype,
    )
    clock = evaluate_clock(
        r=r_grid,
        clock_family=clock_family,
        clock_beta=clock_beta,
        path_family=path_family,
        signal_scale_sq=signal_scale_sq,
    )
    density = clock.ds_dr.square().clamp(min=EPS)
    grid_step = (r_grid[-1] - r_grid[0]) / (r_grid.numel() - 1)
    increments = 0.5 * (density[1:] + density[:-1]) * grid_step
    cdf = torch.zeros_like(r_grid)
    cdf[1:] = torch.cumsum(increments, dim=0)
    cdf = cdf / cdf[-1].clamp(min=EPS)
    cdf[-1] = 1.0

    _CLOCK_IMPORTANCE_CDF_CACHE[cache_key] = (r_grid, cdf)
    return r_grid, cdf


def _evaluate_ft_clock_closed_form(
    r: Tensor,
    path_family: str,
    beta: float,
    signal_scale_sq: Optional[float],
) -> Optional[ClockOutput]:
    one_minus_r = _safe_one_minus(r)

    if path_family == "linear":
        exponent = 1.0 / (2.0 * (1.0 - beta))
        s = 1.0 - one_minus_r.pow(exponent)
        ds_dr = exponent * one_minus_r.pow(exponent - 1.0)
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    if (
        path_family == "trig_vp"
        and signal_scale_sq is not None
        and abs(float(signal_scale_sq) - 1.0) <= FT_TRIG_VP_UNIT_SCALE_TOL
    ):
        exponent = 1.0 / (1.0 - beta)
        inner = 1.0 - one_minus_r.pow(exponent)
        inner = inner.clamp(min=-1.0 + EPS, max=1.0 - EPS)
        d_inner = exponent * one_minus_r.pow(exponent - 1.0)
        s = (2.0 / torch.pi) * torch.asin(inner)
        ds_dr = (2.0 / torch.pi) * d_inner / torch.sqrt((1.0 - inner * inner).clamp(min=EPS))
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    return None


def _evaluate_ft_clock_numeric(
    r: Tensor,
    path_family: str,
    beta: float,
    signal_scale_sq: float,
) -> ClockOutput:
    s_grid, r_grid = _build_ft_clock_grid(
        path_family=path_family,
        beta=beta,
        signal_scale_sq=signal_scale_sq,
        device=r.device,
        dtype=r.dtype,
    )
    s = _interpolate_monotone_inverse(r=r, s_grid=s_grid, r_grid=r_grid)
    path = evaluate_path(s=s, path_family=path_family)
    mean_error = evaluate_mean_terminal_error(path=path, signal_scale_sq=signal_scale_sq)
    mean_error_0 = torch.tensor(signal_scale_sq + 1.0, dtype=r.dtype, device=r.device)
    mean_error_prime = evaluate_mean_terminal_error_derivative(
        path=path,
        signal_scale_sq=signal_scale_sq,
    )
    ds_dr = (
        -torch.pow(mean_error_0, 1.0 - beta)
        * torch.pow(mean_error.clamp(min=EPS), beta)
        / ((1.0 - beta) * mean_error_prime.clamp(max=-EPS))
    )
    return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)


def estimate_signal_scale_sq_from_dataset(
    dataset,
    max_samples: int = 4096,
) -> float:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive. Got {max_samples}.")

    sample_count = min(len(dataset), max_samples)
    total_squared = 0.0
    total_entries = 0
    for index in range(sample_count):
        sample, _ = dataset[index]
        sample = sample.to(torch.float32) * 2.0 - 1.0
        total_squared += float(sample.square().sum().item())
        total_entries += sample.numel()

    if total_entries == 0:
        raise ValueError("Unable to estimate signal_scale_sq from an empty dataset.")
    return total_squared / total_entries


def evaluate_clock(
    r: Tensor,
    clock_family: str,
    clock_beta: Optional[float] = None,
    path_family: Optional[str] = None,
    signal_scale_sq: Optional[float] = None,
) -> ClockOutput:
    if clock_family not in CLOCK_FAMILIES:
        raise ValueError(f"Unsupported clock_family={clock_family}.")

    beta = _validate_beta(clock_family=clock_family, clock_beta=clock_beta)

    if clock_family == "uniform":
        return ClockOutput(s=r, ds_dr=torch.ones_like(r))

    one_minus_r = _safe_one_minus(r)

    if clock_family == "ft_beta":
        if path_family is None:
            raise ValueError("clock_family=ft_beta requires path_family.")
        closed_form = _evaluate_ft_clock_closed_form(
            r=r,
            path_family=path_family,
            beta=beta,
            signal_scale_sq=signal_scale_sq,
        )
        if closed_form is not None:
            return closed_form
        if signal_scale_sq is None:
            raise ValueError(
                "clock_family=ft_beta requires signal_scale_sq when the closed form is unavailable."
            )
        return _evaluate_ft_clock_numeric(
            r=r,
            path_family=path_family,
            beta=beta,
            signal_scale_sq=signal_scale_sq,
        )

    if clock_family == "poly_a0.5":
        exponent = 0.5
        s = 1.0 - one_minus_r.pow(exponent)
        ds_dr = exponent * one_minus_r.pow(exponent - 1.0)
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    if clock_family == "poly_a2.0":
        exponent = 2.0
        s = 1.0 - one_minus_r.pow(exponent)
        ds_dr = exponent * one_minus_r.pow(exponent - 1.0)
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    if clock_family == "cosine":
        s = 1.0 - torch.cos(torch.pi * r / 2.0)
        ds_dr = (torch.pi / 2.0) * torch.sin(torch.pi * r / 2.0)
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    if clock_family == "sigmoid_k8":
        return _normalized_sigmoid(r=r, k=8.0)

    if clock_family == "exp_l3":
        return _normalized_exponential(r=r, lamb=3.0)

    raise AssertionError(f"Unhandled clock_family={clock_family}.")


def evaluate_path(s: Tensor, path_family: str) -> PathOutput:
    if path_family == "linear":
        return PathOutput(
            alpha=s,
            sigma=1.0 - s,
            d_alpha=torch.ones_like(s),
            d_sigma=-torch.ones_like(s),
        )

    if path_family == "trig_vp":
        angle = torch.pi * s / 2.0
        return PathOutput(
            alpha=torch.sin(angle),
            sigma=torch.cos(angle),
            d_alpha=(torch.pi / 2.0) * torch.cos(angle),
            d_sigma=-(torch.pi / 2.0) * torch.sin(angle),
        )

    raise ValueError(f"Unsupported path_family={path_family}.")


def expand_like(time_tensor: Tensor, reference: Tensor) -> Tensor:
    view_shape = [time_tensor.shape[0]] + [1] * (reference.dim() - 1)
    return time_tensor.view(view_shape)


def clamp_time_inside_unit_interval(r: Tensor) -> Tensor:
    return r.clamp(min=TIME_EPS, max=1.0 - TIME_EPS)


def normalize_model_output_type(model_output_type: Optional[str]) -> str:
    resolved = "velocity" if model_output_type is None else str(model_output_type)
    if resolved not in MODEL_OUTPUT_TYPES:
        raise ValueError(
            f"Unsupported model_output_type={model_output_type}. "
            f"Expected one of {MODEL_OUTPUT_TYPES}."
        )
    return resolved


def model_output_to_velocity(
    model_output: Tensor,
    ds_dr: Tensor,
    model_output_type: Optional[str],
) -> Tensor:
    resolved = normalize_model_output_type(model_output_type)
    if resolved == "velocity":
        return model_output
    return expand_like(ds_dr, model_output) * model_output


def model_output_to_base_velocity(
    model_output: Tensor,
    ds_dr: Tensor,
    model_output_type: Optional[str],
) -> Tensor:
    resolved = normalize_model_output_type(model_output_type)
    if resolved == "base_velocity":
        return model_output
    return model_output / expand_like(ds_dr, model_output).clamp(min=EPS)


def sample_strict_unit_interval(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r = torch.rand(batch_size, device=device, dtype=dtype)
    return clamp_time_inside_unit_interval(r * (1.0 - 2.0 * TIME_EPS) + TIME_EPS)


def sample_importance_weighted_time(
    batch_size: int,
    device: torch.device,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if clock_family == "uniform":
        return sample_strict_unit_interval(batch_size=batch_size, device=device, dtype=dtype)

    r_grid, cdf = _build_clock_importance_cdf(
        clock_family=clock_family,
        path_family=path_family,
        clock_beta=clock_beta,
        signal_scale_sq=signal_scale_sq,
        device=device,
        dtype=dtype,
    )
    uniform_samples = sample_strict_unit_interval(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    sampled_r = _interpolate_monotone_inverse(
        r=uniform_samples,
        s_grid=r_grid,
        r_grid=cdf,
    )
    return clamp_time_inside_unit_interval(sampled_r)


def build_continuous_batch(
    x_1: Tensor,
    x_0: Tensor,
    r: Tensor,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
) -> ContinuousPathBatch:
    clock = evaluate_clock(
        r=r,
        clock_family=clock_family,
        clock_beta=clock_beta,
        path_family=path_family,
        signal_scale_sq=signal_scale_sq,
    )
    path = evaluate_path(s=clock.s, path_family=path_family)

    alpha = expand_like(path.alpha, x_1)
    sigma = expand_like(path.sigma, x_1)
    d_alpha = expand_like(path.d_alpha, x_1)
    d_sigma = expand_like(path.d_sigma, x_1)
    ds_dr = expand_like(clock.ds_dr, x_1)

    x_t = sigma * x_0 + alpha * x_1
    base_velocity = d_sigma * x_0 + d_alpha * x_1
    target_velocity = ds_dr * base_velocity

    return ContinuousPathBatch(
        r=r,
        s=clock.s,
        ds_dr=clock.ds_dr,
        x_t=x_t,
        base_velocity=base_velocity,
        target_velocity=target_velocity,
    )
