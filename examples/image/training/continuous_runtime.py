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
TIME_SAMPLING_STRATEGIES = (
    "uniform",
    "ds_dr_sq",
    "mixed_lambda",
    "stratified",
    "stratified_mixed",
    "curriculum",
)
CURRICULUM_SIGNATURE = "warmup0.3_linear_to1"
CURRICULUM_WARMUP_FRACTION = 0.3
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


def _interpolate_monotone_lookup(
    x: Tensor,
    x_grid: Tensor,
    y_grid: Tensor,
) -> Tensor:
    flat_x = x.reshape(-1).clamp(float(x_grid[0].item()), float(x_grid[-1].item()))
    right_indices = torch.searchsorted(x_grid, flat_x, right=True)
    right_indices = right_indices.clamp(min=1, max=x_grid.numel() - 1)
    left_indices = right_indices - 1

    x_left = x_grid[left_indices]
    x_right = x_grid[right_indices]
    y_left = y_grid[left_indices]
    y_right = y_grid[right_indices]

    weight = (flat_x - x_left) / (x_right - x_left).clamp(min=EPS)
    interpolated = y_left + weight * (y_right - y_left)
    return interpolated.reshape_as(x)


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


def normalize_time_sampling_strategy(time_sampling_strategy: Optional[str]) -> str:
    resolved = "uniform" if time_sampling_strategy is None else str(time_sampling_strategy)
    if resolved not in TIME_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unsupported time_sampling_strategy={time_sampling_strategy}. "
            f"Expected one of {TIME_SAMPLING_STRATEGIES}."
        )
    return resolved


def normalize_model_output_type(model_output_type: Optional[str]) -> str:
    resolved = "velocity" if model_output_type is None else str(model_output_type)
    if resolved not in MODEL_OUTPUT_TYPES:
        raise ValueError(
            f"Unsupported model_output_type={model_output_type}. "
            f"Expected one of {MODEL_OUTPUT_TYPES}."
        )
    return resolved


def validate_strategy_configuration(
    model_output_type: Optional[str],
    time_sampling_strategy: Optional[str],
) -> Tuple[str, str]:
    resolved_model_output = normalize_model_output_type(model_output_type)
    resolved_sampling_strategy = normalize_time_sampling_strategy(time_sampling_strategy)
    if resolved_sampling_strategy == "ds_dr_sq":
        if resolved_model_output != "base_velocity":
            raise ValueError(
                "time_sampling_strategy=ds_dr_sq requires model_output_type=base_velocity."
            )
    elif resolved_model_output != "velocity":
        raise ValueError(
            f"time_sampling_strategy={resolved_sampling_strategy} requires "
            "model_output_type=velocity."
        )
    return resolved_model_output, resolved_sampling_strategy


def infer_strategy_id(
    model_output_type: Optional[str],
    time_sampling_strategy: Optional[str],
) -> str:
    _, resolved_sampling_strategy = validate_strategy_configuration(
        model_output_type=model_output_type,
        time_sampling_strategy=time_sampling_strategy,
    )
    mapping = {
        "uniform": "A",
        "ds_dr_sq": "B",
        "mixed_lambda": "C",
        "stratified": "D",
        "stratified_mixed": "E",
        "curriculum": "F",
    }
    return mapping[resolved_sampling_strategy]


def resolve_curriculum_signature(time_sampling_strategy: Optional[str]) -> str:
    if normalize_time_sampling_strategy(time_sampling_strategy) == "curriculum":
        return CURRICULUM_SIGNATURE
    return ""


def resolve_curriculum_lambda(
    current_epoch: Optional[int],
    total_epochs: Optional[int],
) -> float:
    if current_epoch is None or total_epochs is None or int(total_epochs) <= 1:
        return 1.0
    progress = float(current_epoch) / float(max(int(total_epochs) - 1, 1))
    if progress <= CURRICULUM_WARMUP_FRACTION:
        return 0.0
    return min(
        1.0,
        max(
            0.0,
            (progress - CURRICULUM_WARMUP_FRACTION)
            / (1.0 - CURRICULUM_WARMUP_FRACTION),
        ),
    )


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
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    r = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return clamp_time_inside_unit_interval(r * (1.0 - 2.0 * TIME_EPS) + TIME_EPS)


def _sample_uniform_in_interval(
    batch_size: int,
    left: float,
    right: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tensor:
    if batch_size <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    left = float(left)
    right = float(right)
    if not (0.0 <= left < right <= 1.0):
        raise ValueError(f"Invalid interval [{left}, {right}] for time sampling.")
    samples = sample_strict_unit_interval(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return left + (right - left) * samples


def _sample_from_importance_cdf(
    batch_size: int,
    r_grid: Tensor,
    cdf: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    cdf_left: float = 0.0,
    cdf_right: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if batch_size <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if not (0.0 <= cdf_left < cdf_right <= 1.0):
        raise ValueError(f"Invalid CDF interval [{cdf_left}, {cdf_right}] for importance sampling.")
    uniform_samples = sample_strict_unit_interval(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if cdf_left != 0.0 or cdf_right != 1.0:
        uniform_samples = cdf_left + (cdf_right - cdf_left) * uniform_samples
    sampled_r = _interpolate_monotone_inverse(
        r=uniform_samples,
        s_grid=r_grid,
        r_grid=cdf,
    )
    return clamp_time_inside_unit_interval(sampled_r)


def sample_importance_weighted_time(
    batch_size: int,
    device: torch.device,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if clock_family == "uniform":
        return sample_strict_unit_interval(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    r_grid, cdf = _build_clock_importance_cdf(
        clock_family=clock_family,
        path_family=path_family,
        clock_beta=clock_beta,
        signal_scale_sq=signal_scale_sq,
        device=device,
        dtype=dtype,
    )
    return _sample_from_importance_cdf(
        batch_size=batch_size,
        r_grid=r_grid,
        cdf=cdf,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _sample_importance_weighted_time_in_interval(
    batch_size: int,
    left: float,
    right: float,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tensor:
    if batch_size <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if clock_family == "uniform":
        return _sample_uniform_in_interval(
            batch_size=batch_size,
            left=left,
            right=right,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    r_grid, cdf = _build_clock_importance_cdf(
        clock_family=clock_family,
        path_family=path_family,
        clock_beta=clock_beta,
        signal_scale_sq=signal_scale_sq,
        device=device,
        dtype=dtype,
    )
    interval = torch.tensor([left, right], device=device, dtype=dtype)
    cdf_interval = _interpolate_monotone_lookup(
        x=interval,
        x_grid=r_grid,
        y_grid=cdf,
    )
    cdf_left = float(cdf_interval[0].item())
    cdf_right = float(cdf_interval[1].item())
    if cdf_right <= cdf_left + EPS:
        return _sample_uniform_in_interval(
            batch_size=batch_size,
            left=left,
            right=right,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return _sample_from_importance_cdf(
        batch_size=batch_size,
        r_grid=r_grid,
        cdf=cdf,
        device=device,
        dtype=dtype,
        cdf_left=cdf_left,
        cdf_right=cdf_right,
        generator=generator,
    )


def _sample_mixed_lambda_time(
    batch_size: int,
    device: torch.device,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    mixed_lambda: float,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tensor:
    mixed_lambda = float(min(max(mixed_lambda, 0.0), 1.0))
    if mixed_lambda <= 0.0:
        return sample_strict_unit_interval(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if mixed_lambda >= 1.0:
        return sample_importance_weighted_time(
            batch_size=batch_size,
            device=device,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            dtype=dtype,
            generator=generator,
        )
    uniform_samples = sample_strict_unit_interval(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    importance_samples = sample_importance_weighted_time(
        batch_size=batch_size,
        device=device,
        path_family=path_family,
        clock_family=clock_family,
        clock_beta=clock_beta,
        signal_scale_sq=signal_scale_sq,
        dtype=dtype,
        generator=generator,
    )
    selector = torch.rand(batch_size, device=device, generator=generator) < mixed_lambda
    return torch.where(selector, importance_samples, uniform_samples)


def _sample_stratified_time(
    batch_size: int,
    device: torch.device,
    stratified_bins: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tensor:
    if stratified_bins <= 0:
        raise ValueError(f"stratified_bins must be positive. Got {stratified_bins}.")
    counts = [batch_size // stratified_bins for _ in range(stratified_bins)]
    counts[-1] += batch_size - sum(counts)
    samples = []
    for bin_index, count in enumerate(counts):
        left = bin_index / stratified_bins
        right = (bin_index + 1) / stratified_bins
        samples.append(
            _sample_uniform_in_interval(
                batch_size=count,
                left=left,
                right=right,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    stacked = torch.cat(samples, dim=0)
    return stacked[torch.randperm(stacked.shape[0], device=device, generator=generator)]


def _sample_stratified_mixed_time(
    batch_size: int,
    device: torch.device,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    mixed_lambda: float,
    stratified_bins: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tensor:
    if stratified_bins <= 0:
        raise ValueError(f"stratified_bins must be positive. Got {stratified_bins}.")
    mixed_lambda = float(min(max(mixed_lambda, 0.0), 1.0))
    counts = [batch_size // stratified_bins for _ in range(stratified_bins)]
    counts[-1] += batch_size - sum(counts)
    samples = []
    for bin_index, count in enumerate(counts):
        left = bin_index / stratified_bins
        right = (bin_index + 1) / stratified_bins
        uniform_samples = _sample_uniform_in_interval(
            batch_size=count,
            left=left,
            right=right,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        importance_samples = _sample_importance_weighted_time_in_interval(
            batch_size=count,
            left=left,
            right=right,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        selector = torch.rand(count, device=device, generator=generator) < mixed_lambda
        samples.append(torch.where(selector, importance_samples, uniform_samples))
    stacked = torch.cat(samples, dim=0)
    return stacked[torch.randperm(stacked.shape[0], device=device, generator=generator)]


def sample_time_by_strategy(
    batch_size: int,
    device: torch.device,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    signal_scale_sq: Optional[float],
    strategy: Optional[str],
    mixed_lambda: float = 0.5,
    stratified_bins: int = 16,
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    resolved_strategy = normalize_time_sampling_strategy(strategy)
    if resolved_strategy == "uniform":
        return sample_strict_unit_interval(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if resolved_strategy == "ds_dr_sq":
        return sample_importance_weighted_time(
            batch_size=batch_size,
            device=device,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            dtype=dtype,
            generator=generator,
        )
    if resolved_strategy == "mixed_lambda":
        return _sample_mixed_lambda_time(
            batch_size=batch_size,
            device=device,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            mixed_lambda=mixed_lambda,
            dtype=dtype,
            generator=generator,
        )
    if resolved_strategy == "stratified":
        return _sample_stratified_time(
            batch_size=batch_size,
            device=device,
            stratified_bins=stratified_bins,
            dtype=dtype,
            generator=generator,
        )
    if resolved_strategy == "stratified_mixed":
        return _sample_stratified_mixed_time(
            batch_size=batch_size,
            device=device,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            mixed_lambda=mixed_lambda,
            stratified_bins=stratified_bins,
            dtype=dtype,
            generator=generator,
        )
    if resolved_strategy == "curriculum":
        curriculum_lambda = resolve_curriculum_lambda(
            current_epoch=current_epoch,
            total_epochs=total_epochs,
        )
        return _sample_mixed_lambda_time(
            batch_size=batch_size,
            device=device,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
            signal_scale_sq=signal_scale_sq,
            mixed_lambda=curriculum_lambda,
            dtype=dtype,
            generator=generator,
        )
    raise AssertionError(f"Unhandled time sampling strategy {resolved_strategy}.")


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
