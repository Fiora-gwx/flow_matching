from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

EPS = 1e-8
TIME_EPS = 1e-5
PATH_FAMILIES = ("linear", "trig_vp")
CLOCK_FAMILIES = (
    "uniform",
    "ft_linear_beta",
    "ft_vp_beta",
    "poly_a0.5",
    "poly_a2.0",
    "cosine",
    "sigmoid_k8",
    "exp_l3",
)


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


def _validate_beta(clock_family: str, clock_beta: Optional[float]) -> float:
    if clock_family in {"ft_linear_beta", "ft_vp_beta"}:
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


def evaluate_clock(
    r: Tensor, clock_family: str, clock_beta: Optional[float] = None
) -> ClockOutput:
    if clock_family not in CLOCK_FAMILIES:
        raise ValueError(f"Unsupported clock_family={clock_family}.")

    beta = _validate_beta(clock_family=clock_family, clock_beta=clock_beta)
    one_minus_r = _safe_one_minus(r)

    if clock_family == "uniform":
        return ClockOutput(s=r, ds_dr=torch.ones_like(r))

    if clock_family == "ft_linear_beta":
        exponent = 1.0 / (2.0 * (1.0 - beta))
        s = 1.0 - one_minus_r.pow(exponent)
        ds_dr = exponent * one_minus_r.pow(exponent - 1.0)
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

    if clock_family == "ft_vp_beta":
        exponent = 1.0 / (1.0 - beta)
        inner = 1.0 - one_minus_r.pow(exponent)
        inner = inner.clamp(min=-1.0 + EPS, max=1.0 - EPS)
        d_inner = exponent * one_minus_r.pow(exponent - 1.0)
        s = (2.0 / torch.pi) * torch.asin(inner)
        ds_dr = (2.0 / torch.pi) * d_inner / torch.sqrt((1.0 - inner * inner).clamp(min=EPS))
        return _with_exact_endpoints(r=r, s=s, ds_dr=ds_dr)

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


def sample_strict_unit_interval(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r = torch.rand(batch_size, device=device, dtype=dtype)
    return r * (1.0 - 2.0 * TIME_EPS) + TIME_EPS


def build_continuous_batch(
    x_1: Tensor,
    x_0: Tensor,
    r: Tensor,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
) -> ContinuousPathBatch:
    clock = evaluate_clock(r=r, clock_family=clock_family, clock_beta=clock_beta)
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
