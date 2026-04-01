import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from training.continuous_runtime import expand_like
from training.solver_aware.monitors import (
    _cycle_loader,
    _jvp,
    _make_generator,
    _path_sample,
    _prepare_reference_batch,
    _resolve_monitor_microbatch_size,
    _velocity_fn,
)


logger = logging.getLogger(__name__)
PROPAGATION_MICROBATCH = 2
SPECTRAL_EPS = 1e-8
MAX_EXPONENT = 80.0


@dataclass
class PropagationArtifacts:
    s_grid: Tensor
    raw_ell: Tensor
    env_ell: Tensor
    ell_values: Tensor
    g_values: Tensor
    resolved_estimator: str
    theorem_backed: bool
    notes: str
    power_iters: int
    pool_radius: int
    safety_factor: float

    def to_dict(self):
        payload = {
            "s_grid": self.s_grid.detach().cpu(),
            "raw_ell": self.raw_ell.detach().cpu(),
            "env_ell": self.env_ell.detach().cpu(),
            "ell_values": self.ell_values.detach().cpu(),
            "g_values": self.g_values.detach().cpu(),
            "resolved_estimator": self.resolved_estimator,
            "theorem_backed": self.theorem_backed,
            "notes": self.notes,
            "power_iters": int(self.power_iters),
            "pool_radius": int(self.pool_radius),
            "safety_factor": float(self.safety_factor),
        }
        return payload


def _vjp(function, inputs, cotangents):
    if hasattr(torch, "func") and hasattr(torch.func, "vjp"):
        _, vjp_fn = torch.func.vjp(function, inputs)
        return None, vjp_fn(cotangents)[0]
    return torch.autograd.functional.vjp(function, inputs, v=cotangents)


def _safe_norm(values: Tensor) -> Tensor:
    return values.flatten(start_dim=1).pow(2).sum(dim=1).sqrt().clamp(min=SPECTRAL_EPS)


def _spectral_norm_power_iteration(
    velocity_model,
    x: Tensor,
    s: Tensor,
    labels: Tensor,
    cfg_scale: float,
    power_iters: int,
) -> Tensor:
    def wrapped(x_input: Tensor) -> Tensor:
        return _velocity_fn(
            velocity_model=velocity_model,
            x=x_input,
            s=s,
            labels=labels,
            cfg_scale=cfg_scale,
        )

    v = torch.randn_like(x)
    v = v / expand_like(_safe_norm(v), v)

    for _ in range(max(1, int(power_iters))):
        _, jv = _jvp(wrapped, (x,), (v,))
        u = jv / expand_like(_safe_norm(jv), jv)
        _, jtv = _vjp(wrapped, x, u)
        v = jtv / expand_like(_safe_norm(jtv), jtv)
        del jv, u, jtv

    _, jv = _jvp(wrapped, (x,), (v,))
    sigma = _safe_norm(jv)
    del jv, v
    return sigma


def _max_pool_1d(values: Tensor, radius: int) -> Tensor:
    if int(radius) <= 0 or values.numel() <= 2:
        return values
    kernel_size = int(radius) * 2 + 1
    padded = torch.nn.functional.pad(
        values.view(1, 1, -1),
        (int(radius), int(radius)),
        mode="replicate",
    )
    pooled = torch.nn.functional.max_pool1d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    )
    return pooled.view(-1)


def build_propagation_factor(
    s_grid: Tensor,
    ell_values: Tensor,
) -> Tensor:
    """Build G(s)=exp(int_s^1 ell(t)dt) on a monotone s-grid.

    We use an empirical right-endpoint Riemann sum on the pooled envelope
    ell(s). This is a numerical proxy for the tail integral, not a strict
    conservative bound, because ell(s) itself is estimated from a finite batch
    and is not guaranteed to be monotone or globally supremizing.
    """
    if s_grid.ndim != 1 or ell_values.ndim != 1:
        raise ValueError("s_grid and ell_values must be one-dimensional tensors.")
    if s_grid.numel() != ell_values.numel():
        raise ValueError("s_grid and ell_values must have the same length.")
    if s_grid.numel() < 2:
        raise ValueError("At least two grid points are required to build G(s).")

    ds = (s_grid[1:] - s_grid[:-1]).to(dtype=torch.float64)
    if torch.any(ds <= 0.0):
        raise ValueError("s_grid must be strictly increasing.")
    increments = ell_values[1:].to(dtype=torch.float64).clamp(min=0.0) * ds
    reverse_tail = torch.flip(torch.cumsum(torch.flip(increments, dims=[0]), dim=0), dims=[0])
    tail_integral = torch.zeros_like(ell_values, dtype=torch.float64)
    tail_integral[:-1] = reverse_tail
    tail_integral[-1] = 0.0
    return torch.exp(tail_integral.clamp(max=MAX_EXPONENT)).to(dtype=s_grid.dtype)


def estimate_jacobian_spectral_envelope(
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    grid_size: int,
    batch_size: int,
    cfg_scale: float,
    seed: int,
    estimator: str,
    power_iters: int,
    pool_radius: int,
    safety_factor: float,
) -> PropagationArtifacts:
    """Estimate ell(s) and G(s) for propagation-aware solver-aware clocks.

    The propagation-aware construction augments the local monitor with
    G(s)=exp(int_s^1 ell(t)dt), where ell(s) upper-bounds ||J_x u(z,s)||_2 along
    the path ensemble. We estimate ||J_x u||_2 by power iteration using JVP/VJP
    products on a fixed reference batch to keep ell(s) continuous across s.
    """
    if float(safety_factor) < 1.0:
        raise ValueError(
            "solver_aware_g_safety_factor must be >= 1.0 to preserve the intended upper-envelope semantics."
        )

    resolved_estimator = str(estimator)
    theorem_backed = False
    loader_iter = _cycle_loader(data_loader)
    noise_generator = _make_generator(device=device, seed=seed + 28081)
    microbatch_size = max(
        1,
        min(
            int(batch_size),
            min(PROPAGATION_MICROBATCH, _resolve_monitor_microbatch_size(batch_size, estimator="jvp")),
        ),
    )
    s_grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=torch.float32)
    raw_ell = torch.zeros_like(s_grid)
    samples, labels, noise = _prepare_reference_batch(
        loader_iter=loader_iter,
        batch_size=batch_size,
        device=device,
        noise_generator=noise_generator,
    )

    for index, s_value in enumerate(s_grid):
        sigma_chunks = []
        for sample_chunk, label_chunk, noise_chunk in zip(
            samples.split(microbatch_size),
            labels.split(microbatch_size),
            noise.split(microbatch_size),
        ):
            s_batch = torch.full(
                (sample_chunk.shape[0],),
                float(s_value.item()),
                device=device,
                dtype=sample_chunk.dtype,
            )
            z_s = _path_sample(
                samples=sample_chunk,
                noise=noise_chunk,
                s=s_batch,
                path_family=path_family,
            )
            sigma_hat = _spectral_norm_power_iteration(
                velocity_model=velocity_model,
                x=z_s,
                s=s_batch,
                labels=label_chunk,
                cfg_scale=cfg_scale,
                power_iters=power_iters,
            )
            sigma_chunks.append(sigma_hat.detach())
            del sigma_hat, z_s, s_batch

        sigma_values = torch.cat(sigma_chunks, dim=0)
        if resolved_estimator == "spectral_q95":
            raw_ell[index] = torch.quantile(sigma_values, q=0.95)
        elif resolved_estimator in {"spectral_max", "spectral_maxpool"}:
            raw_ell[index] = sigma_values.max()
        else:
            raise ValueError(f"Unsupported solver-aware propagation estimator {resolved_estimator}.")

    env_ell = _max_pool_1d(raw_ell.to(dtype=torch.float64), radius=pool_radius).to(dtype=s_grid.dtype)
    ell_values = (float(safety_factor) * env_ell).to(dtype=s_grid.dtype)
    g_values = build_propagation_factor(
        s_grid=s_grid,
        ell_values=ell_values,
    )
    notes = (
        "Propagation-aware Jacobian envelope uses an empirical batchwise max spectral proxy "
        "ell(s)=max_b ||J_x u(z_b,s)||_2 together with pooling and a right-endpoint tail integral. "
        "This is treated as a heuristic propagation proxy rather than a strict upper bound."
        if resolved_estimator in {"spectral_max", "spectral_maxpool"}
        else "Propagation-aware Jacobian envelope uses a q95 spectral summary for smoother ell(s); "
        "this is explicitly treated as heuristic rather than a strict upper bound."
    )
    logger.info(
        "Computed propagation envelope with estimator=%s on %d grid points "
        "(batch_size=%d, microbatch_size=%d, power_iters=%d, fixed_reference_batch=true).",
        resolved_estimator,
        grid_size,
        batch_size,
        microbatch_size,
        power_iters,
    )
    return PropagationArtifacts(
        s_grid=s_grid,
        raw_ell=raw_ell,
        env_ell=env_ell,
        ell_values=ell_values,
        g_values=g_values,
        resolved_estimator=resolved_estimator,
        theorem_backed=theorem_backed,
        notes=notes,
        power_iters=int(power_iters),
        pool_radius=int(pool_radius),
        safety_factor=float(safety_factor),
    )
