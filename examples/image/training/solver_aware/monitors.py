import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import torch
from torch import Tensor

from training.continuous_runtime import evaluate_path, expand_like


logger = logging.getLogger(__name__)
FD_DELTA_FLOOR = 1e-3
FD_DELTA_SCALE = 0.25


@dataclass
class MonitorArtifacts:
    s_grid: Tensor
    q_values: Tensor
    resolved_estimator: str
    monitor_name: str
    density_exponent: float
    theorem_backed: bool
    notes: str


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _cycle_loader(data_loader: Iterable) -> Iterator[Tuple[Tensor, Tensor]]:
    iterator = iter(data_loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(data_loader)


def _take_monitor_batch(
    loader_iter: Iterator[Tuple[Tensor, Tensor]],
    batch_size: int,
) -> Tuple[Tensor, Tensor]:
    sample_chunks = []
    label_chunks = []
    total = 0
    while total < batch_size:
        samples, labels = next(loader_iter)
        remaining = batch_size - total
        take = min(int(samples.shape[0]), remaining)
        sample_chunks.append(samples[:take])
        label_chunks.append(labels[:take])
        total += take
    return torch.cat(sample_chunks, dim=0), torch.cat(label_chunks, dim=0)


def _path_sample(
    samples: Tensor,
    noise: Tensor,
    s: Tensor,
    path_family: str,
) -> Tensor:
    path = evaluate_path(s=s, path_family=path_family)
    alpha = expand_like(path.alpha, samples)
    sigma = expand_like(path.sigma, samples)
    return sigma * noise + alpha * samples


def _resolve_estimator(target_solver: str, estimator: str) -> str:
    requested = str(estimator or "auto")
    if requested != "auto":
        return requested
    return "jvp" if target_solver == "euler" else "fd"


def _fd_step(s: Tensor, grid_size: int) -> Tensor:
    base_delta = max(FD_DELTA_FLOOR, FD_DELTA_SCALE / max(2, int(grid_size) - 1))
    forward_room = 1.0 - s
    backward_room = s
    use_forward = forward_room >= backward_room
    delta = torch.full_like(s, base_delta)
    delta = torch.minimum(delta, torch.where(use_forward, forward_room, backward_room))
    delta = torch.where(use_forward, delta, -delta)
    zero_delta = torch.zeros_like(delta)
    delta = torch.where(delta == 0.0, zero_delta + base_delta, delta)
    return delta


def _velocity_fn(velocity_model, x: Tensor, s: Tensor, labels: Tensor, cfg_scale: float) -> Tensor:
    return velocity_model(x, s, cfg_scale=cfg_scale, label=labels)


def _jvp(function, inputs, tangents):
    if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
        return torch.func.jvp(function, inputs, tangents)
    return torch.autograd.functional.jvp(function, inputs, tangents)


def _material_derivative_fd(
    velocity_model,
    x: Tensor,
    s: Tensor,
    labels: Tensor,
    cfg_scale: float,
    grid_size: int,
) -> Tensor:
    u = _velocity_fn(velocity_model, x, s, labels, cfg_scale)
    delta = _fd_step(s=s, grid_size=grid_size)
    s_shift = (s + delta).clamp(0.0, 1.0)
    x_shift = x + expand_like(delta, x) * u
    u_shift = _velocity_fn(velocity_model, x_shift, s_shift, labels, cfg_scale)
    delta_expand = expand_like(s_shift - s, x)
    delta_safe = torch.where(
        delta_expand >= 0.0,
        delta_expand.clamp(min=FD_DELTA_FLOOR),
        delta_expand.clamp(max=-FD_DELTA_FLOOR),
    )
    return (u_shift - u) / delta_safe


def _material_derivative_jvp(
    velocity_model,
    x: Tensor,
    s: Tensor,
    labels: Tensor,
    cfg_scale: float,
) -> Tensor:
    u = _velocity_fn(velocity_model, x, s, labels, cfg_scale)

    def wrapped(x_input: Tensor, s_input: Tensor) -> Tensor:
        return _velocity_fn(velocity_model, x_input, s_input, labels, cfg_scale)

    _, derivative = _jvp(
        wrapped,
        (x, s),
        (u, torch.ones_like(s)),
    )
    return derivative


def _material_derivative(
    velocity_model,
    x: Tensor,
    s: Tensor,
    labels: Tensor,
    cfg_scale: float,
    estimator: str,
    grid_size: int,
) -> Tensor:
    if estimator == "fd":
        return _material_derivative_fd(
            velocity_model=velocity_model,
            x=x,
            s=s,
            labels=labels,
            cfg_scale=cfg_scale,
            grid_size=grid_size,
        )
    if estimator == "jvp":
        return _material_derivative_jvp(
            velocity_model=velocity_model,
            x=x,
            s=s,
            labels=labels,
            cfg_scale=cfg_scale,
        )
    raise ValueError(f"Unsupported monitor estimator {estimator}.")


def compute_euler_monitor(
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    grid_size: int,
    batch_size: int,
    estimator: str,
    cfg_scale: float,
    seed: int,
) -> MonitorArtifacts:
    """Estimate the Euler monitor Q_E(s) = E||L_u u(z, s)||^2 on a path sample grid.

    The material derivative is L_u = partial_s + u · grad_x. Euler's local
    truncation error is controlled by L_u u, so the optimal error-proxy density
    satisfies rho_E(s) propto (Q_E(s) + eps)^(1/4).
    """
    resolved_estimator = _resolve_estimator(target_solver="euler", estimator=estimator)
    loader_iter = _cycle_loader(data_loader)
    noise_generator = _make_generator(device=device, seed=seed + 9103)
    s_grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=torch.float32)
    q_values = torch.zeros_like(s_grid)

    for index, s_value in enumerate(s_grid):
        samples, labels = _take_monitor_batch(loader_iter, batch_size=batch_size)
        samples = samples.to(device=device, dtype=torch.float32, non_blocking=True) * 2.0 - 1.0
        labels = labels.to(device=device, non_blocking=True)
        noise = torch.randn(
            samples.shape,
            device=device,
            dtype=samples.dtype,
            generator=noise_generator,
        )
        s_batch = torch.full((samples.shape[0],), float(s_value.item()), device=device, dtype=samples.dtype)
        z_s = _path_sample(samples=samples, noise=noise, s=s_batch, path_family=path_family)
        derivative = _material_derivative(
            velocity_model=velocity_model,
            x=z_s,
            s=s_batch,
            labels=labels,
            cfg_scale=cfg_scale,
            estimator=resolved_estimator,
            grid_size=grid_size,
        )
        squared_norm = derivative.flatten(start_dim=1).pow(2).sum(dim=1)
        q_values[index] = squared_norm.mean()

    logger.info(
        "Computed Euler solver-aware monitor with estimator=%s on %d grid points.",
        resolved_estimator,
        grid_size,
    )
    return MonitorArtifacts(
        s_grid=s_grid,
        q_values=q_values,
        resolved_estimator=resolved_estimator,
        monitor_name="euler_lu_u",
        density_exponent=0.25,
        theorem_backed=True,
        notes=(
            "Euler local truncation error is controlled by L_u u, so the monitor uses "
            "Q_E(s) = E||L_u u||^2 and rho_E(s) propto (Q_E(s)+eps)^(1/4)."
        ),
    )


def compute_heun2_monitor(
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    grid_size: int,
    batch_size: int,
    estimator: str,
    cfg_scale: float,
    seed: int,
) -> MonitorArtifacts:
    """Estimate the Heun2 monitor Q_H(s) = E||L_u^2 u(z, s)||^2 on a path sample grid.

    Heun2 / explicit trapezoid has leading local truncation error controlled by
    L_u^2 u. The phase-1 density therefore uses rho_H(s) propto (Q_H(s) + eps)^(1/6).
    The default estimator follows the requested phase-1 design:
    - auto/fd: along-flow finite differences.
    - jvp: nested JVP evaluation of L_u(L_u u).
    """
    resolved_estimator = _resolve_estimator(target_solver="heun2", estimator=estimator)
    loader_iter = _cycle_loader(data_loader)
    noise_generator = _make_generator(device=device, seed=seed + 17021)
    s_grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=torch.float32)
    q_values = torch.zeros_like(s_grid)

    for index, s_value in enumerate(s_grid):
        samples, labels = _take_monitor_batch(loader_iter, batch_size=batch_size)
        samples = samples.to(device=device, dtype=torch.float32, non_blocking=True) * 2.0 - 1.0
        labels = labels.to(device=device, non_blocking=True)
        noise = torch.randn(
            samples.shape,
            device=device,
            dtype=samples.dtype,
            generator=noise_generator,
        )
        s_batch = torch.full((samples.shape[0],), float(s_value.item()), device=device, dtype=samples.dtype)
        z_s = _path_sample(samples=samples, noise=noise, s=s_batch, path_family=path_family)

        if resolved_estimator == "jvp":
            def a_fn(x_input: Tensor, s_input: Tensor) -> Tensor:
                return _material_derivative_jvp(
                    velocity_model=velocity_model,
                    x=x_input,
                    s=s_input,
                    labels=labels,
                    cfg_scale=cfg_scale,
                )

            u = _velocity_fn(velocity_model, z_s, s_batch, labels, cfg_scale)
            _, second_derivative = _jvp(
                a_fn,
                (z_s, s_batch),
                (u, torch.ones_like(s_batch)),
            )
        else:
            u = _velocity_fn(velocity_model, z_s, s_batch, labels, cfg_scale)
            first_derivative = _material_derivative_fd(
                velocity_model=velocity_model,
                x=z_s,
                s=s_batch,
                labels=labels,
                cfg_scale=cfg_scale,
                grid_size=grid_size,
            )
            delta = _fd_step(s=s_batch, grid_size=grid_size)
            s_shift = (s_batch + delta).clamp(0.0, 1.0)
            z_shift = z_s + expand_like(delta, z_s) * u
            shifted_derivative = _material_derivative_fd(
                velocity_model=velocity_model,
                x=z_shift,
                s=s_shift,
                labels=labels,
                cfg_scale=cfg_scale,
                grid_size=grid_size,
            )
            second_derivative = (
                shifted_derivative - first_derivative
            ) / torch.where(
                expand_like(s_shift - s_batch, z_s) >= 0.0,
                expand_like(s_shift - s_batch, z_s).clamp(min=FD_DELTA_FLOOR),
                expand_like(s_shift - s_batch, z_s).clamp(max=-FD_DELTA_FLOOR),
            )

        squared_norm = second_derivative.flatten(start_dim=1).pow(2).sum(dim=1)
        q_values[index] = squared_norm.mean()

    logger.info(
        "Computed Heun2 solver-aware monitor with estimator=%s on %d grid points.",
        resolved_estimator,
        grid_size,
    )
    return MonitorArtifacts(
        s_grid=s_grid,
        q_values=q_values,
        resolved_estimator=resolved_estimator,
        monitor_name="heun2_lu2_u",
        density_exponent=1.0 / 6.0,
        theorem_backed=True,
        notes=(
            "Heun2 local truncation error is controlled by L_u^2 u, so the monitor uses "
            "Q_H(s) = E||L_u^2 u||^2 and rho_H(s) propto (Q_H(s)+eps)^(1/6)."
        ),
    )
