from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


EPS = 1e-12


@dataclass
class SolverAwareClockArtifacts:
    s_grid: Tensor
    q_raw: Tensor
    q_smoothed: Tensor
    density: Tensor
    phi: Tensor
    r_grid: Tensor
    nodes: Tensor
    density_exponent: float
    smoothing_window: int


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


def _strictly_monotone(values: Tensor) -> Tensor:
    monotone = values.clone()
    min_step = max(torch.finfo(monotone.dtype).eps, EPS)
    for index in range(1, monotone.numel()):
        monotone[index] = torch.maximum(
            monotone[index],
            monotone[index - 1] + min_step,
        )
    monotone = monotone - monotone[0]
    monotone = monotone / monotone[-1].clamp(min=min_step)
    monotone[0] = 0.0
    monotone[-1] = 1.0
    return monotone


def monotone_inverse_lookup(
    x_grid: Tensor,
    y_grid: Tensor,
    query: Tensor,
) -> Tensor:
    flat_query = query.reshape(-1).clamp(float(y_grid[0].item()), float(y_grid[-1].item()))
    right_indices = torch.searchsorted(y_grid, flat_query, right=True)
    right_indices = right_indices.clamp(min=1, max=y_grid.numel() - 1)
    left_indices = right_indices - 1

    y_left = y_grid[left_indices]
    y_right = y_grid[right_indices]
    x_left = x_grid[left_indices]
    x_right = x_grid[right_indices]
    weight = (flat_query - y_left) / (y_right - y_left).clamp(min=EPS)
    values = x_left + weight * (x_right - x_left)
    return values.reshape_as(query)


def build_solver_aware_clock(
    s_grid: Tensor,
    q_values: Tensor,
    density_exponent: float,
    eps: float,
    node_count: int,
    smoothing_window: Optional[int] = None,
) -> SolverAwareClockArtifacts:
    """Build a monotone solver-aware clock from a squared monitor curve Q(s).

    For a solver with leading local error order p+1, the phase-1 density uses
    m(s) = (Q(s) + eps)^gamma, where gamma = 1/4 for Euler and 1/6 for Heun2.
    The cumulative clock is phi(s) = int_0^s m(u) du / int_0^1 m(u) du and the
    sampling nodes are s_n = psi(n / N) with psi = phi^{-1}.
    """
    if s_grid.ndim != 1 or q_values.ndim != 1:
        raise ValueError("s_grid and q_values must be one-dimensional tensors.")
    if s_grid.numel() != q_values.numel():
        raise ValueError("s_grid and q_values must have the same length.")
    if s_grid.numel() < 2:
        raise ValueError("At least two grid points are required to build a solver-aware clock.")
    if node_count < 2:
        raise ValueError("node_count must be at least 2.")

    smoothing_window = (
        max(3, int(s_grid.numel() // 16) * 2 + 1)
        if smoothing_window is None
        else max(1, int(smoothing_window))
    )
    q_raw = q_values.to(dtype=torch.float64)
    q_smoothed = _moving_average(q_raw.clamp(min=0.0), window=smoothing_window)
    density = torch.pow(q_smoothed + float(eps), float(density_exponent))

    ds = s_grid[1:] - s_grid[:-1]
    if torch.any(ds <= 0.0):
        raise ValueError("s_grid must be strictly increasing.")
    # phi(s) is a normalized cumulative integral of the positive density m(s).
    trapezoids = 0.5 * (density[1:] + density[:-1]) * ds.to(dtype=density.dtype)
    phi = torch.zeros_like(s_grid, dtype=density.dtype)
    phi[1:] = torch.cumsum(trapezoids, dim=0)
    phi = phi / phi[-1].clamp(min=EPS)
    phi = _strictly_monotone(phi)

    r_grid = torch.linspace(
        0.0,
        1.0,
        node_count,
        device=s_grid.device,
        dtype=s_grid.dtype,
    )
    nodes = monotone_inverse_lookup(
        x_grid=s_grid.to(dtype=phi.dtype),
        y_grid=phi,
        query=r_grid.to(dtype=phi.dtype),
    ).to(dtype=s_grid.dtype)
    nodes[0] = 0.0
    nodes[-1] = 1.0

    return SolverAwareClockArtifacts(
        s_grid=s_grid,
        q_raw=q_raw.to(dtype=s_grid.dtype),
        q_smoothed=q_smoothed.to(dtype=s_grid.dtype),
        density=density.to(dtype=s_grid.dtype),
        phi=phi.to(dtype=s_grid.dtype),
        r_grid=r_grid,
        nodes=nodes,
        density_exponent=float(density_exponent),
        smoothing_window=int(smoothing_window),
    )
