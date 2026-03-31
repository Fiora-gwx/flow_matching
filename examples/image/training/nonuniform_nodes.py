from typing import Optional

import torch
from torch import Tensor


def build_uniform_time_grid(
    step_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive. Got {step_count}.")
    return torch.linspace(0.0, 1.0, step_count + 1, device=device, dtype=dtype)


def validate_time_grid(time_grid: Tensor, step_count: Optional[int] = None) -> Tensor:
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional.")
    if time_grid.numel() < 2:
        raise ValueError("time_grid must contain at least two nodes.")
    if step_count is not None and time_grid.numel() != int(step_count) + 1:
        raise ValueError(
            f"time_grid length mismatch. expected={int(step_count) + 1}, got={time_grid.numel()}."
        )
    deltas = time_grid[1:] - time_grid[:-1]
    if torch.any(deltas <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    if abs(float(time_grid[0].item())) > 1e-8 or abs(float(time_grid[-1].item()) - 1.0) > 1e-8:
        raise ValueError("time_grid must start at 0 and end at 1.")
    return time_grid


def resolve_time_grid(
    step_count: int,
    device: torch.device,
    dtype: torch.dtype,
    time_grid: Optional[Tensor] = None,
) -> Tensor:
    if time_grid is None:
        return build_uniform_time_grid(step_count=step_count, device=device, dtype=dtype)
    resolved = time_grid.to(device=device, dtype=dtype)
    return validate_time_grid(resolved, step_count=step_count)
