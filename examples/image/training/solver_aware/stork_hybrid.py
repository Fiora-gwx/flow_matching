from typing import Dict, Tuple

import torch
from torch import Tensor

from training.solver_aware.clock import monotone_inverse_lookup
from training.solver_aware.monitors import _path_sample
from training.stork_solver import STORKState, stork4_step


EPS = 1e-12


def stork_cold_start_threshold(step_count: int) -> float:
    resolved_step_count = max(1, int(step_count))
    return 1.0 / float(resolved_step_count)


def build_stork_hybrid_metadata(
    *,
    step_count: int,
    stork_context: bool,
    hybrid_used: bool,
    warm_defect: bool,
    warm_state_heuristic: bool,
) -> Dict[str, object]:
    if not stork_context:
        return {
            "stork_hybrid_clock": False,
            "cold_start_threshold": 0.0,
            "warm_region_start": 0.0,
            "warm_region_enabled": False,
            "warm_macro_step_count": 0,
            "cold_start_fixed_step": False,
            "stork_warm_defect": False,
            "stork_warm_state_heuristic": False,
        }

    resolved_step_count = max(1, int(step_count))
    cold_threshold = stork_cold_start_threshold(resolved_step_count)
    warm_macro_step_count = max(0, resolved_step_count - 1)
    warm_region_enabled = resolved_step_count >= 3
    return {
        "stork_hybrid_clock": bool(hybrid_used and warm_region_enabled),
        "cold_start_threshold": float(cold_threshold),
        "warm_region_start": float(cold_threshold),
        "warm_region_enabled": bool(warm_region_enabled),
        "warm_macro_step_count": int(warm_macro_step_count),
        "cold_start_fixed_step": True,
        "stork_warm_defect": bool(warm_defect),
        "stork_warm_state_heuristic": bool(warm_state_heuristic),
    }


def _interpolate_monotone(x_grid: Tensor, y_grid: Tensor, x_value: float) -> Tensor:
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional.")
    if x_grid.numel() != y_grid.numel():
        raise ValueError("x_grid and y_grid must have the same length.")
    query = torch.as_tensor(
        float(x_value),
        device=x_grid.device,
        dtype=x_grid.dtype,
    ).clamp(float(x_grid[0].item()), float(x_grid[-1].item()))
    right_index = int(torch.searchsorted(x_grid, query, right=True).item())
    right_index = max(1, min(right_index, x_grid.numel() - 1))
    left_index = right_index - 1
    x_left = x_grid[left_index]
    x_right = x_grid[right_index]
    y_left = y_grid[left_index]
    y_right = y_grid[right_index]
    weight = (query - x_left) / (x_right - x_left).clamp(min=EPS)
    return y_left + weight * (y_right - y_left)


def build_stork_hybrid_nodes(
    *,
    s_grid: Tensor,
    phi: Tensor,
    step_count: int,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    resolved_step_count = max(1, int(step_count))
    r_grid = torch.linspace(
        0.0,
        1.0,
        resolved_step_count + 1,
        device=s_grid.device,
        dtype=s_grid.dtype,
    )
    metadata = build_stork_hybrid_metadata(
        step_count=resolved_step_count,
        stork_context=True,
        hybrid_used=resolved_step_count >= 3,
        warm_defect=False,
        warm_state_heuristic=False,
    )
    if resolved_step_count <= 2:
        nodes = r_grid.clone()
        nodes[0] = 0.0
        nodes[-1] = 1.0
        return r_grid, nodes, metadata

    cold_threshold = float(metadata["cold_start_threshold"])
    phi_cold = _interpolate_monotone(
        x_grid=s_grid.to(dtype=torch.float64),
        y_grid=phi.to(dtype=torch.float64),
        x_value=cold_threshold,
    )
    warm_mask = s_grid >= cold_threshold
    warm_s_grid = s_grid[warm_mask].to(dtype=torch.float64)
    warm_phi_grid = phi[warm_mask].to(dtype=torch.float64)
    if warm_s_grid.numel() == 0:
        nodes = r_grid.clone()
        nodes[0] = 0.0
        nodes[-1] = 1.0
        metadata["stork_hybrid_clock"] = False
        metadata["warm_region_enabled"] = False
        return r_grid, nodes, metadata

    tolerance = max(torch.finfo(warm_s_grid.dtype).eps, EPS)
    if float(torch.abs(warm_s_grid[0] - cold_threshold).item()) > tolerance:
        warm_s_grid = torch.cat(
            [
                torch.as_tensor(
                    [cold_threshold],
                    device=warm_s_grid.device,
                    dtype=warm_s_grid.dtype,
                ),
                warm_s_grid,
            ],
            dim=0,
        )
        warm_phi_grid = torch.cat(
            [
                phi_cold.reshape(1),
                warm_phi_grid,
            ],
            dim=0,
        )
    else:
        warm_s_grid = warm_s_grid.clone()
        warm_phi_grid = warm_phi_grid.clone()
        warm_s_grid[0] = cold_threshold
        warm_phi_grid[0] = phi_cold

    warm_phi = (warm_phi_grid - phi_cold) / (1.0 - phi_cold).clamp(min=EPS)
    warm_phi[0] = 0.0
    warm_phi[-1] = 1.0

    warm_r_grid = torch.linspace(
        0.0,
        1.0,
        resolved_step_count,
        device=s_grid.device,
        dtype=torch.float64,
    )
    warm_nodes = monotone_inverse_lookup(
        x_grid=warm_s_grid,
        y_grid=warm_phi,
        query=warm_r_grid,
    ).to(dtype=s_grid.dtype)

    nodes = torch.empty(
        resolved_step_count + 1,
        device=s_grid.device,
        dtype=s_grid.dtype,
    )
    nodes[0] = 0.0
    nodes[1:] = warm_nodes
    nodes[1] = float(cold_threshold)
    nodes[-1] = 1.0
    return r_grid, nodes, metadata


def _velocity_eval(
    *,
    velocity_model,
    x: Tensor,
    t: Tensor,
    labels: Tensor,
    cfg_scale: float,
    step_size: float,
    step_count: int,
) -> Tensor:
    adapt = getattr(velocity_model, "adapt_solver_time", None)
    adapted_t = t
    if callable(adapt):
        adapted_t = adapt(t=t, step_size=float(step_size), step_count=int(step_count))
    try:
        return velocity_model(
            x,
            adapted_t,
            cfg_scale=cfg_scale,
            label=labels,
            use_autocast=False,
        )
    except TypeError:
        return velocity_model(
            x,
            adapted_t,
            cfg_scale=cfg_scale,
            label=labels,
        )


def _clone_stork_state(state: STORKState) -> STORKState:
    return STORKState(
        step_index=int(state.step_index),
        last_velocity=(
            state.last_velocity.detach().clone()
            if state.last_velocity is not None
            else None
        ),
        last_dt=float(state.last_dt) if state.last_dt is not None else None,
        virtual_stage_count=int(state.virtual_stage_count),
        s=int(state.s),
    )


def make_stork_warm_state(
    *,
    velocity_model,
    z_prev: Tensor,
    s_prev: float,
    labels: Tensor,
    cfg_scale: float,
    last_dt: float,
    step_count_hint: int,
) -> STORKState:
    t_prev = torch.full(
        (z_prev.shape[0],),
        float(s_prev),
        device=z_prev.device,
        dtype=z_prev.dtype,
    )
    u_prev = _velocity_eval(
        velocity_model=velocity_model,
        x=z_prev,
        t=t_prev,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=float(last_dt),
        step_count=int(step_count_hint),
    )
    return STORKState(
        step_index=1,
        last_velocity=u_prev.detach().clone(),
        last_dt=float(last_dt),
    )


def _stork_step_with_state(
    *,
    velocity_model,
    z: Tensor,
    s: float,
    h: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
    state: STORKState,
) -> Tensor:
    t = torch.full(
        (z.shape[0],),
        float(s),
        device=z.device,
        dtype=z.dtype,
    )
    model_output = _velocity_eval(
        velocity_model=velocity_model,
        x=z,
        t=t,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=float(h),
        step_count=int(step_count_hint),
    )
    return stork4_step(
        model_output=model_output,
        sample=z,
        t_start=float(s),
        t_end=float(s + h),
        state=state,
    )


def maybe_compute_stork_warm_defect(
    *,
    velocity_model,
    samples: Tensor,
    noise: Tensor,
    z_s: Tensor,
    s: float,
    effective_step: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
    defect_subdivide: int,
    path_family: str,
) -> Tuple[Tensor, Tensor]:
    if defect_subdivide != 2:
        raise NotImplementedError(
            "Phase-1 defect monitor currently supports one warm STORK step vs two half steps only."
        )
    s_prev = max(0.0, float(s) - float(effective_step))
    last_dt = float(s) - float(s_prev)
    s_prev_batch = torch.full(
        (samples.shape[0],),
        float(s_prev),
        device=samples.device,
        dtype=samples.dtype,
    )
    z_prev = _path_sample(
        samples=samples,
        noise=noise,
        s=s_prev_batch,
        path_family=path_family,
    )
    warm_state = make_stork_warm_state(
        velocity_model=velocity_model,
        z_prev=z_prev,
        s_prev=s_prev,
        labels=labels,
        cfg_scale=cfg_scale,
        last_dt=last_dt,
        step_count_hint=step_count_hint,
    )

    full_state = _clone_stork_state(warm_state)
    full_step = _stork_step_with_state(
        velocity_model=velocity_model,
        z=z_s,
        s=s,
        h=effective_step,
        labels=labels,
        cfg_scale=cfg_scale,
        step_count_hint=step_count_hint,
        state=full_state,
    )

    half_h = 0.5 * float(effective_step)
    half_step_hint = max(1, int(step_count_hint) * int(defect_subdivide))
    subdivided_state = _clone_stork_state(warm_state)
    mid = _stork_step_with_state(
        velocity_model=velocity_model,
        z=z_s,
        s=s,
        h=half_h,
        labels=labels,
        cfg_scale=cfg_scale,
        step_count_hint=half_step_hint,
        state=subdivided_state,
    )
    subdivided_step = _stork_step_with_state(
        velocity_model=velocity_model,
        z=mid,
        s=s + half_h,
        h=half_h,
        labels=labels,
        cfg_scale=cfg_scale,
        step_count_hint=half_step_hint,
        state=subdivided_state,
    )
    return full_step, subdivided_step
