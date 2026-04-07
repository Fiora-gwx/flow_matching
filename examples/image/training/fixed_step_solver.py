from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from training.nonuniform_nodes import resolve_time_grid
from training.stork_solver import STORKState, stork4_step


STEP_NFE_COST = {
    "euler": 1,
    "heun2": 2,
    "rk3": 3,
    "stork4": 1,
}
SHARED_FAIR_BUDGETS = frozenset({6, 12, 18, 24, 30, 48, 96})


@dataclass
class ReparameterizedSchedule:
    """Uniform-tau schedule for dx / d tau = g(tau) v_theta(x, t(tau); c)."""

    tau_grid: Tensor
    t_grid: Tensor
    g_grid: Tensor
    dtau: float
    nfe_budget: Optional[int] = None
    step_count: Optional[int] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "ReparameterizedSchedule":
        return ReparameterizedSchedule(
            tau_grid=self.tau_grid.to(device=device, dtype=dtype),
            t_grid=self.t_grid.to(device=device, dtype=dtype),
            g_grid=self.g_grid.to(device=device, dtype=dtype),
            dtau=float(self.dtau),
            nfe_budget=self.nfe_budget,
            step_count=self.step_count,
        )


def _adapt_model_time(
    velocity_model,
    t: Tensor,
    step_size: float,
    step_count: int,
) -> Tensor:
    adapt = getattr(velocity_model, "adapt_solver_time", None)
    if callable(adapt):
        return adapt(t=t, step_size=step_size, step_count=step_count)
    return t


def _expand_scalar_like(reference: Tensor, scalar: Tensor) -> Tensor:
    view_shape = [scalar.shape[0]] + [1] * (reference.dim() - 1)
    return scalar.view(view_shape)


def _validate_reparameterized_schedule(
    schedule: ReparameterizedSchedule,
    *,
    expected_step_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ReparameterizedSchedule:
    resolved = schedule.to(device=device, dtype=dtype)
    if resolved.tau_grid.ndim != 1 or resolved.t_grid.ndim != 1 or resolved.g_grid.ndim != 1:
        raise ValueError("Reparameterized schedule tensors must be one-dimensional.")
    expected_nodes = int(expected_step_count) + 1
    if (
        resolved.tau_grid.numel() != expected_nodes
        or resolved.t_grid.numel() != expected_nodes
        or resolved.g_grid.numel() != expected_nodes
    ):
        raise ValueError(
            "Reparameterized schedule length mismatch. "
            f"expected={expected_nodes}, got tau={resolved.tau_grid.numel()}, "
            f"t={resolved.t_grid.numel()}, g={resolved.g_grid.numel()}."
        )
    tau_deltas = resolved.tau_grid[1:] - resolved.tau_grid[:-1]
    if torch.any(tau_deltas <= 0.0):
        raise ValueError("tau_grid must be strictly increasing.")
    if torch.any(resolved.g_grid <= 0.0):
        raise ValueError("g_grid must stay strictly positive.")
    if abs(float(resolved.tau_grid[0].item())) > 1.0e-8 or abs(float(resolved.tau_grid[-1].item()) - 1.0) > 1.0e-8:
        raise ValueError("tau_grid must start at 0 and end at 1.")
    t_deltas = resolved.t_grid[1:] - resolved.t_grid[:-1]
    if torch.any(t_deltas <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")
    if abs(float(resolved.t_grid[0].item())) > 1.0e-8 or abs(float(resolved.t_grid[-1].item()) - 1.0) > 1.0e-8:
        raise ValueError("t_grid must start at 0 and end at 1.")
    expected_dtau = 1.0 / float(max(1, expected_step_count))
    if not torch.allclose(
        tau_deltas,
        torch.full_like(tau_deltas, expected_dtau),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise ValueError("Reparameterized schedule requires a uniform tau grid.")
    if abs(float(resolved.dtau) - expected_dtau) > 1.0e-6:
        raise ValueError(
            f"Reparameterized schedule dtau mismatch. expected={expected_dtau}, got={resolved.dtau}."
        )
    return resolved


def _evaluate_reparameterized_field(
    velocity_model,
    x: Tensor,
    t: Tensor,
    g: Tensor,
    step_size: float,
    step_count: int,
    **model_extras,
) -> Tensor:
    velocity = velocity_model(
        x,
        _adapt_model_time(
            velocity_model=velocity_model,
            t=t,
            step_size=step_size,
            step_count=step_count,
        ),
        **model_extras,
    )
    return _expand_scalar_like(reference=velocity, scalar=g) * velocity


@dataclass
class FixedStepSample:
    sample: Tensor
    nfe: int
    step_count: int
    time_grid: Tensor
    step_methods: Tuple[str, ...]
    tau_grid: Optional[Tensor] = None
    trajectory: Optional[Tensor] = None
    deltas: Optional[Tensor] = None
    solver_stats: Optional[Dict[str, object]] = None


def build_step_methods(solver_name: str, nfe_budget: int) -> Tuple[str, ...]:
    if nfe_budget <= 0:
        raise ValueError(f"nfe_budget must be positive. Got {nfe_budget}.")

    if solver_name == "euler":
        return tuple("euler" for _ in range(nfe_budget))

    if solver_name == "heun2":
        methods = ["heun2" for _ in range(nfe_budget // 2)]
        if nfe_budget % 2 == 1:
            methods.append("euler")
        return tuple(methods)

    if solver_name == "rk3":
        methods = ["rk3" for _ in range(nfe_budget // 3)]
        remainder = nfe_budget % 3
        if remainder == 1:
            methods.append("euler")
        elif remainder == 2:
            methods.append("heun2")
        return tuple(methods)

    raise ValueError(f"Unsupported solver_name={solver_name}.")


def is_exact_budget(solver_name: str, nfe_budget: int) -> bool:
    if solver_name in {"euler", "stork4"}:
        return True
    if solver_name == "heun2":
        return nfe_budget % 2 == 0
    if solver_name == "rk3":
        return nfe_budget % 3 == 0
    raise ValueError(f"Unsupported solver_name={solver_name}.")


def build_solver_stats(
    solver_name: str,
    requested_nfe_budget: int,
    actual_network_calls: int,
    step_count: int,
    step_methods: Tuple[str, ...],
    virtual_stage_count: int = 0,
) -> Dict[str, object]:
    exact_budget = is_exact_budget(solver_name, requested_nfe_budget)
    tail_step_methods = tuple(method for method in step_methods if method != solver_name)
    used_tail_step = len(tail_step_methods) > 0
    return {
        "solver": solver_name,
        "requested_nfe_budget": int(requested_nfe_budget),
        "actual_network_calls": int(actual_network_calls),
        "step_count": int(step_count),
        "virtual_stage_count": int(virtual_stage_count),
        "used_tail_step": bool(used_tail_step),
        "tail_step_methods": tail_step_methods,
        "is_exact_budget": bool(exact_budget),
        "is_shared_budget": bool(exact_budget and requested_nfe_budget in SHARED_FAIR_BUDGETS),
    }


def get_tail_step_methods(solver_name: str, nfe_budget: int) -> Tuple[str, ...]:
    return tuple(
        method
        for method in build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget)
        if method != solver_name
    )


@torch.no_grad()
def solve_fixed_budget(
    velocity_model,
    x_init: Tensor,
    solver_name: str,
    nfe_budget: int,
    return_trajectory: bool = False,
    time_grid: Optional[Tensor] = None,
    reparameterized_schedule: Optional[ReparameterizedSchedule] = None,
    **model_extras,
) -> FixedStepSample:
    if time_grid is not None and reparameterized_schedule is not None:
        raise ValueError("time_grid and reparameterized_schedule are mutually exclusive.")

    if solver_name == "stork4":
        step_count = nfe_budget
        resolved_schedule = None
        if reparameterized_schedule is not None:
            resolved_schedule = _validate_reparameterized_schedule(
                reparameterized_schedule,
                expected_step_count=step_count,
                device=x_init.device,
                dtype=x_init.dtype,
            )
            resolved_time_grid = resolved_schedule.t_grid
        else:
            resolved_time_grid = resolve_time_grid(
                step_count=step_count,
                device=x_init.device,
                dtype=x_init.dtype,
                time_grid=time_grid,
            )
        x_t = x_init
        states = [x_init.clone()] if return_trajectory else None
        stork_state = STORKState()
        step_methods = tuple("stork4" for _ in range(step_count))

        if hasattr(velocity_model, "reset_nfe_counter"):
            velocity_model.reset_nfe_counter()

        for step_index in range(step_count):
            t_start_value = float(resolved_time_grid[step_index].item())
            t_end_value = float(resolved_time_grid[step_index + 1].item())
            t_start = torch.full(
                (x_t.shape[0],),
                t_start_value,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            if resolved_schedule is None:
                model_output = velocity_model(
                    x_t,
                    _adapt_model_time(
                        velocity_model=velocity_model,
                        t=t_start,
                        step_size=t_end_value - t_start_value,
                        step_count=step_count,
                    ),
                    **model_extras,
                )
                stork_t_start = t_start_value
                stork_t_end = t_end_value
            else:
                g_start = torch.full(
                    (x_t.shape[0],),
                    float(resolved_schedule.g_grid[step_index].item()),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                tau_start_value = float(resolved_schedule.tau_grid[step_index].item())
                tau_end_value = float(resolved_schedule.tau_grid[step_index + 1].item())
                model_output = _evaluate_reparameterized_field(
                    velocity_model=velocity_model,
                    x=x_t,
                    t=t_start,
                    g=g_start,
                    step_size=t_end_value - t_start_value,
                    step_count=step_count,
                    **model_extras,
                )
                stork_t_start = tau_start_value
                stork_t_end = tau_end_value
            x_t = stork4_step(
                model_output=model_output,
                sample=x_t,
                t_start=stork_t_start,
                t_end=stork_t_end,
                state=stork_state,
            )
            if states is not None:
                states.append(x_t.clone())

        if hasattr(velocity_model, "get_nfe"):
            nfe = int(velocity_model.get_nfe())
        else:
            nfe = nfe_budget

        trajectory = None
        deltas = None
        if states is not None:
            trajectory = torch.cat([state.unsqueeze(0) for state in states], dim=0)
            deltas = trajectory[1:] - trajectory[:-1]

        solver_stats = build_solver_stats(
            solver_name=solver_name,
            requested_nfe_budget=nfe_budget,
            actual_network_calls=nfe,
            step_count=step_count,
            step_methods=step_methods,
            virtual_stage_count=stork_state.virtual_stage_count,
        )
        return FixedStepSample(
            sample=x_t,
            nfe=nfe,
            step_count=step_count,
            time_grid=resolved_time_grid,
            step_methods=step_methods,
            tau_grid=None if resolved_schedule is None else resolved_schedule.tau_grid,
            trajectory=trajectory,
            deltas=deltas,
            solver_stats=solver_stats,
        )

    step_methods = build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget)
    step_count = len(step_methods)
    resolved_schedule = None
    if reparameterized_schedule is not None:
        if solver_name not in {"euler", "heun2"}:
            raise ValueError(
                f"reparameterized_schedule currently supports euler/heun2/stork4. Got solver_name={solver_name}."
            )
        resolved_schedule = _validate_reparameterized_schedule(
            reparameterized_schedule,
            expected_step_count=step_count,
            device=x_init.device,
            dtype=x_init.dtype,
        )
        resolved_time_grid = resolved_schedule.t_grid
    else:
        resolved_time_grid = resolve_time_grid(
            step_count=step_count,
            device=x_init.device,
            dtype=x_init.dtype,
            time_grid=time_grid,
        )
    x_t = x_init
    states = [x_init.clone()] if return_trajectory else None

    if hasattr(velocity_model, "reset_nfe_counter"):
        velocity_model.reset_nfe_counter()

    for step_index, step_method in enumerate(step_methods):
        t_start = torch.full(
            (x_t.shape[0],),
            float(resolved_time_grid[step_index].item()),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        t_start_value = float(resolved_time_grid[step_index].item())
        t_end_value = float(resolved_time_grid[step_index + 1].item())
        step_size = t_end_value - t_start_value
        dtau = (
            float(
                resolved_schedule.tau_grid[step_index + 1].item()
                - resolved_schedule.tau_grid[step_index].item()
            )
            if resolved_schedule is not None
            else step_size
        )
        if step_method == "euler":
            if resolved_schedule is None:
                k1 = velocity_model(
                    x_t,
                    _adapt_model_time(
                        velocity_model=velocity_model,
                        t=t_start,
                        step_size=step_size,
                        step_count=step_count,
                    ),
                    **model_extras,
                )
            else:
                g_start = torch.full(
                    (x_t.shape[0],),
                    float(resolved_schedule.g_grid[step_index].item()),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                k1 = _evaluate_reparameterized_field(
                    velocity_model=velocity_model,
                    x=x_t,
                    t=t_start,
                    g=g_start,
                    step_size=step_size,
                    step_count=step_count,
                    **model_extras,
                )
            x_t = x_t + dtau * k1
        elif step_method == "heun2":
            if resolved_schedule is None:
                k1 = velocity_model(
                    x_t,
                    _adapt_model_time(
                        velocity_model=velocity_model,
                        t=t_start,
                        step_size=step_size,
                        step_count=step_count,
                    ),
                    **model_extras,
                )
            else:
                g_start = torch.full(
                    (x_t.shape[0],),
                    float(resolved_schedule.g_grid[step_index].item()),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                k1 = _evaluate_reparameterized_field(
                    velocity_model=velocity_model,
                    x=x_t,
                    t=t_start,
                    g=g_start,
                    step_size=step_size,
                    step_count=step_count,
                    **model_extras,
                )
            x_predict = x_t + dtau * k1
            t_end = torch.full(
                (x_t.shape[0],),
                t_end_value,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            if resolved_schedule is None:
                k2 = velocity_model(
                    x_predict,
                    _adapt_model_time(
                        velocity_model=velocity_model,
                        t=t_end,
                        step_size=step_size,
                        step_count=step_count,
                    ),
                    **model_extras,
                )
            else:
                g_end = torch.full(
                    (x_t.shape[0],),
                    float(resolved_schedule.g_grid[step_index + 1].item()),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                k2 = _evaluate_reparameterized_field(
                    velocity_model=velocity_model,
                    x=x_predict,
                    t=t_end,
                    g=g_end,
                    step_size=step_size,
                    step_count=step_count,
                    **model_extras,
                )
            x_t = x_t + 0.5 * dtau * (k1 + k2)
        elif step_method == "rk3":
            if resolved_schedule is not None:
                raise ValueError("reparameterized_schedule does not support rk3.")
            k1 = velocity_model(
                x_t,
                _adapt_model_time(
                    velocity_model=velocity_model,
                    t=t_start,
                    step_size=step_size,
                    step_count=step_count,
                ),
                **model_extras,
            )
            t_mid = torch.full(
                (x_t.shape[0],),
                float(t_start_value + 0.5 * step_size),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            x_mid = x_t + 0.5 * step_size * k1
            k2 = velocity_model(
                x_mid,
                _adapt_model_time(
                    velocity_model=velocity_model,
                    t=t_mid,
                    step_size=step_size,
                    step_count=step_count,
                ),
                **model_extras,
            )
            t_end = torch.full(
                (x_t.shape[0],),
                t_end_value,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            x_end = x_t - step_size * k1 + 2.0 * step_size * k2
            k3 = velocity_model(
                x_end,
                _adapt_model_time(
                    velocity_model=velocity_model,
                    t=t_end,
                    step_size=step_size,
                    step_count=step_count,
                ),
                **model_extras,
            )
            x_t = x_t + step_size * ((1.0 / 6.0) * k1 + (2.0 / 3.0) * k2 + (1.0 / 6.0) * k3)
        else:
            raise AssertionError(f"Unhandled step_method={step_method}.")

        if states is not None:
            states.append(x_t.clone())

    if hasattr(velocity_model, "get_nfe"):
        nfe = int(velocity_model.get_nfe())
    else:
        nfe = sum(STEP_NFE_COST[method] for method in step_methods)

    trajectory = None
    deltas = None
    if states is not None:
        trajectory = torch.cat([state.unsqueeze(0) for state in states], dim=0)
        deltas = trajectory[1:] - trajectory[:-1]

    solver_stats = build_solver_stats(
        solver_name=solver_name,
        requested_nfe_budget=nfe_budget,
        actual_network_calls=nfe,
        step_count=step_count,
        step_methods=step_methods,
    )

    return FixedStepSample(
        sample=x_t,
        nfe=nfe,
        step_count=step_count,
        time_grid=resolved_time_grid,
        step_methods=step_methods,
        tau_grid=None if resolved_schedule is None else resolved_schedule.tau_grid,
        trajectory=trajectory,
        deltas=deltas,
        solver_stats=solver_stats,
    )
