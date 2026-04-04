from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from training.nonuniform_nodes import resolve_time_grid
from training.stork_solver import STORKState, stork4_step
from training.tack import solve_tack


STEP_NFE_COST = {
    "euler": 1,
    "heun2": 2,
    "rk3": 3,
    "stork4": 1,
}
SHARED_FAIR_BUDGETS = frozenset({6, 12, 18, 24, 30, 48, 96})


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


@dataclass
class FixedStepSample:
    sample: Tensor
    nfe: int
    step_count: int
    time_grid: Tensor
    step_methods: Tuple[str, ...]
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
    **model_extras,
) -> FixedStepSample:
    tack_config = model_extras.pop("tack_config", None)
    tack_profile = model_extras.pop("tack_profile", None)
    artifact_dir = model_extras.pop("artifact_dir", None)

    if solver_name == "tack":
        if time_grid is not None:
            raise ValueError("sampling_solver=tack does not accept an external fixed time_grid.")
        if tack_config is None:
            raise ValueError("sampling_solver=tack requires tack_config.")
        tack_output = solve_tack(
            velocity_model=velocity_model,
            x_init=x_init,
            config=tack_config,
            profile=tack_profile,
            return_trajectory=return_trajectory,
            artifact_dir=artifact_dir,
            **model_extras,
        )
        return FixedStepSample(
            sample=tack_output.sample,
            nfe=tack_output.nfe,
            step_count=tack_output.step_count,
            time_grid=tack_output.time_grid,
            step_methods=tack_output.step_methods,
            trajectory=tack_output.trajectory,
            deltas=tack_output.deltas,
            solver_stats=tack_output.solver_stats,
        )

    if solver_name == "stork4":
        step_count = nfe_budget
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
            x_t = stork4_step(
                model_output=model_output,
                sample=x_t,
                t_start=t_start_value,
                t_end=t_end_value,
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
            trajectory = torch.stack(states, dim=0)
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
            trajectory=trajectory,
            deltas=deltas,
            solver_stats=solver_stats,
        )

    step_methods = build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget)
    step_count = len(step_methods)
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
        if step_method == "euler":
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
            x_t = x_t + step_size * k1
        elif step_method == "heun2":
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
            x_predict = x_t + step_size * k1
            t_end = torch.full(
                (x_t.shape[0],),
                t_end_value,
                device=x_t.device,
                dtype=x_t.dtype,
            )
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
            x_t = x_t + 0.5 * step_size * (k1 + k2)
        elif step_method == "rk3":
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
        trajectory = torch.stack(states, dim=0)
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
        trajectory=trajectory,
        deltas=deltas,
        solver_stats=solver_stats,
    )
