from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class FixedStepSample:
    sample: Tensor
    nfe: int
    step_count: int
    time_grid: Tensor
    step_methods: Tuple[str, ...]
    trajectory: Optional[Tensor] = None
    deltas: Optional[Tensor] = None


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

    raise ValueError(f"Unsupported solver_name={solver_name}.")


@torch.no_grad()
def solve_fixed_budget(
    velocity_model,
    x_init: Tensor,
    solver_name: str,
    nfe_budget: int,
    return_trajectory: bool = False,
    **model_extras,
) -> FixedStepSample:
    step_methods = build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget)
    step_count = len(step_methods)
    time_grid = torch.linspace(
        0.0,
        1.0,
        step_count + 1,
        device=x_init.device,
        dtype=x_init.dtype,
    )
    dt = 1.0 / step_count
    x_t = x_init
    states = [x_init.clone()] if return_trajectory else None

    if hasattr(velocity_model, "reset_nfe_counter"):
        velocity_model.reset_nfe_counter()

    for step_index, step_method in enumerate(step_methods):
        t_start = torch.full(
            (x_t.shape[0],),
            float(time_grid[step_index].item()),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        if step_method == "euler":
            k1 = velocity_model(x_t, t_start, **model_extras)
            x_t = x_t + dt * k1
        elif step_method == "heun2":
            k1 = velocity_model(x_t, t_start, **model_extras)
            x_predict = x_t + dt * k1
            t_end = torch.full(
                (x_t.shape[0],),
                float(time_grid[step_index + 1].item()),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            k2 = velocity_model(x_predict, t_end, **model_extras)
            x_t = x_t + 0.5 * dt * (k1 + k2)
        else:
            raise AssertionError(f"Unhandled step_method={step_method}.")

        if states is not None:
            states.append(x_t.clone())

    if hasattr(velocity_model, "get_nfe"):
        nfe = int(velocity_model.get_nfe())
    else:
        nfe = sum(1 if method == "euler" else 2 for method in step_methods)

    trajectory = None
    deltas = None
    if states is not None:
        trajectory = torch.stack(states, dim=0)
        deltas = trajectory[1:] - trajectory[:-1]

    return FixedStepSample(
        sample=x_t,
        nfe=nfe,
        step_count=step_count,
        time_grid=time_grid,
        step_methods=step_methods,
        trajectory=trajectory,
        deltas=deltas,
    )
