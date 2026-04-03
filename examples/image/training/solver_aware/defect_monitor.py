import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor

from training.fixed_step_solver import build_step_methods
from training.solver_aware.monitors import (
    _cycle_loader,
    _make_generator,
    _path_sample,
    _prepare_reference_batch,
)
from training.stork_solver import STORKState, stork4_step


logger = logging.getLogger(__name__)
DEFECT_MONITOR_MICROBATCH = 8


@dataclass
class DefectMonitorArtifacts:
    target_solver: str
    budget_mode: str
    path_family: str
    defect_subdivide: int
    order: float
    theorem_backed: bool
    monitor_name: str
    notes: str
    s_grid: Tensor
    q_values_by_budget: Dict[str, Tensor]
    budget_step_count_by_nfe: Dict[str, int]
    target_nfe_weights: Dict[str, float]
    distribution_info: Dict[str, object]


def _velocity_eval(
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


def _solver_step_count(solver_name: str, nfe_budget: int) -> int:
    if solver_name == "stork4":
        return int(nfe_budget)
    return len(build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget))


def _resolve_order_and_notes(
    *,
    target_solver: str,
    stork_effective_order: float,
) -> tuple[float, bool, str]:
    theorem = (
        "For a p-th order solver S with "
        "Psi_h^S = Phi + C_S h^(p+1) E_S[u] + O(h^(p+2)), "
        "the self-consistency defect satisfies "
        "Delta_S = Psi_h^S - Psi_{h/2}^{S,(2)} = "
        "C_S (1 - 2^{-p}) h^(p+1) E_S[u] + O(h^(p+2)). "
    )
    if target_solver == "euler":
        return (
            1.0,
            True,
            theorem
            + "The defect monitor is evaluated on z ~ p_s, so "
            "Q_{E,N}^{path}(s) = E_{z~p_s}||Delta_E(z,s;h)||^2 and "
            "rho_{E,N}(s) propto (Q_{E,N}^{path}(s)+eps)^(1/4).",
        )
    if target_solver == "heun2":
        return (
            2.0,
            True,
            theorem
            + "The defect monitor is evaluated on z ~ p_s, so "
            "Q_{H,N}^{path}(s) = E_{z~p_s}||Delta_H(z,s;h)||^2 and "
            "rho_{H,N}(s) propto (Q_{H,N}^{path}(s)+eps)^(1/6).",
        )
    if target_solver == "stork4":
        configured_order = float(stork_effective_order)
        return (
            configured_order,
            False,
            theorem
            + "For STORK we use a configured effective order "
            f"p_stork={configured_order}. The monitor is still evaluated on z ~ p_s, "
            "but the order is assumed / configured rather than theorem-backed.",
        )
    raise ValueError(f"Unsupported target_solver={target_solver}.")


def _euler_step(
    *,
    velocity_model,
    z: Tensor,
    s: float,
    h: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
) -> Tensor:
    t = torch.full((z.shape[0],), float(s), device=z.device, dtype=z.dtype)
    u = _velocity_eval(
        velocity_model=velocity_model,
        x=z,
        t=t,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=h,
        step_count=step_count_hint,
    )
    return z + float(h) * u


def _heun2_step(
    *,
    velocity_model,
    z: Tensor,
    s: float,
    h: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
) -> Tensor:
    t_start = torch.full((z.shape[0],), float(s), device=z.device, dtype=z.dtype)
    k1 = _velocity_eval(
        velocity_model=velocity_model,
        x=z,
        t=t_start,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=h,
        step_count=step_count_hint,
    )
    z_predict = z + float(h) * k1
    t_end = torch.full((z.shape[0],), float(s + h), device=z.device, dtype=z.dtype)
    k2 = _velocity_eval(
        velocity_model=velocity_model,
        x=z_predict,
        t=t_end,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=h,
        step_count=step_count_hint,
    )
    return z + 0.5 * float(h) * (k1 + k2)


def _stork4_step(
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
    t = torch.full((z.shape[0],), float(s), device=z.device, dtype=z.dtype)
    model_output = _velocity_eval(
        velocity_model=velocity_model,
        x=z,
        t=t,
        labels=labels,
        cfg_scale=cfg_scale,
        step_size=h,
        step_count=step_count_hint,
    )
    return stork4_step(
        model_output=model_output,
        sample=z,
        t_start=float(s),
        t_end=float(s + h),
        state=state,
    )


def _apply_single_step(
    *,
    solver_name: str,
    velocity_model,
    z: Tensor,
    s: float,
    h: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
) -> Tensor:
    if solver_name == "euler":
        return _euler_step(
            velocity_model=velocity_model,
            z=z,
            s=s,
            h=h,
            labels=labels,
            cfg_scale=cfg_scale,
            step_count_hint=step_count_hint,
        )
    if solver_name == "heun2":
        return _heun2_step(
            velocity_model=velocity_model,
            z=z,
            s=s,
            h=h,
            labels=labels,
            cfg_scale=cfg_scale,
            step_count_hint=step_count_hint,
        )
    if solver_name == "stork4":
        return _stork4_step(
            velocity_model=velocity_model,
            z=z,
            s=s,
            h=h,
            labels=labels,
            cfg_scale=cfg_scale,
            step_count_hint=step_count_hint,
            state=STORKState(),
        )
    raise ValueError(f"Unsupported solver_name={solver_name}.")


def _apply_subdivided_step(
    *,
    solver_name: str,
    velocity_model,
    z: Tensor,
    s: float,
    h: float,
    labels: Tensor,
    cfg_scale: float,
    step_count_hint: int,
    defect_subdivide: int,
) -> Tensor:
    if defect_subdivide != 2:
        raise NotImplementedError(
            "Phase-1 defect monitor currently supports one step vs two half steps only."
        )
    half_h = 0.5 * float(h)
    half_step_hint = max(1, int(step_count_hint) * int(defect_subdivide))
    if solver_name == "stork4":
        state = STORKState()
        mid = _stork4_step(
            velocity_model=velocity_model,
            z=z,
            s=s,
            h=half_h,
            labels=labels,
            cfg_scale=cfg_scale,
            step_count_hint=half_step_hint,
            state=state,
        )
        return _stork4_step(
            velocity_model=velocity_model,
            z=mid,
            s=s + half_h,
            h=half_h,
            labels=labels,
            cfg_scale=cfg_scale,
            step_count_hint=half_step_hint,
            state=state,
        )
    mid = _apply_single_step(
        solver_name=solver_name,
        velocity_model=velocity_model,
        z=z,
        s=s,
        h=half_h,
        labels=labels,
        cfg_scale=cfg_scale,
        step_count_hint=half_step_hint,
    )
    return _apply_single_step(
        solver_name=solver_name,
        velocity_model=velocity_model,
        z=mid,
        s=s + half_h,
        h=half_h,
        labels=labels,
        cfg_scale=cfg_scale,
        step_count_hint=half_step_hint,
    )


def _compute_budget_curve(
    *,
    velocity_model,
    samples: Tensor,
    labels: Tensor,
    noise: Tensor,
    s_grid: Tensor,
    path_family: str,
    target_solver: str,
    target_nfe: int,
    cfg_scale: float,
    defect_subdivide: int,
) -> Tensor:
    step_count = _solver_step_count(solver_name=target_solver, nfe_budget=target_nfe)
    step_size = 1.0 / float(max(1, step_count))
    microbatch_size = max(1, min(int(samples.shape[0]), DEFECT_MONITOR_MICROBATCH))
    q_values = torch.zeros_like(s_grid, dtype=torch.float32)

    for index, s_value in enumerate(s_grid):
        s_float = float(s_value.item())
        effective_step = min(step_size, max(0.0, 1.0 - s_float))
        if effective_step <= 0.0:
            q_values[index] = 0.0
            continue
        squared_norm_sum = torch.zeros((), device=s_grid.device, dtype=torch.float32)
        sample_count = 0
        for sample_chunk, label_chunk, noise_chunk in zip(
            samples.split(microbatch_size),
            labels.split(microbatch_size),
            noise.split(microbatch_size),
        ):
            s_batch = torch.full(
                (sample_chunk.shape[0],),
                s_float,
                device=sample_chunk.device,
                dtype=sample_chunk.dtype,
            )
            z_s = _path_sample(
                samples=sample_chunk,
                noise=noise_chunk,
                s=s_batch,
                path_family=path_family,
            )
            full_step = _apply_single_step(
                solver_name=target_solver,
                velocity_model=velocity_model,
                z=z_s,
                s=s_float,
                h=effective_step,
                labels=label_chunk,
                cfg_scale=cfg_scale,
                step_count_hint=step_count,
            )
            subdivided = _apply_subdivided_step(
                solver_name=target_solver,
                velocity_model=velocity_model,
                z=z_s,
                s=s_float,
                h=effective_step,
                labels=label_chunk,
                cfg_scale=cfg_scale,
                step_count_hint=step_count,
                defect_subdivide=defect_subdivide,
            )
            defect = full_step - subdivided
            squared_norm = defect.flatten(start_dim=1).pow(2).sum(dim=1)
            squared_norm_sum = squared_norm_sum + squared_norm.sum()
            sample_count += int(squared_norm.shape[0])
        q_values[index] = squared_norm_sum / max(1, sample_count)
    return q_values


def _normalize_budget_weights(
    budgets: Sequence[int],
    target_nfe_weights: Optional[Sequence[float]],
) -> Dict[str, float]:
    if target_nfe_weights is None or len(target_nfe_weights) == 0:
        weight = 1.0 / float(len(budgets))
        return {str(int(budget)): weight for budget in budgets}
    if len(target_nfe_weights) != len(budgets):
        raise ValueError("target_nfe_weights must have the same length as target_nfes.")
    total = float(sum(float(weight) for weight in target_nfe_weights))
    if total <= 0.0:
        raise ValueError("target_nfe_weights must sum to a positive value.")
    return {
        str(int(budget)): float(float(weight) / total)
        for budget, weight in zip(budgets, target_nfe_weights)
    }


def compute_defect_monitor(
    *,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    grid_size: int,
    batch_size: int,
    seed: int,
    target_solver: str,
    budget_mode: str,
    target_nfes: Sequence[int],
    target_nfe_weights: Optional[Sequence[float]],
    cfg_scale: float,
    defect_subdivide: int,
    stork_effective_order: float,
) -> DefectMonitorArtifacts:
    """Compute Q_{S,N}^{path}(s) on the known path distribution z ~ p_s.

    For each s-grid point we sample z_s directly from the analytical path
    distribution p_s rather than from a model rollout distribution:

        z_s = alpha(s) x + sigma(s) epsilon,  x ~ p_data, epsilon ~ p_0.

    The defect proxy then becomes

        Q_{S,N}^{path}(s) = E_{z~p_s} ||Psi_h^S(z,s) - Psi_{h/2}^{S,(2)}(z,s)||^2,

    which is solver-specific, budget-specific, path-distribution-based, and
    training-free.
    """
    budgets = sorted(int(budget) for budget in target_nfes)
    if not budgets:
        raise ValueError("At least one target budget is required.")
    order, theorem_backed, notes = _resolve_order_and_notes(
        target_solver=target_solver,
        stork_effective_order=stork_effective_order,
    )
    loader_iter = _cycle_loader(data_loader)
    noise_generator = _make_generator(device=device, seed=seed + 29071)
    samples, labels, noise = _prepare_reference_batch(
        loader_iter=loader_iter,
        batch_size=batch_size,
        device=device,
        noise_generator=noise_generator,
    )
    s_grid = torch.linspace(0.0, 1.0, int(grid_size), device=device, dtype=torch.float32)

    q_values_by_budget: Dict[str, Tensor] = {}
    budget_step_count_by_nfe: Dict[str, int] = {}
    with torch.no_grad():
        for budget in budgets:
            q_values_by_budget[str(budget)] = _compute_budget_curve(
                velocity_model=velocity_model,
                samples=samples,
                labels=labels,
                noise=noise,
                s_grid=s_grid,
                path_family=path_family,
                target_solver=target_solver,
                target_nfe=budget,
                cfg_scale=cfg_scale,
                defect_subdivide=defect_subdivide,
            ).detach()
            budget_step_count_by_nfe[str(budget)] = _solver_step_count(
                solver_name=target_solver,
                nfe_budget=budget,
            )

    logger.info(
        "Computed path-based defect monitor for solver=%s over budgets=%s on path_family=%s.",
        target_solver,
        budgets,
        path_family,
    )
    if target_solver == "stork4":
        logger.info(
            "Path-based STORK defect monitor uses configured effective order p_stork=%s (assumed/configured).",
            order,
        )

    return DefectMonitorArtifacts(
        target_solver=str(target_solver),
        budget_mode=str(budget_mode),
        path_family=str(path_family),
        defect_subdivide=int(defect_subdivide),
        order=float(order),
        theorem_backed=bool(theorem_backed),
        monitor_name=f"{target_solver}_path_defect",
        notes=notes,
        s_grid=s_grid.detach(),
        q_values_by_budget=q_values_by_budget,
        budget_step_count_by_nfe=budget_step_count_by_nfe,
        target_nfe_weights=_normalize_budget_weights(
            budgets=budgets,
            target_nfe_weights=target_nfe_weights,
        ),
        distribution_info={
            "distribution": "path_distribution",
            "path_family": str(path_family),
            "batch_size": int(batch_size),
            "grid_size": int(grid_size),
            "fixed_path_batch": True,
        },
    )
