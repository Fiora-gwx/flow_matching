from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from training.solver_aware.clock import (
    SolverAwareClockProfile,
    build_solver_aware_clock_profile,
    build_solver_aware_nodes,
)
from training.solver_aware.stork_hybrid import build_stork_hybrid_nodes


@dataclass
class DefectClockProfile:
    budget_mode: str
    order: float
    q_curve_name: str
    aggregation_name: str
    primary_budget: int
    target_nfe_budget: int
    target_step_count: int
    density_exponent: float
    q_raw: Tensor
    q_smoothed: Tensor
    density: Tensor
    phi: Tensor
    s_grid: Tensor
    smoothing_window: int
    q_values_by_budget: Dict[str, Tensor]
    q_normalized_by_budget: Dict[str, Tensor]
    budget_step_count_by_nfe: Dict[str, int]
    budget_weights: Dict[str, float]


@dataclass
class DefectClockArtifacts(DefectClockProfile):
    r_grid: Tensor
    nodes: Tensor


def _normalize_weights(
    budgets: Sequence[int],
    weights: Optional[Sequence[float]],
) -> Dict[int, float]:
    if not budgets:
        raise ValueError("At least one budget is required.")
    if weights is None or len(weights) == 0:
        uniform = 1.0 / float(len(budgets))
        return {int(budget): uniform for budget in budgets}
    if len(weights) != len(budgets):
        raise ValueError("target_nfe_weights must have the same length as the budget list.")
    normalized = [float(weight) for weight in weights]
    weight_sum = float(sum(normalized))
    if weight_sum <= 0.0:
        raise ValueError("target_nfe_weights must sum to a positive value.")
    return {
        int(budget): float(weight / weight_sum)
        for budget, weight in zip(budgets, normalized)
    }


def density_exponent_from_order(order: float) -> float:
    return 1.0 / (2.0 * (float(order) + 1.0))


def _profile_to_defect(
    *,
    budget_mode: str,
    order: float,
    q_curve_name: str,
    aggregation_name: str,
    primary_budget: int,
    target_nfe_budget: int,
    target_step_count: int,
    profile: SolverAwareClockProfile,
    q_values_by_budget: Dict[str, Tensor],
    q_normalized_by_budget: Dict[str, Tensor],
    budget_step_count_by_nfe: Dict[str, int],
    budget_weights: Dict[str, float],
) -> DefectClockProfile:
    return DefectClockProfile(
        budget_mode=str(budget_mode),
        order=float(order),
        q_curve_name=str(q_curve_name),
        aggregation_name=str(aggregation_name),
        primary_budget=int(primary_budget),
        target_nfe_budget=int(target_nfe_budget),
        target_step_count=int(target_step_count),
        density_exponent=float(profile.density_exponent),
        q_raw=profile.q_raw,
        q_smoothed=profile.q_smoothed,
        density=profile.density,
        phi=profile.phi,
        s_grid=profile.s_grid,
        smoothing_window=int(profile.smoothing_window),
        q_values_by_budget=q_values_by_budget,
        q_normalized_by_budget=q_normalized_by_budget,
        budget_step_count_by_nfe=budget_step_count_by_nfe,
        budget_weights=budget_weights,
    )


def build_defect_clock_profile(
    *,
    s_grid: Tensor,
    q_values_by_budget: Dict[int, Tensor],
    budget_step_count_by_nfe: Dict[int, int],
    budget_mode: str,
    order: float,
    eps: float,
    target_nfe_weights: Optional[Sequence[float]] = None,
    smoothing_window: Optional[int] = None,
) -> DefectClockProfile:
    """Build a defect-based clock using effective step counts rather than raw NFEs.

    Let B denote the raw NFE budget and K_S(B) the solver-specific effective
    macro-step count induced by that budget. Multi-budget normalization uses
    K_S(B)^(2p+2), not raw-NFE scaling.
    """
    budgets = sorted(int(budget) for budget in q_values_by_budget)
    if not budgets:
        raise ValueError("q_values_by_budget must contain at least one entry.")
    if budget_mode not in {"single_budget", "multi_budget"}:
        raise ValueError(f"Unsupported budget_mode={budget_mode}.")

    q_values_str = {
        str(budget): q_values_by_budget[int(budget)]
        for budget in budgets
    }
    q_normalized_by_budget = {
        str(budget): torch.pow(
            torch.as_tensor(
                float(budget_step_count_by_nfe[int(budget)]),
                device=s_grid.device,
                dtype=s_grid.dtype,
            ),
            2.0 * (float(order) + 1.0),
        ) * q_values_by_budget[int(budget)]
        for budget in budgets
    }
    budget_step_count_by_nfe_str = {
        str(budget): int(budget_step_count_by_nfe[int(budget)])
        for budget in budgets
    }
    budget_weights = {
        str(key): value
        for key, value in _normalize_weights(
            budgets=budgets,
            weights=target_nfe_weights,
        ).items()
    }

    if budget_mode == "single_budget":
        if len(budgets) != 1:
            raise ValueError("single_budget mode expects exactly one budget.")
        primary_budget = budgets[0]
        profile = build_solver_aware_clock_profile(
            s_grid=s_grid,
            q_values=q_values_by_budget[primary_budget],
            density_exponent=density_exponent_from_order(order=order),
            eps=eps,
            smoothing_window=smoothing_window,
        )
        return _profile_to_defect(
            budget_mode=budget_mode,
            order=order,
            q_curve_name="Q_path_defect",
            aggregation_name="single_budget",
            primary_budget=primary_budget,
            target_nfe_budget=primary_budget,
            target_step_count=budget_step_count_by_nfe[int(primary_budget)],
            profile=profile,
            q_values_by_budget=q_values_str,
            q_normalized_by_budget=q_normalized_by_budget,
            budget_step_count_by_nfe=budget_step_count_by_nfe_str,
            budget_weights=budget_weights,
        )

    aggregated_monitor = torch.zeros_like(s_grid)
    for budget in budgets:
        normalized_curve = q_normalized_by_budget[str(budget)].clamp(min=0.0)
        aggregated_monitor = aggregated_monitor + float(budget_weights[str(budget)]) * torch.sqrt(
            normalized_curve
        )
    profile = build_solver_aware_clock_profile(
        s_grid=s_grid,
        q_values=aggregated_monitor,
        density_exponent=1.0 / (float(order) + 1.0),
        eps=eps,
        smoothing_window=smoothing_window,
    )
    return _profile_to_defect(
        budget_mode=budget_mode,
        order=order,
        q_curve_name="M_tilde_path_defect",
        aggregation_name="normalized_multi_budget",
        primary_budget=budgets[0],
        target_nfe_budget=budgets[0],
        target_step_count=budget_step_count_by_nfe[int(budgets[0])],
        profile=profile,
        q_values_by_budget=q_values_str,
        q_normalized_by_budget=q_normalized_by_budget,
        budget_step_count_by_nfe=budget_step_count_by_nfe_str,
        budget_weights=budget_weights,
    )


def build_defect_clock(
    *,
    s_grid: Tensor,
    q_values_by_budget: Dict[int, Tensor],
    budget_step_count_by_nfe: Dict[int, int],
    budget_mode: str,
    order: float,
    eps: float,
    node_count: int,
    target_solver: Optional[str] = None,
    target_nfe_weights: Optional[Sequence[float]] = None,
    smoothing_window: Optional[int] = None,
) -> DefectClockArtifacts:
    profile = build_defect_clock_profile(
        s_grid=s_grid,
        q_values_by_budget=q_values_by_budget,
        budget_step_count_by_nfe=budget_step_count_by_nfe,
        budget_mode=budget_mode,
        order=order,
        eps=eps,
        target_nfe_weights=target_nfe_weights,
        smoothing_window=smoothing_window,
    )
    if str(target_solver or "") == "stork4":
        r_grid, nodes, _ = build_stork_hybrid_nodes(
            s_grid=profile.s_grid,
            phi=profile.phi,
            step_count=max(1, int(node_count) - 1),
        )
    else:
        r_grid, nodes = build_solver_aware_nodes(
            s_grid=profile.s_grid,
            phi=profile.phi,
            node_count=node_count,
        )
    return DefectClockArtifacts(
        budget_mode=profile.budget_mode,
        order=profile.order,
        q_curve_name=profile.q_curve_name,
        aggregation_name=profile.aggregation_name,
        primary_budget=profile.primary_budget,
        target_nfe_budget=profile.target_nfe_budget,
        target_step_count=profile.target_step_count,
        density_exponent=profile.density_exponent,
        q_raw=profile.q_raw,
        q_smoothed=profile.q_smoothed,
        density=profile.density,
        phi=profile.phi,
        s_grid=profile.s_grid,
        smoothing_window=profile.smoothing_window,
        q_values_by_budget=profile.q_values_by_budget,
        q_normalized_by_budget=profile.q_normalized_by_budget,
        budget_step_count_by_nfe=profile.budget_step_count_by_nfe,
        budget_weights=profile.budget_weights,
        r_grid=r_grid,
        nodes=nodes,
    )
