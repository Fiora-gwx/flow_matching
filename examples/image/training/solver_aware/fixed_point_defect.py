from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from torch import Tensor

from training.solver_aware.defect_clock import build_defect_clock_profile
from training.solver_aware.defect_monitor import compute_defect_monitor


@dataclass
class DefectFixedPointProfile:
    mode: str
    target_solver: str
    budget_mode: str
    target_nfes: Tuple[int, ...]
    target_nfe_budget: int
    target_step_count: int
    target_nfe_weights: Dict[str, float]
    theorem_backed: bool
    notes: str
    checkpoint_source: str
    defect_subdivide: int
    solver_order: float
    q_curve_name: str
    aggregation_name: str
    s_grid: Tensor
    q_raw: Tensor
    q_smoothed: Tensor
    density: Tensor
    phi: Tensor
    density_exponent: float
    smoothing_window: int
    q_values_by_budget: Dict[str, Tensor]
    q_normalized_by_budget: Dict[str, Tensor]
    budget_step_count_by_nfe: Dict[str, int]
    budget_weights: Dict[str, float]
    distribution_info: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in ("s_grid", "q_raw", "q_smoothed", "density", "phi"):
            payload[key] = payload[key].detach().cpu()
        payload["q_values_by_budget"] = {
            key: value.detach().cpu()
            for key, value in self.q_values_by_budget.items()
        }
        payload["q_normalized_by_budget"] = {
            key: value.detach().cpu()
            for key, value in self.q_normalized_by_budget.items()
        }
        return payload


def _append_note(base: str, extra: str) -> str:
    base_text = str(base).strip()
    extra_text = str(extra).strip()
    if not extra_text:
        return base_text
    if not base_text:
        return extra_text
    return f"{base_text} {extra_text}"


def resolve_defect_target_nfes(
    *,
    budget_mode: str,
    current_nfe_budget: int,
    target_nfe: int,
    target_nfe_list: Sequence[int],
) -> Tuple[int, ...]:
    if budget_mode == "single_budget":
        resolved = int(target_nfe) if int(target_nfe) > 0 else int(current_nfe_budget)
        return (resolved,)
    if budget_mode == "multi_budget":
        resolved_list = [int(value) for value in target_nfe_list if int(value) > 0]
        if not resolved_list:
            raise ValueError(
                "solver_aware_budget_mode=multi_budget requires --solver_aware_target_nfe_list."
            )
        return tuple(sorted(dict.fromkeys(resolved_list)))
    raise ValueError(f"Unsupported budget_mode={budget_mode}.")


def build_defect_fixed_point_profile(
    *,
    mode: str,
    current_nfe_budget: int,
    velocity_model,
    data_loader: Iterable,
    device,
    path_family: str,
    target_solver: str,
    budget_mode: str,
    target_nfe: int,
    target_nfe_list: Sequence[int],
    target_nfe_weights: Optional[Sequence[float]],
    grid_size: int,
    batch_size: int,
    eps: float,
    cfg_scale: float,
    checkpoint_source: str,
    seed: int,
    defect_subdivide: int,
    stork_effective_order: float,
    output_dir: Optional[Path] = None,
) -> DefectFixedPointProfile:
    del output_dir
    target_nfes = resolve_defect_target_nfes(
        budget_mode=budget_mode,
        current_nfe_budget=current_nfe_budget,
        target_nfe=target_nfe,
        target_nfe_list=target_nfe_list,
    )
    defect_monitor = compute_defect_monitor(
        velocity_model=velocity_model,
        data_loader=data_loader,
        device=device,
        path_family=path_family,
        grid_size=grid_size,
        batch_size=batch_size,
        seed=seed,
        target_solver=target_solver,
        budget_mode=budget_mode,
        target_nfes=target_nfes,
        target_nfe_weights=target_nfe_weights,
        cfg_scale=cfg_scale,
        defect_subdivide=defect_subdivide,
        stork_effective_order=stork_effective_order,
    )
    defect_clock = build_defect_clock_profile(
        s_grid=defect_monitor.s_grid,
        q_values_by_budget={
            int(key): value
            for key, value in defect_monitor.q_values_by_budget.items()
        },
        budget_step_count_by_nfe={
            int(key): int(value)
            for key, value in defect_monitor.budget_step_count_by_nfe.items()
        },
        budget_mode=budget_mode,
        order=defect_monitor.order,
        eps=eps,
        target_nfe_weights=target_nfe_weights,
    )
    normalization_note = (
        "Raw NFE budget B is mapped to the solver-specific effective step count K_S(B). "
        "Normalized defect curves use effective step count scaling K_S(B)^(2p+2) rather than raw-NFE scaling. "
        f"The primary target_nfe_budget={int(defect_clock.target_nfe_budget)} uses "
        f"target_step_count={int(defect_clock.target_step_count)}."
    )
    return DefectFixedPointProfile(
        mode=str(mode),
        target_solver=str(target_solver),
        budget_mode=str(budget_mode),
        target_nfes=tuple(int(value) for value in target_nfes),
        target_nfe_budget=int(defect_clock.target_nfe_budget),
        target_step_count=int(defect_clock.target_step_count),
        target_nfe_weights=dict(defect_monitor.target_nfe_weights),
        theorem_backed=bool(defect_monitor.theorem_backed),
        notes=_append_note(str(defect_monitor.notes), normalization_note),
        checkpoint_source=str(checkpoint_source),
        defect_subdivide=int(defect_monitor.defect_subdivide),
        solver_order=float(defect_monitor.order),
        q_curve_name=str(defect_clock.q_curve_name),
        aggregation_name=str(defect_clock.aggregation_name),
        s_grid=defect_clock.s_grid,
        q_raw=defect_clock.q_raw,
        q_smoothed=defect_clock.q_smoothed,
        density=defect_clock.density,
        phi=defect_clock.phi,
        density_exponent=float(defect_clock.density_exponent),
        smoothing_window=int(defect_clock.smoothing_window),
        q_values_by_budget=dict(defect_clock.q_values_by_budget),
        q_normalized_by_budget=dict(defect_clock.q_normalized_by_budget),
        budget_step_count_by_nfe={
            key: int(value)
            for key, value in defect_monitor.budget_step_count_by_nfe.items()
        },
        budget_weights=dict(defect_clock.budget_weights),
        distribution_info=dict(defect_monitor.distribution_info),
    )
