import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from training.fixed_step_solver import (
    get_tail_step_methods,
    is_exact_budget,
)
from training.solver_aware.clock import (
    build_solver_aware_clock_profile,
    build_solver_aware_nodes,
)
from training.solver_aware.fixed_point_defect import build_defect_fixed_point_profile
from training.solver_aware.monitors import (
    MonitorArtifacts,
    compute_euler_monitor,
    compute_heun2_monitor,
)
from training.solver_aware.stork_hybrid import (
    build_stork_hybrid_metadata,
    build_stork_hybrid_nodes,
)


logger = logging.getLogger(__name__)


@dataclass
class SolverAwareProfile:
    mode: str
    target_solver: str
    monitor_solver: str
    estimator: str
    theorem_backed: bool
    notes: str
    checkpoint_source: str
    grid_size: int
    batch_size: int
    eps: float
    q_values: Tensor
    q_smoothed: Tensor
    density: Tensor
    s_grid: Tensor
    phi: Tensor
    density_exponent: float
    smoothing_window: int
    monitor_family: str = "legacy_continuous"
    budget_mode: str = "single_budget"
    target_nfe: int = 0
    target_nfe_list: tuple[int, ...] = ()
    target_nfe_weights: Dict[str, float] = field(default_factory=dict)
    target_step_count: int = 0
    budget_step_count_by_nfe: Dict[str, int] = field(default_factory=dict)
    defect_subdivide: int = 2
    solver_order: float = 0.0
    q_curve_name: str = "Q"
    aggregation_name: str = ""
    q_values_by_budget: Dict[str, Tensor] = field(default_factory=dict)
    q_normalized_by_budget: Dict[str, Tensor] = field(default_factory=dict)
    budget_weights: Dict[str, float] = field(default_factory=dict)
    distribution_info: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in ("q_values", "q_smoothed", "density", "s_grid", "phi"):
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


@dataclass
class SolverAwareArtifacts(SolverAwareProfile):
    sampling_solver: str = ""
    sampling_nfe_budget: int = 0
    step_count: int = 0
    r_grid: Tensor = field(default_factory=lambda: torch.empty(0))
    nodes: Tensor = field(default_factory=lambda: torch.empty(0))

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in ("q_values", "q_smoothed", "density", "s_grid", "phi", "r_grid", "nodes"):
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


def _tensor_dict_to_lists(values: Dict[str, Tensor]) -> Dict[str, list[float]]:
    return {
        str(key): [float(item) for item in tensor.detach().cpu().tolist()]
        for key, tensor in values.items()
    }


def _curve_summary(values: Tensor) -> Dict[str, float]:
    detached = values.detach().to(dtype=torch.float64)
    return {
        "min": float(detached.min().item()),
        "mean": float(detached.mean().item()),
        "max": float(detached.max().item()),
    }


def _tail(values: Tensor, count: int = 8) -> list[float]:
    tail = values.detach().cpu().tolist()[-count:]
    return [float(value) for value in tail]


def _append_note_segments(base: str, extra_segments) -> str:
    base_text = str(base).strip()
    segments = [base_text] if base_text else []
    for segment in extra_segments:
        segment_text = str(segment).strip()
        if not segment_text:
            continue
        if base_text and segment_text in base_text:
            continue
        segments.append(segment_text)
    return " ".join(segments)


def _collect_defect_budget_warnings(
    *,
    target_solver: str,
    target_nfe_list: tuple[int, ...],
) -> tuple[list[str], Dict[str, object]]:
    if target_solver not in {"heun2", "rk3"}:
        return [], {
            "exact_budget_only_warning": False,
            "mixed_tail_target_nfe_budgets": [],
            "mixed_tail_step_methods_by_budget": {},
        }

    warnings: list[str] = []
    mixed_tail_budgets: list[int] = []
    mixed_tail_methods: Dict[str, list[str]] = {}
    for budget in target_nfe_list:
        if is_exact_budget(target_solver, int(budget)):
            continue
        tail_methods = list(get_tail_step_methods(target_solver, int(budget)))
        mixed_tail_budgets.append(int(budget))
        mixed_tail_methods[str(int(budget))] = tail_methods
        warnings.append(
            "Current defect theorem is exact-budget only, but "
            f"target_solver={target_solver} uses raw NFE budget {int(budget)} with mixed tail step "
            f"{tail_methods or ['unknown']}. Results are heuristic rather than theorem-backed."
        )
    return warnings, {
        "exact_budget_only_warning": len(warnings) > 0,
        "mixed_tail_target_nfe_budgets": mixed_tail_budgets,
        "mixed_tail_step_methods_by_budget": mixed_tail_methods,
    }


def _collect_cross_solver_warning(
    *,
    monitor_family: str,
    target_solver: str,
    sampling_solver: str,
) -> tuple[list[str], Dict[str, object]]:
    if monitor_family != "defect_based" or not sampling_solver or target_solver == sampling_solver:
        return [], {
            "cross_solver_heuristic_transfer": False,
            "sampling_solver": str(sampling_solver),
        }
    return [
        "Current clock is built for one solver but consumed by another solver. "
        f"target_solver={target_solver}, sampling_solver={sampling_solver}. "
        "The theorem-backed interpretation only applies when target_solver == sampling_solver, "
        "so this run should be treated as cross-solver heuristic transfer."
    ], {
        "cross_solver_heuristic_transfer": True,
        "sampling_solver": str(sampling_solver),
    }


def _compute_step_sizes(nodes: Tensor) -> Tensor:
    step_sizes = torch.zeros_like(nodes)
    if nodes.numel() > 1:
        step_sizes[1:] = nodes[1:] - nodes[:-1]
    return step_sizes


def _stork_payload_fields(artifacts: SolverAwareArtifacts) -> Dict[str, object]:
    distribution_info = artifacts.distribution_info
    return {
        "stork_hybrid_clock": bool(distribution_info.get("stork_hybrid_clock", False)),
        "cold_start_threshold": float(distribution_info.get("cold_start_threshold", 0.0)),
        "warm_region_start": float(distribution_info.get("warm_region_start", 0.0)),
        "warm_region_enabled": bool(distribution_info.get("warm_region_enabled", False)),
        "warm_macro_step_count": int(distribution_info.get("warm_macro_step_count", 0)),
        "cold_start_fixed_step": bool(distribution_info.get("cold_start_fixed_step", False)),
        "stork_warm_defect": bool(distribution_info.get("stork_warm_defect", False)),
        "stork_warm_state_heuristic": bool(
            distribution_info.get("stork_warm_state_heuristic", False)
        ),
    }


def _build_artifact_json_payload(artifacts: SolverAwareArtifacts) -> Dict[str, object]:
    payload = {
        "mode": artifacts.mode,
        "target_solver": artifacts.target_solver,
        "monitor_solver": artifacts.monitor_solver,
        "sampling_solver": artifacts.sampling_solver,
        "estimator": artifacts.estimator,
        "theorem_backed": artifacts.theorem_backed,
        "notes": artifacts.notes,
        "checkpoint_source": artifacts.checkpoint_source,
        "grid_size": artifacts.grid_size,
        "batch_size": artifacts.batch_size,
        "eps": artifacts.eps,
        "sampling_nfe_budget": artifacts.sampling_nfe_budget,
        "sampling_step_count": artifacts.step_count,
        "step_count": artifacts.step_count,
        "monitor_family": artifacts.monitor_family,
        "budget_mode": artifacts.budget_mode,
        "target_nfe": artifacts.target_nfe,
        "target_nfe_budget": artifacts.target_nfe,
        "target_nfe_list": [int(value) for value in artifacts.target_nfe_list],
        "target_nfe_weights": artifacts.target_nfe_weights,
        "target_step_count": artifacts.target_step_count,
        "budget_step_count_by_nfe": artifacts.budget_step_count_by_nfe,
        "defect_subdivide": artifacts.defect_subdivide,
        "solver_order": artifacts.solver_order,
        "q_curve_name": artifacts.q_curve_name,
        "aggregation_name": artifacts.aggregation_name,
        "density_exponent": artifacts.density_exponent,
        "smoothing_window": artifacts.smoothing_window,
        "s_grid": [float(value) for value in artifacts.s_grid.detach().cpu().tolist()],
        "q_values": [float(value) for value in artifacts.q_values.detach().cpu().tolist()],
        "q_smoothed": [float(value) for value in artifacts.q_smoothed.detach().cpu().tolist()],
        "q_values_by_budget": _tensor_dict_to_lists(artifacts.q_values_by_budget),
        "q_normalized_by_budget": _tensor_dict_to_lists(artifacts.q_normalized_by_budget),
        "budget_weights": artifacts.budget_weights,
        "density": [float(value) for value in artifacts.density.detach().cpu().tolist()],
        "phi": [float(value) for value in artifacts.phi.detach().cpu().tolist()],
        "q_summary": _curve_summary(artifacts.q_values),
        "q_smoothed_summary": _curve_summary(artifacts.q_smoothed),
        "density_summary": _curve_summary(artifacts.density),
        "phi_summary": _curve_summary(artifacts.phi),
        "distribution_info": artifacts.distribution_info,
        "terminal_aware_step": bool(artifacts.distribution_info.get("terminal_aware_step", False)),
        "monitor_loaded_from_cache": bool(
            artifacts.distribution_info.get("monitor_loaded_from_cache", False)
        ),
        "monitor_used_eval_loader": bool(
            artifacts.distribution_info.get("monitor_used_eval_loader", False)
        ),
        "warning_messages": list(artifacts.distribution_info.get("warning_messages", [])),
    }
    payload.update(_stork_payload_fields(artifacts))
    return payload


def _build_artifact_csv_text(artifacts: SolverAwareArtifacts) -> str:
    rows = ["grid_index,s_value,q_value,q_smoothed,density,phi,q_curve_name"]
    s_values = artifacts.s_grid.detach().cpu().tolist()
    q_values = artifacts.q_values.detach().cpu().tolist()
    q_smoothed = artifacts.q_smoothed.detach().cpu().tolist()
    density = artifacts.density.detach().cpu().tolist()
    phi = artifacts.phi.detach().cpu().tolist()
    for index, (s_value, q_value, q_smooth, density_value, phi_value) in enumerate(
        zip(s_values, q_values, q_smoothed, density, phi)
    ):
        rows.append(
            f"{index},{float(s_value)},{float(q_value)},{float(q_smooth)},{float(density_value)},{float(phi_value)},{artifacts.q_curve_name}"
        )
    return "\n".join(rows) + "\n"


def _build_node_json_payload(artifacts: SolverAwareArtifacts) -> Dict[str, object]:
    step_sizes = _compute_step_sizes(artifacts.nodes)
    positive_steps = step_sizes[1:]
    uniform_step = 1.0 / float(max(1, artifacts.step_count))
    max_step = float(positive_steps.max().item()) if positive_steps.numel() > 0 else 0.0
    min_positive_step = (
        float(positive_steps.clamp(min=torch.finfo(positive_steps.dtype).eps).min().item())
        if positive_steps.numel() > 0
        else 0.0
    )
    payload = {
        "mode": artifacts.mode,
        "target_solver": artifacts.target_solver,
        "monitor_solver": artifacts.monitor_solver,
        "sampling_solver": artifacts.sampling_solver,
        "estimator": artifacts.estimator,
        "theorem_backed": artifacts.theorem_backed,
        "notes": artifacts.notes,
        "checkpoint_source": artifacts.checkpoint_source,
        "monitor_family": artifacts.monitor_family,
        "budget_mode": artifacts.budget_mode,
        "sampling_nfe_budget": artifacts.sampling_nfe_budget,
        "sampling_step_count": artifacts.step_count,
        "target_nfe": artifacts.target_nfe,
        "target_nfe_budget": artifacts.target_nfe,
        "target_nfe_list": [int(value) for value in artifacts.target_nfe_list],
        "target_step_count": artifacts.target_step_count,
        "budget_step_count_by_nfe": artifacts.budget_step_count_by_nfe,
        "step_count": artifacts.step_count,
        "r_grid": [float(value) for value in artifacts.r_grid.detach().cpu().tolist()],
        "nodes": [float(value) for value in artifacts.nodes.detach().cpu().tolist()],
        "step_sizes": [float(value) for value in step_sizes.detach().cpu().tolist()],
        "warning_messages": list(artifacts.distribution_info.get("warning_messages", [])),
        "diagnostics": {
            "uniform_step": float(uniform_step),
            "max_step": float(max_step),
            "min_positive_step": float(min_positive_step),
            "max_step_over_uniform": float(max_step / uniform_step) if uniform_step > 0.0 else 0.0,
            "max_step_over_min_positive": (
                float(max_step / min_positive_step) if min_positive_step > 0.0 else 0.0
            ),
        },
    }
    payload.update(_stork_payload_fields(artifacts))
    return payload


def _build_node_csv_text(artifacts: SolverAwareArtifacts) -> str:
    rows = ["node_index,r_value,s_value,step_size_from_prev"]
    step_sizes = _compute_step_sizes(artifacts.nodes).detach().cpu().tolist()
    r_values = artifacts.r_grid.detach().cpu().tolist()
    s_values = artifacts.nodes.detach().cpu().tolist()
    for index, (r_value, s_value, step_size) in enumerate(zip(r_values, s_values, step_sizes)):
        rows.append(
            f"{index},{float(r_value)},{float(s_value)},{float(step_size)}"
        )
    return "\n".join(rows) + "\n"


def _build_budget_curve_json_payload(artifacts: SolverAwareArtifacts) -> Dict[str, object]:
    payload = {
        "monitor_family": artifacts.monitor_family,
        "budget_mode": artifacts.budget_mode,
        "target_solver": artifacts.target_solver,
        "sampling_solver": artifacts.sampling_solver,
        "sampling_nfe_budget": artifacts.sampling_nfe_budget,
        "sampling_step_count": artifacts.step_count,
        "solver_order": artifacts.solver_order,
        "target_nfe_budget": artifacts.target_nfe,
        "target_step_count": artifacts.target_step_count,
        "budget_step_count_by_nfe": artifacts.budget_step_count_by_nfe,
        "normalized_by_effective_step_count_scaling": True,
        "effective_step_count_scaling_exponent": float(2.0 * (artifacts.solver_order + 1.0)),
        "s_grid": [float(value) for value in artifacts.s_grid.detach().cpu().tolist()],
        "q_values_by_budget": _tensor_dict_to_lists(artifacts.q_values_by_budget),
        "q_normalized_by_budget": _tensor_dict_to_lists(artifacts.q_normalized_by_budget),
        "budget_weights": artifacts.budget_weights,
        "q_curve_name": artifacts.q_curve_name,
        "aggregation_name": artifacts.aggregation_name,
        "notes": artifacts.notes,
        "distribution_info": artifacts.distribution_info,
    }
    payload.update(_stork_payload_fields(artifacts))
    return payload


def _build_budget_curve_csv_text(artifacts: SolverAwareArtifacts) -> str:
    rows = [
        "grid_index,s_value,budget,target_nfe_budget,target_step_count,q_value,q_normalized_by_effective_step_count_scaling"
    ]
    s_values = artifacts.s_grid.detach().cpu().tolist()
    for budget, q_curve in sorted(artifacts.q_values_by_budget.items(), key=lambda item: int(item[0])):
        normalized_curve = artifacts.q_normalized_by_budget.get(
            budget,
            torch.zeros_like(q_curve),
        )
        target_step_count = int(artifacts.budget_step_count_by_nfe.get(str(budget), 0))
        for index, (s_value, q_value, q_normalized) in enumerate(
            zip(
                s_values,
                q_curve.detach().cpu().tolist(),
                normalized_curve.detach().cpu().tolist(),
            )
        ):
            rows.append(
                f"{index},{float(s_value)},{int(budget)},{int(budget)},{target_step_count},{float(q_value)},{float(q_normalized)}"
            )
    return "\n".join(rows) + "\n"


def _cache_signature(
    *,
    mode: str,
    target_solver: str,
    monitor_solver: str,
    estimator: str,
    monitor_family: str,
    budget_mode: str,
    target_nfe: int,
    target_nfe_list: tuple[int, ...],
    target_nfe_weights: Dict[str, float],
    checkpoint_source: str,
    path_family: str,
    clock_family: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    defect_subdivide: int,
    solver_order: float,
    seed: int,
) -> Dict[str, object]:
    return {
        "mode": mode,
        "target_solver": target_solver,
        "monitor_solver": monitor_solver,
        "estimator": estimator,
        "monitor_family": monitor_family,
        "budget_mode": budget_mode,
        "target_nfe": int(target_nfe),
        "target_nfe_list": [int(value) for value in target_nfe_list],
        "target_nfe_weights": dict(target_nfe_weights),
        "checkpoint_source": checkpoint_source,
        "path_family": path_family,
        "clock_family": clock_family,
        "grid_size": int(grid_size),
        "batch_size": int(batch_size),
        "eps": float(eps),
        "defect_subdivide": int(defect_subdivide),
        "solver_order": float(solver_order),
        "seed": int(seed),
    }


def _normalize_cache_path(cache_path: str) -> Optional[Path]:
    if cache_path in {"", "none", "None", None}:
        return None
    return Path(str(cache_path))


def _resolve_profile_cache_path(
    *,
    cache_path: str,
    output_dir: Optional[Path],
    monitor_family: str,
    target_solver: str,
    monitor_solver: str,
) -> Optional[Path]:
    explicit_path = _normalize_cache_path(cache_path=cache_path)
    if explicit_path is not None:
        return explicit_path
    if output_dir is None:
        return None
    return (
        output_dir.parent
        / f"solver_aware_profile_{monitor_family}_{target_solver}_{monitor_solver}.pt"
    )


def _load_cache(cache_path: Path, signature: Dict[str, object]) -> Optional[SolverAwareProfile]:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("signature") != signature:
        logger.info("Ignoring solver-aware cache %s because the signature no longer matches.", cache_path)
        return None
    artifact_payload = payload["artifacts"]
    expected_fields = {field.name for field in fields(SolverAwareProfile)}
    missing_fields = sorted(expected_fields.difference(artifact_payload.keys()))
    if missing_fields:
        logger.info(
            "Ignoring solver-aware cache %s because it is missing profile fields: %s",
            cache_path,
            ", ".join(missing_fields),
        )
        return None
    normalized_payload = {
        key: artifact_payload[key]
        for key in expected_fields
    }
    return SolverAwareProfile(**normalized_payload)


def _save_cache(cache_path: Path, signature: Dict[str, object], artifacts: SolverAwareProfile) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "artifacts": artifacts.to_dict()}, cache_path)


def _resolve_monitor(target_solver: str) -> Dict[str, object]:
    if target_solver == "euler":
        return {
            "monitor_solver": "euler",
            "theorem_backed": True,
            "notes": (
                "Euler uses the theorem-backed solver-aware clock induced by L_u u and "
                "rho_E(s) propto (Q_E(s)+eps)^(1/4)."
            ),
        }
    if target_solver == "heun2":
        return {
            "monitor_solver": "heun2",
            "theorem_backed": True,
            "notes": (
                "Heun2 uses the theorem-backed solver-aware clock induced by L_u^2 u and "
                "rho_H(s) propto (Q_H(s)+eps)^(1/6)."
            ),
        }
    if target_solver == "stork4":
        return {
            "monitor_solver": "heun2",
            "theorem_backed": False,
            "notes": (
                "Phase-1 STORK4 does not claim a solver-specific optimal monitor theorem. "
                "It reuses the Heun2 monitor as a heuristic node generator, but STORK first macro-step "
                "is a fixed cold-start Euler step and only warm-stage macro-steps participate in hybrid "
                "warm-only solver-aware allocation on [1 / K_STORK(B), 1]. The theorem-backed "
                "interpretation does not apply."
            ),
        }
    raise ValueError(f"Unsupported solver-aware target solver {target_solver}.")


def _compute_monitor(
    *,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    target_solver: str,
    grid_size: int,
    batch_size: int,
    estimator: str,
    cfg_scale: float,
    seed: int,
) -> MonitorArtifacts:
    if target_solver == "euler":
        return compute_euler_monitor(
            velocity_model=velocity_model,
            data_loader=data_loader,
            device=device,
            path_family=path_family,
            grid_size=grid_size,
            batch_size=batch_size,
            estimator=estimator,
            cfg_scale=cfg_scale,
            seed=seed,
        )
    if target_solver == "heun2":
        return compute_heun2_monitor(
            velocity_model=velocity_model,
            data_loader=data_loader,
            device=device,
            path_family=path_family,
            grid_size=grid_size,
            batch_size=batch_size,
            estimator=estimator,
            cfg_scale=cfg_scale,
            seed=seed,
        )
    raise ValueError(f"Unsupported monitor solver {target_solver}.")


def _merge_monitor_and_clock_profile(
    *,
    mode: str,
    target_solver: str,
    monitor_solver: str,
    theorem_backed: bool,
    notes: str,
    checkpoint_source: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    monitor: MonitorArtifacts,
    q_smoothed: Tensor,
    density: Tensor,
    phi: Tensor,
    density_exponent: float,
    smoothing_window: int,
) -> SolverAwareProfile:
    return SolverAwareProfile(
        mode=mode,
        target_solver=target_solver,
        monitor_solver=monitor_solver,
        estimator=monitor.resolved_estimator,
        theorem_backed=theorem_backed,
        notes=notes,
        checkpoint_source=checkpoint_source,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        q_values=monitor.q_values,
        q_smoothed=q_smoothed,
        density=density,
        s_grid=monitor.s_grid,
        phi=phi,
        density_exponent=float(density_exponent),
        smoothing_window=int(smoothing_window),
        monitor_family="legacy_continuous",
        budget_mode="single_budget",
        target_step_count=0,
        solver_order=1.0 if target_solver == "euler" else 2.0,
        q_curve_name="Q_continuous",
        aggregation_name="legacy_continuous",
    )


def _merge_defect_profile(
    defect_profile,
    *,
    estimator: str,
    grid_size: int,
    batch_size: int,
    eps: float,
) -> SolverAwareProfile:
    target_nfe = int(defect_profile.target_nfe_budget)
    target_step_count = int(defect_profile.target_step_count)
    return SolverAwareProfile(
        mode=defect_profile.mode,
        target_solver=defect_profile.target_solver,
        monitor_solver=defect_profile.target_solver,
        estimator=str(estimator),
        theorem_backed=bool(defect_profile.theorem_backed),
        notes=str(defect_profile.notes),
        checkpoint_source=str(defect_profile.checkpoint_source),
        grid_size=int(grid_size),
        batch_size=int(batch_size),
        eps=float(eps),
        q_values=defect_profile.q_raw,
        q_smoothed=defect_profile.q_smoothed,
        density=defect_profile.density,
        s_grid=defect_profile.s_grid,
        phi=defect_profile.phi,
        density_exponent=float(defect_profile.density_exponent),
        smoothing_window=int(defect_profile.smoothing_window),
        monitor_family="defect_based",
        budget_mode=str(defect_profile.budget_mode),
        target_nfe=target_nfe,
        target_nfe_list=tuple(int(value) for value in defect_profile.target_nfes),
        target_nfe_weights=dict(defect_profile.target_nfe_weights),
        target_step_count=target_step_count,
        budget_step_count_by_nfe=dict(defect_profile.budget_step_count_by_nfe),
        defect_subdivide=int(defect_profile.defect_subdivide),
        solver_order=float(defect_profile.solver_order),
        q_curve_name=str(defect_profile.q_curve_name),
        aggregation_name=str(defect_profile.aggregation_name),
        q_values_by_budget=dict(defect_profile.q_values_by_budget),
        q_normalized_by_budget=dict(defect_profile.q_normalized_by_budget),
        budget_weights=dict(defect_profile.budget_weights),
        distribution_info=dict(defect_profile.distribution_info),
    )


def _materialize_solver_aware_artifacts(
    profile: SolverAwareProfile,
    sampling_solver: str,
    sampling_nfe_budget: int,
    step_count: int,
) -> SolverAwareArtifacts:
    stork_context = str(sampling_solver) == "stork4" or str(profile.target_solver) == "stork4"
    distribution_info = dict(profile.distribution_info)
    if stork_context:
        r_grid, nodes, stork_metadata = build_stork_hybrid_nodes(
            s_grid=profile.s_grid,
            phi=profile.phi,
            step_count=step_count,
        )
    else:
        r_grid, nodes = build_solver_aware_nodes(
            s_grid=profile.s_grid,
            phi=profile.phi,
            node_count=step_count + 1,
        )
        stork_metadata = build_stork_hybrid_metadata(
            step_count=step_count,
            stork_context=False,
            hybrid_used=False,
            warm_defect=False,
            warm_state_heuristic=False,
        )
    distribution_info.update(stork_metadata)
    return SolverAwareArtifacts(
        mode=profile.mode,
        target_solver=profile.target_solver,
        monitor_solver=profile.monitor_solver,
        sampling_solver=str(sampling_solver),
        sampling_nfe_budget=int(sampling_nfe_budget),
        estimator=profile.estimator,
        theorem_backed=profile.theorem_backed,
        notes=profile.notes,
        checkpoint_source=profile.checkpoint_source,
        grid_size=profile.grid_size,
        batch_size=profile.batch_size,
        eps=profile.eps,
        q_values=profile.q_values,
        q_smoothed=profile.q_smoothed,
        density=profile.density,
        s_grid=profile.s_grid,
        phi=profile.phi,
        density_exponent=profile.density_exponent,
        smoothing_window=profile.smoothing_window,
        monitor_family=profile.monitor_family,
        budget_mode=profile.budget_mode,
        target_nfe=profile.target_nfe,
        target_nfe_list=profile.target_nfe_list,
        target_nfe_weights=profile.target_nfe_weights,
        target_step_count=profile.target_step_count,
        budget_step_count_by_nfe=profile.budget_step_count_by_nfe,
        defect_subdivide=profile.defect_subdivide,
        solver_order=profile.solver_order,
        q_curve_name=profile.q_curve_name,
        aggregation_name=profile.aggregation_name,
        q_values_by_budget=profile.q_values_by_budget,
        q_normalized_by_budget=profile.q_normalized_by_budget,
        budget_weights=profile.budget_weights,
        distribution_info=distribution_info,
        step_count=step_count,
        r_grid=r_grid,
        nodes=nodes,
    )


def maybe_build_solver_aware_artifacts(
    *,
    mode: str,
    k: int,
    use_nodes: bool,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    clock_family: str,
    target_solver: str,
    estimator: str,
    monitor_family: str,
    budget_mode: str,
    target_nfe: int,
    target_nfe_list,
    target_nfe_weights,
    grid_size: int,
    batch_size: int,
    eps: float,
    cfg_scale: float,
    nfe_budget: int,
    step_count: int,
    checkpoint_source: str,
    seed: int,
    cache_path: str,
    sampling_solver: str,
    using_eval_loader_for_monitor: bool,
    require_cache_hit: bool,
    refuse_recompute_when_cache_exists: bool,
    stork_effective_order: float,
    defect_subdivide: int,
    output_dir: Optional[Path] = None,
) -> Optional[SolverAwareArtifacts]:
    """Build phase-1 solver-aware nodes without touching the legacy FT-clock branch.

    The fixed-point interface is intentionally split from training-free mode:
    - training_free / k=0: use a frozen checkpoint u, estimate Q, build phi, sample.
    - fixed_point / k>=1: reserved for future damped fixed-point retraining.
    """
    if not use_nodes or mode == "off":
        return None

    if mode == "fixed_point" and int(k) > 0:
        raise NotImplementedError(
            "solver_aware_clock_mode=fixed_point with k>=1 is reserved for a future retraining "
            "phase. Phase-1 only implements k=0 training-free updates."
        )

    effective_mode = "training_free" if mode in {"training_free", "fixed_point"} else mode
    if int(k) != 0:
        logger.warning(
            "Phase-1 solver-aware clock only supports k=0. Received k=%s and will use the training-free update.",
            k,
        )

    resolved_monitor_family = str(monitor_family or "legacy_continuous")
    if resolved_monitor_family not in {"legacy_continuous", "defect_based"}:
        raise ValueError(
            f"Unsupported solver-aware monitor family {resolved_monitor_family}."
        )

    if resolved_monitor_family == "legacy_continuous":
        monitor_spec = _resolve_monitor(target_solver=target_solver)
        effective_target_nfe = 0
        effective_target_nfe_list: tuple[int, ...] = ()
        effective_target_nfe_weights: Dict[str, float] = {}
        effective_defect_subdivide = 2
        effective_solver_order = 1.0 if target_solver == "euler" else 2.0
        effective_estimator = estimator
    else:
        monitor_spec = {
            "monitor_solver": target_solver,
            "theorem_backed": target_solver in {"euler", "heun2"},
            "notes": "defect_based solver-aware monitor",
        }
        effective_target_nfe = int(target_nfe) if int(target_nfe) > 0 else int(nfe_budget)
        if str(budget_mode) == "multi_budget":
            effective_target_nfe_list = tuple(
                int(value) for value in (target_nfe_list or ()) if int(value) > 0
            )
        else:
            effective_target_nfe_list = (effective_target_nfe,)
        if target_nfe_weights is not None and len(target_nfe_weights) == len(effective_target_nfe_list):
            effective_target_nfe_weights = {
                str(int(budget)): float(weight)
                for budget, weight in zip(effective_target_nfe_list, target_nfe_weights)
            }
        else:
            effective_target_nfe_weights = {}
        effective_defect_subdivide = int(defect_subdivide)
        effective_solver_order = (
            float(stork_effective_order)
            if target_solver == "stork4"
            else (1.0 if target_solver == "euler" else 2.0)
        )
        effective_estimator = "defect"

    signature = _cache_signature(
        mode=effective_mode,
        target_solver=target_solver,
        monitor_solver=monitor_spec["monitor_solver"],
        estimator=effective_estimator,
        monitor_family=resolved_monitor_family,
        budget_mode=str(budget_mode),
        target_nfe=effective_target_nfe,
        target_nfe_list=effective_target_nfe_list,
        target_nfe_weights=effective_target_nfe_weights,
        checkpoint_source=checkpoint_source,
        path_family=path_family,
        clock_family=clock_family,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        defect_subdivide=effective_defect_subdivide,
        solver_order=effective_solver_order,
        seed=seed,
    )
    resolved_cache_path = _resolve_profile_cache_path(
        cache_path=cache_path,
        output_dir=output_dir,
        monitor_family=resolved_monitor_family,
        target_solver=target_solver,
        monitor_solver=str(monitor_spec["monitor_solver"]),
    )
    profile: Optional[SolverAwareProfile] = None
    cache_file_exists = resolved_cache_path is not None and resolved_cache_path.exists()
    loaded_from_cache = False
    if resolved_cache_path is not None:
        profile = _load_cache(cache_path=resolved_cache_path, signature=signature)
        if profile is not None:
            loaded_from_cache = True
            logger.info(
                "Loaded solver-aware %s profile from cache %s and will rematerialize nodes under raw_nfe_budget=%d, step_count=%d.",
                resolved_monitor_family,
                resolved_cache_path,
                nfe_budget,
                step_count,
            )
    if profile is None and cache_file_exists and refuse_recompute_when_cache_exists:
        raise RuntimeError(
            "Found an existing solver-aware profile cache, but it could not be reused with the current "
            f"configuration: {resolved_cache_path}. Refusing to recompute a solver-aware monitor during eval "
            f"for monitor_family={resolved_monitor_family}. "
            "Please refresh the cache offline or provide a calibration/train loader."
        )
    if profile is None and require_cache_hit:
        raise RuntimeError(
            "Solver-aware evaluation refuses to build a fresh monitor from the current eval/test loader by "
            f"default for monitor_family={resolved_monitor_family} because that may cause evaluation leakage. "
            "Please precompute the profile "
            f"offline (cache path: {resolved_cache_path if resolved_cache_path is not None else 'none'}), "
            "provide a calibration/train loader, or rerun with "
            "--solver_aware_allow_eval_loader_for_monitor if you explicitly accept that leakage."
        )

    if profile is None:
        if resolved_monitor_family == "legacy_continuous":
            monitor = _compute_monitor(
                velocity_model=velocity_model,
                data_loader=data_loader,
                device=device,
                path_family=path_family,
                target_solver=monitor_spec["monitor_solver"],
                grid_size=grid_size,
                batch_size=batch_size,
                estimator=estimator,
                cfg_scale=cfg_scale,
                seed=seed,
            )
            clock_profile = build_solver_aware_clock_profile(
                s_grid=monitor.s_grid,
                q_values=monitor.q_values,
                density_exponent=monitor.density_exponent,
                eps=eps,
            )
            profile = _merge_monitor_and_clock_profile(
                mode=effective_mode,
                target_solver=target_solver,
                monitor_solver=monitor_spec["monitor_solver"],
                theorem_backed=bool(monitor_spec["theorem_backed"]),
                notes=str(monitor_spec["notes"]),
                checkpoint_source=checkpoint_source,
                grid_size=grid_size,
                batch_size=batch_size,
                eps=eps,
                monitor=monitor,
                q_smoothed=clock_profile.q_smoothed,
                density=clock_profile.density,
                phi=clock_profile.phi,
                density_exponent=clock_profile.density_exponent,
                smoothing_window=clock_profile.smoothing_window,
            )
        else:
            defect_profile = build_defect_fixed_point_profile(
                mode=effective_mode,
                current_nfe_budget=nfe_budget,
                velocity_model=velocity_model,
                data_loader=data_loader,
                device=device,
                path_family=path_family,
                target_solver=target_solver,
                budget_mode=str(budget_mode),
                target_nfe=int(target_nfe),
                target_nfe_list=tuple(int(value) for value in (target_nfe_list or ())),
                target_nfe_weights=tuple(float(value) for value in target_nfe_weights)
                if target_nfe_weights is not None
                else None,
                grid_size=grid_size,
                batch_size=batch_size,
                eps=eps,
                cfg_scale=cfg_scale,
                checkpoint_source=checkpoint_source,
                seed=seed,
                defect_subdivide=int(defect_subdivide),
                stork_effective_order=float(stork_effective_order),
                output_dir=output_dir,
            )
            profile = _merge_defect_profile(
                defect_profile,
                estimator=effective_estimator,
                grid_size=grid_size,
                batch_size=batch_size,
                eps=eps,
            )
        if resolved_cache_path is not None:
            _save_cache(cache_path=resolved_cache_path, signature=signature, artifacts=profile)
            logger.info(
                "Saved solver-aware %s profile cache to %s; future NFEs will reuse the same monitor/clock profile.",
                resolved_monitor_family,
                resolved_cache_path,
            )

    artifacts = _materialize_solver_aware_artifacts(
        profile=profile,
        sampling_solver=sampling_solver,
        sampling_nfe_budget=nfe_budget,
        step_count=step_count,
    )

    runtime_warning_messages: list[str] = []
    runtime_metadata = dict(artifacts.distribution_info)
    runtime_metadata.update(
        {
            "monitor_loaded_from_cache": bool(loaded_from_cache),
            "monitor_used_eval_loader": bool(using_eval_loader_for_monitor and not loaded_from_cache),
            "monitor_cache_path": str(resolved_cache_path) if resolved_cache_path is not None else "",
            "sampling_solver": str(sampling_solver),
            "sampling_nfe_budget": int(nfe_budget),
            "sampling_step_count": int(step_count),
            "target_nfe_budget": int(artifacts.target_nfe),
            "target_step_count": int(artifacts.target_step_count),
            "budget_step_count_by_nfe": dict(artifacts.budget_step_count_by_nfe),
        }
    )
    stork_context = bool(
        str(sampling_solver) == "stork4" or str(target_solver) == "stork4"
    )
    if stork_context:
        runtime_metadata.update(
            build_stork_hybrid_metadata(
                step_count=step_count,
                stork_context=True,
                hybrid_used=bool(runtime_metadata.get("stork_hybrid_clock", False)),
                warm_defect=resolved_monitor_family == "defect_based" and target_solver == "stork4",
                warm_state_heuristic=(
                    resolved_monitor_family == "defect_based" and target_solver == "stork4"
                ),
            )
        )

    if using_eval_loader_for_monitor and not loaded_from_cache:
        runtime_warning_messages.append(
            "Current solver-aware monitor was built from the eval/test loader. This may cause "
            "evaluation leakage; prefer a cached offline profile or a calibration/train loader."
        )

    if resolved_monitor_family == "defect_based":
        runtime_metadata["normalized_by_effective_step_count_scaling"] = True
        runtime_metadata["terminal_aware_step"] = True
        runtime_metadata["effective_step_rule"] = "h_eff(s)=min(1/step_count,1-s)"
        budget_warning_messages, budget_warning_metadata = _collect_defect_budget_warnings(
            target_solver=target_solver,
            target_nfe_list=tuple(int(value) for value in artifacts.target_nfe_list),
        )
        cross_solver_messages, cross_solver_metadata = _collect_cross_solver_warning(
            monitor_family=resolved_monitor_family,
            target_solver=target_solver,
            sampling_solver=sampling_solver,
        )
        runtime_warning_messages.extend(budget_warning_messages)
        runtime_warning_messages.extend(cross_solver_messages)
        runtime_metadata.update(budget_warning_metadata)
        runtime_metadata.update(cross_solver_metadata)

    if stork_context:
        runtime_warning_messages.append(
            "STORK first macro-step is a fixed cold-start Euler step, and only warm-stage macro-steps "
            "participate in solver-aware allocation."
        )
        if bool(runtime_metadata.get("stork_hybrid_clock", False)):
            runtime_warning_messages.append(
                "STORK hybrid clock is active: the cold-start interval [0, 1 / K_STORK(B)] is fixed, "
                "and warm-stage allocation only applies on [1 / K_STORK(B), 1]."
            )
        else:
            runtime_warning_messages.append(
                "STORK warm region is too short to optimize, so node allocation falls back to uniform "
                "spacing after keeping the fixed cold-start step."
            )
        if resolved_monitor_family == "legacy_continuous" and target_solver == "stork4":
            runtime_warning_messages.append(
                "Legacy continuous STORK uses a Heun2-monitor heuristic plus hybrid warm-only node "
                "allocation; theorem-backed interpretation does not apply."
            )
        if resolved_monitor_family == "defect_based" and target_solver == "stork4":
            runtime_warning_messages.append(
                "Defect-based STORK uses a configured-order warm-state heuristic defect, not a strict "
                "theorem-backed STORK defect expansion."
            )

    runtime_metadata["warning_messages"] = list(runtime_warning_messages)
    artifacts.distribution_info = runtime_metadata
    semantic_note_segments = [
        "Here B denotes the raw NFE budget and K_S(B) the solver-specific effective macro-step count.",
        f"Current sampling consumes this clock with sampling_solver={artifacts.sampling_solver}, "
        f"raw_nfe_budget={int(artifacts.sampling_nfe_budget)}, and effective step_count=K_S(B)={int(artifacts.step_count)}."
    ]
    if resolved_monitor_family == "defect_based":
        semantic_note_segments.extend(
            [
                "The actual defect step is terminal-aware with h_eff(s) = min(1 / K_S(B), 1 - s).",
                "Raw NFE budget B is mapped to the solver-specific effective step count K_S(B). "
                "Normalized defect curves use effective step count scaling K_S(B)^(2p+2) rather than raw-NFE scaling. "
                f"The primary target_nfe_budget={int(artifacts.target_nfe)} uses target_step_count={int(artifacts.target_step_count)}.",
            ]
        )
    if stork_context:
        semantic_note_segments.extend(
            [
                f"STORK cold-start threshold is 1 / K_STORK(B) = {float(runtime_metadata.get('cold_start_threshold', 0.0))}.",
                "Only warm-stage macro-steps participate in solver-aware allocation on [1 / K_STORK(B), 1].",
            ]
        )
    artifacts.notes = _append_note_segments(
        artifacts.notes,
        semantic_note_segments + runtime_warning_messages,
    )

    logger.info(
        "Solver-aware monitor source: monitor_family=%s, loaded_from_cache=%s, eval_loader_used=%s, cache_path=%s.",
        resolved_monitor_family,
        bool(loaded_from_cache),
        bool(using_eval_loader_for_monitor and not loaded_from_cache),
        str(resolved_cache_path) if resolved_cache_path is not None else "none",
    )

    if resolved_monitor_family == "defect_based":
        logger.info(
            "Path-defect monitor semantic checks: target_solver=%s, sampling_solver=%s.",
            target_solver,
            sampling_solver,
        )
    for warning_message in runtime_warning_messages:
        logger.warning(warning_message)

    logger.info(
        "Materialized solver-aware nodes for target_solver=%s using %s profile under sampling_solver=%s, raw_nfe_budget=%d, step_count=%d.",
        target_solver,
        profile.monitor_family,
        sampling_solver,
        nfe_budget,
        step_count,
    )
    step_sizes = _compute_step_sizes(artifacts.nodes)
    positive_steps = step_sizes[1:]
    uniform_step = 1.0 / float(max(1, artifacts.step_count))
    max_step = float(positive_steps.max().item()) if positive_steps.numel() > 0 else 0.0
    max_step_over_uniform = float(max_step / uniform_step) if uniform_step > 0.0 else 0.0
    logger.info(
        "Solver-aware diagnostics for %s at raw_nfe_budget=%d and step_count=%d: q[min=%.6f, mean=%.6f, max=%.6f], "
        "density[min=%.6f, mean=%.6f, max=%.6f], max_step=%.6f, max_step_over_uniform=%.6f.",
        target_solver,
        nfe_budget,
        step_count,
        float(_curve_summary(artifacts.q_values)["min"]),
        float(_curve_summary(artifacts.q_values)["mean"]),
        float(_curve_summary(artifacts.q_values)["max"]),
        float(_curve_summary(artifacts.density)["min"]),
        float(_curve_summary(artifacts.density)["mean"]),
        float(_curve_summary(artifacts.density)["max"]),
        max_step,
        max_step_over_uniform,
    )
    logger.info(
        "Solver-aware tails for %s at raw_nfe_budget=%d and step_count=%d: q_smoothed_tail=%s, density_tail=%s, nodes_tail=%s.",
        target_solver,
        nfe_budget,
        step_count,
        _tail(artifacts.q_smoothed),
        _tail(artifacts.density),
        _tail(artifacts.nodes),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts.to_dict(), output_dir / "solver_aware_artifacts.pt")
        (output_dir / "solver_aware_artifacts.json").write_text(
            json.dumps(
                _build_artifact_json_payload(artifacts),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "solver_aware_artifacts.csv").write_text(
            _build_artifact_csv_text(artifacts),
            encoding="utf-8",
        )
        (output_dir / "solver_aware_nodes.json").write_text(
            json.dumps(
                _build_node_json_payload(artifacts),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "solver_aware_nodes.csv").write_text(
            _build_node_csv_text(artifacts),
            encoding="utf-8",
        )
        if artifacts.q_values_by_budget:
            (output_dir / "solver_aware_budget_curves.json").write_text(
                json.dumps(
                    _build_budget_curve_json_payload(artifacts),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "solver_aware_budget_curves.csv").write_text(
                _build_budget_curve_csv_text(artifacts),
                encoding="utf-8",
            )
        if artifacts.distribution_info:
            (output_dir / "solver_aware_distribution_info.json").write_text(
                json.dumps(
                    artifacts.distribution_info,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    if resolved_monitor_family == "legacy_continuous":
        logger.info(
            "Built legacy continuous solver-aware nodes for target_solver=%s using monitor_solver=%s and estimator=%s.",
            target_solver,
            monitor_spec["monitor_solver"],
            profile.estimator,
        )
        logger.info("Solver-aware note: %s", monitor_spec["notes"])
    else:
        logger.info(
            "Built path-based defect solver-aware nodes for target_solver=%s with budget_mode=%s, target_nfe_list=%s, and budget_step_count_by_nfe=%s.",
            target_solver,
            artifacts.budget_mode,
            list(artifacts.target_nfe_list),
            artifacts.budget_step_count_by_nfe,
        )
        logger.info("Solver-aware defect note: %s", artifacts.notes)
    return artifacts
