import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Type, TypeVar

import torch
from torch import Tensor

from training.solver_aware.clock import build_solver_aware_clock
from training.solver_aware.monitors import (
    MonitorArtifacts,
    compute_euler_monitor,
    compute_heun2_monitor,
)
from training.solver_aware.propagation import (
    PropagationArtifacts,
    estimate_jacobian_spectral_envelope,
)


logger = logging.getLogger(__name__)
CacheT = TypeVar("CacheT")


@dataclass
class SolverAwareProfile:
    mode: str
    target_solver: str
    monitor_solver: str
    estimator: str
    theorem_backed: bool
    notes: str
    checkpoint_source: str
    cache_path: str
    grid_size: int
    batch_size: int
    eps: float
    eta: float
    floor_mode: str
    floor_eps: float
    compute_qh_for_euler: bool
    legacy_unconstrained: bool
    use_q_h_for_weight: bool
    density_exponent: float
    propagation_exponent: float
    use_propagation: bool
    g_mode: str
    g_estimator: str
    g_power_iters: int
    g_pool_radius: int
    g_safety_factor: float
    q_values: Tensor
    q_h_values: Optional[Tensor]
    ell_raw: Optional[Tensor]
    ell_env: Optional[Tensor]
    ell_values: Optional[Tensor]
    g_values: Optional[Tensor]
    s_grid: Tensor

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Tensor):
                payload[key] = value.detach().cpu()
        return payload


@dataclass
class SolverAwareArtifacts(SolverAwareProfile):
    q_smoothed: Tensor
    q_h_smoothed: Optional[Tensor]
    rho_floor: Tensor
    unconstrained_weight: Tensor
    density: Tensor
    phi: Tensor
    step_count: int
    r_grid: Tensor
    nodes: Tensor
    step_sizes: Tensor

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Tensor):
                payload[key] = value.detach().cpu()
        payload["final_density"] = payload["density"]
        return payload


def _cache_signature(
    *,
    mode: str,
    target_solver: str,
    monitor_solver: str,
    estimator: str,
    checkpoint_source: str,
    path_family: str,
    clock_family: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    eta: float,
    floor_mode: str,
    floor_eps: float,
    compute_qh_for_euler: bool,
    legacy_unconstrained: bool,
    seed: int,
    use_propagation: bool,
    g_mode: str,
    g_estimator: str,
    g_power_iters: int,
    g_pool_radius: int,
    g_safety_factor: float,
) -> Dict[str, object]:
    return {
        "mode": mode,
        "target_solver": target_solver,
        "monitor_solver": monitor_solver,
        "estimator": estimator,
        "checkpoint_source": checkpoint_source,
        "path_family": path_family,
        "clock_family": clock_family,
        "grid_size": int(grid_size),
        "batch_size": int(batch_size),
        "eps": float(eps),
        "eta": float(eta),
        "floor_mode": str(floor_mode),
        "floor_eps": float(floor_eps),
        "compute_qh_for_euler": bool(compute_qh_for_euler),
        "legacy_unconstrained": bool(legacy_unconstrained),
        "seed": int(seed),
        "use_propagation": bool(use_propagation),
        "g_mode": g_mode,
        "g_estimator": g_estimator,
        "g_power_iters": int(g_power_iters),
        "g_pool_radius": int(g_pool_radius),
        "g_safety_factor": float(g_safety_factor),
    }


def _normalize_cache_path(cache_path: str) -> Optional[Path]:
    if cache_path in {"", "none", "None", None}:
        return None
    return Path(str(cache_path))


def _resolve_profile_cache_path(
    *,
    cache_path: str,
    output_dir: Optional[Path],
    target_solver: str,
    monitor_solver: str,
    use_propagation: bool,
) -> Optional[Path]:
    explicit_path = _normalize_cache_path(cache_path=cache_path)
    if explicit_path is not None:
        return explicit_path
    if output_dir is None:
        return None
    suffix = "_propagation" if use_propagation else ""
    return output_dir.parent / f"solver_aware_profile_{target_solver}_{monitor_solver}{suffix}.pt"


def _resolve_propagation_cache_path(
    *,
    cache_path: str,
    output_dir: Optional[Path],
    target_solver: str,
    monitor_solver: str,
) -> Optional[Path]:
    explicit_path = _normalize_cache_path(cache_path=cache_path)
    if explicit_path is not None:
        return explicit_path
    if output_dir is None:
        return None
    return output_dir.parent / f"solver_aware_propagation_{target_solver}_{monitor_solver}.pt"


def _load_cache(
    cache_path: Path,
    signature: Dict[str, object],
    dataclass_type: Type[CacheT],
) -> Optional[CacheT]:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("signature") != signature:
        logger.info("Ignoring solver-aware cache %s because the signature no longer matches.", cache_path)
        return None
    return dataclass_type(**payload["artifacts"])


def _save_cache(cache_path: Path, signature: Dict[str, object], artifacts) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "artifacts": artifacts.to_dict()}, cache_path)


def _validate_mode(mode: str, k: int) -> str:
    if mode == "training_free" and int(k) != 0:
        raise ValueError("solver_aware_clock_mode=training_free requires solver_aware_k=0.")
    if mode == "fixed_point" and int(k) <= 0:
        logger.warning(
            "solver_aware_clock_mode=fixed_point received k=%s. This falls back to the k=0 training-free update.",
            k,
        )
        return "training_free"
    return mode


def _validate_constrained_args(
    *,
    use_propagation: bool,
    g_mode: str,
    eta: float,
    floor_mode: str,
    floor_eps: float,
) -> None:
    if bool(use_propagation) and str(g_mode) == "none":
        raise ValueError(
            "solver_aware_use_propagation=true requires solver_aware_g_mode to be non-none."
        )
    if float(eta) <= 0.0:
        raise ValueError("solver_aware_eta must be positive.")
    if str(floor_mode) not in {"pointwise", "constant"}:
        raise ValueError(f"Unsupported solver_aware_floor_mode={floor_mode}.")
    if float(floor_eps) <= 0.0:
        raise ValueError("solver_aware_floor_eps must be positive.")


def _resolve_monitor_spec(
    *,
    target_solver: str,
    use_propagation: bool,
    legacy_unconstrained: bool,
) -> Dict[str, object]:
    if target_solver == "euler":
        if legacy_unconstrained:
            return {
                "monitor_solver": "euler",
                "use_q_h_for_weight": False,
                "density_exponent": 0.25,
                "propagation_exponent": 0.5,
                "theorem_backed": False,
                "notes": (
                    "Deprecated legacy debug path: Euler uses the old unconstrained density "
                    "rho(s) propto (Q_E(s)+eps)^(1/4) or G(s)^(1/2)(Q_E(s)+eps)^(1/4). "
                    "The main solver-aware method has moved to the constrained formulation."
                ),
            }
        if use_propagation:
            return {
                "monitor_solver": "euler",
                "use_q_h_for_weight": False,
                "density_exponent": 0.25,
                "propagation_exponent": 0.5,
                "theorem_backed": True,
                "notes": (
                    "Euler constrained propagation-aware clock uses the admissible floor "
                    "rho_floor_N(s) ≈ (1/(3 eta N)) sqrt((Q_H(s)+eps)/(Q_E(s)+eps)) together with "
                    "rho_N*(s)=max{rho_floor_N(s), c_N G(s)^(1/2)(Q_E(s)+eps)^(1/4)}. "
                    "This is the theorem-backed constrained formulation with monitor/proxy estimation."
                ),
            }
        return {
            "monitor_solver": "euler",
            "use_q_h_for_weight": False,
            "density_exponent": 0.25,
            "propagation_exponent": 0.0,
            "theorem_backed": True,
            "notes": (
                "Euler constrained solver-aware clock uses the admissible floor "
                "rho_floor_N(s) ≈ (1/(3 eta N)) sqrt((Q_H(s)+eps)/(Q_E(s)+eps)) together with "
                "rho_N*(s)=max{rho_floor_N(s), c_N (Q_E(s)+eps)^(1/4)}. "
                "This replaces the old unconstrained node allocation."
            ),
        }
    if target_solver == "heun2":
        if legacy_unconstrained:
            return {
                "monitor_solver": "heun2",
                "use_q_h_for_weight": True,
                "density_exponent": 1.0 / 6.0,
                "propagation_exponent": 1.0 / 3.0,
                "theorem_backed": False,
                "notes": (
                    "Deprecated legacy debug path: Heun2 uses the old unconstrained density "
                    "rho(s) propto (Q_H(s)+eps)^(1/6) or G(s)^(1/3)(Q_H(s)+eps)^(1/6)."
                ),
            }
        if use_propagation:
            return {
                "monitor_solver": "heun2",
                "use_q_h_for_weight": True,
                "density_exponent": 1.0 / 6.0,
                "propagation_exponent": 1.0 / 3.0,
                "theorem_backed": False,
                "notes": (
                    "Heun2 uses the constrained proxy extension rho_N*(s)=max{rho_floor_N(s), "
                    "c_N G(s)^(1/3)(Q_H(s)+eps)^(1/6)}. The admissible floor is still built from "
                    "the Euler-style Q_H/Q_E proxy ratio, so this branch is marked proxy-based."
                ),
            }
        return {
            "monitor_solver": "heun2",
            "use_q_h_for_weight": True,
            "density_exponent": 1.0 / 6.0,
            "propagation_exponent": 0.0,
            "theorem_backed": False,
            "notes": (
                "Heun2 uses the constrained proxy extension rho_N*(s)=max{rho_floor_N(s), "
                "c_N (Q_H(s)+eps)^(1/6)} with rho_floor_N built from the Euler-style Q_H/Q_E ratio."
            ),
        }
    if target_solver == "stork4":
        if legacy_unconstrained:
            return {
                "monitor_solver": "heun2",
                "use_q_h_for_weight": True,
                "density_exponent": 1.0 / 6.0,
                "propagation_exponent": 1.0 / 3.0,
                "theorem_backed": False,
                "notes": (
                    "Deprecated legacy debug path: STORK4 reuses the old unconstrained Heun2 proxy density."
                ),
            }
        if use_propagation:
            return {
                "monitor_solver": "heun2",
                "use_q_h_for_weight": True,
                "density_exponent": 1.0 / 6.0,
                "propagation_exponent": 1.0 / 3.0,
                "theorem_backed": False,
                "notes": (
                    "STORK4 consumes constrained propagation-aware nodes built from the Heun2 proxy "
                    "monitor and the same admissible floor framework. This remains heuristic."
                ),
            }
        return {
            "monitor_solver": "heun2",
            "use_q_h_for_weight": True,
            "density_exponent": 1.0 / 6.0,
            "propagation_exponent": 0.0,
            "theorem_backed": False,
            "notes": (
                "STORK4 consumes constrained solver-aware nodes built from the Heun2 proxy "
                "monitor and the same admissible floor framework. This remains heuristic."
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


def _compute_required_monitors(
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
    compute_qh_for_euler: bool,
    legacy_unconstrained: bool,
) -> tuple[MonitorArtifacts, Optional[MonitorArtifacts], MonitorArtifacts]:
    q_e_monitor = _compute_monitor(
        velocity_model=velocity_model,
        data_loader=data_loader,
        device=device,
        path_family=path_family,
        target_solver="euler",
        grid_size=grid_size,
        batch_size=batch_size,
        estimator=estimator,
        cfg_scale=cfg_scale,
        seed=seed + 101,
    )
    need_q_h = (
        target_solver in {"heun2", "stork4"}
        or bool(compute_qh_for_euler)
        or not bool(legacy_unconstrained)
    )
    q_h_monitor = None
    if need_q_h:
        q_h_monitor = _compute_monitor(
            velocity_model=velocity_model,
            data_loader=data_loader,
            device=device,
            path_family=path_family,
            target_solver="heun2",
            grid_size=grid_size,
            batch_size=batch_size,
            estimator=estimator,
            cfg_scale=cfg_scale,
            seed=seed + 307,
        )
    if target_solver == "euler":
        primary = q_e_monitor
    else:
        if q_h_monitor is None:
            raise ValueError(
                f"target_solver={target_solver} requires Q_H(s), but it was not computed."
            )
        primary = q_h_monitor
    if not legacy_unconstrained and q_h_monitor is None:
        raise ValueError(
            "Constrained solver-aware clocks require Q_H(s). "
            "Disable --solver_aware_legacy_unconstrained and keep --solver_aware_compute_qh_for_euler enabled."
        )
    return q_e_monitor, q_h_monitor, primary


def _merge_profile(
    *,
    mode: str,
    target_solver: str,
    theorem_backed: bool,
    notes: str,
    checkpoint_source: str,
    cache_path: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    eta: float,
    floor_mode: str,
    floor_eps: float,
    compute_qh_for_euler: bool,
    legacy_unconstrained: bool,
    use_propagation: bool,
    g_mode: str,
    g_estimator: str,
    g_power_iters: int,
    g_pool_radius: int,
    g_safety_factor: float,
    monitor_spec: Dict[str, object],
    q_e_monitor: MonitorArtifacts,
    q_h_monitor: Optional[MonitorArtifacts],
    primary_monitor: MonitorArtifacts,
    propagation: Optional[PropagationArtifacts],
) -> SolverAwareProfile:
    return SolverAwareProfile(
        mode=mode,
        target_solver=target_solver,
        monitor_solver=str(monitor_spec["monitor_solver"]),
        estimator=primary_monitor.resolved_estimator,
        theorem_backed=theorem_backed,
        notes=notes,
        checkpoint_source=checkpoint_source,
        cache_path=cache_path,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=float(eps),
        eta=float(eta),
        floor_mode=str(floor_mode),
        floor_eps=float(floor_eps),
        compute_qh_for_euler=bool(compute_qh_for_euler),
        legacy_unconstrained=bool(legacy_unconstrained),
        use_q_h_for_weight=bool(monitor_spec["use_q_h_for_weight"]),
        density_exponent=float(monitor_spec["density_exponent"]),
        propagation_exponent=float(monitor_spec["propagation_exponent"]),
        use_propagation=bool(use_propagation),
        g_mode=g_mode if use_propagation else "none",
        g_estimator=g_estimator if use_propagation else "",
        g_power_iters=int(g_power_iters) if use_propagation else 0,
        g_pool_radius=int(g_pool_radius) if use_propagation else 0,
        g_safety_factor=float(g_safety_factor) if use_propagation else 0.0,
        q_values=q_e_monitor.q_values,
        q_h_values=None if q_h_monitor is None else q_h_monitor.q_values,
        ell_raw=None if propagation is None else propagation.raw_ell,
        ell_env=None if propagation is None else propagation.env_ell,
        ell_values=None if propagation is None else propagation.ell_values,
        g_values=None if propagation is None else propagation.g_values,
        s_grid=q_e_monitor.s_grid,
    )


def _materialize_solver_aware_artifacts(
    profile: SolverAwareProfile,
    step_count: int,
) -> SolverAwareArtifacts:
    clock = build_solver_aware_clock(
        s_grid=profile.s_grid,
        q_values=profile.q_values,
        q_h_values=profile.q_h_values,
        use_q_h_for_weight=profile.use_q_h_for_weight,
        density_exponent=profile.density_exponent,
        eps=profile.eps,
        step_count=step_count,
        eta=profile.eta,
        floor_mode=profile.floor_mode,
        floor_eps=profile.floor_eps,
        g_values=profile.g_values,
        propagation_exponent=profile.propagation_exponent if profile.use_propagation else 0.0,
        legacy_unconstrained=profile.legacy_unconstrained,
    )
    return SolverAwareArtifacts(
        mode=profile.mode,
        target_solver=profile.target_solver,
        monitor_solver=profile.monitor_solver,
        estimator=profile.estimator,
        theorem_backed=profile.theorem_backed,
        notes=profile.notes,
        checkpoint_source=profile.checkpoint_source,
        cache_path=profile.cache_path,
        grid_size=profile.grid_size,
        batch_size=profile.batch_size,
        eps=profile.eps,
        eta=profile.eta,
        floor_mode=profile.floor_mode,
        floor_eps=profile.floor_eps,
        compute_qh_for_euler=profile.compute_qh_for_euler,
        legacy_unconstrained=profile.legacy_unconstrained,
        use_q_h_for_weight=profile.use_q_h_for_weight,
        density_exponent=profile.density_exponent,
        propagation_exponent=profile.propagation_exponent,
        use_propagation=profile.use_propagation,
        g_mode=profile.g_mode,
        g_estimator=profile.g_estimator,
        g_power_iters=profile.g_power_iters,
        g_pool_radius=profile.g_pool_radius,
        g_safety_factor=profile.g_safety_factor,
        q_values=profile.q_values,
        q_h_values=profile.q_h_values,
        ell_raw=profile.ell_raw,
        ell_env=profile.ell_env,
        ell_values=profile.ell_values,
        g_values=profile.g_values,
        s_grid=profile.s_grid,
        q_smoothed=clock.q_smoothed,
        q_h_smoothed=clock.q_h_smoothed,
        rho_floor=clock.rho_floor,
        unconstrained_weight=clock.unconstrained_weight,
        density=clock.density,
        phi=clock.phi,
        step_count=int(step_count),
        r_grid=clock.r_grid,
        nodes=clock.nodes,
        step_sizes=clock.step_sizes,
    )


def _node_diagnostics(
    step_sizes: Tensor,
    step_count: int,
) -> Dict[str, float]:
    positive_steps = step_sizes[1:][step_sizes[1:] > 0.0]
    uniform_step = 1.0 / float(max(1, step_count))
    max_step = float(step_sizes.max().item()) if step_sizes.numel() > 0 else 0.0
    min_positive_step = (
        float(positive_steps.min().item()) if positive_steps.numel() > 0 else 0.0
    )
    return {
        "uniform_step": uniform_step,
        "max_step": max_step,
        "min_positive_step": min_positive_step,
        "max_step_over_uniform": max_step / max(uniform_step, 1e-12),
        "max_step_over_min_positive": (
            max_step / max(min_positive_step, 1e-12)
            if min_positive_step > 0.0
            else 0.0
        ),
    }


def maybe_build_solver_aware_profile(
    *,
    mode: str,
    k: int,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    clock_family: str,
    target_solver: str,
    estimator: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    eta: float,
    floor_mode: str,
    floor_eps: float,
    compute_qh_for_euler: bool,
    legacy_unconstrained: bool,
    cfg_scale: float,
    checkpoint_source: str,
    seed: int,
    cache_path: str,
    use_propagation: bool,
    g_mode: str,
    g_estimator: str,
    g_power_iters: int,
    g_pool_radius: int,
    g_safety_factor: float,
    g_cache_path: str,
    output_dir: Optional[Path] = None,
) -> Optional[SolverAwareProfile]:
    """Build the shared constrained solver-aware profile.

    This shared profile caches Q_E(s), Q_H(s) and optional G(s), while the
    constrained density itself is materialized per NFE because the admissible
    floor rho_floor_N(s) depends explicitly on the step count N.
    """
    if mode == "off":
        return None

    effective_mode = _validate_mode(mode=mode, k=k)
    _validate_constrained_args(
        use_propagation=use_propagation,
        g_mode=g_mode,
        eta=eta,
        floor_mode=floor_mode,
        floor_eps=floor_eps,
    )

    monitor_spec = _resolve_monitor_spec(
        target_solver=target_solver,
        use_propagation=use_propagation,
        legacy_unconstrained=legacy_unconstrained,
    )
    signature = _cache_signature(
        mode=effective_mode,
        target_solver=target_solver,
        monitor_solver=str(monitor_spec["monitor_solver"]),
        estimator=estimator,
        checkpoint_source=checkpoint_source,
        path_family=path_family,
        clock_family=clock_family,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        eta=eta,
        floor_mode=floor_mode,
        floor_eps=floor_eps,
        compute_qh_for_euler=compute_qh_for_euler,
        legacy_unconstrained=legacy_unconstrained,
        seed=seed,
        use_propagation=use_propagation,
        g_mode=g_mode if use_propagation else "none",
        g_estimator=g_estimator if use_propagation else "",
        g_power_iters=g_power_iters if use_propagation else 0,
        g_pool_radius=g_pool_radius if use_propagation else 0,
        g_safety_factor=g_safety_factor if use_propagation else 0.0,
    )
    resolved_cache_path = _resolve_profile_cache_path(
        cache_path=cache_path,
        output_dir=output_dir,
        target_solver=target_solver,
        monitor_solver=str(monitor_spec["monitor_solver"]),
        use_propagation=use_propagation,
    )

    profile: Optional[SolverAwareProfile] = None
    if resolved_cache_path is not None:
        profile = _load_cache(
            cache_path=resolved_cache_path,
            signature=signature,
            dataclass_type=SolverAwareProfile,
        )
        if profile is not None:
            profile.cache_path = str(resolved_cache_path)
            logger.info(
                "Loaded solver-aware constrained profile from cache %s",
                resolved_cache_path,
            )

    if profile is None:
        q_e_monitor, q_h_monitor, primary_monitor = _compute_required_monitors(
            velocity_model=velocity_model,
            data_loader=data_loader,
            device=device,
            path_family=path_family,
            target_solver=target_solver,
            grid_size=grid_size,
            batch_size=batch_size,
            estimator=estimator,
            cfg_scale=cfg_scale,
            seed=seed,
            compute_qh_for_euler=compute_qh_for_euler,
            legacy_unconstrained=legacy_unconstrained,
        )

        propagation = None
        if use_propagation:
            propagation_signature = dict(signature)
            resolved_g_cache_path = _resolve_propagation_cache_path(
                cache_path=g_cache_path,
                output_dir=output_dir,
                target_solver=target_solver,
                monitor_solver=str(monitor_spec["monitor_solver"]),
            )
            if resolved_g_cache_path is not None:
                propagation = _load_cache(
                    cache_path=resolved_g_cache_path,
                    signature=propagation_signature,
                    dataclass_type=PropagationArtifacts,
                )
                if propagation is not None:
                    logger.info(
                        "Loaded propagation envelope from cache %s",
                        resolved_g_cache_path,
                    )
            if propagation is None:
                if g_mode != "jacobian_envelope":
                    raise ValueError(f"Unsupported solver_aware_g_mode={g_mode}.")
                propagation = estimate_jacobian_spectral_envelope(
                    velocity_model=velocity_model,
                    data_loader=data_loader,
                    device=device,
                    path_family=path_family,
                    grid_size=grid_size,
                    batch_size=batch_size,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    estimator=g_estimator,
                    power_iters=g_power_iters,
                    pool_radius=g_pool_radius,
                    safety_factor=g_safety_factor,
                )
                if resolved_g_cache_path is not None:
                    _save_cache(
                        cache_path=resolved_g_cache_path,
                        signature=propagation_signature,
                        artifacts=propagation,
                    )

        profile = _merge_profile(
            mode=effective_mode,
            target_solver=target_solver,
            theorem_backed=bool(monitor_spec["theorem_backed"]),
            notes=str(monitor_spec["notes"]),
            checkpoint_source=checkpoint_source,
            cache_path="" if resolved_cache_path is None else str(resolved_cache_path),
            grid_size=grid_size,
            batch_size=batch_size,
            eps=eps,
            eta=eta,
            floor_mode=floor_mode,
            floor_eps=floor_eps,
            compute_qh_for_euler=compute_qh_for_euler,
            legacy_unconstrained=legacy_unconstrained,
            use_propagation=use_propagation,
            g_mode=g_mode if use_propagation else "none",
            g_estimator=g_estimator if use_propagation else "",
            g_power_iters=g_power_iters if use_propagation else 0,
            g_pool_radius=g_pool_radius if use_propagation else 0,
            g_safety_factor=g_safety_factor if use_propagation else 0.0,
            monitor_spec=monitor_spec,
            q_e_monitor=q_e_monitor,
            q_h_monitor=q_h_monitor,
            primary_monitor=primary_monitor,
            propagation=propagation,
        )
        if resolved_cache_path is not None:
            profile.cache_path = str(resolved_cache_path)
            _save_cache(
                cache_path=resolved_cache_path,
                signature=signature,
                artifacts=profile,
            )

    logger.info("Solver-aware note: %s", profile.notes)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(profile.to_dict(), output_dir / "solver_aware_profile.pt")
        (output_dir / "solver_aware_profile.json").write_text(
            json.dumps(
                {
                    "mode": profile.mode,
                    "target_solver": profile.target_solver,
                    "monitor_solver": profile.monitor_solver,
                    "monitor_estimator": profile.estimator,
                    "theorem_backed": profile.theorem_backed,
                    "notes": profile.notes,
                    "checkpoint_source": profile.checkpoint_source,
                    "cache_path": profile.cache_path,
                    "grid_size": profile.grid_size,
                    "batch_size": profile.batch_size,
                    "eps": profile.eps,
                    "eta": profile.eta,
                    "floor_mode": profile.floor_mode,
                    "floor_eps": profile.floor_eps,
                    "compute_qh_for_euler": profile.compute_qh_for_euler,
                    "legacy_unconstrained": profile.legacy_unconstrained,
                    "use_propagation": profile.use_propagation,
                    "g_mode": profile.g_mode,
                    "g_estimator": profile.g_estimator,
                    "g_power_iters": profile.g_power_iters,
                    "g_pool_radius": profile.g_pool_radius,
                    "g_safety_factor": profile.g_safety_factor,
                    "q_values": [float(value) for value in profile.q_values.detach().cpu().tolist()],
                    "q_h_values": (
                        []
                        if profile.q_h_values is None
                        else [float(value) for value in profile.q_h_values.detach().cpu().tolist()]
                    ),
                    "g_values": (
                        []
                        if profile.g_values is None
                        else [float(value) for value in profile.g_values.detach().cpu().tolist()]
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return profile


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
    grid_size: int,
    batch_size: int,
    eps: float,
    eta: float,
    floor_mode: str,
    floor_eps: float,
    compute_qh_for_euler: bool,
    legacy_unconstrained: bool,
    cfg_scale: float,
    step_count: int,
    checkpoint_source: str,
    seed: int,
    cache_path: str,
    use_propagation: bool,
    g_mode: str,
    g_estimator: str,
    g_power_iters: int,
    g_pool_radius: int,
    g_safety_factor: float,
    g_cache_path: str,
    output_dir: Optional[Path] = None,
) -> Optional[SolverAwareArtifacts]:
    if not use_nodes or mode == "off":
        return None

    profile = maybe_build_solver_aware_profile(
        mode=mode,
        k=k,
        velocity_model=velocity_model,
        data_loader=data_loader,
        device=device,
        path_family=path_family,
        clock_family=clock_family,
        target_solver=target_solver,
        estimator=estimator,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        eta=eta,
        floor_mode=floor_mode,
        floor_eps=floor_eps,
        compute_qh_for_euler=compute_qh_for_euler,
        legacy_unconstrained=legacy_unconstrained,
        cfg_scale=cfg_scale,
        checkpoint_source=checkpoint_source,
        seed=seed,
        cache_path=cache_path,
        use_propagation=use_propagation,
        g_mode=g_mode,
        g_estimator=g_estimator,
        g_power_iters=g_power_iters,
        g_pool_radius=g_pool_radius,
        g_safety_factor=g_safety_factor,
        g_cache_path=g_cache_path,
        output_dir=output_dir,
    )
    if profile is None:
        return None

    artifacts = _materialize_solver_aware_artifacts(
        profile=profile,
        step_count=step_count,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = _node_diagnostics(
            step_sizes=artifacts.step_sizes,
            step_count=artifacts.step_count,
        )
        if diagnostics["max_step_over_uniform"] > 2.0:
            logger.warning(
                "Constrained solver-aware nodes still look concentrated for %s: "
                "max_step=%.6f, uniform_step=%.6f, ratio=%.2f.",
                artifacts.target_solver,
                diagnostics["max_step"],
                diagnostics["uniform_step"],
                diagnostics["max_step_over_uniform"],
            )
        torch.save(artifacts.to_dict(), output_dir / "solver_aware_artifacts.pt")
        (output_dir / "solver_aware_artifacts.json").write_text(
            json.dumps(
                {
                    "mode": artifacts.mode,
                    "target_solver": artifacts.target_solver,
                    "monitor_solver": artifacts.monitor_solver,
                    "monitor_estimator": artifacts.estimator,
                    "theorem_backed": artifacts.theorem_backed,
                    "notes": artifacts.notes,
                    "checkpoint_source": artifacts.checkpoint_source,
                    "cache_path": artifacts.cache_path,
                    "grid_size": artifacts.grid_size,
                    "batch_size": artifacts.batch_size,
                    "eps": artifacts.eps,
                    "eta": artifacts.eta,
                    "floor_mode": artifacts.floor_mode,
                    "floor_eps": artifacts.floor_eps,
                    "compute_qh_for_euler": artifacts.compute_qh_for_euler,
                    "legacy_unconstrained": artifacts.legacy_unconstrained,
                    "use_propagation": artifacts.use_propagation,
                    "g_mode": artifacts.g_mode,
                    "g_estimator": artifacts.g_estimator,
                    "g_power_iters": artifacts.g_power_iters,
                    "g_pool_radius": artifacts.g_pool_radius,
                    "g_safety_factor": artifacts.g_safety_factor,
                    "step_count": artifacts.step_count,
                    "diagnostics": diagnostics,
                    "q_values": [float(value) for value in artifacts.q_values.detach().cpu().tolist()],
                    "q_smoothed": [float(value) for value in artifacts.q_smoothed.detach().cpu().tolist()],
                    "q_h_values": (
                        []
                        if artifacts.q_h_values is None
                        else [float(value) for value in artifacts.q_h_values.detach().cpu().tolist()]
                    ),
                    "q_h_smoothed": (
                        []
                        if artifacts.q_h_smoothed is None
                        else [float(value) for value in artifacts.q_h_smoothed.detach().cpu().tolist()]
                    ),
                    "rho_floor": [float(value) for value in artifacts.rho_floor.detach().cpu().tolist()],
                    "unconstrained_weight": [
                        float(value) for value in artifacts.unconstrained_weight.detach().cpu().tolist()
                    ],
                    "final_density": [float(value) for value in artifacts.density.detach().cpu().tolist()],
                    "phi": [float(value) for value in artifacts.phi.detach().cpu().tolist()],
                    "g_values": (
                        []
                        if artifacts.g_values is None
                        else [float(value) for value in artifacts.g_values.detach().cpu().tolist()]
                    ),
                    "r_grid": [float(value) for value in artifacts.r_grid.detach().cpu().tolist()],
                    "nodes": [float(value) for value in artifacts.nodes.detach().cpu().tolist()],
                    "step_sizes": [float(value) for value in artifacts.step_sizes.detach().cpu().tolist()],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        uniform_step = diagnostics["uniform_step"]
        node_lines = ["node_index,r_value,s_value,step_size_from_prev,step_size_over_uniform"]
        r_values = artifacts.r_grid.detach().cpu().tolist()
        node_values = artifacts.nodes.detach().cpu().tolist()
        step_values = artifacts.step_sizes.detach().cpu().tolist()
        for index, (r_value, s_value, step_size) in enumerate(zip(r_values, node_values, step_values)):
            ratio = 0.0 if index == 0 else float(step_size) / max(uniform_step, 1e-12)
            node_lines.append(
                f"{index},{float(r_value):.10f},{float(s_value):.10f},{float(step_size):.10f},{ratio:.10f}"
            )
        (output_dir / "solver_aware_nodes.csv").write_text(
            "\n".join(node_lines) + "\n",
            encoding="utf-8",
        )
        (output_dir / "solver_aware_nodes.json").write_text(
            json.dumps(
                {
                    "step_count": int(artifacts.step_count),
                    "r_grid": [float(value) for value in r_values],
                    "nodes": [float(value) for value in node_values],
                    "step_sizes": [float(value) for value in step_values],
                    "diagnostics": diagnostics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return artifacts
