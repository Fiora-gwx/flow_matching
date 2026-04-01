import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Type, TypeVar

import torch
from torch import Tensor

from training.solver_aware.clock import (
    build_solver_aware_clock_profile,
    build_solver_aware_nodes,
)
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
    use_propagation: bool
    g_mode: str
    g_estimator: str
    g_power_iters: int
    g_pool_radius: int
    g_safety_factor: float
    q_values: Tensor
    q_smoothed: Tensor
    ell_raw: Optional[Tensor]
    ell_env: Optional[Tensor]
    ell_values: Optional[Tensor]
    g_values: Optional[Tensor]
    density: Tensor
    s_grid: Tensor
    phi: Tensor

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in (
            "q_values",
            "q_smoothed",
            "ell_raw",
            "ell_env",
            "ell_values",
            "g_values",
            "density",
            "s_grid",
            "phi",
        ):
            value = payload.get(key)
            if isinstance(value, Tensor):
                payload[key] = value.detach().cpu()
        return payload


@dataclass
class SolverAwareArtifacts(SolverAwareProfile):
    step_count: int
    r_grid: Tensor
    nodes: Tensor

    def to_dict(self) -> Dict[str, object]:
        payload = super().to_dict()
        payload["step_count"] = int(self.step_count)
        payload["r_grid"] = self.r_grid.detach().cpu()
        payload["nodes"] = self.nodes.detach().cpu()
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
    artifact_payload = payload["artifacts"]
    return dataclass_type(**artifact_payload)


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


def _validate_propagation_args(
    *,
    use_propagation: bool,
    g_mode: str,
) -> None:
    if bool(use_propagation) and str(g_mode) == "none":
        raise ValueError(
            "solver_aware_use_propagation=true requires solver_aware_g_mode to be non-none."
        )


def _resolve_monitor(target_solver: str, use_propagation: bool, g_estimator: str) -> Dict[str, object]:
    propagation_requested = bool(use_propagation)
    propagation_theorem_backed = not propagation_requested

    if target_solver == "euler":
        return {
            "monitor_solver": "euler",
            "propagation_exponent": 0.5,
            "theorem_backed": propagation_theorem_backed,
            "notes": (
                "Euler local error obeys ||e_{n+1}|| <= (1 + h_n ell_n)||e_n|| + C_E M_E(s_n) h_n^2, "
                "so if a valid propagation upper bound G(s) is available then rho_E*(s) propto "
                "G(s)^(1/2) Q_E(s)^(1/4). The current code uses an empirical propagation proxy "
                "rather than a strict theorem-backed upper bound."
                if propagation_requested
                else "Euler local truncation error is controlled by L_u u, so the monitor uses "
                "Q_E(s)=E||L_u u||^2 and rho_E(s) propto (Q_E(s)+eps)^(1/4)."
            ),
        }
    if target_solver == "heun2":
        return {
            "monitor_solver": "heun2",
            "propagation_exponent": 1.0 / 3.0,
            "theorem_backed": propagation_theorem_backed,
            "notes": (
                "Heun2 uses the theorem-backed local monitor Q_H(s)=E||L_u^2 u||^2. If a valid "
                "propagation upper bound G(s) is available then rho_H*(s) propto "
                "G(s)^(1/3) Q_H(s)^(1/6). The current code uses an empirical propagation proxy "
                "rather than a strict theorem-backed upper bound."
                if propagation_requested
                else "Heun2 local truncation error is controlled by L_u^2 u, so the monitor uses "
                "Q_H(s)=E||L_u^2 u||^2 and rho_H(s) propto (Q_H(s)+eps)^(1/6)."
            ),
        }
    if target_solver == "stork4":
        return {
            "monitor_solver": "heun2",
            "propagation_exponent": 1.0 / 3.0,
            "theorem_backed": False,
            "notes": (
                "Phase-2 STORK4 still does not claim a solver-specific optimal monitor theorem. "
                "It consumes propagation-aware non-uniform nodes built from the Heun2 proxy "
                "monitor and Jacobian envelope as a documented heuristic."
                if propagation_requested
                else "Phase-1 STORK4 does not claim a solver-specific optimal monitor theorem. "
                "It reuses the Heun2 monitor as a heuristic node generator while STORK4 itself "
                "consumes arbitrary non-uniform nodes."
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
    cache_path: str,
    grid_size: int,
    batch_size: int,
    eps: float,
    use_propagation: bool,
    g_mode: str,
    g_estimator: str,
    g_power_iters: int,
    g_pool_radius: int,
    g_safety_factor: float,
    monitor: MonitorArtifacts,
    propagation: Optional[PropagationArtifacts],
    q_smoothed: Tensor,
    density: Tensor,
    phi: Tensor,
) -> SolverAwareProfile:
    return SolverAwareProfile(
        mode=mode,
        target_solver=target_solver,
        monitor_solver=monitor_solver,
        estimator=monitor.resolved_estimator,
        theorem_backed=theorem_backed,
        notes=notes,
        checkpoint_source=checkpoint_source,
        cache_path=cache_path,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        use_propagation=use_propagation,
        g_mode=g_mode,
        g_estimator=g_estimator if use_propagation else "",
        g_power_iters=int(g_power_iters) if use_propagation else 0,
        g_pool_radius=int(g_pool_radius) if use_propagation else 0,
        g_safety_factor=float(g_safety_factor) if use_propagation else 0.0,
        q_values=monitor.q_values,
        q_smoothed=q_smoothed,
        ell_raw=None if propagation is None else propagation.raw_ell,
        ell_env=None if propagation is None else propagation.env_ell,
        ell_values=None if propagation is None else propagation.ell_values,
        g_values=None if propagation is None else propagation.g_values,
        density=density,
        s_grid=monitor.s_grid,
        phi=phi,
    )


def _materialize_solver_aware_artifacts(
    profile: SolverAwareProfile,
    step_count: int,
) -> SolverAwareArtifacts:
    r_grid, nodes = build_solver_aware_nodes(
        s_grid=profile.s_grid,
        phi=profile.phi,
        node_count=step_count + 1,
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
        use_propagation=profile.use_propagation,
        g_mode=profile.g_mode,
        g_estimator=profile.g_estimator,
        g_power_iters=profile.g_power_iters,
        g_pool_radius=profile.g_pool_radius,
        g_safety_factor=profile.g_safety_factor,
        q_values=profile.q_values,
        q_smoothed=profile.q_smoothed,
        ell_raw=profile.ell_raw,
        ell_env=profile.ell_env,
        ell_values=profile.ell_values,
        g_values=profile.g_values,
        density=profile.density,
        s_grid=profile.s_grid,
        phi=profile.phi,
        step_count=step_count,
        r_grid=r_grid,
        nodes=nodes,
    )


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
    """Build the continuous solver-aware profile shared by all NFE values.

    The profile is the continuous object phi(s) induced by Q(s) and, when
    requested, by the propagation factor G(s). Different NFE budgets only
    materialize different node sets s_n = psi(n / N) from this shared profile.
    """
    if mode == "off":
        return None

    effective_mode = _validate_mode(mode=mode, k=k)
    _validate_propagation_args(
        use_propagation=use_propagation,
        g_mode=g_mode,
    )

    monitor_spec = _resolve_monitor(
        target_solver=target_solver,
        use_propagation=use_propagation,
        g_estimator=g_estimator,
    )
    signature = _cache_signature(
        mode=effective_mode,
        target_solver=target_solver,
        monitor_solver=monitor_spec["monitor_solver"],
        estimator=estimator,
        checkpoint_source=checkpoint_source,
        path_family=path_family,
        clock_family=clock_family,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
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
                "Loaded solver-aware continuous profile from cache %s",
                resolved_cache_path,
            )

    if profile is None:
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
                    raise ValueError(f"Unsupported solver-aware g_mode {g_mode}.")
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

        clock_profile = build_solver_aware_clock_profile(
            s_grid=monitor.s_grid,
            q_values=monitor.q_values,
            density_exponent=monitor.density_exponent,
            eps=eps,
            g_values=None if propagation is None else propagation.g_values,
            propagation_exponent=float(monitor_spec["propagation_exponent"]) if use_propagation else 0.0,
        )
        profile = _merge_monitor_and_clock_profile(
            mode=effective_mode,
            target_solver=target_solver,
            monitor_solver=str(monitor_spec["monitor_solver"]),
            theorem_backed=bool(monitor_spec["theorem_backed"]),
            notes=str(monitor_spec["notes"]),
            checkpoint_source=checkpoint_source,
            cache_path="" if resolved_cache_path is None else str(resolved_cache_path),
            grid_size=grid_size,
            batch_size=batch_size,
            eps=eps,
            use_propagation=use_propagation,
            g_mode=g_mode if use_propagation else "none",
            g_estimator=g_estimator if use_propagation else "",
            g_power_iters=g_power_iters if use_propagation else 0,
            g_pool_radius=g_pool_radius if use_propagation else 0,
            g_safety_factor=g_safety_factor if use_propagation else 0.0,
            monitor=monitor,
            propagation=propagation,
            q_smoothed=clock_profile.q_smoothed,
            density=clock_profile.density,
            phi=clock_profile.phi,
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
                    "use_propagation": profile.use_propagation,
                    "g_mode": profile.g_mode,
                    "g_estimator": profile.g_estimator,
                    "g_power_iters": profile.g_power_iters,
                    "g_pool_radius": profile.g_pool_radius,
                    "g_safety_factor": profile.g_safety_factor,
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
                    "use_propagation": artifacts.use_propagation,
                    "g_mode": artifacts.g_mode,
                    "g_estimator": artifacts.g_estimator,
                    "g_power_iters": artifacts.g_power_iters,
                    "g_pool_radius": artifacts.g_pool_radius,
                    "g_safety_factor": artifacts.g_safety_factor,
                    "step_count": artifacts.step_count,
                    "r_grid": [float(value) for value in artifacts.r_grid.detach().cpu().tolist()],
                    "nodes": [float(value) for value in artifacts.nodes.detach().cpu().tolist()],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        node_lines = ["node_index,r_value,s_value,step_size_from_prev"]
        r_values = artifacts.r_grid.detach().cpu().tolist()
        node_values = artifacts.nodes.detach().cpu().tolist()
        for index, (r_value, s_value) in enumerate(zip(r_values, node_values)):
            prev_value = node_values[index - 1] if index > 0 else 0.0
            step_size = float(s_value) - float(prev_value) if index > 0 else 0.0
            node_lines.append(f"{index},{float(r_value):.10f},{float(s_value):.10f},{float(step_size):.10f}")
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
                    "step_sizes": [
                        0.0 if index == 0 else float(node_values[index] - node_values[index - 1])
                        for index in range(len(node_values))
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return artifacts
