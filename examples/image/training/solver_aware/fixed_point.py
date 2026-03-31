import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from training.solver_aware.clock import SolverAwareClockArtifacts, build_solver_aware_clock
from training.solver_aware.monitors import (
    MonitorArtifacts,
    compute_euler_monitor,
    compute_heun2_monitor,
)


logger = logging.getLogger(__name__)


@dataclass
class SolverAwareArtifacts:
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
    step_count: int
    q_values: Tensor
    q_smoothed: Tensor
    density: Tensor
    s_grid: Tensor
    phi: Tensor
    r_grid: Tensor
    nodes: Tensor

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in ("q_values", "q_smoothed", "density", "s_grid", "phi", "r_grid", "nodes"):
            payload[key] = payload[key].detach().cpu()
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
    step_count: int,
    seed: int,
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
        "step_count": int(step_count),
        "seed": int(seed),
    }


def _normalize_cache_path(cache_path: str) -> Optional[Path]:
    if cache_path in {"", "none", "None", None}:
        return None
    return Path(str(cache_path))


def _load_cache(cache_path: Path, signature: Dict[str, object]) -> Optional[SolverAwareArtifacts]:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("signature") != signature:
        logger.info("Ignoring solver-aware cache %s because the signature no longer matches.", cache_path)
        return None
    artifact_payload = payload["artifacts"]
    return SolverAwareArtifacts(**artifact_payload)


def _save_cache(cache_path: Path, signature: Dict[str, object], artifacts: SolverAwareArtifacts) -> None:
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
                "It reuses the Heun2 monitor as a heuristic node generator while STORK4 itself "
                "consumes arbitrary non-uniform nodes with first-order Taylor virtual stages."
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


def _merge_monitor_and_clock(
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
    step_count: int,
    monitor: MonitorArtifacts,
    clock: SolverAwareClockArtifacts,
) -> SolverAwareArtifacts:
    return SolverAwareArtifacts(
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
        step_count=step_count,
        q_values=monitor.q_values,
        q_smoothed=clock.q_smoothed,
        density=clock.density,
        s_grid=monitor.s_grid,
        phi=clock.phi,
        r_grid=clock.r_grid,
        nodes=clock.nodes,
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
    grid_size: int,
    batch_size: int,
    eps: float,
    cfg_scale: float,
    step_count: int,
    checkpoint_source: str,
    seed: int,
    cache_path: str,
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

    monitor_spec = _resolve_monitor(target_solver=target_solver)
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
        step_count=step_count,
        seed=seed,
    )
    resolved_cache_path = _normalize_cache_path(cache_path=cache_path)
    if resolved_cache_path is not None:
        cached = _load_cache(cache_path=resolved_cache_path, signature=signature)
        if cached is not None:
            logger.info("Loaded solver-aware clock artifacts from cache %s", resolved_cache_path)
            return cached

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
    clock = build_solver_aware_clock(
        s_grid=monitor.s_grid,
        q_values=monitor.q_values,
        density_exponent=monitor.density_exponent,
        eps=eps,
        node_count=step_count + 1,
    )
    artifacts = _merge_monitor_and_clock(
        mode=effective_mode,
        target_solver=target_solver,
        monitor_solver=monitor_spec["monitor_solver"],
        theorem_backed=bool(monitor_spec["theorem_backed"]),
        notes=str(monitor_spec["notes"]),
        checkpoint_source=checkpoint_source,
        grid_size=grid_size,
        batch_size=batch_size,
        eps=eps,
        step_count=step_count,
        monitor=monitor,
        clock=clock,
    )

    if resolved_cache_path is not None:
        _save_cache(cache_path=resolved_cache_path, signature=signature, artifacts=artifacts)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts.to_dict(), output_dir / "solver_aware_artifacts.pt")
        (output_dir / "solver_aware_artifacts.json").write_text(
            json.dumps(
                {
                    "mode": artifacts.mode,
                    "target_solver": artifacts.target_solver,
                    "monitor_solver": artifacts.monitor_solver,
                    "estimator": artifacts.estimator,
                    "theorem_backed": artifacts.theorem_backed,
                    "notes": artifacts.notes,
                    "checkpoint_source": artifacts.checkpoint_source,
                    "grid_size": artifacts.grid_size,
                    "batch_size": artifacts.batch_size,
                    "eps": artifacts.eps,
                    "step_count": artifacts.step_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    logger.info(
        "Built solver-aware nodes for target_solver=%s using monitor_solver=%s and estimator=%s.",
        target_solver,
        monitor_spec["monitor_solver"],
        monitor.resolved_estimator,
    )
    logger.info("Solver-aware note: %s", monitor_spec["notes"])
    return artifacts
