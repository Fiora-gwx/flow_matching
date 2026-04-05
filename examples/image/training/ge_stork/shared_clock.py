import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

from training.fixed_step_solver import ReparameterizedSchedule, solve_fixed_budget
from training.nonuniform_nodes import build_uniform_time_grid
from training.solver_aware.monitors import (
    _cycle_loader,
    _jvp,
    _make_generator,
    _take_monitor_batch,
    _velocity_fn,
)


logger = logging.getLogger(__name__)

EPS = 1.0e-12
SHARED_CLOCK_FAMILIES = ("va", "vb", "aa", "ab")
OPTIMIZED_CLOCK_FAMILIES = frozenset({"vb", "ab"})
GENERATOR_CLOCK_FAMILIES = frozenset({"aa", "ab"})
PILOT_SOLVERS = frozenset({"euler", "heun2", "stork4"})
JACOBIAN_BACKENDS = frozenset({"exact", "probe"})


def normalize_shared_clock_family(clock_family: str) -> str:
    resolved = str(clock_family or "").strip().lower().replace("-", "")
    if resolved not in SHARED_CLOCK_FAMILIES:
        raise ValueError(
            f"Unsupported shared clock family={clock_family}. "
            f"Expected one of {SHARED_CLOCK_FAMILIES}."
        )
    return resolved


def normalize_jacobian_backend(jacobian_backend: str) -> str:
    resolved = str(jacobian_backend or "probe").strip().lower()
    if resolved not in JACOBIAN_BACKENDS:
        raise ValueError(
            f"Unsupported Jacobian backend={jacobian_backend}. "
            f"Expected one of {sorted(JACOBIAN_BACKENDS)}."
        )
    return resolved


def _required_pilot_nfe_budget(pilot_solver: str, step_count: int) -> int:
    resolved_solver = str(pilot_solver or "heun2").strip().lower()
    if resolved_solver not in PILOT_SOLVERS:
        raise ValueError(
            f"Unsupported pilot_solver={pilot_solver}. Expected one of {sorted(PILOT_SOLVERS)}."
        )
    if step_count <= 0:
        raise ValueError(f"step_count must be positive. Got {step_count}.")
    if resolved_solver == "heun2":
        return int(step_count) * 2
    return int(step_count)


def _quadrature_weights(grid: Tensor) -> Tensor:
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("physical_grid must be one-dimensional with at least two nodes.")
    dt_values = grid[1:] - grid[:-1]
    if torch.any(dt_values <= 0.0):
        raise ValueError("physical_grid must be strictly increasing.")
    weights = torch.zeros_like(grid)
    weights[0] = 0.5 * dt_values[0]
    weights[-1] = 0.5 * dt_values[-1]
    if grid.numel() > 2:
        weights[1:-1] = 0.5 * (dt_values[:-1] + dt_values[1:])
    return weights


def _normalize_density_from_profile(alpha_profile: Tensor, physical_grid: Tensor) -> Tensor:
    weights = _quadrature_weights(physical_grid)
    total = torch.sum(alpha_profile * weights).clamp(min=EPS)
    return alpha_profile / total


def _build_tau_grid(physical_grid: Tensor, density: Tensor) -> Tensor:
    dt_values = physical_grid[1:] - physical_grid[:-1]
    increments = 0.5 * (density[1:] + density[:-1]) * dt_values
    tau_grid = torch.zeros_like(physical_grid, dtype=density.dtype)
    tau_grid[1:] = torch.cumsum(increments, dim=0)
    tau_grid = tau_grid / tau_grid[-1].clamp(min=EPS)
    tau_grid[0] = 0.0
    tau_grid[-1] = 1.0
    return tau_grid


def _linear_interpolate_monotone(
    x_grid: Tensor,
    y_grid: Tensor,
    query: Tensor,
) -> Tensor:
    flat_query = query.reshape(-1).clamp(float(x_grid[0].item()), float(x_grid[-1].item()))
    right = torch.searchsorted(x_grid, flat_query, right=True)
    right = right.clamp(min=1, max=x_grid.numel() - 1)
    left = right - 1
    x_left = x_grid[left]
    x_right = x_grid[right]
    y_left = y_grid[left]
    y_right = y_grid[right]
    weight = (flat_query - x_left) / (x_right - x_left).clamp(min=EPS)
    interpolated = y_left + weight * (y_right - y_left)
    return interpolated.reshape_as(query)


def _central_time_derivative(values: Tensor, physical_grid: Tensor) -> Tensor:
    derivative = torch.zeros_like(values)
    view_shape = [1] * values.dim()
    view_shape[1] = -1

    left_dt = (physical_grid[1] - physical_grid[0]).clamp(min=EPS)
    right_dt = (physical_grid[-1] - physical_grid[-2]).clamp(min=EPS)
    derivative[:, 0] = (values[:, 1] - values[:, 0]) / left_dt
    derivative[:, -1] = (values[:, -1] - values[:, -2]) / right_dt

    center_dt = (physical_grid[2:] - physical_grid[:-2]).clamp(min=EPS).view(*view_shape)
    derivative[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / center_dt
    return derivative


def _flatten_norm(values: Tensor, start_dim: int = 2) -> Tensor:
    return values.flatten(start_dim=start_dim).norm(dim=-1)


def _state_squared_norm(values: Tensor) -> Tensor:
    return values.flatten(start_dim=2).pow(2).sum(dim=-1)


def _frobenius_sq_from_generator_representation(
    values: Tensor,
    *,
    jacobian_backend: str,
) -> Tensor:
    if jacobian_backend == "exact":
        flattened = values.flatten(start_dim=2)
        return flattened.pow(2).sum(dim=-1)
    return values.flatten(start_dim=3).pow(2).sum(dim=-1).mean(dim=2)


def _sample_probe_vectors(
    example_state: Tensor,
    *,
    probe_count: int,
    seed: int,
) -> Tensor:
    if probe_count <= 0:
        raise ValueError(f"probe_count must be positive. Got {probe_count}.")
    generator = _make_generator(device=example_state.device, seed=int(seed))
    probes = torch.randint(
        low=0,
        high=2,
        size=(int(probe_count),) + tuple(example_state.shape[1:]),
        generator=generator,
        device=example_state.device,
        dtype=torch.int64,
    )
    return probes.to(dtype=example_state.dtype) * 2.0 - 1.0


def _velocity_eval(
    velocity_model,
    x: Tensor,
    t: Tensor,
    labels: Tensor,
    cfg_scale: float,
) -> Tensor:
    return _velocity_fn(
        velocity_model=velocity_model,
        x=x,
        s=t,
        labels=labels,
        cfg_scale=cfg_scale,
    )


def _probe_generator_actions(
    velocity_model,
    x: Tensor,
    t: Tensor,
    labels: Tensor,
    cfg_scale: float,
    probe_vectors: Tensor,
) -> Tensor:
    actions = []
    x_input = x.detach().requires_grad_(True)
    for probe in probe_vectors:
        batch_probe = probe.unsqueeze(0).expand_as(x_input)

        def wrapped(x_input_batch: Tensor) -> Tensor:
            return _velocity_eval(
                velocity_model=velocity_model,
                x=x_input_batch,
                t=t,
                labels=labels,
                cfg_scale=cfg_scale,
            )

        _, action = _jvp(
            wrapped,
            (x_input,),
            (batch_probe,),
        )
        actions.append(action.detach())
    return torch.stack(actions, dim=1)


def _exact_generator_actions(
    velocity_model,
    x: Tensor,
    t: Tensor,
    labels: Tensor,
    cfg_scale: float,
) -> Tensor:
    batch_actions = []
    for sample_index in range(x.shape[0]):
        x_single = x[sample_index].detach().requires_grad_(True)
        t_single = t[sample_index : sample_index + 1]
        label_single = labels[sample_index : sample_index + 1]

        def wrapped(x_input_single: Tensor) -> Tensor:
            velocity = _velocity_eval(
                velocity_model=velocity_model,
                x=x_input_single.unsqueeze(0),
                t=t_single,
                labels=label_single,
                cfg_scale=cfg_scale,
            )
            return velocity.reshape(-1)

        jacobian = torch.autograd.functional.jacobian(
            wrapped,
            x_single,
            vectorize=True,
        )
        batch_actions.append(jacobian.detach())
    return torch.stack(batch_actions, dim=0)


@dataclass
class PilotTrajectoryArtifacts:
    physical_grid: Tensor
    trajectories: Tensor
    labels: Tensor
    pilot_solver: str
    pilot_nfe_budget: int
    pilot_step_count: int
    pilot_batch_size: int
    pilot_num_batches: int
    cfg_scale: float
    seed: int


@dataclass
class SharedClockObservations:
    physical_grid: Tensor
    trajectories: Tensor
    labels: Tensor
    velocity_values: Tensor
    velocity_derivatives: Tensor
    velocity_norms: Tensor
    generator_representation: Optional[Tensor]
    generator_derivatives: Optional[Tensor]
    generator_norms: Optional[Tensor]
    jacobian_backend: str
    probe_vectors: Optional[Tensor]
    num_trajectories: int


@dataclass
class SharedClockProfile:
    clock_family: str
    clock_tag: str
    physical_grid: Tensor
    density: Tensor
    tau_grid: Tensor
    alpha_profile: Tensor
    objective_trace: Tensor
    eps: float
    pilot_solver: str
    pilot_nfe_budget: int
    pilot_step_count: int
    pilot_batch_size: int
    pilot_num_batches: int
    num_trajectories: int
    jacobian_backend: str
    jacobian_num_probes: int
    optimizer_steps: int
    optimizer_lr: float
    checkpoint_source: str
    path_family: str
    cfg_scale: float
    seed: int
    created_at: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "clock_family": self.clock_family,
            "clock_tag": self.clock_tag,
            "physical_grid": self.physical_grid.detach().cpu(),
            "density": self.density.detach().cpu(),
            "tau_grid": self.tau_grid.detach().cpu(),
            "alpha_profile": self.alpha_profile.detach().cpu(),
            "objective_trace": self.objective_trace.detach().cpu(),
            "eps": float(self.eps),
            "pilot_solver": self.pilot_solver,
            "pilot_nfe_budget": int(self.pilot_nfe_budget),
            "pilot_step_count": int(self.pilot_step_count),
            "pilot_batch_size": int(self.pilot_batch_size),
            "pilot_num_batches": int(self.pilot_num_batches),
            "num_trajectories": int(self.num_trajectories),
            "jacobian_backend": self.jacobian_backend,
            "jacobian_num_probes": int(self.jacobian_num_probes),
            "optimizer_steps": int(self.optimizer_steps),
            "optimizer_lr": float(self.optimizer_lr),
            "checkpoint_source": self.checkpoint_source,
            "path_family": self.path_family,
            "cfg_scale": float(self.cfg_scale),
            "seed": int(self.seed),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SharedClockProfile":
        return cls(
            clock_family=str(payload["clock_family"]),
            clock_tag=str(payload["clock_tag"]),
            physical_grid=payload["physical_grid"],
            density=payload["density"],
            tau_grid=payload["tau_grid"],
            alpha_profile=payload["alpha_profile"],
            objective_trace=payload["objective_trace"],
            eps=float(payload["eps"]),
            pilot_solver=str(payload["pilot_solver"]),
            pilot_nfe_budget=int(payload["pilot_nfe_budget"]),
            pilot_step_count=int(payload["pilot_step_count"]),
            pilot_batch_size=int(payload["pilot_batch_size"]),
            pilot_num_batches=int(payload["pilot_num_batches"]),
            num_trajectories=int(payload["num_trajectories"]),
            jacobian_backend=str(payload["jacobian_backend"]),
            jacobian_num_probes=int(payload["jacobian_num_probes"]),
            optimizer_steps=int(payload["optimizer_steps"]),
            optimizer_lr=float(payload["optimizer_lr"]),
            checkpoint_source=str(payload["checkpoint_source"]),
            path_family=str(payload["path_family"]),
            cfg_scale=float(payload["cfg_scale"]),
            seed=int(payload["seed"]),
            created_at=str(payload["created_at"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def make_schedule(
        self,
        *,
        nfe: int,
        step_count: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> ReparameterizedSchedule:
        resolved_step_count = int(step_count if step_count is not None else nfe)
        if resolved_step_count <= 0:
            raise ValueError(f"step_count must be positive. Got {resolved_step_count}.")
        target_device = self.physical_grid.device if device is None else device
        tau_grid = torch.linspace(
            0.0,
            1.0,
            resolved_step_count + 1,
            device=target_device,
            dtype=torch.float64,
        )
        physical_grid = self.physical_grid.to(device=target_device, dtype=torch.float64)
        density = self.density.to(device=target_device, dtype=torch.float64)
        tau_profile = self.tau_grid.to(device=target_device, dtype=torch.float64)
        t_grid = _linear_interpolate_monotone(
            x_grid=tau_profile,
            y_grid=physical_grid,
            query=tau_grid,
        )
        density_query = _linear_interpolate_monotone(
            x_grid=physical_grid,
            y_grid=density,
            query=t_grid,
        ).clamp(min=float(self.eps))
        g_grid = torch.reciprocal(density_query)
        return ReparameterizedSchedule(
            tau_grid=tau_grid.to(dtype=dtype),
            t_grid=t_grid.to(dtype=dtype),
            g_grid=g_grid.to(dtype=dtype),
            dtau=1.0 / float(resolved_step_count),
            nfe_budget=int(nfe),
            step_count=int(resolved_step_count),
        )


def sample_pilot_trajectories(
    *,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    physical_grid: Tensor,
    pilot_solver: str,
    pilot_batch_size: int,
    pilot_num_batches: int,
    cfg_scale: float,
    seed: int,
) -> PilotTrajectoryArtifacts:
    """Sample K pilot trajectories x^{(k)}(t_i) on a fixed physical-time grid."""
    if int(pilot_num_batches) <= 0:
        raise ValueError(f"pilot_num_batches must be positive. Got {pilot_num_batches}.")
    if int(pilot_batch_size) <= 0:
        raise ValueError(f"pilot_batch_size must be positive. Got {pilot_batch_size}.")
    step_count = int(physical_grid.numel()) - 1
    pilot_nfe_budget = _required_pilot_nfe_budget(pilot_solver=pilot_solver, step_count=step_count)
    loader_iter = _cycle_loader(data_loader)
    noise_generator = _make_generator(device=device, seed=int(seed) + 17041)
    trajectory_batches = []
    label_batches = []

    for _ in range(int(pilot_num_batches)):
        samples, labels = _take_monitor_batch(
            loader_iter=loader_iter,
            batch_size=int(pilot_batch_size),
        )
        labels = labels.to(device=device, non_blocking=True)
        x_init = torch.randn(
            samples.shape,
            device=device,
            dtype=torch.float32,
            generator=noise_generator,
        )
        sample = solve_fixed_budget(
            velocity_model=velocity_model,
            x_init=x_init,
            solver_name=pilot_solver,
            nfe_budget=pilot_nfe_budget,
            return_trajectory=True,
            time_grid=physical_grid,
            label=labels,
            cfg_scale=cfg_scale,
        )
        if sample.trajectory is None:
            raise RuntimeError("Pilot trajectory sampling must return full trajectories.")
        trajectory_batches.append(sample.trajectory.permute(1, 0, *range(2, sample.trajectory.dim())).detach())
        label_batches.append(labels.detach())

    trajectories = torch.cat(trajectory_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return PilotTrajectoryArtifacts(
        physical_grid=physical_grid.detach(),
        trajectories=trajectories.detach(),
        labels=labels.detach(),
        pilot_solver=str(pilot_solver),
        pilot_nfe_budget=int(pilot_nfe_budget),
        pilot_step_count=int(step_count),
        pilot_batch_size=int(pilot_batch_size),
        pilot_num_batches=int(pilot_num_batches),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
    )


def extract_local_objects(
    *,
    velocity_model,
    pilot: PilotTrajectoryArtifacts,
    cfg_scale: float,
    require_generator: bool,
    jacobian_backend: str,
    jacobian_num_probes: int,
) -> SharedClockObservations:
    """Extract Q_V, Q_A and their finite-difference D_t Q values on pilot states."""
    jacobian_backend = normalize_jacobian_backend(jacobian_backend)
    trajectories = pilot.trajectories.to(device=pilot.trajectories.device, dtype=torch.float32)
    labels = pilot.labels.to(device=trajectories.device)
    num_trajectories = int(trajectories.shape[0])
    time_count = int(trajectories.shape[1])
    physical_grid = pilot.physical_grid.to(device=trajectories.device, dtype=torch.float32)

    velocity_values = []
    generator_values = []
    probe_vectors = None
    if require_generator and jacobian_backend == "probe":
        probe_vectors = _sample_probe_vectors(
            trajectories[:, 0],
            probe_count=int(jacobian_num_probes),
            seed=int(pilot.seed) + 23057,
        )

    for time_index in range(time_count):
        state_batch = trajectories[:, time_index].detach()
        time_batch = torch.full(
            (num_trajectories,),
            float(physical_grid[time_index].item()),
            device=state_batch.device,
            dtype=state_batch.dtype,
        )
        with torch.enable_grad():
            velocity = _velocity_eval(
                velocity_model=velocity_model,
                x=state_batch,
                t=time_batch,
                labels=labels,
                cfg_scale=cfg_scale,
            ).detach()
        velocity_values.append(velocity)

        if not require_generator:
            continue
        if jacobian_backend == "probe":
            with torch.enable_grad():
                generator_action = _probe_generator_actions(
                    velocity_model=velocity_model,
                    x=state_batch,
                    t=time_batch,
                    labels=labels,
                    cfg_scale=cfg_scale,
                    probe_vectors=probe_vectors,
                )
            generator_values.append(generator_action.detach())
        elif jacobian_backend == "exact":
            with torch.enable_grad():
                generator_action = _exact_generator_actions(
                    velocity_model=velocity_model,
                    x=state_batch,
                    t=time_batch,
                    labels=labels,
                    cfg_scale=cfg_scale,
                )
            generator_values.append(generator_action.detach())

    velocity_tensor = torch.stack(velocity_values, dim=1)
    velocity_derivatives = _central_time_derivative(velocity_tensor, physical_grid)
    velocity_norms = _flatten_norm(velocity_tensor)

    generator_tensor = None
    generator_derivatives = None
    generator_norms = None
    if generator_values:
        generator_tensor = torch.stack(generator_values, dim=1)
        generator_derivatives = _central_time_derivative(generator_tensor, physical_grid)
        if jacobian_backend == "exact":
            generator_norms = generator_tensor.flatten(start_dim=2).norm(dim=-1)
        else:
            generator_norms = torch.sqrt(
                generator_tensor.flatten(start_dim=3).pow(2).sum(dim=-1).mean(dim=2)
            )

    return SharedClockObservations(
        physical_grid=physical_grid.detach(),
        trajectories=trajectories.detach(),
        labels=labels.detach(),
        velocity_values=velocity_tensor.detach(),
        velocity_derivatives=velocity_derivatives.detach(),
        velocity_norms=velocity_norms.detach(),
        generator_representation=None if generator_tensor is None else generator_tensor.detach(),
        generator_derivatives=None if generator_derivatives is None else generator_derivatives.detach(),
        generator_norms=None if generator_norms is None else generator_norms.detach(),
        jacobian_backend=jacobian_backend,
        probe_vectors=None if probe_vectors is None else probe_vectors.detach(),
        num_trajectories=num_trajectories,
    )


def _shared_clock_family_tag(clock_family: str) -> str:
    mapping = {
        "va": "V-a",
        "vb": "V-b",
        "aa": "A-a",
        "ab": "A-b",
    }
    return mapping[normalize_shared_clock_family(clock_family)]


def _analytic_alpha_profile(
    observations: SharedClockObservations,
    *,
    clock_family: str,
    eps: float,
) -> Tensor:
    if clock_family == "va":
        norms = observations.velocity_norms
    elif clock_family == "aa":
        if observations.generator_norms is None:
            raise ValueError("A-a requires generator-layer observations.")
        norms = observations.generator_norms
    else:
        raise ValueError(f"Analytic alpha profile does not support clock_family={clock_family}.")
    return torch.sqrt(torch.mean((norms + float(eps)).pow(2), dim=0))


def _softplus_normalized_density(raw_u: Tensor, physical_grid: Tensor) -> Tensor:
    quad_weights = _quadrature_weights(physical_grid).to(device=raw_u.device, dtype=raw_u.dtype)
    unnormalized = torch.nn.functional.softplus(raw_u)
    normalization = torch.sum(unnormalized * quad_weights).clamp(min=EPS)
    return unnormalized / normalization


def _clock_time_derivative(density: Tensor, physical_grid: Tensor) -> Tensor:
    derivative = torch.zeros_like(density)
    derivative[0] = (density[1] - density[0]) / (physical_grid[1] - physical_grid[0]).clamp(min=EPS)
    derivative[-1] = (density[-1] - density[-2]) / (physical_grid[-1] - physical_grid[-2]).clamp(min=EPS)
    derivative[1:-1] = (density[2:] - density[:-2]) / (physical_grid[2:] - physical_grid[:-2]).clamp(min=EPS)
    return derivative


def _state_level_objective(
    observations: SharedClockObservations,
    *,
    density: Tensor,
) -> Tensor:
    # Discrete V-b objective:
    # || D_t Q_V / m^2 - (m_t / m^3) Q_V ||_2^2 weighted by m(t) dt.
    m = density[1:-1]
    m_t = _clock_time_derivative(density, observations.physical_grid)[1:-1]
    quad_weights = _quadrature_weights(observations.physical_grid)[1:-1].to(device=density.device, dtype=density.dtype)
    view_shape = (1, -1) + (1,) * (observations.velocity_values.dim() - 2)
    term = (
        observations.velocity_derivatives[:, 1:-1] / m.view(*view_shape).pow(2)
        - observations.velocity_values[:, 1:-1]
        * m_t.view(*view_shape)
        / m.view(*view_shape).pow(3)
    )
    sq_norm = _state_squared_norm(term)
    return torch.sum(quad_weights.view(1, -1) * sq_norm * m.view(1, -1)) / float(observations.num_trajectories)


def _generator_level_objective(
    observations: SharedClockObservations,
    *,
    density: Tensor,
) -> Tensor:
    # Discrete A-b objective:
    # || D_t Q_A / m^2 - (m_t / m^3) Q_A ||_F^2 weighted by m(t) dt.
    if observations.generator_representation is None or observations.generator_derivatives is None:
        raise ValueError("Generator-level objective requires generator-layer observations.")
    m = density[1:-1]
    m_t = _clock_time_derivative(density, observations.physical_grid)[1:-1]
    quad_weights = _quadrature_weights(observations.physical_grid)[1:-1].to(device=density.device, dtype=density.dtype)
    view_shape = (1, -1) + (1,) * (observations.generator_representation.dim() - 2)
    term = (
        observations.generator_derivatives[:, 1:-1] / m.view(*view_shape).pow(2)
        - observations.generator_representation[:, 1:-1]
        * m_t.view(*view_shape)
        / m.view(*view_shape).pow(3)
    )
    sq_norm = _frobenius_sq_from_generator_representation(
        term,
        jacobian_backend=observations.jacobian_backend,
    )
    return torch.sum(quad_weights.view(1, -1) * sq_norm * m.view(1, -1)) / float(observations.num_trajectories)


def _optimize_density(
    observations: SharedClockObservations,
    *,
    clock_family: str,
    optimizer_steps: int,
    optimizer_lr: float,
) -> Tuple[Tensor, Tensor]:
    raw_u = torch.zeros_like(observations.physical_grid, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw_u], lr=float(optimizer_lr))
    trace = []
    for _ in range(int(optimizer_steps)):
        optimizer.zero_grad(set_to_none=True)
        density = _softplus_normalized_density(raw_u, observations.physical_grid.to(dtype=torch.float64))
        if clock_family == "vb":
            loss = _state_level_objective(
                observations=observations,
                density=density.to(dtype=observations.velocity_values.dtype),
            )
        elif clock_family == "ab":
            loss = _generator_level_objective(
                observations=observations,
                density=density.to(dtype=observations.velocity_values.dtype),
            )
        else:
            raise ValueError(f"Optimized density does not support clock_family={clock_family}.")
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach().item()))
    density = _softplus_normalized_density(raw_u.detach(), observations.physical_grid.to(dtype=torch.float64))
    return density.to(dtype=observations.physical_grid.dtype), torch.tensor(trace, dtype=observations.physical_grid.dtype)


def _write_shared_clock_profile_summary(
    *,
    profile: SharedClockProfile,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(profile.to_dict(), output_dir / "shared_clock_profile.pt")
    rows = ["grid_index,t_value,m_value,tau_value,alpha_value"]
    for index, (t_value, m_value, tau_value, alpha_value) in enumerate(
        zip(
            profile.physical_grid.detach().cpu().tolist(),
            profile.density.detach().cpu().tolist(),
            profile.tau_grid.detach().cpu().tolist(),
            profile.alpha_profile.detach().cpu().tolist(),
        )
    ):
        rows.append(
            f"{index},{float(t_value)},{float(m_value)},{float(tau_value)},{float(alpha_value)}"
        )
    (output_dir / "shared_clock_profile.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "shared_clock_profile.json").write_text(
        json.dumps(
            {
                **profile.to_dict(),
                "physical_grid": [float(value) for value in profile.physical_grid.detach().cpu().tolist()],
                "density": [float(value) for value in profile.density.detach().cpu().tolist()],
                "tau_grid": [float(value) for value in profile.tau_grid.detach().cpu().tolist()],
                "alpha_profile": [float(value) for value in profile.alpha_profile.detach().cpu().tolist()],
                "objective_trace": [float(value) for value in profile.objective_trace.detach().cpu().tolist()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_shared_clock(
    *,
    clock_family: str,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    pilot_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    cfg_scale: float,
    eps: float,
    jacobian_backend: str,
    jacobian_num_probes: int,
    optimizer_steps: int,
    optimizer_lr: float,
    checkpoint_source: str,
    seed: int,
    output_dir: Optional[Path] = None,
) -> SharedClockProfile:
    """Build one shared clock profile offline, independent of downstream eval NFE."""
    clock_family = normalize_shared_clock_family(clock_family)
    physical_grid = build_uniform_time_grid(
        step_count=max(1, int(physical_grid_size) - 1),
        device=device,
        dtype=torch.float32,
    )
    pilot = sample_pilot_trajectories(
        velocity_model=velocity_model,
        data_loader=data_loader,
        device=device,
        physical_grid=physical_grid,
        pilot_solver=pilot_solver,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    observations = extract_local_objects(
        velocity_model=velocity_model,
        pilot=pilot,
        cfg_scale=cfg_scale,
        require_generator=clock_family in GENERATOR_CLOCK_FAMILIES,
        jacobian_backend=jacobian_backend,
        jacobian_num_probes=jacobian_num_probes,
    )

    if clock_family in {"va", "aa"}:
        alpha_profile = _analytic_alpha_profile(
            observations=observations,
            clock_family=clock_family,
            eps=eps,
        )
        density = _normalize_density_from_profile(alpha_profile, physical_grid)
        objective_trace = torch.empty(0, device=physical_grid.device, dtype=physical_grid.dtype)
    else:
        density, objective_trace = _optimize_density(
            observations=observations,
            clock_family=clock_family,
            optimizer_steps=optimizer_steps,
            optimizer_lr=optimizer_lr,
        )
        alpha_profile = density.detach().clone()

    tau_grid = _build_tau_grid(physical_grid=physical_grid, density=density)
    profile = SharedClockProfile(
        clock_family=clock_family,
        clock_tag=_shared_clock_family_tag(clock_family),
        physical_grid=physical_grid.detach(),
        density=density.detach(),
        tau_grid=tau_grid.detach(),
        alpha_profile=alpha_profile.detach(),
        objective_trace=objective_trace.detach(),
        eps=float(eps),
        pilot_solver=str(pilot_solver),
        pilot_nfe_budget=int(pilot.pilot_nfe_budget),
        pilot_step_count=int(pilot.pilot_step_count),
        pilot_batch_size=int(pilot_batch_size),
        pilot_num_batches=int(pilot_num_batches),
        num_trajectories=int(observations.num_trajectories),
        jacobian_backend=normalize_jacobian_backend(jacobian_backend),
        jacobian_num_probes=int(jacobian_num_probes),
        optimizer_steps=int(optimizer_steps),
        optimizer_lr=float(optimizer_lr),
        checkpoint_source=str(checkpoint_source),
        path_family=str(path_family),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
        created_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        metadata={
            "shared_across_nfe": True,
            "clock_tag": _shared_clock_family_tag(clock_family),
            "pilot_solver": str(pilot_solver),
            "pilot_nfe_budget": int(pilot.pilot_nfe_budget),
            "pilot_step_count": int(pilot.pilot_step_count),
            "physical_grid_size": int(physical_grid_size),
            "jacobian_backend": normalize_jacobian_backend(jacobian_backend),
            "jacobian_num_probes": int(jacobian_num_probes),
            "optimizer_steps": int(optimizer_steps),
            "optimizer_lr": float(optimizer_lr),
            "num_trajectories": int(observations.num_trajectories),
        },
    )

    if output_dir is not None:
        _write_shared_clock_profile_summary(profile=profile, output_dir=output_dir)
        torch.save(
            {
                "physical_grid": pilot.physical_grid.detach().cpu(),
                "trajectories": pilot.trajectories.detach().cpu(),
                "labels": pilot.labels.detach().cpu(),
                "pilot_solver": pilot.pilot_solver,
                "pilot_nfe_budget": pilot.pilot_nfe_budget,
                "pilot_step_count": pilot.pilot_step_count,
            },
            output_dir / "shared_clock_pilot_trajectories.pt",
        )

    return profile


def _shared_clock_signature(
    *,
    clock_family: str,
    path_family: str,
    pilot_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    jacobian_backend: str,
    jacobian_num_probes: int,
    optimizer_steps: int,
    optimizer_lr: float,
    eps: float,
    cfg_scale: float,
    checkpoint_source: str,
    seed: int,
) -> Dict[str, object]:
    return {
        "clock_family": normalize_shared_clock_family(clock_family),
        "path_family": str(path_family),
        "pilot_solver": str(pilot_solver),
        "physical_grid_size": int(physical_grid_size),
        "pilot_batch_size": int(pilot_batch_size),
        "pilot_num_batches": int(pilot_num_batches),
        "jacobian_backend": normalize_jacobian_backend(jacobian_backend),
        "jacobian_num_probes": int(jacobian_num_probes),
        "optimizer_steps": int(optimizer_steps),
        "optimizer_lr": float(optimizer_lr),
        "eps": float(eps),
        "cfg_scale": float(cfg_scale),
        "checkpoint_source": str(checkpoint_source),
        "seed": int(seed),
    }


def _resolve_shared_clock_cache_path(
    *,
    cache_path: str,
    output_dir: Optional[Path],
    signature: Dict[str, object],
) -> Optional[Path]:
    if cache_path not in {"", "none", "None", None}:
        return Path(str(cache_path))
    if output_dir is None:
        return None
    signature_text = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha1(signature_text.encode("utf-8")).hexdigest()[:10]
    return output_dir.parent / f"shared_clock_profile_{signature['clock_family']}_{signature_hash}.pt"


def load_shared_clock(cache_path: Path) -> SharedClockProfile:
    payload = torch.load(cache_path, map_location="cpu")
    if "profile" in payload:
        payload = payload["profile"]
    return SharedClockProfile.from_dict(payload)


def build_or_load_shared_clock(
    *,
    clock_family: str,
    velocity_model,
    data_loader: Iterable,
    device: torch.device,
    path_family: str,
    pilot_solver: str,
    physical_grid_size: int,
    pilot_batch_size: int,
    pilot_num_batches: int,
    cfg_scale: float,
    eps: float,
    jacobian_backend: str,
    jacobian_num_probes: int,
    optimizer_steps: int,
    optimizer_lr: float,
    checkpoint_source: str,
    seed: int,
    cache_path: str,
    output_dir: Optional[Path] = None,
) -> SharedClockProfile:
    """Load a cached shared clock when possible, otherwise build it once offline."""
    signature = _shared_clock_signature(
        clock_family=clock_family,
        path_family=path_family,
        pilot_solver=pilot_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        jacobian_backend=jacobian_backend,
        jacobian_num_probes=jacobian_num_probes,
        optimizer_steps=optimizer_steps,
        optimizer_lr=optimizer_lr,
        eps=eps,
        cfg_scale=cfg_scale,
        checkpoint_source=checkpoint_source,
        seed=seed,
    )
    resolved_cache_path = _resolve_shared_clock_cache_path(
        cache_path=cache_path,
        output_dir=output_dir,
        signature=signature,
    )
    if resolved_cache_path is not None and resolved_cache_path.exists():
        payload = torch.load(resolved_cache_path, map_location="cpu")
        if payload.get("signature") == signature:
            logger.info("Loaded shared clock profile from cache %s.", resolved_cache_path)
            profile = SharedClockProfile.from_dict(payload["profile"])
            if output_dir is not None:
                _write_shared_clock_profile_summary(profile=profile, output_dir=output_dir)
            return profile
        logger.info(
            "Ignoring shared clock cache %s because its signature no longer matches.",
            resolved_cache_path,
        )

    profile = build_shared_clock(
        clock_family=clock_family,
        velocity_model=velocity_model,
        data_loader=data_loader,
        device=device,
        path_family=path_family,
        pilot_solver=pilot_solver,
        physical_grid_size=physical_grid_size,
        pilot_batch_size=pilot_batch_size,
        pilot_num_batches=pilot_num_batches,
        cfg_scale=cfg_scale,
        eps=eps,
        jacobian_backend=jacobian_backend,
        jacobian_num_probes=jacobian_num_probes,
        optimizer_steps=optimizer_steps,
        optimizer_lr=optimizer_lr,
        checkpoint_source=checkpoint_source,
        seed=seed,
        output_dir=output_dir,
    )
    if resolved_cache_path is not None:
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "signature": signature,
                "profile": profile.to_dict(),
            },
            resolved_cache_path,
        )
        logger.info("Saved shared clock profile cache to %s.", resolved_cache_path)
    return profile


def get_time_grid_for_nfe(
    clock: SharedClockProfile,
    nfe: int,
    *,
    step_count: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, object]:
    """Materialize a uniform-tau schedule from one reusable shared clock profile."""
    schedule = clock.make_schedule(
        nfe=int(nfe),
        step_count=step_count,
        device=device,
        dtype=dtype,
    )
    return {
        "tau_grid": schedule.tau_grid,
        "t_grid": schedule.t_grid,
        "g_grid": schedule.g_grid,
        "dtau": float(schedule.dtau),
        "schedule": schedule,
    }


def save_shared_clock_schedule(
    *,
    clock: SharedClockProfile,
    schedule: ReparameterizedSchedule,
    output_dir: Path,
    solver_name: str,
) -> None:
    """Persist the current uniform-tau schedule used by a concrete eval run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = ["grid_index,tau_value,t_value,g_value"]
    for index, (tau_value, t_value, g_value) in enumerate(
        zip(
            schedule.tau_grid.detach().cpu().tolist(),
            schedule.t_grid.detach().cpu().tolist(),
            schedule.g_grid.detach().cpu().tolist(),
        )
    ):
        rows.append(f"{index},{float(tau_value)},{float(t_value)},{float(g_value)}")
    (output_dir / "shared_clock_schedule.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "shared_clock_schedule.json").write_text(
        json.dumps(
            {
                "clock_family": clock.clock_family,
                "clock_tag": clock.clock_tag,
                "solver": str(solver_name),
                "nfe_budget": int(schedule.nfe_budget if schedule.nfe_budget is not None else 0),
                "step_count": int(schedule.step_count if schedule.step_count is not None else 0),
                "dtau": float(schedule.dtau),
                "tau_grid": [float(value) for value in schedule.tau_grid.detach().cpu().tolist()],
                "t_grid": [float(value) for value in schedule.t_grid.detach().cpu().tolist()],
                "g_grid": [float(value) for value in schedule.g_grid.detach().cpu().tolist()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
