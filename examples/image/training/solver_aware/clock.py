from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


EPS = 1e-12
BISECTION_TOL = 1e-8
BISECTION_STEPS = 80
ADAPTIVE_ETA_DELTA = 0.3
ADAPTIVE_ETA_CAP = 0.95

@dataclass
class SolverAwareClockProfile:
    s_grid: Tensor
    q_values: Tensor
    q_smoothed: Tensor
    q_h_values: Optional[Tensor]
    q_h_smoothed: Optional[Tensor]
    g_values: Optional[Tensor]
    rho_floor: Tensor
    unconstrained_weight: Tensor
    density: Tensor
    phi: Tensor
    weight_monitor_name: str
    density_exponent: float
    propagation_exponent: float
    eta: float
    floor_mode: str
    floor_eps: float
    legacy_unconstrained: bool
    used_uniform_fallback: bool
    floor_mass: float
    min_feasible_step_count: int
    step_count: int
    smoothing_window: int


@dataclass
class SolverAwareClockArtifacts(SolverAwareClockProfile):
    r_grid: Tensor
    nodes: Tensor
    step_sizes: Tensor


def _moving_average(values: Tensor, window: int) -> Tensor:
    if window <= 1 or values.numel() <= 2:
        return values
    window = min(int(window), int(values.numel()))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = torch.nn.functional.pad(
        values.view(1, 1, -1),
        (radius, radius),
        mode="replicate",
    )
    kernel = torch.ones(1, 1, window, device=values.device, dtype=values.dtype) / float(window)
    smoothed = torch.nn.functional.conv1d(padded, kernel)
    return smoothed.view(-1)


def _strictly_monotone(values: Tensor) -> Tensor:
    monotone = values.clone()
    min_step = max(torch.finfo(monotone.dtype).eps, EPS)
    for index in range(1, monotone.numel()):
        monotone[index] = torch.maximum(
            monotone[index],
            monotone[index - 1] + min_step,
        )
    monotone = monotone - monotone[0]
    monotone = monotone / monotone[-1].clamp(min=min_step)
    monotone[0] = 0.0
    monotone[-1] = 1.0
    return monotone


def _integrate_on_grid(
    values: Tensor,
    s_grid: Tensor,
) -> Tensor:
    ds = s_grid[1:] - s_grid[:-1]
    if torch.any(ds <= 0.0):
        raise ValueError("s_grid must be strictly increasing.")
    trapezoids = 0.5 * (values[1:] + values[:-1]) * ds.to(dtype=values.dtype)
    return trapezoids.sum()


def monotone_inverse_lookup(
    x_grid: Tensor,
    y_grid: Tensor,
    query: Tensor,
) -> Tensor:
    flat_query = query.reshape(-1).clamp(float(y_grid[0].item()), float(y_grid[-1].item()))
    right_indices = torch.searchsorted(y_grid, flat_query, right=True)
    right_indices = right_indices.clamp(min=1, max=y_grid.numel() - 1)
    left_indices = right_indices - 1

    y_left = y_grid[left_indices]
    y_right = y_grid[right_indices]
    x_left = x_grid[left_indices]
    x_right = x_grid[right_indices]
    weight = (flat_query - y_left) / (y_right - y_left).clamp(min=EPS)
    values = x_left + weight * (x_right - x_left)
    return values.reshape_as(query)


def build_solver_aware_nodes(
    s_grid: Tensor,
    phi: Tensor,
    node_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    r_grid = torch.linspace(
        0.0,
        1.0,
        node_count,
        device=s_grid.device,
        dtype=s_grid.dtype,
    )
    nodes = monotone_inverse_lookup(
        x_grid=s_grid.to(dtype=phi.dtype),
        y_grid=phi.to(dtype=phi.dtype),
        query=r_grid.to(dtype=phi.dtype),
    ).to(dtype=s_grid.dtype)
    nodes[0] = 0.0
    nodes[-1] = 1.0
    step_sizes = torch.zeros_like(nodes)
    step_sizes[1:] = nodes[1:] - nodes[:-1]
    return r_grid, nodes, step_sizes


def _build_unconstrained_weight(
    monitor_smoothed: Tensor,
    density_exponent: float,
    eps: float,
    g_values: Optional[Tensor],
    propagation_exponent: float,
) -> Tensor:
    weight = torch.pow(monitor_smoothed + float(eps), float(density_exponent))
    if g_values is not None:
        weight = weight * torch.pow(
            g_values.clamp(min=EPS),
            float(propagation_exponent),
        )
    return weight


def _build_pointwise_floor(
    q_smoothed: Tensor,
    q_h_smoothed: Tensor,
    step_count: int,
    eta: float,
    floor_eps: float,
) -> Tensor:
    if step_count <= 0:
        raise ValueError("step_count must be positive for constrained solver-aware clocks.")
    if float(eta) <= 0.0:
        raise ValueError("solver_aware_eta must be positive.")
    numerator = (q_h_smoothed + float(floor_eps)).clamp(min=EPS)
    denominator = (q_smoothed + float(floor_eps)).clamp(min=EPS)
    ratio = torch.sqrt(numerator / denominator)
    return ratio / (3.0 * float(eta) * float(step_count))


def _resolve_eta(
    *,
    s_grid: Tensor,
    q_smoothed: Tensor,
    q_h_smoothed: Optional[Tensor],
    step_count: int,
    eta: Optional[float],
    floor_eps: float,
    legacy_unconstrained: bool,
) -> float:
    if eta is not None:
        if float(eta) <= 0.0:
            raise ValueError("solver_aware_eta must be positive.")
        return float(eta)
    if q_h_smoothed is None:
        if not bool(legacy_unconstrained):
            raise ValueError(
                "Adaptive solver-aware eta requires q_h_values so the admissible floor can be built."
            )
        return ADAPTIVE_ETA_CAP
    numerator = (q_h_smoothed + float(floor_eps)).clamp(min=EPS)
    denominator = (q_smoothed + float(floor_eps)).clamp(min=EPS)
    ratio = torch.sqrt(numerator / denominator)
    k_n = float(
        _integrate_on_grid(
            ratio.to(dtype=torch.float64),
            s_grid.to(dtype=torch.float64),
        ).item()
    ) / (3.0 * float(step_count))
    adaptive_eta = k_n / (1.0 - ADAPTIVE_ETA_DELTA)
    return max(float(floor_eps), min(ADAPTIVE_ETA_CAP, float(adaptive_eta)))


def build_density_from_constrained_problem(
    *,
    s_grid: Tensor,
    unconstrained_weight: Tensor,
    rho_floor: Tensor,
    legacy_unconstrained: bool,
) -> Tensor:
    if bool(legacy_unconstrained):
        normalization = _integrate_on_grid(unconstrained_weight, s_grid).clamp(min=EPS)
        return unconstrained_weight / normalization

    floor_mass = float(_integrate_on_grid(rho_floor, s_grid).item())
    if floor_mass >= 1.0 - 1e-6:
        density = rho_floor
        normalization = _integrate_on_grid(density, s_grid).clamp(min=EPS)
        return density / normalization

    lower = torch.zeros((), device=s_grid.device, dtype=unconstrained_weight.dtype)
    upper = torch.ones((), device=s_grid.device, dtype=unconstrained_weight.dtype)

    def _mass(scale: Tensor) -> Tensor:
        density = torch.maximum(rho_floor, scale * unconstrained_weight)
        return _integrate_on_grid(density, s_grid)

    while float(_mass(upper).item()) < 1.0:
        upper = upper * 2.0
        if float(upper.item()) > 1e12:
            raise ValueError("Failed to bracket the constrained normalization constant c_N.")

    for _ in range(BISECTION_STEPS):
        midpoint = 0.5 * (lower + upper)
        if float(_mass(midpoint).item()) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
        if float((upper - lower).abs().item()) <= BISECTION_TOL:
            break

    density = torch.maximum(rho_floor, upper * unconstrained_weight)
    normalization = _integrate_on_grid(density, s_grid).clamp(min=EPS)
    return density / normalization


def build_solver_aware_clock_profile(
    s_grid: Tensor,
    q_values: Tensor,
    q_h_values: Optional[Tensor],
    use_q_h_for_weight: bool,
    density_exponent: float,
    eps: float,
    step_count: int,
    eta: Optional[float],
    floor_mode: str,
    floor_eps: float,
    g_values: Optional[Tensor] = None,
    propagation_exponent: float = 0.0,
    legacy_unconstrained: bool = False,
    smoothing_window: Optional[int] = None,
) -> SolverAwareClockProfile:
    """Build the constrained solver-aware clock profile on a fixed s-grid.

    The main formulation is

        min_rho \int a(s) / rho(s)^p ds
        s.t.   \int rho(s) ds = 1
               rho(s) >= rho_floor_N(s)

    For Euler, the proxy inputs are

        A(s) ≈ sqrt(Q_E(s)+eps)
        B(s) ≈ sqrt(Q_H(s)+eps)

    so the admissible floor becomes

        rho_floor_N(s) ≈ (1 / (3 eta N)) * sqrt((Q_H(s)+eps)/(Q_E(s)+eps)).

    When eta is omitted, this function resolves an adaptive default at the
    current NFE:

        K_N = (1 / (3N)) * \int sqrt((Q_H+eps)/(Q_E+eps)) ds
        eta(N) = min(0.95, K_N / (1 - delta)),

    with delta fixed to 0.3.

    The constrained minimizer has the closed form

        rho_N*(s) = max{ rho_floor_N(s), c_N * w(s) },

    where w(s) is the unconstrained weight built from the solver's primary
    monitor:
    - solver-aware: w(s) = (Q(s)+eps)^gamma
    - propagation-aware: w(s) = G(s)^beta * (Q(s)+eps)^gamma

    and c_N is chosen so that \int rho_N*(s) ds = 1.
    """
    if s_grid.ndim != 1 or q_values.ndim != 1:
        raise ValueError("s_grid and q_values must be one-dimensional tensors.")
    if s_grid.numel() != q_values.numel():
        raise ValueError("s_grid and q_values must have the same length.")
    if s_grid.numel() < 2:
        raise ValueError("At least two grid points are required to build a solver-aware clock.")
    if str(floor_mode) not in {"pointwise", "constant"}:
        raise ValueError(f"Unsupported solver_aware_floor_mode={floor_mode}.")

    smoothing_window = (
        max(3, int(s_grid.numel() // 16) * 2 + 1)
        if smoothing_window is None
        else max(1, int(smoothing_window))
    )
    q_tensor = q_values.to(dtype=torch.float64).clamp(min=0.0)
    q_smoothed = _moving_average(q_tensor, window=smoothing_window)
    q_h_tensor = None
    q_h_smoothed = None
    if q_h_values is not None:
        if q_h_values.ndim != 1 or q_h_values.numel() != s_grid.numel():
            raise ValueError("q_h_values must be one-dimensional and aligned with s_grid.")
        q_h_tensor = q_h_values.to(dtype=torch.float64).clamp(min=0.0)
        q_h_smoothed = _moving_average(q_h_tensor, window=smoothing_window)

    g_tensor = None
    if g_values is not None:
        if g_values.ndim != 1 or g_values.numel() != s_grid.numel():
            raise ValueError("g_values must be one-dimensional and aligned with s_grid.")
        g_tensor = g_values.to(dtype=torch.float64).clamp(min=EPS)

    resolved_eta = _resolve_eta(
        s_grid=s_grid,
        q_smoothed=q_smoothed,
        q_h_smoothed=q_h_smoothed,
        step_count=step_count,
        eta=eta,
        floor_eps=floor_eps,
        legacy_unconstrained=legacy_unconstrained,
    )

    if bool(use_q_h_for_weight):
        if q_h_smoothed is None:
            raise ValueError("use_q_h_for_weight=true requires q_h_values.")
        weight_monitor_smoothed = q_h_smoothed
        weight_monitor_name = "q_h"
    else:
        weight_monitor_smoothed = q_smoothed
        weight_monitor_name = "q_e"

    unconstrained_weight = _build_unconstrained_weight(
        monitor_smoothed=weight_monitor_smoothed,
        density_exponent=density_exponent,
        eps=eps,
        g_values=g_tensor,
        propagation_exponent=propagation_exponent,
    )

    if q_h_smoothed is None:
        if not bool(legacy_unconstrained):
            raise ValueError(
                "Constrained solver-aware clocks require q_h_values so the admissible floor can be built."
            )
        rho_floor = torch.zeros_like(unconstrained_weight)
    else:
        pointwise_floor = _build_pointwise_floor(
            q_smoothed=q_smoothed,
            q_h_smoothed=q_h_smoothed,
            step_count=step_count,
            eta=resolved_eta,
            floor_eps=floor_eps,
        )
        if floor_mode == "pointwise":
            rho_floor = pointwise_floor
        else:
            rho_floor = torch.full_like(pointwise_floor, float(pointwise_floor.max().item()))

    floor_mass = float(
        _integrate_on_grid(
            rho_floor.to(dtype=torch.float64),
            s_grid.to(dtype=torch.float64),
        ).item()
    )
    min_feasible_step_count = max(
        int(step_count),
        int(step_count) if floor_mass <= 1.0 else int(torch.ceil(torch.tensor(float(step_count) * floor_mass)).item()),
    )
    used_uniform_fallback = bool(
        not legacy_unconstrained and floor_mass > 1.0 + 1e-6
    )

    if used_uniform_fallback:
        density = torch.ones_like(unconstrained_weight, dtype=torch.float64)
        density = density / _integrate_on_grid(density, s_grid.to(dtype=torch.float64)).clamp(min=EPS)
    else:
        density = build_density_from_constrained_problem(
            s_grid=s_grid.to(dtype=torch.float64),
            unconstrained_weight=unconstrained_weight,
            rho_floor=rho_floor,
            legacy_unconstrained=legacy_unconstrained,
        )

    trapezoids = 0.5 * (density[1:] + density[:-1]) * (s_grid[1:] - s_grid[:-1]).to(dtype=density.dtype)
    phi = torch.zeros_like(s_grid, dtype=density.dtype)
    phi[1:] = torch.cumsum(trapezoids, dim=0)
    phi = phi / phi[-1].clamp(min=EPS)
    phi = _strictly_monotone(phi)

    return SolverAwareClockProfile(
        s_grid=s_grid,
        q_values=q_tensor.to(dtype=s_grid.dtype),
        q_smoothed=q_smoothed.to(dtype=s_grid.dtype),
        q_h_values=None if q_h_tensor is None else q_h_tensor.to(dtype=s_grid.dtype),
        q_h_smoothed=None if q_h_smoothed is None else q_h_smoothed.to(dtype=s_grid.dtype),
        g_values=None if g_tensor is None else g_tensor.to(dtype=s_grid.dtype),
        rho_floor=rho_floor.to(dtype=s_grid.dtype),
        unconstrained_weight=unconstrained_weight.to(dtype=s_grid.dtype),
        density=density.to(dtype=s_grid.dtype),
        phi=phi.to(dtype=s_grid.dtype),
        weight_monitor_name=weight_monitor_name,
        density_exponent=float(density_exponent),
        propagation_exponent=float(propagation_exponent),
        eta=float(resolved_eta),
        floor_mode=str(floor_mode),
        floor_eps=float(floor_eps),
        legacy_unconstrained=bool(legacy_unconstrained),
        used_uniform_fallback=used_uniform_fallback,
        floor_mass=float(floor_mass),
        min_feasible_step_count=int(min_feasible_step_count),
        step_count=int(step_count),
        smoothing_window=int(smoothing_window),
    )


def build_solver_aware_clock(
    s_grid: Tensor,
    q_values: Tensor,
    q_h_values: Optional[Tensor],
    use_q_h_for_weight: bool,
    density_exponent: float,
    eps: float,
    step_count: int,
    eta: Optional[float],
    floor_mode: str,
    floor_eps: float,
    g_values: Optional[Tensor] = None,
    propagation_exponent: float = 0.0,
    legacy_unconstrained: bool = False,
    smoothing_window: Optional[int] = None,
) -> SolverAwareClockArtifacts:
    profile = build_solver_aware_clock_profile(
        s_grid=s_grid,
        q_values=q_values,
        q_h_values=q_h_values,
        use_q_h_for_weight=use_q_h_for_weight,
        density_exponent=density_exponent,
        eps=eps,
        step_count=step_count,
        eta=eta,
        floor_mode=floor_mode,
        floor_eps=floor_eps,
        g_values=g_values,
        propagation_exponent=propagation_exponent,
        legacy_unconstrained=legacy_unconstrained,
        smoothing_window=smoothing_window,
    )
    r_grid, nodes, step_sizes = build_solver_aware_nodes(
        s_grid=profile.s_grid,
        phi=profile.phi,
        node_count=step_count + 1,
    )
    return SolverAwareClockArtifacts(
        s_grid=profile.s_grid,
        q_values=profile.q_values,
        q_smoothed=profile.q_smoothed,
        q_h_values=profile.q_h_values,
        q_h_smoothed=profile.q_h_smoothed,
        g_values=profile.g_values,
        rho_floor=profile.rho_floor,
        unconstrained_weight=profile.unconstrained_weight,
        density=profile.density,
        phi=profile.phi,
        weight_monitor_name=profile.weight_monitor_name,
        density_exponent=profile.density_exponent,
        propagation_exponent=profile.propagation_exponent,
        eta=profile.eta,
        floor_mode=profile.floor_mode,
        floor_eps=profile.floor_eps,
        legacy_unconstrained=profile.legacy_unconstrained,
        used_uniform_fallback=profile.used_uniform_fallback,
        floor_mass=profile.floor_mass,
        min_feasible_step_count=profile.min_feasible_step_count,
        step_count=profile.step_count,
        smoothing_window=profile.smoothing_window,
        r_grid=r_grid,
        nodes=nodes,
        step_sizes=step_sizes,
    )
