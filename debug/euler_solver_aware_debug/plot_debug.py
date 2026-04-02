from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from monitor_debug import (
    MonitorDebugBundle,
    NodeDiagnostics,
    VariantProfile,
    _sanitize_name,
    _tensor_to_list,
)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:  # pragma: no cover - depends on runtime.
        raise RuntimeError(
            "Missing runtime dependency 'matplotlib'. Install the project runtime environment "
            "before rendering Euler debug plots."
        ) from error
    return plt


def _variant_dir(output_root: Path, variant_name: str) -> Path:
    return output_root / "profiles" / _sanitize_name(variant_name)


def _evaluation_dir(output_root: Path, variant_name: str, nfe: int) -> Path:
    return output_root / "evaluations" / _sanitize_name(variant_name) / f"nfe_{int(nfe):03d}"


def _plot_profile_curves(plt, profile: VariantProfile, output_root: Path) -> None:
    variant_dir = _variant_dir(output_root, profile.variant_name)
    s_grid = _tensor_to_list(profile.s_grid)
    q_raw = _tensor_to_list(profile.q_raw)
    q_smoothed = _tensor_to_list(profile.q_smoothed)
    q_clipped = _tensor_to_list(profile.q_clipped)
    density = _tensor_to_list(profile.density)
    phi = _tensor_to_list(profile.phi)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(s_grid, q_raw, linewidth=2.0, color="#9c755f")
    axis.set_xlabel("s")
    axis.set_ylabel("Q_E(s)")
    axis.set_title(f"{profile.variant_name}: raw Q_E(s)")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(variant_dir / "raw_q.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(s_grid, q_smoothed, linewidth=2.0, color="#2ca02c")
    axis.set_xlabel("s")
    axis.set_ylabel("Q_E(s)")
    axis.set_title(f"{profile.variant_name}: smoothed Q_E(s)")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(variant_dir / "smoothed_q.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(s_grid, density, linewidth=2.0, color="#d62728")
    axis.set_xlabel("s")
    axis.set_ylabel("rho(s)")
    axis.set_title(f"{profile.variant_name}: density")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(variant_dir / "density.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(s_grid, phi, linewidth=2.0, color="#1f77b4")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#7f7f7f")
    axis.set_xlabel("s")
    axis.set_ylabel("phi(s)")
    axis.set_title(f"{profile.variant_name}: cumulative phi(s)")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(variant_dir / "phi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.plot(s_grid, q_raw, linewidth=2.0, color="#9c755f", label="raw")
    axis.plot(s_grid, q_smoothed, linewidth=2.0, color="#2ca02c", label="smoothed")
    axis.plot(s_grid, q_clipped, linewidth=2.0, color="#ff7f0e", label="clipped")
    axis.set_xlabel("s")
    axis.set_ylabel("Q_E(s)")
    axis.set_title(f"{profile.variant_name}: raw vs smoothed vs clipped")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(variant_dir / "raw_vs_smoothed_vs_clipped.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_node_diagnostics(
    plt,
    diagnostics: NodeDiagnostics,
    output_root: Path,
) -> None:
    variant_dir = _variant_dir(output_root, diagnostics.variant_name) / f"nfe_{int(diagnostics.nfe):03d}"
    evaluation_dir = _evaluation_dir(output_root, diagnostics.variant_name, diagnostics.nfe)
    variant_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    r_grid = _tensor_to_list(diagnostics.r_grid)
    nodes = _tensor_to_list(diagnostics.nodes)
    uniform_nodes = _tensor_to_list(diagnostics.uniform_nodes)
    step_sizes = _tensor_to_list(diagnostics.step_sizes)
    step_indices = list(range(1, len(nodes)))

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(r_grid, nodes, linewidth=2.0, color="#d62728")
    axis.scatter(r_grid, nodes, color="#d62728", s=22)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#7f7f7f")
    axis.set_xlabel("r")
    axis.set_ylabel("psi(r)")
    axis.set_title(f"{diagnostics.variant_name}: inverse psi at NFE={diagnostics.nfe}")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(variant_dir / "psi_nodes.png", dpi=220, bbox_inches="tight")
    fig.savefig(evaluation_dir / "psi_nodes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.bar(step_indices, step_sizes, color="#1f77b4", alpha=0.9)
    axis.set_xlabel("step index")
    axis.set_ylabel("step size")
    axis.set_title(f"{diagnostics.variant_name}: step sizes at NFE={diagnostics.nfe}")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(variant_dir / "step_sizes.png", dpi=220, bbox_inches="tight")
    fig.savefig(evaluation_dir / "step_sizes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 1.8))
    axis.scatter(nodes, [0.55] * len(nodes), color="#d62728", s=34, label="solver-aware")
    axis.scatter(uniform_nodes, [0.45] * len(uniform_nodes), color="#1f77b4", s=24, label="uniform")
    axis.set_yticks([])
    axis.set_xlabel("s in [0, 1]")
    axis.set_title(f"{diagnostics.variant_name}: node locations at NFE={diagnostics.nfe}")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8, axis="x")
    axis.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(variant_dir / "nodes_positions.png", dpi=220, bbox_inches="tight")
    fig.savefig(evaluation_dir / "nodes_positions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(r_grid, uniform_nodes, linewidth=2.0, color="#1f77b4", label="uniform")
    axis.plot(r_grid, nodes, linewidth=2.0, color="#d62728", label="solver-aware")
    axis.scatter(r_grid, nodes, color="#d62728", s=20)
    axis.set_xlabel("r")
    axis.set_ylabel("s_n")
    axis.set_title(f"{diagnostics.variant_name}: uniform vs solver-aware nodes (NFE={diagnostics.nfe})")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(variant_dir / "uniform_vs_solver_aware_nodes.png", dpi=220, bbox_inches="tight")
    fig.savefig(evaluation_dir / "uniform_vs_solver_aware_nodes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_stability(plt, bundle: MonitorDebugBundle, output_root: Path) -> None:
    for batch_size, summary in sorted(bundle.stability.items()):
        stability_dir = output_root / "stability" / f"batch_{int(batch_size):03d}"
        stability_dir.mkdir(parents=True, exist_ok=True)
        s_grid = _tensor_to_list(summary.s_grid)
        mean_curve = _tensor_to_list(summary.mean_curve)
        std_curve = _tensor_to_list(summary.std_curve)
        cv_curve = _tensor_to_list(summary.cv_curve)

        lower = [max(0.0, mean - std) for mean, std in zip(mean_curve, std_curve)]
        upper = [mean + std for mean, std in zip(mean_curve, std_curve)]
        fig, axis = plt.subplots(figsize=(7.4, 4.8))
        axis.plot(s_grid, mean_curve, linewidth=2.0, color="#1f77b4", label="mean")
        axis.fill_between(s_grid, lower, upper, color="#1f77b4", alpha=0.18, label="mean ± std")
        axis.set_xlabel("s")
        axis.set_ylabel("Q_E(s)")
        axis.set_title(f"Euler monitor stability: batch_size={batch_size}")
        axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(stability_dir / "q_mean_std.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.4, 4.8))
        axis.plot(s_grid, cv_curve, linewidth=2.0, color="#d62728")
        axis.set_xlabel("s")
        axis.set_ylabel("coefficient of variation")
        axis.set_title(f"Euler monitor CV: batch_size={batch_size}")
        axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(stability_dir / "q_cv.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _plot_grid_sweep(
    plt,
    bundle: MonitorDebugBundle,
    output_root: Path,
    results_rows: Optional[Sequence[Mapping[str, Any]]],
) -> None:
    grid_variants: Dict[str, VariantProfile] = {}
    if "solver_aware_current_impl" in bundle.profiles:
        grid_variants["solver_aware_current_impl"] = bundle.profiles["solver_aware_current_impl"]
    for variant_name, profile in bundle.profiles.items():
        if profile.variant_group == "grid_sweep":
            grid_variants[variant_name] = profile
    if len(grid_variants) <= 1:
        return

    grid_dir = output_root / "grid_sweep"
    grid_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    for variant_name, profile in sorted(
        grid_variants.items(),
        key=lambda item: int(item[1].monitor_grid_size),
    ):
        label = f"{variant_name} (grid={profile.monitor_grid_size})"
        axes[0].plot(
            _tensor_to_list(profile.s_grid),
            _tensor_to_list(profile.phi),
            linewidth=2.0,
            label=label,
        )
    axes[0].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#7f7f7f")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("phi(s)")
    axes[0].set_title("Grid-size sweep: phi(s)")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    for variant_name, profile in sorted(
        grid_variants.items(),
        key=lambda item: int(item[1].monitor_grid_size),
    ):
        if 12 not in bundle.node_diagnostics.get(variant_name, {}):
            continue
        diagnostics = bundle.node_diagnostics[variant_name][12]
        label = f"{profile.monitor_grid_size}"
        axes[1].plot(
            _tensor_to_list(diagnostics.r_grid),
            _tensor_to_list(diagnostics.nodes),
            linewidth=2.0,
            label=label,
        )
    axes[1].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#7f7f7f")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("psi(r)")
    axes[1].set_title("Grid-size sweep: nodes at NFE=12")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[1].legend(frameon=False, title="grid", fontsize=8)
    fig.tight_layout()
    fig.savefig(grid_dir / "phi_and_nodes_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if results_rows:
        rows = [
            row
            for row in results_rows
            if str(row.get("variant_name", "")).startswith("solver_aware_grid_")
            or str(row.get("variant_name", "")) == "solver_aware_current_impl"
        ]
        if rows:
            fig, axis = plt.subplots(figsize=(7.8, 5.0))
            grouped: Dict[str, list] = {}
            for row in rows:
                grouped.setdefault(str(row["variant_name"]), []).append(row)
            for variant_name, variant_rows in sorted(
                grouped.items(),
                key=lambda item: int(
                    bundle.profiles[item[0]].monitor_grid_size if item[0] in bundle.profiles else 0
                ),
            ):
                sorted_rows = sorted(variant_rows, key=lambda item: int(item["nfe"]))
                axis.plot(
                    [int(item["nfe"]) for item in sorted_rows],
                    [float(item["fid"]) for item in sorted_rows],
                    marker="o",
                    linewidth=2.0,
                    label=f"{variant_name} (grid={bundle.profiles[variant_name].monitor_grid_size})",
                )
            axis.set_xlabel("NFE")
            axis.set_ylabel("FID")
            axis.set_title("Grid-size sweep: FID vs NFE")
            axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
            axis.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(grid_dir / "grid_sweep_fid.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def _plot_fid_overview(plt, results_rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    if not results_rows:
        return
    overview_dir = output_root / "evaluations"
    overview_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, list] = {}
    for row in results_rows:
        grouped.setdefault(str(row["variant_name"]), []).append(row)

    fig, axis = plt.subplots(figsize=(9.0, 5.4))
    highlight_names = [
        "uniform_baseline",
        "solver_aware_no_smoothing",
        "solver_aware_current_impl",
        "solver_aware_gaussian",
    ]
    for variant_name in highlight_names:
        if variant_name not in grouped:
            continue
        variant_rows = sorted(grouped[variant_name], key=lambda item: int(item["nfe"]))
        axis.plot(
            [int(item["nfe"]) for item in variant_rows],
            [float(item["fid"]) for item in variant_rows],
            marker="o",
            linewidth=2.0,
            label=variant_name,
        )
    axis.set_xlabel("NFE")
    axis.set_ylabel("FID")
    axis.set_title("Euler baseline vs key solver-aware variants")
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(overview_dir / "fid_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_debug_plots(
    *,
    bundle: MonitorDebugBundle,
    output_root: Path,
    results_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    plt = _load_pyplot()
    for profile in bundle.profiles.values():
        _plot_profile_curves(plt, profile, output_root)
    for diagnostics_by_nfe in bundle.node_diagnostics.values():
        for diagnostics in diagnostics_by_nfe.values():
            _plot_node_diagnostics(plt, diagnostics, output_root)
    _plot_stability(plt, bundle, output_root)
    _plot_grid_sweep(plt, bundle, output_root, results_rows)
    if results_rows:
        _plot_fid_overview(plt, results_rows, output_root)
