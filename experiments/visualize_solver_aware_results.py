#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.result_utils import aggregate_seed_rows, load_result_rows


def _filter_completed_fid_rows(
    rows: Sequence[Dict[str, object]],
    artifact_group: Optional[str] = None,
) -> List[Dict[str, object]]:
    filtered = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        if row.get("metric") != "fid":
            continue
        if artifact_group and row.get("artifact_group") != artifact_group:
            continue
        filtered.append(row)
    return filtered


def _solver_signature(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        row.get("dataset"),
        row.get("path_family"),
        row.get("solver"),
        row.get("strategy_id"),
        row.get("model_output_type"),
        row.get("time_sampling_strategy"),
        row.get("mixed_lambda"),
        row.get("stratified_bins"),
    )


def _collect_plot_rows(
    solver_aware_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    signatures = {_solver_signature(row) for row in solver_aware_rows}
    selected_baselines = []
    for row in baseline_rows:
        if row.get("clock_family") != "uniform":
            continue
        if _solver_signature(row) not in signatures:
            continue
        baseline_row = dict(row)
        baseline_row["node_family"] = "uniform"
        selected_baselines.append(baseline_row)
    return list(solver_aware_rows) + selected_baselines


def _group_solver_rows(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(
        lambda: {"uniform": [], "solver_aware": []}
    )
    for row in rows:
        solver = str(row.get("solver", ""))
        node_family = str(row.get("node_family", "uniform") or "uniform")
        grouped[solver][node_family].append(row)
    return grouped


def plot_fid_curves(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    grouped = _group_solver_rows(rows)
    solvers = [solver for solver in ("euler", "heun2", "stork4") if solver in grouped]
    if not solvers:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(solvers), figsize=(6 * len(solvers), 4.5), squeeze=False)
    for axis, solver in zip(axes[0], solvers):
        for node_family, label, color in (
            ("uniform", "uniform baseline", "#1f77b4"),
            ("solver_aware", "constrained solver-aware / propagation-aware", "#d62728"),
        ):
            series_rows = sorted(grouped[solver][node_family], key=lambda row: int(row["nfe"]))
            if not series_rows:
                continue
            x_values = [int(row["nfe"]) for row in series_rows]
            y_values = [float(row["value_mean"]) for row in series_rows]
            y_std = [float(row.get("value_std", 0.0)) for row in series_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
            axis.fill_between(
                x_values,
                [mean - std for mean, std in zip(y_values, y_std)],
                [mean + std for mean, std in zip(y_values, y_std)],
                color=color,
                alpha=0.12,
            )
        axis.set_title(solver)
        axis.set_xlabel("NFE")
        axis.set_ylabel("FID")
        axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _artifact_path_for_row(results_dir: Path, row: Dict[str, object]) -> Path:
    return (
        results_dir
        / str(row["dataset"])
        / str(row["exp_name"])
        / f"eval_ep{int(row['checkpoint_epoch'])}_nfe{int(row['nfe'])}"
        / "solver_aware_artifacts.pt"
    )


def _load_artifact_for_solver(
    results_dir: Path,
    rows: Sequence[Dict[str, object]],
    solver: str,
) -> Optional[Dict[str, torch.Tensor]]:
    improved_rows = [
        row
        for row in rows
        if row.get("solver") == solver and row.get("node_family") == "solver_aware"
    ]
    improved_rows = sorted(improved_rows, key=lambda row: int(row["nfe"]), reverse=True)
    for row in improved_rows:
        artifact_path = _artifact_path_for_row(results_dir=results_dir, row=row)
        if artifact_path.exists():
            return torch.load(artifact_path, map_location="cpu")
    return None


def plot_solver_artifact(
    artifact: Dict[str, torch.Tensor],
    solver: str,
    output_path: Path,
) -> None:
    s_grid = artifact["s_grid"].cpu()
    q_values = artifact["q_values"].cpu()
    q_smoothed = artifact["q_smoothed"].cpu()
    q_h_values = artifact.get("q_h_values")
    q_h_smoothed = artifact.get("q_h_smoothed")
    if isinstance(q_h_values, torch.Tensor):
        q_h_values = q_h_values.cpu()
    else:
        q_h_values = None
    if isinstance(q_h_smoothed, torch.Tensor):
        q_h_smoothed = q_h_smoothed.cpu()
    else:
        q_h_smoothed = None
    rho_floor = artifact["rho_floor"].cpu()
    unconstrained_weight = artifact["unconstrained_weight"].cpu()
    density = artifact.get("final_density", artifact["density"]).cpu()
    phi = artifact["phi"].cpu()
    r_grid = artifact["r_grid"].cpu()
    nodes = artifact["nodes"].cpu()
    step_sizes = artifact["step_sizes"].cpu()
    ell_values = artifact.get("ell_values")
    g_values = artifact.get("g_values")
    has_propagation = isinstance(ell_values, torch.Tensor) and isinstance(g_values, torch.Tensor)
    if has_propagation:
        ell_values = ell_values.cpu()
        g_values = g_values.cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("Q_E(s)", lambda axis: (
            axis.plot(s_grid, q_values, color="#9c755f", linewidth=1.6, label="raw"),
            axis.plot(s_grid, q_smoothed, color="#2ca02c", linewidth=2.0, label="smoothed"),
            axis.set_ylabel("Q_E"),
            axis.legend(frameon=False),
        )),
    ]
    if q_h_values is not None and q_h_smoothed is not None:
        plot_specs.append(
            ("Q_H(s)", lambda axis: (
                axis.plot(s_grid, q_h_values, color="#8c564b", linewidth=1.6, label="raw"),
                axis.plot(s_grid, q_h_smoothed, color="#17becf", linewidth=2.0, label="smoothed"),
                axis.set_ylabel("Q_H"),
                axis.legend(frameon=False),
            ))
        )
    if has_propagation:
        plot_specs.append(
            ("G(s)", lambda axis: (
                axis.plot(s_grid, g_values, color="#9467bd", linewidth=2.0),
                axis.set_ylabel("G"),
            ))
        )
    plot_specs.extend(
        [
            ("rho_floor(s)", lambda axis: (
                axis.plot(s_grid, rho_floor, color="#ff7f0e", linewidth=2.0),
                axis.set_ylabel("rho_floor"),
            )),
            ("weight(s)", lambda axis: (
                axis.plot(s_grid, unconstrained_weight, color="#bcbd22", linewidth=2.0),
                axis.set_ylabel("unconstrained weight"),
            )),
            ("density(s)", lambda axis: (
                axis.plot(s_grid, density, color="#d62728", linewidth=2.0),
                axis.set_ylabel("constrained density"),
            )),
            ("phi(s)", lambda axis: (
                axis.plot(s_grid, phi, color="#1f77b4", linewidth=2.0),
                axis.plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0),
                axis.set_ylabel("r = phi(s)"),
            )),
            ("nodes", lambda axis: (
                axis.plot(r_grid, nodes, color="#d62728", linewidth=2.0),
                axis.scatter(r_grid, nodes, color="#d62728", s=18),
                axis.plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0),
                axis.set_xlabel("r-grid"),
                axis.set_ylabel("s_n"),
            )),
            ("step sizes", lambda axis: (
                axis.step(r_grid, step_sizes, where="mid", color="#1f77b4", linewidth=2.0),
                axis.scatter(r_grid, step_sizes, color="#1f77b4", s=18),
                axis.set_xlabel("r-grid"),
                axis.set_ylabel("Δs_n"),
            )),
        ]
    )

    panel_count = len(plot_specs)
    ncols = min(3, panel_count)
    nrows = (panel_count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for axis, (title, plotter) in zip(flat_axes, plot_specs):
        plotter(axis)
        axis.set_title(f"{solver}: {title}")
        axis.set_xlabel("s" if title not in {"nodes", "step sizes"} else axis.get_xlabel())
        axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    for axis in flat_axes[len(plot_specs):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    plot_rows: Sequence[Dict[str, object]],
    solver_aware_rows: Sequence[Dict[str, object]],
    output_path: Path,
    baseline_csv: Optional[Path],
) -> None:
    grouped = _group_solver_rows(plot_rows)
    improved_grouped = _group_solver_rows(solver_aware_rows)
    lines = ["# Solver-Aware Clock Summary", ""]
    if baseline_csv is not None:
        lines.append(f"- baseline csv: `{baseline_csv}`")
        lines.append("")
    for solver in ("euler", "heun2", "stork4"):
        payload = grouped.get(solver)
        improved_payload = improved_grouped.get(solver)
        if payload is None or improved_payload is None or not improved_payload["solver_aware"]:
            continue
        baseline_nfes = sorted({int(row["nfe"]) for row in payload["uniform"]})
        improved_rows = sorted(improved_payload["solver_aware"], key=lambda row: int(row["nfe"]))
        checkpoint_sources = sorted(
            {
                str(row.get("monitor_source_checkpoint"))
                for row in improved_rows
                if row.get("monitor_source_checkpoint")
            }
        )
        estimator = str(improved_rows[0].get("solver_aware_monitor_estimator", ""))
        monitor_solver = str(improved_rows[0].get("solver_aware_monitor_solver", ""))
        theorem_backed = str(improved_rows[0].get("solver_aware_theorem_backed", ""))
        propagation = str(improved_rows[0].get("solver_aware_use_propagation", "false"))
        g_mode = str(improved_rows[0].get("solver_aware_g_mode", ""))
        g_estimator = str(improved_rows[0].get("solver_aware_g_estimator", ""))
        eta = str(improved_rows[0].get("solver_aware_eta", ""))
        floor_mode = str(improved_rows[0].get("solver_aware_floor_mode", ""))
        legacy = str(improved_rows[0].get("solver_aware_legacy_unconstrained", "false"))
        status = "constrained theorem-backed proxy" if theorem_backed == "true" else "constrained proxy extension"
        lines.append(f"## {solver}")
        if checkpoint_sources:
            lines.append(f"- checkpoint: `{checkpoint_sources[0]}`")
        lines.append(f"- monitor: `{monitor_solver}` with estimator `{estimator}`")
        lines.append(f"- baseline NFE: {baseline_nfes}")
        lines.append(f"- solver-aware NFE: {[int(row['nfe']) for row in improved_rows]}")
        lines.append(f"- propagation: `{propagation}` / mode `{g_mode}` / estimator `{g_estimator}`")
        lines.append(f"- constrained: `eta={eta}` / floor_mode `{floor_mode}` / legacy `{legacy}`")
        lines.append(f"- status: {status}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_solver_aware_results(
    results_dir: Path,
    csv_path: Optional[Path] = None,
    artifact_group: Optional[str] = None,
    baseline_csv: Optional[Path] = None,
) -> None:
    csv_path = csv_path or results_dir / "results.csv"
    solver_aware_rows = aggregate_seed_rows(
        _filter_completed_fid_rows(load_result_rows(csv_path), artifact_group)
    )
    baseline_rows = []
    if baseline_csv is not None:
        baseline_rows = aggregate_seed_rows(
            _filter_completed_fid_rows(load_result_rows(baseline_csv), artifact_group=None)
        )
    plot_rows = _collect_plot_rows(
        solver_aware_rows=solver_aware_rows,
        baseline_rows=baseline_rows,
    )

    plots_dir = results_dir / "plots"
    plot_fid_curves(rows=plot_rows, output_path=plots_dir / "solver_aware_fid_vs_nfe.png")
    for solver in ("euler", "heun2", "stork4"):
        artifact = _load_artifact_for_solver(results_dir=results_dir, rows=solver_aware_rows, solver=solver)
        if artifact is None:
            continue
        plot_solver_artifact(
            artifact=artifact,
            solver=solver,
            output_path=plots_dir / f"solver_aware_{solver}_monitor_phi_nodes.png",
        )
    write_summary(
        plot_rows=plot_rows,
        solver_aware_rows=solver_aware_rows,
        output_path=results_dir / "solver_aware_summary.md",
        baseline_csv=baseline_csv,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--artifact_group", type=str, default=None)
    parser.add_argument("--baseline_csv", type=Path, default=None)
    args = parser.parse_args()
    visualize_solver_aware_results(
        results_dir=args.results_dir,
        csv_path=args.csv,
        artifact_group=args.artifact_group,
        baseline_csv=args.baseline_csv,
    )


if __name__ == "__main__":
    main()
