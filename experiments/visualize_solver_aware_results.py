#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    artifact_group: Optional[str],
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
            ("uniform", "uniform nodes", "#1f77b4"),
            ("solver_aware", "solver-aware nodes", "#d62728"),
        ):
            series_rows = sorted(grouped[solver][node_family], key=lambda row: int(row["nfe"]))
            if not series_rows:
                continue
            axis.plot(
                [int(row["nfe"]) for row in series_rows],
                [float(row["value_mean"]) for row in series_rows],
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
            axis.fill_between(
                [int(row["nfe"]) for row in series_rows],
                [float(row["value_mean"]) - float(row.get("value_std", 0.0)) for row in series_rows],
                [float(row["value_mean"]) + float(row.get("value_std", 0.0)) for row in series_rows],
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
    compare_artifact: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    s_grid = artifact["s_grid"].cpu()
    q_values = artifact["q_values"].cpu()
    q_smoothed = artifact["q_smoothed"].cpu()
    density = artifact["density"].cpu()
    phi = artifact["phi"].cpu()
    r_grid = artifact["r_grid"].cpu()
    nodes = artifact["nodes"].cpu()
    monitor_family = str(artifact.get("monitor_family", "legacy_continuous"))
    q_curve_name = str(artifact.get("q_curve_name", "Q(s)"))
    q_values_by_budget = artifact.get("q_values_by_budget", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    if isinstance(q_values_by_budget, dict) and q_values_by_budget:
        budget_items = sorted(q_values_by_budget.items(), key=lambda item: int(item[0]))
        for budget, curve in budget_items:
            curve_tensor = curve.cpu() if isinstance(curve, torch.Tensor) else torch.tensor(curve)
            axes[0].plot(
                s_grid,
                curve_tensor,
                linewidth=1.0,
                alpha=0.55,
                label=f"budget {budget}",
            )
    axes[0].plot(s_grid, q_values, color="#9c755f", linewidth=1.8, label=q_curve_name)
    axes[0].plot(s_grid, q_smoothed, color="#2ca02c", linewidth=2.2, label=f"smoothed {q_curve_name}")
    if compare_artifact is not None:
        axes[0].plot(
            compare_artifact["s_grid"].cpu(),
            compare_artifact["q_values"].cpu(),
            color="#7f7f7f",
            linewidth=1.4,
            linestyle="--",
            label="compare q",
        )
    axes[0].set_title(f"{solver}: {monitor_family}")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("monitor value")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False)

    axes[1].plot(s_grid, density, color="#9467bd", linewidth=2.0)
    if compare_artifact is not None:
        axes[1].plot(
            compare_artifact["s_grid"].cpu(),
            compare_artifact["density"].cpu(),
            color="#7f7f7f",
            linewidth=1.4,
            linestyle="--",
        )
    axes[1].set_title(f"{solver}: rho(s)")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    axes[2].plot(s_grid, phi, color="#1f77b4", linewidth=2.2)
    axes[2].plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0)
    if compare_artifact is not None:
        axes[2].plot(
            compare_artifact["s_grid"].cpu(),
            compare_artifact["phi"].cpu(),
            color="#7f7f7f",
            linewidth=1.4,
            linestyle="--",
        )
    axes[2].set_title(f"{solver}: phi(s)")
    axes[2].set_xlabel("s")
    axes[2].set_ylabel("r = phi(s)")
    axes[2].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    axes[3].plot(r_grid, nodes, color="#d62728", linewidth=2.0)
    axes[3].scatter(r_grid, nodes, color="#d62728", s=18)
    axes[3].plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0)
    axes[3].set_title(f"{solver}: nodes")
    axes[3].set_xlabel("r-grid")
    axes[3].set_ylabel("s_n = psi(n / N)")
    axes[3].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    grouped = _group_solver_rows(rows)
    lines = ["# Solver-Aware Clock Phase-1 Summary", ""]
    for solver in ("euler", "heun2", "stork4"):
        payload = grouped.get(solver)
        if payload is None or not payload["solver_aware"]:
            continue
        baseline_nfes = sorted({int(row["nfe"]) for row in payload["uniform"]})
        improved_rows = sorted(payload["solver_aware"], key=lambda row: int(row["nfe"]))
        checkpoint_sources = sorted(
            {
                str(row.get("monitor_source_checkpoint"))
                for row in improved_rows
                if row.get("monitor_source_checkpoint")
            }
        )
        estimator = str(improved_rows[0].get("solver_aware_monitor_estimator", ""))
        monitor_solver = str(improved_rows[0].get("solver_aware_monitor_solver", ""))
        monitor_family = str(improved_rows[0].get("solver_aware_monitor_family", "legacy_continuous"))
        budget_mode = str(improved_rows[0].get("solver_aware_budget_mode", "single_budget"))
        theorem_backed = str(improved_rows[0].get("solver_aware_theorem_backed", ""))
        notes = (
            "theorem-backed"
            if theorem_backed == "true"
            else "heuristic phase-1"
        )
        lines.append(f"## {solver}")
        if checkpoint_sources:
            lines.append(f"- checkpoint: `{checkpoint_sources[0]}`")
        lines.append(f"- monitor_family: `{monitor_family}`")
        lines.append(f"- monitor: `{monitor_solver}` with estimator `{estimator}`")
        lines.append(f"- budget_mode: `{budget_mode}`")
        if monitor_family == "defect_based":
            lines.append("- expectation: `z ~ p_s`")
        lines.append(f"- NFE: {baseline_nfes}")
        lines.append(f"- status: {notes}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_solver_aware_results(
    results_dir: Path,
    csv_path: Optional[Path] = None,
    artifact_group: Optional[str] = None,
    compare_results_dir: Optional[Path] = None,
) -> None:
    csv_path = csv_path or results_dir / "results.csv"
    rows = aggregate_seed_rows(_filter_completed_fid_rows(load_result_rows(csv_path), artifact_group))

    plots_dir = results_dir / "plots"
    plot_fid_curves(rows=rows, output_path=plots_dir / "solver_aware_fid_vs_nfe.png")
    for solver in ("euler", "heun2", "stork4"):
        artifact = _load_artifact_for_solver(results_dir=results_dir, rows=rows, solver=solver)
        if artifact is None:
            continue
        compare_artifact = None
        if compare_results_dir is not None:
            compare_artifact = _load_artifact_for_solver(
                results_dir=compare_results_dir,
                rows=rows,
                solver=solver,
            )
        plot_solver_artifact(
            artifact=artifact,
            solver=solver,
            output_path=plots_dir / f"solver_aware_{solver}_monitor_phi_nodes.png",
            compare_artifact=compare_artifact,
        )
    write_summary(rows=rows, output_path=results_dir / "solver_aware_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--artifact_group", type=str, default=None)
    parser.add_argument("--compare_results_dir", type=Path, default=None)
    args = parser.parse_args()
    visualize_solver_aware_results(
        results_dir=args.results_dir,
        csv_path=args.csv,
        artifact_group=args.artifact_group,
        compare_results_dir=args.compare_results_dir,
    )


if __name__ == "__main__":
    main()
