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
) -> None:
    s_grid = artifact["s_grid"].cpu()
    q_values = artifact["q_values"].cpu()
    q_smoothed = artifact["q_smoothed"].cpu()
    phi = artifact["phi"].cpu()
    r_grid = artifact["r_grid"].cpu()
    nodes = artifact["nodes"].cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(s_grid, q_values, color="#9c755f", linewidth=1.8, label="Q(s)")
    axes[0].plot(s_grid, q_smoothed, color="#2ca02c", linewidth=2.2, label="smoothed Q(s)")
    axes[0].set_title(f"{solver}: monitor")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("monitor value")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False)

    axes[1].plot(s_grid, phi, color="#1f77b4", linewidth=2.2)
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{solver}: phi(s)")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("r = phi(s)")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    axes[2].plot(r_grid, nodes, color="#d62728", linewidth=2.0)
    axes[2].scatter(r_grid, nodes, color="#d62728", s=18)
    axes[2].plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linestyle="--", linewidth=1.0)
    axes[2].set_title(f"{solver}: nodes")
    axes[2].set_xlabel("r-grid")
    axes[2].set_ylabel("s_n = psi(n / N)")
    axes[2].grid(alpha=0.25, linestyle="--", linewidth=0.8)

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
        theorem_backed = str(improved_rows[0].get("solver_aware_theorem_backed", ""))
        notes = (
            "theorem-backed"
            if theorem_backed == "true"
            else "heuristic phase-1"
        )
        lines.append(f"## {solver}")
        if checkpoint_sources:
            lines.append(f"- checkpoint: `{checkpoint_sources[0]}`")
        lines.append(f"- monitor: `{monitor_solver}` with estimator `{estimator}`")
        lines.append(f"- NFE: {baseline_nfes}")
        lines.append(f"- status: {notes}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_solver_aware_results(
    results_dir: Path,
    csv_path: Optional[Path] = None,
    artifact_group: Optional[str] = None,
) -> None:
    csv_path = csv_path or results_dir / "results.csv"
    rows = aggregate_seed_rows(_filter_completed_fid_rows(load_result_rows(csv_path), artifact_group))

    plots_dir = results_dir / "plots"
    plot_fid_curves(rows=rows, output_path=plots_dir / "solver_aware_fid_vs_nfe.png")
    for solver in ("euler", "heun2", "stork4"):
        artifact = _load_artifact_for_solver(results_dir=results_dir, rows=rows, solver=solver)
        if artifact is None:
            continue
        plot_solver_artifact(
            artifact=artifact,
            solver=solver,
            output_path=plots_dir / f"solver_aware_{solver}_monitor_phi_nodes.png",
        )
    write_summary(rows=rows, output_path=results_dir / "solver_aware_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--artifact_group", type=str, default=None)
    args = parser.parse_args()
    visualize_solver_aware_results(
        results_dir=args.results_dir,
        csv_path=args.csv,
        artifact_group=args.artifact_group,
    )


if __name__ == "__main__":
    main()
