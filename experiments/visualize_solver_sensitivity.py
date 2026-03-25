#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.result_utils import (
    aggregate_seed_rows,
    filter_rows,
    is_ft_clock_family,
    load_result_rows,
    write_table_csv,
)
from experiments.plot_style import selected_nfe_ticks, transform_focus_axis_values


SHARED_FAIR_BUDGETS = frozenset({6, 12, 18, 24, 30, 48, 96})
REQUIRED_METRICS = {"fid", "precision", "recall", "is_mean", "is_std"}


def solver_budget_flags(solver: str, nfe: int) -> Dict[str, object]:
    if solver == "euler":
        is_exact_budget = True
    elif solver == "heun2":
        is_exact_budget = nfe % 2 == 0
    elif solver == "rk3":
        is_exact_budget = nfe % 3 == 0
    elif solver == "stork4":
        is_exact_budget = True
    else:
        raise ValueError(f"Unsupported solver={solver}.")
    return {
        "used_tail_step": not is_exact_budget and solver in {"heun2", "rk3"},
        "is_exact_budget": is_exact_budget,
        "is_shared_budget": is_exact_budget and nfe in SHARED_FAIR_BUDGETS,
    }


def method_label(path_family: str, clock_family: str, beta: Optional[float]) -> str:
    if clock_family == "uniform":
        return f"{path_family}_uniform"
    if beta is None:
        return f"{path_family}_{clock_family}"
    beta_tag = str(beta).replace(".", "_")
    return f"{path_family}_ft_beta_{beta_tag}"


def pivot_metric_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("path_family"),
            row.get("clock_family"),
            row.get("clock_param_name"),
            row.get("clock_param_value"),
            row.get("solver"),
            row.get("nfe"),
            row.get("checkpoint_epoch"),
            row.get("artifact_group"),
        )
        if key not in grouped:
            flags = solver_budget_flags(str(row["solver"]), int(row["nfe"]))
            beta = row.get("clock_param_value")
            grouped[key] = {
                "dataset": row.get("dataset"),
                "path_family": row.get("path_family"),
                "clock_family": row.get("clock_family"),
                "clock_param_name": row.get("clock_param_name"),
                "beta": beta,
                "solver": row.get("solver"),
                "nfe": int(row.get("nfe", 0)),
                "actual_network_calls": int(row.get("nfe", 0)),
                "step_count": int(row.get("step_count", 0)),
                "checkpoint_epoch": int(row.get("checkpoint_epoch", 0)),
                "artifact_group": row.get("artifact_group"),
                "method": method_label(
                    str(row.get("path_family")),
                    str(row.get("clock_family")),
                    None if beta is None else float(beta),
                ),
                **flags,
            }
        metric_name = str(row["metric"])
        grouped[key][metric_name] = float(row["value_mean"])
        if metric_name == "fid":
            grouped[key]["fid_mean"] = float(row["value_mean"])
            grouped[key]["fid_std"] = float(row["value_std"])
        elif metric_name in {"precision", "recall"}:
            grouped[key][metric_name] = float(row["value_mean"])
        elif metric_name == "is_mean":
            grouped[key]["is_mean"] = float(row["value_mean"])
        elif metric_name == "is_std":
            grouped[key]["is_std"] = float(row["value_mean"])
    table = []
    for row in grouped.values():
        metrics_present = {metric for metric in REQUIRED_METRICS if metric in row}
        row["has_complete_metrics"] = metrics_present == REQUIRED_METRICS
        table.append(row)
    return sorted(
        table,
        key=lambda row: (
            str(row["dataset"]),
            str(row["path_family"]),
            str(row["method"]),
            str(row["solver"]),
            int(row["nfe"]),
        ),
    )


def fairness_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("is_exact_budget")
        and row.get("is_shared_budget")
        and row.get("has_complete_metrics")
    ]


def appendix_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return list(rows)


def plot_fid_vs_nfe(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    series: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    all_nfes = sorted({int(row["nfe"]) for row in rows})
    tick_values = selected_nfe_ticks(all_nfes)
    for row in rows:
        label = f"{row['method']} / {row['solver']}"
        series[label].append(row)

    for label, series_rows in sorted(series.items()):
        ordered = sorted(series_rows, key=lambda row: int(row["nfe"]))
        nfe_values = [int(row["nfe"]) for row in ordered]
        x = transform_focus_axis_values(nfe_values)
        y = [float(row["fid_mean"]) for row in ordered]
        y_std = [float(row.get("fid_std", 0.0)) for row in ordered]
        lower_label = label.lower()
        if "stork" in lower_label:
            color = "#d62728"
            linestyle = "-"
            marker = "x"
        elif "rk3" in lower_label:
            color = "#2ca02c"
            linestyle = "-."
            marker = "s"
        elif "heun2" in lower_label:
            color = "#1f77b4"
            linestyle = "--"
            marker = "x"
        elif "euler" in lower_label:
            color = "#ff7f0e"
            linestyle = "--"
            marker = "^"
        else:
            color = "#7f7f7f"
            linestyle = "-"
            marker = "o"
        if "uniform" in lower_label:
            linewidth = 1.8
            alpha = 0.9
        else:
            linewidth = 2.2
            alpha = 1.0
        plt.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            markersize=5,
            alpha=alpha,
            label=label,
        )
        lower = [value - std for value, std in zip(y, y_std)]
        upper = [value + std for value, std in zip(y, y_std)]
        plt.fill_between(x, lower, upper, alpha=0.08, color=color)
    plt.xlabel("NFE")
    plt.ylabel("FID")
    plt.xticks(transform_focus_axis_values(tick_values), [str(tick) for tick in tick_values])
    plt.grid(alpha=0.22, linestyle="--", linewidth=0.8)
    plt.legend(fontsize=8, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, object]]]] = defaultdict(
        lambda: {"baseline": [], "ft": []}
    )
    for row in rows:
        key = (str(row["path_family"]), str(row["solver"]))
        if row["clock_family"] == "uniform":
            grouped[key]["baseline"].append(row)
        elif is_ft_clock_family(row["clock_family"]):
            grouped[key]["ft"].append(row)

    lines = ["# Solver Sensitivity Summary", ""]
    if not grouped:
        lines.append("No fairness rows were found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for (path_family, solver), payload in sorted(grouped.items()):
        baseline_by_nfe = {int(row["nfe"]): row for row in payload["baseline"]}
        ft_by_nfe: Dict[int, Dict[str, object]] = {}
        for row in payload["ft"]:
            nfe = int(row["nfe"])
            current = ft_by_nfe.get(nfe)
            if current is None or float(row["fid_mean"]) < float(current["fid_mean"]):
                ft_by_nfe[nfe] = row

        common_nfes = sorted(set(baseline_by_nfe) & set(ft_by_nfe))
        lines.append(f"## {path_family} / {solver}")
        if not common_nfes:
            lines.append("- No overlapping baseline-vs-FT fairness rows.")
            lines.append("")
            continue

        improvements = []
        wins = 0
        beta_votes: Dict[float, int] = defaultdict(int)
        for nfe in common_nfes:
            baseline = baseline_by_nfe[nfe]
            best_ft = ft_by_nfe[nfe]
            improvement = float(baseline["fid_mean"]) - float(best_ft["fid_mean"])
            improvements.append(improvement)
            if improvement > 0:
                wins += 1
            beta = best_ft.get("beta")
            if beta is not None:
                beta_votes[float(beta)] += 1

        mean_improvement = sum(improvements) / len(improvements)
        preferred_beta = None
        if beta_votes:
            preferred_beta = sorted(beta_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        lines.append(
            f"- FT beats baseline on {wins}/{len(common_nfes)} shared exact-NFE budgets; "
            f"mean FID improvement = {mean_improvement:.4f}."
        )
        if preferred_beta is not None:
            lines.append(f"- Most competitive beta on fairness rows: {preferred_beta:.1f}.")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_solver_sensitivity(
    csv_path: Path,
    out_dir: Path,
    artifact_group: Optional[str] = "ft_clock_solver_sensitivity",
) -> None:
    rows = load_result_rows(csv_path)
    rows = filter_rows(rows, status="completed", artifact_group=artifact_group)
    aggregated = aggregate_seed_rows(rows)
    pivoted_rows = pivot_metric_rows(aggregated)

    fair_rows = fairness_rows(pivoted_rows)
    appendix = appendix_rows(pivoted_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_table_csv(out_dir / "solver_fairness_table.csv", fair_rows)
    write_table_csv(out_dir / "solver_appendix_all_budgets.csv", appendix)
    plot_fid_vs_nfe(fair_rows, out_dir / "solver_fairness_fid_vs_nfe.png")
    write_summary(fair_rows, out_dir / "solver_sensitivity_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact_group", type=str, default="ft_clock_solver_sensitivity")
    args = parser.parse_args()
    visualize_solver_sensitivity(args.csv, args.out, artifact_group=args.artifact_group)


if __name__ == "__main__":
    main()
