#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.result_utils import (
    aggregate_seed_rows,
    baseline_vs_best_beta,
    best_rows_by_key,
    filter_rows,
    is_ft_clock_family,
    load_result_rows,
    rows_to_matrix,
    write_table_csv,
)
from experiments.plot_style import selected_nfe_ticks, transform_focus_axis_values


def _clock_label(row: Dict[str, object]) -> str:
    family = str(row["clock_family"])
    if family == "ft_linear_beta":
        family = "ft_beta"
    param_name = row.get("clock_param_name")
    param_value = row.get("clock_param_value")
    strategy_id = str(row.get("strategy_id", ""))
    if param_name not in {None, "", "none"} and param_value is not None:
        label = f"{family} ({param_name}={param_value})"
    else:
        label = family
    if strategy_id:
        return f"strategy {strategy_id}: {label}"
    return label


def _plot_main_curve(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_clock_label(row), []).append(row)
    all_nfes = sorted({int(row["nfe"]) for row in rows})
    tick_values = selected_nfe_ticks(all_nfes)
    plt.figure(figsize=(10, 6))
    ft_labels = [label for label in sorted(grouped.keys()) if label.startswith("ft_beta")]
    ft_palette = ["#d62728", "#ff7f0e", "#c44e52", "#e377c2", "#8c564b", "#bcbd22"]
    ft_colors = {label: ft_palette[index % len(ft_palette)] for index, label in enumerate(ft_labels)}
    other_palette = ["#7f7f7f", "#17becf", "#9467bd", "#8c8c8c"]
    other_labels = [
        label
        for label in sorted(grouped.keys())
        if label != "uniform" and not label.startswith("ft_beta")
    ]
    other_colors = {
        label: other_palette[index % len(other_palette)]
        for index, label in enumerate(other_labels)
    }
    for label, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: row["nfe"])
        nfe_values = [int(row["nfe"]) for row in ordered]
        x = transform_focus_axis_values(nfe_values)
        y = [row["value_mean"] for row in ordered]
        if label == "uniform":
            color = "#1f77b4"
            linestyle = "--"
            marker = "x"
            linewidth = 2.2
        elif label.startswith("ft_beta"):
            color = ft_colors[label]
            linestyle = "-"
            marker = "x"
            linewidth = 2.0
        else:
            color = other_colors.get(label, "#7f7f7f")
            linestyle = "--"
            marker = "o"
            linewidth = 1.8
        plt.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            markersize=5,
            label=label,
        )
        if any(float(row["value_std"]) > 0.0 for row in ordered):
            lower = [row["value_mean"] - row["value_std"] for row in ordered]
            upper = [row["value_mean"] + row["value_std"] for row in ordered]
            plt.fill_between(x, lower, upper, alpha=0.10, color=color)
    plt.xlabel("NFE")
    plt.ylabel("FID")
    plt.xticks(transform_focus_axis_values(tick_values), [str(tick) for tick in tick_values])
    plt.grid(alpha=0.22, linestyle="--", linewidth=0.8)
    plt.legend(loc="upper right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "fid_vs_nfe.png", dpi=300)
    plt.close()


def _plot_heatmap(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("strategy_id", "")), []).append(row)
    for strategy_id, grouped_rows in grouped.items():
        matrix = rows_to_matrix(grouped_rows, row_field="clock_param_value", col_field="nfe")
        beta_values = sorted(matrix.keys())
        nfe_values = sorted({nfe for values in matrix.values() for nfe in values.keys()})
        if not beta_values or not nfe_values:
            continue
        heatmap = [[matrix[beta].get(nfe, float("nan")) for nfe in nfe_values] for beta in beta_values]
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap, aspect="auto", origin="lower")
        plt.colorbar(label="FID")
        plt.xticks(range(len(nfe_values)), [str(nfe) for nfe in nfe_values])
        plt.yticks(range(len(beta_values)), [str(beta) for beta in beta_values])
        plt.xlabel("NFE")
        plt.ylabel("beta")
        plt.tight_layout()
        filename = "fid_heatmap_beta_nfe.png"
        if strategy_id:
            filename = f"fid_heatmap_beta_nfe_strategy_{strategy_id}.png"
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()


def _write_ablation_tables(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    if not rows or not any(str(row.get("strategy_id", "")) for row in rows):
        return
    best_rows = []
    grouped: Dict[Tuple[object, object, object], List[Dict[str, object]]] = {}
    for row in rows:
        if not is_ft_clock_family(row["clock_family"]):
            continue
        key = (row.get("clock_param_value"), row.get("solver"), row.get("nfe"))
        grouped.setdefault(key, []).append(row)
    comparison_rows = []
    for (beta, solver, nfe), group_rows in sorted(grouped.items()):
        best_row = min(group_rows, key=lambda row: float(row["value_mean"]))
        best_rows.append(
            {
                "beta": beta,
                "solver": solver,
                "nfe": nfe,
                "best_strategy_id": best_row.get("strategy_id", ""),
                "best_fid_mean": best_row["value_mean"],
                "best_fid_std": best_row["value_std"],
            }
        )
        by_strategy = {str(row.get("strategy_id", "")): row for row in group_rows}
        if "A" in by_strategy and "B" in by_strategy:
            comparison_rows.append(
                {
                    "beta": beta,
                    "solver": solver,
                    "nfe": nfe,
                    "strategy_A_fid_mean": by_strategy["A"]["value_mean"],
                    "strategy_B_fid_mean": by_strategy["B"]["value_mean"],
                    "strategy_B_minus_A": by_strategy["B"]["value_mean"] - by_strategy["A"]["value_mean"],
                }
            )
    write_table_csv(output_dir / "ablation_best_strategy_by_beta_nfe.csv", best_rows)
    write_table_csv(output_dir / "ablation_strategy_b_vs_a.csv", comparison_rows)


def _schedule_family_table(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered = [row for row in rows if row["nfe"] in {10, 20, 50}]
    best = best_rows_by_key(filtered, ["dataset", "path_family", "solver", "clock_family", "nfe"])
    return sorted(best, key=lambda row: (row["dataset"], row["path_family"], row["nfe"], row["clock_family"]))


def _cross_path_table(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, object, object], Dict[str, object]] = {}
    for row in rows:
        if row["nfe"] not in {10, 20, 50}:
            continue
        key = (row["dataset"], row["solver"], row["nfe"])
        entry = grouped.setdefault(key, {"dataset": row["dataset"], "solver": row["solver"], "nfe": row["nfe"]})
        value = row["value_mean"]
        value_std = row["value_std"]
        if row["clock_family"] == "uniform":
            entry[f"{row['path_family']}:uniform"] = (value, value_std)
            continue
        if not is_ft_clock_family(row["clock_family"]):
            continue
        label = f"{row['path_family']}:ft"
        current = entry.get(label)
        if current is None or float(value) < float(current[0]):
            entry[label] = (value, value_std)
    output = []
    for _, entry in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        output.append(
            {
                "dataset": entry["dataset"],
                "solver": entry["solver"],
                "nfe": entry["nfe"],
                "linear_uniform_mean": entry.get("linear:uniform", (None, None))[0],
                "linear_uniform_std": entry.get("linear:uniform", (None, None))[1],
                "linear_ft_mean": entry.get("linear:ft", (None, None))[0],
                "linear_ft_std": entry.get("linear:ft", (None, None))[1],
                "trig_vp_uniform_mean": entry.get("trig_vp:uniform", (None, None))[0],
                "trig_vp_uniform_std": entry.get("trig_vp:uniform", (None, None))[1],
                "trig_vp_ft_mean": entry.get("trig_vp:ft", (None, None))[0],
                "trig_vp_ft_std": entry.get("trig_vp:ft", (None, None))[1],
            }
        )
    return output


def visualize_results(
    csv_path: Path,
    output_dir: Path,
    artifact_group: str = "",
    plot_heatmap_only: bool = False,
) -> None:
    rows = load_result_rows(csv_path)
    fid_rows = filter_rows(rows, metric="fid", status="completed")
    if artifact_group:
        fid_rows = [row for row in fid_rows if row["artifact_group"] == artifact_group]
    fid_rows = aggregate_seed_rows(fid_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_rows = [row for row in fid_rows if row["path_family"] == "linear"]
    ft_rows = [row for row in linear_rows if is_ft_clock_family(row["clock_family"])]
    _plot_heatmap(ft_rows, output_dir)
    if plot_heatmap_only:
        return

    _plot_main_curve(linear_rows, output_dir)

    main_table = baseline_vs_best_beta(linear_rows, already_aggregated=True)
    write_table_csv(output_dir / "baseline_vs_best_ft.csv", main_table)
    write_table_csv(output_dir / "schedule_family_table.csv", _schedule_family_table(linear_rows))
    write_table_csv(output_dir / "cross_path_table.csv", _cross_path_table(fid_rows))
    _write_ablation_tables(linear_rows, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact_group", type=str, default="")
    parser.add_argument("--plot_heatmap_only", action="store_true")
    args = parser.parse_args()
    visualize_results(
        args.csv,
        args.out,
        artifact_group=args.artifact_group,
        plot_heatmap_only=args.plot_heatmap_only,
    )
