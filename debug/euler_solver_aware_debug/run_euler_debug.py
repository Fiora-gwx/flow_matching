from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_variants import compare_euler_variants
from monitor_debug import (
    DEFAULT_NFE_LIST,
    MonitorDebugBundle,
    _json_dump,
    deep_merge,
    load_debug_config,
    prepare_runtime_context,
    run_monitor_debug,
)
from plot_debug import render_debug_plots


def _default_config_path() -> Path:
    return SCRIPT_DIR / "configs" / "default.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Euler solver-aware debug toolchain")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--output_root", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fid_samples", type=int, default=None)
    parser.add_argument("--nfe_list", nargs="*", type=int, default=None)
    parser.add_argument("--checkpoint_from_artifact_group", type=str, default=None)
    parser.add_argument("--checkpoint_from_source_exp_name", type=str, default=None)
    parser.add_argument("--checkpoint_from_epoch", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--skip_sampling_eval", action="store_true")
    parser.add_argument("--skip_stability_check", action="store_true")
    parser.add_argument("--skip_grid_size_sweep_eval", action="store_true")
    return parser


def _apply_cli_overrides(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    override: Dict[str, Any] = {}
    if args.dataset is not None:
        override["dataset"] = args.dataset
    if args.data_path is not None:
        override["data_path"] = args.data_path
    if args.device is not None:
        override["device"] = args.device
    if args.allow_cpu:
        override["allow_cpu"] = True
    if args.seed is not None:
        override["seed"] = int(args.seed)
    if args.fid_samples is not None:
        override["fid_samples"] = int(args.fid_samples)
    if args.nfe_list:
        override["nfe_list"] = [int(value) for value in args.nfe_list]

    checkpoint_override: Dict[str, Any] = {}
    if args.checkpoint_from_artifact_group is not None:
        checkpoint_override["artifact_group"] = args.checkpoint_from_artifact_group
    if args.checkpoint_from_source_exp_name is not None:
        checkpoint_override["source_exp_name"] = args.checkpoint_from_source_exp_name
    if args.checkpoint_from_epoch is not None:
        checkpoint_override["checkpoint_epoch"] = int(args.checkpoint_from_epoch)
    if args.checkpoint_path is not None:
        checkpoint_override["checkpoint_path"] = args.checkpoint_path
    if checkpoint_override:
        override["checkpoint"] = checkpoint_override

    execution_override: Dict[str, Any] = {}
    if args.skip_sampling_eval:
        execution_override["run_sampling_eval"] = False
    if args.skip_stability_check:
        execution_override["run_stability_check"] = False
    if args.skip_grid_size_sweep_eval:
        execution_override["run_grid_size_sweep_eval"] = False
    if execution_override:
        override["execution"] = execution_override

    merged = deep_merge(dict(config), override)
    if "nfe_list" not in merged or not merged["nfe_list"]:
        merged["nfe_list"] = list(DEFAULT_NFE_LIST)
    return merged


def _group_rows_by_variant(results_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in results_rows:
        grouped.setdefault(str(row["variant_name"]), []).append(row)
    return grouped


def _safe_mean(values: Sequence[float]) -> float:
    usable = [float(value) for value in values if not math.isnan(float(value))]
    if not usable:
        return float("nan")
    return sum(usable) / float(len(usable))


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    usable = [
        row
        for row in rows
        if not math.isnan(float(row.get("fid", float("nan"))))
    ]
    if not usable:
        return rows[0]
    return min(usable, key=lambda item: float(item["fid"]))


def _implementation_bug_signals(bundle: MonitorDebugBundle, results_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    issues: List[str] = []
    for row in bundle.numerical_rows:
        if not bool(row["nodes_strictly_increasing"]):
            issues.append("存在非严格递增 nodes。")
            break
    for row in bundle.numerical_rows:
        if not bool(row["step_sizes_positive"]):
            issues.append("存在非正 step size。")
            break
    for row in bundle.numerical_rows:
        if not bool(row["phi_strictly_monotone"]):
            issues.append("存在非严格单调 phi(s)。")
            break
    for row in results_rows:
        if int(row.get("actual_step_count", row["nfe"])) != int(row["nfe"]):
            issues.append("Euler 实际 step_count 与请求 NFE 不一致。")
            break
    return issues


def _summarize_variant_changes(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    current_variant: str,
) -> Dict[str, List[str]]:
    current_rows = grouped.get(current_variant, [])
    current_by_nfe = {int(row["nfe"]): row for row in current_rows}
    improved: List[str] = []
    unimproved: List[str] = []
    for variant_name, rows in grouped.items():
        if variant_name in {"uniform_baseline", current_variant}:
            continue
        gains = []
        for row in rows:
            nfe = int(row["nfe"])
            if nfe not in current_by_nfe:
                continue
            fid_value = float(row["fid"])
            current_fid = float(current_by_nfe[nfe]["fid"])
            if math.isnan(fid_value) or math.isnan(current_fid):
                continue
            gains.append(current_fid - fid_value)
        if gains and max(gains) > 0.0:
            improved.append(f"{variant_name}: best ΔFID={max(gains):.4f}")
        else:
            unimproved.append(variant_name)
    return {"improved": improved, "unimproved": unimproved}


def _build_summary_markdown(
    *,
    bundle: MonitorDebugBundle,
    results_rows: Sequence[Mapping[str, Any]],
) -> str:
    grouped = _group_rows_by_variant(results_rows)
    checkpoint = bundle.context.checkpoint
    current_variant = "solver_aware_current_impl"
    baseline_rows = grouped.get("uniform_baseline", [])
    current_rows = grouped.get(current_variant, [])
    lines: List[str] = ["# Euler Debug Summary", ""]
    lines.append("## Run Metadata")
    lines.append(f"- checkpoint_path: `{checkpoint.checkpoint_path}`")
    lines.append(f"- artifact_group: `{checkpoint.artifact_group}`")
    lines.append(f"- source_exp_name: `{checkpoint.source_exp_name}`")
    lines.append(f"- checkpoint_epoch: `{checkpoint.checkpoint_epoch}`")
    lines.append("")

    if baseline_rows and current_rows:
        lines.append("## Baseline vs Current Solver-Aware")
        baseline_by_nfe = {int(row["nfe"]): row for row in baseline_rows}
        current_by_nfe = {int(row["nfe"]): row for row in current_rows}
        for nfe in sorted(set(baseline_by_nfe).intersection(current_by_nfe)):
            baseline_fid = float(baseline_by_nfe[nfe]["fid"])
            current_fid = float(current_by_nfe[nfe]["fid"])
            delta = current_fid - baseline_fid
            lines.append(
                f"- NFE={nfe}: uniform={_format_metric(baseline_fid)}, "
                f"current_solver_aware={_format_metric(current_fid)}, ΔFID={delta:+.4f}"
            )
        lines.append("")

    if results_rows:
        lines.append("## Best Variants")
        fid_enabled = any(not math.isnan(float(row.get("fid", float("nan")))) for row in results_rows)
        if fid_enabled:
            for nfe in sorted({int(row["nfe"]) for row in results_rows}):
                candidates = [
                    row
                    for row in results_rows
                    if int(row["nfe"]) == nfe and not math.isnan(float(row["fid"]))
                ]
                if not candidates:
                    continue
                best = min(candidates, key=lambda item: float(item["fid"]))
                lines.append(
                    f"- NFE={nfe}: best={best['variant_name']} with FID={_format_metric(float(best['fid']))}"
                )
        else:
            lines.append("- 本次运行未开启采样评估，FID 为空。")
        lines.append("")

    variant_changes = _summarize_variant_changes(grouped, current_variant=current_variant)
    lines.append("## What Improved")
    if variant_changes["improved"]:
        for item in variant_changes["improved"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 没有看到任何变体稳定优于当前 solver-aware 实现。")
    lines.append("")

    lines.append("## What Did Not Improve")
    if variant_changes["unimproved"]:
        for item in variant_changes["unimproved"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 所有纳入比较的变体都至少在某个 NFE 上优于当前实现。")
    lines.append("")

    implementation_signals = _implementation_bug_signals(bundle, results_rows)
    current_diag_rows = bundle.node_diagnostics.get(current_variant, {})
    current_diag = None
    if current_diag_rows:
        usable_nfes = sorted(current_diag_rows)
        current_diag = current_diag_rows[usable_nfes[0]]
        if current_rows:
            current_by_nfe = {int(row["nfe"]): row for row in current_rows}
            if current_by_nfe:
                worst_nfe = max(
                    current_by_nfe,
                    key=lambda nfe: float(current_by_nfe[nfe]["fid"])
                    if not math.isnan(float(current_by_nfe[nfe]["fid"]))
                    else -1.0,
                )
                current_diag = current_diag_rows.get(worst_nfe, current_diag)

    max_cv = float("nan")
    if bundle.stability:
        max_cv = max(
            float(summary.cv_curve.max().item())
            for summary in bundle.stability.values()
        )

    grid_sweep_node_diff = _safe_mean(
        [float(row["node_linf_diff"]) for row in bundle.grid_sweep_rows]
    )
    current_max_step_over_uniform = (
        float(current_diag.max_step_over_uniform) if current_diag is not None else float("nan")
    )
    monitor_sharp_issue = (
        current_diag is not None
        and float(current_diag.q_spike_ratio_max_over_p95) > 2.0
        and bool(variant_changes["improved"])
    )
    node_focus_issue = (
        current_diag is not None
        and float(current_diag.max_step_over_uniform) > 1.5
    )
    grid_issue = not math.isnan(grid_sweep_node_diff) and grid_sweep_node_diff > 0.02
    variance_issue = not math.isnan(max_cv) and max_cv > 0.35

    lines.append("## Diagnosis")
    if current_diag is not None:
        lines.append(f"- 当前 solver-aware 代表性节点诊断: {current_diag.summary_sentence}")
        lines.append(
            f"- 代表性 max_step_over_uniform={current_max_step_over_uniform:.4f}, "
            f"q_spike_ratio_max_over_p95={float(current_diag.q_spike_ratio_max_over_p95):.4f}"
        )
    if not math.isnan(grid_sweep_node_diff):
        lines.append(f"- grid sweep 的平均 node L_inf 偏差={grid_sweep_node_diff:.4f}")
    if not math.isnan(max_cv):
        lines.append(f"- stability 检查里最大的 monitor CV={max_cv:.4f}")
    if implementation_signals:
        for item in implementation_signals:
            lines.append(f"- 实现异常信号: {item}")
    else:
        lines.append("- 数值自检没有发现明显实现级硬错误。")
    lines.append("")

    lines.append("## Final Judgment")
    if implementation_signals:
        lines.append(
            "- 最终判断更像实现问题，因为基础数值自检已经出现硬性异常，优先应该先清掉这些实现级错误再讨论 monitor 与 Euler 的理论错配。"
        )
    else:
        reasons = []
        if monitor_sharp_issue:
            reasons.append("monitor 尖峰/平滑问题")
        if node_focus_issue:
            reasons.append("node 过度聚焦与中段大步长问题")
        if grid_issue:
            reasons.append("monitor 离散化/continuous-to-discrete 问题")
        if variance_issue:
            reasons.append("JVP 方差问题")
        if not reasons:
            reasons.append("continuous solver-aware target 与 finite-step Euler 的目标错配")
        lines.append(
            "- 最终判断更像理论/目标错配，而不是显式实现 bug。"
        )
        lines.append(
            f"- 更具体地说，本次 run 最像由以下因素主导: {', '.join(reasons)}。"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_debug_config(Path(args.config))
    resolved_config = _apply_cli_overrides(config, args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _json_dump(output_root / "run_config_resolved.json", resolved_config)

    context = prepare_runtime_context(resolved_config, output_root=output_root)
    bundle = run_monitor_debug(
        context=context,
        config=resolved_config,
        output_root=output_root,
    )
    results_rows = compare_euler_variants(
        bundle=bundle,
        config=resolved_config,
        output_root=output_root,
    )
    render_debug_plots(
        bundle=bundle,
        output_root=output_root,
        results_rows=results_rows,
    )
    summary = _build_summary_markdown(bundle=bundle, results_rows=results_rows)
    (output_root / "euler_debug_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
