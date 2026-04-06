import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_RESULT_FIELDS = [
    "run_id",
    "exp_name",
    "dataset",
    "seed",
    "stage",
    "checkpoint_epoch",
    "path_family",
    "clock_family",
    "clock_param_name",
    "clock_param_value",
    "solver",
    "nfe",
    "step_count",
    "real_samples",
    "synthetic_samples",
    "metric",
    "value",
    "status",
    "artifact_group",
    "strategy_id",
    "model_output_type",
    "time_sampling_strategy",
    "mixed_lambda",
    "stratified_bins",
]
SOLVER_AWARE_RESULT_FIELDS = [
    "solver_aware_clock_mode",
    "solver_aware_target_solver",
    "solver_aware_monitor_solver",
    "solver_aware_monitor_family",
    "solver_aware_budget_mode",
    "solver_aware_target_nfe",
    "solver_aware_target_nfe_list",
    "solver_aware_target_nfe_weights",
    "solver_aware_target_step_count",
    "solver_aware_budget_step_counts",
    "solver_aware_k",
    "solver_aware_monitor_estimator",
    "solver_aware_eps",
    "solver_aware_use_nodes",
    "node_family",
    "monitor_source_checkpoint",
    "monitor_grid_size",
    "solver_aware_monitor_batch_size",
    "solver_aware_theorem_backed",
    "solver_aware_reference_solver",
    "solver_aware_reference_nfe",
    "solver_aware_reference_grid_size",
    "solver_aware_reference_cache_path",
    "solver_aware_reference_source",
    "solver_aware_reference_cache_hit",
    "solver_aware_defect_subdivide",
    "solver_aware_stork_effective_order",
    "solver_aware_q_curve_name",
]
SHARED_CLOCK_RESULT_FIELDS = [
    "shared_clock_mode",
    "shared_clock_family",
    "shared_clock_tag",
    "shared_clock_pilot_solver",
    "shared_clock_pilot_nfe_budget",
    "shared_clock_pilot_step_count",
    "shared_clock_physical_grid_size",
    "shared_clock_pilot_batch_size",
    "shared_clock_pilot_num_batches",
    "shared_clock_observation_microbatch",
    "shared_clock_eps",
    "shared_clock_jacobian_backend",
    "shared_clock_jacobian_num_probes",
    "shared_clock_optimizer_steps",
    "shared_clock_optimizer_lr",
    "shared_clock_cache_path",
]
TACK_RESULT_FIELDS = [
    "requested_eval_nfe",
    "realized_nfe",
    "tack_num_accepted_steps",
    "tack_num_heun_steps",
    "tack_num_ab2_steps",
    "tack_num_ab3_steps",
    "tack_num_halvings",
    "tack_num_doublings",
    "tack_mean_defect",
    "tack_mean_chi",
]
LEGACY_CURRENT_RESULT_FIELDS = BASE_RESULT_FIELDS + SOLVER_AWARE_RESULT_FIELDS
CURRENT_RESULT_FIELDS = LEGACY_CURRENT_RESULT_FIELDS + SHARED_CLOCK_RESULT_FIELDS
RESULT_FIELDS = CURRENT_RESULT_FIELDS + TACK_RESULT_FIELDS

NUMERIC_FIELDS = {
    "seed": int,
    "checkpoint_epoch": int,
    "nfe": int,
    "step_count": int,
    "real_samples": int,
    "synthetic_samples": int,
    "value": float,
}

OPTIONAL_NUMERIC_FIELDS = {
    "clock_param_value": float,
    "mixed_lambda": float,
    "stratified_bins": int,
    "solver_aware_k": int,
    "solver_aware_eps": float,
    "solver_aware_target_nfe": int,
    "solver_aware_target_step_count": int,
    "monitor_grid_size": int,
    "solver_aware_monitor_batch_size": int,
    "solver_aware_reference_nfe": int,
    "solver_aware_reference_grid_size": int,
    "solver_aware_defect_subdivide": int,
    "solver_aware_stork_effective_order": float,
    "shared_clock_pilot_nfe_budget": int,
    "shared_clock_pilot_step_count": int,
    "shared_clock_physical_grid_size": int,
    "shared_clock_pilot_batch_size": int,
    "shared_clock_pilot_num_batches": int,
    "shared_clock_observation_microbatch": int,
    "shared_clock_eps": float,
    "shared_clock_jacobian_num_probes": int,
    "shared_clock_optimizer_steps": int,
    "shared_clock_optimizer_lr": float,
    "requested_eval_nfe": float,
    "realized_nfe": float,
    "tack_num_accepted_steps": float,
    "tack_num_heun_steps": float,
    "tack_num_ab2_steps": float,
    "tack_num_ab3_steps": float,
    "tack_num_halvings": float,
    "tack_num_doublings": float,
    "tack_mean_defect": float,
    "tack_mean_chi": float,
}
METRIC_OUTPUTS = {
    "fid": ("fid",),
    "precision_recall": ("precision", "recall"),
    "inception_score": ("is_mean", "is_std"),
}
FT_CLOCK_FAMILIES = frozenset({"ft_beta", "ft_linear_beta"})


def is_ft_clock_family(clock_family: object) -> bool:
    return str(clock_family) in FT_CLOCK_FAMILIES


def _write_results_header(csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def _read_header(csv_path: Path) -> List[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def validate_results_schema(csv_path: Path) -> None:
    header = _read_header(csv_path)
    if not header:
        return
    legacy_solver_aware_fields = [
        "solver_aware_clock_mode",
        "solver_aware_target_solver",
        "solver_aware_monitor_solver",
        "solver_aware_k",
        "solver_aware_monitor_estimator",
        "solver_aware_eps",
        "solver_aware_use_nodes",
        "node_family",
        "monitor_source_checkpoint",
        "monitor_grid_size",
        "solver_aware_monitor_batch_size",
        "solver_aware_theorem_backed",
    ]
    accepted_headers = {
        tuple(BASE_RESULT_FIELDS),
        tuple(BASE_RESULT_FIELDS + legacy_solver_aware_fields),
        tuple(LEGACY_CURRENT_RESULT_FIELDS),
        tuple(CURRENT_RESULT_FIELDS),
        tuple(RESULT_FIELDS),
    }
    if tuple(header) not in accepted_headers:
        raise ValueError(
            f"Result schema mismatch in {csv_path}. "
            f"Expected header {BASE_RESULT_FIELDS} or {RESULT_FIELDS}, got {header}. "
            "Migrate or remove the legacy results.csv before writing new FT-clock results."
        )


def ensure_results_file(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        _write_results_header(csv_path)
        return
    if not _read_header(csv_path):
        _write_results_header(csv_path)
        return
    validate_results_schema(csv_path)


def append_result_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    ensure_results_file(csv_path)
    fieldnames = _read_header(csv_path) or RESULT_FIELDS
    missing_extra_fields = [field for field in RESULT_FIELDS if field not in fieldnames]
    if missing_extra_fields:
        for row in rows:
            if any(row.get(field) not in {"", None, False} for field in missing_extra_fields):
                raise ValueError(
                    f"Cannot append rows with fields {missing_extra_fields} to legacy-schema CSV {csv_path}. "
                    "Use a fresh artifact_group/results.csv so the extended schema can be written."
                )
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def metric_output_names(metric_name: str) -> Tuple[str, ...]:
    return METRIC_OUTPUTS.get(metric_name, (metric_name,))


def _coerce_row(row: Dict[str, str]) -> Dict[str, object]:
    result: Dict[str, object] = dict(row)
    for field, caster in NUMERIC_FIELDS.items():
        if row.get(field, "") != "":
            result[field] = caster(row[field])
    for field, caster in OPTIONAL_NUMERIC_FIELDS.items():
        if row.get(field, "") not in {"", None}:
            result[field] = caster(row[field])
        else:
            result[field] = None
    for field in SOLVER_AWARE_RESULT_FIELDS:
        result.setdefault(field, "")
    for field in SHARED_CLOCK_RESULT_FIELDS:
        result.setdefault(field, "")
    for field in TACK_RESULT_FIELDS:
        result.setdefault(field, None)
    return result


def load_result_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    validate_results_schema(csv_path)
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_coerce_row(row) for row in reader]


def filter_rows(
    rows: Sequence[Dict[str, object]],
    metric: Optional[str] = None,
    status: Optional[str] = "completed",
    artifact_group: Optional[str] = None,
    dataset: Optional[str] = None,
    path_family: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    filtered = []
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        if status is not None and row.get("status") != status:
            continue
        if artifact_group is not None and row.get("artifact_group") != artifact_group:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if path_family is not None and row.get("path_family") != path_family:
            continue
        if strategy_id is not None and row.get("strategy_id") != strategy_id:
            continue
        filtered.append(row)
    return filtered


def best_rows_by_key(
    rows: Sequence[Dict[str, object]], key_fields: Sequence[str]
) -> List[Dict[str, object]]:
    best: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        current = best.get(key)
        if current is None or _row_value(row) < _row_value(current):
            best[key] = row
    return list(best.values())


def baseline_vs_best_beta(
    rows: Sequence[Dict[str, object]],
    baseline_clock: str = "uniform",
    already_aggregated: bool = False,
) -> List[Dict[str, object]]:
    aggregated_rows = list(rows) if already_aggregated else aggregate_seed_rows(rows)
    grouped: Dict[tuple, Dict[str, object]] = defaultdict(dict)
    for row in aggregated_rows:
        key = (
            row["dataset"],
            row["path_family"],
            row.get("strategy_id", ""),
            row["solver"],
            row["checkpoint_epoch"],
            row["nfe"],
        )
        if row["clock_family"] == baseline_clock:
            grouped[key]["baseline"] = row
        elif is_ft_clock_family(row["clock_family"]):
            best_ft = grouped[key].get("best_ft")
            if best_ft is None or _row_value(row) < _row_value(best_ft):
                grouped[key]["best_ft"] = row

    table = []
    for key, values in grouped.items():
        if "baseline" not in values or "best_ft" not in values:
            continue
        baseline = values["baseline"]
        best_ft = values["best_ft"]
        table.append(
            {
                "dataset": baseline["dataset"],
                "path_family": baseline["path_family"],
                "strategy_id": baseline.get("strategy_id", ""),
                "solver": baseline["solver"],
                "checkpoint_epoch": baseline["checkpoint_epoch"],
                "nfe": baseline["nfe"],
                "baseline_fid_mean": baseline["value_mean"],
                "baseline_fid_std": baseline["value_std"],
                "baseline_num_seeds": baseline["num_seeds"],
                "best_clock_family": best_ft["clock_family"],
                "best_clock_param_name": best_ft["clock_param_name"],
                "best_clock_param_value": best_ft["clock_param_value"],
                "best_fid_mean": best_ft["value_mean"],
                "best_fid_std": best_ft["value_std"],
                "best_num_seeds": best_ft["num_seeds"],
            }
        )
    return sorted(
        table,
        key=lambda row: (
            row["dataset"],
            row["path_family"],
            row.get("strategy_id", ""),
            row["checkpoint_epoch"],
            row["nfe"],
        ),
    )


def rows_to_matrix(
    rows: Sequence[Dict[str, object]], row_field: str, col_field: str
) -> Dict[object, Dict[object, float]]:
    matrix: Dict[object, Dict[object, float]] = defaultdict(dict)
    for row in rows:
        matrix[row[row_field]][row[col_field]] = _row_value(row)
    return matrix


def write_table_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_clock_parameter(
    clock_family: str,
    clock_beta: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    if is_ft_clock_family(clock_family):
        return "beta", clock_beta
    if clock_family == "poly_a0.5":
        return "a", 0.5
    if clock_family == "poly_a2.0":
        return "a", 2.0
    if clock_family == "sigmoid_k8":
        return "k", 8.0
    if clock_family == "exp_l3":
        return "lambda", 3.0
    return "none", None


def _row_value(row: Dict[str, object]) -> float:
    if "value_mean" in row:
        return float(row["value_mean"])
    return float(row["value"])


def aggregate_seed_rows(
    rows: Sequence[Dict[str, object]],
    group_fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    if not rows:
        return []
    if group_fields is None:
        average_fields = {
            "real_samples",
            "synthetic_samples",
            "step_count",
            "requested_eval_nfe",
            "realized_nfe",
            "tack_num_accepted_steps",
            "tack_num_heun_steps",
            "tack_num_ab2_steps",
            "tack_num_ab3_steps",
            "tack_num_halvings",
            "tack_num_doublings",
            "tack_mean_defect",
            "tack_mean_chi",
        }
        group_fields = [
            field
            for field in RESULT_FIELDS
            if field not in {"run_id", "seed", "value"} and field not in average_fields
        ]
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)

    aggregated_rows = []
    for key, group_rows in grouped.items():
        result = {field: group_rows[0].get(field) for field in group_fields}
        values = [float(row["value"]) for row in group_rows]
        result["run_id"] = f"aggregate:{len(aggregated_rows)}"
        result["seed"] = "aggregate"
        result["value"] = statistics.mean(values)
        result["value_mean"] = statistics.mean(values)
        result["value_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result["num_seeds"] = len(values)
        for sample_field in (
            "real_samples",
            "synthetic_samples",
            "step_count",
            "requested_eval_nfe",
            "realized_nfe",
            "tack_num_accepted_steps",
            "tack_num_heun_steps",
            "tack_num_ab2_steps",
            "tack_num_ab3_steps",
            "tack_num_halvings",
            "tack_num_doublings",
            "tack_mean_defect",
            "tack_mean_chi",
        ):
            observed = [
                float(row[sample_field])
                for row in group_rows
                if row.get(sample_field) not in {"", None}
            ]
            if observed:
                mean_value = statistics.mean(observed)
                if sample_field in {
                    "real_samples",
                    "synthetic_samples",
                    "step_count",
                }:
                    result[sample_field] = round(mean_value)
                else:
                    result[sample_field] = mean_value
        aggregated_rows.append(result)
    return aggregated_rows


def resolve_best_beta_reference(
    reference: Dict[str, object],
    workspace_root: Optional[Path] = None,
) -> float:
    workspace_root = workspace_root or Path.cwd()
    if reference.get("results_csv"):
        csv_path = Path(str(reference["results_csv"]))
        if not csv_path.is_absolute():
            csv_path = workspace_root / csv_path
    else:
        artifact_group = str(reference["artifact_group"])
        csv_path = workspace_root / "experiments" / "results" / artifact_group / "results.csv"
    rows = load_result_rows(csv_path)
    rows = filter_rows(
        rows,
        metric=str(reference.get("metric", "fid")),
        status=str(reference.get("status", "completed")),
        artifact_group=reference.get("artifact_group"),
        dataset=reference.get("dataset"),
        path_family=reference.get("path_family"),
        strategy_id=reference.get("strategy_id"),
    )
    solver = reference.get("solver")
    if solver is not None:
        rows = [row for row in rows if row.get("solver") == solver]
    checkpoint_epoch = reference.get("checkpoint_epoch")
    if checkpoint_epoch is not None:
        rows = [row for row in rows if row.get("checkpoint_epoch") == checkpoint_epoch]
    model_output_type = reference.get("model_output_type")
    if model_output_type is not None:
        rows = [row for row in rows if row.get("model_output_type") == model_output_type]
    time_sampling_strategy = reference.get("time_sampling_strategy")
    if time_sampling_strategy is not None:
        rows = [
            row
            for row in rows
            if row.get("time_sampling_strategy") == time_sampling_strategy
        ]
    clock_family = reference.get("clock_family")
    if clock_family is not None:
        if clock_family == "ft_beta" and reference.get("path_family") == "linear":
            rows = [row for row in rows if str(row.get("clock_family")) in {"ft_beta", "ft_linear_beta"}]
        else:
            rows = [row for row in rows if row.get("clock_family") == clock_family]
    else:
        rows = [row for row in rows if is_ft_clock_family(row.get("clock_family", ""))]
    selection_nfes = reference.get("selection_nfes")
    if selection_nfes is not None:
        selection_nfes = {int(nfe) for nfe in selection_nfes}
        rows = [row for row in rows if int(row.get("nfe", -1)) in selection_nfes]
    aggregated = aggregate_seed_rows(rows)
    if not aggregated:
        raise ValueError(f"Unable to resolve best beta from {csv_path}; no matching rows found.")

    grouped: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in aggregated:
        beta = row.get("clock_param_value")
        if beta is None:
            continue
        grouped[float(beta)].append(row)
    if not grouped:
        raise ValueError(f"Unable to resolve best beta from {csv_path}; no beta rows found.")

    best_beta = None
    best_score = None
    for beta, beta_rows in grouped.items():
        score = statistics.mean(_row_value(row) for row in beta_rows)
        if best_score is None or score < best_score:
            best_beta = beta
            best_score = score
    assert best_beta is not None
    return best_beta
