import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

RESULT_FIELDS = [
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
    "metric",
    "value",
    "status",
    "artifact_group",
]

NUMERIC_FIELDS = {
    "seed": int,
    "checkpoint_epoch": int,
    "nfe": int,
    "step_count": int,
    "value": float,
}

OPTIONAL_NUMERIC_FIELDS = {
    "clock_param_value": float,
}


def ensure_results_file(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()


def append_result_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_results_file(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


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
    return result


def load_result_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
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
        filtered.append(row)
    return filtered


def best_rows_by_key(
    rows: Sequence[Dict[str, object]], key_fields: Sequence[str]
) -> List[Dict[str, object]]:
    best: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        current = best.get(key)
        if current is None or float(row["value"]) < float(current["value"]):
            best[key] = row
    return list(best.values())


def baseline_vs_best_beta(
    rows: Sequence[Dict[str, object]],
    baseline_clock: str = "uniform",
    ft_clock_prefix: str = "ft_",
) -> List[Dict[str, object]]:
    grouped: Dict[tuple, Dict[str, object]] = defaultdict(dict)
    for row in rows:
        key = (
            row["dataset"],
            row["path_family"],
            row["solver"],
            row["nfe"],
        )
        if row["clock_family"] == baseline_clock:
            grouped[key]["baseline"] = row
        elif str(row["clock_family"]).startswith(ft_clock_prefix):
            best_ft = grouped[key].get("best_ft")
            if best_ft is None or float(row["value"]) < float(best_ft["value"]):
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
                "solver": baseline["solver"],
                "nfe": baseline["nfe"],
                "baseline_fid": baseline["value"],
                "best_clock_family": best_ft["clock_family"],
                "best_clock_param_name": best_ft["clock_param_name"],
                "best_clock_param_value": best_ft["clock_param_value"],
                "best_fid": best_ft["value"],
            }
        )
    return sorted(table, key=lambda row: (row["dataset"], row["path_family"], row["nfe"]))


def rows_to_matrix(
    rows: Sequence[Dict[str, object]], row_field: str, col_field: str
) -> Dict[object, Dict[object, float]]:
    matrix: Dict[object, Dict[object, float]] = defaultdict(dict)
    for row in rows:
        matrix[row[row_field]][row[col_field]] = float(row["value"])
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
