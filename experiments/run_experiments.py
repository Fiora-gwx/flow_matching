#!/usr/bin/env python3
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_utils import (
    checkpoint_matches_spec,
    find_checkpoint,
    load_checkpoint_args,
    resolve_reused_checkpoint,
)
from experiments.result_utils import (
    append_result_rows,
    ensure_results_file,
    infer_clock_parameter,
    load_result_rows,
    metric_output_names,
    resolve_best_beta_reference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
LEGACY_CONFIG_KEYS = {"alpha", "use_ft_eqm", "use_nt_ft_fm", "importance_weighting"}
LEGACY_CLOCK_FAMILIES = {"ft_linear_beta", "ft_vp_beta"}
CURRICULUM_SIGNATURE = "warmup0.3_linear_to1"
SOLVER_AWARE_DEFAULTS = {
    "solver_aware_clock_mode": "off",
    "solver_aware_target_solver": "",
    "solver_aware_monitor_solver": "",
    "solver_aware_monitor_family": "",
    "solver_aware_budget_mode": "",
    "solver_aware_target_nfe": None,
    "solver_aware_target_nfe_list": "",
    "solver_aware_target_nfe_weights": "",
    "solver_aware_target_step_count": None,
    "solver_aware_budget_step_counts": "",
    "solver_aware_k": 0,
    "solver_aware_monitor_estimator": "",
    "solver_aware_eps": None,
    "solver_aware_use_nodes": False,
    "node_family": "uniform",
    "monitor_source_checkpoint": "",
    "monitor_grid_size": None,
    "solver_aware_monitor_batch_size": None,
    "solver_aware_theorem_backed": "",
    "solver_aware_reference_solver": "",
    "solver_aware_reference_nfe": None,
    "solver_aware_reference_grid_size": None,
    "solver_aware_reference_cache_path": "",
    "solver_aware_reference_source": "",
    "solver_aware_reference_cache_hit": "",
    "solver_aware_defect_subdivide": None,
    "solver_aware_stork_effective_order": None,
    "solver_aware_q_curve_name": "",
}


def _solver_aware_step_count(target_solver: str, nfe_budget: int) -> int:
    if target_solver in {"euler", "stork4"}:
        return int(nfe_budget)
    if target_solver == "heun2":
        return int(nfe_budget // 2 + (1 if nfe_budget % 2 else 0))
    return int(nfe_budget)


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_dicts(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    merged.update(override)
    return merged


def run_command(cmd: str, log_file: Path, retries: int = 0) -> bool:
    for attempt in range(retries + 1):
        try:
            logger.info("Executing: %s", cmd)
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(f"\n--- Execution Attempt {attempt + 1} at {time.ctime()} ---\n")
                handle.write(f"Command: {cmd}\n\n")
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    handle.write(line)
                process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            return True
        except subprocess.CalledProcessError as error:
            logger.error("Command failed with exit code %s", error.returncode)
            if attempt < retries:
                time.sleep(5)
            else:
                return False
    return False


def extract_eval_stats(log_path: Path) -> Dict[str, float]:
    if not log_path.exists():
        return {}
    latest = {}
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            eval_stats = {
                key.removeprefix("eval_"): value
                for key, value in data.items()
                if key.startswith("eval_")
            }
            if eval_stats:
                latest = eval_stats
    return latest


def assert_no_legacy_keys(spec: Dict) -> None:
    legacy_keys = sorted(key for key in LEGACY_CONFIG_KEYS if key in spec)
    if legacy_keys:
        raise ValueError(
            f"Legacy config keys are no longer supported: {legacy_keys}. "
            "Use path_family/clock_family/clock_beta instead."
        )
    legacy_clock_family = spec.get("clock_family")
    if legacy_clock_family in LEGACY_CLOCK_FAMILIES:
        raise ValueError(
            f"Legacy clock_family={legacy_clock_family} is no longer supported. "
            "Use clock_family=ft_beta and let path_family determine the bridge."
        )


def resolve_dynamic_spec_fields(spec: Dict, workspace_root: Optional[Path] = None) -> Dict:
    resolved = dict(spec)
    best_beta_from = resolved.get("best_beta_from")
    if best_beta_from is not None:
        resolved["clock_beta"] = resolve_best_beta_reference(
            reference=best_beta_from,
            workspace_root=workspace_root,
        )
        logger.info(
            "Resolved best beta for %s to %.4f from %s",
            resolved.get("name", "<unnamed>"),
            resolved["clock_beta"],
            best_beta_from,
        )
    return resolved


def _resolve_strategy_fields(spec: Dict) -> Dict:
    model_output_type = str(spec.get("model_output_type", "velocity"))
    time_sampling_strategy = str(spec.get("time_sampling_strategy", "uniform"))
    mixed_lambda = float(spec.get("mixed_lambda", 0.5))
    stratified_bins = int(spec.get("stratified_bins", 16))
    curriculum_signature = str(
        spec.get(
            "curriculum_signature",
            CURRICULUM_SIGNATURE if time_sampling_strategy == "curriculum" else "",
        )
    )
    strategy_mapping = {
        ("velocity", "uniform"): "A",
        ("base_velocity", "ds_dr_sq"): "B",
        ("velocity", "mixed_lambda"): "C",
        ("velocity", "stratified"): "D",
        ("velocity", "stratified_mixed"): "E",
        ("velocity", "curriculum"): "F",
    }
    strategy_id = strategy_mapping.get((model_output_type, time_sampling_strategy))
    if strategy_id is None:
        raise ValueError(
            "Unsupported ablation strategy semantics: "
            f"model_output_type={model_output_type}, "
            f"time_sampling_strategy={time_sampling_strategy}"
        )
    resolved = dict(spec)
    resolved["model_output_type"] = model_output_type
    resolved["time_sampling_strategy"] = time_sampling_strategy
    resolved["mixed_lambda"] = mixed_lambda
    resolved["stratified_bins"] = stratified_bins
    resolved["curriculum_signature"] = curriculum_signature
    provided_strategy_id = spec.get("strategy_id")
    resolved["strategy_id"] = (
        strategy_id if provided_strategy_id in {"", None} else str(provided_strategy_id)
    )
    return resolved


def _row_float_or_default(row: Dict[str, object], field: str, default: float) -> float:
    value = row.get(field, default)
    return default if value in {"", None} else float(value)


def _row_int_or_default(row: Dict[str, object], field: str, default: int) -> int:
    value = row.get(field, default)
    return default if value in {"", None} else int(value)


def _row_strategy_fields(row: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    normalized["model_output_type"] = row.get("model_output_type") or "velocity"
    normalized["time_sampling_strategy"] = row.get("time_sampling_strategy") or (
        "ds_dr_sq" if normalized["model_output_type"] == "base_velocity" else "uniform"
    )
    normalized["mixed_lambda"] = _row_float_or_default(row, "mixed_lambda", 0.5)
    normalized["stratified_bins"] = _row_int_or_default(row, "stratified_bins", 16)
    return _resolve_strategy_fields(normalized)


def _checkpoint_semantics_for_results(
    checkpoint_path: Optional[Path],
    spec: Dict[str, object],
) -> Dict[str, object]:
    if checkpoint_path is None:
        return dict(spec)

    checkpoint_args = load_checkpoint_args(checkpoint_path)
    if not checkpoint_args:
        return dict(spec)

    effective = dict(spec)
    for field in (
        "path_family",
        "clock_family",
        "clock_beta",
        "model_output_type",
        "time_sampling_strategy",
        "mixed_lambda",
        "stratified_bins",
        "strategy_id",
        "curriculum_signature",
        "clock_semantics_tag",
    ):
        if checkpoint_args.get(field) is not None:
            effective[field] = checkpoint_args[field]
    return _resolve_strategy_fields(effective)


def _resolve_solver_aware_result_fields(
    spec: Dict[str, object],
    checkpoint_path: Optional[Path],
    current_nfe: Optional[int] = None,
) -> Dict[str, object]:
    mode = str(spec.get("solver_aware_clock_mode", "off"))
    use_nodes = bool(spec.get("solver_aware_use_nodes", False))
    target_solver = str(spec.get("solver_aware_target_solver", spec.get("sampling_solver", "")))
    if mode == "off" or not use_nodes:
        return dict(SOLVER_AWARE_DEFAULTS)

    monitor_solver = target_solver
    monitor_family = str(
        spec.get("solver_aware_monitor_family", "legacy_continuous")
    )
    budget_mode = str(spec.get("solver_aware_budget_mode", "single_budget"))
    theorem_backed = ""
    if target_solver == "stork4":
        theorem_backed = "false" if monitor_family == "legacy_continuous" else "false"
    elif target_solver in {"euler", "heun2"}:
        theorem_backed = "true"
    if monitor_family == "legacy_continuous" and target_solver == "stork4":
        monitor_solver = "heun2"

    resolved_target_nfe = None
    target_nfe_list = ""
    target_nfe_weights = ""
    target_step_count = None
    budget_step_counts = ""
    reference_solver = ""
    reference_nfe = None
    reference_grid_size = None
    reference_cache_path = ""
    reference_source = ""
    defect_subdivide = None
    stork_effective_order = None
    q_curve_name = ""
    if monitor_family == "defect_based":
        if budget_mode == "single_budget":
            resolved_target_nfe = int(
                spec.get("solver_aware_target_nfe", 0) or current_nfe or 0
            )
            target_step_count = _solver_aware_step_count(
                target_solver=target_solver,
                nfe_budget=resolved_target_nfe,
            )
            target_nfe_list = str(resolved_target_nfe)
            budget_step_counts = f"{resolved_target_nfe}:{target_step_count}"
            q_curve_name = "Q_path_defect"
        else:
            raw_list = [
                int(value)
                for value in spec.get("solver_aware_target_nfe_list", [])
                if int(value) > 0
            ]
            target_nfe_list = "|".join(str(value) for value in raw_list)
            raw_weights = [float(value) for value in spec.get("solver_aware_target_nfe_weights", [])]
            if raw_weights:
                target_nfe_weights = "|".join(str(value) for value in raw_weights)
            budget_step_counts = "|".join(
                f"{budget}:{_solver_aware_step_count(target_solver=target_solver, nfe_budget=budget)}"
                for budget in raw_list
            )
            q_curve_name = "M_tilde_path_defect"
        defect_subdivide = int(spec.get("solver_aware_defect_subdivide", 2))
        stork_effective_order = float(spec.get("solver_aware_stork_effective_order", 4.0))

    return {
        "solver_aware_clock_mode": mode,
        "solver_aware_target_solver": target_solver,
        "solver_aware_monitor_solver": monitor_solver,
        "solver_aware_monitor_family": monitor_family,
        "solver_aware_budget_mode": budget_mode if monitor_family == "defect_based" else "single_budget",
        "solver_aware_target_nfe": resolved_target_nfe,
        "solver_aware_target_nfe_list": target_nfe_list,
        "solver_aware_target_nfe_weights": target_nfe_weights,
        "solver_aware_target_step_count": target_step_count,
        "solver_aware_budget_step_counts": budget_step_counts,
        "solver_aware_k": int(spec.get("solver_aware_k", 0)),
        "solver_aware_monitor_estimator": (
            "defect"
            if monitor_family == "defect_based"
            else str(spec.get("solver_aware_monitor_estimator", "auto"))
        ),
        "solver_aware_eps": spec.get("solver_aware_eps"),
        "solver_aware_use_nodes": "true" if use_nodes else "false",
        "node_family": "solver_aware",
        "monitor_source_checkpoint": str(
            checkpoint_path
            or spec.get("solver_aware_checkpoint_path")
            or ""
        ),
        "monitor_grid_size": spec.get("solver_aware_monitor_grid_size"),
        "solver_aware_monitor_batch_size": spec.get("solver_aware_monitor_batch_size"),
        "solver_aware_theorem_backed": theorem_backed,
        "solver_aware_reference_solver": reference_solver,
        "solver_aware_reference_nfe": reference_nfe,
        "solver_aware_reference_grid_size": reference_grid_size,
        "solver_aware_reference_cache_path": reference_cache_path,
        "solver_aware_reference_source": reference_source,
        "solver_aware_reference_cache_hit": "",
        "solver_aware_defect_subdivide": defect_subdivide,
        "solver_aware_stork_effective_order": stork_effective_order,
        "solver_aware_q_curve_name": q_curve_name,
    }


def _solver_aware_fields_match(row: Dict[str, object], spec: Dict[str, object]) -> bool:
    expected = _resolve_solver_aware_result_fields(
        spec=spec,
        checkpoint_path=None,
        current_nfe=int(row.get("nfe", 0) or 0),
    )
    for field, expected_value in expected.items():
        observed = row.get(field, SOLVER_AWARE_DEFAULTS.get(field, ""))
        if observed in {"", None}:
            observed = SOLVER_AWARE_DEFAULTS.get(field, "")
        if expected_value in {"", None}:
            if observed not in {"", None}:
                return False
            continue
        if field in {
            "solver_aware_k",
            "solver_aware_target_nfe",
            "solver_aware_target_step_count",
            "monitor_grid_size",
            "solver_aware_monitor_batch_size",
            "solver_aware_reference_nfe",
            "solver_aware_reference_grid_size",
            "solver_aware_defect_subdivide",
        }:
            if int(observed) != int(expected_value):
                return False
            continue
        if field in {"solver_aware_eps", "solver_aware_stork_effective_order"}:
            if float(observed) != float(expected_value):
                return False
            continue
        if str(observed) != str(expected_value):
            return False
    return True


class ExperimentManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.exp_group_name = self.config.get("experiment_name", config_path.stem)
        self.base_dir = Path(f"./experiments/results/{self.exp_group_name}")
        self.logs_dir = Path(f"./experiments/logs/{self.exp_group_name}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.base_dir / "experiment_status.json"
        self.state = self._load_state()
        self.results_csv = self.base_dir / "results.csv"
        ensure_results_file(self.results_csv)

    def _load_state(self) -> Dict[str, str]:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _save_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def _launcher(self, num_gpus: int) -> str:
        if num_gpus <= 1:
            return "python3"
        return f"torchrun --standalone --nproc_per_node={num_gpus}"

    def _base_flags(self, spec: Dict, output_dir: Path) -> List[str]:
        flags = [
            f"--dataset {spec['dataset']}",
            f"--data_path {spec['data_path']}",
            f"--batch_size {spec['batch_size']}",
            f"--output_dir {output_dir}",
            f"--model_output_type {spec.get('model_output_type', 'velocity')}",
            f"--time_sampling_strategy {spec.get('time_sampling_strategy', 'uniform')}",
            f"--mixed_lambda {spec.get('mixed_lambda', 0.5)}",
            f"--stratified_bins {spec.get('stratified_bins', 16)}",
        ]
        if spec.get("use_ema", False):
            flags.append("--use_ema")
        if spec.get("discrete_flow_matching", False):
            flags.append("--discrete_flow_matching")
        else:
            flags.extend(
                [
                    f"--path_family {spec.get('path_family', 'linear')}",
                    f"--clock_family {spec.get('clock_family', 'uniform')}",
                    f"--sampling_solver {spec.get('sampling_solver', 'heun2')}",
                ]
            )
            if spec.get("clock_beta") is not None:
                flags.append(f"--clock_beta {spec['clock_beta']}")
        if spec.get("metrics"):
            flags.append("--metrics " + " ".join(spec["metrics"]))
        if spec.get("precision_recall_neighbors") is not None:
            flags.append(
                f"--precision_recall_neighbors {spec['precision_recall_neighbors']}"
            )
        if spec.get("precision_recall_max_samples") is not None:
            flags.append(
                f"--precision_recall_max_samples {spec['precision_recall_max_samples']}"
            )
        if spec.get("inception_score_splits") is not None:
            flags.append(
                f"--inception_score_splits {spec['inception_score_splits']}"
            )
        if spec.get("cfg_scale") is not None:
            flags.append(f"--cfg_scale {spec['cfg_scale']}")
        if spec.get("class_drop_prob") is not None:
            flags.append(f"--class_drop_prob {spec['class_drop_prob']}")
        if spec.get("fid_samples") is not None:
            flags.append(f"--fid_samples {spec['fid_samples']}")
        if spec.get("solver_aware_clock_mode") not in {None, "", "off"}:
            flags.append(f"--solver_aware_clock_mode {spec['solver_aware_clock_mode']}")
            flags.append(
                f"--solver_aware_target_solver {spec.get('solver_aware_target_solver', spec.get('sampling_solver', 'euler'))}"
            )
            flags.append(
                f"--solver_aware_monitor_family {spec.get('solver_aware_monitor_family', 'legacy_continuous')}"
            )
            flags.append(
                f"--solver_aware_budget_mode {spec.get('solver_aware_budget_mode', 'single_budget')}"
            )
            if spec.get("solver_aware_target_nfe") is not None:
                flags.append(f"--solver_aware_target_nfe {spec.get('solver_aware_target_nfe', 0)}")
            if spec.get("solver_aware_target_nfe_list"):
                flags.append(
                    "--solver_aware_target_nfe_list "
                    + " ".join(str(value) for value in spec["solver_aware_target_nfe_list"])
                )
            if spec.get("solver_aware_target_nfe_weights"):
                flags.append(
                    "--solver_aware_target_nfe_weights "
                    + " ".join(str(value) for value in spec["solver_aware_target_nfe_weights"])
                )
            flags.append(f"--solver_aware_k {spec.get('solver_aware_k', 0)}")
            flags.append(
                f"--solver_aware_monitor_estimator {spec.get('solver_aware_monitor_estimator', 'auto')}"
            )
            if spec.get("solver_aware_monitor_grid_size") is not None:
                flags.append(
                    f"--solver_aware_monitor_grid_size {spec['solver_aware_monitor_grid_size']}"
                )
            if spec.get("solver_aware_monitor_batch_size") is not None:
                flags.append(
                    f"--solver_aware_monitor_batch_size {spec['solver_aware_monitor_batch_size']}"
                )
            if spec.get("solver_aware_eps") is not None:
                flags.append(f"--solver_aware_eps {spec['solver_aware_eps']}")
            if spec.get("solver_aware_cache_path") not in {None, ""}:
                flags.append(f"--solver_aware_cache_path {spec['solver_aware_cache_path']}")
            if spec.get("solver_aware_stork_effective_order") is not None:
                flags.append(
                    f"--solver_aware_stork_effective_order {spec.get('solver_aware_stork_effective_order', 4.0)}"
                )
            if spec.get("solver_aware_defect_subdivide") is not None:
                flags.append(
                    f"--solver_aware_defect_subdivide {spec.get('solver_aware_defect_subdivide', 2)}"
                )
            if spec.get("solver_aware_use_nodes", False):
                flags.append("--solver_aware_use_nodes")
            if spec.get("solver_aware_checkpoint_path") not in {None, ""}:
                flags.append(
                    f"--solver_aware_checkpoint_path {spec['solver_aware_checkpoint_path']}"
                )
            if spec.get("solver_aware_checkpoint_from_experiment") not in {None, ""}:
                flags.append(
                    "--solver_aware_checkpoint_from_experiment "
                    + str(spec["solver_aware_checkpoint_from_experiment"])
                )
            if spec.get("solver_aware_checkpoint_epoch") is not None:
                flags.append(
                    f"--solver_aware_checkpoint_epoch {spec['solver_aware_checkpoint_epoch']}"
                )
        return flags

    def build_train_cmd(
        self,
        spec: Dict,
        output_dir: Path,
        resume_checkpoint: Optional[Path] = None,
    ) -> str:
        flags = self._base_flags(spec, output_dir)
        flags.extend(
            [
                f"--epochs {spec['epochs']}",
                f"--lr {spec.get('lr', 0.0001)}",
                f"--seed {spec.get('seed', 0)}",
            ]
        )
        if spec.get("accum_iter") is not None:
            flags.append(f"--accum_iter {spec['accum_iter']}")
        if resume_checkpoint is not None:
            flags.append(f"--resume {resume_checkpoint}")
        flags.append(f"--eval_frequency {spec.get('eval_frequency', -1)}")
        if spec.get("decay_lr", False):
            flags.append("--decay_lr")
        return f"{self._launcher(spec.get('num_gpus', 1))} examples/image/train.py " + " ".join(flags)

    def build_eval_cmd(
        self,
        spec: Dict,
        output_dir: Path,
        checkpoint: Path,
        nfe: int,
        metrics_override: Optional[List[str]] = None,
    ) -> str:
        eval_spec = dict(spec)
        if metrics_override is not None:
            eval_spec["metrics"] = metrics_override
        flags = self._base_flags(eval_spec, output_dir)
        flags.extend(
            [
                "--eval_only",
                f"--resume {checkpoint}",
                f"--eval_nfe {nfe}",
                f"--seed {spec.get('seed', 0)}",
            ]
        )
        if "fid" in eval_spec.get("metrics", ["fid"]):
            flags.append("--compute_fid")
        return f"{self._launcher(spec.get('num_gpus', 1))} examples/image/train.py " + " ".join(flags)

    def _result_rows(
        self,
        spec: Dict,
        epoch: int,
        nfe: int,
        stats: Dict[str, float],
        checkpoint_path: Optional[Path] = None,
    ) -> List[Dict[str, object]]:
        rows = []
        effective_spec = _checkpoint_semantics_for_results(
            checkpoint_path=checkpoint_path,
            spec=spec,
        )
        solver_aware_fields = _resolve_solver_aware_result_fields(
            spec=spec,
            checkpoint_path=checkpoint_path,
            current_nfe=int(stats.get("nfe", nfe)),
        )
        clock_param_name, clock_param_value = infer_clock_parameter(
            effective_spec.get("clock_family", "uniform"),
            effective_spec.get("clock_beta"),
        )
        for metric_name, value in stats.items():
            if metric_name in {"nfe", "step_count", "real_samples", "synthetic_samples"}:
                continue
            rows.append(
                {
                    "run_id": f"{spec['name']}:ep{epoch}:nfe{nfe}:{metric_name}",
                    "exp_name": spec["name"],
                    "dataset": spec["dataset"],
                    "seed": spec.get("seed", 0),
                    "stage": "eval",
                    "checkpoint_epoch": epoch,
                    "path_family": effective_spec.get("path_family", "linear"),
                    "clock_family": effective_spec.get("clock_family", "uniform"),
                    "clock_param_name": clock_param_name,
                    "clock_param_value": clock_param_value,
                    "solver": spec.get("sampling_solver", "heun2"),
                    "nfe": int(stats.get("nfe", nfe)),
                    "step_count": int(stats.get("step_count", 0)),
                    "real_samples": int(stats.get("real_samples", 0)),
                    "synthetic_samples": int(stats.get("synthetic_samples", 0)),
                    "metric": metric_name,
                    "value": float(value),
                    "status": "completed",
                    "artifact_group": spec.get("artifact_group", self.exp_group_name),
                    "strategy_id": effective_spec.get("strategy_id", ""),
                    "model_output_type": effective_spec.get("model_output_type", "velocity"),
                    "time_sampling_strategy": effective_spec.get("time_sampling_strategy", "uniform"),
                    "mixed_lambda": effective_spec.get("mixed_lambda", 0.5),
                    "stratified_bins": effective_spec.get("stratified_bins", 16),
                    **solver_aware_fields,
                }
            )
        return rows

    def _existing_metric_outputs(
        self,
        rows: List[Dict[str, object]],
        spec: Dict,
        epoch: int,
        nfe: int,
    ) -> List[str]:
        clock_param_name, clock_param_value = infer_clock_parameter(
            spec.get("clock_family", "uniform"),
            spec.get("clock_beta"),
        )
        matching_rows = [
            row
            for row in rows
            if _row_strategy_fields(row)
            if row.get("exp_name") == spec["name"]
            and row.get("dataset") == spec["dataset"]
            and row.get("seed") == spec.get("seed", 0)
            and row.get("stage") == "eval"
            and row.get("checkpoint_epoch") == epoch
            and row.get("path_family") == spec.get("path_family", "linear")
            and row.get("clock_family") == spec.get("clock_family", "uniform")
            and row.get("clock_param_name") == clock_param_name
            and row.get("clock_param_value") == clock_param_value
            and row.get("solver") == spec.get("sampling_solver", "heun2")
            and row.get("nfe") == nfe
            and row.get("status") == "completed"
            and row.get("artifact_group") == spec.get("artifact_group", self.exp_group_name)
            and _row_strategy_fields(row).get("strategy_id") == spec.get("strategy_id", "")
            and _row_strategy_fields(row).get("model_output_type") == spec.get("model_output_type", "velocity")
            and _row_strategy_fields(row).get("time_sampling_strategy") == spec.get("time_sampling_strategy", "uniform")
            and float(_row_strategy_fields(row).get("mixed_lambda", 0.5)) == float(spec.get("mixed_lambda", 0.5))
            and int(_row_strategy_fields(row).get("stratified_bins", 16)) == int(spec.get("stratified_bins", 16))
            and _solver_aware_fields_match(row=row, spec=spec)
        ]
        return sorted({str(row["metric"]) for row in matching_rows})

    def _missing_eval_metrics(
        self,
        rows: List[Dict[str, object]],
        spec: Dict,
        epoch: int,
        nfe: int,
    ) -> List[str]:
        existing_outputs = set(self._existing_metric_outputs(rows, spec, epoch, nfe))
        missing_metrics = []
        for metric_name in spec.get("metrics", ["fid"]):
            required_outputs = metric_output_names(str(metric_name))
            if any(output_name not in existing_outputs for output_name in required_outputs):
                missing_metrics.append(str(metric_name))
        return missing_metrics

    def run_all(self) -> None:
        base_config = self.config.get("base_config", {})
        assert_no_legacy_keys(base_config)
        experiments = self.config.get("experiments", [])
        existing_rows = load_result_rows(self.results_csv)
        for experiment in experiments:
            spec = merge_dicts(base_config, experiment)
            assert_no_legacy_keys(spec)
            spec = resolve_dynamic_spec_fields(spec, workspace_root=Path.cwd())
            spec.setdefault("dataset", base_config.get("dataset", "cifar10"))
            spec.setdefault("data_path", base_config.get("data_path", "./data/cifar10"))
            spec.setdefault("epochs", base_config.get("epochs", 500))
            spec.setdefault("batch_size", base_config.get("batch_size", 128))
            spec.setdefault("num_gpus", base_config.get("num_gpus", 1))
            spec.setdefault("path_family", base_config.get("path_family", "linear"))
            spec.setdefault("sampling_solver", base_config.get("sampling_solver", "heun2"))
            spec.setdefault("metrics", base_config.get("metrics", ["fid"]))
            spec.setdefault("eval_epochs", [spec["epochs"] - 1])
            spec.setdefault("eval_nfes", [base_config.get("eval_nfe", 50)])
            spec.setdefault("artifact_group", self.exp_group_name)
            spec = _resolve_strategy_fields(spec)

            exp_dir = self.base_dir / spec["dataset"] / spec["name"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            train_log = self.logs_dir / f"{spec['name']}_train.log"
            train_key = f"{spec['name']}:train"
            local_checkpoint = exp_dir / "checkpoint.pth"
            reused_checkpoint = None
            if spec.get("checkpoint_from") is not None:
                reused_checkpoint = resolve_reused_checkpoint(
                    reference=spec["checkpoint_from"],
                    spec=spec,
                    workspace_root=ROOT,
                )
                if reused_checkpoint is not None:
                    logger.info(
                        "Reusing checkpoint for %s from %s",
                        spec["name"],
                        reused_checkpoint,
                    )
                else:
                    logger.warning(
                        "Configured checkpoint reuse for %s could not be resolved from reference: %s",
                        spec["name"],
                        spec["checkpoint_from"],
                    )
            checkpoint = local_checkpoint
            local_checkpoint_compatible = False
            if local_checkpoint.exists():
                local_checkpoint_compatible = checkpoint_matches_spec(
                    checkpoint_path=local_checkpoint,
                    spec=spec,
                )
                if not local_checkpoint_compatible:
                    logger.warning(
                        "Ignoring stale local checkpoint for %s because its saved args do not match the current spec: %s",
                        spec["name"],
                        local_checkpoint,
                    )

            if (
                spec.get("checkpoint_from") is not None
                and reused_checkpoint is None
                and not (local_checkpoint.exists() and local_checkpoint_compatible)
            ):
                logger.error(
                    "Checkpoint reuse is required for %s, but no compatible checkpoint was found. "
                    "reference=%s. Refusing to fall back to training.",
                    spec["name"],
                    spec["checkpoint_from"],
                )
                self.state[train_key] = "failed_missing_reused_checkpoint"
                self._save_state()
                continue

            if (
                local_checkpoint.exists()
                and local_checkpoint_compatible
                and self.state.get(train_key) == "completed"
            ):
                checkpoint = local_checkpoint
            elif local_checkpoint.exists() and local_checkpoint_compatible:
                self.state[train_key] = "running"
                self._save_state()
                success = run_command(
                    self.build_train_cmd(spec, exp_dir, resume_checkpoint=local_checkpoint),
                    train_log,
                    retries=1,
                )
                self.state[train_key] = "completed" if success else "failed"
                self._save_state()
                if not success:
                    continue
                checkpoint = local_checkpoint
            elif reused_checkpoint is not None:
                checkpoint = reused_checkpoint
                self.state[train_key] = "completed"
                self._save_state()
            elif (
                self.state.get(train_key) != "completed"
                or not local_checkpoint.exists()
                or not local_checkpoint_compatible
            ):
                self.state[train_key] = "running"
                self._save_state()
                resume_checkpoint = (
                    local_checkpoint if local_checkpoint.exists() and local_checkpoint_compatible else None
                )
                success = run_command(
                    self.build_train_cmd(spec, exp_dir, resume_checkpoint=resume_checkpoint),
                    train_log,
                    retries=1,
                )
                self.state[train_key] = "completed" if success else "failed"
                self._save_state()
                if not success:
                    continue
                checkpoint = local_checkpoint

            for epoch in spec["eval_epochs"]:
                checkpoint_for_eval = checkpoint
                if checkpoint_for_eval == local_checkpoint:
                    checkpoint_for_eval = find_checkpoint(exp_dir, epoch)
                if checkpoint_for_eval is None:
                    logger.warning("Checkpoint for epoch %s not found in %s", epoch, exp_dir)
                    continue
                for nfe in spec["eval_nfes"]:
                    eval_key = f"{spec['name']}:ep{epoch}:nfe{nfe}"
                    missing_metrics = self._missing_eval_metrics(
                        existing_rows,
                        spec,
                        epoch,
                        nfe,
                    )
                    if not missing_metrics:
                        if self.state.get(eval_key) != "completed":
                            self.state[eval_key] = "completed"
                            self._save_state()
                        continue
                    eval_dir = exp_dir / f"eval_ep{epoch}_nfe{nfe}"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_log = self.logs_dir / f"{spec['name']}_ep{epoch}_nfe{nfe}.log"
                    self.state[eval_key] = "running"
                    self._save_state()
                    success = run_command(
                        self.build_eval_cmd(
                            spec,
                            eval_dir,
                            checkpoint_for_eval,
                            nfe,
                            metrics_override=missing_metrics,
                        ),
                        eval_log,
                        retries=0,
                    )
                    if not success:
                        self.state[eval_key] = "failed"
                        self._save_state()
                        continue
                    stats = extract_eval_stats(eval_dir / "log.txt")
                    if not stats:
                        self.state[eval_key] = "failed_no_eval_stats"
                        self._save_state()
                        continue
                    new_rows = self._result_rows(
                        spec,
                        epoch,
                        nfe,
                        stats,
                        checkpoint_path=checkpoint_for_eval,
                    )
                    append_result_rows(self.results_csv, new_rows)
                    existing_rows.extend(new_rows)
                    self.state[eval_key] = "completed"
                    self._save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    ExperimentManager(args.config).run_all()
