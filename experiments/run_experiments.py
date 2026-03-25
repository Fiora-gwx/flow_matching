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

from experiments.checkpoint_utils import find_checkpoint, resolve_reused_checkpoint
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

    def _result_rows(self, spec: Dict, epoch: int, nfe: int, stats: Dict[str, float]) -> List[Dict[str, object]]:
        rows = []
        clock_param_name, clock_param_value = infer_clock_parameter(
            spec.get("clock_family", "uniform"),
            spec.get("clock_beta"),
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
                    "path_family": spec.get("path_family", "linear"),
                    "clock_family": spec.get("clock_family", "uniform"),
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
                    workspace_root=Path.cwd(),
                )
                if reused_checkpoint is not None:
                    logger.info(
                        "Reusing checkpoint for %s from %s",
                        spec["name"],
                        reused_checkpoint,
                    )
                else:
                    logger.warning(
                        "Configured checkpoint reuse for %s but no external checkpoint was found: %s",
                        spec["name"],
                        spec["checkpoint_from"],
                    )
            checkpoint = local_checkpoint

            if local_checkpoint.exists() and self.state.get(train_key) == "completed":
                checkpoint = local_checkpoint
            elif local_checkpoint.exists():
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
            elif self.state.get(train_key) != "completed" or not local_checkpoint.exists():
                self.state[train_key] = "running"
                self._save_state()
                resume_checkpoint = local_checkpoint if local_checkpoint.exists() else None
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
                    new_rows = self._result_rows(spec, epoch, nfe, stats)
                    append_result_rows(self.results_csv, new_rows)
                    existing_rows.extend(new_rows)
                    self.state[eval_key] = "completed"
                    self._save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    ExperimentManager(args.config).run_all()
