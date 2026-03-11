#!/usr/bin/env python3
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from experiments.result_utils import append_result_rows, ensure_results_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def find_checkpoint(exp_dir: Path, epoch: int) -> Optional[Path]:
    candidates = [
        exp_dir / f"checkpoint-{epoch}.pth",
        exp_dir / f"checkpoint{epoch}.pth",
        exp_dir / f"checkpoint{epoch:04d}.pth",
        exp_dir / "checkpoint.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def infer_clock_param(experiment: Dict) -> Tuple[str, Optional[float]]:
    clock_family = experiment.get("clock_family", "uniform")
    if clock_family.startswith("ft_"):
        return "beta", experiment.get("clock_beta")
    if clock_family == "poly_a0.5":
        return "a", 0.5
    if clock_family == "poly_a2.0":
        return "a", 2.0
    if clock_family == "sigmoid_k8":
        return "k", 8.0
    if clock_family == "exp_l3":
        return "lambda", 3.0
    return "none", None


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
        if spec.get("cfg_scale") is not None:
            flags.append(f"--cfg_scale {spec['cfg_scale']}")
        if spec.get("class_drop_prob") is not None:
            flags.append(f"--class_drop_prob {spec['class_drop_prob']}")
        if spec.get("fid_samples") is not None:
            flags.append(f"--fid_samples {spec['fid_samples']}")
        return flags

    def build_train_cmd(self, spec: Dict, output_dir: Path) -> str:
        flags = self._base_flags(spec, output_dir)
        flags.extend(
            [
                f"--epochs {spec['epochs']}",
                f"--lr {spec.get('lr', 0.0001)}",
                f"--seed {spec.get('seed', 0)}",
            ]
        )
        if spec.get("eval_frequency") is not None:
            flags.append(f"--eval_frequency {spec['eval_frequency']}")
        if spec.get("decay_lr", False):
            flags.append("--decay_lr")
        return f"{self._launcher(spec.get('num_gpus', 1))} examples/image/train.py " + " ".join(flags)

    def build_eval_cmd(self, spec: Dict, output_dir: Path, checkpoint: Path, nfe: int) -> str:
        flags = self._base_flags(spec, output_dir)
        flags.extend(
            [
                "--eval_only",
                f"--resume {checkpoint}",
                f"--eval_nfe {nfe}",
                f"--seed {spec.get('seed', 0)}",
            ]
        )
        if "fid" in spec.get("metrics", ["fid"]):
            flags.append("--compute_fid")
        return f"{self._launcher(spec.get('num_gpus', 1))} examples/image/train.py " + " ".join(flags)

    def _result_rows(self, spec: Dict, epoch: int, nfe: int, stats: Dict[str, float]) -> List[Dict[str, object]]:
        rows = []
        clock_param_name, clock_param_value = infer_clock_param(spec)
        for metric_name, value in stats.items():
            if metric_name in {"nfe", "step_count"}:
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
                    "metric": metric_name,
                    "value": float(value),
                    "status": "completed",
                    "artifact_group": spec.get("artifact_group", self.exp_group_name),
                }
            )
        return rows

    def run_all(self) -> None:
        base_config = self.config.get("base_config", {})
        experiments = self.config.get("experiments", [])
        for experiment in experiments:
            spec = merge_dicts(base_config, experiment)
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
            checkpoint = exp_dir / "checkpoint.pth"

            if self.state.get(train_key) != "completed" or not checkpoint.exists():
                self.state[train_key] = "running"
                self._save_state()
                success = run_command(self.build_train_cmd(spec, exp_dir), train_log, retries=1)
                self.state[train_key] = "completed" if success else "failed"
                self._save_state()
                if not success:
                    continue

            for epoch in spec["eval_epochs"]:
                checkpoint = find_checkpoint(exp_dir, epoch)
                if checkpoint is None:
                    logger.warning("Checkpoint for epoch %s not found in %s", epoch, exp_dir)
                    continue
                for nfe in spec["eval_nfes"]:
                    eval_key = f"{spec['name']}:ep{epoch}:nfe{nfe}"
                    if self.state.get(eval_key) == "completed":
                        continue
                    eval_dir = exp_dir / f"eval_ep{epoch}_nfe{nfe}"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    eval_log = self.logs_dir / f"{spec['name']}_ep{epoch}_nfe{nfe}.log"
                    self.state[eval_key] = "running"
                    self._save_state()
                    success = run_command(
                        self.build_eval_cmd(spec, eval_dir, checkpoint, nfe),
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
                    append_result_rows(self.results_csv, self._result_rows(spec, epoch, nfe, stats))
                    self.state[eval_key] = "completed"
                    self._save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    ExperimentManager(args.config).run_all()
