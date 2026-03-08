#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import time
import glob
from pathlib import Path
import yaml

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd, log_file=None, env=None, retries=0):
    """Execute a shell command with optional retries and file logging."""
    for attempt in range(retries + 1):
        try:
            logger.info(f"Executing: {cmd}")
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"\n--- Execution Attempt {attempt + 1} at {time.ctime()} ---\n")
                    f.write(f"Command: {cmd}\n\n")
                    
                    process = subprocess.Popen(
                        cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
                    )
                    for line in process.stdout:
                        print(line, end="") 
                        f.write(line)       
                    process.wait()
            else:
                process = subprocess.run(cmd, shell=True, env=env, check=True)
                
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            return True 
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            if attempt < retries:
                logger.warning(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return False

def extract_fid_from_log(log_path):
    """Parses log.txt to find the latest eval_fid."""
    if not os.path.exists(log_path):
        return None
    latest_fid = None
    try:
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "eval_fid" in data:
                        latest_fid = data["eval_fid"]
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return latest_fid

def find_checkpoint(exp_dir, epoch):
    """
    尝试寻找对应 epoch 的 checkpoint。
    逻辑：
    1. 如果 epoch 是 'latest' 或配置的最大 epoch，优先找 checkpoint.pth
    2. 否则找 checkpoint{epoch}.pth 或 checkpoint_{epoch}.pth
    """
    # 常见命名模式
    candidates = [
        f"checkpoint{epoch:04d}.pth", # checkpoint0100.pth
        f"checkpoint{epoch}.pth",     # checkpoint100.pth
        f"checkpoint-{epoch}.pth",    # checkpoint_100.pth
        "checkpoint.pth"               # 默认最新
    ]
    
    for fname in candidates:
        p = exp_dir / fname
        if p.exists():
            return p
            
    # 如果没找到，且请求的是最大 epoch，可能 checkpoint.pth 就是我们要的
    default_ckpt = exp_dir / "checkpoint.pth"
    if default_ckpt.exists():
        logger.warning(f"Specific checkpoint for epoch {epoch} not found. Using default checkpoint.pth (Warning: Ensure this is the correct epoch).")
        return default_ckpt
        
    return None

# --- Experiment Manager ---

class ExperimentManager:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.exp_group_name = self.config.get("experiment_name", "unnamed")
        
        self.base_dir = Path(f"./experiments/results/{self.exp_group_name}")
        self.logs_dir = Path(f"./experiments/logs/{self.exp_group_name}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.base_dir / "experiment_status.json"
        self.state = self._load_state()
        
        self.results_csv = self.base_dir / "results.csv"
        if not self.results_csv.exists():
            with open(self.results_csv, "w") as f:
                # 新增 epoch 列
                f.write("exp_name,dataset,alpha,lambda_scale,epoch,nfe,fid,status\n")

    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def _record_result(self, exp_name, dataset, alpha, lamb, epoch, nfe, fid, status):
        with open(self.results_csv, "a") as f:
            f.write(f"{exp_name},{dataset},{alpha},{lamb},{epoch},{nfe},{fid},{status}\n")

    def build_train_cmd(self, exp_cfg, out_dir):
        base_cfg = self.config["base_config"]
        num_gpus = base_cfg.get("num_gpus", 1)
        
        cmd = f"torchrun --standalone --nproc_per_node={num_gpus} examples/image/train.py "
        cmd += f"--dataset {exp_cfg.get('dataset', base_cfg.get('dataset'))} "
        cmd += f"--data_path {base_cfg['data_path']} "
        cmd += f"--epochs {base_cfg['epochs']} "
        cmd += f"--batch_size {base_cfg['batch_size']} "
        cmd += f"--output_dir {out_dir} "
        
        if exp_cfg.get("use_ema", base_cfg.get("use_ema", False)):
            cmd += "--use_ema "
        if exp_cfg.get("use_ft_eqm", False):
            cmd += "--use_ft_eqm "
            cmd += f"--alpha {exp_cfg['alpha']} "
            if "lambda_scale" in exp_cfg and exp_cfg["lambda_scale"] is not None:
                cmd += f"--lambda_scale {exp_cfg['lambda_scale']} "
        
        return cmd

    def build_eval_cmd(self, exp_cfg, out_dir, ckpt_path, nfe):
        base_cfg = self.config["base_config"]
        num_gpus = base_cfg.get("num_gpus", 1)
        
        # 假设 midpoint (2 NFE per step)
        steps = nfe // 2 if nfe >= 2 else 1
        step_size = 1.0 / steps
        ode_options = f"'{{\"step_size\": {step_size}}}'"

        cmd = f"torchrun --standalone --nproc_per_node={num_gpus} examples/image/train.py "
        cmd += f"--dataset {exp_cfg.get('dataset', base_cfg.get('dataset'))} "
        cmd += f"--data_path {base_cfg['data_path']} "
        cmd += f"--batch_size 64 "
        cmd += f"--eval_only --compute_fid "
        cmd += f"--resume {ckpt_path} "
        cmd += f"--output_dir {out_dir} "
        cmd += f"--ode_options {ode_options} "

        if exp_cfg.get("use_ema", self.config["base_config"].get("use_ema", False)):
            cmd += "--use_ema "
            
        if exp_cfg.get("use_ft_eqm", False):
            cmd += "--use_ft_eqm "
            cmd += f"--alpha {exp_cfg['alpha']} "
            if "lambda_scale" in exp_cfg and exp_cfg["lambda_scale"] is not None:
                cmd += f"--lambda_scale {exp_cfg['lambda_scale']} "
        
        return cmd

    def run_all(self):
        experiments = self.config.get("experiments", [])
        
        for exp in experiments:
            exp_name = exp["name"]
            dataset = exp.get("dataset", "cifar10")
            alpha = exp.get("alpha", 0.5)
            lamb = exp.get("lambda_scale", "auto")
            
            logger.info(f"========== Processing: {exp_name} (Alpha={alpha}) ==========")
            exp_dir = self.base_dir / dataset / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 1. Training ---
            train_log = self.logs_dir / f"{exp_name}_train.log"
            # 训练只看最终 checkpoint 是否存在来决定是否跳过
            final_ckpt = exp_dir / "checkpoint.pth"
            state_key = f"{exp_name}_train"
            
            if self.state.get(state_key) == "completed" and final_ckpt.exists():
                logger.info(f"Training already completed.")
            else:
                self.state[state_key] = "running"
                self._save_state()
                success = run_command(self.build_train_cmd(exp, exp_dir), log_file=train_log, retries=1)
                if success:
                    self.state[state_key] = "completed"
                else:
                    self.state[state_key] = "failed"
                    self._save_state()
                    continue 

            # --- 2. Evaluation Loop (Epoch -> NFE) ---
            # 获取需要评估的 epochs 列表，默认只评估最终 epoch
            target_epochs = exp.get("eval_epochs", [self.config["base_config"]["epochs"]])
            eval_nfes = exp.get("eval_nfes", [100])

            for epoch in target_epochs:
                # 寻找该 epoch 的权重文件
                ckpt_path = find_checkpoint(exp_dir, epoch)
                
                if ckpt_path is None:
                    logger.warning(f"Checkpoint for epoch {epoch} not found in {exp_dir}. Skipping.")
                    continue

                for nfe in eval_nfes:
                    eval_task_name = f"{exp_name}_ep{epoch}_nfe{nfe}"
                    eval_out_dir = exp_dir / f"eval_ep{epoch}_nfe{nfe}"
                    eval_out_dir.mkdir(parents=True, exist_ok=True)
                    eval_log = self.logs_dir / f"{eval_task_name}.log"

                    if self.state.get(eval_task_name) == "completed":
                        continue

                    logger.info(f"Evaluating: Alpha={alpha}, Epoch={epoch}, NFE={nfe}")
                    self.state[eval_task_name] = "running"
                    self._save_state()

                    success = run_command(
                        self.build_eval_cmd(exp, eval_out_dir, ckpt_path, nfe), 
                        log_file=eval_log, retries=0
                    )

                    if success:
                        fid = extract_fid_from_log(eval_out_dir / "log.txt")
                        if fid is not None:
                            self.state[eval_task_name] = "completed"
                            self._record_result(exp_name, dataset, alpha, lamb, epoch, nfe, fid, "completed")
                            logger.info(f"Result: FID={fid:.2f}")
                        else:
                            self.state[eval_task_name] = "failed_no_fid"
                    else:
                        self.state[eval_task_name] = "failed"
                    
                    self._save_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    ExperimentManager(args.config).run_all()