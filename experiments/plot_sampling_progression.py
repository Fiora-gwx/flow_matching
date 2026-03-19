#!/usr/bin/env python3
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.utils import make_grid

from experiments.checkpoint_utils import resolve_checkpoint_path, resolve_reused_checkpoint
from experiments.result_utils import resolve_best_beta_reference

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_ROOT = ROOT / "examples" / "image"
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))

from models.model_configs import instantiate_model
from training.eval_loop import CFGScaledModel
from training.fixed_step_solver import FixedStepSample, solve_fixed_budget

DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "ft_clock" / "sampling_progression.yaml"


@dataclass
class MethodSpec:
    label: str
    dataset: str
    name: str
    path_family: str
    clock_family: str
    sampling_solver: str
    eval_nfe: int
    checkpoint_epoch: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_base_dir: Optional[str] = None
    artifact_group: Optional[str] = None
    clock_beta: Optional[float] = None
    use_ema: bool = False
    cfg_scale: float = 0.0
    fixed_label: int = 0


def load_config(config_path: Path) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_method_fields(method: Dict[str, object]) -> Dict[str, object]:
    resolved = dict(method)
    best_beta_from = resolved.pop("best_beta_from", None)
    checkpoint_from = resolved.pop("checkpoint_from", None)
    if best_beta_from is not None:
        resolved["clock_beta"] = resolve_best_beta_reference(
            reference=best_beta_from,
            workspace_root=ROOT,
        )
    if checkpoint_from is not None:
        checkpoint = resolve_reused_checkpoint(
            reference=checkpoint_from,
            spec=resolved,
            workspace_root=ROOT,
        )
        if checkpoint is None:
            raise FileNotFoundError(
                f"Checkpoint not found for method={resolved.get('label', resolved.get('name', '<unnamed>'))} "
                f"from reference {checkpoint_from}"
            )
        resolved["checkpoint_path"] = str(checkpoint)
    return resolved


def resolve_method_checkpoint(method: MethodSpec) -> Path:
    if method.checkpoint_base_dir:
        base_dir = Path(method.checkpoint_base_dir)
        if not base_dir.is_absolute():
            base_dir = ROOT / base_dir
    elif method.artifact_group:
        base_dir = ROOT / "experiments" / "results" / method.artifact_group
    else:
        base_dir = ROOT / "experiments" / "results"

    checkpoint = resolve_checkpoint_path(
        base_dir=base_dir,
        spec={
            "dataset": method.dataset,
            "name": method.name,
            "checkpoint_epoch": method.checkpoint_epoch,
            "checkpoint_path": method.checkpoint_path,
        },
        workspace_root=ROOT,
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"Checkpoint not found for method={method.label}. "
            f"dataset={method.dataset}, name={method.name}, "
            f"artifact_group={method.artifact_group}, checkpoint_path={method.checkpoint_path}"
        )
    return checkpoint


def load_model_from_checkpoint(
    method: MethodSpec,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = instantiate_model(
        architechture=method.dataset,
        is_discrete=False,
        use_ema=method.use_ema,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def build_snapshot_indices(num_states: int, snapshot_ratios: Sequence[float]) -> List[int]:
    if num_states <= 0:
        raise ValueError(f"num_states must be positive. Got {num_states}.")
    indices = []
    for ratio in snapshot_ratios:
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError(f"snapshot ratio must be in [0, 1]. Got {ratio}.")
        indices.append(int(round(float(ratio) * (num_states - 1))))
    return indices


def sample_method_trajectory(
    method: MethodSpec,
    checkpoint_path: Path,
    x_init: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> FixedStepSample:
    model = load_model_from_checkpoint(method, checkpoint_path, device)
    model_wrapper = CFGScaledModel(model)
    model_wrapper.train(False)
    result = solve_fixed_budget(
        velocity_model=model_wrapper,
        x_init=x_init.clone(),
        solver_name=method.sampling_solver,
        nfe_budget=int(method.eval_nfe),
        return_trajectory=True,
        label=labels,
        cfg_scale=method.cfg_scale,
    )
    if result.trajectory is None:
        raise RuntimeError(f"Trajectory capture failed for method={method.label}.")
    return result


def make_snapshot_grid(
    trajectory: torch.Tensor,
    state_index: int,
    grid_nrow: int,
) -> torch.Tensor:
    images = torch.clamp(trajectory[state_index] * 0.5 + 0.5, min=0.0, max=1.0)
    return make_grid(images, nrow=grid_nrow, padding=2)


def plot_snapshot_comparison(
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    grids: Sequence[Sequence[torch.Tensor]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = len(row_labels)
    num_cols = len(column_labels)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.2 * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [list(axes)]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    for row_index, row_label in enumerate(row_labels):
        for col_index, col_label in enumerate(column_labels):
            ax = axes[row_index][col_index]
            grid = grids[row_index][col_index].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(grid)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(col_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(row_label, fontsize=11, rotation=0, labelpad=38, va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_snapshot_metadata(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_method_specs(config: Dict[str, object]) -> List[MethodSpec]:
    methods = config.get("methods", [])
    if len(methods) < 2:
        raise ValueError("At least two methods are required for comparison.")
    specs = []
    for method in methods:
        specs.append(MethodSpec(**resolve_method_fields(method)))
    return specs


def run(config_path: Path) -> Tuple[Path, Path]:
    config = load_config(config_path)
    method_specs = build_method_specs(config)
    device = torch.device(str(config.get("device", "cuda")))
    seed = int(config.get("seed", 0))
    num_samples = int(config.get("num_samples", 8))
    grid_nrow = int(config.get("grid_nrow", 4))
    image_size = int(config.get("image_size", 32))
    snapshot_ratios = list(config.get("snapshot_ratios", [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]))
    output_dir = Path(str(config.get("output_dir", "experiments/results/sampling_progression")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    x_init = torch.randn((num_samples, 3, image_size, image_size), dtype=torch.float32, device=device)

    rows = []
    metadata_rows = []
    column_labels = [f"r={float(ratio):.2f}" for ratio in snapshot_ratios]
    for method in method_specs:
        checkpoint_path = resolve_method_checkpoint(method)
        labels = torch.full(
            (num_samples,),
            fill_value=int(method.fixed_label),
            dtype=torch.long,
            device=device,
        )
        sample = sample_method_trajectory(
            method,
            checkpoint_path=checkpoint_path,
            x_init=x_init,
            labels=labels,
            device=device,
        )
        snapshot_indices = build_snapshot_indices(sample.trajectory.shape[0], snapshot_ratios)
        row_grids = []
        for ratio, state_index in zip(snapshot_ratios, snapshot_indices):
            row_grids.append(make_snapshot_grid(sample.trajectory, state_index, grid_nrow))
            metadata_rows.append(
                {
                    "method": method.label,
                    "checkpoint": str(checkpoint_path),
                    "snapshot_ratio": float(ratio),
                    "state_index": int(state_index),
                    "time_value": float(sample.time_grid[state_index].item()),
                    "nfe_budget": int(method.eval_nfe),
                    "actual_nfe": int(sample.nfe),
                    "step_count": int(sample.step_count),
                    "path_family": method.path_family,
                    "clock_family": method.clock_family,
                    "clock_beta": method.clock_beta,
                    "sampling_solver": method.sampling_solver,
                }
            )
        rows.append(row_grids)

    figure_path = output_dir / "sampling_progression_comparison.png"
    plot_snapshot_comparison(
        row_labels=[spec.label for spec in method_specs],
        column_labels=column_labels,
        grids=rows,
        output_path=figure_path,
    )
    metadata_path = output_dir / "sampling_progression_metadata.csv"
    write_snapshot_metadata(metadata_path, metadata_rows)
    return figure_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
