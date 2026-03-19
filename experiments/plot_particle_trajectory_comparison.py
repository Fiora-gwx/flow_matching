#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_ROOT = ROOT / "examples" / "image"
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))

from experiments.checkpoint_utils import resolve_checkpoint_path, resolve_reused_checkpoint
from experiments.result_utils import resolve_best_beta_reference
from training.continuous_runtime import build_continuous_batch

DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "ft_clock" / "particle_trajectory_comparison.yaml"


@dataclass
class MethodSpec:
    label: str
    pairing: str
    path_family: str
    clock_family: str
    clock_beta: Optional[float]
    dataset: str = "cifar10"
    name: str = ""
    artifact_group: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    checkpoint_path: Optional[str] = None


PAIRING_CHOICES = ("oracle", "random", "reverse", "cyclic_shift")
PALETTE = ("#1f77b4", "#2ca02c", "#d62728")


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
    checkpoint_path = None
    if checkpoint_from is not None:
        checkpoint_path = resolve_reused_checkpoint(
            reference=checkpoint_from,
            spec=resolved,
            workspace_root=ROOT,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Checkpoint not found for method={resolved.get('label', resolved.get('name', '<unnamed>'))} "
                f"from reference {checkpoint_from}"
            )
    elif resolved.get("name"):
        base_dir = ROOT / "experiments" / "results"
        artifact_group = resolved.get("artifact_group")
        if artifact_group:
            base_dir = base_dir / str(artifact_group)
        checkpoint_path = resolve_checkpoint_path(
            base_dir=base_dir,
            spec=resolved,
            workspace_root=ROOT,
        )
    if checkpoint_path is not None:
        resolved["checkpoint_path"] = str(checkpoint_path)
    return resolved


def build_method_spec(method: Dict[str, object]) -> MethodSpec:
    return MethodSpec(**resolve_method_fields(method))


def make_demo_points(
    num_points_per_group: int,
    seed: int,
    jitter: float = 0.16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source_anchors = torch.tensor(
        [[-1.8, 1.05], [-1.8, -1.05], [-0.35, 0.0]],
        dtype=torch.float32,
    )
    target_anchors = torch.tensor(
        [[1.8, 1.05], [1.8, -1.05], [0.35, 0.0]],
        dtype=torch.float32,
    )
    group_points = []
    group_targets = []
    group_ids = []
    rotation_angles = [0.35, -0.30, 0.15]
    for group_index, (source_anchor, target_anchor, angle) in enumerate(
        zip(source_anchors, target_anchors, rotation_angles)
    ):
        offsets = torch.randn(
            (num_points_per_group, 2),
            generator=generator,
            dtype=torch.float32,
        ) * jitter
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        rotation = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=torch.float32,
        )
        transformed_offsets = offsets @ rotation.T
        group_points.append(source_anchor.unsqueeze(0) + offsets)
        group_targets.append(target_anchor.unsqueeze(0) + transformed_offsets)
        group_ids.append(torch.full((num_points_per_group,), group_index, dtype=torch.long))
    return (
        torch.cat(group_points, dim=0),
        torch.cat(group_targets, dim=0),
        torch.cat(group_ids, dim=0),
    )


def build_pair_indices(strategy: str, count: int, seed: int) -> torch.Tensor:
    if strategy not in PAIRING_CHOICES:
        raise ValueError(f"Unsupported pairing strategy: {strategy}")
    if strategy == "oracle":
        return torch.arange(count, dtype=torch.long)
    if strategy == "reverse":
        return torch.arange(count - 1, -1, -1, dtype=torch.long)
    if strategy == "cyclic_shift":
        return torch.roll(torch.arange(count, dtype=torch.long), shifts=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator)


def compute_trajectory(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    pair_indices: torch.Tensor,
    path_family: str,
    clock_family: str,
    clock_beta: Optional[float],
    num_steps: int,
) -> torch.Tensor:
    paired_targets = target_points.index_select(0, pair_indices)
    trajectory = []
    r_grid = torch.linspace(0.0, 1.0, num_steps, dtype=source_points.dtype)
    for r_value in r_grid:
        r = torch.full((source_points.shape[0],), float(r_value.item()), dtype=source_points.dtype)
        batch = build_continuous_batch(
            x_1=paired_targets,
            x_0=source_points,
            r=r,
            path_family=path_family,
            clock_family=clock_family,
            clock_beta=clock_beta,
        )
        trajectory.append(batch.x_t)
    return torch.stack(trajectory, dim=0)


def summarize_trajectory(
    label: str,
    trajectory: torch.Tensor,
    oracle_targets: torch.Tensor,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, object]:
    deltas = trajectory[1:] - trajectory[:-1]
    path_length = deltas.norm(dim=2).sum(dim=0)
    mean_curvature = 0.0
    if trajectory.shape[0] >= 3:
        second_diff = trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
        mean_curvature = float(second_diff.norm(dim=2).mean().item())
    endpoint_error = (trajectory[-1] - oracle_targets).norm(dim=1)
    return {
        "method": label,
        "checkpoint": checkpoint_path or "",
        "mean_path_length": float(path_length.mean().item()),
        "mean_curvature": mean_curvature,
        "mean_oracle_endpoint_error": float(endpoint_error.mean().item()),
    }


def write_summary_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_reference(ax, oracle_trajectory: torch.Tensor) -> None:
    for point_index in range(oracle_trajectory.shape[1]):
        ax.plot(
            oracle_trajectory[:, point_index, 0].cpu(),
            oracle_trajectory[:, point_index, 1].cpu(),
            color="#bbbbbb",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
        )


def draw_panel(
    ax,
    method_spec: MethodSpec,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    group_ids: torch.Tensor,
    pair_indices: torch.Tensor,
    trajectory: torch.Tensor,
    oracle_trajectory: torch.Tensor,
) -> None:
    _draw_reference(ax, oracle_trajectory)
    marker_steps = [0, trajectory.shape[0] // 3, 2 * trajectory.shape[0] // 3, trajectory.shape[0] - 1]
    for point_index in range(trajectory.shape[1]):
        color = PALETTE[int(group_ids[point_index].item()) % len(PALETTE)]
        path = trajectory[:, point_index].cpu()
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.6, alpha=0.95)
        markers = path[marker_steps]
        ax.scatter(markers[:, 0], markers[:, 1], color=color, s=14, alpha=0.85, zorder=3)

    ax.scatter(
        source_points[:, 0].cpu(),
        source_points[:, 1].cpu(),
        c=[PALETTE[int(group_id.item()) % len(PALETTE)] for group_id in group_ids],
        marker="o",
        s=40,
        edgecolors="black",
        linewidths=0.5,
        label="source",
        zorder=4,
    )
    matched_targets = target_points.index_select(0, pair_indices)
    ax.scatter(
        matched_targets[:, 0].cpu(),
        matched_targets[:, 1].cpu(),
        c=[PALETTE[int(group_id.item()) % len(PALETTE)] for group_id in group_ids],
        marker="D",
        s=34,
        edgecolors="black",
        linewidths=0.4,
        label="matched target",
        zorder=4,
    )
    ax.set_title(
        f"{method_spec.label}\n"
        f"pair={method_spec.pairing}, path={method_spec.path_family}, clock={method_spec.clock_family}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_comparison(
    output_dir: Path,
    baseline_spec: MethodSpec,
    proposed_spec: MethodSpec,
    num_points_per_group: int,
    num_steps: int,
    seed: int,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_points, oracle_targets, group_ids = make_demo_points(
        num_points_per_group=num_points_per_group,
        seed=seed,
    )
    oracle_pair = build_pair_indices("oracle", source_points.shape[0], seed=seed)
    baseline_pair = build_pair_indices(
        baseline_spec.pairing,
        source_points.shape[0],
        seed=seed + 11,
    )
    proposed_pair = build_pair_indices(
        proposed_spec.pairing,
        source_points.shape[0],
        seed=seed + 29,
    )

    perfect_spec = MethodSpec(
        label="Theoretical Perfect Pairing",
        pairing="oracle",
        path_family="linear",
        clock_family="uniform",
        clock_beta=None,
    )
    perfect_trajectory = compute_trajectory(
        source_points,
        oracle_targets,
        oracle_pair,
        path_family=perfect_spec.path_family,
        clock_family=perfect_spec.clock_family,
        clock_beta=perfect_spec.clock_beta,
        num_steps=num_steps,
    )
    baseline_trajectory = compute_trajectory(
        source_points,
        oracle_targets,
        baseline_pair,
        path_family=baseline_spec.path_family,
        clock_family=baseline_spec.clock_family,
        clock_beta=baseline_spec.clock_beta,
        num_steps=num_steps,
    )
    proposed_trajectory = compute_trajectory(
        source_points,
        oracle_targets,
        proposed_pair,
        path_family=proposed_spec.path_family,
        clock_family=proposed_spec.clock_family,
        clock_beta=proposed_spec.clock_beta,
        num_steps=num_steps,
    )

    figure_path = output_dir / "particle_trajectory_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_panel(
        axes[0],
        perfect_spec,
        source_points,
        oracle_targets,
        group_ids,
        oracle_pair,
        perfect_trajectory,
        perfect_trajectory,
    )
    draw_panel(
        axes[1],
        baseline_spec,
        source_points,
        oracle_targets,
        group_ids,
        baseline_pair,
        baseline_trajectory,
        perfect_trajectory,
    )
    draw_panel(
        axes[2],
        proposed_spec,
        source_points,
        oracle_targets,
        group_ids,
        proposed_pair,
        proposed_trajectory,
        perfect_trajectory,
    )
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(figure_path, dpi=300)
    plt.close(fig)

    summary_rows = [
        summarize_trajectory(perfect_spec.label, perfect_trajectory, oracle_targets),
        summarize_trajectory(
            baseline_spec.label,
            baseline_trajectory,
            oracle_targets,
            checkpoint_path=baseline_spec.checkpoint_path,
        ),
        summarize_trajectory(
            proposed_spec.label,
            proposed_trajectory,
            oracle_targets,
            checkpoint_path=proposed_spec.checkpoint_path,
        ),
    ]
    summary_path = output_dir / "particle_trajectory_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    return figure_path, summary_path


def run_from_config(config_path: Path) -> Tuple[Path, Path]:
    config = load_config(config_path)
    output_dir = Path(str(config.get("output_dir", "experiments/results/particle_trajectory_compare")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return plot_comparison(
        output_dir=output_dir,
        baseline_spec=build_method_spec(config["baseline"]),
        proposed_spec=build_method_spec(config["proposed"]),
        num_points_per_group=int(config.get("num_points_per_group", 6)),
        num_steps=int(config.get("num_steps", 25)),
        seed=int(config.get("seed", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--num_points_per_group", type=int, default=6)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline_label", type=str, default="Traditional Method")
    parser.add_argument("--baseline_pairing", choices=PAIRING_CHOICES, default="random")
    parser.add_argument("--baseline_path_family", choices=("linear", "trig_vp"), default="linear")
    parser.add_argument(
        "--baseline_clock_family",
        choices=(
            "uniform",
            "ft_linear_beta",
            "ft_vp_beta",
            "poly_a0.5",
            "poly_a2.0",
            "cosine",
            "sigmoid_k8",
            "exp_l3",
        ),
        default="uniform",
    )
    parser.add_argument("--baseline_clock_beta", type=float, default=None)
    parser.add_argument("--proposed_label", type=str, default="Our Method")
    parser.add_argument("--proposed_pairing", choices=PAIRING_CHOICES, default="oracle")
    parser.add_argument("--proposed_path_family", choices=("linear", "trig_vp"), default="linear")
    parser.add_argument(
        "--proposed_clock_family",
        choices=(
            "uniform",
            "ft_linear_beta",
            "ft_vp_beta",
            "poly_a0.5",
            "poly_a2.0",
            "cosine",
            "sigmoid_k8",
            "exp_l3",
        ),
        default="ft_linear_beta",
    )
    parser.add_argument("--proposed_clock_beta", type=float, default=0.5)
    args = parser.parse_args()

    if args.config is not None or args.out is None:
        run_from_config(args.config or DEFAULT_CONFIG)
        return

    baseline_spec = MethodSpec(
        label=args.baseline_label,
        pairing=args.baseline_pairing,
        path_family=args.baseline_path_family,
        clock_family=args.baseline_clock_family,
        clock_beta=args.baseline_clock_beta,
    )
    proposed_spec = MethodSpec(
        label=args.proposed_label,
        pairing=args.proposed_pairing,
        path_family=args.proposed_path_family,
        clock_family=args.proposed_clock_family,
        clock_beta=args.proposed_clock_beta,
    )
    plot_comparison(
        output_dir=args.out,
        baseline_spec=baseline_spec,
        proposed_spec=proposed_spec,
        num_points_per_group=args.num_points_per_group,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
