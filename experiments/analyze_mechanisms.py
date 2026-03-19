#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from experiments.checkpoint_utils import resolve_checkpoint_path, resolve_reused_checkpoint
from experiments.result_utils import resolve_best_beta_reference
IMAGE_ROOT = ROOT / 'examples' / 'image'
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))

from models.model_configs import instantiate_model
from training.analysis_utils import collect_loss_and_velocity_profile, collect_trajectory_profile
from training.data_transform import get_train_transform
from training.eval_loop import CFGScaledModel
from experiments.plot_style import selected_nfe_ticks, transform_focus_axis_values


def merge_dicts(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    merged.update(override)
    return merged


def resolve_dynamic_spec_fields(spec: Dict) -> Dict:
    resolved = dict(spec)
    best_beta_from = resolved.get('best_beta_from')
    if best_beta_from is not None:
        resolved['clock_beta'] = resolve_best_beta_reference(
            reference=best_beta_from,
            workspace_root=ROOT,
        )
    return resolved


def resolve_analysis_checkpoint(base_dir: Path, spec: Dict):
    checkpoint_from = spec.get('checkpoint_from')
    if checkpoint_from is not None:
        reused_checkpoint = resolve_reused_checkpoint(
            reference=checkpoint_from,
            spec=spec,
            workspace_root=ROOT,
        )
        if reused_checkpoint is not None:
            return reused_checkpoint
    return resolve_checkpoint_path(
        base_dir=base_dir,
        spec=spec,
        workspace_root=ROOT,
    )


def build_dataset(spec: Dict):
    transform = get_train_transform()
    if spec['dataset'] == 'cifar10':
        return datasets.CIFAR10(root=spec['data_path'], train=True, download=True, transform=transform)
    if spec['dataset'] == 'cifar100':
        return datasets.CIFAR100(root=spec['data_path'], train=True, download=True, transform=transform)
    raise NotImplementedError(f"Unsupported dataset {spec['dataset']}")


def load_model_from_checkpoint(spec: Dict, checkpoint_path: Path, device: torch.device):
    model = instantiate_model(
        architechture=spec['dataset'],
        is_discrete=spec.get('discrete_flow_matching', False),
        use_ema=spec.get('use_ema', False),
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_profile(rows, output_path: Path, y_field: str, ylabel: str):
    if not rows:
        return
    x = [0.5 * (row['r_left'] + row['r_right']) for row in rows]
    y = [row[y_field] for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker='o', color='#d62728', linewidth=2.1, markersize=4)
    plt.xlabel('r')
    plt.ylabel(ylabel)
    plt.grid(alpha=0.22, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_trajectory(rows, output_path: Path):
    if not rows:
        return
    original_x = [int(row['nfe']) for row in rows]
    x = transform_focus_axis_values(original_x)
    y = [row['terminal_budget_ratio'] for row in rows]
    tick_values = selected_nfe_ticks(original_x)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker='o', color='#1f77b4', linestyle='--', linewidth=2.0, markersize=4)
    plt.xlabel('NFE')
    plt.ylabel('Terminal Budget Ratio')
    plt.xticks(transform_focus_axis_values(tick_values), [str(tick) for tick in tick_values])
    plt.grid(alpha=0.22, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(config_path: Path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    base_config = config.get('base_config', {})
    experiment_name = config.get('experiment_name', config_path.stem)
    base_dir = ROOT / 'experiments' / 'results' / experiment_name
    device = torch.device(base_config.get('device', 'cuda'))

    for experiment in config.get('experiments', []):
        spec = resolve_dynamic_spec_fields(merge_dicts(base_config, experiment))
        checkpoint_path = resolve_analysis_checkpoint(base_dir=base_dir, spec=spec)
        if checkpoint_path is None:
            print(f'Skipping {spec["name"]}: missing checkpoint {checkpoint_path}')
            continue

        dataset = build_dataset(spec)
        batch_size = int(spec.get('batch_size', 64))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=spec.get('num_workers', 2),
            drop_last=True,
        )
        model = load_model_from_checkpoint(spec, checkpoint_path, device)
        args = SimpleNamespace(
            path_family=spec.get('path_family', 'linear'),
            clock_family=spec.get('clock_family', 'uniform'),
            clock_beta=spec.get('clock_beta'),
            analysis_num_bins=spec.get('analysis_num_bins', 20),
            analysis_num_batches=spec.get('analysis_num_batches', 8),
            analysis_num_samples=spec.get('analysis_num_samples', 512),
            eval_nfes=spec.get('eval_nfes', [10, 20, 50]),
            sampling_solver=spec.get('sampling_solver', 'heun2'),
            cfg_scale=spec.get('cfg_scale', 0.0),
        )
        output_dir = base_dir / spec['dataset'] / spec['name'] / 'analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        loss_rows = collect_loss_and_velocity_profile(model, loader, device, args)
        write_csv(output_dir / 'loss_velocity_profile.csv', loss_rows)
        plot_profile(loss_rows, output_dir / 'target_velocity_norm_vs_r.png', 'mean_target_velocity_norm', 'Mean ||target velocity||')
        plot_profile(loss_rows, output_dir / 'loss_density_vs_r.png', 'mean_loss', 'Mean Loss')

        labels = torch.zeros(args.analysis_num_samples, dtype=torch.long, device=device)
        image_size = int(spec.get('image_size', 32))
        sample_shape = (args.analysis_num_samples, 3, image_size, image_size)
        trajectory_rows = collect_trajectory_profile(
            CFGScaledModel(model),
            device,
            args,
            sample_shape=sample_shape,
            labels=labels,
        )
        write_csv(output_dir / 'trajectory_profile.csv', trajectory_rows)
        plot_trajectory(trajectory_rows, output_dir / 'terminal_budget_ratio_vs_nfe.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    main(args.config)
