from typing import Dict, List

import torch

from training.continuous_runtime import build_continuous_batch
from training.fixed_step_solver import solve_fixed_budget


@torch.no_grad()
def collect_loss_and_velocity_profile(model, data_loader, device, args) -> List[Dict[str, float]]:
    num_bins = int(args.analysis_num_bins)
    batches = int(args.analysis_num_batches)
    bin_counts = torch.zeros(num_bins, dtype=torch.float64)
    loss_sums = torch.zeros(num_bins, dtype=torch.float64)
    velocity_sums = torch.zeros(num_bins, dtype=torch.float64)

    for batch_index, (samples, labels) in enumerate(data_loader):
        if batch_index >= batches:
            break
        samples = samples.to(device, non_blocking=True) * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)
        noise = torch.randn_like(samples)
        r = torch.rand(samples.shape[0], device=device)
        batch = build_continuous_batch(
            x_1=samples,
            x_0=noise,
            r=r,
            path_family=args.path_family,
            clock_family=args.clock_family,
            clock_beta=args.clock_beta,
        )
        pred = model(batch.x_t, batch.r, extra={"label": labels})
        losses = torch.pow(pred - batch.target_velocity, 2).flatten(start_dim=1).mean(dim=1)
        velocity_norm = batch.target_velocity.flatten(start_dim=1).norm(dim=1)
        bins = torch.clamp((r * num_bins).long(), max=num_bins - 1)
        for bin_index in range(num_bins):
            mask = bins == bin_index
            if not mask.any():
                continue
            bin_counts[bin_index] += int(mask.sum())
            loss_sums[bin_index] += losses[mask].sum().cpu()
            velocity_sums[bin_index] += velocity_norm[mask].sum().cpu()

    rows = []
    for bin_index in range(num_bins):
        count = max(float(bin_counts[bin_index].item()), 1.0)
        rows.append(
            {
                "bin_index": bin_index,
                "r_left": bin_index / num_bins,
                "r_right": (bin_index + 1) / num_bins,
                "count": float(bin_counts[bin_index].item()),
                "mean_loss": float(loss_sums[bin_index].item() / count),
                "mean_target_velocity_norm": float(velocity_sums[bin_index].item() / count),
            }
        )
    return rows


@torch.no_grad()
def collect_trajectory_profile(model_wrapper, device, args, sample_shape, labels=None) -> List[Dict[str, float]]:
    rows = []
    labels = labels if labels is not None else torch.zeros(sample_shape[0], dtype=torch.long, device=device)
    for nfe in args.eval_nfes:
        x_init = torch.randn(sample_shape, dtype=torch.float32, device=device)
        sample = solve_fixed_budget(
            velocity_model=model_wrapper,
            x_init=x_init,
            solver_name=args.sampling_solver,
            nfe_budget=int(nfe),
            return_trajectory=True,
            label=labels,
            cfg_scale=args.cfg_scale,
        )
        if sample.deltas is None:
            continue
        step_norms = sample.deltas.flatten(start_dim=2).norm(dim=2).mean(dim=1).cpu()
        times = sample.time_grid[1:].cpu()
        terminal_mask = times > 0.8
        terminal_budget = float(step_norms[terminal_mask].sum().item()) if terminal_mask.any() else 0.0
        total_budget = float(step_norms.sum().item())
        rows.append(
            {
                "nfe": int(nfe),
                "actual_nfe": int(sample.nfe),
                "step_count": int(sample.step_count),
                "mean_step_length": float(step_norms.mean().item()),
                "terminal_budget": terminal_budget,
                "terminal_budget_ratio": terminal_budget / total_budget if total_budget > 0 else 0.0,
                "terminal_steps": int(terminal_mask.sum().item()),
            }
        )
    return rows
