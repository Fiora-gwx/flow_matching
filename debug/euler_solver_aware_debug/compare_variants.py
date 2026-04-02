from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from monitor_debug import (
    DEFAULT_NFE_LIST,
    MonitorDebugBundle,
    NodeDiagnostics,
    VariantProfile,
    _csv_dump,
    _json_dump,
    _sanitize_name,
    _tensor_to_list,
    bootstrap_repo_paths,
)


def _feature_stats_to_cpu(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "count": int(payload["count"]),
        "feature_dim": int(payload["feature_dim"]),
        "sum": payload["sum"].detach().cpu(),
        "sum_outer": payload["sum_outer"].detach().cpu(),
    }


def _feature_stats_to_device(torch_mod, payload: Mapping[str, Any], device) -> Dict[str, Any]:
    return {
        "count": int(payload["count"]),
        "feature_dim": int(payload["feature_dim"]),
        "sum": payload["sum"].to(device=device, dtype=torch_mod.float64),
        "sum_outer": payload["sum_outer"].to(device=device, dtype=torch_mod.float64),
    }


def _update_feature_stats(torch_mod, stats: Optional[Dict[str, Any]], features) -> Dict[str, Any]:
    features = features.to(dtype=torch_mod.float64)
    if stats is None:
        feature_dim = int(features.shape[1])
        stats = {
            "count": 0,
            "feature_dim": feature_dim,
            "sum": torch_mod.zeros(feature_dim, device=features.device, dtype=torch_mod.float64),
            "sum_outer": torch_mod.zeros(
                feature_dim,
                feature_dim,
                device=features.device,
                dtype=torch_mod.float64,
            ),
        }
    stats["count"] += int(features.shape[0])
    stats["sum"] = stats["sum"] + features.sum(dim=0)
    stats["sum_outer"] = stats["sum_outer"] + features.transpose(0, 1) @ features
    return stats


def _finalize_moments(torch_mod, stats: Mapping[str, Any]):
    count = max(1, int(stats["count"]))
    mean = stats["sum"] / float(count)
    centered_outer = stats["sum_outer"] - float(count) * torch_mod.outer(mean, mean)
    denom = max(1, count - 1)
    covariance = centered_outer / float(denom)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return mean, covariance


def _matrix_sqrt_psd(torch_mod, matrix):
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch_mod.linalg.eigh(matrix)
    clipped = eigenvalues.clamp(min=0.0)
    return (eigenvectors * clipped.sqrt().unsqueeze(0)) @ eigenvectors.transpose(0, 1)


def _compute_fid_from_stats(torch_mod, real_stats: Mapping[str, Any], fake_stats: Mapping[str, Any]) -> float:
    real_mean, real_cov = _finalize_moments(torch_mod, real_stats)
    fake_mean, fake_cov = _finalize_moments(torch_mod, fake_stats)
    mean_diff = real_mean - fake_mean
    cov_sqrt = _matrix_sqrt_psd(torch_mod, real_cov)
    mixed = cov_sqrt @ fake_cov @ cov_sqrt
    trace_sqrt = torch_mod.trace(_matrix_sqrt_psd(torch_mod, mixed))
    fid = (
        mean_diff.dot(mean_diff)
        + torch_mod.trace(real_cov)
        + torch_mod.trace(fake_cov)
        - 2.0 * trace_sqrt
    )
    return float(fid.clamp(min=0.0).item())


def _build_feature_backend(device):
    _ = bootstrap_repo_paths()
    torchmetrics = None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        torchmetrics = FrechetInceptionDistance
    except ModuleNotFoundError as error:  # pragma: no cover - depends on runtime.
        raise RuntimeError(
            "Missing runtime dependency 'torchmetrics'. Install the project runtime environment "
            "before running Euler FID comparisons."
        ) from error
    return torchmetrics(normalize=True).to(device=device, non_blocking=True)


def compute_real_feature_stats(
    *,
    bundle: MonitorDebugBundle,
    fid_samples: int,
    output_root: Path,
) -> Dict[str, Any]:
    bootstrap_repo_paths()
    from training.eval_utils import iter_batches_until_target
    from training.metric_utils import extract_inception_features

    torch_mod = bundle.context.torch
    cache_path = output_root / "cache" / f"real_feature_stats_{int(fid_samples)}.pt"
    if cache_path.exists():
        loaded = torch_mod.load(cache_path, map_location=bundle.context.device)
        return _feature_stats_to_device(torch_mod, loaded, bundle.context.device)

    backend = _build_feature_backend(device=bundle.context.device)
    stats = None
    processed = 0
    for _, batch in iter_batches_until_target(
        data_loader=bundle.context.data_loader,
        target_samples=int(fid_samples),
        test_run=False,
    ):
        samples, _labels = batch
        samples = samples.to(bundle.context.device, non_blocking=True)
        if processed + int(samples.shape[0]) > int(fid_samples):
            samples = samples[: int(fid_samples) - processed]
        features = extract_inception_features(backend, samples)
        stats = _update_feature_stats(torch_mod, stats, features)
        processed += int(samples.shape[0])

    if stats is None:
        raise RuntimeError("Failed to accumulate real feature stats because the eval loader is empty.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(_feature_stats_to_cpu(stats), cache_path)
    return stats


def _evaluation_output_dir(output_root: Path, variant_name: str, nfe: int) -> Path:
    return output_root / "evaluations" / _sanitize_name(variant_name) / f"nfe_{int(nfe):03d}"


def _compute_fake_feature_stats(
    *,
    bundle: MonitorDebugBundle,
    variant_name: str,
    diagnostics: Optional[NodeDiagnostics],
    nfe: int,
    fid_samples: int,
    output_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bootstrap_repo_paths()
    from training.eval_utils import iter_batches_until_target
    from training.fixed_step_solver import solve_fixed_budget
    from training.metric_utils import extract_inception_features

    torch_mod = bundle.context.torch
    cache_path = _evaluation_output_dir(output_root, variant_name, nfe) / "fake_feature_stats.pt"
    if cache_path.exists():
        loaded = torch_mod.load(cache_path, map_location=bundle.context.device)
        fake_stats = _feature_stats_to_device(torch_mod, loaded["feature_stats"], bundle.context.device)
        return fake_stats, dict(loaded["eval_info"])

    backend = _build_feature_backend(device=bundle.context.device)
    generator = torch_mod.Generator(device=bundle.context.device)
    generator.manual_seed(int(bundle.context.seed) + int(nfe) * 997 + len(str(variant_name)))
    time_grid = None
    if diagnostics is not None:
        time_grid = diagnostics.nodes.to(device=bundle.context.device, dtype=torch_mod.float32)

    fake_stats = None
    actual_nfe = 0
    actual_step_count = 0
    generated = 0
    last_solver_stats = None
    for _, batch in iter_batches_until_target(
        data_loader=bundle.context.data_loader,
        target_samples=int(fid_samples),
        test_run=False,
    ):
        samples, labels = batch
        samples = samples.to(bundle.context.device, non_blocking=True)
        labels = labels.to(bundle.context.device, non_blocking=True)
        x_init = torch_mod.randn(
            samples.shape,
            device=bundle.context.device,
            dtype=torch_mod.float32,
            generator=generator,
        )
        sampling = solve_fixed_budget(
            velocity_model=bundle.context.velocity_model,
            x_init=x_init,
            solver_name="euler",
            nfe_budget=int(nfe),
            return_trajectory=False,
            time_grid=time_grid,
            label=labels,
            cfg_scale=bundle.context.cfg_scale,
        )
        synthetic = (sampling.sample * 0.5 + 0.5).clamp(0.0, 1.0)
        if generated + int(synthetic.shape[0]) > int(fid_samples):
            synthetic = synthetic[: int(fid_samples) - generated]
        features = extract_inception_features(backend, synthetic)
        fake_stats = _update_feature_stats(torch_mod, fake_stats, features)
        generated += int(synthetic.shape[0])
        actual_nfe = int(sampling.nfe)
        actual_step_count = int(sampling.step_count)
        last_solver_stats = sampling.solver_stats

    if fake_stats is None:
        raise RuntimeError(f"Failed to accumulate fake feature stats for {variant_name} at NFE={nfe}.")
    eval_info = {
        "actual_nfe": actual_nfe,
        "actual_step_count": actual_step_count,
        "generated_samples": generated,
        "solver_stats": last_solver_stats or {},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(
        {
            "feature_stats": _feature_stats_to_cpu(fake_stats),
            "eval_info": eval_info,
        },
        cache_path,
    )
    _json_dump(cache_path.parent / "eval_info.json", eval_info)
    return fake_stats, eval_info


def _uniform_diagnostics(torch_mod, nfe: int) -> NodeDiagnostics:
    nodes = torch_mod.linspace(0.0, 1.0, int(nfe) + 1, dtype=torch_mod.float32)
    step_sizes = nodes[1:] - nodes[:-1]
    summary = (
        f"Uniform Euler nodes keep every step fixed at 1/{int(nfe)}，"
        "作为 solver-aware 变体的对照组。"
    )
    return NodeDiagnostics(
        variant_name="uniform_baseline",
        nfe=int(nfe),
        step_count=int(nfe),
        smoothing_mode="uniform",
        clipping_mode="none",
        lambda_mix=0.0,
        monitor_grid_size=0,
        monitor_batch_size=0,
        constraint_name="uniform",
        uniform_nodes=nodes.detach().cpu(),
        nodes_unconstrained=nodes.detach().cpu(),
        nodes=nodes.detach().cpu(),
        r_grid=nodes.detach().cpu(),
        step_sizes=step_sizes.detach().cpu(),
        qe_non_negative=True,
        q_spike_ratio_max_over_p95=1.0,
        phi_strictly_monotone=True,
        psi_roundtrip_max_abs_error=0.0,
        nodes_strictly_increasing=True,
        step_sizes_positive=True,
        nodes_in_unit_interval=True,
        step_count_matches_requested=True,
        max_step=float(step_sizes.max().item()),
        min_positive_step=float(step_sizes.min().item()),
        max_step_over_uniform=1.0,
        max_step_over_min_positive=1.0,
        q_peak_interval=(0.0, 0.0),
        density_peak_interval=(0.0, 0.0),
        min_step_interval=(0.0, 1.0 / float(max(1, int(nfe)))),
        max_step_interval=(0.0, 1.0 / float(max(1, int(nfe)))),
        summary_sentence=summary,
    )


def _result_row(
    *,
    bundle: MonitorDebugBundle,
    variant_name: str,
    profile: Optional[VariantProfile],
    diagnostics: NodeDiagnostics,
    fid_value: float,
    eval_info: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint = bundle.context.checkpoint
    clipping_mode = diagnostics.clipping_mode
    if variant_name == "uniform_baseline":
        smoothing_mode = "uniform"
        lambda_mix = 0.0
        monitor_grid_size = 0
        monitor_batch_size = 0
    else:
        assert profile is not None
        smoothing_mode = profile.smoothing_mode
        lambda_mix = profile.lambda_mix
        monitor_grid_size = profile.monitor_grid_size
        monitor_batch_size = profile.monitor_batch_size
    return {
        "checkpoint_path": checkpoint.checkpoint_path,
        "artifact_group": checkpoint.artifact_group,
        "source_exp_name": checkpoint.source_exp_name,
        "checkpoint_epoch": checkpoint.checkpoint_epoch,
        "nfe": diagnostics.nfe,
        "variant_name": variant_name,
        "smoothing_mode": smoothing_mode,
        "clipping_mode": clipping_mode,
        "lambda_mix": lambda_mix,
        "monitor_grid_size": monitor_grid_size,
        "batch_size": monitor_batch_size,
        "constraint_name": diagnostics.constraint_name,
        "max_step": diagnostics.max_step,
        "min_positive_step": diagnostics.min_positive_step,
        "max_step_over_uniform": diagnostics.max_step_over_uniform,
        "max_step_over_min_positive": diagnostics.max_step_over_min_positive,
        "fid": fid_value,
        "actual_nfe": int(eval_info.get("actual_nfe", diagnostics.nfe)),
        "actual_step_count": int(eval_info.get("actual_step_count", diagnostics.step_count)),
        "generated_samples": int(eval_info.get("generated_samples", 0)),
        "step_count_matches_requested": diagnostics.step_count_matches_requested,
        "psi_roundtrip_max_abs_error": diagnostics.psi_roundtrip_max_abs_error,
        "summary_sentence": diagnostics.summary_sentence,
    }


def _variant_sequence(bundle: MonitorDebugBundle, config: Mapping[str, Any]) -> List[Tuple[str, Optional[VariantProfile]]]:
    variants: List[Tuple[str, Optional[VariantProfile]]] = [("uniform_baseline", None)]
    for name in (
        "solver_aware_no_smoothing",
        "solver_aware_current_impl",
        "solver_aware_gaussian",
    ):
        if name in bundle.profiles:
            variants.append((name, bundle.profiles[name]))
    for name in sorted(bundle.profiles):
        if name.startswith("solver_aware_qclip_"):
            variants.append((name, bundle.profiles[name]))
    for name in sorted(bundle.profiles):
        if name.startswith("solver_aware_densitycap_"):
            variants.append((name, bundle.profiles[name]))
    for name in sorted(bundle.profiles):
        if name.startswith("solver_aware_mix_lambda_"):
            variants.append((name, bundle.profiles[name]))
    for name in sorted(bundle.profiles):
        if name.startswith("solver_aware_constraint_"):
            variants.append((name, bundle.profiles[name]))
    if bool(dict(config.get("execution", {})).get("run_grid_size_sweep_eval", True)):
        for name in sorted(bundle.profiles):
            if name.startswith("solver_aware_grid_"):
                variants.append((name, bundle.profiles[name]))
    return variants


def compare_euler_variants(
    *,
    bundle: MonitorDebugBundle,
    config: Mapping[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    torch_mod = bundle.context.torch
    execution_cfg = dict(config.get("execution", {}))
    nfe_list = [int(value) for value in config.get("nfe_list", DEFAULT_NFE_LIST)]
    results: List[Dict[str, Any]] = []
    run_sampling_eval = bool(execution_cfg.get("run_sampling_eval", True))
    fid_samples = int(config.get("fid_samples", 0))
    real_stats = None
    if run_sampling_eval:
        real_stats = compute_real_feature_stats(
            bundle=bundle,
            fid_samples=fid_samples,
            output_root=output_root,
        )

    for variant_name, profile in _variant_sequence(bundle, config):
        for nfe in nfe_list:
            if variant_name == "uniform_baseline":
                diagnostics = _uniform_diagnostics(torch_mod, nfe=int(nfe))
            else:
                diagnostics = bundle.node_diagnostics[variant_name][int(nfe)]

            if run_sampling_eval:
                fake_stats, eval_info = _compute_fake_feature_stats(
                    bundle=bundle,
                    variant_name=variant_name,
                    diagnostics=None if variant_name == "uniform_baseline" else diagnostics,
                    nfe=int(nfe),
                    fid_samples=fid_samples,
                    output_root=output_root,
                )
                fid_value = _compute_fid_from_stats(torch_mod, real_stats, fake_stats)
            else:
                eval_info = {
                    "actual_nfe": diagnostics.nfe,
                    "actual_step_count": diagnostics.step_count,
                    "generated_samples": 0,
                }
                fid_value = float("nan")

            row = _result_row(
                bundle=bundle,
                variant_name=variant_name,
                profile=profile,
                diagnostics=diagnostics,
                fid_value=fid_value,
                eval_info=eval_info,
            )
            results.append(row)
            _json_dump(
                _evaluation_output_dir(output_root, variant_name, nfe) / "result.json",
                row,
            )

    _csv_dump(
        output_root / "euler_debug_results.csv",
        tuple(results[0].keys()) if results else (),
        results,
    )
    return results
