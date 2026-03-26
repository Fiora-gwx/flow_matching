# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Optional

import PIL.Image
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image

from training import distributed_mode
from training.continuous_runtime import (
    clamp_time_inside_unit_interval,
    evaluate_clock,
    model_output_to_velocity,
)
from training.eval_utils import iter_batches_until_target
from training.fixed_step_solver import solve_fixed_budget
from training.metric_utils import (
    compute_precision_recall,
    extract_inception_features,
    prepare_inception_input,
    requested_metrics,
)
from training.train_loop import MASK_TOKEN

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:  # pragma: no cover - depends on runtime environment.
    FrechetInceptionDistance = None

try:
    from torchmetrics.image.inception import InceptionScore
except ImportError:  # pragma: no cover - depends on runtime environment.
    InceptionScore = None

logger = logging.getLogger(__name__)
PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(
        self,
        model: Module,
        path_family: str = "linear",
        clock_family: str = "uniform",
        clock_beta: Optional[float] = None,
        signal_scale_sq: Optional[float] = None,
        model_output_type: str = "velocity",
    ):
        super().__init__(model)
        self.nfe_counter = 0
        self.path_family = path_family
        self.clock_family = clock_family
        self.clock_beta = clock_beta
        self.signal_scale_sq = signal_scale_sq
        self.model_output_type = model_output_type

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            raw_result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
            self.nfe_counter += 2
        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                raw_result = self.model(x, t, extra={"label": label})
            self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(raw_result.to(dtype=torch.float32), dim=-1)

        result = raw_result.to(dtype=torch.float32)
        if self.model_output_type == "velocity":
            return result

        clock = evaluate_clock(
            r=clamp_time_inside_unit_interval(t.to(dtype=torch.float32)),
            clock_family=self.clock_family,
            clock_beta=self.clock_beta,
            path_family=self.path_family,
            signal_scale_sq=self.signal_scale_sq,
        )
        return model_output_to_velocity(
            model_output=result,
            ds_dr=clock.ds_dr,
            model_output_type=self.model_output_type,
        )

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def _build_fid_metric(device: torch.device):
    if FrechetInceptionDistance is None:
        raise RuntimeError(
            "torchmetrics[image] is required for evaluation metrics. Install project dependencies before running eval."
        )
    return FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)


def _build_inception_score_metric(device: torch.device, splits: int):
    if InceptionScore is None:
        raise RuntimeError(
            "torchmetrics[image] is required for evaluation metrics. Install project dependencies before running eval."
        )
    return InceptionScore(splits=max(1, splits), normalize=False).to(
        device=device, non_blocking=True
    )


@torch.no_grad()
def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    active_metrics = requested_metrics(args)

    cfg_scaled_model = CFGScaledModel(
        model=model,
        path_family=args.path_family,
        clock_family=args.clock_family,
        clock_beta=args.clock_beta,
        signal_scale_sq=getattr(args, "signal_scale_sq", None),
        model_output_type=getattr(args, "model_output_type", "velocity"),
    )
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        discrete_solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        discrete_solver = None

    metric_backend = None
    if "fid" in active_metrics or "precision_recall" in active_metrics:
        metric_backend = _build_fid_metric(device=device)
    inception_score_metric = None
    if "inception_score" in active_metrics:
        requested_splits = int(getattr(args, "inception_score_splits", 10))
        inception_score_metric = _build_inception_score_metric(
            device=device,
            splits=min(max(1, fid_samples), requested_splits),
        )

    real_features = []
    fake_features = []
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    num_synthetic = 0
    num_real = 0
    snapshots_saved = False
    last_nfe = 0
    last_step_count = 0
    last_solver_stats = None

    loader_length = len(data_loader) if hasattr(data_loader, "__len__") else "?"
    for data_iter_step, batch in iter_batches_until_target(
        data_loader=data_loader,
        target_samples=fid_samples,
        test_run=args.test_run,
    ):
        samples, labels = batch
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        num_real += samples.shape[0]

        if metric_backend is not None and "fid" in active_metrics:
            metric_backend.update(samples, real=True)
        if metric_backend is not None and "precision_recall" in active_metrics:
            real_features.append(extract_inception_features(metric_backend, samples).cpu())

        cfg_scaled_model.reset_nfe_counter()
        if args.discrete_flow_matching:
            x_0 = torch.full(samples.shape, fill_value=MASK_TOKEN, dtype=torch.long, device=device)
            if args.sym_func:
                sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
            else:
                sym = args.sym
            dtype = torch.float32 if args.sampling_dtype == "float32" else torch.float64
            synthetic_samples = discrete_solver.sample(
                x_init=x_0,
                step_size=1.0 / args.discrete_fm_steps,
                verbose=False,
                div_free=sym,
                dtype_categorical=dtype,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
            last_nfe = cfg_scaled_model.get_nfe()
            last_step_count = args.discrete_fm_steps
        else:
            x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)
            sampling = solve_fixed_budget(
                velocity_model=cfg_scaled_model,
                x_init=x_0,
                solver_name=args.sampling_solver,
                nfe_budget=args.eval_nfe,
                return_trajectory=False,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
            synthetic_samples = sampling.sample
            last_nfe = sampling.nfe
            last_step_count = sampling.step_count
            last_solver_stats = getattr(sampling, "solver_stats", None)
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)

        synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
        logger.info(
            f"{samples.shape[0]} samples generated in {last_nfe} evaluations and {last_step_count} steps."
        )
        if num_synthetic + synthetic_samples.shape[0] > fid_samples:
            synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]

        if metric_backend is not None and "fid" in active_metrics:
            metric_backend.update(synthetic_samples, real=False)
        if metric_backend is not None and "precision_recall" in active_metrics:
            fake_features.append(
                extract_inception_features(metric_backend, synthetic_samples).cpu()
            )
        if inception_score_metric is not None:
            inception_score_metric.update(prepare_inception_input(synthetic_samples))
        num_synthetic += synthetic_samples.shape[0]

        if not snapshots_saved and args.output_dir:
            save_image(
                synthetic_samples,
                fp=Path(args.output_dir) / "snapshots" / f"{epoch}_{data_iter_step}.png",
            )
            snapshots_saved = True

        if args.save_fid_samples and args.output_dir:
            images_np = (
                (synthetic_samples * 255.0)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for batch_index, image_np in enumerate(images_np):
                image_dir = Path(args.output_dir) / "fid_samples"
                os.makedirs(image_dir, exist_ok=True)
                image_path = (
                    image_dir
                    / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                )
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if data_iter_step % PRINT_FREQUENCY == 0 and metric_backend is not None and "fid" in active_metrics:
            gc.collect()
            running_fid = metric_backend.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{loader_length}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

    if args.output_dir and last_solver_stats is not None:
        solver_stats_path = Path(args.output_dir) / "solver_stats.json"
        with open(solver_stats_path, "w", encoding="utf-8") as handle:
            json.dump(last_solver_stats, handle, indent=2, sort_keys=True)

    results = {
        "nfe": float(last_nfe),
        "step_count": float(last_step_count),
        "real_samples": float(num_real),
        "synthetic_samples": float(num_synthetic),
    }
    if metric_backend is not None and "fid" in active_metrics:
        results["fid"] = float(metric_backend.compute().detach().cpu())
    if metric_backend is not None and "precision_recall" in active_metrics:
        pr_metrics = compute_precision_recall(
            real_features=torch.cat(real_features, dim=0).to(device=device),
            fake_features=torch.cat(fake_features, dim=0).to(device=device),
            neighbors=args.precision_recall_neighbors,
            max_samples=args.precision_recall_max_samples,
        )
        results.update(pr_metrics)
    if inception_score_metric is not None:
        is_mean, is_std = inception_score_metric.compute()
        results["is_mean"] = float(is_mean.detach().cpu())
        results["is_std"] = float(is_std.detach().cpu())
    return results
