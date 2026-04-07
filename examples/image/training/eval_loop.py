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
from typing import Dict, Iterable, Optional

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
    TIME_EPS,
    evaluate_clock,
    model_output_to_velocity,
)
from training.eval_utils import iter_batches_until_target
from training.fixed_step_solver import ReparameterizedSchedule, build_step_methods, solve_fixed_budget
from training.ge_stork import (
    build_or_load_shared_clock,
    get_time_grid_for_nfe,
    save_shared_clock_schedule,
)
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
SOLVER_STATS_DICT_KEYS = {"mode_histogram", "dyadic_step_histogram"}
SOLVER_STATS_BOOL_KEYS = {
    "used_tail_step",
    "is_exact_budget",
    "is_shared_budget",
}


def _autocast_context(device: torch.device, enabled: bool):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


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

    def adapt_solver_time(
        self,
        t: torch.Tensor,
        step_size: float,
        step_count: Optional[int] = None,
    ) -> torch.Tensor:
        adapted = t.to(dtype=torch.float32)
        if self.model_output_type == "velocity" or self.clock_family == "uniform":
            return adapted

        resolved_step_count = max(1, int(step_count if step_count is not None else 0))
        sample_eps = min(0.5 - TIME_EPS, max(TIME_EPS, 1.0 / float(resolved_step_count)))
        return adapted.clamp(min=sample_eps, max=1.0 - sample_eps)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
        use_autocast: bool = True,
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

        autocast_enabled = bool(use_autocast) and bool(x.is_cuda)
        if cfg_scale != 0.0:
            with _autocast_context(device=x.device, enabled=autocast_enabled):
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            raw_result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
            self.nfe_counter += 2
        else:
            with _autocast_context(device=x.device, enabled=autocast_enabled):
                raw_result = self.model(x, t, extra={"label": label})
            self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(raw_result.to(dtype=torch.float32), dim=-1)

        result = raw_result.to(dtype=torch.float32)
        if self.model_output_type == "velocity":
            return result

        clock = evaluate_clock(
            r=t.to(dtype=torch.float32),
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


def _unwrap_eval_model(model: Module) -> Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _solver_step_count(solver_name: str, nfe_budget: int) -> int:
    if solver_name == "stork4":
        return int(nfe_budget)
    return len(build_step_methods(solver_name=solver_name, nfe_budget=nfe_budget))


def _build_monitor_loader(
    args: Namespace,
    data_loader: Iterable,
    *,
    batch_size: Optional[int] = None,
):
    dataset = getattr(data_loader, "dataset", None)
    resolved_batch_size = max(
        1,
        int(
            batch_size
            if batch_size is not None
            else args.batch_size
        ),
    )
    if dataset is None or not distributed_mode.is_dist_avail_and_initialized():
        return data_loader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(args, "pin_mem", True)),
        drop_last=False,
    )


def _broadcast_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not distributed_mode.is_dist_avail_and_initialized():
        return tensor.to(device=device)
    broadcast = tensor.to(device=device)
    torch.distributed.broadcast(broadcast, src=0)
    return broadcast


def _broadcast_optional_object(payload):
    if not distributed_mode.is_dist_avail_and_initialized():
        return payload
    objects = [payload if distributed_mode.is_main_process() else None]
    torch.distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def _update_solver_stats_accumulator(
    accumulator: Optional[Dict[str, object]],
    solver_stats: Optional[Dict[str, object]],
    weight: int,
) -> Optional[Dict[str, object]]:
    if solver_stats is None or weight <= 0:
        return accumulator
    if accumulator is None:
        accumulator = {
            "weight": 0.0,
            "scalar_sums": {},
            "dict_sums": {},
        }
    accumulator["weight"] = float(accumulator["weight"]) + float(weight)
    scalar_sums = dict(accumulator["scalar_sums"])
    dict_sums = {
        key: dict(value)
        for key, value in dict(accumulator["dict_sums"]).items()
    }
    for key, value in solver_stats.items():
        if key in SOLVER_STATS_DICT_KEYS and isinstance(value, dict):
            current = dict(dict_sums.get(key, {}))
            for sub_key, sub_value in value.items():
                current[str(sub_key)] = current.get(str(sub_key), 0.0) + float(sub_value)
            dict_sums[key] = current
            continue
        if isinstance(value, bool):
            scalar_sums[key] = scalar_sums.get(key, 0.0) + float(int(value)) * float(weight)
            continue
        if isinstance(value, (int, float)):
            scalar_sums[key] = scalar_sums.get(key, 0.0) + float(value) * float(weight)
    accumulator["scalar_sums"] = scalar_sums
    accumulator["dict_sums"] = dict_sums
    return accumulator


def _finalize_solver_stats(
    accumulator: Optional[Dict[str, object]],
    last_solver_stats: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if last_solver_stats is None:
        return None
    if accumulator is None or float(accumulator.get("weight", 0.0)) <= 0.0:
        return dict(last_solver_stats)
    result = dict(last_solver_stats)
    weight = float(accumulator["weight"])
    for key, value in dict(accumulator["scalar_sums"]).items():
        averaged = float(value) / weight
        if key in {
            "requested_nfe_budget",
            "requested_eval_nfe",
        }:
            result[key] = int(round(averaged))
        elif key in SOLVER_STATS_BOOL_KEYS:
            result[key] = bool(round(averaged))
        else:
            result[key] = averaged
    for key, value in dict(accumulator["dict_sums"]).items():
        result[key] = {
            sub_key: int(round(float(sub_value)))
            for sub_key, sub_value in value.items()
        }
    return result


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
    monitor_data_loader: Optional[Iterable] = None,
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
    clock_monitor_model = CFGScaledModel(
        model=_unwrap_eval_model(model),
        path_family=args.path_family,
        clock_family=args.clock_family,
        clock_beta=args.clock_beta,
        signal_scale_sq=getattr(args, "signal_scale_sq", None),
        model_output_type=getattr(args, "model_output_type", "velocity"),
    )
    clock_monitor_model.train(False)

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

    real_features = []
    fake_features = []
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    shared_clock_mode = str(getattr(args, "shared_clock_mode", "off"))
    shared_clock_schedule = None
    if not args.discrete_flow_matching and shared_clock_mode != "off":
        if monitor_data_loader is None:
            raise ValueError(
                "shared_clock_mode requires an explicit calibration/train loader. "
                "Refusing to build a shared clock from the current eval/test loader."
            )
        if str(getattr(args, "sampling_solver", "")) not in {"euler", "heun2", "stork4"}:
            raise ValueError(
                "shared_clock_mode currently supports sampling_solver in {euler, heun2, stork4}."
            )
        step_count = _solver_step_count(
            solver_name=args.sampling_solver,
            nfe_budget=args.eval_nfe,
        )
        schedule_payload = None
        if distributed_mode.is_main_process():
            logger.info(
                "Shared clock build will use the provided calibration/train loader; "
                "the current eval/test loader is not allowed for shared clock construction."
            )
            shared_clock_loader = _build_monitor_loader(
                args=args,
                data_loader=monitor_data_loader,
                batch_size=int(getattr(args, "shared_clock_pilot_batch_size", args.batch_size)),
            )
            with torch.enable_grad():
                shared_clock_profile = build_or_load_shared_clock(
                    clock_family=str(getattr(args, "shared_clock_family", "ab")),
                    velocity_model=clock_monitor_model,
                    data_loader=shared_clock_loader,
                    device=device,
                    path_family=args.path_family,
                    pilot_solver=str(getattr(args, "shared_clock_pilot_solver", "heun2")),
                    physical_grid_size=int(getattr(args, "shared_clock_physical_grid_size", 65)),
                    pilot_batch_size=int(getattr(args, "shared_clock_pilot_batch_size", 16)),
                    pilot_num_batches=int(getattr(args, "shared_clock_pilot_num_batches", 4)),
                    observation_microbatch=int(
                        getattr(args, "shared_clock_observation_microbatch", 4)
                    ),
                    cfg_scale=float(args.cfg_scale),
                    eps=float(getattr(args, "shared_clock_eps", 1.0e-6)),
                    jacobian_backend=str(getattr(args, "shared_clock_jacobian_backend", "probe")),
                    jacobian_num_probes=int(getattr(args, "shared_clock_jacobian_num_probes", 4)),
                    optimizer_steps=int(getattr(args, "shared_clock_optimizer_steps", 200)),
                    optimizer_lr=float(getattr(args, "shared_clock_optimizer_lr", 0.05)),
                    checkpoint_source=str(getattr(args, "resume", "") or ""),
                    seed=int(getattr(args, "seed", 0)),
                    cache_path=str(getattr(args, "shared_clock_cache_path", "none")),
                    output_dir=Path(args.output_dir) if args.output_dir else None,
                )
            schedule_bundle = get_time_grid_for_nfe(
                shared_clock_profile,
                int(args.eval_nfe),
                step_count=step_count,
                device=device,
                dtype=torch.float32,
            )
            shared_clock_schedule = schedule_bundle["schedule"]
            if args.output_dir:
                save_shared_clock_schedule(
                    clock=shared_clock_profile,
                    schedule=shared_clock_schedule,
                    output_dir=Path(args.output_dir),
                    solver_name=str(args.sampling_solver),
                )
            schedule_payload = {
                "tau_grid": shared_clock_schedule.tau_grid.detach().cpu(),
                "t_grid": shared_clock_schedule.t_grid.detach().cpu(),
                "g_grid": shared_clock_schedule.g_grid.detach().cpu(),
                "dtau": float(shared_clock_schedule.dtau),
                "nfe_budget": int(shared_clock_schedule.nfe_budget or 0),
                "step_count": int(shared_clock_schedule.step_count or 0),
            }
        schedule_payload = _broadcast_optional_object(schedule_payload)
        if schedule_payload is not None and not distributed_mode.is_main_process():
            shared_clock_schedule = ReparameterizedSchedule(
                tau_grid=schedule_payload["tau_grid"].to(device=device, dtype=torch.float32),
                t_grid=schedule_payload["t_grid"].to(device=device, dtype=torch.float32),
                g_grid=schedule_payload["g_grid"].to(device=device, dtype=torch.float32),
                dtau=float(schedule_payload["dtau"]),
                nfe_budget=int(schedule_payload["nfe_budget"]),
                step_count=int(schedule_payload["step_count"]),
            )

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

    num_synthetic = 0
    num_real = 0
    snapshots_saved = False
    last_nfe = 0
    last_step_count = 0
    last_solver_stats = None
    solver_stats_accumulator = None

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
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
        else:
            x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)
            solve_kwargs = {
                "label": labels,
                "cfg_scale": args.cfg_scale,
            }
            sampling = solve_fixed_budget(
                velocity_model=cfg_scaled_model,
                x_init=x_0,
                solver_name=args.sampling_solver,
                nfe_budget=args.eval_nfe,
                return_trajectory=False,
                time_grid=None,
                reparameterized_schedule=shared_clock_schedule,
                **solve_kwargs,
            )
            synthetic_samples = sampling.sample
            last_nfe = sampling.nfe
            last_step_count = sampling.step_count
            last_solver_stats = getattr(sampling, "solver_stats", None)
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
        logger.info(
            f"{samples.shape[0]} samples generated in {last_nfe} actual evaluations under raw_nfe_budget={args.eval_nfe} and {last_step_count} macro-steps."
        )
        if num_synthetic + synthetic_samples.shape[0] > fid_samples:
            synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
        solver_stats_accumulator = _update_solver_stats_accumulator(
            solver_stats_accumulator,
            last_solver_stats,
            weight=int(synthetic_samples.shape[0]),
        )

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

    finalized_solver_stats = _finalize_solver_stats(
        solver_stats_accumulator,
        last_solver_stats,
    )
    if args.output_dir and finalized_solver_stats is not None:
        solver_stats_path = Path(args.output_dir) / "solver_stats.json"
        with open(solver_stats_path, "w", encoding="utf-8") as handle:
            json.dump(finalized_solver_stats, handle, indent=2, sort_keys=True)

    realized_nfe = float(
        finalized_solver_stats.get("realized_nfe")
        if finalized_solver_stats is not None and finalized_solver_stats.get("realized_nfe") is not None
        else last_nfe
    )
    summarized_step_count = float(
        finalized_solver_stats.get("step_count")
        if finalized_solver_stats is not None and finalized_solver_stats.get("step_count") is not None
        else last_step_count
    )

    results = {
        "nfe": float(args.eval_nfe),
        "requested_nfe_budget": float(args.eval_nfe),
        "requested_eval_nfe": float(args.eval_nfe),
        "realized_nfe": realized_nfe,
        "step_count": summarized_step_count,
        "real_samples": float(num_real),
        "synthetic_samples": float(num_synthetic),
    }
    if finalized_solver_stats is not None:
        for key, value in finalized_solver_stats.items():
            if isinstance(value, (int, float)) and key not in results:
                results[key] = float(value)
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
