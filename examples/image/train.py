# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.continuous_runtime import (
    estimate_signal_scale_sq_from_dataset,
    infer_strategy_id,
    resolve_clock_semantics_tag,
    resolve_curriculum_signature,
    validate_strategy_configuration,
)
from training.data_transform import get_eval_transform, get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def build_dataset(args, transform):
    if args.dataset == "imagenet":
        return datasets.ImageFolder(args.data_path, transform=transform)
    if args.dataset == "cifar10":
        return datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform,
        )
    if args.dataset == "cifar100":
        return datasets.CIFAR100(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform,
        )
    raise NotImplementedError(f"Unsupported dataset {args.dataset}")


def should_run_eval(args, epoch: int) -> bool:
    return (
        args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0
    ) or args.eval_only or args.test_run


def should_save_checkpoint(args, epoch: int) -> bool:
    if not args.output_dir or args.eval_only:
        return False
    return (
        args.test_run
        or (epoch + 1) == args.epochs
        or (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
    )


def _find_checkpoint_from_directory(
    exp_dir: Path,
    epoch: Optional[int],
) -> Optional[Path]:
    if epoch is not None:
        candidates = [
            exp_dir / f"checkpoint-{epoch}.pth",
            exp_dir / f"checkpoint{epoch}.pth",
            exp_dir / f"checkpoint{epoch:04d}.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    latest = exp_dir / "checkpoint.pth"
    if latest.exists():
        return latest
    return None


def _resolve_solver_aware_checkpoint_from_experiment(
    reference: str,
    dataset: str,
    epoch: int,
) -> Optional[Path]:
    tokens = [token.strip() for token in str(reference).split(":") if token.strip()]
    if len(tokens) == 2:
        artifact_group, exp_name = tokens
        resolved_dataset = dataset
    elif len(tokens) == 3:
        artifact_group, resolved_dataset, exp_name = tokens
    else:
        raise ValueError(
            "solver_aware_checkpoint_from_experiment must have the form "
            "'artifact_group:exp_name' or 'artifact_group:dataset:exp_name'."
        )
    exp_dir = WORKSPACE_ROOT / "experiments" / "results" / artifact_group / resolved_dataset / exp_name
    resolved_epoch = None if epoch < 0 else int(epoch)
    return _find_checkpoint_from_directory(exp_dir=exp_dir, epoch=resolved_epoch)


def _resolve_solver_aware_checkpoint(args) -> Optional[Path]:
    explicit_path = str(getattr(args, "solver_aware_checkpoint_path", "") or "").strip()
    experiment_reference = str(
        getattr(args, "solver_aware_checkpoint_from_experiment", "") or ""
    ).strip()
    checkpoint_epoch = int(getattr(args, "solver_aware_checkpoint_epoch", -1))

    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        return checkpoint_path if checkpoint_path.exists() else None
    if experiment_reference:
        return _resolve_solver_aware_checkpoint_from_experiment(
            reference=experiment_reference,
            dataset=str(args.dataset),
            epoch=checkpoint_epoch,
        )
    return None


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    transform_eval = get_eval_transform()
    dataset_train = build_dataset(args=args, transform=transform_train)
    dataset_eval = build_dataset(args=args, transform=transform_eval)
    logger.info(dataset_train)
    args.signal_scale_sq = estimate_signal_scale_sq_from_dataset(dataset_train)
    args.clock_semantics_tag = resolve_clock_semantics_tag(
        path_family=args.path_family,
        clock_family=args.clock_family,
        signal_scale_sq=args.signal_scale_sq,
    )
    (
        args.model_output_type,
        args.time_sampling_strategy,
    ) = validate_strategy_configuration(
        model_output_type=args.model_output_type,
        time_sampling_strategy=args.time_sampling_strategy,
    )
    args.strategy_id = infer_strategy_id(
        model_output_type=args.model_output_type,
        time_sampling_strategy=args.time_sampling_strategy,
    )
    args.curriculum_signature = resolve_curriculum_signature(
        time_sampling_strategy=args.time_sampling_strategy,
    )
    logger.info(f"Estimated signal_scale_sq={args.signal_scale_sq:.6f}")

    logger.info("Initializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_eval = torch.utils.data.DistributedSampler(
        dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    logger.info(str(sampler_train))
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )
    model.to(device)
    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    if (
        args.eval_only
        and getattr(args, "solver_aware_clock_mode", "off") != "off"
    ):
        resolved_solver_aware_checkpoint = _resolve_solver_aware_checkpoint(args)
        if args.resume and resolved_solver_aware_checkpoint is not None:
            resume_path = Path(str(args.resume)).expanduser().resolve()
            if resume_path != resolved_solver_aware_checkpoint.resolve():
                raise ValueError(
                    "solver-aware eval received two different checkpoint sources: "
                    f"--resume={resume_path} and "
                    f"--solver_aware checkpoint={resolved_solver_aware_checkpoint}."
                )
        elif not args.resume and resolved_solver_aware_checkpoint is not None:
            args.resume = str(resolved_solver_aware_checkpoint)
        args.solver_aware_monitor_source_checkpoint = str(
            resolved_solver_aware_checkpoint or args.resume or ""
        )
        if getattr(args, "solver_aware_use_nodes", False) and not args.resume:
            raise ValueError(
                "solver-aware training-free evaluation requires a checkpoint. "
                "Provide --resume, --solver_aware_checkpoint_path, or "
                "--solver_aware_checkpoint_from_experiment."
            )

    loss_scaler = NativeScaler()
    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    if args.eval_only:
        epoch_iterator = [int(args.start_epoch)]
        logger.info(
            "Eval-only mode at checkpoint epoch %s.",
            args.start_epoch,
        )
    else:
        epoch_iterator = range(args.start_epoch, args.epochs)
        logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in epoch_iterator:
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {"epoch": epoch}

        if should_save_checkpoint(args, epoch):
            save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        if args.output_dir and should_run_eval(args, epoch):
            if args.distributed:
                data_loader_eval.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                data_loader_eval,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
