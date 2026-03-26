# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from pathlib import Path

import torch
from training.distributed_mode import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, epoch, model, model_without_ddp, optimizer, lr_schedule, loss_scaler
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % epoch_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def _checkpoint_args_to_dict(payload):
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, Namespace):
        return vars(payload).copy()
    if hasattr(payload, "__dict__"):
        return dict(vars(payload))
    return {}


def _normalize_checkpoint_clock_family(clock_family):
    if clock_family in {"ft_linear_beta", "ft_vp_beta"}:
        return "ft_beta"
    return clock_family


def _float_matches(expected, observed, tol: float = 1e-8):
    if expected is None:
        return observed is None
    if observed is None:
        return False
    try:
        return abs(float(expected) - float(observed)) <= tol
    except (TypeError, ValueError):
        return False


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        requested_path_family = getattr(args, "path_family", None)
        requested_clock_family = getattr(args, "clock_family", None)
        requested_clock_beta = getattr(args, "clock_beta", None)
        checkpoint_args = _checkpoint_args_to_dict(checkpoint.get("args"))
        checkpoint_model_output_type = str(
            checkpoint_args.get("model_output_type", "velocity")
        )
        checkpoint_time_sampling = checkpoint_args.get("time_sampling_strategy")
        is_eval_only = bool(hasattr(args, "eval") and args.eval) or bool(
            getattr(args, "eval_only", False)
        )
        if checkpoint_args.get("signal_scale_sq") is not None:
            args.signal_scale_sq = float(checkpoint_args["signal_scale_sq"])
        if checkpoint_args.get("clock_semantics_tag") is not None:
            args.clock_semantics_tag = checkpoint_args["clock_semantics_tag"]
        checkpoint_clock_beta = checkpoint_args.get("clock_beta")
        checkpoint_clock_family = None
        if checkpoint_args.get("clock_family") is not None:
            checkpoint_clock_family = _normalize_checkpoint_clock_family(
                checkpoint_args["clock_family"]
            )
        checkpoint_path_family = checkpoint_args.get("path_family")

        if not is_eval_only:
            if (
                checkpoint_path_family is not None
                and requested_path_family is not None
                and checkpoint_path_family != requested_path_family
            ):
                raise ValueError(
                    "Refusing to resume training from a checkpoint with a different "
                    f"path_family. expected={requested_path_family}, "
                    f"checkpoint={checkpoint_path_family}"
                )
            if (
                checkpoint_clock_family is not None
                and requested_clock_family is not None
                and checkpoint_clock_family != requested_clock_family
            ):
                raise ValueError(
                    "Refusing to resume training from a checkpoint with a different "
                    f"clock_family. expected={requested_clock_family}, "
                    f"checkpoint={checkpoint_clock_family}"
                )
            if checkpoint_clock_beta is not None and not _float_matches(
                requested_clock_beta,
                checkpoint_clock_beta,
            ):
                raise ValueError(
                    "Refusing to resume training from a checkpoint with a different "
                    f"clock_beta. expected={requested_clock_beta}, "
                    f"checkpoint={checkpoint_clock_beta}"
                )

        if checkpoint_clock_beta is not None:
            args.clock_beta = checkpoint_clock_beta
        if checkpoint_clock_family is not None:
            args.clock_family = checkpoint_clock_family
        if checkpoint_path_family is not None:
            args.path_family = checkpoint_path_family

        if not is_eval_only and args.clock_family != "uniform":
            if (
                checkpoint_model_output_type != "base_velocity"
                or checkpoint_time_sampling != "ds_dr_sq"
            ):
                raise ValueError(
                    "Refusing to continue non-uniform-clock training from a legacy "
                    "checkpoint with velocity-output semantics. Re-run the experiment "
                    "from scratch under the new u-parameterized objective."
                )

        args.model_output_type = checkpoint_model_output_type
        if checkpoint_time_sampling is not None:
            args.time_sampling_strategy = checkpoint_time_sampling

        print("Resume checkpoint %s" % args.resume)
        if "epoch" in checkpoint and is_eval_only:
            args.start_epoch = checkpoint["epoch"]
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not is_eval_only
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
