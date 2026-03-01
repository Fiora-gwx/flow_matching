# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        if args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.torch.rand(samples.shape[0]).to(device)

            # sample probability path
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            # discrete flow matching loss
            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            
            # 采样时间 t (即公式中的 gamma)
            # 你的理论建议使用均匀分布 U(0,1)，因此这里不要开启 --skewed_timesteps
            if args.skewed_timesteps:
                 t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.rand(samples.shape[0], device=device)

            # 采样路径 x_t
            # CondOTProbPath 默认就是线性插值: x_t = (1-t)*x_0 + t*x_1
            # path.sample 返回的 dx_t 默认是 (x_1 - x_0)，即 (x - epsilon)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t # 原始速度场 (x - epsilon)

            # FT-EqM 逻辑
            if args.use_ft_eqm:
                lamb = args.lambda_scale if args.lambda_scale is not None else (2.0 * args.alpha)
                gamma_term = (1 - t).clamp(min=1e-6) 
                c_gamma_raw = lamb * gamma_term.pow(2 * args.alpha - 1) # 形状: [Batch]
                
                # 为了和图像广播，调整形状
                view_shape = [t.shape[0]] + [1] * (samples.dim() - 1)
                c_gamma = c_gamma_raw.view(view_shape)
                
                # 2. 计算目标速度
                target_v = u_t * c_gamma
                
                # 3. === 核心：计算方差归一化权重 (Min-SNR 策略) ===
                # w = min( 5.0, 1 / c(gamma)^2 )
                # 阈值 5.0 是业界常用的超参数，可以根据训练情况微调 (e.g., 5.0 ~ 10.0)
                max_weight = 5.0 
                loss_weight = torch.clamp(1.0 / (c_gamma_raw ** 2 + 1e-6), max=max_weight)
                
                with torch.cuda.amp.autocast():
                    v_pred = model(x_t, t, extra=conditioning)
                    
                    # 4. 计算加权的 MSE
                    # 先计算每个样本的均方误差 (在空间和通道维度上平均)
                    raw_loss = torch.pow(v_pred - target_v, 2).mean(dim=[1, 2, 3]) # 形状: [Batch]
                    
                    # 乘以权重后求 Batch 的平均
                    loss = (raw_loss * loss_weight).mean()
            else:
                # 原有的 Flow Matching Loss
                with torch.cuda.amp.autocast():
                    loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
