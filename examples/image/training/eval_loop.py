# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from torchdiffeq import odeint
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def rho(gamma: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - gamma).clamp(min=1e-6).pow(2.0 * alpha - 1.0)


def radial_correction(
    v: torch.Tensor,
    z: torch.Tensor,
    x_hat: torch.Tensor,
    kappa: float,
    alpha: float,
    eps_safe: float = 1e-6,
) -> torch.Tensor:
    delta = z - x_hat
    norm_sq = (delta**2).flatten(start_dim=1).sum(dim=-1, keepdim=True)
    norm_2a = norm_sq.pow(alpha)
    radial_v = (delta * v).flatten(start_dim=1).sum(dim=-1, keepdim=True)
    beta = (-kappa * norm_2a - 2.0 * radial_v) / (2.0 * norm_sq + eps_safe)
    view_shape = [beta.shape[0]] + [1] * (z.dim() - 1)
    return v + beta.view(view_shape) * delta


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

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
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter



def exact_gamma_from_tau(tau: float, alpha: float, lamb: float) -> float:
    exponent = 1.0 / (2.0 - 2.0 * alpha)
    # 防御性编程：防止浮点误差导致的极小负数
    inner = 1.0 - lamb * (2.0 - 2.0 * alpha) * tau
    inner = max(0.0, inner) 
    return 1.0 - inner ** exponent

def exact_tau_from_gamma(gamma, alpha, lamb):
    """
    精确计算 τ(γ)
    
    公式: τ(γ) = [1 - (1-γ)^(2-2α)] / [λ(2-2α)]
    
    Args:
        gamma: 路径参数 γ ∈ [0,1]
        alpha: FT-EqM 参数
        lamb: 调度参数 λ
    Returns:
        tau: 物理时间 τ
    """
    gamma_clamped = gamma.clamp(min=0.0, max=1.0)
    numerator = 1.0 - (1.0 - gamma_clamped).pow(2.0 - 2.0 * alpha)
    denominator = lamb * (2.0 - 2.0 * alpha)
    tau = numerator / denominator
    return tau


@torch.no_grad()
def sample_nt_ft_fm(
    solver: ODESolver,
    x_0: torch.Tensor,
    labels: torch.Tensor,
    args: Namespace,
) -> torch.Tensor:
    if args.nt_solver == "midpoint":
        steps = 100
        if args.ode_options and "step_size" in args.ode_options:
            steps = int(1.0 / args.ode_options["step_size"])
        steps = max(steps, 1)

        z = x_0
        dg = 1.0 / steps
        gamma = 0.0

        for _ in range(steps):
            gamma_t = torch.full(
                (z.shape[0],),
                gamma,
                dtype=z.dtype,
                device=z.device,
            )
            rho_val = rho(gamma_t, args.alpha).view([z.shape[0]] + [1] * (z.dim() - 1))

            v_start = solver.velocity_model(
                z,
                gamma_t,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
            k1 = radial_correction(
                v=v_start,
                z=z,
                x_hat=z + v_start / rho_val,
                kappa=args.kappa,
                alpha=args.alpha,
            ) / rho_val

            gamma_mid = min(gamma + 0.5 * dg, 1.0)
            gamma_mid_t = torch.full(
                (z.shape[0],),
                gamma_mid,
                dtype=z.dtype,
                device=z.device,
            )
            z_mid = z + 0.5 * dg * k1
            rho_mid = rho(gamma_mid_t, args.alpha).view([z.shape[0]] + [1] * (z.dim() - 1))

            v_mid = solver.velocity_model(
                z_mid,
                gamma_mid_t,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
            k2 = radial_correction(
                v=v_mid,
                z=z_mid,
                x_hat=z_mid + v_mid / rho_mid,
                kappa=args.kappa,
                alpha=args.alpha,
            ) / rho_mid

            z = z + dg * k2
            gamma += dg

        return z

    if args.nt_solver == "rk45":
        atol = args.ode_options.get("atol", 1e-5) if args.ode_options else 1e-5
        rtol = args.ode_options.get("rtol", 1e-5) if args.ode_options else 1e-5

        def vector_field(gamma_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
            gamma_batch = torch.full(
                (z_t.shape[0],),
                float(gamma_t.item()),
                dtype=z_t.dtype,
                device=z_t.device,
            )
            rho_val = rho(gamma_batch, args.alpha).view(
                [z_t.shape[0]] + [1] * (z_t.dim() - 1)
            )
            v = solver.velocity_model(
                z_t,
                gamma_batch,
                label=labels,
                cfg_scale=args.cfg_scale,
            )
            u = radial_correction(
                v=v,
                z=z_t,
                x_hat=z_t + v / rho_val,
                kappa=args.kappa,
                alpha=args.alpha,
            )
            return u / rho_val

        gamma_grid = torch.tensor([0.0, 1.0], device=x_0.device, dtype=x_0.dtype)
        result = odeint(vector_field, x_0, gamma_grid, method="dopri5", atol=atol, rtol=rtol)
        return result[-1]

    raise ValueError(f"Unsupported --nt_solver value: {args.nt_solver}")


@torch.no_grad()
def sample_ft_eqm(
    solver: ODESolver,
    x_0: torch.Tensor,
    labels: torch.Tensor,
    args: Namespace,
) -> torch.Tensor:
    lamb = args.lambda_scale if args.lambda_scale is not None else (2.0 * args.alpha)
    if abs(args.alpha - 1.0) < 0.01:
        t_total = 10.0
    else:
        t_total = 1.0 / (lamb * (2.0 - 2.0 * args.alpha))

    if args.ft_eqm_solver == "midpoint":
        steps = 100
        if args.ode_options and "step_size" in args.ode_options:
            steps = int(1.0 / args.ode_options["step_size"])
        steps = max(steps, 1)

        z = x_0
        tau_grid = torch.linspace(0.0, t_total, steps + 1, device=x_0.device)

        for i in range(steps):
            tau_start = tau_grid[i].item()
            tau_end = tau_grid[i + 1].item()
            dt = tau_end - tau_start

            gamma_start = exact_gamma_from_tau(tau_start, args.alpha, lamb)
            gamma_start_batch = torch.full(
                (z.shape[0],), gamma_start, dtype=z.dtype, device=z.device
            )
            v_start = solver.velocity_model(
                z,
                gamma_start_batch,
                label=labels,
                cfg_scale=args.cfg_scale,
            )

            z_mid = z + v_start * (dt / 2.0)
            tau_mid = tau_start + (dt / 2.0)
            gamma_mid = exact_gamma_from_tau(tau_mid, args.alpha, lamb)
            gamma_mid_batch = torch.full(
                (z.shape[0],), gamma_mid, dtype=z.dtype, device=z.device
            )
            v_mid = solver.velocity_model(
                z_mid,
                gamma_mid_batch,
                label=labels,
                cfg_scale=args.cfg_scale,
            )

            z = z + v_mid * dt

        return z

    if args.ft_eqm_solver == "rk45":
        atol = args.ode_options.get("atol", 1e-5) if args.ode_options else 1e-5
        rtol = args.ode_options.get("rtol", 1e-5) if args.ode_options else 1e-5

        def vector_field(tau_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
            gamma = exact_gamma_from_tau(float(tau_t.item()), args.alpha, lamb)
            gamma_batch = torch.full(
                (z_t.shape[0],), gamma, dtype=z_t.dtype, device=z_t.device
            )
            return solver.velocity_model(
                z_t,
                gamma_batch,
                label=labels,
                cfg_scale=args.cfg_scale,
            )

        tau_grid = torch.tensor([0.0, t_total], device=x_0.device, dtype=x_0.dtype)
        result = odeint(vector_field, x_0, tau_grid, method="dopri5", atol=atol, rtol=rtol)
        return result[-1]

    raise ValueError(f"Unsupported --ft_eqm_solver value: {args.ft_eqm_solver}")


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    if args.use_ft_eqm and args.use_nt_ft_fm:
        raise ValueError("--use_ft_eqm and --use_nt_ft_fm are mutually exclusive.")

    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    label=labels,
                    cfg_scale=args.cfg_scale,
                )
            else:
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                if args.use_nt_ft_fm:
                    synthetic_samples = sample_nt_ft_fm(
                        solver=solver,
                        x_0=x_0,
                        labels=labels,
                        args=args,
                    )

                elif args.use_ft_eqm:
                    synthetic_samples = sample_ft_eqm(
                        solver=solver,
                        x_0=x_0,
                        labels=labels,
                        args=args,
                    )

                else:
                    if args.edm_schedule:
                        time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                    else:
                        time_grid = torch.tensor([0.0, 1.0], device=device)

                    synthetic_samples = solver.sample(
                        time_grid=time_grid,
                        x_init=x_0,
                        method=args.ode_method,
                        return_intermediates=False,
                        atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                        rtol=ode_opts["rtol"] if "rtol" in ode_opts else 1e-5,
                        step_size=ode_opts["step_size"]
                        if "step_size" in ode_opts
                        else None,
                        label=labels,
                        cfg_scale=args.cfg_scale,
                    )



                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
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

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    return {"fid": float(fid_metric.compute().detach().cpu())}
