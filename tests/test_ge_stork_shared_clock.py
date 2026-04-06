import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, "examples", "image")
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

if torch is not None:
    from training.ge_stork.shared_clock import (
        build_or_load_shared_clock,
        build_shared_clock,
        get_time_grid_for_nfe,
    )


class TinyVelocityModel:
    def __init__(self):
        self.nfe = 0

    def __call__(self, x, t, cfg_scale=0.0, label=None, use_autocast=False):
        del cfg_scale, label, use_autocast
        self.nfe += 1
        return (1.0 + t.view(-1, 1, 1)) * x + 0.1


def trapz_weights(grid):
    deltas = grid[1:] - grid[:-1]
    weights = torch.zeros_like(grid)
    weights[0] = 0.5 * deltas[0]
    weights[-1] = 0.5 * deltas[-1]
    if grid.numel() > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return weights


@unittest.skipIf(torch is None, "torch is required for shared clock tests")
class SharedClockProfileTest(unittest.TestCase):
    def setUp(self):
        batch = (
            torch.zeros(2, 1, 1, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
        self.data_loader = [batch]
        self.device = torch.device("cpu")

    def test_analytic_shared_clock_density_is_positive_and_normalized(self):
        profile = build_shared_clock(
            clock_family="va",
            velocity_model=TinyVelocityModel(),
            data_loader=self.data_loader,
            device=self.device,
            path_family="linear",
            pilot_solver="euler",
            physical_grid_size=5,
            pilot_batch_size=2,
            pilot_num_batches=1,
            observation_microbatch=1,
            cfg_scale=0.0,
            eps=1.0e-6,
            jacobian_backend="probe",
            jacobian_num_probes=2,
            optimizer_steps=4,
            optimizer_lr=0.05,
            checkpoint_source="checkpoint-499.pth",
            seed=0,
        )
        self.assertTrue(torch.all(profile.density > 0.0))
        total_mass = torch.sum(profile.density * trapz_weights(profile.physical_grid))
        self.assertAlmostEqual(float(total_mass.item()), 1.0, places=5)
        self.assertTrue(torch.all(profile.tau_grid[1:] > profile.tau_grid[:-1]))
        self.assertEqual(profile.observation_microbatch, 1)

    def test_shared_clock_schedule_reuses_same_profile_across_step_counts(self):
        profile = build_shared_clock(
            clock_family="ab",
            velocity_model=TinyVelocityModel(),
            data_loader=self.data_loader,
            device=self.device,
            path_family="linear",
            pilot_solver="euler",
            physical_grid_size=5,
            pilot_batch_size=2,
            pilot_num_batches=1,
            observation_microbatch=1,
            cfg_scale=0.0,
            eps=1.0e-6,
            jacobian_backend="exact",
            jacobian_num_probes=1,
            optimizer_steps=6,
            optimizer_lr=0.05,
            checkpoint_source="checkpoint-499.pth",
            seed=0,
        )
        schedule4 = get_time_grid_for_nfe(profile, 4, step_count=4)["schedule"]
        schedule8 = get_time_grid_for_nfe(profile, 8, step_count=8)["schedule"]
        self.assertAlmostEqual(float(schedule4.t_grid[2].item()), float(schedule8.t_grid[4].item()), places=5)
        self.assertTrue(torch.all(schedule4.g_grid > 0.0))
        self.assertTrue(torch.all(schedule8.g_grid > 0.0))
        self.assertAlmostEqual(float(schedule4.tau_grid[-1].item()), 1.0, places=6)
        self.assertAlmostEqual(float(schedule8.tau_grid[-1].item()), 1.0, places=6)

    def test_vb_uses_va_density_as_optimizer_init(self):
        shared_kwargs = dict(
            velocity_model=TinyVelocityModel(),
            data_loader=self.data_loader,
            device=self.device,
            path_family="linear",
            pilot_solver="euler",
            physical_grid_size=5,
            pilot_batch_size=2,
            pilot_num_batches=1,
            observation_microbatch=1,
            cfg_scale=0.0,
            eps=1.0e-6,
            jacobian_backend="probe",
            jacobian_num_probes=2,
            checkpoint_source="checkpoint-499.pth",
            seed=0,
        )
        va_profile = build_shared_clock(
            clock_family="va",
            optimizer_steps=4,
            optimizer_lr=0.05,
            **shared_kwargs,
        )
        vb_profile = build_shared_clock(
            clock_family="vb",
            optimizer_steps=0,
            optimizer_lr=0.05,
            **shared_kwargs,
        )
        self.assertTrue(torch.allclose(vb_profile.density, va_profile.density, atol=1.0e-5, rtol=1.0e-5))

    def test_build_or_load_shared_clock_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "eval_ep499_nfe12"
            cache_path = str(Path(tmpdir) / "shared_clock_cache.pt")
            kwargs = dict(
                clock_family="va",
                velocity_model=TinyVelocityModel(),
                data_loader=self.data_loader,
                device=self.device,
                path_family="linear",
                pilot_solver="euler",
                physical_grid_size=5,
                pilot_batch_size=2,
                pilot_num_batches=1,
                observation_microbatch=1,
                cfg_scale=0.0,
                eps=1.0e-6,
                jacobian_backend="probe",
                jacobian_num_probes=2,
                optimizer_steps=4,
                optimizer_lr=0.05,
                checkpoint_source="checkpoint-499.pth",
                seed=0,
                cache_path=cache_path,
                output_dir=output_dir,
            )
            first = build_or_load_shared_clock(**kwargs)
            second = build_or_load_shared_clock(**kwargs)
            self.assertTrue(Path(cache_path).exists())
            self.assertEqual(first.created_at, second.created_at)
            self.assertTrue((output_dir / "shared_clock_profile.json").exists())


if __name__ == "__main__":
    unittest.main()
