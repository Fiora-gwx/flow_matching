import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.fixed_step_solver import build_step_methods, solve_fixed_budget
from training.tack import (
    _ab2_predict_nonuniform,
    _ab2_predict_uniform,
    _ab3_predict_nonuniform,
    _ab3_predict_uniform,
    _stabilize_psi_prime,
    _stabilize_rho_star,
    build_tack_config_from_namespace,
)


class DummyModel:
    def __init__(self):
        self.nfe = 0

    def __call__(self, x, t, **kwargs):
        self.nfe += 1
        return torch.ones_like(x)

    def reset_nfe_counter(self):
        self.nfe = 0

    def get_nfe(self):
        return self.nfe


class LinearModel(DummyModel):
    def __init__(self, rate):
        super().__init__()
        self.rate = float(rate)

    def __call__(self, x, t, **kwargs):
        self.nfe += 1
        return self.rate * x


class RecordingModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.queried_times = []

    def __call__(self, x, t, **kwargs):
        self.nfe += 1
        self.queried_times.append(t.detach().clone())
        return torch.ones_like(x)


class EndpointAdaptiveModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.queried_times = []

    def adapt_solver_time(self, t, step_size, step_count=None):
        del step_size
        if step_count is None:
            return t
        sample_eps = 1.0 / float(step_count)
        return t.clamp(min=sample_eps, max=1.0 - sample_eps)

    def __call__(self, x, t, **kwargs):
        self.nfe += 1
        self.queried_times.append(t.detach().clone())
        return torch.ones_like(x)


class FixedStepSolverTest(unittest.TestCase):
    def test_nonuniform_ab_coefficients_match_uniform_on_equal_steps(self):
        z_n = torch.zeros(1, 1)
        g_n = torch.full((1, 1), 3.0)
        g_nm1 = torch.full((1, 1), 2.0)
        g_nm2 = torch.full((1, 1), 1.0)
        h = 0.1

        ab2_uniform = _ab2_predict_uniform(z_n, g_n, g_nm1, h=h)
        ab2_nonuniform = _ab2_predict_nonuniform(
            z_n=z_n,
            g_n=g_n,
            g_nm1=g_nm1,
            h=h,
            beta1=h,
            eps=1.0e-8,
        )
        ab3_uniform = _ab3_predict_uniform(z_n, g_n, g_nm1, g_nm2, h=h)
        ab3_nonuniform = _ab3_predict_nonuniform(
            z_n=z_n,
            g_n=g_n,
            g_nm1=g_nm1,
            g_nm2=g_nm2,
            h=h,
            beta1=h,
            beta0=h,
            eps=1.0e-8,
        )

        self.assertIsNotNone(ab2_nonuniform)
        self.assertIsNotNone(ab3_nonuniform)
        self.assertTrue(torch.allclose(ab2_uniform, ab2_nonuniform))
        self.assertTrue(torch.allclose(ab3_uniform, ab3_nonuniform))

    def test_rho_star_and_psi_prime_are_stabilized(self):
        rho_raw = torch.tensor([6.0, 1.0, 6.0], dtype=torch.float64)
        rho_floor_raw = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        floor_smoothed, rho_smoothed, rho_star = _stabilize_rho_star(
            rho_raw=rho_raw,
            rho_floor_raw=rho_floor_raw,
            smoothing_window=3,
            eps=1.0e-8,
        )
        self.assertTrue(torch.all(rho_star >= rho_floor_raw))
        self.assertTrue(torch.all(rho_star >= rho_smoothed))
        self.assertTrue(torch.all(floor_smoothed > 0.0))

        psi_prime_values, regularizer, cap = _stabilize_psi_prime(
            psi_values=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
            r_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
            rho_star=rho_star,
            rho_floor_raw=rho_floor_raw,
            total_mass=1.0,
            eps=1.0e-8,
        )
        self.assertTrue(torch.isfinite(psi_prime_values).all())
        self.assertGreater(regularizer, 0.0)
        self.assertGreaterEqual(cap, 1.0)
        self.assertLessEqual(float(psi_prime_values.max().item()), cap + 1.0e-6)

    def test_build_step_methods_exact_budget(self):
        self.assertEqual(build_step_methods('euler', 3), ('euler', 'euler', 'euler'))
        self.assertEqual(build_step_methods('heun2', 4), ('heun2', 'heun2'))
        self.assertEqual(build_step_methods('heun2', 5), ('heun2', 'heun2', 'euler'))
        self.assertEqual(build_step_methods('rk3', 6), ('rk3', 'rk3'))
        self.assertEqual(build_step_methods('rk3', 7), ('rk3', 'rk3', 'euler'))
        self.assertEqual(build_step_methods('rk3', 8), ('rk3', 'rk3', 'heun2'))

    def test_solver_counts_real_nfe(self):
        model = DummyModel()
        x_init = torch.zeros(2, 3)
        result = solve_fixed_budget(model, x_init, 'heun2', 5)
        self.assertEqual(result.nfe, 5)
        self.assertEqual(result.step_count, 3)
        self.assertFalse(result.solver_stats['is_exact_budget'])
        self.assertTrue(result.solver_stats['used_tail_step'])

    def test_rk3_converges_on_linear_ode(self):
        x_init = torch.ones(1, 1)
        exact = torch.exp(torch.tensor(-1.0))

        coarse = solve_fixed_budget(LinearModel(rate=-1.0), x_init, 'rk3', 3)
        fine = solve_fixed_budget(LinearModel(rate=-1.0), x_init, 'rk3', 6)

        coarse_error = torch.abs(coarse.sample.squeeze() - exact)
        fine_error = torch.abs(fine.sample.squeeze() - exact)
        self.assertEqual(coarse.nfe, 3)
        self.assertEqual(fine.nfe, 6)
        self.assertLess(fine_error.item(), coarse_error.item())
        self.assertTrue(fine.solver_stats['is_exact_budget'])
        self.assertFalse(fine.solver_stats['used_tail_step'])

    def test_stork4_reports_virtual_stages(self):
        model = DummyModel()
        x_init = torch.zeros(2, 3)
        result = solve_fixed_budget(model, x_init, 'stork4', 6)
        self.assertEqual(result.nfe, 6)
        self.assertEqual(result.step_count, 6)
        self.assertTrue(torch.isfinite(result.sample).all())
        self.assertGreater(result.solver_stats['virtual_stage_count'], 0)
        self.assertTrue(result.solver_stats['is_exact_budget'])
        self.assertTrue(result.solver_stats['is_shared_budget'])
        self.assertFalse(result.solver_stats['used_tail_step'])

    def test_heun2_uses_adapted_terminal_stage_time(self):
        model = EndpointAdaptiveModel()
        x_init = torch.zeros(1, 1)
        solve_fixed_budget(model, x_init, 'heun2', 8)
        queried_times = torch.cat(model.queried_times)
        self.assertTrue(torch.all(queried_times < 1.0))
        self.assertAlmostEqual(float(queried_times[-1]), 0.75, places=6)

    def test_rk3_uses_adapted_terminal_stage_time(self):
        model = EndpointAdaptiveModel()
        x_init = torch.zeros(1, 1)
        solve_fixed_budget(model, x_init, 'rk3', 12)
        queried_times = torch.cat(model.queried_times)
        self.assertTrue(torch.all(queried_times < 1.0))
        self.assertAlmostEqual(float(queried_times[-1]), 0.75, places=6)

    def test_velocity_model_without_adapt_hook_keeps_terminal_time(self):
        model = RecordingModel()
        x_init = torch.zeros(1, 1)
        solve_fixed_budget(model, x_init, 'heun2', 8)
        queried_times = torch.cat(model.queried_times)
        self.assertAlmostEqual(float(queried_times[-1]), 1.0, places=6)

    def test_euler_accepts_nonuniform_time_grid(self):
        model = RecordingModel()
        x_init = torch.zeros(1, 1)
        time_grid = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float32)
        result = solve_fixed_budget(
            model,
            x_init,
            'euler',
            3,
            time_grid=time_grid,
        )
        queried_times = torch.cat(model.queried_times)
        self.assertTrue(torch.allclose(result.time_grid, time_grid))
        self.assertTrue(torch.allclose(queried_times, time_grid[:-1]))

    def test_stork4_accepts_nonuniform_time_grid(self):
        model = DummyModel()
        x_init = torch.zeros(1, 1)
        time_grid = torch.tensor([0.0, 0.05, 0.25, 1.0], dtype=torch.float32)
        result = solve_fixed_budget(
            model,
            x_init,
            'stork4',
            3,
            time_grid=time_grid,
        )
        self.assertTrue(torch.allclose(result.time_grid, time_grid))
        self.assertEqual(result.step_count, 3)
        self.assertGreater(result.solver_stats['virtual_stage_count'], 0)

    def test_tack_online_only_respects_requested_nfe_without_profile(self):
        model = DummyModel()
        x_init = torch.zeros(1, 1)
        args = types.SimpleNamespace(
            path_family="linear",
            clock_family="uniform",
            clock_beta=None,
            signal_scale_sq=None,
            cfg_scale=0.0,
            eval_nfe=6,
            tack_profile_grid_size=16,
            tack_profile_batch_size=8,
            tack_profile_num_batches=2,
            tack_profile_eps=1.0e-8,
            tack_lambda=1.0,
            tack_eta=0.25,
            tack_profile_cache=False,
            tack_force_recompute_profile=False,
            tack_chi_lo=0.10,
            tack_chi_hi=0.50,
            tack_tau=0.05,
            tack_startup_steps=2,
            tack_enable_dyadic=False,
            tack_batch_shared_adapt=True,
            tack_min_dr_scale=0.25,
            tack_max_dr_scale=4.0,
            tack_monitor_estimator="auto",
            tack_mode="online_only",
            seed=0,
            resume="checkpoint-499.pth",
        )
        result = solve_fixed_budget(
            model,
            x_init,
            "tack",
            6,
            tack_config=build_tack_config_from_namespace(args),
        )
        self.assertEqual(result.nfe, 6)
        self.assertEqual(result.step_count, 5)
        self.assertEqual(result.solver_stats["requested_eval_nfe"], 6)
        self.assertEqual(result.solver_stats["realized_nfe"], 6)
        self.assertEqual(result.solver_stats["tack_num_accepted_steps"], 5)
        self.assertEqual(result.solver_stats["tack_num_heun_steps"], 5)
        self.assertEqual(result.solver_stats["tack_num_valid_chi_steps"], 0)
        self.assertTrue(all(row["chi"] is None for row in result.solver_stats["step_records"]))
        self.assertTrue(all(not row["chi_valid"] for row in result.solver_stats["step_records"]))
        self.assertEqual(result.time_grid.shape[0], result.step_count + 1)

    def test_tack_clock_only_keeps_uniform_r_steps_without_dyadic_updates(self):
        model = LinearModel(rate=-1.0)
        x_init = torch.ones(1, 1)
        args = types.SimpleNamespace(
            path_family="linear",
            clock_family="uniform",
            clock_beta=None,
            signal_scale_sq=None,
            cfg_scale=0.0,
            eval_nfe=8,
            tack_profile_grid_size=16,
            tack_profile_batch_size=8,
            tack_profile_num_batches=2,
            tack_profile_eps=1.0e-8,
            tack_lambda=1.0,
            tack_eta=0.25,
            tack_profile_cache=False,
            tack_force_recompute_profile=False,
            tack_chi_lo=1.0e6,
            tack_chi_hi=1.0e6,
            tack_tau=0.01,
            tack_startup_steps=1,
            tack_enable_dyadic=True,
            tack_batch_shared_adapt=True,
            tack_min_dr_scale=0.25,
            tack_max_dr_scale=4.0,
            tack_monitor_estimator="auto",
            tack_mode="clock_only",
            seed=0,
            resume="checkpoint-499.pth",
        )
        result = solve_fixed_budget(
            model,
            x_init,
            "tack",
            8,
            tack_config=build_tack_config_from_namespace(args),
        )

        dq_values = [row["dq"] for row in result.solver_stats["step_records"]]
        self.assertTrue(all(abs(dq - dq_values[0]) < 1.0e-8 for dq in dq_values))
        self.assertEqual(result.solver_stats["tack_num_halvings"], 0)
        self.assertEqual(result.solver_stats["tack_num_doublings"], 0)

    def test_tack_full_uses_nonuniform_history_with_dyadic_steps(self):
        model = LinearModel(rate=-1.0)
        x_init = torch.ones(1, 1)
        args = types.SimpleNamespace(
            path_family="linear",
            clock_family="uniform",
            clock_beta=None,
            signal_scale_sq=None,
            cfg_scale=0.0,
            eval_nfe=10,
            tack_profile_grid_size=16,
            tack_profile_batch_size=8,
            tack_profile_num_batches=2,
            tack_profile_eps=1.0e-8,
            tack_lambda=1.0,
            tack_eta=0.25,
            tack_profile_cache=False,
            tack_force_recompute_profile=False,
            tack_chi_lo=1.0e6,
            tack_chi_hi=1.0e6,
            tack_tau=10.0,
            tack_startup_steps=1,
            tack_enable_dyadic=True,
            tack_batch_shared_adapt=True,
            tack_min_dr_scale=0.25,
            tack_max_dr_scale=4.0,
            tack_monitor_estimator="auto",
            tack_mode="full",
            seed=0,
            resume="checkpoint-499.pth",
        )
        result = solve_fixed_budget(
            model,
            x_init,
            "tack",
            10,
            tack_config=build_tack_config_from_namespace(args),
        )

        dq_values = [row["dq"] for row in result.solver_stats["step_records"]]
        self.assertGreater(len({round(float(dq), 8) for dq in dq_values}), 1)
        self.assertGreater(result.solver_stats["tack_num_doublings"], 0)
        self.assertGreater(result.solver_stats["tack_num_ab3_steps"], 0)
        self.assertTrue(
            any(
                row["mode"] == "ab3" and row["dq_history_depth"] >= 2
                for row in result.solver_stats["step_records"]
            )
        )


if __name__ == '__main__':
    unittest.main()
