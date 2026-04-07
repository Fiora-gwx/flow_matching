import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

try:
    import torch

    from training.fixed_step_solver import ReparameterizedSchedule, build_step_methods, solve_fixed_budget
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


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


@unittest.skipIf(torch is None, "torch is required for fixed-step solver tests")
class FixedStepSolverTest(unittest.TestCase):
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

    def test_euler_reparameterized_schedule_uses_uniform_tau_step(self):
        model = RecordingModel()
        x_init = torch.zeros(1, 1)
        schedule = ReparameterizedSchedule(
            tau_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            t_grid=torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
            g_grid=torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32),
            dtau=0.5,
            nfe_budget=2,
            step_count=2,
        )
        result = solve_fixed_budget(
            model,
            x_init,
            'euler',
            2,
            reparameterized_schedule=schedule,
        )
        self.assertAlmostEqual(float(result.sample.squeeze().item()), 3.0, places=6)
        self.assertTrue(torch.allclose(result.time_grid, schedule.t_grid))
        self.assertTrue(torch.allclose(result.tau_grid, schedule.tau_grid))
        queried_times = torch.cat(model.queried_times)
        self.assertTrue(torch.allclose(queried_times, schedule.t_grid[:-1]))

    def test_heun2_reparameterized_schedule_uses_gn_and_gn1(self):
        model = RecordingModel()
        x_init = torch.zeros(1, 1)
        schedule = ReparameterizedSchedule(
            tau_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            t_grid=torch.tensor([0.0, 0.3, 1.0], dtype=torch.float32),
            g_grid=torch.tensor([2.0, 6.0, 10.0], dtype=torch.float32),
            dtau=0.5,
            nfe_budget=4,
            step_count=2,
        )
        result = solve_fixed_budget(
            model,
            x_init,
            'heun2',
            4,
            reparameterized_schedule=schedule,
        )
        self.assertAlmostEqual(float(result.sample.squeeze().item()), 6.0, places=6)
        self.assertTrue(torch.allclose(result.tau_grid, schedule.tau_grid))
        queried_times = torch.cat(model.queried_times)
        self.assertTrue(
            torch.allclose(
                queried_times,
                torch.tensor([0.0, 0.3, 0.3, 1.0], dtype=torch.float32),
            )
        )

    def test_stork4_reparameterized_schedule_tracks_tau_grid(self):
        model = DummyModel()
        x_init = torch.zeros(1, 1)
        schedule = ReparameterizedSchedule(
            tau_grid=torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32),
            t_grid=torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float32),
            g_grid=torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32),
            dtau=1.0 / 3.0,
            nfe_budget=3,
            step_count=3,
        )
        result = solve_fixed_budget(
            model,
            x_init,
            'stork4',
            3,
            reparameterized_schedule=schedule,
        )
        self.assertTrue(torch.allclose(result.time_grid, schedule.t_grid))
        self.assertTrue(torch.allclose(result.tau_grid, schedule.tau_grid))
        self.assertGreater(result.solver_stats['virtual_stage_count'], 0)


if __name__ == '__main__':
    unittest.main()
