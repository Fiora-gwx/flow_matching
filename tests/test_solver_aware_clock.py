import unittest

try:
    import torch

    from examples.image.training.solver_aware.clock import build_solver_aware_clock_profile
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is required for solver-aware clock tests")
class SolverAwareClockTest(unittest.TestCase):
    def test_constrained_profile_falls_back_to_uniform_when_floor_is_infeasible(self):
        s_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
        q_e = torch.full((5,), 1.0e-4, dtype=torch.float32)
        q_h = torch.full((5,), 1.0, dtype=torch.float32)

        profile = build_solver_aware_clock_profile(
            s_grid=s_grid,
            q_values=q_e,
            q_h_values=q_h,
            use_q_h_for_weight=False,
            density_exponent=0.25,
            eps=1.0e-6,
            step_count=2,
            eta=0.25,
            floor_mode="pointwise",
            floor_eps=1.0e-6,
            legacy_unconstrained=False,
        )

        self.assertTrue(profile.used_uniform_fallback)
        self.assertGreater(profile.floor_mass, 1.0)
        self.assertGreaterEqual(profile.min_feasible_step_count, 3)
        self.assertTrue(torch.allclose(profile.density, torch.ones_like(profile.density), atol=1e-5))
