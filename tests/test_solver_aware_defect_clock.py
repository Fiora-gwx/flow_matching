import unittest

try:
    import torch

    from examples.image.training.solver_aware.defect_clock import (
        build_defect_clock_profile,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is required for solver-aware defect clock tests")
class SolverAwareDefectClockTest(unittest.TestCase):
    def test_single_budget_uses_order_dependent_exponent(self):
        profile = build_defect_clock_profile(
            s_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            q_values_by_budget={12: torch.tensor([1.0, 16.0, 81.0], dtype=torch.float32)},
            budget_scale_by_nfe={12: 12},
            budget_mode="single_budget",
            order=1.0,
            eps=1.0e-6,
        )

        self.assertAlmostEqual(profile.density_exponent, 0.25, places=6)
        self.assertEqual(profile.q_curve_name, "Q_path_defect")
        self.assertEqual(profile.primary_budget, 12)
        self.assertIn("12", profile.q_values_by_budget)
        self.assertIn("12", profile.q_normalized_by_budget)

    def test_multi_budget_normalizes_before_aggregation(self):
        base_curve = torch.ones(3, dtype=torch.float32)
        profile = build_defect_clock_profile(
            s_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            q_values_by_budget={
                10: base_curve / (10.0 ** 4),
                20: base_curve / (20.0 ** 4),
            },
            budget_scale_by_nfe={10: 10, 20: 20},
            budget_mode="multi_budget",
            order=1.0,
            eps=1.0e-6,
            target_nfe_weights=[0.5, 0.5],
        )

        self.assertAlmostEqual(profile.density_exponent, 0.5, places=6)
        self.assertEqual(profile.q_curve_name, "M_tilde_path_defect")
        self.assertTrue(torch.allclose(profile.q_raw, base_curve, atol=1.0e-5))
        self.assertEqual(profile.budget_weights["10"], 0.5)
        self.assertEqual(profile.budget_weights["20"], 0.5)


if __name__ == "__main__":
    unittest.main()
