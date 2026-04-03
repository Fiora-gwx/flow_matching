import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "ft_clock"
    / "defect_based_solver_aware_linear_uniform.yaml"
)


class SolverAwareDefectConfigTest(unittest.TestCase):
    def test_config_contains_requested_phase1_experiments(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- name: linear_uniform_euler_defect_single_budget", source)
        self.assertIn("- name: linear_uniform_heun2_defect_single_budget", source)
        self.assertIn("- name: linear_uniform_stork4_defect_single_budget", source)
        self.assertIn("- name: linear_uniform_euler_defect_multi_budget", source)
        self.assertIn("- name: linear_uniform_heun2_defect_multi_budget", source)

    def test_config_uses_defect_based_monitor_family(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("solver_aware_monitor_family: defect_based", source)
        self.assertIn("solver_aware_defect_subdivide: 2", source)
        self.assertIn("solver_aware_stork_effective_order: 4.0", source)

    def test_config_declares_multi_budget_target_list(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("solver_aware_target_nfe_list: [6, 12, 18, 24, 30, 48, 96]", source)
        self.assertIn("solver_aware_target_nfe_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]", source)


if __name__ == "__main__":
    unittest.main()
