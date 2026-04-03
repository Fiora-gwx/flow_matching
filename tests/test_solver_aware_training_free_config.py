import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "ft_clock"
    / "solver_aware_training_free_linear_uniform.yaml"
)


class SolverAwareTrainingFreeConfigTest(unittest.TestCase):
    def test_config_runs_only_solver_aware_experiments(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- name: linear_uniform_euler_solver_aware", source)
        self.assertIn("- name: linear_uniform_heun2_solver_aware", source)
        self.assertIn("- name: linear_uniform_stork4_solver_aware", source)
        self.assertNotIn("linear_uniform_euler_baseline", source)
        self.assertNotIn("linear_uniform_heun2_baseline", source)
        self.assertNotIn("linear_uniform_stork4_baseline", source)

    def test_config_reuses_trained_linear_uniform_checkpoint(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("artifact_group: ft_clock_linear_main", source)
        self.assertIn("source_exp_name: linear_uniform", source)
        self.assertEqual(source.count("checkpoint_from:"), 1)

    def test_stork4_solver_aware_uses_solver_sensitivity_nfe_grid(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("eval_nfes: [6, 8, 10, 12, 18, 20, 24, 30, 48, 50, 96]", source)


if __name__ == "__main__":
    unittest.main()
