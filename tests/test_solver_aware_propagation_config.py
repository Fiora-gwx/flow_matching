import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "ft_clock"
    / "solver_aware_propagation_training_free_linear_uniform.yaml"
)


class SolverAwarePropagationConfigTest(unittest.TestCase):
    def test_config_runs_only_solver_aware_experiments(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- name: linear_uniform_euler_propagation_solver_aware", source)
        self.assertIn("- name: linear_uniform_heun2_propagation_solver_aware", source)
        self.assertIn("- name: linear_uniform_stork4_propagation_solver_aware", source)
        self.assertNotIn("linear_uniform_euler_baseline", source)
        self.assertNotIn("linear_uniform_heun2_baseline", source)
        self.assertNotIn("linear_uniform_stork4_baseline", source)

    def test_config_enables_propagation(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("solver_aware_use_propagation: true", source)
        self.assertIn("solver_aware_g_mode: jacobian_envelope", source)
        self.assertIn("solver_aware_g_estimator: spectral_q95", source)
        self.assertNotIn("solver_aware_eta:", source)
        self.assertIn("solver_aware_floor_mode: pointwise", source)
        self.assertIn("solver_aware_floor_eps: 1.0e-6", source)
        self.assertIn("solver_aware_compute_qh_for_euler: true", source)
        self.assertIn("solver_aware_legacy_unconstrained: false", source)

    def test_config_reuses_linear_uniform_checkpoint(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("artifact_group: ft_clock_linear_main_uparam", source)
        self.assertIn("source_exp_name: linear_uniform", source)


if __name__ == "__main__":
    unittest.main()
