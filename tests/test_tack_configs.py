import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER_CONFIG = ROOT / "experiments" / "configs" / "ft_clock" / "tack_solver_sensitivity.yaml"
ABLATION_CONFIG = ROOT / "experiments" / "configs" / "ft_clock" / "tack_ablation.yaml"


class TackConfigTest(unittest.TestCase):
    def test_solver_sensitivity_config_contains_tack_experiments_only(self):
        source = SOLVER_CONFIG.read_text(encoding="utf-8")
        self.assertIn("sampling_solver: tack", source)
        self.assertIn("- name: linear_uniform_tack", source)
        self.assertIn("- name: trig_vp_uniform_tack", source)
        self.assertNotIn("sampling_solver: euler", source)
        self.assertNotIn("sampling_solver: heun2", source)
        self.assertNotIn("sampling_solver: rk3", source)
        self.assertNotIn("sampling_solver: stork4", source)

    def test_ablation_config_contains_all_three_modes(self):
        source = ABLATION_CONFIG.read_text(encoding="utf-8")
        self.assertIn("tack_mode: full", source)
        self.assertIn("tack_mode: clock_only", source)
        self.assertIn("tack_mode: online_only", source)
        self.assertIn("- name: linear_uniform_tack_full", source)
        self.assertIn("- name: trig_vp_uniform_tack_online_only", source)


if __name__ == "__main__":
    unittest.main()
