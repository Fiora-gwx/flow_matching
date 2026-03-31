import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "ft_clock"
    / "variance_reduction_ablation.yaml"
)


class VarianceReductionAblationConfigTest(unittest.TestCase):
    def test_stage1_ablation_keeps_legacy_baseline_conditioning(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("cfg_scale: 0.2", source)
        self.assertIn("class_drop_prob: 0.2", source)

    def test_stage1_ablation_uses_two_gpus_with_accumulation_to_preserve_effective_batch(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("num_gpus: 2", source)
        self.assertIn("accum_iter: 2", source)

    def test_stage1_ablation_excludes_rk3(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sampling_solver: rk3", source)

    def test_strategy_a_non_euler_solvers_reuse_euler_checkpoint(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("source_name_template: linear_ft_beta_{clock_beta_tag}_strategy_A_euler"), 6)
        self.assertEqual(source.count("checkpoint_from:"), 6)


if __name__ == "__main__":
    unittest.main()
