import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "ft_clock"
    / "ge_stork_linear_uniform_12way.yaml"
)


class GeStorkConfigTest(unittest.TestCase):
    def test_config_contains_all_12_main_experiments(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("shared_clock_family:"), 12)
        self.assertEqual(source.count("sampling_solver: euler"), 4)
        self.assertEqual(source.count("sampling_solver: heun2"), 4)
        self.assertEqual(source.count("sampling_solver: stork4"), 4)
        self.assertIn("linear_uniform_ge_stork_ab_stork4", source)
        self.assertIn("linear_uniform_ge_stork_vb_stork4", source)

    def test_config_reuses_linear_uniform_checkpoint(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("artifact_group: ft_clock_linear_main", source)
        self.assertIn("source_exp_name: linear_uniform", source)
        self.assertEqual(source.count("checkpoint_from:"), 1)


if __name__ == "__main__":
    unittest.main()
