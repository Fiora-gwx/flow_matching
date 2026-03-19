import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

if HAS_TORCH and HAS_MATPLOTLIB:
    from experiments.plot_particle_trajectory_comparison import (
        MethodSpec,
        plot_comparison,
    )
else:  # pragma: no cover - environment dependent
    MethodSpec = None
    plot_comparison = None


@unittest.skipUnless(HAS_TORCH and HAS_MATPLOTLIB, "torch and matplotlib are required")
class ParticleTrajectoryPlotTest(unittest.TestCase):
    def test_plot_comparison_writes_figure_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "plots"
            figure_path, summary_path = plot_comparison(
                output_dir=out_dir,
                baseline_spec=MethodSpec(
                    label="Baseline",
                    pairing="random",
                    path_family="linear",
                    clock_family="uniform",
                    clock_beta=None,
                ),
                proposed_spec=MethodSpec(
                    label="Ours",
                    pairing="oracle",
                    path_family="linear",
                    clock_family="ft_linear_beta",
                    clock_beta=0.5,
                ),
                num_points_per_group=4,
                num_steps=12,
                seed=3,
            )

            self.assertTrue(figure_path.exists())
            self.assertTrue(summary_path.exists())
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn("Theoretical Perfect Pairing", summary_text)
            self.assertIn("Ours", summary_text)


if __name__ == "__main__":
    unittest.main()
