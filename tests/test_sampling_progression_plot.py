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
    import torch

    from experiments.plot_sampling_progression import (
        build_snapshot_indices,
        plot_snapshot_comparison,
    )
else:  # pragma: no cover - environment dependent
    torch = None
    build_snapshot_indices = None
    plot_snapshot_comparison = None


@unittest.skipUnless(HAS_TORCH and HAS_MATPLOTLIB, "torch and matplotlib are required")
class SamplingProgressionPlotTest(unittest.TestCase):
    def test_build_snapshot_indices_includes_start_and_end(self):
        indices = build_snapshot_indices(11, [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 10)

    def test_plot_snapshot_comparison_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "progression.png"
            grids = [
                [torch.rand(3, 16, 16) for _ in range(3)],
                [torch.rand(3, 16, 16) for _ in range(3)],
            ]
            plot_snapshot_comparison(
                row_labels=["Baseline", "Ours"],
                column_labels=["r=0.0", "r=0.5", "r=1.0"],
                grids=grids,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
