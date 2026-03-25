import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import analyze_mechanisms


class AnalyzeMechanismsTest(unittest.TestCase):
    def test_resolve_analysis_checkpoint_supports_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "experiments"
                / "results"
                / "ft_clock_linear_main"
                / "cifar10"
                / "linear_uniform"
                / "checkpoint-499.pth"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            (checkpoint.parent / "args.json").write_text(
                json.dumps(
                    {
                        "path_family": "linear",
                        "clock_family": "uniform",
                    }
                ),
                encoding="utf-8",
            )
            spec = {
                "dataset": "cifar10",
                "name": "linear_uniform",
                "checkpoint_from": {
                    "artifact_group": "ft_clock_linear_main",
                    "checkpoint_epoch": 499,
                },
            }

            with mock.patch.object(analyze_mechanisms, "ROOT", root):
                resolved = analyze_mechanisms.resolve_analysis_checkpoint(
                    base_dir=root / "experiments" / "results" / "ft_clock_mechanism_analysis",
                    spec=spec,
                )

            self.assertEqual(resolved, checkpoint)


if __name__ == "__main__":
    unittest.main()
