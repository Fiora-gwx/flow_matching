import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

if HAS_TORCH and HAS_MATPLOTLIB:
    import torch

    from experiments import plot_sampling_progression as sampling_progression
    from experiments.result_utils import append_result_rows, ensure_results_file
    from experiments.plot_sampling_progression import (
        build_snapshot_indices,
        plot_snapshot_comparison,
        resolve_method_fields,
    )
else:  # pragma: no cover - environment dependent
    torch = None
    append_result_rows = None
    build_snapshot_indices = None
    plot_snapshot_comparison = None
    resolve_method_fields = None
    sampling_progression = None
    ensure_results_file = None


def make_source_row(run_id, seed, beta, value, nfe):
    return {
        "run_id": run_id,
        "exp_name": f"linear_ft_beta_{str(beta).replace('.', '_')}",
        "dataset": "cifar10",
        "seed": seed,
        "stage": "eval",
        "checkpoint_epoch": 499,
        "path_family": "linear",
        "clock_family": "ft_linear_beta",
        "clock_param_name": "beta",
        "clock_param_value": beta,
        "solver": "heun2",
        "nfe": nfe,
        "step_count": 5,
        "real_samples": 50000,
        "synthetic_samples": 50000,
        "metric": "fid",
        "value": value,
        "status": "completed",
        "artifact_group": "ft_clock_linear_main",
    }


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

    def test_resolve_method_fields_supports_best_beta_and_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "experiments" / "results" / "ft_clock_linear_main" / "results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            ensure_results_file(csv_path)
            append_result_rows(
                csv_path,
                [
                    make_source_row("a", 0, 0.3, 9.5, 10),
                    make_source_row("b", 1, 0.3, 9.1, 20),
                    make_source_row("c", 0, 0.5, 8.5, 10),
                    make_source_row("d", 1, 0.5, 8.1, 20),
                ],
            )
            checkpoint = (
                root
                / "experiments"
                / "results"
                / "ft_clock_linear_main"
                / "cifar10"
                / "linear_ft_beta_0_5"
                / "checkpoint-499.pth"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()

            with mock.patch.object(sampling_progression, "ROOT", root):
                resolved = resolve_method_fields(
                    {
                        "label": "Ours",
                        "dataset": "cifar10",
                        "name": "linear_ft_best",
                        "path_family": "linear",
                        "clock_family": "ft_linear_beta",
                        "sampling_solver": "heun2",
                        "eval_nfe": 50,
                        "best_beta_from": {
                            "results_csv": "experiments/results/ft_clock_linear_main/results.csv",
                            "artifact_group": "ft_clock_linear_main",
                            "dataset": "cifar10",
                            "path_family": "linear",
                            "solver": "heun2",
                            "metric": "fid",
                            "selection_nfes": [10, 20],
                            "clock_family": "ft_linear_beta",
                        },
                        "checkpoint_from": {
                            "artifact_group": "ft_clock_linear_main",
                            "source_name_template": "linear_ft_beta_{clock_beta_tag}",
                            "checkpoint_epoch": 499,
                        },
                    }
                )

            self.assertEqual(resolved["clock_beta"], 0.5)
            self.assertEqual(Path(resolved["checkpoint_path"]), checkpoint)


if __name__ == "__main__":
    unittest.main()
