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
    from experiments import plot_particle_trajectory_comparison as particle_trajectory
    from experiments.result_utils import append_result_rows, ensure_results_file
    from experiments.plot_particle_trajectory_comparison import (
        MethodSpec,
        build_method_spec,
        plot_comparison,
    )
else:  # pragma: no cover - environment dependent
    MethodSpec = None
    append_result_rows = None
    build_method_spec = None
    ensure_results_file = None
    plot_comparison = None
    particle_trajectory = None


def make_source_row(run_id, seed, beta, value, nfe):
    return {
        "run_id": run_id,
        "exp_name": f"linear_ft_beta_{str(beta).replace('.', '_')}",
        "dataset": "cifar10",
        "seed": seed,
        "stage": "eval",
        "checkpoint_epoch": 499,
        "path_family": "linear",
        "clock_family": "ft_beta",
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
                    clock_family="ft_beta",
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

    def test_build_method_spec_resolves_best_beta_and_checkpoint(self):
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
            (checkpoint.parent / "args.json").write_text(
                '{"path_family":"linear","clock_family":"ft_beta","clock_beta":0.5,'
                '"clock_semantics_tag":"ft_global_v2_linear_closed_form",'
                '"model_output_type":"base_velocity","time_sampling_strategy":"ds_dr_sq"}',
                encoding="utf-8",
            )

            with mock.patch.object(particle_trajectory, "ROOT", root):
                spec = build_method_spec(
                    {
                        "label": "Ours",
                        "pairing": "oracle",
                        "dataset": "cifar10",
                        "name": "linear_ft_best",
                        "path_family": "linear",
                        "clock_family": "ft_beta",
                        "best_beta_from": {
                            "results_csv": "experiments/results/ft_clock_linear_main/results.csv",
                            "artifact_group": "ft_clock_linear_main",
                            "dataset": "cifar10",
                            "path_family": "linear",
                            "solver": "heun2",
                            "metric": "fid",
                            "selection_nfes": [10, 20],
                            "clock_family": "ft_beta",
                        },
                        "checkpoint_from": {
                            "artifact_group": "ft_clock_linear_main",
                            "source_name_template": "linear_ft_beta_{clock_beta_tag}",
                            "checkpoint_epoch": 499,
                        },
                    }
                )

            self.assertEqual(spec.clock_beta, 0.5)
            self.assertEqual(Path(spec.checkpoint_path), checkpoint)

    def test_build_method_spec_defaults_missing_clock_beta_to_none(self):
        spec = build_method_spec(
            {
                "label": "Baseline",
                "pairing": "random",
                "dataset": "cifar10",
                "name": "linear_uniform",
                "path_family": "linear",
                "clock_family": "uniform",
            }
        )
        self.assertIsNone(spec.clock_beta)


if __name__ == "__main__":
    unittest.main()
