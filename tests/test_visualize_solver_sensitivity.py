import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _FakePyplot(types.ModuleType):
    def __init__(self):
        super().__init__("matplotlib.pyplot")

    def figure(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def fill_between(self, *args, **kwargs):
        return None

    def xlabel(self, *args, **kwargs):
        return None

    def ylabel(self, *args, **kwargs):
        return None

    def xticks(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def savefig(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-figure", encoding="utf-8")

    def close(self, *args, **kwargs):
        return None


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

fake_matplotlib = types.ModuleType("matplotlib")
fake_matplotlib.use = lambda *args, **kwargs: None
fake_pyplot = _FakePyplot()
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)

from experiments.result_utils import RESULT_FIELDS
from experiments.visualize_solver_sensitivity import visualize_solver_sensitivity


def make_metric_rows(exp_name, clock_family, beta, solver, nfe, metric_values):
    rows = []
    for metric_name, value in metric_values.items():
        rows.append(
            {
                "run_id": f"{exp_name}:{solver}:{nfe}:{metric_name}",
                "exp_name": exp_name,
                "dataset": "cifar10",
                "seed": 0,
                "stage": "eval",
                "checkpoint_epoch": 499,
                "path_family": "linear",
                "clock_family": clock_family,
                "clock_param_name": "none" if beta is None else "beta",
                "clock_param_value": "" if beta is None else beta,
                "solver": solver,
                "nfe": nfe,
                "step_count": nfe if solver == "stork4" else max(1, nfe // 2),
                "real_samples": 50000,
                "synthetic_samples": 50000,
                "metric": metric_name,
                "value": value,
                "status": "completed",
                "artifact_group": "group",
            }
        )
    return rows


class VisualizeSolverSensitivityTest(unittest.TestCase):
    def test_visualize_solver_sensitivity_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            out_dir = Path(tmpdir) / "plots"
            rows = []
            rows.extend(
                make_metric_rows(
                    "linear_uniform_heun2",
                    "uniform",
                    None,
                    "heun2",
                    12,
                    {
                        "fid": 10.0,
                        "precision": 0.70,
                        "recall": 0.60,
                        "is_mean": 8.2,
                        "is_std": 0.2,
                    },
                )
            )
            rows.extend(
                make_metric_rows(
                    "linear_ft_beta_0_3_heun2",
                    "ft_linear_beta",
                    0.3,
                    "heun2",
                    12,
                    {
                        "fid": 8.5,
                        "precision": 0.74,
                        "recall": 0.63,
                        "is_mean": 8.8,
                        "is_std": 0.1,
                    },
                )
            )
            rows.extend(
                make_metric_rows(
                    "linear_uniform_rk3",
                    "uniform",
                    None,
                    "rk3",
                    12,
                    {
                        "fid": 9.8,
                        "precision": 0.71,
                        "recall": 0.59,
                        "is_mean": 8.1,
                        "is_std": 0.2,
                    },
                )
            )
            rows.extend(
                make_metric_rows(
                    "linear_ft_beta_0_3_stork4",
                    "ft_linear_beta",
                    0.3,
                    "stork4",
                    10,
                    {
                        "fid": 8.1,
                        "precision": 0.75,
                        "recall": 0.64,
                        "is_mean": 8.9,
                        "is_std": 0.1,
                    },
                )
            )

            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            visualize_solver_sensitivity(csv_path, out_dir, artifact_group="group")

            self.assertTrue((out_dir / "solver_fairness_fid_vs_nfe.png").exists())
            self.assertTrue((out_dir / "solver_fairness_table.csv").exists())
            self.assertTrue((out_dir / "solver_appendix_all_budgets.csv").exists())
            self.assertTrue((out_dir / "solver_sensitivity_summary.md").exists())

            with open(out_dir / "solver_fairness_table.csv", "r", newline="", encoding="utf-8") as handle:
                fairness_rows = list(csv.DictReader(handle))
            self.assertTrue(fairness_rows)
            self.assertTrue(all(row["nfe"] != "10" for row in fairness_rows))
            self.assertIn("actual_network_calls", fairness_rows[0])

            with open(out_dir / "solver_appendix_all_budgets.csv", "r", newline="", encoding="utf-8") as handle:
                appendix_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["nfe"] == "10" for row in appendix_rows))


if __name__ == "__main__":
    unittest.main()
