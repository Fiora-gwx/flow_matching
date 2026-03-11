import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _FakePyplot(types.ModuleType):
    def __init__(self):
        super().__init__('matplotlib.pyplot')
        self._target = None

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

    def xscale(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def imshow(self, *args, **kwargs):
        return None

    def colorbar(self, *args, **kwargs):
        return None

    def xticks(self, *args, **kwargs):
        return None

    def yticks(self, *args, **kwargs):
        return None

    def savefig(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text('fake-figure', encoding='utf-8')

    def close(self, *args, **kwargs):
        return None


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

fake_matplotlib = types.ModuleType('matplotlib')
fake_matplotlib.use = lambda *args, **kwargs: None
fake_pyplot = _FakePyplot()
sys.modules.setdefault('matplotlib', fake_matplotlib)
sys.modules.setdefault('matplotlib.pyplot', fake_pyplot)

from experiments.result_utils import RESULT_FIELDS
from experiments.visualize_results import visualize_results


class VisualizeResultsTest(unittest.TestCase):
    def test_visualize_results_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            out_dir = Path(tmpdir) / 'plots'
            rows = [
                {
                    'run_id': 'a0',
                    'exp_name': 'linear_uniform',
                    'dataset': 'cifar10',
                    'seed': 0,
                    'stage': 'eval',
                    'checkpoint_epoch': 499,
                    'path_family': 'linear',
                    'clock_family': 'uniform',
                    'clock_param_name': 'none',
                    'clock_param_value': '',
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 12.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
                {
                    'run_id': 'a1',
                    'exp_name': 'linear_uniform',
                    'dataset': 'cifar10',
                    'seed': 1,
                    'stage': 'eval',
                    'checkpoint_epoch': 499,
                    'path_family': 'linear',
                    'clock_family': 'uniform',
                    'clock_param_name': 'none',
                    'clock_param_value': '',
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 10.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
                {
                    'run_id': 'b0',
                    'exp_name': 'linear_ft',
                    'dataset': 'cifar10',
                    'seed': 0,
                    'stage': 'eval',
                    'checkpoint_epoch': 499,
                    'path_family': 'linear',
                    'clock_family': 'ft_linear_beta',
                    'clock_param_name': 'beta',
                    'clock_param_value': 0.3,
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 10.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
                {
                    'run_id': 'c',
                    'exp_name': 'vp_uniform',
                    'dataset': 'cifar10',
                    'seed': 0,
                    'stage': 'eval',
                    'checkpoint_epoch': 499,
                    'path_family': 'trig_vp',
                    'clock_family': 'uniform',
                    'clock_param_name': 'none',
                    'clock_param_value': '',
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 11.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
            ]
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            visualize_results(csv_path, out_dir, artifact_group='group')
            self.assertTrue((out_dir / 'baseline_vs_best_ft.csv').exists())
            self.assertTrue((out_dir / 'cross_path_table.csv').exists())
            self.assertTrue((out_dir / 'fid_vs_nfe.png').exists())
            with open(out_dir / 'baseline_vs_best_ft.csv', 'r', newline='', encoding='utf-8') as handle:
                table_rows = list(csv.DictReader(handle))
            self.assertEqual(len(table_rows), 1)
            self.assertEqual(table_rows[0]['baseline_num_seeds'], '2')
            self.assertNotEqual(table_rows[0]['baseline_fid_std'], '0.0')

    def test_visualize_results_heatmap_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            out_dir = Path(tmpdir) / 'plots'
            rows = [
                {
                    'run_id': 'a',
                    'exp_name': 'linear_ft',
                    'dataset': 'cifar10',
                    'seed': 0,
                    'stage': 'eval',
                    'checkpoint_epoch': 499,
                    'path_family': 'linear',
                    'clock_family': 'ft_linear_beta',
                    'clock_param_name': 'beta',
                    'clock_param_value': 0.3,
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 10.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
            ]
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            visualize_results(csv_path, out_dir, artifact_group='group', plot_heatmap_only=True)
            self.assertTrue((out_dir / 'fid_heatmap_beta_nfe.png').exists())
            self.assertFalse((out_dir / 'baseline_vs_best_ft.csv').exists())


if __name__ == '__main__':
    unittest.main()
