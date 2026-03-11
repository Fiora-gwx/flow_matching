import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.result_utils import RESULT_FIELDS
from experiments.visualize_results import visualize_results


class VisualizeResultsTest(unittest.TestCase):
    def test_visualize_results_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            out_dir = Path(tmpdir) / 'plots'
            rows = [
                {
                    'run_id': 'a',
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
                    'metric': 'fid',
                    'value': 12.0,
                    'status': 'completed',
                    'artifact_group': 'group',
                },
                {
                    'run_id': 'b',
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


if __name__ == '__main__':
    unittest.main()
