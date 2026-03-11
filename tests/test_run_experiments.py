import json
import os
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

fake_yaml = types.ModuleType('yaml')
fake_yaml.safe_load = lambda handle: json.load(handle)
sys.modules.setdefault('yaml', fake_yaml)

from experiments.result_utils import append_result_rows, ensure_results_file, load_result_rows
from experiments.run_experiments import ExperimentManager, resolve_dynamic_spec_fields


def make_source_row(run_id, seed, beta, value, nfe):
    return {
        'run_id': run_id,
        'exp_name': f'linear_ft_beta_{beta}',
        'dataset': 'cifar10',
        'seed': seed,
        'stage': 'eval',
        'checkpoint_epoch': 499,
        'path_family': 'linear',
        'clock_family': 'ft_linear_beta',
        'clock_param_name': 'beta',
        'clock_param_value': beta,
        'solver': 'heun2',
        'nfe': nfe,
        'step_count': 5,
        'real_samples': 50000,
        'synthetic_samples': 50000,
        'metric': 'fid',
        'value': value,
        'status': 'completed',
        'artifact_group': 'ft_clock_linear_main',
    }


class RunExperimentsTest(unittest.TestCase):
    def test_resolve_dynamic_spec_fields_reads_best_beta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'source.csv'
            ensure_results_file(csv_path)
            append_result_rows(csv_path, [
                make_source_row('a', 0, 0.3, 11.0, 10),
                make_source_row('b', 1, 0.3, 9.0, 10),
                make_source_row('c', 0, 0.5, 10.0, 10),
                make_source_row('d', 1, 0.5, 8.0, 10),
            ])
            spec = resolve_dynamic_spec_fields(
                {
                    'name': 'target',
                    'clock_family': 'ft_linear_beta',
                    'best_beta_from': {
                        'results_csv': str(csv_path),
                        'dataset': 'cifar10',
                        'path_family': 'linear',
                        'solver': 'heun2',
                        'metric': 'fid',
                        'selection_nfes': [10],
                        'clock_family': 'ft_linear_beta',
                    },
                }
            )
            self.assertEqual(spec['clock_beta'], 0.5)

    def test_experiment_manager_writes_results_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            source_csv = workspace / 'source.csv'
            ensure_results_file(source_csv)
            append_result_rows(source_csv, [
                make_source_row('a', 0, 0.3, 11.0, 10),
                make_source_row('b', 1, 0.3, 9.0, 10),
                make_source_row('c', 0, 0.5, 10.0, 10),
                make_source_row('d', 1, 0.5, 8.0, 10),
            ])
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'demo_group',
                'base_config': {
                    'dataset': 'cifar10',
                    'data_path': './data/cifar10',
                    'epochs': 2,
                    'batch_size': 8,
                    'num_gpus': 1,
                    'path_family': 'linear',
                    'sampling_solver': 'heun2',
                    'metrics': ['fid'],
                    'eval_epochs': [1],
                    'eval_nfes': [10],
                },
                'experiments': [
                    {
                        'name': 'demo_ft',
                        'clock_family': 'ft_linear_beta',
                        'best_beta_from': {
                            'results_csv': str(source_csv),
                            'dataset': 'cifar10',
                            'path_family': 'linear',
                            'solver': 'heun2',
                            'metric': 'fid',
                            'selection_nfes': [10],
                            'clock_family': 'ft_linear_beta',
                        },
                    }
                ],
            }), encoding='utf-8')

            def fake_run_command(cmd, log_file, retries=0):
                tokens = shlex.split(cmd)
                self.assertIn('--output_dir', tokens)
                output_dir = Path(tokens[tokens.index('--output_dir') + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                if '--eval_only' not in tokens:
                    (output_dir / 'checkpoint.pth').touch()
                return True

            with mock.patch('experiments.run_experiments.run_command', side_effect=fake_run_command):
                with mock.patch('experiments.run_experiments.extract_eval_stats', return_value={
                    'nfe': 10.0,
                    'step_count': 5.0,
                    'real_samples': 50000.0,
                    'synthetic_samples': 50000.0,
                    'fid': 9.5,
                }):
                    cwd = os.getcwd()
                    os.chdir(workspace)
                    try:
                        ExperimentManager(config_path).run_all()
                    finally:
                        os.chdir(cwd)

            rows = load_result_rows(workspace / 'experiments' / 'results' / 'demo_group' / 'results.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['clock_param_value'], 0.5)
            self.assertEqual(rows[0]['real_samples'], 50000)
            self.assertEqual(rows[0]['synthetic_samples'], 50000)
