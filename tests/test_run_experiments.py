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
        'clock_family': 'ft_beta',
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
                    'clock_family': 'ft_beta',
                    'best_beta_from': {
                        'results_csv': str(csv_path),
                        'dataset': 'cifar10',
                        'path_family': 'linear',
                        'solver': 'heun2',
                        'metric': 'fid',
                        'selection_nfes': [10],
                        'clock_family': 'ft_beta',
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
                        'clock_family': 'ft_beta',
                        'best_beta_from': {
                            'results_csv': str(source_csv),
                            'dataset': 'cifar10',
                            'path_family': 'linear',
                            'solver': 'heun2',
                            'metric': 'fid',
                            'selection_nfes': [10],
                            'clock_family': 'ft_beta',
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
            self.assertEqual(rows[0]['strategy_id'], 'A')
            self.assertEqual(rows[0]['time_sampling_strategy'], 'uniform')

    def test_build_train_cmd_adds_resume_for_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'demo_group',
                'base_config': {},
                'experiments': [],
            }), encoding='utf-8')
            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                checkpoint = workspace / 'checkpoint.pth'
                cmd = manager.build_train_cmd(
                    {
                        'dataset': 'cifar10',
                        'data_path': './data/cifar10',
                        'batch_size': 8,
                        'epochs': 2,
                        'seed': 0,
                        'num_gpus': 1,
                        'path_family': 'linear',
                        'clock_family': 'uniform',
                        'sampling_solver': 'heun2',
                        'model_output_type': 'velocity',
                        'time_sampling_strategy': 'uniform',
                    },
                    workspace / 'out',
                    resume_checkpoint=checkpoint,
                )
            finally:
                os.chdir(cwd)
            self.assertIn(f'--resume {checkpoint}', cmd)
            self.assertIn('--eval_frequency -1', cmd)
            self.assertIn('--model_output_type velocity', cmd)
            self.assertIn('--time_sampling_strategy uniform', cmd)

    def test_build_train_cmd_preserves_strategy_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'demo_group',
                'base_config': {},
                'experiments': [],
            }), encoding='utf-8')
            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                cmd = manager.build_train_cmd(
                    {
                        'dataset': 'cifar10',
                        'data_path': './data/cifar10',
                        'batch_size': 8,
                        'epochs': 2,
                        'seed': 0,
                        'num_gpus': 1,
                        'path_family': 'linear',
                        'clock_family': 'ft_beta',
                        'clock_beta': 0.3,
                        'sampling_solver': 'heun2',
                        'model_output_type': 'velocity',
                        'time_sampling_strategy': 'stratified_mixed',
                        'mixed_lambda': 0.7,
                        'stratified_bins': 16,
                    },
                    workspace / 'out',
                )
            finally:
                os.chdir(cwd)
            self.assertIn('--model_output_type velocity', cmd)
            self.assertIn('--time_sampling_strategy stratified_mixed', cmd)
            self.assertIn('--mixed_lambda 0.7', cmd)
            self.assertIn('--stratified_bins 16', cmd)

    def test_build_train_cmd_preserves_accum_iter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'demo_group',
                'base_config': {},
                'experiments': [],
            }), encoding='utf-8')
            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                cmd = manager.build_train_cmd(
                    {
                        'dataset': 'cifar10',
                        'data_path': './data/cifar10',
                        'batch_size': 128,
                        'epochs': 2,
                        'seed': 0,
                        'num_gpus': 2,
                        'accum_iter': 2,
                        'path_family': 'linear',
                        'clock_family': 'ft_beta',
                        'clock_beta': 0.3,
                        'sampling_solver': 'heun2',
                        'model_output_type': 'velocity',
                        'time_sampling_strategy': 'uniform',
                    },
                    workspace / 'out',
                )
            finally:
                os.chdir(cwd)
            self.assertIn('--accum_iter 2', cmd)

    def test_build_eval_cmd_includes_solver_aware_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'demo_group',
                'base_config': {},
                'experiments': [],
            }), encoding='utf-8')
            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                cmd = manager.build_eval_cmd(
                    {
                        'dataset': 'cifar10',
                        'data_path': './data/cifar10',
                        'batch_size': 8,
                        'epochs': 2,
                        'seed': 0,
                        'num_gpus': 1,
                        'path_family': 'linear',
                        'clock_family': 'uniform',
                        'sampling_solver': 'stork4',
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                        'metrics': ['fid'],
                        'solver_aware_clock_mode': 'training_free',
                        'solver_aware_target_solver': 'stork4',
                        'solver_aware_k': 0,
                        'solver_aware_monitor_estimator': 'auto',
                        'solver_aware_monitor_grid_size': 65,
                        'solver_aware_monitor_batch_size': 64,
                        'solver_aware_eps': 1e-6,
                        'solver_aware_use_nodes': True,
                    },
                    workspace / 'out',
                    workspace / 'checkpoint.pth',
                    12,
                )
            finally:
                os.chdir(cwd)
            self.assertIn('--solver_aware_clock_mode training_free', cmd)
            self.assertIn('--solver_aware_target_solver stork4', cmd)
            self.assertIn('--solver_aware_use_nodes', cmd)
            self.assertIn('--solver_aware_monitor_estimator auto', cmd)

    def test_experiment_manager_resumes_training_when_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
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
                        'clock_family': 'uniform',
                    }
                ],
            }), encoding='utf-8')

            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                checkpoint = manager.base_dir / 'cifar10' / 'demo_ft' / 'checkpoint.pth'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                (checkpoint.parent / 'args.json').write_text(
                    json.dumps(
                        {
                            'path_family': 'linear',
                            'clock_family': 'uniform',
                            'model_output_type': 'velocity',
                            'time_sampling_strategy': 'uniform',
                        }
                    ),
                    encoding='utf-8',
                )
                manager.state['demo_ft:train'] = 'failed'
                manager._save_state()
                seen_resume = {'value': False}

                def fake_run_command(cmd, log_file, retries=0):
                    tokens = shlex.split(cmd)
                    if '--eval_only' not in tokens:
                        self.assertIn('--resume', tokens)
                        self.assertEqual(tokens[tokens.index('--resume') + 1], str(checkpoint))
                        seen_resume['value'] = True
                    return True

                with mock.patch('experiments.run_experiments.run_command', side_effect=fake_run_command):
                    with mock.patch('experiments.run_experiments.extract_eval_stats', return_value={
                        'nfe': 10.0,
                        'step_count': 5.0,
                        'real_samples': 50000.0,
                        'synthetic_samples': 50000.0,
                        'fid': 9.5,
                    }):
                        manager.run_all()
            finally:
                os.chdir(cwd)

            self.assertTrue(seen_resume['value'])

    def test_experiment_manager_backfills_missing_inception_score_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
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
                    'metrics': ['fid', 'inception_score'],
                    'eval_epochs': [1],
                    'eval_nfes': [10],
                },
                'experiments': [
                    {
                        'name': 'demo_ft',
                        'clock_family': 'uniform',
                    }
                ],
            }), encoding='utf-8')

            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                checkpoint = manager.base_dir / 'cifar10' / 'demo_ft' / 'checkpoint.pth'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                (checkpoint.parent / 'args.json').write_text(
                    json.dumps(
                        {
                            'path_family': 'linear',
                            'clock_family': 'uniform',
                            'model_output_type': 'velocity',
                            'time_sampling_strategy': 'uniform',
                        }
                    ),
                    encoding='utf-8',
                )
                manager.state['demo_ft:train'] = 'completed'
                manager.state['demo_ft:ep1:nfe10'] = 'completed'
                manager._save_state()
                append_result_rows(manager.results_csv, [{
                    'run_id': 'demo_ft:ep1:nfe10:fid',
                    'exp_name': 'demo_ft',
                    'dataset': 'cifar10',
                    'seed': 0,
                    'stage': 'eval',
                    'checkpoint_epoch': 1,
                    'path_family': 'linear',
                    'clock_family': 'uniform',
                    'clock_param_name': 'none',
                    'clock_param_value': None,
                    'solver': 'heun2',
                    'nfe': 10,
                    'step_count': 5,
                    'real_samples': 50000,
                    'synthetic_samples': 50000,
                    'metric': 'fid',
                    'value': 9.5,
                    'status': 'completed',
                    'artifact_group': 'demo_group',
                }])
                seen_commands = []

                def fake_run_command(cmd, log_file, retries=0):
                    seen_commands.append(cmd)
                    tokens = shlex.split(cmd)
                    self.assertIn('--eval_only', tokens)
                    self.assertIn('--metrics', tokens)
                    self.assertEqual(tokens[tokens.index('--metrics') + 1], 'inception_score')
                    self.assertNotIn('--compute_fid', tokens)
                    return True

                with mock.patch('experiments.run_experiments.run_command', side_effect=fake_run_command):
                    with mock.patch('experiments.run_experiments.extract_eval_stats', return_value={
                        'nfe': 10.0,
                        'step_count': 5.0,
                        'real_samples': 50000.0,
                        'synthetic_samples': 50000.0,
                        'is_mean': 8.3,
                        'is_std': 0.2,
                    }):
                        manager.run_all()
            finally:
                os.chdir(cwd)

            self.assertEqual(len(seen_commands), 1)
            rows = load_result_rows(workspace / 'experiments' / 'results' / 'demo_group' / 'results.csv')
            metric_names = sorted(row['metric'] for row in rows)
            self.assertEqual(metric_names, ['fid', 'is_mean', 'is_std'])

    def test_experiment_manager_ignores_stale_local_checkpoint_with_mismatched_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
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
                        'clock_family': 'ft_beta',
                        'clock_beta': 0.5,
                        'model_output_type': 'velocity',
                        'time_sampling_strategy': 'uniform',
                    }
                ],
            }), encoding='utf-8')

            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                manager = ExperimentManager(config_path)
                checkpoint = manager.base_dir / 'cifar10' / 'demo_ft' / 'checkpoint.pth'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                (checkpoint.parent / 'args.json').write_text(
                    json.dumps(
                        {
                            'path_family': 'linear',
                            'clock_family': 'ft_beta',
                            'clock_beta': 0.5,
                            'clock_semantics_tag': 'ft_global_v2_linear_closed_form',
                            'model_output_type': 'velocity',
                            'time_sampling_strategy': 'mixed_lambda',
                            'mixed_lambda': 0.5,
                        }
                    ),
                    encoding='utf-8',
                )
                manager.state['demo_ft:train'] = 'completed'
                manager._save_state()
                seen_commands = []

                def fake_run_command(cmd, log_file, retries=0):
                    seen_commands.append(cmd)
                    tokens = shlex.split(cmd)
                    output_dir = Path(tokens[tokens.index('--output_dir') + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if '--eval_only' not in tokens:
                        self.assertNotIn('--resume', tokens)
                        (output_dir / 'checkpoint.pth').touch()
                        (output_dir / 'args.json').write_text(
                            json.dumps(
                                {
                                    'path_family': 'linear',
                                    'clock_family': 'ft_beta',
                                    'clock_beta': 0.5,
                                    'clock_semantics_tag': 'ft_global_v2_linear_closed_form',
                                    'model_output_type': 'velocity',
                                    'time_sampling_strategy': 'uniform',
                                }
                            ),
                            encoding='utf-8',
                        )
                    return True

                with mock.patch('experiments.run_experiments.run_command', side_effect=fake_run_command):
                    with mock.patch('experiments.run_experiments.extract_eval_stats', return_value={
                        'nfe': 10.0,
                        'step_count': 5.0,
                        'real_samples': 50000.0,
                        'synthetic_samples': 50000.0,
                        'fid': 9.5,
                    }):
                        manager.run_all()
            finally:
                os.chdir(cwd)

            self.assertEqual(len(seen_commands), 2)
            self.assertNotIn('--eval_only', seen_commands[0])
            rows = load_result_rows(workspace / 'experiments' / 'results' / 'demo_group' / 'results.csv')
            self.assertEqual(rows[0]['strategy_id'], 'A')
            self.assertEqual(rows[0]['time_sampling_strategy'], 'uniform')

    def test_experiment_manager_reuses_external_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            config_path = workspace / 'config.json'
            config_path.write_text(json.dumps({
                'experiment_name': 'cross_path_group',
                'base_config': {
                    'dataset': 'cifar10',
                    'data_path': './data/cifar10',
                    'epochs': 500,
                    'batch_size': 8,
                    'num_gpus': 1,
                    'path_family': 'linear',
                    'sampling_solver': 'heun2',
                    'metrics': ['fid'],
                    'eval_epochs': [499],
                    'eval_nfes': [10],
                },
                'experiments': [
                    {
                        'name': 'linear_ft_best',
                        'path_family': 'linear',
                        'clock_family': 'ft_beta',
                        'clock_beta': 0.5,
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                        'checkpoint_from': {
                            'artifact_group': 'ft_clock_linear_main',
                            'source_name_template': 'linear_ft_beta_{clock_beta_tag}',
                            'checkpoint_epoch': 499,
                        },
                    }
                ],
            }), encoding='utf-8')

            external_checkpoint = (
                workspace
                / 'experiments'
                / 'results'
                / 'ft_clock_linear_main'
                / 'cifar10'
                / 'linear_ft_beta_0_5'
                / 'checkpoint-499.pth'
            )
            external_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            external_checkpoint.touch()
            (external_checkpoint.parent / 'args.json').write_text(
                json.dumps(
                    {
                        'path_family': 'linear',
                        'clock_family': 'ft_beta',
                        'clock_beta': 0.5,
                        'clock_semantics_tag': 'ft_global_v2_linear_closed_form',
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                    }
                ),
                encoding='utf-8',
            )

            cwd = os.getcwd()
            os.chdir(workspace)
            try:
                seen_commands = []

                def fake_run_command(cmd, log_file, retries=0):
                    seen_commands.append(cmd)
                    tokens = shlex.split(cmd)
                    self.assertIn('--eval_only', tokens)
                    self.assertIn('--resume', tokens)
                    self.assertEqual(
                        Path(tokens[tokens.index('--resume') + 1]).resolve(),
                        external_checkpoint.resolve(),
                    )
                    return True

                with mock.patch('experiments.run_experiments.run_command', side_effect=fake_run_command):
                    with mock.patch('experiments.run_experiments.extract_eval_stats', return_value={
                        'nfe': 10.0,
                        'step_count': 5.0,
                        'real_samples': 50000.0,
                        'synthetic_samples': 50000.0,
                        'fid': 9.1,
                    }):
                        ExperimentManager(config_path).run_all()
            finally:
                os.chdir(cwd)

            self.assertEqual(len(seen_commands), 1)
            self.assertIn('--eval_only', seen_commands[0])
