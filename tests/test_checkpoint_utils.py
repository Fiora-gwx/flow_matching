import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.checkpoint_utils import (
    find_checkpoint,
    resolve_checkpoint_path,
    resolve_reused_checkpoint,
)


class CheckpointUtilsTest(unittest.TestCase):
    def test_find_checkpoint_prefers_epoch_specific(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            epoch_path = exp_dir / 'checkpoint-10.pth'
            latest_path = exp_dir / 'checkpoint.pth'
            epoch_path.touch()
            latest_path.touch()
            self.assertEqual(find_checkpoint(exp_dir, 10), epoch_path)

    def test_resolve_checkpoint_path_supports_explicit_and_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / 'experiments' / 'results' / 'group'
            exp_dir = base_dir / 'cifar10' / 'demo'
            exp_dir.mkdir(parents=True)
            epoch_path = exp_dir / 'checkpoint-20.pth'
            explicit_path = root / 'manual.pth'
            epoch_path.touch()
            explicit_path.touch()
            self.assertEqual(
                resolve_checkpoint_path(base_dir, {'dataset': 'cifar10', 'name': 'demo', 'checkpoint_epoch': 20}),
                epoch_path,
            )
            self.assertEqual(
                resolve_checkpoint_path(base_dir, {'dataset': 'cifar10', 'name': 'demo', 'checkpoint_path': str(explicit_path)}, workspace_root=root),
                explicit_path,
            )

    def test_find_checkpoint_supports_underscore_and_latest_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            underscore_epoch_path = exp_dir / 'checkpoint_20.pth'
            latest_alias_path = exp_dir / 'checkpoint_latest.pth'
            underscore_epoch_path.touch()
            latest_alias_path.touch()
            self.assertEqual(find_checkpoint(exp_dir, 20), underscore_epoch_path)
            underscore_epoch_path.unlink()
            self.assertEqual(find_checkpoint(exp_dir, 20, warn_on_fallback=False), latest_alias_path)

    def test_resolve_reused_checkpoint_supports_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / 'experiments' / 'results' / 'ft_clock_linear_main' / 'cifar10' / 'linear_ft_beta_0_5'
            base_dir.mkdir(parents=True)
            checkpoint = base_dir / 'checkpoint-499.pth'
            checkpoint.touch()
            (base_dir / 'args.json').write_text(
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
            resolved = resolve_reused_checkpoint(
                reference={
                    'artifact_group': 'ft_clock_linear_main',
                    'source_name_template': 'linear_ft_beta_{clock_beta_tag}',
                    'checkpoint_epoch': 499,
                },
                spec={
                    'dataset': 'cifar10',
                    'name': 'linear_ft_best',
                    'path_family': 'linear',
                    'clock_family': 'ft_beta',
                    'clock_beta': 0.5,
                    'model_output_type': 'base_velocity',
                    'time_sampling_strategy': 'ds_dr_sq',
                    'mixed_lambda': 0.5,
                    'stratified_bins': 16,
                },
                workspace_root=root,
            )
            self.assertEqual(resolved, checkpoint)

    def test_resolve_reused_checkpoint_supports_flat_result_directory_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / 'experiments' / 'results' / 'ft_clock_linear_main' / 'linear_uniform'
            exp_dir.mkdir(parents=True)
            checkpoint = exp_dir / 'checkpoint-499.pth'
            checkpoint.touch()
            (exp_dir / 'args.json').write_text(
                json.dumps(
                    {
                        'path_family': 'linear',
                        'clock_family': 'uniform',
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                    }
                ),
                encoding='utf-8',
            )
            resolved = resolve_reused_checkpoint(
                reference={
                    'artifact_group': 'ft_clock_linear_main',
                    'source_exp_name': 'linear_uniform',
                    'checkpoint_epoch': 499,
                },
                spec={
                    'dataset': 'cifar10',
                    'name': 'linear_uniform_euler_shared_clock',
                    'path_family': 'linear',
                    'clock_family': 'uniform',
                    'model_output_type': 'base_velocity',
                    'time_sampling_strategy': 'ds_dr_sq',
                    'mixed_lambda': 0.5,
                    'stratified_bins': 16,
                },
                workspace_root=root,
            )
            self.assertEqual(resolved, checkpoint)

    def test_resolve_reused_checkpoint_rejects_legacy_linear_ft_training_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / 'experiments' / 'results' / 'ft_clock_linear_main' / 'cifar10' / 'linear_ft_beta_0_5'
            exp_dir.mkdir(parents=True)
            checkpoint = exp_dir / 'checkpoint-499.pth'
            checkpoint.touch()
            (exp_dir / 'args.json').write_text(
                json.dumps(
                    {
                        'path_family': 'linear',
                        'clock_family': 'ft_linear_beta',
                        'clock_beta': 0.5,
                    }
                ),
                encoding='utf-8',
            )
            resolved = resolve_reused_checkpoint(
                reference={
                    'artifact_group': 'ft_clock_linear_main',
                    'source_name_template': 'linear_ft_beta_{clock_beta_tag}',
                    'checkpoint_epoch': 499,
                },
                spec={
                    'dataset': 'cifar10',
                    'name': 'linear_ft_best',
                    'path_family': 'linear',
                    'clock_family': 'ft_beta',
                    'clock_beta': 0.5,
                    'model_output_type': 'base_velocity',
                    'time_sampling_strategy': 'ds_dr_sq',
                    'mixed_lambda': 0.5,
                    'stratified_bins': 16,
                },
                workspace_root=root,
            )
            self.assertIsNone(resolved)

    def test_resolve_reused_checkpoint_logs_missing_path_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertLogs('experiments.checkpoint_utils', level='WARNING') as logs:
                resolved = resolve_reused_checkpoint(
                    reference={
                        'artifact_group': 'ft_clock_linear_main',
                        'source_exp_name': 'linear_uniform',
                        'checkpoint_epoch': 499,
                    },
                    spec={
                        'dataset': 'cifar10',
                        'name': 'linear_uniform_euler_shared_clock',
                        'path_family': 'linear',
                        'clock_family': 'uniform',
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                        'mixed_lambda': 0.5,
                        'stratified_bins': 16,
                    },
                    workspace_root=root,
                )
            self.assertIsNone(resolved)
            joined_logs = "\n".join(logs.output)
            self.assertIn('did not resolve to an existing file', joined_logs)
            self.assertIn("artifact_group=ft_clock_linear_main", joined_logs)
            self.assertIn("source_name=linear_uniform", joined_logs)
            self.assertIn("candidate_paths=", joined_logs)

    def test_resolve_reused_checkpoint_logs_semantic_mismatch_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / 'experiments' / 'results' / 'ft_clock_linear_main' / 'cifar10' / 'linear_uniform'
            exp_dir.mkdir(parents=True)
            checkpoint = exp_dir / 'checkpoint-499.pth'
            checkpoint.touch()
            (exp_dir / 'args.json').write_text(
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
            with self.assertLogs('experiments.checkpoint_utils', level='WARNING') as logs:
                resolved = resolve_reused_checkpoint(
                    reference={
                        'artifact_group': 'ft_clock_linear_main',
                        'source_exp_name': 'linear_uniform',
                        'checkpoint_epoch': 499,
                    },
                    spec={
                        'dataset': 'cifar10',
                        'name': 'linear_uniform_euler_shared_clock',
                        'path_family': 'linear',
                        'clock_family': 'uniform',
                        'model_output_type': 'base_velocity',
                        'time_sampling_strategy': 'ds_dr_sq',
                        'mixed_lambda': 0.5,
                        'stratified_bins': 16,
                    },
                    workspace_root=root,
                )
            self.assertIsNone(resolved)
            joined_logs = "\n".join(logs.output)
            self.assertIn('checkpoint semantics do not match the requested spec', joined_logs)
            self.assertIn("artifact_group=ft_clock_linear_main", joined_logs)
            self.assertIn("source_name=linear_uniform", joined_logs)
            self.assertIn("candidate_paths=", joined_logs)

    def test_resolve_reused_checkpoint_rejects_legacy_trig_vp_ft_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / 'experiments' / 'results' / 'ft_clock_trig_vp' / 'cifar10' / 'trig_vp_ft_beta_0_5'
            exp_dir.mkdir(parents=True)
            checkpoint = exp_dir / 'checkpoint-499.pth'
            checkpoint.touch()
            (exp_dir / 'args.json').write_text(
                json.dumps(
                    {
                        'path_family': 'trig_vp',
                        'clock_family': 'ft_vp_beta',
                        'clock_beta': 0.5,
                    }
                ),
                encoding='utf-8',
            )
            resolved = resolve_reused_checkpoint(
                reference={
                    'artifact_group': 'ft_clock_trig_vp',
                    'source_name_template': 'trig_vp_ft_beta_{clock_beta_tag}',
                    'checkpoint_epoch': 499,
                },
                spec={
                    'dataset': 'cifar10',
                    'name': 'trig_vp_ft_best',
                    'path_family': 'trig_vp',
                    'clock_family': 'ft_beta',
                    'clock_beta': 0.5,
                    'model_output_type': 'base_velocity',
                    'time_sampling_strategy': 'ds_dr_sq',
                    'mixed_lambda': 0.5,
                    'stratified_bins': 16,
                },
                workspace_root=root,
            )
            self.assertIsNone(resolved)
