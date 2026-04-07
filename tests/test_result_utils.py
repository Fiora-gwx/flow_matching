import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.result_utils import (
    BASE_RESULT_FIELDS,
    RESULT_FIELDS,
    aggregate_seed_rows,
    append_result_rows,
    baseline_vs_best_beta,
    ensure_results_file,
    infer_clock_parameter,
    resolve_best_beta_reference,
    validate_results_schema,
)


def make_row(
    run_id,
    seed,
    clock_family,
    clock_param_value,
    value,
    nfe=10,
    strategy_id='A',
    model_output_type='velocity',
    time_sampling_strategy='uniform',
    mixed_lambda='',
    stratified_bins='',
):
    return {
        'run_id': run_id,
        'exp_name': f'exp_{clock_family}_{clock_param_value}',
        'dataset': 'cifar10',
        'seed': seed,
        'stage': 'eval',
        'checkpoint_epoch': 499,
        'path_family': 'linear',
        'clock_family': clock_family,
        'clock_param_name': 'beta' if clock_param_value is not None else 'none',
        'clock_param_value': clock_param_value,
        'solver': 'heun2',
        'nfe': nfe,
        'step_count': 5,
        'real_samples': 50000,
        'synthetic_samples': 50000,
        'metric': 'fid',
        'value': value,
        'status': 'completed',
        'artifact_group': 'ft_clock_linear_main',
        'strategy_id': strategy_id,
        'model_output_type': model_output_type,
        'time_sampling_strategy': time_sampling_strategy,
        'mixed_lambda': mixed_lambda,
        'stratified_bins': stratified_bins,
        'requested_eval_nfe': float(nfe),
        'realized_nfe': float(nfe),
    }


class ResultUtilsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            make_row('a0', 0, 'uniform', None, 12.0),
            make_row('a1', 1, 'uniform', None, 10.0),
            make_row('b0', 0, 'ft_beta', 0.3, 11.0),
            make_row('b1', 1, 'ft_beta', 0.3, 9.0),
            make_row('c0', 0, 'ft_beta', 0.5, 10.0),
            make_row('c1', 1, 'ft_beta', 0.5, 8.0),
        ]

    def test_aggregate_seed_rows(self):
        aggregated = aggregate_seed_rows(self.rows)
        self.assertEqual(len(aggregated), 3)
        ft_row = [row for row in aggregated if row['clock_param_value'] == 0.5][0]
        self.assertEqual(ft_row['num_seeds'], 2)
        self.assertEqual(ft_row['value_mean'], 9.0)
        self.assertGreater(ft_row['value_std'], 0.0)

    def test_baseline_vs_best_beta_uses_seed_mean(self):
        table = baseline_vs_best_beta(self.rows)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['baseline_fid_mean'], 11.0)
        self.assertEqual(table[0]['best_fid_mean'], 9.0)
        self.assertEqual(table[0]['best_clock_param_value'], 0.5)
        self.assertEqual(table[0]['best_num_seeds'], 2)

    def test_baseline_vs_best_beta_preserves_aggregated_std(self):
        aggregated = aggregate_seed_rows(self.rows)
        table = baseline_vs_best_beta(aggregated, already_aggregated=True)
        self.assertEqual(len(table), 1)
        self.assertGreater(table[0]['baseline_fid_std'], 0.0)
        self.assertEqual(table[0]['baseline_num_seeds'], 2)
        self.assertGreater(table[0]['best_fid_std'], 0.0)
        self.assertEqual(table[0]['best_num_seeds'], 2)

    def test_infer_clock_parameter(self):
        self.assertEqual(infer_clock_parameter('uniform'), ('none', None))
        self.assertEqual(infer_clock_parameter('ft_beta', 0.4), ('beta', 0.4))
        self.assertEqual(infer_clock_parameter('sigmoid_k8'), ('k', 8.0))

    def test_resolve_best_beta_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            ensure_results_file(csv_path)
            append_result_rows(csv_path, self.rows + [
                make_row('d0', 0, 'ft_beta', 0.3, 15.0, nfe=20),
                make_row('d1', 1, 'ft_beta', 0.3, 13.0, nfe=20),
                make_row('e0', 0, 'ft_beta', 0.5, 9.0, nfe=20),
                make_row('e1', 1, 'ft_beta', 0.5, 11.0, nfe=20),
            ])
            beta = resolve_best_beta_reference(
                {
                    'results_csv': str(csv_path),
                    'dataset': 'cifar10',
                    'path_family': 'linear',
                    'solver': 'heun2',
                    'metric': 'fid',
                    'selection_nfes': [10, 20],
                    'clock_family': 'ft_beta',
                }
            )
            self.assertEqual(beta, 0.5)

    def test_validate_results_schema_rejects_legacy_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            csv_path.write_text(
                'exp_name,dataset,alpha,lambda_scale,epoch,nfe,fid,status\\n'
                'base_0.5,cifar10,0.5,auto,99,4,38.7,completed\\n',
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                validate_results_schema(csv_path)
            with self.assertRaises(ValueError):
                ensure_results_file(csv_path)

    def test_ensure_results_file_writes_header_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            csv_path.write_text('', encoding='utf-8')
            ensure_results_file(csv_path)
            header = csv_path.read_text(encoding='utf-8').splitlines()[0]
            self.assertEqual(header, ','.join(RESULT_FIELDS))

    def test_append_result_rows_after_empty_file_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            csv_path.write_text('', encoding='utf-8')
            append_result_rows(csv_path, [self.rows[0]])
            rows = csv_path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(rows[0], ','.join(RESULT_FIELDS))
            self.assertEqual(len(rows), 2)

    def test_validate_results_schema_accepts_base_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'results.csv'
            csv_path.write_text(','.join(BASE_RESULT_FIELDS) + '\n', encoding='utf-8')
            validate_results_schema(csv_path)

    def test_aggregate_seed_rows_averages_runtime_fields(self):
        rows = [
            {
                **make_row('r0', 0, 'uniform', None, 10.0),
                'solver': 'heun2',
                'requested_eval_nfe': 12.0,
                'realized_nfe': 11.0,
            },
            {
                **make_row('r1', 1, 'uniform', None, 8.0),
                'solver': 'heun2',
                'requested_eval_nfe': 12.0,
                'realized_nfe': 13.0,
            },
        ]
        aggregated = aggregate_seed_rows(rows)
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]['solver'], 'heun2')
        self.assertEqual(aggregated[0]['requested_eval_nfe'], 12.0)
        self.assertEqual(aggregated[0]['realized_nfe'], 12.0)


if __name__ == '__main__':
    unittest.main()
