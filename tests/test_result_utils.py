import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.result_utils import baseline_vs_best_beta, best_rows_by_key


class ResultUtilsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                'dataset': 'cifar10',
                'path_family': 'linear',
                'solver': 'heun2',
                'nfe': 10,
                'clock_family': 'uniform',
                'clock_param_name': 'none',
                'clock_param_value': None,
                'value': 12.0,
            },
            {
                'dataset': 'cifar10',
                'path_family': 'linear',
                'solver': 'heun2',
                'nfe': 10,
                'clock_family': 'ft_linear_beta',
                'clock_param_name': 'beta',
                'clock_param_value': 0.3,
                'value': 11.0,
            },
            {
                'dataset': 'cifar10',
                'path_family': 'linear',
                'solver': 'heun2',
                'nfe': 10,
                'clock_family': 'ft_linear_beta',
                'clock_param_name': 'beta',
                'clock_param_value': 0.5,
                'value': 10.5,
            },
        ]

    def test_best_rows_by_key(self):
        rows = self.rows + [dict(self.rows[-1], value=10.8)]
        best = best_rows_by_key(rows, ['dataset', 'path_family', 'solver', 'clock_family', 'nfe'])
        self.assertEqual(len(best), 2)
        ft_row = [row for row in best if row['clock_family'] == 'ft_linear_beta'][0]
        self.assertEqual(ft_row['value'], 10.5)

    def test_baseline_vs_best_beta(self):
        table = baseline_vs_best_beta(self.rows)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]['baseline_fid'], 12.0)
        self.assertEqual(table[0]['best_fid'], 10.5)
        self.assertEqual(table[0]['best_clock_param_value'], 0.5)


if __name__ == '__main__':
    unittest.main()
