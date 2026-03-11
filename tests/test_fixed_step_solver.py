import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.fixed_step_solver import build_step_methods, solve_fixed_budget


class DummyModel:
    def __init__(self):
        self.nfe = 0

    def __call__(self, x, t, **kwargs):
        self.nfe += 1
        return torch.ones_like(x)

    def reset_nfe_counter(self):
        self.nfe = 0

    def get_nfe(self):
        return self.nfe


class FixedStepSolverTest(unittest.TestCase):
    def test_build_step_methods_exact_budget(self):
        self.assertEqual(build_step_methods('euler', 3), ('euler', 'euler', 'euler'))
        self.assertEqual(build_step_methods('heun2', 4), ('heun2', 'heun2'))
        self.assertEqual(build_step_methods('heun2', 5), ('heun2', 'heun2', 'euler'))

    def test_solver_counts_real_nfe(self):
        model = DummyModel()
        x_init = torch.zeros(2, 3)
        result = solve_fixed_budget(model, x_init, 'heun2', 5)
        self.assertEqual(result.nfe, 5)
        self.assertEqual(result.step_count, 3)


if __name__ == '__main__':
    unittest.main()
