import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.eval_utils import iter_batches_until_target


class EvalUtilsTest(unittest.TestCase):
    def test_iter_batches_until_target_restarts_loader(self):
        data_loader = [
            ([0, 1, 2], [0, 0, 0]),
            ([3, 4], [0, 0]),
        ]
        steps = list(iter_batches_until_target(data_loader, target_samples=7))
        self.assertEqual([step for step, _ in steps], [0, 1, 2])
        total = sum(len(batch[0]) for _, batch in steps)
        self.assertGreaterEqual(total, 7)

    def test_iter_batches_until_target_rejects_empty_loader(self):
        with self.assertRaises(ValueError):
            list(iter_batches_until_target([], target_samples=1))
