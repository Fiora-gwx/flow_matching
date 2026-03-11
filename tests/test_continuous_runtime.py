import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.continuous_runtime import CLOCK_FAMILIES, build_continuous_batch, evaluate_clock


class ContinuousRuntimeTest(unittest.TestCase):
    def test_clock_endpoints_and_monotonicity(self):
        r = torch.linspace(0.0, 1.0, 257)
        for family in CLOCK_FAMILIES:
            beta = 0.3 if family.startswith('ft_') else None
            clock = evaluate_clock(r=r, clock_family=family, clock_beta=beta)
            self.assertAlmostEqual(float(clock.s[0]), 0.0, places=5)
            self.assertAlmostEqual(float(clock.s[-1]), 1.0, places=5)
            self.assertTrue(torch.all(clock.s[1:] >= clock.s[:-1]))
            self.assertTrue(torch.all(clock.ds_dr >= 0.0))

    def test_build_continuous_batch_matches_linear_endpoint(self):
        samples = torch.ones(4, 3, 2, 2)
        noise = torch.zeros_like(samples)
        r = torch.zeros(4)
        batch = build_continuous_batch(
            x_1=samples,
            x_0=noise,
            r=r,
            path_family='linear',
            clock_family='uniform',
            clock_beta=None,
        )
        self.assertTrue(torch.allclose(batch.x_t, noise))
        self.assertEqual(batch.target_velocity.shape, samples.shape)


if __name__ == '__main__':
    unittest.main()
