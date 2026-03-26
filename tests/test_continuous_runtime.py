import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, 'examples', 'image')
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.continuous_runtime import (
    CLOCK_FAMILIES,
    build_continuous_batch,
    clamp_time_inside_unit_interval,
    evaluate_clock,
    evaluate_mean_terminal_error,
    evaluate_path,
    model_output_to_base_velocity,
    model_output_to_velocity,
    sample_importance_weighted_time,
    sample_strict_unit_interval,
)


class ContinuousRuntimeTest(unittest.TestCase):
    def test_clock_endpoints_and_monotonicity(self):
        r = torch.linspace(0.0, 1.0, 257)
        for family in CLOCK_FAMILIES:
            beta = 0.3 if family == 'ft_beta' else None
            clock = evaluate_clock(
                r=r,
                clock_family=family,
                clock_beta=beta,
                path_family='linear',
                signal_scale_sq=0.75,
            )
            self.assertAlmostEqual(float(clock.s[0]), 0.0, places=5)
            self.assertAlmostEqual(float(clock.s[-1]), 1.0, places=5)
            self.assertTrue(torch.all(clock.s[1:] >= clock.s[:-1]))
            self.assertTrue(torch.all(clock.ds_dr >= 0.0))
            self.assertFalse(torch.isnan(clock.ds_dr).any())
            self.assertFalse(torch.isinf(clock.ds_dr).any())

    def test_ft_beta_matches_target_error_curve_for_linear_path(self):
        r = torch.linspace(0.0, 1.0, 257)
        beta = 0.35
        signal_scale_sq = 1.7
        clock = evaluate_clock(
            r=r,
            clock_family='ft_beta',
            clock_beta=beta,
            path_family='linear',
            signal_scale_sq=signal_scale_sq,
        )
        path = evaluate_path(clock.s, path_family='linear')
        mean_error = evaluate_mean_terminal_error(path, signal_scale_sq=signal_scale_sq)
        expected = mean_error[0] * torch.pow(1.0 - r, 1.0 / (1.0 - beta))
        self.assertTrue(torch.allclose(mean_error, expected, atol=1e-5, rtol=1e-5))

    def test_ft_beta_matches_target_error_curve_for_trig_vp_numeric_inverse(self):
        r = torch.linspace(0.0, 1.0, 257)
        beta = 0.4
        signal_scale_sq = 0.25
        clock = evaluate_clock(
            r=r,
            clock_family='ft_beta',
            clock_beta=beta,
            path_family='trig_vp',
            signal_scale_sq=signal_scale_sq,
        )
        path = evaluate_path(clock.s, path_family='trig_vp')
        mean_error = evaluate_mean_terminal_error(path, signal_scale_sq=signal_scale_sq)
        expected = mean_error[0] * torch.pow(1.0 - r, 1.0 / (1.0 - beta))
        self.assertTrue(torch.allclose(mean_error, expected, atol=1e-4, rtol=1e-4))

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
            signal_scale_sq=1.0,
        )
        self.assertTrue(torch.allclose(batch.x_t, noise))
        self.assertEqual(batch.target_velocity.shape, samples.shape)

    def test_sample_strict_unit_interval_avoids_endpoints(self):
        r = sample_strict_unit_interval(1024, device=torch.device('cpu'))
        self.assertTrue(torch.all(r > 0.0))
        self.assertTrue(torch.all(r < 1.0))

    def test_clamp_time_inside_unit_interval_avoids_endpoint_zeroing(self):
        r = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        clamped = clamp_time_inside_unit_interval(r)
        self.assertGreater(float(clamped[0]), 0.0)
        self.assertLess(float(clamped[-1]), 1.0)
        self.assertAlmostEqual(float(clamped[1]), 0.5, places=6)

    def test_sample_importance_weighted_time_avoids_endpoints(self):
        r = sample_importance_weighted_time(
            batch_size=1024,
            device=torch.device('cpu'),
            path_family='linear',
            clock_family='ft_beta',
            clock_beta=0.5,
            signal_scale_sq=1.0,
        )
        self.assertTrue(torch.all(r > 0.0))
        self.assertTrue(torch.all(r < 1.0))

    def test_low_beta_importance_sampling_does_not_collapse_to_boundary_atom(self):
        r = sample_importance_weighted_time(
            batch_size=4096,
            device=torch.device('cpu'),
            path_family='linear',
            clock_family='ft_beta',
            clock_beta=0.2,
            signal_scale_sq=1.0,
        )
        boundary = torch.tensor(1.0 - 1e-5, dtype=r.dtype)
        self.assertEqual(int((r == boundary).sum().item()), 0)

    def test_model_output_conversion_helpers_are_inverses(self):
        ds_dr = torch.tensor([0.5, 2.0], dtype=torch.float32)
        base_velocity = torch.tensor(
            [[1.0, -2.0], [0.25, 0.5]],
            dtype=torch.float32,
        )
        velocity = model_output_to_velocity(
            model_output=base_velocity,
            ds_dr=ds_dr,
            model_output_type='base_velocity',
        )
        recovered = model_output_to_base_velocity(
            model_output=velocity,
            ds_dr=ds_dr,
            model_output_type='velocity',
        )
        self.assertTrue(torch.allclose(recovered, base_velocity))


if __name__ == '__main__':
    unittest.main()
