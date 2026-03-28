import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, "examples", "image")
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

from training.continuous_runtime import evaluate_clock, sample_time_by_strategy


class TimeSamplingStrategiesTest(unittest.TestCase):
    def test_all_strategies_stay_inside_open_interval(self):
        strategies = (
            "uniform",
            "ds_dr_sq",
            "mixed_lambda",
            "stratified",
            "stratified_mixed",
            "curriculum",
        )
        for strategy in strategies:
            samples = sample_time_by_strategy(
                batch_size=256,
                device=torch.device("cpu"),
                path_family="linear",
                clock_family="ft_beta",
                clock_beta=0.3,
                signal_scale_sq=1.0,
                strategy=strategy,
                mixed_lambda=0.5,
                stratified_bins=16,
                current_epoch=499,
                total_epochs=500,
            )
            self.assertTrue(torch.all(samples > 0.0), msg=strategy)
            self.assertTrue(torch.all(samples < 1.0), msg=strategy)

    def test_stratified_sampling_hits_every_bin(self):
        samples = sample_time_by_strategy(
            batch_size=160,
            device=torch.device("cpu"),
            path_family="linear",
            clock_family="ft_beta",
            clock_beta=0.3,
            signal_scale_sq=1.0,
            strategy="stratified",
            stratified_bins=16,
        )
        counts = torch.bincount(torch.clamp((samples * 16).long(), max=15), minlength=16)
        self.assertTrue(torch.all(counts >= 5))

    def test_terminal_backoff_keeps_linear_beta_point_below_threshold(self):
        clock = evaluate_clock(
            r=torch.tensor([0.75], dtype=torch.float32),
            clock_family="ft_beta",
            clock_beta=0.3,
            path_family="linear",
            signal_scale_sq=1.0,
        )
        self.assertLess(float(clock.ds_dr.item()), 10.0)


if __name__ == "__main__":
    unittest.main()
