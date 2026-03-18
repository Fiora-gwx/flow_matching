import importlib.util
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, "examples", "image")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import train as image_train
else:  # pragma: no cover - environment dependent
    image_train = None


@unittest.skipUnless(HAS_TORCH, "torch is required for train schedule tests")
class TrainScheduleTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "output_dir": "./output",
            "eval_only": False,
            "test_run": False,
            "epochs": 500,
            "eval_frequency": -1,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_final_checkpoint_is_saved_when_periodic_eval_disabled(self):
        args = self._args(eval_frequency=-1, epochs=500)

        self.assertFalse(image_train.should_run_eval(args, 498))
        self.assertTrue(image_train.should_save_checkpoint(args, 499))

    def test_periodic_eval_epoch_still_saves_and_evaluates(self):
        args = self._args(eval_frequency=50, epochs=500)

        self.assertTrue(image_train.should_run_eval(args, 49))
        self.assertTrue(image_train.should_save_checkpoint(args, 49))

    def test_eval_only_never_saves_checkpoint(self):
        args = self._args(eval_only=True, eval_frequency=-1)

        self.assertTrue(image_train.should_run_eval(args, 0))
        self.assertFalse(image_train.should_save_checkpoint(args, 0))


if __name__ == "__main__":
    unittest.main()
