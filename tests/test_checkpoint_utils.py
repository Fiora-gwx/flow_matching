import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.checkpoint_utils import find_checkpoint, resolve_checkpoint_path


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
