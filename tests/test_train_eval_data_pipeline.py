import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "examples" / "image" / "train.py"
TRANSFORM_SOURCE = ROOT / "examples" / "image" / "training" / "data_transform.py"


class TrainEvalDataPipelineSourceTest(unittest.TestCase):
    def test_eval_transform_excludes_random_flip(self):
        source = TRANSFORM_SOURCE.read_text(encoding="utf-8")
        self.assertIn("def get_eval_transform()", source)
        eval_section = source.split("def get_eval_transform()", maxsplit=1)[1]
        self.assertNotIn("RandomHorizontalFlip()", eval_section)

    def test_eval_loader_uses_eval_dataset(self):
        source = TRAIN_SOURCE.read_text(encoding="utf-8")
        self.assertIn("transform_eval = get_eval_transform()", source)
        self.assertIn(
            "dataset_eval = build_dataset(args=args, transform=transform_eval)",
            source,
        )
        self.assertIn(
            "dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False",
            source,
        )
        self.assertIn("data_loader_eval = torch.utils.data.DataLoader(", source)
        self.assertIn("        dataset_eval,", source)

    def test_eval_path_passes_train_loader_for_solver_aware_monitor(self):
        source = TRAIN_SOURCE.read_text(encoding="utf-8")
        self.assertIn("monitor_data_loader=data_loader_train", source)


if __name__ == "__main__":
    unittest.main()
