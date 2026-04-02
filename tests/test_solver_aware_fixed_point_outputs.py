import unittest

try:
    import torch

    from examples.image.training.solver_aware.fixed_point import (
        SolverAwareArtifacts,
        _build_artifact_csv_text,
        _build_artifact_json_payload,
        _build_node_csv_text,
        _build_node_json_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is required for solver-aware output tests")
class SolverAwareFixedPointOutputsTest(unittest.TestCase):
    def test_payloads_include_curves_and_step_sizes(self):
        artifacts = SolverAwareArtifacts(
            mode="training_free",
            target_solver="euler",
            monitor_solver="euler",
            estimator="jvp",
            theorem_backed=True,
            notes="test",
            checkpoint_source="checkpoint-499.pth",
            grid_size=5,
            batch_size=4,
            eps=1.0e-6,
            q_values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            q_smoothed=torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32),
            density=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
            s_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            phi=torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
            density_exponent=0.25,
            smoothing_window=3,
            step_count=2,
            r_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            nodes=torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
        )

        artifact_payload = _build_artifact_json_payload(artifacts)
        node_payload = _build_node_json_payload(artifacts)
        artifact_csv = _build_artifact_csv_text(artifacts)
        node_csv = _build_node_csv_text(artifacts)

        self.assertEqual(artifact_payload["s_grid"], [0.0, 0.5, 1.0])
        self.assertEqual(artifact_payload["q_values"], [1.0, 2.0, 3.0])
        self.assertEqual(artifact_payload["density"], [0.4000000059604645, 0.5, 0.6000000238418579])
        self.assertEqual(node_payload["nodes"], [0.0, 0.25, 1.0])
        self.assertEqual(node_payload["step_sizes"], [0.0, 0.25, 0.75])
        self.assertAlmostEqual(node_payload["diagnostics"]["max_step"], 0.75, places=6)
        self.assertAlmostEqual(node_payload["diagnostics"]["max_step_over_uniform"], 1.5, places=6)
        self.assertIn("grid_index,s_value,q_value,q_smoothed,density,phi", artifact_csv)
        self.assertIn("node_index,r_value,s_value,step_size_from_prev", node_csv)


if __name__ == "__main__":
    unittest.main()
