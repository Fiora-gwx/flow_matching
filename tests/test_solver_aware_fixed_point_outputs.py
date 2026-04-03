import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch

    from examples.image.training.solver_aware.fixed_point import (
        SolverAwareArtifacts,
        _build_artifact_csv_text,
        _build_artifact_json_payload,
        _build_node_csv_text,
        _build_node_json_payload,
        _load_cache,
        _resolve_profile_cache_path,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is required for solver-aware output tests")
class SolverAwareFixedPointOutputsTest(unittest.TestCase):
    def test_resolve_profile_cache_path_uses_budget_specific_suffixes_for_defect_profiles(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "eval_ep499_nfe12"
            single_budget_path = _resolve_profile_cache_path(
                cache_path="none",
                output_dir=output_dir,
                monitor_family="defect_based",
                target_solver="euler",
                monitor_solver="euler",
                budget_mode="single_budget",
                target_nfe=12,
                target_nfe_list=(12,),
                target_nfe_weights={"12": 1.0},
            )
            multi_budget_path = _resolve_profile_cache_path(
                cache_path="none",
                output_dir=output_dir,
                monitor_family="defect_based",
                target_solver="euler",
                monitor_solver="euler",
                budget_mode="multi_budget",
                target_nfe=12,
                target_nfe_list=(6, 12, 18, 24),
                target_nfe_weights={"6": 1.0, "12": 1.0, "18": 1.0, "24": 1.0},
            )

        self.assertEqual(
            single_budget_path,
            output_dir.parent / "solver_aware_profile_defect_based_euler_euler_single_nfe12.pt",
        )
        self.assertIsNotNone(multi_budget_path)
        self.assertIn(
            "solver_aware_profile_defect_based_euler_euler_multi_",
            multi_budget_path.name,
        )
        self.assertNotEqual(single_budget_path, multi_budget_path)

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

    def test_load_cache_ignores_legacy_profile_without_new_fields(self):
        signature = {"mode": "training_free"}
        legacy_payload = {
            "mode": "training_free",
            "target_solver": "euler",
            "monitor_solver": "euler",
            "estimator": "jvp",
            "theorem_backed": True,
            "notes": "legacy",
            "checkpoint_source": "checkpoint-499.pth",
            "grid_size": 5,
            "batch_size": 4,
            "eps": 1.0e-6,
            "q_values": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "q_smoothed": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "density": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "s_grid": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "phi": torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pt"
            torch.save({"signature": signature, "artifacts": legacy_payload}, path)
            self.assertIsNone(_load_cache(path, signature))

    def test_defect_payloads_include_budget_curves_and_reference_metadata(self):
        artifacts = SolverAwareArtifacts(
            mode="training_free",
            target_solver="heun2",
            monitor_solver="heun2",
            estimator="defect",
            theorem_backed=True,
            notes="defect",
            checkpoint_source="checkpoint-499.pth",
            grid_size=5,
            batch_size=4,
            eps=1.0e-6,
            q_values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            q_smoothed=torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32),
            density=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
            s_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            phi=torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
            density_exponent=1.0 / 6.0,
            smoothing_window=3,
            monitor_family="defect_based",
            budget_mode="multi_budget",
            target_nfe=12,
            target_nfe_list=(12, 24),
            target_nfe_weights={"12": 0.5, "24": 0.5},
            target_step_count=6,
            budget_step_count_by_nfe={"12": 6, "24": 12},
            defect_subdivide=2,
            solver_order=2.0,
            q_curve_name="M_tilde_path_defect",
            aggregation_name="normalized_multi_budget",
            q_values_by_budget={
                "12": torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32),
                "24": torch.tensor([0.5, 0.75, 1.0], dtype=torch.float32),
            },
            q_normalized_by_budget={
                "12": torch.tensor([4.0, 6.0, 8.0], dtype=torch.float32),
                "24": torch.tensor([4.0, 6.0, 8.0], dtype=torch.float32),
            },
            budget_weights={"12": 0.5, "24": 0.5},
            distribution_info={"distribution": "path_distribution", "path_family": "linear"},
            step_count=6,
            r_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            nodes=torch.tensor([0.0, 0.2, 1.0], dtype=torch.float32),
        )

        artifact_payload = _build_artifact_json_payload(artifacts)
        node_payload = _build_node_json_payload(artifacts)

        self.assertEqual(artifact_payload["monitor_family"], "defect_based")
        self.assertEqual(artifact_payload["budget_mode"], "multi_budget")
        self.assertEqual(artifact_payload["q_curve_name"], "M_tilde_path_defect")
        self.assertEqual(artifact_payload["q_values_by_budget"]["12"], [1.0, 1.5, 2.0])
        self.assertEqual(artifact_payload["distribution_info"]["distribution"], "path_distribution")
        self.assertEqual(node_payload["budget_step_count_by_nfe"]["24"], 12)


if __name__ == "__main__":
    unittest.main()
