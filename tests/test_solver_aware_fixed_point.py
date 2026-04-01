import unittest
from types import SimpleNamespace

try:
    import torch

    from examples.image.training.solver_aware.fixed_point import _build_node_json_payload
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is required for solver-aware fixed-point tests")
class SolverAwareFixedPointTest(unittest.TestCase):
    def test_node_payload_includes_eta_and_floor_diagnostics(self):
        artifacts = SimpleNamespace(
            step_count=6,
            eta=0.4,
            used_uniform_fallback=False,
            floor_mass=0.8,
            min_feasible_step_count=6,
            rho_floor=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            r_grid=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            nodes=torch.tensor([0.0, 0.4, 1.0], dtype=torch.float32),
            step_sizes=torch.tensor([0.0, 0.4, 0.6], dtype=torch.float32),
        )
        diagnostics = {"uniform_step": 1.0 / 6.0}

        payload = _build_node_json_payload(artifacts=artifacts, diagnostics=diagnostics)

        self.assertEqual(payload["eta"], 0.4)
        self.assertEqual(payload["floor_mass"], 0.8)
        self.assertEqual(payload["rho_floor"], [0.1, 0.2, 0.3])
        self.assertEqual(payload["rho_floor_summary"]["min"], 0.1)
        self.assertEqual(payload["rho_floor_summary"]["max"], 0.3)


if __name__ == "__main__":
    unittest.main()
