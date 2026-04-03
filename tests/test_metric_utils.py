import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_ROOT = os.path.join(ROOT, "examples", "image")
if IMAGE_ROOT not in sys.path:
    sys.path.insert(0, IMAGE_ROOT)

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from training.metric_utils import extract_inception_features
else:  # pragma: no cover - environment dependent
    torch = None
    extract_inception_features = None


@unittest.skipUnless(HAS_TORCH, "torch is required for metric utils tests")
class MetricUtilsTest(unittest.TestCase):
    class FakeMetricBackend:
        def __init__(self):
            self.last_input = None

        def inception(self, images):
            self.last_input = images
            return torch.ones(images.shape[0], 8, 1, 1, dtype=torch.float32)

    def test_extract_inception_features_converts_float_images_to_uint8(self):
        backend = self.FakeMetricBackend()
        images = torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 1.1]]],
                [[[0.25, 0.75], [-0.1, 0.9]]],
            ],
            dtype=torch.float32,
        )

        features = extract_inception_features(backend, images)

        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(tuple(features.shape), (2, 8))
        self.assertEqual(backend.last_input.dtype, torch.uint8)
        expected = (
            images.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        )
        self.assertTrue(torch.equal(backend.last_input, expected))

    def test_extract_inception_features_keeps_uint8_images(self):
        backend = self.FakeMetricBackend()
        images = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)

        extract_inception_features(backend, images)

        self.assertIs(backend.last_input, images)
        self.assertEqual(backend.last_input.dtype, torch.uint8)


@unittest.skipUnless(HAS_TORCH, "torch is required for eval precision/recall tests")
class EvalLoopPrecisionRecallTest(unittest.TestCase):
    def test_eval_model_precision_recall_accepts_float_images(self):
        from training import eval_loop

        class FakeMetricBackend:
            def __init__(self):
                self.real_dtypes = []
                self.fake_dtypes = []

            def update(self, images, real):
                if real:
                    self.real_dtypes.append(images.dtype)
                else:
                    self.fake_dtypes.append(images.dtype)

            def inception(self, images):
                self.assertEqual(images.dtype, torch.uint8)
                return torch.ones(images.shape[0], 4, 1, 1, dtype=torch.float32)

            def compute(self):
                return torch.tensor(0.0)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left} != {right}")

        fake_backend = FakeMetricBackend()
        sample = torch.rand(2, 3, 32, 32, dtype=torch.float32)
        labels = torch.zeros(2, dtype=torch.long)
        data_loader = [(sample, labels)]
        args = types.SimpleNamespace(
            metrics=["precision_recall"],
            compute_fid=False,
            discrete_flow_matching=False,
            output_dir=None,
            test_run=True,
            save_fid_samples=False,
            sampling_solver="heun2",
            eval_nfe=10,
            cfg_scale=0.0,
            precision_recall_neighbors=3,
            precision_recall_max_samples=16,
        )

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, extra=None):
                return x

        fake_sampling = types.SimpleNamespace(
            sample=torch.rand(2, 3, 32, 32, dtype=torch.float32),
            nfe=10,
            step_count=5,
        )

        with mock.patch.object(eval_loop, "_build_fid_metric", return_value=fake_backend):
            with mock.patch.object(eval_loop, "solve_fixed_budget", return_value=fake_sampling):
                with mock.patch.object(
                    eval_loop,
                    "compute_precision_recall",
                    return_value={"precision": 0.7, "recall": 0.6},
                ):
                    results = eval_loop.eval_model(
                        model=DummyModel(),
                        data_loader=data_loader,
                        device=torch.device("cpu"),
                        epoch=0,
                        fid_samples=2,
                        args=args,
                    )

        self.assertEqual(fake_backend.real_dtypes, [])
        self.assertEqual(fake_backend.fake_dtypes, [])
        self.assertEqual(results["precision"], 0.7)
        self.assertEqual(results["recall"], 0.6)
        self.assertEqual(results["synthetic_samples"], 2.0)


@unittest.skipUnless(HAS_TORCH, "torch is required for eval fid tests")
class EvalLoopFidInputTest(unittest.TestCase):
    def test_eval_model_keeps_continuous_fake_samples_as_float_for_metrics(self):
        from training import eval_loop

        class FakeMetricBackend:
            def __init__(self):
                self.real_inputs = []
                self.fake_inputs = []

            def update(self, images, real):
                if real:
                    self.real_inputs.append(images.detach().clone())
                else:
                    self.fake_inputs.append(images.detach().clone())

            def compute(self):
                return torch.tensor(0.0)

        fake_backend = FakeMetricBackend()
        real_sample = torch.full((1, 3, 4, 4), 0.25, dtype=torch.float32)
        labels = torch.zeros(1, dtype=torch.long)
        data_loader = [(real_sample, labels)]
        args = types.SimpleNamespace(
            metrics=["fid"],
            compute_fid=False,
            discrete_flow_matching=False,
            output_dir=None,
            test_run=True,
            save_fid_samples=False,
            sampling_solver="heun2",
            eval_nfe=10,
            cfg_scale=0.0,
            path_family="linear",
            clock_family="uniform",
            clock_beta=None,
            signal_scale_sq=None,
            precision_recall_neighbors=3,
            precision_recall_max_samples=16,
        )

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, extra=None):
                return x

        fake_sampling = types.SimpleNamespace(
            sample=torch.full((1, 3, 4, 4), -0.7532, dtype=torch.float32),
            nfe=10,
            step_count=5,
        )

        with mock.patch.object(eval_loop, "_build_fid_metric", return_value=fake_backend):
            with mock.patch.object(eval_loop, "solve_fixed_budget", return_value=fake_sampling):
                results = eval_loop.eval_model(
                    model=DummyModel(),
                    data_loader=data_loader,
                    device=torch.device("cpu"),
                    epoch=0,
                    fid_samples=1,
                    args=args,
                )

        self.assertEqual(len(fake_backend.real_inputs), 1)
        self.assertEqual(len(fake_backend.fake_inputs), 1)
        self.assertAlmostEqual(fake_backend.real_inputs[0][0, 0, 0, 0].item(), 0.25, places=6)
        self.assertAlmostEqual(fake_backend.fake_inputs[0][0, 0, 0, 0].item(), 0.1234, places=4)
        self.assertNotAlmostEqual(
            fake_backend.fake_inputs[0][0, 0, 0, 0].item(),
            31.0 / 255.0,
            places=6,
        )
        self.assertEqual(results["synthetic_samples"], 1.0)


@unittest.skipUnless(HAS_TORCH, "torch is required for eval inception score tests")
class EvalLoopInceptionScoreTest(unittest.TestCase):
    def test_eval_model_inception_score_accepts_float_images(self):
        from training import eval_loop

        class FakeInceptionScoreBackend:
            def __init__(self):
                self.input_dtypes = []

            def update(self, images):
                self.input_dtypes.append(images.dtype)

            def compute(self):
                return torch.tensor(3.2), torch.tensor(0.4)

        fake_backend = FakeInceptionScoreBackend()
        sample = torch.rand(2, 3, 32, 32, dtype=torch.float32)
        labels = torch.zeros(2, dtype=torch.long)
        data_loader = [(sample, labels)]
        args = types.SimpleNamespace(
            metrics=["inception_score"],
            compute_fid=False,
            discrete_flow_matching=False,
            output_dir=None,
            test_run=True,
            save_fid_samples=False,
            sampling_solver="heun2",
            eval_nfe=10,
            cfg_scale=0.0,
            precision_recall_neighbors=3,
            precision_recall_max_samples=16,
            inception_score_splits=2,
        )

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, extra=None):
                return x

        fake_sampling = types.SimpleNamespace(
            sample=torch.rand(2, 3, 32, 32, dtype=torch.float32),
            nfe=10,
            step_count=5,
        )

        with mock.patch.object(
            eval_loop,
            "_build_inception_score_metric",
            return_value=fake_backend,
        ):
            with mock.patch.object(eval_loop, "solve_fixed_budget", return_value=fake_sampling):
                results = eval_loop.eval_model(
                    model=DummyModel(),
                    data_loader=data_loader,
                    device=torch.device("cpu"),
                    epoch=0,
                    fid_samples=2,
                    args=args,
                )

        self.assertEqual(fake_backend.input_dtypes, [torch.uint8])
        self.assertAlmostEqual(results["is_mean"], 3.2, places=6)
        self.assertAlmostEqual(results["is_std"], 0.4, places=6)
        self.assertEqual(results["synthetic_samples"], 2.0)


@unittest.skipUnless(HAS_TORCH, "torch is required for eval sidecar tests")
class EvalLoopSolverStatsSidecarTest(unittest.TestCase):
    def test_eval_model_writes_solver_stats_sidecar(self):
        from training import eval_loop

        sample = torch.rand(2, 3, 32, 32, dtype=torch.float32)
        labels = torch.zeros(2, dtype=torch.long)
        data_loader = [(sample, labels)]

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, extra=None):
                return x

        fake_sampling = types.SimpleNamespace(
            sample=torch.rand(2, 3, 32, 32, dtype=torch.float32),
            nfe=12,
            step_count=4,
            solver_stats={
                "solver": "rk3",
                "requested_nfe_budget": 12,
                "actual_network_calls": 12,
                "step_count": 4,
                "virtual_stage_count": 0,
                "used_tail_step": False,
                "is_exact_budget": True,
                "is_shared_budget": True,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                metrics=["inception_score"],
                compute_fid=False,
                discrete_flow_matching=False,
                output_dir=tmpdir,
                test_run=True,
                save_fid_samples=False,
                sampling_solver="rk3",
                eval_nfe=12,
                cfg_scale=0.0,
                precision_recall_neighbors=3,
                precision_recall_max_samples=16,
                inception_score_splits=2,
            )

            class FakeInceptionScoreBackend:
                def update(self, images):
                    return None

                def compute(self):
                    return torch.tensor(2.0), torch.tensor(0.1)

            with mock.patch.object(
                eval_loop,
                "_build_inception_score_metric",
                return_value=FakeInceptionScoreBackend(),
            ):
                with mock.patch.object(eval_loop, "solve_fixed_budget", return_value=fake_sampling):
                    eval_loop.eval_model(
                        model=DummyModel(),
                        data_loader=data_loader,
                        device=torch.device("cpu"),
                        epoch=0,
                        fid_samples=2,
                        args=args,
                    )

            sidecar_path = os.path.join(tmpdir, "solver_stats.json")
            self.assertTrue(os.path.exists(sidecar_path))
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            self.assertEqual(sidecar["solver"], "rk3")
            self.assertEqual(sidecar["actual_network_calls"], 12)
            self.assertTrue(sidecar["is_shared_budget"])


@unittest.skipUnless(HAS_TORCH, "torch is required for solver-aware eval tests")
class EvalLoopSolverAwareMonitorLoaderTest(unittest.TestCase):
    def _run_monitor_loader_case(self, monitor_family: str):
        from training import eval_loop

        sample = torch.rand(1, 3, 8, 8, dtype=torch.float32)
        labels = torch.zeros(1, dtype=torch.long)
        data_loader = [(sample, labels)]
        monitor_loader = object()

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, extra=None):
                return x

        class FakeMetricBackend:
            def update(self, images, real):
                return None

            def compute(self):
                return torch.tensor(0.0)

        fake_sampling = types.SimpleNamespace(
            sample=torch.rand(1, 3, 8, 8, dtype=torch.float32),
            nfe=12,
            step_count=12,
            solver_stats=None,
        )
        fake_artifacts = types.SimpleNamespace(
            nodes=torch.tensor([0.0, 1.0], dtype=torch.float32),
            distribution_info={},
        )
        args = types.SimpleNamespace(
            metrics=["fid"],
            compute_fid=False,
            discrete_flow_matching=False,
            output_dir=None,
            test_run=True,
            save_fid_samples=False,
            sampling_solver="euler",
            eval_nfe=12,
            cfg_scale=0.0,
            path_family="linear",
            clock_family="uniform",
            clock_beta=None,
            signal_scale_sq=None,
            precision_recall_neighbors=3,
            precision_recall_max_samples=16,
            solver_aware_clock_mode="training_free",
            solver_aware_target_solver="euler",
            solver_aware_monitor_family=monitor_family,
            solver_aware_budget_mode="single_budget",
            solver_aware_target_nfe=0,
            solver_aware_target_nfe_list=[],
            solver_aware_target_nfe_weights=[],
            solver_aware_k=0,
            solver_aware_monitor_estimator="auto",
            solver_aware_monitor_grid_size=65,
            solver_aware_monitor_batch_size=64,
            solver_aware_eps=1.0e-6,
            solver_aware_cache_path="none",
            solver_aware_allow_eval_loader_for_monitor=False,
            solver_aware_stork_effective_order=4.0,
            solver_aware_defect_subdivide=2,
            solver_aware_use_nodes=True,
            seed=0,
            resume="checkpoint-499.pth",
        )

        with mock.patch.object(eval_loop, "_build_fid_metric", return_value=FakeMetricBackend()):
            with mock.patch.object(
                eval_loop,
                "maybe_build_solver_aware_artifacts",
                return_value=fake_artifacts,
            ) as build_artifacts:
                with mock.patch.object(
                    eval_loop,
                    "solve_fixed_budget",
                    return_value=fake_sampling,
                ):
                    results = eval_loop.eval_model(
                        model=DummyModel(),
                        data_loader=data_loader,
                        device=torch.device("cpu"),
                        epoch=0,
                        fid_samples=1,
                        args=args,
                        monitor_data_loader=monitor_loader,
                    )

        build_kwargs = build_artifacts.call_args.kwargs
        self.assertIs(build_kwargs["data_loader"], monitor_loader)
        self.assertFalse(build_kwargs["using_eval_loader_for_monitor"])
        self.assertFalse(build_kwargs["require_cache_hit"])
        self.assertEqual(results["nfe"], 12.0)
        self.assertEqual(results["synthetic_samples"], 1.0)

    def test_eval_model_uses_provided_monitor_loader_for_defect_based_solver_aware(self):
        self._run_monitor_loader_case(monitor_family="defect_based")

    def test_eval_model_uses_provided_monitor_loader_for_legacy_solver_aware(self):
        self._run_monitor_loader_case(monitor_family="legacy_continuous")
