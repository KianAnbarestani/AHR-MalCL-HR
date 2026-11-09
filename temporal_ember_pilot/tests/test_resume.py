import random
import tempfile
import unittest
from pathlib import Path

from temporal_ember_pilot.checkpoint import load_checkpoint, load_latest_checkpoint, save_checkpoint
from temporal_ember_pilot.engine import (
    StopRequested, capture_rng_state, planned_pool_counts, planned_updates,
    restore_rng_state, run_window,
)
from temporal_ember_pilot.config import ARMS, EXPERIMENT_ID, SEEDS, validate_run_matrix
from temporal_ember_pilot.data import load_manifest
from temporal_ember_pilot.runner import (
    _load_run_state, _rng_stream_identity, _rng_streams,
    load_or_freeze_budget_schedule, run_production_matrix,
)


def _state(seed=42):
    random.seed(seed)
    torch_model = None
    torch_optimizer = None
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch_model = torch.nn.Linear(2, 1)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    except ImportError:
        pass
    training_state = {"weights": [0.25, -0.5]}
    optimizer_state = {"step": 0, "momentum": [0.0, 0.0]}
    if torch_model is not None:
        training_state = {"backend": "torch", "model": torch_model.state_dict()}
        optimizer_state = {"backend": "torch", "model": torch_optimizer.state_dict(), "step": 0}
    return {
        "stream_cursor": {"next_window_index": 1, "last_completed_window": 0, "window_id": "2018-01", "phase": "completed", "batch_index": 0},
        "training_state": training_state,
        "optimizer_state": optimizer_state,
        "scaler_state": {"n": 8, "mean": [1.0, 2.0], "scale": [2.0, 3.0]},
        "anchor_state": {"sample_ids": ["anchor-a"]},
        "class_map": {"old": 0},
        "gan_state": {"weights": [0.1], "steps": 1},
        "rng_state": capture_rng_state(),
        "metrics": [],
    }


def _runtime_versions(torch):
    import lief
    import numpy as np
    import platform
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "scikit_learn": sklearn.__version__,
        "lief": str(lief.__version__),
    }


def _train_batch(state, window, batch_index):
    draw = random.random()
    try:
        import numpy as np
        draw += float(np.random.random())
    except ImportError:
        pass
    try:
        import torch
        draw += float(torch.rand(1).item())
    except ImportError:
        pass
    if state["training_state"].get("backend") == "torch":
        import torch
        model = torch.nn.Linear(2, 1)
        model.load_state_dict(state["training_state"]["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.load_state_dict(state["optimizer_state"]["model"])
        x = torch.randn(4, 2)
        target = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        state["training_state"]["model"] = model.state_dict()
        state["optimizer_state"]["model"] = optimizer.state_dict()
        state["optimizer_state"]["step"] += 1
    else:
        state["training_state"]["weights"][0] += draw + batch_index
        state["optimizer_state"]["step"] += 1
        state["optimizer_state"]["momentum"][0] = draw
    state["metrics"].append((window["window_id"], batch_index, draw))
    return state


def _finalize_window(state, window):
    state["gan_state"]["steps"] += 1
    state["metrics"].append({"completed": window["window_id"]})
    return state


def _assert_nested_equal(test_case, left, right):
    try:
        import torch
        if torch.is_tensor(left) or torch.is_tensor(right):
            test_case.assertTrue(torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right))
            return
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right)
            return
    except ImportError:
        pass
    if isinstance(left, dict) and isinstance(right, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_nested_equal(test_case, a, b)
    else:
        test_case.assertEqual(left, right)


class ResumeTests(unittest.TestCase):
    def test_exactly_nine_approved_production_configurations(self):
        configs = validate_run_matrix()
        self.assertEqual(len(configs), 9)
        self.assertEqual({config.arm for config in configs}, set(ARMS))
        self.assertEqual({config.seed for config in configs}, set(SEEDS))
        self.assertTrue(all(config.experiment_id == EXPERIMENT_ID for config in configs))

    def test_production_runner_is_disabled_without_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "production training is disabled"):
            run_production_matrix(archive_path="missing-archive", manifest_path="missing-manifest")

    def test_planned_budget_uses_fixed_hybrid_pool_quota(self):
        counts = planned_pool_counts(128, {"old_a": 40, "old_b": 200})
        self.assertEqual(counts, {"current": 128, "anchor": 140, "generated": 200, "full_hybrid": 468})
        self.assertEqual(planned_updates(128, {"old_a": 40, "old_b": 200}), 6)
        for arm in ("full_hybrid", "anchor_only", "current_only"):
            for seed in (42, 43, 44):
                self.assertEqual(planned_updates(128, {"old_a": 40, "old_b": 200}), 6)

    def test_runner_loads_the_release_frozen_six_window_budget(self):
        manifest = load_manifest(Path(__file__).parents[1] / "split_manifest.json.gz")
        with tempfile.TemporaryDirectory() as tmp:
            schedule = load_or_freeze_budget_schedule(manifest, tmp)
            self.assertEqual(len(schedule["windows"]), 6)
            self.assertEqual(schedule["schedule_sha256"], "1d7ec52ff9988b6991106da05a9d96329c5179d71d1a934ce1fb431d4bce9170")
            self.assertEqual(load_or_freeze_budget_schedule(manifest, tmp), schedule)
            self.assertTrue((Path(tmp) / "frozen_budget_schedule.json").exists())

    def test_fresh_resume_restores_and_checks_deterministic_backend_settings(self):
        import torch

        config = validate_run_matrix()[0]
        manifest = {"manifest_sha256": "a" * 64}
        schedule = {"schedule_sha256": "b" * 64}
        state = _state()
        state["numpy_rng_generators"] = {}
        state["run_metadata"] = {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "arm": config.arm,
            "seed": config.seed,
            "manifest_sha256": manifest["manifest_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "device_type": "cpu",
            "rng_stream_identity": _rng_stream_identity(_rng_streams(config)),
            "runtime_versions": _runtime_versions(torch),
            "deterministic_backend": {
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp) / "window_0.pkl", state)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            _load_run_state(config, manifest, schedule, tmp, "cpu")
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)

    def test_resume_rejects_checkpoint_with_different_backend_settings(self):
        import torch

        config = validate_run_matrix()[0]
        manifest = {"manifest_sha256": "a" * 64}
        schedule = {"schedule_sha256": "b" * 64}
        state = _state()
        state["numpy_rng_generators"] = {}
        state["run_metadata"] = {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "arm": config.arm,
            "seed": config.seed,
            "manifest_sha256": manifest["manifest_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "device_type": "cpu",
            "rng_stream_identity": _rng_stream_identity(_rng_streams(config)),
            "runtime_versions": _runtime_versions(torch),
            "deterministic_backend": {
                "cudnn_deterministic": False,
                "cudnn_benchmark": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp) / "window_0.pkl", state)
            with self.assertRaisesRegex(ValueError, "deterministic backend"):
                _load_run_state(config, manifest, schedule, tmp, "cpu")

    def test_resume_rejects_changed_runtime_versions(self):
        import torch

        config = validate_run_matrix()[0]
        manifest = {"manifest_sha256": "a" * 64}
        schedule = {"schedule_sha256": "b" * 64}
        state = _state()
        state["numpy_rng_generators"] = {}
        state["run_metadata"] = {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "arm": config.arm,
            "seed": config.seed,
            "manifest_sha256": manifest["manifest_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "device_type": "cpu",
            "rng_stream_identity": _rng_stream_identity(_rng_streams(config)),
            "runtime_versions": {**_runtime_versions(torch), "torch": "different-version"},
            "deterministic_backend": {
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp) / "window_0.pkl", state)
            with self.assertRaisesRegex(ValueError, "runtime versions differ"):
                _load_run_state(config, manifest, schedule, tmp, "cpu")

    def test_stop_resume_matches_uninterrupted_training_optimizer_cursor_and_rng(self):
        window = {"window_id": "2018-02_03", "window_index": 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boundary = root / "window_0.pkl"
            window_path = root / "window_1.pkl"
            base = _state()
            save_checkpoint(boundary, base)

            uninterrupted = run_window(base, window, updates=5, train_batch=_train_batch, checkpoint_path=window_path,
                                       finalize_window=_finalize_window)
            self.assertEqual(len(list(root.glob("window_*.pkl"))), 2)
            latest_path, latest_state = load_latest_checkpoint(root)
            self.assertEqual(latest_path.name, "window_1.pkl")
            self.assertEqual(latest_state["stream_cursor"]["next_window_index"], 2)

            resumed_path = root / "resumed"
            resumed_path.mkdir()
            resumed_boundary = resumed_path / "window_0.pkl"
            save_checkpoint(resumed_boundary, _state())
            with self.assertRaises(StopRequested):
                run_window(load_checkpoint(resumed_boundary), window, updates=5, train_batch=_train_batch,
                           checkpoint_path=resumed_path / "window_1.pkl", stop_after_batches=2)
            self.assertFalse((resumed_path / "window_1.pkl").exists())
            resumed = run_window(load_checkpoint(resumed_boundary), window, updates=5, train_batch=_train_batch,
                                 checkpoint_path=resumed_path / "window_1.pkl", finalize_window=_finalize_window)

            _assert_nested_equal(self, uninterrupted["training_state"], resumed["training_state"])
            _assert_nested_equal(self, uninterrupted["optimizer_state"], resumed["optimizer_state"])
            self.assertEqual(uninterrupted["scaler_state"], resumed["scaler_state"])
            self.assertEqual(uninterrupted["anchor_state"], resumed["anchor_state"])
            self.assertEqual(uninterrupted["class_map"], resumed["class_map"])
            self.assertEqual(uninterrupted["gan_state"], resumed["gan_state"])
            self.assertEqual(uninterrupted["metrics"], resumed["metrics"])
            self.assertEqual(uninterrupted["stream_cursor"], resumed["stream_cursor"])
            rng_a = uninterrupted["rng_state"]
            rng_b = resumed["rng_state"]
            _assert_nested_equal(self, rng_a, rng_b)
            restore_rng_state(uninterrupted["rng_state"])
            uninterrupted_next_rng = [random.random()]
            resumed_next_rng = []
            try:
                import numpy as np
                uninterrupted_next_rng.append(float(np.random.random()))
            except ImportError:
                pass
            try:
                import torch
                uninterrupted_next_rng.append(float(torch.rand(1).item()))
            except ImportError:
                pass
            restore_rng_state(resumed["rng_state"])
            resumed_next_rng.append(random.random())
            try:
                import numpy as np
                resumed_next_rng.append(float(np.random.random()))
            except ImportError:
                pass
            try:
                import torch
                resumed_next_rng.append(float(torch.rand(1).item()))
            except ImportError:
                pass
            self.assertEqual(uninterrupted_next_rng, resumed_next_rng)


if __name__ == "__main__":
    unittest.main()
