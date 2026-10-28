from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from revision_r1.config import RunConfig
from revision_r1.data import DatasetBundle
from revision_r1.generative import class_mean_mid, generate_class_candidates
from revision_r1.models import ExpandingClassifier, V155ConditionalGenerator
from revision_r1.preprocessing import RunningScaler
from revision_r1.runner import run_one


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(404)
    x_train = np.vstack([
        rng.normal(loc=class_id / 20.0, scale=0.2, size=(2, 4))
        for class_id in range(100)
    ]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([
        rng.normal(loc=class_id / 20.0, scale=0.2, size=(1, 4))
        for class_id in range(100)
    ]).astype(np.float32)
    y_test = np.arange(100, dtype=np.int64)
    marker = {"sha256": "dropout-resume-fixture", "md5": "dropout-resume-fixture"}
    return DatasetBundle("ember", x_train, y_train, x_test, y_test, marker, marker)


def _config() -> RunConfig:
    return RunConfig(
        dataset="ember", seed=42, arm="current_only", update_schedule_sha256="a" * 64,
        classifier_hidden=(8, 8, 4), classifier_embedding_dim=8, dropout=0.45,
    )


def _assert_models_exact(left_path: Path, right_path: Path) -> None:
    left = torch.load(left_path, map_location="cpu", weights_only=False)["classifier"]
    right = torch.load(right_path, map_location="cpu", weights_only=False)["classifier"]
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_evaluation_helpers_restore_classifier_and_generator_modes():
    model = ExpandingClassifier(4, initial_classes=2, hidden=(6, 5, 3), embedding_dim=3, dropout=0.4)
    generator = V155ConditionalGenerator(4, z_dim=3, hidden=4)
    raw = np.asarray([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    scaler = RunningScaler().update(raw)

    model.train()
    generator.train()
    means = class_mean_mid(model, raw, labels, scaler, [0, 1], "cpu")
    generate_class_candidates(
        generator, model, means[0], 0, candidates_per_class=3,
        z_dim=3, device="cpu", chunk_size=3,
    )
    assert model.training is True
    assert generator.training is True

    model.eval()
    generator.eval()
    class_mean_mid(model, raw, labels, scaler, [0, 1], "cpu")
    generate_class_candidates(
        generator, model, means[0], 0, candidates_per_class=3,
        z_dim=3, device="cpu", chunk_size=3,
    )
    assert model.training is False
    assert generator.training is False


def test_nonzero_dropout_modes_and_later_task_resume_match_uninterrupted_after_head_expansion(tmp_path):
    baseline = tmp_path / "baseline"
    resumed = tmp_path / "resumed"
    bundle = _bundle()
    config = _config()

    full = run_one(
        config, bundle, baseline, synthetic=True,
        update_counts=[1] * 11, device="cpu",
    )
    assert full["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"

    def interrupt_after_expanded_head_update(state):
        runtime = state.extra["runtime"]
        return (
            state.cursor.task_index == 1
            and state.cursor.phase == "classifier"
            and int(runtime.get("classifier_updates_this_task", 0)) >= 1
        )

    interrupted = run_one(
        config, bundle, resumed, synthetic=True,
        update_counts=[1] * 11, device="cpu",
        stop_hook=interrupt_after_expanded_head_update,
    )
    assert interrupted["status"] == "INTERRUPTED_RESUMABLE"
    assert interrupted["cursor"]["task_index"] == 1
    assert interrupted["cursor"]["phase"] == "classifier"
    recovered = run_one(
        config, bundle, resumed, synthetic=True,
        update_counts=[1] * 11, device="cpu",
    )
    assert recovered["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"

    baseline_modes = json.loads((baseline / "classifier_mode_trace.json").read_text())
    resumed_modes = json.loads((resumed / "classifier_mode_trace.json").read_text())
    assert resumed_modes == baseline_modes
    training = [event for event in resumed_modes if event["phase"] == "classifier_training"]
    evaluation = [event for event in resumed_modes if event["phase"] == "evaluation"]
    assert {event["task_id"] for event in training} == set(range(1, 12))
    assert all(event["classifier_training"] for event in training)
    assert all(event["dropout_probabilities"] and all(p > 0 for p in event["dropout_probabilities"]) for event in training)
    assert all(event["dropout_training"] == [True] * len(event["dropout_training"]) for event in training)
    assert {event["task_id"] for event in evaluation} == set(range(1, 12))
    assert all(not event["classifier_training"] for event in evaluation)
    assert all(event["dropout_training"] == [False] * len(event["dropout_training"]) for event in evaluation)

    _assert_models_exact(baseline / "final_models.pt", resumed / "final_models.pt")
    with np.load(baseline / "predictions.npz", allow_pickle=False) as expected:
        with np.load(resumed / "predictions.npz", allow_pickle=False) as observed:
            assert all(np.array_equal(expected[name], observed[name]) for name in expected.files)
    assert (baseline / "trace.jsonl").read_bytes() == (resumed / "trace.jsonl").read_bytes()
    resources = json.loads((resumed / "resource_metrics.json").read_text())
    assert resources["classifier_seconds"] > 0
    assert resources["timing_metadata"]["classifier_boundary"] == "forward/loss through optimizer step, synchronized CUDA wall time"

