from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

from revision_r1.archives import build_checkpoint_archive, build_lightweight_results_archive
from revision_r1.checkpointing import save_generation
from revision_r1.config import RunConfig
from revision_r1.data import DatasetBundle
from revision_r1.runner import run_one
from revision_r1.state import PhaseCursor, RunState


def _state(identity):
    return RunState(identity=identity, cursor=PhaseCursor(task_index=0, phase="classifier"))


def test_lightweight_results_archive_contains_hash_manifest_and_excludes_weights(tmp_path):
    rng = np.random.default_rng(31)
    x_train = np.vstack([rng.normal(size=(2, 4)) + class_id for class_id in range(100)]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([rng.normal(size=(1, 4)) + class_id for class_id in range(100)]).astype(np.float32)
    y_test = np.arange(100, dtype=np.int64)
    metadata = {"sha256": "synthetic-test", "md5": "synthetic-test"}
    bundle = DatasetBundle("ember", x_train, y_train, x_test, y_test, metadata, metadata)
    config = RunConfig("ember", 42, "current_only", "a" * 64, classifier_hidden=(8, 8, 4), classifier_embedding_dim=8, dropout=0.0)
    root = tmp_path / "run_one"
    run_one(config, bundle, root, synthetic=True, update_counts=[1] * 11, device="cpu")
    archive_path = tmp_path / "results.zip"

    result = build_lightweight_results_archive([root], archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "run_one/per_task_metrics.csv" in names
        assert not any(name.endswith((".pt", ".pth", ".ckpt")) for name in names)
        manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
    assert manifest["archive_kind"] == "LIGHTWEIGHT_RESULTS_NO_MODEL_CHECKPOINTS"
    assert manifest["files"]["run_one/per_task_metrics.csv"]["sha256"]
    assert (tmp_path / "results.zip.manifest.json").is_file()
    assert result["sha256"]


def test_checkpoint_archive_keeps_all_verified_recovery_and_task_boundaries(tmp_path):
    identity = {"run": "synthetic"}
    checkpoint_root = tmp_path / "checkpoints"
    save_generation(checkpoint_root / "recovery", _state(identity), identity, kind="recovery")
    save_generation(checkpoint_root / "task_boundaries", _state(identity), identity, kind="task-boundary", retain_generations=None)
    archive_path = tmp_path / "checkpoints.zip"

    result = build_checkpoint_archive(checkpoint_root, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        assert len([name for name in archive.namelist() if name.endswith(".pt")]) == 2
    assert manifest["generation_count"] == 2
    assert result["file_count"] == 4
    assert (tmp_path / "checkpoints.zip.manifest.json").is_file()


def test_checkpoint_archive_rejects_missing_or_modified_payload(tmp_path):
    identity = {"run": "synthetic"}
    checkpoint_root = tmp_path / "checkpoints"
    save_generation(checkpoint_root, _state(identity), identity, kind="recovery")
    payload = next(checkpoint_root.glob("generation-*.pt"))
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_checkpoint_archive(checkpoint_root, tmp_path / "bad.zip")
