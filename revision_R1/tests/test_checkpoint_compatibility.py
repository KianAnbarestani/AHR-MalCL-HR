from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from revision_r1.compatibility import training_behavior_fingerprint, verify_checkpoint_compatibility
from scripts.build_checkpoint_compatibility import (
    AUTHENTIC_BASELINE_COMMIT,
    build_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _has_git_history(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def test_checkpoint_compatibility_is_anchored_to_authentic_baseline_and_separate_runtime_hash():
    # The build-time check authenticates against the historical Git baseline.
    # A distributed clean package has no repository history; there the same
    # test verifies its immutable compatibility manifest against the actual
    # packaged runtime modules instead of requiring unavailable Git metadata.
    manifest = build_manifest(ROOT) if _has_git_history(ROOT) else verify_checkpoint_compatibility(ROOT)

    assert manifest["baseline_commit_sha"] == AUTHENTIC_BASELINE_COMMIT
    assert manifest["baseline_training_source_sha256"] == "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
    assert manifest["runtime_source_sha256"] != manifest["baseline_training_source_sha256"]
    assert manifest["changed_infrastructure_modules"] == [
        "checkpointing.py", "resources.py", "runner.py", "storage.py", "validation_workflow.py",
    ]
    assert manifest["added_compatibility_modules"] == ["compatibility.py", "drive_storage.py"]
    assert manifest["training_behavior_matches_baseline"] is True
    assert manifest["checkpoint_schema_unchanged"] is True


def test_runtime_compatibility_verifier_rejects_any_unmanifested_module_change(tmp_path):
    package = tmp_path / "revision_R1"
    shutil.copytree(ROOT / "revision_r1", package / "revision_r1")
    shutil.copyfile(ROOT / "checkpoint_compatibility.json", package / "checkpoint_compatibility.json")
    module = package / "revision_r1" / "sampling.py"
    module.write_bytes(module.read_bytes() + b"\n# tampered\n")

    with pytest.raises(RuntimeError, match="runtime source modules do not match"):
        verify_checkpoint_compatibility(package)


def test_compatibility_manifest_is_machine_readable_and_matches_verified_runtime():
    manifest = json.loads((ROOT / "checkpoint_compatibility.json").read_text(encoding="utf-8"))
    observed_fingerprint = training_behavior_fingerprint((ROOT / "revision_r1" / "runner.py").read_bytes())
    assert observed_fingerprint == manifest["training_behavior_fingerprint_sha256"]
    verified = verify_checkpoint_compatibility(ROOT)
    assert manifest == verified
