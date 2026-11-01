from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from revision_r1.compatibility import verify_checkpoint_compatibility


ROOT = Path(__file__).resolve().parents[1]


def test_distributed_runtime_manifest_binds_legacy_identity_and_actual_code_hash():
    manifest = verify_checkpoint_compatibility(ROOT)
    on_disk = json.loads((ROOT / "checkpoint_compatibility.json").read_text(encoding="utf-8"))

    assert manifest == on_disk
    assert manifest["baseline_training_source_sha256"] == "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
    assert manifest["runtime_source_sha256"] != manifest["baseline_training_source_sha256"]
    assert manifest["training_behavior_matches_baseline"] is True


def test_distributed_runtime_manifest_rejects_changed_training_module(tmp_path):
    package = tmp_path / "package"
    shutil.copytree(ROOT / "revision_r1", package / "revision_r1")
    shutil.copyfile(ROOT / "checkpoint_compatibility.json", package / "checkpoint_compatibility.json")
    source = package / "revision_r1" / "sampling.py"
    source.write_bytes(source.read_bytes() + b"\n# incompatible test mutation\n")

    with pytest.raises(RuntimeError, match="runtime source modules do not match"):
        verify_checkpoint_compatibility(package)
