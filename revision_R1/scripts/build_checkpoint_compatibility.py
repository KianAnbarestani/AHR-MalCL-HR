"""Build the explicit legacy-checkpoint/runtime-source compatibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from revision_r1.compatibility import (
    CHECKPOINT_IDENTITY_SHA256,
    ADDED_INFRASTRUCTURE_MODULES,
    INFRASTRUCTURE_ONLY_CHANGES,
    runtime_module_hashes,
    runtime_source_sha256,
    training_behavior_fingerprint,
    verify_checkpoint_compatibility,
)


AUTHENTIC_BASELINE_COMMIT = "e7bea685827137a1ea48c5886db22518d68a6b06"


def _run_git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo_root)


def _baseline_modules(repo_root: Path, commit: str) -> dict[str, bytes]:
    listed = _run_git(repo_root, "ls-tree", "-r", "--name-only", commit, "revision_R1/revision_r1").decode().splitlines()
    modules: dict[str, bytes] = {}
    for relative in listed:
        if relative.endswith(".py"):
            modules[Path(relative).name] = _run_git(repo_root, "show", f"{commit}:{relative}")
    if not modules:
        raise ValueError(f"baseline commit {commit} contains no revision_R1 Python modules")
    return modules


def _source_digest(modules: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(modules):
        digest.update(name.encode())
        digest.update(modules[name])
    return digest.hexdigest()


def build_manifest(package_root: str | Path, *, baseline_commit: str = AUTHENTIC_BASELINE_COMMIT) -> dict[str, Any]:
    root = Path(package_root).resolve()
    repo_root = root.parent
    modules = _baseline_modules(repo_root, baseline_commit)
    baseline_source_hash = _source_digest(modules)
    if baseline_source_hash != CHECKPOINT_IDENTITY_SHA256:
        raise ValueError(
            "the selected Git baseline does not authenticate the existing checkpoint identity: "
            f"expected {CHECKPOINT_IDENTITY_SHA256}, observed {baseline_source_hash}"
        )

    current_hashes = runtime_module_hashes(root)
    baseline_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in modules.items()}
    changed = {
        name for name, baseline_hash in baseline_hashes.items()
        if current_hashes.get(name) != baseline_hash
    }
    added = set(current_hashes) - set(baseline_hashes)
    if changed != INFRASTRUCTURE_ONLY_CHANGES or added != ADDED_INFRASTRUCTURE_MODULES:
        raise ValueError(
            "working tree contains changes outside the reviewed checkpoint-only scope: "
            f"changed={sorted(changed)}, added={sorted(added)}"
        )

    baseline_fingerprint = training_behavior_fingerprint(modules["runner.py"])
    current_fingerprint = training_behavior_fingerprint((root / "revision_r1" / "runner.py").read_bytes())
    if baseline_fingerprint != current_fingerprint:
        raise ValueError("training phase state-machine code changed from the authentic checkpoint baseline")

    return {
        "schema_version": 1,
        "compatibility_kind": "CHECKPOINT_ONLY_RUNTIME_UPDATE",
        "baseline_commit_sha": baseline_commit,
        "baseline_training_source_sha256": baseline_source_hash,
        "baseline_module_sha256": baseline_hashes,
        "runtime_source_sha256": runtime_source_sha256(root),
        "runtime_module_sha256": current_hashes,
        "changed_infrastructure_modules": sorted(changed),
        "added_compatibility_modules": sorted(added),
        "training_behavior_fingerprint_sha256": current_fingerprint,
        "baseline_training_behavior_fingerprint_sha256": baseline_fingerprint,
        "training_behavior_matches_baseline": True,
        "checkpoint_schema_unchanged": True,
        "resume_identity_policy": "preserve the authentic historical identity only after exact baseline and runtime source verification; never rewrite checkpoint identity",
        "checkpoint_cadence_policy": {
            "task_boundary_checkpoints": 11,
            "mid_task_checkpoints": False,
            "publication_order": "checkpoint_then_matrix_progress_json",
            "retention": "retain_each_task_boundary_checkpoint",
        },
    }


def write_manifest(package_root: str | Path, *, baseline_commit: str = AUTHENTIC_BASELINE_COMMIT) -> dict[str, Any]:
    root = Path(package_root).resolve()
    manifest = build_manifest(root, baseline_commit=baseline_commit)
    payload = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    target = root / "checkpoint_compatibility.json"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    checked = verify_checkpoint_compatibility(root)
    if checked != manifest:
        raise RuntimeError("written checkpoint compatibility manifest failed read-back")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-commit", default=AUTHENTIC_BASELINE_COMMIT)
    args = parser.parse_args()
    manifest = write_manifest(args.package_root, baseline_commit=args.baseline_commit)
    print(json.dumps({
        "status": "PASS",
        "baseline_commit_sha": manifest["baseline_commit_sha"],
        "baseline_training_source_sha256": manifest["baseline_training_source_sha256"],
        "runtime_source_sha256": manifest["runtime_source_sha256"],
        "changed_infrastructure_modules": manifest["changed_infrastructure_modules"],
        "training_behavior_fingerprint_sha256": manifest["training_behavior_fingerprint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
