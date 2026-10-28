"""Separate lightweight result exports from complete checkpoint archives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from .validation import validate_run_artifacts


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_zip(path: Path, files: list[tuple[Path, str]], manifest: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source, relative in files:
                archive.write(source, relative)
            archive.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    result = {
        "archive": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "file_count": len(files),
        "manifest": manifest,
    }
    _atomic_json(Path(str(path) + ".manifest.json"), result)
    return result


def build_lightweight_results_archive(run_roots: list[str | Path], output_zip: str | Path) -> dict:
    """Bundle validated metrics/predictions while excluding all model/checkpoint weights."""
    files: list[tuple[Path, str]] = []
    entries: dict[str, dict] = {}
    excluded_dirs = {"checkpoints", "registry", "__pycache__"}
    for raw_root in run_roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        status_path = root / "status.json"
        if not status_path.is_file():
            raise ValueError(f"run has no status.json: {root}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") not in {"COMPLETED_VALIDATED", "SYNTHETIC_VALIDATED_NON_REPORTABLE"}:
            raise ValueError(f"run is not validated: {root}")
        validate_run_artifacts(root, require_production=status["status"] == "COMPLETED_VALIDATED")
        for source in sorted(root.rglob("*")):
            if not source.is_file() or any(part in excluded_dirs for part in source.relative_to(root).parts):
                continue
            if source.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
                continue
            relative = (Path(root.name) / source.relative_to(root)).as_posix()
            if relative in entries:
                raise ValueError(f"archive path collision: {relative}")
            files.append((source, relative))
            entries[relative] = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    if not files:
        raise ValueError("no validated result artifacts found")
    manifest = {
        "schema_version": 1,
        "archive_kind": "LIGHTWEIGHT_RESULTS_NO_MODEL_CHECKPOINTS",
        "training_reproduction_claimed": False,
        "files": entries,
    }
    return _atomic_zip(Path(output_zip), files, manifest)


def build_checkpoint_archive(checkpoint_root: str | Path, output_zip: str | Path) -> dict:
    """Archive every verified recovery and task-boundary checkpoint generation."""
    root = Path(checkpoint_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifests = sorted(root.rglob("generation-*.manifest.json"))
    if not manifests:
        raise ValueError(f"no checkpoint generation manifests under {root}")
    files: list[tuple[Path, str]] = []
    entries: dict[str, dict] = {}
    for manifest_path in manifests:
        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint manifest: {manifest_path}") from exc
        payload_name = record.get("payload_file")
        if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
            raise ValueError(f"unsafe or missing payload reference in {manifest_path}")
        payload_path = manifest_path.parent / payload_name
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        payload_hash = sha256_file(payload_path)
        if payload_path.stat().st_size != int(record.get("payload_bytes", -1)) or payload_hash != record.get("payload_sha256"):
            raise ValueError(f"checkpoint payload checksum mismatch: {payload_path}")
        for source in (manifest_path, payload_path):
            relative = source.relative_to(root).as_posix()
            files.append((source, relative))
            entries[relative] = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    manifest = {
        "schema_version": 1,
        "archive_kind": "FULL_CHECKPOINT_GENERATIONS",
        "checkpoint_root": str(root),
        "generation_count": len(manifests),
        "files": entries,
    }
    return _atomic_zip(Path(output_zip), files, manifest)
