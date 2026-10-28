"""Integrity-checked, manifest-last checkpoint generations and Drive mirroring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .state import RunState
from .storage import (
    DEFAULT_STORAGE_RESERVE_BYTES,
    cleanup_unpublished_checkpoint_temps,
    effective_generation_limit,
    estimate_checkpoint_bytes,
    require_path_capacity,
)
from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory


class CheckpointError(RuntimeError):
    pass


@dataclass
class LoadedCheckpoint:
    state: RunState
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CheckpointPolicy:
    update_interval: int = 25
    seconds_interval: float = 60.0

    def __post_init__(self) -> None:
        if self.update_interval < 1 or self.seconds_interval <= 0:
            raise ValueError("checkpoint intervals must be positive")

    def should_checkpoint(
        self,
        *,
        last_checkpoint_update: int,
        update: int,
        last_checkpoint_time: float,
        now: float,
        phase_boundary: bool = False,
    ) -> bool:
        return bool(
            phase_boundary
            or update - last_checkpoint_update >= self.update_interval
            or now - last_checkpoint_time >= self.seconds_interval
        )


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"run identity must be finite JSON data: {exc}") from exc


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some mounted Drive/FUSE filesystems do not expose directory fsync.
        pass


def _torch_load(path: Path) -> RunState:
    # Checkpoints are trusted only when produced by this run package. Never load
    # user-supplied or downloaded pickle files through this path.
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"checkpoint payload cannot be decoded: {exc}") from exc
    if not isinstance(state, RunState):
        raise CheckpointError("checkpoint payload has an unexpected object type")
    state.validate()
    return state


def _generation_paths(root: Path, generation: int) -> tuple[Path, Path]:
    stem = f"generation-{generation:06d}"
    return root / f"{stem}.pt", root / f"{stem}.manifest.json"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Read back before publication to reject truncated/invalid serialization.
        with open(temp_name, encoding="utf-8") as stream:
            if json.load(stream) != value:
                raise CheckpointError("manifest read-back differs before publication")
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_generation(
    root: str | Path,
    state: RunState,
    identity: dict[str, Any],
    kind: str,
    *,
    retain_generations: int | None = 2,
) -> dict[str, Any]:
    """Publish payload first and its hash manifest last; unreferenced files are ignored."""
    directory = Path(root)
    retention_limit = effective_generation_limit(identity, retain_generations)
    if retention_limit is not None and retention_limit < 2:
        raise ValueError("at least two verified recovery generations must be retained")
    directory.mkdir(parents=True, exist_ok=True)
    if identity.get("execution_scope") == "synthetic_validation":
        cleanup_unpublished_checkpoint_temps(directory)
    state.validate()
    if _canonical(state.identity) != _canonical(identity):
        raise CheckpointError("state identity differs from checkpoint identity")
    require_path_capacity(
        directory,
        required_bytes=estimate_checkpoint_bytes(state),
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    require_host_memory(
        estimate_checkpoint_bytes(state), reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="checkpoint serialization and integrity read-back",
    )
    generations = [
        int(match.group(1))
        for path in directory.glob("generation-*.manifest.json")
        if (match := re.fullmatch(r"generation-(\d+)\.manifest\.json", path.name))
    ]
    generation = max(generations, default=0) + 1
    payload, manifest_path = _generation_paths(directory, generation)
    fd, temp_name = tempfile.mkstemp(prefix=f".{payload.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp_payload = Path(temp_name)
        decoded = _torch_load(temp_payload)
        if _canonical(decoded.identity) != _canonical(identity):
            raise CheckpointError("checkpoint read-back identity mismatch")
        # The full identity/hash validation below deserializes the published
        # payload again. Release this first read-back before that second load
        # so two complete RunState object graphs do not overlap in host RAM.
        del decoded
        payload_hash = _sha256(temp_payload)
        payload_size = temp_payload.stat().st_size
        os.replace(temp_payload, payload)
        _fsync_dir(directory)
        manifest = {
            "schema_version": 1,
            "generation": generation,
            "kind": str(kind),
            "created_unix": time.time(),
            "identity": identity,
            "identity_sha256": _digest_bytes(_canonical(identity)),
            "payload_file": payload.name,
            "payload_bytes": payload_size,
            "payload_sha256": payload_hash,
        }
        _write_json_atomic(manifest_path, manifest)
        _validate_manifest(directory, manifest, identity)
        if retention_limit is not None:
            _prune_verified_generations(directory, identity, keep=retention_limit)
        return manifest
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return None
        if not isinstance(value.get("generation"), int):
            return None
        return value
    except (OSError, json.JSONDecodeError):
        return None


def _validate_manifest(root: Path, manifest: dict[str, Any], identity: dict[str, Any]) -> RunState:
    if _canonical(manifest.get("identity")) != _canonical(identity):
        raise CheckpointError("checkpoint identity mismatch; use a new run namespace")
    if manifest.get("identity_sha256") != _digest_bytes(_canonical(identity)):
        raise CheckpointError("manifest identity digest is invalid")
    name = manifest.get("payload_file", "")
    if not isinstance(name, str) or not re.fullmatch(r"generation-\d+\.pt", name):
        raise CheckpointError("manifest payload path is invalid")
    payload = root / name
    if not payload.is_file():
        raise CheckpointError("manifest payload is missing")
    if payload.stat().st_size != manifest.get("payload_bytes") or _sha256(payload) != manifest.get("payload_sha256"):
        raise CheckpointError("manifest payload hash/size validation failed")
    require_host_memory(
        int(manifest["payload_bytes"]), reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="checkpoint payload deserialization for recovery",
    )
    state = _torch_load(payload)
    if _canonical(state.identity) != _canonical(identity):
        raise CheckpointError("payload identity does not match run namespace")
    return state


def _prune_verified_generations(root: Path, identity: dict[str, Any], *, keep: int) -> None:
    valid = []
    for path in root.glob("generation-*.manifest.json"):
        manifest = _read_manifest(path)
        if manifest is None:
            continue
        if _canonical(manifest.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch while retaining generations")
        try:
            _validate_manifest(root, manifest, identity)
        except CheckpointError:
            continue
        valid.append((manifest["generation"], path, manifest))
    valid.sort(key=lambda item: item[0], reverse=True)
    for _, manifest_path, manifest in valid[keep:]:
        payload_name = manifest.get("payload_file")
        if isinstance(payload_name, str) and re.fullmatch(r"generation-\d+\.pt", payload_name):
            (root / payload_name).unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)


def load_latest_valid(root: str | Path, identity: dict[str, Any]) -> LoadedCheckpoint:
    directory = Path(root)
    manifests = []
    for path in directory.glob("generation-*.manifest.json"):
        manifest = _read_manifest(path)
        if manifest is not None:
            manifests.append((manifest["generation"], path, manifest))
    manifests.sort(key=lambda item: item[0], reverse=True)
    if not manifests:
        raise CheckpointError(f"no valid checkpoint manifest found under {directory}")
    # A namespace is immutable; do not silently fall back into another run.
    for _, _, manifest in manifests:
        if _canonical(manifest.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch; use a new run namespace")
    errors = []
    for _, _, manifest in manifests:
        try:
            return LoadedCheckpoint(_validate_manifest(directory, manifest, identity), manifest)
        except CheckpointError as exc:
            errors.append(str(exc))
    raise CheckpointError("no valid checkpoint generation: " + "; ".join(errors))


def mirror_checkpoint_generations(
    local_root: str | Path,
    drive_root: str | Path,
    identity: dict[str, Any],
    *,
    keep: int | None = 2,
    prune_destination: bool = True,
) -> list[dict[str, Any]]:
    """Copy newest valid local generations and verify Drive reads before reporting success.

    Google Drive/FUSE does not offer transactional rename/locking guarantees;
    payloads are copied to temporary names and manifests are published last.
    """
    effective_keep = effective_generation_limit(identity, keep)
    synthetic = identity.get("execution_scope") == "synthetic_validation"
    if effective_keep is not None and effective_keep < 2:
        raise ValueError("at least two generations must be retained")
    source, destination = Path(local_root), Path(drive_root)
    destination.mkdir(parents=True, exist_ok=True)
    if synthetic:
        cleanup_unpublished_checkpoint_temps(destination)
    candidates = []
    for path in source.glob("generation-*.manifest.json"):
        manifest = _read_manifest(path)
        if manifest is None:
            continue
        if _canonical(manifest.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch during Drive sync")
        try:
            _validate_manifest(source, manifest, identity)
        except CheckpointError:
            continue
        candidates.append(manifest)
    candidates.sort(key=lambda item: item["generation"], reverse=True)
    copied = []
    selected_manifests = candidates if effective_keep is None else candidates[:effective_keep]
    for manifest in selected_manifests:
        source_payload = source / manifest["payload_file"]
        dest_payload = destination / manifest["payload_file"]
        require_path_capacity(
            destination,
            required_bytes=int(manifest["payload_bytes"]),
            reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        )
        fd, temp_name = tempfile.mkstemp(prefix=f".{dest_payload.name}.", suffix=".tmp", dir=destination)
        try:
            with os.fdopen(fd, "wb") as output, source_payload.open("rb") as inp:
                shutil.copyfileobj(inp, output)
                output.flush()
                os.fsync(output.fileno())
            temporary = Path(temp_name)
            if temporary.stat().st_size != manifest["payload_bytes"] or _sha256(temporary) != manifest["payload_sha256"]:
                raise CheckpointError("Drive staging payload differs from local verified payload")
            os.replace(temporary, dest_payload)
            _fsync_dir(destination)
            _, dest_manifest = _generation_paths(destination, manifest["generation"])
            _write_json_atomic(dest_manifest, manifest)
            _validate_manifest(destination, manifest, identity)
            copied.append(manifest)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    if candidates and not copied:
        raise CheckpointError("no checkpoint generation survived Drive read-back validation")
    if (prune_destination or synthetic) and effective_keep is not None:
        _prune_verified_generations(destination, identity, keep=effective_keep)
    return sorted(copied, key=lambda item: item["generation"])
