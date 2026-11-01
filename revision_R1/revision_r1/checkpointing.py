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

DEFAULT_LOCAL_RECOVERY_INTERVAL_SECONDS = 300.0
DEFAULT_DRIVE_PUBLISH_INTERVAL_SECONDS = 900.0


class CheckpointError(RuntimeError):
    pass


@dataclass
class LoadedCheckpoint:
    state: RunState
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CheckpointPolicy:
    update_interval: int | None = None
    seconds_interval: float = DEFAULT_LOCAL_RECOVERY_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if self.update_interval is not None and self.update_interval < 1:
            raise ValueError("legacy update interval must be positive when supplied")
        if self.seconds_interval <= 0:
            raise ValueError("checkpoint seconds interval must be positive")

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
            or now - last_checkpoint_time >= self.seconds_interval
        )


class CheckpointSchedule:
    """Completion-based local-save and Drive-publication deadlines."""

    def __init__(
        self, *, local_interval_seconds: float = DEFAULT_LOCAL_RECOVERY_INTERVAL_SECONDS,
        drive_interval_seconds: float = DEFAULT_DRIVE_PUBLISH_INTERVAL_SECONDS, clock=time.monotonic,
    ) -> None:
        if local_interval_seconds <= 0 or drive_interval_seconds <= 0:
            raise ValueError("checkpoint cadence intervals must be positive")
        self.local_interval_seconds = float(local_interval_seconds)
        self.drive_interval_seconds = float(drive_interval_seconds)
        self._clock = clock
        completed = float(clock())
        self.last_local_completion = completed
        self.last_drive_completion = completed

    def local_due(self, *, force: bool = False) -> bool:
        return bool(force or float(self._clock()) - self.last_local_completion >= self.local_interval_seconds)

    def drive_due(self, *, force: bool = False) -> bool:
        return bool(force or float(self._clock()) - self.last_drive_completion >= self.drive_interval_seconds)

    def mark_local_complete(self) -> None:
        self.last_local_completion = float(self._clock())

    def mark_drive_complete(self) -> None:
        self.last_drive_completion = float(self._clock())

    def metadata(self, *, legacy_config_fields: dict[str, int] | None = None) -> dict[str, Any]:
        return {
            "local_recovery_interval_seconds": self.local_interval_seconds,
            "drive_publish_interval_seconds": self.drive_interval_seconds,
            "clock": "time.monotonic",
            "deadline_anchor": "previous_checkpoint_cycle_completion",
            "forced_events": [
                "initial", "task-boundary", "graceful-interruption",
                "failed-recovery", "terminal-completion",
            ],
            "legacy_config_fields_preserved_for_run_identity": dict(legacy_config_fields or {}),
            "runtime_policy_overrides_legacy_operational_fields": True,
        }


class CheckpointMirrorResult(list):
    """List-compatible mirror result carrying separated transfer measurements."""

    def __init__(self, manifests=(), *, timings=None, bytes_written=0, retained_bytes=None):
        super().__init__(manifests)
        self.timings = dict(timings or {})
        self.bytes_written = int(bytes_written)
        self.retained_bytes = retained_bytes


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


def next_generation_number(root: str | Path, identity: dict[str, Any]) -> int:
    """Return the next unused generation, including orphan payload/manifest names.

    The manifest scan is metadata-only, so a fresh Colab runtime can continue a
    separate task-boundary archive without downloading its large payloads.
    """
    directory = Path(root)
    highest = 0
    if not directory.is_dir():
        return 1
    for path in directory.glob("generation-*"):
        match = re.fullmatch(r"generation-(\d+)\.(?:pt|manifest\.json)", path.name)
        if match is None:
            continue
        generation = int(match.group(1))
        highest = max(highest, generation)
        if path.name.endswith(".manifest.json"):
            manifest = _read_manifest(path)
            if manifest is not None:
                if int(manifest["generation"]) != generation:
                    raise CheckpointError(f"generation manifest/name mismatch: {path.name}")
                if _canonical(manifest.get("identity")) != _canonical(identity):
                    raise CheckpointError("checkpoint identity mismatch in task-boundary archive")
    return highest + 1


def _manifest_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _set_self_sized_byte_field(value: dict[str, Any], key: str, base_bytes: int = 0) -> int:
    """Solve the small JSON-size fixed point for a manifest's own write counter."""
    accounting = value["checkpoint_accounting"]
    candidate = int(base_bytes)
    while True:
        accounting[key] = candidate
        observed = int(base_bytes) + len(_manifest_bytes(value))
        if observed == candidate:
            return candidate
        candidate = observed


def _write_json_atomic(path: Path, value: dict[str, Any]) -> int:
    payload = _manifest_bytes(value)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload.decode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        # Read back before publication to reject truncated/invalid serialization.
        with open(temp_name, encoding="utf-8") as stream:
            if json.load(stream) != value:
                raise CheckpointError("manifest read-back differs before publication")
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
        return len(payload)
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
    prior_manifests = []
    for path in directory.glob("generation-*.manifest.json"):
        previous = _read_manifest(path)
        if previous is None:
            continue
        if _canonical(previous.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch in target recovery directory")
        prior_manifests.append((int(previous["generation"]), previous))
    prior_manifests.sort(key=lambda item: item[0], reverse=True)
    prior_accounting = (
        prior_manifests[0][1].get("checkpoint_accounting", {}) if prior_manifests else {}
    )
    generation = (prior_manifests[0][0] if prior_manifests else 0) + 1
    payload, manifest_path = _generation_paths(directory, generation)
    fd, temp_name = tempfile.mkstemp(prefix=f".{payload.name}.", suffix=".tmp", dir=directory)
    serialization_started = time.monotonic()
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp_payload = Path(temp_name)
        serialization_seconds = time.monotonic() - serialization_started
        verification_started = time.monotonic()
        payload_hash = _sha256(temp_payload)
        payload_size = temp_payload.stat().st_size
        local_verification_seconds = time.monotonic() - verification_started
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
            "checkpoint_accounting": {
                "serialization_seconds": serialization_seconds,
                "local_verification_seconds": local_verification_seconds,
                "pruning_seconds": 0.0,
                "local_payload_write_bytes": payload_size,
                "local_payload_write_bytes_since_policy_start": (
                    int(prior_accounting.get("local_payload_write_bytes_since_policy_start", 0)) + payload_size
                ),
                "local_manifest_write_bytes": 0,
                "local_manifest_write_bytes_since_policy_start": int(
                    prior_accounting.get("local_manifest_write_bytes_since_policy_start", 0)
                ),
                "cumulative_local_write_history_complete": bool(
                    prior_accounting.get("cumulative_local_write_history_complete", False)
                ),
                "recovery_artifact_bytes_retained_after_pruning": None,
            },
        }
        first_manifest_write_bytes = _write_json_atomic(manifest_path, manifest)
        _validate_manifest(directory, manifest, identity, load_state=False, verify_payload=False)
        if retention_limit is not None:
            prune = _prune_verified_generations(
                directory, identity, keep=retention_limit, already_verified=(manifest,),
            )
            manifest["checkpoint_accounting"]["pruning_seconds"] = prune["pruning_seconds"]
            manifest["checkpoint_accounting"]["local_verification_seconds"] += prune["verification_seconds"]
        retained_except_current_manifest = sum(
            path.stat().st_size for path in directory.glob("generation-*")
            if path.is_file() and path != manifest_path
        )
        cumulative_manifest_bytes = int(
            prior_accounting.get("local_manifest_write_bytes_since_policy_start", 0)
        )
        current_manifest_bytes = _set_self_sized_byte_field(
            manifest, "local_manifest_write_bytes",
            base_bytes=first_manifest_write_bytes,
        )
        manifest["checkpoint_accounting"]["local_manifest_write_bytes_since_policy_start"] = (
            cumulative_manifest_bytes + current_manifest_bytes
        )
        while True:
            final_manifest_size = len(_manifest_bytes(manifest))
            expected_manifest_bytes = first_manifest_write_bytes + final_manifest_size
            expected_retained_bytes = retained_except_current_manifest + final_manifest_size
            retained_value = manifest["checkpoint_accounting"]["recovery_artifact_bytes_retained_after_pruning"]
            if (
                expected_manifest_bytes == current_manifest_bytes
                and expected_retained_bytes == retained_value
            ):
                break
            current_manifest_bytes = expected_manifest_bytes
            manifest["checkpoint_accounting"]["local_manifest_write_bytes"] = current_manifest_bytes
            manifest["checkpoint_accounting"]["local_manifest_write_bytes_since_policy_start"] = (
                cumulative_manifest_bytes + current_manifest_bytes
            )
            manifest["checkpoint_accounting"]["recovery_artifact_bytes_retained_after_pruning"] = expected_retained_bytes
        final_manifest_write_bytes = _write_json_atomic(manifest_path, manifest)
        if first_manifest_write_bytes + final_manifest_write_bytes != current_manifest_bytes:
            raise CheckpointError("local checkpoint manifest write-byte accounting did not reconcile")
        return manifest
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def copy_generation_payload(
    source_root: str | Path, source_manifest: dict[str, Any],
    destination_root: str | Path, identity: dict[str, Any], *,
    kind: str = "task-boundary", retain_generations: int | None = None,
    generation: int | None = None,
) -> dict[str, Any]:
    """Create a separate artifact stream by copying an already serialized payload."""
    source, destination = Path(source_root), Path(destination_root)
    if _canonical(source_manifest.get("identity")) != _canonical(identity):
        raise CheckpointError("copied checkpoint source identity mismatch")
    _validate_manifest(source, source_manifest, identity, load_state=False, verify_payload=False)
    retention_limit = effective_generation_limit(identity, retain_generations)
    if retention_limit is not None and retention_limit < 2:
        raise ValueError("at least two generations must be retained")
    destination.mkdir(parents=True, exist_ok=True)
    if identity.get("execution_scope") == "synthetic_validation":
        cleanup_unpublished_checkpoint_temps(destination)
    generation = next_generation_number(destination, identity) if generation is None else int(generation)
    if generation < 1:
        raise ValueError("checkpoint generation must be positive")
    payload, manifest_path = _generation_paths(destination, generation)
    if payload.exists() or manifest_path.exists():
        raise CheckpointError(f"refusing to overwrite existing task-boundary generation {generation}")
    source_payload = source / str(source_manifest["payload_file"])
    require_path_capacity(
        destination, required_bytes=int(source_manifest["payload_bytes"]),
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    fd, temp_name = tempfile.mkstemp(prefix=f".{payload.name}.", suffix=".tmp", dir=destination)
    copy_started = time.monotonic()
    try:
        with os.fdopen(fd, "wb") as output, source_payload.open("rb") as inp:
            shutil.copyfileobj(inp, output)
            output.flush()
            os.fsync(output.fileno())
        copy_seconds = time.monotonic() - copy_started
        verify_started = time.monotonic()
        temporary = Path(temp_name)
        payload_size = temporary.stat().st_size
        payload_hash = _sha256(temporary)
        verification_seconds = time.monotonic() - verify_started
        if payload_size != int(source_manifest["payload_bytes"]) or payload_hash != source_manifest["payload_sha256"]:
            raise CheckpointError("task-boundary payload copy failed hash/size verification")
        os.replace(temporary, payload)
        _fsync_dir(destination)
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
            "copied_from_recovery_generation": int(source_manifest["generation"]),
            "checkpoint_accounting": {
                "serialization_seconds": 0.0,
                "local_artifact_copy_seconds": copy_seconds,
                "local_verification_seconds": verification_seconds,
                "local_artifact_copy_bytes": payload_size,
                "local_manifest_write_bytes": 0,
            },
        }
        manifest_write_bytes = _set_self_sized_byte_field(
            manifest, "local_manifest_write_bytes",
        )
        if _write_json_atomic(manifest_path, manifest) != manifest_write_bytes:
            raise CheckpointError("task-boundary manifest write-byte accounting did not reconcile")
        _validate_manifest(destination, manifest, identity, load_state=False, verify_payload=False)
        if retention_limit is not None:
            _prune_verified_generations(destination, identity, keep=retention_limit, already_verified=(manifest,))
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


def _validate_manifest(
    root: Path, manifest: dict[str, Any], identity: dict[str, Any], *,
    load_state: bool = True, verify_payload: bool = True,
) -> RunState | None:
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
    if payload.stat().st_size != manifest.get("payload_bytes"):
        raise CheckpointError("manifest payload size validation failed")
    if verify_payload and _sha256(payload) != manifest.get("payload_sha256"):
        raise CheckpointError("manifest payload hash validation failed")
    if not load_state:
        return None
    require_host_memory(
        int(manifest["payload_bytes"]), reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="checkpoint payload deserialization for recovery",
    )
    state = _torch_load(payload)
    if _canonical(state.identity) != _canonical(identity):
        raise CheckpointError("payload identity does not match run namespace")
    return state


def _prune_verified_generations(
    root: Path, identity: dict[str, Any], *, keep: int,
    already_verified: tuple[dict[str, Any], ...] = (), delete_callback=None,
) -> dict[str, Any]:
    """Prune only after `keep` surviving payloads are verified; never delete the last point."""
    if keep < 1:
        raise ValueError("checkpoint pruning must retain at least one generation")
    verification_seconds = 0.0
    records = []
    for path in root.glob("generation-*.manifest.json"):
        manifest = _read_manifest(path)
        if manifest is None:
            continue
        if _canonical(manifest.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch while retaining generations")
        records.append((int(manifest["generation"]), path, manifest))
    records.sort(key=lambda item: item[0], reverse=True)
    known = {int(manifest["generation"]): manifest for manifest in already_verified}
    valid = []
    for generation, path, manifest in records:
        try:
            started = time.monotonic()
            _validate_manifest(
                root, manifest, identity,
                load_state=False,
                verify_payload=generation not in known,
            )
            if generation not in known:
                verification_seconds += time.monotonic() - started
        except CheckpointError:
            continue
        valid.append((generation, path, manifest))
        if len(valid) >= keep:
            break
    if len(valid) < keep:
        return {
            "pruning_seconds": 0.0,
            "verification_seconds": verification_seconds,
            "deleted_bytes": 0,
            "retained_bytes": sum(path.stat().st_size for path in root.glob("generation-*") if path.is_file()),
            "pruning_skipped": "fewer than keep verified recovery generations",
            "pruning_deferred_files": [],
        }

    retained = {generation for generation, _, _ in valid[:keep]}
    deleted_bytes = 0
    deferred_files: list[dict[str, Any]] = []
    pruning_started = time.monotonic()
    for generation, manifest_path, manifest in records:
        if generation in retained:
            continue
        payload_name = manifest.get("payload_file")
        if isinstance(payload_name, str) and re.fullmatch(r"generation-\d+\.pt", payload_name):
            payload_path = root / payload_name
            if payload_path.is_file():
                payload_size = payload_path.stat().st_size
                if delete_callback is None:
                    payload_path.unlink()
                else:
                    if delete_callback(payload_path) is False:
                        deferred_files.append({"generation": generation, "file": payload_path.name})
                        # Leave the manifest paired with a payload that Drive
                        # says this account cannot delete.
                        continue
                deleted_bytes += payload_size
        if manifest_path.is_file():
            manifest_size = manifest_path.stat().st_size
            if delete_callback is None:
                manifest_path.unlink()
            else:
                if delete_callback(manifest_path) is False:
                    deferred_files.append({"generation": generation, "file": manifest_path.name})
                    continue
            deleted_bytes += manifest_size
    deferred_files.sort(key=lambda item: (item["generation"], item["file"]))
    return {
        "pruning_seconds": time.monotonic() - pruning_started,
        "verification_seconds": verification_seconds,
        "deleted_bytes": deleted_bytes,
        "retained_bytes": sum(path.stat().st_size for path in root.glob("generation-*") if path.is_file()),
        "pruning_skipped": (
            "Drive deletion capability was not granted for one or more stale checkpoint files"
            if deferred_files else None
        ),
        "pruning_deferred_files": deferred_files,
    }


def latest_valid_manifest(root: str | Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Find the newest hash-verified generation without deserializing its state."""
    directory = Path(root)
    manifests = []
    for path in directory.glob("generation-*.manifest.json"):
        manifest = _read_manifest(path)
        if manifest is not None:
            manifests.append((int(manifest["generation"]), manifest))
    manifests.sort(key=lambda item: item[0], reverse=True)
    if not manifests:
        raise CheckpointError(f"no valid checkpoint manifest found under {directory}")
    for _, manifest in manifests:
        if _canonical(manifest.get("identity")) != _canonical(identity):
            raise CheckpointError("checkpoint identity mismatch; use a new run namespace")
    errors = []
    for _, manifest in manifests:
        try:
            _validate_manifest(directory, manifest, identity, load_state=False, verify_payload=True)
            return manifest
        except CheckpointError as exc:
            errors.append(str(exc))
    raise CheckpointError("no valid checkpoint generation: " + "; ".join(errors))


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


def load_verified_manifest(
    root: str | Path, identity: dict[str, Any], manifest: dict[str, Any],
) -> LoadedCheckpoint:
    """Deserialize a generation already hash-verified in this checkpoint cycle."""
    directory = Path(root)
    state = _validate_manifest(directory, manifest, identity, load_state=True, verify_payload=False)
    assert state is not None
    return LoadedCheckpoint(state, manifest)


def mirror_checkpoint_generations(
    local_root: str | Path,
    drive_root: str | Path,
    identity: dict[str, Any],
    *,
    keep: int | None = 2,
    prune_destination: bool = True,
    generation: int | None = None,
    delete_callback=None,
    preverified_manifests: tuple[dict[str, Any], ...] = (),
) -> CheckpointMirrorResult:
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
        candidates.append((path, manifest))
    candidates.sort(key=lambda item: item[1]["generation"], reverse=True)
    copied = []
    if generation is None:
        selected = candidates if effective_keep is None else candidates[:effective_keep]
    else:
        selected = [item for item in candidates if int(item[1]["generation"]) == int(generation)]
        if not selected:
            raise CheckpointError(f"requested generation {generation} is absent from {source}")
    timings = {"upload_seconds": 0.0, "verification_seconds": 0.0, "manifest_seconds": 0.0, "pruning_seconds": 0.0}
    bytes_written = 0
    preverified = {int(item["generation"]): item for item in preverified_manifests}
    verified_source = []
    source_started = time.monotonic()
    for source_manifest_path, manifest in selected:
        supplied = preverified.get(int(manifest["generation"]))
        if supplied is not None:
            if supplied != manifest:
                raise CheckpointError("preverified checkpoint manifest differs from the local source manifest")
            try:
                _validate_manifest(source, manifest, identity, load_state=False, verify_payload=False)
            except CheckpointError:
                continue
            verified_source.append((source_manifest_path, manifest))
            continue
        try:
            _validate_manifest(source, manifest, identity, load_state=False, verify_payload=True)
        except CheckpointError:
            continue
        timings["verification_seconds"] += time.monotonic() - source_started
        verified_source.append((source_manifest_path, manifest))
        source_started = time.monotonic()
    if selected and not verified_source:
        raise CheckpointError("no valid checkpoint generation survived source hash validation")

    for source_manifest_path, manifest in verified_source:
        source_payload = source / manifest["payload_file"]
        dest_payload = destination / manifest["payload_file"]
        _, dest_manifest = _generation_paths(destination, manifest["generation"])
        existing = _read_manifest(dest_manifest)
        if existing is not None and existing != manifest:
            raise CheckpointError(
                f"refusing to overwrite a different published manifest for generation {manifest['generation']}"
            )
        if existing == manifest and dest_payload.is_file() and dest_payload.stat().st_size == int(manifest["payload_bytes"]):
            verify_started = time.monotonic()
            if _sha256(dest_payload) != manifest["payload_sha256"]:
                existing = None
            timings["verification_seconds"] += time.monotonic() - verify_started
            if existing == manifest:
                continue
        require_path_capacity(
            destination,
            required_bytes=int(manifest["payload_bytes"]),
            reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        )
        fd, temp_name = tempfile.mkstemp(prefix=f".{dest_payload.name}.", suffix=".tmp", dir=destination)
        try:
            upload_started = time.monotonic()
            with os.fdopen(fd, "wb") as output, source_payload.open("rb") as inp:
                shutil.copyfileobj(inp, output)
                output.flush()
                os.fsync(output.fileno())
            timings["upload_seconds"] += time.monotonic() - upload_started
            temporary = Path(temp_name)
            if temporary.stat().st_size != manifest["payload_bytes"]:
                raise CheckpointError("staged checkpoint payload size differs from verified source")
            os.replace(temporary, dest_payload)
            _fsync_dir(destination)
            verify_started = time.monotonic()
            if _sha256(dest_payload) != manifest["payload_sha256"]:
                raise CheckpointError("published checkpoint payload differs from verified source")
            timings["verification_seconds"] += time.monotonic() - verify_started
            manifest_started = time.monotonic()
            _write_json_atomic(dest_manifest, manifest)
            _validate_manifest(destination, manifest, identity, load_state=False, verify_payload=False)
            timings["manifest_seconds"] += time.monotonic() - manifest_started
            copied.append(manifest)
            bytes_written += int(manifest["payload_bytes"]) + dest_manifest.stat().st_size
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    if (prune_destination or synthetic) and effective_keep is not None:
        prune = _prune_verified_generations(
            destination, identity, keep=effective_keep,
            already_verified=tuple(copied), delete_callback=delete_callback,
        )
        timings["verification_seconds"] += prune["verification_seconds"]
        timings["pruning_seconds"] += prune["pruning_seconds"]
        timings["pruning_deleted_bytes"] = float(prune["deleted_bytes"])
        timings["pruning_skipped"] = prune["pruning_skipped"]
        timings["pruning_deferred_files"] = prune["pruning_deferred_files"]
    retained_bytes = sum(
        path.stat().st_size for path in destination.glob("generation-*") if path.is_file()
    )
    return CheckpointMirrorResult(
        sorted(copied, key=lambda item: item["generation"]),
        timings=timings, bytes_written=bytes_written, retained_bytes=retained_bytes,
    )
