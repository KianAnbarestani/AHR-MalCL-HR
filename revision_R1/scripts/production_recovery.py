"""Prepare an existing checkpoint for Cell F without constructing a runner."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import shutil

from scripts import production_workflow as workflow


CANONICAL_SOURCE = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"


def api_run_registry_class(storage_adapter, base_registry=None):
    """The same artifact-first registry publication used by training and retry."""
    from revision_r1.registry import RunRegistry
    base = base_registry or RunRegistry

    class ApiRunRegistry(base):
        def begin_attempt(self, run_id, *, identity, completion_only=False):
            if _canonical_completion_documents(storage_adapter, self, run_id, identity) is not None:
                raise ValueError("canonical run is already completed; no replacement attempt may be started")
            result = super().begin_attempt(run_id, identity=identity)
            if completion_only:
                state = self.status(run_id)
                state["attempts"][-1]["operation"] = "completion_only"
                workflow._atomic_json(self._state_path(run_id), state)
            workflow._publish_api_tree(storage_adapter, self.root)
            return result

        def fail_attempt(self, run_id, reason):
            result = super().fail_attempt(run_id, reason)
            workflow._publish_api_tree(storage_adapter, self.root)
            return result

        def complete(self, run_id, artifacts):
            _publish_export_tree(storage_adapter, self.root.parent)
            result = super().complete(run_id, artifacts)
            workflow._publish_api_tree(storage_adapter, self.root)
            return result

    return ApiRunRegistry


def _publish_export_tree(storage, output):
    logical = storage.logical_path_for_local(output)
    return storage.publish_tree(output, logical, include=lambda path, _local:
        workflow._api_storage_include_nonlease(path, {}) and Path(path).relative_to(logical).parts[0] != "registry")


def _canonical_completion_documents(storage, registry, run_id, identity):
    evidence = storage.local_staging_root / ".completion_registry_readback" / run_id
    evidence.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in (registry._state_path(run_id), registry._run_dir(run_id) / "COMPLETE.json"):
        try:
            target = evidence / path.name
            storage.download_file(storage.logical_path_for_local(path), target)
            documents.append(json.loads(target.read_text()))
        except FileNotFoundError:
            documents.append(None)
    state, marker = documents
    if not isinstance(state, dict) or state.get("status") != "completed":
        return None
    if (state.get("identity") != identity or not isinstance(marker, dict)
            or marker.get("identity") != identity or marker.get("run_id") != run_id
            or not isinstance(marker.get("artifacts"), dict) or not marker["artifacts"]
            or state.get("completion_artifacts") != marker.get("artifacts")):
        raise ValueError("canonical completed registry/marker mismatch; preserved")
    for name, item in marker["artifacts"].items():
        path = registry.root.parent / name
        if (Path(name).name != name or not path.is_file() or path.is_symlink()
                or path.stat().st_size != item["bytes"] or workflow._hash_file_stream(path) != item["sha256"]):
            raise ValueError("canonical completion differs from local exported bytes; preserved")
    return state, marker


def _assert_remote_bytes_match(storage, local):
    """Never let a retry replace conflicting canonical immutable evidence."""
    logical = storage.logical_path_for_local(local)
    try:
        remote = storage.exact_file(logical)
    except FileNotFoundError:
        return
    if remote is None:
        return
    try:
        remote_size = int(remote["size"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"canonical artifact size metadata is unavailable; preserved: {logical}") from None
    if remote_size != local.stat().st_size:
        raise ValueError(f"canonical artifact size conflicts; preserved: {logical}")
    if remote.get("md5Checksum") is not None:
        if remote["md5Checksum"] != workflow._checkpoint_md5(local):
            raise ValueError(f"canonical artifact checksum conflicts; preserved: {logical}")
    else:
        evidence = storage.local_staging_root / ".completion_content_readback" / hashlib.sha256(str(logical).encode()).hexdigest()
        storage.download_file(logical, evidence, expected_sha256=workflow._hash_file_stream(local))


def _hydrate_missing(storage, logical, local):
    """Do not replace surviving same-runtime exports with stale remote bytes."""
    results = []
    for item in storage.list_files(logical):
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe canonical artifact name")
        path, target = Path(logical) / name, Path(local) / name
        if not workflow._api_storage_include_nonlease(path, {}):
            continue
        if target.is_symlink():
            raise ValueError(f"symlinked staging artifact; preserved: {target}")
        if item.get("mimeType") == "application/vnd.google-apps.folder":
            results.extend(_hydrate_missing(storage, path, target))
        elif item.get("mimeType") == "application/vnd.google-apps.shortcut":
            raise ValueError("implicit shortcut inside canonical artifacts is not permitted")
        elif not target.exists():
            # Per-file capacity checks reserve only the missing bytes, not a
            # second full copy of an already staged output/checkpoint tree.
            results.append(storage.download_file(path, target))
    return results


def _generation_files(directory):
    directory = Path(directory)
    names = set()
    if directory.is_dir():
        for path in directory.iterdir():
            if path.name.startswith("generation-"):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"checkpoint generation entry is not a regular file; preserved: {path}")
                names.add(path.name)
    return names


def _checkpoint_source_inventory(storage, staged_dir, local_dir):
    staged_dir, local_dir = Path(staged_dir), Path(local_dir)
    logical_dir = storage.logical_path_for_local(staged_dir)
    local_names = _generation_files(local_dir)
    staged_names = _generation_files(staged_dir)
    local_all = sorted(path.name for path in local_dir.iterdir() if path.is_file()) if local_dir.is_dir() else []
    staged_all = sorted(path.name for path in staged_dir.iterdir() if path.is_file()) if staged_dir.is_dir() else []
    try:
        remote_rows = storage.list_files(logical_dir)
    except FileNotFoundError:
        remote_rows = []
    remote = {}
    remote_journals = []
    journals = {"runtime_local": [], "api_staging": [], "canonical_drive": []}
    def local_journal_rows(directory):
        rows = []
        if not directory.is_dir():
            return rows
        for path in sorted(directory.iterdir()):
            if "journal" not in path.name.lower() and "publication" not in path.name.lower():
                continue
            row = {"name": path.name, "path": str(path), "is_symlink": path.is_symlink(),
                   "is_file": path.is_file()}
            if path.is_file() and not path.is_symlink():
                row.update({"size_bytes": path.stat().st_size,
                            "sha256": workflow._hash_file_stream(path)})
            rows.append(row)
        return rows
    for name in local_all:
        if "journal" in name.lower() or "publication" in name.lower():
            journals["runtime_local"].append(name)
    for name in staged_all:
        if "journal" in name.lower() or "publication" in name.lower():
            journals["api_staging"].append(name)
    for item in remote_rows:
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unsafe filename in canonical checkpoint directory")
        if "journal" in name.lower() or "publication" in name.lower():
            journals["canonical_drive"].append(name)
            remote_journals.append({
                "name": name, "file_id": str(item.get("id") or ""),
                "parent_ids": [str(value) for value in item.get("parents", [])],
                "size_bytes": item.get("size"), "md5_checksum": item.get("md5Checksum"),
                "mime_type": item.get("mimeType"),
                "owner": storage._owner(item) if hasattr(storage, "_owner") else None,
            })
        if not name.startswith("generation-"):
            continue
        if item.get("mimeType") in {"application/vnd.google-apps.folder", "application/vnd.google-apps.shortcut"}:
            raise ValueError(f"unexpected folder/shortcut in checkpoint directory: {name}")
        if name in remote:
            raise ValueError(f"ambiguous duplicate canonical checkpoint name; preserved: {name}")
        remote[name] = {
            "file_id": str(item.get("id") or ""), "size_bytes": item.get("size"),
            "md5_checksum": item.get("md5Checksum"), "mime_type": item.get("mimeType"),
            "parent_ids": [str(value) for value in item.get("parents", [])],
            "owner": storage._owner(item) if hasattr(storage, "_owner") else None,
        }
    return {
        "runtime_local_root": str(local_dir),
        "api_staging_root": str(staged_dir),
        "canonical_drive_path": str(logical_dir),
        "runtime_local_files": sorted(local_names),
        "api_staging_files_before_reconciliation": sorted(staged_names),
        "runtime_local_journal_files": sorted(journals["runtime_local"]),
        "api_staging_journal_files": sorted(journals["api_staging"]),
        "canonical_drive_journal_files": sorted(journals["canonical_drive"]),
        "canonical_drive_files_before_reconciliation": [
            {"name": name, **remote[name]} for name in sorted(remote)
        ],
        "publication_journals": {
            source: sorted(names) for source, names in journals.items()
        },
        "publication_journal_evidence": {
            "runtime_local": local_journal_rows(local_dir),
            "api_staging": local_journal_rows(staged_dir),
            "canonical_drive": sorted(remote_journals, key=lambda row: row["name"]),
        },
        "_remote_by_name": remote,
    }


def _copy_checkpoint_file_exclusive(source, target, *, expected_bytes, expected_sha256):
    """Copy a verified file without replacing an existing checkpoint artifact."""
    from revision_r1.storage import DEFAULT_STORAGE_RESERVE_BYTES, require_path_capacity

    source, target = Path(source), Path(target)
    verify_started = time.monotonic()
    if (not source.is_file() or source.is_symlink()
            or source.stat().st_size != int(expected_bytes)
            or workflow._hash_file_stream(source) != expected_sha256):
        raise ValueError(f"checkpoint source fails its manifest integrity; preserved: {source}")
    source_verify_seconds = time.monotonic() - verify_started
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        verify_started = time.monotonic()
        if (target.is_symlink() or not target.is_file()
                or target.stat().st_size != int(expected_bytes)
                or workflow._hash_file_stream(target) != expected_sha256):
            raise ValueError(f"conflicting checkpoint destination; preserved: {target}")
        return {"copied": False, "copy_seconds": 0.0,
                "verification_seconds": source_verify_seconds + time.monotonic() - verify_started}
    require_path_capacity(
        target.parent, required_bytes=int(expected_bytes),
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        copy_started = time.monotonic()
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        copy_seconds = time.monotonic() - copy_started
        verify_started = time.monotonic()
        if (temporary.stat().st_size != int(expected_bytes)
                or workflow._hash_file_stream(temporary) != expected_sha256):
            raise ValueError(f"checkpoint copy failed integrity verification; preserved: {target}")
        verification_seconds = source_verify_seconds + time.monotonic() - verify_started
        try:
            os.link(temporary, target)
        except FileExistsError:
            verify_started = time.monotonic()
            if (target.is_symlink() or not target.is_file()
                    or target.stat().st_size != int(expected_bytes)
                    or workflow._hash_file_stream(target) != expected_sha256):
                raise ValueError(f"checkpoint destination changed during copy; preserved: {target}")
            verification_seconds += time.monotonic() - verify_started
            return {"copied": False, "copy_seconds": copy_seconds,
                    "verification_seconds": verification_seconds}
        return {"copied": True, "copy_seconds": copy_seconds,
                "verification_seconds": verification_seconds}
    finally:
        temporary.unlink(missing_ok=True)


def _remote_checkpoint_matches(storage, local_path):
    local_path = Path(local_path)
    logical = storage.logical_path_for_local(local_path)
    try:
        metadata = storage.exact_file(logical)
    except FileNotFoundError:
        return None
    if metadata is None:
        return None
    parent_id = storage.folder_id(logical.parent, create=False)
    if ([str(value) for value in metadata.get("parents", [])] != [str(parent_id)]
            or metadata.get("mimeType", "").startswith("application/vnd.google-apps.")):
        raise ValueError(f"canonical checkpoint parent/type mismatch; preserved: {logical}")
    try:
        remote_size = int(metadata.get("size"))
    except (TypeError, ValueError):
        raise ValueError(f"canonical checkpoint size is unavailable; preserved: {logical}") from None
    if remote_size != local_path.stat().st_size:
        raise ValueError(f"canonical checkpoint size conflicts with staged bytes; preserved: {logical}")
    md5 = metadata.get("md5Checksum")
    if md5 is not None:
        if md5 != workflow._checkpoint_md5(local_path):
            raise ValueError(f"canonical checkpoint checksum conflicts with staged bytes; preserved: {logical}")
    else:
        temp_root = storage.local_staging_root / ".checkpoint_content_readback"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = temp_root / (hashlib.sha256(str(logical).encode()).hexdigest() + ".verify")
        try:
            storage.download_file(logical, temp, expected_sha256=workflow._hash_file_stream(local_path),
                                  expected_bytes=local_path.stat().st_size)
        finally:
            temp.unlink(missing_ok=True)
    return {
        "file_id": str(metadata.get("id") or ""),
        "parent_id": str(parent_id), "size_bytes": remote_size,
        "md5_checksum": md5, "owner": storage._owner(metadata) if hasattr(storage, "_owner") else None,
    }


def _load_verified_run_state(directory, manifest, identity):
    """Read cursor metadata only; mmap avoids copying tensor storage into host RAM."""
    import torch
    from revision_r1.state import RunState

    payload = Path(directory) / manifest["payload_file"]
    if (not payload.is_file() or payload.is_symlink()
            or payload.stat().st_size != manifest["payload_bytes"]
            or workflow._hash_file_stream(payload) != manifest["payload_sha256"]):
        raise ValueError(f"checkpoint payload fails size/SHA-256 before cursor inspection: {payload}")
    try:
        state = torch.load(payload, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        raise ValueError(f"verified checkpoint payload cannot be decoded for cursor inspection: {payload}") from exc
    if not isinstance(state, RunState):
        raise ValueError(f"checkpoint does not deserialize to the declared RunState type: {payload}")
    state.validate()
    if state.identity != identity:
        raise ValueError(f"checkpoint payload identity differs from its frozen run identity: {payload}")
    return state


def _validated_task_rows(state, *, through_task):
    runtime = state.extra.get("runtime") if isinstance(state.extra, dict) else None
    rows = runtime.get("task_rows") if isinstance(runtime, dict) else None
    if not isinstance(rows, list):
        raise ValueError("checkpoint cursor provenance lacks serialized runtime.task_rows")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("task_id")) is not int:
            raise ValueError("checkpoint runtime task rows contain an invalid task_id")
        ids.append(row["task_id"])
    expected = list(range(1, int(through_task) + 1))
    if ids != expected:
        raise ValueError(f"checkpoint cursor/task-row provenance mismatch: expected={expected}, observed={ids}")
    return ids


def _boundary_cursor_evidence(directory, manifest, identity, *, recovery_directory=None, run_record=None):
    state = _load_verified_run_state(directory, manifest, identity)
    cursor = state.cursor
    provenance = manifest.get("recovery_provenance")
    if provenance is not None:
        if type(cursor.task_index) is not int or type(cursor.phase) is not str:
            raise ValueError("recovered task-boundary cursor fields must be exact integers/strings")
        if any(type(getattr(cursor, name)) is not int or getattr(cursor, name) != 0
               for name in ("item_index", "substep_index", "update_index")):
            raise ValueError("recovered task-boundary cursor fields must be exact zero integers")
        source_generation = manifest.get("copied_from_recovery_generation")
        source_kind = provenance.get("source_kind") if isinstance(provenance, dict) else None
        if type(source_generation) is not int or source_generation <= 0:
            raise ValueError("recovered task-boundary source generation is invalid")
        if source_kind == "complete":
            expected_provenance = {
                "source_kind": "complete", "source_generation": source_generation,
                "task_id": 11, "validated_cursor": {"task_index": 11, "phase": "complete"},
            }
            if (provenance != expected_provenance or cursor.task_index != 11 or cursor.phase != "complete"):
                raise ValueError("derived final task-boundary recovery provenance is invalid")
            if recovery_directory is None or run_record is None:
                raise ValueError("derived final task-boundary lacks its recovery-source validation context")
            source_path = Path(recovery_directory) / f"generation-{source_generation:06d}.manifest.json"
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"terminal recovery source generation {source_generation} is unavailable")
            source_manifest = _manifest(source_path, identity)
            if source_manifest.get("generation") != source_generation or source_manifest.get("kind") != "complete":
                raise ValueError(f"terminal recovery source generation {source_generation} is not the exact complete state")
            if (source_manifest.get("payload_bytes") != manifest.get("payload_bytes")
                    or source_manifest.get("payload_sha256") != manifest.get("payload_sha256")):
                raise ValueError(f"terminal recovery source generation {source_generation} payload differs from derived boundary")
            _validate_terminal_completion_checkpoint(recovery_directory, source_manifest, identity, run_record)
            _validated_task_rows(state, through_task=11)
        elif source_kind == "task-boundary":
            task_id = provenance.get("task_id")
            cursor_record = provenance.get("validated_cursor")
            source_manifest_sha256 = provenance.get("source_manifest_sha256")
            source_payload_sha256 = provenance.get("source_payload_sha256")
            expected_phase = "export" if task_id == 11 else "scaler"
            expected_provenance = {
                "source_kind": "task-boundary", "source_generation": source_generation,
                "task_id": task_id, "validated_cursor": {"task_index": task_id, "phase": expected_phase},
                "source_manifest_sha256": source_manifest_sha256,
                "source_payload_sha256": source_payload_sha256,
            }
            if (type(task_id) is not int or not 1 <= task_id <= 11
                    or cursor_record != expected_provenance["validated_cursor"]
                    or provenance != expected_provenance
                    or not workflow._is_sha256(source_manifest_sha256)
                    or not workflow._is_sha256(source_payload_sha256)
                    or cursor.task_index != task_id or cursor.phase != expected_phase
                    or manifest.get("payload_sha256") != source_payload_sha256):
                raise ValueError("recovered task-boundary provenance/cursor is inconsistent")
            if state.sampler_state:
                raise ValueError("recovered task-boundary retains an active sampler")
            _validated_task_rows(state, through_task=task_id)
        else:
            raise ValueError("unsupported recovered task-boundary provenance kind")
        recovered = True
    else:
        if type(cursor.task_index) is not int or type(cursor.phase) is not str:
            raise ValueError("task-boundary cursor fields must be exact integers/strings")
        task_id = cursor.task_index
        if not 1 <= task_id <= 11:
            raise ValueError(f"task-boundary checkpoint cursor is outside tasks 1..11: {task_id}")
        expected_phase = "export" if task_id == 11 else "scaler"
        if cursor.phase != expected_phase:
            raise ValueError(
                f"task-boundary generation {manifest['generation']} has cursor phase {cursor.phase!r}; "
                f"task {task_id} requires {expected_phase!r}"
            )
        if any(type(getattr(cursor, name)) is not int or getattr(cursor, name) != 0
               for name in ("item_index", "substep_index", "update_index")):
            raise ValueError(f"task-boundary cursor is not at a clean task boundary: generation {manifest['generation']}")
        if state.sampler_state:
            raise ValueError(f"task-boundary cursor retains an active sampler: generation {manifest['generation']}")
        _validated_task_rows(state, through_task=task_id)
        recovered = False
    result = {
        "task_id": int(cursor.task_index),
        "cursor": {"task_index": int(cursor.task_index), "phase": str(cursor.phase)},
        "recovered_from_terminal_state": recovered,
    }
    del state
    return result


def _validate_terminal_completion_checkpoint(directory, manifest, identity, record):
    if manifest.get("kind") != "complete":
        raise ValueError("only a kind=complete recovery generation can supply final-task provenance")
    state = _load_verified_run_state(directory, manifest, identity)
    cursor = state.cursor
    if (type(cursor.task_index) is not int or type(cursor.phase) is not str
            or cursor.task_index != 11 or cursor.phase != "complete"):
        raise ValueError(
            "terminal recovery cursor does not prove completion of task 11: "
            f"observed task_index={cursor.task_index}, phase={cursor.phase!r}"
        )
    if any(type(getattr(cursor, name)) is not int or getattr(cursor, name) != 0
           for name in ("item_index", "substep_index", "update_index")):
        raise ValueError("terminal recovery cursor is not at the exact completed task boundary")
    _validated_task_rows(state, through_task=11)
    expected_updates = sum(int(value) for value in record["update_budgets_by_task"])
    if int(state.resources.get("classifier_updates", -1)) != expected_updates:
        raise ValueError("terminal recovery classifier-update count differs from the frozen schedule")
    truth, prediction = state.predictions.get("y_true_global"), state.predictions.get("y_pred_global")
    if truth is None or prediction is None or len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("terminal recovery checkpoint lacks validated final predictions")
    runtime = state.extra["runtime"]
    if runtime.get("class_order_sha256") not in (None, record.get("class_order_sha256")):
        raise ValueError("terminal recovery class-order hash differs from the frozen run record")
    result = {"task_index": 11, "phase": "complete"}
    del state
    return result


def _write_recovered_final_boundary(boundary_dir, recovery_dir, source_manifest, identity, record):
    """Derive only task 11's boundary artifact from an authenticated terminal state."""
    from revision_r1 import checkpointing

    validated_cursor = _validate_terminal_completion_checkpoint(
        recovery_dir, source_manifest, identity, record,
    )
    payload_source = Path(recovery_dir) / source_manifest["payload_file"]
    if (not payload_source.is_file() or payload_source.is_symlink()
            or payload_source.stat().st_size != source_manifest["payload_bytes"]
            or workflow._hash_file_stream(payload_source) != source_manifest["payload_sha256"]):
        raise ValueError("terminal recovery payload changed after cursor validation; preserved")
    boundary_dir.mkdir(parents=True, exist_ok=True)
    generation_files = _generation_files(boundary_dir)
    generation_ids = set()
    for name in generation_files:
        match = re.fullmatch(r"generation-(\d+)\.(?:pt|manifest\.json)", name)
        if match is None:
            raise ValueError(f"unrecognized task-boundary generation artifact; preserved: {name}")
        generation_ids.add(int(match.group(1)))
    valid_pair_ids = {
        int(path.name.removeprefix("generation-").removesuffix(".manifest.json"))
        for path in boundary_dir.glob("generation-*.manifest.json")
        if (boundary_dir / path.name.replace(".manifest.json", ".pt")).is_file()
    }
    incomplete = sorted(generation_ids - valid_pair_ids)
    recovery_provenance = {
        "source_kind": "complete", "source_generation": int(source_manifest["generation"]),
        "task_id": 11, "validated_cursor": validated_cursor,
    }
    candidate_generations = []
    for generation in incomplete:
        payload = boundary_dir / f"generation-{generation:06d}.pt"
        manifest_path = boundary_dir / f"generation-{generation:06d}.manifest.json"
        if payload.exists():
            if (payload.is_symlink() or payload.stat().st_size != source_manifest["payload_bytes"]
                    or workflow._hash_file_stream(payload) != source_manifest["payload_sha256"]):
                raise ValueError(f"unpaired task-boundary payload conflicts with terminal recovery; preserved: {payload}")
        if manifest_path.exists():
            existing = _manifest(manifest_path, identity)
            if (existing.get("kind") != "task-boundary"
                    or existing.get("payload_sha256") != source_manifest["payload_sha256"]
                    or existing.get("payload_bytes") != source_manifest["payload_bytes"]
                    or existing.get("copied_from_recovery_generation") != source_manifest["generation"]
                    or existing.get("recovery_provenance") != recovery_provenance):
                raise ValueError(f"unpaired task-boundary manifest is not this terminal recovery; preserved: {manifest_path}")
        candidate_generations.append(generation)
    if len(candidate_generations) > 1:
        raise ValueError(f"ambiguous incomplete final-boundary generations; preserved: {candidate_generations}")
    generation = candidate_generations[0] if candidate_generations else max(generation_ids, default=0) + 1
    payload_path = boundary_dir / f"generation-{generation:06d}.pt"
    manifest_path = boundary_dir / f"generation-{generation:06d}.manifest.json"
    copy_accounting = _copy_checkpoint_file_exclusive(
        payload_source, payload_path, expected_bytes=source_manifest["payload_bytes"],
        expected_sha256=source_manifest["payload_sha256"],
    )
    if manifest_path.exists():
        value = _manifest(manifest_path, identity)
        if value.get("recovery_provenance") != recovery_provenance:
            raise ValueError(f"existing final-boundary manifest provenance conflicts; preserved: {manifest_path}")
    else:
        manifest = {
            "schema_version": 1, "generation": generation, "kind": "task-boundary",
            "created_unix": time.time(), "identity": identity,
            "identity_sha256": hashlib.sha256(workflow._canonical_bytes(identity)).hexdigest(),
            "payload_file": payload_path.name, "payload_bytes": int(source_manifest["payload_bytes"]),
            "payload_sha256": str(source_manifest["payload_sha256"]),
            "copied_from_recovery_generation": int(source_manifest["generation"]),
            "recovery_provenance": recovery_provenance,
            "checkpoint_accounting": {
                "serialization_seconds": 0.0,
                "local_artifact_copy_seconds": copy_accounting["copy_seconds"],
                "local_verification_seconds": copy_accounting["verification_seconds"],
                "local_artifact_copy_bytes": int(source_manifest["payload_bytes"])
                    if copy_accounting["copied"] else 0,
                "local_artifact_copy_timing_complete": copy_accounting["copied"],
                "local_manifest_write_bytes": 0,
            },
        }
        manifest_bytes = checkpointing._set_self_sized_byte_field(manifest, "local_manifest_write_bytes")
        if checkpointing._write_json_atomic(manifest_path, manifest) != manifest_bytes:
            raise ValueError("derived final-boundary manifest write-byte accounting did not reconcile")
    return generation, _manifest(manifest_path, identity)


def _validate_export_evidence(plan, record, output):
    from revision_r1.validation import REQUIRED_ARTIFACTS, validate_run_artifacts
    validation = validate_run_artifacts(output, require_production=True)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    if not (set(REQUIRED_ARTIFACTS) - {"artifact_manifest.json"}) <= set(manifest["files"]):
        raise ValueError("export manifest does not authenticate every required artifact")
    for name in manifest["files"]:
        if Path(name).name != name or (output / name).is_symlink():
            raise ValueError("unsafe exported artifact path; preserved")
    config = json.loads((output / "config.json").read_text())
    result = json.loads((output / "full-result.json").read_text())
    order = json.loads((output / "class_order.json").read_text())
    if (config.get("config_sha256") != record["config_sha256"]
            or config.get("code_sha256") != plan["training_source_sha256"]
            or config.get("namespace") != "PRODUCTION"
            or result.get("dataset") != record["dataset"]
            or result.get("seed") != record["seed"] or result.get("arm") != record["arm"]
            or result.get("schedule_sha256") != plan["schedule_sha256"]
            or result.get("classifier_updates") != sum(record["update_budgets_by_task"])
            or order.get("sha256") != record["class_order_sha256"]
            or order.get("class_order") != record["class_order"]):
        raise ValueError("completed exports differ from frozen run identity/budgets")
    return validation


def _publish_existing_completion_registry(storage, registry, run_id, identity):
    """Never upload fresh-staging rebound paths over a verified old marker."""
    documents = _canonical_completion_documents(storage, registry, run_id, identity)
    if documents is not None:
        remote_state, remote_marker = documents
        local_artifacts = registry.status(run_id)["completion_artifacts"]
        remote_artifacts = remote_marker["artifacts"]
        signature = lambda rows: {name: (row["bytes"], row["sha256"]) for name, row in rows.items()}
        if signature(remote_artifacts) != signature(local_artifacts):
            raise ValueError("canonical completion artifact hashes differ; preserved")
        return
    workflow._publish_api_tree(storage, registry.root)


def _checkpoint_source_payload_evidence(inventory, name):
    remote = inventory["_remote_by_name"].get(name)
    return {
        "runtime_local": name in inventory["runtime_local_files"],
        "api_staging_before_reconciliation": name in inventory["api_staging_files_before_reconciliation"],
        "canonical_drive": remote is not None,
        "drive_file_id": None if remote is None else remote["file_id"],
        "drive_size_bytes": None if remote is None else remote["size_bytes"],
        "drive_md5_checksum": None if remote is None else remote["md5_checksum"],
    }


def _read_remote_checkpoint_manifest(storage, logical_dir, metadata, identity, temp_dir):
    name = metadata["name"]
    size = metadata.get("size_bytes", metadata.get("size"))
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise ValueError(f"canonical checkpoint manifest size is unavailable: {name}") from None
    if not 0 < size <= 1024 * 1024:
        raise ValueError(f"canonical checkpoint manifest size is invalid: {name} ({size} bytes)")
    path = Path(temp_dir) / name
    storage.download_file(Path(logical_dir) / name, path, expected_bytes=size)
    value = _manifest(path, identity)
    return value, path.read_bytes()


def _quarantine_staged_checkpoint(path, storage):
    """Preserve a bad staging copy before using verified evidence from another source."""
    path = Path(path)
    quarantine = storage.local_staging_root / ".checkpoint_reconciliation_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"{time.time_ns()}-{path.name}"
    os.replace(path, target)
    return target


def _reconcile_checkpoint_tree(storage, staged_dir, local_dir, identity, kind):
    """Merge local, staging and canonical Drive generations before count validation."""
    inventory = _checkpoint_source_inventory(storage, staged_dir, local_dir)
    logical_dir = Path(inventory["canonical_drive_path"])
    remote_by_name = inventory["_remote_by_name"]
    parent_id = None
    if remote_by_name:
        parent_id = storage.folder_id(logical_dir, create=False)
        for name, remote in remote_by_name.items():
            if remote["parent_ids"] != [str(parent_id)]:
                raise ValueError(f"canonical checkpoint is outside its exact folder parent: {name}")

    expected_kind = "task-boundary" if kind == "task_boundaries" else None
    manifests = {}
    manifest_sources = {}
    temp_root = storage.local_staging_root / ".checkpoint_manifest_readback"
    temp_root.mkdir(parents=True, exist_ok=True)
    invalid_manifests = []
    for source_name, directory in (("runtime_local", Path(local_dir)), ("api_staging", Path(staged_dir))):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("generation-*.manifest.json")):
            if path.is_symlink():
                raise ValueError(f"symlinked checkpoint manifest; preserved: {path}")
            try:
                value = _manifest(path, identity)
            except (OSError, ValueError, UnicodeError) as exc:
                match = re.fullmatch(r"generation-(\d+)\.manifest\.json", path.name)
                if match is None:
                    raise ValueError(f"unparseable checkpoint manifest name; preserved: {path}") from exc
                invalid_manifests.append({
                    "generation": int(match.group(1)), "source": source_name,
                    "name": path.name, "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": workflow._hash_file_stream(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if expected_kind and value["kind"] != expected_kind:
                raise ValueError(f"unexpected {kind} manifest kind in {source_name}: {path.name}")
            if kind == "recovery" and value["kind"] not in {"recovery", "complete"}:
                raise ValueError(f"unexpected recovery manifest kind in {source_name}: {path.name}")
            generation = value["generation"]
            if generation in manifests and _signature(manifests[generation]) != _signature(value):
                raise ValueError(f"conflicting checkpoint manifests for generation {generation}; preserved")
            manifests[generation] = value
            manifest_sources.setdefault(generation, {})[source_name] = path.read_bytes()

    remote_manifest_rows = sorted(
        ({**row, "name": name} for name, row in remote_by_name.items() if name.endswith(".manifest.json")),
        key=lambda row: row["name"],
    )
    with tempfile.TemporaryDirectory(prefix="completion-checkpoint-manifests-", dir=temp_root) as temp_dir:
        for metadata in remote_manifest_rows:
            value, payload = _read_remote_checkpoint_manifest(storage, logical_dir, metadata, identity, temp_dir)
            if expected_kind and value["kind"] != expected_kind:
                raise ValueError(f"unexpected canonical {kind} manifest kind: {metadata['name']}")
            if kind == "recovery" and value["kind"] not in {"recovery", "complete"}:
                raise ValueError(f"unexpected canonical recovery manifest kind: {metadata['name']}")
            generation = value["generation"]
            if generation in manifests and _signature(manifests[generation]) != _signature(value):
                raise ValueError(f"canonical checkpoint manifest conflicts for generation {generation}; preserved")
            manifests[generation] = value
            manifest_sources.setdefault(generation, {})["canonical_drive"] = payload

    if not manifests:
        raise ValueError(f"no identity-valid {kind} checkpoint manifests found across local, staging, or Drive; "
                         + json.dumps({"invalid_manifests": invalid_manifests}, sort_keys=True))

    unpaired_invalid = [row for row in invalid_manifests if row["generation"] not in manifests]
    if unpaired_invalid:
        raise ValueError(f"invalid checkpoint manifests have no verified alternate source; preserved: "
                         + json.dumps({"invalid_manifests": unpaired_invalid,
                                       "verified_generations": sorted(manifests)}, sort_keys=True))
    inventory["invalid_manifests_recovered_from_verified_alternate"] = invalid_manifests

    staged_dir = Path(staged_dir)
    local_dir = Path(local_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    generation_report = []
    referenced_names = set()
    for generation, value in sorted(manifests.items()):
        expected_payload = value["payload_file"]
        referenced_names.update((expected_payload, f"generation-{generation:06d}.manifest.json"))
        source_paths = {
            "runtime_local": local_dir / expected_payload,
            "api_staging": staged_dir / expected_payload,
        }
        valid_sources = {}
        payload_integrity = {}
        for source_name, candidate in source_paths.items():
            if not candidate.exists():
                payload_integrity[source_name] = "missing"
                continue
            if candidate.is_symlink() or not candidate.is_file():
                payload_integrity[source_name] = "invalid_file_type"
                continue
            if candidate.stat().st_size != value["payload_bytes"]:
                payload_integrity[source_name] = "size_mismatch"
                continue
            digest = workflow._hash_file_stream(candidate)
            if digest != value["payload_sha256"]:
                payload_integrity[source_name] = "sha256_mismatch"
                continue
            payload_integrity[source_name] = "PASS"
            valid_sources[source_name] = candidate

        remote_payload = remote_by_name.get(expected_payload)
        remote_payload_integrity = "missing"
        if remote_payload is not None:
            try:
                remote_size = int(remote_payload["size_bytes"])
            except (TypeError, ValueError):
                remote_size = -1
            if remote_size != value["payload_bytes"]:
                remote_payload_integrity = "size_mismatch"
                raise ValueError(
                    f"canonical checkpoint payload conflicts with generation {generation}; "
                    f"file_id={remote_payload['file_id']} observed_size={remote_payload['size_bytes']} "
                    f"expected_size={value['payload_bytes']}; preserved"
                )

        if not valid_sources and remote_payload is not None:
            target = staged_dir / expected_payload
            if target.exists() or target.is_symlink():
                inventory.setdefault("staging_quarantined_paths", []).append(
                    str(_quarantine_staged_checkpoint(target, storage))
                )
            storage.download_file(
                logical_dir / expected_payload, target,
                expected_sha256=value["payload_sha256"], expected_bytes=value["payload_bytes"],
            )
            valid_sources["canonical_drive"] = target
            payload_integrity["canonical_drive"] = "PASS_SHA256_READBACK"
            remote_payload_integrity = "PASS_SHA256_READBACK"
        if not valid_sources and kind == "recovery" and value["kind"] == "recovery":
            payload_present = any((directory / expected_payload).exists()
                                  for directory in (local_dir, staged_dir)) or remote_payload is not None
            if not payload_present:
                manifest_name = f"generation-{generation:06d}.manifest.json"
                staged_manifest = staged_dir / manifest_name
                expected_manifest_bytes = manifest_sources[generation].get("api_staging")
                if expected_manifest_bytes is None:
                    expected_manifest_bytes = manifest_sources[generation].get("runtime_local")
                if expected_manifest_bytes is None:
                    expected_manifest_bytes = manifest_sources[generation]["canonical_drive"]
                if not staged_manifest.exists():
                    with staged_manifest.open("xb") as target:
                        target.write(expected_manifest_bytes)
                        target.flush()
                        os.fsync(target.fileno())
                generation_report.append({
                    "generation": generation, "kind": value["kind"],
                    "identity_sha256": value["identity_sha256"], "payload_file": expected_payload,
                    "payload_exists": {"runtime_local": False,
                        "api_staging_before_reconciliation": expected_payload in inventory["api_staging_files_before_reconciliation"],
                        "api_staging_after_reconciliation": False, "canonical_drive": False},
                    "payload_bytes": value["payload_bytes"], "payload_sha256": value["payload_sha256"],
                    "payload_integrity": {**payload_integrity, "canonical_drive": "missing"},
                    "source_provenance": {"runtime_local": False, "api_staging_before_reconciliation": False,
                        "canonical_drive": "canonical_drive" in manifest_sources[generation],
                        "manifest_sources": sorted(manifest_sources[generation])},
                    "manifest_bytes": staged_manifest.stat().st_size, "manifest_only_orphan": True,
                })
                continue
        if not valid_sources:
            raise ValueError(
                f"checkpoint generation {generation} has a manifest but no valid payload; "
                f"expected_bytes={value['payload_bytes']} expected_sha256={value['payload_sha256']} "
                f"payload_integrity={json.dumps(payload_integrity, sort_keys=True)}"
            )
        staged_payload = staged_dir / expected_payload
        if staged_payload not in valid_sources.values():
            if staged_payload.exists() or staged_payload.is_symlink():
                inventory.setdefault("staging_quarantined_paths", []).append(
                    str(_quarantine_staged_checkpoint(staged_payload, storage))
                )
            source_payload = valid_sources.get("runtime_local") or valid_sources.get("canonical_drive")
            if source_payload is None:
                source_payload = valid_sources["api_staging"]
            if source_payload != staged_payload:
                if source_payload.parent == staged_dir and source_payload == staged_payload:
                    pass
                elif source_payload.name == expected_payload and source_payload.exists():
                    _copy_checkpoint_file_exclusive(
                        source_payload, staged_payload, expected_bytes=value["payload_bytes"],
                        expected_sha256=value["payload_sha256"],
                    )
        if (not staged_payload.is_file() or staged_payload.is_symlink()
                or staged_payload.stat().st_size != value["payload_bytes"]
                or workflow._hash_file_stream(staged_payload) != value["payload_sha256"]):
            raise ValueError(f"reconciled checkpoint payload failed final integrity check: {staged_payload}")
        if remote_payload is not None:
            # The adapter reuses an identical local copy only after checking
            # Drive's exact file ID, size, and MD5, then computes its SHA-256.
            # If no Drive MD5 is available it downloads and hashes the bytes.
            storage.download_file(
                logical_dir / expected_payload, staged_payload,
                expected_sha256=value["payload_sha256"], expected_bytes=value["payload_bytes"],
            )
            remote_payload_integrity = "PASS_SHA256_AND_CANONICAL_READBACK"

        manifest_name = f"generation-{generation:06d}.manifest.json"
        staged_manifest = staged_dir / manifest_name
        expected_manifest_bytes = manifest_sources[generation].get("api_staging")
        if expected_manifest_bytes is None:
            expected_manifest_bytes = manifest_sources[generation].get("runtime_local")
        if expected_manifest_bytes is None:
            expected_manifest_bytes = manifest_sources[generation]["canonical_drive"]
        if staged_manifest.exists() or staged_manifest.is_symlink():
            if staged_manifest.is_symlink() or not staged_manifest.is_file():
                inventory.setdefault("staging_quarantined_paths", []).append(
                    str(_quarantine_staged_checkpoint(staged_manifest, storage))
                )
            else:
                try:
                    staged_value = _manifest(staged_manifest, identity)
                except ValueError:
                    inventory.setdefault("staging_quarantined_paths", []).append(
                        str(_quarantine_staged_checkpoint(staged_manifest, storage))
                    )
                else:
                    if _signature(staged_value) != _signature(value):
                        raise ValueError(f"staged manifest conflicts for generation {generation}; preserved")
        if not staged_manifest.exists():
            with staged_manifest.open("xb") as target:
                target.write(expected_manifest_bytes)
                target.flush()
                os.fsync(target.fileno())
        _manifest(staged_manifest, identity)

        generation_report.append({
            "generation": generation, "kind": value["kind"], "identity_sha256": value["identity_sha256"],
            "manifest_exists": {
                "runtime_local": f"generation-{generation:06d}.manifest.json" in inventory["runtime_local_files"],
                "api_staging_before_reconciliation": f"generation-{generation:06d}.manifest.json" in inventory["api_staging_files_before_reconciliation"],
                "canonical_drive": f"generation-{generation:06d}.manifest.json" in remote_by_name,
            },
            "manifest_source_sha256": {source: hashlib.sha256(data).hexdigest()
                                        for source, data in manifest_sources[generation].items()},
            "payload_file": expected_payload, "payload_exists": {
                "runtime_local": (local_dir / expected_payload).is_file(),
                "api_staging_before_reconciliation": expected_payload in inventory["api_staging_files_before_reconciliation"],
                "api_staging_after_reconciliation": staged_payload.is_file(),
                "canonical_drive": remote_payload is not None,
            },
            "payload_bytes": value["payload_bytes"], "payload_sha256": value["payload_sha256"],
            "payload_observed_sizes_bytes": {
                "runtime_local": (local_dir / expected_payload).stat().st_size
                    if (local_dir / expected_payload).is_file() else None,
                "api_staging_before_reconciliation": (staged_dir / expected_payload).stat().st_size
                    if (staged_dir / expected_payload).is_file() else None,
                "canonical_drive": None if remote_payload is None else remote_payload["size_bytes"],
            },
            "payload_integrity": {**payload_integrity, "canonical_drive": remote_payload_integrity},
            "source_provenance": {
                "runtime_local": payload_integrity.get("runtime_local") == "PASS",
                "api_staging_before_reconciliation": payload_integrity.get("api_staging") == "PASS",
                "canonical_drive": remote_payload is not None or "canonical_drive" in manifest_sources[generation],
                "manifest_sources": sorted(manifest_sources[generation]),
                "drive_payload_file_id": None if remote_payload is None else remote_payload["file_id"],
                "drive_manifest_file_id": None if remote_by_name.get(manifest_name) is None
                    else remote_by_name[manifest_name]["file_id"],
                "drive_file_ids": [file_id for file_id in (
                    None if remote_payload is None else remote_payload["file_id"],
                    None if remote_by_name.get(manifest_name) is None
                        else remote_by_name[manifest_name]["file_id"],
                ) if file_id],
            },
            "manifest_bytes": staged_manifest.stat().st_size,
        })

    # Orphan payloads cannot establish task identity without a valid manifest.
    for directory, label in ((local_dir, "runtime local"), (staged_dir, "API staging")):
        for path in _generation_files(directory):
            if path.endswith(".pt") and path not in referenced_names:
                raise ValueError(f"unreferenced {label} checkpoint payload; preserved: {directory / path}")
    for name in remote_by_name:
        if name.endswith(".pt") and name not in referenced_names:
            raise ValueError(f"unreferenced canonical checkpoint payload; preserved: {logical_dir / name}")

    inventory["verified_generations"] = generation_report
    inventory.pop("_remote_by_name", None)
    return inventory


def _prepare_completion_checkpoints(storage, staged_root, local_root, identity, run_record):
    """Reconcile all existing checkpoint sources, then enforce the frozen retention contract."""
    source_inventories = {}
    records = []
    # Reconcile every namespace first. A partial staging tree is not evidence
    # that a canonical/local generation is absent, and no count is enforced
    # until all three source locations have been compared.
    for kind, _count in (("recovery", 2), ("task_boundaries", 11)):
        staged, local = staged_root / kind, local_root / kind
        source_inventories[kind] = _reconcile_checkpoint_tree(storage, staged, local, identity, kind)

    # Resolve task cursors (and a possible strictly validated task-11 recovery)
    # before the generic recovery-retention check can mask the exact task gap.
    for kind, count in (("task_boundaries", 11), ("recovery", 2)):
        staged, local = staged_root / kind, local_root / kind
        if kind == "task_boundaries":
            task_rows = []
            for item in source_inventories[kind]["verified_generations"]:
                manifest_path = staged / f"generation-{item['generation']:06d}.manifest.json"
                manifest = _manifest(manifest_path, identity)
                cursor = _boundary_cursor_evidence(
                    staged, manifest, identity,
                    recovery_directory=staged_root / "recovery", run_record=run_record,
                )
                task_rows.append({**item, **cursor})
            by_task = {}
            duplicates = {}
            for item in task_rows:
                by_task.setdefault(item["task_id"], []).append(item)
            duplicates = {key: rows for key, rows in by_task.items() if len(rows) != 1}
            missing = sorted(set(range(1, 12)) - set(by_task))
            if duplicates:
                raise ValueError("duplicate task-boundary cursor evidence: " + json.dumps(
                    {"duplicates": {key: [row["generation"] for row in rows] for key, rows in duplicates.items()},
                     "inventory": task_rows}, sort_keys=True))
            if missing == [11]:
                recovery = staged_root / "recovery"
                terminal = []
                for candidate in sorted(recovery.glob("generation-*.manifest.json")):
                    value = _manifest(candidate, identity)
                    if value["kind"] != "complete":
                        continue
                    _validate_terminal_completion_checkpoint(recovery, value, identity, run_record)
                    terminal.append(value)
                if not terminal:
                    missing_boundary = {
                        "task_id": 11, "generation": None,
                        "identity_sha256": hashlib.sha256(workflow._canonical_bytes(identity)).hexdigest(),
                        "payload_exists": {"runtime_local": None, "api_staging": None, "canonical_drive": None},
                        "payload_size_bytes": None, "payload_integrity": "NO_AUTHENTICATED_TASK_BOUNDARY",
                        "note": "No identity- and cursor-validated task-11 boundary or kind=complete recovery generation exists; see source inventory for rejected candidates.",
                    }
                    raise ValueError(
                        "missing final task-boundary evidence and no identity/cursor-validated complete recovery state; "
                        + json.dumps({"missing_task_ids": [11], "missing_task_boundaries": [missing_boundary],
                                      "task_boundary_inventory": task_rows,
                                      "recovery_generation_inventory": source_inventories["recovery"]["verified_generations"],
                                      "recovery_source_inventory": source_inventories["recovery"],
                                      "publication_journals": source_inventories["recovery"].get("publication_journals")},
                                     sort_keys=True)
                    )
                source_manifest = max(terminal, key=lambda row: row["generation"])
                generation, value = _write_recovered_final_boundary(
                    staged, recovery, source_manifest, identity, run_record,
                )
                recovered_row = next(
                    row for row in source_inventories["recovery"]["verified_generations"]
                    if row["generation"] == source_manifest["generation"]
                )
                boundary_item = {
                    "generation": generation, "kind": "task-boundary",
                    "identity_sha256": value["identity_sha256"],
                    "payload_file": value["payload_file"], "payload_exists": {
                        "runtime_local": False, "api_staging_before_reconciliation": False,
                        "api_staging_after_reconciliation": True, "canonical_drive": False,
                    }, "payload_bytes": value["payload_bytes"], "payload_sha256": value["payload_sha256"],
                    "payload_integrity": {"api_staging": "PASS", "recovery_source": "PASS"},
                    "source_provenance": {"runtime_local": False, "api_staging_before_reconciliation": False,
                        "canonical_drive": False, "recovered_from_terminal_generation": source_manifest["generation"],
                        "recovery_source_provenance": recovered_row["source_provenance"]},
                    "manifest_bytes": (staged / f"generation-{generation:06d}.manifest.json").stat().st_size,
                    "recovery_provenance": value["recovery_provenance"],
                    **_boundary_cursor_evidence(
                        staged, value, identity,
                        recovery_directory=staged_root / "recovery", run_record=run_record,
                    ),
                }
                task_rows.append(boundary_item)
                source_inventories[kind]["verified_generations"].append(boundary_item)
            if missing and missing != [11]:
                missing_evidence = [{
                    "task_id": task_id, "generation": None,
                    "identity_sha256": hashlib.sha256(workflow._canonical_bytes(identity)).hexdigest(),
                    "payload_exists": {"runtime_local": None, "api_staging": None, "canonical_drive": None},
                    "payload_size_bytes": None, "payload_integrity": "NO_AUTHENTICATED_TASK_BOUNDARY",
                    "note": "No identity-valid manifest/payload pair proved this task cursor; inspect listed generations, do not infer from generation number.",
                } for task_id in missing]
                raise ValueError(
                    "required task-boundary evidence is missing; refusing to infer an earlier task from final state: "
                    + json.dumps({"missing_task_ids": missing, "missing_task_boundaries": missing_evidence,
                                  "task_boundary_inventory": task_rows,
                                  "recovery_generation_inventory": source_inventories["recovery"]["verified_generations"],
                                  "source_inventory": source_inventories[kind],
                                  "publication_journals": source_inventories[kind].get("publication_journals")},
                                 sort_keys=True)
                )
            if len(task_rows) != 11 or sorted(item["task_id"] for item in task_rows) != list(range(1, 12)):
                raise ValueError("task-boundary generation count/cursor inventory does not match tasks 1..11: "
                                 + json.dumps({"task_boundary_inventory": task_rows}, sort_keys=True))
            source_inventories[kind]["verified_generations"] = task_rows
        measured = workflow._checkpoint_tree_measurement(
            staged_root,
            identity,
            kind=kind,
            expected_count=count,
            allow_recovery_overflow=(kind == "recovery"),
        )
        for generation in measured["generations"]:
            path = staged / f"generation-{generation:06d}.manifest.json"
            value = _manifest(path, identity)
            payload = staged / value["payload_file"]
            if workflow._hash_file_stream(payload) != value["payload_sha256"]:
                raise ValueError(f"completion checkpoint SHA-256 mismatch: {payload}")
            detail = next((row for row in source_inventories[kind]["verified_generations"]
                           if row["generation"] == generation), {})
            records.append((staged, generation, value, detail))
    return records, source_inventories


def complete_exported_run(plan, record, *, output_root, checkpoint_root,
                          local_checkpoint_root, storage_adapter, api_factory,
                          acknowledge_no_other_worker=False, print_fn=print):
    """Recover publication/registration only; read verified cursor metadata without training."""
    from revision_r1.registry import RunRegistry
    from revision_r1.drive_storage import DriveApiLease, _classify_drive_api_lease_record
    workflow._require_canonical_storage_api_factory(api_factory)
    if storage_adapter.service is not api_factory():
        raise ValueError("completion storage adapter is not bound to canonical OAuth")
    if acknowledge_no_other_worker is not True:
        raise ValueError("check Manage sessions and acknowledge no other worker")
    workflow.validate_frozen_plan(plan, expected_source_sha256=CANONICAL_SOURCE)
    if record not in plan["runs"] or record == plan["runs"][0]:
        raise ValueError("completion retry requires an exact non-Run-1 frozen record")
    root = Path(output_root).resolve()
    output = workflow.run_output_root(root, record)
    checkpoints = workflow.run_checkpoint_root(checkpoint_root, record)
    local = workflow.run_checkpoint_root(local_checkpoint_root, record)
    identity = workflow._expected_plan_runner_identity(plan, record)
    run_id = identity["run_id"]
    lock = root / ".matrix_execution.lease.json"
    logical_lock = storage_adapter.logical_path_for_local(lock)
    storage_adapter.ensure_lease_record(logical_lock)
    current, _, _ = DriveApiLease(storage_adapter, logical_lock)._read()
    if (_classify_drive_api_lease_record(current, now=time.time())["status"] == "expired"
            and current.get("host") == socket.gethostname() and current.get("pid") == os.getpid()):
        workflow.recover_failed_matrix_lease(plan, record, storage_adapter=storage_adapter,
            output_root=root, api_factory=api_factory, expected_matrix_record=current,
            acknowledge_no_other_worker=True)
    with workflow._MatrixLeaseHeartbeat(lock, storage_adapter=storage_adapter) as lease:
        workflow._hydrate_api_matrix_registry(storage_adapter, root, plan)
        # The canonical registry must exist; never create an empty replacement.
        evidence = workflow.matrix_registry_evidence(storage_adapter, root, plan)
        if plan["runs"][0]["run_key"] not in evidence["completed_run_keys"]:
            raise ValueError("canonical registry lacks finalized Run 1; preserved")
        try:
            _hydrate_missing(storage_adapter, storage_adapter.logical_path_for_local(output), output)
        except FileNotFoundError:
            pass
        workflow._rebind_staged_completion_paths(output)
        _validate_export_evidence(plan, record, output)
        base = RunRegistry(output / "registry")
        canonical_completion = _canonical_completion_documents(storage_adapter, base, run_id, identity)
        if canonical_completion is not None and not base.is_completed(run_id):
            # Preserve stale local failure evidence before adopting the verified
            # canonical control pair. Only local paths are rebound, never Drive.
            old_path = base._state_path(run_id)
            if old_path.is_file():
                archive = root / "completion_recovery" / plan["plan_sha256"] / f"{record['run_key']}_prior_registry_{workflow._hash_file_stream(old_path)}.json"
                if not archive.exists():
                    workflow._atomic_json(archive, json.loads(old_path.read_text()))
                    workflow._api_publish_file(storage_adapter, archive)
            workflow._atomic_json(base._state_path(run_id), canonical_completion[0])
            workflow._atomic_json(base._run_dir(run_id) / "COMPLETE.json", canonical_completion[1])
            workflow._rebind_staged_completion_paths(output)
        state = base.status(run_id)
        if state.get("identity") != identity or not state.get("attempts"):
            raise ValueError("export retry lacks the existing identity-bound attempt; no replacement run created")
        run_lock = output / ".run.lease.json"
        logical_run_lock = storage_adapter.logical_path_for_local(run_lock)
        storage_adapter.ensure_lease_record(logical_run_lock)
        with workflow._MatrixLeaseHeartbeat(run_lock, storage_adapter=storage_adapter) as run_lease:
            print_fn("Training complete: 11/11 tasks")
            matrix = workflow._read_matrix_registry(root, plan)
            existing = matrix["completed_runs"].get(record["run_key"])
            added = False
            checkpoint_evidence = []
            if existing is None:
                checkpoint_records, checkpoint_source_inventory = _prepare_completion_checkpoints(
                    storage_adapter, checkpoints, local, identity, record,
                )
                print_fn("Publishing artifacts: existing exported files and checkpoints")
                for directory, generation, _value, source_detail in checkpoint_records:
                    if lease.lost.is_set() or run_lease.lost.is_set():
                        raise RuntimeError("completion matrix lease lost; all artifacts preserved")
                    _assert_remote_bytes_match(storage_adapter, directory / _value["payload_file"])
                    _assert_remote_bytes_match(storage_adapter, directory / f"generation-{generation:06d}.manifest.json")
                    publication = workflow._api_publish_checkpoint_generation(storage_adapter, directory, identity, generation)
                    checkpoint_evidence.append({"generation": generation, "kind": _value["kind"],
                        "task_id": source_detail.get("task_id"),
                        "identity_sha256": _value["identity_sha256"],
                        "logical_payload_path": str(storage_adapter.logical_path_for_local(directory / _value["payload_file"])),
                        "payload_bytes": _value["payload_bytes"], "payload_sha256": _value["payload_sha256"],
                        "payload_file_id": publication[0]["file_id"], "manifest_file_id": publication[1]["file_id"],
                        "source_provenance": source_detail.get("source_provenance"),
                        "recovery_provenance": source_detail.get("recovery_provenance"),
                        "payload_integrity": source_detail.get("payload_integrity")})
                for path in output.iterdir():
                    if path.is_file() and path.name != ".run.lease.json":
                        _assert_remote_bytes_match(storage_adapter, path)
                _publish_export_tree(storage_adapter, output)
                print_fn("Verifying artifacts: complete manifest, hashes and frozen identity")
                _validate_export_evidence(plan, record, output)
                registry = api_run_registry_class(storage_adapter)(output / "registry")
                if not registry.is_completed(run_id):
                    # A new publication-only attempt preserves the failed training
                    # attempt and uses the normal hash-verifying completion API.
                    current_attempt = registry.status(run_id)
                    if current_attempt.get("status") == "running":
                        if current_attempt["attempts"][-1].get("operation") != "completion_only":
                            raise ValueError("ambiguous running training attempt; completion retry refused")
                    else:
                        registry.begin_attempt(run_id, identity=identity, completion_only=True)
                    try:
                        artifacts = {name: output / name for name in (
                            "artifact_manifest.json", "status.json", "full-result.json", "task_matrix.json",
                            "predictions.npz", "resource_metrics.json", "config.json", "environment.json")}
                        if not registry.complete(run_id, artifacts):
                            raise ValueError("completion retry failed artifact revalidation")
                    except Exception as error:
                        if registry.status(run_id).get("status") == "running":
                            try:
                                registry.fail_attempt(run_id, f"completion-only publication: {type(error).__name__}")
                            except Exception as cleanup:
                                error.add_note(f"failure reporting also failed: {type(cleanup).__name__}")
                        raise
                else:
                    _publish_existing_completion_registry(storage_adapter, registry, run_id, identity)
                validated = workflow._validate_completed_run(plan, record, None, output, synthetic=False)
                measurement = workflow.measure_completed_production_run(
                    plan,
                    record,
                    None,
                    output,
                    checkpoints,
                    allow_recovery_overflow=True,
                    storage_adapter=storage_adapter,
                )
                if lease.lost.is_set() or run_lease.lost.is_set():
                    raise RuntimeError("completion matrix lease lost before registration")
                print_fn("Registering completion: validated artifacts only")
                matrix, added = workflow._record_completed_run(root, plan, record, validated, storage_measurement=measurement)
                workflow._api_publish_file(storage_adapter, workflow._matrix_registry_path(root, plan))
            else:
                print_fn("Publishing artifacts: already durably registered; reusing existing evidence")
                validated = workflow._validate_completed_run(plan, record, None, output, synthetic=False)
                print_fn("Verifying artifacts: canonical completion registry and hashes")
                matrix, _ = workflow._record_completed_run(root, plan, record, validated,
                    storage_measurement=existing.get("storage_measurement"))
                print_fn("Registering completion: already registered once; no duplicate")
            next_record = next((row for row in plan["runs"] if row["run_key"] not in matrix["completed_runs"]), None)
            result = {"status": "COMPLETED_VALIDATED", "run_key": record["run_key"],
                "completion_registry_added_now": added, "training_launched": False,
                "model_initialized": False, "files_deleted": False,
                "checkpoint_evidence": checkpoint_evidence,
                "checkpoint_source_inventory": checkpoint_source_inventory if existing is None else None,
                "checkpoint_validation_scope": "identity/size/local SHA-256 and API publication verification" if checkpoint_evidence
                    else "reused previously registered completion/measurement; checkpoints not downloaded again",
                "artifact_manifest_sha256": validated["artifact_manifest_sha256"],
                "completed_run_keys": [row["run_key"] for row in plan["runs"] if row["run_key"] in matrix["completed_runs"]],
                "next_pending_run": None if next_record is None else next_record["run_key"],
                "matrix_registry_evidence": workflow.matrix_registry_evidence(storage_adapter, root, plan)}
            report = root / "completion_recovery" / plan["plan_sha256"] / f"{record['run_key']}_{time.time_ns()}.json"
            workflow._atomic_json(report, result)
            workflow._api_publish_file(storage_adapter, report)
            print_fn(f"Run {record['run_index']} finalized")
            print_fn("Next pending run: " + str(result["next_pending_run"]))
            return result


def _manifest(path, identity):
    value = json.loads(path.read_text(encoding="utf-8"))
    generation = value.get("generation")
    if (type(generation) is not int or generation <= 0
            or path.name != f"generation-{generation:06d}.manifest.json"
            or value.get("schema_version") != 1
            or value.get("identity") != identity
            or value.get("identity_sha256") != hashlib.sha256(workflow._canonical_bytes(identity)).hexdigest()
            or value.get("payload_file") != f"generation-{generation:06d}.pt"
            or type(value.get("payload_bytes")) is not int or value["payload_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("payload_sha256", "")))):
        raise ValueError(f"incompatible recovery manifest: {path}")
    return value


def _signature(value):
    return workflow._canonical_bytes(value)


def prepare_pending_checkpoint_publication(
    plan, record, *, output_root, checkpoint_root, local_checkpoint_root,
    storage_adapter, api_factory, acknowledge_no_other_worker=False, progress_callback=None,
):
    """Prepare existing bytes and report sanitized publication progress."""
    from revision_r1.drive_storage import describe_upload_failure
    previous_callback = storage_adapter.progress_callback

    def emit(row):
        if progress_callback is not None:
            if str(row.get("status", "")).startswith("CHECKPOINT_"):
                row = {**row, "checkpoint_upload": {
                    **{name: row[name] for name in (
                        "generation", "filename", "uploaded_bytes", "total_bytes", "diagnostics",
                    ) if name in row},
                    "status": row["status"].removeprefix("CHECKPOINT_").lower(),
                }}
            try:
                progress_callback(row)
            except Exception:
                # A display failure must not interrupt publication or mask its cause.
                pass

    storage_adapter.progress_callback = lambda row: emit({
        **row, "stage": "CHECKPOINT_UPLOADING", "status": "CHECKPOINT_UPLOADING",
    })
    try:
        if acknowledge_no_other_worker is not True:
            raise ValueError("explicit single-worker acknowledgement is required")
        return _prepare_pending_checkpoint_publication(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            local_checkpoint_root=local_checkpoint_root, storage_adapter=storage_adapter,
            api_factory=api_factory, acknowledge_no_other_worker=acknowledge_no_other_worker,
            progress_callback=emit,
        )
    except Exception as exc:
        emit({"stage": "CHECKPOINT_FAILED", "status": "CHECKPOINT_FAILED",
              "diagnostics": describe_upload_failure(exc), "training_launched": False})
        raise
    finally:
        storage_adapter.progress_callback = previous_callback


def _prepare_pending_checkpoint_publication(
    plan, record, *, output_root, checkpoint_root, local_checkpoint_root,
    storage_adapter, api_factory, acknowledge_no_other_worker, progress_callback,
):
    """Publish the newest verified existing generation, preserving all fallbacks.

    No payload is deserialized. Errors preserve checkpoint files and propagate;
    an expired lease is recovered only through the workflow's exact-owner gate.
    """
    workflow._require_canonical_storage_api_factory(api_factory)
    if storage_adapter.service is not api_factory():
        raise ValueError("adapter must use the canonical bound API service")
    workflow.validate_frozen_plan(plan, expected_source_sha256=CANONICAL_SOURCE)
    if record not in plan["runs"] or not isinstance(record.get("dataset_identity"), dict) or not record["dataset_identity"]:
        raise ValueError("an exact frozen record with dataset identity is required")
    root = Path(output_root).expanduser().resolve()
    staged = workflow.run_checkpoint_root(checkpoint_root, record) / "recovery"
    local = workflow.run_checkpoint_root(local_checkpoint_root, record) / "recovery"
    logical = storage_adapter.logical_path_for_local(staged)
    identity = workflow._expected_plan_runner_identity(plan, record)

    def pending():
        workflow._hydrate_api_matrix_registry(storage_adapter, root, plan)
        if record["run_key"] in workflow._read_matrix_registry(root, plan)["completed_runs"]:
            raise ValueError("completed run is not eligible for checkpoint publication preparation")

    pending()
    lock = root / ".matrix_execution.lease.json"
    lock_logical = storage_adapter.logical_path_for_local(lock)
    storage_adapter.ensure_lease_record(lock_logical)
    from revision_r1.drive_storage import DriveApiLease, _classify_drive_api_lease_record
    current, _, _ = DriveApiLease(storage_adapter, lock_logical)._read()
    if (_classify_drive_api_lease_record(current, now=time.time())["status"] == "expired"
            and current.get("host") == socket.gethostname() and current.get("pid") == os.getpid()):
        workflow.recover_failed_matrix_lease(
            plan, record, storage_adapter=storage_adapter, output_root=root, api_factory=api_factory,
            expected_matrix_record=current, acknowledge_no_other_worker=acknowledge_no_other_worker,
        )
    with workflow._MatrixLeaseHeartbeat(lock, storage_adapter=storage_adapter) as lease:
        pending()
        candidates = []
        signatures = {}

        def remember(value):
            generation = value["generation"]
            signature = _signature(value)
            if generation in signatures and signatures[generation] != signature:
                raise ValueError(f"conflicting recovery generation {generation}; all files preserved")
            signatures[generation] = signature

        for directory in dict.fromkeys((local, staged)):
            for path in directory.glob("generation-*.manifest.json"):
                if path.is_symlink():
                    raise ValueError("symlinked recovery manifest")
                value = _manifest(path, identity)
                remember(value)
                payload = path.parent / value["payload_file"]
                if (payload.is_file() and not payload.is_symlink()
                        and payload.stat().st_size == value["payload_bytes"]
                        and workflow._hash_file_stream(payload) == value["payload_sha256"]):
                    candidates.append((value, path, payload))
        try:
            files = storage_adapter.list_files(logical)
        except FileNotFoundError:
            files = []
        names = {}
        for item in files:
            name = item["name"]
            if name in names:
                raise ValueError(f"ambiguous remote recovery name: {name}")
            names[name] = item
        # Keep downloaded evidence separate so existing staged files are never replaced.
        remote = []
        if any(re.fullmatch(r"generation-\d+\.manifest.json", name) for name in names):
            staged.mkdir(parents=True, exist_ok=True)
            evidence = Path(tempfile.mkdtemp(prefix="publication-evidence-", dir=staged))
            for name in sorted(names):
                if not re.fullmatch(r"generation-\d+\.manifest.json", name):
                    continue
                if not 0 < int(names[name].get("size", 0)) <= 1024 * 1024:
                    raise ValueError("remote recovery manifest exceeds small metadata limit")
                path = evidence / name
                storage_adapter.download_file(logical / name, path)
                value = _manifest(path, identity)
                remember(value)
                metadata = names.get(value["payload_file"])
                if metadata and int(metadata.get("size", -1)) == value["payload_bytes"]:
                    remote.append((value, path))
            best = max((item[0]["generation"] for item in candidates), default=0)
            for value, path in sorted(remote, key=lambda item: item[0]["generation"], reverse=True):
                if value["generation"] <= best:
                    break
                payload = staged / value["payload_file"]
                if payload.exists() or payload.is_symlink():
                    if (payload.is_symlink() or not payload.is_file()
                            or payload.stat().st_size != value["payload_bytes"]
                            or workflow._hash_file_stream(payload) != value["payload_sha256"]):
                        raise ValueError("incompatible existing staged payload; preserved")
                else:
                    storage_adapter.download_file(
                        logical / payload.name, payload, expected_sha256=value["payload_sha256"],
                        expected_bytes=value["payload_bytes"],
                    )
                canonical_manifest = staged / path.name
                if canonical_manifest.exists() or canonical_manifest.is_symlink():
                    if (canonical_manifest.is_symlink()
                            or _signature(_manifest(canonical_manifest, identity)) != _signature(value)):
                        raise ValueError("incompatible existing staged manifest; preserved")
                else:
                    # Exclusive creation preserves every existing staged manifest.
                    with canonical_manifest.open("xb") as target:
                        target.write(path.read_bytes())
                candidates.append((value, canonical_manifest, payload))
                break
        if not candidates:
            raise ValueError("no valid compatible checkpoint; initialization is forbidden")
        value, manifest, payload = max(candidates, key=lambda item: item[0]["generation"])
        manifest_hash = workflow._hash_file_stream(manifest)
        existing_payload = names.get(payload.name)
        if existing_payload is not None and (
            int(existing_payload.get("size", -1)) != value["payload_bytes"]
            or existing_payload.get("md5Checksum") != workflow._checkpoint_md5(payload)
        ):
            raise ValueError("conflicting or unverifiable remote payload; preserved without replacement")
        if lease.lost.is_set():
            raise RuntimeError("canonical matrix lease lost before publication")
        progress_callback({"stage": "CHECKPOINT_UPLOADING", "status": "CHECKPOINT_UPLOADING",
                           "generation": value["generation"], "payload_bytes": value["payload_bytes"]})
        payload_result = storage_adapter.publish_file(payload, logical / payload.name)
        if workflow._hash_file_stream(payload) != value["payload_sha256"] or payload_result["sha256"] != value["payload_sha256"]:
            raise ValueError("payload changed during publication; manifest preserved")
        if lease.lost.is_set():
            raise RuntimeError("canonical matrix lease lost before manifest publication")
        if workflow._hash_file_stream(manifest) != manifest_hash or _manifest(manifest, identity) != value:
            raise ValueError("manifest changed during publication; remote manifest preserved")
        manifest_result = storage_adapter.publish_file(manifest, logical / manifest.name)
        if manifest_result["sha256"] != manifest_hash:
            raise ValueError("manifest publication digest was not verified")
        result = {
            "status": "PREPARED_CHECKPOINT_BEFORE_USER_F", "generation": value["generation"],
            "kind": value.get("kind"), "local_path": str(payload), "drive_path": str(logical / payload.name),
            "manifest_path": str(manifest), "payload_sha256": value["payload_sha256"],
            "payload_bytes": value["payload_bytes"], "payload_file_id": payload_result["file_id"],
            "manifest_file_id": manifest_result["file_id"], "recovery_cursor": None,
            "recovery_cursor_status": "UNKNOWN_UNTIL_RUNNER_RESTORE", "no_training": True,
            "training_launched": False, "model_initialized": False, "dataset_initialized": False,
            "files_pruned": False,
        }
        if progress_callback is not None:
            progress_callback({**result, "stage": "CHECKPOINT_VERIFIED", "status": "CHECKPOINT_VERIFIED"})
            progress_callback(result)
        return result
