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

from scripts import production_workflow as workflow


CANONICAL_SOURCE = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"


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
