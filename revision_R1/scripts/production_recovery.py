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


def _prepare_completion_checkpoints(storage, staged_root, local_root, identity):
    """Hash checkpoint bytes without loading tensor state, and never prune."""
    from revision_r1.storage import require_path_capacity, DEFAULT_STORAGE_RESERVE_BYTES
    records = []
    for kind, count in (("recovery", 2), ("task_boundaries", 11)):
        staged, local = staged_root / kind, local_root / kind
        staged.mkdir(parents=True, exist_ok=True)
        for path in local.glob("generation-*.manifest.json"):
            value = _manifest(path, identity)
            source = path.parent / value["payload_file"]
            if (not source.is_file() or source.is_symlink()
                    or source.stat().st_size != value["payload_bytes"]
                    or workflow._hash_file_stream(source) != value["payload_sha256"]):
                raise ValueError(f"local completion checkpoint fails integrity: {path}")
            target = staged / source.name
            if target.exists() and (target.is_symlink() or workflow._hash_file_stream(target) != value["payload_sha256"]):
                raise ValueError(f"conflicting staged checkpoint; preserved: {target}")
            if not target.exists():
                require_path_capacity(staged, required_bytes=value["payload_bytes"], reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES)
                shutil.copyfile(source, target)
            target_manifest = staged / path.name
            if target_manifest.exists() and _signature(_manifest(target_manifest, identity)) != _signature(value):
                raise ValueError(f"conflicting staged checkpoint manifest; preserved: {target_manifest}")
            if not target_manifest.exists():
                shutil.copyfile(path, target_manifest)
        try:
            _hydrate_missing(storage, storage.logical_path_for_local(staged), staged)
        except FileNotFoundError:
            pass
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
            records.append((staged, generation, value))
    return records


def complete_exported_run(plan, record, *, output_root, checkpoint_root,
                          local_checkpoint_root, storage_adapter, api_factory,
                          acknowledge_no_other_worker=False, print_fn=print):
    """Recover publication/registration only. Never call a runner or torch.load."""
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
                checkpoint_records = _prepare_completion_checkpoints(storage_adapter, checkpoints, local, identity)
                print_fn("Publishing artifacts: existing exported files and checkpoints")
                for directory, generation, _value in checkpoint_records:
                    if lease.lost.is_set() or run_lease.lost.is_set():
                        raise RuntimeError("completion matrix lease lost; all artifacts preserved")
                    _assert_remote_bytes_match(storage_adapter, directory / _value["payload_file"])
                    _assert_remote_bytes_match(storage_adapter, directory / f"generation-{generation:06d}.manifest.json")
                    publication = workflow._api_publish_checkpoint_generation(storage_adapter, directory, identity, generation)
                    checkpoint_evidence.append({"generation": generation, "kind": _value["kind"],
                        "logical_payload_path": str(storage_adapter.logical_path_for_local(directory / _value["payload_file"])),
                        "payload_bytes": _value["payload_bytes"], "payload_sha256": _value["payload_sha256"],
                        "payload_file_id": publication[0]["file_id"], "manifest_file_id": publication[1]["file_id"]})
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
