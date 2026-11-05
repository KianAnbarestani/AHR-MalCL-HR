import hashlib
import json
import os
from pathlib import Path
import socket
import time

import pytest
import torch

from scripts import production_recovery as recovery
from scripts import production_workflow as workflow
from test_drive_api_storage import _FakeService, _storage


@pytest.fixture
def setup(tmp_path, monkeypatch):
    service = _FakeService()
    stage = tmp_path / "stage"
    adapter = _storage(service, stage, monkeypatch)
    output = stage / "results"
    checkpoints = stage / "checkpoints"
    adapter.bind_root("/drive/MyDrive/ReVision/results", output)
    adapter.bind_root("/drive/MyDrive/ReVision/checkpoints", checkpoints)
    monkeypatch.setattr(workflow, "_BOUND_DRIVE_API_SERVICE", service)
    monkeypatch.setattr(workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": workflow.CANONICAL_STORAGE_ACCOUNT_EMAIL,
        "canonical_revision_folder_id": workflow.CANONICAL_REVISION_FOLDER_ID,
    })
    plan = json.loads((Path(__file__).parents[1] / "configs/production_run_plan_v1.json").read_text())
    for record in plan["runs"]:
        record["dataset_identity"] = {"train_md5": "a" * 32, "test_md5": "b" * 32}
    plan = workflow._refresh_plan_hash(plan)
    record = plan["runs"][1]
    args = dict(output_root=output, checkpoint_root=checkpoints,
                local_checkpoint_root=tmp_path / "local", storage_adapter=adapter,
                api_factory=lambda: service, acknowledge_no_other_worker=True)
    return plan, record, args, service


def checkpoint(plan, record, root, generation, *, tensor=1):
    directory = workflow.run_checkpoint_root(root, record) / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f"generation-{generation:06d}.pt"
    torch.save({"tensor": torch.tensor([tensor]), "cursor": {"task": 2}}, payload)
    identity = workflow._expected_plan_runner_identity(plan, record)
    value = dict(schema_version=1, generation=generation, identity=identity,
                 identity_sha256=hashlib.sha256(workflow._canonical_bytes(identity)).hexdigest(),
                 payload_file=payload.name, payload_bytes=payload.stat().st_size,
                 payload_sha256=workflow._hash_file_stream(payload), kind="periodic")
    manifest = directory / f"generation-{generation:06d}.manifest.json"
    manifest.write_text(json.dumps(value))
    return payload, manifest


def test_failed_payload_preserves_files_and_retry_is_idempotent(setup, monkeypatch):
    plan, record, args, service = setup
    payload, manifest = checkpoint(plan, record, args["local_checkpoint_root"], 3)
    adapter = args["storage_adapter"]
    original = adapter.publish_file
    failed = []

    def fail(path, logical, **kwargs):
        if Path(path).suffix == ".pt":
            failed.append(str(path))
            raise OSError("upload interrupted")
        return original(path, logical, **kwargs)

    monkeypatch.setattr(adapter, "publish_file", fail)
    with pytest.raises(OSError, match="upload interrupted"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert failed and payload.exists() and manifest.exists()
    assert not any(row["name"].endswith(".manifest.json") for row in service.records.values())
    monkeypatch.setattr(adapter, "publish_file", original)
    monkeypatch.setattr(torch, "load", lambda *_a, **_kw: pytest.fail("must not deserialize"))
    first = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    second = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert first["payload_file_id"] == second["payload_file_id"]
    assert first["manifest_file_id"] == second["manifest_file_id"]
    assert first["no_training"] and first["recovery_cursor"] is None
    assert sum(row["name"] == payload.name for row in service.records.values()) == 1


def test_newer_local_wins_over_old_drive(setup):
    plan, record, args, service = setup
    old_payload, old_manifest = checkpoint(plan, record, args["checkpoint_root"], 2)
    adapter = args["storage_adapter"]
    for path in (old_payload, old_manifest):
        adapter.publish_file(path, adapter.logical_path_for_local(path))
    checkpoint(plan, record, args["local_checkpoint_root"], 5)
    result = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert result["generation"] == 5
    assert old_payload.exists() and old_manifest.exists()
    assert sum(row["name"].endswith(".pt") for row in service.records.values()) == 2


def test_conflicting_remote_orphan_payload_is_never_replaced(setup, tmp_path):
    plan, record, args, service = setup
    payload, _ = checkpoint(plan, record, args["local_checkpoint_root"], 3, tensor=1)
    other, _ = checkpoint(plan, record, tmp_path / "other", 3, tensor=2)
    adapter = args["storage_adapter"]
    logical = adapter.logical_path_for_local(workflow.run_checkpoint_root(args["checkpoint_root"], record) / "recovery" / payload.name)
    published = adapter.publish_file(other, logical)
    before = bytes(service.records[published["file_id"]]["content"])
    with pytest.raises(ValueError, match="remote payload; preserved"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert bytes(service.records[published["file_id"]]["content"]) == before
    assert not any(row["name"].endswith("manifest.json") for row in service.records.values())


def test_newer_remote_staged_with_hash_without_overwriting_local(setup):
    plan, record, args, _ = setup
    old, _ = checkpoint(plan, record, args["local_checkpoint_root"], 2)
    payload, manifest = checkpoint(plan, record, args["checkpoint_root"], 4)
    adapter = args["storage_adapter"]
    for path in (payload, manifest):
        adapter.publish_file(path, adapter.logical_path_for_local(path))
    result = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert result["generation"] == 4 and old.exists()


def test_completed_run_one_rejected(setup):
    plan, _, args, _ = setup
    record = plan["runs"][0]
    root = Path(args["output_root"])
    registry = workflow._read_matrix_registry(root, plan)
    registry["completed_runs"][record["run_key"]] = {
        **{key: record[key] for key in ("run_key", "run_index", "config_sha256", "class_order_sha256", "dataset_identity")},
        "training_source_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "run_id": workflow._run_id_for(record, synthetic=False), "artifact_manifest_sha256": "a" * 64,
    }
    path = workflow._matrix_registry_path(root, plan)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    adapter = args["storage_adapter"]
    adapter.publish_file(path, adapter.logical_path_for_local(path))
    with pytest.raises(ValueError, match="completed run"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)


def test_same_generation_conflict_preserved(setup):
    plan, record, args, _ = setup
    checkpoint(plan, record, args["local_checkpoint_root"], 2, tensor=1)
    checkpoint(plan, record, args["checkpoint_root"], 2, tensor=9)
    with pytest.raises(ValueError, match="conflicting recovery generation"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)


def test_manifest_upload_failure_keeps_payload_for_exact_id_retry(setup, monkeypatch):
    plan, record, args, service = setup
    payload, manifest = checkpoint(plan, record, args["local_checkpoint_root"], 7)
    adapter = args["storage_adapter"]
    original = adapter.publish_file

    def fail(path, logical, **kwargs):
        if Path(path).name.endswith(".manifest.json"):
            raise OSError("manifest interrupted")
        return original(path, logical, **kwargs)

    monkeypatch.setattr(adapter, "publish_file", fail)
    with pytest.raises(OSError, match="manifest interrupted"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    file_id = next(key for key, row in service.records.items() if row["name"] == payload.name)
    assert payload.exists() and manifest.exists()
    monkeypatch.setattr(adapter, "publish_file", original)
    result = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert result["payload_file_id"] == file_id


def test_wrong_identity_digest_fails_without_publication(setup):
    plan, record, args, service = setup
    _, manifest = checkpoint(plan, record, args["local_checkpoint_root"], 3)
    value = json.loads(manifest.read_text())
    value["identity_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="incompatible recovery manifest"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert not any(row["name"].endswith(".pt") for row in service.records.values())


def test_remote_payload_downloaded_once_to_canonical_stage(setup, monkeypatch, tmp_path):
    plan, record, args, _ = setup
    adapter = args["storage_adapter"]
    payload, manifest = checkpoint(plan, record, tmp_path / "remote-source", 8)
    canonical = workflow.run_checkpoint_root(args["checkpoint_root"], record) / "recovery"
    logical = adapter.logical_path_for_local(canonical)
    for path in (payload, manifest):
        adapter.publish_file(path, logical / path.name)
    original = adapter.download_to_path
    downloads = []

    def download(request, path):
        if ".pt." in Path(path).name:
            downloads.append(str(path))
        return original(request, path)

    monkeypatch.setattr(adapter, "download_to_path", download)
    monkeypatch.setattr(torch, "load", lambda *_a, **_kw: pytest.fail("no tensor load"))
    first = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    second = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert first["local_path"] == second["local_path"] == str(canonical / payload.name)
    assert len(downloads) == 1
    assert (canonical / manifest.name).read_bytes() == manifest.read_bytes()
    assert not list(canonical.glob("publication-evidence-*/*.pt"))


def test_full_manifest_conflict_preserves_remote_accounting(setup):
    plan, record, args, service = setup
    payload, manifest = checkpoint(plan, record, args["local_checkpoint_root"], 6)
    adapter = args["storage_adapter"]
    canonical = workflow.run_checkpoint_root(args["checkpoint_root"], record) / "recovery"
    logical = adapter.logical_path_for_local(canonical)
    adapter.publish_file(payload, logical / payload.name)
    value = json.loads(manifest.read_text())
    value["created_unix"] = 123
    manifest.write_text(json.dumps(value))
    published = adapter.publish_file(manifest, logical / manifest.name)
    before = service.records[published["file_id"]].copy()
    value["created_unix"] = 456
    value["accounting"] = {"bytes": 1000}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="conflicting recovery generation"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert service.records[published["file_id"]] == before


def test_acknowledgement_required_before_any_storage_write(setup):
    plan, record, args, service = setup
    before = dict(service.records)
    args["acknowledge_no_other_worker"] = False
    with pytest.raises(ValueError, match="single-worker acknowledgement"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert service.records == before


def test_upload_progress_restored_and_failure_sanitized(setup, monkeypatch):
    plan, record, args, _ = setup
    checkpoint(plan, record, args["local_checkpoint_root"], 1)
    adapter = args["storage_adapter"]
    previous = lambda row: None
    adapter.progress_callback = previous
    events = []
    original = adapter.publish_file

    def upload(path, logical, **kwargs):
        if Path(path).suffix == ".pt":
            adapter.progress_callback({"uploaded_bytes": 12, "total_bytes": 24})
        return original(path, logical, **kwargs)

    monkeypatch.setattr(adapter, "publish_file", upload)
    recovery.prepare_pending_checkpoint_publication(plan, record, **args, progress_callback=events.append)
    assert adapter.progress_callback is previous
    assert any(row.get("uploaded_bytes") == 12 for row in events)
    assert any(row.get("stage") == "CHECKPOINT_VERIFIED" for row in events)

    def fail(*_args, **_kwargs):
        raise OSError("secret https://upload.invalid/private-token")

    monkeypatch.setattr(adapter, "publish_file", fail)
    with pytest.raises(OSError):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args, progress_callback=events.append)
    assert adapter.progress_callback is previous
    assert events[-1]["stage"] == "CHECKPOINT_FAILED"
    assert "private-token" not in json.dumps(events)


@pytest.mark.parametrize("same_runtime", [False, True])
def test_expired_lease_uses_idle_kernel_recovery_only_for_same_runtime(setup, monkeypatch, same_runtime):
    plan, record, args, _ = setup
    checkpoint(plan, record, args["local_checkpoint_root"], 3)
    adapter = args["storage_adapter"]
    lock = Path(args["output_root"]) / ".matrix_execution.lease.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    document = dict(schema_version=1, active=True, owner_token="old-owner",
                    host=socket.gethostname() if same_runtime else "different-host",
                    pid=os.getpid(), created_at=now - 1000, heartbeat_at=now - 900,
                    expires_at=now - 600)
    lock.write_text(json.dumps(document))
    adapter.publish_file(lock, adapter.logical_path_for_local(lock))
    calls = []

    def recover(*_args, **kwargs):
        calls.append(kwargs["expected_matrix_record"])
        # Simulate the workflow refusing missing canonical FAILED evidence.
        raise ValueError("idle-kernel proof required")

    monkeypatch.setattr(workflow, "recover_failed_matrix_lease", recover)
    if same_runtime:
        with pytest.raises(ValueError, match="idle-kernel proof required"):
            recovery.prepare_pending_checkpoint_publication(plan, record, **args)
        assert calls == [document]
    else:
        result = recovery.prepare_pending_checkpoint_publication(plan, record, **args)
        assert result["generation"] == 3 and calls == []


def test_incompatible_existing_canonical_payload_preserved(setup, tmp_path):
    plan, record, args, _ = setup
    adapter = args["storage_adapter"]
    payload, manifest = checkpoint(plan, record, tmp_path / "remote-source", 8)
    canonical = workflow.run_checkpoint_root(args["checkpoint_root"], record) / "recovery"
    logical = adapter.logical_path_for_local(canonical)
    for path in (payload, manifest):
        adapter.publish_file(path, logical / path.name)
    canonical.mkdir(parents=True, exist_ok=True)
    existing = canonical / payload.name
    existing.write_bytes(b"incompatible existing bytes")
    with pytest.raises(ValueError, match="incompatible existing staged payload"):
        recovery.prepare_pending_checkpoint_publication(plan, record, **args)
    assert existing.read_bytes() == b"incompatible existing bytes"
    assert not (canonical / manifest.name).exists()
