"""Real exported files, registry/checkpoint serialization; external Drive only is fake."""
import json
import hashlib
from pathlib import Path
import shutil

import pytest
import torch

from scripts import production_recovery as recovery
from scripts import production_workflow as workflow
from revision_r1.registry import RunRegistry
from revision_r1.state import RunState, PhaseCursor
from revision_r1.checkpointing import save_generation
from revision_r1.drive_storage import DriveStorageError
from test_drive_api_storage import _FakeService, _storage
from test_production_workflow import _identity, _write_completed_production_fixture


def _checkpoints(root, identity):
    for task in range(1, 12):
        state = RunState(
            identity=identity,
            cursor=PhaseCursor(task_index=task, phase="export" if task == 11 else "scaler"),
            extra={"runtime": {"task_rows": [{"task_id": value} for value in range(1, task + 1)]}},
        )
        save_generation(root / "task_boundaries", state, identity,
                        kind="task-boundary", retain_generations=None)
    for kind in ("recovery", "complete"):
        state = RunState(
            identity=identity,
            cursor=PhaseCursor(task_index=11, phase="complete" if kind == "complete" else "export"),
            resources={"classifier_updates": 11},
            predictions={"y_true_global": [0], "y_pred_global": [0]},
            extra={"runtime": {"task_rows": [{"task_id": value} for value in range(1, 12)]}},
        )
        save_generation(root / "recovery", state, identity, kind=kind, retain_generations=None)


def _setup(tmp_path, monkeypatch):
    import revision_r1.resources as resources
    monkeypatch.setattr(resources, "read_host_memory_snapshot", lambda: {"effective_available_bytes": 8 << 30})
    service = _FakeService()
    stage = tmp_path / "stage"
    adapter = _storage(service, stage, monkeypatch)
    root, checkpoints = stage / "results", stage / "checkpoints"
    adapter.bind_root("/drive/MyDrive/ReVision/production/results", root)
    adapter.bind_root("/drive/MyDrive/ReVision/production/checkpoints", checkpoints)
    monkeypatch.setattr(workflow, "_BOUND_DRIVE_API_SERVICE", service)
    monkeypatch.setattr(workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": workflow.CANONICAL_STORAGE_ACCOUNT_EMAIL,
        "canonical_revision_folder_id": workflow.CANONICAL_REVISION_FOLDER_ID,
    })
    plan = json.loads((Path(__file__).parents[1] / "configs/production_run_plan_v1.json").read_text())
    for record in plan["runs"]:
        record["dataset_identity"] = _identity(record["dataset"])
        record["update_budgets_by_task"] = [1] * 11
        record["effective_schedule_sha256"] = plan["schedule_sha256"]
    plan = workflow._refresh_plan_hash(plan)
    first, second = plan["runs"][:2]
    out1 = workflow.run_output_root(root, first)
    identity1 = _write_completed_production_fixture(out1, plan, first)
    ckpt1 = workflow.run_checkpoint_root(checkpoints, first)
    _checkpoints(ckpt1, identity1)
    validated = workflow._validate_completed_run(plan, first, None, out1, synthetic=False)
    measurement = workflow.measure_completed_production_run(plan, first, None, out1, ckpt1)
    workflow._record_completed_run(root, plan, first, validated, storage_measurement=measurement)
    workflow._publish_api_tree(adapter, root)
    out2 = workflow.run_output_root(root, second)
    identity2 = _write_completed_production_fixture(out2, plan, second)
    _checkpoints(workflow.run_checkpoint_root(checkpoints, second), identity2)
    # Test fixture starts immediately before the real runner's completion call.
    shutil.rmtree(out2 / "registry")
    registry = recovery.api_run_registry_class(adapter)(out2 / "registry")
    registry.begin_attempt(workflow._run_id_for(second, synthetic=False), identity=identity2)
    return plan, second, root, checkpoints, adapter, service, out2, registry


def test_failed_final_publication_completion_only_retry_and_fresh_stage_skip(tmp_path, monkeypatch):
    finalizer = getattr(recovery, "complete_exported_run", None)
    assert callable(finalizer), "missing completion-only recovery; must not rerun trained tasks"
    plan, second, root, checkpoints, adapter, service, output, registry = _setup(tmp_path, monkeypatch)
    from revision_r1.runner import _write_completion
    from revision_r1.config import make_run_config
    config = make_run_config(second["dataset"], second["seed"], second["arm"], plan["schedule_sha256"])
    run_id = workflow._run_id_for(second, synthetic=False)
    identity = workflow._expected_plan_runner_identity(plan, second)
    service.files().fail_create_name = "selection_records.jsonl"
    with pytest.raises(Exception, match="upload failed"):
        _write_completion(RunState(identity=identity, cursor=PhaseCursor(task_index=11, phase="complete")), config, output, registry, run_id, False)
    registry.fail_attempt(run_id, "final artifact publication failed; training already complete")
    assert not RunRegistry(output / "registry").is_completed(run_id)
    service.files().fail_create_name = None
    artifact_hashes = {path.name: workflow._hash_file_stream(path) for path in output.iterdir() if path.is_file()}
    checkpoint_hashes = {str(path): workflow._hash_file_stream(path) for path in checkpoints.rglob("*.pt")}
    # Completion may read serialized cursor state, but must not construct a
    # model or execute any task.
    import revision_r1.runner as runner
    monkeypatch.setattr(runner, "run_one", lambda *_a, **_kw: pytest.fail("must not retrain"))
    args = dict(output_root=root, checkpoint_root=checkpoints, local_checkpoint_root=tmp_path / "local",
                storage_adapter=adapter, api_factory=lambda: service, acknowledge_no_other_worker=True)
    result = finalizer(plan, second, **args)
    assert result["status"] == "COMPLETED_VALIDATED"
    assert result["next_pending_run"] == "ember_seed42_anchor_only"
    assert result["training_launched"] is False
    assert len(result["checkpoint_evidence"]) == 13
    assert sorted(row["task_id"] for row in result["checkpoint_evidence"] if row.get("kind") == "task-boundary") == list(range(1, 12))
    assert {path.name: workflow._hash_file_stream(path) for path in output.iterdir() if path.is_file()} == artifact_hashes
    assert all(workflow._hash_file_stream(Path(path)) == digest for path, digest in checkpoint_hashes.items())
    entry = RunRegistry(output / "registry").status(run_id)
    assert entry["attempts"][-1]["operation"] == "completion_only"
    assert entry["attempts"][0]["status"] == "failed"
    attempts = len(entry["attempts"])
    again = finalizer(plan, second, **args)
    assert again["completion_registry_added_now"] is False
    assert len(RunRegistry(output / "registry").status(run_id)["attempts"]) == attempts
    empty_log = adapter.exact_file(adapter.logical_path_for_local(output / "selection_records.jsonl"))
    assert int(empty_log["size"]) == 0
    assert int(empty_log["quotaBytesUsed"]) == 0
    assert empty_log["md5Checksum"] == "d41d8cd98f00b204e9800998ecf8427e"
    # A fresh staging directory hydrates the canonical completion and skips both runs.
    stage2 = tmp_path / "fresh"
    fresh = _storage(service, stage2, monkeypatch)
    fresh.bind_root("/drive/MyDrive/ReVision/production/results", stage2 / "results")
    fresh.bind_root("/drive/MyDrive/ReVision/production/checkpoints", stage2 / "checkpoints")
    rerun = finalizer(plan, second, **{**args, "output_root": stage2 / "results",
                     "checkpoint_root": stage2 / "checkpoints", "storage_adapter": fresh})
    assert rerun["completion_registry_added_now"] is False
    assert rerun["next_pending_run"] == "ember_seed42_anchor_only"
    assert set(rerun["completed_run_keys"]) == {"ember_seed42_full_hybrid", "ember_seed42_current_only"}


def test_completion_only_validates_and_preserves_fifteen_recovery_generations(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, _adapter, _service, _output, _registry = setup
    checkpoint_namespace = workflow.run_checkpoint_root(checkpoint_root, record)
    recovery_dir = checkpoint_namespace / "recovery"
    for path in recovery_dir.iterdir():
        path.unlink()
    identity = workflow._expected_plan_runner_identity(plan, record)
    for index in range(15):
        save_generation(
            recovery_dir,
            RunState(identity=identity, cursor=PhaseCursor(task_index=11, phase="complete")),
            identity,
            kind="complete" if index == 14 else "recovery",
            retain_generations=None,
        )
    checkpoint_hashes = {
        str(path): workflow._hash_file_stream(path)
        for path in checkpoint_namespace.rglob("*") if path.is_file()
    }

    # The ordinary production measurement remains strict; only completion
    # recovery may account for a verified overflow without pruning it.
    with pytest.raises(ValueError, match=r"recovery: expected 2, found 15"):
        workflow._checkpoint_tree_measurement(
            checkpoint_namespace, identity, kind="recovery", expected_count=2,
        )

    result = recovery.complete_exported_run(plan, record, **args)

    assert result["status"] == "COMPLETED_VALIDATED"
    assert result["training_launched"] is False
    assert result["files_deleted"] is False
    assert result["next_pending_run"] == "ember_seed42_anchor_only"
    assert len(result["checkpoint_evidence"]) == 26
    measurement = workflow._read_matrix_registry(root, plan)["completed_runs"][record["run_key"]]["storage_measurement"]
    assert measurement["recovery_checkpoints"]["generation_count"] == 15
    assert measurement["recovery_checkpoints"]["generations"] == list(range(1, 16))
    assert measurement["task_boundary_checkpoints"]["generation_count"] == 11
    assert {
        str(path): workflow._hash_file_stream(path)
        for path in checkpoint_namespace.rglob("*") if path.is_file()
    } == checkpoint_hashes


def _failed_exports(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    plan, record, root, ckpt, adapter, service, output, registry = setup
    registry.fail_attempt(workflow._run_id_for(record, synthetic=False), "artifact upload failed")
    args = dict(output_root=root, checkpoint_root=ckpt, local_checkpoint_root=tmp_path / "local",
                storage_adapter=adapter, api_factory=lambda: service, acknowledge_no_other_worker=True)
    return setup, args


def _remove_boundary(checkpoint_root, record, generation):
    boundary_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    (boundary_dir / f"generation-{generation:06d}.pt").unlink()
    (boundary_dir / f"generation-{generation:06d}.manifest.json").unlink()
    return boundary_dir


def test_completion_reconciles_boundary_11_from_runtime_local_checkpoints(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, _adapter, _service, _output, _registry = setup
    staged = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    local_root = workflow.run_checkpoint_root(tmp_path / "local", record) / "task_boundaries"
    local_root.mkdir(parents=True)
    for suffix in ("pt", "manifest.json"):
        shutil.copyfile(staged / f"generation-000011.{suffix}", local_root / f"generation-000011.{suffix}")
    _remove_boundary(checkpoint_root, record, 11)
    result = recovery.complete_exported_run(
        plan, record, **{**args, "local_checkpoint_root": tmp_path / "local"},
    )
    final_boundary = next(row for row in result["checkpoint_evidence"]
                          if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert final_boundary["generation"] == 11
    assert final_boundary["source_provenance"]["runtime_local"] is True
    assert result["next_pending_run"] == "ember_seed42_anchor_only"


def test_completion_repairs_partial_staging_from_verified_runtime_payload(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, _adapter, _service, _output, _registry = setup
    boundary_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    local_payload = workflow.run_checkpoint_root(tmp_path / "local", record) / "task_boundaries" / "generation-000011.pt"
    local_payload.parent.mkdir(parents=True)
    shutil.copyfile(boundary_dir / local_payload.name, local_payload)
    (boundary_dir / "generation-000011.pt").unlink()

    result = recovery.complete_exported_run(
        plan, record, **{**args, "local_checkpoint_root": tmp_path / "local"},
    )

    boundary = next(row for row in result["checkpoint_evidence"]
                    if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert boundary["source_provenance"]["runtime_local"] is True
    assert boundary["source_provenance"]["api_staging_before_reconciliation"] is False
    assert (boundary_dir / "generation-000011.pt").is_file()
    assert result["next_pending_run"] == "ember_seed42_anchor_only"


def test_completion_repairs_corrupt_staging_manifest_from_verified_runtime_copy(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, _adapter, _service, _output, _registry = setup
    boundary_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    local_dir = workflow.run_checkpoint_root(tmp_path / "local", record) / "task_boundaries"
    local_dir.mkdir(parents=True)
    for suffix in ("pt", "manifest.json"):
        source = boundary_dir / f"generation-000011.{suffix}"
        shutil.copyfile(source, local_dir / source.name)
    (boundary_dir / "generation-000011.manifest.json").write_text("{truncated", encoding="utf-8")

    result = recovery.complete_exported_run(
        plan, record, **{**args, "local_checkpoint_root": tmp_path / "local"},
    )

    boundary = next(row for row in result["checkpoint_evidence"]
                    if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert boundary["source_provenance"]["runtime_local"] is True
    assert boundary["task_id"] == 11
    assert (boundary_dir / "generation-000011.manifest.json").is_file()
    assert "generation-000011.manifest.json" in str(result["checkpoint_source_inventory"]["task_boundaries"]["staging_quarantined_paths"][0])


def test_completion_reconciles_boundary_11_from_canonical_drive(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, adapter, _service, _output, _registry = setup
    staged = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    identity = workflow._expected_plan_runner_identity(plan, record)
    workflow._api_publish_checkpoint_generation(adapter, staged, identity, 11)
    _remove_boundary(checkpoint_root, record, 11)
    result = recovery.complete_exported_run(plan, record, **args)
    final_boundary = next(row for row in result["checkpoint_evidence"]
                          if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert final_boundary["source_provenance"]["canonical_drive"] is True
    assert final_boundary["source_provenance"]["drive_file_ids"]
    assert result["next_pending_run"] == "ember_seed42_anchor_only"


def test_completion_derives_only_final_boundary_from_valid_terminal_recovery(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, _adapter, _service, _output, _registry = setup
    boundary_dir = _remove_boundary(checkpoint_root, record, 11)
    result = recovery.complete_exported_run(plan, record, **args)
    final_boundary = next(row for row in result["checkpoint_evidence"]
                          if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert final_boundary["recovery_provenance"] == {
        "source_kind": "complete", "source_generation": 2, "task_id": 11,
        "validated_cursor": {"task_index": 11, "phase": "complete"},
    }
    assert (boundary_dir / f"generation-{final_boundary['generation']:06d}.manifest.json").is_file()
    assert result["next_pending_run"] == "ember_seed42_anchor_only"


def test_completion_derives_final_boundary_from_canonical_terminal_recovery(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, adapter, _service, _output, _registry = setup
    boundary_dir = _remove_boundary(checkpoint_root, record, 11)
    recovery_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "recovery"
    identity = workflow._expected_plan_runner_identity(plan, record)
    workflow._api_publish_checkpoint_generation(adapter, recovery_dir, identity, 2)
    (recovery_dir / "generation-000002.pt").unlink()
    (recovery_dir / "generation-000002.manifest.json").unlink()

    result = recovery.complete_exported_run(plan, record, **args)

    final_boundary = next(row for row in result["checkpoint_evidence"]
                          if row.get("kind") == "task-boundary" and row.get("task_id") == 11)
    assert final_boundary["recovery_provenance"]["source_generation"] == 2
    assert final_boundary["source_provenance"]["recovery_source_provenance"]["canonical_drive"] is True
    assert (boundary_dir / f"generation-{final_boundary['generation']:06d}.pt").is_file()
    assert result["completion_registry_added_now"] is True


def test_completion_reports_missing_earlier_boundary_without_registration(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, _root, checkpoint_root, _adapter, _service, _output, _registry = setup
    _remove_boundary(checkpoint_root, record, 5)
    with pytest.raises(ValueError, match=r'"missing_task_ids": \[5\]') as error:
        recovery.complete_exported_run(plan, record, **args)
    message = str(error.value)
    assert '"task_id": 5' in message
    assert '"identity_sha256"' in message
    assert '"payload_integrity": "NO_AUTHENTICATED_TASK_BOUNDARY"' in message


def test_terminal_recovery_with_wrong_task_rows_cannot_supply_final_boundary(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, _adapter, _service, _output, _registry = setup
    boundary_dir = _remove_boundary(checkpoint_root, record, 11)
    identity = workflow._expected_plan_runner_identity(plan, record)
    recovery_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "recovery"
    (recovery_dir / "generation-000002.pt").unlink()
    (recovery_dir / "generation-000002.manifest.json").unlink()
    state = RunState(
        identity=identity,
        cursor=PhaseCursor(task_index=11, phase="complete"),
        resources={"classifier_updates": 11},
        predictions={"y_true_global": [0], "y_pred_global": [0]},
        extra={"runtime": {"task_rows": [{"task_id": value} for value in range(1, 11)]}},
    )
    save_generation(recovery_dir, state, identity, kind="complete", retain_generations=None)

    with pytest.raises(ValueError, match="cursor/task-row provenance mismatch"):
        recovery.complete_exported_run(plan, record, **args)

    assert not (boundary_dir / "generation-000011.manifest.json").exists()
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]


def test_missing_task_eleven_reports_explicit_gap_without_terminal_recovery(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, _adapter, _service, _output, _registry = setup
    _remove_boundary(checkpoint_root, record, 11)
    recovery_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "recovery"
    (recovery_dir / "generation-000002.pt").unlink()
    (recovery_dir / "generation-000002.manifest.json").unlink()
    identity = workflow._expected_plan_runner_identity(plan, record)
    save_generation(
        recovery_dir,
        RunState(identity=identity, cursor=PhaseCursor(task_index=11, phase="complete"),
                 resources={"classifier_updates": 11}, predictions={"y_true_global": [0], "y_pred_global": [0]},
                 extra={"runtime": {"task_rows": [{"task_id": value} for value in range(1, 12)]}}),
        identity, kind="recovery", retain_generations=None,
    )

    with pytest.raises(ValueError, match=r'"missing_task_ids": \[11\]') as error:
        recovery.complete_exported_run(plan, record, **args)
    message = str(error.value)
    assert '"task_id": 11' in message
    assert '"payload_integrity": "NO_AUTHENTICATED_TASK_BOUNDARY"' in message
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]


def test_recovered_final_boundary_requires_its_exact_terminal_source_generation(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, adapter, service, _output, _registry = setup
    _remove_boundary(checkpoint_root, record, 11)
    original_validate = recovery._validate_export_evidence
    calls = {"count": 0}

    def interrupt_after_checkpoint_publication(*values):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("interrupted after checkpoint publication")
        return original_validate(*values)

    monkeypatch.setattr(recovery, "_validate_export_evidence", interrupt_after_checkpoint_publication)
    with pytest.raises(OSError, match="interrupted after checkpoint publication"):
        recovery.complete_exported_run(plan, record, **args)
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]
    monkeypatch.setattr(recovery, "_validate_export_evidence", original_validate)

    identity = workflow._expected_plan_runner_identity(plan, record)
    recovery_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "recovery"
    terminal = recovery._manifest(recovery_dir / "generation-000002.manifest.json", identity)
    from revision_r1.checkpointing import copy_generation_payload
    copy_generation_payload(recovery_dir, terminal, recovery_dir, identity,
                            kind="complete", retain_generations=None, generation=3)
    workflow._api_publish_checkpoint_generation(adapter, recovery_dir, identity, 3)
    for suffix in ("pt", "manifest.json"):
        (recovery_dir / f"generation-000002.{suffix}").unlink()
        remote = adapter.exact_file(adapter.logical_path_for_local(recovery_dir / f"generation-000002.{suffix}"))
        del service.records[remote["id"]]

    with pytest.raises(ValueError, match="terminal recovery source generation 2 is unavailable"):
        recovery.complete_exported_run(plan, record, **args)
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]


def test_nonintegral_boundary_cursor_is_rejected_before_completion_registration(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, _adapter, _service, _output, _registry = setup
    boundary_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    manifest_path = boundary_dir / "generation-000001.manifest.json"
    manifest = recovery._manifest(manifest_path, workflow._expected_plan_runner_identity(plan, record))
    identity = workflow._expected_plan_runner_identity(plan, record)
    state = RunState(
        identity=identity,
        cursor=PhaseCursor(task_index=1.5, phase="scaler", item_index=0.5),
        extra={"runtime": {"task_rows": [{"task_id": 1}]}},
    )
    payload_path = boundary_dir / manifest["payload_file"]
    torch.save(state, payload_path)
    manifest["payload_bytes"] = payload_path.stat().st_size
    manifest["payload_sha256"] = workflow._hash_file_stream(payload_path)
    workflow._atomic_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="cursor fields must be exact integers"):
        recovery.complete_exported_run(plan, record, **args)
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]


def test_checkpoint_publication_interruption_retries_without_duplicate_file_or_registration(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, adapter, service, _output, _registry = setup
    staged = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    local_root = workflow.run_checkpoint_root(tmp_path / "local", record) / "task_boundaries"
    local_root.mkdir(parents=True)
    for suffix in ("pt", "manifest.json"):
        shutil.copyfile(staged / f"generation-000011.{suffix}", local_root / f"generation-000011.{suffix}")
    _remove_boundary(checkpoint_root, record, 11)
    original_publish = adapter.publish_file
    fail_once = {"armed": True}

    def interrupt_manifest(local_path, logical_path, **kwargs):
        if Path(local_path).name == "generation-000011.manifest.json" and fail_once["armed"]:
            fail_once["armed"] = False
            raise OSError("interrupted final-boundary manifest publication")
        return original_publish(local_path, logical_path, **kwargs)

    monkeypatch.setattr(adapter, "publish_file", interrupt_manifest)
    retry_args = {**args, "local_checkpoint_root": tmp_path / "local"}
    with pytest.raises(OSError, match="interrupted final-boundary manifest publication"):
        recovery.complete_exported_run(plan, record, **retry_args)
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]
    assert len([row for row in service.records.values() if row.get("name") == "generation-000011.pt"]) == 1
    result = recovery.complete_exported_run(plan, record, **retry_args)
    assert result["completion_registry_added_now"] is True
    assert any(row.get("kind") == "task-boundary" and row.get("task_id") == 11
               for row in result["checkpoint_evidence"])
    assert len([row for row in service.records.values() if row.get("name") == "generation-000011.pt"]) == 1
    assert result["next_pending_run"] == "ember_seed42_anchor_only"


@pytest.mark.parametrize("fault", ["wrong_identity", "corrupt_payload"])
def test_invalid_task_boundary_blocks_completion_without_modifying_canonical_checkpoint(tmp_path, monkeypatch, fault):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, checkpoint_root, adapter, service, _output, _registry = setup
    boundary_dir = workflow.run_checkpoint_root(checkpoint_root, record) / "task_boundaries"
    manifest = boundary_dir / "generation-000011.manifest.json"
    payload = boundary_dir / "generation-000011.pt"
    remote_payload = workflow._api_publish_file(adapter, payload)
    remote_before = bytes(service.records[remote_payload["file_id"]]["content"])
    if fault == "wrong_identity":
        value = json.loads(manifest.read_text())
        value["identity"]["code_sha256"] = "0" * 64
        manifest.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            recovery.complete_exported_run(plan, record, **args)
    else:
        payload.write_bytes(b"corrupt-local-payload")
        remote_record = service.records[remote_payload["file_id"]]
        remote_record["content"] = b"corrupt-canonical-payload"
        remote_record["size"] = str(len(remote_record["content"]))
        remote_record["md5Checksum"] = hashlib.md5(remote_record["content"], usedforsecurity=False).hexdigest()
        remote_record["quotaBytesUsed"] = remote_record["size"]
        remote_before = bytes(remote_record["content"])
        with pytest.raises((ValueError, DriveStorageError)):
            recovery.complete_exported_run(plan, record, **args)
    assert record["run_key"] not in workflow._read_matrix_registry(root, plan)["completed_runs"]
    assert bytes(service.records[remote_payload["file_id"]]["content"]) == remote_before


def test_matrix_publication_failure_fresh_retry_preserves_canonical_marker(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, ckpt, adapter, service, output, _registry = setup
    publish = adapter.publish_file
    def fail_matrix(path, logical, **kwargs):
        if Path(path).name == "matrix_completion_registry.json":
            raise OSError("matrix publication interrupted")
        return publish(path, logical, **kwargs)
    monkeypatch.setattr(adapter, "publish_file", fail_matrix)
    with pytest.raises(OSError, match="matrix publication interrupted"):
        recovery.complete_exported_run(plan, record, **args)
    assert record["run_key"] not in json.loads(next(row["content"] for row in service.records.values()
        if row["name"] == "matrix_completion_registry.json"))["completed_runs"]
    marker_bytes = {row["id"]: bytes(row["content"]) for row in service.records.values() if row["name"] == "COMPLETE.json"}
    stage = tmp_path / "fresh"
    fresh = _storage(service, stage, monkeypatch)
    fresh.bind_root("/drive/MyDrive/ReVision/production/results", stage / "results")
    fresh.bind_root("/drive/MyDrive/ReVision/production/checkpoints", stage / "checkpoints")
    result = recovery.complete_exported_run(plan, record, **{**args, "storage_adapter": fresh,
        "output_root": stage / "results", "checkpoint_root": stage / "checkpoints"})
    assert result["completion_registry_added_now"] is True
    assert result["next_pending_run"] == "ember_seed42_anchor_only"
    assert all(bytes(service.records[file_id]["content"]) == payload for file_id, payload in marker_bytes.items())


@pytest.mark.parametrize("fault", ["missing", "corrupt", "checkpoint", "wrong_identity"])
def test_incomplete_or_corrupt_evidence_blocks_without_training_or_registration(tmp_path, monkeypatch, fault):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, ckpt, adapter, service, output, registry = setup
    if fault == "missing":
        (output / "predictions.npz").unlink()
    elif fault == "corrupt":
        (output / "selection_records.jsonl").write_text("fabricated")
    elif fault == "checkpoint":
        payload = sorted(workflow.run_checkpoint_root(ckpt, record).glob("recovery/*.pt"))[-1]
        payload.write_bytes(b"truncated")
    else:
        path = registry._state_path(workflow._run_id_for(record, synthetic=False))
        state = json.loads(path.read_text())
        state["identity"]["code_sha256"] = "0" * 64
        path.write_text(json.dumps(state))
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.complete_exported_run(plan, record, **args)
    remote = json.loads(next(row["content"] for row in service.records.values() if row["name"] == "matrix_completion_registry.json"))
    assert set(remote["completed_runs"]) == {"ember_seed42_full_hybrid"}
    assert not any(row["name"] == "COMPLETE.json" and record["arm"] in str(row.get("content", b"")) for row in service.records.values())


def test_active_worker_excludes_completion_retry(tmp_path, monkeypatch):
    from revision_r1.drive_storage import DriveApiLease, DriveLeaseError
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, _ckpt, adapter, _service, _output, _registry = setup
    logical_lock = adapter.logical_path_for_local(root / ".matrix_execution.lease.json")
    adapter.ensure_lease_record(logical_lock)
    with DriveApiLease(adapter, logical_lock):
        with pytest.raises(DriveLeaseError):
            recovery.complete_exported_run(plan, record, **args)


@pytest.mark.parametrize("quota", [0, None])
def test_storage_probe_uses_content_proof_not_positive_quota(tmp_path, quota):
    from test_production_workflow import _SessionProbeAdapter
    class AccountingProbe(_SessionProbeAdapter):
        def probe_write(self, *args, **kwargs):
            report = super().probe_write(*args, **kwargs)
            report["quotaBytesUsed"] = quota
            return report
    logical = tmp_path / "drive/MyDrive/ReVision"
    adapter = AccountingProbe(object(), "revision", logical)
    evidence = workflow._write_capability_probe(logical, {"path_exists": True, "folder_id": "revision"},
        workflow.CANONICAL_STORAGE_ACCOUNT_EMAIL, adapter.service, drive_root=tmp_path / "drive/MyDrive", storage_adapter=adapter)
    assert evidence["status"] == "PASS"
    assert evidence["new_file_quota_bytes"] == quota
    assert evidence["new_file_owner"] == workflow.CANONICAL_STORAGE_ACCOUNT_EMAIL


def test_actual_notebook_matrix_cell_is_disabled_by_default(capsys):
    import ast
    from scripts.build_github_handoff import _github_matrix_cell
    tree = ast.parse(_github_matrix_cell())
    namespace = {}
    exec(compile(tree, "distributed-Cell-F-default", "exec"), namespace)
    assert namespace["LAUNCH_REMAINING_MATRIX"] is False
    assert namespace["ACKNOWLEDGE_NO_OTHER_COLAB_WORKER"] is False
    assert namespace["CLEAN_RESTART_RUN2"] is True
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("NOT_LAUNCHED:") for line in lines)
    assert any("Run 1 remains finalized/skipped" in line for line in lines)
    assert not any(line.startswith(("Training complete:", "Publishing artifacts:", "Matrix status:"))
                   for line in lines)


@pytest.mark.parametrize("name", ["selection_records.jsonl", "resource_metrics.json", "artifact_manifest.json", "checkpoint"])
def test_conflicting_canonical_bytes_are_preserved(tmp_path, monkeypatch, name):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, ckpt, adapter, service, output, _registry = setup
    if name == "checkpoint":
        source = sorted(workflow.run_checkpoint_root(ckpt, record).glob("recovery/*.pt"))[-1]
    else:
        source = output / name
    logical = adapter.logical_path_for_local(source)
    other = tmp_path / "conflicting-bytes"
    other.write_bytes(b"different historical bytes")
    published = adapter.publish_file(other, logical)
    before = bytes(service.records[published["file_id"]]["content"])
    with pytest.raises(ValueError, match="conflict"):
        recovery.complete_exported_run(plan, record, **args)
    assert bytes(service.records[published["file_id"]]["content"]) == before
    remote = json.loads(next(row["content"] for row in service.records.values() if row["name"] == "matrix_completion_registry.json"))
    assert set(remote["completed_runs"]) == {"ember_seed42_full_hybrid"}


def test_stale_local_failed_registry_cannot_overwrite_canonical_completion(tmp_path, monkeypatch):
    setup, args = _failed_exports(tmp_path, monkeypatch)
    plan, record, root, _ckpt, adapter, service, output, registry = setup
    run_id = workflow._run_id_for(record, synthetic=False)
    stale_state = registry._state_path(run_id).read_bytes()
    recovery.complete_exported_run(plan, record, **args)
    remote_before = {row["id"]: bytes(row["content"]) for row in service.records.values()
                     if row["name"] in {"registry.json", "COMPLETE.json", "matrix_completion_registry.json"}}
    registry._state_path(run_id).write_bytes(stale_state)
    result = recovery.complete_exported_run(plan, record, **args)
    assert result["completion_registry_added_now"] is False
    assert all(bytes(service.records[key]["content"]) == value for key, value in remote_before.items())
    assert RunRegistry(output / "registry").is_completed(run_id)
    assert any("prior_registry" in row["name"] for row in service.records.values())
