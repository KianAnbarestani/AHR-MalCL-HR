from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest
import numpy as np
import torch

from revision_r1.config import ARMS, DATASETS, SEEDS, TASK_SIZES, make_run_config
from revision_r1.schedule import PoolCounts
from revision_r1.storage import DEFAULT_STORAGE_RESERVE_BYTES, StorageCapacityError
from revision_r1.task_stream import class_order, order_sha256, split_order
from revision_r1.runner import _code_sha256
from revision_r1.data import DatasetBundle
from scripts.production_workflow import (
    _SequentialDatasetBundleCache,
    _checkpoint_payload_size_reference,
    bind_runtime_schedule,
    build_frozen_run_plan,
    _logical_drive_path_components,
    continue_production_matrix,
    execute_frozen_plan_record,
    execute_first_production_run,
    export_production_progress,
    inspect_drive_destination,
    load_v62_recovery_checkpoint_bound,
    load_run_progress,
    project_storage,
    require_run_capacity,
    run_checkpoint_root,
    run_output_root,
    resolve_drive_folder,
    training_source_sha256,
    validate_frozen_plan,
    validate_runtime_bound_plan,
)
import scripts.production_workflow as production_workflow


SCHEDULE_SHA = "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212"
DATASET_REFERENCES = {
    "ember": {
        "train_md5": DATASETS["ember"]["train"]["md5"],
        "test_md5": DATASETS["ember"]["test"]["md5"],
        "feature_dim": 2381,
        "train_shape": [303331, 2381],
        "test_shape": [33704, 2381],
    },
    "az_class": {
        "train_md5": DATASETS["az_class"]["train"]["md5"],
        "test_md5": DATASETS["az_class"]["test"]["md5"],
        "feature_dim": 2439,
        "train_shape": [257023, 2439],
        "test_shape": [28559, 2439],
    },
}


def _identity(dataset: str) -> dict:
    reference = DATASET_REFERENCES[dataset]
    return {
        "train_md5": reference["train_md5"],
        "train_sha256": hashlib.sha256(f"{dataset}:train".encode()).hexdigest(),
        "test_md5": reference["test_md5"],
        "test_sha256": hashlib.sha256(f"{dataset}:test".encode()).hexdigest(),
    }


def _preflight() -> dict:
    result = {
        "status": "PREFLIGHT_VALIDATED_NO_TRAINING",
        "schedule": {
            "sha256": SCHEDULE_SHA,
            "row_count": 220,
            "provenance": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
            "reconciliation": {"adjusted_rows": 0, "changed_update_rows": 0},
        },
    }
    for dataset in DATASETS:
        identity = _identity(dataset)
        counts = {str(label): 1000 for label in range(100)}
        result[dataset] = {
            "train": {"md5": identity["train_md5"], "sha256": identity["train_sha256"], "class_counts": counts},
            "test": {"md5": identity["test_md5"], "sha256": identity["test_sha256"], "class_counts": counts},
        }
    return result


def _write_effective_schedule(tmp_path: Path, preflight: dict) -> tuple[Path, Path]:
    schedule_path = tmp_path / "effective.csv"
    manifest_path = tmp_path / "effective.manifest.json"
    fields = [
        "dataset", "seed", "task", "current_pool_size", "anchor_pool_size",
        "generated_pool_size", "full_hybrid_pool_size", "batch_size",
        "classifier_epochs_equivalent", "proposed_classifier_updates",
        "budget_provenance", "training_reproduction_claimed",
    ]
    rows = []
    actual = {}
    for dataset in ("ember", "az_class"):
        quota = DATASETS[dataset]["anchor_k"]
        for seed in SEEDS:
            seen: list[int] = []
            for task_id, group in enumerate(split_order(class_order(seed), TASK_SIZES), start=1):
                current = len(group) * 1000
                anchors = sum(min(quota, 1000) for _ in seen)
                generated = len(seen) * 100
                pools = PoolCounts(current, anchors, generated)
                actual[(dataset, seed, task_id)] = pools
                full = pools.total
                rows.append({
                    "dataset": dataset, "seed": seed, "task": task_id,
                    "current_pool_size": pools.current, "anchor_pool_size": pools.anchor,
                    "generated_pool_size": pools.generated, "full_hybrid_pool_size": full,
                    "batch_size": 256, "classifier_epochs_equivalent": 3,
                    "proposed_classifier_updates": 3 * ((full + 255) // 256),
                    "budget_provenance": "TEST_FIXTURE_DATA_DERIVED",
                    "training_reproduction_claimed": "NO",
                })
                seen.extend(group)
    with schedule_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    identities = {dataset: _identity(dataset) for dataset in DATASETS}
    manifest_path.write_text(json.dumps({
        "status": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
        "training_reproduction_claimed": False,
        "row_count": 220,
        "source_csv_sha256": digest,
        "dataset_identities": identities,
        "reconciliation": {"adjusted_rows": 0, "changed_update_rows": 0},
    }), encoding="utf-8")
    return schedule_path, manifest_path


def _prepare_package(package_root: Path) -> None:
    source = package_root / "revision_r1"
    source.mkdir(parents=True, exist_ok=True)
    (source / "fixture_core.py").write_text("# deterministic source fixture\n", encoding="utf-8")


def _plan(package_root: Path) -> dict:
    _prepare_package(package_root)
    return build_frozen_run_plan(
        package_root,
        schedule_sha256=SCHEDULE_SHA,
        dataset_references=DATASET_REFERENCES,
    )


def test_matrix_bundle_cache_loads_each_contiguous_dataset_once_and_holds_only_one():
    calls = []
    bundles = {"ember": object(), "az_class": object()}
    cache = _SequentialDatasetBundleCache(lambda dataset: calls.append(dataset) or bundles[dataset])

    assert cache.get("ember") is bundles["ember"]
    assert cache.get("ember") is bundles["ember"]
    assert cache.get("az_class") is bundles["az_class"]
    assert cache.get("az_class") is bundles["az_class"]
    assert cache.get("ember") is bundles["ember"]
    assert calls == ["ember", "az_class", "ember"]


def test_plan_has_exactly_100_unique_runs_and_required_first_run(tmp_path):
    plan = _plan(tmp_path)

    assert plan["run_count"] == 100
    assert len(plan["runs"]) == 100
    keys = [(row["dataset"], row["seed"], row["arm"]) for row in plan["runs"]]
    assert len(set(keys)) == 100
    assert keys[0] == ("ember", 42, "full_hybrid")
    assert set(keys) == {
        (dataset, seed, arm)
        for dataset in DATASETS for seed in SEEDS for arm in ARMS
    }


def test_run_records_bind_existing_config_schedule_source_and_class_order(tmp_path):
    plan = _plan(tmp_path)
    first = plan["runs"][0]
    expected_config = make_run_config("ember", 42, "full_hybrid", SCHEDULE_SHA)

    assert first["run_index"] == 1
    assert first["run_key"] == "ember_seed42_full_hybrid"
    assert first["config_sha256"] == expected_config.sha256()
    assert first["config"] == expected_config.payload()
    assert first["schedule_sha256"] == SCHEDULE_SHA
    assert first["training_source_sha256"] == training_source_sha256(tmp_path)
    assert first["class_order_sha256"] == order_sha256(class_order(42))
    assert first["class_order"] == class_order(42)


def test_source_hash_matches_the_unmodified_runner_identity_algorithm(tmp_path):
    source = tmp_path / "revision_r1"
    source.mkdir()
    (source / "b.py").write_bytes(b"second")
    (source / "a.py").write_bytes(b"first")
    expected = hashlib.sha256(b"a.pyfirstb.pysecond").hexdigest()

    assert training_source_sha256(tmp_path) == expected


def test_source_hash_matches_actual_runner_hash_and_contains_no_legacy_override():
    project_root = Path(__file__).resolve().parents[1]

    assert training_source_sha256(project_root) == _code_sha256()


def test_readonly_lease_inspector_reports_live_expired_malformed_and_missing(tmp_path):
    from scripts.production_workflow import inspect_lease_readonly

    now = 2_000_000_000.0
    live_path = tmp_path / "live.lease.json"
    live = {"host": "other-runtime", "pid": 77, "heartbeat_at": now - 20, "expires_at": now + 280}
    live_path.write_text(json.dumps(live), encoding="utf-8")
    live_report = inspect_lease_readonly(live_path, now_unix=now, current_host="this-runtime", current_pid=10)
    assert live_report["status"] == "live"
    assert live_report["record"] == live
    assert live_report["raw_text"] == json.dumps(live)
    assert live_report["owner_process_visibility"] == "different_runtime_unverifiable"
    assert live_path.read_text(encoding="utf-8") == json.dumps(live)

    expired_path = tmp_path / "expired.lease.json"
    expired_path.write_text(json.dumps({"host": "old-runtime", "pid": 77, "heartbeat_at": now - 400, "expires_at": now - 100}))
    expired = inspect_lease_readonly(expired_path, now_unix=now, current_host="this-runtime", current_pid=10)
    assert expired["status"] == "expired"
    assert expired["recovery_may_be_attempted_by_existing_lease_logic"] is True

    malformed_path = tmp_path / "malformed.lease.json"
    malformed_path.write_text("{broken")
    malformed = inspect_lease_readonly(malformed_path, now_unix=now)
    assert malformed["status"] == "malformed"
    assert malformed["recovery_may_be_attempted_by_existing_lease_logic"] is False

    missing = inspect_lease_readonly(tmp_path / "absent.lease.json", now_unix=now)
    assert missing["status"] == "missing"
    assert missing["raw_text"] is None


def test_first_run_lease_diagnostic_preserves_lock_and_verifies_recent_checkpoint(tmp_path):
    from scripts.production_workflow import inspect_first_run_lease_and_progress, run_checkpoint_root, run_output_root
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState

    plan = _plan(tmp_path / "pkg")
    current = dict(plan["runs"][0])
    legacy = dict(current)
    legacy["training_source_sha256"] = "b" * 64
    output_root, checkpoint_root = tmp_path / "results", tmp_path / "checkpoints"
    output_root.mkdir()
    matrix_lock = output_root / ".matrix_execution.lease.json"
    matrix_raw = json.dumps({"host": "other-runtime", "pid": 41, "heartbeat_at": 1_900_000_000, "expires_at": 2_100_000_000})
    matrix_lock.write_text(matrix_raw)

    legacy_output = run_output_root(output_root, legacy, runner_version="revision_R1.0.1")
    legacy_output.mkdir(parents=True)
    (legacy_output / "status.json").write_text(json.dumps({"status": "running", "task": 3, "phase": "classifier"}))
    progress_path = output_root / "progress" / f"{legacy['run_key']}_{legacy['config_sha256'][:12]}.json"
    progress_path.parent.mkdir()
    progress_path.write_text(json.dumps({"status": "running", "task": 3, "phase": "classifier", "cursor": 9}))
    identity = production_workflow._diagnostic_run_identity(
        legacy, _preflight(), runner_version="revision_R1.0.1",
    )
    recovery = run_checkpoint_root(checkpoint_root, legacy, runner_version="revision_R1.0.1") / "recovery"
    state = RunState(identity=identity, cursor=PhaseCursor(task_index=2, phase="classifier"))
    saved = save_generation(recovery, state, identity, kind="recovery")

    report = inspect_first_run_lease_and_progress(
        plan, preflight=_preflight(), output_root=output_root,
        checkpoint_root=checkpoint_root, legacy_first_run_record=legacy,
        legacy_build_identity={"training_source_sha256": "b" * 64, "zip_sha256": "c" * 64},
        current_runtime={"runtime_instance_id": "fixture-runtime"}, now_unix=2_000_000_000,
    )
    old = report["run_namespaces"][0]
    assert report["diagnostic_mode"] == "READ_ONLY_NO_LOCK_RECOVERY_NO_TRAINING"
    assert "/revision_R1.0.1/" in old["run_output_root"]
    assert "/revision_R1.0.1/" in old["task_boundary_checkpoint_directory"]
    assert "/revision_R1.0.2/" in report["run_namespaces"][1]["run_output_root"]
    assert report["matrix_lease"]["status"] == "live"
    assert report["matrix_lease"]["raw_text"] == matrix_raw
    assert old["run_status_json"]["record"]["phase"] == "classifier"
    assert old["progress"]["record"]["cursor"] == 9
    candidate = old["recovery_checkpoints"]["candidates"][0]
    assert candidate["generation"] == saved["generation"]
    assert candidate["integrity_status"] == "PASS_MANIFEST_AND_PAYLOAD_HASH"
    assert report["run_namespaces"][1]["first_run_resume_gate"]["status"] == "BLOCKED_NO_VALID_RECOVERY"
    assert matrix_lock.read_text() == matrix_raw
    assert old["training_source_matches_current_package"] is False
    assert report["matrix_registry_paths"]["legacy_v1_2"] == str(output_root / "registry" / "matrix_completion_registry.json")
    assert report["matrix_registry_paths"]["current_package"] == str(
        output_root / "registry" / plan["plan_sha256"] / "matrix_completion_registry.json"
    )
    assert report["legacy_v1_2_matrix_registry"] is None
    assert report["matrix_registry"] is None
    assert report["remote_colab_process_visibility"].startswith("UNAVAILABLE")


def test_first_run_resume_gate_requires_exact_hashed_v13_recovery(tmp_path):
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState
    from revision_r1.checkpointing import CheckpointError, load_latest_valid
    from scripts.production_workflow import inspect_first_run_resume_checkpoint, _stage_required_resume_manifest

    plan = _production_source_plan()
    preflight = _preflight()
    # The fixture's first run is bound to the same preflight hashes.
    checkpoint_root = tmp_path / "checkpoints"
    record = plan["runs"][0]
    identity = production_workflow._diagnostic_run_identity(record, preflight)
    directory = run_checkpoint_root(checkpoint_root, record) / "recovery"
    saved = save_generation(
        directory, RunState(identity=identity, cursor=PhaseCursor(task_index=0, phase="gan_critic")),
        identity, kind="recovery",
    )

    passed = inspect_first_run_resume_checkpoint(plan, preflight=preflight, checkpoint_root=checkpoint_root)
    assert passed["status"] == "PASS_RESUME_CANDIDATE"
    assert passed["automatic_fresh_start_allowed"] is False
    assert passed["verified_candidates"][0]["generation"] == saved["generation"]

    local_recovery = tmp_path / "empty-local-staging" / "recovery"
    staged = _stage_required_resume_manifest(
        directory, local_recovery, expected_identity=identity,
        generation=saved["generation"], manifest_sha256=passed["selected_integrity_verified_candidate"]["manifest_sha256"],
    )
    assert staged.read_bytes() == (directory / staged.name).read_bytes()
    assert not (local_recovery / saved["payload_file"]).exists()
    with pytest.raises(CheckpointError, match="no valid checkpoint generation"):
        load_latest_valid(local_recovery, identity)
    (local_recovery / saved["payload_file"]).write_bytes((directory / saved["payload_file"]).read_bytes())
    restored = load_latest_valid(local_recovery, identity)
    assert restored.state.identity == identity
    assert restored.state.cursor.phase == "gan_critic"
    with pytest.raises(RuntimeError, match="manifest changed"):
        _stage_required_resume_manifest(
            directory, tmp_path / "wrong-local", expected_identity=identity,
            generation=saved["generation"], manifest_sha256="0" * 64,
        )

    manifest_path = directory / f"generation-{saved['generation']:06d}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rejected = inspect_first_run_resume_checkpoint(plan, preflight=preflight, checkpoint_root=checkpoint_root)
    assert rejected["status"] == "BLOCKED_IDENTITY_MISMATCH"
    assert rejected["automatic_fresh_start_allowed"] is False


def test_first_run_resume_gate_accepts_verified_candidate_with_protected_extra_manifests(tmp_path):
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState
    from scripts.production_workflow import inspect_first_run_resume_checkpoint

    plan = _production_source_plan()
    preflight = _preflight()
    record = plan["runs"][0]
    identity = production_workflow._diagnostic_run_identity(record, preflight)
    directory = run_checkpoint_root(tmp_path / "checkpoints", record) / "recovery"
    for task in range(3):
        save_generation(
            directory,
            RunState(identity=identity, cursor=PhaseCursor(task_index=task, phase="gan_critic")),
            identity,
            kind="recovery",
            retain_generations=None,
        )
    # Model the observed orphan: its manifest remains but its payload is absent.
    (directory / "generation-000001.pt").unlink()

    result = inspect_first_run_resume_checkpoint(
        plan, preflight=preflight, checkpoint_root=tmp_path / "checkpoints",
    )

    assert result["status"] == "PASS_RESUME_CANDIDATE"
    assert result["automatic_fresh_start_allowed"] is False
    assert result["manifest_generation_count"] == 3
    assert result["retention_overflow_count"] == 1
    assert result["uninspected_manifest_generations"] == [1]
    assert result["selected_integrity_verified_candidate"]["generation"] == 3


def test_first_run_resume_gate_still_rejects_wrong_identity_in_excess_manifest(tmp_path):
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState
    from scripts.production_workflow import inspect_first_run_resume_checkpoint

    plan = _production_source_plan()
    preflight = _preflight()
    record = plan["runs"][0]
    identity = production_workflow._diagnostic_run_identity(record, preflight)
    directory = run_checkpoint_root(tmp_path / "checkpoints", record) / "recovery"
    for task in range(3):
        save_generation(
            directory,
            RunState(identity=identity, cursor=PhaseCursor(task_index=task, phase="gan_critic")),
            identity,
            kind="recovery",
            retain_generations=None,
        )
    oldest_manifest = directory / "generation-000001.manifest.json"
    document = json.loads(oldest_manifest.read_text(encoding="utf-8"))
    document["identity"]["code_sha256"] = "0" * 64
    oldest_manifest.write_text(json.dumps(document), encoding="utf-8")

    result = inspect_first_run_resume_checkpoint(
        plan, preflight=preflight, checkpoint_root=tmp_path / "checkpoints",
    )

    assert result["status"] == "BLOCKED_IDENTITY_MISMATCH"
    assert result["automatic_fresh_start_allowed"] is False


def test_first_run_resume_gate_blocks_absent_recovery_instead_of_fresh_start(tmp_path):
    from scripts.production_workflow import inspect_first_run_resume_checkpoint

    plan = _production_source_plan()
    result = inspect_first_run_resume_checkpoint(
        plan, preflight=_preflight(), checkpoint_root=tmp_path / "empty-checkpoints",
    )
    assert result["status"] == "BLOCKED_NO_VALID_RECOVERY"
    assert result["automatic_fresh_start_allowed"] is False


def test_matrix_completion_registries_are_plan_scoped_and_legacy_path_is_preserved(tmp_path):
    first = {"plan_sha256": "a" * 64}
    second = {"plan_sha256": "b" * 64}
    root = tmp_path / "results"
    legacy = production_workflow._matrix_registry_path(root)
    assert legacy == root / "registry" / "matrix_completion_registry.json"
    assert production_workflow._matrix_registry_path(root, first) != production_workflow._matrix_registry_path(root, second)
    assert production_workflow._matrix_registry_path(root, first).parent == root / "registry" / ("a" * 64)


def test_first_run_holds_matrix_lease_before_and_during_dataset_load(tmp_path, monkeypatch):
    from revision_r1.data import DatasetBundle
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState

    preflight = _preflight()
    schedule_dir = tmp_path / "effective"
    schedule_dir.mkdir()
    schedule_path, manifest_path = _write_effective_schedule(schedule_dir, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    monkeypatch.setattr(production_workflow, "EXPECTED_ANCHOR_CORRECTIONS", 0)
    monkeypatch.setattr(production_workflow, "EXPECTED_UPDATE_BUDGET_CHANGES", 0)
    preflight["schedule"]["sha256"] = schedule_sha
    base_plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=schedule_sha,
        dataset_references=DATASET_REFERENCES,
    )
    plan = bind_runtime_schedule(
        base_plan, schedule_path, manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    monkeypatch.setattr(production_workflow, "validate_runtime_bound_plan", lambda observed, *_a, **_k: observed)
    monkeypatch.setattr(production_workflow, "_require_canonical_storage_api_factory", lambda _factory: None)
    v62 = _write_v62_producer_report_fixture(tmp_path)
    monkeypatch.setattr(production_workflow, "load_v62_recovery_checkpoint_bound", lambda _path: {
        "max_checkpoint_payload_bytes": 180_000_000,
        "provenance": "test-only size fixture", "size_is_already_conservative_bound": True,
    })
    output_root, checkpoint_root, local_root, cache_root = (
        tmp_path / "out", tmp_path / "drive-checkpoints", tmp_path / "local", tmp_path / "cache",
    )
    cache_root.mkdir()
    features = np.zeros((10, 2381), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)
    identity = _identity("ember")
    bundle = DatasetBundle(
        "ember", features, labels, features, labels,
        {"md5": identity["train_md5"], "sha256": identity["train_sha256"], "shape": [10, 2381]},
        {"md5": identity["test_md5"], "sha256": identity["test_sha256"], "shape": [10, 2381]},
    )
    resume_identity = production_workflow._expected_plan_runner_identity(plan, plan["runs"][0])
    assert resume_identity == production_workflow._diagnostic_run_identity(plan["runs"][0], preflight)
    resume_root = run_checkpoint_root(checkpoint_root, plan["runs"][0]) / "recovery"
    save_generation(
        resume_root,
        RunState(identity=resume_identity, cursor=PhaseCursor(task_index=0, phase="gan_critic")),
        resume_identity, kind="recovery",
    )
    snapshot = {
        "drive_available_bytes": 10**12, "local_available_bytes": 10**12,
        "local_existing_dataset_cache_bytes": 0, "current_run_existing_drive_bytes": 0,
        "current_run_existing_local_checkpoint_bytes": 0, "destination_owner_verified": True,
        "write_capability_verified": True, "new_file_owner_verified": True,
        "quota_metadata_verified": True,
        "output_path": str(output_root.resolve()), "checkpoint_path": str(checkpoint_root.resolve()),
    }
    monkeypatch.setattr(production_workflow, "inspect_production_capacity", lambda *_a, **_k: dict(snapshot))
    monkeypatch.setattr(production_workflow, "_preflight_addressed_cache_bytes", lambda *_a, **_k: 0)
    monkeypatch.setattr(production_workflow, "_bundle_cache_bytes", lambda *_a, **_k: 0)
    monkeypatch.setattr(production_workflow, "require_run_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(production_workflow, "require_available_bytes", lambda *_a, **_k: None)
    events = []

    def load_bundle(*_args, **kwargs):
        assert (output_root / ".matrix_execution.lease.json").is_file()
        assert (output_root / "preparation_progress" / plan["plan_sha256"] / f"{plan['runs'][0]['run_key']}_{plan['runs'][0]['config_sha256'][:12]}.json").is_file()
        kwargs["progress_callback"]({"stage": "fixture_loaded", "rows": 10})
        return bundle

    monkeypatch.setattr(production_workflow, "load_validated_dataset_bundle", load_bundle)

    def stop_before_runner(*_args, **kwargs):
        assert (output_root / ".matrix_execution.lease.json").is_file()
        assert kwargs["matrix_lease"].lease._held is True
        return {"status": "SYNTHETIC_TEST_STOP_BEFORE_RUNNER"}

    monkeypatch.setattr(production_workflow, "execute_frozen_plan_record", stop_before_runner)
    result = production_workflow.execute_first_production_run(
        plan, preflight=preflight, schedule_path=schedule_path,
        schedule_manifest_path=manifest_path, v62_recovery_report_path=v62,
        data_roots={"ember": tmp_path, "az_class": tmp_path}, cache_root=cache_root,
        output_root=output_root, checkpoint_root=checkpoint_root,
        local_checkpoint_root=local_root, launch_enabled=True,
        progress_callback=events.append,
    )
    assert result["status"] == "SYNTHETIC_TEST_STOP_BEFORE_RUNNER"
    assert not (output_root / ".matrix_execution.lease.json").exists()
    prep = json.loads((output_root / "preparation_progress" / plan["plan_sha256"] / f"{plan['runs'][0]['run_key']}_{plan['runs'][0]['config_sha256'][:12]}.json").read_text())
    assert prep["stage"] == "dataset_load_complete"
    assert any(event["stage"] == "dataset_fixture_loaded" for event in events)


def test_plan_rejects_protocol_tampering_even_with_a_recomputed_manifest_hash(tmp_path):
    plan = _plan(tmp_path)
    plan["protocol"]["arms"] = ["current_only"]
    digest_payload = dict(plan)
    digest_payload.pop("plan_sha256")
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="scientific protocol"):
        validate_frozen_plan(plan)


def test_plan_integrity_rejects_duplicate_or_mutated_run_records(tmp_path):
    plan = _plan(tmp_path)
    duplicate = json.loads(json.dumps(plan))
    duplicate["runs"][1] = duplicate["runs"][0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_frozen_plan(duplicate)

    changed = json.loads(json.dumps(plan))
    changed["runs"][0]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash|identity|config"):
        validate_frozen_plan(changed)


def test_plan_rejects_wrong_first_run_and_changed_source_identity(tmp_path):
    plan = _plan(tmp_path)
    wrong_first = json.loads(json.dumps(plan))
    wrong_first["runs"][0], wrong_first["runs"][1] = wrong_first["runs"][1], wrong_first["runs"][0]
    wrong_first["plan_sha256"] = ""
    with pytest.raises(ValueError, match="first run"):
        validate_frozen_plan(wrong_first, verify_digest=False)

    with pytest.raises(ValueError, match="source"):
        validate_frozen_plan(plan, expected_source_sha256="c" * 64)


def test_runtime_binding_validates_all_rows_and_attaches_effective_task_budgets(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule, manifest = _write_effective_schedule(tmp_path, preflight)
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    _prepare_package(tmp_path)
    plan = build_frozen_run_plan(
        tmp_path, schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES,
    )
    bound = bind_runtime_schedule(
        plan, schedule, manifest, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )

    first = bound["runs"][0]
    assert first["update_budgets_by_task"] == [588, 177, 189, 201, 213, 225, 237, 249, 258, 270, 282]
    assert first["effective_schedule_sha256"] == schedule_sha
    assert bound["runtime_binding"]["schedule_rows"] == 220
    assert bound["runtime_binding"]["training_reproduction_claimed"] is False
    assert bound["runtime_binding"]["dataset_identities"]["ember"]["train_sha256"] == _identity("ember")["train_sha256"]
    assert plan["runs"][0].get("update_budgets_by_task") is None


def test_runtime_bound_plan_recomputes_exact_schedule_manifest_preflight_binding(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule, manifest = _write_effective_schedule(tmp_path, preflight)
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    project_root = Path(__file__).resolve().parents[1]
    plan = build_frozen_run_plan(project_root, schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES)
    bound = bind_runtime_schedule(
        plan, schedule, manifest, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    checked = validate_runtime_bound_plan(
        bound, schedule, manifest, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    assert checked["plan_sha256"] == bound["plan_sha256"]

    changed = json.loads(json.dumps(bound))
    changed["runs"][0]["update_budgets_by_task"][0] += 1
    production_workflow._refresh_plan_hash(changed)
    with pytest.raises(ValueError, match="runtime plan does not reproduce"):
        validate_runtime_bound_plan(
            changed, schedule, manifest, preflight,
            expected_anchor_corrections=0, expected_budget_changes=0,
        )


def test_runtime_plan_hash_is_stable_across_mount_alias_paths(tmp_path, monkeypatch):
    preflight = _preflight()
    first_dir = tmp_path / "drive-alias-a"
    second_dir = tmp_path / "drive-alias-b"
    first_dir.mkdir()
    second_dir.mkdir()
    schedule_a, manifest_a = _write_effective_schedule(first_dir, preflight)
    schedule_b, manifest_b = _write_effective_schedule(second_dir, preflight)
    schedule_sha = hashlib.sha256(schedule_a.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    base = build_frozen_run_plan(
        Path(__file__).resolve().parents[1],
        schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES,
    )

    bound_a = bind_runtime_schedule(
        base, schedule_a, manifest_a, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    bound_b = bind_runtime_schedule(
        base, schedule_b, manifest_b, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )

    assert bound_a["plan_sha256"] == bound_b["plan_sha256"]
    assert bound_a["runtime_binding"] == bound_b["runtime_binding"]


def test_runtime_binding_rejects_wrong_schedule_hash_and_dataset_identity(tmp_path, monkeypatch):
    preflight = _preflight()
    plan = _plan(tmp_path)
    schedule, manifest = _write_effective_schedule(tmp_path, preflight)
    preflight["schedule"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="schedule SHA-256"):
        bind_runtime_schedule(plan, schedule, manifest, preflight)

    preflight = _preflight()
    preflight["ember"]["train"]["md5"] = "0" * 32
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    _prepare_package(tmp_path)
    plan = build_frozen_run_plan(
        tmp_path, schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES,
    )
    with pytest.raises(ValueError, match="dataset|MD5"):
        bind_runtime_schedule(plan, schedule, manifest, preflight)


def test_runtime_binding_rejects_unexpected_training_reconciliation_counts(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule, manifest = _write_effective_schedule(tmp_path, preflight)
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    _prepare_package(tmp_path)
    plan = build_frozen_run_plan(
        tmp_path, schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES,
    )

    with pytest.raises(ValueError, match="47-anchor/5-budget"):
        bind_runtime_schedule(plan, schedule, manifest, preflight)


class _FakeRequest:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _FakeMediaRequest(_FakeRequest):
    """Drive-media test request exposing raw HTTP bytes to the real downloader."""

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.uri = "https://www.googleapis.com/drive/v3/files/test-file?alt=media"
        self.headers = {}

        class _Http:
            def request(_self, uri, method, body=None, headers=None, **_kwargs):
                import httplib2

                assert method == "GET"
                response = httplib2.Response({
                    "status": "200", "content-type": "application/json",
                    "content-length": str(len(payload)),
                })
                return response, payload

        self.http = _Http()


class _FakeFiles:
    def __init__(self, records, deleted_ids):
        self.records = records
        self.deleted_ids = deleted_ids
        self.queries = []
        self.list_fields = []

    def list(self, *, q, **_kwargs):
        self.queries.append(q)
        self.list_fields.append(_kwargs.get("fields", ""))
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name_match = re.search(r"name='((?:\\'|[^'])*)'", q)
        name = name_match.group(1).replace("\\'", "'") if name_match else None
        values = [
            record for record in self.records.values()
            if parent in record.get("parents", [])
            and (name is None or record["name"] == name)
            and not record.get("trashed", False)
        ]
        return _FakeRequest({"files": values})

    def get(self, *, fileId, alt=None, **_kwargs):
        if alt == "media":
            record = self.records[fileId]
            path = getattr(self, "local_paths", {}).get(fileId)
            if path is None or not path.is_file():
                raise FileNotFoundError(record.get("name", fileId))
            return _FakeMediaRequest(path.read_bytes())
        path = getattr(self, "local_paths", {}).get(fileId)
        if path is not None and path.is_file():
            self.records[fileId]["size"] = str(path.stat().st_size)
            self.records[fileId]["quotaBytesUsed"] = str(path.stat().st_size)
        return _FakeRequest(self.records[fileId])

    def get_media(self, *, fileId, **_kwargs):
        # Mirrors googleapiclient's generated media method used for blob bytes.
        return self.get(fileId=fileId, alt="media")

    def delete(self, *, fileId, **_kwargs):
        self.deleted_ids.add(fileId)
        self.records.pop(fileId, None)
        return _FakeRequest({})


class _FakeDriveService:
    def __init__(self, records, about):
        self.records = records
        self.deleted_ids = set()
        self._files = _FakeFiles(records, self.deleted_ids)
        self._about = about

    def files(self):
        return self._files

    def about(self):
        return self

    def get(self, **_kwargs):
        return _FakeRequest(self._about)


def _drive_service(
    *, target_owner="researcher@example.org", authenticated_user="researcher@example.org",
    shortcut=False, duplicate=False, shared_drive=None, quota=True,
    can_add_children=True, quota_limit="10000000000", quota_usage="1000000000",
):
    owner = [{"emailAddress": target_owner, "displayName": "Researcher"}] if target_owner else []
    root_record = {
        "id": "revision-folder", "name": "ReVision", "mimeType": "application/vnd.google-apps.folder",
        "parents": ["root"], "owners": owner, "capabilities": {"canAddChildren": can_add_children},
    }
    if shared_drive:
        root_record["driveId"] = shared_drive
        root_record["owners"] = []
    if shortcut:
        root_record = {
            "id": "revision-shortcut", "name": "ReVision", "mimeType": "application/vnd.google-apps.shortcut",
            "parents": ["root"],
            "shortcutDetails": {"targetId": "revision-folder", "targetMimeType": "application/vnd.google-apps.folder"},
        }
    records = {
        root_record["id"]: root_record,
        "revision-folder": {
            "id": "revision-folder", "name": "ReVision", "mimeType": "application/vnd.google-apps.folder",
            "parents": ["root"], "owners": owner, "capabilities": {"canAddChildren": can_add_children},
            **({"driveId": shared_drive} if shared_drive else {}),
        },
        "results-folder": {
            "id": "results-folder", "name": "results", "mimeType": "application/vnd.google-apps.folder",
            "parents": ["revision-folder"], "owners": owner, "capabilities": {"canAddChildren": can_add_children},
            **({"driveId": shared_drive} if shared_drive else {}),
        },
    }
    if shortcut:
        records["revision-folder"]["parents"] = ["unrelated-parent"]
    if duplicate:
        records["duplicate-revision"] = {
            **records["revision-folder"], "id": "duplicate-revision",
        }
    about = {"user": {"emailAddress": authenticated_user}}
    if quota:
        about["storageQuota"] = {
            "limit": quota_limit, "usage": quota_usage,
            "usageInDrive": "900000000", "usageInDriveTrash": "10000000",
        }
    return _FakeDriveService(records, about)


def _verified_probe_metadata(service, parent_id, name):
    """Mock only Drive's remote metadata response; probe file I/O stays real."""
    probe_id = f"probe-file:{name}"
    if probe_id in service.deleted_ids:
        return None
    authenticated_user = service._about["user"]["emailAddress"]
    return {
        "id": probe_id, "name": name, "mimeType": "application/octet-stream",
        "parents": [parent_id], "owners": [{"emailAddress": authenticated_user}],
        "ownedByMe": True, "quotaBytesUsed": "64", "size": "64",
    }


def _bind_main_storage_for_delete_test(monkeypatch, service):
    monkeypatch.setattr(production_workflow, "CANONICAL_REVISION_FOLDER_ID", "revision-folder")
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", service, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": "kian.anb777@gmail.com",
        "canonical_revision_folder_id": "revision-folder",
    }, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, components: {
        "folder_id": "revision-folder" if tuple(components) == ("ReVision",) else "recovery-folder",
        "path_exists": True, "can_add_children": True,
    })


def test_drive_path_resolution_checks_owner_and_follows_shortcut_targets():
    owned = resolve_drive_folder(_drive_service(), ("ReVision", "results"))
    assert owned["folder_id"] == "results-folder"
    assert owned["owner_email"] == "researcher@example.org"
    assert owned["drive_id"] is None
    assert owned["can_add_children"] is True

    via_shortcut = resolve_drive_folder(_drive_service(shortcut=True), ("ReVision", "results"))
    assert via_shortcut["folder_id"] == "results-folder"
    assert via_shortcut["traversed_shortcut"] is True

    future = resolve_drive_folder(_drive_service(), ("ReVision", "new-production-root", "checkpoints"))
    assert future["path_exists"] is False
    assert future["parent_id"] == "revision-folder"
    assert future["unresolved_components"] == ["new-production-root", "checkpoints"]


def test_canonical_storage_session_binds_main_account_exact_folder_and_mount_probe(tmp_path, monkeypatch):
    service = _drive_service(
        target_owner="kian.anb777@gmail.com",
        authenticated_user="kian.anb777@gmail.com",
    )
    drive_root = tmp_path / "MyDrive"
    revision_path = drive_root / "ReVision"
    revision_path.mkdir(parents=True)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", _verified_probe_metadata)
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, components: {
        "path_exists": True,
        "folder_id": production_workflow.CANONICAL_REVISION_FOLDER_ID,
        "owner_email": "kian.anb777@gmail.com", "can_add_children": True,
    })
    monkeypatch.setattr(production_workflow, "CANONICAL_REVISION_FOLDER_ID", "revision-folder")

    evidence = production_workflow.verify_canonical_storage_session(
        api_factory=lambda: service,
        drive_mount_root=drive_root,
        expected_account_email="kian.anb777@gmail.com",
        expected_revision_folder_id="revision-folder",
    )

    assert evidence["status"] == "PASS"
    assert evidence["storage_account_email"] == "kian.anb777@gmail.com"
    assert evidence["canonical_revision_folder_id"] == "revision-folder"
    assert evidence["probe"]["probe_parent_id"] == "revision-folder"
    assert evidence["probe"]["new_file_owner"] == "kian.anb777@gmail.com"
    assert evidence["probe"]["permanent_recovery_delete_verified"] is True
    assert production_workflow._BOUND_DRIVE_API_SERVICE is service
    assert list(revision_path.iterdir()) == []


def test_canonical_storage_account_survives_compute_account_switch_with_same_resume_identity(tmp_path, monkeypatch):
    """Runtime billing identity is not part of the persistent run/checkpoint identity."""
    from revision_r1.checkpointing import save_generation, load_latest_valid
    from revision_r1.state import PhaseCursor, RunState

    storage_service = _drive_service(
        target_owner="kian.anb777@gmail.com",
        authenticated_user="kian.anb777@gmail.com",
    )
    drive_root = tmp_path / "MyDrive"
    (drive_root / "ReVision").mkdir(parents=True)
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", _verified_probe_metadata)
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, _components: {
        "path_exists": True,
        "folder_id": production_workflow.CANONICAL_REVISION_FOLDER_ID,
        "owner_email": "kian.anb777@gmail.com", "can_add_children": True,
    })
    identity = {"run_id": "existing-run", "training_source_sha256": "a" * 64,
                "schedule_sha256": SCHEDULE_SHA, "config_sha256": "b" * 64}
    local_recovery = tmp_path / "local" / "recovery"
    save_generation(
        local_recovery,
        RunState(identity=identity, cursor=PhaseCursor(task_index=4, phase="classifier")),
        identity, kind="recovery",
    )
    generation_before = (local_recovery / "generation-000001.manifest.json").read_bytes()

    for compute_account in ("compute-one@example.org", "compute-two@example.org"):
        evidence = production_workflow.verify_canonical_storage_session(
            api_factory=lambda: storage_service,
            drive_mount_root=drive_root,
            expected_account_email="kian.anb777@gmail.com",
            expected_revision_folder_id=production_workflow.CANONICAL_REVISION_FOLDER_ID,
        )
        production_workflow._require_canonical_storage_api_factory(lambda: storage_service)
        loaded = load_latest_valid(local_recovery, identity)
        assert evidence["storage_account_email"] == "kian.anb777@gmail.com"
        assert compute_account != evidence["storage_account_email"]
        assert loaded.state.cursor.task_index == 4
        assert loaded.state.identity == identity
        assert (local_recovery / "generation-000001.manifest.json").read_bytes() == generation_before


def test_checkpoint_publication_uses_real_serialization_verifies_shortcut_owner_and_prunes_after_api_readback(
    tmp_path, monkeypatch,
):
    from revision_r1.checkpointing import mirror_checkpoint_generations, save_generation
    from revision_r1.state import PhaseCursor, RunState

    storage_service = _drive_service(
        target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com",
        quota_limit="100000000000", quota_usage="1000000",
    )
    drive_root = tmp_path / "MyDrive"
    actual_revision = tmp_path / "shortcut-target" / "ReVision"
    actual_revision.mkdir(parents=True)
    (drive_root).mkdir()
    (drive_root / "ReVision").symlink_to(actual_revision, target_is_directory=True)
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    destination = checkpoint_root / "run" / "recovery"
    destination.mkdir(parents=True)
    source = tmp_path / "local" / "recovery"
    identity = {"run_id": "checkpoint-run", "training_source_sha256": "a" * 64,
                "schedule_sha256": SCHEDULE_SHA, "config_sha256": "b" * 64}
    for task in range(3):
        save_generation(
            source,
            RunState(identity=identity, cursor=PhaseCursor(task_index=task, phase="classifier")),
            identity, kind="recovery", retain_generations=None,
        )
    initial_first = mirror_checkpoint_generations(
        source, destination, identity, keep=2, prune_destination=False, generation=1,
    )
    initial_second = mirror_checkpoint_generations(
        source, destination, identity, keep=2, prune_destination=False, generation=2,
    )
    assert initial_first and initial_second
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", storage_service, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": "kian.anb777@gmail.com",
        "canonical_revision_folder_id": "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x",
    }, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, components: {
        "path_exists": True,
        "folder_id": production_workflow.CANONICAL_REVISION_FOLDER_ID if tuple(components) == ("ReVision",) else "recovery-folder",
        "parent_id": "parent-folder", "owner_email": "kian.anb777@gmail.com",
        "can_add_children": True, "traversed_shortcut": True,
    })

    api_events = []

    def api_metadata(_service, parent_id, name):
        path = destination / name
        if not path.is_file():
            return None
        payload = path.read_bytes()
        record = {
            "id": f"drive-{name}", "name": name, "mimeType": "application/octet-stream",
            "parents": [parent_id], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
            "ownedByMe": True, "size": str(len(payload)), "quotaBytesUsed": str(len(payload)),
            "md5Checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "capabilities": {"canDelete": True},
        }
        api_events.append((name, record["owners"][0]["emailAddress"], record["id"]))
        return record

    monkeypatch.setattr(production_workflow, "_lookup_drive_child", api_metadata)
    deletion_events = []

    def exact_delete(path):
        assert any(name == "generation-000003.pt" for name, _owner, _id in api_events)
        assert any(name == "generation-000003.manifest.json" for name, _owner, _id in api_events)
        deletion_events.append(path.name)
        path.unlink()
        return True

    result = production_workflow.publish_verified_drive_checkpoint_generation(
        mirror_checkpoint_generations, source, destination, identity,
        drive_checkpoint_root=checkpoint_root, drive_mount_root=drive_root,
        api_factory=lambda: storage_service, generation=3, keep=2,
        delete_callback=exact_delete, preverified_manifests=(),
    )

    assert result
    assert all(owner == "kian.anb777@gmail.com" for _name, owner, _id in api_events)
    assert {name for name, _owner, _id in api_events} >= {
        "generation-000003.pt", "generation-000003.manifest.json",
    }
    assert deletion_events == ["generation-000001.pt", "generation-000001.manifest.json"]
    assert {int(path.name[11:17]) for path in destination.glob("generation-*.manifest.json")} == {2, 3}


def test_checkpoint_publication_retries_partial_drive_copy_and_idempotent_replay(tmp_path, monkeypatch):
    import errno
    import shutil
    from revision_r1 import checkpointing
    from revision_r1.state import PhaseCursor, RunState

    storage_service = _drive_service(
        target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com",
        quota_limit="100000000000", quota_usage="1000000",
    )
    drive_root = tmp_path / "MyDrive"
    destination = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints" / "run" / "recovery"
    destination.mkdir(parents=True)
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    source = tmp_path / "local" / "recovery"
    identity = {"run_id": "retry-run", "training_source_sha256": "a" * 64,
                "schedule_sha256": SCHEDULE_SHA, "config_sha256": "b" * 64}
    from revision_r1.checkpointing import save_generation, mirror_checkpoint_generations
    manifest = save_generation(
        source, RunState(identity=identity, cursor=PhaseCursor(task_index=1, phase="classifier")),
        identity, kind="recovery",
    )
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", storage_service, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": "kian.anb777@gmail.com",
        "canonical_revision_folder_id": "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x",
    }, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, components: {
        "path_exists": True,
        "folder_id": production_workflow.CANONICAL_REVISION_FOLDER_ID if tuple(components) == ("ReVision",) else "recovery-folder",
        "parent_id": "parent-folder", "owner_email": "kian.anb777@gmail.com",
        "can_add_children": True,
    })
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _s, parent, name: (
        {"id": f"drive-{name}", "name": name, "mimeType": "application/octet-stream",
         "parents": [parent], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
         "ownedByMe": True, "size": str((destination / name).stat().st_size),
         "quotaBytesUsed": str((destination / name).stat().st_size),
         "md5Checksum": hashlib.md5((destination / name).read_bytes(), usedforsecurity=False).hexdigest()}
        if (destination / name).is_file() else None
    ))
    real_copyfileobj = shutil.copyfileobj
    copies = {"count": 0}

    def interrupted_copy(source_stream, destination_stream, length=0):
        copies["count"] += 1
        if copies["count"] == 1:
            destination_stream.write(source_stream.read(128))
            raise OSError(errno.EIO, "simulated interrupted mounted Drive write")
        return real_copyfileobj(source_stream, destination_stream, length)

    monkeypatch.setattr(checkpointing.shutil, "copyfileobj", interrupted_copy)
    published = production_workflow.publish_verified_drive_checkpoint_generation(
        mirror_checkpoint_generations, source, destination, identity,
        drive_checkpoint_root=checkpoint_root, drive_mount_root=drive_root,
        api_factory=lambda: storage_service, generation=int(manifest["generation"]),
        keep=2, preverified_manifests=(manifest,), delete_callback=lambda _path: False,
    )
    assert published
    assert copies["count"] == 2
    assert (destination / manifest["payload_file"]).stat().st_size == manifest["payload_bytes"]
    repeated = production_workflow.publish_verified_drive_checkpoint_generation(
        mirror_checkpoint_generations, source, destination, identity,
        drive_checkpoint_root=checkpoint_root, drive_mount_root=drive_root,
        api_factory=lambda: storage_service, generation=int(manifest["generation"]),
        keep=2, preverified_manifests=(manifest,), delete_callback=lambda _path: False,
    )
    assert not repeated
    assert copies["count"] == 2


def test_checkpoint_publication_capacity_rejects_before_serialized_bytes_are_copied(tmp_path, monkeypatch):
    from revision_r1.checkpointing import save_generation, mirror_checkpoint_generations
    from revision_r1.state import PhaseCursor, RunState

    low_quota_service = _drive_service(
        target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com",
        quota_limit="1000000000", quota_usage="999999999",
    )
    drive_root = tmp_path / "MyDrive"
    destination = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints" / "run" / "recovery"
    destination.mkdir(parents=True)
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    source = tmp_path / "local" / "recovery"
    identity = {"run_id": "low-quota-run", "training_source_sha256": "a" * 64,
                "schedule_sha256": SCHEDULE_SHA, "config_sha256": "b" * 64}
    manifest = save_generation(
        source, RunState(identity=identity, cursor=PhaseCursor(task_index=1, phase="classifier")),
        identity, kind="recovery",
    )
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", low_quota_service, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", {
        "status": "PASS", "storage_account_email": "kian.anb777@gmail.com",
        "canonical_revision_folder_id": "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x",
    }, raising=False)
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, components: {
        "path_exists": True,
        "folder_id": production_workflow.CANONICAL_REVISION_FOLDER_ID if tuple(components) == ("ReVision",) else "recovery-folder",
        "parent_id": "parent-folder", "owner_email": "kian.anb777@gmail.com",
        "can_add_children": True,
    })
    attempted = []

    def should_not_mirror(*_args, **_kwargs):
        attempted.append(True)
        return mirror_checkpoint_generations(*_args, **_kwargs)

    with pytest.raises(StorageCapacityError, match="Drive capacity gate stopped before checkpoint publication"):
        production_workflow.publish_verified_drive_checkpoint_generation(
            should_not_mirror, source, destination, identity,
            drive_checkpoint_root=checkpoint_root, drive_mount_root=drive_root,
            api_factory=lambda: low_quota_service, generation=int(manifest["generation"]),
            keep=2, preverified_manifests=(manifest,), delete_callback=lambda _path: False,
        )
    assert not attempted
    assert not list(destination.glob("generation-*.pt"))


def test_checkpoint_publication_retry_classifies_only_transient_errors():
    import errno

    transient = StorageCapacityError("temporary DriveFS error")
    transient.__cause__ = OSError(errno.EIO, "transient DriveFS I/O")
    class HttpErrorLike(Exception):
        def __init__(self, status):
            self.resp = type("Response", (), {"status": status})()

    assert production_workflow._retryable_checkpoint_publication_error(transient)
    assert production_workflow._retryable_checkpoint_publication_error(HttpErrorLike(429))
    assert production_workflow._retryable_checkpoint_publication_error(HttpErrorLike(503))
    assert not production_workflow._retryable_checkpoint_publication_error(HttpErrorLike(403))
    assert not production_workflow._retryable_checkpoint_publication_error(StorageCapacityError("wrong identity"))


@pytest.mark.parametrize(
    "authenticated_user,expected_folder_id,match",
    [
        ("other@example.org", "revision-folder", "storage account mismatch"),
        ("kian.anb777@gmail.com", "wrong-folder", "canonical ReVision folder ID mismatch"),
    ],
)
def test_canonical_storage_session_rejects_wrong_account_or_folder_before_probe(
    tmp_path, monkeypatch, authenticated_user, expected_folder_id, match,
):
    service = _drive_service(
        target_owner="kian.anb777@gmail.com", authenticated_user=authenticated_user,
    )
    drive_root = tmp_path / "MyDrive"
    (drive_root / "ReVision").mkdir(parents=True)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", _verified_probe_metadata)
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None, raising=False)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None, raising=False)

    with pytest.raises(StorageCapacityError, match=match):
        production_workflow.verify_canonical_storage_session(
            api_factory=lambda: service,
            drive_mount_root=drive_root,
            expected_account_email="kian.anb777@gmail.com",
            expected_revision_folder_id=expected_folder_id,
        )

    assert list((drive_root / "ReVision").iterdir()) == []
    assert production_workflow._BOUND_DRIVE_API_SERVICE is None


def test_matrix_capacity_gate_requires_only_next_run_incremental_storage_and_reserve():
    plan = project_storage(_production_source_plan(), observed_checkpoint_bytes=1_000_000)
    snapshot = {
        "drive_available_bytes": plan["drive"]["first_run_required_without_reserve_bytes"]
        + DEFAULT_STORAGE_RESERVE_BYTES,
        "local_available_bytes": plan["local"]["next_run_required_without_reserve_bytes"]
        + DEFAULT_STORAGE_RESERVE_BYTES,
        "local_existing_dataset_cache_bytes": plan["local"]["dataset_cache_upper_bytes"],
        "production_measurements_validated": True,
    }

    require_run_capacity(snapshot, plan, scope="matrix", completed_runs=1)

    snapshot["drive_available_bytes"] -= 1
    with pytest.raises(StorageCapacityError, match="matrix.*required"):
        require_run_capacity(snapshot, plan, scope="matrix", completed_runs=1)


def test_dataset_bundle_reuses_verified_local_cache_when_drive_npz_is_missing(tmp_path, monkeypatch):
    import revision_r1.data as data_module

    rng = np.random.default_rng(23)
    labels = np.arange(100, dtype=np.int64)
    train_x = rng.normal(size=(100, 4)).astype(np.float32)
    test_x = rng.normal(size=(100, 4)).astype(np.float32)
    data_root = tmp_path / "Drive" / "datasets"
    data_root.mkdir(parents=True)
    train_path, test_path = data_root / "train.npz", data_root / "test.npz"
    np.savez(train_path, X_train=train_x, y_train=labels)
    np.savez(test_path, X_test=test_x, y_test=labels)
    mini_spec = {
        "ember": {
            "train": {"filename": "train.npz", "md5": data_module.file_digest(train_path, "md5"), "shape": [100, 4]},
            "test": {"filename": "test.npz", "md5": data_module.file_digest(test_path, "md5"), "shape": [100, 4]},
            "feature_dim": 4,
            "anchor_k": 1,
        }
    }
    monkeypatch.setattr(production_workflow, "DATASETS", mini_spec)
    monkeypatch.setattr(data_module, "DATASETS", mini_spec)
    cache_root = tmp_path / "local-cache"
    from revision_r1.data import preflight_dataset

    cached_bundle = preflight_dataset(data_root, "ember", cache_dir=cache_root)
    preflight = {
        "ember": {
            "train": {"md5": cached_bundle.train_metadata["md5"], "sha256": cached_bundle.train_metadata["sha256"]},
            "test": {"md5": cached_bundle.test_metadata["md5"], "sha256": cached_bundle.test_metadata["sha256"]},
        }
    }
    train_path.unlink()
    test_path.unlink()

    actual = production_workflow.load_validated_dataset_bundle(data_root, "ember", cache_root, preflight)

    assert isinstance(actual.x_train, np.memmap)
    assert isinstance(actual.y_test, np.memmap)
    np.testing.assert_array_equal(actual.x_train, train_x)
    np.testing.assert_array_equal(actual.x_test, test_x)
    assert actual.train_metadata["sha256"] == preflight["ember"]["train"]["sha256"]
    assert actual.test_metadata["sha256"] == preflight["ember"]["test"]["sha256"]
    cache_paths = data_module._cache_paths(
        cache_root.resolve(), actual.train_metadata["md5"],
        actual.train_metadata["x_key"], actual.train_metadata["y_key"],
    )
    cache_paths[0].write_bytes(cache_paths[0].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        production_workflow.load_validated_dataset_bundle(data_root, "ember", cache_root, preflight)


def test_dataset_source_inventory_uses_drive_npz_and_never_downloads(tmp_path):
    preflight = _preflight()
    roots = {name: tmp_path / name for name in production_workflow.DATASET_ORDER}
    for dataset, root in roots.items():
        root.mkdir()
        spec = production_workflow.DATASETS[dataset]
        for split in ("train", "test"):
            (root / spec[split]["filename"]).write_bytes(b"fixture presence only")

    report = production_workflow.inspect_dataset_sources(roots, tmp_path / "cache", preflight)

    assert report["ember"]["splits"]["train"]["source"] == "CANONICAL_DRIVE_NPZ"
    assert report["az_class"]["splits"]["test"]["sha256_verification"] == "performed_before_dataset_use"
    assert all(row["external_downloads"] == "DISABLED" for row in report.values())


def test_dataset_source_inventory_fails_if_drive_and_verified_cache_are_both_missing(tmp_path):
    preflight = _preflight()
    roots = {name: tmp_path / name for name in production_workflow.DATASET_ORDER}
    for root in roots.values():
        root.mkdir()

    with pytest.raises(FileNotFoundError, match="no external download will be attempted"):
        production_workflow.inspect_dataset_sources(roots, tmp_path / "cache", preflight)


def test_completed_first_run_is_validated_and_reused_without_remeasurement_or_reregistration(tmp_path, monkeypatch):
    plan = _production_source_plan()
    first = plan["runs"][0]
    artifact_sha = "a" * 64
    measurement = {
        "status": "MEASURED_VALIDATED_PRODUCTION_RUN",
        "run_key": first["run_key"],
        "run_id": production_workflow._run_id_for(first, synthetic=False),
        "max_checkpoint_payload_bytes": 100,
        "run_output_bytes": 200,
    }
    entry = {
        "run_key": first["run_key"], "run_index": first["run_index"],
        "config_sha256": first["config_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "class_order_sha256": first["class_order_sha256"],
        "dataset_identity": first.get("dataset_identity"),
        "run_id": measurement["run_id"], "artifact_manifest_sha256": artifact_sha,
        "storage_measurement": measurement,
    }
    monkeypatch.setattr(
        production_workflow, "_read_matrix_registry",
        lambda *_args: {"completed_runs": {first["run_key"]: entry}},
    )
    monkeypatch.setattr(
        production_workflow, "_validate_completed_run",
        lambda *_args, **_kwargs: {"run_id": measurement["run_id"], "artifact_manifest_sha256": artifact_sha},
    )
    monkeypatch.setattr(
        production_workflow, "measure_completed_production_run",
        lambda *_args, **_kwargs: pytest.fail("completed run must not be measured/finalized again"),
    )
    monkeypatch.setattr(
        production_workflow, "_record_completed_run",
        lambda *_args, **_kwargs: pytest.fail("completed run must not be registered again"),
    )

    reused = production_workflow.reuse_validated_first_run_measurement(
        plan, first, tmp_path / "run1-output",
    )

    assert reused == measurement


def test_execute_first_run_skips_registered_finalized_run_before_reference_or_runner(tmp_path, monkeypatch):
    plan = _production_source_plan()
    first = plan["runs"][0]
    measurement = {
        "status": "MEASURED_VALIDATED_PRODUCTION_RUN", "run_key": first["run_key"],
        "run_id": "existing-run-id", "max_checkpoint_payload_bytes": 1_000,
        "run_output_bytes": 2_000,
    }
    monkeypatch.setattr(production_workflow, "_require_canonical_storage_api_factory", lambda _factory: None)
    monkeypatch.setattr(production_workflow, "validate_runtime_bound_plan", lambda value, *_a, **_k: value)
    monkeypatch.setattr(
        production_workflow, "_read_matrix_registry",
        lambda *_args: {"completed_runs": {first["run_key"]: {"already": "registered"}}},
    )
    monkeypatch.setattr(production_workflow, "reuse_validated_first_run_measurement", lambda *_a: measurement)
    monkeypatch.setattr(
        production_workflow, "load_v62_recovery_checkpoint_bound",
        lambda *_a, **_k: pytest.fail("finalized run must not reload the old checkpoint reference"),
    )
    monkeypatch.setattr(
        production_workflow, "measure_completed_production_run",
        lambda *_a, **_k: pytest.fail("finalized run must not be remeasured"),
    )
    monkeypatch.setattr(
        production_workflow, "_record_completed_run",
        lambda *_a, **_k: pytest.fail("finalized run must not be registered again"),
    )

    result = execute_first_production_run(
        plan, preflight={}, schedule_path="unused", schedule_manifest_path="unused",
        v62_recovery_report_path="unused", data_roots={}, cache_root=tmp_path / "cache",
        output_root=tmp_path / "results", checkpoint_root=tmp_path / "checkpoints",
        local_checkpoint_root=tmp_path / "local", launch_enabled=True,
    )

    assert result["status"] == "ALREADY_FINALIZED_COMPLETED_RUN"
    assert result["run_key"] == "ember_seed42_full_hybrid"
    assert result["storage_measurement"] is measurement


def test_drive_folder_resolution_emits_explicit_mime_type_comparisons_and_escaped_name():
    service = _FakeDriveService({}, {"user": {"emailAddress": "researcher@example.org"}})

    result = resolve_drive_folder(service, ("A'B",))

    assert result["path_exists"] is False
    assert service.files().queries == [
        r"'root' in parents and name='A\'B' and trashed=false and "
        r"(mimeType = 'application/vnd.google-apps.folder' or "
        r"mimeType = 'application/vnd.google-apps.shortcut')"
    ]


def test_drive_path_components_follow_logical_mydrive_path_without_resolving_shortcut(tmp_path):
    drive_root = tmp_path / "MyDrive"
    target = tmp_path / "shortcut-target" / "ReVision"
    drive_root.mkdir()
    target.mkdir(parents=True)
    (drive_root / "ReVision").symlink_to(target, target_is_directory=True)

    configured, components = _logical_drive_path_components(
        drive_root / "ReVision" / "revision_R1" / "production" / "results",
        drive_root=drive_root,
    )

    assert components == ("ReVision", "revision_R1", "production", "results")
    assert configured.startswith(str(drive_root))
    with pytest.raises(StorageCapacityError, match="must be configured below"):
        _logical_drive_path_components(tmp_path / "elsewhere", drive_root=drive_root)


def test_drive_destination_fails_closed_for_ambiguous_and_shared_drive():
    with pytest.raises(StorageCapacityError, match="ambiguous"):
        resolve_drive_folder(_drive_service(duplicate=True), ("ReVision",))
    with pytest.raises(StorageCapacityError, match="shared drive"):
        inspect_drive_destination(
            "/content/drive/MyDrive/ReVision/results",
            api_factory=lambda: _drive_service(shared_drive="org-drive"),
        )
def test_drive_destination_verifies_owner_and_shared_editor_by_new_file_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)

    owner = inspect_drive_destination(
        destination_path,
        drive_root=drive_root,
        api_factory=lambda: _drive_service(target_owner="owner@example.org", authenticated_user="owner@example.org"),
        probe_metadata_lookup=_verified_probe_metadata,
    )
    shared_editor = inspect_drive_destination(
        destination_path,
        drive_root=drive_root,
        api_factory=lambda: _drive_service(target_owner="owner@example.org", authenticated_user="editor@example.org"),
        probe_metadata_lookup=_verified_probe_metadata,
    )
    owner_cached = inspect_drive_destination(
        destination_path,
        drive_root=drive_root,
        api_factory=lambda: _drive_service(target_owner="owner@example.org", authenticated_user="owner@example.org"),
        probe_metadata_lookup=lambda *_args: pytest.fail("same-runtime probe should be reused for the same account and folder"),
    )

    assert owner["authenticated_user"] == "owner@example.org"
    assert owner["destination_owner"] == "owner@example.org"
    assert owner["new_file_owner"] == "owner@example.org"
    assert owner["write_capability_verified"] is True
    assert owner["permanent_recovery_delete_verified"] is True
    assert owner["available_bytes"] == 9_000_000_000
    assert owner["usage_in_drive_trash_bytes"] == 10_000_000
    assert owner["folder_id"] == "results-folder"
    assert shared_editor["authenticated_user"] == "editor@example.org"
    assert shared_editor["destination_owner"] == "owner@example.org"
    assert shared_editor["destination_owned_by_authenticated_user"] is False
    assert shared_editor["new_file_owner"] == "editor@example.org"
    assert shared_editor["new_file_owned_by_authenticated_user"] is True
    assert shared_editor["permanent_recovery_delete_verified"] is True
    assert shared_editor["quota_account_email"] == "editor@example.org"
    assert shared_editor["available_bytes"] == 9_000_000_000
    assert owner_cached["write_probe"]["probe_performed_this_check"] is False
    assert owner["quota_change_observation"] == "NO_POSITIVE_CHANGE_OBSERVED_OR_API_EVENTUAL_CONSISTENCY"
    assert owner["quota_usage_before_probe_bytes"] == 1_000_000_000
    assert owner["quota_usage_during_probe_bytes"] == 1_000_000_000
    assert owner["quota_usage_after_probe_cleanup_bytes"] == 1_000_000_000
    assert list(destination_path.iterdir()) == []


def test_recovery_pruner_deletes_exact_generation_when_drive_confirms_delete_capability(monkeypatch):
    service = _drive_service(target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com")
    service.records["checkpoint-file"] = {
        "id": "checkpoint-file", "name": "generation-000001.pt",
        "mimeType": "application/octet-stream", "parents": ["recovery-folder"],
        "owners": [{"emailAddress": "kian.anb777@gmail.com"}], "ownedByMe": True,
        "capabilities": {"canDelete": True},
    }
    service.records["recovery-folder"] = {
        "id": "recovery-folder", "name": "recovery", "mimeType": "application/vnd.google-apps.folder",
        "parents": ["checkpoints-folder"], "owners": [{"emailAddress": "folder-owner@example.org"}],
        "capabilities": {"canAddChildren": True},
    }
    _bind_main_storage_for_delete_test(monkeypatch, service)
    monkeypatch.setattr(production_workflow, "_logical_drive_path_components", lambda _path, **_kwargs: ("/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/recovery", ("ReVision", "revision_R1", "production", "checkpoints", "recovery")))
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _service, parent, name: next((row for row in service.records.values() if parent in row.get("parents", []) and row.get("name") == name), None))

    result = production_workflow.permanently_delete_recovery_checkpoint(
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/run/recovery/generation-000001.pt",
        api_factory=lambda: service,
    )

    assert result is True
    assert "checkpoint-file" not in service.records
    assert service.deleted_ids == {"checkpoint-file"}
    with pytest.raises(StorageCapacityError, match="outside ReVision"):
        production_workflow.permanently_delete_recovery_checkpoint(
            "/content/drive/MyDrive/ReVision/revision_R1/production/results/generation-000001.pt",
            api_factory=lambda: service,
        )


def test_recovery_pruner_preserves_mixed_owner_history_even_when_can_delete_is_true(monkeypatch):
    service = _drive_service(target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com")
    service.records["historical-checkpoint"] = {
        "id": "historical-checkpoint", "name": "generation-000001.pt",
        "mimeType": "application/octet-stream", "parents": ["recovery-folder"],
        "owners": [{"emailAddress": "previous-editor@example.org"}], "ownedByMe": False,
        "capabilities": {"canDelete": True},
    }
    _bind_main_storage_for_delete_test(monkeypatch, service)
    monkeypatch.setattr(production_workflow, "_logical_drive_path_components", lambda _path, **_kwargs: (
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/recovery",
        ("ReVision", "revision_R1", "production", "checkpoints", "recovery"),
    ))
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _service, parent, name: next((
        row for row in service.records.values() if parent in row.get("parents", []) and row.get("name") == name
    ), None))

    result = production_workflow.permanently_delete_recovery_checkpoint(
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/run/recovery/generation-000001.pt",
        api_factory=lambda: service,
    )

    assert result is False
    assert "historical-checkpoint" in service.records
    assert service.deleted_ids == set()


def test_recovery_pruner_defers_exact_legacy_file_when_drive_denies_delete(monkeypatch):
    service = _drive_service(target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com")
    service.records["legacy-checkpoint"] = {
        "id": "legacy-checkpoint", "name": "generation-000279.pt",
        "mimeType": "application/x-zip", "parents": ["recovery-folder"],
        "owners": [{"emailAddress": "previous-editor@example.org"}], "ownedByMe": False,
        "capabilities": {"canDelete": False},
    }
    service.records["recovery-folder"] = {
        "id": "recovery-folder", "name": "recovery", "mimeType": "application/vnd.google-apps.folder",
        "parents": ["checkpoints-folder"], "owners": [{"emailAddress": "folder-owner@example.org"}],
        "capabilities": {"canAddChildren": True},
    }
    _bind_main_storage_for_delete_test(monkeypatch, service)
    monkeypatch.setattr(production_workflow, "_logical_drive_path_components", lambda _path, **_kwargs: (
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/recovery",
        ("ReVision", "revision_R1", "production", "checkpoints", "recovery"),
    ))
    monkeypatch.setattr(
        production_workflow, "_lookup_drive_child",
        lambda _service, parent, name: next((
            row for row in service.records.values()
            if parent in row.get("parents", []) and row.get("name") == name
        ), None),
    )

    result = production_workflow.permanently_delete_recovery_checkpoint(
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/run/recovery/generation-000279.pt",
        api_factory=lambda: service,
    )

    assert result is False
    assert "legacy-checkpoint" in service.records
    assert service.deleted_ids == set()


def test_recovery_pruner_rejects_replaced_file_id_before_delete(monkeypatch):
    service = _drive_service(target_owner="kian.anb777@gmail.com", authenticated_user="kian.anb777@gmail.com")
    service.records["replacement-file"] = {
        "id": "replacement-file", "name": "generation-000001.pt",
        "mimeType": "application/octet-stream", "parents": ["recovery-folder"],
        "owners": [{"emailAddress": "editor@example.org"}], "ownedByMe": True,
        "capabilities": {"canDelete": True},
    }
    _bind_main_storage_for_delete_test(monkeypatch, service)
    monkeypatch.setattr(production_workflow, "_logical_drive_path_components", lambda _path, **_kwargs: (
        "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/recovery",
        ("ReVision", "revision_R1", "production", "checkpoints", "recovery"),
    ))
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _service, _parent, _name:
                        service.records["replacement-file"])

    with pytest.raises(StorageCapacityError, match="file ID differs from the retention journal"):
        production_workflow.permanently_delete_recovery_checkpoint(
            "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints/run/recovery/generation-000001.pt",
            api_factory=lambda: service, expected_drive_file_id="original-file",
            expected_parent_id="recovery-folder",
        )
    assert service.deleted_ids == set()


def test_recovery_inventory_requests_drive_delete_and_edit_capabilities():
    service = _drive_service()

    production_workflow._lookup_drive_child(service, "results-folder", "generation-000279.pt")
    production_workflow._list_drive_folder_files(service, "results-folder")

    assert len(service.files().list_fields) == 2
    assert all(
        "capabilities(canDelete,canEdit)" in fields
        for fields in service.files().list_fields
    )


def test_viewer_is_rejected_before_probe_and_unknown_quota_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)
    called = []

    with pytest.raises(StorageCapacityError, match="canAddChildren"):
        inspect_drive_destination(
            destination_path, drive_root=drive_root,
            api_factory=lambda: _drive_service(
                target_owner="owner@example.org", authenticated_user="viewer@example.org",
                can_add_children=False,
            ),
            probe_metadata_lookup=lambda *_args: called.append(True),
        )
    assert called == []

    unknown = inspect_drive_destination(
        destination_path, drive_root=drive_root,
        api_factory=lambda: _drive_service(
            target_owner="owner@example.org", authenticated_user="editor@example.org",
            quota_limit=None,
        ),
        probe_metadata_lookup=_verified_probe_metadata,
    )
    assert unknown["quota_status"] == "UNKNOWN_CAPACITY"
    assert unknown["available_bytes"] is None
    assert "did not report" in unknown["quota_capacity_note"].lower()
    projection = project_storage({"run_count": 100}, observed_checkpoint_bytes=1000)
    with pytest.raises(StorageCapacityError, match="cannot establish Drive capacity"):
        require_run_capacity({
            "drive_available_bytes": unknown["available_bytes"],
            "local_available_bytes": projection["local"]["first_run_peak_with_reserve_bytes"],
            "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
        }, projection, scope="first_run")


def test_drive_probe_retries_exact_parent_lookup_after_mount_write(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    sleeps = []
    monkeypatch.setattr(production_workflow.time, "sleep", sleeps.append)
    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)
    attempts = []

    def delayed_metadata(service, parent_id, name):
        attempts.append((parent_id, name))
        if len(attempts) < 3:
            return None
        return _verified_probe_metadata(service, parent_id, name)

    destination = inspect_drive_destination(
        destination_path,
        drive_root=drive_root,
        api_factory=lambda: _drive_service(
            target_owner="owner@example.org", authenticated_user="editor@example.org",
        ),
        probe_metadata_lookup=delayed_metadata,
    )

    assert destination["write_capability_verified"] is True
    assert len(attempts) == 4  # 3 visibility retries plus deletion read-back.
    assert all(parent == "results-folder" for parent, _name in attempts)
    assert len({name for _parent, name in attempts}) == 1
    assert sleeps == [1.0, 2.0]
    assert list(destination_path.iterdir()) == []


def test_drive_probe_visibility_timeout_reports_target_and_cleans_only_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    sleeps = []
    monkeypatch.setattr(production_workflow.time, "sleep", sleeps.append)
    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)
    (destination_path / "keep.txt").write_text("preserve", encoding="utf-8")
    attempts = []

    def invisible_metadata(_service, parent_id, name):
        attempts.append((parent_id, name))
        return None

    with pytest.raises(StorageCapacityError, match=r"6 attempts.*results-folder") as error:
        inspect_drive_destination(
            destination_path,
            drive_root=drive_root,
            api_factory=lambda: _drive_service(
                target_owner="owner@example.org", authenticated_user="editor@example.org",
            ),
            probe_metadata_lookup=invisible_metadata,
        )

    assert len(attempts) == 6
    assert len({name for _parent, name in attempts}) == 1
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 15.0]
    assert attempts[0][1] in str(error.value)
    assert sorted(path.name for path in destination_path.iterdir()) == ["keep.txt"]


def test_drive_insufficient_quota_stops_with_reserve_preserved():
    projection = project_storage({"run_count": 100}, observed_checkpoint_bytes=1000)
    low_quota_snapshot = {
        "drive_available_bytes": 1,
        "local_available_bytes": projection["local"]["first_run_peak_with_reserve_bytes"],
        "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
    }
    with pytest.raises(StorageCapacityError, match="insufficient Drive storage.*safety reserve"):
        require_run_capacity(low_quota_snapshot, projection, scope="first_run")


def test_account_switch_keeps_existing_checkpoint_resume_identity(tmp_path, monkeypatch):
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState
    from scripts.production_workflow import inspect_first_run_resume_checkpoint

    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    plan = _production_source_plan()
    preflight = _preflight()
    record = plan["runs"][0]
    checkpoint_root = tmp_path / "shared-checkpoints"
    identity = production_workflow._diagnostic_run_identity(record, preflight)
    recovery = run_checkpoint_root(checkpoint_root, record) / "recovery"
    saved = save_generation(
        recovery, RunState(identity=identity, cursor=PhaseCursor(task_index=2, phase="classifier")),
        identity, kind="recovery",
    )
    first_gate = inspect_first_run_resume_checkpoint(plan, preflight=preflight, checkpoint_root=checkpoint_root)

    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)
    owner_snapshot = inspect_drive_destination(
        destination_path, drive_root=drive_root,
        api_factory=lambda: _drive_service(target_owner="owner@example.org", authenticated_user="owner@example.org"),
        probe_metadata_lookup=_verified_probe_metadata,
    )
    editor_snapshot = inspect_drive_destination(
        destination_path, drive_root=drive_root,
        api_factory=lambda: _drive_service(target_owner="owner@example.org", authenticated_user="different-editor@example.org"),
        probe_metadata_lookup=_verified_probe_metadata,
    )
    second_gate = inspect_first_run_resume_checkpoint(plan, preflight=preflight, checkpoint_root=checkpoint_root)

    assert first_gate["status"] == second_gate["status"] == "PASS_RESUME_CANDIDATE"
    assert first_gate["verified_candidates"][0]["generation"] == saved["generation"]
    assert second_gate["verified_candidates"][0]["generation"] == saved["generation"]
    assert identity == production_workflow._diagnostic_run_identity(record, preflight)
    assert run_checkpoint_root(checkpoint_root, record) == recovery.parent
    assert owner_snapshot["quota_account_email"] != editor_snapshot["quota_account_email"]
    assert owner_snapshot["new_file_owner"] == "owner@example.org"
    assert editor_snapshot["new_file_owner"] == "different-editor@example.org"


def test_drive_destination_reports_authenticated_quota_account_and_storage_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    drive_root = tmp_path / "MyDrive"
    destination_path = drive_root / "ReVision" / "results"
    destination_path.mkdir(parents=True)
    destination = inspect_drive_destination(
        destination_path, drive_root=drive_root,
        api_factory=_drive_service,
        probe_metadata_lookup=_verified_probe_metadata,
    )
    assert destination["authenticated_user"] == "researcher@example.org"
    assert destination["destination_owner"] == "researcher@example.org"
    assert destination["quota_account_email"] == "researcher@example.org"
    assert destination["new_file_quota_bytes"] == 64
    assert destination["quota_status"] == "FINITE_VERIFIED"
    assert destination["available_bytes"] == 9_000_000_000
    assert destination["folder_id"] == "results-folder"


def test_storage_projection_accounts_for_all_retained_slots_and_atomic_headroom():
    projection = project_storage(
        {"run_count": 100}, observed_checkpoint_bytes=1000, observed_output_bytes=2000,
        checkpoint_size_is_already_conservative_bound=True,
    )
    per_checkpoint = 1000
    per_run_checkpoints = 13 * per_checkpoint
    output_bound = 256 * 1024 * 1024

    assert projection["checkpoint_layout"]["task_boundary_generations_per_run"] == 11
    assert projection["checkpoint_layout"]["recovery_generations_per_run"] == 2
    assert projection["checkpoint_layout"]["total_retained_generation_slots_per_run"] == 13
    assert projection["checkpoint_payload_upper_bytes"] == per_checkpoint
    assert projection["accounting"]["checkpoint_size_reference_already_conservative_bound"] is True
    assert projection["drive"]["first_run_retained_bytes"] == per_run_checkpoints + output_bound
    assert projection["drive"]["first_run_peak_with_reserve_bytes"] == (
        per_run_checkpoints + output_bound + per_checkpoint + output_bound + DEFAULT_STORAGE_RESERVE_BYTES
    )
    assert projection["drive"]["matrix_retained_bytes"] == 100 * (per_run_checkpoints + output_bound)
    assert projection["checkpoint_write_volume"]["matrix_retained_payload_lower_bound_bytes"] == 100 * per_run_checkpoints
    assert projection["local"]["dataset_cache_upper_bytes"] == 11_997_099_600
    assert projection["local"]["matrix_peak_bytes"] > projection["local"]["next_run_peak_with_reserve_bytes"]
    assert projection["checkpoint_size_provenance"].startswith("v6.2 GPU synthetic recovery reference")


def test_storage_projection_reports_recovery_pruning_but_preserves_boundaries():
    projection = project_storage(
        {"run_count": 100}, observed_checkpoint_bytes=1_080_000_000,
        checkpoint_reference_provenance="observed exact-identity production checkpoint",
        checkpoint_size_evidence={"selected_recovery_generation": 202},
    )

    assert projection["checkpoint_payload_upper_bytes"] == (
        int(np.ceil(1_080_000_000 * 1.5)) + production_workflow.CHECKPOINT_SIZE_FIXED_OVERHEAD_BYTES
    )
    assert projection["checkpoint_size_evidence"]["selected_recovery_generation"] == 202
    assert projection["checkpoint_layout"]["recovery_generations_pruned"] is True
    assert projection["checkpoint_layout"]["task_boundary_payloads_pruned"] is False
    assert projection["accounting"]["task_boundary_archives_pruned"] is False


def test_checkpoint_capacity_reference_uses_actual_matching_checkpoint_sizes(tmp_path):
    identity = {"code_sha256": "a" * 64, "config_sha256": "b" * 64}
    recovery = tmp_path / "recovery"
    boundaries = tmp_path / "task_boundaries"
    recovery.mkdir()
    boundaries.mkdir()
    candidate_payload = recovery / "generation-000042.pt"
    candidate_payload.write_bytes(b"r" * 128)
    candidate_manifest = {
        "schema_version": 1, "generation": 42, "identity": identity,
        "payload_file": candidate_payload.name, "payload_bytes": 128,
        "payload_sha256": hashlib.sha256(candidate_payload.read_bytes()).hexdigest(),
    }
    candidate_manifest_path = recovery / "generation-000042.manifest.json"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")
    boundary_payload = boundaries / "generation-000003.pt"
    boundary_payload.write_bytes(b"t" * 512)
    boundary_manifest_path = boundaries / "generation-000003.manifest.json"
    boundary_manifest_path.write_text(json.dumps({
        "schema_version": 1, "generation": 3, "kind": "task-boundary", "identity": identity,
        "payload_file": boundary_payload.name, "payload_bytes": 512,
        "payload_sha256": hashlib.sha256(boundary_payload.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    result = _checkpoint_payload_size_reference(
        tmp_path, expected_identity=identity,
        selected_recovery_candidate={
            "generation": 42, "manifest_sha256": hashlib.sha256(candidate_manifest_path.read_bytes()).hexdigest(),
            "payload_bytes_expected": 128, "payload_bytes_observed": 128,
            "payload_sha256_expected": candidate_manifest["payload_sha256"],
            "payload_sha256_observed": candidate_manifest["payload_sha256"],
            "payload_hash_matches": True, "manifest_identity_matches": True,
            "integrity_status": "PASS_MANIFEST_AND_PAYLOAD_HASH",
        },
    )

    assert result["max_payload_bytes"] == 512
    assert result["selected_recovery_payload_bytes"] == 128
    assert result["matching_manifest_count"] == 2
    assert result["size_evidence"] == "matching identity manifests plus max(manifest-declared, observed payload sizes); selected recovery candidate hash verified"

    read_only_size = _checkpoint_payload_size_reference(tmp_path, expected_identity=identity)
    assert read_only_size["max_payload_bytes"] == 512
    assert read_only_size["latest_recovery_manifest_generation"] == 42
    assert "payload hashes not checked here" in read_only_size["size_evidence"]


def test_checkpoint_capacity_reference_uses_manifest_size_when_newest_payload_is_truncated(tmp_path):
    identity = {"code_sha256": "a" * 64, "config_sha256": "b" * 64}
    recovery = tmp_path / "recovery"
    recovery.mkdir()

    def write_generation(generation: int, payload_bytes: bytes, declared_bytes: int):
        payload = recovery / f"generation-{generation:06d}.pt"
        payload.write_bytes(payload_bytes)
        manifest = {
            "schema_version": 1, "generation": generation, "kind": "recovery", "identity": identity,
            "payload_file": payload.name, "payload_bytes": declared_bytes,
            "payload_sha256": hashlib.sha256(b"full-payload-placeholder").hexdigest(),
        }
        path = recovery / f"generation-{generation:06d}.manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest, payload

    selected_path, selected_manifest, selected_payload = write_generation(41, b"v" * 64, 64)
    _, _, damaged_newest_payload = write_generation(42, b"short", 128)
    candidate = {
        "generation": 41, "manifest_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "payload_bytes_expected": 64, "payload_bytes_observed": 64,
        "payload_sha256_expected": hashlib.sha256(selected_payload.read_bytes()).hexdigest(),
        "payload_sha256_observed": hashlib.sha256(selected_payload.read_bytes()).hexdigest(),
        "payload_hash_matches": True, "manifest_identity_matches": True,
        "integrity_status": "PASS_MANIFEST_AND_PAYLOAD_HASH",
    }
    selected_manifest["payload_sha256"] = candidate["payload_sha256_expected"]
    selected_path.write_text(json.dumps(selected_manifest), encoding="utf-8")
    candidate["manifest_sha256"] = hashlib.sha256(selected_path.read_bytes()).hexdigest()

    result = _checkpoint_payload_size_reference(
        tmp_path, expected_identity=identity, selected_recovery_candidate=candidate,
    )

    assert result["selected_recovery_generation"] == 41
    assert result["latest_recovery_manifest_generation"] == 42
    assert result["max_payload_bytes"] == 128
    newest = next(item for item in result["observed_generations"] if item["generation"] == 42)
    assert newest["payload_bytes"] == damaged_newest_payload.stat().st_size
    assert newest["manifest_payload_bytes"] == 128
    assert newest["payload_size_matches_manifest"] is False


def test_checkpoint_capacity_reference_fails_closed_on_identity_or_size_mismatch(tmp_path):
    identity = {"code_sha256": "a" * 64}
    folder = tmp_path / "recovery"
    folder.mkdir()
    payload = folder / "generation-000001.pt"
    payload.write_bytes(b"payload")
    manifest_path = folder / "generation-000001.manifest.json"
    manifest = {
        "schema_version": 1, "generation": 1, "identity": identity,
        "payload_file": payload.name, "payload_bytes": 7, "payload_sha256": "c" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = {
        "generation": 1, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "payload_bytes_expected": 7, "payload_bytes_observed": 7,
        "payload_sha256_expected": "c" * 64, "payload_sha256_observed": "c" * 64, "payload_hash_matches": True,
        "manifest_identity_matches": True, "integrity_status": "PASS_MANIFEST_AND_PAYLOAD_HASH",
    }

    manifest["identity"] = {"code_sha256": "d" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageCapacityError, match="identity"):
        _checkpoint_payload_size_reference(tmp_path, expected_identity=identity, selected_recovery_candidate=candidate)

    manifest["identity"] = identity
    manifest["payload_bytes"] = 8
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageCapacityError, match="changed"):
        _checkpoint_payload_size_reference(tmp_path, expected_identity=identity, selected_recovery_candidate=candidate)


def _write_v62_producer_report_fixture(tmp_path: Path) -> Path:
    """Build a report-shaped fixture from the real v6.2 publication verifier."""
    from revision_r1.validation_workflow import verify_drive_checkpoint_publication

    code_hash = "adb4ee76f1d7fb82a35ffaf4f38f5ddaa9b7649c36098f6110f47a938f9faf88"
    manifest_path = tmp_path / "generation-000079.manifest.json"
    payload_path = tmp_path / "generation-000079.pt"
    manifest_path.write_bytes(b'{"generation":79,"kind":"interrupted-resumable"}\n')
    payload_path.write_bytes(b"synthetic checkpoint payload fixture")
    generation = 79
    worker_publication = {
        "status": "PERSISTED_AND_READ_BACK_VERIFIED",
        "published_generation_count": 1,
        "generation": generation,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "payload_bytes": payload_path.stat().st_size,
    }
    publication_evidence = verify_drive_checkpoint_publication(
        worker_publication,
        manifest_path=manifest_path,
        payload_path=payload_path,
        generation=generation,
    )
    report = {
        "validation_build_id": "revision_R1_colab_validation_v6_2_final_gan_recovery_qa",
        "validation_workflow_id": "2b4d8f5f1f29a938",
        "schedule_sha256": SCHEDULE_SHA,
        "status": "PASS",
        "execution_scope": "COLAB_CUDA_SYNTHETIC_VALIDATION",
        "classifier_phase": {"status": "PASS", "rerun": False},
        "gan_phase": {
            "status": "PASS",
            "interruption_drive_checkpoint": {
                **{key: publication_evidence[key] for key in (
                    "generation", "manifest_sha256", "payload_sha256", "payload_bytes",
                )},
                "worker_publication_matches_drive_readback": publication_evidence,
                "fresh_process_restored_from_drive": True,
                "local_staging_empty_before_restore": True,
            },
        },
        "reference_checkpoint": {
            "status": "PASS", "max_checkpoint_payload_bytes": 180_000_000,
            "reference_manifest_verified": True, "reference_payload_hash_verified": True,
            "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "payload_sha256": hashlib.sha256(b"payload").hexdigest(),
            "generation": 79,
            "code_sha256": code_hash, "schedule_sha256": SCHEDULE_SHA,
            "identity": {
                "code_sha256": code_hash, "schedule_sha256": SCHEDULE_SHA,
                "execution_scope": "synthetic_validation",
            },
        },
    }
    report_path = tmp_path / "resume_validation_2b4d8f5f1f29a938_20261026T142320Z_946699443.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_v62_report_loader_accepts_only_the_exact_verified_gan_recovery_report(tmp_path):
    report_path = _write_v62_producer_report_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bound = load_v62_recovery_checkpoint_bound(report_path)
    assert bound["max_checkpoint_payload_bytes"] == 180_000_000
    assert bound["provenance"] == "v6.2 synthetic GPU recovery report; not a production measurement"
    assert bound["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()

    import copy

    invalid_reports = []
    for path, value in (
        (("status",), "NOT_RUN"),
        (("validation_workflow_id",), "other"),
        (("reference_checkpoint", "reference_payload_hash_verified"), False),
        (("gan_phase", "status"), "FAIL"),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "status"), "FAIL"),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "generation_count"), 2),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "manifest_sha256"), "0" * 64),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "payload_sha256"), "1" * 64),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "generation"), 80),
        (("gan_phase", "interruption_drive_checkpoint", "worker_publication_matches_drive_readback", "payload_bytes"), 999),
        (("gan_phase", "interruption_drive_checkpoint", "fresh_process_restored_from_drive"), False),
        (("gan_phase", "interruption_drive_checkpoint", "local_staging_empty_before_restore"), False),
    ):
        invalid = copy.deepcopy(report)
        target = invalid
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        invalid_reports.append(invalid)
    for invalid in invalid_reports:
        report_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            load_v62_recovery_checkpoint_bound(report_path)


def test_capacity_gates_keep_reserve_and_separate_first_run_from_matrix():
    plan = {"run_count": 100}
    projection = project_storage(plan, observed_checkpoint_bytes=1000, observed_output_bytes=0)
    first_run_required = projection["drive"]["first_run_peak_with_reserve_bytes"]
    local_required = projection["local"]["next_run_peak_with_reserve_bytes"]
    require_run_capacity({
        "drive_available_bytes": first_run_required,
        "local_available_bytes": local_required,
        "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
    }, projection, scope="first_run")
    with pytest.raises(StorageCapacityError, match="first_run.*required"):
        require_run_capacity({"drive_available_bytes": first_run_required - 1, "local_available_bytes": local_required}, projection, scope="first_run")
    with pytest.raises(StorageCapacityError, match="validated first-run"):
        require_run_capacity({"drive_available_bytes": first_run_required, "local_available_bytes": local_required}, projection, scope="matrix")
    require_run_capacity({
        "drive_available_bytes": first_run_required,
        "local_available_bytes": projection["local"]["next_run_peak_with_reserve_bytes"],
        "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
        "production_measurements_validated": True,
    }, projection, scope="matrix", completed_runs=1)
    with pytest.raises(StorageCapacityError, match="matrix.*required"):
        require_run_capacity({
            "drive_available_bytes": first_run_required - 1,
            "local_available_bytes": projection["local"]["next_run_peak_with_reserve_bytes"],
            "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
            "production_measurements_validated": True,
        }, projection, scope="matrix", completed_runs=1)


def test_production_launch_switches_default_off_and_fail_before_any_filesystem_change(tmp_path):
    output_root = tmp_path / "drive-output"
    checkpoint_root = tmp_path / "drive-checkpoints"
    local_root = tmp_path / "local-stage"
    args = {
        "plan": {}, "preflight": {}, "schedule_path": tmp_path / "effective.csv",
        "schedule_manifest_path": tmp_path / "effective.manifest.json",
        "data_roots": {}, "cache_root": tmp_path / "cache",
        "output_root": output_root, "checkpoint_root": checkpoint_root,
        "local_checkpoint_root": local_root,
    }
    with pytest.raises(RuntimeError, match="LAUNCH_FIRST_PRODUCTION_RUN is False"):
        execute_first_production_run(
            args["plan"], preflight=args["preflight"], schedule_path=args["schedule_path"],
            schedule_manifest_path=args["schedule_manifest_path"], v62_recovery_report_path=tmp_path / "report.json",
            data_roots=args["data_roots"], cache_root=args["cache_root"], output_root=output_root,
            checkpoint_root=checkpoint_root, local_checkpoint_root=local_root,
        )
    with pytest.raises(RuntimeError, match="LAUNCH_REMAINING_MATRIX is False"):
        continue_production_matrix(
            args["plan"], preflight=args["preflight"], schedule_path=args["schedule_path"],
            schedule_manifest_path=args["schedule_manifest_path"], data_roots=args["data_roots"],
            cache_root=args["cache_root"], output_root=output_root, checkpoint_root=checkpoint_root,
            local_checkpoint_root=local_root,
        )
    assert not output_root.exists() and not checkpoint_root.exists() and not local_root.exists()


def test_progress_identity_binds_plan_schedule_source_and_run(tmp_path):
    plan = _production_source_plan()
    record = plan["runs"][0]
    progress_path = tmp_path / "progress.json"
    identity = {
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "run_key": record["run_key"], "run_index": record["run_index"],
        "config_sha256": record["config_sha256"],
        "status": "INTERRUPTED_RESUMABLE",
        "drive_checkpoint_write_bytes_total": 1234,
    }
    progress_path.write_text(json.dumps(identity), encoding="utf-8")

    assert load_run_progress(progress_path, plan, record) == identity

    for field in ("plan_sha256", "schedule_sha256", "training_source_sha256", "config_sha256"):
        changed = {**identity, field: "0" * 64}
        progress_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="progress identity"):
            load_run_progress(progress_path, plan, record)

    invalid_count = {**identity, "drive_checkpoint_write_bytes_total": -1}
    progress_path.write_text(json.dumps(invalid_count), encoding="utf-8")
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_run_progress(progress_path, plan, record)


def test_progress_export_labels_registered_validation_as_historical_not_fresh(tmp_path):
    plan = _production_source_plan()
    record = plan["runs"][0]
    root = tmp_path / "production-output"
    validated = {
        "run_id": production_workflow._run_id_for(record, synthetic=False),
        "artifact_manifest_sha256": hashlib.sha256(b"validated-artifact-index").hexdigest(),
    }
    production_workflow._record_completed_run(root, plan, record, validated)

    exported = export_production_progress(plan, root)

    assert exported["status"] == "IN_PROGRESS"
    assert exported["validated_completion_count"] == 1
    first = exported["runs"][0]
    assert first["status"] == "REGISTERED_COMPLETION"
    assert first["validation_freshness"] == "VALIDATED_AT_REGISTRATION_NOT_RECHECKED"
    assert first["progress_log"] is None


def test_progress_export_counts_only_identity_bound_drive_mirror_readback_bytes(tmp_path):
    plan = _production_source_plan()
    record = plan["runs"][0]
    root = tmp_path / "production-output"
    progress_path = root / "progress" / plan["plan_sha256"] / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
    progress = {
        **production_workflow._run_progress_identity(plan, record),
        "status": "INTERRUPTED_RESUMABLE",
        "drive_checkpoint_write_bytes_total": 123456,
        "drive_checkpoint_write_bytes_this_attempt": 123456,
    }
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    exported = export_production_progress(plan, root)

    assert exported["progress_records_found"] == 1
    assert exported["successfully_mirrored_checkpoint_payload_and_manifest_bytes"] == 123456
    assert "partial failed-upload bytes" in exported["checkpoint_write_measurement_scope"]


def test_matrix_executor_refuses_an_existing_live_matrix_lease(tmp_path):
    from revision_r1.locking import LeaseError, RunLease

    lease_path = tmp_path / "output" / ".matrix_execution.lease.json"
    owner = RunLease(lease_path, ttl_seconds=60, owner_token="already-running-colab")
    owner.acquire()
    contender = production_workflow._MatrixLeaseHeartbeat(lease_path, interval_seconds=.01, ttl_seconds=60)
    try:
        with pytest.raises(LeaseError, match="held"):
            contender.__enter__()
    finally:
        owner.release()


def test_production_capacity_failure_precedes_run_executor_and_writes(tmp_path, monkeypatch):
    from revision_r1 import runner as runner_module

    preflight = _preflight()
    schedule_path, manifest_path = _write_effective_schedule(tmp_path, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1],
        schedule_sha256=schedule_sha, dataset_references=DATASET_REFERENCES,
    )
    plan = bind_runtime_schedule(
        plan, schedule_path, manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    record = plan["runs"][0]
    output_root = tmp_path / "drive-output"
    checkpoint_root = tmp_path / "drive-checkpoints"
    projection = project_storage(plan, observed_checkpoint_bytes=1000)
    snapshot = {
        "canonical_storage_identity_verified": True,
        "destination_owner_verified": True, "write_capability_verified": True,
        "permanent_recovery_delete_verified": True,
        "new_file_owner_verified": True, "quota_metadata_verified": True,
        "output_path": str(output_root.resolve()), "checkpoint_path": str(checkpoint_root.resolve()),
        "drive_available_bytes": 0, "local_available_bytes": 0,
    }
    monkeypatch.setattr(runner_module, "run_one", lambda *args, **kwargs: pytest.fail("runner must not start when capacity gate fails"))
    monkeypatch.setattr(production_workflow, "_require_canonical_storage_api_factory", lambda _factory: None)

    with pytest.raises(StorageCapacityError, match="first_run capacity gate stopped before writes"):
        execute_frozen_plan_record(
            plan, record, _small_synthetic_bundle(),
            output_root=output_root, checkpoint_root=checkpoint_root,
            local_checkpoint_root=tmp_path / "local-stage", schedule_path=schedule_path,
            production_authorized=True, device="cuda", storage_snapshot=snapshot,
            storage_projection=projection, capacity_scope="first_run",
        )

    assert not output_root.exists()
    assert not checkpoint_root.exists()
    assert not (tmp_path / "local-stage").exists()


def _small_synthetic_bundle():
    rng = np.random.default_rng(77)
    x_train = np.vstack([rng.normal(class_id / 20, .2, size=(2, 4)) for class_id in range(100)]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([rng.normal(class_id / 20, .2, size=(1, 4)) for class_id in range(100)]).astype(np.float32)
    y_test = np.arange(100, dtype=np.int64)
    marker = {"md5": "synthetic-only", "sha256": "synthetic-only"}
    return DatasetBundle("ember", x_train, y_train, x_test, y_test, marker, marker)


def _production_source_plan():
    project_root = Path(__file__).resolve().parents[1]
    return build_frozen_run_plan(
        project_root, schedule_sha256=SCHEDULE_SHA, dataset_references=DATASET_REFERENCES,
    )


@pytest.fixture
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_frozen_plan_record_executes_real_run_one_and_completion_registration_is_idempotent(tmp_path, one_torch_thread):
    plan = _production_source_plan()
    record = plan["runs"][0]
    events = []
    result = execute_frozen_plan_record(
        plan, record, _small_synthetic_bundle(),
        output_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY",
        checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "checkpoints",
        local_checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "local",
        synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
        progress_callback=events.append,
    )
    assert result["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    assert result["completion_count"] == 1
    output = Path(result["output_path"])
    assert json.loads((output / "full-result.json").read_text())["namespace"] == "SYNTHETIC_NON_REPORTABLE"
    run_progress = tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "progress" / plan["plan_sha256"]
    event_logs = list((run_progress / "events").glob("ember_seed42_full_hybrid_*.jsonl"))
    assert len(event_logs) == 1
    log_events = [json.loads(line) for line in event_logs[0].read_text().splitlines()]
    assert len(log_events) >= 2
    assert all(event["attempt_id"] == log_events[0]["attempt_id"] for event in log_events)
    assert any(event.get("live_progress", {}).get("identity", {}).get("arm") == "full_hybrid" for event in log_events)
    assert all("live_progress" in event for event in events if event.get("status") == "RUNNING")

    repeated = execute_frozen_plan_record(
        plan, record, _small_synthetic_bundle(),
        output_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY",
        checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "checkpoints",
        local_checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "local",
        synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
    )
    assert repeated["status"] == "ALREADY_VALIDATED"
    assert repeated["completion_count"] == 1
    (output / "predictions.npz").write_bytes(b"corrupted only in temporary test fixture")
    with pytest.raises((ValueError, OSError, zipfile.BadZipFile)):
        production_workflow._validate_completed_run(plan, record, _small_synthetic_bundle(), output, synthetic=True)


def test_frozen_record_interruption_resumes_same_identity_and_does_not_register_early(tmp_path, one_torch_thread):
    plan = _production_source_plan()
    record = plan["runs"][0]
    root = tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY"
    kwargs = dict(
        output_root=root, checkpoint_root=root / "checkpoints", local_checkpoint_root=root / "local",
        synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
    )

    def stop_after_first_task(state):
        return state.cursor.task_index == 1 and state.cursor.phase == "scaler"

    first_events = []
    interrupted = execute_frozen_plan_record(
        plan, record, _small_synthetic_bundle(), stop_hook=stop_after_first_task,
        progress_callback=first_events.append, **kwargs,
    )
    assert interrupted["status"] == "INTERRUPTED_RESUMABLE"
    assert interrupted["completion_count"] == 0
    recovery = list((Path(interrupted["local_checkpoint_root"]) / "recovery").glob("generation-*.manifest.json"))
    assert recovery

    resumed_events = []
    resumed = execute_frozen_plan_record(
        plan, record, _small_synthetic_bundle(), progress_callback=resumed_events.append, **kwargs,
    )
    assert resumed["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    assert resumed["completion_count"] == 1
    assert resumed["run_identity_sha256"] == interrupted["run_identity_sha256"]
    event_root = root / "progress" / plan["plan_sha256"] / "events"
    event_logs = sorted(event_root.glob("ember_seed42_full_hybrid_*.jsonl"))
    assert len(event_logs) == 2
    attempt_ids = [json.loads(path.read_text().splitlines()[0])["attempt_id"] for path in event_logs]
    assert len(set(attempt_ids)) == 2
    assert first_events and resumed_events


def test_frozen_record_rejects_source_identity_mismatch_before_creating_outputs(tmp_path):
    plan = _production_source_plan()
    record = dict(plan["runs"][0], training_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="source identity"):
        execute_frozen_plan_record(
            plan, record, _small_synthetic_bundle(),
            output_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY",
            checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "checkpoints",
            local_checkpoint_root=tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY" / "local",
            synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
        )
    assert not (tmp_path / "SYNTHETIC_NONREPORTABLE_TEST_ONLY").exists()


def _finalization_plan(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    preflight = _preflight()
    schedule_path, schedule_manifest_path = _write_effective_schedule(tmp_path, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=schedule_sha,
        dataset_references=DATASET_REFERENCES,
    )
    plan = bind_runtime_schedule(
        plan, schedule_path, schedule_manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    return plan, plan["runs"][0]


def _allow_test_checkpoint_memory(monkeypatch):
    """Make fixture serialization deterministic; production RAM guards remain unmodified."""
    import revision_r1.resources as resources

    monkeypatch.setattr(resources, "read_host_memory_snapshot", lambda: {
        "effective_available_bytes": 8 << 30,
        "host_available_bytes": 8 << 30,
        "cgroup_available_bytes": 8 << 30,
        "memory_status": "synthetic_fixture_capacity_only",
    })


def _write_completed_production_fixture(output: Path, plan: dict, record: dict) -> dict:
    from revision_r1.registry import RunRegistry
    from revision_r1.validation import build_artifact_manifest

    output.mkdir(parents=True)
    order = record["class_order"]
    task_matrix = [[0.5 if column <= row else None for column in range(11)] for row in range(11)]
    boundary_stages = [
        {"stage": stage, "test_example_count": 1, "prediction_prefix": f"stage_{stage:02d}"}
        for stage in range(1, 12)
    ]
    for name, value in {
        "config.json": {
            "config_sha256": record["config_sha256"],
            "code_sha256": plan["training_source_sha256"], "namespace": "PRODUCTION",
        },
        "environment.json": {"fixture": "completed-output-only; no model was initialized"},
        "class_order.json": {"class_order": order, "sha256": record["class_order_sha256"]},
        "task_matrix.json": {"task_matrix": task_matrix},
        "task_boundary_prediction_manifest.json": {"stages": boundary_stages},
        "source_proportions.json": [
            {"sampled_examples": 1, "realized_draws": {"current": 1}} for _ in range(11)
        ],
        "classifier_mode_trace.json": [
            {"task_id": task, "phase": phase, "classifier_training": training,
             "dropout_training": [training]}
            for task in range(1, 12)
            for phase, training in (("classifier_training", True), ("evaluation", False))
        ],
        "resource_metrics.json": {
            "classifier_seconds": 11.0, "generator_seconds": 12.0, "critic_seconds": 13.0,
            "checkpoint_seconds": 14.0, "active_seconds": 50.0,
            "peak_cuda_allocated_bytes": 100, "peak_cuda_reserved_bytes": 200,
        },
        "full-result.json": {
            "dataset": record["dataset"], "seed": record["seed"], "arm": record["arm"],
            "schedule_sha256": plan["schedule_sha256"], "training_reproduction_claimed": False,
            "namespace": "PRODUCTION", "classifier_updates": sum(record["update_budgets_by_task"]),
            "generator_steps": 22, "critic_steps": 33,
        },
        "status.json": {"status": "COMPLETED_VALIDATED", "namespace": "PRODUCTION"},
    }.items():
        (output / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    (output / "per_task_metrics.csv").write_text(
        "task_id\n" + "".join(f"{task}\n" for task in range(1, 12)), encoding="utf-8",
    )
    (output / "classification_report.csv").write_text("class\n0\n", encoding="utf-8")
    (output / "confusion_matrix.csv").write_text("0\n1\n", encoding="utf-8")
    (output / "trace.jsonl").write_text("", encoding="utf-8")
    (output / "selection_records.jsonl").write_text("", encoding="utf-8")
    np.savez(output / "predictions.npz", y_true_global=np.array([0]), y_pred_global=np.array([0]))
    boundary_arrays = {}
    for stage in range(1, 12):
        prefix = f"stage_{stage:02d}"
        boundary_arrays[f"{prefix}_y_true_global"] = np.array([0])
        boundary_arrays[f"{prefix}_y_pred_global"] = np.array([0])
        boundary_arrays[f"{prefix}_test_indices"] = np.array([0])
    np.savez(output / "task_boundary_predictions.npz", **boundary_arrays)
    from revision_r1.validation import build_artifact_manifest
    (output / "artifact_manifest.json").write_text(
        json.dumps(build_artifact_manifest(output), sort_keys=True), encoding="utf-8",
    )

    run_id = production_workflow._run_id_for(record, synthetic=False)
    identity = {
        "runner_version": "revision_R1.0.2", "config_sha256": record["config_sha256"],
        "dataset": {"namespace": "PRODUCTION", "dataset": record["dataset"], **record["dataset_identity"]},
        "code_sha256": plan["training_source_sha256"], "schedule_sha256": plan["schedule_sha256"],
        "execution_scope": "production", "run_id": run_id,
    }
    registry = RunRegistry(output / "registry")
    registry.begin_attempt(run_id, identity=identity)
    registry.complete(run_id, {
        name: output / name for name in (
            "artifact_manifest.json", "status.json", "full-result.json", "task_matrix.json",
            "predictions.npz", "resource_metrics.json", "config.json", "environment.json",
        )
    })
    return identity


def _write_completed_checkpoint_trees(checkpoint_namespace: Path, identity: dict) -> tuple[Path, Path]:
    from revision_r1.checkpointing import save_generation
    from revision_r1.state import PhaseCursor, RunState

    recovery = checkpoint_namespace / "recovery"
    boundaries = checkpoint_namespace / "task_boundaries"
    for generation in range(1, 5):
        state = RunState(identity=identity, cursor=PhaseCursor(task_index=10, phase=f"recovery_{generation}"))
        save_generation(
            recovery, state, identity,
            kind="complete" if generation == 4 else "recovery",
            retain_generations=None,
        )
    for task in range(1, 12):
        state = RunState(identity=identity, cursor=PhaseCursor(task_index=task, phase="task-boundary"))
        save_generation(boundaries, state, identity, kind="task-boundary", retain_generations=None)
    return recovery, boundaries


class _FinalizationDriveFiles(_FakeFiles):
    def __init__(
        self, records, deleted_ids, local_paths, fail_once_name=None, drive_root=None,
        stale_metadata_reads_after_update=0, interrupt_after_update_once=False,
        fail_updates_after_accept=0,
    ):
        super().__init__(records, deleted_ids)
        self.local_paths = local_paths
        self.drive_root = drive_root
        self.fail_once_name = fail_once_name
        self.failed_once = False
        self.fail_media_readback = False
        self.discover_local_folders = True
        self.created_folder_ids = []
        self.created_file_ids = []
        self.remote_contents = {}
        self.update_calls = []
        self.stale_metadata_reads_after_update = stale_metadata_reads_after_update
        self.stale_metadata_snapshots = {}
        self.interrupt_after_update_once = interrupt_after_update_once
        self.fail_updates_after_accept = fail_updates_after_accept
        self.generated_ids = 0

    def generateIds(self, *, count=1, **_kwargs):
        ids = []
        for _ in range(count):
            self.generated_ids += 1
            ids.append(f"generated-journal-{self.generated_ids}")
        return _FakeRequest({"ids": ids})

    def create(self, *, body, fields=None, **_kwargs):
        parent_ids = body.get("parents", [])
        if len(parent_ids) != 1:
            raise AssertionError("test Drive file creation must bind exactly one parent")
        parent_id = str(parent_ids[0])
        if parent_id not in self.records or self.records[parent_id].get("capabilities", {}).get("canAddChildren") is not True:
            raise PermissionError("test Drive parent does not permit adding children")
        is_folder = body.get("mimeType") == "application/vnd.google-apps.folder"
        file_id = str(body.get("id") or (
            f"created-folder-{len(self.created_folder_ids) + 1}" if is_folder
            else f"created-drive-file-{len(self.created_file_ids) + 1}"
        ))
        record = {
            "id": file_id, "name": body["name"], "mimeType": body["mimeType"],
            "parents": [parent_id], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
            "ownedByMe": True, "capabilities": {"canAddChildren": True},
            "appProperties": dict(body.get("appProperties", {})),
            "size": "0", "quotaBytesUsed": "0", "version": "1",
        }
        self.records[file_id] = record
        if is_folder:
            self.created_folder_ids.append(file_id)
        else:
            self.created_file_ids.append(file_id)
            self.remote_contents[file_id] = b""
        parent_components = []
        cursor = self.records[parent_id]
        while cursor.get("id") != "root":
            parent_components.append(cursor["name"])
            ancestors = cursor.get("parents", [])
            cursor = self.records.get(ancestors[0], {"id": "root"}) if ancestors else {"id": "root"}
        target_dir = self.drive_root.joinpath(*reversed(parent_components), body["name"])
        if is_folder:
            target_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRequest(dict(record))

    def update(self, *, fileId, media_body=None, body=None, **_kwargs):
        previous = dict(self.records[fileId])
        if isinstance(media_body, (bytes, bytearray)):
            content = bytes(media_body)
        else:
            content = media_body.getbytes(0, media_body.size())
        self.remote_contents[fileId] = content
        record = self.records[fileId]
        if body:
            record.update(body)
        record["size"] = str(len(content))
        record["quotaBytesUsed"] = str(len(content))
        record["md5Checksum"] = hashlib.md5(content, usedforsecurity=False).hexdigest()
        record["version"] = str(int(record.get("version", "0")) + 1)
        self.update_calls.append({"file_id": fileId, "bytes": content})
        if self.stale_metadata_reads_after_update:
            self.stale_metadata_snapshots[fileId] = [
                dict(previous) for _ in range(self.stale_metadata_reads_after_update)
            ]
        if self.interrupt_after_update_once:
            self.interrupt_after_update_once = False
            raise KeyboardInterrupt("injected interruption after Drive accepted journal bytes")
        if self.fail_updates_after_accept:
            self.fail_updates_after_accept -= 1
            raise OSError("injected transport failure after Drive accepted journal bytes")
        return _FakeRequest(dict(record))

    def get(self, *, fileId, alt=None, **kwargs):
        if alt == "media" and self.fail_media_readback:
            raise OSError("injected Drive media read-back failure")
        if alt == "media" and fileId in self.remote_contents:
            return _FakeMediaRequest(self.remote_contents[fileId])
        if alt is None and self.stale_metadata_snapshots.get(fileId):
            return _FakeRequest(self.stale_metadata_snapshots[fileId].pop(0))
        if alt is None and fileId in self.remote_contents:
            return _FakeRequest(dict(self.records[fileId]))
        return super().get(fileId=fileId, alt=alt, **kwargs)

    def delete(self, *, fileId, **kwargs):
        name = self.records[fileId]["name"]
        if name == self.fail_once_name and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("injected transient Drive delete failure")
        request = super().delete(fileId=fileId, **kwargs)
        local_path = self.local_paths.get(fileId)
        if local_path is not None:
            local_path.unlink(missing_ok=True)
        self.remote_contents.pop(fileId, None)
        return request

    def list(self, *, q, **kwargs):
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        if self.drive_root is not None:
            if parent == "root":
                local_dir = self.drive_root
            else:
                folder = self.records.get(parent)
                components = []
                cursor = folder
                while cursor and cursor.get("id") != "root":
                    components.append(cursor["name"])
                    parent_ids = cursor.get("parents", [])
                    cursor = self.records.get(parent_ids[0]) if parent_ids else {"id": "root"}
                local_dir = self.drive_root.joinpath(*reversed(components))
            existing = {
                record.get("name"): record for record in self.records.values()
                if parent in record.get("parents", [])
            }
            for path in local_dir.iterdir() if local_dir.is_dir() else ():
                record = existing.get(path.name)
                if path.is_dir() and self.discover_local_folders:
                    if record is None:
                        file_id = f"mounted-folder-{parent}-{path.name}"
                        self.records[file_id] = {
                            "id": file_id, "name": path.name,
                            "mimeType": "application/vnd.google-apps.folder", "parents": [parent],
                            "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
                            "ownedByMe": True, "capabilities": {"canAddChildren": True},
                        }
                    continue
                if not path.is_file():
                    continue
                if record is None:
                    file_id = f"mounted-{parent}-{path.name}"
                    record = {
                        "id": file_id, "name": path.name, "mimeType": "application/json",
                        "parents": [parent], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
                        "ownedByMe": True, "capabilities": {"canDelete": True},
                    }
                    self.records[file_id] = record
                if record.get("id") in self.remote_contents:
                    continue
                record["size"] = str(path.stat().st_size)
                record["quotaBytesUsed"] = str(path.stat().st_size)
                self.local_paths[record["id"]] = path
        return super().list(q=q, **kwargs)


def _finalization_drive_service(
    recovery: Path, drive_root: Path, *, denied_names=(), fail_once_name=None,
    stale_metadata_reads_after_update=0, interrupt_after_update_once=False,
    fail_updates_after_accept=0,
):
    components = _logical_drive_path_components(recovery, drive_root=drive_root)[1]
    records = {}
    parent = "root"
    for index, name in enumerate(components):
        folder_id = production_workflow.CANONICAL_REVISION_FOLDER_ID if index == 0 else f"folder-{index}"
        records[folder_id] = {
            "id": folder_id, "name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
            "ownedByMe": True, "capabilities": {"canAddChildren": True},
        }
        parent = folder_id
    local_paths = {}
    for path in sorted(recovery.iterdir()):
        file_id = f"file-{path.name}"
        local_paths[file_id] = path
        records[file_id] = {
            "id": file_id, "name": path.name, "mimeType": "application/octet-stream",
            "parents": [parent], "owners": [{"emailAddress": "kian.anb777@gmail.com"}],
            "ownedByMe": True, "size": str(path.stat().st_size),
            "quotaBytesUsed": str(path.stat().st_size),
            "capabilities": {"canDelete": path.name not in denied_names},
        }
    service = _FakeDriveService(records, {
        "user": {"emailAddress": "kian.anb777@gmail.com"},
        "storageQuota": {"limit": "10000000000", "usage": "1000000000"},
    })
    service._files = _FinalizationDriveFiles(
        records, service.deleted_ids, local_paths, fail_once_name=fail_once_name,
        drive_root=drive_root,
        stale_metadata_reads_after_update=stale_metadata_reads_after_update,
        interrupt_after_update_once=interrupt_after_update_once,
        fail_updates_after_accept=fail_updates_after_accept,
    )
    return service


def test_drive_journal_payload_uses_media_endpoint_not_metadata_json_endpoint():
    import httplib2
    from googleapiclient.model import JsonModel

    raw_payload = b'{\n  "journal_revision": 7,\n  "note": "preserve exact bytes"\n}'
    metadata_payload = (
        b'{"id":"journal-id","kind":"drive#file",'
        b'"mimeType":"application/json","name":"journal.json"}'
    )

    class _RawHttp:
        def __init__(self, payload, *, content_location=None):
            self.payload = payload
            self.content_location = content_location

        def request(self, uri, method, body=None, headers=None, **_kwargs):
            assert method == "GET"
            assert headers["range"].startswith("bytes=0-")
            response_headers = {
                "status": "200", "content-type": "application/json",
                "content-length": str(len(self.payload)),
            }
            if self.content_location:
                response_headers["content-location"] = self.content_location
            return httplib2.Response(response_headers), self.payload

    class _MediaRequest:
        def __init__(self, payload, uri, *, content_location=None):
            self.uri = uri
            self.headers = {}
            self.http = _RawHttp(payload, content_location=content_location)

        def execute(self):
            return JsonModel().deserialize(self.http.payload)

    metadata_uri = "https://www.googleapis.com/drive/v3/files/journal-id?alt=json&supportsAllDrives=true"
    media_uri = "https://www.googleapis.com/drive/v3/files/journal-id?alt=media&supportsAllDrives=true"
    metadata_request = _MediaRequest(
        metadata_payload, metadata_uri, content_location=metadata_uri,
    )
    media_request = _MediaRequest(raw_payload, media_uri)
    calls = []

    class _Files:
        def get(self, *, fileId, alt=None, **_kwargs):
            calls.append(("get", fileId, alt))
            return metadata_request

        def get_media(self, *, fileId, **_kwargs):
            assert fileId == "journal-id"
            calls.append(("get_media", fileId, None))
            return media_request

    class _Service:
        def files(self):
            return _Files()

    observed = production_workflow._get_drive_journal_payload(
        _Service(), "journal-id", expected={"file_id": "journal-id"},
    )

    assert isinstance(metadata_request.execute(), dict)
    assert calls == [("get_media", "journal-id", None)]
    assert observed == raw_payload
    assert observed != metadata_payload


@pytest.mark.parametrize(
    "shared_shortcut", [False, True], ids=["ordinary-mydrive", "shared-folder-shortcut"],
)
def test_completed_run_with_four_recovery_generations_finalizes_and_reregisters_idempotently(
    tmp_path, monkeypatch, shared_shortcut,
):
    from revision_r1.validation import file_sha256

    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_mount = tmp_path / "content" / "drive"
    drive_root = drive_mount / "MyDrive"
    drive_root.mkdir(parents=True)
    if shared_shortcut:
        shortcut_target = drive_mount / ".shortcut-targets-by-id" / "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x" / "ReVision"
        shortcut_target.mkdir(parents=True)
        (drive_root / "ReVision").symlink_to(shortcut_target, target_is_directory=True)
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    boundary_hashes = {path.name: file_sha256(path) for path in boundaries.iterdir()}
    events = output_root / "progress" / plan["plan_sha256"] / "events"
    events.mkdir(parents=True)
    event_path = events / f"{record['run_key']}_{record['config_sha256'][:12]}_attempt.jsonl"
    event_path.write_text(json.dumps({
        "last_verified_drive_checkpoint": {
            "pruning_deferred_files": [
                {"generation": 1, "file": "generation-000001.pt"},
                {"generation": 1, "file": "generation-000001.manifest.json"},
            ],
        },
    }) + "\n", encoding="utf-8")
    assert production_workflow._validate_completed_run(plan, record, None, output, synthetic=False)["validation"]["valid"]
    with pytest.raises(ValueError, match=r"recovery: expected 2, found 4"):
        production_workflow.measure_completed_production_run(plan, record, None, output, checkpoint_namespace)

    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer), "finalization-only recovery must reconcile the four verified generations"
    logical_recovery = (
        checkpoint_root / "revision_R1.0.2" / record["training_source_sha256"][:16]
        / record["dataset"] / f"seed_{int(record['seed'])}" / record["arm"]
        / record["config_sha256"][:16] / "recovery"
    )
    assert logical_recovery.is_relative_to(drive_root)
    assert logical_recovery.resolve() == recovery.resolve()
    if shared_shortcut:
        assert ".shortcut-targets-by-id/1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x/ReVision/" in str(recovery)
    service = _finalization_drive_service(logical_recovery, drive_root)
    monkeypatch.setattr(
        production_workflow, "execute_frozen_plan_record",
        lambda *_args, **_kwargs: pytest.fail("finalization must never invoke training"),
    )
    progress = []
    first = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root, progress_callback=progress.append,
    )
    second = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    assert first["status"] == "FINALIZED_COMPLETED_RUN"
    assert first["completion_registered"] is True
    assert first["recovery_retention"]["retained_generations"] == [3, 4]
    assert first["recovery_retention"]["rotation_explanation"]["status"] == "CONFIRMED_FROM_RUN_PROGRESS"
    assert len(first["recovery_retention"]["rotation_explanation"]["deferred_files_still_present"]) == 2
    assert [row["generation"] for row in first["recovery_inventory"]] == [1, 2, 3, 4]
    assert all(row["integrity_status"] == "PASS_MANIFEST_AND_PAYLOAD_HASH" for row in first["recovery_inventory"])
    assert all(file["owners"] for row in first["recovery_inventory"] for file in row["files"].values())
    assert len(list(recovery.glob("generation-*.manifest.json"))) == 2
    assert sorted(path.name for path in recovery.glob("generation-*.pt")) == [
        "generation-000003.pt", "generation-000004.pt",
    ]
    assert {path.name: file_sha256(path) for path in boundaries.iterdir()} == boundary_hashes
    assert second["completion_registered"] is False
    assert first["storage_measurement"] == second["storage_measurement"]
    assert production_workflow._read_matrix_registry(output_root, plan)["completed_runs"][record["run_key"]]["storage_measurement"] == first["storage_measurement"]
    assert any(row.get("stage") == "recovery_inventory_complete" for row in progress)


def test_finalization_reconciles_older_orphan_manifests_and_retries_safely(tmp_path, monkeypatch):
    from revision_r1.validation import file_sha256

    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    (recovery / "generation-000001.pt").unlink()
    (recovery / "generation-000002.pt").unlink()
    boundary_hashes = {path.name: file_sha256(path) for path in boundaries.iterdir()}
    output_manifest_hash = file_sha256(output / "artifact_manifest.json")
    service = _finalization_drive_service(
        recovery, drive_root, fail_once_name="generation-000002.manifest.json",
    )
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer)
    monkeypatch.setattr(
        production_workflow, "execute_frozen_plan_record",
        lambda *_args, **_kwargs: pytest.fail("finalization must never invoke training"),
    )

    with pytest.raises(StorageCapacityError, match="Drive delete failure"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    journal = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    assert journal.is_file()
    partial_journal = json.loads(journal.read_text(encoding="utf-8"))
    assert partial_journal["deleted_files"] == ["generation-000001.manifest.json"]
    assert not (recovery / "generation-000001.manifest.json").exists()
    assert (recovery / "generation-000002.manifest.json").is_file()
    assert all((recovery / f"generation-{generation:06d}.pt").is_file() for generation in (3, 4))
    assert all((recovery / f"generation-{generation:06d}.manifest.json").is_file() for generation in (3, 4))

    retried = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    repeated = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    assert retried["status"] == "FINALIZED_COMPLETED_RUN"
    assert retried["completion_registered"] is True
    assert repeated["completion_registered"] is False
    assert retried["recovery_retention"]["retained_generations"] == [3, 4]
    assert retried["recovery_retention"]["deleted_orphan_manifests"] == [
        "generation-000001.manifest.json", "generation-000002.manifest.json",
    ]
    assert retried["recovery_retention"]["orphan_manifest_reconciliation"]["status"] == "VERIFIED_DELETED"
    assert retried["recovery_retention"]["orphan_manifest_reconciliation"]["specific_cause_established_from_run_logs"] is False
    assert [row["generation"] for row in retried["recovery_retention"]["orphan_manifest_reconciliation"]["manifest_records"]] == [1, 2]
    assert all(row["can_delete"] is True and row["identity_sha256"] for row in retried["recovery_retention"]["orphan_manifest_reconciliation"]["manifest_records"])
    assert all(row["integrity_status"] == "PASS_MANIFEST_AND_PAYLOAD_HASH" for row in retried["recovery_inventory"])
    assert len(list(recovery.glob("generation-*.manifest.json"))) == 2
    assert sorted(path.name for path in recovery.glob("generation-*.pt")) == [
        "generation-000003.pt", "generation-000004.pt",
    ]
    assert {path.name: file_sha256(path) for path in boundaries.iterdir()} == boundary_hashes
    assert file_sha256(output / "artifact_manifest.json") == output_manifest_hash
    assert retried["storage_measurement"] == repeated["storage_measurement"]


def test_finalization_creates_and_verifies_missing_drive_journal_folders(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    _identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, _identity)
    (recovery / "generation-000001.pt").unlink()
    (recovery / "generation-000002.pt").unlink()
    service = _finalization_drive_service(recovery, drive_root)
    service.files().discover_local_folders = False

    result = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    journal = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    assert result["status"] == "FINALIZED_COMPLETED_RUN"
    assert result["recovery_retention"]["retained_generations"] == [3, 4]
    assert len(service.files().created_folder_ids) == 2
    assert [service.records[file_id]["name"] for file_id in service.files().created_folder_ids] == [
        "_finalization", plan["plan_sha256"],
    ]
    location = production_workflow.resolve_drive_folder(
        service, production_workflow._logical_drive_path_components(journal.parent, drive_root=drive_root)[1],
    )
    assert location["path_exists"] is True
    metadata = production_workflow._lookup_drive_child(service, location["folder_id"], journal.name)
    assert metadata is not None
    assert service.files().get(fileId=metadata["id"], alt="media").execute() == journal.read_bytes()


def test_finalization_journal_api_publication_handles_shared_shortcut_and_stale_metadata(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_mount = tmp_path / "content" / "drive"
    drive_root = drive_mount / "MyDrive"
    drive_root.mkdir(parents=True)
    shortcut_target = drive_mount / ".shortcut-targets-by-id" / "shared-folder-id" / "ReVision"
    shortcut_target.mkdir(parents=True)
    (drive_root / "ReVision").symlink_to(shortcut_target, target_is_directory=True)
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    from revision_r1.validation import file_sha256
    boundary_hashes = {path.name: file_sha256(path) for path in boundaries.iterdir()}
    logical_recovery = (
        checkpoint_root / "revision_R1.0.2" / record["training_source_sha256"][:16]
        / record["dataset"] / f"seed_{int(record['seed'])}" / record["arm"]
        / record["config_sha256"][:16] / "recovery"
    )
    service = _finalization_drive_service(
        logical_recovery, drive_root, stale_metadata_reads_after_update=1,
    )
    service.files().discover_local_folders = False
    monkeypatch.setattr(
        production_workflow, "execute_frozen_plan_record",
        lambda *_args, **_kwargs: pytest.fail("finalization must never invoke training"),
    )

    first = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal_file_id = journal["journal_drive_file_id"]
    remote_metadata = service.files().get(fileId=journal_file_id).execute()
    remote_bytes = service.files().get(fileId=journal_file_id, alt="media").execute()
    folder_id = journal["journal_drive_parent_id"]
    assert first["status"] == "FINALIZED_COMPLETED_RUN"
    assert first["recovery_retention"]["retained_generations"] == [3, 4]
    assert service.files().update_calls, "journal bytes must be published through Drive API update"
    assert journal["journal_revision"] >= 2
    assert len(service.files().update_calls) >= 2
    assert remote_metadata["id"] == journal_file_id
    assert remote_metadata["parents"] == [folder_id]
    assert remote_metadata["owners"][0]["emailAddress"] != "folder-owner@example.org"
    assert remote_bytes == journal_path.read_bytes()
    assert hashlib.sha256(remote_bytes).hexdigest() == hashlib.sha256(journal_path.read_bytes()).hexdigest()
    assert service.files().stale_metadata_snapshots[journal_file_id] == []
    assert {path.name: file_sha256(path) for path in boundaries.iterdir()} == boundary_hashes

    update_count = len(service.files().update_calls)
    second = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    assert second["completion_registered"] is False
    assert len(service.files().update_calls) == update_count
    assert second["storage_measurement"] == first["storage_measurement"]


def test_finalization_recovers_journal_after_interruption_post_drive_publication(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    service = _finalization_drive_service(recovery, drive_root, interrupt_after_update_once=True)
    service.files().discover_local_folders = False

    with pytest.raises(KeyboardInterrupt, match="after Drive accepted journal bytes"):
        production_workflow.finalize_completed_production_run(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    assert not journal_path.exists()
    assert len(list(recovery.glob("generation-*.manifest.json"))) == 4
    assert service.deleted_ids == set()
    assert len(service.files().created_file_ids) == 1
    remote_file_id = service.files().created_file_ids[0]
    assert json.loads(service.files().get(fileId=remote_file_id, alt="media").execute())["journal_sha256"]

    resumed = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "FINALIZED_COMPLETED_RUN"
    assert resumed["recovery_retention"]["retained_generations"] == [3, 4]
    assert persisted["journal_drive_file_id"] == remote_file_id
    assert len(service.files().created_file_ids) == 1


def test_finalization_accepts_ambiguous_transport_failure_only_after_exact_api_readback(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    monkeypatch.setattr(production_workflow, "RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS", (0.0,) * 5)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    service = _finalization_drive_service(recovery, drive_root, fail_updates_after_accept=5)

    result = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    remote_bytes = service.files().get(
        fileId=journal["journal_drive_file_id"], alt="media",
    ).execute()
    assert result["status"] == "FINALIZED_COMPLETED_RUN"
    assert len(service.files().update_calls) >= 5
    assert all(
        item["bytes"] == service.files().update_calls[0]["bytes"]
        for item in service.files().update_calls[:5]
    )
    assert remote_bytes == journal_path.read_bytes()
    assert service.deleted_ids


def test_finalization_rejects_duplicate_journal_names_without_pruning(tmp_path, monkeypatch):
    from scripts.production_workflow import RecoveryJournalPublicationError

    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    service = _finalization_drive_service(recovery, drive_root)
    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    components = production_workflow._logical_drive_path_components(
        journal_path.parent, drive_root=drive_root,
    )[1]
    folder = production_workflow._ensure_drive_folder_path(service, components)
    for file_id in ("ambiguous-journal-a", "ambiguous-journal-b"):
        service.records[file_id] = {
            "id": file_id, "name": journal_path.name, "mimeType": "application/json",
            "parents": [folder["folder_id"]], "size": "0", "owners": [],
            "capabilities": {},
        }
        service.files().remote_contents[file_id] = b""

    with pytest.raises(RecoveryJournalPublicationError, match="duplicate.*journal"):
        production_workflow.finalize_completed_production_run(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    assert len(list(recovery.glob("generation-*.manifest.json"))) == 4
    assert service.deleted_ids == set()
    assert not journal_path.exists()


def test_finalization_recovers_preexisting_local_only_retention_journal(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    (recovery / "generation-000001.pt").unlink()
    (recovery / "generation-000002.pt").unlink()
    service = _finalization_drive_service(recovery, drive_root)
    service.files().discover_local_folders = False
    recovery_folder = production_workflow._drive_recovery_folder(service, recovery, drive_root)
    inspected = production_workflow._inspect_verified_recovery_tree(
        recovery, identity, service, recovery_folder["folder_id"],
    )
    orphan_rows = []
    for orphan in inspected["orphan_manifests"]:
        manifest_file = orphan["manifest_file"]
        orphan_rows.append({
            "generation": orphan["generation"], "name": manifest_file["name"],
            "drive_file_id": manifest_file["drive_file_id"], "size_bytes": manifest_file["size_bytes"],
            "local_sha256": manifest_file["local_sha256"], "can_delete": manifest_file["can_delete"],
            "identity_sha256": orphan["identity_sha256"], "payload_file": orphan["payload_file"],
            "payload_bytes": orphan["payload_bytes"], "payload_sha256": orphan["payload_sha256"],
        })
    journal = {
        "schema_version": 1, "status": "PREPARED", "plan_sha256": plan["plan_sha256"],
        "run_key": record["run_key"], "run_id": production_workflow._run_id_for(record, synthetic=False),
        "identity": identity, "identity_sha256": production_workflow._sha256_json(identity),
        "recovery_folder": recovery_folder, "protected_generations": [3, 4],
        "initial_recovery_inventory": inspected["verified_generations"], "delete_generations": [],
        "delete_orphan_manifests": orphan_rows, "deleted_files": [],
        "orphan_manifest_reconciliation": {
            "status": "PREPARED", "mechanism": "test fixture matching a local-only prior attempt",
            "specific_cause_established_from_run_logs": False,
            "reason": "older manifest-only generations are followed by two verified recovery points",
        },
        "rotation_explanation": production_workflow._rotation_deferred_pruning_evidence(
            output_root, plan, record, [],
        ),
        "prepared_utc": "2026-11-02T00:00:00+00:00",
    }
    journal["journal_sha256"] = production_workflow._finalization_journal_digest(journal)
    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    production_workflow._atomic_json(journal_path, journal)

    result = production_workflow.finalize_completed_production_run(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    published_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert result["status"] == "FINALIZED_COMPLETED_RUN"
    assert result["recovery_retention"]["retained_generations"] == [3, 4]
    assert published_journal["journal_drive_file_id"]
    assert published_journal["journal_drive_parent_id"]
    assert [service.records[file_id]["name"] for file_id in service.files().created_folder_ids] == [
        "_finalization", plan["plan_sha256"],
    ]


def test_finalization_journal_folder_creation_fails_closed_without_parent_permission(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    _identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, _identity)
    (recovery / "generation-000001.pt").unlink()
    (recovery / "generation-000002.pt").unlink()
    before = {path.name: path.read_bytes() for path in recovery.iterdir()}
    service = _finalization_drive_service(recovery, drive_root)
    service.files().discover_local_folders = False
    checkpoint_record = next(
        row for row in service.records.values() if row.get("name") == "checkpoints"
    )
    checkpoint_record["capabilities"]["canAddChildren"] = False

    with pytest.raises(StorageCapacityError, match="canAddChildren"):
        production_workflow.finalize_completed_production_run(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    journal = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    assert not journal.exists()
    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == before
    assert service.deleted_ids == set()
    assert service.files().created_folder_ids == []


@pytest.mark.parametrize("orphan_case", ["newer_than_fallback", "delete_denied_without_edit"])
def test_finalization_preserves_ambiguous_or_non_deletable_orphan_manifests(tmp_path, monkeypatch, orphan_case):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    denied_names = ()
    if orphan_case == "newer_than_fallback":
        newer = json.loads((recovery / "generation-000004.manifest.json").read_text(encoding="utf-8"))
        newer.update({"generation": 5, "payload_file": "generation-000005.pt", "kind": "recovery"})
        (recovery / "generation-000005.manifest.json").write_text(
            json.dumps(newer, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        error = "as new as a required retained recovery point"
    else:
        (recovery / "generation-000001.pt").unlink()
        (recovery / "generation-000002.pt").unlink()
        denied_names = {"generation-000001.manifest.json"}
        error = "does not verify delete or editor capability"
    before = {path.name: path.read_bytes() for path in recovery.iterdir()}
    service = _finalization_drive_service(recovery, drive_root, denied_names=denied_names)
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer)

    with pytest.raises((ValueError, StorageCapacityError), match=error):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == before
    assert service.deleted_ids == set()
    assert not (checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json").exists()


def test_finalization_shared_editor_preserves_orphans_after_account_switch(tmp_path, monkeypatch):
    """A shared-folder editor may retain owned orphan manifests but not permanently delete them."""
    from revision_r1.validation import file_sha256

    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    for generation in (1, 2):
        (recovery / f"generation-{generation:06d}.pt").unlink()
    manifest_names = [f"generation-{generation:06d}.manifest.json" for generation in (1, 2)]
    manifest_bytes = {name: (recovery / name).read_bytes() for name in manifest_names}
    boundary_hashes = {path.name: file_sha256(path) for path in boundaries.iterdir()}
    service = _finalization_drive_service(
        recovery, drive_root, fail_once_name=manifest_names[0],
    )
    for file_id, metadata in service.records.items():
        if metadata.get("name") in manifest_names:
            metadata["capabilities"]["canEdit"] = True

    finalizer = production_workflow.finalize_completed_production_run
    with pytest.raises(StorageCapacityError, match="Drive delete failure"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    journal_path = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    prepared = json.loads(journal_path.read_text(encoding="utf-8"))
    assert prepared["status"] == "PREPARED"
    assert prepared["deleted_files"] == []
    for metadata in service.records.values():
        if metadata.get("name") in manifest_names:
            metadata["capabilities"]["canDelete"] = False

    result = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    repeated = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )

    assert result["status"] == "FINALIZED_COMPLETED_RUN"
    assert result["completion_registered"] is True
    assert repeated["completion_registered"] is False
    assert result["recovery_retention"]["retained_generations"] == [3, 4]
    assert result["recovery_retention"]["deleted_orphan_manifests"] == []
    assert result["recovery_retention"]["preserved_orphan_manifests"] == manifest_names
    assert result["recovery_retention"]["orphan_manifest_reconciliation"]["status"] == "VERIFIED_PRESERVED"
    assert result["storage_measurement"]["recovery_checkpoints"]["generation_count"] == 2
    assert result["storage_measurement"]["recovery_checkpoints"]["manifest_only_orphan_generations"] == [1, 2]
    assert {name: (recovery / name).read_bytes() for name in manifest_names} == manifest_bytes
    assert all(not (recovery / f"generation-{generation:06d}.pt").exists() for generation in (1, 2))
    assert all((recovery / f"generation-{generation:06d}.pt").is_file() for generation in (3, 4))
    assert {path.name: file_sha256(path) for path in boundaries.iterdir()} == boundary_hashes
    assert service.deleted_ids == set()


def test_finalization_partial_drive_cleanup_is_safe_to_retry(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    service = _finalization_drive_service(
        recovery, drive_root, fail_once_name="generation-000001.manifest.json",
    )
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer), "partial-cleanup retry requires the finalization-only operation"

    with pytest.raises(StorageCapacityError, match="Drive delete failure"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )
    journal = checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    assert journal.is_file()
    assert not (recovery / "generation-000001.pt").exists()
    assert (recovery / "generation-000001.manifest.json").is_file()

    retried = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    assert retried["status"] == "FINALIZED_COMPLETED_RUN"
    assert retried["completion_registered"] is True
    assert retried["recovery_retention"]["retained_generations"] == [3, 4]
    assert len(list(recovery.glob("generation-*.manifest.json"))) == 2


def test_finalization_requires_drive_journal_payload_readback_before_pruning(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    service = _finalization_drive_service(recovery, drive_root)
    service.files().fail_media_readback = True
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer)

    monkeypatch.setattr(production_workflow, "RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    with pytest.raises(production_workflow.RecoveryJournalPublicationError, match="journal-payload-read"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )
    assert len(list(recovery.glob("generation-*.manifest.json"))) == 4
    assert service.deleted_ids == set()

    service.files().fail_media_readback = False
    retried = finalizer(
        plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
        api_factory=lambda: service, drive_root=drive_root,
    )
    assert retried["status"] == "FINALIZED_COMPLETED_RUN"
    assert retried["recovery_retention"]["retained_generations"] == [3, 4]


def test_finalization_refuses_unverified_drive_delete_capability_without_mutation(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    before = {path.name: path.read_bytes() for path in recovery.iterdir()}
    service = _finalization_drive_service(
        recovery, drive_root, denied_names={"generation-000001.pt"},
    )
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer), "retention must have a fail-closed finalization gate"

    with pytest.raises(StorageCapacityError, match="canDelete"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == before
    assert not (checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json").exists()
    assert service.deleted_ids == set()


def test_finalization_hash_failure_preserves_every_recovery_file(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, _boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    damaged = recovery / "generation-000001.pt"
    payload = bytearray(damaged.read_bytes())
    payload[-1] ^= 0x01
    damaged.write_bytes(payload)
    before = {path.name: path.read_bytes() for path in recovery.iterdir()}
    service = _finalization_drive_service(recovery, drive_root)
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer)

    with pytest.raises(ValueError, match="SHA-256 differs from its manifest"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == before
    assert service.deleted_ids == set()
    assert not (checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json").exists()


def test_finalization_preflights_all_task_boundaries_before_pruning_recovery(tmp_path, monkeypatch):
    _allow_test_checkpoint_memory(monkeypatch)
    plan, record = _finalization_plan(tmp_path / "schedule", monkeypatch)
    drive_root = tmp_path / "mount" / "MyDrive"
    output_root = drive_root / "ReVision" / "revision_R1" / "production" / "results"
    checkpoint_root = drive_root / "ReVision" / "revision_R1" / "production" / "checkpoints"
    output = run_output_root(output_root, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root, record)
    identity = _write_completed_production_fixture(output, plan, record)
    recovery, boundaries = _write_completed_checkpoint_trees(checkpoint_namespace, identity)
    (boundaries / "generation-000001.pt").unlink()
    recovery_before = {path.name: path.read_bytes() for path in recovery.iterdir()}
    service = _finalization_drive_service(recovery, drive_root)
    finalizer = getattr(production_workflow, "finalize_completed_production_run", None)
    assert callable(finalizer)

    with pytest.raises(ValueError, match="payload is missing or size-mismatched"):
        finalizer(
            plan, record, output_root=output_root, checkpoint_root=checkpoint_root,
            api_factory=lambda: service, drive_root=drive_root,
        )

    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == recovery_before
    assert service.deleted_ids == set()
    assert not (checkpoint_root / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json").exists()

