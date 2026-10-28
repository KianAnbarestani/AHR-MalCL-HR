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


class _FakeFiles:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def list(self, *, q, **_kwargs):
        self.queries.append(q)
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name='((?:\\'|[^'])*)'", q).group(1).replace("\\'", "'")
        values = [
            record for record in self.records.values()
            if parent in record.get("parents", []) and record["name"] == name and not record.get("trashed", False)
        ]
        return _FakeRequest({"files": values})

    def get(self, *, fileId, **_kwargs):
        return _FakeRequest(self.records[fileId])


class _FakeDriveService:
    def __init__(self, records, about):
        self.records = records
        self._files = _FakeFiles(records)
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
    authenticated_user = service._about["user"]["emailAddress"]
    return {
        "id": "probe-file-id", "name": name, "mimeType": "application/octet-stream",
        "parents": [parent_id], "owners": [{"emailAddress": authenticated_user}],
        "ownedByMe": True, "quotaBytesUsed": "64", "size": "64",
    }


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
    assert owner["available_bytes"] == 9_000_000_000
    assert owner["usage_in_drive_trash_bytes"] == 10_000_000
    assert owner["folder_id"] == "results-folder"
    assert shared_editor["authenticated_user"] == "editor@example.org"
    assert shared_editor["destination_owner"] == "owner@example.org"
    assert shared_editor["destination_owned_by_authenticated_user"] is False
    assert shared_editor["new_file_owner"] == "editor@example.org"
    assert shared_editor["new_file_owned_by_authenticated_user"] is True
    assert shared_editor["quota_account_email"] == "editor@example.org"
    assert shared_editor["available_bytes"] == 9_000_000_000
    assert owner_cached["write_probe"]["probe_performed_this_check"] is False
    assert owner["quota_change_observation"] == "NO_POSITIVE_CHANGE_OBSERVED_OR_API_EVENTUAL_CONSISTENCY"
    assert owner["quota_usage_before_probe_bytes"] == 1_000_000_000
    assert owner["quota_usage_during_probe_bytes"] == 1_000_000_000
    assert owner["quota_usage_after_probe_cleanup_bytes"] == 1_000_000_000
    assert list(destination_path.iterdir()) == []


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
    with pytest.raises(StorageCapacityError, match="matrix.*required"):
        require_run_capacity({
            "drive_available_bytes": first_run_required,
            "local_available_bytes": local_required,
            "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
            "production_measurements_validated": True,
        }, projection, scope="matrix")
    remaining_drive = (
        99 * projection["drive"]["first_run_retained_bytes"]
        + projection["drive"]["matrix_temporary_checkpoint_headroom_bytes"]
        + projection["drive"]["matrix_temporary_output_headroom_bytes"]
        + DEFAULT_STORAGE_RESERVE_BYTES
    )
    require_run_capacity({
        "drive_available_bytes": remaining_drive,
        "local_available_bytes": projection["local"]["next_run_peak_with_reserve_bytes"],
        "local_existing_dataset_cache_bytes": projection["local"]["dataset_cache_upper_bytes"],
        "production_measurements_validated": True,
    }, projection, scope="matrix", completed_runs=1)
    with pytest.raises(StorageCapacityError, match="matrix.*required"):
        require_run_capacity({
            "drive_available_bytes": remaining_drive - 1,
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
        "destination_owner_verified": True, "write_capability_verified": True,
        "new_file_owner_verified": True, "quota_metadata_verified": True,
        "output_path": str(output_root.resolve()), "checkpoint_path": str(checkpoint_root.resolve()),
        "drive_available_bytes": 0, "local_available_bytes": 0,
    }
    monkeypatch.setattr(runner_module, "run_one", lambda *args, **kwargs: pytest.fail("runner must not start when capacity gate fails"))

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

