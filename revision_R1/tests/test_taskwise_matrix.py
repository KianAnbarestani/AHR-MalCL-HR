from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from revision_r1.config import RunConfig
from revision_r1.data import DatasetBundle
from revision_r1.runner import run_one
from revision_r1.state import PhaseCursor, RunState
from scripts import taskwise_matrix as matrix


def _plan():
    rows = []
    for index in range(1, 101):
        rows.append({
            "run_index": index,
            "run_key": "ember_seed42_full_hybrid" if index == 1 else f"run_{index:03d}",
            "dataset": "ember", "seed": 42, "arm": "full_hybrid" if index == 1 else "current_only",
            "config_sha256": f"{index:064x}",
        })
    return {
        "plan_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "training_source_sha256": "c" * 64,
        "run_count": 100,
        "runs": rows,
    }


def _state(task=1):
    return RunState(
        identity={"config_sha256": "d" * 64, "schedule_sha256": "b" * 64},
        cursor=PhaseCursor(task_index=task, phase="scaler" if task < 11 else "export"),
        classifier_state={"weight": torch.tensor([float(task)])},
        rng_state={"torch_cpu": torch.get_rng_state()},
    )


class _MemoryStorage:
    def __init__(self):
        self.files = {}
        self.published_file_order = []
        self.fail_tree_after_first = False

    def bind_root(self, *_args):
        return None

    def is_mutable_control_path(self, _path):
        return True

    def publish_file(self, local_path, logical_path, **_kwargs):
        payload = open(local_path, "rb").read()
        self.files[str(logical_path)] = payload
        self.published_file_order.append(str(logical_path))
        return {"status": "PASS", "bytes": len(payload), "file_id": f"id-{len(self.files)}"}

    def download_file(self, logical_path, local_path, **_kwargs):
        key = str(logical_path)
        if key not in self.files:
            raise FileNotFoundError(key)
        from pathlib import Path
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[key])
        return {"path": str(target), "bytes": len(self.files[key])}

    def publish_tree(self, local_root, logical_root, *, include=None):
        from pathlib import Path
        local_root, logical_root = Path(local_root), Path(logical_root)
        published = []
        for path in sorted(item for item in local_root.rglob("*") if item.is_file()):
            logical = logical_root / path.relative_to(local_root)
            if include is not None and not include(logical, path):
                continue
            published.append(self.publish_file(path, logical))
            if self.fail_tree_after_first:
                self.fail_tree_after_first = False
                raise OSError("simulated interruption during result publication")
        return published


def test_progress_imports_run1_and_keeps_all_100_runs_in_frozen_order():
    plan = _plan()
    paths = [f"/drive/run1/task-{task:02d}.pt" for task in range(1, 12)]
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=paths,
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    assert len(progress["runs"]) == 100
    assert [row["run_key"] for row in progress["runs"]] == [row["run_key"] for row in plan["runs"]]
    assert all(row["run_complete"] for row in progress["runs"][:1])
    assert all(task["done"] and task["checkpoint_path"] for task in progress["runs"][0]["tasks"])
    assert not progress["runs"][1]["run_complete"]
    assert all(not task["done"] and task["checkpoint_path"] is None for task in progress["runs"][1]["tasks"])
    assert matrix.first_unfinished_run(progress)["run_key"] == "run_002"
    matrix.validate_matrix_progress(progress, plan)
    assert matrix.next_task_id(progress["runs"][1]) == 1


def test_checkpoint_is_published_before_progress_and_retry_reuses_its_path(tmp_path):
    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    storage = _MemoryStorage()
    progress_path = tmp_path / "matrix_progress.json"
    logical_progress = "/drive/matrix_progress.json"
    matrix.atomic_write_json(progress_path, progress)
    storage.files[logical_progress] = progress_path.read_bytes()

    def fail_between_checkpoint_and_progress(stage, _context):
        if stage == "after_checkpoint_published":
            raise RuntimeError("simulated process loss before progress update")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        matrix.persist_task_completion(
            progress, run_index=2, task_id=1, state=_state(),
            checkpoint_path="/drive/taskwise/run-002/task-01.pt",
            local_checkpoint_path=tmp_path / "task-01.pt", progress_path=progress_path,
            logical_progress_path=logical_progress, storage_adapter=storage,
            fault_hook=fail_between_checkpoint_and_progress,
        )
    assert "/drive/taskwise/run-002/task-01.pt" in storage.files
    assert not progress["runs"][1]["tasks"][0]["done"]
    assert json.loads(storage.files[logical_progress]) == progress

    matrix.persist_task_completion(
        progress, run_index=2, task_id=1, state=_state(),
        checkpoint_path="/drive/taskwise/run-002/task-01.pt",
        local_checkpoint_path=tmp_path / "task-01.pt", progress_path=progress_path,
        logical_progress_path=logical_progress, storage_adapter=storage,
    )
    task = progress["runs"][1]["tasks"][0]
    assert task["done"] is True
    assert task["checkpoint_path"] == "/drive/taskwise/run-002/task-01.pt"
    restored = matrix.load_task_checkpoint(
        task, local_checkpoint_path=tmp_path / "restored.pt", storage_adapter=storage,
    )
    assert restored.cursor.task_index == 1
    assert torch.equal(restored.classifier_state["weight"], torch.tensor([1.0]))


def test_progress_already_published_survives_interruption_after_json_update(tmp_path):
    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    storage = _MemoryStorage()
    progress_path = tmp_path / "matrix_progress.json"
    logical_progress = "/drive/matrix_progress.json"
    matrix.atomic_write_json(progress_path, progress)
    storage.files[logical_progress] = progress_path.read_bytes()

    def fail_after_json(stage, _context):
        if stage == "after_progress_published":
            raise RuntimeError("simulated loss after atomic progress commit")

    with pytest.raises(RuntimeError, match="after atomic progress commit"):
        matrix.persist_task_completion(
            progress, run_index=2, task_id=1, state=_state(),
            checkpoint_path="/drive/taskwise/run-002/task-01.pt",
            local_checkpoint_path=tmp_path / "task-01.pt", progress_path=progress_path,
            logical_progress_path=logical_progress, storage_adapter=storage, fault_hook=fail_after_json,
        )
    fresh = matrix.read_canonical_progress(
        tmp_path / "fresh.json", logical_progress_path=logical_progress, storage_adapter=storage,
    )
    assert fresh["runs"][1]["tasks"][0]["done"] is True
    assert matrix.first_unfinished_run(fresh)["run_key"] == "run_002"
    assert matrix.first_unfinished_task(fresh["runs"][1])["task_id"] == 2


def test_all_task_checkpoints_with_incomplete_run_selects_task11_export_retry():
    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    run = progress["runs"][1]
    for task in run["tasks"]:
        task.update({"done": True, "checkpoint_path": f"/drive/run2/task-{task['task_id']:02d}.pt"})
    assert run["run_complete"] is False
    assert matrix.next_task_id(run) == 11
    run["run_complete"] = True
    assert matrix.next_task_id(run) is None


def _tiny_bundle():
    rng = np.random.default_rng(4)
    x_train = np.vstack([rng.normal(class_id / 20, 0.2, size=(2, 4)) for class_id in range(100)]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([rng.normal(class_id / 20, 0.2, size=(1, 4)) for class_id in range(100)]).astype(np.float32)
    y_test = np.arange(100, dtype=np.int64)
    marker = {"sha256": "synthetic", "md5": "synthetic"}
    return DatasetBundle("ember", x_train, y_train, x_test, y_test, marker, marker)


def test_task11_checkpoint_retries_export_without_retraining(tmp_path, monkeypatch):
    from revision_r1 import runner

    config = RunConfig(
        dataset="ember", seed=42, arm="current_only", update_schedule_sha256="f" * 64,
        anchor_k=0, classifier_hidden=(8, 8, 4), classifier_embedding_dim=8,
        dropout=0.0, generated_candidates_per_class=2, generated_k=1,
    )
    bundle = _tiny_bundle()
    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    storage = _MemoryStorage()
    progress_path = tmp_path / "matrix_progress.json"
    logical_progress = "/drive/matrix_progress.json"
    matrix.atomic_write_json(progress_path, progress)
    storage.files[logical_progress] = progress_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoints"
    output = tmp_path / "run-output"
    paths = [f"/drive/task-{task:02d}.pt" for task in range(1, 12)]
    saves = []

    def save_boundary(state, task_id):
        saves.append(task_id)
        matrix.persist_task_completion(
            progress, run_index=2, task_id=task_id, state=state,
            checkpoint_path=paths[task_id - 1],
            local_checkpoint_path=checkpoint_dir / f"task-{task_id:02d}.pt",
            progress_path=progress_path, logical_progress_path=logical_progress,
            storage_adapter=storage,
        )

    original_export = runner._export_run
    fail_once = {"enabled": True}

    def interrupted_export(*args, **kwargs):
        if fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("simulated Task 11 result-export interruption")
        return original_export(*args, **kwargs)

    monkeypatch.setattr(runner, "_export_run", interrupted_export)
    with pytest.raises(RuntimeError, match="Task 11 result-export interruption"):
        run_one(
            config, bundle, output, synthetic=True, update_counts=[1] * 11, device="cpu",
            taskwise_mode=True, taskwise_checkpoint_callback=save_boundary,
        )
    assert saves == list(range(1, 12))
    assert all(item["done"] for item in progress["runs"][1]["tasks"])
    last = matrix.load_task_checkpoint(
        progress["runs"][1]["tasks"][10], local_checkpoint_path=tmp_path / "task11-resume.pt",
        storage_adapter=storage,
    )
    assert last.cursor.task_index == 11 and last.cursor.phase == "export"

    resumed = run_one(
        config, bundle, output, synthetic=True, update_counts=[1] * 11, device="cpu",
        taskwise_mode=True, taskwise_resume_state=last,
        taskwise_checkpoint_callback=save_boundary,
    )
    assert resumed["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    assert resumed["classifier_updates"] == last.resources["classifier_updates"]
    assert saves == list(range(1, 12))
    assert (output / "artifact_manifest.json").is_file()


def test_drive_storage_allows_only_taskwise_payload_and_single_matrix_progress_updates():
    from pathlib import Path
    from revision_r1.drive_storage import DriveApiStorage

    adapter = object.__new__(DriveApiStorage)
    adapter.logical_revision_root = Path("/content/drive/MyDrive/ReVision")
    base = adapter.logical_revision_root / "revision_R1" / "production"
    assert adapter.is_mutable_control_path(base / "results" / "matrix_progress.json")
    assert adapter.is_mutable_control_path(
        base / "checkpoints" / "taskwise_v1" / ("a" * 64)
        / ("002_ember_seed42_current_only_" + "b" * 16) / "task-11.pt"
    )
    assert adapter.is_mutable_control_path(
        base / "results" / "taskwise_v1" / ("a" * 64) / "runs"
        / ("002_ember_seed42_current_only_" + "b" * 16) / "artifact_manifest.json"
    )
    assert not adapter.is_mutable_control_path(base / "checkpoints" / "legacy" / "task-11.pt")
    assert not adapter.is_mutable_control_path(base / "results" / "some_other_progress.json")


def test_matrix_runner_retries_task11_export_after_partial_result_publication(tmp_path, monkeypatch):
    from scripts import production_workflow

    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    # Test the first pending run in isolation while preserving the real 100-row order.
    for row in progress["runs"][2:]:
        row["run_complete"] = True
        for task in row["tasks"]:
            task.update({"done": True, "checkpoint_path": f"/drive/fixture/{row['run_index']}/{task['task_id']}"})
    storage = _MemoryStorage()
    progress_path = tmp_path / "matrix_progress.json"
    logical_progress = "/drive/matrix_progress.json"
    matrix.atomic_write_json(progress_path, progress)
    storage.files[logical_progress] = progress_path.read_bytes()
    storage.fail_tree_after_first = True
    monkeypatch.setattr(production_workflow, "validate_runtime_bound_plan", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner_calls = []
    output_root = tmp_path / "results"
    checkpoint_root = tmp_path / "checkpoints"
    logical_output_root = "/drive/results"
    logical_checkpoint_root = "/drive/checkpoints"

    def fake_runner(_config, _bundle, output, **kwargs):
        resume_state = kwargs["taskwise_resume_state"]
        runner_calls.append(None if resume_state is None else (resume_state.cursor.task_index, resume_state.cursor.phase))
        output.mkdir(parents=True, exist_ok=True)
        if resume_state is None:
            record = plan["runs"][1]
            for task_id in range(1, 12):
                state = RunState(
                    identity={
                        "config_sha256": record["config_sha256"],
                        "schedule_sha256": plan["schedule_sha256"],
                        "code_sha256": plan["training_source_sha256"],
                    },
                    cursor=PhaseCursor(task_index=task_id, phase="export" if task_id == 11 else "scaler"),
                    classifier_state={"weight": torch.tensor([float(task_id)])},
                    rng_state={"torch_cpu": torch.get_rng_state()},
                )
                kwargs["taskwise_checkpoint_callback"](state, task_id)
        (output / "full-result.json").write_text('{"fixture":true}\n')
        (output / "artifact_manifest.json").write_text('{"fixture":"complete"}\n')
        return {"status": "COMPLETED_VALIDATED", "validated": True}

    common = dict(
        plan=plan, progress_path=progress_path, logical_progress_path=logical_progress,
        data_roots={"ember": tmp_path, "az_class": tmp_path}, cache_root=tmp_path / "cache",
        preflight={}, schedule_path=tmp_path / "schedule.csv", schedule_manifest_path=tmp_path / "schedule.json",
        output_root=output_root, logical_output_root=logical_output_root,
        checkpoint_root=checkpoint_root, logical_checkpoint_root=logical_checkpoint_root,
        storage_adapter=storage, launch_enabled=True, acknowledge_single_worker=True,
        bundle_loader=lambda *_args, **_kwargs: _tiny_bundle(), runner_fn=fake_runner,
    )
    with pytest.raises(OSError, match="simulated interruption during result publication"):
        matrix.run_taskwise_matrix(**common)
    remote_after_failure = json.loads(storage.files[logical_progress])
    assert all(task["done"] for task in remote_after_failure["runs"][1]["tasks"])
    assert remote_after_failure["runs"][1]["run_complete"] is False
    storage.published_file_order.clear()

    result = matrix.run_taskwise_matrix(**common)
    assert result["status"] == "MATRIX_COMPLETE"
    assert runner_calls == [None, (11, "export")]
    recovered = json.loads(storage.files[logical_progress])
    assert recovered["runs"][1]["run_complete"] is True
    result_files = [path for path in storage.published_file_order if "/results/taskwise_v1/" in path]
    assert result_files[-1].endswith("/artifact_manifest.json")


def test_missing_recorded_checkpoint_reports_run_task_and_path(tmp_path, monkeypatch):
    from scripts import production_workflow

    plan = _plan()
    progress = matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f"/drive/run1/task-{i:02d}.pt" for i in range(1, 12)],
        run1_registry_evidence={"run_id": "verified-run-1", "artifact_manifest_sha256": "e" * 64},
    )
    run = progress["runs"][1]
    expected_path = (
        "/drive/checkpoints/taskwise_v1/" + plan["plan_sha256"]
        + "/002_run_002_0000000000000000/task-01.pt"
    )
    run["tasks"][0].update({"done": True, "checkpoint_path": expected_path})
    storage = _MemoryStorage()
    progress_path = tmp_path / "matrix_progress.json"
    matrix.atomic_write_json(progress_path, progress)
    storage.files["/drive/matrix_progress.json"] = progress_path.read_bytes()
    monkeypatch.setattr(production_workflow, "validate_runtime_bound_plan", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError) as error:
        matrix.run_taskwise_matrix(
            plan, progress_path=progress_path, logical_progress_path="/drive/matrix_progress.json",
            data_roots={"ember": tmp_path, "az_class": tmp_path}, cache_root=tmp_path / "cache",
            preflight={}, schedule_path=tmp_path / "schedule.csv", schedule_manifest_path=tmp_path / "schedule.json",
            output_root=tmp_path / "results", logical_output_root="/drive/results",
            checkpoint_root=tmp_path / "checkpoints", logical_checkpoint_root="/drive/checkpoints",
            storage_adapter=storage, launch_enabled=True, acknowledge_single_worker=True,
            bundle_loader=lambda *_args, **_kwargs: _tiny_bundle(), runner_fn=lambda *_args, **_kwargs: pytest.fail("runner must not start"),
        )
    assert "run run_002, task 1" in str(error.value)
    assert expected_path in str(error.value)
