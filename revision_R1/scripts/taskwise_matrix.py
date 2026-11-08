"""One-progress-file, one-checkpoint-per-task production orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from revision_r1.state import RunState


PROGRESS_NAME = "matrix_progress.json"
TASK_COUNT = 11
RUN_COUNT = 100
TASKWISE_NAMESPACE = "taskwise_v1"


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        try:
            directory_fd = os.open(target.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def create_matrix_progress(
    plan: dict[str, Any], *, run1_checkpoint_paths: list[str],
    run1_registry_evidence: dict[str, Any],
) -> dict[str, Any]:
    runs = plan.get("runs")
    if (plan.get("run_count") != RUN_COUNT or not isinstance(runs, list)
            or len(runs) != RUN_COUNT or runs[0].get("run_key") != "ember_seed42_full_hybrid"):
        raise ValueError("matrix progress requires the exact ordered 100-run frozen plan")
    if len(run1_checkpoint_paths) != TASK_COUNT or len(set(run1_checkpoint_paths)) != TASK_COUNT:
        raise ValueError("Run 1 import requires eleven distinct existing task-boundary checkpoint paths")
    if not run1_registry_evidence.get("run_id") or not run1_registry_evidence.get("artifact_manifest_sha256"):
        raise ValueError("Run 1 import requires its existing validated registry identity")
    entries = []
    for index, row in enumerate(runs, start=1):
        imported = index == 1
        entries.append({
            "run_index": index,
            "run_key": row["run_key"],
            "config_sha256": row["config_sha256"],
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "arm": row["arm"],
            "run_complete": imported,
            "tasks": [
                {
                    "task_id": task_id,
                    "done": imported,
                    "checkpoint_path": run1_checkpoint_paths[task_id - 1] if imported else None,
                    **({"provenance": "imported_from_existing_completed_run_registry"} if imported else {}),
                }
                for task_id in range(1, TASK_COUNT + 1)
            ],
            **({"completion_provenance": {
                "kind": "existing_registry_entry",
                "run_id": run1_registry_evidence["run_id"],
                "artifact_manifest_sha256": run1_registry_evidence["artifact_manifest_sha256"],
            }} if imported else {}),
        })
    return {
        "schema_version": 1,
        "progress_kind": "TASKWISE_MATRIX_PROGRESS",
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "run_count": RUN_COUNT,
        "task_count": TASK_COUNT,
        "checkpoint_policy": "one_full_state_checkpoint_after_each_completed_task",
        "runs": entries,
    }


def validate_matrix_progress(progress: dict[str, Any], plan: dict[str, Any]) -> None:
    if not isinstance(progress, dict) or (
        progress.get("schema_version") != 1
        or progress.get("progress_kind") != "TASKWISE_MATRIX_PROGRESS"
        or progress.get("plan_sha256") != plan.get("plan_sha256")
        or progress.get("schedule_sha256") != plan.get("schedule_sha256")
        or progress.get("training_source_sha256") != plan.get("training_source_sha256")
        or progress.get("run_count") != RUN_COUNT
        or progress.get("task_count") != TASK_COUNT
        or not isinstance(progress.get("runs"), list)
        or len(progress["runs"]) != RUN_COUNT
    ):
        raise ValueError("matrix_progress.json is malformed or bound to different frozen identities")
    for index, (item, record) in enumerate(zip(progress["runs"], plan["runs"]), start=1):
        if any(item.get(field) != expected for field, expected in (
            ("run_index", index), ("run_key", record["run_key"]),
            ("config_sha256", record["config_sha256"]), ("dataset", record["dataset"]),
            ("seed", int(record["seed"])), ("arm", record["arm"]),
        )):
            raise ValueError(f"matrix_progress.json run order/identity mismatch at run {index}")
        tasks = item.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != TASK_COUNT:
            raise ValueError(f"matrix_progress.json must contain eleven task entries for run {index}")
        first_pending = None
        for task_id, task in enumerate(tasks, start=1):
            if task.get("task_id") != task_id or type(task.get("done")) is not bool:
                raise ValueError(f"matrix_progress.json task identity/state is invalid at run {index}, task {task_id}")
            if task["done"]:
                if not isinstance(task.get("checkpoint_path"), str) or not task["checkpoint_path"]:
                    raise ValueError(f"completed task has no checkpoint path at run {index}, task {task_id}")
                if first_pending is not None:
                    raise ValueError(f"task completion is non-contiguous at run {index}, task {task_id}")
                sha = task.get("checkpoint_sha256")
                size = task.get("checkpoint_bytes")
                if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
                    raise ValueError(f"checkpoint SHA-256 is malformed at run {index}, task {task_id}")
                if size is not None and (type(size) is not int or size < 1):
                    raise ValueError(f"checkpoint size is malformed at run {index}, task {task_id}")
            else:
                if task.get("checkpoint_path") is not None:
                    raise ValueError(f"unfinished task has a checkpoint path at run {index}, task {task_id}")
                if first_pending is None:
                    first_pending = task_id
        if type(item.get("run_complete")) is not bool:
            raise ValueError(f"run completion state is malformed at run {index}")
        if item["run_complete"] and (first_pending is not None or not all(task["done"] for task in tasks)):
            raise ValueError(f"completed run has unfinished tasks at run {index}")
    if not progress["runs"][0]["run_complete"]:
        raise ValueError("Run 1 must remain imported as already complete from its existing registry")


def first_unfinished_run(progress: dict[str, Any]) -> dict[str, Any] | None:
    return next((row for row in progress["runs"] if not row["run_complete"]), None)


def first_unfinished_task(run: dict[str, Any]) -> dict[str, Any] | None:
    return next((task for task in run["tasks"] if not task["done"]), None)


def next_task_id(run: dict[str, Any]) -> int | None:
    """Return the next training task, or Task 11 for an export-only retry."""
    if run.get("run_complete") is True:
        return None
    pending = first_unfinished_task(run)
    if pending is not None:
        return int(pending["task_id"])
    if len(run.get("tasks", [])) == TASK_COUNT and all(item.get("done") is True for item in run["tasks"]):
        return TASK_COUNT
    raise ValueError(f"incomplete run {run.get('run_key')} has no valid next task")


def atomic_save_state(path: str | Path, state: RunState) -> tuple[int, str]:
    state.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        try:
            directory_fd = os.open(target.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        return target.stat().st_size, digest.hexdigest()
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def persist_task_completion(
    progress: dict[str, Any], *, run_index: int, task_id: int, state: RunState,
    checkpoint_path: str, local_checkpoint_path: str | Path,
    progress_path: str | Path, logical_progress_path: str | Path,
    storage_adapter=None, fault_hook=None,
) -> dict[str, Any]:
    if not 1 <= int(run_index) <= RUN_COUNT or not 1 <= int(task_id) <= TASK_COUNT:
        raise ValueError("run/task index is outside the frozen matrix")
    run = progress["runs"][int(run_index) - 1]
    pending = first_unfinished_task(run)
    if pending is None or pending["task_id"] != int(task_id):
        raise ValueError(f"refusing out-of-order task completion at run {run_index}, task {task_id}")
    if state.cursor.task_index != int(task_id):
        raise ValueError(f"checkpoint cursor does not represent completed run {run_index}, task {task_id}")
    expected_phase = "export" if int(task_id) == TASK_COUNT else "scaler"
    if state.cursor.phase != expected_phase:
        raise ValueError(
            f"checkpoint cursor phase {state.cursor.phase!r} does not represent completed task {task_id}; "
            f"expected {expected_phase!r}"
        )
    size, digest = atomic_save_state(local_checkpoint_path, state)
    if storage_adapter is not None:
        publication = storage_adapter.publish_file(
            local_checkpoint_path, checkpoint_path,
            allow_foreign_owner_update=storage_adapter.is_mutable_control_path(checkpoint_path),
        )
        if publication.get("status") not in {"PASS", "UNCHANGED_VERIFIED_METADATA"}:
            raise RuntimeError(f"Drive did not verify task checkpoint publication: {publication}")
    task = run["tasks"][int(task_id) - 1]
    update_context = {"run_index": run_index, "run_key": run["run_key"], "task_id": task_id,
                      "checkpoint_path": checkpoint_path, "checkpoint_bytes": size, "checkpoint_sha256": digest}
    if fault_hook is not None:
        fault_hook("after_checkpoint_published", update_context)
    task.update({"done": True, "checkpoint_path": str(checkpoint_path),
                 "checkpoint_bytes": size, "checkpoint_sha256": digest})
    atomic_write_json(progress_path, progress)
    if storage_adapter is not None:
        publication = storage_adapter.publish_file(
            progress_path, logical_progress_path,
            allow_foreign_owner_update=storage_adapter.is_mutable_control_path(logical_progress_path),
        )
        if publication.get("status") not in {"PASS", "UNCHANGED_VERIFIED_METADATA"}:
            raise RuntimeError(f"Drive did not verify matrix progress publication: {publication}")
    if fault_hook is not None:
        fault_hook("after_progress_published", update_context)
    return update_context


def read_canonical_progress(local_path: str | Path, *, logical_progress_path: str | Path, storage_adapter=None) -> dict[str, Any]:
    path = Path(local_path)
    if storage_adapter is not None:
        storage_adapter.download_file(logical_progress_path, path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"canonical matrix progress is unavailable: {logical_progress_path} -> {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"matrix progress is unreadable: {logical_progress_path} -> {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"matrix progress root is not an object: {logical_progress_path}")
    return value


def load_task_checkpoint(
    task: dict[str, Any], *, local_checkpoint_path: str | Path, storage_adapter=None,
) -> RunState:
    logical = task.get("checkpoint_path")
    run_task = f"run task checkpoint {logical!r}"
    if not isinstance(logical, str) or not logical:
        raise ValueError(f"{run_task} has no recorded path")
    local = Path(local_checkpoint_path)
    try:
        if storage_adapter is not None:
            storage_adapter.download_file(
                logical, local,
                expected_sha256=task.get("checkpoint_sha256"),
                expected_bytes=task.get("checkpoint_bytes"),
            )
        if local.is_symlink() or not local.is_file():
            raise FileNotFoundError(local)
        if task.get("checkpoint_bytes") is not None and local.stat().st_size != task["checkpoint_bytes"]:
            raise ValueError("checkpoint byte count differs from matrix_progress.json")
        if task.get("checkpoint_sha256") is not None:
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            if digest != task["checkpoint_sha256"]:
                raise ValueError("checkpoint SHA-256 differs from matrix_progress.json")
        state = torch.load(local, map_location="cpu", weights_only=False)
        if not isinstance(state, RunState):
            raise TypeError(f"checkpoint contains {type(state).__name__}, not RunState")
        state.validate()
        expected_task = int(task["task_id"])
        phase = "export" if expected_task == TASK_COUNT else "scaler"
        if state.cursor.task_index != expected_task or state.cursor.phase != phase:
            raise ValueError(
                f"checkpoint cursor is task={state.cursor.task_index}, phase={state.cursor.phase!r}; "
                f"expected task={expected_task}, phase={phase!r}"
            )
        return state
    except Exception as exc:
        raise RuntimeError(
            f"cannot load checkpoint for run {task.get('run_key', '<unknown>')}, "
            f"task {task.get('task_id', '<unknown>')}, path {logical}: {type(exc).__name__}: {exc}"
        ) from exc


def publish_taskwise_results(storage_adapter, local_root: str | Path, logical_root: str | Path) -> list[dict[str, Any]]:
    """Publish result files first and the completion manifest last."""
    local = Path(local_root)
    logical = Path(logical_root)
    manifest = local / "artifact_manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise FileNotFoundError(f"validated result artifact manifest is missing: {manifest}")
    expected_count = sum(
        1 for path in local.rglob("*")
        if path.is_file() and path != manifest
    )
    if expected_count < 1:
        raise ValueError(f"result tree has no artifacts to publish before its manifest: {local}")
    results = storage_adapter.publish_tree(
        local, logical,
        include=lambda remote_path, _local_path: Path(remote_path).name != "artifact_manifest.json",
    )
    if len(results) != expected_count:
        raise RuntimeError(
            f"result publication count differs from local artifact tree: expected {expected_count}, got {len(results)}"
        )
    accepted = {"PASS", "UNCHANGED_VERIFIED_METADATA"}
    failures = [item for item in results if item.get("status") not in accepted]
    if failures:
        raise RuntimeError(f"Drive did not verify all taskwise result artifacts: {failures}")
    manifest_result = storage_adapter.publish_file(
        manifest, logical / manifest.name,
        allow_foreign_owner_update=storage_adapter.is_mutable_control_path(logical / manifest.name),
    )
    if manifest_result.get("status") not in accepted:
        raise RuntimeError(f"Drive did not verify the taskwise result manifest: {manifest_result}")
    return [*results, manifest_result]


def discover_run1_task_checkpoint_paths(storage_adapter, logical_checkpoint_root: str | Path, run1_record: dict[str, Any]) -> list[str]:
    # Keep the canonical MyDrive lexical path. Path.resolve() can turn the
    # shared-folder shortcut into .shortcut-targets-by-id, invalid for API traversal.
    logical_dir = (
        Path(os.path.abspath(str(Path(logical_checkpoint_root).expanduser())))
        / "revision_R1.0.2" / str(run1_record["training_source_sha256"])[:16]
        / str(run1_record["dataset"]) / f"seed_{int(run1_record['seed'])}"
        / str(run1_record["arm"]) / str(run1_record["config_sha256"])[:16]
        / "task_boundaries"
    )
    files = storage_adapter.list_files(logical_dir)
    payloads = [item for item in files if re.fullmatch(r"generation-\d+\.pt", str(item.get("name", "")))]
    manifests = [item for item in files if re.fullmatch(r"generation-\d+\.manifest\.json", str(item.get("name", "")))]
    if len(payloads) != TASK_COUNT or len(manifests) != TASK_COUNT:
        raise ValueError(
            "Run 1 registry import requires eleven existing task-boundary checkpoint payload/manifest pairs; "
            f"found payloads={len(payloads)}, manifests={len(manifests)} at {logical_dir}"
        )
    payload_names = {str(item["name"]) for item in payloads}
    manifest_generations = {
        re.fullmatch(r"generation-(\d+)\.manifest\.json", str(item["name"])).group(1)
        for item in manifests
    }
    payload_generations = {
        re.fullmatch(r"generation-(\d+)\.pt", name).group(1) for name in payload_names
    }
    if payload_generations != manifest_generations:
        raise ValueError(f"Run 1 task-boundary payload/manifest generations do not pair at {logical_dir}")
    for item in payloads:
        if int(item.get("size", 0)) <= 0 or not item.get("id"):
            raise ValueError(f"Run 1 task-boundary payload metadata is incomplete: {item.get('name')}")
    return [str(logical_dir / name) for name in sorted(payload_names, key=lambda name: int(name.split("-")[1].split(".")[0]))]


def prepare_matrix_progress(
    plan: dict[str, Any], *, local_progress_path: str | Path,
    logical_progress_path: str | Path, storage_adapter,
    run1_checkpoint_paths: list[str] | None = None,
    run1_registry_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local = Path(local_progress_path)
    try:
        progress = read_canonical_progress(
            local, logical_progress_path=logical_progress_path, storage_adapter=storage_adapter,
        )
    except FileNotFoundError:
        if run1_checkpoint_paths is None or run1_registry_evidence is None:
            raise ValueError("matrix_progress.json is absent and verified Run 1 registry import evidence was not supplied")
        if local.exists() or local.is_symlink():
            raise ValueError(
                "canonical matrix_progress.json is absent but a local staging copy exists; preserving it instead of "
                "silently initializing a new matrix"
            )
        progress = create_matrix_progress(
            plan, run1_checkpoint_paths=run1_checkpoint_paths,
            run1_registry_evidence=run1_registry_evidence,
        )
        atomic_write_json(local, progress)
        publication = storage_adapter.publish_file(
            local, logical_progress_path,
            allow_foreign_owner_update=storage_adapter.is_mutable_control_path(logical_progress_path),
        )
        if publication.get("status") not in {"PASS", "UNCHANGED_VERIFIED_METADATA"}:
            raise RuntimeError(f"Drive did not verify initial matrix progress publication: {publication}")
    validate_matrix_progress(progress, plan)
    return progress


def _record_directory(record: dict[str, Any]) -> str:
    return f"{int(record['run_index']):03d}_{record['run_key']}_{record['config_sha256'][:16]}"


def _load_bundle(dataset: str, *, data_roots, cache_root, preflight):
    from revision_r1.config import DATASETS
    from revision_r1.data import DatasetBundle, load_npz_arrays
    from scripts.production_workflow import _load_verified_cache_split, _verified_cache_candidate

    spec = DATASETS[dataset]
    outputs = {}
    for split in ("train", "test"):
        if _verified_cache_candidate(cache_root, dataset, split, preflight) is not None:
            outputs[split] = _load_verified_cache_split(cache_root, dataset, split, preflight)
            continue
        metadata_spec = spec[split]
        outputs[split] = load_npz_arrays(
            Path(data_roots[dataset]) / metadata_spec["filename"],
            split=split, expected_md5=metadata_spec["md5"],
            expected_shape=metadata_spec["shape"], expected_labels=range(100),
            cache_dir=cache_root,
        )
    train_x, train_y, train_meta = outputs["train"]
    test_x, test_y, test_meta = outputs["test"]
    return DatasetBundle(dataset, train_x, train_y, test_x, test_y, train_meta, test_meta)


def run_taskwise_matrix(
    plan: dict[str, Any], *, progress_path: str | Path, logical_progress_path: str | Path,
    data_roots, cache_root: str | Path, preflight: dict[str, Any],
    schedule_path: str | Path, schedule_manifest_path: str | Path,
    output_root: str | Path, logical_output_root: str | Path,
    checkpoint_root: str | Path, logical_checkpoint_root: str | Path,
    storage_adapter, launch_enabled: bool = False, acknowledge_single_worker: bool = False,
    device: str = "cuda", progress_callback=None, fault_hook=None,
    bundle_loader=None, runner_fn=None,
) -> dict[str, Any]:
    """Run/resume the first unfinished frozen run, persisting only task boundaries."""
    from revision_r1 import runner as runner_module
    from revision_r1.config import make_run_config
    from scripts.production_workflow import validate_runtime_bound_plan

    validate_runtime_bound_plan(plan, schedule_path, schedule_manifest_path, preflight)
    progress = read_canonical_progress(
        progress_path, logical_progress_path=logical_progress_path, storage_adapter=storage_adapter,
    )
    validate_matrix_progress(progress, plan)
    if not launch_enabled:
        return {"status": "NOT_LAUNCHED", "next_run": first_unfinished_run(progress), "progress_path": str(logical_progress_path)}
    if acknowledge_single_worker is not True:
        raise RuntimeError("explicitly confirm only one Colab worker is using this matrix")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production matrix execution requires an available CUDA device")
    storage_adapter.bind_root(logical_output_root, output_root)
    storage_adapter.bind_root(logical_checkpoint_root, checkpoint_root)
    root = Path(output_root)
    checkpoint_base = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    callback = progress_callback or (lambda _event: None)
    load_bundle = bundle_loader or _load_bundle
    run = first_unfinished_run(progress)
    while run is not None:
        next_task = next_task_id(run)
        if next_task is None:
            raise ValueError(f"run {run['run_key']} was selected despite being complete")
        record = plan["runs"][int(run["run_index"]) - 1]
        task = first_unfinished_task(run)
        n_done = int(task["task_id"]) - 1 if task is not None else TASK_COUNT
        run_dir_name = _record_directory(record)
        output = root / TASKWISE_NAMESPACE / plan["plan_sha256"] / "runs" / run_dir_name
        local_run_checkpoint_root = checkpoint_base / TASKWISE_NAMESPACE / plan["plan_sha256"] / run_dir_name
        bundle = load_bundle(record["dataset"], data_roots=data_roots, cache_root=cache_root, preflight=preflight)
        config = make_run_config(record["dataset"], int(record["seed"]), record["arm"], plan["schedule_sha256"])
        state = None
        if n_done:
            last = run["tasks"][n_done - 1]
            expected_local_path = local_run_checkpoint_root / f"task-{n_done:02d}.pt"
            expected_logical_path = Path(logical_checkpoint_root) / TASKWISE_NAMESPACE / plan["plan_sha256"] / run_dir_name / expected_local_path.name
            if last.get("checkpoint_path") != str(expected_logical_path):
                raise ValueError(
                    f"recorded checkpoint path differs from the deterministic route for run {run['run_key']}, "
                    f"task {n_done}: {last.get('checkpoint_path')} != {expected_logical_path}"
                )
            state = load_task_checkpoint(
                {**last, "run_key": run["run_key"]},
                local_checkpoint_path=expected_local_path,
                storage_adapter=storage_adapter,
            )
            if (state.identity.get("config_sha256") != record["config_sha256"]
                    or state.identity.get("schedule_sha256") != plan["schedule_sha256"]
                    or state.identity.get("code_sha256") != plan["training_source_sha256"]):
                raise RuntimeError(
                    f"checkpoint identity mismatch for run {run['run_key']}, task {n_done}, path {last['checkpoint_path']}"
                )

        def save_boundary(boundary_state, task_id: int):
            local_path = local_run_checkpoint_root / f"task-{int(task_id):02d}.pt"
            logical_path = Path(logical_checkpoint_root) / TASKWISE_NAMESPACE / plan["plan_sha256"] / run_dir_name / local_path.name
            result = persist_task_completion(
                progress, run_index=int(run["run_index"]), task_id=int(task_id), state=boundary_state,
                checkpoint_path=str(logical_path), local_checkpoint_path=local_path,
                progress_path=progress_path, logical_progress_path=logical_progress_path,
                storage_adapter=storage_adapter, fault_hook=fault_hook,
            )
            callback({"status": "TASK_CHECKPOINT_SAVED", "run_index": run["run_index"],
                      "run_key": run["run_key"], "task": task_id, **result})

        execute = runner_fn or runner_module.run_one
        result = execute(
            config, bundle, output, update_schedule_path=schedule_path,
            synthetic=False, device=device, taskwise_mode=True,
            taskwise_resume_state=state, taskwise_checkpoint_callback=save_boundary,
            progress_callback=lambda current: callback({
                "status": "RUNNING", "run_index": run["run_index"], "run_key": run["run_key"],
                "task": min(TASK_COUNT, int(current.cursor.task_index) + 1),
                "phase": current.cursor.phase,
                "classifier_updates": int(current.resources.get("classifier_updates", 0)),
            }),
        )
        if result.get("status") == "INTERRUPTED_TASK":
            return {"status": "INTERRUPTED_TASK", "run_key": run["run_key"],
                    "cursor": result.get("cursor"), "progress_path": str(logical_progress_path)}
        if result.get("validated") is not True or result.get("status") != "COMPLETED_VALIDATED":
            raise RuntimeError(f"runner did not validate completed outputs for run {run['run_key']}: {result}")
        logical_run_output = Path(logical_output_root) / TASKWISE_NAMESPACE / plan["plan_sha256"] / "runs" / run_dir_name
        callback({"status": "PUBLISHING_RESULTS", "run_index": run["run_index"], "run_key": run["run_key"]})
        publish_taskwise_results(storage_adapter, output, logical_run_output)
        callback({"status": "RESULTS_PUBLISHED", "run_index": run["run_index"], "run_key": run["run_key"]})
        run["run_complete"] = True
        run["completion_provenance"] = {"kind": "taskwise_runner", "status": result["status"]}
        atomic_write_json(progress_path, progress)
        completion_publication = storage_adapter.publish_file(
            progress_path, logical_progress_path,
            allow_foreign_owner_update=storage_adapter.is_mutable_control_path(logical_progress_path),
        )
        if completion_publication.get("status") not in {"PASS", "UNCHANGED_VERIFIED_METADATA"}:
            raise RuntimeError(f"Drive did not verify completed-run progress publication: {completion_publication}")
        callback({"status": "RUN_COMPLETE", "run_index": run["run_index"], "run_key": run["run_key"]})
        run = first_unfinished_run(progress)
    return {"status": "MATRIX_COMPLETE", "completed_runs": RUN_COUNT,
            "progress_path": str(logical_progress_path)}
