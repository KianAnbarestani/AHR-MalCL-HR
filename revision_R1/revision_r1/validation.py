"""Fail-closed result validation before completion markers or reportable exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARTIFACTS = (
    "config.json", "environment.json", "class_order.json", "task_matrix.json",
    "per_task_metrics.csv", "classification_report.csv", "confusion_matrix.csv",
    "predictions.npz", "task_boundary_predictions.npz", "task_boundary_prediction_manifest.json",
    "source_proportions.json", "trace.jsonl", "selection_records.jsonl",
    "classifier_mode_trace.json", "resource_metrics.json", "full-result.json", "artifact_manifest.json",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_manifest(root: str | Path, *, exclude: set[str] | None = None) -> dict:
    directory = Path(root)
    excluded = {"artifact_manifest.json", "COMPLETE.json", ".run.lease.json"} | (exclude or set())
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in excluded and path.name != "checkpoint.pt":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {"schema_version": 1, "files": files}


def validate_run_artifacts(
    root: str | Path,
    *,
    require_production: bool = False,
    allow_pending: bool = False,
    verify_hashes: bool = True,
) -> dict:
    directory = Path(root)
    if not directory.is_dir():
        if require_production:
            raise ValueError("synthetic or missing run directory cannot satisfy production validation")
        raise FileNotFoundError(directory)
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"run artifact set is incomplete: {missing}")
    try:
        summary = json.loads((directory / "full-result.json").read_text(encoding="utf-8"))
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        matrix_payload = json.loads((directory / "task_matrix.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
        class_order_payload = json.loads((directory / "class_order.json").read_text(encoding="utf-8"))
        source_rows = json.loads((directory / "source_proportions.json").read_text(encoding="utf-8"))
        mode_trace = json.loads((directory / "classifier_mode_trace.json").read_text(encoding="utf-8"))
        boundary_manifest = json.loads((directory / "task_boundary_prediction_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"run metadata is not valid JSON: {exc}") from exc
    namespace = summary.get("namespace")
    if require_production and (namespace != "PRODUCTION" or summary.get("training_reproduction_claimed") is not False):
        raise ValueError("synthetic/non-production output cannot be marked as production evidence")
    accepted_statuses = {"SYNTHETIC_VALIDATED_NON_REPORTABLE", "COMPLETED_VALIDATED"}
    if allow_pending:
        accepted_statuses.add("ARTIFACTS_WRITTEN_UNVALIDATED")
    if status.get("status") not in accepted_statuses:
        raise ValueError("run status is not a validated completion state")
    matrix = np.asarray(matrix_payload.get("task_matrix"), dtype=object)
    if matrix.shape != (11, 11):
        raise ValueError(f"task matrix must be 11x11, got {matrix.shape}")
    for stage in range(11):
        if any(matrix[stage, task] is None for task in range(stage + 1)):
            raise ValueError(f"task matrix has missing observed cell at stage {stage + 1}")
        if any(matrix[stage, task] is not None for task in range(stage + 1, 11)):
            raise ValueError(f"task matrix has non-null future cell at stage {stage + 1}")
    order = class_order_payload.get("class_order", [])
    if len(order) != 100 or sorted(order) != list(range(100)):
        raise ValueError("class-order artifact is not a permutation of 0..99")
    if len(source_rows) != 11:
        raise ValueError("source-proportion output must contain one row per task")
    training_modes = [row for row in mode_trace if row.get("phase") == "classifier_training"]
    evaluation_modes = [row for row in mode_trace if row.get("phase") == "evaluation"]
    if {int(row.get("task_id", -1)) for row in training_modes} != set(range(1, 12)):
        raise ValueError("classifier mode trace must record training mode for all 11 tasks")
    if {int(row.get("task_id", -1)) for row in evaluation_modes} != set(range(1, 12)):
        raise ValueError("classifier mode trace must record evaluation mode for all 11 tasks")
    for row in training_modes:
        if row.get("classifier_training") is not True or not row.get("dropout_training") or not all(row["dropout_training"]):
            raise ValueError(f"classifier/dropout training mode was not active in task {row.get('task_id')}")
    for row in evaluation_modes:
        if row.get("classifier_training") is not False or not row.get("dropout_training") or any(row["dropout_training"]):
            raise ValueError(f"classifier/dropout evaluation mode was not active in task {row.get('task_id')}")
    if len(boundary_manifest.get("stages", [])) != 11:
        raise ValueError("task-boundary prediction mapping must describe all 11 evaluation stages")
    if [int(row.get("stage", -1)) for row in boundary_manifest["stages"]] != list(range(1, 12)):
        raise ValueError("task-boundary prediction stages must be ordered 1 through 11")
    for row in source_rows:
        if sum(int(value) for value in row.get("realized_draws", {}).values()) != int(row.get("sampled_examples", -1)):
            raise ValueError("realized source draw counts do not sum to full-batch draw count")
    with (directory / "per_task_metrics.csv").open(newline="", encoding="utf-8") as stream:
        task_rows = list(csv.DictReader(stream))
    if len(task_rows) != 11 or {int(row["task_id"]) for row in task_rows} != set(range(1, 12)):
        raise ValueError("per-task metrics must have one unique row for each task")
    prediction_file = np.load(directory / "predictions.npz", allow_pickle=False)
    if prediction_file["y_true_global"].shape != prediction_file["y_pred_global"].shape:
        raise ValueError("final predictions and labels are not aligned")
    boundary_file = np.load(directory / "task_boundary_predictions.npz", allow_pickle=False)
    for stage in range(1, 12):
        prefix = f"stage_{stage:02d}"
        true = boundary_file[f"{prefix}_y_true_global"]
        pred = boundary_file[f"{prefix}_y_pred_global"]
        indices = boundary_file[f"{prefix}_test_indices"]
        if true.shape != pred.shape or true.shape != indices.shape:
            raise ValueError(f"task-boundary predictions are misaligned at stage {stage}")
        if len(true) != int(boundary_manifest["stages"][stage - 1].get("test_example_count", -1)):
            raise ValueError(f"task-boundary prediction count disagrees with manifest at stage {stage}")
    if verify_hashes:
        for relative, expected in manifest.get("files", {}).items():
            path = directory / relative
            if not path.is_file() or path.stat().st_size != expected.get("bytes") or file_sha256(path) != expected.get("sha256"):
                raise ValueError(f"artifact hash/size mismatch: {relative}")
    return {
        "valid": True,
        "namespace": namespace,
        "training_reproduction_claimed": summary.get("training_reproduction_claimed"),
        "task_rows": len(task_rows),
        "artifact_count": len(manifest.get("files", {})),
    }
