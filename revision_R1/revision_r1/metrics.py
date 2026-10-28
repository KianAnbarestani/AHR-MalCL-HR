"""Self-contained classification and V15.4-harmonized continual metrics."""

from typing import Sequence

import numpy as np


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    class_ids = [int(value) for value in labels]
    if truth.ndim != 1 or prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("classification metrics need nonempty, aligned 1-D truth/prediction arrays")
    if len(set(class_ids)) != len(class_ids) or not class_ids:
        raise ValueError("metric labels must be a nonempty unique sequence")
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for true_label, predicted_label in zip(truth, prediction):
        if int(true_label) not in index or int(predicted_label) not in index:
            raise ValueError("truth or prediction contains a label outside the explicit metric label set")
        confusion[index[int(true_label)], index[int(predicted_label)]] += 1
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = int(support.sum())
    present = support > 0
    return {
        "accuracy": float(tp.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(), "support": support.astype(int).tolist(),
        "labels": class_ids, "confusion_matrix": confusion.astype(int).tolist(),
    }


def _validated_matrix(values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
    matrix = [[None if value is None else float(value) for value in row] for row in values]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("continual task matrix must be nonempty and square")
    for stage, row in enumerate(matrix):
        for task, value in enumerate(row):
            if task <= stage and value is None:
                raise ValueError(f"missing observed task-matrix cell stage={stage + 1}, task={task + 1}")
            if task > stage and value is not None:
                raise ValueError(f"future task-matrix cell must be null stage={stage + 1}, task={task + 1}")
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError("task-matrix accuracies must be finite fractions in [0,1]")
    return matrix


def continual_metrics(task_matrix: Sequence[Sequence[float | None]], task_supports: Sequence[int]) -> dict:
    matrix = _validated_matrix(task_matrix)
    n = len(matrix)
    supports = np.asarray(task_supports, dtype=np.float64)
    if supports.shape != (n,) or np.any(~np.isfinite(supports)) or np.any(supports <= 0):
        raise ValueError(f"continual metrics require {n} positive finite task supports")
    stage_seen = [
        float(np.average(np.asarray(row[: stage + 1], dtype=np.float64), weights=supports[: stage + 1]))
        for stage, row in enumerate(matrix)
    ]
    forgetting = float(np.mean([
        max(float(matrix[stage][task]) for stage in range(task, n - 1)) - float(matrix[-1][task])
        for task in range(n - 1)
    ])) if n > 1 else 0.0
    bwt = float(np.mean([float(matrix[-1][task]) - float(matrix[task][task]) for task in range(n - 1)])) if n > 1 else 0.0
    final = [float(value) for value in matrix[-1]]
    return {
        "stage_seen_accuracy": stage_seen,
        "aia": float(np.mean(stage_seen)),
        "forgetting": forgetting,
        "bwt": bwt,
        "task_acquisition": float(np.mean([float(matrix[i][i]) for i in range(n)])),
        "task_acquisition_definition": (
            "mean diagonal task accuracy measured immediately after each task; "
            "this is an accuracy summary, not a measure of adaptation speed or elapsed time"
        ),
        "final_per_task_accuracy": final,
        "final_old_task_accuracy": float(np.average(final[:-1], weights=supports[:-1])) if n > 1 else float("nan"),
        "final_new_task_accuracy": final[-1],
        "definition_version": "V15.4_HARMONIZED_TASK_MATRIX",
    }

