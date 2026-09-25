#!/usr/bin/env python3
"""Authoritative V15.4 task-matrix-derived continual-learning metrics.

Matrix rows are post-training stages and columns are evaluation tasks.  Cells
are sample accuracies expressed as fractions.  Unavailable future-task cells
are represented by ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFINITION_VERSION = "V15.4"


@dataclass(frozen=True)
class HarmonizedMetrics:
    stage_seen_accuracy: tuple[float, ...]
    aia: float
    forgetting: float
    bwt: float
    task_acquisition: float
    final_per_task_accuracy: tuple[float, ...]
    final_old_task_accuracy: float
    final_new_task_accuracy: float


def _matrix(values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
    matrix = [[None if value is None else float(value) for value in row] for row in values]
    if not matrix:
        raise ValueError("task matrix is empty")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError(f"expected square {n}x{n} matrix")
    for stage, row in enumerate(matrix):
        for task, value in enumerate(row):
            if task <= stage and value is None:
                raise ValueError(f"missing observed cell stage={stage + 1} task={task + 1}")
            if task > stage and value is not None:
                raise ValueError(f"future-task cell must be unavailable: stage={stage + 1} task={task + 1}")
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"accuracy outside [0,1]: {value}")
    return matrix


def _supports(values: Iterable[int], n: int) -> np.ndarray:
    supports = np.asarray(list(values), dtype=float)
    if supports.shape != (n,):
        raise ValueError(f"expected {n} task supports")
    if np.any(supports <= 0) or np.any(~np.isfinite(supports)):
        raise ValueError("task supports must be positive finite values")
    return supports


def stage_seen_class_accuracy(
    task_matrix: Sequence[Sequence[float | None]], task_supports: Iterable[int]
) -> tuple[float, ...]:
    """Return sample-weighted accuracy over all tasks seen at every stage."""
    matrix = _matrix(task_matrix)
    supports = _supports(task_supports, len(matrix))
    output: list[float] = []
    for stage, row in enumerate(matrix):
        accuracies = np.asarray(row[: stage + 1], dtype=float)
        weights = supports[: stage + 1]
        output.append(float(np.average(accuracies, weights=weights)))
    return tuple(output)


def average_incremental_accuracy(
    task_matrix: Sequence[Sequence[float | None]], task_supports: Iterable[int]
) -> float:
    return float(np.mean(stage_seen_class_accuracy(task_matrix, task_supports)))


def conventional_forgetting(task_matrix: Sequence[Sequence[float | None]]) -> float:
    """Mean max-pre-final minus final accuracy over tasks 1..N-1.

    Negative task contributions are retained.  The final stage is excluded
    from the prior maximum.
    """
    matrix = _matrix(task_matrix)
    n = len(matrix)
    values = []
    for task in range(n - 1):
        prior = [float(matrix[stage][task]) for stage in range(task, n - 1)]
        values.append(max(prior) - float(matrix[n - 1][task]))
    return float(np.mean(values))


def backward_transfer(task_matrix: Sequence[Sequence[float | None]]) -> float:
    matrix = _matrix(task_matrix)
    n = len(matrix)
    return float(np.mean([float(matrix[-1][task]) - float(matrix[task][task]) for task in range(n - 1)]))


def task_acquisition_accuracy(task_matrix: Sequence[Sequence[float | None]]) -> float:
    matrix = _matrix(task_matrix)
    return float(np.mean([float(matrix[index][index]) for index in range(len(matrix))]))


def final_per_task_accuracy(task_matrix: Sequence[Sequence[float | None]]) -> tuple[float, ...]:
    matrix = _matrix(task_matrix)
    return tuple(float(value) for value in matrix[-1] if value is not None)


def derive(
    task_matrix: Sequence[Sequence[float | None]], task_supports: Iterable[int]
) -> HarmonizedMetrics:
    matrix = _matrix(task_matrix)
    supports = _supports(task_supports, len(matrix))
    stage = stage_seen_class_accuracy(matrix, supports.astype(int))
    final = np.asarray(final_per_task_accuracy(matrix), dtype=float)
    old_weights = supports[:-1]
    return HarmonizedMetrics(
        stage_seen_accuracy=stage,
        aia=float(np.mean(stage)),
        forgetting=conventional_forgetting(matrix),
        bwt=backward_transfer(matrix),
        task_acquisition=task_acquisition_accuracy(matrix),
        final_per_task_accuracy=tuple(float(value) for value in final),
        final_old_task_accuracy=float(np.average(final[:-1], weights=old_weights)),
        final_new_task_accuracy=float(final[-1]),
    )
