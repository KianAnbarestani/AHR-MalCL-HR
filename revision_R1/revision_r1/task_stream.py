"""Legacy-compatible class order and data-derived revision_R1 task pools."""

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .schedule import PoolCounts


@dataclass(frozen=True)
class TaskIndices:
    task_id: int
    classes: tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


def class_order(seed: int) -> list[int]:
    return np.random.RandomState(int(seed)).permutation(100).astype(int).tolist()


def order_sha256(order: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in order], separators=(",", ":")).encode()).hexdigest()


def split_order(order: Sequence[int], task_sizes: Sequence[int]) -> list[tuple[int, ...]]:
    order = [int(x) for x in order]
    if len(order) != 100 or sorted(order) != list(range(100)):
        raise ValueError("class order must be a permutation of global class IDs 0..99")
    if sum(map(int, task_sizes)) != 100 or any(int(size) <= 0 for size in task_sizes):
        raise ValueError("task sizes must be positive and sum to 100 classes")
    tasks, cursor = [], 0
    for size in task_sizes:
        tasks.append(tuple(order[cursor:cursor + int(size)]))
        cursor += int(size)
    return tasks


def build_tasks(y_train: np.ndarray, y_test: np.ndarray, seed: int, task_sizes: Sequence[int]) -> list[TaskIndices]:
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("train and test labels must be one-dimensional")
    order = class_order(seed)
    tasks = []
    for task_id, classes in enumerate(split_order(order, task_sizes), start=1):
        tr = np.flatnonzero(np.isin(y_train, classes)).astype(np.int64)
        te = np.flatnonzero(np.isin(y_test, classes)).astype(np.int64)
        if tr.size == 0 or te.size == 0:
            raise ValueError(f"task {task_id} has empty train or test support")
        tasks.append(TaskIndices(task_id, classes, tr, te))
    return tasks


def reconstruct_pool_counts(
    y_train: np.ndarray,
    order: Sequence[int],
    task_sizes: Sequence[int],
    anchor_k: int,
    generated_k: int = 100,
) -> dict[int, PoolCounts]:
    y_train = np.asarray(y_train, dtype=np.int64)
    if y_train.ndim != 1 or not np.isin(y_train, np.arange(100)).all():
        raise ValueError("training labels must be global class IDs 0..99")
    tasks = split_order(order, task_sizes)
    counts = np.bincount(y_train, minlength=100)
    result: dict[int, PoolCounts] = {}
    seen: list[int] = []
    for task_id, classes in enumerate(tasks, start=1):
        current = int(sum(counts[class_id] for class_id in classes))
        anchors = int(sum(min(int(anchor_k), int(counts[class_id])) for class_id in seen)) if anchor_k else 0
        generated = len(seen) * int(generated_k) if generated_k else 0
        result[task_id] = PoolCounts(current, anchors, generated)
        seen.extend(classes)
    return result

