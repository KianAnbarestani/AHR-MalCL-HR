"""Atomic, machine-readable revision_R1 result exports."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"refusing to write empty CSV artifact {path}")
    fields = list(dict.fromkeys(key for row in records for key in row))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_predictions(path: str | Path, y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> None:
    target = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                y_true_global=np.asarray(y_true, dtype=np.int64),
                y_pred_global=np.asarray(y_pred, dtype=np.int64),
                class_order=np.asarray(class_ids, dtype=np.int64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_task_boundary_predictions(path: str | Path, stage_records: dict[str, dict[str, Any]]) -> None:
    if set(stage_records) != {str(stage) for stage in range(1, 12)}:
        raise ValueError("task-boundary predictions must contain all 11 stages")
    target = Path(path)
    arrays = {}
    for stage, record in sorted(stage_records.items(), key=lambda item: int(item[0])):
        prefix = f"stage_{int(stage):02d}"
        arrays[f"{prefix}_test_indices"] = np.asarray(record["test_indices"], dtype=np.int64)
        arrays[f"{prefix}_y_true_global"] = np.asarray(record["y_true_global"], dtype=np.int64)
        arrays[f"{prefix}_y_pred_global"] = np.asarray(record["y_pred_global"], dtype=np.int64)
        arrays[f"{prefix}_head_class_ids"] = np.asarray(record["head_class_ids"], dtype=np.int64)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
