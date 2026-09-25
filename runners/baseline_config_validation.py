#!/usr/bin/env python3
"""Dependency-light, non-training validation for the historical baseline runners."""

from __future__ import annotations

import json
from pathlib import Path


def validate(kind: str, args: object) -> int:
    config_path = Path(str(getattr(args, "config")))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_name = {"er": "experience replay", "derpp": "der"}[kind]
    name = str(config.get("baseline_name", "")).lower()
    if expected_name not in name:
        raise ValueError(f"configuration method mismatch: expected {kind}, got {config.get('baseline_name')!r}")
    dataset = str(config.get("dataset", "")).lower().replace("-", "_")
    if dataset not in {"ember", "az_class"}:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    schedule = config.get("task_ordering", {})
    expected_schedule = {
        "initial_classes": 50, "increment_classes_per_task": 5,
        "total_classes": 100, "task_count": 11,
    }
    for key, value in expected_schedule.items():
        if int(schedule.get(key, -1)) != value:
            raise ValueError(f"invalid task schedule {key}: {schedule.get(key)!r}")
    seeds = [int(seed) for seed in (getattr(args, "seeds", None) or config.get("seeds", []))]
    single = getattr(args, "single_seed", None)
    if single is not None:
        seeds = [int(single)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed configuration is empty or duplicated")
    output = Path(str(getattr(args, "output_root", "") or config.get("output_directory", "")))
    if not str(output) or "result/" in output.as_posix():
        raise ValueError(f"unsafe or empty output path: {output}")
    dataset_config = {
        "ember": ("EMBER_Class_train.npz", "EMBER_Class_test.npz"),
        "az_class": ("AZ_Class_Train.npz", "AZ_Class_Test.npz"),
    }[dataset]
    roots = [Path("data"), Path("datasets"), Path("/content/data")]
    present = any((root / dataset_config[0]).is_file() and (root / dataset_config[1]).is_file() for root in roots)
    print(json.dumps({
        "validation_mode": "config_only", "method": kind, "dataset": dataset,
        "dataset_files_expected": list(dataset_config), "dataset_files_present": present,
        "dataset_files_loaded": False, "seeds": seeds,
        "task_sizes": [50] + [5] * 10, "output_path": output.as_posix(),
        "training_started": False, "result_files_written": False, "status": "passed",
    }, indent=2))
    return 0
