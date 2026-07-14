#!/usr/bin/env python3
"""Check the AHR-MalCL baseline data adapter for one dataset/seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "v1 - data adapter check"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_suite import data_adapter

OUT_DIR = PROJECT_ROOT / "paper_ready_baselines"


def slug_dataset(name: str) -> str:
    canonical = data_adapter.canonical_dataset_name(name)
    return "az" if canonical == "az_class" else canonical


def finite_check_stream(stream: dict[str, Any]) -> dict[str, Any]:
    checked_rows = 0
    has_nonfinite = False
    for task in stream["tasks"]:
        for key in ("X_train", "X_test"):
            view = task[key]
            if not len(view):
                continue
            sample = view.take_array(max_rows=min(len(view), 1024))
            checked_rows += int(sample.shape[0])
            if sample.size and not np.isfinite(sample).all():
                has_nonfinite = True
    return {"checked_rows": checked_rows, "has_nan_or_inf": has_nonfinite}


def markdown_report(summary: dict[str, Any], finite: dict[str, Any], matches_known: str) -> str:
    lines = [
        f"# Data Adapter Check: {summary['dataset']} Seed {summary['seed']}",
        "",
        f"- Dataset name: `{summary['dataset']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Feature dimension: `{summary['feature_dim']}`",
        f"- Number of classes: `{summary['num_classes']}`",
        f"- Number of tasks: `{summary['task_count']}`",
        f"- Classes per task: `{summary['classes_per_task']}`",
        f"- Train samples per task: `{summary['train_samples_per_task']}`",
        f"- Test samples per task: `{summary['test_samples_per_task']}`",
        f"- Label range: `{summary['label_min']}` to `{summary['label_max']}`",
        f"- NaN/inf check rows: `{finite['checked_rows']}`",
        f"- NaN/inf detected: `{'YES' if finite['has_nan_or_inf'] else 'NO'}`",
        f"- Task ordering matches known metadata if available: `{matches_known}`",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(summary["metadata"], indent=2),
        "```",
        "",
        "## Class Order",
        "",
        "```json",
        json.dumps(summary["class_order"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(dataset: str, seed: int, config_path: Path | None = None) -> int:
    config: dict[str, Any] = {}
    if config_path:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"data_adapter_check_{slug_dataset(dataset)}_seed{int(seed)}.md"
    try:
        stream = data_adapter.load_class_incremental_stream(dataset, int(seed), config=config)
        summary = data_adapter.summarize_stream(stream)
        finite = finite_check_stream(stream)
        expected_feature_dim = data_adapter.get_dataset_config(dataset)["expected_feature_dim"]
        matches_known = "YES" if int(summary["feature_dim"]) == int(expected_feature_dim) else "NO"
        report_path.write_text(markdown_report(summary, finite, matches_known), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")
        return 0
    except Exception as exc:  # noqa: BLE001 - check command must produce a report.
        todo = data_adapter.write_todo(dataset, f"{type(exc).__name__}: {exc}")
        lines = [
            f"# Data Adapter Check Failed: {dataset} Seed {seed}",
            "",
            f"- Error: `{type(exc).__name__}: {exc}`",
            f"- TODO: `{todo.relative_to(PROJECT_ROOT)}`",
            "",
            "The adapter did not download data or run training.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Data adapter check failed: {type(exc).__name__}: {exc}")
        print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {todo.relative_to(PROJECT_ROOT)}")
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check baseline data adapter.")
    parser.add_argument("--dataset", required=True, help="EMBER or AZ-Class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="Optional baseline config JSON.")
    return parser.parse_args()


def main() -> None:
    print(VERSION, flush=True)
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    raise SystemExit(run(args.dataset, args.seed, config_path=config_path))


if __name__ == "__main__":
    main()
