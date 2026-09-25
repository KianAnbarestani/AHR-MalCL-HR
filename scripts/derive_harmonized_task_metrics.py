#!/usr/bin/env python3
"""Derive V15.4 metrics only from released task matrices and supports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.harmonized_continual_metrics import DEFINITION_VERSION, derive


def read_supports(path: Path) -> dict[tuple[str, int], list[int]]:
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["dataset"], int(row["seed"]))].append((int(row["task_id"]), int(row["test_support"])))
    result = {}
    for key, rows in grouped.items():
        rows.sort()
        if [task for task, _ in rows] != list(range(1, 12)):
            raise ValueError(f"incomplete support schedule: {key}")
        result[key] = [support for _, support in rows]
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-matrix-dir", type=Path, default=ROOT / "task_matrices")
    parser.add_argument("--support-file", type=Path, default=ROOT / "analysis_outputs" / "TASK_TEST_SUPPORTS_V15_4.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs")
    args = parser.parse_args()
    supports = read_supports(args.support_file)
    per_seed: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    files = sorted(args.task_matrix_dir.glob("*.json"))
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["dataset"], int(record["seed"]))
        metric = derive(record["task_level_accuracies"], supports[key])
        row = {
            "dataset": record["dataset"],
            "method": record["method"],
            "seed": int(record["seed"]),
            "aia": metric.aia * 100,
            "forgetting": metric.forgetting * 100,
            "bwt": metric.bwt * 100,
            "task_acquisition": metric.task_acquisition * 100,
            "final_old_task_accuracy": metric.final_old_task_accuracy * 100,
            "final_new_task_accuracy": metric.final_new_task_accuracy * 100,
            "matrix_source": path.name,
            "support_source": args.support_file.name,
            "definition_version": DEFINITION_VERSION,
        }
        per_seed.append(row)
        for stage, value in enumerate(metric.stage_seen_accuracy, 1):
            stage_rows.append({
                "dataset": record["dataset"], "method": record["method"], "seed": int(record["seed"]),
                "stage_id": stage, "sample_weighted_seen_accuracy": value * 100,
                "cumulative_seen_support": sum(supports[key][:stage]),
                "matrix_source": path.name, "definition_version": DEFINITION_VERSION,
            })
    if len(per_seed) != 80:
        raise ValueError(f"expected 80 method-dataset-seed records, found {len(per_seed)}")
    keys = ["aia", "forgetting", "bwt", "task_acquisition", "final_old_task_accuracy", "final_new_task_accuracy"]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)
    summary: list[dict[str, object]] = []
    for (dataset, method), rows in sorted(grouped.items()):
        for metric in keys:
            values = [float(row[metric]) for row in rows]
            summary.append({
                "dataset": dataset, "method": method, "metric": metric, "n": len(values),
                "mean": mean(values), "sample_sd": stdev(values), "definition_version": DEFINITION_VERSION,
            })
    per_fields = ["dataset", "method", "seed", *keys, "matrix_source", "support_source", "definition_version"]
    stage_fields = ["dataset", "method", "seed", "stage_id", "sample_weighted_seen_accuracy", "cumulative_seen_support", "matrix_source", "definition_version"]
    write_csv(args.output_dir / "HARMONIZED_TASK_METRICS_PER_SEED_V15_4.csv", per_seed, per_fields)
    write_csv(args.output_dir / "HARMONIZED_TASK_METRICS_SUMMARY_V15_4.csv", summary, ["dataset", "method", "metric", "n", "mean", "sample_sd", "definition_version"])
    write_csv(args.output_dir / "HARMONIZED_STAGE_ACCURACY_V15_4.csv", stage_rows, stage_fields)
    (args.output_dir / "harmonized_derivation_audit.log").write_text(
        f"definition={DEFINITION_VERSION}\nmatrices={len(files)}\nper_seed_records={len(per_seed)}\nstage_records={len(stage_rows)}\n",
        encoding="utf-8",
    )
    print(f"derived {len(per_seed)} harmonized records from {len(files)} released matrices")


if __name__ == "__main__":
    main()
