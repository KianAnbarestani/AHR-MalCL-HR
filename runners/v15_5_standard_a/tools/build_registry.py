#!/usr/bin/env python3
"""Build the immutable 40-row execution template from the authoritative gap CSV."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
GAP = ROOT / "V15_5_FINAL_NEW_RUN_GAP.csv"
OUT = ROOT / "V15_5_STANDARD_A_40_RUN_REGISTRY.csv"
FIELDS = [
    "run_id", "dataset", "seed", "variant", "K", "status", "attempt",
    "start_time", "end_time", "host", "gpu", "output_dir", "config_sha256",
    "class_order_sha256", "checkpoint", "result_sha256", "validation_status",
    "failure_reason",
]

with GAP.open(newline="") as handle:
    gap = list(csv.DictReader(handle))

rows = []
for source in gap:
    rows.append({
        "run_id": source["run_id"],
        "dataset": source["dataset"],
        "seed": source["seed"],
        "variant": source["variant"],
        "K": source["K"],
        "status": "NOT_STARTED",
        "attempt": "0",
        "start_time": "",
        "end_time": "",
        "host": "",
        "gpu": "",
        "output_dir": f"runs/{source['variant']}/{source['dataset']}/seed_{source['seed']}",
        "config_sha256": "",
        "class_order_sha256": "",
        "checkpoint": "",
        "result_sha256": "",
        "validation_status": "NOT_VALIDATED",
        "failure_reason": "",
    })

keys = [(r["dataset"], r["seed"], r["variant"], r["K"]) for r in rows]
assert len(rows) == 40
assert len(keys) == len(set(keys))

with OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows: {OUT}")
