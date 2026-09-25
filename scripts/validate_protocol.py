#!/usr/bin/env python3
"""Validate the released final protocol manifest without running training."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs" / "final_protocol.json").read_text())
assert config["seeds"] == list(range(42, 52))
assert config["task_sizes"] == [50] + [5] * 10
assert config["full_data"] is True
assert config["evaluate_all_seen_classes_after_each_task"] is True
for dataset, expected_dim in (("EMBER", 2381), ("AZ-Class", 2439)):
    assert config["datasets"][dataset]["feature_dim"] == expected_dim
print("protocol validation passed: 100 classes, 11 tasks, seeds 42-51, full data, all-seen evaluation")
