from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from revision_r1.cli import _verified_schedule_hash, make_parser, preflight_all
from revision_r1.schedule import PoolCounts


def test_schedule_gate_requires_exact_manifest_hash_and_nonhistorical_marker(tmp_path):
    schedule = tmp_path / "budget.csv"
    schedule.write_text("dataset,seed,task\n")
    digest = hashlib.sha256(schedule.read_bytes()).hexdigest()
    manifest = tmp_path / "budget.json"
    manifest.write_text(json.dumps({
        "source_csv_sha256": digest,
        "row_count": 220,
        "training_reproduction_claimed": False,
        "status": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
    }))
    assert _verified_schedule_hash(schedule, manifest) == digest

    manifest.write_text(json.dumps({
        "source_csv_sha256": "0" * 64,
        "row_count": 220,
        "training_reproduction_claimed": False,
        "status": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
    }))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verified_schedule_hash(schedule, manifest)

    manifest.write_text(json.dumps({
        "source_csv_sha256": digest,
        "row_count": 220,
        "training_reproduction_claimed": True,
        "status": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
    }))
    with pytest.raises(ValueError, match="disclaim"):
        _verified_schedule_hash(schedule, manifest)


def test_production_parser_requires_separate_authorization_schedule_manifest_and_paths():
    parser = make_parser()
    args = parser.parse_args([
        "run", "--dataset", "ember", "--seed", "42", "--arm", "current_only",
        "--data-root", "/drive/ember", "--cache-root", "/tmp/cache",
        "--output-root", "/drive/results", "--schedule", "/package/budget.csv",
        "--schedule-manifest", "/package/budget.json",
        "--drive-checkpoint-root", "/drive/checkpoints",
    ])
    assert not args.production_authorized
    assert args.local_checkpoint_root == "/content/revision_r1_checkpoints"


def test_preflight_derives_verified_schedule_without_mutating_phase1_source(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    source = project / "configs" / "update_budget.csv"
    source_manifest = project / "configs" / "update_budget_manifest.json"
    source_bytes = source.read_bytes()
    with source.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    actual = {
        (row["dataset"], int(row["seed"]), int(row["task"])): PoolCounts(
            int(row["current_pool_size"]), int(row["anchor_pool_size"]),
            int(row["generated_pool_size"]),
        )
        for row in records
    }
    actual[("az_class", 42, 2)] = PoolCounts(16230, 9984, 5000)

    def fake_preflight(data_root, dataset, cache_dir):
        return SimpleNamespace(
            train_metadata={"md5": f"{dataset}-train-md5", "sha256": f"{dataset}-train-sha"},
            test_metadata={"md5": f"{dataset}-test-md5", "sha256": f"{dataset}-test-sha"},
        )

    monkeypatch.setattr("revision_r1.cli.preflight_dataset", fake_preflight)
    monkeypatch.setattr("revision_r1.cli.build_actual_pool_matrix", lambda ember, az: actual)
    output = tmp_path / "preflight" / "preflight.json"
    result = preflight_all(SimpleNamespace(
        ember_data_root="/fixture/ember", az_data_root="/fixture/az",
        cache_root=str(tmp_path / "cache"), schedule=str(source),
        schedule_manifest=str(source_manifest), output=str(output),
    ))

    assert source.read_bytes() == source_bytes
    assert result["status"] == "PREFLIGHT_VALIDATED_NO_TRAINING"
    assert result["schedule"]["row_count"] == 220
    assert result["schedule"]["reconciliation"]["adjusted_rows"] == 1
    effective = Path(result["schedule"]["path"])
    manifest = Path(result["schedule"]["manifest_path"])
    with effective.open(newline="", encoding="utf-8") as stream:
        effective_rows = list(csv.DictReader(stream))
    row = next(row for row in effective_rows if row["dataset"] == "az_class" and row["seed"] == "42" and row["task"] == "2")
    assert row["source_anchor_pool_size"] == "10000"
    assert row["anchor_pool_size"] == "9984"
    assert result["schedule"]["parent_source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert json.loads(manifest.read_text(encoding="utf-8"))["training_reproduction_claimed"] is False
    assert output.is_file()
