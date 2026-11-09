"""Compact paired report export for the frozen temporal EMBER pilot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .config import ARMS, EXPERIMENT_ID, SEEDS, validate_run_matrix
from .stream import WINDOW_IDS, manifest_sha256, window_partitions


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _finite_score(metric, name: str) -> float:
    if not isinstance(metric, dict) or metric.get("macro_f1") is None:
        raise ValueError(f"completed report metric {name} is missing macro-F1")
    value = float(metric["macro_f1"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"report metric {name} macro-F1 must be finite in [0, 1]")
    return value


def _metric_summary(metric):
    if metric is None:
        return None
    result = {
        "macro_f1": None if metric.get("macro_f1") is None else float(metric["macro_f1"]),
        "sample_count": int(metric.get("sample_count", 0)),
        "scoring_ids_sha256": metric.get("scoring_ids_sha256"),
        "per_family_sample_counts": {
            str(family): int(count)
            for family, count in sorted(metric.get("per_family_sample_counts", {}).items())
        },
        "per_family_f1": {
            str(family): None if score is None else float(score)
            for family, score in sorted(metric.get("per_family_f1", {}).items())
        },
    }
    if metric.get("metric_family_count") is not None:
        result["metric_family_count"] = int(metric["metric_family_count"])
    return result


def _validate_inputs(manifest: dict, run_records: list[dict]) -> dict[tuple[str, int], dict]:
    if not isinstance(manifest, dict) or manifest_sha256(manifest) != manifest.get("manifest_sha256"):
        raise ValueError("report input manifest failed its frozen split hash")
    if manifest.get("window_ids") != list(WINDOW_IDS):
        raise ValueError("report input must use the frozen six-window stream")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(run_records) != 9:
        raise ValueError("report requires exactly nine completed production runs")
    by_pair = {}
    schedule_hashes = set()
    for record in run_records:
        if not isinstance(record, dict) or record.get("status") != "completed":
            raise ValueError("report requires completed run result records")
        pair = (record.get("arm"), int(record.get("seed", -1)))
        if pair in by_pair:
            raise ValueError(f"duplicate production run record for {pair}")
        if pair not in expected:
            raise ValueError(f"unexpected production run record for {pair}")
        if record.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("run result belongs to a different experiment ID")
        if record.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("run result was produced from a different split manifest")
        if not record.get("schedule_sha256") or len(str(record["schedule_sha256"])) != 64:
            raise ValueError("run result is missing its frozen schedule identity")
        if not record.get("final_checkpoint_sha256"):
            raise ValueError("run result is missing final checkpoint provenance")
        windows = record.get("window_records")
        if not isinstance(windows, list) or [row.get("window_id") for row in windows] != list(WINDOW_IDS):
            raise ValueError(f"run {record.get('run_id')} must contain all six windows in frozen order")
        if record.get("run_id") != f"{pair[0]}_seed{pair[1]}":
            raise ValueError("run ID does not match its arm and seed")
        schedule_hashes.add(record["schedule_sha256"])
        by_pair[pair] = record
    if set(by_pair) != expected:
        raise ValueError("report matrix is incomplete; expected exactly the three arms by seeds 42, 43, and 44")
    if len(schedule_hashes) != 1:
        raise ValueError("all nine runs must share one frozen classifier-update schedule")

    score_hash_by_window = {}
    update_count_by_window = {}
    parts_by_window = {window_id: window_partitions(manifest, window_id) for window_id in WINDOW_IDS}
    family_for = {record["sha256"]: record["family"] for record in manifest["rows"]}
    for record in by_pair.values():
        for index, row in enumerate(record["window_records"]):
            score_hash = row.get("scoring_ids_sha256")
            if not isinstance(score_hash, str) or len(score_hash) != 64:
                raise ValueError(f"missing frozen scoring-set identity for {row.get('window_id')}")
            previous_hash = score_hash_by_window.setdefault(index, score_hash)
            if previous_hash != score_hash:
                raise ValueError(f"paired arms/seeds do not use the same scoring set for {row['window_id']}")
            expected_ids = parts_by_window[row["window_id"]]["scoring_ids"]
            expected_hash = hashlib.sha256("\n".join(expected_ids).encode("ascii")).hexdigest()
            if score_hash != expected_hash:
                raise ValueError(f"scoring-set hash differs from the frozen manifest for {row['window_id']}")
            if int(row.get("scored_sample_count", -1)) != len(expected_ids):
                raise ValueError(f"scored sample count differs from the frozen manifest for {row['window_id']}")
            updates = int(row.get("classifier_updates", 0))
            if updates < 1:
                raise ValueError(f"missing classifier update budget for {row['window_id']}")
            previous_updates = update_count_by_window.setdefault(index, updates)
            if previous_updates != updates:
                raise ValueError(f"classifier update budget differs by arm or seed for {row['window_id']}")
            if index == 0:
                if row.get("pre_update") is not None or row.get("january_post_fit") is None:
                    raise ValueError("January must report its post-fit score separately")
                _finite_score(row["january_post_fit"], "January post-fit")
                metric = row["january_post_fit"]
            else:
                if row.get("pre_update") is None:
                    raise ValueError(f"missing pre-update score for {row['window_id']}")
                _finite_score(row["pre_update"], f"{row['window_id']} pre-update")
                if row.get("old_family_retention") is None:
                    raise ValueError(f"missing old-family retention metric for {row['window_id']}")
                if row.get("new_family_acquisition") is None:
                    raise ValueError(f"missing new-family acquisition status for {row['window_id']}")
                metric = row["pre_update"]
                old_families = set(manifest["admissions"][WINDOW_IDS[0]])
                old_ids = [sample_id for sample_id in expected_ids if family_for[sample_id] in old_families]
                if int(row["old_family_retention"].get("sample_count", -1)) != len(old_ids):
                    raise ValueError(f"old-family retention sample count differs from the frozen manifest for {row['window_id']}")
                acquisition = row["new_family_acquisition"]
                expected_families = manifest["admissions"][WINDOW_IDS[index - 1]] if index > 1 else []
                expected_support = manifest["family_support_by_window"][row["window_id"]]["heldout"]
                observed_counts = acquisition.get("per_family_sample_counts", {})
                for family in expected_families:
                    expected_count = int(expected_support.get(family, 0))
                    if int(observed_counts.get(family, 0)) != expected_count:
                        raise ValueError(
                            f"new-family sample count for {family} differs from the frozen manifest in {row['window_id']}"
                        )
                if index > 1 and int(acquisition.get("sample_count", -1)) != sum(
                    int(expected_support.get(family, 0)) for family in expected_families
                ):
                    raise ValueError(f"new-family acquisition sample count differs from the frozen manifest in {row['window_id']}")
            if int(metric.get("sample_count", -1)) != len(expected_ids):
                raise ValueError(f"metric sample count differs from the frozen scoring set for {row['window_id']}")
            if sum(int(value) for value in metric.get("per_family_sample_counts", {}).values()) != len(expected_ids):
                raise ValueError(f"per-family counts differ from the frozen scoring set for {row['window_id']}")
            if metric.get("scoring_ids_sha256") != score_hash:
                raise ValueError(f"metric scoring-set hash differs from frozen window row for {row['window_id']}")
    return by_pair


def _acquisition_rows(manifest: dict, by_pair: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            record = by_pair[(arm, seed)]
            for evaluation_index in range(2, len(WINDOW_IDS)):
                evaluation_window = WINDOW_IDS[evaluation_index]
                admission_window = WINDOW_IDS[evaluation_index - 1]
                metric = record["window_records"][evaluation_index]["new_family_acquisition"]
                families = manifest["admissions"][admission_window]
                counts = metric.get("per_family_sample_counts", {})
                f1_values = metric.get("per_family_f1", {})
                for family in families:
                    sample_count = int(counts.get(family, 0))
                    rows.append({
                        "run_id": record["run_id"],
                        "arm": arm,
                        "seed": seed,
                        "admission_window": admission_window,
                        "evaluation_window": evaluation_window,
                        "family": family,
                        "status": metric.get("status", "missing"),
                        "sample_count": sample_count,
                        "family_f1": None if f1_values.get(family) is None else float(f1_values[family]),
                        "window_acquisition_macro_f1": None if metric.get("macro_f1") is None else float(metric["macro_f1"]),
                    })
    october_window = WINDOW_IDS[-1]
    october_train_support = manifest["family_support_by_window"][october_window]["train"]
    for family in manifest["admissions"][october_window]:
        rows.append({
            "run_id": None,
            "arm": None,
            "seed": None,
            "admission_window": october_window,
            "evaluation_window": None,
            "family": family,
            "status": "unavailable_no_following_window",
            "sample_count": None,
            "family_f1": None,
            "window_acquisition_macro_f1": None,
            "training_support_at_admission": int(october_train_support.get(family, 0)),
        })
    return rows


def export_report(manifest: dict, run_records, output_dir: str | Path) -> dict:
    """Validate paired runs and export JSON/CSV plus the exact frozen manifest."""
    records = list(run_records)
    by_pair = _validate_inputs(manifest, records)
    schedule_sha = next(iter({record["schedule_sha256"] for record in records}))
    metrics_by_run = []
    csv_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            record = by_pair[(arm, seed)]
            run_windows = []
            for index, row in enumerate(record["window_records"]):
                pre = _metric_summary(row.get("pre_update"))
                january = _metric_summary(row.get("january_post_fit"))
                retention = _metric_summary(row.get("old_family_retention"))
                acquisition = row.get("new_family_acquisition") or {}
                window = {
                    "window_id": row["window_id"],
                    "window_index": index,
                    "scoring_ids_sha256": row["scoring_ids_sha256"],
                    "scored_sample_count": int(row.get("scored_sample_count", 0)),
                    "heldout_sample_count": int(row.get("heldout_sample_count", 0)),
                    "unscored_unknown_sample_count": int(row.get("unscored_unknown_sample_count", 0)),
                    "pre_update_macro_f1": None if pre is None else pre["macro_f1"],
                    "pre_update_metric": pre,
                    "january_postfit_macro_f1": None if january is None else january["macro_f1"],
                    "january_postfit_metric": january,
                    "old_family_retention": retention,
                    "new_family_acquisition": {
                        "status": acquisition.get("status"),
                        "macro_f1": None if acquisition.get("macro_f1") is None else float(acquisition["macro_f1"]),
                        "sample_count": int(acquisition.get("sample_count", 0)),
                        "scoring_ids_sha256": acquisition.get("scoring_ids_sha256"),
                        "per_family_sample_counts": {
                            str(family): int(count)
                            for family, count in sorted(acquisition.get("per_family_sample_counts", {}).items())
                        },
                        "per_family_f1": {
                            str(family): None if score is None else float(score)
                            for family, score in sorted(acquisition.get("per_family_f1", {}).items())
                        },
                    },
                    "classifier_updates": int(row["classifier_updates"]),
                    "planned_classifier_updates": int(row["classifier_updates"]),
                    "planned_pool": row.get("planned_pool", {}),
                    "realized_pool": row.get("realized_pool", {}),
                    "planned_replay": {
                        "anchor": int(row.get("planned_pool", {}).get("anchor", 0)),
                        "generated": int(row.get("planned_pool", {}).get("generated", 0)),
                    },
                    "realized_replay": row.get("realized_replay", {}),
                    "generated_replay_output_count": int(row.get("generated_replay_output_count", 0)),
                    "classifier_sampled_source_counts": row.get("classifier_sampled_source_counts", {}),
                    "gan_compute_seconds": float(row.get("gan_compute_seconds", 0.0)),
                    "gan_wall_seconds": float(row.get("gan_wall_seconds", 0.0)),
                    "generation_seconds": float(row.get("generation_seconds", 0.0)),
                }
                run_windows.append(window)
                csv_rows.append({
                    "run_id": record["run_id"], "arm": arm, "seed": seed,
                    "window_id": row["window_id"], "scored_sample_count": window["scored_sample_count"],
                    "heldout_sample_count": window["heldout_sample_count"],
                    "unscored_unknown_sample_count": window["unscored_unknown_sample_count"],
                    "pre_update_macro_f1": window["pre_update_macro_f1"],
                    "january_postfit_macro_f1": window["january_postfit_macro_f1"],
                    "old_family_retention_macro_f1": None if retention is None else retention["macro_f1"],
                    "new_family_acquisition_macro_f1": window["new_family_acquisition"]["macro_f1"],
                    "new_family_acquisition_status": window["new_family_acquisition"]["status"],
                    "classifier_updates": window["classifier_updates"],
                    "planned_anchor_replay": window["planned_replay"]["anchor"],
                    "planned_generated_replay": window["planned_replay"]["generated"],
                    "realized_anchor_replay": int(window["realized_replay"].get("anchor", 0)),
                    "realized_generated_replay": int(window["realized_replay"].get("generated", 0)),
                    "generated_replay_output_count": window["generated_replay_output_count"],
                    "gan_compute_seconds": window["gan_compute_seconds"],
                    "gan_wall_seconds": window["gan_wall_seconds"],
                    "generation_seconds": window["generation_seconds"],
                })
            metrics_by_run.append({
                "run_id": record["run_id"], "arm": arm, "seed": seed,
                "final_checkpoint_sha256": record["final_checkpoint_sha256"],
                "windows": run_windows,
            })

    per_seed = {}
    window_differences = {window_id: [] for window_id in WINDOW_IDS[1:]}
    seed_means = []
    for seed in SEEDS:
        differences = {}
        for window_id in WINDOW_IDS[1:]:
            index = WINDOW_IDS.index(window_id)
            full = _finite_score(by_pair[("full_hybrid", seed)]["window_records"][index]["pre_update"],
                                f"{window_id} Full Hybrid")
            anchor = _finite_score(by_pair[("anchor_only", seed)]["window_records"][index]["pre_update"],
                                  f"{window_id} Anchor-only")
            difference = full - anchor
            differences[window_id] = difference
            window_differences[window_id].append(difference)
        seed_mean = sum(differences.values()) / len(WINDOW_IDS[1:])
        seed_means.append(seed_mean)
        per_seed[str(seed)] = {
            "window_differences": differences,
            "mean_difference": seed_mean,
        }
    primary = {
        "name": "paired Full Hybrid minus Anchor-only mean pre-update macro-F1",
        "window_ids": list(WINDOW_IDS[1:]),
        "window_weighting": "equal",
        "seed_weighting": "equal",
        "per_seed": per_seed,
        "mean_difference_by_window_across_seeds": {
            window_id: sum(values) / len(values) for window_id, values in window_differences.items()
        },
        "mean_of_seed_means": sum(seed_means) / len(seed_means),
        "seed_count": len(SEEDS),
        "window_count": len(WINDOW_IDS[1:]),
        "january_postfit_included": False,
    }
    acquisition_rows = _acquisition_rows(manifest, by_pair)
    windows = []
    for index, window_id in enumerate(WINDOW_IDS):
        windows.append({
            "window_id": window_id,
            "months": manifest["windows"][index]["months"],
            "admitted_families": manifest["admissions"][window_id],
            "train_family_support": manifest["family_support_by_window"][window_id]["train"],
            "heldout_family_support": manifest["family_support_by_window"][window_id]["heldout"],
            "scoring_ids_sha256": next(
                row["scoring_ids_sha256"] for row in by_pair[(ARMS[0], SEEDS[0])]["window_records"]
                if row["window_id"] == window_id
            ),
        })
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "result_status": "production_results_pending" if not records else "completed",
        "source_dataset": manifest.get("source", {}),
        "date_field": manifest.get("date_field", {}),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule_sha,
        "split_policy": manifest.get("split_policy", {}),
        "admission_interpretation": (
            "chronological new-to-learner family admission; first pilot training appearance is not a real-world emergence date; "
            "January top 50 are selected from January training-partition support only"
        ),
        "macro_f1_definition": (
            "mean per-class F1 over the full family cohort known before the scoring cutoff; "
            "families with no held-out support in that window contribute zero and retain a reported sample count of zero"
        ),
        "windows": windows,
        "primary_result": primary,
        "january_postfit": {
            "window_id": WINDOW_IDS[0],
            "reported_separately_from_primary": True,
            "by_arm_seed": {
                f"{arm}_seed{seed}": _metric_summary(
                    by_pair[(arm, seed)]["window_records"][0]["january_post_fit"]
                )
                for arm in ARMS for seed in SEEDS
            },
        },
        "acquisition_sparse_support_threshold": 30,
        "acquisition": acquisition_rows,
        "runs": metrics_by_run,
        "report_files": {
            "json": "temporal_report.json",
            "window_csv": "window_metrics.csv",
            "acquisition_csv": "new_family_acquisition.csv",
            "frozen_manifest": "frozen_split_manifest.json.gz",
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_bytes = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n").encode("utf-8")
    manifest_bytes = gzip.compress(_canonical_json(manifest), compresslevel=9, mtime=0)
    report["report_files"]["frozen_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    report_bytes = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(output_dir / "temporal_report.json", report_bytes)
    _atomic_bytes(output_dir / "frozen_split_manifest.json.gz", manifest_bytes)
    _atomic_bytes(output_dir / "frozen_split_manifest.json.gz.sha256",
                  (hashlib.sha256(manifest_bytes).hexdigest() + "  frozen_split_manifest.json.gz\n").encode())

    window_fields = list(csv_rows[0])
    window_output = _csv_bytes(window_fields, csv_rows)
    _atomic_bytes(output_dir / "window_metrics.csv", window_output)
    acquisition_fields = [
        "run_id", "arm", "seed", "admission_window", "evaluation_window", "family", "status",
        "sample_count", "family_f1", "window_acquisition_macro_f1", "training_support_at_admission",
    ]
    _atomic_bytes(output_dir / "new_family_acquisition.csv", _csv_bytes(acquisition_fields, acquisition_rows))
    return report


def _csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def load_completed_run_records(output_root: str | Path) -> list[dict]:
    """Load only result records from the isolated nine-run output root."""
    root = Path(output_root) / "runs"
    records = []
    for config in validate_run_matrix():
        path = root / config.run_id / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"missing completed run result: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records
