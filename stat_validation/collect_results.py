#!/usr/bin/env python3
"""Collect raw result metrics from existing JSON artifacts."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import (
    apply_validity_fields,
    ensure_output_dirs,
    extract_rows_from_json,
    infer_setting_type,
    load_json_object,
    merge_duplicate_rows,
    rel_path,
    write_csv,
)


VERSION = "v1 - collect statistical validation results"


RUN_FIELDS = [
    "dataset",
    "method",
    "method_raw",
    "seed",
    "config_hash",
    "setting_type",
    "k",
    "source_file",
    "source_files",
    "duplicate_source_count",
    "source_kind",
    "context_label",
    "early_stop_reason",
    "task_count",
    "accs_seen_per_task",
    "acc_matrix_taskwise",
    "use_diversity_buffer",
    "use_kd",
    "use_proto_align",
    "gan_train_on_real_buffer",
    "use_wgan_gp",
    "use_projection_critic",
    "main_method_variant",
    "classifier_backbone",
    "clf_optimizer",
    "scaler_mode",
    "buffer_update_mode",
    "result_validity",
    "paper_use_category",
    "validity_reason",
] + config.METRICS

AGGREGATE_FIELDS = [
    "dataset",
    "method",
    "method_raw",
    "setting_type",
    "k",
    "runs",
    "seeds",
    "source_file",
    "source_files",
    "duplicate_source_count",
    "source_kind",
    "context_label",
    "per_task_seen_mean",
    "per_task_seen_min",
    "per_task_seen_max",
    "use_diversity_buffer",
    "use_kd",
    "use_proto_align",
    "gan_train_on_real_buffer",
    "use_wgan_gp",
    "use_projection_critic",
    "main_method_variant",
    "classifier_backbone",
    "clf_optimizer",
    "scaler_mode",
    "buffer_update_mode",
    "result_validity",
    "paper_use_category",
    "validity_reason",
] + config.METRICS


def discover_json_files() -> list[Path]:
    if not config.RESULT_ROOT.exists():
        return []
    return sorted(config.RESULT_ROOT.rglob("*.json"), key=lambda p: p.as_posix().lower())


def missing_report_rows(table_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for setting in sorted({str(row.get("setting_type", "")) for row in rows}):
        scopes.append((f"setting_type={setting}", [row for row in rows if row.get("setting_type") == setting]))
    for scope, scoped_rows in scopes:
        total = len(scoped_rows)
        for metric in config.METRICS:
            missing = sum(1 for row in scoped_rows if row.get(metric) in (None, ""))
            report.append(
                {
                    "table": table_name,
                    "scope": scope,
                    "metric": metric,
                    "rows": total,
                    "missing": missing,
                    "available": total - missing,
                    "missing_percent": (100.0 * missing / total) if total else "",
                }
            )
    return report


def run() -> dict[str, int]:
    ensure_output_dirs()
    json_files = discover_json_files()
    manifest: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for path in json_files:
        parsed_runs = 0
        parsed_aggregates = 0
        error = ""
        status = "ok"
        try:
            data = load_json_object(path)
            new_runs, new_aggregates = extract_rows_from_json(path, data)
            run_rows.extend(new_runs)
            aggregate_rows.extend(new_aggregates)
            parsed_runs = len(new_runs)
            parsed_aggregates = len(new_aggregates)
        except Exception as exc:  # noqa: BLE001 - manifest must preserve malformed-file errors.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        manifest.append(
            {
                "source_file": rel_path(path),
                "setting_type": infer_setting_type(path),
                "status": status,
                "parsed_run_rows": parsed_runs,
                "parsed_aggregate_rows": parsed_aggregates,
                "file_size_bytes": path.stat().st_size if path.exists() else "",
                "error": error,
            }
        )

    merged_runs = [apply_validity_fields(row) for row in merge_duplicate_rows(run_rows)]
    merged_aggregates = [apply_validity_fields(row) for row in merge_duplicate_rows(aggregate_rows)]
    missing_rows = (
        missing_report_rows("all_runs_long", merged_runs)
        + missing_report_rows("all_aggregates_long", merged_aggregates)
    )

    write_csv(merged_runs, config.RAW_DIR / "all_runs_long.csv", RUN_FIELDS)
    write_csv(merged_aggregates, config.RAW_DIR / "all_aggregates_long.csv", AGGREGATE_FIELDS)
    write_csv(
        manifest,
        config.RAW_DIR / "file_manifest.csv",
        [
            "source_file",
            "setting_type",
            "status",
            "parsed_run_rows",
            "parsed_aggregate_rows",
            "file_size_bytes",
            "error",
        ],
    )
    write_csv(
        missing_rows,
        config.RAW_DIR / "missing_metrics_report.csv",
        ["table", "scope", "metric", "rows", "missing", "available", "missing_percent"],
    )

    return {
        "json_files": len(json_files),
        "runs": len(merged_runs),
        "aggregates": len(merged_aggregates),
        "manifest_errors": sum(1 for row in manifest if row["status"] != "ok"),
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Collected {runs} runs and {aggregates} aggregates from {json_files} JSON files "
        "({manifest_errors} manifest errors).".format(**summary)
    )


if __name__ == "__main__":
    main()
