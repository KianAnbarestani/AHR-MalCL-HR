#!/usr/bin/env python3
"""Run safety and consistency checks on collected AHR-MalCL metrics."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs, write_csv


VERSION = "v1 - sanity checks for statistical validation"


CHECK_FIELDS = [
    "check",
    "severity",
    "dataset",
    "setting_type",
    "method",
    "k",
    "seed",
    "result_validity",
    "paper_use_category",
    "metric",
    "status",
    "detail",
]


def bool_state(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def value_present(value: Any) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None", "none", "NaN"}


def add_issue(
    rows: list[dict[str, Any]],
    check: str,
    severity: str,
    status: str,
    detail: str,
    row: Any | None = None,
    metric: str = "",
) -> None:
    rows.append(
        {
            "check": check,
            "severity": severity,
            "dataset": "" if row is None else row.get("dataset", ""),
            "setting_type": "" if row is None else row.get("setting_type", ""),
            "method": "" if row is None else row.get("method", ""),
            "k": "" if row is None else row.get("k", ""),
            "seed": "" if row is None else row.get("seed", ""),
            "result_validity": "" if row is None else row.get("result_validity", ""),
            "paper_use_category": "" if row is None else row.get("paper_use_category", ""),
            "metric": metric,
            "status": status,
            "detail": detail,
        }
    )


def row_stub(
    dataset: Any,
    setting_type: Any,
    method: Any,
    k: Any,
    seed: Any = "",
    result_validity: Any = "",
    paper_use_category: Any = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "setting_type": setting_type,
        "method": method,
        "k": k,
        "seed": seed,
        "result_validity": result_validity,
        "paper_use_category": paper_use_category,
    }


def issue_subset(rows: list[dict[str, Any]], severity: str, status: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("severity") == severity and (status is None or row.get("status") == status)
    ]


def run() -> dict[str, int]:
    import pandas as pd

    ensure_output_dirs()
    runs_path = config.RAW_DIR / "all_runs_long.csv"
    aggregates_path = config.RAW_DIR / "all_aggregates_long.csv"
    if not runs_path.exists() or not aggregates_path.exists():
        raise FileNotFoundError("Run collect_results.py before sanity_checks.py")

    runs = pd.read_csv(runs_path)
    aggregates = pd.read_csv(aggregates_path)
    issues: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []

    for column in config.METRICS:
        if column not in runs.columns:
            add_issue(issues, "numeric_metric_columns", "error", "missing", f"Missing run metric column {column}", metric=column)
            continue
        converted = pd.to_numeric(runs[column], errors="coerce")
        non_numeric = runs[column].notna() & converted.isna()
        if non_numeric.any():
            add_issue(
                issues,
                "numeric_metric_columns",
                "error",
                "fail",
                f"{int(non_numeric.sum())} non-numeric values in run column {column}",
                metric=column,
            )
        else:
            add_issue(issues, "numeric_metric_columns", "info", "pass", f"{column} is numeric where present", metric=column)

    five_seed_validities = {"valid_main_hr", "valid_memory_efficient_hr", "valid_ablation"}
    scoped = runs[runs["result_validity"].isin(five_seed_validities)].copy()
    if not scoped.empty:
        for keys, group in scoped.groupby(
            ["dataset", "setting_type", "method", "k", "result_validity", "paper_use_category"],
            dropna=False,
        ):
            dataset, setting_type, method, k, result_validity, paper_use_category = keys
            seeds = sorted(int(x) for x in pd.to_numeric(group["seed"], errors="coerce").dropna().unique())
            missing = [seed for seed in config.EXPECTED_SEEDS if seed not in seeds]
            extra = [seed for seed in seeds if seed not in config.EXPECTED_SEEDS]
            pseudo_row = row_stub(dataset, setting_type, method, k, "", result_validity, paper_use_category)
            if missing or extra:
                add_issue(
                    issues,
                    "expected_seed_coverage",
                    "critical",
                    "fail",
                    f"seeds={seeds}; missing={missing}; extra={extra}",
                    pseudo_row,
                )
            else:
                add_issue(issues, "expected_seed_coverage", "info", "pass", f"seeds={seeds}", pseudo_row)

    paper_claim_rows = runs[
        runs["paper_use_category"].isin(["main_final_table", "memory_efficient_hr_table", "ablation_table"])
    ]
    for _, row in paper_claim_rows.iterrows():
        if value_present(row.get("early_stop_reason")):
            add_issue(
                issues,
                "paper_claim_early_stop",
                "critical",
                "fail",
                f"early_stop_reason={row.get('early_stop_reason')}",
                row,
            )
    if paper_claim_rows.empty or not any(value_present(x) for x in paper_claim_rows.get("early_stop_reason", [])):
        add_issue(issues, "paper_claim_early_stop", "info", "pass", "No early stopped paper-claim rows detected")

    final_main = runs[runs["result_validity"] == "valid_main_hr"]
    expected_flags = {
        "use_diversity_buffer": False,
        "use_kd": False,
        "use_proto_align": False,
        "gan_train_on_real_buffer": True,
    }
    for _, row in final_main.iterrows():
        for flag, expected in expected_flags.items():
            actual = bool_state(row.get(flag))
            if actual is None:
                add_issue(issues, "final_config_flags", "warning", "missing", f"{flag} missing", row, metric=flag)
            elif actual != expected:
                add_issue(
                    issues,
                    "final_config_flags",
                    "critical",
                    "fail",
                    f"{flag} expected {expected} but found {actual}",
                    row,
                    metric=flag,
                )
    if not final_main.empty:
        add_issue(issues, "final_config_flags", "info", "pass", "Checked final main-method config flags")

    memory_efficient_hr = runs[runs["result_validity"] == "valid_memory_efficient_hr"]
    for _, row in memory_efficient_hr.iterrows():
        for flag, expected in expected_flags.items():
            actual = bool_state(row.get(flag))
            if actual is None:
                add_issue(issues, "memory_efficient_hr_config_flags", "warning", "missing", f"{flag} missing", row, metric=flag)
            elif actual != expected:
                add_issue(
                    issues,
                    "memory_efficient_hr_config_flags",
                    "critical",
                    "fail",
                    f"{flag} expected {expected} but found {actual}",
                    row,
                    metric=flag,
                )
    if not memory_efficient_hr.empty:
        add_issue(
            issues,
            "memory_efficient_hr_config_flags",
            "info",
            "pass",
            "Checked valid memory-efficient HR config flags",
        )

    for _, row in runs[runs["result_validity"] == "legacy_or_config_mismatch"].iterrows():
        add_issue(
            issues,
            "legacy_or_config_mismatch",
            "legacy",
            "record",
            str(row.get("validity_reason", "")),
            row,
        )

    for _, row in runs[runs["result_validity"] == "excluded_incomplete"].iterrows():
        add_issue(
            issues,
            "expected_exclusion",
            "expected_exclusion",
            "record",
            f"{row.get('validity_reason', '')}; source={row.get('source_files') or row.get('source_file')}",
            row,
        )

    ablation = runs[runs["setting_type"] == "ablation"]
    for keys, group in ablation.groupby(["dataset", "k"], dropna=False):
        dataset, k = keys
        methods = sorted(str(x) for x in group["method"].dropna().unique())
        ref_group = group[group["method"] == "hybrid_random_buffer"]
        ref_seeds = sorted(int(x) for x in pd.to_numeric(ref_group["seed"], errors="coerce").dropna().unique())
        for method in methods:
            method_group = group[group["method"] == method]
            seeds = sorted(int(x) for x in pd.to_numeric(method_group["seed"], errors="coerce").dropna().unique())
            matches_ref = seeds == ref_seeds
            pairing_rows.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "reference_method": "hybrid_random_buffer",
                    "method": method,
                    "reference_seeds": ",".join(map(str, ref_seeds)),
                    "method_seeds": ",".join(map(str, seeds)),
                    "n_reference": len(ref_seeds),
                    "n_method": len(seeds),
                    "matches_reference": matches_ref,
                }
            )
            if method != "hybrid_random_buffer" and not matches_ref:
                add_issue(
                    issues,
                    "ablation_pairing",
                    "critical",
                    "fail",
                    f"{method} seeds {seeds} do not match reference seeds {ref_seeds}",
                    row_stub(dataset, "ablation", method, k),
                )
    if pairing_rows and not any(row["check"] == "ablation_pairing" and row["status"] == "fail" for row in issues):
        add_issue(issues, "ablation_pairing", "info", "pass", "Ablation methods share reference seed sets")

    for metric in config.METRICS:
        missing_runs = int(runs[metric].isna().sum()) if metric in runs.columns else len(runs)
        missing_aggs = int(aggregates[metric].isna().sum()) if metric in aggregates.columns else len(aggregates)
        add_issue(
            issues,
            "nan_report",
            "info",
            "record",
            f"runs_missing={missing_runs}; aggregates_missing={missing_aggs}",
            metric=metric,
        )

    if "duplicate_source_count" in runs.columns:
        duplicated = runs[pd.to_numeric(runs["duplicate_source_count"], errors="coerce").fillna(1) > 1]
        for _, row in duplicated.iterrows():
            add_issue(
                issues,
                "duplicate_run_report",
                "info",
                "record",
                f"merged duplicate sources={row.get('duplicate_source_count')}",
                row,
            )

    for keys, group in runs.groupby(["dataset", "setting_type", "method", "k"], dropna=False):
        hashes = sorted(str(x) for x in group["config_hash"].dropna().unique() if str(x).strip())
        if len(hashes) > 1:
            dataset, setting_type, method, k = keys
            add_issue(
                issues,
                "inconsistent_config_report",
                "warning",
                "record",
                f"Multiple config hashes in group: {hashes}",
                {"dataset": dataset, "setting_type": setting_type, "method": method, "k": k, "seed": ""},
            )

    memory_cols = {"memory_MB", "replay_memory_MB", "model_memory_MB"}
    if memory_cols.issubset(set(runs.columns)):
        for _, row in runs.iterrows():
            memory = pd.to_numeric(row.get("memory_MB"), errors="coerce")
            replay = pd.to_numeric(row.get("replay_memory_MB"), errors="coerce")
            model = pd.to_numeric(row.get("model_memory_MB"), errors="coerce")
            if pd.notna(memory) and pd.notna(replay) and pd.notna(model):
                delta = abs(float(memory) - (float(replay) + float(model)))
                if delta > 1.0:
                    add_issue(
                        issues,
                        "memory_consistency_report",
                        "warning",
                        "record",
                        f"memory_MB differs from replay+model by {delta:.3f} MB",
                        row,
                    )
    add_issue(issues, "memory_consistency_report", "info", "record", "Memory consistency checked where component values exist")

    write_csv(issues, config.CHECKS_DIR / "sanity_check_report.csv", CHECK_FIELDS)
    write_csv(
        pairing_rows,
        config.CHECKS_DIR / "pairing_matrix.csv",
        [
            "dataset",
            "k",
            "reference_method",
            "method",
            "reference_seeds",
            "method_seeds",
            "n_reference",
            "n_method",
            "matches_reference",
        ],
    )

    critical_failures = issue_subset(issues, "critical", "fail")
    warnings = issue_subset(issues, "warning")
    expected_exclusions = issue_subset(issues, "expected_exclusion")
    legacy_rows = issue_subset(issues, "legacy")
    main_final_rows = runs[
        (runs["result_validity"] == "valid_main_hr")
        & (runs["paper_use_category"] == "main_final_table")
    ]
    main_final_safe = len(critical_failures) == 0
    for dataset, expected_k in config.FINAL_BEST_K.items():
        group = main_final_rows[
            (main_final_rows["dataset"] == dataset)
            & (pd.to_numeric(main_final_rows["k"], errors="coerce") == expected_k)
            & (main_final_rows["method"] == config.MAIN_METHOD_CANONICAL)
        ]
        seeds = sorted(int(x) for x in pd.to_numeric(group["seed"], errors="coerce").dropna().unique())
        if seeds != config.EXPECTED_SEEDS:
            main_final_safe = False
            add_issue(
                issues,
                "main_final_decision",
                "critical",
                "fail",
                f"{dataset} final main HR seeds are {seeds}, expected {config.EXPECTED_SEEDS}",
                row_stub(dataset, "final_best_performance", config.MAIN_METHOD_CANONICAL, expected_k, "", "valid_main_hr", "main_final_table"),
            )
    critical_failures = issue_subset(issues, "critical", "fail")
    main_final_safe = main_final_safe and len(critical_failures) == 0
    write_csv(issues, config.CHECKS_DIR / "sanity_check_report.csv", CHECK_FIELDS)
    lines = [
        "# Sanity Check Report",
        "",
        f"- Run rows checked: {len(runs)}",
        f"- Aggregate rows checked: {len(aggregates)}",
        f"- Critical failures: {len(critical_failures)}",
        f"- Warnings: {len(warnings)}",
        f"- Expected exclusions: {len(expected_exclusions)}",
        f"- Legacy/config mismatch rows: {len(legacy_rows)}",
        f"- Final main HR table is safe to use: {'YES' if main_final_safe else 'NO'}",
        "",
        "## Critical Failures",
        "",
    ]
    if critical_failures:
        for row in critical_failures:
            lines.append(
                f"- {row['check']}: {row['dataset']} {row['setting_type']} {row['method']} "
                f"K={row['k']} seed={row['seed']} {row['detail']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for row in warnings:
            lines.append(
                f"- {row['check']}: {row['dataset']} {row['setting_type']} {row['method']} "
                f"K={row['k']} seed={row['seed']} {row['detail']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Expected Exclusions", ""])
    if expected_exclusions:
        for row in expected_exclusions:
            lines.append(f"- {row['detail']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Legacy / Config Mismatch Rows", ""])
    if legacy_rows:
        for row in legacy_rows:
            lines.append(
                f"- {row['dataset']} {row['setting_type']} {row['method']} K={row['k']} "
                f"seed={row['seed']}: {row['detail']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Final Decision", ""])
    if main_final_safe:
        lines.append("- Main final HR results are safe to use.")
    else:
        lines.append("- Main final HR results are not safe to use until critical failures are resolved.")
    lines.extend(["", "## Notes", ""])
    lines.append("- Missing metrics are recorded in `raw/missing_metrics_report.csv` and summarized in the CSV report.")
    lines.append("- Pairing compatibility for ablation tests is written to `checks/pairing_matrix.csv`.")
    (config.CHECKS_DIR / "sanity_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "issues": len(issues),
        "critical_failures": len(critical_failures),
        "pairing_rows": len(pairing_rows),
        "legacy_rows": len(legacy_rows),
        "expected_exclusions": len(expected_exclusions),
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote sanity checks with {issues} records, {critical_failures} critical failures, "
        "{legacy_rows} legacy rows, {expected_exclusions} expected exclusions, "
        "and {pairing_rows} pairing rows.".format(**summary)
    )


if __name__ == "__main__":
    main()
