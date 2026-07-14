#!/usr/bin/env python3
"""Run the complete statistical validation pipeline in order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs, rel_path


VERSION = "v1 - run complete statistical validation pipeline"

PIPELINE_SCRIPTS = [
    "collect_results.py",
    "sanity_checks.py",
    "descriptive_stats.py",
    "stat_tests.py",
    "memory_budget_analysis.py",
    "replay_drift_analysis.py",
    "make_tables.py",
    "make_figures.py",
]


def read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def bool_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def generated_files() -> list[str]:
    if not config.OUTPUT_ROOT.exists():
        return []
    return sorted(rel_path(path) for path in config.OUTPUT_ROOT.rglob("*") if path.is_file())


def generate_summary_report() -> None:
    ensure_output_dirs()
    table1 = read_csv(config.TABLES_DIR / "table_1_final_main_results.csv")
    corrected = read_csv(config.STATS_DIR / "holm_corrected_tests.csv")
    skipped = read_csv(config.STATS_DIR / "skipped_tests.csv")
    sanity = read_csv(config.CHECKS_DIR / "sanity_check_report.csv")
    elbow = read_csv(config.TABLES_DIR / "table_memory_budget_elbow.csv")
    replay_corr = read_csv(config.STATS_DIR / "replay_drift_correlations.csv")
    manifest = read_csv(config.RAW_DIR / "file_manifest.csv")
    missing = read_csv(config.RAW_DIR / "missing_metrics_report.csv")
    runs = read_csv(config.RAW_DIR / "all_runs_long.csv")
    validity_audit = read_csv(config.TABLES_DIR / "table_validity_audit.csv")
    memory_efficient_hr = read_csv(config.TABLES_DIR / "table_memory_efficient_hr_results.csv")
    legacy_memory_efficient = read_csv(config.TABLES_DIR / "table_legacy_memory_efficient_results.csv")

    critical_failures = (
        sanity[(sanity["severity"] == "critical") & (sanity["status"] == "fail")]
        if not sanity.empty and {"severity", "status"}.issubset(set(sanity.columns))
        else sanity.iloc[0:0]
    )
    expected_exclusions = (
        sanity[sanity["severity"] == "expected_exclusion"]
        if not sanity.empty and "severity" in sanity.columns
        else sanity.iloc[0:0]
    )
    legacy_rows = (
        runs[runs["result_validity"] == "legacy_or_config_mismatch"]
        if not runs.empty and "result_validity" in runs.columns
        else runs.iloc[0:0]
    )
    excluded_rows = (
        runs[runs["result_validity"] == "excluded_incomplete"]
        if not runs.empty and "result_validity" in runs.columns
        else runs.iloc[0:0]
    )
    main_final_safe = len(critical_failures) == 0 and len(table1) == 2
    memory_efficient_used_as_main = (
        not table1.empty
        and "Setting" in table1.columns
        and table1["Setting"].astype(str).str.contains("memory efficient", case=False, na=False).any()
    )
    memory_efficient_expected = {("az_class", 100), ("ember", 25)}
    if not memory_efficient_hr.empty and {"Dataset", "K"}.issubset(set(memory_efficient_hr.columns)):
        memory_efficient_seen = {
            (str(row.get("Dataset")), int(row.get("K")))
            for _, row in memory_efficient_hr.iterrows()
        }
    else:
        memory_efficient_seen = set()
    memory_efficient_hr_safe = len(critical_failures) == 0 and memory_efficient_seen == memory_efficient_expected
    result_files_modified = False
    training_rerun_needed = False

    lines: list[str] = [
        "# Statistical Validation Summary",
        "",
        "## Evidence Decisions",
        "",
        f"Final main HR table is safe to use: {'YES' if main_final_safe else 'NO'}",
        f"Memory-efficient HR table is safe to use: {'YES' if memory_efficient_hr_safe else 'NO'}",
        f"Memory-efficient legacy/config-mismatch rows used as main HR results: {'YES' if memory_efficient_used_as_main else 'NO'}",
        f"Training rerun needed: {'YES' if training_rerun_needed else 'NO'}",
        f"Result files modified: {'YES' if result_files_modified else 'NO'}",
        "",
        "## Final Main Result Summary",
        "",
    ]
    if table1.empty:
        lines.append("- No final-main result table was generated.")
    else:
        for _, row in table1.iterrows():
            lines.append(
                f"- {row.get('Dataset')} {row.get('Setting')} K={row.get('K')}: "
                f"mean seen accuracy {row.get('Mean Seen Accuracy')}, "
                f"final taskwise accuracy {row.get('Final Taskwise Accuracy')}, "
                f"forgetting {row.get('Forgetting')}."
            )
    lines.extend(
        [
            "",
            f"- Final main HR table is safe to use: {'YES' if main_final_safe else 'NO'}",
            f"- Memory-efficient legacy/config-mismatch rows used as main HR results: {'YES' if memory_efficient_used_as_main else 'NO'}",
        ]
    )

    lines.extend(["", "## Memory-Efficient Simplified HR Summary", ""])
    if memory_efficient_hr.empty:
        lines.append("- No valid simplified HR memory-efficient rows were generated.")
    else:
        for _, row in memory_efficient_hr.iterrows():
            source_label = (
                "ablation hybrid_random_buffer"
                if str(row.get("Dataset")) == "az_class"
                else "ablation hybrid-random-buffer"
            )
            lines.append(
                f"- {row.get('Dataset')} K={row.get('K')} from {source_label}: "
                f"mean seen accuracy {row.get('Mean Seen Accuracy')}, "
                f"final taskwise accuracy {row.get('Final Taskwise Accuracy')}, "
                f"forgetting {row.get('Forgetting')}."
            )
        lines.append(f"- Memory-efficient HR table is safe to use: {'YES' if memory_efficient_hr_safe else 'NO'}")
        lines.append("- No rerun is needed for memory-efficient AHR-MalCL-HR results.")

    lines.extend(["", "## Statistical Test Highlights", ""])
    if corrected.empty:
        lines.append("- No paired statistical tests were available.")
    else:
        sig = corrected[corrected["holm_reject_0_05"].map(bool_true)] if "holm_reject_0_05" in corrected.columns else corrected.iloc[0:0]
        nonsig = corrected[~corrected.index.isin(sig.index)] if not sig.empty else corrected
        lines.append(f"- Holm-significant test rows: {len(sig)}")
        lines.append(f"- Non-significant or uncorrected test rows: {len(nonsig)}")
        for _, row in sig.head(12).iterrows():
            lines.append(
                f"- Significant: {row.get('dataset')} K={row.get('k')} {row.get('test')} "
                f"{row.get('comparison_label')} on {row.get('metric')} "
                f"(Holm p={row.get('holm_p_value')})."
            )
    if not skipped.empty:
        lines.append(f"- Skipped metric-comparison rows: {len(skipped)}; see `stats/skipped_tests.csv`.")

    lines.extend(["", "## Memory-Budget Elbow Summary", ""])
    lines.append(
        "- Memory-budget rows come from older/full-complexity configuration and should be interpreted as "
        "budget-sensitivity evidence, not as the official final simplified AHR-MalCL-HR main result."
    )
    if elbow.empty:
        lines.append("- No memory-budget elbow rows were generated.")
    else:
        for _, row in elbow.iterrows():
            lines.append(
                f"- {row.get('dataset')}: recommended practical K={row.get('recommended_practical_k')}; "
                f"best accuracy K={row.get('best_accuracy_k')}; "
                f"lowest forgetting K={row.get('lowest_forgetting_k')}."
            )

    lines.extend(["", "## Replay-Drift Summary", ""])
    if replay_corr.empty:
        lines.append("- No replay-drift correlations were generated.")
    else:
        insufficient_replay = False
        for _, row in replay_corr.iterrows():
            note = row.get("note", "")
            detail = f" note={note}" if isinstance(note, str) and note else ""
            if isinstance(note, str) and "not enough points" in note:
                insufficient_replay = True
            lines.append(
                f"- {row.get('dataset')}: {row.get('x_metric')} vs {row.get('y_metric')} "
                f"Pearson={row.get('pearson_correlation')}, Spearman={row.get('spearman_correlation')}.{detail}"
            )
        if insufficient_replay:
            lines.append("- Replay-drift correlations still have insufficient points and should not be used as correlation claims.")

    lines.extend(["", "## Validity Audit Summary", ""])
    if validity_audit.empty:
        lines.append("- No validity audit table was generated.")
    else:
        lines.append(f"- Validity audit rows: {len(validity_audit)}.")
    lines.append(f"- Critical failures: {len(critical_failures)}.")
    lines.append(f"- Legacy/config-mismatch run rows: {len(legacy_rows)}.")
    lines.append(f"- Expected exclusion run rows: {len(excluded_rows)}.")
    if not memory_efficient_hr.empty:
        lines.append(
            f"- Valid simplified HR memory-efficient table rows: {len(memory_efficient_hr)}."
        )
        lines.append("- No rerun is needed for memory-efficient AHR-MalCL-HR results.")
    else:
        lines.append("- No valid simplified HR memory-efficient table rows were found.")
    if not legacy_memory_efficient.empty:
        lines.append(
            f"- Legacy memory-efficient result rows kept as supplementary only: {len(legacy_memory_efficient)}."
        )

    lines.extend(["", "## Excluded Rows", ""])
    if excluded_rows.empty:
        lines.append("- None.")
    else:
        for _, row in excluded_rows.iterrows():
            lines.append(
                f"- {row.get('dataset')} {row.get('setting_type')} {row.get('method')} "
                f"K={row.get('k')} seed={row.get('seed')}: {row.get('paper_use_category')} "
                f"({row.get('result_validity')}) - {row.get('validity_reason')} "
                f"[{row.get('source_files') or row.get('source_file')}]"
            )

    lines.extend(["", "## Supplementary-Only Legacy Rows", ""])
    if legacy_rows.empty:
        lines.append("- None.")
    else:
        for _, row in legacy_rows.iterrows():
            lines.append(
                f"- {row.get('dataset')} {row.get('setting_type')} {row.get('method')} "
                f"K={row.get('k')} seed={row.get('seed')}: {row.get('paper_use_category')} "
                f"({row.get('result_validity')}) - {row.get('validity_reason')} "
                f"[{row.get('source_files') or row.get('source_file')}]"
            )

    manual_review_rows = runs[
        runs["paper_use_category"] == "manual_review_needed"
    ] if not runs.empty and "paper_use_category" in runs.columns else runs.iloc[0:0]
    lines.extend(["", "## Manual-Review Rows Not Used For Main Claims", ""])
    if manual_review_rows.empty:
        lines.append("- None.")
    else:
        for _, row in manual_review_rows.iterrows():
            lines.append(
                f"- {row.get('dataset')} {row.get('setting_type')} {row.get('method')} "
                f"K={row.get('k')} seed={row.get('seed')}: {row.get('validity_reason')} "
                f"[{row.get('source_files') or row.get('source_file')}]"
            )

    lines.extend(["", "## Warnings / Limitations", ""])
    warnings = sanity[sanity["severity"] == "warning"] if not sanity.empty and "severity" in sanity.columns else sanity.iloc[0:0]
    manifest_errors = manifest[manifest["status"] != "ok"] if not manifest.empty and "status" in manifest.columns else manifest.iloc[0:0]
    lines.append(f"- Sanity-check critical failures: {len(critical_failures)}.")
    lines.append(f"- Sanity-check warnings: {len(warnings)}.")
    lines.append(f"- Expected exclusions: {len(expected_exclusions)}.")
    lines.append(f"- Manifest parse errors: {len(manifest_errors)}.")
    if not missing.empty:
        all_missing = missing[(missing["scope"] == "all") & (missing["missing"] > 0)].head(12)
        if not all_missing.empty:
            lines.append("- Missing metrics are expected for some older result files; top records:")
            for _, row in all_missing.iterrows():
                lines.append(f"  - {row.get('table')} {row.get('metric')}: missing {row.get('missing')} of {row.get('rows')}.")
    lines.append("- Paired p-values are restricted to ablation rows with matching dataset, setting, K, and seed sets.")
    lines.append("- Literature reference numbers are not tested for significance because no raw per-seed values are present.")

    lines.extend(["", "## Files Generated", ""])
    for file_path in generated_files():
        lines.append(f"- `{file_path}`")

    lines.extend(
        [
            "",
            "## Next Steps For Paper Writing",
            "",
            "- Inspect `checks/sanity_check_report.md` before quoting any result.",
            "- Use `tables/table_validity_audit.md` to confirm which result rows support each paper claim category.",
            "- Use `tables/table_1_final_main_results.md` through `table_7_runtime_memory.md` as the paper table drafts.",
            "- Use `tables/table_memory_efficient_hr_results.md` for simplified HR memory-efficient evidence.",
            "- Use `stats/holm_corrected_tests.csv` and `tables/table_4_component_effects.md` for ablation claims.",
            "- Use `figure_data/*.csv` as the source of truth if figures need journal-specific restyling.",
        ]
    )
    (config.OUTPUT_ROOT / "STATISTICAL_VALIDATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    ensure_output_dirs()
    for script in PIPELINE_SCRIPTS:
        script_path = PROJECT_ROOT / "stat_validation" / script
        print(f"running {script}", flush=True)
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
    generate_summary_report()


def main() -> None:
    print(VERSION, flush=True)
    run()
    print(f"Wrote {config.OUTPUT_ROOT / 'STATISTICAL_VALIDATION_SUMMARY.md'}")


if __name__ == "__main__":
    main()
