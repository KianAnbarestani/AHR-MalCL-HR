#!/usr/bin/env python3
"""Integrate future external-baseline outputs without touching trusted results."""

from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path
from statistics import mean
from typing import Any


VERSION = "v2 - external baseline result integration with input-root"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "result_external_baselines"
OUT_DIR = PROJECT_ROOT / "paper_ready_baselines"
TRUSTED_RUNS = PROJECT_ROOT / "paper_ready_stats" / "raw" / "all_runs_long.csv"

EXPECTED_BASELINES = [
    "DERPP",
    "MADAR",
    "MalCL-faithful",
    "Joint",
    "ER",
    "LwF",
    "EWC",
    "iCaRL",
    "FreeMOCA",
]

SUMMARY_COLUMNS = [
    "baseline",
    "dataset",
    "k",
    "seed_count",
    "seeds",
    "mean_acc_seen_mean",
    "final_taskwise_average_accuracy_mean",
    "forgetting_mean",
    "macro_f1_mean",
    "balanced_accuracy_mean",
    "memory_MB_mean",
    "source_files",
    "comparison_mode",
]

TEST_COLUMNS = [
    "baseline",
    "dataset",
    "k",
    "metric",
    "comparison_target",
    "paired_seed_count",
    "test",
    "statistic",
    "p_value",
    "note",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def metric_from_payload(payload: dict[str, Any], metric: str) -> float | None:
    aliases = {
        "mean_acc_seen": ["mean_acc_seen", "mean_accuracy", "mean_acc_seen_mean"],
        "final_taskwise_average_accuracy": [
            "final_taskwise_average_accuracy",
            "final_average_accuracy_taskwise",
            "final_taskwise_accuracy",
            "final_taskwise_average_accuracy_mean",
        ],
        "forgetting": ["forgetting", "forgetting_mean"],
        "macro_f1": ["macro_f1", "macro_f1_mean", "f1_macro"],
        "balanced_accuracy": ["balanced_accuracy", "balanced_accuracy_mean"],
        "memory_MB": ["memory_MB", "memory_MB_mean"],
    }
    for key in aliases.get(metric, [metric]):
        if key in payload:
            return safe_float(payload.get(key))
    for nested_key in ("metrics", "cl_metrics", "task_metrics", "aggregate"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            value = metric_from_payload(nested, metric)
            if value is not None:
                return value
    return None


def infer_from_path(path: Path, input_root: Path) -> dict[str, str]:
    parts = path.relative_to(input_root).parts
    baseline = parts[0] if parts else ""
    dataset = ""
    k = ""
    seed = ""
    for part in parts:
        lower = part.lower()
        if lower in {"ember", "az-class", "az_class", "az"}:
            dataset = "ember" if lower == "ember" else "az_class"
        if lower.startswith("k="):
            k = part.split("=", 1)[1]
        if lower.startswith("seed="):
            seed = part.split("=", 1)[1]
    return {"baseline": baseline, "dataset": dataset, "k": k, "seed": seed}


def discover_external_runs(input_root: Path) -> list[dict[str, Any]]:
    if not input_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("*.json")):
        if path.name not in {"full-result.json", "aggregate.json"} and "result" not in path.name.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            rows.append({"source_file": str(path.relative_to(PROJECT_ROOT)), "parse_error": f"{type(exc).__name__}: {exc}"})
            continue
        info = infer_from_path(path, input_root)
        if isinstance(payload, dict):
            info.update(
                {
                    "seed": str(payload.get("seed", info.get("seed", ""))),
                    "source_file": str(path.relative_to(PROJECT_ROOT)),
                    "mean_acc_seen": metric_from_payload(payload, "mean_acc_seen"),
                    "final_taskwise_average_accuracy": metric_from_payload(payload, "final_taskwise_average_accuracy"),
                    "forgetting": metric_from_payload(payload, "forgetting"),
                    "macro_f1": metric_from_payload(payload, "macro_f1"),
                    "balanced_accuracy": metric_from_payload(payload, "balanced_accuracy"),
                    "memory_MB": metric_from_payload(payload, "memory_MB"),
                }
            )
            rows.append(info)
    return rows


def aggregate_external(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("parse_error"):
            continue
        key = (str(row.get("baseline", "")), str(row.get("dataset", "")), str(row.get("k", "")))
        groups.setdefault(key, []).append(row)
    out = []
    for (baseline, dataset, k), items in sorted(groups.items()):
        seeds = sorted({str(item.get("seed", "")) for item in items if str(item.get("seed", ""))})
        summary: dict[str, Any] = {
            "baseline": baseline,
            "dataset": dataset,
            "k": k,
            "seed_count": len(seeds),
            "seeds": ";".join(seeds),
            "source_files": ";".join(str(item.get("source_file", "")) for item in items),
            "comparison_mode": "paired tests only if seed sets match trusted rows",
        }
        for metric in [
            "mean_acc_seen",
            "final_taskwise_average_accuracy",
            "forgetting",
            "macro_f1",
            "balanced_accuracy",
            "memory_MB",
        ]:
            values = [safe_float(item.get(metric)) for item in items]
            values = [value for value in values if value is not None]
            summary[f"{metric}_mean"] = mean(values) if values else ""
        out.append(summary)
    return out


def trusted_target_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(TRUSTED_RUNS)
    out = []
    for row in rows:
        if row.get("method") != "hybrid_random_buffer":
            continue
        if row.get("paper_use_category") in {"main_hr_table", "memory_efficient_hr_table"}:
            out.append(row)
    return out


def build_stat_tests(external_rows: list[dict[str, Any]], trusted_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not external_rows:
        return []
    tests: list[dict[str, Any]] = []
    metrics = ["mean_acc_seen", "final_taskwise_average_accuracy", "forgetting", "macro_f1", "balanced_accuracy"]
    trusted_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in trusted_rows:
        trusted_by_key.setdefault((row.get("dataset", ""), str(row.get("k", ""))), []).append(row)
    external_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in external_rows:
        external_groups.setdefault((str(row.get("baseline", "")), str(row.get("dataset", "")), str(row.get("k", ""))), []).append(row)

    try:
        from scipy import stats
    except Exception:  # noqa: BLE001
        stats = None

    for (baseline, dataset, k), items in sorted(external_groups.items()):
        trusted = trusted_by_key.get((dataset, k), [])
        if not trusted:
            tests.append(
                {
                    "baseline": baseline,
                    "dataset": dataset,
                    "k": k,
                    "metric": "",
                    "comparison_target": "AHR-MalCL-HR",
                    "paired_seed_count": 0,
                    "test": "descriptive_only",
                    "statistic": "",
                    "p_value": "",
                    "note": "No trusted target rows with matching dataset and K.",
                }
            )
            continue
        for metric in metrics:
            ext_by_seed = {str(row.get("seed")): safe_float(row.get(metric)) for row in items}
            trusted_by_seed = {str(row.get("seed")): safe_float(row.get(metric)) for row in trusted}
            paired = [
                (ext_by_seed[seed], trusted_by_seed[seed])
                for seed in sorted(set(ext_by_seed) & set(trusted_by_seed))
                if ext_by_seed.get(seed) is not None and trusted_by_seed.get(seed) is not None
            ]
            if len(paired) >= 2 and stats is not None:
                ext_values = [pair[0] for pair in paired]
                trusted_values = [pair[1] for pair in paired]
                result = stats.ttest_rel(ext_values, trusted_values)
                tests.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "k": k,
                        "metric": metric,
                        "comparison_target": "AHR-MalCL-HR",
                        "paired_seed_count": len(paired),
                        "test": "paired_t",
                        "statistic": result.statistic,
                        "p_value": result.pvalue,
                        "note": "Seeds matched; external minus target direction is not Holm-corrected here.",
                    }
                )
            else:
                tests.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "k": k,
                        "metric": metric,
                        "comparison_target": "AHR-MalCL-HR",
                        "paired_seed_count": len(paired),
                        "test": "descriptive_only",
                        "statistic": "",
                        "p_value": "",
                        "note": "Not enough matching seed values for paired test.",
                    }
                )
    return tests


def write_missing_report(external_rows: list[dict[str, Any]], input_root: Path) -> None:
    present = {str(row.get("baseline", "")) for row in external_rows if row.get("baseline")}
    missing = [name for name in EXPECTED_BASELINES if name not in present]
    missing_lines = [f"- {name}" for name in missing] if missing else ["- none"]
    lines = [
        "# External Baseline Missing Results",
        "",
        f"Input root: `{input_root}`.",
        f"External result JSON files found: {len(external_rows)}.",
        "",
        "## Missing Expected Baselines",
        "",
        *missing_lines,
        "",
        "## Notes",
        "",
        "- This is expected before DER++, MADAR, ER, Joint, or optional external baseline training has been run.",
        "- Paired statistical tests will only be created when external and trusted seed sets match.",
        "- Do not place external baseline outputs under `result/`; use `result_external_baselines/`.",
        "",
    ]
    (OUT_DIR / "external_baseline_missing_results.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# External Baseline Summary",
        "",
    ]
    if not summary_rows:
        lines.extend(
            [
                "No external baseline result rows were found yet.",
                "",
                "Run external baselines into `result_external_baselines/` and rerun `baseline_suite/integrate_external_baseline_results.py`.",
                "",
            ]
        )
    else:
        lines.append("| baseline | dataset | k | seeds | final_taskwise_average_accuracy_mean | forgetting_mean |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in summary_rows:
            lines.append(
                "| {baseline} | {dataset} | {k} | {seeds} | {acc} | {forgetting} |".format(
                    baseline=row.get("baseline", ""),
                    dataset=row.get("dataset", ""),
                    k=row.get("k", ""),
                    seeds=row.get("seeds", ""),
                    acc=row.get("final_taskwise_average_accuracy_mean", ""),
                    forgetting=row.get("forgetting_mean", ""),
                )
            )
        lines.append("")
    (OUT_DIR / "external_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_root: Path | None = None) -> dict[str, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = (input_root or EXTERNAL_ROOT)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    external_rows = discover_external_runs(root)
    summary_rows = aggregate_external(external_rows)
    tests = build_stat_tests(external_rows, trusted_target_rows())
    write_csv(OUT_DIR / "external_baseline_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_summary_markdown(summary_rows)
    write_csv(OUT_DIR / "external_baseline_stat_tests.csv", tests, TEST_COLUMNS)
    write_missing_report(external_rows, root)
    return {
        "external_json_rows": len(external_rows),
        "summary_rows": len(summary_rows),
        "test_rows": len(tests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate external baseline outputs.")
    parser.add_argument("--input-root", default="result_external_baselines", help="Local or copied Colab result root.")
    return parser.parse_args()


def main() -> None:
    print(VERSION, flush=True)
    args = parse_args()
    summary = run(Path(args.input_root))
    print(
        "Integrated {external_json_rows} external JSON rows into {summary_rows} summaries and {test_rows} tests.".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
