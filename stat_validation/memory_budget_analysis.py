#!/usr/bin/env python3
"""Analyze K-budget behavior for memory, accuracy, forgetting, and replay quality."""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs


VERSION = "v1 - memory budget analysis"


def mean_or_nan(series: Any) -> float:
    values = series.dropna()
    if values.empty:
        return math.nan
    return float(values.mean())


def summarize_memory_budget(runs: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(["dataset", "k"], dropna=False):
        dataset, k = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "k": int(k) if pd.notna(k) else math.nan,
            "n": int(group.shape[0]),
            "seeds": ",".join(
                map(str, sorted(int(x) for x in pd.to_numeric(group["seed"], errors="coerce").dropna().unique()))
            ),
        }
        for metric in config.MEMORY_BUDGET_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce") if metric in group.columns else pd.Series(dtype=float)
            row[metric] = mean_or_nan(values)
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.dropna().shape[0] > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "k"], na_position="last")


def add_k0_comparisons(table: Any) -> Any:
    import pandas as pd

    table = table.copy()
    for column in [
        "accuracy_gain_vs_k0",
        "final_accuracy_gain_vs_k0",
        "forgetting_reduction_vs_k0",
        "memory_overhead_vs_k0",
        "accuracy_gain_per_MB",
        "forgetting_reduction_per_MB",
        "mmd_reduction_percent",
        "wasserstein_reduction_percent",
    ]:
        table[column] = math.nan

    for dataset, group in table.groupby("dataset", dropna=False):
        base_rows = group[pd.to_numeric(group["k"], errors="coerce") == 0]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        idxs = group.index
        memory_overhead = table.loc[idxs, "memory_MB"] - base["memory_MB"]
        nonzero_memory_overhead = memory_overhead.mask(memory_overhead == 0)
        table.loc[idxs, "accuracy_gain_vs_k0"] = table.loc[idxs, "mean_acc_seen"] - base["mean_acc_seen"]
        table.loc[idxs, "final_accuracy_gain_vs_k0"] = (
            table.loc[idxs, "final_taskwise_average_accuracy"] - base["final_taskwise_average_accuracy"]
        )
        table.loc[idxs, "forgetting_reduction_vs_k0"] = base["forgetting"] - table.loc[idxs, "forgetting"]
        table.loc[idxs, "memory_overhead_vs_k0"] = memory_overhead
        table.loc[idxs, "accuracy_gain_per_MB"] = table.loc[idxs, "accuracy_gain_vs_k0"] / nonzero_memory_overhead
        table.loc[idxs, "forgetting_reduction_per_MB"] = (
            table.loc[idxs, "forgetting_reduction_vs_k0"] / nonzero_memory_overhead
        )
        if pd.notna(base["mmd_real_generated"]) and base["mmd_real_generated"] != 0:
            table.loc[idxs, "mmd_reduction_percent"] = (
                (base["mmd_real_generated"] - table.loc[idxs, "mmd_real_generated"]) / abs(base["mmd_real_generated"]) * 100.0
            )
        if pd.notna(base["wasserstein_real_generated"]) and base["wasserstein_real_generated"] != 0:
            table.loc[idxs, "wasserstein_reduction_percent"] = (
                (base["wasserstein_real_generated"] - table.loc[idxs, "wasserstein_real_generated"])
                / abs(base["wasserstein_real_generated"])
                * 100.0
            )
    return table


def elbow_rows(memory_table: Any, all_runs: Any) -> list[dict[str, Any]]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    final_best = all_runs[all_runs["setting_type"] == "final_best_performance"].copy()
    if not final_best.empty:
        final_best["mean_acc_seen"] = pd.to_numeric(final_best["mean_acc_seen"], errors="coerce")

    for dataset, group in memory_table.groupby("dataset", dropna=False):
        group = group.copy()
        group["k"] = pd.to_numeric(group["k"], errors="coerce")
        best_accuracy_k = math.nan
        lowest_forgetting_k = math.nan
        best_gain_per_mb_k = math.nan
        k_95 = math.nan
        recommended = math.nan
        threshold_source = "memory_budget_best"

        valid_acc = group.dropna(subset=["mean_acc_seen"])
        if not valid_acc.empty:
            best_accuracy_k = int(valid_acc.loc[valid_acc["mean_acc_seen"].idxmax(), "k"])
            best_memory_budget_acc = float(valid_acc["mean_acc_seen"].max())
        else:
            best_memory_budget_acc = math.nan

        valid_forgetting = group.dropna(subset=["forgetting"])
        if not valid_forgetting.empty:
            lowest_forgetting_k = int(valid_forgetting.loc[valid_forgetting["forgetting"].idxmin(), "k"])

        valid_gain = group.dropna(subset=["accuracy_gain_per_MB"])
        valid_gain = valid_gain[valid_gain["memory_overhead_vs_k0"] > 0]
        if not valid_gain.empty:
            best_gain_per_mb_k = int(valid_gain.loc[valid_gain["accuracy_gain_per_MB"].idxmax(), "k"])

        dataset_final = final_best[final_best["dataset"] == dataset]
        if not dataset_final.empty and dataset_final["mean_acc_seen"].notna().any():
            threshold = 0.95 * float(dataset_final["mean_acc_seen"].mean())
            threshold_source = "final_best_performance"
        elif not math.isnan(best_memory_budget_acc):
            threshold = 0.95 * best_memory_budget_acc
        else:
            threshold = math.nan

        if not math.isnan(threshold):
            eligible = group[group["mean_acc_seen"] >= threshold].sort_values("k")
            if not eligible.empty:
                k_95 = int(eligible.iloc[0]["k"])

        if not math.isnan(k_95):
            recommended = k_95
        elif not math.isnan(best_gain_per_mb_k):
            recommended = best_gain_per_mb_k
        else:
            recommended = best_accuracy_k

        rows.append(
            {
                "dataset": dataset,
                "best_accuracy_k": best_accuracy_k,
                "lowest_forgetting_k": lowest_forgetting_k,
                "best_gain_per_MB_k": best_gain_per_mb_k,
                "k_95_percent_of_best_performance": k_95,
                "recommended_practical_k": recommended,
                "threshold_source": threshold_source,
                "threshold_mean_acc_seen": threshold if not math.isnan(threshold) else math.nan,
            }
        )
    return rows


def run() -> dict[str, int]:
    import pandas as pd

    ensure_output_dirs()
    runs_path = config.RAW_DIR / "all_runs_long.csv"
    if not runs_path.exists():
        raise FileNotFoundError("Run collect_results.py before memory_budget_analysis.py")
    runs = pd.read_csv(runs_path)
    for metric in config.MEMORY_BUDGET_METRICS:
        if metric not in runs.columns:
            runs[metric] = math.nan
    memory_runs = runs[runs["setting_type"] == "memory_budget"].copy()
    if not memory_runs.empty:
        memory_runs["k"] = pd.to_numeric(memory_runs["k"], errors="coerce")

    table = summarize_memory_budget(memory_runs) if not memory_runs.empty else pd.DataFrame()
    if not table.empty:
        table = add_k0_comparisons(table)
    elbows = elbow_rows(table, runs) if not table.empty else []

    table.to_csv(config.TABLES_DIR / "table_memory_budget.csv", index=False)
    pd.DataFrame(elbows).to_csv(config.TABLES_DIR / "table_memory_budget_elbow.csv", index=False)
    table.to_csv(config.FIGURE_DATA_DIR / "memory_budget_curve.csv", index=False)
    return {"budget_rows": len(table), "elbow_rows": len(elbows)}


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print("Wrote memory-budget analysis with {budget_rows} budget rows and {elbow_rows} elbow rows.".format(**summary))


if __name__ == "__main__":
    main()
