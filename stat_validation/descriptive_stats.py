#!/usr/bin/env python3
"""Compute descriptive statistics for collected AHR-MalCL runs."""

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


VERSION = "v1 - descriptive statistics for statistical validation"


def t_critical_95(n: int) -> float:
    if n <= 1:
        return math.nan
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df=n - 1))
    except Exception:  # noqa: BLE001 - normal fallback keeps the table usable without scipy.
        return 1.96


def summarize_group(group: Any, metric: str) -> dict[str, Any]:
    import pandas as pd

    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    n = int(values.shape[0])
    seeds = sorted(int(x) for x in pd.to_numeric(group.loc[values.index, "seed"], errors="coerce").dropna().unique())
    if n == 0:
        return {
            "metric": metric,
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "seeds": "",
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else math.nan
    se = float(std / math.sqrt(n)) if n > 1 else math.nan
    crit = t_critical_95(n)
    ci_low = mean - crit * se if n > 1 else math.nan
    ci_high = mean + crit * se if n > 1 else math.nan
    return {
        "metric": metric,
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "seeds": ",".join(map(str, seeds)),
    }


def descriptive_rows(runs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in [
            "dataset",
            "method",
            "setting_type",
            "k",
            "result_validity",
            "paper_use_category",
            "validity_reason",
        ]
        if column in runs.columns
    ]
    for keys, group in runs.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys, strict=True))
        for metric in config.DESCRIPTIVE_METRICS:
            summary = summarize_group(group, metric)
            rows.append({**base, **summary})
    return rows


def run() -> dict[str, int]:
    import pandas as pd

    ensure_output_dirs()
    runs_path = config.RAW_DIR / "all_runs_long.csv"
    if not runs_path.exists():
        raise FileNotFoundError("Run collect_results.py before descriptive_stats.py")
    runs = pd.read_csv(runs_path)
    for metric in config.DESCRIPTIVE_METRICS:
        if metric not in runs.columns:
            runs[metric] = math.nan

    rows = descriptive_rows(runs)
    all_df = pd.DataFrame(rows)
    all_path = config.TABLES_DIR / "descriptive_stats_all.csv"
    all_df.to_csv(all_path, index=False)

    final_df = all_df[all_df["setting_type"].isin(["final_best_performance", "final_memory_efficient"])]
    ablation_df = all_df[all_df["setting_type"] == "ablation"]
    memory_df = all_df[all_df["setting_type"] == "memory_budget"]

    final_df.to_csv(config.TABLES_DIR / "descriptive_stats_final.csv", index=False)
    ablation_df.to_csv(config.TABLES_DIR / "descriptive_stats_ablation.csv", index=False)
    memory_df.to_csv(config.TABLES_DIR / "descriptive_stats_memory_budget.csv", index=False)

    return {
        "all_rows": len(all_df),
        "final_rows": len(final_df),
        "ablation_rows": len(ablation_df),
        "memory_rows": len(memory_df),
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote descriptive stats: {all_rows} all, {final_rows} final, "
        "{ablation_rows} ablation, {memory_rows} memory-budget rows.".format(**summary)
    )


if __name__ == "__main__":
    main()
