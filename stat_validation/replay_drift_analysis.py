#!/usr/bin/env python3
"""Analyze replay drift metrics and their relationship to final performance."""

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


VERSION = "v1 - replay drift analysis"


DRIFT_METRICS = [
    "mean_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "mmd_real_generated",
    "wasserstein_real_generated",
    "memory_MB",
]


def build_drift_table(runs: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for keys, group in runs.groupby(["dataset", "k"], dropna=False):
        dataset, k = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "k": int(k) if pd.notna(k) else math.nan,
            "n": int(group.shape[0]),
            "methods": ",".join(sorted(str(x) for x in group["method"].dropna().unique())),
            "seeds": ",".join(
                map(str, sorted(int(x) for x in pd.to_numeric(group["seed"], errors="coerce").dropna().unique()))
            ),
        }
        for metric in DRIFT_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce") if metric in group.columns else pd.Series(dtype=float)
            row[metric] = float(values.mean()) if values.notna().any() else math.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["dataset", "k"], na_position="last")
    table["mmd_reduction_vs_k0_percent"] = math.nan
    table["wasserstein_reduction_vs_k0_percent"] = math.nan
    for dataset, group in table.groupby("dataset", dropna=False):
        base_rows = group[pd.to_numeric(group["k"], errors="coerce") == 0]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        idxs = group.index
        if pd.notna(base["mmd_real_generated"]) and base["mmd_real_generated"] != 0:
            table.loc[idxs, "mmd_reduction_vs_k0_percent"] = (
                (base["mmd_real_generated"] - table.loc[idxs, "mmd_real_generated"]) / abs(base["mmd_real_generated"]) * 100.0
            )
        if pd.notna(base["wasserstein_real_generated"]) and base["wasserstein_real_generated"] != 0:
            table.loc[idxs, "wasserstein_reduction_vs_k0_percent"] = (
                (base["wasserstein_real_generated"] - table.loc[idxs, "wasserstein_real_generated"])
                / abs(base["wasserstein_real_generated"])
                * 100.0
            )
    return table


def correlation_rows(table: Any) -> list[dict[str, Any]]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    pairs = [
        ("mmd_real_generated", "final_taskwise_average_accuracy"),
        ("mmd_real_generated", "forgetting"),
        ("wasserstein_real_generated", "final_taskwise_average_accuracy"),
        ("wasserstein_real_generated", "forgetting"),
    ]
    if table.empty:
        return rows
    for dataset, group in table.groupby("dataset", dropna=False):
        for x_metric, y_metric in pairs:
            x = pd.to_numeric(group[x_metric], errors="coerce")
            y = pd.to_numeric(group[y_metric], errors="coerce")
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            note = ""
            pearson = math.nan
            spearman = math.nan
            if len(valid) >= 3:
                pearson = float(valid["x"].corr(valid["y"], method="pearson"))
                spearman = float(valid["x"].corr(valid["y"], method="spearman"))
            else:
                note = f"not enough points for correlation: n={len(valid)}"
            rows.append(
                {
                    "dataset": dataset,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": len(valid),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "note": note,
                }
            )
    return rows


def run() -> dict[str, int]:
    import pandas as pd

    ensure_output_dirs()
    runs_path = config.RAW_DIR / "all_runs_long.csv"
    if not runs_path.exists():
        raise FileNotFoundError("Run collect_results.py before replay_drift_analysis.py")
    runs = pd.read_csv(runs_path)
    for metric in DRIFT_METRICS:
        if metric not in runs.columns:
            runs[metric] = math.nan
    drift_runs = runs[runs["setting_type"] == "replay_drift"].copy()
    if not drift_runs.empty:
        drift_runs["k"] = pd.to_numeric(drift_runs["k"], errors="coerce")

    table = build_drift_table(drift_runs) if not drift_runs.empty else pd.DataFrame()
    correlations = correlation_rows(table)
    table.to_csv(config.TABLES_DIR / "table_replay_drift.csv", index=False)
    pd.DataFrame(correlations).to_csv(config.STATS_DIR / "replay_drift_correlations.csv", index=False)
    table.to_csv(config.FIGURE_DATA_DIR / "replay_drift_curve.csv", index=False)
    return {"drift_rows": len(table), "correlation_rows": len(correlations)}


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote replay-drift analysis with {drift_rows} drift rows and "
        "{correlation_rows} correlation rows.".format(**summary)
    )


if __name__ == "__main__":
    main()
