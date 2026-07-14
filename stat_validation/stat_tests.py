#!/usr/bin/env python3
"""Run paired statistical tests for controlled ablation comparisons."""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs, safe_float, write_csv


VERSION = "v1 - paired statistical tests for statistical validation"


MAIN_ABLATION_METHODS = [
    "classifier_real_only",
    "malcl_like",
    "wgan_projection_generated_only",
    "real_buffer_only",
    "hybrid_diversity_buffer",
    "plus_kd",
    "full_ahr_malcl",
]

COMPONENT_EFFECTS = [
    ("Generated replay effect", "wgan_projection_generated_only", "classifier_real_only"),
    ("Real anchoring effect", "hybrid_random_buffer", "wgan_projection_generated_only"),
    ("Hybrid vs real-only", "hybrid_random_buffer", "real_buffer_only"),
    ("Diversity effect", "hybrid_diversity_buffer", "hybrid_random_buffer"),
    ("KD effect", "plus_kd", "hybrid_random_buffer"),
    ("Full complexity effect", "full_ahr_malcl", "hybrid_random_buffer"),
]


def try_scipy() -> Any | None:
    try:
        from scipy import stats

        return stats
    except Exception:  # noqa: BLE001 - downstream records explain missing p-values.
        return None


def finite(value: Any) -> bool:
    number = safe_float(value)
    return number is not None and math.isfinite(number)


def t_ci(differences: list[float], stats: Any | None) -> tuple[float, float]:
    n = len(differences)
    if n <= 1:
        return math.nan, math.nan
    mean_diff = sum(differences) / n
    std_diff = sample_std(differences)
    if std_diff is None:
        return mean_diff, mean_diff
    se = std_diff / math.sqrt(n)
    if stats is not None:
        crit = float(stats.t.ppf(0.975, df=n - 1))
    else:
        crit = 1.96
    return mean_diff - crit * se, mean_diff + crit * se


def sample_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0:
        return 0.0
    return math.sqrt(variance)


def percent_improvement(metric: str, mean_a: float, mean_b: float) -> float:
    if mean_b == 0:
        return math.nan
    if metric in config.LOWER_IS_BETTER:
        return (mean_b - mean_a) / abs(mean_b) * 100.0
    return (mean_a - mean_b) / abs(mean_b) * 100.0


def comparison_specs(methods: set[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for other in MAIN_ABLATION_METHODS:
        if other in methods:
            specs.append(
                {
                    "comparison_family": "main_ablation_reference",
                    "comparison_label": f"hybrid_random_buffer vs {other}",
                    "method_a": "hybrid_random_buffer",
                    "method_b": other,
                }
            )
    for label, method_a, method_b in COMPONENT_EFFECTS:
        if method_a in methods and method_b in methods:
            specs.append(
                {
                    "comparison_family": "component_effect",
                    "comparison_label": label,
                    "method_a": method_a,
                    "method_b": method_b,
                }
            )
    return specs


def paired_metric_values(group: Any, method_a: str, method_b: str, metric: str) -> tuple[list[int], list[float], list[float], str | None]:
    import pandas as pd

    a = group[group["method"] == method_a].copy()
    b = group[group["method"] == method_b].copy()
    if a.empty or b.empty:
        return [], [], [], "method missing"
    a["seed"] = pd.to_numeric(a["seed"], errors="coerce")
    b["seed"] = pd.to_numeric(b["seed"], errors="coerce")
    seed_a = sorted(int(x) for x in a["seed"].dropna().unique())
    seed_b = sorted(int(x) for x in b["seed"].dropna().unique())
    if seed_a != seed_b:
        return [], [], [], f"seed sets differ: {method_a}={seed_a}; {method_b}={seed_b}"
    if len(seed_a) < 2:
        return [], [], [], f"too few paired seeds: {seed_a}"

    a_metric = a[["seed", metric]].copy()
    b_metric = b[["seed", metric]].copy()
    a_metric[metric] = pd.to_numeric(a_metric[metric], errors="coerce")
    b_metric[metric] = pd.to_numeric(b_metric[metric], errors="coerce")
    paired = a_metric.merge(b_metric, on="seed", suffixes=("_a", "_b"))
    paired = paired.dropna(subset=[f"{metric}_a", f"{metric}_b"])
    if len(paired) != len(seed_a):
        return [], [], [], f"missing paired metric values for {metric}; valid_pairs={len(paired)} expected={len(seed_a)}"
    seeds = [int(x) for x in paired["seed"].tolist()]
    values_a = [float(x) for x in paired[f"{metric}_a"].tolist()]
    values_b = [float(x) for x in paired[f"{metric}_b"].tolist()]
    return seeds, values_a, values_b, None


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "p_value") -> list[dict[str, Any]]:
    adjusted_rows = [dict(row) for row in rows]
    finite_items = [(idx, safe_float(row.get(p_key))) for idx, row in enumerate(adjusted_rows)]
    finite_items = [(idx, p) for idx, p in finite_items if p is not None]
    finite_items.sort(key=lambda item: item[1])
    m = len(finite_items)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(finite_items):
        adjusted = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_rows[idx]["holm_p_value"] = running_max
        adjusted_rows[idx]["holm_reject_0_05"] = running_max <= 0.05
    for row in adjusted_rows:
        row.setdefault("holm_p_value", "")
        row.setdefault("holm_reject_0_05", "")
    return adjusted_rows


def run() -> dict[str, int]:
    import pandas as pd

    ensure_output_dirs()
    stats = try_scipy()
    runs_path = config.RAW_DIR / "all_runs_long.csv"
    if not runs_path.exists():
        raise FileNotFoundError("Run collect_results.py before stat_tests.py")
    runs = pd.read_csv(runs_path)
    ablation = runs[runs["setting_type"] == "ablation"].copy()

    t_rows: list[dict[str, Any]] = []
    w_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for keys, group in ablation.groupby(["dataset", "k"], dropna=False):
        dataset, k = keys
        methods = set(str(x) for x in group["method"].dropna().unique())
        for spec in comparison_specs(methods):
            for metric in config.PRIMARY_METRICS:
                seeds, values_a, values_b, skip_reason = paired_metric_values(
                    group, spec["method_a"], spec["method_b"], metric
                )
                base = {
                    "dataset": dataset,
                    "setting_type": "ablation",
                    "k": k,
                    "metric": metric,
                    **spec,
                }
                if skip_reason:
                    skipped_rows.append({**base, "reason": skip_reason})
                    continue

                differences = [a - b for a, b in zip(values_a, values_b, strict=True)]
                n = len(differences)
                mean_a = sum(values_a) / n
                mean_b = sum(values_b) / n
                mean_diff = sum(differences) / n
                std_diff = sample_std(differences)
                cohen_dz = mean_diff / std_diff if std_diff and std_diff > 0 else math.nan
                ci_low, ci_high = t_ci(differences, stats)
                improvement = percent_improvement(metric, mean_a, mean_b)
                shapiro_p = math.nan
                shapiro_note = ""
                if stats is None:
                    shapiro_note = "scipy unavailable"
                elif n >= 3:
                    try:
                        shapiro_p = float(stats.shapiro(differences).pvalue)
                    except Exception as exc:  # noqa: BLE001
                        shapiro_note = f"{type(exc).__name__}: {exc}"
                else:
                    shapiro_note = "n<3"

                p_value = math.nan
                statistic = math.nan
                t_note = ""
                if stats is None:
                    t_note = "scipy unavailable"
                elif all(abs(diff) < 1e-15 for diff in differences):
                    statistic = 0.0
                    p_value = 1.0
                    t_note = "all paired differences are zero"
                else:
                    try:
                        result = stats.ttest_rel(values_a, values_b, nan_policy="omit")
                        statistic = float(result.statistic)
                        p_value = float(result.pvalue)
                    except Exception as exc:  # noqa: BLE001
                        t_note = f"{type(exc).__name__}: {exc}"
                t_rows.append(
                    {
                        **base,
                        "n_pairs": n,
                        "seeds": ",".join(map(str, seeds)),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_difference_a_minus_b": mean_diff,
                        "t_statistic": statistic,
                        "p_value": p_value,
                        "shapiro_p_value": shapiro_p,
                        "note": "; ".join(x for x in [t_note, shapiro_note] if x),
                    }
                )

                w_p_value = math.nan
                w_statistic = math.nan
                w_note = ""
                if stats is None:
                    w_note = "scipy unavailable"
                elif all(abs(diff) < 1e-15 for diff in differences):
                    w_note = "Wilcoxon skipped because all paired differences are zero"
                elif n < 2:
                    w_note = "Wilcoxon skipped because n<2"
                else:
                    try:
                        result = stats.wilcoxon(values_a, values_b, zero_method="wilcox")
                        w_statistic = float(result.statistic)
                        w_p_value = float(result.pvalue)
                    except Exception as exc:  # noqa: BLE001
                        w_note = f"{type(exc).__name__}: {exc}"
                w_rows.append(
                    {
                        **base,
                        "n_pairs": n,
                        "seeds": ",".join(map(str, seeds)),
                        "wilcoxon_statistic": w_statistic,
                        "p_value": w_p_value,
                        "note": w_note,
                    }
                )

                effect_rows.append(
                    {
                        **base,
                        "n_pairs": n,
                        "seeds": ",".join(map(str, seeds)),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_difference_a_minus_b": mean_diff,
                        "percent_improvement": improvement,
                        "ci95_low_difference": ci_low,
                        "ci95_high_difference": ci_high,
                        "cohens_dz": cohen_dz,
                    }
                )

    t_rows = holm_adjust(t_rows)
    w_rows = holm_adjust(w_rows)
    corrected_rows: list[dict[str, Any]] = []
    for test_name, rows in [("paired_t", t_rows), ("wilcoxon", w_rows)]:
        for row in rows:
            corrected_rows.append(
                {
                    "test": test_name,
                    "dataset": row.get("dataset"),
                    "setting_type": row.get("setting_type"),
                    "k": row.get("k"),
                    "metric": row.get("metric"),
                    "comparison_family": row.get("comparison_family"),
                    "comparison_label": row.get("comparison_label"),
                    "method_a": row.get("method_a"),
                    "method_b": row.get("method_b"),
                    "raw_p_value": row.get("p_value"),
                    "holm_p_value": row.get("holm_p_value"),
                    "holm_reject_0_05": row.get("holm_reject_0_05"),
                    "note": row.get("note", ""),
                }
            )

    t_lookup = {
        (row["dataset"], row["k"], row["metric"], row["comparison_label"]): row
        for row in t_rows
    }
    component_rows: list[dict[str, Any]] = []
    for row in effect_rows:
        if row["comparison_family"] != "component_effect":
            continue
        t_row = t_lookup.get((row["dataset"], row["k"], row["metric"], row["comparison_label"]), {})
        component_rows.append(
            {
                **row,
                "paired_t_p_value": t_row.get("p_value", ""),
                "paired_t_holm_p_value": t_row.get("holm_p_value", ""),
                "paired_t_holm_reject_0_05": t_row.get("holm_reject_0_05", ""),
            }
        )

    pd.DataFrame(t_rows).to_csv(config.STATS_DIR / "paired_t_tests.csv", index=False)
    pd.DataFrame(w_rows).to_csv(config.STATS_DIR / "wilcoxon_tests.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(config.STATS_DIR / "effect_sizes.csv", index=False)
    pd.DataFrame(corrected_rows).to_csv(config.STATS_DIR / "holm_corrected_tests.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(config.STATS_DIR / "skipped_tests.csv", index=False)
    pd.DataFrame(component_rows).to_csv(config.TABLES_DIR / "table_component_effects.csv", index=False)

    return {
        "paired_t": len(t_rows),
        "wilcoxon": len(w_rows),
        "effects": len(effect_rows),
        "skipped": len(skipped_rows),
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote paired tests: {paired_t} t-tests, {wilcoxon} Wilcoxon tests, "
        "{effects} effect rows, {skipped} skipped tests.".format(**summary)
    )


if __name__ == "__main__":
    main()
