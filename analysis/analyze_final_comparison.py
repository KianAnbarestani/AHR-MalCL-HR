#!/usr/bin/env python3
"""Recompute the released final comparison from canonical seed-level CSV input.

This script intentionally reads only ``results/final_all_methods_seed_level.csv``
or an equivalent input directory. It neither accesses the private project tree nor
runs training.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEEDS = list(range(42, 52))
DATASETS = ["EMBER", "AZ-Class"]
METHODS = [
    "AHR-MalCL-HR (hybrid_random_buffer)",
    "ER (standard)",
    "DER++ (standard)",
    "protocol-matched adapted MalCL baseline",
]
AHR = METHODS[0]
COMPARATORS = METHODS[1:]
METRICS = [
    "final_taskwise_accuracy",
    "mean_seen_accuracy",
    "forgetting",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
]


def input_csv(input_root: Path) -> Path:
    return input_root if input_root.is_file() else input_root / "final_all_methods_seed_level.csv"


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"method", "dataset", "seed", *METRICS}
    missing = expected - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = dict(row)
        item["seed"] = int(str(row["seed"]))
        for metric in METRICS:
            value = float(str(row[metric]))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} in {row['method']} {row['dataset']} seed {row['seed']}")
            item[metric] = value
        parsed.append(item)
    expected_keys = {(method, dataset, seed) for method in METHODS for dataset in DATASETS for seed in SEEDS}
    found_keys = {(str(row["method"]), str(row["dataset"]), int(row["seed"])) for row in parsed}
    if found_keys != expected_keys or len(parsed) != len(expected_keys):
        missing = sorted(expected_keys - found_keys)
        extra = sorted(found_keys - expected_keys)
        raise ValueError(f"seed-level protocol mismatch: missing={missing[:4]} extra={extra[:4]} rows={len(parsed)}")
    return parsed


def mean_sd_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    n = len(values)
    average = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return average, sd, average - half_width, average + half_width


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 20000) -> tuple[float, float]:
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pairs: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary: list[dict[str, object]], figures: Path) -> Path:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    colors = ["#007c91", "#d05a32", "#6c7a36", "#7a5195"]
    for axis, dataset in zip(axes, DATASETS):
        records = [row for row in summary if row["dataset"] == dataset and row["metric"] == "final_taskwise_accuracy"]
        records.sort(key=lambda row: METHODS.index(str(row["method"])))
        means = [float(row["mean"]) for row in records]
        errors = [float(row["ci95_high"]) - float(row["mean"]) for row in records]
        axis.bar(range(len(records)), means, yerr=errors, capsize=4, color=colors, edgecolor="#202020", linewidth=0.5)
        axis.set_title(dataset)
        axis.set_xticks(range(len(records)), ["AHR", "ER", "DER++", "MalCL"], rotation=0)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final taskwise accuracy (%)")
    fig.tight_layout()
    output = figures / "analysis_validation_figure.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def recompute(input_root: Path, output_root: Path) -> dict[str, object]:
    rows = load_rows(input_csv(input_root))
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["dataset"]), str(row["method"]))].append(row)
    for values in by_key.values():
        values.sort(key=lambda item: int(item["seed"]))

    summary: list[dict[str, object]] = []
    main_table: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            values = by_key[(dataset, method)]
            combined: dict[str, object] = {"dataset": dataset, "method": method, "n": len(values)}
            for metric in METRICS:
                scores = np.asarray([float(value[metric]) for value in values], dtype=float)
                avg, sd, low, high = mean_sd_ci(scores)
                summary.append({"dataset": dataset, "method": method, "metric": metric, "metric_scale": "percentage_points", "n": len(scores), "mean": avg, "sample_sd": sd, "ci95_low": low, "ci95_high": high})
                for suffix, value in (("mean", avg), ("sample_sd", sd), ("ci95_low", low), ("ci95_high", high)):
                    combined[f"{metric}_{suffix}"] = value
            main_table.append(combined)

    paired: list[dict[str, object]] = []
    win_tie_loss: list[dict[str, object]] = []
    rng = np.random.default_rng(20260715)
    for dataset in DATASETS:
        for metric in METRICS:
            family: list[tuple[int, float]] = []
            for comparator in COMPARATORS:
                a = np.asarray([float(row[metric]) for row in by_key[(dataset, AHR)]], dtype=float)
                b = np.asarray([float(row[metric]) for row in by_key[(dataset, comparator)]], dtype=float)
                differences = a - b if metric != "forgetting" else b - a
                try:
                    wilcoxon = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
                    w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                except ValueError:
                    w_stat, w_p = 0.0, 1.0
                t_test = stats.ttest_1samp(differences, 0.0)
                mean_diff = float(differences.mean())
                diff_sd = float(differences.std(ddof=1))
                bootstrap_low, bootstrap_high = bootstrap_ci(differences, rng)
                row = {
                    "dataset": dataset,
                    "metric": metric,
                    "primary_comparison": comparator == "protocol-matched adapted MalCL baseline",
                    "method_a": AHR,
                    "method_b": comparator,
                    "n_pairs": len(differences),
                    "difference_definition": "AHR minus comparator; positive favors AHR" if metric != "forgetting" else "comparator minus AHR forgetting; positive favors AHR",
                    "mean_paired_difference": mean_diff,
                    "paired_difference_sample_sd": diff_sd,
                    "bootstrap_ci95_low": bootstrap_low,
                    "bootstrap_ci95_high": bootstrap_high,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p": w_p,
                    "paired_t_statistic": float(t_test.statistic),
                    "paired_t_p": float(t_test.pvalue),
                    "paired_effect_size_dz": mean_diff / diff_sd if diff_sd else 0.0,
                    "wins_ahr": int(np.sum(differences > 0)),
                    "ties": int(np.sum(differences == 0)),
                    "losses_ahr": int(np.sum(differences < 0)),
                    "holm_family": f"{dataset}::{metric}",
                }
                for seed, difference in zip(SEEDS, differences):
                    row[f"seed_{seed}_difference"] = float(difference)
                paired.append(row)
                family.append((len(paired) - 1, w_p))
                win_tie_loss.append({key: row[key] for key in ("dataset", "metric", "method_a", "method_b", "difference_definition", "wins_ahr", "ties", "losses_ahr")})
            for index, adjusted in holm_adjust(family).items():
                paired[index]["wilcoxon_holm_p"] = adjusted
                paired[index]["wilcoxon_holm_significant_0_05"] = adjusted < 0.05

    summary_fields = ["dataset", "method", "metric", "metric_scale", "n", "mean", "sample_sd", "ci95_low", "ci95_high"]
    paired_fields = list(paired[0])
    main_fields = list(main_table[0])
    write_csv(tables / "final_all_methods_seed_level.csv", rows, list(rows[0]))
    write_csv(tables / "final_all_methods_dataset_summary.csv", summary, summary_fields)
    write_csv(tables / "final_paired_statistical_tests.csv", paired, paired_fields)
    write_csv(tables / "final_win_tie_loss.csv", win_tie_loss, list(win_tie_loss[0]))
    write_csv(tables / "final_main_paper_table.csv", main_table, main_fields)
    figure = make_figure(summary, figures)
    return {"seed_rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired), "figure": str(figure)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "generated")
    args = parser.parse_args()
    result = recompute(args.input_root, args.output_root)
    print(f"recomputed {result['seed_rows']} seed rows, {result['summary_rows']} summary rows, and {result['paired_rows']} paired tests")
    print(f"figure: {result['figure']}")


if __name__ == "__main__":
    main()
