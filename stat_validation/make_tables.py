#!/usr/bin/env python3
"""Create paper-ready CSV and Markdown tables from validation outputs."""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs, safe_float


VERSION = "v1 - make paper-ready tables"

PERCENT_METRICS = {
    "mean_acc_seen",
    "min_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "plasticity_stability_gap",
    "accuracy_gain_vs_k0",
    "final_accuracy_gain_vs_k0",
    "forgetting_reduction_vs_k0",
}

METRIC_LABELS = {
    "mean_acc_seen": "Mean Seen Accuracy",
    "min_acc_seen": "Min Seen Accuracy",
    "final_taskwise_average_accuracy": "Final Taskwise Accuracy",
    "forgetting": "Forgetting",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
    "old_class_accuracy": "Old-Class Accuracy",
    "new_class_accuracy": "New-Class Accuracy",
    "plasticity_stability_gap": "Plasticity-Stability Gap",
    "mmd_real_generated": "MMD",
    "wasserstein_real_generated": "Wasserstein",
    "memory_MB": "Memory MB",
    "gpu_peak_memory_MB": "GPU Peak MB",
    "elapsed_minutes": "Elapsed Minutes",
}


def read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def method_display(method: Any) -> str:
    if str(method) == "hybrid_random_buffer":
        return config.MAIN_METHOD
    return str(method)


def setting_display(setting: Any) -> str:
    return str(setting).replace("_", " ")


def format_stat(mean: Any, std: Any, metric: str, n: Any = "") -> str:
    mean_value = safe_float(mean)
    std_value = safe_float(std)
    n_value = safe_float(n)
    if mean_value is None:
        return ""
    if metric in PERCENT_METRICS:
        mean_value *= 100.0
        if std_value is not None:
            std_value *= 100.0
        digits = 2
    elif metric in {"memory_MB", "gpu_peak_memory_MB", "elapsed_minutes"}:
        digits = 1
    else:
        digits = 4
    if std_value is not None and n_value is not None and n_value > 1:
        return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"
    if n_value == 1:
        return f"{mean_value:.{digits}f} (n=1)"
    return f"{mean_value:.{digits}f}"


def format_plain(value: Any, metric: str = "") -> str:
    number = safe_float(value)
    if number is None:
        return ""
    if metric in PERCENT_METRICS:
        return f"{number * 100.0:.2f}"
    if "percent" in metric:
        return f"{number:.2f}"
    if "MB" in metric or "memory" in metric.lower():
        return f"{number:.1f}"
    return f"{number:.4f}"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def write_table(df: Any, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    columns = list(df.columns)
    lines = ["| " + " | ".join(markdown_escape(col) for col in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(markdown_escape(row.get(col, "")) for col in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def descriptive_wide(desc: Any, metrics: list[str]) -> Any:
    import pandas as pd

    if desc.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "method", "setting_type", "k"]
    for keys, group in desc.groupby(group_cols, dropna=False):
        dataset, method, setting_type, k = keys
        out: dict[str, Any] = {
            "Method": method_display(method),
            "Dataset": dataset,
            "Setting": setting_display(setting_type),
            "K": k,
        }
        for metric in metrics:
            metric_row = group[group["metric"] == metric]
            label = METRIC_LABELS.get(metric, metric)
            if metric_row.empty:
                out[label] = ""
            else:
                first = metric_row.iloc[0]
                out[label] = format_stat(first.get("mean"), first.get("std"), metric, first.get("n"))
        rows.append(out)
    return pd.DataFrame(rows)


def append_note(table: Any, note: str) -> Any:
    if not table.empty:
        table = table.copy()
        table["Paper Use Note"] = note
    return table


def full_metric_table(desc: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for _, row in desc.iterrows():
        metric = row.get("metric", "")
        rows.append(
            {
                "Method": method_display(row.get("method")),
                "Dataset": row.get("dataset"),
                "Setting": setting_display(row.get("setting_type")),
                "K": row.get("k"),
                "Metric": METRIC_LABELS.get(metric, metric),
                "Mean ± SD": format_stat(row.get("mean"), row.get("std"), metric, row.get("n")),
                "95% CI Low": format_plain(row.get("ci95_low"), metric),
                "95% CI High": format_plain(row.get("ci95_high"), metric),
                "Median": format_plain(row.get("median"), metric),
                "Min": format_plain(row.get("min"), metric),
                "Max": format_plain(row.get("max"), metric),
                "n": row.get("n"),
                "Seeds": row.get("seeds", ""),
            }
        )
    return pd.DataFrame(rows)


def component_effect_table(component: Any) -> Any:
    import pandas as pd

    if component.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in component.iterrows():
        metric = row.get("metric", "")
        rows.append(
            {
                "Dataset": row.get("dataset"),
                "K": row.get("k"),
                "Effect": row.get("comparison_label"),
                "Metric": METRIC_LABELS.get(metric, metric),
                "Method A": row.get("method_a"),
                "Method B": row.get("method_b"),
                "Mean A": format_plain(row.get("mean_a"), metric),
                "Mean B": format_plain(row.get("mean_b"), metric),
                "A - B": format_plain(row.get("mean_difference_a_minus_b"), metric),
                "Improvement %": format_plain(row.get("percent_improvement"), "percent"),
                "Cohen dz": format_plain(row.get("cohens_dz")),
                "Holm p": format_plain(row.get("paired_t_holm_p_value")),
                "Significant 0.05": row.get("paired_t_holm_reject_0_05", ""),
            }
        )
    return pd.DataFrame(rows)


def memory_table(memory: Any) -> Any:
    import pandas as pd

    if memory.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in memory.iterrows():
        rows.append(
            {
                "Dataset": row.get("dataset"),
                "K": row.get("k"),
                "n": row.get("n"),
                "Mean Seen Accuracy": format_plain(row.get("mean_acc_seen"), "mean_acc_seen"),
                "Final Taskwise Accuracy": format_plain(
                    row.get("final_taskwise_average_accuracy"), "final_taskwise_average_accuracy"
                ),
                "Forgetting": format_plain(row.get("forgetting"), "forgetting"),
                "Macro-F1": format_plain(row.get("macro_f1"), "macro_f1"),
                "Memory MB": format_plain(row.get("memory_MB"), "memory_MB"),
                "Accuracy Gain vs K=0": format_plain(row.get("accuracy_gain_vs_k0"), "accuracy_gain_vs_k0"),
                "Forgetting Reduction vs K=0": format_plain(
                    row.get("forgetting_reduction_vs_k0"), "forgetting_reduction_vs_k0"
                ),
                "Accuracy Gain per MB": format_plain(row.get("accuracy_gain_per_MB")),
                "MMD Reduction %": format_plain(row.get("mmd_reduction_percent"), "percent"),
                "Wasserstein Reduction %": format_plain(row.get("wasserstein_reduction_percent"), "percent"),
            }
        )
    return pd.DataFrame(rows)


def replay_table(replay: Any) -> Any:
    import pandas as pd

    if replay.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in replay.iterrows():
        rows.append(
            {
                "Dataset": row.get("dataset"),
                "K": row.get("k"),
                "n": row.get("n"),
                "Final Taskwise Accuracy": format_plain(
                    row.get("final_taskwise_average_accuracy"), "final_taskwise_average_accuracy"
                ),
                "Forgetting": format_plain(row.get("forgetting"), "forgetting"),
                "MMD": format_plain(row.get("mmd_real_generated")),
                "MMD Reduction vs K=0 %": format_plain(row.get("mmd_reduction_vs_k0_percent"), "percent"),
                "Wasserstein": format_plain(row.get("wasserstein_real_generated")),
                "Wasserstein Reduction vs K=0 %": format_plain(
                    row.get("wasserstein_reduction_vs_k0_percent"), "percent"
                ),
                "Methods": row.get("methods", ""),
            }
        )
    return pd.DataFrame(rows)


def runtime_memory_table(desc: Any) -> Any:
    metrics = ["memory_MB", "gpu_peak_memory_MB", "elapsed_minutes"]
    return descriptive_wide(desc[desc["metric"].isin(metrics)] if not desc.empty else desc, metrics)


def validity_audit_table(runs: Any) -> Any:
    import pandas as pd

    if runs.empty:
        return pd.DataFrame()
    columns = [
        ("source_files", "File Path"),
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("setting_type", "Setting Type"),
        ("k", "K"),
        ("seed", "Seed"),
        ("use_diversity_buffer", "use_diversity_buffer"),
        ("use_kd", "use_kd"),
        ("use_proto_align", "use_proto_align"),
        ("gan_train_on_real_buffer", "gan_train_on_real_buffer"),
        ("result_validity", "result_validity"),
        ("paper_use_category", "paper_use_category"),
        ("validity_reason", "validity_reason"),
    ]
    out = pd.DataFrame()
    for source, label in columns:
        out[label] = runs[source] if source in runs.columns else ""
    return out.sort_values(["Dataset", "Setting Type", "Method", "K", "Seed"], na_position="last")


def run() -> dict[str, int]:
    ensure_output_dirs()
    desc_final = read_csv(config.TABLES_DIR / "descriptive_stats_final.csv")
    desc_ablation = read_csv(config.TABLES_DIR / "descriptive_stats_ablation.csv")
    desc_all = read_csv(config.TABLES_DIR / "descriptive_stats_all.csv")
    component = read_csv(config.TABLES_DIR / "table_component_effects.csv")
    memory = read_csv(config.TABLES_DIR / "table_memory_budget.csv")
    replay = read_csv(config.TABLES_DIR / "table_replay_drift.csv")
    runs = read_csv(config.RAW_DIR / "all_runs_long.csv")

    table1_metrics = [
        "mean_acc_seen",
        "final_taskwise_average_accuracy",
        "forgetting",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "old_class_accuracy",
        "new_class_accuracy",
        "mmd_real_generated",
        "wasserstein_real_generated",
        "memory_MB",
    ]
    if not desc_final.empty and "result_validity" in desc_final.columns:
        table1_source = desc_final[
            (desc_final["result_validity"] == "valid_main_hr")
            & (desc_final["paper_use_category"] == "main_final_table")
        ]
    else:
        table1_source = desc_final[desc_final["method"] == config.MAIN_METHOD_CANONICAL] if not desc_final.empty else desc_final
    table1 = descriptive_wide(table1_source, table1_metrics)
    table2 = full_metric_table(desc_final)
    table3 = descriptive_wide(
        desc_ablation,
        [
            "mean_acc_seen",
            "final_taskwise_average_accuracy",
            "forgetting",
            "macro_f1",
            "balanced_accuracy",
            "old_class_accuracy",
            "new_class_accuracy",
            "mmd_real_generated",
            "wasserstein_real_generated",
        ],
    )
    table4 = component_effect_table(component)
    table5 = memory_table(memory)
    table6 = replay_table(replay)
    table7 = runtime_memory_table(desc_all)
    table8 = validity_audit_table(runs)
    legacy_source = (
        desc_final[desc_final["result_validity"] == "legacy_or_config_mismatch"]
        if not desc_final.empty and "result_validity" in desc_final.columns
        else desc_final.iloc[0:0]
    )
    table9 = append_note(
        descriptive_wide(legacy_source, table1_metrics),
        "Not used as final AHR-MalCL-HR main results because config flags differ.",
    )
    memory_efficient_hr_source = (
        desc_all[desc_all["result_validity"] == "valid_memory_efficient_hr"]
        if not desc_all.empty and "result_validity" in desc_all.columns
        else desc_all.iloc[0:0]
    )
    table10 = descriptive_wide(memory_efficient_hr_source, table1_metrics)

    tables = [
        ("table_1_final_main_results", table1),
        ("table_2_full_metrics", table2),
        ("table_3_ablation", table3),
        ("table_4_component_effects", table4),
        ("table_5_memory_budget", table5),
        ("table_6_replay_drift", table6),
        ("table_7_runtime_memory", table7),
        ("table_validity_audit", table8),
        ("table_legacy_memory_efficient_results", table9),
        ("table_memory_efficient_hr_results", table10),
    ]
    for stem, table in tables:
        write_table(table, config.TABLES_DIR / f"{stem}.csv", config.TABLES_DIR / f"{stem}.md")
    return {stem: len(table) for stem, table in tables}


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print("Wrote paper-ready tables: " + ", ".join(f"{name}={rows}" for name, rows in summary.items()))


if __name__ == "__main__":
    main()
