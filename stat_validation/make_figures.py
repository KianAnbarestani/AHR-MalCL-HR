#!/usr/bin/env python3
"""Create basic matplotlib figures and figure-ready CSVs."""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stat_validation import config
from stat_validation.common import ensure_output_dirs


VERSION = "v1 - make statistical validation figures"


def read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def save_placeholder(path: Path, message: str, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def finalize(fig: Any, path: Path, plt: Any) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def accuracy_over_tasks_data(runs: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    if runs.empty or "accs_seen_per_task" not in runs.columns:
        return pd.DataFrame(rows)
    final = runs[runs["setting_type"].isin(["final_best_performance", "final_memory_efficient"])].copy()
    for _, row in final.iterrows():
        text = row.get("accs_seen_per_task")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "setting_type": row.get("setting_type"),
                    "method": row.get("method"),
                    "k": row.get("k"),
                    "seed": row.get("seed"),
                    "task": index,
                    "accuracy": value,
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    return (
        data.groupby(["dataset", "setting_type", "method", "k", "task"], dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_accuracy", "std": "std_accuracy", "count": "n"})
    )


def old_new_tradeoff_data(desc: Any) -> Any:
    import pandas as pd

    if desc.empty:
        return pd.DataFrame()
    subset = desc[desc["metric"].isin(["old_class_accuracy", "new_class_accuracy"])].copy()
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot_table(
        index=["dataset", "method", "setting_type", "k"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    return pivot


def ablation_barplot_data(desc: Any) -> Any:
    import pandas as pd

    if desc.empty:
        return pd.DataFrame()
    return desc[desc["metric"].isin(["mean_acc_seen", "forgetting"])].copy()


def runtime_memory_data(desc: Any) -> Any:
    import pandas as pd

    if desc.empty:
        return pd.DataFrame()
    subset = desc[desc["metric"].isin(["memory_MB", "elapsed_minutes", "gpu_peak_memory_MB"])].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset.pivot_table(
        index=["dataset", "method", "setting_type", "k"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()


def plot_accuracy_over_tasks(data: Any, path: Path, plt: Any) -> None:
    if data.empty:
        save_placeholder(path, "No accuracy-over-task data available", plt)
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for keys, group in data.groupby(["dataset", "setting_type", "k"], dropna=False):
        dataset, setting_type, k = keys
        group = group.sort_values("task")
        ax.plot(group["task"], group["mean_accuracy"] * 100.0, marker="o", label=f"{dataset} {setting_type} K={k}")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean seen accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize(fig, path, plt)


def plot_memory_budget(memory: Any, metric: str, path: Path, ylabel: str, plt: Any) -> None:
    if memory.empty or metric not in memory.columns:
        save_placeholder(path, f"No {metric} memory-budget data available", plt)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset, group in memory.groupby("dataset", dropna=False):
        group = group.sort_values("k")
        y = group[metric] * 100.0 if metric in {"mean_acc_seen", "forgetting"} else group[metric]
        ax.plot(group["k"], y, marker="o", label=str(dataset))
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize(fig, path, plt)


def plot_replay_drift(replay: Any, metric: str, path: Path, ylabel: str, plt: Any) -> None:
    if replay.empty or metric not in replay.columns:
        save_placeholder(path, f"No {metric} replay-drift data available", plt)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset, group in replay.groupby("dataset", dropna=False):
        group = group.sort_values("k")
        ax.plot(group["k"], group[metric], marker="o", label=str(dataset))
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize(fig, path, plt)


def plot_ablation_bar(data: Any, metric: str, path: Path, ylabel: str, plt: Any) -> None:
    if data.empty:
        save_placeholder(path, f"No ablation {metric} data available", plt)
        return
    subset = data[data["metric"] == metric].copy()
    if subset.empty:
        save_placeholder(path, f"No ablation {metric} data available", plt)
        return
    subset["label"] = subset["dataset"].astype(str) + " | " + subset["method"].astype(str)
    subset = subset.sort_values(["dataset", "mean"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = subset["mean"] * 100.0 if metric in {"mean_acc_seen", "forgetting"} else subset["mean"]
    ax.bar(subset["label"], y)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    finalize(fig, path, plt)


def plot_old_new(data: Any, path: Path, plt: Any) -> None:
    if data.empty or not {"old_class_accuracy", "new_class_accuracy"}.issubset(set(data.columns)):
        save_placeholder(path, "No old/new tradeoff data available", plt)
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for dataset, group in data.groupby("dataset", dropna=False):
        ax.scatter(group["old_class_accuracy"] * 100.0, group["new_class_accuracy"] * 100.0, label=str(dataset), alpha=0.8)
    ax.set_xlabel("Old-class accuracy (%)")
    ax.set_ylabel("New-class accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize(fig, path, plt)


def plot_runtime_memory(data: Any, path: Path, plt: Any) -> None:
    if data.empty or not {"memory_MB", "elapsed_minutes"}.issubset(set(data.columns)):
        save_placeholder(path, "No runtime/memory data available", plt)
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for setting, group in data.groupby("setting_type", dropna=False):
        ax.scatter(group["memory_MB"], group["elapsed_minutes"], label=str(setting), alpha=0.8)
    ax.set_xlabel("Memory MB")
    ax.set_ylabel("Elapsed minutes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize(fig, path, plt)


def run() -> dict[str, int]:
    import pandas as pd

    mpl_cache = Path("/tmp/ahr_malcl_matplotlib_cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dirs()
    runs = read_csv(config.RAW_DIR / "all_runs_long.csv")
    desc_all = read_csv(config.TABLES_DIR / "descriptive_stats_all.csv")
    desc_ablation = read_csv(config.TABLES_DIR / "descriptive_stats_ablation.csv")
    memory = read_csv(config.FIGURE_DATA_DIR / "memory_budget_curve.csv")
    replay = read_csv(config.FIGURE_DATA_DIR / "replay_drift_curve.csv")

    accuracy_tasks = accuracy_over_tasks_data(runs)
    old_new = old_new_tradeoff_data(desc_all)
    ablation_data = ablation_barplot_data(desc_ablation)
    runtime_data = runtime_memory_data(desc_all)

    accuracy_tasks.to_csv(config.FIGURE_DATA_DIR / "accuracy_over_tasks.csv", index=False)
    old_new.to_csv(config.FIGURE_DATA_DIR / "old_new_tradeoff.csv", index=False)
    ablation_data.to_csv(config.FIGURE_DATA_DIR / "ablation_barplot_data.csv", index=False)
    runtime_data.to_csv(config.FIGURE_DATA_DIR / "runtime_memory_data.csv", index=False)

    plot_accuracy_over_tasks(accuracy_tasks, config.FIGURES_DIR / "fig_accuracy_over_tasks.png", plt)
    plot_memory_budget(memory, "mean_acc_seen", config.FIGURES_DIR / "fig_memory_budget_mean_accuracy.png", "Mean seen accuracy (%)", plt)
    plot_memory_budget(memory, "forgetting", config.FIGURES_DIR / "fig_memory_budget_forgetting.png", "Forgetting (%)", plt)
    plot_replay_drift(replay, "mmd_real_generated", config.FIGURES_DIR / "fig_replay_drift_mmd.png", "MMD", plt)
    plot_replay_drift(
        replay,
        "wasserstein_real_generated",
        config.FIGURES_DIR / "fig_replay_drift_wasserstein.png",
        "Wasserstein",
        plt,
    )
    plot_ablation_bar(ablation_data, "mean_acc_seen", config.FIGURES_DIR / "fig_ablation_mean_accuracy.png", "Mean seen accuracy (%)", plt)
    plot_ablation_bar(ablation_data, "forgetting", config.FIGURES_DIR / "fig_ablation_forgetting.png", "Forgetting (%)", plt)
    plot_old_new(old_new, config.FIGURES_DIR / "fig_old_new_tradeoff.png", plt)
    plot_runtime_memory(runtime_data, config.FIGURES_DIR / "fig_runtime_memory.png", plt)

    return {
        "accuracy_task_rows": len(accuracy_tasks),
        "old_new_rows": len(old_new),
        "ablation_rows": len(ablation_data),
        "runtime_rows": len(runtime_data),
        "figures": 9,
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote {figures} figures and figure-data rows: accuracy={accuracy_task_rows}, "
        "old_new={old_new_rows}, ablation={ablation_rows}, runtime={runtime_rows}.".format(**summary)
    )


if __name__ == "__main__":
    main()
