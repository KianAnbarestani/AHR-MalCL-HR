#!/usr/bin/env python3
"""Analyze final AHR-MalCL ablation result JSON/CSV files.

This script is read-only with respect to experiment artifacts. It writes only
analysis outputs under result/final-run/.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


VERSION = "v1 - Analyze final AHR-MalCL ablation results"

EXPECTED_METHODS = [
    "classifier_real_only",
    "malcl_like",
    "wgan_projection_generated_only",
    "real_buffer_only",
    "hybrid_random_buffer",
    "hybrid_diversity_buffer",
    "plus_kd",
    "full_ahr_malcl",
]

SEEDS = [42, 43, 44, 45, 46]

METHOD_DESCRIPTIONS = {
    "classifier_real_only": "Classifier trained only on available real current-task data without generative replay or real-memory hybrid anchoring.",
    "malcl_like": "MalCL-like generative replay baseline.",
    "wgan_projection_generated_only": "WGAN-GP + projection critic generated replay only, without real-memory anchoring.",
    "real_buffer_only": "Real replay buffer only, without generated replay.",
    "hybrid_random_buffer": "Hybrid generated replay plus random real-memory buffer.",
    "hybrid_diversity_buffer": "Hybrid generated replay plus diversity-aware real-memory buffer.",
    "plus_kd": "Hybrid replay with knowledge distillation added.",
    "full_ahr_malcl": "Full method with WGAN-GP, projection critic, FML, diversity-aware real buffer, KD, and prototype alignment.",
}

COMPARISONS = [
    ("full_ahr_malcl", "malcl_like"),
    ("full_ahr_malcl", "wgan_projection_generated_only"),
    ("hybrid_diversity_buffer", "hybrid_random_buffer"),
    ("plus_kd", "hybrid_diversity_buffer"),
    ("full_ahr_malcl", "plus_kd"),
    ("real_buffer_only", "wgan_projection_generated_only"),
    ("hybrid_random_buffer", "real_buffer_only"),
    ("hybrid_diversity_buffer", "real_buffer_only"),
]

SUMMARY_METRICS = [
    "mean_acc_seen",
    "min_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "bwt",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "plasticity_stability_gap",
    "memory_MB",
    "gpu_peak_memory_MB",
    "mmd_real_generated",
    "wasserstein_real_generated",
    "feature_center_l2",
]

FINAL_CLASSIFICATION_METRICS = {
    "mean_acc_seen",
    "min_acc_seen",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
}

METRIC_ALIASES = {
    "mean_acc_seen": ["mean_acc_seen", "final_accuracy", "final_acc", "accuracy"],
    "min_acc_seen": ["min_acc_seen", "min_accuracy", "min_acc"],
    "final_taskwise_average_accuracy": [
        "final_taskwise_average_accuracy",
        "final_average_accuracy_taskwise",
        "final_taskwise_accuracy",
    ],
    "forgetting": ["forgetting", "forgetting_mean"],
    "bwt": ["bwt", "backward_transfer"],
    "precision_macro": ["precision_macro"],
    "precision_weighted": ["precision_weighted"],
    "recall_macro": ["recall_macro"],
    "recall_weighted": ["recall_weighted"],
    "macro_f1": ["macro_f1", "f1_macro"],
    "weighted_f1": ["weighted_f1", "f1_weighted"],
    "balanced_accuracy": ["balanced_accuracy"],
    "old_class_accuracy": ["old_class_accuracy"],
    "new_class_accuracy": ["new_class_accuracy"],
    "plasticity_stability_gap": ["plasticity_stability_gap"],
    "memory_MB": ["memory_MB"],
    "gpu_peak_memory_MB": ["gpu_peak_memory_MB"],
    "mmd_real_generated": ["mmd_real_generated"],
    "wasserstein_real_generated": ["wasserstein_real_generated"],
    "feature_center_l2": ["feature_center_l2"],
}

SPECIAL_AZ_METHODS = {"hybrid_diversity_buffer", "plus_kd"}


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


METHOD_ALIASES = {
    "classifier_real_only": ["classifier_real_only", "classifier real only", "classifier-real-only"],
    "malcl_like": ["malcl_like", "malcl-like", "malcl like"],
    "wgan_projection_generated_only": [
        "wgan_projection_generated_only",
        "wgan-projection-generated-only",
        "wgan projection generated only",
    ],
    "real_buffer_only": ["real_buffer_only", "real-buffer -only", "real-buffer-only", "real buffer only"],
    "hybrid_random_buffer": ["hybrid_random_buffer", "hybrid-random-buffer", "hybrid random buffer"],
    "hybrid_diversity_buffer": [
        "hybrid_diversity_buffer",
        "hybrid-diversity-buffer",
        "hybrid diversity buffer",
    ],
    "plus_kd": ["plus_kd", "plus-kd", "plus kd"],
    "full_ahr_malcl": ["full_ahr_malcl", "full ahr malcl", "final_paper", "memory-efficient setting"],
}

ALIAS_TO_METHOD = {
    norm_name(alias): method
    for method, aliases in METHOD_ALIASES.items()
    for alias in aliases
}


def canonical_method_from_path(path: Path) -> str | None:
    text = " ".join(path.parts[-4:]) + " " + path.name
    ntext = norm_name(text)
    best: tuple[int, str] | None = None
    for alias, method in ALIAS_TO_METHOD.items():
        if alias in ntext:
            score = len(alias)
            if best is None or score > best[0]:
                best = (score, method)
    return best[1] if best else None


def load_json_first_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def first_present_metric(source: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES.get(metric, [metric]):
        value = safe_float(source.get(key))
        if value is not None:
            return value
    return None


def nested_metric(run: dict[str, Any], metric: str) -> float | None:
    value = first_present_metric(run, metric)
    if value is not None:
        return value
    for nested_key in ("cl_metrics", "task_metrics", "metrics", "aggregate"):
        nested = run.get(nested_key)
        if isinstance(nested, dict):
            value = first_present_metric(nested, metric)
            if value is not None:
                return value
    return None


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return statistics.fmean(nums) if nums else None


def std(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    avg = statistics.fmean(nums)
    return math.sqrt(sum((x - avg) ** 2 for x in nums) / len(nums))


def mean_std_text(values: list[float | None]) -> str:
    avg = mean(values)
    sd = std(values)
    if avg is None:
        return "NA"
    if sd is None:
        return fmt(avg)
    return f"{fmt(avg)} +/- {fmt(sd)}"


def join_list(values: Any) -> str:
    if not isinstance(values, list):
        return "NA"
    cleaned = [fmt(v, 6) for v in values]
    return ";".join(cleaned)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_macro_weighted_rows(report_path: Path) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    rows = read_csv_rows(report_path)
    macro = next((r for r in rows if r.get("class_id") == "macro_avg"), None)
    weighted = next((r for r in rows if r.get("class_id") == "weighted_avg"), None)
    class_rows = [r for r in rows if r.get("class_id") not in ("macro_avg", "weighted_avg")]
    if macro is None or weighted is None:
        raise ValueError(f"{report_path} is missing macro_avg or weighted_avg")
    return macro, weighted, class_rows


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed[=_-]?(\d+)", str(path))
    return int(match.group(1)) if match else None


def reconstructed_seed_metrics(seed_dir: Path, dataset: str, method: str) -> dict[str, Any]:
    report_path = seed_dir / "final_classification_report.csv"
    acc_path = seed_dir / "final_per_class_accuracy.csv"
    worst_path = seed_dir / "worst_10_classes.csv"
    macro, weighted, _ = find_macro_weighted_rows(report_path)
    acc_rows = read_csv_rows(acc_path) if acc_path.exists() else []
    accuracies = [safe_float(r.get("accuracy")) for r in acc_rows]
    worst_rows = read_csv_rows(worst_path) if worst_path.exists() else []
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed_from_path(seed_dir),
        "source_status": "reconstructed_from_final_csv",
        "source_path": str(seed_dir),
        "mean_acc_seen": safe_float(weighted.get("recall")),
        "min_acc_seen": min([x for x in accuracies if x is not None], default=None),
        "final_taskwise_average_accuracy": None,
        "forgetting": None,
        "bwt": None,
        "precision_macro": safe_float(macro.get("precision")),
        "precision_weighted": safe_float(weighted.get("precision")),
        "recall_macro": safe_float(macro.get("recall")),
        "recall_weighted": safe_float(weighted.get("recall")),
        "macro_f1": safe_float(macro.get("f1_score")),
        "weighted_f1": safe_float(weighted.get("f1_score")),
        "balanced_accuracy": safe_float(macro.get("recall")),
        "old_class_accuracy": None,
        "new_class_accuracy": None,
        "plasticity_stability_gap": None,
        "memory_MB": None,
        "gpu_peak_memory_MB": None,
        "mmd_real_generated": None,
        "wasserstein_real_generated": None,
        "feature_center_l2": None,
        "accs_seen_per_task": None,
        "train_time_per_task": None,
        "worst_10_classes": "; ".join(
            f"{r.get('class_label')}:{fmt(r.get('accuracy'), 6)}" for r in worst_rows[:10]
        ) if worst_rows else "NA",
    }


def is_reconstructed_source(path: Path, run: dict[str, Any]) -> bool:
    if "reconstructed" in path.name.lower():
        return True
    cl_metrics = run.get("cl_metrics")
    if isinstance(cl_metrics, dict) and "artifact_fallback_note" in cl_metrics:
        return True
    if run.get("config_hash") is None and run.get("checkpoint_dir") is None:
        return True
    return False


def augment_reconstructed_run_from_csv(run: dict[str, Any]) -> dict[str, Any]:
    """Fill safe final-classification metrics from report_path when present."""
    report_value = run.get("report_path")
    if not report_value:
        return run
    report_path = Path(str(report_value))
    if not report_path.exists():
        return run
    try:
        macro, weighted, _ = find_macro_weighted_rows(report_path)
    except Exception:
        return run
    out = dict(run)
    out["mean_acc_seen"] = safe_float(out.get("mean_acc_seen")) or safe_float(out.get("final_accuracy")) or safe_float(weighted.get("recall"))
    out["precision_macro"] = safe_float(out.get("precision_macro")) or safe_float(macro.get("precision"))
    out["precision_weighted"] = safe_float(out.get("precision_weighted")) or safe_float(weighted.get("precision"))
    out["recall_macro"] = safe_float(out.get("recall_macro")) or safe_float(macro.get("recall"))
    out["recall_weighted"] = safe_float(out.get("recall_weighted")) or safe_float(weighted.get("recall"))
    out["macro_f1"] = safe_float(out.get("macro_f1")) or safe_float(macro.get("f1_score"))
    out["weighted_f1"] = safe_float(out.get("weighted_f1")) or safe_float(weighted.get("f1_score"))
    out["balanced_accuracy"] = safe_float(out.get("balanced_accuracy")) or safe_float(macro.get("recall"))

    acc_value = out.get("per_class_accuracy_path")
    if acc_value:
        acc_path = Path(str(acc_value))
        if acc_path.exists():
            try:
                accs = [safe_float(r.get("accuracy")) for r in read_csv_rows(acc_path)]
                out["min_acc_seen"] = safe_float(out.get("min_acc_seen")) or min(
                    [x for x in accs if x is not None],
                    default=None,
                )
            except Exception:
                pass
    return out


def run_to_seed_metrics(
    run: dict[str, Any],
    dataset: str,
    method: str,
    source_path: Path,
    source_status: str,
    reconstructed: bool,
) -> dict[str, Any]:
    cl_metrics = run.get("cl_metrics") if isinstance(run.get("cl_metrics"), dict) else {}
    row = {
        "dataset": dataset,
        "method": method,
        "seed": run.get("seed"),
        "source_status": source_status,
        "source_path": str(source_path),
        "mean_acc_seen": nested_metric(run, "mean_acc_seen"),
        "min_acc_seen": nested_metric(run, "min_acc_seen"),
        "final_taskwise_average_accuracy": nested_metric(run, "final_taskwise_average_accuracy"),
        "forgetting": nested_metric(run, "forgetting"),
        "bwt": nested_metric(run, "bwt"),
        "precision_macro": nested_metric(run, "precision_macro"),
        "precision_weighted": nested_metric(run, "precision_weighted"),
        "recall_macro": nested_metric(run, "recall_macro"),
        "recall_weighted": nested_metric(run, "recall_weighted"),
        "macro_f1": nested_metric(run, "macro_f1"),
        "weighted_f1": nested_metric(run, "weighted_f1"),
        "balanced_accuracy": nested_metric(run, "balanced_accuracy"),
        "old_class_accuracy": nested_metric(run, "old_class_accuracy"),
        "new_class_accuracy": nested_metric(run, "new_class_accuracy"),
        "plasticity_stability_gap": nested_metric(run, "plasticity_stability_gap"),
        "memory_MB": nested_metric(run, "memory_MB"),
        "gpu_peak_memory_MB": nested_metric(run, "gpu_peak_memory_MB"),
        "mmd_real_generated": nested_metric(run, "mmd_real_generated"),
        "wasserstein_real_generated": nested_metric(run, "wasserstein_real_generated"),
        "feature_center_l2": nested_metric(run, "feature_center_l2"),
        "accs_seen_per_task": join_list(run.get("accs_seen_per_task")),
        "train_time_per_task": join_list(run.get("train_time_per_task")),
        "worst_10_classes": "; ".join(
            f"{item.get('class_label')}:{fmt(item.get('accuracy'), 6)}"
            for item in run.get("worst_10_classes", [])[:10]
            if isinstance(item, dict)
        ) or "NA",
    }
    if reconstructed:
        row["source_status"] = "reconstructed_from_final_csv"
        for key in SUMMARY_METRICS:
            if key not in FINAL_CLASSIFICATION_METRICS:
                row[key] = None
        row["accs_seen_per_task"] = "NA"
        row["train_time_per_task"] = "NA"
    return row


def parse_result_json(path: Path, dataset: str, method: str) -> list[dict[str, Any]]:
    data = load_json_first_object(path)
    runs = extract_runs_from_json_data(data)
    if not runs:
        raise ValueError(f"{path} does not have the expected dataset -> runs shape")
    rows = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        reconstructed = is_reconstructed_source(path, run)
        if reconstructed:
            run = augment_reconstructed_run_from_csv(run)
        source_status = "json_reconstructed" if reconstructed else "json_complete"
        rows.append(run_to_seed_metrics(run, dataset, method, path, source_status, reconstructed))
    return rows


def extract_runs_from_json_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("runs"), list):
        return [r for r in data["runs"] if isinstance(r, dict)]

    dataset_key = next((k for k, v in data.items() if isinstance(v, dict) and "runs" in v), None)
    if dataset_key is not None and isinstance(data[dataset_key].get("runs"), list):
        return [r for r in data[dataset_key]["runs"] if isinstance(r, dict)]

    for key in ("seed_results", "seeds", "per_seed", "per_seed_results"):
        value = data.get(key)
        if isinstance(value, dict):
            runs = []
            for seed_key, seed_value in value.items():
                if isinstance(seed_value, dict):
                    row = dict(seed_value)
                    row.setdefault("seed", seed_from_path(Path(str(seed_key))) or safe_float(seed_key))
                    runs.append(row)
            if runs:
                return runs
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]

    top_seed_dicts = []
    for key, value in data.items():
        if isinstance(value, dict) and re.search(r"seed[=_-]?\d+|^\d+$", str(key)):
            row = dict(value)
            row.setdefault("seed", seed_from_path(Path(str(key))) or safe_float(key))
            top_seed_dicts.append(row)
    return top_seed_dicts


def find_json_for_method(root: Path, method: str) -> Path | None:
    candidates = []
    for path in root.rglob("*.json"):
        if path.name.startswith("IMPUTED_"):
            continue
        canonical = canonical_method_from_path(path)
        if canonical == method and "results" in path.name.lower():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: ("reconstructed" not in p.name.lower(), len(str(p))))
    return candidates[0]


def find_method_seed_dirs(root: Path, method: str) -> list[Path]:
    dirs = []
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and canonical_method_from_path(child) == method:
            dirs.extend(sorted([p for p in child.glob("seed=*") if p.is_dir()], key=lambda p: str(p)))
    return dirs


def locate_dataset_rows(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs = {
        "az": {
            "label": "AZ-Class",
            "root": repo_root / "result/final-run/AZ-Class/ablation",
            "full_json": repo_root / "result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json",
            "target": "K=100",
        },
        "ember": {
            "label": "EMBER",
            "root": repo_root / "result/final-run/EMBER/ablations",
            "full_json": repo_root / "result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json",
            "target": "K=25",
        },
    }
    all_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    for dataset, cfg in configs.items():
        found_methods: list[str] = []
        missing_methods: list[str] = []
        reconstructed_methods: list[str] = []
        warnings: list[str] = []
        method_sources: dict[str, str] = {}
        for method in EXPECTED_METHODS:
            rows: list[dict[str, Any]] = []
            if method == "full_ahr_malcl":
                path = cfg["full_json"]
                if path.exists():
                    rows = parse_result_json(path, dataset, method)
                    method_sources[method] = str(path)
                else:
                    warnings.append(f"{dataset}/{method}: missing full-method JSON at {path}")
            else:
                json_path = find_json_for_method(cfg["root"], method)
                if json_path is not None:
                    rows = parse_result_json(json_path, dataset, method)
                    method_sources[method] = str(json_path)
                    if any(r["source_status"] != "json_complete" for r in rows):
                        reconstructed_methods.append(method)
                else:
                    seed_dirs = find_method_seed_dirs(cfg["root"], method)
                    csv_dirs = [
                        d for d in seed_dirs
                        if (d / "final_classification_report.csv").exists()
                        and (d / "final_per_class_accuracy.csv").exists()
                    ]
                    if csv_dirs:
                        rows = [reconstructed_seed_metrics(d, dataset, method) for d in csv_dirs]
                        reconstructed_methods.append(method)
                        method_sources[method] = str(csv_dirs[0].parent)
                    else:
                        warnings.append(f"{dataset}/{method}: no result JSON or reconstructable final CSV artifacts found")
            if rows:
                found_methods.append(method)
                seed_values = sorted({safe_float(r.get("seed")) for r in rows if r.get("seed") is not None})
                if len(rows) != 5:
                    warnings.append(f"{dataset}/{method}: expected 5 seed rows but found {len(rows)}")
                missing_seeds = [s for s in SEEDS if float(s) not in seed_values]
                if missing_seeds:
                    warnings.append(f"{dataset}/{method}: missing seeds {missing_seeds}")
                all_rows.extend(rows)
            else:
                missing_methods.append(method)
        integrity[dataset] = {
            "label": cfg["label"],
            "target": cfg["target"],
            "root": str(cfg["root"]),
            "found_methods": found_methods,
            "missing_methods": missing_methods,
            "reconstructed_methods": sorted(set(reconstructed_methods)),
            "warnings": warnings,
            "method_sources": method_sources,
        }
    return all_rows, integrity


def summarize(rows: list[dict[str, Any]], integrity: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    for dataset in ["ember", "az"]:
        for method in EXPECTED_METHODS:
            items = grouped.get((dataset, method), [])
            if not items:
                continue
            summary = {
                "dataset": dataset,
                "target_setting": integrity[dataset]["target"],
                "method": method,
                "n_seeds": len(items),
                "source_status": ";".join(sorted({str(r["source_status"]) for r in items})),
                "source_path": items[0].get("source_path", ""),
            }
            for metric in SUMMARY_METRICS:
                vals = [safe_float(r.get(metric)) for r in items]
                summary[f"{metric}_mean"] = mean(vals)
                summary[f"{metric}_std"] = std(vals)
            summary["per_task_seen_mean"] = "NA"
            complete_item = next((r for r in items if r.get("source_status") == "json_complete"), None)
            if complete_item is not None:
                summary["per_task_seen_mean"] = complete_item.get("accs_seen_per_task", "NA")
            summary["worst_10_classes_available"] = sum(1 for r in items if r.get("worst_10_classes") not in (None, "NA", ""))
            out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if row.get(k) is None else row.get(k) for k in fieldnames})


def make_paper_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def ms(row: dict[str, Any], metric: str) -> str:
        avg = safe_float(row.get(f"{metric}_mean"))
        sd = safe_float(row.get(f"{metric}_std"))
        if avg is None:
            return "NA"
        if sd is None:
            return fmt(avg)
        return f"{fmt(avg)} +/- {fmt(sd)}"

    rows = []
    grouped = {(r["dataset"], r["method"]): r for r in summary_rows}
    for dataset in ["ember", "az"]:
        for method in EXPECTED_METHODS:
            r = grouped.get((dataset, method))
            if not r:
                rows.append({"dataset": dataset, "method": method, "available": "no"})
                continue
            rows.append({
                "dataset": dataset,
                "method": method,
                "available": "yes",
                "n_seeds": r["n_seeds"],
                "mean_acc_seen": ms(r, "mean_acc_seen"),
                "min_acc_seen": ms(r, "min_acc_seen"),
                "final_taskwise_accuracy": fmt(r["final_taskwise_average_accuracy_mean"]),
                "forgetting": fmt(r["forgetting_mean"]),
                "bwt": fmt(r["bwt_mean"]),
                "macro_f1": fmt(r["macro_f1_mean"]),
                "balanced_accuracy": fmt(r["balanced_accuracy_mean"]),
                "old_class_accuracy": fmt(r["old_class_accuracy_mean"]),
                "mmd_real_generated": fmt(r["mmd_real_generated_mean"]),
                "wasserstein_real_generated": fmt(r["wasserstein_real_generated_mean"]),
                "memory_MB": fmt(r["memory_MB_mean"]),
                "source_status": r["source_status"],
            })
    return rows


def metric_delta(summary_by_key: dict[tuple[str, str], dict[str, Any]], dataset: str, a: str, b: str, metric: str) -> float | None:
    ra = summary_by_key.get((dataset, a))
    rb = summary_by_key.get((dataset, b))
    if not ra or not rb:
        return None
    va = safe_float(ra.get(f"{metric}_mean"))
    vb = safe_float(rb.get(f"{metric}_mean"))
    if va is None or vb is None:
        return None
    return va - vb


def comparison_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], r["method"]): r for r in summary_rows}
    rows = []
    for dataset in ["ember", "az"]:
        for a, b in COMPARISONS:
            row = {"dataset": dataset, "comparison": f"{a} vs {b}", "method_a": a, "method_b": b}
            for metric in [
                "mean_acc_seen",
                "final_taskwise_average_accuracy",
                "forgetting",
                "macro_f1",
                "balanced_accuracy",
                "old_class_accuracy",
                "mmd_real_generated",
                "wasserstein_real_generated",
                "memory_MB",
            ]:
                row[f"{metric}_delta"] = metric_delta(by_key, dataset, a, b, metric)
            rows.append(row)
    return rows


def best_by_metric(summary_rows: list[dict[str, Any]], dataset: str, metric: str, lower: bool = False) -> dict[str, Any] | None:
    candidates = [r for r in summary_rows if r["dataset"] == dataset and safe_float(r.get(f"{metric}_mean")) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: safe_float(r[f"{metric}_mean"]), reverse=not lower)[0]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(fmt(value))
            elif value is None:
                vals.append("NA")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def inspect_json_structure(path: Path) -> dict[str, Any]:
    data = load_json_first_object(path)
    top_keys = list(data.keys())
    runs = extract_runs_from_json_data(data)

    aggregate = None
    if isinstance(data.get("aggregate"), dict):
        aggregate = data["aggregate"]
    else:
        for value in data.values():
            if isinstance(value, dict) and isinstance(value.get("aggregate"), dict):
                aggregate = value["aggregate"]
                break

    direct_metric_fields = sorted(
        key for key in top_keys
        if any(key in aliases for aliases in METRIC_ALIASES.values())
    )
    aggregate_metric_fields = sorted(
        key for key in (aggregate or {}).keys()
        if any(key == alias or key.startswith(f"{alias}_") for aliases in METRIC_ALIASES.values() for alias in aliases)
    )
    per_seed_metric_fields = sorted({
        key
        for run in runs
        for key in run.keys()
        if any(key in aliases for aliases in METRIC_ALIASES.values())
    })
    seeds = sorted({
        int(seed)
        for seed in (safe_float(run.get("seed")) for run in runs)
        if seed is not None
    })
    nested_seed_dict_keys = sorted(
        str(key)
        for key, value in data.items()
        if isinstance(value, dict) and re.search(r"seed[=_-]?\d+|^\d+$", str(key))
    )
    is_reconstructed = (
        "reconstructed" in path.name.lower()
        or safe_float(data.get("k")) is not None and "reconstructed" in str(data.get("source", "")).lower()
        or "final classification metrics only" in str(data.get("warning", "")).lower()
    )
    has_direct_metric_fields = bool(direct_metric_fields)
    has_seed_level_entries = bool(runs or nested_seed_dict_keys)
    return {
        "path": str(path),
        "top_level_keys": top_keys,
        "has_aggregate": aggregate is not None,
        "aggregate_keys": list(aggregate.keys()) if isinstance(aggregate, dict) else [],
        "aggregate_metric_fields": aggregate_metric_fields,
        "has_runs": isinstance(data.get("runs"), list) or any(isinstance(v, dict) and isinstance(v.get("runs"), list) for v in data.values()),
        "has_seed_level_entries": has_seed_level_entries,
        "detected_number_of_seeds": len(seeds) if seeds else len(runs),
        "detected_seeds": seeds,
        "has_direct_metric_fields": has_direct_metric_fields,
        "direct_metric_fields": direct_metric_fields,
        "has_nested_per_seed_dictionaries": bool(nested_seed_dict_keys),
        "nested_per_seed_dictionary_keys": nested_seed_dict_keys,
        "per_seed_metric_fields": per_seed_metric_fields,
        "is_reconstructed_or_final_only": is_reconstructed,
    }


def build_az_structure_debug(
    repo_root: Path,
    integrity: dict[str, Any],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    az_root = repo_root / "result/final-run/AZ-Class/ablation"
    entries: list[dict[str, Any]] = []
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        if row.get("dataset") == "az":
            rows_by_method.setdefault(str(row.get("method")), []).append(row)

    for method in EXPECTED_METHODS:
        if method == "full_ahr_malcl":
            path = repo_root / "result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json"
        else:
            path = find_json_for_method(az_root, method)
        if path is None or not path.exists():
            entry = {
                "method": method,
                "path": "MISSING",
                "top_level_keys": [],
                "detected_number_of_seeds": 0,
                "has_aggregate": False,
                "has_seed_level_entries": False,
                "missing_metrics": SUMMARY_METRICS,
                "special_note": "missing JSON",
            }
        else:
            info = inspect_json_structure(path)
            method_rows = rows_by_method.get(method, [])
            missing_metrics = [
                metric
                for metric in SUMMARY_METRICS
                if not any(safe_float(row.get(metric)) is not None for row in method_rows)
            ]
            special_note = ""
            if method in SPECIAL_AZ_METHODS:
                special_note = (
                    "special AZ handling: nonstandard/reconstructed final-only JSON was inspected separately; "
                    "available metrics only were normalized; unavailable continual-learning metrics remain NA"
                    if info["is_reconstructed_or_final_only"]
                    else "special AZ handling: inspected separately and normalized"
                )
            entry = {
                "method": method,
                **info,
                "missing_metrics": missing_metrics,
                "special_note": special_note,
            }
        entries.append(entry)

    integrity.setdefault("az", {})["json_structure_debug"] = entries
    return entries


def write_az_structure_debug(out_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = out_dir / "az_ablation_json_structure_debug.txt"
    lines = [
        "v1 - AZ-Class ablation JSON structure debug",
        "",
        "This file records JSON structure inspection before/alongside metric normalization.",
        "",
    ]
    for entry in entries:
        lines.extend([
            f"method: {entry['method']}",
            f"json_path: {entry.get('path', 'MISSING')}",
            f"top_level_keys: {entry.get('top_level_keys', [])}",
            f"detected_number_of_seeds: {entry.get('detected_number_of_seeds', 0)}",
            f"detected_seeds: {entry.get('detected_seeds', [])}",
            f"aggregate_metrics_found: {entry.get('has_aggregate', False)}",
            f"aggregate_metric_fields: {entry.get('aggregate_metric_fields', [])}",
            f"per_seed_metrics_found: {entry.get('has_seed_level_entries', False)}",
            f"per_seed_metric_fields: {entry.get('per_seed_metric_fields', [])}",
            f"direct_metric_fields_found: {entry.get('has_direct_metric_fields', False)}",
            f"nested_per_seed_dictionaries_found: {entry.get('has_nested_per_seed_dictionaries', False)}",
            f"reconstructed_or_final_only: {entry.get('is_reconstructed_or_final_only', False)}",
            f"missing_metrics_after_normalization: {entry.get('missing_metrics', [])}",
        ])
        if entry.get("special_note"):
            lines.append(f"special_note: {entry['special_note']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_az_structure_validation(entries: list[dict[str, Any]]) -> None:
    print("AZ-Class ablation JSON structure validation:")
    for entry in entries:
        print(f"- method: {entry['method']}")
        print(f"  JSON path: {entry.get('path', 'MISSING')}")
        print(f"  top-level keys: {entry.get('top_level_keys', [])}")
        print(f"  detected number of seeds: {entry.get('detected_number_of_seeds', 0)}")
        print(f"  aggregate metrics found: {entry.get('has_aggregate', False)}")
        print(f"  per-seed metrics found: {entry.get('has_seed_level_entries', False)}")
        print(f"  missing metrics: {entry.get('missing_metrics', [])}")


def write_reports(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    comp_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
    figure_status: str,
) -> None:
    by_dataset = {d: [r for r in summary_rows if r["dataset"] == d] for d in ["ember", "az"]}
    best_lines = []
    for dataset in ["ember", "az"]:
        best = best_by_metric(summary_rows, dataset, "macro_f1")
        if best:
            best_lines.append(f"- {dataset.upper()}: best macro-F1 is `{best['method']}` with macro-F1 {fmt(best['macro_f1_mean'])}.")
        else:
            best_lines.append(f"- {dataset.upper()}: macro-F1 was unavailable.")

    warnings = []
    for dataset, info in integrity.items():
        for warning in info["warnings"]:
            warnings.append(f"- {warning}")
        if info["reconstructed_methods"]:
            warnings.append(
                f"- {dataset}: reconstructed final-classification-only methods: "
                + ", ".join(info["reconstructed_methods"])
            )
        if info["missing_methods"]:
            warnings.append(f"- {dataset}: missing methods: " + ", ".join(info["missing_methods"]))
    if figure_status:
        warnings.append(f"- Optional figures: {figure_status}")

    main_table_rows = []
    for row in summary_rows:
        main_table_rows.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "n": row["n_seeds"],
            "macro_f1": fmt(row["macro_f1_mean"]),
            "mean_acc": fmt(row["mean_acc_seen_mean"]),
            "final_taskwise": fmt(row["final_taskwise_average_accuracy_mean"]),
            "forgetting": fmt(row["forgetting_mean"]),
            "old_acc": fmt(row["old_class_accuracy_mean"]),
            "mmd": fmt(row["mmd_real_generated_mean"]),
            "source": row["source_status"],
        })

    md = [
        "# Final Ablation Analysis Report: AHR-MalCL",
        "",
        "## 1. Executive Summary",
        "",
        "This report analyzes the final AHR-MalCL ablation artifacts for EMBER (K=25) and AZ-Class (K=100). The analysis uses existing JSON and CSV result files only; no training or GPU computation was run.",
        "",
        *best_lines,
        "",
        "The strongest conclusions should be drawn from methods with complete JSON metrics. Methods reconstructed from final classification CSVs support final classification comparisons but do not support claims about forgetting, BWT, memory, MMD, Wasserstein distance, or per-task dynamics.",
        "",
        "## 2. Experimental Setup",
        "",
        "- EMBER ablation target: K=25.",
        "- AZ-Class ablation target: K=100.",
        "- Expected seeds: 42, 43, 44, 45, 46.",
        "- Analysis policy: aggregate JSONs are preferred; final CSV reconstruction is used only for safe final classification metrics.",
        "",
        "## 3. Ablation Methods",
        "",
    ]
    for method in EXPECTED_METHODS:
        md.append(f"- `{method}`: {METHOD_DESCRIPTIONS[method]}")
    md.extend([
        "",
        "## 4. Data Availability and File Integrity Check",
        "",
    ])
    for dataset, info in integrity.items():
        md.append(f"### {info['label']} ({info['target']})")
        md.append(f"- Root: `{info['root']}`")
        md.append(f"- Methods found: {', '.join(info['found_methods']) or 'none'}")
        md.append(f"- Missing methods: {', '.join(info['missing_methods']) or 'none'}")
        md.append(f"- Reconstructed methods: {', '.join(info['reconstructed_methods']) or 'none'}")
        md.append("")
    md.extend([
        "## 5. EMBER K=25 Ablation Results",
        "",
        markdown_table(
            [r for r in main_table_rows if r["dataset"] == "ember"],
            ["method", "n", "macro_f1", "mean_acc", "final_taskwise", "forgetting", "old_acc", "mmd", "source"],
        ),
        "",
        "## 6. AZ-Class K=100 Ablation Results",
        "",
        markdown_table(
            [r for r in main_table_rows if r["dataset"] == "az"],
            ["method", "n", "macro_f1", "mean_acc", "final_taskwise", "forgetting", "old_acc", "mmd", "source"],
        ),
        "",
        "## Special Handling of AZ-Class hybrid_diversity_buffer and plus_kd JSON Files",
        "",
        "The AZ-Class K=100 JSON files for `hybrid_diversity_buffer` and `plus_kd` had a different layout from the other ablation outputs. The script inspected these files separately, including top-level keys, aggregate availability, run/seed entries, direct metric fields, and reconstructed/final-only markers.",
        "",
        "Only metrics available in the JSON, or safe final-classification metrics recoverable because the reconstructed JSON explicitly points to final CSV reports, were used. Missing continual-learning and replay-drift metrics such as forgetting, BWT, MMD, Wasserstein distance, memory, and per-task curves were not fabricated; they remain `NA` in the normalized tables.",
        "",
        "The final ablation tables normalize all methods into a common schema, so these two methods can still be compared fairly on available final classification metrics against `hybrid_random_buffer`, `real_buffer_only`, and `full_ahr_malcl`. Comparisons involving unavailable metrics should be interpreted as unavailable rather than neutral or zero.",
        "",
        "A detailed structure audit was written to `result/final-run/az_ablation_json_structure_debug.txt`.",
        "",
        "## 7. Component Contribution Analysis",
        "",
        markdown_table(
            comp_rows,
            [
                "dataset",
                "comparison",
                "mean_acc_seen_delta",
                "final_taskwise_average_accuracy_delta",
                "forgetting_delta",
                "macro_f1_delta",
                "balanced_accuracy_delta",
                "old_class_accuracy_delta",
                "mmd_real_generated_delta",
                "wasserstein_real_generated_delta",
                "memory_MB_delta",
            ],
        ),
        "",
        "Positive deltas indicate that the first method is larger than the second. For forgetting, MMD, Wasserstein distance, and memory, lower values are generally better, so negative deltas are favorable.",
        "",
        "## 8. Seed-Level Stability Analysis",
        "",
        "Seed-level stability is summarized by population standard deviations in the machine-readable summary CSV. Lower standard deviation indicates more stable behavior across seeds. Reconstructed methods can still be compared for final macro-F1 stability, but not for unavailable continual-learning metrics.",
        "",
        "## 9. Replay Drift and Representation Preservation Analysis",
        "",
        "Replay-drift evidence is available only when MMD, Wasserstein, or feature-center metrics are present in complete JSON files. Where these fields are NA, the source artifacts do not support drift claims. Under the evaluated protocol, improvements in MMD or Wasserstein for hybrid/full methods over generated-only baselines support replay-drift stabilization.",
        "",
        "## 10. Memory and Runtime Analysis",
        "",
        "Memory and runtime fields are included when present in complete JSON files. They are intentionally left as NA for reconstructed final-classification-only methods.",
        "",
        "## 11. Paper-Ready Ablation Table",
        "",
        markdown_table(
            paper_rows,
            [
                "dataset",
                "method",
                "available",
                "n_seeds",
                "mean_acc_seen",
                "final_taskwise_accuracy",
                "forgetting",
                "macro_f1",
                "balanced_accuracy",
                "old_class_accuracy",
                "mmd_real_generated",
                "memory_MB",
                "source_status",
            ],
        ),
        "",
        "## 12. Scientific Interpretation",
        "",
        "- If `hybrid_diversity_buffer` improves over `hybrid_random_buffer`, the results support the diversity-aware buffer claim.",
        "- If `plus_kd` improves over `hybrid_diversity_buffer`, the results support the contribution of knowledge distillation.",
        "- If `full_ahr_malcl` improves over `plus_kd`, the results support the value of prototype alignment and full integration.",
        "- If hybrid methods improve over generated-only baselines, the results support real-memory anchoring.",
        "- If generated-only WGAN/projection improves over `malcl_like`, the results support the WGAN-GP/projection critic contribution.",
        "- Claims about forgetting and replay drift should only use rows with complete JSON metrics.",
        "",
        "## 13. Risks / Missing Metrics / Limitations",
        "",
        "\n".join(warnings) if warnings else "No missing data warnings were detected.",
        "",
        "## 14. Final Recommendation for Manuscript",
        "",
        "Use the complete JSON methods as the main evidence for continual-learning claims. Use reconstructed methods only as final-classification supplemental evidence, clearly labeling them as reconstructed. The manuscript should avoid claiming guaranteed acceptance or broad SOTA superiority unless supported by external baselines and complete protocols.",
        "",
    ])

    md_path = out_dir / "AHR_MalCL_Final_Ablation_Report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    html_path = out_dir / "AHR_MalCL_Final_Ablation_Report.html"
    html_path.write_text(markdown_to_html("\n".join(md)), encoding="utf-8")


def markdown_to_html(text: str) -> str:
    body = []
    in_ul = False
    in_table = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("| ") and line.endswith(" |"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if set(cells) == {"---"}:
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = "th" if cells and cells[0] in ("dataset", "method") else "td"
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif not line:
            if in_ul:
                body.append("</ul>")
                in_ul = False
        else:
            if in_ul:
                body.append("</ul>")
                in_ul = False
            body.append(f"<p>{html.escape(line)}</p>")
    if in_ul:
        body.append("</ul>")
    if in_table:
        body.append("</table>")
    css = """
    body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1180px; line-height: 1.55; color: #1f2933; }
    h1, h2, h3 { color: #111827; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid #d0d7de; padding: 0.35rem 0.5rem; vertical-align: top; }
    th { background: #f3f4f6; }
    code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 3px; }
    """
    return "<!doctype html><html><head><meta charset='utf-8'><title>Final Ablation Analysis Report: AHR-MalCL</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>"


def maybe_make_figures(out_dir: Path, summary_rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        return f"skipped because matplotlib is unavailable ({exc})"

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("macro_f1", "Macro-F1", "ablation_macro_f1.png"),
        ("forgetting", "Forgetting", "ablation_forgetting.png"),
        ("old_class_accuracy", "Old-class accuracy", "ablation_old_accuracy.png"),
        ("mmd_real_generated", "MMD real/generated", "ablation_mmd.png"),
    ]
    for dataset in ["az", "ember"]:
        rows = [r for r in summary_rows if r["dataset"] == dataset]
        for metric, title, suffix in metrics:
            vals = [(r["method"], safe_float(r.get(f"{metric}_mean"))) for r in rows]
            vals = [(m, v) for m, v in vals if v is not None]
            if not vals:
                continue
            plt.figure(figsize=(10, 5))
            plt.bar([m for m, _ in vals], [v for _, v in vals])
            plt.xticks(rotation=35, ha="right")
            plt.title(f"{dataset.upper()} {title}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{dataset}_{suffix}", dpi=160)
            plt.close()
    return "generated under result/final-run/figures/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze final AHR-MalCL ablation results.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    return parser.parse_args()


def main() -> None:
    print(VERSION)
    args = parse_args()
    repo_root = args.root.resolve()
    out_dir = repo_root / "result/final-run"
    seed_rows, integrity = locate_dataset_rows(repo_root)
    az_debug_entries = build_az_structure_debug(repo_root, integrity, seed_rows)
    az_debug_path = write_az_structure_debug(out_dir, az_debug_entries)
    summary_rows = summarize(seed_rows, integrity)
    paper_rows = make_paper_table(summary_rows)
    comp_rows = comparison_rows(summary_rows)

    summary_fields = ["dataset", "target_setting", "method", "n_seeds", "source_status", "source_path"]
    for metric in SUMMARY_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summary_fields.extend(["per_task_seen_mean", "worst_10_classes_available"])
    write_csv(out_dir / "ablation_analysis_summary.csv", summary_rows, summary_fields)

    seed_fields = ["dataset", "method", "seed", "source_status", "source_path"] + SUMMARY_METRICS + [
        "accs_seen_per_task",
        "train_time_per_task",
        "worst_10_classes",
    ]
    write_csv(out_dir / "ablation_seed_level_metrics.csv", seed_rows, seed_fields)

    paper_fields = [
        "dataset",
        "method",
        "available",
        "n_seeds",
        "mean_acc_seen",
        "min_acc_seen",
        "final_taskwise_accuracy",
        "forgetting",
        "bwt",
        "macro_f1",
        "balanced_accuracy",
        "old_class_accuracy",
        "mmd_real_generated",
        "wasserstein_real_generated",
        "memory_MB",
        "source_status",
    ]
    write_csv(out_dir / "ablation_table_for_paper.csv", paper_rows, paper_fields)

    figure_status = maybe_make_figures(out_dir, summary_rows)
    write_reports(out_dir, summary_rows, seed_rows, paper_rows, comp_rows, integrity, figure_status)

    print_az_structure_validation(az_debug_entries)
    print(f"total datasets found: {len([d for d, info in integrity.items() if info['found_methods']])}")
    for dataset, info in integrity.items():
        print(f"{dataset}: total methods found: {len(info['found_methods'])}")
        print(f"{dataset}: missing methods: {', '.join(info['missing_methods']) or 'none'}")
        print(f"{dataset}: reconstructed methods: {', '.join(info['reconstructed_methods']) or 'none'}")
        best = best_by_metric(summary_rows, dataset, "macro_f1")
        if best:
            print(f"{dataset}: best method by macro-F1: {best['method']} ({fmt(best['macro_f1_mean'])})")
        else:
            print(f"{dataset}: best method by macro-F1: NA")
    print("outputs written:")
    print(f"- {out_dir / 'ablation_analysis_summary.csv'}")
    print(f"- {out_dir / 'ablation_seed_level_metrics.csv'}")
    print(f"- {out_dir / 'ablation_table_for_paper.csv'}")
    print(f"- {az_debug_path}")
    print(f"- {out_dir / 'AHR_MalCL_Final_Ablation_Report.md'}")
    print(f"- {out_dir / 'AHR_MalCL_Final_Ablation_Report.html'}")
    print(f"figures: {figure_status}")


if __name__ == "__main__":
    main()
