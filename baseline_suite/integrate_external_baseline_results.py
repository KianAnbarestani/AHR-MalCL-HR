#!/usr/bin/env python3
"""Integrate external baseline outputs without touching trusted results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VERSION = "v3 - external baseline paired integration"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "result_external_baselines"
OUT_DIR = PROJECT_ROOT / "paper_ready_baselines"
TRUSTED_RUNS = PROJECT_ROOT / "paper_ready_stats" / "raw" / "all_runs_long.csv"

TRUSTED_METHOD_LABEL = "AHR-MalCL-HR"
EXPECTED_SEEDS = ["42", "43", "44", "45", "46"]

METRICS = [
    "mean_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "memory_MB",
    "model_memory_MB",
]

METRIC_LABELS = {
    "mean_acc_seen": "mean seen accuracy",
    "final_taskwise_average_accuracy": "final taskwise accuracy",
    "forgetting": "forgetting",
    "macro_f1": "macro-F1",
    "weighted_f1": "weighted-F1",
    "balanced_accuracy": "balanced accuracy",
    "old_class_accuracy": "old-class accuracy",
    "new_class_accuracy": "new-class accuracy",
    "memory_MB": "memory MB",
    "model_memory_MB": "model memory MB",
}

LOWER_IS_BETTER = {"forgetting", "memory_MB", "model_memory_MB"}

METRIC_ALIASES = {
    "mean_acc_seen": ["mean_acc_seen", "mean_accuracy", "mean_acc_seen_mean"],
    "final_taskwise_average_accuracy": [
        "final_taskwise_average_accuracy",
        "final_average_accuracy_taskwise",
        "final_taskwise_accuracy",
        "final_taskwise_average_accuracy_mean",
    ],
    "forgetting": ["forgetting", "forgetting_mean"],
    "macro_f1": ["macro_f1", "f1_macro", "macro_f1_mean"],
    "weighted_f1": ["weighted_f1", "f1_weighted", "weighted_f1_mean"],
    "balanced_accuracy": ["balanced_accuracy", "balanced_accuracy_mean"],
    "old_class_accuracy": ["old_class_accuracy", "old_class_accuracy_mean"],
    "new_class_accuracy": ["new_class_accuracy", "new_class_accuracy_mean"],
    "memory_MB": ["memory_MB", "memory_mb", "memory_MB_mean"],
    "model_memory_MB": ["model_memory_MB", "model_memory_mb", "model_memory_MB_mean"],
}

EXPECTED_COMPARISONS = [
    {"baseline": "ER", "dataset": "ember", "k": "100", "target_k": "100"},
    {"baseline": "DERPP", "dataset": "ember", "k": "100", "target_k": "100"},
    {"baseline": "Joint", "dataset": "ember", "k": "", "target_k": "100"},
    {"baseline": "ER", "dataset": "az_class", "k": "200", "target_k": "200"},
    {"baseline": "DERPP", "dataset": "az_class", "k": "200", "target_k": "200"},
    {"baseline": "Joint", "dataset": "az_class", "k": "", "target_k": "200"},
]

SUMMARY_COLUMNS = [
    "baseline",
    "dataset",
    "k",
    "seed_count",
    "seeds",
    "source_files",
    "comparison_mode",
]
for _metric in METRICS:
    SUMMARY_COLUMNS.extend([f"{_metric}_mean", f"{_metric}_std", f"{_metric}_available_count"])

PAIR_COLUMNS = [
    "baseline",
    "dataset",
    "k",
    "metric",
    "metric_label",
    "metric_direction",
    "comparison_target",
    "comparison_basis",
    "paired_seed_count",
    "seeds",
    "mean_ahr",
    "mean_baseline",
    "mean_difference_ahr_minus_baseline",
    "percentage_difference_ahr_minus_baseline",
    "ci95_low",
    "ci95_high",
    "cohen_dz",
    "winner_by_mean",
    "ahr_wins_by_mean",
    "baseline_wins_by_mean",
    "paired_t_statistic",
    "paired_t_p_value",
    "paired_t_p_holm",
    "paired_t_significant_0_05",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "wilcoxon_p_holm",
    "wilcoxon_significant_0_05",
    "test_status",
    "note",
]

EFFECT_COLUMNS = [
    "baseline",
    "dataset",
    "k",
    "metric",
    "metric_label",
    "metric_direction",
    "n",
    "mean_ahr",
    "mean_baseline",
    "mean_difference_ahr_minus_baseline",
    "percentage_difference_ahr_minus_baseline",
    "ci95_low",
    "ci95_high",
    "cohen_dz",
    "winner_by_mean",
    "paired_t_p_holm",
    "wilcoxon_p_holm",
]

AVAILABILITY_COLUMNS = [
    "method",
    "baseline",
    "dataset",
    "k",
    "metric",
    "expected_seed_count",
    "available_count",
    "missing_count",
    "seeds_available",
    "seeds_missing",
    "source_files",
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


def clean_number(value: Any) -> Any:
    number = safe_float(value)
    if number is None:
        return ""
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def normalize_seed(value: Any) -> str:
    if value in (None, "", "None", "null"):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isnan(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return str(value).strip()


def normalize_k(value: Any) -> str:
    if value in (None, "", "None", "null"):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().replace("K=", "").replace("k=", "")
    if math.isnan(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return str(value).strip()


def seed_sort_key(seed: str) -> tuple[int, Any]:
    try:
        return (0, int(seed))
    except (TypeError, ValueError):
        return (1, str(seed))


def sorted_seeds(seeds: set[str] | list[str]) -> list[str]:
    return sorted([seed for seed in seeds if seed], key=seed_sort_key)


def canonical_dataset(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"ember", "ember2018"}:
        return "ember"
    if text in {"az", "az_class", "azclass", "az_classification"}:
        return "az_class"
    if "ember" in text:
        return "ember"
    if "az" in text:
        return "az_class"
    return text


def canonical_baseline(value: Any) -> str:
    raw = str(value or "").strip()
    compact = raw.lower().replace("+", "p").replace("-", "").replace("_", "").replace(" ", "")
    if compact in {"er", "experiencereplay", "replay"}:
        return "ER"
    if compact in {"derpp", "derp", "derplusplus", "darkexperiencereplay"}:
        return "DERPP"
    if compact in {"joint", "jointofflineupperbound", "offlineoracle", "oracle"}:
        return "Joint"
    return raw


def friendly_dataset(dataset: str) -> str:
    return "AZ-Class" if dataset == "az_class" else "EMBER" if dataset == "ember" else dataset


def metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json_object(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = text.lstrip()
        payload, _ = json.JSONDecoder().raw_decode(stripped)
        return payload


def iter_dicts(payload: Any) -> list[dict[str, Any]]:
    stack = [payload]
    out: list[dict[str, Any]] = []
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            out.append(current)
            for value in current.values():
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend(item for item in value if isinstance(item, dict))
    return out


def nested_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for mapping in iter_dicts(payload):
        for key in keys:
            if key in mapping and mapping.get(key) not in (None, ""):
                return mapping.get(key)
    return None


def metric_from_payload(payload: dict[str, Any], metric: str) -> float | None:
    for mapping in iter_dicts(payload):
        for key in METRIC_ALIASES.get(metric, [metric]):
            if key in mapping:
                value = safe_float(mapping.get(key))
                if value is not None:
                    return value
    return None


def list_from_payload(payload: dict[str, Any], keys: list[str]) -> list[float]:
    for mapping in iter_dicts(payload):
        for key in keys:
            raw = mapping.get(key)
            if isinstance(raw, list):
                values = [safe_float(item) for item in raw]
                return [value for value in values if value is not None]
    return []


def final_accs_from_matrix(payload: dict[str, Any]) -> list[float]:
    matrix = nested_first(payload, ["acc_matrix_taskwise", "accuracy_matrix", "task_accuracy_matrix"])
    if not isinstance(matrix, list) or not matrix:
        return []
    final_row = matrix[-1]
    if not isinstance(final_row, list):
        return []
    values = [safe_float(item) for item in final_row]
    return [value for value in values if value is not None and value >= 0.0]


def set_metric(row: dict[str, Any], metric: str, value: Any, source: str) -> None:
    number = safe_float(value)
    if number is None:
        return
    row[metric] = number
    row.setdefault("_metric_sources", {})[metric] = source


def fill_derived_metrics(row: dict[str, Any], payload: dict[str, Any]) -> None:
    if safe_float(row.get("mean_acc_seen")) is None:
        accs = list_from_payload(payload, ["accs_seen_per_task", "seen_accuracies", "mean_acc_seen_per_task"])
        if accs:
            set_metric(row, "mean_acc_seen", mean(accs), "derived_from_accs_seen_per_task")

    final_accs = final_accs_from_matrix(payload)
    if safe_float(row.get("final_taskwise_average_accuracy")) is None and final_accs:
        set_metric(row, "final_taskwise_average_accuracy", mean(final_accs), "derived_from_acc_matrix_taskwise")

    if safe_float(row.get("forgetting")) is None:
        accs = list_from_payload(payload, ["accs_seen_per_task", "seen_accuracies", "mean_acc_seen_per_task"])
        if len(accs) > 1:
            set_metric(row, "forgetting", max(accs) - accs[-1], "derived_from_accs_seen_per_task")
        elif row.get("baseline") == "Joint" and accs:
            set_metric(row, "forgetting", 0.0, "joint_single_pass_default")


def parse_classification_report(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    report: dict[str, dict[str, str]] = {}
    for row in rows:
        label = str(row.get("label", row.get("", ""))).strip().lower().replace("_", " ")
        if label:
            report[label] = row
    return report


def mean_per_class_accuracy(path: Path) -> float | None:
    values = []
    for row in read_csv_rows(path):
        value = safe_float(row.get("accuracy"))
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def balanced_accuracy_from_confusion(path: Path) -> float | None:
    matrix: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            values = [safe_float(item) for item in row]
            if values and all(value is not None for value in values):
                matrix.append([float(value) for value in values if value is not None])
    if not matrix:
        return None
    recalls = []
    for index, row in enumerate(matrix):
        total = sum(row)
        if total <= 0 or index >= len(row):
            continue
        recalls.append(row[index] / total)
    return mean(recalls) if recalls else None


def recover_from_sibling_csvs(row: dict[str, Any], payload_path: Path) -> None:
    seed = normalize_seed(row.get("seed"))
    candidate_dirs = [payload_path.parent]
    if seed:
        candidate_dirs.append(payload_path.parent / f"seed={seed}")
        candidate_dirs.append(payload_path.parent.parent / f"seed={seed}")

    seen_dirs: set[Path] = set()
    for directory in candidate_dirs:
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)

        report_path = directory / "final_classification_report.csv"
        if report_path.exists():
            report = parse_classification_report(report_path)
            macro = report.get("macro avg") or report.get("macro")
            weighted = report.get("weighted avg") or report.get("weighted")
            if safe_float(row.get("macro_f1")) is None and macro:
                set_metric(row, "macro_f1", macro.get("f1-score"), "final_classification_report.csv")
            if safe_float(row.get("weighted_f1")) is None and weighted:
                set_metric(row, "weighted_f1", weighted.get("f1-score"), "final_classification_report.csv")
            if safe_float(row.get("balanced_accuracy")) is None and macro:
                set_metric(row, "balanced_accuracy", macro.get("recall"), "final_classification_report.csv")

        per_class_path = directory / "final_per_class_accuracy.csv"
        if per_class_path.exists() and safe_float(row.get("balanced_accuracy")) is None:
            set_metric(row, "balanced_accuracy", mean_per_class_accuracy(per_class_path), "final_per_class_accuracy.csv")

        confusion_path = directory / "final_confusion_matrix.csv"
        if confusion_path.exists() and safe_float(row.get("balanced_accuracy")) is None:
            set_metric(row, "balanced_accuracy", balanced_accuracy_from_confusion(confusion_path), "final_confusion_matrix.csv")


def infer_from_path(path: Path, input_root: Path) -> dict[str, str]:
    try:
        parts = path.relative_to(input_root).parts
    except ValueError:
        parts = path.parts
    info = {"baseline": "", "dataset": "", "k": "", "seed": ""}
    if parts:
        info["baseline"] = canonical_baseline(parts[0])
    for part in parts:
        lower = part.lower()
        if canonical_dataset(part) in {"ember", "az_class"}:
            info["dataset"] = canonical_dataset(part)
        if lower.startswith("k="):
            info["k"] = normalize_k(part.split("=", 1)[1])
        if lower.startswith("seed="):
            info["seed"] = normalize_seed(part.split("=", 1)[1])
    return info


def row_from_payload(payload: dict[str, Any], path: Path, input_root: Path, source_type: str) -> dict[str, Any]:
    path_info = infer_from_path(path, input_root)
    baseline = canonical_baseline(nested_first(payload, ["method", "baseline", "baseline_name"]) or path_info["baseline"])
    dataset = canonical_dataset(
        nested_first(payload, ["dataset"])
        or nested_first(payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}, ["dataset"])
        or path_info["dataset"]
    )
    k = normalize_k(nested_first(payload, ["k", "k_per_class"]) or path_info["k"])
    if not k and isinstance(payload.get("config"), dict):
        memory_budget = payload["config"].get("memory_budget")
        if isinstance(memory_budget, dict):
            k = normalize_k(memory_budget.get("k_per_class"))
    if baseline == "Joint":
        k = ""

    row: dict[str, Any] = {
        "baseline": baseline,
        "dataset": dataset,
        "k": k,
        "seed": normalize_seed(nested_first(payload, ["seed"]) or path_info["seed"]),
        "source_file": relative_path(path),
        "source_type": source_type,
        "_metric_sources": {},
    }
    for metric in METRICS:
        set_metric(row, metric, metric_from_payload(payload, metric), "json_payload")
    fill_derived_metrics(row, payload)
    recover_from_sibling_csvs(row, path)
    return row


def is_result_json(path: Path) -> bool:
    name = path.name.lower()
    return name in {"full-result.json", "aggregate.json"} or "result" in name


def dedupe_external_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"full_result": 3, "json_result": 2, "aggregate_run": 1}
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("parse_error"):
            continue
        key = (
            str(row.get("baseline", "")),
            str(row.get("dataset", "")),
            str(row.get("k", "")),
            str(row.get("seed", "")),
        )
        if not key[0] or not key[1] or not key[3]:
            continue
        current = deduped.get(key)
        if current is None or rank.get(str(row.get("source_type")), 0) > rank.get(str(current.get("source_type")), 0):
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: (item["dataset"], item["baseline"], item["k"], seed_sort_key(item["seed"])))


def discover_external_runs(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not input_root.exists():
        return [], []

    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("*.json")):
        if not is_result_json(path):
            continue
        try:
            payload = load_json_object(path)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append({"source_file": relative_path(path), "parse_error": f"{type(exc).__name__}: {exc}"})
            continue

        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            for run_payload in payload["runs"]:
                if isinstance(run_payload, dict):
                    rows.append(row_from_payload(run_payload, path, input_root, "aggregate_run"))
            continue
        if isinstance(payload, dict):
            source_type = "full_result" if path.name == "full-result.json" else "json_result"
            rows.append(row_from_payload(payload, path, input_root, source_type))
    return dedupe_external_rows(rows), parse_errors


def aggregate_external(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("baseline", "")), str(row.get("dataset", "")), str(row.get("k", "")))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (baseline, dataset, k), items in sorted(groups.items()):
        seeds = sorted_seeds({str(item.get("seed", "")) for item in items})
        summary: dict[str, Any] = {
            "baseline": baseline,
            "dataset": dataset,
            "k": k,
            "seed_count": len(seeds),
            "seeds": ";".join(seeds),
            "source_files": ";".join(sorted({str(item.get("source_file", "")) for item in items})),
            "comparison_mode": "paired against trusted valid_main_hr/main_final_table rows when seed and metric sets match",
        }
        for metric in METRICS:
            values = [safe_float(item.get(metric)) for item in items]
            values = [value for value in values if value is not None]
            summary[f"{metric}_available_count"] = len(values)
            summary[f"{metric}_mean"] = mean(values) if values else ""
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else "")
        out.append(summary)
    return out


def trusted_target_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(TRUSTED_RUNS):
        dataset = canonical_dataset(row.get("dataset"))
        k = normalize_k(row.get("k"))
        if row.get("method") != "hybrid_random_buffer":
            continue
        if row.get("result_validity") != "valid_main_hr":
            continue
        if row.get("paper_use_category") != "main_final_table":
            continue
        if row.get("setting_type") != "final_best_performance":
            continue
        if (dataset, k) not in {("ember", "100"), ("az_class", "200")}:
            continue
        trusted = dict(row)
        trusted["dataset"] = dataset
        trusted["k"] = k
        trusted["seed"] = normalize_seed(row.get("seed"))
        out.append(trusted)
    return sorted(out, key=lambda item: (item["dataset"], item["k"], seed_sort_key(item["seed"])))


def grouped_by_dataset_k(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("k", ""))), []).append(row)
    return groups


def external_group_for(rows: list[dict[str, Any]], expected: dict[str, str]) -> list[dict[str, Any]]:
    baseline = expected["baseline"]
    dataset = expected["dataset"]
    k = expected["k"]
    out = []
    for row in rows:
        if row.get("baseline") != baseline or row.get("dataset") != dataset:
            continue
        if baseline != "Joint" and str(row.get("k", "")) != k:
            continue
        out.append(row)
    return sorted(out, key=lambda item: seed_sort_key(str(item.get("seed", ""))))


def values_by_seed(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    return {normalize_seed(row.get("seed")): safe_float(row.get(metric)) for row in rows if normalize_seed(row.get("seed"))}


def winner_for(metric: str, mean_diff: float | None, baseline: str) -> tuple[str, bool, bool]:
    if mean_diff is None or abs(mean_diff) < 1e-12:
        return "tie", False, False
    ahr_wins = mean_diff < 0 if metric in LOWER_IS_BETTER else mean_diff > 0
    return (TRUSTED_METHOD_LABEL if ahr_wins else baseline, ahr_wins, not ahr_wins)


def maybe_import_scipy_stats() -> Any:
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001
        return None
    return stats


def paired_statistics(ahr_values: list[float], baseline_values: list[float], stats_module: Any) -> dict[str, Any]:
    diffs = [ahr - baseline for ahr, baseline in zip(ahr_values, baseline_values)]
    mean_ahr = mean(ahr_values)
    mean_baseline = mean(baseline_values)
    mean_diff = mean(diffs)
    sd_diff = stdev(diffs) if len(diffs) > 1 else 0.0
    sem = sd_diff / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    if len(diffs) > 1 and sem > 0 and stats_module is not None:
        t_crit = float(stats_module.t.ppf(0.975, len(diffs) - 1))
        ci_low = mean_diff - t_crit * sem
        ci_high = mean_diff + t_crit * sem
    else:
        ci_low = mean_diff
        ci_high = mean_diff

    percentage_diff = ""
    if abs(mean_baseline) > 1e-12:
        percentage_diff = (mean_diff / abs(mean_baseline)) * 100.0

    cohen_dz = ""
    if sd_diff > 0:
        cohen_dz = mean_diff / sd_diff

    result: dict[str, Any] = {
        "mean_ahr": mean_ahr,
        "mean_baseline": mean_baseline,
        "mean_difference_ahr_minus_baseline": mean_diff,
        "percentage_difference_ahr_minus_baseline": percentage_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_dz,
        "paired_t_statistic": "",
        "paired_t_p_value": "",
        "wilcoxon_statistic": "",
        "wilcoxon_p_value": "",
        "note": "",
    }
    if stats_module is None:
        result["note"] = "SciPy unavailable; p-values not computed."
        return result

    if all(abs(diff) < 1e-12 for diff in diffs):
        result.update({"paired_t_statistic": 0.0, "paired_t_p_value": 1.0})
        result["note"] = "All paired differences are zero; Wilcoxon is undefined."
        return result

    try:
        t_res = stats_module.ttest_rel(ahr_values, baseline_values)
        result["paired_t_statistic"] = clean_number(t_res.statistic)
        result["paired_t_p_value"] = clean_number(t_res.pvalue)
    except Exception as exc:  # noqa: BLE001
        result["note"] = f"paired t-test failed: {type(exc).__name__}: {exc}"

    try:
        w_res = stats_module.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        result["wilcoxon_statistic"] = clean_number(w_res.statistic)
        result["wilcoxon_p_value"] = clean_number(w_res.pvalue)
    except Exception as exc:  # noqa: BLE001
        suffix = f" Wilcoxon failed: {type(exc).__name__}: {exc}"
        result["note"] = (str(result.get("note", "")) + suffix).strip()
    return result


def holm_adjust(rows: list[dict[str, Any]], p_col: str, out_col: str, sig_col: str) -> None:
    indexed: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        p_value = safe_float(row.get(p_col))
        if p_value is not None and 0 <= p_value <= 1:
            indexed.append((index, p_value))
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running = max(running, adjusted)
        rows[index][out_col] = running
        rows[index][sig_col] = running <= 0.05


def build_paired_tests(external_rows: list[dict[str, Any]], trusted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats_module = maybe_import_scipy_stats()
    trusted_groups = grouped_by_dataset_k(trusted_rows)
    tests: list[dict[str, Any]] = []

    for expected in EXPECTED_COMPARISONS:
        baseline = expected["baseline"]
        dataset = expected["dataset"]
        k = expected["k"]
        trusted = trusted_groups.get((dataset, expected["target_k"]), [])
        external = external_group_for(external_rows, expected)
        comparison_basis = "dataset+seed" if baseline == "Joint" else "dataset+K+seed"

        for metric in METRICS:
            row: dict[str, Any] = {
                "baseline": baseline,
                "dataset": dataset,
                "k": k,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "metric_direction": metric_direction(metric),
                "comparison_target": TRUSTED_METHOD_LABEL,
                "comparison_basis": comparison_basis,
                "test_status": "skipped",
            }
            if not trusted:
                row["note"] = "No trusted valid_main_hr/main_final_table target rows were found."
                tests.append(row)
                continue
            if not external:
                row["note"] = "No external seed-level rows were found for this expected comparison."
                tests.append(row)
                continue

            trusted_seeds = {str(item.get("seed", "")) for item in trusted if str(item.get("seed", ""))}
            external_seeds = {str(item.get("seed", "")) for item in external if str(item.get("seed", ""))}
            row["paired_seed_count"] = len(trusted_seeds & external_seeds)
            row["seeds"] = ";".join(sorted_seeds(trusted_seeds & external_seeds))
            if trusted_seeds != external_seeds:
                row["note"] = (
                    "Seed set mismatch; paired tests require identical seeds. "
                    f"trusted={';'.join(sorted_seeds(trusted_seeds))}; "
                    f"external={';'.join(sorted_seeds(external_seeds))}."
                )
                tests.append(row)
                continue

            trusted_values = values_by_seed(trusted, metric)
            external_values = values_by_seed(external, metric)
            missing = [
                seed
                for seed in sorted_seeds(trusted_seeds)
                if trusted_values.get(seed) is None or external_values.get(seed) is None
            ]
            if missing:
                row["note"] = f"Metric missing for paired seed(s): {';'.join(missing)}."
                tests.append(row)
                continue

            seeds = sorted_seeds(trusted_seeds)
            ahr_values = [float(trusted_values[seed]) for seed in seeds if trusted_values[seed] is not None]
            baseline_values = [float(external_values[seed]) for seed in seeds if external_values[seed] is not None]
            if len(ahr_values) < 2:
                row["note"] = "At least two paired seed values are required."
                tests.append(row)
                continue

            stats_row = paired_statistics(ahr_values, baseline_values, stats_module)
            row.update(stats_row)
            row["paired_seed_count"] = len(seeds)
            row["seeds"] = ";".join(seeds)
            winner, ahr_wins, baseline_wins = winner_for(metric, safe_float(row.get("mean_difference_ahr_minus_baseline")), baseline)
            row["winner_by_mean"] = winner
            row["ahr_wins_by_mean"] = ahr_wins
            row["baseline_wins_by_mean"] = baseline_wins
            row["test_status"] = "tested" if safe_float(row.get("paired_t_p_value")) is not None else "effect_only"
            tests.append(row)

    holm_adjust(tests, "paired_t_p_value", "paired_t_p_holm", "paired_t_significant_0_05")
    holm_adjust(tests, "wilcoxon_p_value", "wilcoxon_p_holm", "wilcoxon_significant_0_05")
    return tests


def build_effect_sizes(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in paired_rows:
        if safe_float(row.get("mean_difference_ahr_minus_baseline")) is None:
            continue
        out.append(
            {
                "baseline": row.get("baseline", ""),
                "dataset": row.get("dataset", ""),
                "k": row.get("k", ""),
                "metric": row.get("metric", ""),
                "metric_label": row.get("metric_label", ""),
                "metric_direction": row.get("metric_direction", ""),
                "n": row.get("paired_seed_count", ""),
                "mean_ahr": row.get("mean_ahr", ""),
                "mean_baseline": row.get("mean_baseline", ""),
                "mean_difference_ahr_minus_baseline": row.get("mean_difference_ahr_minus_baseline", ""),
                "percentage_difference_ahr_minus_baseline": row.get("percentage_difference_ahr_minus_baseline", ""),
                "ci95_low": row.get("ci95_low", ""),
                "ci95_high": row.get("ci95_high", ""),
                "cohen_dz": row.get("cohen_dz", ""),
                "winner_by_mean": row.get("winner_by_mean", ""),
                "paired_t_p_holm": row.get("paired_t_p_holm", ""),
                "wilcoxon_p_holm": row.get("wilcoxon_p_holm", ""),
            }
        )
    return out


def expected_external_seed_set(dataset: str) -> set[str]:
    trusted = [row for row in trusted_target_rows() if row.get("dataset") == dataset]
    seeds = {str(row.get("seed", "")) for row in trusted if str(row.get("seed", ""))}
    return seeds or set(EXPECTED_SEEDS)


def availability_for_group(
    method: str,
    baseline: str,
    dataset: str,
    k: str,
    rows: list[dict[str, Any]],
    expected_seeds: set[str],
    note: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_files = ";".join(sorted({str(row.get("source_file", "")) for row in rows if row.get("source_file")}))
    for metric in METRICS:
        values = values_by_seed(rows, metric)
        available = {seed for seed, value in values.items() if value is not None}
        missing = expected_seeds - available
        out.append(
            {
                "method": method,
                "baseline": baseline,
                "dataset": dataset,
                "k": k,
                "metric": metric,
                "expected_seed_count": len(expected_seeds),
                "available_count": len(available),
                "missing_count": len(missing),
                "seeds_available": ";".join(sorted_seeds(available)),
                "seeds_missing": ";".join(sorted_seeds(missing)),
                "source_files": source_files,
                "note": note,
            }
        )
    return out


def build_metric_availability(external_rows: list[dict[str, Any]], trusted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trusted_groups = grouped_by_dataset_k(trusted_rows)
    for (dataset, k), items in sorted(trusted_groups.items()):
        seeds = {str(item.get("seed", "")) for item in items if str(item.get("seed", ""))}
        rows.extend(
            availability_for_group(
                TRUSTED_METHOD_LABEL,
                "",
                dataset,
                k,
                items,
                seeds,
                "trusted valid_main_hr/main_final_table rows",
            )
        )

    handled: set[tuple[str, str, str]] = set()
    for expected in EXPECTED_COMPARISONS:
        baseline = expected["baseline"]
        dataset = expected["dataset"]
        k = expected["k"]
        items = external_group_for(external_rows, expected)
        expected_seeds = expected_external_seed_set(dataset)
        handled.add((baseline, dataset, k))
        note = (
            "external seed-level rows found"
            if items
            else "no external seed-level rows found for this expected comparison"
        )
        rows.extend(availability_for_group(baseline, baseline, dataset, k, items, expected_seeds, note))

    for group in sorted({(row["baseline"], row["dataset"], row["k"]) for row in external_rows} - handled):
        baseline, dataset, k = group
        items = [row for row in external_rows if (row["baseline"], row["dataset"], row["k"]) == group]
        seeds = {str(item.get("seed", "")) for item in items if str(item.get("seed", ""))}
        rows.extend(availability_for_group(baseline, baseline, dataset, k, items, seeds, "additional external group"))
    return rows


def format_value(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean_value = format_value(row.get(f"{metric}_mean"))
    if not mean_value:
        return ""
    std_value = format_value(row.get(f"{metric}_std"))
    return f"{mean_value} +/- {std_value}" if std_value else mean_value


def write_summary_markdown(summary_rows: list[dict[str, Any]], input_root: Path) -> None:
    lines = [
        "# External Baseline Summary",
        "",
        f"Input root: `{input_root}`.",
        "",
        "Forgetting, memory_MB, and model_memory_MB are lower-is-better. Accuracy and F1 metrics are higher-is-better.",
        "",
    ]
    if not summary_rows:
        lines.extend(
            [
                "No external seed-level baseline result rows were found.",
                "",
                "Copy Colab-produced `full-result.json` seed directories into `result_external_baselines/` and rerun the integrator.",
                "",
            ]
        )
    else:
        lines.append(
            "| baseline | dataset | k | seeds | final taskwise accuracy | forgetting | macro-F1 | weighted-F1 | balanced accuracy | memory MB |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in summary_rows:
            lines.append(
                "| {baseline} | {dataset} | {k} | {seeds} | {final_acc} | {forgetting} | {macro} | {weighted} | {balanced} | {memory} |".format(
                    baseline=row.get("baseline", ""),
                    dataset=friendly_dataset(str(row.get("dataset", ""))),
                    k=row.get("k", ""),
                    seeds=row.get("seeds", ""),
                    final_acc=format_mean_std(row, "final_taskwise_average_accuracy"),
                    forgetting=format_mean_std(row, "forgetting"),
                    macro=format_mean_std(row, "macro_f1"),
                    weighted=format_mean_std(row, "weighted_f1"),
                    balanced=format_mean_std(row, "balanced_accuracy"),
                    memory=format_mean_std(row, "memory_MB"),
                )
            )
        lines.append("")
    (OUT_DIR / "external_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def significant_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in paired_rows if row.get("paired_t_significant_0_05") is True]


def tested_rows_for(paired_rows: list[dict[str, Any]], dataset: str, baseline: str) -> list[dict[str, Any]]:
    return [
        row
        for row in paired_rows
        if row.get("dataset") == dataset
        and row.get("baseline") == baseline
        and safe_float(row.get("mean_difference_ahr_minus_baseline")) is not None
    ]


def describe_metric_list(rows: list[dict[str, Any]], winner: str) -> str:
    metrics = [str(row.get("metric", "")) for row in rows if row.get("winner_by_mean") == winner]
    return ", ".join(metrics) if metrics else "none"


def trusted_summary(trusted_rows: list[dict[str, Any]], dataset: str, k: str) -> dict[str, float]:
    rows = [row for row in trusted_rows if row.get("dataset") == dataset and row.get("k") == k]
    out: dict[str, float] = {}
    for metric in METRICS:
        values = [safe_float(row.get(metric)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            out[metric] = mean(values)
    return out


def summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["baseline"], row["dataset"], row["k"]): row for row in summary_rows}


def write_claim_safety(
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    trusted_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    input_root: Path,
) -> None:
    lines = [
        "# External Baseline Claim Safety",
        "",
        f"Input root: `{input_root}`.",
        f"Trusted AHR-MalCL-HR seed rows found: {len(trusted_rows)}.",
        f"External seed-level rows found: {len(external_rows)}.",
        f"External parse errors: {len(parse_errors)}.",
        "",
        "Interpretation rule: `mean_difference_ahr_minus_baseline = AHR-MalCL-HR - external baseline`. For forgetting, lower is better, so a negative difference favors AHR-MalCL-HR. For accuracy, F1, and balanced accuracy, a positive difference favors AHR-MalCL-HR.",
        "",
    ]

    if not external_rows:
        lines.extend(
            [
                "## EMBER",
                "",
                "- External ER, DERPP, and Joint seed-level rows are unavailable locally, so no EMBER external comparison is claim-safe yet.",
                "",
                "## AZ-Class",
                "",
                "- External ER, DERPP, and Joint seed-level rows are unavailable locally, so no AZ-Class external comparison is claim-safe yet.",
                "",
                "## Wins And Losses",
                "",
                "- AHR-MalCL-HR wins: unavailable.",
                "- ER/DERPP/Joint wins: unavailable.",
                "- Holm-significant differences: none computed.",
                "",
                "## Safe Claims",
                "",
                "- Safe: the integration pipeline is prepared to compare ER, DERPP, and Joint against the trusted valid_main_hr/main_final_table AHR-MalCL-HR rows.",
                "- Safe: current paper claims should continue to rely on the already validated internal statistical tables until external seed files are copied in.",
                "",
                "## Unsafe Claims",
                "",
                "- Unsafe: do not claim AHR-MalCL-HR beats ER, DERPP, or Joint on EMBER or AZ-Class from this workspace yet.",
                "- Unsafe: do not quote prompt-level Colab summary numbers as integrated evidence until the corresponding seed JSON/CSV files exist under `result_external_baselines/`.",
                "",
                "## MADAR",
                "",
                "- MADAR is still recommended as an external reference baseline; no MADAR result pack is integrated here.",
                "",
            ]
        )
        (OUT_DIR / "external_baseline_claim_safety.md").write_text("\n".join(lines), encoding="utf-8")
        return

    summaries = summary_lookup(summary_rows)
    for dataset, target_k in [("ember", "100"), ("az_class", "200")]:
        trusted = trusted_summary(trusted_rows, dataset, target_k)
        lines.extend(
            [
                f"## {friendly_dataset(dataset)}",
                "",
                f"- AHR-MalCL-HR trusted K={target_k}: final taskwise accuracy {format_value(trusted.get('final_taskwise_average_accuracy'))}, forgetting {format_value(trusted.get('forgetting'))}.",
            ]
        )
        for expected in [item for item in EXPECTED_COMPARISONS if item["dataset"] == dataset]:
            key = (expected["baseline"], dataset, expected["k"])
            summary = summaries.get(key)
            if not summary:
                lines.append(f"- {expected['baseline']}: no seed-level rows integrated.")
                continue
            lines.append(
                "- {baseline} K={k}: final taskwise accuracy {final_acc}, forgetting {forgetting}, seeds {seeds}.".format(
                    baseline=expected["baseline"],
                    k=expected["k"] or "offline",
                    final_acc=format_mean_std(summary, "final_taskwise_average_accuracy"),
                    forgetting=format_mean_std(summary, "forgetting"),
                    seeds=summary.get("seeds", ""),
                )
            )
        lines.append("")

        for expected in [item for item in EXPECTED_COMPARISONS if item["dataset"] == dataset]:
            rows = tested_rows_for(paired_rows, dataset, expected["baseline"])
            if not rows:
                lines.append(f"- {expected['baseline']}: no tested paired metric rows.")
                continue
            lines.append(
                f"- {expected['baseline']} metrics favoring AHR-MalCL-HR by mean: {describe_metric_list(rows, TRUSTED_METHOD_LABEL)}."
            )
            lines.append(
                f"- {expected['baseline']} metrics favoring {expected['baseline']} by mean: {describe_metric_list(rows, expected['baseline'])}."
            )
        lines.append("")

    sig = significant_rows(paired_rows)
    lines.extend(["## Holm-Significant Differences", ""])
    if not sig:
        lines.append("- None at alpha=0.05 after Holm correction.")
    else:
        for row in sig:
            direction = "favored AHR-MalCL-HR" if row.get("winner_by_mean") == TRUSTED_METHOD_LABEL else f"favored {row.get('baseline')}"
            lines.append(
                "- {dataset} {baseline} {metric}: Holm p={p}; {direction}; mean diff={diff}.".format(
                    dataset=friendly_dataset(str(row.get("dataset", ""))),
                    baseline=row.get("baseline", ""),
                    metric=row.get("metric", ""),
                    p=format_value(row.get("paired_t_p_holm"), 6),
                    direction=direction,
                    diff=format_value(row.get("mean_difference_ahr_minus_baseline"), 6),
                )
            )
    lines.append("")

    safe = [
        row
        for row in sig
        if row.get("winner_by_mean") == TRUSTED_METHOD_LABEL and row.get("metric") not in {"memory_MB", "model_memory_MB"}
    ]
    unsafe = [row for row in paired_rows if row.get("baseline_wins_by_mean") is True and row.get("metric") in {"final_taskwise_average_accuracy", "mean_acc_seen", "macro_f1", "weighted_f1", "balanced_accuracy", "forgetting"}]

    lines.extend(["## Safe Claims", ""])
    if safe:
        for row in safe:
            verb = "lower" if row.get("metric") == "forgetting" else "higher"
            lines.append(
                f"- AHR-MalCL-HR has {verb} {row.get('metric')} than {row.get('baseline')} on {friendly_dataset(str(row.get('dataset', '')))} under paired, Holm-corrected testing."
            )
    else:
        lines.append("- No external superiority claim is Holm-significant for AHR-MalCL-HR yet.")
    lines.append("- Joint should be described as an offline upper bound, not a memory-limited continual-learning method.")
    lines.append("")

    lines.extend(["## Unsafe Claims", ""])
    if unsafe:
        seen = set()
        for row in unsafe:
            key = (row.get("dataset"), row.get("baseline"), row.get("metric"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- Do not claim AHR-MalCL-HR beats {row.get('baseline')} on {friendly_dataset(str(row.get('dataset', '')))} {row.get('metric')}; the paired mean favors {row.get('baseline')}."
            )
    else:
        lines.append("- Avoid broad superiority claims unless they are tied to the metric-specific paired tests above.")
    lines.append("")

    lines.extend(
        [
            "## MADAR",
            "",
            "- MADAR is still recommended as an external reference baseline unless a comparable MADAR seed-level result pack is integrated and audited.",
            "",
        ]
    )
    (OUT_DIR / "external_baseline_claim_safety.md").write_text("\n".join(lines), encoding="utf-8")


def write_missing_report(external_rows: list[dict[str, Any]], parse_errors: list[dict[str, Any]], input_root: Path) -> None:
    lines = [
        "# External Baseline Missing Results",
        "",
        f"Input root: `{input_root}`.",
        f"External seed-level rows found: {len(external_rows)}.",
        f"Parse errors: {len(parse_errors)}.",
        "",
        "## Expected Comparisons",
        "",
    ]
    for expected in EXPECTED_COMPARISONS:
        items = external_group_for(external_rows, expected)
        status = "present" if items else "missing"
        lines.append(
            f"- {expected['baseline']} {friendly_dataset(expected['dataset'])} K={expected['k'] or 'offline'}: {status} ({len(items)} seed rows)."
        )
    if parse_errors:
        lines.extend(["", "## Parse Errors", ""])
        for row in parse_errors:
            lines.append(f"- `{row.get('source_file')}`: {row.get('parse_error')}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Paired tests require matching seeds and metric availability for AHR-MalCL-HR and the external baseline.",
            "- Joint is compared by dataset and seed only; ER and DERPP are compared by dataset, K, and seed.",
            "- Keep external outputs under `result_external_baselines/`; do not place them under trusted `result/`.",
            "",
        ]
    )
    (OUT_DIR / "external_baseline_missing_results.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_root: Path | None = None) -> dict[str, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = input_root or EXTERNAL_ROOT
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    external_rows, parse_errors = discover_external_runs(root)
    trusted_rows = trusted_target_rows()
    summary_rows = aggregate_external(external_rows)
    paired_rows = build_paired_tests(external_rows, trusted_rows)
    effect_rows = build_effect_sizes(paired_rows)
    availability_rows = build_metric_availability(external_rows, trusted_rows)

    write_csv(OUT_DIR / "external_baseline_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_summary_markdown(summary_rows, root)
    write_csv(OUT_DIR / "external_baseline_paired_tests.csv", paired_rows, PAIR_COLUMNS)
    write_csv(OUT_DIR / "external_baseline_stat_tests.csv", paired_rows, PAIR_COLUMNS)
    write_csv(OUT_DIR / "external_baseline_effect_sizes.csv", effect_rows, EFFECT_COLUMNS)
    write_csv(OUT_DIR / "external_baseline_metric_availability.csv", availability_rows, AVAILABILITY_COLUMNS)
    write_claim_safety(summary_rows, paired_rows, trusted_rows, external_rows, parse_errors, root)
    write_missing_report(external_rows, parse_errors, root)

    return {
        "trusted_rows": len(trusted_rows),
        "external_seed_rows": len(external_rows),
        "parse_errors": len(parse_errors),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
        "tested_pair_rows": sum(1 for row in paired_rows if row.get("test_status") == "tested"),
        "effect_rows": len(effect_rows),
        "availability_rows": len(availability_rows),
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
        "Trusted rows: {trusted_rows}; external seed rows: {external_seed_rows}; "
        "summary rows: {summary_rows}; paired rows: {paired_rows}; tested paired rows: {tested_pair_rows}; "
        "effect rows: {effect_rows}; availability rows: {availability_rows}; parse errors: {parse_errors}.".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
