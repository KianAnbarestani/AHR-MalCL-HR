#!/usr/bin/env python3
"""Shared utilities for safe AHR-MalCL result parsing and output writing."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from stat_validation import config


VERSION = "v1 - statistical validation common helpers"


def ensure_output_dirs() -> None:
    for directory in config.OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_")


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def safe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def is_missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def canonical_dataset(*candidates: Any) -> str | None:
    haystack = " ".join(str(c) for c in candidates if c is not None)
    normalized = compact(haystack)
    if "azclass" in normalized or re.search(r"\baz\b", haystack.lower()):
        return "az_class"
    if "ember" in normalized:
        return "ember"
    return None


def canonical_method(*candidates: Any) -> str | None:
    haystack = " ".join(str(c) for c in candidates if c is not None)
    normalized = compact(haystack)
    best: tuple[int, str] | None = None
    for method, aliases in config.METHOD_ALIASES.items():
        for alias in aliases:
            alias_key = compact(alias)
            if alias_key and alias_key in normalized:
                score = len(alias_key)
                if best is None or score > best[0]:
                    best = (score, method)
    if best:
        return best[1]
    if "replaydriftk0noanchor" in normalized:
        return "replay_drift_k0_no_anchor"
    if "replaydriftk25anchor" in normalized:
        return "replay_drift_k25_anchor"
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            cleaned = slug(candidate)
            if cleaned:
                return cleaned
    return None


def infer_setting_type(path: Path, context_labels: Iterable[Any] = ()) -> str:
    text = compact(" ".join([path.as_posix(), *[str(x) for x in context_labels]]))
    if "bestperformancesetting" in text:
        return "final_best_performance"
    if "memoryefficientsetting" in text:
        return "final_memory_efficient"
    if "ablation" in text or "ablations" in text:
        return "ablation"
    if "memorybudget" in text:
        return "memory_budget"
    if "replydrift" in text or "replaydrift" in text:
        return "replay_drift"
    if "oraclefidelity" in text:
        return "oracle_fidelity"
    if "orderingsensitivity" in text or "/ordering/" in path.as_posix().lower():
        return "ordering_sensitivity"
    if "scalersensitivity" in text:
        return "scaler_sensitivity"
    return "other"


def infer_k(path: Path, *candidates: Any) -> int | None:
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("real_buffer_k_per_class", "replay_k_per_class", "k"):
                value = safe_int(candidate.get(key))
                if value is not None:
                    return value
        else:
            value = safe_int(candidate)
            if value is not None:
                return value
    text = path.as_posix()
    patterns = [
        r"(?i)(?:^|[/_\-\s])k\s*=\s*(\d+)(?:$|[/_\-\s])",
        r"(?i)(?:^|[/_\-\s])k(\d+)(?:$|[/_\-\s])",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return int(matches[-1])
    return None


def load_json_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def first_present(source: dict[str, Any], aliases: Iterable[str]) -> float | None:
    for key in aliases:
        value = safe_float(source.get(key))
        if value is not None:
            return value
    return None


def replay_quality_metric(run: dict[str, Any], metric: str) -> float | None:
    aliases = config.METRIC_ALIASES.get(metric, [metric])
    replay_quality = run.get("replay_quality")
    if not isinstance(replay_quality, list):
        return None
    values: list[float] = []
    for item in replay_quality:
        if isinstance(item, dict):
            value = first_present(item, aliases)
            if value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def extract_metric(source: dict[str, Any], metric: str, aggregate: bool = False) -> float | None:
    if metric == "final_taskwise_average_accuracy":
        priority = ["final_taskwise_average_accuracy"]
        if not aggregate:
            nested = source.get("cl_metrics")
            if isinstance(nested, dict):
                value = safe_float(nested.get("final_average_accuracy_taskwise"))
                if value is not None:
                    return value
        priority.append("final_taskwise_average_accuracy_mean")
        value = first_present(source, priority)
        if value is not None:
            return value
    if metric == "forgetting":
        priority = ["forgetting"]
        if not aggregate:
            nested = source.get("cl_metrics")
            if isinstance(nested, dict):
                value = safe_float(nested.get("forgetting"))
                if value is not None:
                    return value
        priority.append("forgetting_mean")
        value = first_present(source, priority)
        if value is not None:
            return value

    aliases = config.METRIC_ALIASES.get(metric, [metric])
    value = first_present(source, aliases)
    if value is not None:
        return value
    for nested_key in ("cl_metrics", "aggregate", "task_metrics", "metrics"):
        nested = source.get(nested_key)
        if isinstance(nested, dict):
            value = first_present(nested, aliases)
            if value is not None:
                return value
    return replay_quality_metric(source, metric)


def json_dumps(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def update_context_for_key(context: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    child = dict(context)
    labels = list(child.get("labels", []))
    labels.append(key)
    child["labels"] = labels
    dataset = canonical_dataset(key)
    if dataset:
        child["dataset"] = dataset
    key_k = infer_k(Path(key), key)
    if key_k is not None:
        child["k"] = key_k
    method = canonical_method(key)
    if method and method not in {"ember", "az_class"}:
        child.setdefault("method", method)
    if isinstance(value, dict):
        config_block = value.get("config")
        if isinstance(config_block, dict):
            child["config"] = config_block
        if "real_buffer_k_per_class" in value:
            child["k"] = safe_int(value.get("real_buffer_k_per_class"))
    return child


def iter_result_blocks(
    obj: Any,
    path: Path,
    context: dict[str, Any] | None = None,
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    if context is None:
        context = {"labels": []}
    if isinstance(obj, dict):
        has_block = isinstance(obj.get("aggregate"), dict) or isinstance(obj.get("runs"), list)
        if has_block:
            yield "block", obj, context
            return
        is_single_run = (
            "seed" in obj
            and ("method" in obj or "mean_acc_seen" in obj)
            and not isinstance(obj.get("runs"), list)
        )
        if is_single_run:
            yield "run", obj, context
            return
        for key, value in obj.items():
            if key in {"acc_matrix_taskwise", "per_class_accuracy", "per_class_f1"}:
                continue
            yield from iter_result_blocks(value, path, update_context_for_key(context, key, value))
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_result_blocks(item, path, context)


def row_base(path: Path, context: dict[str, Any], source_kind: str) -> dict[str, Any]:
    labels = context.get("labels", [])
    return {
        "source_file": rel_path(path),
        "source_kind": source_kind,
        "setting_type": infer_setting_type(path, labels),
        "context_label": "/".join(str(x) for x in labels),
    }


def config_values(run: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    cfg = run.get("config")
    if not isinstance(cfg, dict):
        cfg = context.get("config") if isinstance(context.get("config"), dict) else {}
    return {
        "use_diversity_buffer": cfg.get("use_diversity_buffer"),
        "use_kd": cfg.get("use_kd"),
        "use_proto_align": cfg.get("use_proto_align"),
        "gan_train_on_real_buffer": cfg.get("gan_train_on_real_buffer"),
        "use_wgan_gp": cfg.get("use_wgan_gp"),
        "use_projection_critic": cfg.get("use_projection_critic"),
        "main_method_variant": cfg.get("main_method_variant"),
        "classifier_backbone": cfg.get("classifier_backbone"),
        "clf_optimizer": cfg.get("clf_optimizer"),
        "scaler_mode": cfg.get("scaler_mode"),
        "buffer_update_mode": cfg.get("buffer_update_mode"),
    }


def source_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in ("source_file", "source_files", "context_label", "method_raw")
    ).lower()


def has_final_hr_flags(row: dict[str, Any]) -> bool:
    return (
        safe_bool(row.get("use_diversity_buffer")) is False
        and safe_bool(row.get("use_kd")) is False
        and safe_bool(row.get("use_proto_align")) is False
        and safe_bool(row.get("gan_train_on_real_buffer")) is True
    )


def has_full_complexity_flags(row: dict[str, Any]) -> bool:
    return any(
        safe_bool(row.get(flag)) is True
        for flag in ("use_diversity_buffer", "use_kd", "use_proto_align")
    )


def is_exact_final_best_path(row: dict[str, Any]) -> bool:
    text = source_text(row)
    return (
        "final-run/ember/best-performance setting/k=100/ahr_malcl_hr_hybrid_random_buffer_ember_k100_results.json"
        in text
        or "final-run/az-class/best-performance setting/k=200/ahr_malcl_hr_hybrid_random_buffer_az_k200_results.json"
        in text
    )


def is_hybrid_random_ablation_memory_efficient_source(row: dict[str, Any]) -> bool:
    text = source_text(row)
    return (
        "final-run/ember/ablations/hybrid-random-buffer/ahr_malcl_v15_1_ablation_critic5_hybrid_random_buffer_results.json"
        in text
        or "final-run/az-class/ablation/hybrid_random_buffer/ahr_malcl_v15_1_final_ablation_critic5_az_k100_hybrid_random_buffer_results.json"
        in text
    )


def classify_result(row: dict[str, Any]) -> dict[str, str]:
    dataset = row.get("dataset")
    method = row.get("method")
    setting = row.get("setting_type")
    k = safe_int(row.get("k"))
    text = source_text(row)

    if "critic2" in text and setting == "final_memory_efficient":
        return {
            "result_validity": "excluded_incomplete",
            "paper_use_category": "exclude_from_paper_claims",
            "validity_reason": "stray incomplete EMBER critic2 memory-efficient seed file; not part of declared final result set",
        }

    if (
        is_exact_final_best_path(row)
        and setting == "final_best_performance"
        and method == config.MAIN_METHOD_CANONICAL
        and k == config.FINAL_BEST_K.get(str(dataset))
    ):
        if has_final_hr_flags(row) or row.get("source_kind") == "aggregate":
            return {
                "result_validity": "valid_main_hr",
                "paper_use_category": "main_final_table",
                "validity_reason": "official best-performance simplified AHR-MalCL-HR result",
            }
        return {
            "result_validity": "legacy_or_config_mismatch",
            "paper_use_category": "manual_review_needed",
            "validity_reason": "official best-performance path but required simplified HR config flags are missing or mismatched",
        }

    if (
        is_hybrid_random_ablation_memory_efficient_source(row)
        and setting == "ablation"
        and method == config.MAIN_METHOD_CANONICAL
        and k == config.FINAL_MEMORY_EFFICIENT_K.get(str(dataset))
    ):
        if has_final_hr_flags(row) or row.get("source_kind") == "aggregate":
            return {
                "result_validity": "valid_memory_efficient_hr",
                "paper_use_category": "memory_efficient_hr_table",
                "validity_reason": "ablation hybrid_random_buffer aggregate provides simplified HR memory-efficient result",
            }
        return {
            "result_validity": "unknown_review_needed",
            "paper_use_category": "manual_review_needed",
            "validity_reason": "hybrid_random_buffer ablation memory-efficient source has missing or mismatched simplified HR flags",
        }

    if setting == "final_memory_efficient":
        if has_full_complexity_flags(row):
            return {
                "result_validity": "legacy_or_config_mismatch",
                "paper_use_category": "supplementary_only",
                "validity_reason": "memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method",
            }
        return {
            "result_validity": "unknown_review_needed",
            "paper_use_category": "manual_review_needed",
            "validity_reason": "memory-efficient final file is not an official simplified HR main-result source",
        }

    if setting == "ablation":
        return {
            "result_validity": "valid_ablation",
            "paper_use_category": "ablation_table",
            "validity_reason": "controlled ablation result under final ablation folders",
        }

    if setting == "memory_budget":
        return {
            "result_validity": "valid_memory_budget",
            "paper_use_category": "memory_budget_table",
            "validity_reason": "older memory-budget result used only for memory-budget analysis",
        }

    if setting == "replay_drift":
        return {
            "result_validity": "valid_replay_drift",
            "paper_use_category": "replay_drift_table",
            "validity_reason": "replay-drift result used only for replay-drift analysis",
        }

    return {
        "result_validity": "unknown_review_needed",
        "paper_use_category": "manual_review_needed",
        "validity_reason": "not part of main final, ablation, memory-budget, or replay-drift evidence categories",
    }


def apply_validity_fields(row: dict[str, Any]) -> dict[str, Any]:
    row.update(classify_result(row))
    return row


def run_to_row(run: dict[str, Any], path: Path, context: dict[str, Any]) -> dict[str, Any]:
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    labels = " ".join(str(x) for x in context.get("labels", []))
    row = row_base(path, context, "run")
    dataset = canonical_dataset(run.get("dataset"), context.get("dataset"), path.as_posix(), labels)
    method = canonical_method(
        cfg.get("main_method_variant"),
        run.get("method"),
        context.get("method"),
        labels,
        path.as_posix(),
    )
    row.update(
        {
            "dataset": dataset,
            "method": method,
            "method_raw": run.get("method", ""),
            "seed": safe_int(run.get("seed")),
            "config_hash": run.get("config_hash", ""),
            "k": infer_k(path, cfg, run, context.get("k")),
            "early_stop_reason": run.get("early_stop_reason"),
            "task_count": len(run.get("accs_seen_per_task") or []),
            "accs_seen_per_task": json_dumps(run.get("accs_seen_per_task")),
            "acc_matrix_taskwise": json_dumps(run.get("acc_matrix_taskwise")),
        }
    )
    row.update(config_values(run, context))
    for metric in config.METRICS:
        row[metric] = extract_metric(run, metric, aggregate=False)
    return apply_validity_fields(row)


def aggregate_to_row(block: dict[str, Any], path: Path, context: dict[str, Any]) -> dict[str, Any]:
    aggregate = block.get("aggregate") if isinstance(block.get("aggregate"), dict) else {}
    runs = block.get("runs") if isinstance(block.get("runs"), list) else []
    first_run = next((x for x in runs if isinstance(x, dict)), {})
    cfg = first_run.get("config") if isinstance(first_run.get("config"), dict) else {}
    labels = " ".join(str(x) for x in context.get("labels", []))
    row = row_base(path, context, "aggregate")
    dataset = canonical_dataset(first_run.get("dataset"), context.get("dataset"), path.as_posix(), labels)
    method = canonical_method(
        cfg.get("main_method_variant"),
        first_run.get("method"),
        context.get("method"),
        labels,
        path.as_posix(),
    )
    row.update(
        {
            "dataset": dataset,
            "method": method,
            "method_raw": first_run.get("method", ""),
            "k": infer_k(path, cfg, first_run, context.get("k")),
            "runs": aggregate.get("runs", len(runs)),
            "seeds": json_dumps(
                sorted(
                    safe_int(run.get("seed"))
                    for run in runs
                    if isinstance(run, dict) and safe_int(run.get("seed")) is not None
                )
            ),
            "per_task_seen_mean": json_dumps(aggregate.get("per_task_seen_mean")),
            "per_task_seen_min": json_dumps(aggregate.get("per_task_seen_min")),
            "per_task_seen_max": json_dumps(aggregate.get("per_task_seen_max")),
        }
    )
    row.update(config_values(first_run, context))
    for metric in config.METRICS:
        row[metric] = extract_metric(aggregate, metric, aggregate=True)
    return apply_validity_fields(row)


def extract_rows_from_json(path: Path, data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for kind, obj, context in iter_result_blocks(data, path):
        if kind == "run":
            run_rows.append(run_to_row(obj, path, context))
        elif kind == "block":
            aggregate_rows.append(aggregate_to_row(obj, path, context))
            for run in obj.get("runs", []) or []:
                if isinstance(run, dict):
                    run_rows.append(run_to_row(run, path, context))
    return run_rows, aggregate_rows


def merge_duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("setting_type"),
            row.get("method"),
            row.get("k"),
            row.get("seed"),
            row.get("config_hash") or "",
        )
        existing = merged.get(key)
        if existing is None:
            new_row = dict(row)
            new_row["source_files"] = row.get("source_file", "")
            new_row["duplicate_source_count"] = 1
            merged[key] = new_row
            continue
        for field, value in row.items():
            if is_missing(existing.get(field)) and not is_missing(value):
                existing[field] = value
        files = set(str(existing.get("source_files", "")).split(";"))
        files.add(str(row.get("source_file", "")))
        existing["source_files"] = ";".join(sorted(f for f in files if f))
        existing["duplicate_source_count"] = safe_int(existing.get("duplicate_source_count")) or 1
        existing["duplicate_source_count"] += 1
    return list(merged.values())


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> None:
    print(VERSION, flush=True)
    ensure_output_dirs()
    print(f"ready={config.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
