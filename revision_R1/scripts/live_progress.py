"""Read-only, bounded progress summaries for production orchestration.

This module deliberately does not import NumPy, PyTorch, datasets, or models.
It formats values already persisted in RunState and frozen plan records.
"""

from __future__ import annotations

import math
import time
from typing import Any


_SOURCE_NAMES = ("current", "anchor", "generated")
DEBUG_LOGGING = False


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe_number(item())
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        number = _safe_number(item)
        if number is not None:
            result[name] = number
        elif isinstance(item, str) or item is None or isinstance(item, bool):
            result[name] = item
    return result


def _latest_for_task(rows: Any, task_id: int) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    return next((row for row in reversed(rows)
                 if isinstance(row, dict) and row.get("task_id") == task_id), None)


def _progress(cursor: Any, runtime: dict[str, Any], schedule_budget: int | None,
              arm: str, task_sizes: list[int]) -> dict[str, Any]:
    task_index = max(0, int(getattr(cursor, "task_index", 0)))
    phase = str(getattr(cursor, "phase", "unknown"))
    item_index = max(0, int(getattr(cursor, "item_index", 0)))
    substep_index = max(0, int(getattr(cursor, "substep_index", 0)))
    update_index = max(0, int(getattr(cursor, "update_index", 0)))

    if phase == "classifier" and schedule_budget is not None:
        completed = int(runtime.get("classifier_updates_this_task", update_index))
        return {"completed": min(completed, schedule_budget), "total": schedule_budget,
                "unit": "classifier_updates"}
    if phase == "gan_critic" and arm in {"generated_only", "full_hybrid"}:
        epochs = int(runtime.get("gan_epochs_for_run", 0))
        batches_per_epoch = int(runtime.get("gan_batches_per_epoch", 0))
        critic_steps = int(runtime.get("gan_critic_steps_for_run", 0))
        total_batches = epochs * batches_per_epoch
        if total_batches > 0 and critic_steps > 0:
            total_critic = total_batches * critic_steps
            completed_critic = min(total_critic, item_index * critic_steps + min(substep_index, critic_steps))
            completed_generator = min(total_batches, item_index)
            return {
                "completed": completed_critic, "total": total_critic, "unit": "critic_steps",
                "generator_updates_completed": completed_generator,
                "generator_updates_total": total_batches,
                "epoch_number": min(epochs, item_index // batches_per_epoch + 1),
                "epoch_count": epochs,
                "batch_number": item_index % batches_per_epoch + 1,
                "batches_in_epoch": batches_per_epoch,
                "critic_substep_completed": substep_index,
                "critic_substeps_per_batch": critic_steps,
            }
    if phase == "generation":
        total = len(runtime.get("generation_classes", []))
        if total:
            return {"completed": min(item_index, total), "total": total, "unit": "classes"}
    if phase == "selection":
        total = len(runtime.get("generation_classes", []))
        if total:
            completed = item_index + (1 if substep_index > 0 else 0)
            return {"completed": min(completed, total), "total": total, "unit": "classes"}
    return {
        "completed": None, "total": None, "unit": None,
        "status": "not_quantifiable_from_persisted_cursor",
        "reason": f"the {phase} phase has no persisted substep counter",
    }


def format_live_progress_event(
    event: dict[str, Any], state: Any, *, attempt_elapsed_seconds: float,
    matrix_counts: dict[str, Any], memory_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact JSON-safe view without changing or probing training state."""
    runtime = state.extra.get("runtime", {}) if isinstance(state.extra, dict) else {}
    cursor = state.cursor
    task_index = max(0, int(cursor.task_index))
    phase = str(cursor.phase)
    run_key = str(event.get("run_key", "unknown"))
    run_index = int(event.get("run_index", 0))
    task_sizes = [int(value) for value in event.get("task_sizes", [50] + [5] * 10)]
    task_count = len(task_sizes)
    task_number = min(task_count, task_index + 1)
    update_budgets = event.get("update_budgets_by_task", [])
    schedule_budget = None
    if isinstance(update_budgets, list) and task_number > 0 and len(update_budgets) >= task_number:
        schedule_budget = int(update_budgets[task_number - 1])

    class_order = runtime.get("class_order", event.get("class_order", []))
    task_start = sum(task_sizes[:max(0, task_number - 1)])
    task_end = task_start + (task_sizes[task_number - 1] if task_number else 0)
    task_classes = class_order[task_start:task_end] if isinstance(class_order, list) else []
    seen_classes = class_order[:task_end] if isinstance(class_order, list) else []

    tracker = state.resources.get("tracker", {}) if isinstance(state.resources, dict) else {}
    tracker = tracker if isinstance(tracker, dict) else {}
    current_task_updates = int(runtime.get("classifier_updates_this_task", 0))
    latest_update = _latest_for_task(runtime.get("update_trace"), task_number)
    gan_rows = runtime.get("gan_loss_trace", [])
    losses: dict[str, Any] = {
        "classifier_latest": None if latest_update is None else _safe_number(latest_update.get("loss")),
        "classifier_terms_mean_this_task": {},
        "generator_latest": None,
        "critic_latest": None,
        "gradient_penalty": {"status": "N/A", "reason": "the runner records total critic loss, not a separate gradient-penalty trace"},
    }
    terms = runtime.get("loss_terms_this_task", {})
    if isinstance(terms, dict):
        for name, values in terms.items():
            if isinstance(values, list) and values:
                numeric = [float(value) for value in values if _safe_number(value) is not None]
                if numeric:
                    losses["classifier_terms_mean_this_task"][str(name)] = sum(numeric) / len(numeric)
    if isinstance(gan_rows, list):
        for kind in ("generator", "critic"):
            latest_gan = next((row for row in reversed(gan_rows)
                               if isinstance(row, dict) and row.get("task_id") == task_number
                               and row.get("kind") == kind), None)
            if latest_gan is not None:
                losses[f"{kind}_latest"] = _safe_number(latest_gan.get("loss"))
    # These are completed evaluations, not necessarily the cursor's current task.
    evaluation_ids = [row["task_id"] for name in ("task_rows", "task_details")
                      for row in runtime.get(name, [])
                      if isinstance(row, dict) and isinstance(row.get("task_id"), int)
                      and row["task_id"] <= task_number]
    evaluation_task = max(evaluation_ids, default=None)
    task_row = _latest_for_task(runtime.get("task_rows"), evaluation_task)
    task_detail = _latest_for_task(runtime.get("task_details"), evaluation_task)
    metrics = task_detail.get("metrics") if isinstance(task_detail, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    metrics = dict(metrics)
    for source, target in (("seen_accuracy", "accuracy"), ("macro_f1", "macro_f1"),
                           ("weighted_f1", "weighted_f1"), ("balanced_accuracy", "balanced_accuracy")):
        if target not in metrics and task_row and source in task_row:
            metrics[target] = task_row[source]

    arm = str(event.get("arm", "unknown"))
    gan = {"status": "N/A", "reason": "arm has no GAN component"}
    phase_progress = _progress(cursor, runtime, schedule_budget, arm, task_sizes)
    if arm in {"generated_only", "full_hybrid"}:
        epochs = int(runtime.get("gan_epochs_for_run", 0))
        batches = int(runtime.get("gan_batches_per_epoch", 0))
        critic_steps = int(runtime.get("gan_critic_steps_for_run", 0))
        total_batches = epochs * batches
        gan = {
            "status": "available" if total_batches and critic_steps else "awaiting_task_setup",
            "epochs": epochs or None, "batches_per_epoch": batches or None,
            "total_batches": total_batches or None,
            "critic_steps_per_batch": critic_steps or None,
            "critic_steps_completed_this_task": phase_progress.get("completed") if phase == "gan_critic" else None,
            "critic_steps_total_this_task": phase_progress.get("total") if phase == "gan_critic" else (total_batches * critic_steps or None),
            "generator_updates_completed_this_task": phase_progress.get("generator_updates_completed") if phase == "gan_critic" else None,
            "generator_updates_total_this_task": phase_progress.get("generator_updates_total") if phase == "gan_critic" else (total_batches or None),
            "epoch_number": phase_progress.get("epoch_number") if phase == "gan_critic" else None,
            "epoch_count": phase_progress.get("epoch_count") if phase == "gan_critic" else (epochs or None),
            "batch_number": phase_progress.get("batch_number") if phase == "gan_critic" else None,
            "batches_in_epoch": phase_progress.get("batches_in_epoch") if phase == "gan_critic" else (batches or None),
            "critic_substep_completed": phase_progress.get("critic_substep_completed") if phase == "gan_critic" else None,
            "critic_substeps_per_batch": phase_progress.get("critic_substeps_per_batch") if phase == "gan_critic" else (critic_steps or None),
            "pool_source_counts": _safe_mapping(runtime.get("gan_pool_sources")),
        }

    sampler_state = state.sampler_state if isinstance(getattr(state, "sampler_state", None), dict) else {}
    realized = sampler_state.get("realized_source_counts", {})
    latest_draws = latest_update.get("source_draws", {}) if latest_update else {}
    expected = _safe_mapping(runtime.get("expected_source_proportions"))
    pool_sizes = _safe_mapping(runtime.get("source_pool_sizes"))
    realized_counts = _safe_mapping(realized)
    realized_total = sum(float(realized_counts.get(name, 0)) for name in _SOURCE_NAMES)
    latest_draw_counts = _safe_mapping(latest_draws)
    latest_draw_total = sum(float(latest_draw_counts.get(name, 0)) for name in _SOURCE_NAMES)
    source_sampling = {
        "expected_pool_proportions": expected or None,
        "unique_pool_sizes": pool_sizes or None,
        "realized_draw_counts_this_task": realized_counts or None,
        "realized_proportions_this_task": ({name: realized_counts.get(name, 0) / realized_total for name in _SOURCE_NAMES} if realized_total else None),
        "latest_draws": latest_draw_counts or None,
        "latest_draw_proportions": ({name: latest_draw_counts.get(name, 0) / latest_draw_total for name in _SOURCE_NAMES} if latest_draw_total else None),
        "replacement": True,
        "sampler": "mixed_pool_inverse_class_frequency_with_replacement",
    }

    cumulative = _safe_number(tracker.get("active_seconds"))
    time_info = {
        "attempt_elapsed_seconds": float(attempt_elapsed_seconds),
        "cumulative_active_seconds": cumulative,
        "phase_eta_seconds": None,
        "phase_eta_status": "awaiting_observed_progress_within_this_attempt_and_phase",
    }
    resources = {
        "steps": _safe_mapping(tracker.get("steps")),
        "component_seconds": _safe_mapping(tracker.get("seconds")),
        "checkpoint_seconds": _safe_number(tracker.get("seconds", {}).get("checkpoint")) if isinstance(tracker.get("seconds"), dict) else None,
        "instrumentation_seconds": _safe_number(tracker.get("instrumentation_seconds")),
        "memory": _safe_mapping(memory_snapshot),
        "classifier_updates_cumulative": _safe_number(state.resources.get("classifier_updates")),
        "generator_updates_cumulative": _safe_number(state.resources.get("generator_updates")),
        "critic_steps_cumulative": _safe_number(state.resources.get("critic_steps")),
        "progress_observer_overhead_seconds": _safe_number(event.get("progress_observer_overhead_seconds")),
    }
    checkpoint = dict(event.get("last_verified_drive_checkpoint") or {})
    if checkpoint.get("verified_at_unix") is not None:
        checkpoint["age_seconds"] = max(0.0, float(event.get("now_unix", 0.0)) - float(checkpoint["verified_at_unix"]))
    checkpoint.update({
        "local_saved": dict(event.get("latest_local_checkpoint") or {}),
        "write_bytes_this_attempt": int(event.get("drive_checkpoint_write_bytes_this_attempt", 0)),
        "write_bytes_total": int(event.get("drive_checkpoint_write_bytes_total", 0)),
        "write_seconds_this_attempt": _safe_number(event.get("drive_checkpoint_write_seconds_this_attempt")),
    })

    return {
        "schema_version": 1,
        "status": str(event.get("status", "RUNNING")),
        "attempt_id": event.get("attempt_id"),
        "identity": {
            "plan_sha256": event.get("plan_sha256"),
            "run_key": run_key, "run_index": run_index,
            "run_count": int(event.get("run_count", 100)),
            "dataset": event.get("dataset"), "seed": event.get("seed"), "arm": arm,
            "config_sha256": event.get("config_sha256", getattr(state, "identity", {}).get("config_sha256")),
            "training_source_sha256": event.get("training_source_sha256", getattr(state, "identity", {}).get("code_sha256")),
            "schedule_sha256": event.get("schedule_sha256", getattr(state, "identity", {}).get("schedule_sha256")),
        },
        "matrix": _safe_mapping(matrix_counts),
        "task": {
            "number": task_number, "count": task_count, "phase": phase,
            "classes_new": task_classes, "classes_seen": seen_classes,
            "new_class_count": len(task_classes), "seen_class_count": len(seen_classes),
        },
        "cursor": {"task_index": task_index, "phase": phase,
                   "item_index": int(cursor.item_index), "substep_index": int(cursor.substep_index),
                   "update_index": int(cursor.update_index)},
        "phase_progress": phase_progress,
        "classifier": {
            "task_updates_completed": current_task_updates,
            "task_update_budget": schedule_budget,
            "task_updates_total": schedule_budget,
            "updates_cumulative": int(state.resources.get("classifier_updates", 0)),
            "expected_loss": None if latest_update is None else _safe_number(latest_update.get("loss")),
        },
        "gan": gan,
        "losses": losses,
        "source_sampling": source_sampling,
        "metrics": {"latest_task": _safe_mapping(metrics), "task_row": _safe_mapping(task_row),
                    "task_id": evaluation_task,
                    "classes_seen": ((task_row or {}).get("classes_seen")
                                     if (task_row or {}).get("classes_seen") is not None
                                     else (sum(task_sizes[:evaluation_task]) if evaluation_task else None)),
                    "scope": "all seen test classes"},
        "training_accuracy": {"status": "N/A", "reason": "not stored by the runner"},
        "resources": resources,
        "time": time_info,
        "checkpoint": checkpoint,
        "memory": _safe_mapping(memory_snapshot),
    }


def _duration(value: Any) -> str:
    value = _safe_number(value)
    if value is None:
        return "N/A"
    seconds = max(0, int(value))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _bytes(value: Any) -> str:
    value = _safe_number(value)
    if value is None:
        return "N/A"
    unit, scale = ("GiB", 1024 ** 3) if value >= 1024 ** 3 else ("MiB", 1024 ** 2)
    return f"{value / scale:.1f} {unit}"


def _value(value: Any, *, percent: bool = False) -> str:
    number = _safe_number(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%" if percent else f"{number:.4g}"


def _pairs(values: Any, *, percent: bool = False) -> str:
    if not isinstance(values, dict):
        return "N/A"
    return ", ".join(f"{key}: {_value(value, percent=percent)}"
                     for key, value in values.items() if _safe_number(value) is not None) or "N/A"


def _upload_text(record: dict[str, Any], progress: dict[str, Any]) -> str:
    upload = record.get("checkpoint_upload") or progress.get("checkpoint", {}).get("upload")
    if not isinstance(upload, dict):
        return ""
    uploaded, total = _safe_number(upload.get("uploaded_bytes")), _safe_number(upload.get("total_bytes"))
    percentage = _value(uploaded / total, percent=True) if uploaded is not None and total and total > 0 else "N/A"
    text = (f"Checkpoint upload: {record.get('status', upload.get('status', 'unknown'))} | "
            f"{upload.get('status', 'unknown')} | generation {upload.get('generation', 'N/A')} | "
            f"{upload.get('filename', 'N/A')} | {percentage} | {_bytes(uploaded)}/{_bytes(total)}")
    for key in ("error_type", "http_status"):
        if upload.get(key) is not None:
            text += f" | {key}: {upload[key]}"
    return text


def render_progress_line(record: dict[str, Any]) -> str:
    """Render a latest-state panel as actual multiline plain text."""
    progress = record.get("live_progress") or {}
    status = str(record.get("status", progress.get("status", "RUNNING")))
    stage = str(record.get("stage", ""))
    error = record.get("error") or record.get("reason")
    header = f"{status} | {stage}".rstrip(" |")
    if isinstance(error, str):
        header += f" | {error}"
    if not progress:
        elapsed = record.get("preparation_elapsed_seconds")
        if elapsed is not None:
            header += f" | elapsed {_duration(elapsed)}"
        memory = record.get("memory") or {}
        available = memory.get("effective_available_bytes", memory.get("host_available_bytes"))
        if available is not None:
            header += f" | RAM available {_bytes(available)}"
        upload = _upload_text(record, progress)
        return header + ("\n" + upload if upload else "")
    ident, task = progress.get("identity", {}), progress.get("task", {})
    phase = task.get("phase")
    phase_name = {"classifier": "classifier training", "gan_critic": "GAN training",
                  "evaluation": "task evaluation"}.get(phase, phase)
    lines = [header,
             f"Run {ident.get('run_index')}/{ident.get('run_count')} | {ident.get('dataset')} | "
             f"{ident.get('arm')} | seed {ident.get('seed')}",
             f"Task {task.get('number')}/{task.get('count')} | new classes {task.get('new_class_count')} | "
             f"seen classes {task.get('seen_class_count')} | {phase_name}"]
    classifier = progress.get("classifier", {})
    completed = _safe_number(classifier.get("task_updates_completed"))
    total = _safe_number(classifier.get("task_updates_total"))
    percentage = f" ({100 * completed / total:.1f}%)" if completed is not None and total is not None and total > 0 else ""
    lines.append(f"Classifier updates {classifier.get('task_updates_completed')}/"
                 f"{classifier.get('task_updates_total')}{percentage} | batch 256 | latest classifier loss "
                 f"{_value(progress.get('losses', {}).get('classifier_latest'))}")
    phase_progress = progress.get("phase_progress", {})
    if ident.get("arm") in {"full_hybrid", "generated_only"}:
        gan = progress.get("gan", {})
        lines.append(f"GAN epoch {gan.get('epoch_number')}/{gan.get('epoch_count')} | "
                     f"batch {gan.get('batch_number')}/{gan.get('batches_in_epoch')} | "
                     f"critic steps {gan.get('critic_steps_completed_this_task')}/"
                     f"{gan.get('critic_steps_total_this_task')} | generator updates "
                     f"{gan.get('generator_updates_completed_this_task')}/{gan.get('generator_updates_total_this_task')}")
        losses = progress.get("losses", {})
        lines.append(f"Latest generator loss {_value(losses.get('generator_latest'))} | "
                     f"latest critic loss {_value(losses.get('critic_latest'))}")
    elif phase_progress.get("total") is not None and phase != "classifier":
        lines.append(f"Progress {phase_progress.get('completed')}/{phase_progress.get('total')} "
                     f"{phase_progress.get('unit')}")
    sampling = progress.get("source_sampling", {})
    lines.append("Source draws: " + _pairs(sampling.get("latest_draws")) +
                 " | realized: " + _pairs(sampling.get("latest_draw_proportions"), percent=True))
    evaluation = progress.get("metrics", {})
    metrics = evaluation.get("latest_task", {})
    labels = (("accuracy", "accuracy"), ("macro_f1", "macro F1"),
              ("weighted_f1", "weighted F1"), ("balanced_accuracy", "balanced accuracy"))
    available_metrics = [f"{label} {_value(metrics[key], percent=True)}"
                         for key, label in labels if _safe_number(metrics.get(key)) is not None]
    if available_metrics:
        lines.append(f"Latest completed evaluation: task {evaluation.get('task_id')} | "
                     f"{evaluation.get('scope', 'all seen test classes')} | "
                     f"classes {evaluation.get('classes_seen', 'N/A')} | " + " | ".join(available_metrics))
    else:
        lines.append("Latest completed evaluation: N/A")
    timing, resources = progress.get("time", {}), progress.get("resources", {})
    memory = resources.get("memory", progress.get("memory", {}))
    eta = timing.get("phase_eta_seconds")
    eta_text = _duration(eta) if _safe_number(eta) is not None else "estimating"
    lines.append(f"Time: attempt {_duration(timing.get('attempt_elapsed_seconds'))} | active "
                 f"{_duration(timing.get('cumulative_active_seconds'))} | Phase ETA {eta_text}")
    lines.append(f"RAM available {_bytes(memory.get('effective_available_bytes', memory.get('host_available_bytes')))} | "
                 f"GPU allocated {_bytes(memory.get('gpu_allocated_bytes'))}")
    checkpoint = progress.get("checkpoint", {})
    lines.append(f"Checkpoint: local generation {checkpoint.get('local_saved', {}).get('generation', 'N/A')} | "
                 f"Drive verified generation {checkpoint.get('generation', 'N/A')} | "
                 f"sync age {_duration(checkpoint.get('age_seconds'))} | "
                 f"writes {_bytes(checkpoint.get('write_bytes_this_attempt'))}")
    matrix = progress.get("matrix", {})
    lines.append(f"Matrix validated {matrix.get('completed', 'N/A')}/{matrix.get('total', 'N/A')}")
    upload = _upload_text(record, progress)
    if upload:
        lines.append(upload)
    return "\n".join(lines)


class ColabProgressDisplay:
    """Callable observer for records containing live_progress.

    Inject display_fn/update_fn with IPython display/update_display signatures,
    print_fn and a monotonic clock. Pass display_fn=False for plain text output.
    An injected display_fn without update_fn also uses compact plain text output.
    No records or panel history are retained.
    """

    def __init__(self, *, display_fn=None, update_fn=None, print_fn=print,
                 clock=time.monotonic, interval_seconds=7.0, debug_logging=DEBUG_LOGGING):
        if not 5 <= interval_seconds <= 7:
            raise ValueError("display interval must be between 5 and 7 seconds")
        if display_fn is None:
            try:
                from IPython import get_ipython
                from IPython.display import display, update_display
                if get_ipython() is not None:
                    display_fn, update_fn = display, update_display
            except ImportError:
                pass
        self.display_fn, self.update_fn = display_fn, update_fn
        self.print_fn, self.clock = print_fn, clock
        self.debug_logging = debug_logging
        self.throttle = ProgressThrottle(interval_seconds)
        self.latest_text = ""
        self._display_id = None
        # Task completion keys are bounded to the current run.
        self._run_key = None
        self._summaries = set()

    def __call__(self, record: dict[str, Any]) -> None:
        progress = record.get("live_progress") or {}
        ident, task = progress.get("identity", {}), progress.get("task", {})
        run_key = record.get("run_key", ident.get("run_key"))
        if run_key is not None and run_key != self._run_key:
            self._run_key = run_key
            self._summaries.clear()
        status = str(record.get("status", progress.get("status", "RUNNING")))
        stage = str(record.get("stage", ""))
        marker = (status + " " + stage).lower()
        completion = (any(word in marker for word in ("task_complete", "run_complete"))
                      or status in {"FIRST_COUNTED_PRODUCTION_RUN_VALIDATED",
                                    "SYNTHETIC_VALIDATED_NON_REPORTABLE", "VALIDATED",
                                    "COMPLETED_VALIDATED"})
        important = completion or status.startswith("CHECKPOINT_") or any(
            word in marker for word in ("fail", "error", "interrupt", "blocked"))
        if not progress and not important and not self.debug_logging:
            return
        upload = record.get("checkpoint_upload") or progress.get("checkpoint", {}).get("upload") or {}
        signature = (run_key, task.get("number"), task.get("phase"), status,
                     upload.get("generation"), upload.get("filename"), upload.get("status"))
        due = self.throttle.should_emit(signature, self.clock())
        if completion:
            kind = "task" if "task_complet" in marker else "run"
            key = (kind, task.get("number") if kind == "task" else None)
            if key not in self._summaries:
                self._summaries.add(key)
                summary = f"Completed run {run_key}"
                if kind == "task":
                    summary = f"Completed task {task.get('number')} | run {run_key}"
                    evaluation = progress.get("metrics", {})
                    if evaluation.get("task_id") == task.get("number"):
                        metrics = evaluation.get("latest_task", {})
                        measured = [f"{label} {_value(metrics[name], percent=True)}"
                                    for name, label in (("accuracy", "accuracy"), ("macro_f1", "macro F1"))
                                    if _safe_number(metrics.get(name)) is not None]
                        if measured:
                            summary += " | all seen test classes | " + " | ".join(measured)
                self.print_fn(f"{summary} | {status}")
        if not due:
            return
        self.latest_text = render_progress_line(record)
        if self.display_fn and self.update_fn:
            try:
                mime = {"text/plain": self.latest_text}
                if self._display_id is None:
                    handle = self.display_fn(mime, raw=True, display_id=True)
                    self._display_id = getattr(handle, "display_id", handle)
                    if self._display_id is None:
                        self.display_fn = False
                else:
                    self.update_fn(mime, raw=True, display_id=self._display_id)
                return
            except Exception:
                self.display_fn = False
        # A compact current snapshot; never replay past panels in a terminal.
        compact = " | ".join(self.latest_text.splitlines()[:3])
        upload = _upload_text(record, progress)
        if upload and upload not in compact:
            compact += " | " + upload
        self.print_fn(compact)

class BoundedProgressBuffer:
    """Keep only the most recent few rendered lines for a single notebook display."""

    def __init__(self, max_events: int = 5):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.max_events = max_events
        self._lines: list[str] = []

    def append(self, record: dict[str, Any]) -> str:
        self._lines.append(render_progress_line(record))
        if len(self._lines) > self.max_events:
            del self._lines[:-self.max_events]
        return "\n\n".join(self._lines)


class ProgressThrottle:
    """Emit on initial state, phase/task transition, and bounded intervals."""

    def __init__(self, interval_seconds: float = 12.0):
        if not math.isfinite(float(interval_seconds)) or interval_seconds <= 0:
            raise ValueError("progress interval must be a finite positive number")
        self.interval_seconds = float(interval_seconds)
        self._last_signature: Any = None
        self._last_emit: float | None = None

    def should_emit(self, signature: Any, now: float) -> bool:
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("progress clock must be finite")
        transition = self._last_emit is None or signature != self._last_signature
        due = self._last_emit is None or now - self._last_emit >= self.interval_seconds
        if transition or due:
            self._last_signature = signature
            self._last_emit = now
            return True
        return False


class PhaseRateEstimator:
    """Estimate remaining work only from observed progress in this exact phase."""

    def __init__(self, *, stale_after_seconds: float = 120.0):
        if stale_after_seconds <= 0:
            raise ValueError("stale interval must be positive")
        self.stale_after_seconds = float(stale_after_seconds)
        self._signature: Any = None
        self._anchor: tuple[float, float] | None = None
        self._last_observed_progress_at: float | None = None
        self._rate: float | None = None

    def estimate(self, signature: Any, completed: float | None, total: float | None, *, now: float) -> float | None:
        now = float(now)
        if not math.isfinite(now):
            return None
        if signature != self._signature:
            self._signature = signature
            self._anchor = None
            self._last_observed_progress_at = None
            self._rate = None
        if completed is None or total is None:
            return None
        completed, total = float(completed), float(total)
        if not math.isfinite(completed) or not math.isfinite(total) or total <= completed:
            return None
        if self._anchor is None:
            self._anchor = (completed, now)
            return None
        previous_completed, previous_time = self._anchor
        if completed > previous_completed and now > previous_time:
            self._rate = (completed - previous_completed) / (now - previous_time)
            self._anchor = (completed, now)
            self._last_observed_progress_at = now
        if self._rate is None or self._last_observed_progress_at is None:
            return None
        if now - self._last_observed_progress_at > self.stale_after_seconds:
            return None
        return max(0.0, (total - completed) / self._rate)
