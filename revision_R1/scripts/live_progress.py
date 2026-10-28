"""Read-only, bounded progress summaries for production orchestration.

This module deliberately does not import NumPy, PyTorch, datasets, or models.
It formats values already persisted in RunState and frozen plan records.
"""

from __future__ import annotations

import math
from typing import Any


_SOURCE_NAMES = ("current", "anchor", "generated")


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
    task_row = _latest_for_task(runtime.get("task_rows"), task_number)
    task_detail = _latest_for_task(runtime.get("task_details"), task_number)
    metrics = task_detail.get("metrics") if isinstance(task_detail, dict) else None

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
        "metrics": {"latest_task": _safe_mapping(metrics), "task_row": _safe_mapping(task_row)},
        "training_accuracy": {"status": "N/A", "reason": "not stored by the runner"},
        "resources": resources,
        "time": time_info,
        "checkpoint": checkpoint,
        "memory": _safe_mapping(memory_snapshot),
    }


def render_progress_line(record: dict[str, Any]) -> str:
    """Render one concise text-only dashboard update from an observer event."""
    progress = record.get("live_progress") or {}
    if not progress:
        elapsed = record.get("preparation_elapsed_seconds")
        elapsed_text = "" if elapsed is None else f" | preparation elapsed {float(elapsed):.1f}s"
        memory = record.get("memory", {})
        ram = memory.get("effective_available_bytes", memory.get("host_available_bytes")) if isinstance(memory, dict) else None
        ram_text = "" if ram is None else f" | host/cgroup available {ram} bytes"
        return f"{record.get('status', record.get('stage', 'progress'))} | {record.get('stage', '')} {record.get('error', '')}{elapsed_text}{ram_text}".strip()
    ident = progress.get("identity", {})
    task = progress.get("task", {})
    phase_progress = progress.get("phase_progress", {})
    fraction = "N/A"
    if phase_progress.get("total"):
        fraction = f"{phase_progress.get('completed')}/{phase_progress.get('total')} {phase_progress.get('unit')}"
    time_info = progress.get("time", {})
    eta = time_info.get("phase_eta_seconds")
    eta_text = "ETA N/A" if eta is None else f"ETA ~{float(eta) / 60:.1f} min (this attempt/phase)"
    losses = progress.get("losses", {})
    sampling = progress.get("source_sampling", {})
    metrics = progress.get("metrics", {}).get("latest_task", {})
    attempt = time_info.get("attempt_elapsed_seconds")
    dataset = {"ember": "EMBER", "az_class": "AZ-Class"}.get(ident.get("dataset"), ident.get("dataset", "N/A"))
    arm = {"current_only": "Current-only", "anchor_only": "Anchor-only",
           "generated_only": "Generated-only", "full_hybrid": "Full Hybrid",
           "derpp_matched": "DER++-matched"}.get(ident.get("arm"), ident.get("arm", "N/A"))
    human_phase = {
        "scaler": "data preparation/scaler", "gan_critic": "GAN training",
        "classifier": "classifier training", "evaluation": "task evaluation",
        "memory": "replay memory update", "generation": "replay generation",
        "selection": "replay selection", "refresh": "stored-logit refresh",
        "export": "final export",
    }.get(task.get("phase"), task.get("phase", "unknown"))
    gan = progress.get("gan", {})
    gan_label = ""
    if gan.get("status") == "N/A":
        gan_label = " | GAN: N/A (arm has no GAN component)"
    task_line = (
        f"Task {task.get('number')}/{task.get('count')} | new classes: {task.get('new_class_count')} | "
        f"seen classes: {task.get('seen_class_count')} | Phase: {human_phase}: {fraction} | {eta_text}{gan_label}"
    )
    if gan.get("status") == "available" and task.get("phase") == "gan_critic":
        task_line += (
            f" | Epoch {phase_progress.get('epoch_number')}/{phase_progress.get('epoch_count')}"
            f" | batch {phase_progress.get('batch_number')}/{phase_progress.get('batches_in_epoch')}"
            f" | generator updates {phase_progress.get('generator_updates_completed')}/"
            f"{phase_progress.get('generator_updates_total')}"
            f" | critic substep {phase_progress.get('critic_substep_completed')}/"
            f"{phase_progress.get('critic_substeps_per_batch')}"
        )
    matrix = progress.get("matrix", {})
    time_info = progress.get("time", {})
    resources = progress.get("resources", {})
    memory = resources.get("memory", {})
    checkpoint = progress.get("checkpoint", {})
    local_checkpoint = checkpoint.get("local_saved", {})
    elapsed = time_info.get("attempt_elapsed_seconds")
    cumulative = time_info.get("cumulative_active_seconds")
    return (
        f"Run {ident.get('run_index')}/{ident.get('run_count')} | {dataset} | {arm} | seed {ident.get('seed')}\n"
        f"{task_line}\n"
        f"Losses: classifier {losses.get('classifier_latest', 'N/A')} | generator {losses.get('generator_latest', 'N/A')} | critic {losses.get('critic_latest', 'N/A')} | gradient penalty {losses.get('gradient_penalty', {}).get('status', 'N/A')}\n"
        f"Source draws: {sampling.get('latest_draws', 'N/A')} | expected proportions: {sampling.get('expected_pool_proportions', 'N/A')} | "
        f"realized proportions: {sampling.get('latest_draw_proportions', 'N/A')} | metrics: {metrics or 'N/A'}\n"
        f"Time: this attempt {elapsed}s | cumulative active {cumulative}s | "
        f"observer overhead {resources.get('progress_observer_overhead_seconds', 'N/A')}s | "
        f"RAM available {memory.get('effective_available_bytes', memory.get('host_available_bytes', 'N/A'))} bytes | "
        f"GPU allocated {memory.get('gpu_allocated_bytes', 'N/A')} bytes\n"
        f"Local checkpoint: generation {local_checkpoint.get('generation', 'N/A')} | "
        f"Drive verified: generation {checkpoint.get('generation', 'N/A')} | "
        f"sync age {checkpoint.get('age_seconds', 'N/A')}s | writes this attempt {checkpoint.get('write_bytes_this_attempt', 'N/A')} bytes\n"
        f"Matrix: validated {matrix.get('completed', 'N/A')}/{matrix.get('total', 'N/A')} | "
        f"not yet validated {matrix.get('not_yet_validated', 'N/A')} | "
        f"failed count {matrix.get('failed_count', 'N/A')} ({matrix.get('failed_count_status', 'unknown')})"
    )


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
