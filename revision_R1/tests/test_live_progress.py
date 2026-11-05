from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.live_progress import (
    BoundedProgressBuffer, PhaseRateEstimator, ProgressThrottle,
    format_live_progress_event,
)


def _state(*, arm="full_hybrid", task_index=2, phase="classifier"):
    runtime = {
        "class_order": list(range(100)),
        "new_classes": [10, 11, 12, 13, 14],
        "seen_classes": list(range(15)),
        "classifier_updates_this_task": 13,
        "source_pool_sizes": {"current": 250, "anchor": 100, "generated": 500},
        "expected_source_proportions": {"current": 0.3, "anchor": 0.2, "generated": 0.5},
        "update_trace": [{
            "task_id": task_index + 1, "update": 13,
            "source_draws": {"current": 75, "anchor": 51, "generated": 130},
            "loss": 0.42,
        }],
        "loss_terms_this_task": {"current_ce": [0.5, 0.4], "replay_ce": [0.3], "logit_mse": []},
        "gan_batches_per_epoch": 20,
        "gan_epochs_for_run": 3,
        "gan_critic_steps_for_run": 5,
        "gan_loss_trace": [{"task_id": task_index + 1, "kind": "critic", "loss": 1.25}],
        "generation_classes": [0, 1, 2, 3, 4],
        "task_rows": [{"task_id": task_index + 1, "seen_accuracy": 0.75, "macro_f1": 0.7}],
        "task_details": [{"task_id": task_index + 1, "metrics": {"accuracy": 0.75, "macro_f1": 0.7}}],
    }
    return SimpleNamespace(
        identity={"run_id": "run-1", "config_sha256": "c" * 64,
                  "schedule_sha256": "s" * 64, "code_sha256": "t" * 64},
        cursor=SimpleNamespace(task_index=task_index, phase=phase, item_index=7,
                               substep_index=2, update_index=13),
        extra={"runtime": runtime},
        resources={"classifier_updates": 88, "generator_updates": 40,
                   "critic_steps": 200, "tracker": {
                       "active_seconds": 321.0,
                       "steps": {"classifier": 88, "generator": 40, "critic": 200},
                       "seconds": {"classifier": 123.0, "generator": 456.0,
                                   "critic": 789.0, "evaluation": 12.0,
                                   "checkpoint": 34.0},
                       "peak_cuda_allocated_bytes": 1234,
                       "peak_cuda_reserved_bytes": 5678,
                       "instrumentation_seconds": 0.02,
                   }},
        metrics={"task_matrix": [[None, None], [0.75, None]]},
    )


@pytest.mark.parametrize("arm", ["current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched"])
def test_formatter_reports_frozen_run_identity_and_arm_specific_gan_status(arm):
    state = _state(arm=arm)
    event = {
        "run_key": f"ember_seed42_{arm}", "run_index": 1, "run_count": 100,
        "attempt_id": "attempt-1", "plan_sha256": "p" * 64, "arm": arm,
        "training_source_sha256": "t" * 64, "schedule_sha256": "s" * 64,
        "update_budgets_by_task": [100, 90, 80] + [70] * 8,
        "last_verified_drive_checkpoint": {"generation": 102, "payload_sha256": "a" * 64},
        "drive_checkpoint_write_bytes_this_attempt": 1024,
        "drive_checkpoint_write_bytes_total": 2048,
        "progress_observer_overhead_seconds": 0.75,
    }
    before = deepcopy(state.__dict__)

    result = format_live_progress_event(
        event, state, attempt_elapsed_seconds=23.5,
        matrix_counts={"completed": 0, "total": 100},
        memory_snapshot={"host_available_bytes": 99},
    )

    assert result["identity"]["arm"] == arm
    assert result["identity"]["run_key"] == f"ember_seed42_{arm}"
    assert result["time"]["attempt_elapsed_seconds"] == 23.5
    assert result["time"]["cumulative_active_seconds"] == 321.0
    assert result["classifier"]["task_update_budget"] == 80
    assert result["classifier"]["task_updates_completed"] == 13
    assert result["classifier"]["task_updates_total"] == 80
    assert result["source_sampling"]["expected_pool_proportions"] == {
        "current": 0.3, "anchor": 0.2, "generated": 0.5,
    }
    assert result["source_sampling"]["latest_draws"] == {
        "current": 75, "anchor": 51, "generated": 130,
    }
    assert result["source_sampling"]["latest_draw_proportions"]["generated"] == 130 / 256
    assert result["losses"]["classifier_latest"] == 0.42
    assert result["losses"]["gradient_penalty"]["status"] == "N/A"
    assert result["metrics"]["latest_task"] == {"accuracy": 0.75, "macro_f1": 0.7}
    assert result["checkpoint"]["generation"] == 102
    assert result["resources"]["progress_observer_overhead_seconds"] == 0.75
    assert result["matrix"] == {"completed": 0, "total": 100}
    assert state.__dict__ == before
    if arm in {"generated_only", "full_hybrid"}:
        assert result["gan"]["status"] == "available"
    else:
        assert result["gan"] == {"status": "N/A", "reason": "arm has no GAN component"}


def test_classifier_progress_uses_resumed_task_cursor_and_exact_schedule_budget():
    result = format_live_progress_event(
        {"run_key": "ember_seed42_full_hybrid", "update_budgets_by_task": [125, 100, 80] + [1] * 8},
        _state(task_index=2, phase="classifier"), attempt_elapsed_seconds=4,
        matrix_counts={"completed": 0, "total": 100}, memory_snapshot={},
    )
    assert result["task"]["number"] == 3
    assert result["phase_progress"] == {"completed": 13, "total": 80, "unit": "classifier_updates"}


def test_gan_progress_uses_epochs_batches_and_critic_substeps():
    state = _state(task_index=0, phase="gan_critic")
    state.cursor.item_index = 7
    state.cursor.substep_index = 2
    result = format_live_progress_event(
        {"run_key": "ember_seed42_full_hybrid", "arm": "full_hybrid",
         "update_budgets_by_task": [1] * 11}, state,
        attempt_elapsed_seconds=7, matrix_counts={}, memory_snapshot={},
    )
    assert result["gan"]["total_batches"] == 60
    assert result["gan"]["critic_steps_completed_this_task"] == 37
    assert result["gan"]["critic_steps_total_this_task"] == 300
    assert result["gan"]["generator_updates_completed_this_task"] == 7
    assert result["gan"]["generator_updates_total_this_task"] == 60
    assert result["gan"]["epoch_number"] == 1
    assert result["gan"]["batch_number"] == 8
    assert result["gan"]["critic_substep_completed"] == 2
    assert {key: result["phase_progress"][key] for key in ("completed", "total", "unit")} == {
        "completed": 37, "total": 300, "unit": "critic_steps",
    }


@pytest.mark.parametrize("phase", ["scaler", "evaluation", "memory", "refresh", "export"])
def test_unquantified_phases_are_explicitly_not_fabricated(phase):
    result = format_live_progress_event(
        {"run_key": "ember_seed42_anchor_only", "arm": "anchor_only",
         "update_budgets_by_task": [10] * 11}, _state(arm="anchor_only", phase=phase),
        attempt_elapsed_seconds=9, matrix_counts={}, memory_snapshot={},
    )
    assert result["phase_progress"]["total"] is None
    assert result["phase_progress"]["status"] == "not_quantifiable_from_persisted_cursor"
    assert result["training_accuracy"] == {"status": "N/A", "reason": "not stored by the runner"}


def test_generation_and_selection_denominators_use_class_cursor():
    for phase, substep, expected in (("generation", 0, 2), ("selection", 1, 3)):
        state = _state(phase=phase)
        state.cursor.item_index = 2
        state.cursor.substep_index = substep
        result = format_live_progress_event(
            {"run_key": "x", "update_budgets_by_task": [1] * 11}, state,
            attempt_elapsed_seconds=1, matrix_counts={}, memory_snapshot={},
        )
        assert result["phase_progress"] == {"completed": expected, "total": 5, "unit": "classes"}


def test_progress_throttle_emits_first_and_phase_changes_immediately():
    throttle = ProgressThrottle(interval_seconds=12)
    assert throttle.should_emit((1, "gan_critic"), 100.0)
    assert not throttle.should_emit((1, "gan_critic"), 111.9)
    assert throttle.should_emit((1, "gan_critic"), 112.0)
    assert throttle.should_emit((1, "classifier"), 113.0)


def test_phase_rate_estimator_never_carries_rate_across_phase_or_arm():
    estimator = PhaseRateEstimator()
    assert estimator.estimate(("full_hybrid", 1, "gan_critic"), 1, 10, now=100) is None
    eta = estimator.estimate(("full_hybrid", 1, "gan_critic"), 11, 20, now=110)
    assert eta == 9.0
    assert estimator.estimate(("anchor_only", 1, "classifier"), 1, 10, now=111) is None


def test_notebook_display_buffer_is_bounded_and_renders_only_available_values():
    buffer = BoundedProgressBuffer(max_events=2)
    events = [
        {"status": "RUNNING", "stage": f"preflight-{index}"}
        for index in range(3)
    ]
    text = ""
    for event in events:
        text = buffer.append(event)
    assert "preflight-0" not in text
    assert "preflight-1" in text and "preflight-2" in text

    progress = format_live_progress_event(
        {"status": "RUNNING", "run_key": "ember_seed42_anchor_only", "arm": "anchor_only",
         "run_index": 1, "run_count": 100, "update_budgets_by_task": [1] * 11},
        _state(arm="anchor_only"), attempt_elapsed_seconds=2,
        matrix_counts={"completed": 0, "total": 100}, memory_snapshot={},
    )
    rendered = BoundedProgressBuffer().append({"status": "RUNNING", "live_progress": progress})
    assert "Task 3/11" in rendered and "classifier training" in rendered
    assert "Phase ETA estimating" in rendered
    assert "GAN" not in rendered
    assert "accuracy" in rendered or "metrics=N/A" in rendered


def _record(arm="current_only", **event):
    state = _state(arm=arm)
    progress = format_live_progress_event(
        {"arm": arm, "run_key": "run-1", "run_index": 1, "dataset": "ember",
         "update_budgets_by_task": [100, 90, 80] + [70] * 8}, state,
        attempt_elapsed_seconds=3661, matrix_counts={"completed": 1, "total": 100},
        memory_snapshot={"host_available_bytes": 2 * 1024 ** 3,
                         "gpu_allocated_bytes": 512 * 1024 ** 2})
    return {"status": "RUNNING", "live_progress": progress, **event}


@pytest.mark.parametrize("arm", ["current_only", "anchor_only", "derpp_matched"])
def test_panel_multiline_units_exact_budget_and_arm_fields(arm):
    from scripts.live_progress import render_progress_line
    text = render_progress_line(_record(arm))
    assert len(text.splitlines()) >= 8
    assert "\\n" not in text and "{" not in text and "}" not in text
    assert "13/80 (16.2%) | batch 256 | latest classifier loss 0.42" in text
    assert "01:01:01" in text and "2.0 GiB" in text and "512.0 MiB" in text
    assert "75.00%" in text and "macro F1 70.00%" in text
    for field in ("GAN", "generator", "critic", "gradient penalty", "epoch"):
        assert field not in text


def test_latest_real_evaluation_survives_transition_and_row_fallback():
    from scripts.live_progress import render_progress_line
    state = _state(task_index=3)
    runtime = state.extra["runtime"]
    runtime["task_details"] = [{"task_id": 3, "evaluations": [
        {"evaluated_task_id": 1, "accuracy": 0.99, "support": 12}],
        "metrics": {"accuracy": 0.73, "macro_f1": 0.65}}]
    runtime["task_rows"] = [{"task_id": 3, "classes_seen": 60, "seen_accuracy": 0.73,
                             "macro_f1": 0.65, "weighted_f1": 0.71}]
    def format_state():
        return format_live_progress_event({"arm": "current_only"}, state,
            attempt_elapsed_seconds=0, matrix_counts={}, memory_snapshot={})
    progress = format_state()
    text = render_progress_line({"live_progress": progress})
    assert "Task 4/11" in text
    assert "evaluation: task 3 | all seen test classes | classes 60" in text
    assert "accuracy 73.00%" in text and "99.00%" not in text
    assert "balanced accuracy" not in text
    runtime["task_details"] = []
    runtime["task_rows"][0]["balanced_accuracy"] = 0.68
    text = render_progress_line({"live_progress": format_state()})
    assert "balanced accuracy 68.00%" in text
    runtime["task_rows"] = []
    assert "Latest completed evaluation: N/A" in render_progress_line({"live_progress": format_state()})


def test_callable_raw_mime_single_panel_bounded_refresh_and_quiet_debug():
    from scripts.live_progress import ColabProgressDisplay
    now = [0.0]
    displayed, updated, printed = [], [], []
    def display(mime, **kwargs):
        displayed.append((mime, kwargs))
        return SimpleNamespace(display_id="panel-1")
    callback = ColabProgressDisplay(display_fn=display,
        update_fn=lambda mime, **kw: updated.append((mime, kw)),
        print_fn=printed.append, clock=lambda: now[0])
    callback({"status": "RUNNING", "stage": "checkpoint_hash_start"})
    assert not displayed
    record = _record()
    callback(record)
    for tick in range(1, 100):
        now[0] = tick / 100
        callback(record)
    assert len(displayed) == 1 and not updated
    now[0] = 7
    callback(record)
    assert len(updated) == 1
    assert displayed[0][1] == {"raw": True, "display_id": True}
    assert updated[0][1] == {"raw": True, "display_id": "panel-1"}
    assert set(updated[0][0]) == {"text/plain"}
    assert updated[0][0]["text/plain"] == callback.latest_text
    assert callback.latest_text.count("Run 1/") == 1
    assert "\n" in callback.latest_text and not printed


@pytest.mark.parametrize("location", ["record", "progress"])
def test_upload_failures_are_visible_with_cached_progress(location):
    from scripts.live_progress import ColabProgressDisplay, render_progress_line
    upload = {"generation": 5, "filename": "state.pt", "uploaded_bytes": 1024 ** 2,
              "total_bytes": 2 * 1024 ** 2, "status": "failed",
              "error_type": "HttpError", "http_status": 503}
    record = _record(status="CHECKPOINT_FAILED")
    if location == "record":
        record["checkpoint_upload"] = upload
    else:
        record["live_progress"]["checkpoint"]["upload"] = upload
    text = render_progress_line(record)
    assert "CHECKPOINT_FAILED" in text and "HttpError" in text and "503" in text
    assert "50.00% | 1.0 MiB/2.0 MiB" in text
    printed = []
    ColabProgressDisplay(display_fn=False, print_fn=printed.append)(record)
    assert "CHECKPOINT_FAILED" in printed[0] and "HttpError" in printed[0]


def test_plain_fallback_bounded_output_and_deduplicated_permanent_summaries():
    from scripts.live_progress import ColabProgressDisplay
    printed, now = [], [0]
    callback = ColabProgressDisplay(display_fn=False, print_fn=printed.append, clock=lambda: now[0])
    record = _record()
    for _ in range(100):
        callback(record)
    assert len(printed) == 1 and "\n" not in printed[0]
    now[0] = 7
    callback(record)
    assert len(printed) == 2
    for status in ("TASK_COMPLETED", "FIRST_COUNTED_PRODUCTION_RUN_VALIDATED",
                   "SYNTHETIC_VALIDATED_NON_REPORTABLE", "VALIDATED"):
        for _ in range(5):
            callback({**record, "status": status})
    summaries = [line for line in printed if line.startswith("Completed")]
    assert len(summaries) == 2
    assert len(callback._summaries) == 2


def test_upload_progress_is_throttled_but_state_transitions_are_immediate():
    from scripts.live_progress import ColabProgressDisplay
    printed = []
    callback = ColabProgressDisplay(display_fn=False, print_fn=printed.append, clock=lambda: 0)
    record = _record(status="CHECKPOINT_UPLOADING", checkpoint_upload={
        "generation": 1, "filename": "state.pt", "uploaded_bytes": 0,
        "total_bytes": 1024 ** 2, "status": "uploading"})
    for index in range(100):
        record["checkpoint_upload"]["uploaded_bytes"] = index
        callback(record)
    assert len(printed) == 1
    callback({**record, "status": "CHECKPOINT_VERIFIED"})
    assert len(printed) == 2


def test_display_failure_switches_to_compact_bounded_fallback():
    from scripts.live_progress import ColabProgressDisplay
    printed, calls = [], []
    def broken_display(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("no frontend")
    callback = ColabProgressDisplay(display_fn=broken_display, update_fn=lambda *a, **k: None,
                                   print_fn=printed.append, clock=lambda: 0)
    for _ in range(10):
        callback(_record())
    assert len(calls) == 1 and len(printed) == 1
    assert "\n" not in printed[0] and "{" not in printed[0]


def test_gan_arm_displays_actual_gan_counters_and_never_balanced_accuracy_placeholder():
    from scripts.live_progress import render_progress_line
    state = _state(task_index=0, phase="gan_critic")
    progress = format_live_progress_event({"arm": "full_hybrid"}, state,
        attempt_elapsed_seconds=0, matrix_counts={}, memory_snapshot={})
    text = render_progress_line({"live_progress": progress})
    assert "GAN epoch 1/3 | batch 8/20 | critic steps 37/300 | generator updates 7/60" in text
    assert "latest critic loss 1.25" in text
    assert "balanced accuracy" not in text


def test_unavailable_nonfinite_metrics_are_not_reported_as_measured():
    from scripts.live_progress import render_progress_line
    record = _record()
    record["live_progress"]["metrics"]["latest_task"] = {
        "accuracy": None, "balanced_accuracy": float("nan")}
    assert "Latest completed evaluation: N/A" in render_progress_line(record)


def test_task_completion_summary_uses_latest_actual_metrics_once():
    from scripts.live_progress import ColabProgressDisplay
    printed, now = [], [0]
    callback = ColabProgressDisplay(display_fn=False, print_fn=printed.append, clock=lambda: now[0])
    record = _record(status="TASK_COMPLETED")
    record["live_progress"]["metrics"]["latest_task"] = {"accuracy": 0.81, "macro_f1": 0.76}
    before = deepcopy(record)
    for tick in (0, 1, 7, 14):
        now[0] = tick
        callback(record)
    summaries = [line for line in printed if line.startswith("Completed task")]
    assert summaries == ["Completed task 3 | run run-1 | all seen test classes | "
                         "accuracy 81.00% | macro F1 76.00% | TASK_COMPLETED"]
    assert record == before


@pytest.mark.parametrize("metrics, evaluation_task, expected", [
    ({"accuracy": 0.8}, 3, "accuracy 80.00%"),
    ({"macro_f1": 0.7}, 3, "macro F1 70.00%"),
    ({"accuracy": None, "macro_f1": float("nan")}, 3, None),
    ({"accuracy": 0.8, "macro_f1": 0.7}, 2, None),
])
def test_task_summary_omits_unavailable_or_previous_task_metrics(metrics, evaluation_task, expected):
    from scripts.live_progress import ColabProgressDisplay
    printed = []
    callback = ColabProgressDisplay(display_fn=False, print_fn=printed.append)
    record = _record(status="TASK_COMPLETED")
    record["live_progress"]["metrics"].update(task_id=evaluation_task, latest_task=metrics)
    callback(record)
    summary = printed[0]
    if expected:
        assert expected in summary and "all seen test classes" in summary
    else:
        assert "accuracy" not in summary and "macro F1" not in summary
    assert "N/A" not in summary and "nan" not in summary


def test_injected_display_without_update_uses_compact_bounded_fallback():
    from scripts.live_progress import ColabProgressDisplay
    printed, displayed = [], []
    callback = ColabProgressDisplay(display_fn=lambda *a, **k: displayed.append(a),
        print_fn=printed.append, clock=lambda: 0)
    for _ in range(10):
        callback(_record())
    assert not displayed
    assert len(printed) == 1 and "\n" not in printed[0]
    assert callback.throttle.interval_seconds == 7
    with pytest.raises(ValueError):
        ColabProgressDisplay(interval_seconds=8)


@pytest.mark.parametrize("completed,total,expected", [
    (187, 200, "187/200 (93.5%)"),
    (0, 200, "0/200 (0.0%)"),
    (200, 200, "200/200 (100.0%)"),
    (0, 0, "0/0 |"),
    (13, None, "13/None |"),
])
def test_classifier_percentage_uses_exact_update_denominator(completed, total, expected):
    from scripts.live_progress import render_progress_line
    record = _record()
    record["live_progress"]["classifier"].update(
        task_updates_completed=completed, task_updates_total=total)
    assert f"Classifier updates {expected}" in render_progress_line(record)


@pytest.mark.parametrize("eta,expected", [
    (None, "estimating"), (float("nan"), "estimating"),
    (3661, "01:01:01"), (0, "00:00:00"),
])
def test_phase_eta_label_and_estimation_state(eta, expected):
    from scripts.live_progress import render_progress_line
    record = _record()
    record["live_progress"]["time"]["phase_eta_seconds"] = eta
    assert f"Phase ETA {expected}" in render_progress_line(record)
