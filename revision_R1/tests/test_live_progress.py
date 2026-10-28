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
    assert "ETA N/A" in rendered
    assert "GAN: N/A" in rendered
    assert "accuracy" in rendered or "metrics=N/A" in rendered
