from __future__ import annotations

import csv
import json
import shutil

import numpy as np
import pytest
import torch

from revision_r1.config import RunConfig
from revision_r1.data import DatasetBundle
from revision_r1.runner import _with_gan_torch_rng, require_production_runtime, run_one
from revision_r1.validation import validate_run_artifacts


def _bundle(*, far_away_test=False):
    rng = np.random.default_rng(123)
    x_train = np.vstack([rng.normal(loc=class_id / 20, scale=0.2, size=(2, 4)) for class_id in range(100)]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([rng.normal(loc=class_id / 20, scale=0.2, size=(1, 4)) for class_id in range(100)]).astype(np.float32)
    if far_away_test:
        x_test += 10000.0
    y_test = np.arange(100, dtype=np.int64)
    marker = {"sha256": "synthetic-fixture", "md5": "synthetic-fixture"}
    return DatasetBundle("ember", x_train, y_train, x_test, y_test, marker, marker)


def _config(arm):
    anchor_k = 100 if arm in {"anchor_only", "full_hybrid", "derpp_matched"} else 0
    return RunConfig(
        dataset="ember", seed=42, arm=arm, update_schedule_sha256="a" * 64,
        anchor_k=anchor_k, classifier_hidden=(8, 8, 4), classifier_embedding_dim=8,
        dropout=0.0, generated_candidates_per_class=2, generated_k=1,
    )


def _run_all(tmp_path):
    results = {}
    for arm in ("current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched"):
        results[arm] = run_one(
            _config(arm), _bundle(), tmp_path / arm,
            synthetic=True, update_counts=[1] * 11, device="cpu",
        )
    return results


def test_synthetic_matrix_exports_exact_updates_source_traces_and_component_isolation(tmp_path):
    results = _run_all(tmp_path)
    hashes = {result["initial_state_sha256"] for result in results.values()}
    assert len(hashes) == 1

    for arm, result in results.items():
        root = tmp_path / arm
        assert result["namespace"] == "SYNTHETIC_NON_REPORTABLE"
        assert result["classifier_updates"] == 11
        assert result["sampled_examples"] == 11 * 256
        assert result["task_matrix_shape"] == [11, 11]
        assert result["validated"] is True
        assert (root / "predictions.npz").is_file()
        assert (root / "classification_report.csv").is_file()
        assert (root / "source_proportions.json").is_file()
        assert (root / "trace.jsonl").is_file()
        assert (root / "resource_metrics.json").is_file()
        assert (root / "task_boundary_predictions.npz").is_file()
        assert (root / "task_boundary_prediction_manifest.json").is_file()
        assert (root / "selection_records.jsonl").is_file()
        recovery_manifest = next((root / "checkpoints" / "recovery").glob("generation-*.manifest.json"))
        recovery_identity = json.loads(recovery_manifest.read_text())["identity"]
        assert recovery_identity["schedule_sha256"] == "a" * 64
        assert recovery_identity["execution_scope"] == "synthetic_validation"
        summary = json.loads((root / "full-result.json").read_text())
        assert summary["training_reproduction_claimed"] is False
        assert summary["namespace"] == "SYNTHETIC_NON_REPORTABLE"
        predictions = np.load(root / "predictions.npz")
        assert predictions["y_true_global"].shape == (100,)
        assert predictions["y_pred_global"].shape == (100,)
        boundary = np.load(root / "task_boundary_predictions.npz", allow_pickle=False)
        boundary_manifest = json.loads((root / "task_boundary_prediction_manifest.json").read_text())
        assert len(boundary_manifest["stages"]) == 11
        for stage in range(1, 12):
            prefix = f"stage_{stage:02d}"
            assert len(boundary[f"{prefix}_y_true_global"]) == 50 + (stage - 1) * 5
            assert len(boundary[f"{prefix}_y_pred_global"]) == 50 + (stage - 1) * 5
            assert len(boundary[f"{prefix}_test_indices"]) == 50 + (stage - 1) * 5
        proportion_rows = json.loads((root / "source_proportions.json").read_text())
        assert len(proportion_rows) == 11
        assert all(sum(row["realized_draws"].values()) == 256 for row in proportion_rows)
        details = json.loads((root / "per_task_details.json").read_text())
        assert len(details) == 11
        assert all(row["optimizer_lifecycle"]["classifier_step_before_first_update"] == 0 for row in details)
        if arm in {"generated_only", "full_hybrid"}:
            assert all(row["optimizer_lifecycle"]["critic_step_before_first_update"] == 0 for row in details)
            assert all(row["optimizer_lifecycle"]["generator_step_before_first_update"] == 0 for row in details)
    assert results["current_only"]["generator_steps"] == 0
    assert results["anchor_only"]["generator_steps"] == 0
    assert results["derpp_matched"]["generator_steps"] == 0
    assert results["current_only"]["critic_steps"] == 0
    assert results["anchor_only"]["critic_steps"] == 0
    assert results["derpp_matched"]["critic_steps"] == 0
    assert results["generated_only"]["generator_steps"] > 0
    assert results["full_hybrid"]["generator_steps"] > 0
    assert results["current_only"]["anchor_examples"] == 0
    assert results["generated_only"]["anchor_examples"] == 0
    assert results["anchor_only"]["anchor_examples"] > 0
    assert results["derpp_matched"]["derpp_replay_updates"] > 0
    assert results["generated_only"]["generated_examples"] > 0
    for arm in ("generated_only", "full_hybrid"):
        selection_rows = [json.loads(line) for line in (tmp_path / arm / "selection_records.jsonl").read_text().splitlines()]
        assert selection_rows
        assert all(row["selected_count"] == 1 for row in selection_rows)
        assert all("candidate_features_sha256" in row for row in selection_rows)
    later_sources = json.loads((tmp_path / "generated_only" / "source_proportions.json").read_text())[1]
    assert later_sources["expected_proportions"]["generated"] > 0
    all_traces = {
        arm: [json.loads(line) for line in (tmp_path / arm / "trace.jsonl").read_text().splitlines()]
        for arm in results
    }
    reference_first = all_traces["current_only"][0]
    for trace in all_traces.values():
        assert trace[0]["source_draws"] == {"anchor": 0, "current": 256, "generated": 0}
        assert trace[0]["class_draws"] == reference_first["class_draws"]
        assert trace[0]["loss"] == pytest.approx(reference_first["loss"], abs=1e-7)
    hybrid_details = json.loads((tmp_path / "full_hybrid" / "per_task_details.json").read_text())
    assert hybrid_details[1]["generator_training_pool_sources"]["anchor"] > 0
    derpp_details = json.loads((tmp_path / "derpp_matched" / "per_task_details.json").read_text())
    assert derpp_details[1]["loss_terms"]["replay_ce"] > 0
    assert derpp_details[1]["loss_terms"]["logit_mse"] > 0


def test_cpu_interrupt_and_resume_do_not_duplicate_task_or_update_rows(tmp_path):
    root = tmp_path / "resume"
    calls = 0

    def stop_after_several_units(_state):
        nonlocal calls
        calls += 1
        return calls == 8

    first = run_one(
        _config("derpp_matched"), _bundle(), root, synthetic=True,
        update_counts=[1] * 11, device="cpu", stop_hook=stop_after_several_units,
    )
    assert first["status"] == "INTERRUPTED_RESUMABLE"
    resumed = run_one(_config("derpp_matched"), _bundle(), root, synthetic=True, update_counts=[1] * 11, device="cpu")
    assert resumed["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    with (root / "per_task_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11
    assert len({int(row["task_id"]) for row in rows}) == 11
    assert resumed["classifier_updates"] == 11


def test_generated_run_resumes_from_inside_critic_substeps_without_duplicate_updates(tmp_path):
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted = run_one(_config("generated_only"), _bundle(), uninterrupted_root, synthetic=True, update_counts=[1] * 11, device="cpu")
    requested = False

    def interrupt_in_critic(state):
        nonlocal requested
        if state.cursor.phase == "gan_critic" and state.cursor.substep_index == 1:
            requested = True
            return True
        return False

    partial = run_one(
        _config("generated_only"), _bundle(), resumed_root, synthetic=True,
        update_counts=[1] * 11, device="cpu", stop_hook=interrupt_in_critic,
    )
    assert requested
    assert partial["status"] == "INTERRUPTED_RESUMABLE"
    resumed = run_one(_config("generated_only"), _bundle(), resumed_root, synthetic=True, update_counts=[1] * 11, device="cpu")
    assert resumed["classifier_updates"] == uninterrupted["classifier_updates"] == 11
    assert resumed["critic_steps"] == uninterrupted["critic_steps"]
    assert resumed["generator_steps"] == uninterrupted["generator_steps"]
    assert json.loads((resumed_root / "source_proportions.json").read_text()) == json.loads((uninterrupted_root / "source_proportions.json").read_text())
    assert (resumed_root / "predictions.npz").read_bytes() == (uninterrupted_root / "predictions.npz").read_bytes()


def test_scaler_uses_only_current_training_rows_not_test_features(tmp_path):
    train = _bundle(far_away_test=False)
    altered_test = _bundle(far_away_test=True)
    first = run_one(_config("current_only"), train, tmp_path / "train", synthetic=True, update_counts=[1] * 11, device="cpu")
    second = run_one(_config("current_only"), altered_test, tmp_path / "test-shifted", synthetic=True, update_counts=[1] * 11, device="cpu")
    first_rows = json.loads((tmp_path / "train" / "scaler_history.json").read_text())
    second_rows = json.loads((tmp_path / "test-shifted" / "scaler_history.json").read_text())
    assert first_rows == second_rows
    assert first["initial_state_sha256"] == second["initial_state_sha256"]


def test_production_execution_refuses_cpu_and_synthetic_namespace_cannot_be_reported():
    with pytest.raises(RuntimeError, match="CUDA"):
        require_production_runtime("cpu")
    with pytest.raises(ValueError, match="synthetic"):
        validate_run_artifacts("/not/a/run", require_production=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only synthetic runner check")
def test_synthetic_validation_can_run_on_cuda_without_becoming_production(tmp_path):
    result = run_one(
        _config("current_only"), _bundle(), tmp_path / "cuda-synthetic",
        synthetic=True, update_counts=[1] * 11, device="cuda",
    )
    assert result["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    assert result["namespace"] == "SYNTHETIC_NON_REPORTABLE"
    summary = json.loads((tmp_path / "cuda-synthetic" / "full-result.json").read_text())
    assert summary["schedule_sha256"] == "a" * 64
    assert json.loads(next((tmp_path / "cuda-synthetic" / "checkpoints" / "recovery").glob("generation-*.manifest.json")).read_text())["identity"]["execution_scope"] == "synthetic_validation"


def test_gan_random_draws_do_not_advance_classifier_torch_rng():
    torch.manual_seed(1234)
    expected = torch.rand(12)
    torch.manual_seed(1234)
    state = {"gan_torch_cpu_rng": torch.Generator().manual_seed(9).get_state()}
    _with_gan_torch_rng(state, lambda: torch.rand(1000), device=torch.device("cpu"))
    actual = torch.rand(12)
    assert torch.equal(actual, expected)


def test_forced_task_boundary_resumes_from_bounded_recovery_after_local_checkpoint_loss(tmp_path):
    output = tmp_path / "result"
    local = tmp_path / "local-checkpoints" / "ember" / "seed_42" / "current_only"
    drive = tmp_path / "drive-checkpoints" / "ember" / "seed_42" / "current_only"

    def stop_after_first_task(state):
        return state.cursor.task_index == 1 and state.cursor.phase == "scaler"

    interrupted = run_one(
        _config("current_only"), _bundle(), output, synthetic=True,
        update_counts=[1] * 11, device="cpu", local_checkpoint_root=local,
        drive_checkpoint_root=drive, stop_hook=stop_after_first_task,
    )
    assert interrupted["status"] == "INTERRUPTED_RESUMABLE"
    assert list((drive / "recovery").glob("generation-*.manifest.json"))
    assert not list((drive / "task_boundaries").glob("generation-*.manifest.json"))
    shutil.rmtree(local)

    resumed = run_one(
        _config("current_only"), _bundle(), output, synthetic=True,
        update_counts=[1] * 11, device="cpu", local_checkpoint_root=local,
        drive_checkpoint_root=drive,
    )
    assert resumed["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
    assert resumed["classifier_updates"] == 11
    assert not list((drive / "task_boundaries").glob("generation-*.manifest.json"))
    recovery = list((drive / "recovery").glob("generation-*.manifest.json"))
    assert len(recovery) <= 2
    latest_generation = max(recovery, key=lambda path: json.loads(path.read_text())["generation"])
    assert json.loads(latest_generation.read_text())["kind"] == "complete"
