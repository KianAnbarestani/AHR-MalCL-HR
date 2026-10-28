from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _api(module_name: str, symbol: str):
    assert find_spec(f"revision_r1.{module_name}") is not None, f"missing revision_r1.{module_name}"
    module = import_module(f"revision_r1.{module_name}")
    assert hasattr(module, symbol), f"missing revision_r1.{module_name}.{symbol}"
    return getattr(module, symbol)


def test_derpp_keeps_separate_current_ce_replay_ce_and_stored_logit_mse():
    derpp_loss = _api("losses", "derpp_loss")
    logits = torch.tensor([[2.0, 0.0], [0.25, 0.75], [1.0, 0.5]], requires_grad=True)
    labels = torch.tensor([10, 20, 10])
    sources = ["current", "anchor", "anchor"]
    stored = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    stored_mask = torch.tensor([[False, False], [True, True], [True, True]])
    seen = torch.zeros(100, dtype=torch.bool)
    seen[[10, 20]] = True
    result = derpp_loss(
        logits, labels, sources, stored, [10, 20], [10, 20], seen, stored_mask,
        beta=1.0, alpha=0.5,
    )
    current = F.cross_entropy(logits[[0]], torch.tensor([0]))
    replay = F.cross_entropy(logits[[1, 2]], torch.tensor([1, 0]))
    mse = F.mse_loss(logits[[1, 2]], stored[[1, 2]])
    assert result.current_ce.item() == pytest.approx(current.item())
    assert result.replay_ce.item() == pytest.approx(replay.item())
    assert result.logit_mse.item() == pytest.approx(mse.item())
    assert result.total.item() == pytest.approx((current + replay + 0.5 * mse).item())
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_derpp_handles_empty_source_terms_and_rejects_missing_or_unmapped_targets():
    derpp_loss = _api("losses", "derpp_loss")
    logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    seen = torch.zeros(100, dtype=torch.bool)
    seen[[7, 19]] = True
    current_only = derpp_loss(logits, [7], ["current"], None, [], [7, 19], seen, None)
    assert current_only.replay_ce.item() == 0.0
    assert current_only.logit_mse.item() == 0.0
    assert torch.isfinite(current_only.total)
    with pytest.raises(ValueError, match="stored logits"):
        derpp_loss(logits, [19], ["anchor"], None, [], [7, 19], seen, None)
    with pytest.raises(ValueError, match="seen"):
        derpp_loss(logits, [8], ["current"], None, [], [7, 19], seen, None)
    with pytest.raises(ValueError, match="source"):
        derpp_loss(logits, [7], ["generated"], None, [], [7, 19], seen, None)


def test_seen_class_mask_and_noncontiguous_global_to_head_mapping_are_applied():
    derpp_loss = _api("losses", "derpp_loss")
    logits = torch.tensor([[0.0, 2.0, 1000.0]], requires_grad=True)
    seen = torch.zeros(100, dtype=torch.bool)
    seen[[7, 19]] = True
    result = derpp_loss(logits, [19], ["current"], None, [], [31, 19, 88], seen, None)
    assert torch.isfinite(result.total)
    expected = F.cross_entropy(torch.tensor([[-float("inf"), 2.0, -float("inf")]]), torch.tensor([1]))
    assert result.total.item() == pytest.approx(expected.item())


def test_anchor_only_and_er_matched_labels_have_identical_ce_and_gradients():
    classification_loss = _api("losses", "classification_loss")
    inputs = torch.tensor([[0.4, 1.2], [1.1, -0.2], [0.3, 0.8]])
    labels = torch.tensor([10, 19, 10])
    sources = ["current", "anchor", "anchor"]
    seen = torch.zeros(100, dtype=torch.bool)
    seen[[10, 19]] = True
    w = torch.tensor([[0.2, -0.3], [-0.1, 0.5]], requires_grad=True)
    w2 = w.detach().clone().requires_grad_(True)
    logits1, logits2 = inputs @ w.T, inputs @ w2.T
    args = (labels, sources, None, [], [10, 19], seen, None)
    loss1 = classification_loss("anchor_only", logits1, *args)
    loss2 = classification_loss("er_matched", logits2, *args)
    loss1.backward()
    loss2.backward()
    assert loss1.item() == pytest.approx(loss2.item())
    torch.testing.assert_close(w.grad, w2.grad)


def test_classifier_initialization_and_head_expansion_are_paired_and_rng_isolated():
    build_classifier = _api("models", "build_classifier")
    initial_state_sha256 = _api("models", "initial_state_sha256")
    before = torch.random.get_rng_state().clone()
    model1 = build_classifier(4, seed=123, initial_classes=50, hidden=(8, 4, 3), embedding_dim=2, dropout=0.0)
    assert torch.equal(before, torch.random.get_rng_state())
    model2 = build_classifier(4, seed=123, initial_classes=50, hidden=(8, 4, 3), embedding_dim=2, dropout=0.0)
    assert initial_state_sha256(model1) == initial_state_sha256(model2)
    model1.expand(55, seed=991)
    model2.expand(55, seed=991)
    assert initial_state_sha256(model1) == initial_state_sha256(model2)
    assert model1(torch.ones(2, 4)).shape == (2, 55)


def test_metrics_match_harmonized_task_matrix_and_final_classification_definitions():
    classification_metrics = _api("metrics", "classification_metrics")
    continual_metrics = _api("metrics", "continual_metrics")
    classification = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2], labels=[0, 1, 2])
    assert classification["accuracy"] == pytest.approx(0.75)
    assert classification["macro_f1"] == pytest.approx(7 / 9)
    assert classification["weighted_f1"] == pytest.approx(0.75)
    assert classification["balanced_accuracy"] == pytest.approx(5 / 6)
    assert classification["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    matrix = [[0.8, None, None], [0.7, 0.9, None], [0.6, 0.8, 1.0]]
    metrics = continual_metrics(matrix, [2, 2, 1])
    assert metrics["aia"] == pytest.approx((0.8 + 0.8 + 0.76) / 3)
    assert metrics["forgetting"] == pytest.approx(0.15)
    assert metrics["bwt"] == pytest.approx(-0.15)
    assert metrics["task_acquisition"] == pytest.approx(0.9)
    assert "not a measure of adaptation speed" in metrics["task_acquisition_definition"]
    assert metrics["final_old_task_accuracy"] == pytest.approx(0.7)
    assert metrics["final_new_task_accuracy"] == pytest.approx(1.0)


def test_generated_selection_records_candidate_scores_and_selected_candidate_indices():
    select_generated = _api("generative", "select_generated")

    class TinyGenerator(torch.nn.Module):
        def forward(self, noise, labels):
            return noise + labels.float().unsqueeze(1)

    class TinyClassifier(torch.nn.Module):
        def forward(self, values, return_mid=False):
            middle = values
            logits = torch.stack((values[:, 0], -values[:, 0]), dim=1)
            return (logits, middle) if return_mid else logits

    torch.manual_seed(99)
    features, labels, records = select_generated(
        TinyGenerator(), TinyClassifier(), {3: torch.zeros(2)}, [3],
        candidates_per_class=5, quota=2, z_dim=2, diversity_lambda=0.25,
        device="cpu", return_records=True,
    )
    record = records[0]
    assert features.shape == (2, 2)
    assert labels.tolist() == [3, 3]
    assert record["candidate_count"] == 5
    assert len(record["candidate_l1_distances"]) == 5
    assert len(record["selected_candidate_indices"]) == 2
    assert len(record["candidate_features_sha256"]) == 64
    assert len(record["selected_features_sha256"]) == 64


def test_unique_arm_component_activation_has_no_redundant_er_arm():
    arm_components = _api("config", "arm_components")
    expected = {
        "current_only": (False, False, False, False),
        "anchor_only": (True, False, False, False),
        "generated_only": (False, True, True, False),
        "full_hybrid": (True, True, True, False),
        "derpp_matched": (True, False, False, True),
    }
    for arm, values in expected.items():
        actual = arm_components(arm)
        assert (actual["anchors"], actual["generated"], actual["gan"], actual["derpp_loss"]) == values
    with pytest.raises(ValueError, match="unsupported arm"):
        arm_components("er_matched")

