from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
import torch


def _api(module_name: str, symbol: str):
    assert find_spec(f"revision_r1.{module_name}") is not None, f"missing revision_r1.{module_name}"
    module = import_module(f"revision_r1.{module_name}")
    assert hasattr(module, symbol), f"missing revision_r1.{module_name}.{symbol}"
    return getattr(module, symbol)


def test_real_anchor_and_generated_replay_are_transformed_from_raw_under_current_scaler():
    RunningScaler = _api("preprocessing", "RunningScaler")
    build_mixed_pool = _api("replay", "build_mixed_pool")
    to_raw_generator_output = _api("preprocessing", "to_raw_generator_output")

    scaler_at_generation = RunningScaler()
    scaler_at_generation.update(np.array([[2.0, 4.0], [4.0, 8.0]]))
    generated_standardized = np.array([[0.5, -0.5]], dtype=np.float32)
    generated_raw = to_raw_generator_output(generated_standardized, scaler_at_generation.state_dict())
    np.testing.assert_allclose(generated_raw, [[3.5, 5.0]], atol=1e-6)

    scaler_current = RunningScaler()
    scaler_current.update(np.array([[10.0, 20.0], [14.0, 24.0]]))
    raw_known = np.array([[3.5, 5.0]], dtype=np.float32)
    parts = {
        "current": (raw_known, np.array([17]), ["train:17"]),
        "anchor": (raw_known, np.array([3]), ["train:3"]),
        "generated": (generated_raw, np.array([3]), ["task1:generated:3:0"]),
    }
    pool = build_mixed_pool(parts, scaler_current)
    np.testing.assert_allclose(pool.x_raw, np.repeat(raw_known, 3, axis=0), atol=1e-6)
    np.testing.assert_allclose(pool.x_standardized, np.repeat([[-4.25, -8.5]], 3, axis=0), atol=1e-6)
    assert pool.sources.tolist() == ["current", "anchor", "generated"]
    assert pool.sample_ids == ["train:17", "train:3", "task1:generated:3:0"]


def test_lazy_mixed_pool_preserves_order_and_materializes_only_requested_rows():
    RunningScaler = _api("preprocessing", "RunningScaler")
    from revision_r1.replay import IndexedFeatureRows, build_mixed_pool

    class ReadProbe:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.dtype = values.dtype
            self.max_requested_rows = 0

        def __getitem__(self, indices):
            self.max_requested_rows = max(self.max_requested_rows, len(indices))
            return self.values[indices]

    values = np.arange(240, dtype=np.float32).reshape(60, 4)
    source = ReadProbe(values)
    selected = np.array([11, 2, 53, 8, 29], dtype=np.int64)
    features = IndexedFeatureRows(source, selected)
    labels = np.array([7, 4, 7, 2, 4], dtype=np.int64)
    ids = [f"train:{value}" for value in selected]
    scaler = RunningScaler().update(values[:20])

    pool = build_mixed_pool({"current": (features, labels, ids)}, scaler)
    assert source.max_requested_rows == 0
    assert pool.sample_ids == ids
    assert pool.sources.tolist() == ["current"] * len(selected)
    requested = np.array([3, 0], dtype=np.int64)
    np.testing.assert_array_equal(pool.x_raw[requested], values[selected[requested]])
    np.testing.assert_array_equal(pool.y_global[requested], labels[requested])
    np.testing.assert_array_equal(
        pool.x_standardized[requested], scaler.transform(values[selected[requested]])
    )
    assert source.max_requested_rows <= len(requested)


def test_stored_logit_refresh_uses_raw_examples_current_scaler_and_global_class_order():
    RunningScaler = _api("preprocessing", "RunningScaler")
    refresh_stored_logits = _api("replay", "refresh_stored_logits")
    scaler = RunningScaler()
    scaler.update(np.array([[10.0, 20.0], [14.0, 24.0]]))
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    record = refresh_stored_logits(
        model, np.array([[3.5, 5.0]], dtype=np.float32), [7], scaler,
        class_ids=[19, 7], device="cpu",
    )
    np.testing.assert_allclose(record["logits"], [[-4.25, -8.5]], atol=1e-6)
    assert record["class_ids"] == [19, 7]
    assert record["global_labels"] == [7]
    assert record["scaler_sha256"] == scaler.sha256()
    assert record["seen_class_mask"][7] is True
    assert record["seen_class_mask"][19] is True


def test_scaler_updates_are_training_only_inputs_and_state_round_trips_exactly():
    RunningScaler = _api("preprocessing", "RunningScaler")
    scaler = RunningScaler()
    scaler.update(np.array([[2.0, 4.0], [4.0, 8.0]]))
    scaler.update(np.array([[6.0, 12.0]]))
    np.testing.assert_allclose(scaler.mean_, [4.0, 8.0])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(8 / 3), np.sqrt(32 / 3)], rtol=1e-6)
    restored = RunningScaler.from_state_dict(scaler.state_dict())
    np.testing.assert_array_equal(restored.transform([[4.0, 8.0]]), scaler.transform([[4.0, 8.0]]))
    assert restored.sha256() == scaler.sha256()
    with pytest.raises(ValueError, match="finite"):
        scaler.update(np.array([[np.nan, 0.0]]))


def test_chunked_indexed_scaler_matches_full_float32_training_rows_and_bounds_reads():
    RunningScaler = _api("preprocessing", "RunningScaler")

    class ReadProbe:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.max_requested_rows = 0

        def __getitem__(self, indices):
            self.max_requested_rows = max(self.max_requested_rows, len(indices))
            return self.values[indices]

    values = (1000 + np.random.default_rng(41).normal(size=(1200, 9))).astype(np.float32)
    indices = np.random.default_rng(42).permutation(len(values))
    probe = ReadProbe(values)
    bounded = RunningScaler()
    bounded.update_indexed(probe, indices, chunk_rows=73)
    reference = RunningScaler()
    reference.update(values[indices])

    assert probe.max_requested_rows <= 73
    np.testing.assert_allclose(bounded.mean_, reference.mean_, rtol=1e-7, atol=1e-5)
    np.testing.assert_allclose(bounded.scale_, reference.scale_, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(bounded.M2_, reference.M2_, rtol=1e-6, atol=1e-5)


