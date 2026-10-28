from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest


def _api(module_name: str, symbol: str):
    assert find_spec(f"revision_r1.{module_name}") is not None, f"missing revision_r1.{module_name}"
    module = import_module(f"revision_r1.{module_name}")
    assert hasattr(module, symbol), f"missing revision_r1.{module_name}.{symbol}"
    return getattr(module, symbol)


def test_revision_class_order_preserves_historical_seed_42_order():
    class_order = _api("task_stream", "class_order")
    order = class_order(42)
    assert order[:10] == [83, 53, 70, 45, 44, 39, 22, 80, 10, 0]
    assert sorted(order) == list(range(100))


def test_task_splits_follow_50_then_ten_five_class_increments():
    split_order = _api("task_stream", "split_order")
    order = list(range(100))
    task_sizes = (50,) + (5,) * 10
    tasks = split_order(order, task_sizes)
    assert len(tasks) == 11
    assert tasks[0] == tuple(range(50))
    assert tasks[1] == tuple(range(50, 55))
    assert tasks[-1] == tuple(range(95, 100))
    with pytest.raises(ValueError, match="sum to 100"):
        split_order(order, (50, 5))


def test_tasks_keep_original_dataset_row_indices_and_nonempty_test_support():
    build_tasks = _api("task_stream", "build_tasks")
    train_y = np.repeat(np.arange(100), 2)
    test_y = np.arange(100)
    tasks = build_tasks(train_y, test_y, 42, (50,) + (5,) * 10)
    first_classes = set(np.random.RandomState(42).permutation(100)[:50].tolist())
    assert set(tasks[0].classes) == first_classes
    assert len(tasks[0].train_indices) == 100
    assert len(tasks[0].test_indices) == 50
    assert set(train_y[tasks[0].train_indices]) == first_classes


def test_actual_pool_reconstruction_uses_class_counts_and_main_quotas():
    reconstruct_pool_counts = _api("task_stream", "reconstruct_pool_counts")
    labels = np.repeat(np.arange(100), 4)
    actual = reconstruct_pool_counts(labels, list(range(100)), (50,) + (5,) * 10, anchor_k=3, generated_k=100)
    assert actual[1].current == 200
    assert actual[1].anchor == 0
    assert actual[1].generated == 0
    assert actual[2].current == 20
    assert actual[2].anchor == 150
    assert actual[2].generated == 5000
    assert actual[11].anchor == 285
    assert actual[11].generated == 9500


def test_run_config_fixes_arm_dataset_batch_tasks_and_schedule_identity():
    config_module = import_module("revision_r1.config") if find_spec("revision_r1.config") else None
    assert config_module is not None, "missing revision_r1.config"
    make_run_config = getattr(config_module, "make_run_config", None)
    assert callable(make_run_config), "missing make_run_config"
    schedule_hash = "a" * 64
    config = make_run_config("ember", 42, "full_hybrid", schedule_hash)
    assert config.anchor_k == 100
    assert config.batch_size == 256
    assert config.task_sizes == (50,) + (5,) * 10
    assert config.sha256() == make_run_config("ember", 42, "full_hybrid", schedule_hash).sha256()
    assert make_run_config("ember", 42, "anchor_only", schedule_hash).payload()["sampler"] == "mixed_pool_inverse_class_frequency_with_replacement"
    with pytest.raises(ValueError, match="batch size"):
        config_module.RunConfig("ember", 42, "full_hybrid", schedule_hash, batch_size=128, anchor_k=100)
    with pytest.raises(ValueError, match="unsupported arm"):
        make_run_config("ember", 42, "optional_k_sweep", schedule_hash)

