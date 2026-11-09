"""Causal update budgets and deterministic completed-window execution."""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Callable, Mapping
from pathlib import Path

from .checkpoint import save_checkpoint

ANCHOR_K = 100
GENERATED_K = 100
BATCH_SIZE = 256
CLASSIFIER_EPOCH_EQUIVALENT = 3


class StopRequested(RuntimeError):
    """Synthetic or operator stop; resume from the preceding completed window."""


def planned_pool_counts(current_n: int, prior_family_support: Mapping[str, int]) -> dict[str, int]:
    """Freeze Full Hybrid pool size from source rows, anchor quota and gen quota."""
    current_n = int(current_n)
    if current_n < 1:
        raise ValueError("current training pool must be nonempty")
    if any(int(support) < 0 for support in prior_family_support.values()):
        raise ValueError("prior family support cannot be negative")
    anchor_n = sum(min(ANCHOR_K, int(support)) for support in prior_family_support.values())
    generated_n = len(prior_family_support) * GENERATED_K
    return {
        "current": current_n,
        "anchor": anchor_n,
        "generated": generated_n,
        "full_hybrid": current_n + anchor_n + generated_n,
    }


def planned_updates(current_n: int, prior_family_support: Mapping[str, int]) -> int:
    pool = planned_pool_counts(current_n, prior_family_support)["full_hybrid"]
    return CLASSIFIER_EPOCH_EQUIVALENT * math.ceil(pool / BATCH_SIZE)


def _required_state_keys(state: dict) -> None:
    required = {"stream_cursor", "training_state", "optimizer_state", "scaler_state",
                "anchor_state", "class_map", "gan_state", "rng_state"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(f"run state lacks required keys: {sorted(required - set(state if isinstance(state, dict) else ())) }")


def capture_rng_state(numpy_generators: Mapping | None = None) -> dict:
    """Capture every installed global RNG and named NumPy Generator state."""
    result = {"python": random.getstate()}
    try:
        import numpy as np
        result["numpy_global"] = deepcopy(np.random.get_state())
        result["numpy_generators"] = {
            str(name): deepcopy(generator.bit_generator.state)
            for name, generator in (numpy_generators or {}).items()
        }
    except ImportError:
        result["numpy_global"] = None
        result["numpy_generators"] = {}
    try:
        import torch
        result["torch_cpu"] = torch.get_rng_state().cpu()
        result["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    except ImportError:
        result["torch_cpu"] = None
        result["torch_cuda"] = []
    return result


def restore_rng_state(state: dict, numpy_generators: Mapping | None = None) -> None:
    if not isinstance(state, dict) or "python" not in state:
        raise ValueError("RNG checkpoint must contain Python random state")
    random.setstate(state["python"])
    if state.get("numpy_global") is not None:
        import numpy as np
        np.random.set_state(deepcopy(state["numpy_global"]))
        saved = state.get("numpy_generators", {})
        generators = numpy_generators or {}
        if set(saved) != {str(name) for name in generators}:
            raise ValueError("named NumPy RNG streams do not match their checkpoint")
        for name, generator in generators.items():
            generator.bit_generator.state = deepcopy(saved[str(name)])
    if state.get("torch_cpu") is not None:
        import torch
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_states = state.get("torch_cuda", [])
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError("checkpoint CUDA RNG state does not match the active runtime")
            torch.cuda.set_rng_state_all(cuda_states)


def run_window(
    state: dict,
    window: Mapping,
    *,
    updates: int,
    train_batch: Callable[[dict, Mapping, int], dict],
    checkpoint_path,
    finalize_window: Callable[[dict, Mapping], dict] | None = None,
    stop_after_batches: int | None = None,
) -> dict:
    """Train one update window and save exactly one completed-window checkpoint.

    An interrupted partial window is intentionally unsaved. The caller resumes
    from the previous window checkpoint, restoring that boundary's RNG state.
    """
    _required_state_keys(state)
    index = int(window["window_index"])
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != f"window_{index}.pkl":
        raise ValueError("checkpoint path must identify the completed window index")
    cursor = state["stream_cursor"]
    if int(cursor["next_window_index"]) != index:
        raise ValueError("window order/cursor mismatch")
    if int(updates) < 1:
        raise ValueError("each nonempty window needs at least one classifier update")
    if stop_after_batches is not None and int(stop_after_batches) < 1:
        raise ValueError("stop_after_batches must be positive")
    generators = state.get("numpy_rng_generators", {})
    restore_rng_state(state["rng_state"], generators)
    cursor.update({"window_id": str(window["window_id"]), "phase": "classifier", "batch_index": 0})
    for batch_index in range(int(updates)):
        cursor["batch_index"] = batch_index
        state = train_batch(state, window, batch_index)
        state["rng_state"] = capture_rng_state(generators)
        cursor["batch_index"] = batch_index + 1
        if stop_after_batches is not None and batch_index + 1 >= int(stop_after_batches):
            raise StopRequested(f"stopped in {window['window_id']} after {batch_index + 1} updates")
    if finalize_window is not None:
        state = finalize_window(state, window)
    cursor.update({
        "last_completed_window": index,
        "next_window_index": index + 1,
        "window_id": str(window["window_id"]),
        "phase": "completed",
        "batch_index": 0,
    })
    state["rng_state"] = capture_rng_state(generators)
    save_checkpoint(checkpoint_path, state)
    return state
