"""Phase-resumable revision_R1 matched runner; production entry is CUDA-gated."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import resource as process_resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import CheckpointPolicy, CheckpointError, load_latest_valid, mirror_checkpoint_generations, save_generation
from .storage import should_persist_task_boundary_duplicate
from .config import ARMS, DATASETS, RunConfig, arm_components
from .engine import StopRequested
from .exports import write_csv, write_json, write_jsonl, write_predictions, write_task_boundary_predictions
from .generative import class_mean_mid, critic_update, generate_class_candidates, generator_update, select_candidate_features
from .losses import classification_loss, derpp_loss
from .locking import RunLease
from .metrics import classification_metrics, continual_metrics
from .models import ExpandingClassifier, V155ConditionalGenerator, V155ProjectionCritic, build_classifier, initial_state_sha256
from .preprocessing import RunningScaler
from .registry import RunRegistry
from .replay import IndexedFeatureRows, MixedPool, build_mixed_pool, refresh_stored_logits
from .resources import ResourceTracker
from .rng import RNGStreams, make_rng_streams
from .sampling import MixedPoolSampler, expected_source_proportions
from .state import PhaseCursor, RunState
from .task_stream import build_tasks, order_sha256
from .validation import build_artifact_manifest, validate_run_artifacts, file_sha256


SYNTHETIC_NAMESPACE = "SYNTHETIC_NON_REPORTABLE"
PHASES = ("scaler", "gan_critic", "classifier", "evaluation", "generation", "selection", "refresh", "export")


def require_production_runtime(device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production execution requires an explicitly selected, available CUDA device")


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    values = np.asarray(array)
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    rows_per_block = max(1, (8 << 20) // max(1, values[0:1].nbytes if values.ndim else values.nbytes))
    for start in range(0, len(values), rows_per_block):
        digest.update(np.ascontiguousarray(values[start : start + rows_per_block]).tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _dataset_identity(bundle, *, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {
            "namespace": SYNTHETIC_NAMESPACE,
            "dataset": bundle.dataset,
            "train_x_sha256": _array_hash(bundle.x_train),
            "train_y_sha256": _array_hash(bundle.y_train),
            "test_x_sha256": _array_hash(bundle.x_test),
            "test_y_sha256": _array_hash(bundle.y_test),
        }
    return {
        "namespace": "PRODUCTION",
        "dataset": bundle.dataset,
        "train_sha256": bundle.train_metadata["sha256"],
        "test_sha256": bundle.test_metadata["sha256"],
        "train_md5": bundle.train_metadata["md5"],
        "test_md5": bundle.test_metadata["md5"],
    }


def _code_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _capture_rng(streams: RNGStreams) -> dict[str, Any]:
    result = {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "named_streams": streams.state_dict(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state: dict[str, Any], streams: RNGStreams) -> None:
    if not state:
        return
    random.setstate(state["python_global"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    streams.load_state_dict(state["named_streams"])


def _with_gan_torch_rng(state: dict[str, Any], operation, *, device: torch.device):
    """Run GAN randomness on a separate Torch stream, preserving classifier state."""
    classifier_cpu = torch.get_rng_state()
    classifier_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    if "gan_torch_cpu_rng" not in state:
        raise ValueError("separate GAN Torch RNG state was not initialized")
    torch.set_rng_state(state["gan_torch_cpu_rng"])
    if device.type == "cuda":
        gan_cuda = state.get("gan_torch_cuda_rng")
        if gan_cuda is None:
            raise ValueError("separate GAN CUDA RNG state was not initialized")
        torch.cuda.set_rng_state_all(gan_cuda)
    try:
        result = operation()
        state["gan_torch_cpu_rng"] = torch.get_rng_state()
        if device.type == "cuda":
            state["gan_torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        return result
    finally:
        torch.set_rng_state(classifier_cpu)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(classifier_cuda)


def _initial_gan_torch_rng(seed: int, device: torch.device) -> dict[str, Any]:
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        result = {"gan_torch_cpu_rng": torch.get_rng_state()}
        if device.type == "cuda":
            result["gan_torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return result


def _max_optimizer_step(optimizer) -> int:
    steps = []
    for values in optimizer.state.values():
        step = values.get("step")
        if step is not None:
            steps.append(int(step.item() if isinstance(step, torch.Tensor) else step))
    return max(steps, default=0)


def _record_classifier_mode(runtime: dict[str, Any], classifier, *, phase: str, task_id: int) -> None:
    dropout_modules = [module for module in classifier.modules() if isinstance(module, torch.nn.Dropout)]
    expected = phase == "classifier_training"
    dropout_training = [bool(module.training) for module in dropout_modules]
    if bool(classifier.training) is not expected or not dropout_training or any(value is not expected for value in dropout_training):
        raise RuntimeError(f"classifier/dropout mode mismatch in {phase} for task {task_id}")
    trace = runtime.setdefault("classifier_mode_trace", [])
    if any(row.get("phase") == phase and int(row.get("task_id", -1)) == int(task_id) for row in trace):
        return
    trace.append({
        "phase": phase,
        "task_id": int(task_id),
        "classifier_training": bool(classifier.training),
        "dropout_training": dropout_training,
        "dropout_probabilities": [float(module.p) for module in dropout_modules],
    })


def _seed_new_run(streams: RNGStreams) -> None:
    random.seed(streams.dropout_seed)
    np.random.seed(streams.dropout_seed % (2**32 - 1))
    torch.manual_seed(streams.dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(streams.dropout_seed)


def _make_gan_models(feature_dim: int, config: RunConfig, seed: int, device: torch.device, *, small: bool):
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        generator = V155ConditionalGenerator(
            feature_dim, z_dim=config.noise_dim, hidden=(16 if small else 256)
        ).to(device)
        critic = V155ProjectionCritic(feature_dim).to(device)
    return generator, critic


def _run_schedule(path: str | Path, config: RunConfig, train_labels: np.ndarray) -> list[int]:
    path = Path(path)
    observed_hash = file_sha256(path)
    if observed_hash != config.update_schedule_sha256:
        raise ValueError(f"proposed update schedule hash mismatch: {observed_hash}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    tasks = build_tasks(train_labels, np.arange(100, dtype=np.int64), config.seed, config.task_sizes)
    counts = np.bincount(train_labels, minlength=100)
    selected = {}
    for row in rows:
        if row.get("dataset") == config.dataset and int(row.get("seed", -1)) == config.seed:
            task_id = int(row["task"])
            if task_id in selected:
                raise ValueError(f"duplicate proposed schedule row for task {task_id}")
            selected[task_id] = row
    if set(selected) != set(range(1, 12)):
        raise ValueError("proposed update schedule must contain exactly tasks 1..11 for this run")
    output = []
    seen: list[int] = []
    anchor_quota = int(DATASETS[config.dataset]["anchor_k"])
    for task_id, task in enumerate(tasks, start=1):
        current = int(sum(counts[class_id] for class_id in task.classes))
        anchors = int(sum(min(anchor_quota, int(counts[class_id])) for class_id in seen))
        generated = len(seen) * int(config.generated_k)
        row = selected[task_id]
        declared = (
            int(row["current_pool_size"]),
            int(row["anchor_pool_size"]),
            int(row["generated_pool_size"]),
            int(row["full_hybrid_pool_size"]),
        )
        actual = (current, anchors, generated, current + anchors + generated)
        if declared != actual:
            raise ValueError(f"proposed schedule pools differ from verified dataset at task {task_id}: {declared} != {actual}")
        updates = int(row["proposed_classifier_updates"])
        if updates != 3 * ((actual[3] + 255) // 256):
            raise ValueError(f"proposed update formula mismatch at task {task_id}")
        if row.get("training_reproduction_claimed", "").strip().upper() not in {"NO", "FALSE", "0"}:
            raise ValueError("proposed schedule unexpectedly claims historical reproduction")
        output.append(updates)
        seen.extend(task.classes)
    return output


def _task_supports(bundle, order, task_sizes) -> list[int]:
    groups = []
    cursor = 0
    for size in task_sizes:
        groups.append(order[cursor : cursor + int(size)])
        cursor += int(size)
    return [int(np.count_nonzero(np.isin(bundle.y_test, group))) for group in groups]


def _seen_classes(order: list[int], task_index: int) -> list[int]:
    return order[: 50 + 5 * int(task_index)]


def _active_parts(state: RunState, bundle, task_index: int, current_indices: np.ndarray) -> dict[str, tuple[Any, np.ndarray, list[str]]]:
    parts: dict[str, tuple[Any, np.ndarray, list[str]]] = {
        "current": (
            IndexedFeatureRows(bundle.x_train, current_indices),
            np.asarray(bundle.y_train[current_indices], dtype=np.int64),
            [f"train:{int(index)}" for index in current_indices],
        )
    }
    by_class = state.anchor_replay_state.get("indices_by_class", {})
    anchor_indices = [int(index) for label, values in by_class.items() if int(label) in _seen_classes(state.extra["runtime"]["class_order"], task_index) for index in values]
    if anchor_indices:
        indices = np.asarray(anchor_indices, dtype=np.int64)
        parts["anchor"] = (
            IndexedFeatureRows(bundle.x_train, indices),
            np.asarray(bundle.y_train[indices], dtype=np.int64),
            [f"train:{int(index)}" for index in indices],
        )
    replay = state.generated_replay_state
    if replay.get("x_raw") is not None and len(replay.get("global_labels", [])):
        x_raw = np.asarray(replay["x_raw"], dtype=np.float32)
        labels = np.asarray(replay["global_labels"], dtype=np.int64)
        parts["generated"] = (x_raw, labels, [f"generated:{i}" for i in range(len(labels))])
    return parts


def _class_targets(state: RunState, pool: MixedPool, batch_indices: np.ndarray, head_class_ids: list[int]):
    targets = np.zeros((len(batch_indices), len(head_class_ids)), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    head_position = {int(class_id): index for index, class_id in enumerate(head_class_ids)}
    anchor_state = state.anchor_replay_state
    anchor_classes = [int(value) for value in anchor_state.get("class_ids", [])]
    target_by_index = anchor_state.get("targets_by_index", {})
    generated_state = state.generated_replay_state
    generated_classes = [int(value) for value in generated_state.get("class_ids", [])]
    gen_targets = generated_state.get("stored_logits", [])
    source_rows = pool.sources[batch_indices]
    sample_ids = [pool.sample_ids[int(index)] for index in batch_indices]
    for row, (source, sample_id) in enumerate(zip(source_rows, sample_ids)):
        if source == "anchor":
            old = target_by_index.get(str(int(sample_id.split(":", 1)[1])))
            if old is None:
                continue
            for col, class_id in enumerate(anchor_classes):
                if class_id in head_position:
                    target_value = np.asarray(old, dtype=np.float32)[col]
                    targets[row, head_position[class_id]] = target_value
                    mask[row, head_position[class_id]] = True
        elif source == "generated" and gen_targets:
            generated_index = int(sample_id.split(":", 1)[1])
            for col, class_id in enumerate(generated_classes):
                if class_id in head_position:
                    targets[row, head_position[class_id]] = float(gen_targets[generated_index][col])
                    mask[row, head_position[class_id]] = True
    return targets, mask, head_class_ids


def _prepare_logits(state: RunState, model, bundle, scaler: RunningScaler, head_class_ids: list[int], device: torch.device) -> None:
    anchor_indices = [int(i) for label in sorted(state.anchor_replay_state.get("indices_by_class", {}), key=int) for i in state.anchor_replay_state["indices_by_class"][label]]
    if anchor_indices:
        raw = IndexedFeatureRows(bundle.x_train, anchor_indices)
        labels = np.asarray(bundle.y_train[anchor_indices], dtype=np.int64)
        refreshed = refresh_stored_logits(model, raw, labels, scaler, head_class_ids, device=device)
        state.anchor_replay_state["ordered_indices"] = anchor_indices
        state.anchor_replay_state["class_ids"] = refreshed["class_ids"]
        state.anchor_replay_state["seen_class_mask"] = refreshed["seen_class_mask"]
        state.anchor_replay_state["scaler_sha256"] = refreshed["scaler_sha256"]
        state.anchor_replay_state["targets_by_index"] = {
            str(index): refreshed["logits"][position] for position, index in enumerate(anchor_indices)
        }
    replay = state.generated_replay_state
    if replay.get("x_raw") is not None and len(replay.get("global_labels", [])):
        refreshed = refresh_stored_logits(
            model, replay["x_raw"], replay["global_labels"], scaler,
            head_class_ids, device=device,
        )
        replay["stored_logits"] = refreshed["logits"]
        replay["class_ids"] = refreshed["class_ids"]
        replay["seen_class_mask"] = refreshed["seen_class_mask"]
        replay["scaler_sha256"] = refreshed["scaler_sha256"]


def _predict_test_indices_batched(
    model, x_test, indices: np.ndarray, scaler: RunningScaler, device: torch.device,
    *, batch_rows: int = 2048,
) -> np.ndarray:
    if batch_rows < 1:
        raise ValueError("evaluation batch_rows must be positive")
    from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory
    require_host_memory(
        min(len(indices), batch_rows) * int(x_test.shape[1]) * 8 + (16 << 20),
        reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="bounded test-evaluation batch",
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_rows):
            selected = indices[start : start + batch_rows]
            raw = np.asarray(x_test[selected], dtype=np.float32)
            standardized = scaler.transform(raw)
            tensor = torch.as_tensor(standardized, dtype=torch.float32, device=device)
            predictions.append(model(tensor).argmax(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False) if predictions else np.empty(0, dtype=np.int64)


def _atomic_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _environment(device: torch.device, synthetic: bool) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "pytorch": str(torch.__version__),
        "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "platform": platform.platform(), "namespace": SYNTHETIC_NAMESPACE if synthetic else "PRODUCTION",
    }


def _export_run(state: RunState, config: RunConfig, bundle, output_dir: Path, device: torch.device, synthetic: bool, initial_hash: str, code_hash: str, supports: list[int], tracker: ResourceTracker, classifier, generator, critic) -> dict:
    runtime = state.extra["runtime"]
    order = runtime["class_order"]
    matrix = runtime["task_matrix"]
    continual = continual_metrics(matrix, supports)
    y_true = np.asarray(state.predictions["y_true_global"], dtype=np.int64)
    y_pred = np.asarray(state.predictions["y_pred_global"], dtype=np.int64)
    classification = classification_metrics(y_true, y_pred, labels=order)
    namespace = SYNTHETIC_NAMESPACE if synthetic else "PRODUCTION"
    result = {
        "namespace": namespace,
        "method": "revision_R1_matched_runner",
        "training_reproduction_claimed": False,
        "dataset": config.dataset, "seed": config.seed, "arm": config.arm,
        "initial_state_sha256": initial_hash,
        "class_order_sha256": order_sha256(order),
        "schedule_sha256": config.update_schedule_sha256,
        "classifier_updates": int(state.resources.get("classifier_updates", 0)),
        "sampled_examples": int(state.resources.get("sampled_examples", 0)),
        "generator_steps": int(state.resources.get("generator_updates", 0)),
        "critic_steps": int(state.resources.get("critic_steps", 0)),
        "anchor_examples": int(sum(len(v) for v in state.anchor_replay_state.get("indices_by_class", {}).values())),
        "generated_examples": int(len(state.generated_replay_state.get("global_labels", []))),
        "derpp_replay_updates": int(state.resources.get("derpp_replay_updates", 0)),
        "continual_metrics": continual,
        "final_metrics": {key: classification[key] for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")},
    }
    write_json(output_dir / "config.json", {**config.payload(), "config_sha256": config.sha256(), "code_sha256": code_hash, "namespace": namespace})
    write_json(output_dir / "environment.json", _environment(device, synthetic))
    write_json(output_dir / "class_order.json", {"class_order": order, "sha256": order_sha256(order)})
    write_json(output_dir / "task_supports.json", {"task_supports": supports})
    write_json(output_dir / "task_matrix.json", {"task_matrix": matrix, "task_supports": supports, "continual_metrics": continual})
    write_json(output_dir / "per_task_details.json", runtime["task_details"])
    write_csv(output_dir / "per_task_metrics.csv", runtime["task_rows"])
    write_csv(output_dir / "classification_report.csv", [
        {"arrival_id": arrival, "global_class_id": int(class_id), "precision": classification["per_class_precision"][arrival],
         "recall": classification["per_class_recall"][arrival], "f1": classification["per_class_f1"][arrival],
         "support": classification["support"][arrival]}
        for arrival, class_id in enumerate(order)
    ])
    confusion_rows = []
    for row_index, class_id in enumerate(order):
        row = {"true_global_class_id": int(class_id)}
        row.update({f"pred_{int(predicted)}": int(classification["confusion_matrix"][row_index][column]) for column, predicted in enumerate(order)})
        confusion_rows.append(row)
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    write_json(output_dir / "source_proportions.json", runtime["source_rows"])
    _atomic_trace(output_dir / "trace.jsonl", runtime["update_trace"])
    write_json(output_dir / "classifier_mode_trace.json", runtime.get("classifier_mode_trace", []))
    boundary_records = runtime.get("task_boundary_predictions", {})
    write_task_boundary_predictions(output_dir / "task_boundary_predictions.npz", boundary_records)
    boundary_stages = []
    for stage in range(1, 12):
        record = boundary_records[str(stage)]
        boundary_stages.append({
            "stage": stage,
            "test_example_count": len(record["y_true_global"]),
            "test_indices_sha256": _array_hash(np.asarray(record["test_indices"], dtype=np.int64)),
            "head_class_ids": [int(value) for value in record["head_class_ids"]],
            "evaluated_task_ids": list(range(1, stage + 1)),
        })
    write_json(output_dir / "task_boundary_prediction_manifest.json", {
        "schema_version": 1,
        "prediction_file": "task_boundary_predictions.npz",
        "arrays_per_stage": ["test_indices", "y_true_global", "y_pred_global", "head_class_ids"],
        "stages": boundary_stages,
    })
    write_jsonl(output_dir / "selection_records.jsonl", runtime.get("selection_records", []))
    write_json(output_dir / "scaler_history.json", runtime["scaler_history"])
    host_rss = int(process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        host_rss *= 1024
    resource_output = tracker.snapshot(artifact_root=output_dir, host_ram_bytes=host_rss)
    resource_output.update({
        "classifier_updates": result["classifier_updates"],
        "generator_updates": result["generator_steps"],
        "critic_steps": result["critic_steps"],
        "classifier_seconds": tracker.seconds.get("classifier", 0.0),
        "generator_seconds": tracker.seconds.get("generator", 0.0),
        "critic_seconds": tracker.seconds.get("critic", 0.0),
        "checkpoint_seconds": tracker.seconds.get("checkpoint", 0.0),
        "training_reproduction_claimed": False,
    })
    write_json(output_dir / "resource_metrics.json", resource_output)
    write_predictions(output_dir / "predictions.npz", y_true, y_pred, np.asarray(order, dtype=np.int64))
    model_payload = {"classifier": classifier.state_dict(), "generator": None if generator is None else generator.state_dict(), "critic": None if critic is None else critic.state_dict()}
    torch.save(model_payload, output_dir / "final_models.pt")
    write_json(output_dir / "full-result.json", result)
    write_json(output_dir / "status.json", {"status": "ARTIFACTS_WRITTEN_UNVALIDATED", "namespace": namespace})
    resource_output["artifact_storage_bytes"] = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    write_json(output_dir / "resource_metrics.json", resource_output)
    artifact_manifest = build_artifact_manifest(output_dir)
    write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return result


def _refresh_terminal_resource_metrics(output_dir: Path, tracker: ResourceTracker) -> None:
    """Publish final terminal-checkpoint timing without recursively checkpointing it."""
    metrics_path = output_dir / "resource_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["seconds"] = dict(tracker.seconds)
    payload["classifier_seconds"] = float(tracker.seconds.get("classifier", 0.0))
    payload["generator_seconds"] = float(tracker.seconds.get("generator", 0.0))
    payload["critic_seconds"] = float(tracker.seconds.get("critic", 0.0))
    payload["checkpoint_seconds"] = float(tracker.seconds.get("checkpoint", 0.0))
    payload["active_seconds"] = float(tracker.active_seconds)
    payload["peak_cuda_allocated_bytes"] = int(tracker.peak_cuda_allocated_bytes)
    payload["peak_cuda_reserved_bytes"] = int(tracker.peak_cuda_reserved_bytes)
    payload["timing_metadata"] = tracker.timing_metadata()
    payload["cuda_synchronization_calls"] = int(tracker.cuda_synchronization_calls)
    payload["cuda_synchronization_seconds"] = float(tracker.cuda_synchronization_seconds)
    payload["resource_instrumentation_seconds"] = float(tracker.instrumentation_seconds)
    write_json(metrics_path, payload)
    write_json(output_dir / "artifact_manifest.json", build_artifact_manifest(output_dir))


def _write_completion(state: RunState, config: RunConfig, output_dir: Path, registry: RunRegistry, run_id: str, synthetic: bool) -> dict:
    validation = validate_run_artifacts(
        output_dir, require_production=not synthetic, allow_pending=True, verify_hashes=False,
    )
    namespace = SYNTHETIC_NAMESPACE if synthetic else "PRODUCTION"
    final_status = "SYNTHETIC_VALIDATED_NON_REPORTABLE" if synthetic else "COMPLETED_VALIDATED"
    write_json(output_dir / "status.json", {"status": final_status, "namespace": namespace, "validated_at_unix": time.time()})
    write_json(output_dir / "artifact_manifest.json", build_artifact_manifest(output_dir))
    validation = validate_run_artifacts(output_dir, require_production=not synthetic)
    artifacts = {
        name: output_dir / name
        for name in (
            "artifact_manifest.json", "status.json", "full-result.json", "task_matrix.json",
            "predictions.npz", "resource_metrics.json", "config.json", "environment.json",
        )
    }
    if not registry.complete(run_id, artifacts):
        raise ValueError("run completion marker failed artifact revalidation")
    state.exports["validated_artifacts"] = validation
    return validation


def run_one(
    run_config: RunConfig,
    dataset_bundle,
    output_dir: str | Path,
    stop_hook=None,
    *,
    update_schedule_path: str | Path | None = None,
    update_counts: list[int] | None = None,
    synthetic: bool = False,
    device: str | torch.device = "cuda",
    drive_checkpoint_root: str | Path | None = None,
    local_checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one immutable arm; all synthetic CPU/GPU modes are non-reportable.

    Production requires an already NPZ-validated DatasetBundle and the exact
    Phase 1 proposed schedule, plus an explicit CUDA device. No GPU operation is
    started by module import, package validation, or synthetic execution.
    """
    if run_config.arm not in ARMS:
        raise ValueError("unsupported revision_R1 run arm")
    if dataset_bundle.dataset != run_config.dataset:
        raise ValueError("run config and dataset bundle identity differ")
    target_device = torch.device(device)
    if synthetic:
        if target_device.type not in {"cpu", "cuda"}:
            raise ValueError("synthetic validation supports CPU or CUDA devices only")
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("synthetic GPU validation requested but CUDA is unavailable")
        if update_counts is None or len(update_counts) != 11 or any(int(value) < 1 for value in update_counts):
            raise ValueError("synthetic validation requires 11 explicit positive update counts")
        update_counts = [int(value) for value in update_counts]
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
    else:
        from .data import require_memory_mapped_production_bundle

        require_memory_mapped_production_bundle(dataset_bundle)
        require_production_runtime(target_device)
        expected = DATASETS[run_config.dataset]
        if dataset_bundle.x_train.shape[1] != expected["feature_dim"]:
            raise ValueError("production feature dimension differs from known dataset reference")
        if dataset_bundle.train_metadata.get("md5") != expected["train"]["md5"] or dataset_bundle.test_metadata.get("md5") != expected["test"]["md5"]:
            raise ValueError("production dataset metadata does not match known checksum references")
        if update_schedule_path is None:
            raise ValueError("production requires the preserved proposed update schedule CSV")
        update_counts = _run_schedule(update_schedule_path, run_config, dataset_bundle.y_train)
        torch.cuda.reset_peak_memory_stats(target_device)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    execution_scope = "synthetic_validation" if synthetic else "production"
    code_hash = _code_sha256()
    run_id = (
        f"{run_config.dataset}_seed{run_config.seed}_{run_config.arm}_"
        f"{run_config.sha256()[:10]}_{code_hash[:8]}_{execution_scope}"
    )
    identity = {
        "runner_version": "revision_R1.0.2",
        "config_sha256": run_config.sha256(),
        "dataset": _dataset_identity(dataset_bundle, synthetic=synthetic),
        "code_sha256": code_hash,
        "schedule_sha256": run_config.update_schedule_sha256,
        "execution_scope": execution_scope,
        "run_id": run_id,
    }
    registry = RunRegistry(output / "registry")
    if registry.is_completed(run_id):
        validation = validate_run_artifacts(output, require_production=not synthetic)
        return {"status": "ALREADY_VALIDATED", "validated": True, **validation}
    lease = RunLease(output / ".run.lease.json", ttl_seconds=900)
    lease.acquire()
    try:
        attempt = registry.begin_attempt(run_id, identity=identity)
    except Exception:
        lease.release()
        raise
    streams = make_rng_streams(run_config.dataset, run_config.seed, run_config.arm)
    _seed_new_run(streams)
    order = list(streams.class_order)
    tasks = build_tasks(dataset_bundle.y_train, dataset_bundle.y_test, run_config.seed, run_config.task_sizes)
    supports = _task_supports(dataset_bundle, order, run_config.task_sizes)
    components = arm_components(run_config.arm)
    classifier = build_classifier(
        dataset_bundle.feature_dim, streams.classifier_init_seed, initial_classes=50,
        hidden=((8, 8, 4) if synthetic else run_config.classifier_hidden),
        embedding_dim=(8 if synthetic else run_config.classifier_embedding_dim),
        dropout=run_config.dropout, device=target_device,
    )
    classifier_initial_hash = initial_state_sha256(classifier)
    generator = critic = None
    if components["gan"]:
        generator, critic = _make_gan_models(dataset_bundle.feature_dim, run_config, streams.gan_seed, target_device, small=synthetic)
    checkpoint_base = Path(local_checkpoint_root) if local_checkpoint_root is not None else output / "checkpoints"
    checkpoints = checkpoint_base / "recovery"
    task_boundary_checkpoints = checkpoint_base / "task_boundaries"
    persist_task_boundary_duplicate = should_persist_task_boundary_duplicate(identity)
    if drive_checkpoint_root is not None:
        remote_recovery = Path(drive_checkpoint_root) / "recovery"
        remote_boundaries = Path(drive_checkpoint_root) / "task_boundaries"
        if list(remote_recovery.glob("generation-*.manifest.json")):
            mirror_checkpoint_generations(remote_recovery, checkpoints, identity, keep=2)
        if persist_task_boundary_duplicate and list(remote_boundaries.glob("generation-*.manifest.json")):
            mirror_checkpoint_generations(
                remote_boundaries, task_boundary_checkpoints, identity,
                keep=None, prune_destination=False,
            )
    state = None
    tracker = ResourceTracker()
    checkpoint_policy = CheckpointPolicy(run_config.checkpoint_interval_updates, run_config.checkpoint_interval_seconds)
    last_checkpoint_update = 0
    last_checkpoint_time = time.time()
    last_drive_sync = time.time()
    code_hash = identity["code_sha256"]
    classifier_optimizer = None
    generator_optimizer = None
    critic_optimizer = None

    if list(checkpoints.glob("generation-*.manifest.json")):
        loaded = load_latest_valid(checkpoints, identity)
        state = loaded.state
        if state.classifier_state:
            head_size = int(state.classifier_state["out.weight"].shape[0])
            if head_size > classifier.num_classes:
                classifier.expand(head_size, streams.classifier_init_seed)
            classifier.load_state_dict(state.classifier_state)
        if generator is not None:
            generator.load_state_dict(state.generator_state)
            critic.load_state_dict(state.critic_state)
        if state.resources.get("tracker"):
            tracker.load_state_dict(state.resources["tracker"])
            tracker.resume_active_clock()
        last_checkpoint_update = int(state.resources.get("total_optimizer_updates", 0))
        classifier_optimizer = None
        generator_optimizer = None
        critic_optimizer = None
        runtime_at_resume = state.extra.get("runtime", {})
        if state.cursor.phase == "classifier":
            classifier_optimizer = torch.optim.AdamW(
                classifier.parameters(), lr=run_config.classifier_lr,
                weight_decay=run_config.classifier_weight_decay,
            )
            if state.classifier_optimizer_state and runtime_at_resume.get("classifier_optimizer_task") == state.cursor.task_index:
                classifier_optimizer.load_state_dict(state.classifier_optimizer_state)
        if state.cursor.phase == "gan_critic" and generator is not None and critic is not None:
            generator_optimizer = torch.optim.Adam(
                generator.parameters(), lr=run_config.generator_lr, betas=run_config.gan_betas,
            )
            critic_optimizer = torch.optim.Adam(
                critic.parameters(), lr=run_config.critic_lr, betas=run_config.gan_betas,
            )
            if state.generator_optimizer_state:
                generator_optimizer.load_state_dict(state.generator_optimizer_state)
            if state.critic_optimizer_state:
                critic_optimizer.load_state_dict(state.critic_optimizer_state)
        # Constructors (including an expanded output head), model state loads,
        # and any phase optimizer state restoration all precede RNG restoration.
        _restore_rng(state.rng_state, streams)
    else:
        runtime = {
            "class_order": order,
            "class_order_sha256": order_sha256(order),
            "task_matrix": [[None] * 11 for _ in range(11)],
            "task_rows": [], "task_details": [], "source_rows": [],
            "update_trace": [], "scaler_history": [],
            "task_boundary_predictions": {}, "selection_records": [],
            "classifier_mode_trace": [],
            "task_supports": supports,
            "initial_state_sha256": classifier_initial_hash,
        }
        state = RunState(
            identity=identity,
            cursor=PhaseCursor(task_index=0, phase="scaler"),
            classifier_state=classifier.state_dict(),
            rng_state={},
            metrics={"task_matrix": runtime["task_matrix"]},
            resources={"classifier_updates": 0, "sampled_examples": 0, "generator_updates": 0, "critic_steps": 0, "derpp_replay_updates": 0, "total_optimizer_updates": 0},
            extra={"runtime": runtime, "initial_state_sha256": classifier_initial_hash},
        )
        state.extra.update(_initial_gan_torch_rng(streams.gan_seed, target_device))

    if state.cursor.phase == "complete":
        try:
            validation = _write_completion(state, run_config, output, registry, run_id, synthetic)
            return {"status": "SYNTHETIC_VALIDATED_NON_REPORTABLE" if synthetic else "COMPLETED_VALIDATED", "validated": True, **validation}
        finally:
            lease.release()

    tracker.resume_active_clock()
    runtime = state.extra["runtime"]
    runtime["class_order"] = order
    runtime.setdefault("task_matrix", [[None] * 11 for _ in range(11)])
    runtime.setdefault("task_rows", [])
    runtime.setdefault("task_details", [])
    runtime.setdefault("source_rows", [])
    runtime.setdefault("update_trace", [])
    runtime.setdefault("scaler_history", [])
    runtime.setdefault("task_boundary_predictions", {})
    runtime.setdefault("selection_records", [])
    runtime.setdefault("classifier_mode_trace", [])
    if not state.anchor_replay_state:
        state.anchor_replay_state = {"indices_by_class": {}, "targets_by_index": {}, "class_ids": []}
    if not state.generated_replay_state:
        state.generated_replay_state = {"x_raw": None, "global_labels": [], "stored_logits": [], "class_ids": []}
    rng_restored = bool(state.rng_state)
    if not rng_restored:
        state.rng_state = _capture_rng(streams)
    state.resources.setdefault("classifier_updates", 0)
    state.resources.setdefault("sampled_examples", 0)
    state.resources.setdefault("generator_updates", 0)
    state.resources.setdefault("critic_steps", 0)
    state.resources.setdefault("derpp_replay_updates", 0)
    state.resources.setdefault("total_optimizer_updates", 0)

    synthetic_gan_epochs = 1 if synthetic else run_config.gan_epochs
    synthetic_critic_steps = 1 if synthetic else run_config.critic_steps
    synthetic_candidates = 2 if synthetic else run_config.generated_candidates_per_class
    synthetic_generated_k = 1 if synthetic else run_config.generated_k
    cached_context = None
    cached_task = None

    def context_for(task_index: int):
        nonlocal cached_context, cached_task
        if cached_task == task_index and cached_context is not None:
            return cached_context
        task = tasks[task_index]
        current_indices = np.flatnonzero(np.isin(dataset_bundle.y_train, task.classes)).astype(np.int64)
        scaler = RunningScaler.from_state_dict(state.scaler_state) if state.scaler_state else RunningScaler()
        head_classes = _seen_classes(order, task_index)
        parts = _active_parts(state, dataset_bundle, task_index, current_indices)
        # Before the taskwise scaler update, expose only raw pools; transforming
        # with an unfitted scaler would violate the training-only fit ordering.
        mixed = build_mixed_pool(parts, scaler) if scaler.n > 0 and state.cursor.phase != "scaler" else None
        if components["gan"] and mixed is not None:
            # Full Hybrid and Generated-only GAN pools have the same ordered
            # current/anchor/generated rows as the classifier pool. Reuse its
            # lazy feature view instead of materializing a second full copy.
            gan_x, gan_y, gan_sources = mixed.x_raw, mixed.y_global, mixed.sources
        else:
            gan_x, gan_y, gan_sources = None, np.empty(0, dtype=np.int64), np.empty(0, dtype=str)
        cached_context = {
            "task": task, "current_indices": current_indices, "scaler": scaler,
            "head_classes": head_classes, "mixed": mixed,
            "gan_x_raw": gan_x, "gan_y": gan_y, "gan_sources": gan_sources,
        }
        cached_task = task_index
        return cached_context

    def capture_state():
        state.classifier_state = classifier.state_dict()
        if generator is not None:
            state.generator_state = generator.state_dict()
            state.critic_state = critic.state_dict()
        state.rng_state = _capture_rng(streams)
        state.resources["tracker"] = tracker.state_dict()

    def checkpoint(*, force: bool = False, kind: str = "recovery"):
        nonlocal last_checkpoint_update, last_checkpoint_time, last_drive_sync
        now = time.time()
        updates = int(state.resources.get("total_optimizer_updates", 0))
        due = force or checkpoint_policy.should_checkpoint(
            last_checkpoint_update=last_checkpoint_update, update=updates,
            last_checkpoint_time=last_checkpoint_time, now=now, phase_boundary=False,
        )
        if not due:
            return
        checkpoint_started = tracker.start_timer()
        try:
            capture_state()
            runtime["last_checkpoint_phase"] = state.cursor.phase
            runtime["last_checkpoint_task"] = state.cursor.task_index
            save_generation(checkpoints, state, identity, kind=kind)
            if drive_checkpoint_root is not None and (force or now - last_drive_sync >= run_config.drive_sync_interval_seconds):
                mirror_checkpoint_generations(checkpoints, Path(drive_checkpoint_root) / "recovery", identity, keep=2)
                last_drive_sync = now
            if kind == "task-boundary" and persist_task_boundary_duplicate:
                save_generation(
                    task_boundary_checkpoints, state, identity, kind="task-boundary",
                    retain_generations=None,
                )
                if drive_checkpoint_root is not None:
                    mirror_checkpoint_generations(
                        task_boundary_checkpoints, Path(drive_checkpoint_root) / "task_boundaries",
                        identity, keep=2, prune_destination=False,
                    )
            last_checkpoint_update = updates
            last_checkpoint_time = now
            if drive_checkpoint_root is not None and now - last_drive_sync >= run_config.drive_sync_interval_seconds:
                mirror_checkpoint_generations(checkpoints, Path(drive_checkpoint_root) / "recovery", identity, keep=2)
                mirror_checkpoint_generations(task_boundary_checkpoints, Path(drive_checkpoint_root) / "task_boundaries", identity, keep=2, prune_destination=False)
                last_drive_sync = now
        finally:
            tracker.finish_timer("checkpoint", checkpoint_started)

    if not list(checkpoints.glob("generation-*.manifest.json")):
        checkpoint(force=True, kind="initial")

    def safe_stop() -> bool:
        if stop_hook is None:
            return False
        return bool(stop_hook(state))

    try:
        while state.cursor.phase not in {"export", "complete"}:
            task_index = state.cursor.task_index
            if task_index >= 11:
                state.cursor.phase = "export"
                checkpoint(force=True, kind="phase-boundary")
                if safe_stop():
                    raise StopRequested("stop requested before final export")
                break
            context_started = tracker.start_timer()
            context = context_for(task_index)
            tracker.finish_timer("data", context_started)
            task = context["task"]
            phase_before = state.cursor.phase

            if phase_before == "scaler":
                new_scaler = context["scaler"]
                new_scaler.update_indexed(dataset_bundle.x_train, context["current_indices"])
                state.scaler_state = new_scaler.state_dict()
                if classifier.num_classes < len(context["head_classes"]):
                    classifier.expand(len(context["head_classes"]), streams.classifier_init_seed + task_index + 1)
                cached_context = None
                cached_task = None
                context = context_for(task_index)
                context["mixed"] = build_mixed_pool(
                    _active_parts(state, dataset_bundle, task_index, context["current_indices"]),
                    new_scaler,
                )
                if components["gan"]:
                    context["gan_x_raw"] = context["mixed"].x_raw
                    context["gan_y"] = context["mixed"].y_global
                    context["gan_sources"] = context["mixed"].sources
                state.classifier_optimizer_state = {}
                state.generator_optimizer_state = {}
                state.critic_optimizer_state = {}
                classifier_optimizer = None
                generator_optimizer = None
                critic_optimizer = None
                runtime["optimizer_task_index"] = task_index
                runtime["classifier_updates_this_task"] = 0
                runtime["loss_terms_this_task"] = {"current_ce": [], "replay_ce": [], "logit_mse": []}
                runtime["current_indices"] = context["current_indices"].tolist()
                runtime["new_classes"] = list(task.classes)
                runtime["seen_classes"] = context["head_classes"]
                runtime["source_pool_sizes"] = {name: int(np.count_nonzero(context["mixed"].sources == name)) for name in ("current", "anchor", "generated")}
                runtime["expected_source_proportions"] = expected_source_proportions(context["mixed"].y_global, context["mixed"].sources)
                runtime["gan_pool_sources"] = {}
                if components["gan"]:
                    pool_labels = context["gan_y"]
                    nbatches = int(np.ceil(len(pool_labels) / 256))
                    runtime["gan_orders"] = [streams.gan.permutation(len(pool_labels)).astype(np.int64) for _ in range(synthetic_gan_epochs)]
                    runtime["gan_batches_per_epoch"] = nbatches
                    runtime["gan_epochs_for_run"] = synthetic_gan_epochs
                    runtime["gan_critic_steps_for_run"] = synthetic_critic_steps
                    runtime["gan_candidates_for_run"] = synthetic_candidates
                    runtime["gan_generated_k_for_run"] = synthetic_generated_k
                    runtime["gan_pool_sources"] = {str(name): int(np.count_nonzero(context["gan_sources"] == name)) for name in sorted(set(context["gan_sources"].tolist()))}
                    state.cursor.phase = "gan_critic"
                else:
                    state.cursor.phase = "classifier"
                state.cursor.item_index = 0
                state.cursor.substep_index = 0
                state.cursor.update_index = 0
                runtime["scaler_history"].append({"task_id": task_index + 1, "sample_count": new_scaler.n, "scaler_sha256": new_scaler.sha256(), "training_indices_only": True})
                checkpoint(force=True, kind="phase-boundary")
                if safe_stop():
                    raise StopRequested("stop requested after taskwise scaler update")
                cached_context = None
                cached_task = None
                continue

            if phase_before == "gan_critic":
                nbatches = int(runtime["gan_batches_per_epoch"])
                total_batches = nbatches * int(runtime["gan_epochs_for_run"])
                if state.cursor.item_index >= total_batches:
                    state.cursor.phase = "classifier"
                    state.cursor.item_index = 0
                    state.cursor.substep_index = 0
                    checkpoint(force=True, kind="phase-boundary")
                    if safe_stop():
                        raise StopRequested("stop requested at GAN/classifier phase boundary")
                    continue
                epoch = state.cursor.item_index // nbatches
                batch_in_epoch = state.cursor.item_index % nbatches
                perm = runtime["gan_orders"][epoch]
                batch_indices = perm[batch_in_epoch * 256 : (batch_in_epoch + 1) * 256]
                if len(batch_indices) < 2:
                    state.cursor.item_index += 1
                    state.cursor.substep_index = 0
                    checkpoint()
                    if safe_stop():
                        raise StopRequested("stop requested after skipped singleton GAN minibatch")
                    continue
                x_std = context["scaler"].transform(context["gan_x_raw"][batch_indices])
                real = torch.as_tensor(x_std, dtype=torch.float32, device=target_device)
                labels = torch.as_tensor(context["gan_y"][batch_indices], dtype=torch.long, device=target_device)
                if generator_optimizer is None:
                    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=run_config.generator_lr, betas=run_config.gan_betas)
                    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=run_config.critic_lr, betas=run_config.gan_betas)
                    if state.generator_optimizer_state:
                        generator_optimizer.load_state_dict(state.generator_optimizer_state)
                    if state.critic_optimizer_state:
                        critic_optimizer.load_state_dict(state.critic_optimizer_state)
                    lifecycle = runtime.setdefault("optimizer_lifecycle_by_task", {}).setdefault(str(task_index + 1), {})
                    lifecycle.setdefault("critic_step_before_first_update", _max_optimizer_step(critic_optimizer))
                    lifecycle.setdefault("generator_step_before_first_update", _max_optimizer_step(generator_optimizer))
                if state.cursor.substep_index < int(runtime["gan_critic_steps_for_run"]):
                    with tracker.measure("critic"):
                        critic_loss = _with_gan_torch_rng(
                            state.extra,
                            lambda: critic_update(
                                generator, critic, real, labels, critic_optimizer,
                                z_dim=run_config.noise_dim, gp_weight=run_config.gradient_penalty_weight,
                                clip_norm=run_config.gradient_clip_norm,
                            ),
                            device=target_device,
                        )
                    tracker.add_steps("critic")
                    state.resources["critic_steps"] += 1
                    state.resources["total_optimizer_updates"] += 1
                    state.critic_optimizer_state = critic_optimizer.state_dict()
                    state.cursor.substep_index += 1
                    runtime.setdefault("gan_loss_trace", []).append({"task_id": task_index + 1, "kind": "critic", "loss": critic_loss, "minibatch": state.cursor.item_index, "substep": state.cursor.substep_index})
                else:
                    with tracker.measure("generator"):
                        generator_loss = _with_gan_torch_rng(
                            state.extra,
                            lambda: generator_update(
                                generator, critic, real, labels, generator_optimizer,
                                z_dim=run_config.noise_dim,
                                feature_matching_weight=run_config.feature_matching_weight,
                                clip_norm=run_config.gradient_clip_norm,
                            ),
                            device=target_device,
                        )
                    tracker.add_steps("generator")
                    state.resources["generator_updates"] += 1
                    state.resources["total_optimizer_updates"] += 1
                    state.generator_optimizer_state = generator_optimizer.state_dict()
                    state.cursor.item_index += 1
                    state.cursor.substep_index = 0
                    runtime.setdefault("gan_loss_trace", []).append({"task_id": task_index + 1, "kind": "generator", "loss": generator_loss, "minibatch": state.cursor.item_index - 1})
                checkpoint()
                if safe_stop():
                    raise StopRequested("stop requested at a checkpointable GAN optimizer substep")
                continue

            if phase_before == "classifier":
                classifier.train()
                _record_classifier_mode(runtime, classifier, phase="classifier_training", task_id=task_index + 1)
                updates_this_task = int(runtime["classifier_updates_this_task"])
                if updates_this_task >= int(update_counts[task_index]):
                    state.cursor.phase = "evaluation"
                    state.cursor.item_index = 0
                    checkpoint(force=True, kind="phase-boundary")
                    if safe_stop():
                        raise StopRequested("stop requested at classifier/evaluation phase boundary")
                    continue
                if not state.sampler_state or state.sampler_state.get("task_index") != task_index:
                    sampler = MixedPoolSampler(context["mixed"].y_global, context["mixed"].sources, streams.sampling)
                    state.sampler_state = sampler.state_dict()
                    state.sampler_state["task_index"] = task_index
                else:
                    sampler = MixedPoolSampler.from_state(context["mixed"].y_global, context["mixed"].sources, state.sampler_state)
                draw = sampler.draw_batch(256)
                state.sampler_state = sampler.state_dict()
                state.sampler_state["task_index"] = task_index
                streams.sampling.bit_generator.state = state.sampler_state["rng_state"]
                selected = draw.indices
                xb = torch.as_tensor(context["mixed"].x_standardized[selected], dtype=torch.float32, device=target_device)
                yb = torch.as_tensor(draw.labels, dtype=torch.long, device=target_device)
                source_list = draw.sources.tolist()
                if classifier_optimizer is None:
                    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=run_config.classifier_lr, weight_decay=run_config.classifier_weight_decay)
                    if state.classifier_optimizer_state and runtime.get("classifier_optimizer_task") == task_index:
                        classifier_optimizer.load_state_dict(state.classifier_optimizer_state)
                    lifecycle = runtime.setdefault("optimizer_lifecycle_by_task", {}).setdefault(str(task_index + 1), {})
                    lifecycle.setdefault("classifier_step_before_first_update", _max_optimizer_step(classifier_optimizer))
                with tracker.measure("classifier"):
                    logits = classifier(xb)
                    if run_config.arm == "derpp_matched":
                        stored, stored_mask, stored_ids = _class_targets(state, context["mixed"], selected, context["head_classes"])
                        terms = derpp_loss(
                            logits, yb, source_list,
                            torch.as_tensor(stored, dtype=logits.dtype, device=target_device),
                            stored_ids, context["head_classes"],
                            torch.as_tensor([class_id in context["head_classes"] for class_id in range(100)], dtype=torch.bool, device=target_device),
                            torch.as_tensor(stored_mask, dtype=torch.bool, device=target_device),
                            beta=1.0, alpha=0.5,
                        )
                        loss = terms.total
                        for key in ("current_ce", "replay_ce", "logit_mse"):
                            runtime["loss_terms_this_task"][key].append(float(getattr(terms, key).detach().cpu()))
                        if any(source == "anchor" for source in source_list):
                            state.resources["derpp_replay_updates"] += 1
                    else:
                        loss = classification_loss(
                            run_config.arm, logits, yb, source_list, None, [], context["head_classes"],
                            torch.as_tensor([class_id in context["head_classes"] for class_id in range(100)], dtype=torch.bool, device=target_device),
                        )
                    if not torch.isfinite(xb).all() or not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise ValueError("non-finite classifier input/logit/loss")
                    classifier_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), run_config.gradient_clip_norm)
                    classifier_optimizer.step()
                tracker.add_steps("classifier")
                state.resources["classifier_updates"] += 1
                state.resources["sampled_examples"] += 256
                state.resources["total_optimizer_updates"] += 1
                runtime["classifier_optimizer_task"] = task_index
                state.classifier_optimizer_state = classifier_optimizer.state_dict()
                runtime["classifier_updates_this_task"] += 1
                state.cursor.update_index += 1
                state.cursor.item_index = state.cursor.update_index
                source_draws = {name: int(np.count_nonzero(draw.sources == name)) for name in ("current", "anchor", "generated")}
                class_draws = {str(int(label)): int(count) for label, count in zip(*np.unique(draw.labels, return_counts=True))}
                runtime["update_trace"].append({
                    "task_id": task_index + 1, "update": updates_this_task + 1,
                    "batch_size": 256, "source_draws": source_draws,
                    "class_draws": class_draws, "loss": float(loss.detach().cpu()),
                    "sampler_cursor": int(state.sampler_state["cursor"]),
                })
                checkpoint()
                if safe_stop():
                    raise StopRequested("stop requested after fixed-size classifier update")
                continue

            if phase_before == "evaluation":
                classifier.eval()
                _record_classifier_mode(runtime, classifier, phase="evaluation", task_id=task_index + 1)
                evaluation_started = tracker.start_timer()
                all_true, all_pred = [], []
                stage_row = []
                stage_indices, stage_true, stage_pred = [], [], []
                for prior_task_index in range(task_index + 1):
                    prior_task = tasks[prior_task_index]
                    test_indices = np.flatnonzero(np.isin(dataset_bundle.y_test, prior_task.classes))
                    head_prediction = _predict_test_indices_batched(
                        classifier, dataset_bundle.x_test, test_indices,
                        context["scaler"], target_device,
                    )
                    global_prediction = np.asarray(context["head_classes"], dtype=np.int64)[head_prediction]
                    truth = np.asarray(dataset_bundle.y_test[test_indices], dtype=np.int64)
                    accuracy = float(np.mean(global_prediction == truth))
                    runtime["task_matrix"][task_index][prior_task_index] = accuracy
                    stage_row.append({"evaluated_task_id": prior_task_index + 1, "accuracy": accuracy, "support": len(truth)})
                    all_true.extend(truth.tolist())
                    all_pred.extend(global_prediction.tolist())
                    stage_indices.extend(test_indices.tolist())
                    stage_true.extend(truth.tolist())
                    stage_pred.extend(global_prediction.tolist())
                runtime.setdefault("task_boundary_predictions", {})[str(task_index + 1)] = {
                    "test_indices": np.asarray(stage_indices, dtype=np.int64),
                    "y_true_global": np.asarray(stage_true, dtype=np.int64),
                    "y_pred_global": np.asarray(stage_pred, dtype=np.int64),
                    "head_class_ids": np.asarray(context["head_classes"], dtype=np.int64),
                }
                metrics = classification_metrics(all_true, all_pred, context["head_classes"])
                row = {
                    "task_id": task_index + 1,
                    "classes_seen": len(context["head_classes"]),
                    "seen_accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "scaler_sha256": context["scaler"].sha256(),
                }
                old = [row for row in runtime["task_rows"] if int(row["task_id"]) != task_index + 1]
                runtime["task_rows"] = old + [row]
                runtime["task_details"] = [row for row in runtime["task_details"] if int(row["task_id"]) != task_index + 1] + [{"task_id": task_index + 1, "evaluations": stage_row, "metrics": metrics}]
                if task_index == 10:
                    final_indices = np.arange(len(dataset_bundle.y_test), dtype=np.int64)
                    final_pred_head = _predict_test_indices_batched(
                        classifier, dataset_bundle.x_test, final_indices,
                        context["scaler"], target_device,
                    )
                    state.predictions = {
                        "y_true_global": np.asarray(dataset_bundle.y_test, dtype=np.int64),
                        "y_pred_global": np.asarray(context["head_classes"], dtype=np.int64)[final_pred_head],
                    }
                tracker.finish_timer("evaluation", evaluation_started)
                state.metrics["task_matrix"] = runtime["task_matrix"]
                state.cursor.phase = "memory"
                state.cursor.item_index = 0
                checkpoint(force=True, kind="phase-boundary")
                if safe_stop():
                    raise StopRequested("stop requested after task evaluation")
                cached_context = None
                cached_task = None
                continue

            if phase_before == "memory":
                if runtime.get("memory_stage_task") == task_index:
                    raise RuntimeError("memory phase resumed without a persisted generation/selection cursor")
                if components["anchors"]:
                    by_class = {str(key): list(values) for key, values in state.anchor_replay_state.get("indices_by_class", {}).items()}
                    rng = np.random.RandomState((streams.anchor_seed + run_config.seed + task_index) % (2**32 - 1))
                    for class_id in task.classes:
                        class_indices = np.flatnonzero(dataset_bundle.y_train == int(class_id)).astype(np.int64)
                        quota = min(int(run_config.anchor_k), len(class_indices))
                        selected = class_indices if len(class_indices) <= quota else rng.choice(class_indices, size=quota, replace=False)
                        by_class[str(int(class_id))] = [int(value) for value in selected]
                    state.anchor_replay_state["indices_by_class"] = by_class
                runtime["memory_stage_task"] = task_index
                runtime["selection_records_this_task"] = []
                runtime["pending_generated_raw"] = []
                runtime["pending_generated_labels"] = []
                if components["generated"]:
                    means = {}
                    mean_sources = {}
                    new_set = set(int(value) for value in task.classes)
                    mean_rng = np.random.RandomState(run_config.seed + 999 + task_index)
                    for class_id in context["head_classes"]:
                        if class_id in new_set:
                            indices = np.flatnonzero(dataset_bundle.y_train == int(class_id)).astype(np.int64)
                            if len(indices) > 512:
                                indices = mean_rng.choice(indices, size=512, replace=False)
                            selected_x = np.asarray(dataset_bundle.x_train[indices], dtype=np.float32)
                            selected_y = np.full(len(indices), int(class_id), dtype=np.int64)
                            mean_sources[str(int(class_id))] = "current_real_training_rows"
                        else:
                            replay = state.generated_replay_state
                            positions = np.flatnonzero(np.asarray(replay.get("global_labels", []), dtype=np.int64) == int(class_id))
                            selected_x = np.asarray(replay["x_raw"], dtype=np.float32)[positions]
                            selected_y = np.full(len(positions), int(class_id), dtype=np.int64)
                            mean_sources[str(int(class_id))] = "carried_generated_replay"
                        if len(selected_x):
                            class_mean = class_mean_mid(
                                classifier, selected_x, selected_y, context["scaler"],
                                [int(class_id)], str(target_device),
                            )
                            means.update(class_mean)
                    if not means:
                        raise ValueError("generated replay selection has no class-mean support")
                    runtime["generation_classes"] = list(context["head_classes"])
                    runtime["generation_means"] = {str(class_id): value.detach().cpu().tolist() for class_id, value in means.items()}
                    runtime["generation_mean_sources"] = mean_sources
                    runtime["generation_model_sha256"] = initial_state_sha256(generator)
                    runtime.setdefault("selection_records", [])
                    state.cursor.phase = "generation"
                else:
                    state.cursor.phase = "refresh"
                state.cursor.item_index = 0
                state.cursor.substep_index = 0
                checkpoint(force=True, kind="phase-boundary")
                if safe_stop():
                    raise StopRequested("stop requested after anchor selection and before replay generation")
                continue

            if phase_before == "generation":
                generation_classes = runtime["generation_classes"]
                if state.cursor.item_index >= len(generation_classes):
                    pending_x = runtime["pending_generated_raw"]
                    pending_y = runtime["pending_generated_labels"]
                    state.generated_replay_state = {
                        "x_raw": np.concatenate(pending_x).astype(np.float32),
                        "global_labels": np.concatenate(pending_y).astype(np.int64).tolist(),
                        "stored_logits": [], "class_ids": [],
                        "source_mean_provenance": runtime["generation_mean_sources"],
                    }
                    runtime.pop("pending_generated_raw", None)
                    runtime.pop("pending_generated_labels", None)
                    runtime["memory_stage_task"] = task_index
                    state.cursor.phase = "refresh"
                    state.cursor.item_index = 0
                    state.cursor.substep_index = 0
                    checkpoint(force=True, kind="phase-boundary")
                    if safe_stop():
                        raise StopRequested("stop requested after candidate generation/selection phase")
                    continue
                class_id = int(generation_classes[state.cursor.item_index])
                mean = torch.as_tensor(runtime["generation_means"][str(class_id)], dtype=torch.float32, device=target_device)
                rng_before = _tensor_sha256(state.extra["gan_torch_cpu_rng"])
                generation_started = tracker.start_timer()
                candidates, distances = _with_gan_torch_rng(
                    state.extra,
                    lambda: generate_class_candidates(
                        generator, classifier, mean, class_id,
                        candidates_per_class=synthetic_candidates,
                        z_dim=run_config.noise_dim, device=str(target_device),
                    ),
                    device=target_device,
                )
                runtime["pending_candidate_features"] = candidates.detach().cpu().numpy().astype(np.float32)
                runtime["pending_candidate_distances"] = distances.detach().cpu().tolist()
                runtime["pending_candidate_class_id"] = class_id
                runtime["pending_candidate_record"] = {
                    "candidate_rng_start_sha256": rng_before,
                    "candidate_rng_end_sha256": _tensor_sha256(state.extra["gan_torch_cpu_rng"]),
                    "generator_state_sha256": runtime["generation_model_sha256"],
                    "scaler_sha256": context["scaler"].sha256(),
                    "mean_source": runtime["generation_mean_sources"][str(class_id)],
                    "task_id": task_index + 1,
                }
                tracker.finish_timer("generation", generation_started)
                state.cursor.phase = "selection"
                state.cursor.substep_index = 0
                checkpoint()
                if safe_stop():
                    raise StopRequested("stop requested after class candidate generation")
                continue

            if phase_before == "selection":
                if state.cursor.substep_index == 0:
                    class_id = int(runtime["pending_candidate_class_id"])
                    selection_started = tracker.start_timer()
                    features = torch.as_tensor(runtime["pending_candidate_features"], dtype=torch.float32, device=target_device)
                    distances = torch.as_tensor(runtime["pending_candidate_distances"], dtype=torch.float32, device=target_device)
                    selected, record = select_candidate_features(
                        features, distances, class_id=class_id,
                        quota=synthetic_generated_k, diversity_lambda=run_config.diversity_lambda,
                    )
                    raw_selected = context["scaler"].inverse(selected)
                    runtime["pending_generated_raw"].append(np.asarray(raw_selected, dtype=np.float32))
                    runtime["pending_generated_labels"].append(np.full(len(raw_selected), class_id, dtype=np.int64))
                    record.update(runtime["pending_candidate_record"])
                    record["candidate_features_sha256"] = hashlib.sha256(
                        np.asarray(runtime["pending_candidate_features"], dtype=np.float32).tobytes()
                    ).hexdigest()
                    record["selected_candidate_indices"] = record["selected_candidate_indices"]
                    runtime["selection_records_this_task"].append(record)
                    runtime["selection_records"].append(record)
                    tracker.finish_timer("selection", selection_started)
                    for name in ("pending_candidate_features", "pending_candidate_distances", "pending_candidate_class_id", "pending_candidate_record"):
                        runtime.pop(name, None)
                    state.cursor.substep_index = 1
                    checkpoint()
                    if safe_stop():
                        raise StopRequested("stop requested after class replay selection")
                    continue
                state.cursor.item_index += 1
                state.cursor.substep_index = 0
                state.cursor.phase = "generation"
                checkpoint()
                if safe_stop():
                    raise StopRequested("stop requested after selection cursor advance")
                continue

            if phase_before == "refresh":
                _prepare_logits(state, classifier, dataset_bundle, context["scaler"], context["head_classes"], target_device)
                expected = runtime["expected_source_proportions"]
                sampler_state = state.sampler_state
                realized_counts = {name: int(sampler_state.get("realized_source_counts", {}).get(name, 0)) for name in ("current", "anchor", "generated")}
                sampled_examples = int(update_counts[task_index]) * 256
                source_row = {
                    "task_id": task_index + 1,
                    "pool_unique_examples": runtime["source_pool_sizes"],
                    "expected_proportions": expected,
                    "realized_draws": realized_counts,
                    "realized_proportions": {name: value / sampled_examples for name, value in realized_counts.items()},
                    "sampled_examples": sampled_examples,
                    "replacement": True,
                    "sampler": "mixed_pool_inverse_class_frequency",
                    "update_count": int(update_counts[task_index]),
                    "source_first_balancing": False,
                }
                runtime["source_rows"] = [row for row in runtime["source_rows"] if int(row["task_id"]) != task_index + 1] + [source_row]
                details = next(row for row in runtime["task_details"] if int(row["task_id"]) == task_index + 1)
                details["sampler_and_replay"] = source_row
                details["classifier_update_budget"] = int(update_counts[task_index])
                details["generator_training_pool_sources"] = runtime.get("gan_pool_sources", {})
                details["loss_terms"] = {
                    key: float(np.mean(values)) if values else 0.0
                    for key, values in runtime["loss_terms_this_task"].items()
                }
                details["optimizer_lifecycle"] = runtime.get("optimizer_lifecycle_by_task", {}).get(str(task_index + 1), {})
                details["gan_optimizer_steps_this_task"] = {
                    "generator": sum(row.get("kind") == "generator" and row.get("task_id") == task_index + 1 for row in runtime.get("gan_loss_trace", [])),
                    "critic": sum(row.get("kind") == "critic" and row.get("task_id") == task_index + 1 for row in runtime.get("gan_loss_trace", [])),
                }
                details["generation_class_mean_sources"] = runtime.get("generation_mean_sources", {})
                details["selection_record_count"] = len(runtime.get("selection_records_this_task", []))
                state.cursor.task_index += 1
                state.cursor.phase = "scaler" if state.cursor.task_index < 11 else "export"
                state.cursor.item_index = 0
                state.cursor.substep_index = 0
                state.cursor.update_index = 0
                state.sampler_state = {}
                runtime.pop("gan_orders", None)
                runtime.pop("gan_batches_per_epoch", None)
                runtime.pop("gan_pool_sources", None)
                runtime.pop("generation_classes", None)
                runtime.pop("generation_means", None)
                runtime.pop("generation_mean_sources", None)
                runtime.pop("generation_model_sha256", None)
                runtime.pop("selection_records_this_task", None)
                runtime.pop("memory_stage_task", None)
                cached_context = None
                cached_task = None
                checkpoint(force=True, kind="task-boundary")
                if safe_stop():
                    raise StopRequested("stop requested after task memory and replay update")
                continue

        if state.cursor.phase == "export":
            classifier.eval()
            if state.predictions.get("y_true_global") is None:
                raise ValueError("final test predictions were not produced")
            _export_run(state, run_config, dataset_bundle, output, target_device, synthetic, runtime["initial_state_sha256"], code_hash, supports, tracker, classifier, generator, critic)
            state.cursor.phase = "complete"
            checkpoint(force=True, kind="complete")
            _refresh_terminal_resource_metrics(output, tracker)
        validation = _write_completion(state, run_config, output, registry, run_id, synthetic)
        return {
            "status": "SYNTHETIC_VALIDATED_NON_REPORTABLE" if synthetic else "COMPLETED_VALIDATED",
            "validated": True,
            "namespace": SYNTHETIC_NAMESPACE if synthetic else "PRODUCTION",
            "initial_state_sha256": runtime["initial_state_sha256"],
            "classifier_updates": int(state.resources["classifier_updates"]),
            "sampled_examples": int(state.resources["sampled_examples"]),
            "generator_steps": int(state.resources["generator_updates"]),
            "critic_steps": int(state.resources["critic_steps"]),
            "anchor_examples": sum(len(values) for values in state.anchor_replay_state.get("indices_by_class", {}).values()),
            "generated_examples": len(state.generated_replay_state.get("global_labels", [])),
            "derpp_replay_updates": int(state.resources["derpp_replay_updates"]),
            "task_matrix_shape": [11, 11],
            **validation,
        }
    except StopRequested as exc:
        tracker.pause_active_clock()
        state.resources["tracker"] = tracker.state_dict()
        checkpoint(force=True, kind="interrupted-resumable")
        write_json(output / "status.json", {"status": "INTERRUPTED_RESUMABLE", "reason": str(exc), "cursor": state.cursor.__dict__})
        registry.fail_attempt(run_id, str(exc))
        return {"status": "INTERRUPTED_RESUMABLE", "cursor": state.cursor.__dict__, "classifier_updates": int(state.resources["classifier_updates"])}
    except Exception as exc:
        try:
            checkpoint(force=True, kind="failed-recovery")
        finally:
            registry.fail_attempt(run_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        tracker.pause_active_clock()
        lease.release()
