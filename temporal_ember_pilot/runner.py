"""Independent causal temporal runner using the vendored revision_R1 methods."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import time
import random

from .checkpoint import load_latest_checkpoint, save_checkpoint, sha256_file
from .config import (
    ANCHOR_K, ARMS, DEFAULT_OUTPUT_ROOT, EXPERIMENT_ID, FEATURE_DIM,
    FROZEN_SCHEDULE_SHA256, GAN_CANDIDATES_PER_CLASS, GENERATED_K, PilotRunConfig, SEEDS,
    validate_run_matrix,
)
from .data import FeatureStore, load_manifest, materialize_feature_store
from .engine import (
    capture_rng_state, planned_pool_counts, planned_updates, restore_rng_state,
    run_window,
)
from .progress import ProgressLine, ProgressRegistry
from .stream import WINDOW_IDS, window_partitions

_FAMILY_CACHE: dict[str, dict[str, str]] = {}
_TRAIN_FAMILY_CACHE: dict[str, dict[tuple[int, str], list[str]]] = {}


def _family_by_id(manifest: dict) -> dict[str, str]:
    key = str(manifest["manifest_sha256"])
    if key not in _FAMILY_CACHE:
        _FAMILY_CACHE[key] = {row["sha256"]: row["family"] for row in manifest["rows"]}
    return _FAMILY_CACHE[key]


def _train_ids_by_window_family(manifest: dict) -> dict[tuple[int, str], list[str]]:
    key = str(manifest["manifest_sha256"])
    if key not in _TRAIN_FAMILY_CACHE:
        values: dict[tuple[int, str], list[str]] = {}
        for row in manifest["rows"]:
            if row["split"] == "train":
                values.setdefault((int(row["window_index"]), row["family"]), []).append(row["sha256"])
        for ids in values.values():
            ids.sort()
        _TRAIN_FAMILY_CACHE[key] = values
    return _TRAIN_FAMILY_CACHE[key]


def _canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(payload, target, sort_keys=True, indent=2, ensure_ascii=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def freeze_budget_schedule(manifest: dict, output_root: str | Path) -> dict:
    """Compute and persist the seed/arm-invariant Full Hybrid budget in advance."""
    if manifest.get("manifest_sha256") != _canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("manifest hash changed before budget freeze")
    family_admission = manifest["family_admission_index"]
    budgets = []
    for window_index, window_id in enumerate(manifest["window_ids"]):
        parts = window_partitions(manifest, window_id)
        prior_support = {}
        for family, introduced_at in family_admission.items():
            if int(introduced_at) < window_index:
                admission_window = manifest["window_ids"][int(introduced_at)]
                prior_support[family] = manifest["family_support_by_window"][admission_window]["train"][family]
        pool_counts = planned_pool_counts(len(parts["train_ids"]), prior_support)
        budgets.append({
            "window_id": window_id,
            "window_index": window_index,
            "known_families_before": len(prior_support),
            "new_families_this_window": len(parts["admitted_families_this_window"]),
            "current_train_rows": pool_counts["current"],
            "planned_anchor_rows": pool_counts["anchor"],
            "planned_generated_rows": pool_counts["generated"],
            "planned_full_hybrid_pool": pool_counts["full_hybrid"],
            "classifier_updates_all_arms": planned_updates(len(parts["train_ids"]), prior_support),
            "classifier_epochs_equivalent": 3,
            "batch_size": 256,
            "anchor_quota_per_prior_family": ANCHOR_K,
            "generated_quota_per_prior_family": GENERATED_K,
        })
    schedule = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "policy": "3 * ceil(planned Full Hybrid pool / 256), frozen before any run; actual generated count never changes update count",
        "windows": budgets,
    }
    schedule["schedule_sha256"] = _canonical_sha256(schedule)
    path = Path(output_root) / "frozen_budget_schedule.json"
    _atomic_json(path, schedule)
    return schedule


def load_or_freeze_budget_schedule(manifest: dict, output_root: str | Path) -> dict:
    path = Path(output_root) / "frozen_budget_schedule.json"
    release_schedule_path = Path(__file__).parent / "frozen_budget_schedule.json"
    release_schedule = json.loads(release_schedule_path.read_text(encoding="utf-8"))
    release_sha = release_schedule.get("schedule_sha256")
    if (release_sha != FROZEN_SCHEDULE_SHA256
            or _canonical_sha256({k: v for k, v in release_schedule.items() if k != "schedule_sha256"}) != release_sha):
        raise ValueError("packaged frozen classifier budget schedule checksum failed")
    if release_schedule.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("packaged frozen classifier budget schedule belongs to a different split manifest")
    if path.exists():
        schedule = json.loads(path.read_text(encoding="utf-8"))
        recorded = schedule.get("schedule_sha256")
        if _canonical_sha256({k: v for k, v in schedule.items() if k != "schedule_sha256"}) != recorded:
            raise ValueError("frozen classifier budget schedule checksum failed")
        if recorded != FROZEN_SCHEDULE_SHA256 or schedule != release_schedule:
            raise ValueError("saved classifier budget schedule differs from the release-frozen schedule")
        return schedule
    _atomic_json(path, release_schedule)
    return release_schedule


def _torch_numpy():
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("Run/Resume requires NumPy and PyTorch in the Colab runtime") from error
    return np, torch


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    return value


def _make_gan_models(feature_dim: int, defaults, seed: int, device):
    import torch
    from .vendor.revision_r1.models import V155ConditionalGenerator, V155ProjectionCritic

    target = torch.device(device)
    devices = [target.index or torch.cuda.current_device()] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if target.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        generator = V155ConditionalGenerator(
            feature_dim, num_classes=100, z_dim=defaults.noise_dim,
            hidden=256,
        ).to(target)
        critic = V155ProjectionCritic(feature_dim, num_classes=100, hidden_dim=256).to(target)
    return generator, critic


def _rng_streams(config: PilotRunConfig):
    from .vendor.revision_r1.rng import make_rng_streams

    return make_rng_streams("ember", config.seed, config.arm)


def _rng_stream_identity(streams) -> dict:
    return {
        "class_order_policy": "not_used; family order is fixed by chronological admissions",
        "classifier_init_seed": int(streams.classifier_init_seed),
        "dropout_seed": int(streams.dropout_seed),
        "anchor_seed": int(streams.anchor_seed),
        "gan_seed": int(streams.gan_seed),
        "policy": "revision_R1 named streams; family order follows frozen temporal admissions",
    }


def _initialize_gan_torch_rng(seed: int, device, torch) -> dict:
    target = torch.device(device)
    devices = [target.index or torch.cuda.current_device()] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if target.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        result = {"cpu": torch.get_rng_state().cpu()}
        if target.type == "cuda":
            result["cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return result


def _with_gan_torch_rng(state: dict, torch, device, operation):
    """Advance a separate GAN Torch RNG while preserving classifier dropout RNG."""
    target = torch.device(device)
    classifier_cpu = torch.get_rng_state()
    classifier_cuda = torch.cuda.get_rng_state_all() if target.type == "cuda" else None
    gan_state = state.get("gan_torch_rng")
    if not gan_state or "cpu" not in gan_state:
        raise ValueError("checkpoint is missing its independent GAN Torch RNG")
    torch.set_rng_state(gan_state["cpu"].cpu())
    if target.type == "cuda":
        cuda_state = gan_state.get("cuda")
        if cuda_state is None or len(cuda_state) != torch.cuda.device_count():
            raise ValueError("GAN CUDA RNG checkpoint does not match the active runtime")
        torch.cuda.set_rng_state_all(cuda_state)
    try:
        result = operation()
        gan_state["cpu"] = torch.get_rng_state().cpu()
        if target.type == "cuda":
            gan_state["cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
        return result
    finally:
        torch.set_rng_state(classifier_cpu)
        if target.type == "cuda":
            torch.cuda.set_rng_state_all(classifier_cuda)


def _build_models(config: PilotRunConfig, state: dict, device):
    import torch
    from .vendor.revision_r1.models import build_classifier

    defaults = config.verified_defaults
    identity = state["run_metadata"]["rng_stream_identity"]
    class_count = len(state["class_map"])
    classifier = build_classifier(
        FEATURE_DIM, seed=identity["classifier_init_seed"], initial_classes=class_count,
        hidden=defaults.classifier_hidden, embedding_dim=defaults.classifier_embedding_dim,
        dropout=defaults.dropout, device=device,
    )
    generator, critic = _make_gan_models(FEATURE_DIM, defaults, identity["gan_seed"], device)
    classifier.load_state_dict(state["training_state"]["classifier"])
    generator.load_state_dict(state["training_state"]["generator"])
    critic.load_state_dict(state["training_state"]["critic"])
    if classifier.num_classes != class_count:
        raise ValueError("saved classifier width and causal class map differ")
    return classifier, generator, critic, defaults


def _new_state(config: PilotRunConfig, manifest: dict, schedule: dict, device) -> dict:
    np, torch = _torch_numpy()
    import lief
    import sklearn
    from .vendor.revision_r1.models import build_classifier

    streams = _rng_streams(config)
    random.seed(streams.dropout_seed)
    np.random.seed(streams.dropout_seed % (2**32 - 1))
    torch.manual_seed(streams.dropout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(streams.dropout_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    defaults = config.verified_defaults
    jan_families = manifest["admissions"][WINDOW_IDS[0]]
    class_map = {family: class_id for class_id, family in enumerate(jan_families)}
    classifier = build_classifier(
        FEATURE_DIM, seed=streams.classifier_init_seed, initial_classes=len(class_map),
        hidden=defaults.classifier_hidden, embedding_dim=defaults.classifier_embedding_dim,
        dropout=defaults.dropout, device=device,
    )
    generator, critic = _make_gan_models(FEATURE_DIM, defaults, streams.gan_seed, device)
    rng_generators = {
        "classifier_sampler": streams.sampling,
        "anchor_selection": streams.anchor_selection,
        "gan_sampler": streams.gan,
    }
    state = {
        "schema_version": 1,
        "run_metadata": {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "arm": config.arm,
            "seed": config.seed,
            "manifest_sha256": manifest["manifest_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "device_type": torch.device(device).type,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "scikit_learn": sklearn.__version__,
                "lief": str(lief.__version__),
            },
            "verified_config": defaults.payload(),
            "rng_stream_identity": _rng_stream_identity(streams),
        },
        "stream_cursor": {
            "next_window_index": 0, "last_completed_window": -1,
            "window_id": None, "phase": "initial", "batch_index": 0,
        },
        "training_state": {
            "classifier": _cpu_tree(classifier.state_dict(), torch),
            "generator": _cpu_tree(generator.state_dict(), torch),
            "critic": _cpu_tree(critic.state_dict(), torch),
        },
        "optimizer_state": {"classifier": {}, "generator": {}, "critic": {}},
        "scaler_state": None,
        "anchor_state": {"sample_ids_by_family": {}},
        "class_map": class_map,
        "gan_state": {
            "generated_x_raw": np.empty((0, FEATURE_DIM), dtype=np.float32),
            "generated_class_ids": np.empty((0,), dtype=np.int64),
        },
        "numpy_rng_generators": rng_generators,
        "gan_torch_rng": _initialize_gan_torch_rng(streams.gan_seed, device, torch),
        "rng_state": None,
        "window_records": [],
        "evaluations": {},
    }
    state["rng_state"] = capture_rng_state(rng_generators)
    return state


def _metric_record(sample_ids, manifest, class_map, model, scaler, feature_store, *, device, labels=None):
    import numpy as np
    import torch
    from .vendor.revision_r1.metrics import classification_metrics

    ids = list(sample_ids)
    if not ids:
        return {"macro_f1": None, "sample_count": 0, "scoring_ids_sha256": hashlib.sha256(b"").hexdigest(),
                "per_family_sample_counts": {}, "per_family_f1": {}}
    family_for = _family_by_id(manifest)
    try:
        truth = np.asarray([class_map[family_for[sample_id]] for sample_id in ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"evaluation sample has no class known before its cutoff: {error.args[0]}") from error
    x_raw = feature_store.get(ids)
    x = torch.as_tensor(scaler.transform(x_raw), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(x).argmax(1).detach().cpu().numpy().astype(np.int64)
    finally:
        model.train(was_training)
    metric_labels = sorted(class_map.values())
    result = classification_metrics(truth, prediction, metric_labels)
    family_by_class = {class_id: family for family, class_id in class_map.items()}
    target_labels = metric_labels if labels is None else list(labels)
    class_position = {class_id: position for position, class_id in enumerate(metric_labels)}
    per_family_counts = {}
    per_family_f1 = {}
    for class_id in target_labels:
        index = class_position[int(class_id)]
        family = family_by_class[class_id]
        per_family_counts[family] = int(result["support"][index])
        per_family_f1[family] = float(result["per_class_f1"][index])
    target_f1 = [per_family_f1[family_by_class[int(class_id)]] for class_id in target_labels]
    return {
        "macro_f1": float(np.mean(target_f1)),
        "sample_count": len(ids),
        "scoring_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest(),
        "per_family_sample_counts": per_family_counts,
        "per_family_f1": per_family_f1,
        "metric_family_count": len(target_labels),
    }


def _current_train_ids(manifest: dict, index: int) -> list[str]:
    return [row["sha256"] for row in manifest["rows"]
            if row["window_index"] == index and row["split"] == "train"]


def _update_scaler(state: dict, manifest: dict, index: int, parts: dict, store):
    import numpy as np
    from .vendor.revision_r1.preprocessing import RunningScaler

    scaler = RunningScaler() if state["scaler_state"] is None else RunningScaler.from_state_dict(state["scaler_state"])
    current_ids = _current_train_ids(manifest, index)
    positions = np.asarray([store.row_index[sample_id] for sample_id in current_ids], dtype=np.int64)
    scaler.update_indexed(store.features, positions, chunk_rows=4096)
    if scaler.n != len(parts["fit_ids"]):
        raise ValueError(f"scaler fit count {scaler.n} differs from causal training rows {len(parts['fit_ids'])}")
    expected_digest = hashlib.sha256("\n".join(parts["fit_ids"]).encode("ascii")).hexdigest()
    state["scaler_fit_provenance"] = {
        "through_window": parts["window_id"],
        "sample_count": scaler.n,
        "sample_ids_sha256": expected_digest,
        "training_rows_only": True,
    }
    state["scaler_state"] = scaler.state_dict()
    return scaler


def _load_run_state(config, manifest, schedule, run_dir, device):
    loaded = load_latest_checkpoint(run_dir)
    if loaded is None:
        return _new_state(config, manifest, schedule, device)
    _, state = loaded
    meta = state["run_metadata"]
    expected = (config.experiment_id, config.run_id, config.arm, config.seed,
                manifest["manifest_sha256"], schedule["schedule_sha256"])
    observed = (meta.get("experiment_id"), meta.get("run_id"), meta.get("arm"), meta.get("seed"),
                meta.get("manifest_sha256"), meta.get("schedule_sha256"))
    if observed != expected:
        raise ValueError(f"checkpoint belongs to a different run/configuration: {observed}")
    expected_rng_identity = _rng_stream_identity(_rng_streams(config))
    if meta.get("rng_stream_identity") != expected_rng_identity:
        raise ValueError("checkpoint named RNG stream identity differs from this run")
    import torch
    if meta.get("device_type") != torch.device(device).type:
        raise ValueError("resume device type differs from the saved run; resume with the same CPU/GPU mode")
    restore_rng_state(state["rng_state"], state.get("numpy_rng_generators", {}))
    return state


def _pool_for_arm(config, state, parts, feature_store, manifest, class_map):
    import numpy as np

    current_ids = parts["train_ids"]
    current_x = feature_store.get(current_ids)
    family_for = _family_by_id(manifest)
    current_y = np.asarray([class_map[family_for[sample_id]] for sample_id in current_ids], dtype=np.int64)
    x_parts = [current_x]
    y_parts = [current_y]
    source_parts = [np.full(len(current_ids), "current", dtype="U9")]
    anchor_count = 0
    generated_count = 0
    if config.arm in {"full_hybrid", "anchor_only"}:
        anchor_ids = [sample_id for family in sorted(state["anchor_state"]["sample_ids_by_family"])
                      for sample_id in state["anchor_state"]["sample_ids_by_family"][family]]
        if anchor_ids:
            anchor_x = feature_store.get(anchor_ids)
            anchor_y = np.asarray([class_map[family_for[sample_id]] for sample_id in anchor_ids], dtype=np.int64)
            x_parts.append(anchor_x)
            y_parts.append(anchor_y)
            source_parts.append(np.full(len(anchor_ids), "anchor", dtype="U9"))
            anchor_count = len(anchor_ids)
    if config.arm == "full_hybrid":
        generated_x = np.asarray(state["gan_state"]["generated_x_raw"], dtype=np.float32)
        generated_y = np.asarray(state["gan_state"]["generated_class_ids"], dtype=np.int64)
        if len(generated_x):
            x_parts.append(generated_x)
            y_parts.append(generated_y)
            source_parts.append(np.full(len(generated_x), "generated", dtype="U9"))
            generated_count = len(generated_x)
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(source_parts), anchor_count, generated_count


def _fit_gan_inner(config, defaults, generator, critic, x_raw, y_global, scaler, state,
                   device, run_id, window_id, window_index, progress_line, progress_registry):
    import numpy as np
    import torch
    from .vendor.revision_r1.generative import critic_update, generator_update

    started = time.perf_counter()
    compute_seconds = 0.0
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=defaults.generator_lr, betas=defaults.gan_betas)
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=defaults.critic_lr, betas=defaults.gan_betas)
    x_pool = np.asarray(x_raw, dtype=np.float32)
    y_pool = np.asarray(y_global, dtype=np.int64)
    batches_per_epoch = math.ceil(len(y_pool) / defaults.gan_batch_size)
    rng = state["numpy_rng_generators"]["gan_sampler"]
    epoch_orders = [rng.permutation(len(y_pool)) for _ in range(defaults.gan_epochs)]
    for epoch, order in enumerate(epoch_orders, start=1):
        for batch_number, offset in enumerate(range(0, len(order), defaults.gan_batch_size), start=1):
            indices = order[offset:offset + defaults.gan_batch_size]
            if len(indices) < 2:
                continue
            real = torch.as_tensor(scaler.transform(x_pool[indices]), dtype=torch.float32, device=device)
            labels = torch.as_tensor(y_pool[indices], dtype=torch.long, device=device)
            for _ in range(defaults.critic_steps):
                step_started = time.perf_counter()
                critic_update(generator, critic, real, labels, optimizer_c,
                              z_dim=defaults.noise_dim, gp_weight=defaults.gradient_penalty_weight,
                              clip_norm=defaults.gradient_clip_norm)
                compute_seconds += time.perf_counter() - step_started
            step_started = time.perf_counter()
            generator_update(generator, critic, real, labels, optimizer_g,
                             z_dim=defaults.noise_dim,
                             feature_matching_weight=defaults.feature_matching_weight,
                             clip_norm=defaults.gradient_clip_norm)
            compute_seconds += time.perf_counter() - step_started
            progress_line.update(run_id, window_id, "gan", epoch, defaults.gan_epochs,
                                 batch_number, batches_per_epoch)
            progress_registry.update(run_id, window_id, "gan", epoch=epoch,
                                     epochs=defaults.gan_epochs, batch=batch_number,
                                     batches=batches_per_epoch)
        progress_line.finish_phase()
    return optimizer_g, optimizer_c, compute_seconds, time.perf_counter() - started


def _fit_gan(config, defaults, generator, critic, x_raw, y_global, scaler, state,
             device, run_id, window_id, window_index, progress_line, progress_registry):
    _, torch = _torch_numpy()
    return _with_gan_torch_rng(
        state, torch, device,
        lambda: _fit_gan_inner(
            config, defaults, generator, critic, x_raw, y_global, scaler, state,
            device, run_id, window_id, window_index, progress_line, progress_registry,
        ),
    )


def _run_one(config, manifest, schedule, store, output_root, *, device, progress_line, progress_registry):
    np, torch = _torch_numpy()
    from .vendor.revision_r1.generative import class_mean_mid, select_generated
    from .vendor.revision_r1.losses import classification_loss
    from .vendor.revision_r1.preprocessing import RunningScaler
    from .vendor.revision_r1.sampling import MixedPoolSampler

    run_dir = Path(output_root) / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_run_state(config, manifest, schedule, run_dir, device)
    result_path = run_dir / "result.json"
    latest = load_latest_checkpoint(run_dir)
    if latest and state["stream_cursor"]["next_window_index"] == len(WINDOW_IDS) and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        latest_sha = sha256_file(latest[0])
        if (result.get("status") == "completed" and result.get("manifest_sha256") == manifest["manifest_sha256"]
                and result.get("schedule_sha256") == schedule["schedule_sha256"]
                and result.get("final_checkpoint_sha256") == latest_sha):
            return {"run_id": config.run_id, "status": "skipped_finished"}

    classifier, generator, critic, defaults = _build_models(config, state, device)
    progress_line = progress_line or ProgressLine()
    for window_index in range(int(state["stream_cursor"]["next_window_index"]), len(WINDOW_IDS)):
        window_id = WINDOW_IDS[window_index]
        parts = window_partitions(manifest, window_id)
        budget = schedule["windows"][window_index]
        if len(parts["train_ids"]) != int(budget["current_train_rows"]):
            raise ValueError("window train IDs changed after budget freeze")
        restore_rng_state(state["rng_state"], state.get("numpy_rng_generators", {}))

        if window_index > 0:
            progress_line.update(config.run_id, window_id, "pre_update_evaluation", 1, 1, 1, 1)
            progress_registry.update(config.run_id, window_id, "pre_update_evaluation", epoch=1, epochs=1, batch=1, batches=1)
            pre_metric = _metric_record(parts["scoring_ids"], manifest, state["class_map"],
                                        classifier, RunningScaler.from_state_dict(state["scaler_state"]),
                                        store, device=device)
            state["evaluations"][window_id] = {
                "phase": "pre_update",
                "known_family_macro_f1": pre_metric,
                "old_family_retention": _metric_record(
                    [sample_id for sample_id in parts["scoring_ids"]
                     if _family_by_id(manifest)[sample_id] in set(manifest["admissions"][WINDOW_IDS[0]])],
                    manifest, state["class_map"], classifier,
                    RunningScaler.from_state_dict(state["scaler_state"]), store, device=device,
                    labels=[state["class_map"][family] for family in manifest["admissions"][WINDOW_IDS[0]]]),
                "new_family_acquisition": _acquisition_metric(
                    parts["acquisition_ids"], manifest, state["class_map"], classifier,
                    RunningScaler.from_state_dict(state["scaler_state"]), store, device,
                    parts["acquisition_families"]),
                "scored_count": len(parts["scoring_ids"]),
                "heldout_count": sum(manifest["family_support_by_window"][window_id]["heldout"].values()),
                "unscored_unknown_count": sum(manifest["family_support_by_window"][window_id]["heldout"].values()) - len(parts["scoring_ids"]),
            }
            progress_line.finish_phase()

        # Expand only from the current training partition's admitted labels.
        prior_class_count = classifier.num_classes
        for family in parts["admitted_families_this_window"]:
            if family not in state["class_map"]:
                state["class_map"][family] = len(state["class_map"])
        if classifier.num_classes != len(state["class_map"]):
            init_seed = state["run_metadata"]["rng_stream_identity"]["classifier_init_seed"]
            classifier.expand(len(state["class_map"]), seed=init_seed + window_index)
        if classifier.num_classes < prior_class_count:
            raise ValueError("classifier class head unexpectedly shrank")

        scaler = _update_scaler(state, manifest, window_index, parts, store)
        pool_x, pool_y, pool_sources, realized_anchor_count, realized_generated_count = _pool_for_arm(
            config, state, parts, store, manifest, state["class_map"]
        )
        actual_pool_counts = {
            "current": int(np.count_nonzero(pool_sources == "current")),
            "anchor": int(np.count_nonzero(pool_sources == "anchor")),
            "generated": int(np.count_nonzero(pool_sources == "generated")),
            "total": len(pool_y),
        }
        gan_compute_seconds = 0.0
        gan_wall_seconds = 0.0
        optimizer_g = optimizer_c = None
        if config.arm == "full_hybrid":
            optimizer_g, optimizer_c, gan_compute_seconds, gan_wall_seconds = _fit_gan(
                config, defaults, generator, critic, pool_x, pool_y, scaler, state, device,
                config.run_id, window_id, window_index, progress_line, progress_registry,
            )
        state["rng_state"] = capture_rng_state(state.get("numpy_rng_generators", {}))

        classifier_optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=defaults.classifier_lr,
            weight_decay=defaults.classifier_weight_decay,
        )
        sampler = MixedPoolSampler(pool_y, pool_sources, state["numpy_rng_generators"]["classifier_sampler"])
        head_class_ids = list(range(classifier.num_classes))
        seen_mask = torch.zeros(100, dtype=torch.bool, device=device)
        seen_mask[head_class_ids] = True
        pool_sizes = math.ceil(int(budget["planned_full_hybrid_pool"]) / defaults.batch_size)
        realized_samples = {"current": 0, "anchor": 0, "generated": 0}
        loss_values = []

        def train_batch(run_state, window, batch_index):
            draw = sampler.draw_batch(defaults.batch_size)
            xb_raw = pool_x[draw.indices]
            xb = torch.as_tensor(scaler.transform(xb_raw), dtype=torch.float32, device=device)
            yb = torch.as_tensor(draw.labels, dtype=torch.long, device=device)
            classifier.train()
            logits = classifier(xb)
            loss = classification_loss(
                config.arm, logits, yb, draw.sources.tolist(), None, [],
                head_class_ids, seen_mask,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite classifier loss")
            classifier_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), defaults.gradient_clip_norm)
            classifier_optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
            for name, count in draw.source_counts.items():
                realized_samples[name] = realized_samples.get(name, 0) + int(count)
            epoch = min(defaults.classifier_epochs_equivalent, batch_index // pool_sizes + 1)
            batch = batch_index % pool_sizes + 1
            progress_line.update(config.run_id, window_id, "classifier", epoch,
                                 defaults.classifier_epochs_equivalent, batch, pool_sizes)
            progress_registry.update(config.run_id, window_id, "classifier", epoch=epoch,
                                     epochs=defaults.classifier_epochs_equivalent, batch=batch,
                                     batches=pool_sizes)
            return run_state

        def finalize_window(run_state, window):
            progress_line.finish_phase()
            classifier.eval()
            if window_index == 0:
                jan_metric = _metric_record(
                    parts["scoring_ids"], manifest, state["class_map"], classifier,
                    scaler, store, device=device,
                )
                state["evaluations"][window_id] = {
                    "phase": "post_fit_january",
                    "january_post_fit": jan_metric,
                    "scored_count": len(parts["scoring_ids"]),
                    "heldout_count": sum(manifest["family_support_by_window"][window_id]["heldout"].values()),
                    "unscored_unknown_count": sum(manifest["family_support_by_window"][window_id]["heldout"].values()) - len(parts["scoring_ids"]),
                }

            generation_seconds = 0.0
            generated_output_count = 0
            if config.arm in {"full_hybrid", "anchor_only"}:
                anchor_state = {key: list(value) for key, value in state["anchor_state"]["sample_ids_by_family"].items()}
                anchor_rng = np.random.RandomState(
                    (state["run_metadata"]["rng_stream_identity"]["anchor_seed"]
                     + config.seed + window_index) % (2**32 - 1)
                )
                for family in parts["admitted_families_this_window"]:
                    class_ids = _train_ids_by_window_family(manifest).get((window_index, family), [])
                    quota = min(ANCHOR_K, len(class_ids))
                    selected = class_ids if len(class_ids) <= quota else [class_ids[i] for i in anchor_rng.choice(len(class_ids), size=quota, replace=False)]
                    anchor_state[family] = sorted(selected)
                state["anchor_state"] = {"sample_ids_by_family": anchor_state}

            if config.arm == "full_hybrid":
                generation_started = time.perf_counter()
                new_families = set(parts["admitted_families_this_window"])
                means = {}
                mean_rng = np.random.RandomState(config.seed + 999 + window_index)
                generated_raw = np.asarray(state["gan_state"]["generated_x_raw"], dtype=np.float32)
                generated_ids = np.asarray(state["gan_state"]["generated_class_ids"], dtype=np.int64)
                for family, class_id in sorted(state["class_map"].items(), key=lambda item: item[1]):
                    if family in new_families:
                        sample_ids = _train_ids_by_window_family(manifest).get((window_index, family), [])
                        if len(sample_ids) > 512:
                            choose = mean_rng.choice(len(sample_ids), size=512, replace=False)
                            sample_ids = [sample_ids[int(position)] for position in choose]
                        raw = store.get(sample_ids)
                    else:
                        positions = np.flatnonzero(generated_ids == int(class_id))
                        raw = generated_raw[positions]
                    if not len(raw):
                        raise ValueError(f"no causal training/replay rows available to form mean for {family}")
                    labels = np.full(len(raw), int(class_id), dtype=np.int64)
                    class_means = class_mean_mid(
                        classifier, raw, labels, scaler, [int(class_id)], device=str(device),
                    )
                    means[int(class_id)] = class_means[int(class_id)]
                selected_x, selected_y = _with_gan_torch_rng(
                    state, torch, device,
                    lambda: select_generated(
                        generator, classifier, means, list(range(classifier.num_classes)),
                        candidates_per_class=GAN_CANDIDATES_PER_CLASS,
                        quota=GENERATED_K, z_dim=defaults.noise_dim,
                        diversity_lambda=defaults.diversity_lambda,
                        device=str(device), return_records=False,
                    ),
                )
                raw_generated = scaler.inverse(np.asarray(selected_x, dtype=np.float32))
                state["gan_state"] = {
                    "generated_x_raw": np.asarray(raw_generated, dtype=np.float32),
                    "generated_class_ids": np.asarray(selected_y, dtype=np.int64),
                    "generated_at_window": window_id,
                }
                generated_output_count = len(raw_generated)
                generation_seconds = time.perf_counter() - generation_started

            state["training_state"] = {
                "classifier": _cpu_tree(classifier.state_dict(), torch),
                "generator": _cpu_tree(generator.state_dict(), torch),
                "critic": _cpu_tree(critic.state_dict(), torch),
            }
            state["optimizer_state"] = {
                "classifier": _cpu_tree(classifier_optimizer.state_dict(), torch),
                "generator": _cpu_tree(optimizer_g.state_dict(), torch) if optimizer_g is not None else {},
                "critic": _cpu_tree(optimizer_c.state_dict(), torch) if optimizer_c is not None else {},
            }
            state["scaler_state"] = scaler.state_dict()
            planned = schedule["windows"][window_index]
            record = {
                "window_id": window_id,
                "window_index": window_index,
                "pre_update": state["evaluations"].get(window_id, {}).get("known_family_macro_f1"),
                "january_post_fit": state["evaluations"].get(window_id, {}).get("january_post_fit"),
                "old_family_retention": state["evaluations"].get(window_id, {}).get("old_family_retention"),
                "new_family_acquisition": state["evaluations"].get(window_id, {}).get("new_family_acquisition"),
                "scored_sample_count": state["evaluations"][window_id]["scored_count"],
                "heldout_sample_count": state["evaluations"][window_id]["heldout_count"],
                "unscored_unknown_sample_count": state["evaluations"][window_id]["unscored_unknown_count"],
                "scoring_ids_sha256": (state["evaluations"].get(window_id, {}).get("known_family_macro_f1")
                                        or state["evaluations"].get(window_id, {}).get("january_post_fit", {})).get("scoring_ids_sha256"),
                "classifier_updates": int(planned["classifier_updates_all_arms"]),
                "planned_pool": {
                    "full_hybrid": int(planned["planned_full_hybrid_pool"]),
                    "current": int(planned["current_train_rows"]),
                    "anchor": int(planned["planned_anchor_rows"]),
                    "generated": int(planned["planned_generated_rows"]),
                },
                "planned_replay": {
                    "anchor": int(planned["planned_anchor_rows"])
                    if config.arm in {"full_hybrid", "anchor_only"} else 0,
                    "generated": int(planned["planned_generated_rows"])
                    if config.arm == "full_hybrid" else 0,
                },
                "realized_pool": actual_pool_counts,
                "realized_replay": {"anchor": realized_anchor_count, "generated": realized_generated_count},
                "generated_replay_output_count": generated_output_count,
                "classifier_sampled_source_counts": realized_samples,
                "classifier_mean_loss": float(np.mean(loss_values)) if loss_values else None,
                "gan_compute_seconds": float(gan_compute_seconds),
                "gan_wall_seconds": float(gan_wall_seconds),
                "generation_seconds": float(generation_seconds),
                "scaler_fit_sample_count": int(scaler.n),
                "scaler_fit_ids_sha256": state["scaler_fit_provenance"]["sample_ids_sha256"],
                "scaler_fit_training_only": True,
            }
            state["window_records"].append(record)
            state["rng_state"] = capture_rng_state(state.get("numpy_rng_generators", {}))
            progress_registry.update(config.run_id, window_id, "window_complete",
                                     epoch=defaults.classifier_epochs_equivalent,
                                     epochs=defaults.classifier_epochs_equivalent,
                                     batch=int(planned["classifier_updates_all_arms"]),
                                     batches=int(planned["classifier_updates_all_arms"]))
            return run_state

        checkpoint_path = run_dir / f"window_{window_index}.pkl"
        state = run_window(
            state,
            {"window_id": window_id, "window_index": window_index},
            updates=int(budget["classifier_updates_all_arms"]),
            train_batch=train_batch,
            checkpoint_path=checkpoint_path,
            finalize_window=finalize_window,
        )

    final_checkpoint = run_dir / f"window_{len(WINDOW_IDS) - 1}.pkl"
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config.experiment_id,
        "run_id": config.run_id,
        "arm": config.arm,
        "seed": config.seed,
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "window_records": state["window_records"],
        "evaluations": state["evaluations"],
    }
    _atomic_json(result_path, result)
    return {"run_id": config.run_id, "status": "completed", "result_path": str(result_path)}


def _acquisition_metric(sample_ids, manifest, class_map, classifier, scaler, feature_store, device, families):
    family_for = _family_by_id(manifest)
    family_set = set(families)
    selected = [sample_id for sample_id in sample_ids if family_for[sample_id] in family_set]
    if not families:
        return {"status": "not_applicable_initial_cohort", "macro_f1": None, "sample_count": 0,
                "per_family_sample_counts": {}, "per_family_f1": {}}
    if not selected:
        return {"status": "sparse_no_heldout_support", "macro_f1": None, "sample_count": 0,
                "per_family_sample_counts": {family: 0 for family in families},
                "per_family_f1": {family: None for family in families}}
    labels = [class_map[family] for family in families if family in class_map]
    record = _metric_record(selected, manifest, class_map, classifier, scaler, feature_store,
                            device=device, labels=labels)
    sparse = any(count < 30 for count in record["per_family_sample_counts"].values())
    return {
        "status": "exploratory_sparse_support" if sparse else "measured",
        "macro_f1": record["macro_f1"],
        "sample_count": record["sample_count"],
        "scoring_ids_sha256": record["scoring_ids_sha256"],
        "per_family_sample_counts": record["per_family_sample_counts"],
        "per_family_f1": record["per_family_f1"],
    }


def run_production_matrix(
    *,
    archive_path,
    manifest_path,
    output_root=DEFAULT_OUTPUT_ROOT,
    feature_store_path=None,
    enable_training: bool = False,
    confirmed_experiment_id: str = "",
    device: str | None = None,
    progress_line=None,
) -> list[dict]:
    """Run/resume the exact nine-run matrix after an explicit Colab opt-in."""
    if enable_training is not True or confirmed_experiment_id != EXPERIMENT_ID:
        raise RuntimeError(
            f"production training is disabled; set enable_training=True and confirm {EXPERIMENT_ID!r} in the notebook"
        )
    np, torch = _torch_numpy()
    manifest = load_manifest(manifest_path)
    if manifest.get("source", {}).get("archive_sha256") != sha256_file(archive_path):
        raise ValueError("run archive differs from the audited manifest source")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    schedule = load_or_freeze_budget_schedule(manifest, output_root)
    feature_store_path = Path(feature_store_path or (output_root / "feature_store"))
    metadata = materialize_feature_store(archive_path, manifest, feature_store_path)
    store = FeatureStore(feature_store_path, manifest)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; choose CPU explicitly or attach a GPU")
    progress_registry = ProgressRegistry(output_root, EXPERIMENT_ID)
    progress_line = progress_line or ProgressLine()
    results = []
    for config in validate_run_matrix():
        run_dir = output_root / "runs" / config.run_id
        latest = load_latest_checkpoint(run_dir)
        result_path = run_dir / "result.json"
        if latest and latest[1]["stream_cursor"]["next_window_index"] == len(WINDOW_IDS) and result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if (prior.get("status") == "completed"
                    and prior.get("manifest_sha256") == manifest["manifest_sha256"]
                    and prior.get("schedule_sha256") == schedule["schedule_sha256"]
                    and prior.get("final_checkpoint_sha256") == sha256_file(latest[0])):
                results.append({"run_id": config.run_id, "status": "skipped_finished"})
                continue
        results.append(_run_one(config, manifest, schedule, store, output_root,
                                device=target_device, progress_line=progress_line,
                                progress_registry=progress_registry))
    if len(results) != 9:
        raise AssertionError("pilot matrix did not process exactly nine configurations")
    return results
