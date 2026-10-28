"""Fresh-process workers for Colab GPU smoke, resume, and engineering checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .config import DATASETS, make_run_config
from .data import DatasetBundle
from .validation_workflow import EXPECTED_EFFECTIVE_SCHEDULE_SHA256


LEGACY_V5_RUNNER_CODE_SHA256 = "adb4ee76f1d7fb82a35ffaf4f38f5ddaa9b7649c36098f6110f47a938f9faf88"
CHECKPOINT_SYNC_POLICIES = {"all", "local_only", "publish_interruption_once", "restore_only"}
CHECKPOINT_WRITE_POLICIES = {"all", "recovery_and_terminal_only"}


def checkpoint_mirror_action(
    policy: str, *, source_is_drive: bool, destination_is_drive: bool,
) -> str:
    """Select a validation mirror action; production runner defaults use `all`."""
    if policy not in CHECKPOINT_SYNC_POLICIES:
        raise ValueError(f"unsupported validation checkpoint sync policy {policy!r}")
    if source_is_drive and not destination_is_drive:
        return "restore"
    if destination_is_drive:
        return "mirror" if policy == "all" else "skip"
    return "local"


def checkpoint_write_action(policy: str, kind: str) -> str:
    """Bound validation-only checkpoint writes without changing runner/model behavior."""
    if policy not in CHECKPOINT_WRITE_POLICIES:
        raise ValueError(f"unsupported validation checkpoint write policy {policy!r}")
    if policy == "all" or kind in {"initial", "interrupted-resumable", "complete", "failed-recovery"}:
        return "write"
    return "skip"


def estimate_schedule_resource_counts(
    schedule_rows: list[dict[str, Any]], *, arm_count: int,
    gan_epochs: int, gan_batch_size: int, critic_steps: int,
) -> dict[str, int]:
    """Count all rows supplied exactly once; rows may already span every seed."""
    if min(arm_count, gan_epochs, gan_batch_size, critic_steps) < 1:
        raise ValueError("resource projection counts and batch sizes must be positive")
    classifier_updates = sum(int(row["proposed_classifier_updates"]) for row in schedule_rows) * arm_count
    generated_only = 0
    full_hybrid = 0
    for row in schedule_rows:
        current = int(row["current_pool_size"])
        anchor = int(row["anchor_pool_size"])
        generated_only += gan_epochs * math.ceil(current / gan_batch_size)
        full_hybrid += gan_epochs * math.ceil((current + anchor) / gan_batch_size)
    generator_total = generated_only + full_hybrid
    return {
        "classifier_updates": classifier_updates,
        "generated_only_generator_updates": generated_only,
        "full_hybrid_generator_updates": full_hybrid,
        "generator_updates_total": generator_total,
        "critic_updates_total": generator_total * critic_steps,
    }


def make_synthetic_bundle(dataset: str) -> DatasetBundle:
    """Small non-reportable fixture using the dataset's exact feature dimension."""
    if dataset not in DATASETS:
        raise ValueError(f"unsupported synthetic dataset {dataset!r}")
    dimension = int(DATASETS[dataset]["feature_dim"])
    seed = 142 if dataset == "ember" else 2439
    rng = np.random.default_rng(seed)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 3)
    y_test = np.arange(100, dtype=np.int64)
    x_train = rng.normal(0.0, 0.2, size=(len(y_train), dimension)).astype(np.float32)
    x_test = rng.normal(0.0, 0.2, size=(len(y_test), dimension)).astype(np.float32)
    x_train[:, :8] += (y_train[:, None] / 100.0).astype(np.float32)
    x_test[:, :8] += (y_test[:, None] / 100.0).astype(np.float32)
    marker = {"namespace": "SYNTHETIC_NON_REPORTABLE", "sha256": "synthetic-fixture", "md5": "synthetic-fixture"}
    return DatasetBundle(dataset, x_train, y_train, x_test, y_test, dict(marker), dict(marker))


def make_cpu_test_bundle(dataset: str) -> DatasetBundle:
    """Tiny deterministic fixture for CPU integration tests; never reportable."""
    if dataset not in DATASETS:
        raise ValueError(f"unsupported synthetic dataset {dataset!r}")
    rng = np.random.default_rng(7401 if dataset == "ember" else 7402)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 3)
    y_test = np.arange(100, dtype=np.int64)
    x_train = rng.normal(0.0, 0.2, size=(len(y_train), 16)).astype(np.float32)
    x_test = rng.normal(0.0, 0.2, size=(len(y_test), 16)).astype(np.float32)
    x_train[:, :4] += (y_train[:, None] / 100.0).astype(np.float32)
    x_test[:, :4] += (y_test[:, None] / 100.0).astype(np.float32)
    marker = {
        "namespace": "LOCAL_CPU_TEST_FIXTURE_NOT_DATASET_EVIDENCE",
        "sha256": "local-cpu-test-fixture",
        "md5": "local-cpu-test-fixture",
    }
    return DatasetBundle(dataset, x_train, y_train, x_test, y_test, dict(marker), dict(marker))


def _ensure_validation_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if "validation" not in resolved.parts:
        raise ValueError(f"{label} must be beneath an isolated validation directory: {resolved}")
    return resolved


def _stop_hook(mode: str | None, *, initial_generator_updates: int | None = None):
    if mode is None:
        return None
    if mode == "classifier_after_update":
        return lambda state: (
            state.cursor.phase == "classifier"
            and int(state.extra.get("runtime", {}).get("classifier_updates_this_task", 0)) >= 1
        )
    if mode == "gan_after_generator":
        return lambda state: int(state.resources.get("generator_updates", 0)) >= 1
    if mode == "gan_after_next_generator":
        if initial_generator_updates is None:
            raise ValueError("GAN next-generator interruption needs a verified initial checkpoint")
        return lambda state: int(state.resources.get("generator_updates", 0)) > initial_generator_updates
    raise ValueError(f"unsupported synthetic interruption point {mode!r}")


def _identity_from_checkpoint(checkpoint_root: Path, *, schedule_sha256: str, code_sha256: str) -> dict[str, Any]:
    from .checkpointing import load_latest_valid

    manifests = sorted((checkpoint_root / "recovery").glob("generation-*.manifest.json"))
    if not manifests:
        raise ValueError(f"expected a verified starting checkpoint in {checkpoint_root}")
    identity = json.loads(manifests[-1].read_text(encoding="utf-8")).get("identity", {})
    if identity.get("schedule_sha256") != schedule_sha256 or identity.get("code_sha256") != code_sha256:
        raise ValueError("starting checkpoint schedule/code identity differs from the targeted GAN validation")
    loaded = load_latest_valid(checkpoint_root / "recovery", identity)
    return {"identity": identity, "state": loaded.state, "manifest": loaded.manifest}


def _publish_latest_interruption_checkpoint(
    local_root: Path, drive_root: Path, identity: dict[str, Any],
) -> dict[str, Any]:
    """Publish only the newest verified interruption generation, then verify Drive read-back."""
    from .checkpointing import _validate_manifest, load_latest_valid, mirror_checkpoint_generations
    from .validation_workflow import sha256_file

    local_recovery = local_root / "recovery"
    drive_recovery = drive_root / "recovery"
    latest = load_latest_valid(local_recovery, identity)
    generation = int(latest.manifest["generation"])
    manifest_name = f"generation-{generation:06d}.manifest.json"
    payload_name = str(latest.manifest["payload_file"])
    source_manifest = local_recovery / manifest_name
    source_payload = local_recovery / payload_name

    existing_manifest = drive_recovery / manifest_name
    if existing_manifest.is_file():
        existing = load_latest_valid(drive_recovery, identity)
        if (
            int(existing.manifest["generation"]) == generation
            and existing.manifest.get("payload_sha256") == latest.manifest.get("payload_sha256")
        ):
            return {
                "status": "ALREADY_PERSISTED",
                "generation": generation,
                "manifest_sha256": sha256_file(existing_manifest),
                "payload_sha256": sha256_file(drive_recovery / payload_name),
                "payload_bytes": (drive_recovery / payload_name).stat().st_size,
                "drive_recovery_root": str(drive_recovery),
            }
        raise ValueError(f"Drive already contains a different generation {generation}; refusing to overwrite it")

    drive_recovery.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="publish-one-gan-generation-", dir=local_root) as temp_name:
        source = Path(temp_name)
        for original in (source_manifest, source_payload):
            linked = source / original.name
            try:
                os.link(original, linked)
            except OSError:
                shutil.copy2(original, linked)
        copied = mirror_checkpoint_generations(source, drive_recovery, identity, keep=2)
        if len(copied) != 1 or int(copied[0]["generation"]) != generation:
            raise ValueError("Drive publication did not contain exactly the interrupted generation")

    verified = load_latest_valid(drive_recovery, identity)
    manifest = verified.manifest
    _validate_manifest(drive_recovery, manifest, identity)
    manifest_path = drive_recovery / manifest_name
    payload_path = drive_recovery / payload_name
    return {
        "status": "PERSISTED_AND_READ_BACK_VERIFIED",
        "generation": generation,
        "manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": sha256_file(payload_path),
        "payload_bytes": payload_path.stat().st_size,
        "drive_recovery_root": str(drive_recovery),
        "published_generation_count": 1,
    }


def run_synthetic_job(
    *,
    dataset: str,
    arm: str,
    schedule_sha256: str,
    output_dir: str | Path,
    drive_checkpoint_root: str | Path,
    local_checkpoint_root: str | Path,
    interruption: str | None = None,
    checkpoint_sync_policy: str = "all",
    checkpoint_write_policy: str = "all",
    runner_identity_code_sha256: str | None = None,
    cpu_test_only: bool = False,
) -> dict[str, Any]:
    """Run a synthetic worker; CPU execution is an isolated non-reportable test fixture."""
    if schedule_sha256 != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("validation worker must use the approved effective schedule SHA-256")
    import torch

    if not cpu_test_only and not torch.cuda.is_available():
        raise RuntimeError("synthetic GPU validation requires CUDA; no CPU PASS can substitute")
    device = "cpu" if cpu_test_only else "cuda"
    output = _ensure_validation_path(output_dir, "synthetic output")
    drive_root = _ensure_validation_path(drive_checkpoint_root, "Drive checkpoint root")
    local_root = _ensure_validation_path(local_checkpoint_root, "local validation checkpoint root")
    if checkpoint_sync_policy not in CHECKPOINT_SYNC_POLICIES:
        raise ValueError(f"unsupported validation checkpoint sync policy {checkpoint_sync_policy!r}")
    if checkpoint_write_policy not in CHECKPOINT_WRITE_POLICIES:
        raise ValueError(f"unsupported validation checkpoint write policy {checkpoint_write_policy!r}")
    from . import runner as runner_module

    runner_code_sha256_original = runner_module._code_sha256
    if runner_identity_code_sha256 is not None:
        from .validation_workflow import verify_legacy_runner_source_compatibility

        compatibility = verify_legacy_runner_source_compatibility(Path(__file__).resolve().parents[1])
        if runner_identity_code_sha256 != LEGACY_V5_RUNNER_CODE_SHA256:
            raise ValueError("only the exact verified v5 runner checkpoint identity may be reused")
        if compatibility.get("status") != "PASS" or compatibility.get("baseline_code_sha256") != runner_identity_code_sha256:
            raise ValueError(f"legacy checkpoint source compatibility failed: {compatibility}")
        runner_module._code_sha256 = lambda: runner_identity_code_sha256

    restore_events: list[dict[str, Any]] = []
    mirror_original = runner_module.mirror_checkpoint_generations
    save_original = runner_module.save_generation
    checkpoint_write_events: list[dict[str, Any]] = []
    suppressed_checkpoint_kinds: dict[str, int] = {}

    def controlled_mirror(source, destination, identity, **kwargs):
        source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
        source_is_drive = source_path.is_relative_to(drive_root.resolve())
        destination_is_drive = destination_path.is_relative_to(drive_root.resolve())
        action = checkpoint_mirror_action(
            checkpoint_sync_policy,
            source_is_drive=source_is_drive,
            destination_is_drive=destination_is_drive,
        )
        if action == "skip":
            return []
        if action == "restore":
            before = not destination_path.exists() or not any(destination_path.glob("generation-*.manifest.json"))
            copied = mirror_original(source_path, destination_path, identity, **kwargs)
            restore_events.append({
                "source_drive_root": str(source_path),
                "local_destination": str(destination_path),
                "local_staging_empty_before_restore": before,
                "generations_restored": [int(item["generation"]) for item in copied],
            })
            return copied
        return mirror_original(source_path, destination_path, identity, **kwargs)

    runner_module.mirror_checkpoint_generations = controlled_mirror

    def controlled_save_generation(root, state, identity, kind, **kwargs):
        action = checkpoint_write_action(checkpoint_write_policy, str(kind))
        if action == "skip":
            suppressed_checkpoint_kinds[str(kind)] = suppressed_checkpoint_kinds.get(str(kind), 0) + 1
            return {}
        manifest = save_original(root, state, identity, kind, **kwargs)
        checkpoint_write_events.append({
            "kind": str(kind), "generation": int(manifest["generation"]),
            "payload_bytes": int(manifest["payload_bytes"]),
            "manifest_bytes": len(json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")),
            "root": str(Path(root)),
        })
        return manifest

    runner_module.save_generation = controlled_save_generation

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = make_run_config(dataset, 42, arm, schedule_sha256)
    bundle = make_cpu_test_bundle(dataset) if cpu_test_only else make_synthetic_bundle(dataset)
    start_generator_updates = None
    if interruption == "gan_after_next_generator":
        checkpoint = _identity_from_checkpoint(
            local_root, schedule_sha256=schedule_sha256,
            code_sha256=runner_identity_code_sha256 or runner_code_sha256_original(),
        )
        state_identity = checkpoint["identity"]
        if (
            state_identity.get("config_sha256") != config.sha256()
            or state_identity.get("dataset") != runner_module._dataset_identity(bundle, synthetic=True)
        ):
            raise ValueError("GAN reference checkpoint config/dataset identity is incompatible")
        start_generator_updates = int(checkpoint["state"].resources.get("generator_updates", 0))

    try:
        result = runner_module.run_one(
            config, bundle, output,
            stop_hook=_stop_hook(interruption, initial_generator_updates=start_generator_updates),
            update_counts=[1] * 11,
            synthetic=True,
            device=device,
            drive_checkpoint_root=drive_root,
            local_checkpoint_root=local_root,
        )
        publication = None
        if checkpoint_sync_policy == "publish_interruption_once":
            if result.get("status") != "INTERRUPTED_RESUMABLE":
                raise ValueError("single-generation Drive publication requires a resumable interrupted worker")
            local_checkpoint = _identity_from_checkpoint(
                local_root, schedule_sha256=schedule_sha256,
                code_sha256=runner_identity_code_sha256 or runner_code_sha256_original(),
            )
            if interruption == "gan_after_next_generator" and int(local_checkpoint["state"].resources.get("generator_updates", 0)) <= int(start_generator_updates or 0):
                raise ValueError("interrupted checkpoint did not advance through a GAN generator update")
            publication = _publish_latest_interruption_checkpoint(local_root, drive_root, local_checkpoint["identity"])
        if checkpoint_sync_policy == "restore_only" and not any(
            item["local_staging_empty_before_restore"] for item in restore_events
        ):
            raise ValueError("fresh resume worker did not restore from Drive into empty local checkpoint staging")
        result["validation_checkpoint_storage"] = {
            "policy": checkpoint_sync_policy,
            "checkpoint_write_policy": checkpoint_write_policy,
            "checkpoint_write_events": checkpoint_write_events,
            "checkpoint_payload_bytes_written": sum(row["payload_bytes"] for row in checkpoint_write_events),
            "checkpoint_manifest_bytes_written": sum(row["manifest_bytes"] for row in checkpoint_write_events),
            "suppressed_checkpoint_kinds": suppressed_checkpoint_kinds,
            "routine_drive_uploads_during_run": 0 if checkpoint_sync_policy != "all" else None,
            "drive_restore_events": restore_events,
            "interruption_checkpoint_publication": publication,
            "runner_identity_code_sha256": runner_identity_code_sha256 or runner_code_sha256_original(),
            "legacy_identity_override": runner_identity_code_sha256 is not None,
            "execution_device": device,
            "cpu_test_fixture": cpu_test_only,
        }
        return result
    finally:
        runner_module.mirror_checkpoint_generations = mirror_original
        runner_module.save_generation = save_original
        runner_module._code_sha256 = runner_code_sha256_original


def _sync_cuda(torch) -> None:
    torch.cuda.synchronize()


def _pilot_projection(dataset: str, feature_dim: int, base_checkpoint_bytes: int) -> dict[str, Any]:
    """Transparent checkpoint-storage scenarios; replay growth is a protocol estimate."""
    task_seen = [50 + 5 * index for index in range(11)]
    generated_k = 100
    anchor_k = int(DATASETS[dataset]["anchor_k"])
    generated_bytes_per_task = [
        (count * generated_k * feature_dim * 4) + (count * generated_k * count * 4)
        for count in task_seen
    ]
    anchor_index_bytes_per_task = [count * anchor_k * 8 for count in task_seen]
    boundaries = [base_checkpoint_bytes + gen + anchor for gen, anchor in zip(generated_bytes_per_task, anchor_index_bytes_per_task)]
    recovery = [base_checkpoint_bytes + generated_bytes_per_task[-1] + anchor_index_bytes_per_task[-1]] * 2
    runs_per_dataset = 10 * 5
    snapshot_count_per_run = 11 + 2
    base_snapshot_scenario = base_checkpoint_bytes * snapshot_count_per_run * runs_per_dataset
    persistent_upper_per_run = sum(boundaries) + sum(recovery)
    full_hybrid_upper_scenario = persistent_upper_per_run * runs_per_dataset
    return {
        "runs_assumed_for_dataset": runs_per_dataset,
        "drive_snapshots_assumed_per_run": snapshot_count_per_run,
        "base_snapshot_scenario_bytes": int(base_snapshot_scenario),
        "conservative_full_hybrid_replay_upper_scenario_bytes": int(full_hybrid_upper_scenario),
        "persistent_upper_scenario_bytes_per_run": int(persistent_upper_per_run),
        "max_single_checkpoint_payload_bytes": int(max(boundaries + recovery)),
        "production_checkpoint_policy_unchanged": True,
        "upper_scenario_assumes_all_five_arms_retain_full_hybrid_replay": True,
        "replay_formula": "per task boundary: generated_k*seen_classes*feature_dim*4 + generated_k*seen_classes^2*4 + anchor_k*seen_classes*8; plus two retained recovery generations",
        "assumptions": [
            "There are 2 retained Drive recovery generations and 11 task-boundary generations per completed run.",
            "The base is the measured actual-data first-task model/optimizer checkpoint without later replay growth.",
            "The upper scenario assumes generated replay quota 100/class, stored logits for every seen class, and K anchor indices at every boundary for every arm; it is intentionally conservative.",
            "Storage for datasets, caches, logs, result exports, filesystem metadata, and Google Drive overhead is excluded.",
        ],
        "is_full_run_measurement": False,
    }


def run_actual_data_pilot(
    *,
    dataset: str,
    data_root: str | Path,
    cache_root: str | Path,
    schedule_sha256: str,
    effective_schedule_path: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    classifier_steps: int = 5,
) -> dict[str, Any]:
    """Measure a bounded train-split-only GPU engineering profile; no test evaluation."""
    if schedule_sha256 != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("engineering pilot must use the approved effective schedule SHA-256")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("actual-data engineering pilot requires CUDA")
    output = _ensure_validation_path(output_dir, "pilot output")
    checkpoint_root = _ensure_validation_path(checkpoint_dir, "pilot checkpoint root")
    result_path = output / "pilot_measurements.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("schedule_sha256") != schedule_sha256:
            raise ValueError("existing pilot result is bound to a different effective schedule")
        from .checkpointing import load_latest_valid
        manifests = sorted(checkpoint_root.glob("generation-*.manifest.json"))
        if not manifests:
            raise ValueError("existing pilot result has no corresponding Drive checkpoint generation")
        latest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        identity = latest.get("identity", {})
        if identity.get("schedule_sha256") != schedule_sha256 or identity.get("execution_scope") != "engineering_pilot":
            raise ValueError("existing pilot checkpoint identity does not match the expected schedule/profile")
        loaded = load_latest_valid(checkpoint_root, identity)
        if loaded.manifest.get("payload_sha256") != existing.get("checkpoint", {}).get("payload_sha256"):
            raise ValueError("existing pilot result/checkpoint hashes differ")
        return {"status": "ALREADY_VALIDATED", "result_path": str(result_path), "measurement": existing}

    from .checkpointing import load_latest_valid, save_generation
    from .config import make_run_config
    from .data import load_npz_arrays
    from .losses import pooled_ce_loss
    from .models import build_classifier
    from .preprocessing import RunningScaler

    spec = DATASETS[dataset]
    x_train, y_train, train_metadata = load_npz_arrays(
        Path(data_root) / spec["train"]["filename"], split="train",
        expected_md5=spec["train"]["md5"], expected_shape=tuple(spec["train"]["shape"]),
        expected_labels=range(100), cache_dir=cache_root,
    )
    from .rng import make_rng_streams
    from .generative import critic_update, generator_update
    from .models import V155ConditionalGenerator, V155ProjectionCritic
    from .state import PhaseCursor, RunState
    from .task_stream import split_order
    from .validation_workflow import sha256_file, write_immutable_json

    if sha256_file(effective_schedule_path) != schedule_sha256:
        raise ValueError("pilot effective schedule SHA-256 does not match the expected schedule identity")
    with Path(effective_schedule_path).open("r", newline="", encoding="utf-8-sig") as stream:
        schedule_rows = [row for row in csv.DictReader(stream) if row.get("dataset") == dataset]
    if len(schedule_rows) != 110:
        raise ValueError(f"pilot expected 110 effective schedule rows for {dataset}, found {len(schedule_rows)}")

    seed = 42
    config = make_run_config(dataset, seed, "full_hybrid", schedule_sha256)
    streams = make_rng_streams(dataset, seed, "full_hybrid")
    current_classes = split_order(streams.class_order, config.task_sizes)[0]
    current_indices = np.flatnonzero(np.isin(y_train, current_classes)).astype(np.int64)
    if len(current_indices) < 256:
        raise ValueError("task-1 training support is too small for the fixed 256-example pilot batch")
    scaler = RunningScaler()
    for start in range(0, len(current_indices), 8192):
        batch_indices = current_indices[start : start + 8192]
        scaler.update(np.asarray(x_train[batch_indices], dtype=np.float32))
    sources = np.full(len(current_indices), "current", dtype=str)
    from .sampling import MixedPoolSampler
    sample_rng = np.random.default_rng(242042)
    sampler_obj = MixedPoolSampler(np.asarray(y_train[current_indices]), sources, sample_rng)
    device = torch.device("cuda")
    torch.manual_seed(42042)
    torch.cuda.manual_seed_all(42042)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    classifier = build_classifier(
        int(spec["feature_dim"]), streams.classifier_init_seed,
        initial_classes=50, hidden=config.classifier_hidden,
        embedding_dim=config.classifier_embedding_dim, dropout=config.dropout,
        device=device,
    )
    classifier_optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=config.classifier_lr, weight_decay=config.classifier_weight_decay,
    )
    seen_mask = torch.zeros(100, dtype=torch.bool, device=device)
    seen_mask[torch.as_tensor(current_classes, dtype=torch.long, device=device)] = True

    def classifier_step() -> None:
        draw = sampler_obj.draw_batch(256)
        raw = np.asarray(x_train[current_indices[draw.indices]], dtype=np.float32)
        x = torch.as_tensor(scaler.transform(raw), dtype=torch.float32, device=device)
        y = torch.as_tensor(draw.labels, dtype=torch.long, device=device)
        logits = classifier(x)
        loss = pooled_ce_loss(logits, y, current_classes, seen_mask)
        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.gradient_clip_norm)
        classifier_optimizer.step()

    classifier_step()  # warm-up, deliberately excluded from elapsed throughput
    _sync_cuda(torch)
    classifier_started = time.perf_counter()
    for _ in range(classifier_steps):
        classifier_step()
    _sync_cuda(torch)
    classifier_seconds = time.perf_counter() - classifier_started

    # Use one fixed, training-only batch to measure the configured WGAN-GP costs.
    draw = sampler_obj.draw_batch(256)
    real_raw = np.asarray(x_train[current_indices[draw.indices]], dtype=np.float32)
    real = torch.as_tensor(scaler.transform(real_raw), dtype=torch.float32, device=device)
    labels = torch.as_tensor(draw.labels, dtype=torch.long, device=device)
    generator = V155ConditionalGenerator(int(spec["feature_dim"]), z_dim=config.noise_dim, hidden=256).to(device)
    critic = V155ProjectionCritic(int(spec["feature_dim"])).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr, betas=config.gan_betas)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr, betas=config.gan_betas)
    critic_update(generator, critic, real, labels, critic_optimizer, z_dim=config.noise_dim, gp_weight=config.gradient_penalty_weight, clip_norm=config.gradient_clip_norm)
    _sync_cuda(torch)
    critic_started = time.perf_counter()
    for _ in range(config.critic_steps):
        critic_update(generator, critic, real, labels, critic_optimizer, z_dim=config.noise_dim, gp_weight=config.gradient_penalty_weight, clip_norm=config.gradient_clip_norm)
    _sync_cuda(torch)
    critic_seconds = time.perf_counter() - critic_started
    generator_update(generator, critic, real, labels, generator_optimizer, z_dim=config.noise_dim, feature_matching_weight=config.feature_matching_weight, clip_norm=config.gradient_clip_norm)
    _sync_cuda(torch)
    generator_started = time.perf_counter()
    generator_update(generator, critic, real, labels, generator_optimizer, z_dim=config.noise_dim, feature_matching_weight=config.feature_matching_weight, clip_norm=config.gradient_clip_norm)
    _sync_cuda(torch)
    generator_seconds = time.perf_counter() - generator_started

    from .runner import _code_sha256
    profile = {
        "name": "actual_training_split_first_task_microbenchmark",
        "classifier_warmup_updates": 1,
        "classifier_measured_updates": int(classifier_steps),
        "critic_warmup_updates": 1,
        "critic_measured_updates": int(config.critic_steps),
        "generator_warmup_updates": 1,
        "generator_measured_updates": 1,
        "batch_size": 256,
        "dataset": dataset,
        "seed": seed,
        "arm_reference": "full_hybrid component cost profile; engineering only",
        "schedule_sha256": schedule_sha256,
        "uses_test_performance": False,
        "uses_test_features_or_labels": False,
    }
    profile_sha = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = {
        "runner_version": "revision_R1.0.2",
        "config_sha256": config.sha256(),
        "dataset": {
            "namespace": "ACTUAL_DATA_ENGINEERING_PILOT_TRAIN_SPLIT_ONLY",
            "dataset": dataset,
            "train_sha256": train_metadata["sha256"],
            "train_md5": train_metadata["md5"],
        },
        "code_sha256": _code_sha256(),
        "schedule_sha256": schedule_sha256,
        "execution_scope": "engineering_pilot",
        "profile_sha256": profile_sha,
    }
    state = RunState(
        identity=identity,
        cursor=PhaseCursor(task_index=0, phase="engineering_pilot_checkpoint"),
        classifier_state=classifier.state_dict(),
        classifier_optimizer_state=classifier_optimizer.state_dict(),
        generator_state=generator.state_dict(),
        generator_optimizer_state=generator_optimizer.state_dict(),
        critic_state=critic.state_dict(),
        critic_optimizer_state=critic_optimizer.state_dict(),
        scaler_state=scaler.state_dict(),
        rng_state={"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
        resources={"classifier_updates": classifier_steps, "critic_steps": config.critic_steps, "generator_updates": 1},
        extra={"engineering_profile": profile},
    )
    checkpoint_started = time.perf_counter()
    checkpoint_manifest = save_generation(checkpoint_root, state, identity, kind="engineering-pilot")
    _sync_cuda(torch)
    checkpoint_write_seconds = time.perf_counter() - checkpoint_started
    readback_started = time.perf_counter()
    loaded = load_latest_valid(checkpoint_root, identity)
    checkpoint_readback_seconds = time.perf_counter() - readback_started
    if loaded.manifest["payload_sha256"] != checkpoint_manifest["payload_sha256"]:
        raise ValueError("Drive checkpoint read-back identity/hash differs from the write")
    _sync_cuda(torch)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    checkpoint_bytes = int(checkpoint_manifest["payload_bytes"])
    projection = _pilot_projection(dataset, int(spec["feature_dim"]), checkpoint_bytes)
    resource_counts = estimate_schedule_resource_counts(
        schedule_rows, arm_count=5, gan_epochs=config.gan_epochs,
        gan_batch_size=config.gan_batch_size, critic_steps=config.critic_steps,
    )
    classifier_updates_total = resource_counts["classifier_updates"]
    generator_updates_total = resource_counts["generator_updates_total"]
    critic_updates_total = resource_counts["critic_updates_total"]
    runtime_estimate = {
        "status": "ESTIMATE_NOT_FULL_RUN_MEASUREMENT",
        "scope": "this dataset's 10 seeds and five arms, projected from a first-task microbenchmark",
        "effective_classifier_updates": classifier_updates_total,
        "generated_only_and_full_hybrid_generator_updates": generator_updates_total,
        "generator_updates_by_arm": {
            "generated_only": resource_counts["generated_only_generator_updates"],
            "full_hybrid": resource_counts["full_hybrid_generator_updates"],
        },
        "generated_only_and_full_hybrid_critic_updates": critic_updates_total,
        "classifier_seconds_estimate": classifier_updates_total * classifier_seconds / classifier_steps,
        "generator_seconds_estimate": generator_updates_total * generator_seconds,
        "critic_seconds_estimate": critic_updates_total * critic_seconds / config.critic_steps,
        "basis": "effective schedule row counts and GAN real-pool batches multiplied by measured first-task per-update time",
        "excludes": ["candidate generation", "replay selection", "evaluation", "data preparation", "Drive checkpoint synchronization beyond the measured save", "later-task head growth and replay-dependent effects"],
        "classifier_update_matching_is_not_total_compute_matching": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_type": "ACTUAL_DATA_ENGINEERING_PILOT_NOT_SCIENTIFIC_RESULT",
        "dataset": dataset,
        "feature_dim": int(spec["feature_dim"]),
        "schedule_sha256": schedule_sha256,
        "effective_schedule_path": str(Path(effective_schedule_path).resolve()),
        "effective_schedule_file_sha256": sha256_file(effective_schedule_path),
        "train_file_sha256": train_metadata["sha256"],
        "train_rows_in_task_1": int(len(current_indices)),
        "train_split_only": True,
        "test_split_read_by_pilot": False,
        "test_performance_used_for_setting_selection": False,
        "validation_profile": profile,
        "profile_sha256": profile_sha,
        "config_sha256": config.sha256(),
        "classifier": {"updates": classifier_steps, "examples": int(classifier_steps * 256), "seconds": classifier_seconds, "updates_per_second": classifier_steps / classifier_seconds if classifier_seconds else None, "examples_per_second": classifier_steps * 256 / classifier_seconds if classifier_seconds else None},
        "critic": {"updates": config.critic_steps, "seconds": critic_seconds, "updates_per_second": config.critic_steps / critic_seconds if critic_seconds else None, "examples_per_second": config.critic_steps * 256 / critic_seconds if critic_seconds else None},
        "generator": {"updates": 1, "seconds": generator_seconds, "updates_per_second": 1 / generator_seconds if generator_seconds else None, "examples_per_second": 256 / generator_seconds if generator_seconds else None},
        "peak_gpu_memory_bytes": {"allocated": peak_allocated, "reserved": peak_reserved},
        "checkpoint": {
            "identity_sha256": checkpoint_manifest["identity_sha256"],
            "payload_sha256": checkpoint_manifest["payload_sha256"],
            "payload_bytes": checkpoint_bytes,
            "drive_write_and_readback_seconds": checkpoint_write_seconds,
            "drive_readback_validation_seconds": checkpoint_readback_seconds,
            "persisted_and_verified": True,
        },
        "projected_storage": projection,
        "runtime_extrapolation": runtime_estimate,
    }
    write_immutable_json(result_path, report)
    return {"status": "PASS", "result_path": str(result_path), "measurement": report}


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="revision-r1-validation-worker")
    modes = parser.add_subparsers(dest="command", required=True)
    synthetic = modes.add_parser("synthetic", help="run/interrupt a synthetic CUDA validation job")
    synthetic.add_argument("--mode", choices=("run", "interrupt"), required=True)
    synthetic.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    synthetic.add_argument("--arm", choices=("current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched"), required=True)
    synthetic.add_argument("--schedule-sha256", required=True)
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--drive-checkpoint-root", required=True)
    synthetic.add_argument("--local-checkpoint-root", required=True)
    synthetic.add_argument("--interruption", choices=("classifier_after_update", "gan_after_generator", "gan_after_next_generator"))
    synthetic.add_argument("--checkpoint-sync-policy", choices=tuple(sorted(CHECKPOINT_SYNC_POLICIES)), default="all")
    synthetic.add_argument("--checkpoint-write-policy", choices=tuple(sorted(CHECKPOINT_WRITE_POLICIES)), default="all")
    synthetic.add_argument("--runner-identity-code-sha256")
    synthetic.add_argument("--cpu-test-only", action="store_true", help="run a tiny non-reportable CPU fixture; never substitutes for Colab CUDA validation")
    pilot = modes.add_parser("pilot", help="run the actual training-split-only engineering pilot")
    pilot.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    pilot.add_argument("--data-root", required=True)
    pilot.add_argument("--cache-root", required=True)
    pilot.add_argument("--schedule-sha256", required=True)
    pilot.add_argument("--effective-schedule", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--checkpoint-dir", required=True)
    pilot.add_argument("--classifier-steps", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    if args.command == "synthetic" and args.mode == "interrupt" and args.interruption is None:
        parser.error("--interruption is required in synthetic interrupt mode")
    if args.command == "synthetic" and args.mode == "run" and args.interruption is not None:
        parser.error("--interruption is only valid in synthetic interrupt mode")
    try:
        if args.command == "synthetic":
            result = run_synthetic_job(
                dataset=args.dataset, arm=args.arm, schedule_sha256=args.schedule_sha256,
                output_dir=args.output_dir, drive_checkpoint_root=args.drive_checkpoint_root,
                local_checkpoint_root=args.local_checkpoint_root,
                interruption=args.interruption if args.mode == "interrupt" else None,
                checkpoint_sync_policy=args.checkpoint_sync_policy,
                checkpoint_write_policy=args.checkpoint_write_policy,
                runner_identity_code_sha256=args.runner_identity_code_sha256,
                cpu_test_only=args.cpu_test_only,
            )
        else:
            result = run_actual_data_pilot(
                dataset=args.dataset, data_root=args.data_root, cache_root=args.cache_root,
                schedule_sha256=args.schedule_sha256, effective_schedule_path=args.effective_schedule,
                output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir,
                classifier_steps=args.classifier_steps,
            )
        print(json.dumps({"worker_pid": os.getpid(), **result}, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
