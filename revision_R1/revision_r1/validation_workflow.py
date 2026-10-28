"""Drive-safe Colab validation orchestration for the revision_R1 package.

This module deliberately imports no PyTorch at import time. Schedule and report
checks remain usable in a CPU-only builder environment; CUDA work is deferred to
the dedicated worker process.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .storage import (
    DEFAULT_STORAGE_RESERVE_BYTES,
    StorageCapacityError,
    available_bytes,
    require_available_bytes,
    require_path_capacity,
)


EXPECTED_EFFECTIVE_SCHEDULE_SHA256 = "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212"
VALIDATION_BUILD_ID = "revision_R1_colab_validation_v6_2_final_gan_recovery_qa"
EXPECTED_ANCHOR_CORRECTIONS = 47
EXPECTED_UPDATE_BUDGET_CHANGES = 5
SCHEDULE_ROWS = 220
TASK_SIZES = (50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
DATASET_ANCHOR_K = {"ember": 100, "az_class": 200}
SCHEDULE_BATCH_SIZE = 256
SCHEDULE_EPOCHS = 3
VALIDATION_CHECKPOINT_PREFLIGHT_BYTES = 512 * 1024 * 1024
PRIOR_V5_WORKFLOW_ID = "9f9d258801b1c86a"
PRIOR_V5_CODE_SHA256 = "adb4ee76f1d7fb82a35ffaf4f38f5ddaa9b7649c36098f6110f47a938f9faf88"
VALIDATION_WORKER_LOG_LIMIT_BYTES = 32 * 1024 * 1024
GAN_RESUME_REPORT_BUDGET_BYTES = 8 * 1024 * 1024
GAN_REPORT_ATOMIC_OVERHEAD_BYTES = 32 * 1024 * 1024
GAN_LOCAL_RESULT_OVERHEAD_BYTES = 256 * 1024 * 1024
# Conservative logical local writes: baseline seed+complete, two fault copies,
# interrupted seed+save+publication staging, and resumed Drive restore+complete.
GAN_LOCAL_CHECKPOINT_SLOTS = 8
GAN_LOCAL_CHECKPOINT_WRITE_SLOTS = 9
GAN_CHECKPOINT_MANIFEST_BUDGET_BYTES = 64 * 1024
GAN_CLEANUP_RECEIPT_WRITE_SPACE_BOUND_BYTES = GAN_REPORT_ATOMIC_OVERHEAD_BYTES + GAN_CHECKPOINT_MANIFEST_BUDGET_BYTES


def summarize_drive_storage_inventory(
    quota: dict[str, Any], trashed_files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize account quota and trashed checkpoint files without changing Drive."""
    def optional_int(name: str) -> int | None:
        value = quota.get(name)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if parsed >= 0 else None

    limit = optional_int("limit")
    usage = optional_int("usage")
    generations = []
    for item in trashed_files:
        name = str(item.get("name", ""))
        if not re.fullmatch(r"generation-\d+\.(?:pt|manifest\.json)", name):
            continue
        try:
            size = max(0, int(item.get("size", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            size = 0
        generations.append({
            "id": item.get("id"), "name": name, "bytes": size,
            "parents": item.get("parents", []), "modified_time": item.get("modifiedTime"),
        })

    return {
        "account_limit_bytes": limit,
        "account_usage_bytes": usage,
        "available_account_bytes": max(0, limit - usage) if limit is not None and usage is not None else None,
        "usage_in_drive_bytes": optional_int("usageInDrive"),
        "usage_in_drive_trash_bytes": optional_int("usageInDriveTrash"),
        "trashed_checkpoint_generation_files": len(generations),
        "trashed_checkpoint_generation_bytes": sum(item["bytes"] for item in generations),
        "trashed_checkpoint_payload_files": sum(item["name"].endswith(".pt") for item in generations),
        "trashed_checkpoint_manifest_files": sum(item["name"].endswith(".manifest.json") for item in generations),
        "trash_file_sample": sorted(generations, key=lambda item: item["bytes"], reverse=True)[:100],
        "trash_filename_attribution": "generation-name matches only; parent metadata is retained, but validation ownership is not inferred",
        "inventory_is_read_only": True,
        "trash_was_modified": False,
    }


def inventory_validation_files(validation_root: str | Path, *, largest_count: int = 20) -> dict[str, Any]:
    """Inventory visible files in the validation tree; never follows symlinks or deletes."""
    root = Path(validation_root).expanduser().resolve()
    if largest_count < 0:
        raise ValueError("largest_count cannot be negative")
    count = 0
    total_bytes = 0
    by_top_level: dict[str, dict[str, int]] = {}
    largest: list[tuple[int, str]] = []
    if root.is_dir():
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not (Path(directory) / name).is_symlink()]
            for filename in filenames:
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
                relative = path.relative_to(root).as_posix()
                category = relative.split("/", 1)[0]
                item = by_top_level.setdefault(category, {"files": 0, "bytes": 0})
                item["files"] += 1
                item["bytes"] += size
                count += 1
                total_bytes += size
                largest.append((size, relative))
    return {
        "root": str(root), "file_count": count, "visible_bytes": total_bytes,
        "by_top_level": by_top_level,
        "largest_files": [
            {"relative_path": relative, "bytes": size}
            for size, relative in sorted(largest, reverse=True)[:largest_count]
        ],
        "inventory_is_read_only": True,
    }


def query_drive_validation_inventory(validation_root: str | Path) -> dict[str, Any]:
    """Read quota and trashed-generation metadata via Drive API; no mutations are issued."""
    try:
        import google.auth
        from googleapiclient.discovery import build

        from .storage import DRIVE_METADATA_SCOPE

        credentials, _ = google.auth.default(scopes=[DRIVE_METADATA_SCOPE])
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        quota = service.about().get(
            fields="storageQuota(limit,usage,usageInDrive,usageInDriveTrash)",
        ).execute().get("storageQuota", {})
        trashed_files: list[dict[str, Any]] = []
        page_token = None
        while True:
            page = service.files().list(
                q="trashed = true and name contains 'generation-'",
                spaces="drive", pageSize=1000, pageToken=page_token,
                fields="nextPageToken,files(id,name,size,mimeType,modifiedTime,parents)",
            ).execute()
            trashed_files.extend(page.get("files", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        raise RuntimeError(
            "read-only Drive quota/Trash inventory failed; no GPU worker was started"
        ) from exc
    return {
        "drive": summarize_drive_storage_inventory(quota, trashed_files),
        "visible_validation_files": inventory_validation_files(validation_root),
        "trash_search_scope": "trashed files whose names match checkpoint generation files",
        "account_trash_usage_scope": "account-wide usageInDriveTrash from Drive API",
    }


def plan_gan_resume_storage(
    *, max_checkpoint_payload_bytes: int,
    available_local_bytes: int, available_drive_bytes: int,
    reserve_bytes: int = DEFAULT_STORAGE_RESERVE_BYTES,
    worker_count: int = 4,
    existing_workspace_bytes: int = 0,
    existing_report_bytes: int = 0,
    existing_visible_validation_bytes: int = 0,
    existing_local_validation_bytes: int = 0,
    existing_drive_account_usage_bytes: int | None = None,
    existing_drive_trash_usage_bytes: int | None = None,
    remaining_worker_log_bytes: int | None = None,
    worker_log_capacity_passed: bool = True,
) -> dict[str, Any]:
    """Fail-closed bound for the targeted GAN-only recovery validation."""
    payload_bound = int(max_checkpoint_payload_bytes)
    local_free = int(available_local_bytes)
    drive_free = int(available_drive_bytes)
    reserve = int(reserve_bytes)
    existing_workspace = int(existing_workspace_bytes)
    existing_reports = int(existing_report_bytes)
    visible_drive_bytes = int(existing_visible_validation_bytes)
    visible_local_bytes = int(existing_local_validation_bytes)
    account_usage = None if existing_drive_account_usage_bytes is None else int(existing_drive_account_usage_bytes)
    trash_usage = None if existing_drive_trash_usage_bytes is None else int(existing_drive_trash_usage_bytes)
    if min(
        payload_bound, local_free, drive_free, reserve, existing_workspace,
        existing_reports, visible_drive_bytes, visible_local_bytes,
        *([account_usage] if account_usage is not None else []),
        *([trash_usage] if trash_usage is not None else []),
        *([int(remaining_worker_log_bytes)] if remaining_worker_log_bytes is not None else []),
    ) < 0 or worker_count < 1:
        raise ValueError("storage plan inputs must be nonnegative and worker_count positive")
    drive_atomic_payload_peak = 2 * payload_bound  # one new payload plus atomic staging overlap
    drive_manifest_bound = GAN_CHECKPOINT_MANIFEST_BUDGET_BYTES
    drive_atomic_peak = drive_atomic_payload_peak + drive_manifest_bound
    log_budget = (
        int(remaining_worker_log_bytes) if remaining_worker_log_bytes is not None
        else int(worker_count) * VALIDATION_WORKER_LOG_LIMIT_BYTES
    )
    report_budget_remaining = max(0, GAN_RESUME_REPORT_BUDGET_BYTES - existing_reports)
    report_atomic_headroom = (
        GAN_REPORT_ATOMIC_OVERHEAD_BYTES + GAN_CLEANUP_RECEIPT_WRITE_SPACE_BOUND_BYTES
        if report_budget_remaining else 0
    )
    report_write_space_bound = (
        2 * report_budget_remaining + report_atomic_headroom
        if report_budget_remaining else 0
    )
    all_worker_logs_have_capacity = worker_log_capacity_passed and (remaining_worker_log_bytes is None or log_budget > 0)
    drive_write_volume_bound = drive_atomic_peak + log_budget + 2 * report_budget_remaining
    drive_peak_additional = drive_write_volume_bound + report_atomic_headroom
    drive_required = drive_peak_additional + reserve
    local_checkpoint_peak = GAN_LOCAL_CHECKPOINT_SLOTS * payload_bound
    local_peak = local_checkpoint_peak + GAN_LOCAL_RESULT_OVERHEAD_BYTES
    local_required = max(0, local_peak - existing_workspace) + reserve
    local_other_validation_bytes = max(0, visible_local_bytes - existing_workspace)
    local_workspace_within_bound = existing_workspace <= local_peak
    drive_persistent_additional = (
        payload_bound + drive_manifest_bound + log_budget + report_budget_remaining
    )
    local_checkpoint_payload_write_bound = GAN_LOCAL_CHECKPOINT_WRITE_SLOTS * payload_bound
    local_checkpoint_manifest_write_bound = GAN_LOCAL_CHECKPOINT_WRITE_SLOTS * drive_manifest_bound
    local_write_volume_bound = (
        local_checkpoint_payload_write_bound + local_checkpoint_manifest_write_bound
        + GAN_LOCAL_RESULT_OVERHEAD_BYTES
    )
    total_validation_write_volume_bound = drive_write_volume_bound + local_write_volume_bound
    drive_passed = drive_free >= drive_required and all_worker_logs_have_capacity and report_budget_remaining > 0
    local_passed = local_free >= local_required and local_workspace_within_bound
    return {
        "status": "PASS" if drive_passed and local_passed else "INSUFFICIENT",
        "max_checkpoint_payload_bound_bytes": payload_bound,
        "persistent_drive_checkpoint_generations": 1,
        "drive_checkpoint_write_volume_bound_bytes": drive_atomic_peak,
        "drive_checkpoint_payload_write_volume_bound_bytes": drive_atomic_payload_peak,
        "drive_checkpoint_persistent_payload_bytes": payload_bound,
        "drive_checkpoint_manifest_write_bound_bytes": drive_manifest_bound,
        "drive_checkpoint_atomic_temp_peak_bytes": payload_bound,
        "drive_atomic_checkpoint_peak_bytes": drive_atomic_payload_peak,
        "drive_atomic_checkpoint_peak_including_manifest_bytes": drive_atomic_peak,
        "drive_worker_log_write_budget_bytes": log_budget,
        "drive_reports_write_budget_bytes": report_budget_remaining,
        "existing_workflow_report_bytes": existing_reports,
        "report_atomic_write_space_bound_bytes": report_write_space_bound,
        "report_atomic_headroom_bytes": report_atomic_headroom,
        "total_drive_write_volume_bound_bytes": drive_write_volume_bound,
        "persistent_drive_validation_write_volume_bound_bytes": drive_persistent_additional,
        "expected_peak_additional_drive_bytes": drive_peak_additional,
        "existing_visible_validation_bytes": visible_drive_bytes,
        "projected_drive_validation_visible_peak_bytes": visible_drive_bytes + drive_peak_additional,
        "projected_drive_validation_visible_after_bytes": visible_drive_bytes + drive_persistent_additional,
        "existing_drive_account_usage_bytes": account_usage,
        "existing_drive_trash_usage_bytes": trash_usage,
        "projected_drive_account_usage_peak_bytes": account_usage + drive_peak_additional if account_usage is not None else None,
        "projected_drive_account_usage_after_bytes": account_usage + drive_persistent_additional if account_usage is not None else None,
        "projected_drive_account_usage_peak_includes_trash": account_usage is not None,
        "drive_required_bytes_including_reserve": drive_required,
        "available_drive_bytes": drive_free,
        "drive_capacity_passed": drive_passed,
        "worker_log_capacity_passed": all_worker_logs_have_capacity,
        "workflow_report_budget_passed": report_budget_remaining > 0,
        "expected_local_checkpoint_slots": GAN_LOCAL_CHECKPOINT_SLOTS,
        "expected_local_checkpoint_peak_bytes": local_checkpoint_peak,
        "expected_local_result_and_fault_scratch_bytes": GAN_LOCAL_RESULT_OVERHEAD_BYTES,
        "local_checkpoint_write_slots_bound": GAN_LOCAL_CHECKPOINT_WRITE_SLOTS,
        "local_checkpoint_payload_write_volume_bound_bytes": local_checkpoint_payload_write_bound,
        "local_checkpoint_manifest_write_volume_bound_bytes": local_checkpoint_manifest_write_bound,
        "local_result_and_fault_scratch_write_volume_bound_bytes": GAN_LOCAL_RESULT_OVERHEAD_BYTES,
        "total_local_write_volume_bound_bytes": local_write_volume_bound,
        "total_validation_write_volume_bound_bytes": total_validation_write_volume_bound,
        "target_worker_checkpoint_write_policy": "recovery_and_terminal_only",
        "expected_local_peak_bytes": local_peak,
        "existing_owned_workspace_bytes": existing_workspace,
        "existing_local_validation_bytes": visible_local_bytes,
        "other_existing_local_validation_bytes": local_other_validation_bytes,
        "local_workspace_within_declared_bound": local_workspace_within_bound,
        "projected_local_validation_peak_bytes": local_other_validation_bytes + max(existing_workspace, local_peak),
        "additional_local_bytes_required_including_reserve": local_required,
        "local_required_bytes_including_reserve": local_required,
        "available_local_bytes": local_free,
        "local_capacity_passed": local_passed,
        "safety_reserve_bytes": reserve,
        "routine_checkpoints_drive_synced": False,
        "routine_checkpoints_kept_local": True,
        "classifier_case_rerun": False,
        "smoke_suite_rerun": False,
        "manual_runtime_reset_rerun": False,
        "trash_cleanup_performed": False,
        "drive_accounting_includes_trash": True,
    }


def _gan_resume_failure_status(*, comparison_performed: bool, is_storage: bool) -> str:
    """A pre-comparison capacity stop is NOT_RUN, not evidence of recovery failure."""
    if is_storage and not comparison_performed:
        return "NOT_RUN"
    return "FAIL"


def _available_local_validation_bytes(path: str | Path) -> int:
    probe = Path(path).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _cuda_device_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def _has_future_generator_update(state: Any) -> bool:
    cursor = state.cursor
    task_index = int(cursor.task_index)
    phase = str(cursor.phase)
    if task_index < 0 or task_index >= 11 or phase in {"export", "complete"}:
        return False
    if task_index < 10:
        return True
    runtime = state.extra.get("runtime", {})
    if phase == "scaler":
        return True
    if phase == "gan_critic":
        total_batches = int(runtime.get("gan_batches_per_epoch", 0)) * int(runtime.get("gan_epochs_for_run", 0))
        return int(cursor.item_index) < total_batches
    generation_classes = runtime.get("generation_classes", [])
    if phase == "generation":
        return int(cursor.item_index) < len(generation_classes)
    if phase == "selection":
        return int(cursor.item_index) + 1 < len(generation_classes)
    return False


def _inspect_legacy_gan_reference(
    validation_root: str | Path, *, schedule_sha256: str, cpu_test_only: bool = False,
) -> dict[str, Any]:
    """Select the newest verified v5 GAN checkpoint that can still reach a generator update."""
    from . import runner as runner_module
    from .checkpointing import CheckpointError, _validate_manifest
    from .config import make_run_config
    from .storage import estimate_checkpoint_bytes
    from .validation_worker import make_cpu_test_bundle, make_synthetic_bundle

    root = Path(validation_root).expanduser().resolve()
    reference_root = root / "checkpoints" / PRIOR_V5_WORKFLOW_ID / "resume" / "gan_phase" / "baseline"
    recovery = reference_root / "recovery"
    manifest_paths = sorted(recovery.glob("generation-*.manifest.json"), reverse=True)
    if not manifest_paths:
        raise FileNotFoundError(f"unfinished v5 GAN reference checkpoint is missing: {recovery}")
    config = make_run_config("ember", 42, "generated_only", schedule_sha256)
    bundle = make_cpu_test_bundle("ember") if cpu_test_only else make_synthetic_bundle("ember")
    expected_dataset = runner_module._dataset_identity(bundle, synthetic=True)
    selected = None
    max_payload_bytes = 0
    errors: list[str] = []
    for manifest_path in manifest_paths:
        manifest = _load_json(manifest_path, "v5 GAN reference manifest")
        identity = manifest.get("identity", {})
        if (
            identity.get("schedule_sha256") != schedule_sha256
            or identity.get("code_sha256") != PRIOR_V5_CODE_SHA256
            or identity.get("execution_scope") != "synthetic_validation"
            or identity.get("config_sha256") != config.sha256()
            or identity.get("dataset") != expected_dataset
        ):
            raise ValueError(f"v5 GAN reference checkpoint identity is incompatible: {manifest_path}")
        try:
            state = _validate_manifest(recovery, manifest, identity)
        except CheckpointError as exc:
            errors.append(f"{manifest_path.name}: {exc}")
            continue
        max_payload_bytes = max(max_payload_bytes, int(manifest.get("payload_bytes", 0)))
        if int(state.resources.get("generator_updates", 0)) > 0 and _has_future_generator_update(state):
            selected = (manifest_path, manifest, identity, state)
            break
    if selected is None:
        detail = "; ".join(errors) if errors else "no valid checkpoint has a future GAN generator update"
        raise ValueError(f"no compatible unfinished v5 GAN reference can exercise interruption: {detail}")
    manifest_path, manifest, identity, state = selected
    max_payload_bytes = max(max_payload_bytes, estimate_checkpoint_bytes(state))
    return {
        "status": "PASS", "checkpoint_root": str(reference_root),
        "checkpoint_recovery_root": str(recovery),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "payload_path": str(recovery / manifest["payload_file"]),
        "payload_sha256": manifest["payload_sha256"],
        "checkpoint_payload_bytes": int(manifest["payload_bytes"]),
        "max_checkpoint_payload_bytes": int(max_payload_bytes),
        "generation": int(manifest["generation"]), "kind": manifest.get("kind"),
        "identity": identity, "cursor": dict(state.cursor.__dict__),
        "generator_updates_before_interruption": int(state.resources.get("generator_updates", 0)),
        "reference_manifest_verified": True, "reference_payload_hash_verified": True,
        "schedule_sha256": schedule_sha256, "code_sha256": PRIOR_V5_CODE_SHA256,
        "v5_checkpoint_tree_modified": False,
        "local_cpu_test_fixture": bool(cpu_test_only),
    }


def _seed_reference_checkpoint(reference: dict[str, Any], local_root: str | Path) -> dict[str, Any]:
    """Copy one verified v5 checkpoint generation into a local-only branch."""
    from .checkpointing import _validate_manifest

    local = Path(local_root).expanduser().resolve() / "recovery"
    local.mkdir(parents=True, exist_ok=True)
    source_root = Path(reference["checkpoint_recovery_root"]).expanduser().resolve()
    source_manifest = Path(reference["manifest_path"]).expanduser().resolve()
    if source_manifest.parent != source_root:
        raise ValueError("selected GAN reference manifest is not directly beneath its recovery root")
    manifest = _load_json(source_manifest, "selected v5 GAN reference manifest")
    identity = reference["identity"]
    _validate_manifest(source_root, manifest, identity)
    payload_name = manifest.get("payload_file")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError("selected GAN reference manifest has an invalid relative payload path")
    source_payload = (source_manifest.parent / payload_name).resolve()
    if source_payload.parent != source_root:
        raise ValueError("selected GAN reference payload is not relative to its manifest directory")
    if Path(reference["payload_path"]).expanduser().resolve() != source_payload:
        raise ValueError("selected GAN reference payload path differs from its manifest")
    target_manifest, target_payload = local / source_manifest.name, local / source_payload.name
    for source, target, expected_sha in (
        (source_payload, target_payload, reference["payload_sha256"]),
        (source_manifest, target_manifest, reference["manifest_sha256"]),
    ):
        if target.exists():
            if sha256_file(target) != expected_sha:
                raise ValueError(f"local GAN validation staging contains conflicting checkpoint bytes: {target}")
            continue
        shutil.copy2(source, target)
        if sha256_file(target) != expected_sha:
            raise ValueError(f"local GAN validation checkpoint copy failed SHA-256 verification: {target}")
    _validate_manifest(local, manifest, identity)
    return {
        "status": "PASS", "local_recovery_root": str(local),
        "generation": int(manifest["generation"]),
        "manifest_sha256": sha256_file(target_manifest),
        "payload_sha256": sha256_file(target_payload),
        "payload_bytes": target_payload.stat().st_size,
        "copied_from_v5_drive_checkpoint": True,
    }


def _cleanup_local_checkpoint_staging(
    checkpoint_root: str | Path, *, workspace: Path, workflow_id: str,
    schedule_sha256: str, code_sha256: str, role: str,
    allowed_staging_root: str | Path = "/content/revision_r1_validation_staging",
) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().absolute()
    boundary = workspace.expanduser().absolute()
    current = root
    while current != boundary and boundary in current.parents:
        if current.is_symlink():
            raise ValueError(f"symlinked checkpoint staging is not allowed: {current}")
        current = current.parent
    if boundary.is_symlink() or not root.resolve().is_relative_to(boundary.resolve()):
        raise ValueError(f"refusing to clean checkpoint root outside the owned workspace: {root}")
    owner = _load_json(root / "checkpoint_owner.json", "local checkpoint owner")
    if owner != {
        "validation_build_id": VALIDATION_BUILD_ID, "workflow_id": workflow_id,
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha256,
        "role": role, "artifact_type": "SYNTHETIC_GAN_RESUME_VALIDATION_ONLY",
    }:
        raise ValueError("local checkpoint ownership identity differs; refusing cleanup")
    allowed_dirs = {"recovery", "task_boundaries"}
    generation_pattern = re.compile(r"generation-\d+\.(?:pt|manifest\.json)")
    temp_pattern = re.compile(r"\.generation-\d+\.(?:pt|manifest\.json)\..+\.tmp")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for dirname in dirnames:
            child = base / dirname
            if child.is_symlink() or dirname not in allowed_dirs:
                raise ValueError(f"unexpected/symlinked directory in owned checkpoint staging: {child}")
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                raise ValueError(f"symlink in owned checkpoint staging: {path}")
            relative_parent = path.parent.relative_to(root).parts
            if path == root / "checkpoint_owner.json":
                continue
            if len(relative_parent) != 1 or relative_parent[0] not in allowed_dirs:
                raise ValueError(f"unexpected file in owned checkpoint staging: {path}")
            if not (generation_pattern.fullmatch(filename) or temp_pattern.fullmatch(filename)):
                raise ValueError(f"unexpected file in owned checkpoint staging: {path}")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    shutil.rmtree(root)
    return {"status": "PASS_LOCAL_CHECKPOINTS_RELEASED_AFTER_DRIVE_VERIFY", "path": str(root), "bytes_removed": size}


def _cleanup_local_gan_workspace(
    workspace: str | Path, *, workflow_id: str, schedule_sha256: str,
    code_sha256: str, local_staging_root: str | Path,
) -> dict[str, Any]:
    """Remove only a passing build-owned local validation workspace after evidence is durable."""
    raw_path = Path(workspace).expanduser().absolute()
    raw_staging_root = Path(local_staging_root).expanduser().absolute()
    safe_root = raw_staging_root.resolve() / "validation"
    path = raw_path.resolve()
    current = raw_path
    while current != raw_staging_root and raw_staging_root in current.parents:
        if current.is_symlink():
            raise ValueError(f"symlinked local validation workspace is not allowed: {current}")
        current = current.parent
    if raw_staging_root.is_symlink() or not path.is_relative_to(safe_root) or path == safe_root:
        raise ValueError(f"refusing to clean non-validation local path: {raw_path}")
    owner = _load_json(path / "workspace_owner.json", "local GAN validation workspace owner")
    expected_owner = {
        "validation_build_id": VALIDATION_BUILD_ID,
        "workflow_id": workflow_id,
        "schedule_sha256": schedule_sha256,
        "code_sha256": code_sha256,
        "artifact_type": "SYNTHETIC_GAN_RESUME_VALIDATION_ONLY",
    }
    if owner != expected_owner:
        raise ValueError("local GAN validation workspace ownership identity differs; refusing cleanup")
    expected_children = {"workspace_owner.json", "checkpoints", "outputs"}
    if {child.name for child in path.iterdir()} - expected_children:
        raise ValueError("unexpected content in GAN validation workspace; refusing cleanup")
    if any(child.is_symlink() for child in path.iterdir()):
        raise ValueError("symlink in GAN validation workspace; refusing cleanup")
    size = inventory_validation_files(path)["visible_bytes"]
    shutil.rmtree(path)
    return {"status": "PASS_CLEANED_OWNED_LOCAL_VALIDATION_ONLY", "path": str(path), "bytes_removed": size}


def verify_legacy_runner_source_compatibility(
    package_root: str | Path, manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify unchanged v5 training/checkpoint sources before using its checkpoint identity."""
    root = Path(package_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve() if manifest_path else root / "legacy_checkpoint_source_compatibility.json"
    try:
        manifest = _load_json(manifest_file, "legacy checkpoint source compatibility manifest")
        source_root = root / "revision_r1"
        excluded = set(manifest.get("excluded_validation_modules", []))
        expected = manifest.get("core_files", {})
        actual = {
            f"revision_r1/{path.name}": path
            for path in source_root.glob("*.py") if path.name not in excluded
        }
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name, path in actual.items()
            if name in expected and sha256_file(path) != expected[name]
        )
        ok = not missing and not unexpected and not mismatched
        return {
            "status": "PASS" if ok else "FAIL",
            "baseline_build_id": manifest.get("baseline_build_id"),
            "baseline_code_sha256": manifest.get("baseline_code_sha256"),
            "verified_training_core_files": len(expected) - len(missing) - len(mismatched),
            "mismatched_files": mismatched,
            "missing_files": missing,
            "unexpected_files": unexpected,
            "excluded_validation_only_modules": sorted(excluded),
            "manifest_path": str(manifest_file),
            "reason": "v5 model, runner, checkpoint, config, data, and support modules must remain byte-identical",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "status": "FAIL", "baseline_code_sha256": PRIOR_V5_CODE_SHA256,
            "verified_training_core_files": 0,
            "mismatched_files": [], "missing_files": [], "unexpected_files": [],
            "error": str(exc), "manifest_path": str(manifest_file),
        }


def validate_prior_classifier_resume_evidence(validation_root: str | Path) -> dict[str, Any]:
    """Adopt the Cell 10 classifier PASS only when its previous report is identity-bound."""
    root = Path(validation_root).expanduser().resolve()
    reports = sorted(
        (root / "reports").glob(f"resume_validation_{PRIOR_V5_WORKFLOW_ID}_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for report_path in reports:
        try:
            report = _load_json(report_path, "prior v5 resume report")
        except ValueError:
            continue
        if (
            report.get("validation_build_id") != "revision_R1_colab_validation_v5"
            or report.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
            or report.get("code_sha256") != PRIOR_V5_CODE_SHA256
        ):
            continue
        cases = report.get("cases", [])
        classifier_case = next((item for item in cases if item.get("scenario") == "classifier_phase"), None)
        comparison = (classifier_case or {}).get("comparison", {})
        if (
            (classifier_case or {}).get("status") != "PASS"
            or comparison.get("status") != "PASS"
            or comparison.get("final_predictions_exact") is not True
            or comparison.get("exact_counters_and_sampling") is not True
            or comparison.get("rng_state_exact") is not True
            or comparison.get("separate_gan_rng_state_exact") is not True
            or comparison.get("max_classifier_loss_absolute_delta") != 0.0
            or comparison.get("max_task_matrix_finite_absolute_error") != 0.0
            or comparison.get("max_model_state_absolute_delta") != 0.0
            or any(value != 0.0 for value in comparison.get("max_optimizer_state_absolute_deltas", {}).values())
        ):
            continue
        if "StorageCapacityError" not in str(report.get("error", "")):
            continue
        evidence_path = root / "reports" / f"resume_case_{PRIOR_V5_WORKFLOW_ID}_classifier_phase.json"
        evidence_digest = None
        if evidence_path.is_file():
            evidence = _load_json(evidence_path, "persisted classifier PASS evidence")
            if (
                evidence.get("status") != "PASS"
                or evidence.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
                or evidence.get("code_sha256") != PRIOR_V5_CODE_SHA256
                or (evidence.get("case") or {}).get("scenario") != "classifier_phase"
            ):
                continue
            evidence_digest = sha256_file(evidence_path)
        return {
            "status": "PASS", "evidence": classifier_case,
            "source_report": str(report_path), "source_report_sha256": sha256_file(report_path),
            "persisted_case_evidence": str(evidence_path) if evidence_path.is_file() else None,
            "persisted_case_evidence_sha256": evidence_digest,
            "prior_validation_build_id": report.get("validation_build_id"),
            "prior_validation_code_sha256": report.get("code_sha256"),
            "schedule_sha256": report.get("schedule_sha256"),
            "evidence_provenance": (
                "LOCAL_SYNTHETIC_CPU_TEST_FIXTURE"
                if report.get("local_cpu_test_fixture") is True
                else "REUSED_VERIFIED_V5_CELL10_EVIDENCE"
            ),
            "rerun": False,
        }
    raise ValueError(
        "no identity-matched v5 Cell 10 report with a classifier PASS and GAN StorageCapacityError was found"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_validation_storage_capacity(
    *, local_path: str | Path, drive_path: str | Path,
    checkpoint_preflight_bytes: int = VALIDATION_CHECKPOINT_PREFLIGHT_BYTES,
) -> dict[str, Any]:
    """Fail before GPU work unless both local staging and Drive can stage a checkpoint."""
    local = require_path_capacity(
        local_path, required_bytes=checkpoint_preflight_bytes,
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    drive = require_path_capacity(
        drive_path, required_bytes=checkpoint_preflight_bytes,
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    return {
        "status": "PASS",
        "checkpoint_write_preflight_bytes": checkpoint_preflight_bytes,
        "reserve_bytes": DEFAULT_STORAGE_RESERVE_BYTES,
        "local": local,
        "drive": drive,
        "production_runs_launched": False,
    }


def write_immutable_bytes(path: str | Path, payload: bytes) -> Path:
    """Create once; an identical rerun is harmless, differing content is refused."""
    target = Path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite a different validation artifact: {target}")
        return target
    require_path_capacity(
        target,
        required_bytes=max(1, len(payload) * 2),
        reserve_bytes=32 * 1024 * 1024,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_name, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite a different validation artifact: {target}")
        except OSError:
            created = False
            try:
                fd_out = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                created = True
                with os.fdopen(fd_out, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise ValueError(f"refusing to overwrite a different validation artifact: {target}")
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                raise
        return target
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_immutable_json(path: str | Path, value: dict[str, Any]) -> Path:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return write_immutable_bytes(path, payload)


def prune_checkpoint_payloads_after_pass(
    checkpoint_root: str | Path, evidence_path: str | Path,
) -> dict[str, Any]:
    """Remove synthetic checkpoint payloads only after a PASS report authenticates all files.

    Comparison reports and logs are deliberately outside this function's deletion
    scope. Both manifests and payloads are authenticated in a first pass before
    any checkpoint file is unlinked.
    """
    root = Path(checkpoint_root).expanduser().resolve()
    evidence_file = Path(evidence_path).expanduser().resolve()
    evidence = _load_json(evidence_file, "checkpoint comparison evidence")
    if evidence.get("status") != "PASS":
        raise ValueError("checkpoint cleanup requires a persisted PASS comparison report")
    schedule_sha256 = evidence.get("schedule_sha256")
    code_sha256 = evidence.get("code_sha256")
    if not isinstance(schedule_sha256, str) or len(schedule_sha256) != 64:
        raise ValueError("checkpoint cleanup report is missing its schedule identity")
    if not isinstance(code_sha256, str) or len(code_sha256) != 64:
        raise ValueError("checkpoint cleanup report is missing its code identity")
    records = evidence.get("checkpoint_evidence")
    if not isinstance(records, list):
        raise ValueError("checkpoint cleanup report has no authenticated checkpoint evidence")

    matching_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            recorded_root = Path(str(record["checkpoint_root"])).expanduser().resolve()
        except (KeyError, OSError, RuntimeError):
            continue
        if recorded_root == root:
            matching_records.append(record)

    manifests = sorted(root.rglob("generation-*.manifest.json")) if root.is_dir() else []
    payloads = sorted(root.rglob("generation-*.pt")) if root.is_dir() else []
    actual_manifests = {path.resolve() for path in manifests if path.is_file()}
    actual_payloads = {path.resolve() for path in payloads if path.is_file()}
    if not actual_payloads:
        return {
            "status": "ALREADY_PRUNED", "payloads_removed": 0,
            "manifests_retained": len(actual_manifests),
        }

    authenticated_manifests: set[Path] = set()
    authenticated_payloads: set[Path] = set()
    verified_records = []
    for record in matching_records:
        if record.get("execution_scope") != "synthetic_validation":
            raise ValueError("checkpoint cleanup refuses non-synthetic execution scope")
        if record.get("schedule_sha256") != schedule_sha256:
            raise ValueError("checkpoint cleanup schedule identity mismatch between report and evidence")
        if record.get("code_sha256") != code_sha256:
            raise ValueError("checkpoint cleanup code identity mismatch between report and evidence")
        for key, expected_root, digest_key, size_key in (
            ("manifest_relative_path", authenticated_manifests, "manifest_sha256", None),
            ("payload_relative_path", authenticated_payloads, "payload_sha256", "payload_bytes"),
        ):
            relative = Path(str(record.get(key, "")))
            candidate = (root / relative).resolve()
            if relative.is_absolute() or not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError(f"checkpoint cleanup evidence has an invalid/missing {key}: {relative}")
            if sha256_file(candidate) != record.get(digest_key):
                raise ValueError(f"checkpoint cleanup evidence hash mismatch for {candidate}")
            if size_key and candidate.stat().st_size != int(record.get(size_key, -1)):
                raise ValueError(f"checkpoint cleanup evidence size mismatch for {candidate}")
            expected_root.add(candidate)
        manifest_path = (root / str(record["manifest_relative_path"])).resolve()
        manifest = _load_json(manifest_path, "verified checkpoint manifest")
        identity = manifest.get("identity", {})
        if identity.get("execution_scope") != "synthetic_validation":
            raise ValueError("checkpoint cleanup manifest is not synthetic validation")
        if identity.get("schedule_sha256") != schedule_sha256:
            raise ValueError("checkpoint cleanup schedule identity mismatch in checkpoint manifest")
        if identity.get("code_sha256") != code_sha256:
            raise ValueError("checkpoint cleanup code identity mismatch in checkpoint manifest")
        payload_name = manifest.get("payload_file")
        payload_path = (manifest_path.parent / str(payload_name)).resolve()
        if not payload_path.is_relative_to(root) or payload_path not in authenticated_payloads:
            raise ValueError("checkpoint cleanup manifest does not reference its authenticated payload")
        if manifest.get("payload_sha256") != record.get("payload_sha256"):
            raise ValueError("checkpoint cleanup manifest and evidence payload hashes differ")
        verified_records.append({"manifest": manifest_path, "payload": payload_path})

    if authenticated_manifests != actual_manifests or authenticated_payloads != actual_payloads:
        raise ValueError("PASS comparison evidence does not authenticate every checkpoint file")

    # All hashes, sizes, paths, and execution scopes are checked before this point.
    # Small manifests remain as immutable checkpoint-hash evidence; only bulky
    # payloads are discarded after a successful PASS comparison.
    for item in verified_records:
        item["payload"].unlink()
    return {
        "status": "PRUNED", "payloads_removed": len(authenticated_payloads),
        "manifests_retained": len(authenticated_manifests),
        "removed_payload_sha256": sorted(record["payload_sha256"] for record in matching_records),
        "comparison_evidence": str(evidence_file),
    }


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _rows(path: Path) -> tuple[list[str], dict[tuple[str, int, int], dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"schedule has no CSV header: {path}")
        result: dict[tuple[str, int, int], dict[str, str]] = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed schedule row {line}: {path}")
            try:
                key = (row["dataset"].strip(), int(row["seed"]), int(row["task"]))
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid schedule key on line {line}: {path}") from exc
            if key in result:
                raise ValueError(f"duplicate schedule key {key}: {path}")
            result[key] = row
    return list(reader.fieldnames), result


def _schedule_candidates(directory: Path, expected_sha256: str) -> list[Path]:
    if not directory.is_dir():
        return []
    matches = []
    for candidate in sorted(directory.rglob("*.csv")):
        try:
            if sha256_file(candidate) == expected_sha256:
                matches.append(candidate.resolve())
        except OSError:
            continue
    return matches


def resolve_effective_artifacts(
    preflight_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    discover_manifest: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve the exact schedule/manifest referenced by preflight or fail closed."""
    preflight_file = Path(preflight_path).expanduser().resolve()
    preflight = _load_json(preflight_file, "preflight report")
    if preflight.get("status") != "PREFLIGHT_VALIDATED_NO_TRAINING":
        raise ValueError("preflight report is not a successful no-training validation")
    if preflight.get("training_reproduction_claimed") is not False:
        raise ValueError("preflight report must disclaim historical training reproduction")
    info = preflight.get("schedule")
    if not isinstance(info, dict) or info.get("row_count") != SCHEDULE_ROWS:
        raise ValueError("preflight does not confirm all 220 effective schedule rows")
    if info.get("sha256") != expected_sha256:
        raise ValueError(f"preflight schedule SHA-256 mismatch: {info.get('sha256')} != {expected_sha256}")

    base = preflight_file.parent.resolve()
    stated_schedule = Path(str(info.get("path", ""))).expanduser()
    schedule_path: Path | None = None
    if stated_schedule.is_file() and sha256_file(stated_schedule) == expected_sha256:
        candidate = stated_schedule.resolve()
        if not candidate.is_relative_to(base):
            raise ValueError("effective schedule referenced by preflight is outside its preflight directory")
        schedule_path = candidate
    else:
        candidates = _schedule_candidates(base, expected_sha256)
        if len(candidates) != 1:
            raise ValueError(
                f"effective schedule discovery is {'ambiguous' if candidates else 'empty'} under {base}: "
                f"{[str(path) for path in candidates]}"
            )
        schedule_path = candidates[0]

    stated_manifest = Path(str(info.get("manifest_path", ""))).expanduser()
    candidates: list[Path] = []
    for candidate in (stated_manifest, schedule_path.with_suffix(".manifest.json")):
        if candidate.is_file() and candidate.resolve() not in candidates:
            if not candidate.resolve().is_relative_to(base):
                raise ValueError("effective schedule manifest referenced by preflight is outside its preflight directory")
            manifest = _load_json(candidate, "effective schedule manifest")
            if manifest.get("source_csv_sha256") == expected_sha256:
                candidates.append(candidate.resolve())
    explicit_manifest_found = bool(candidates)
    if discover_manifest or not explicit_manifest_found:
        for candidate in sorted(base.rglob("*.manifest.json")):
            if candidate.resolve() in candidates:
                continue
            try:
                manifest = _load_json(candidate, "effective schedule manifest")
            except ValueError:
                continue
            if manifest.get("source_csv_sha256") == expected_sha256:
                candidates.append(candidate.resolve())
    # When paths are absent, discovery must identify exactly one companion.
    if not candidates:
        raise ValueError(f"no manifest matching effective schedule SHA-256 under {base}")
    if (discover_manifest or not explicit_manifest_found) and len(candidates) != 1:
        raise ValueError(f"effective manifest discovery is ambiguous under {base}: {[str(p) for p in candidates]}")
    manifest_path = candidates[0]
    manifest = _load_json(manifest_path, "effective schedule manifest")
    if sha256_file(schedule_path) != expected_sha256:
        raise ValueError("resolved effective schedule failed SHA-256 verification")
    if manifest.get("source_csv_sha256") != expected_sha256:
        raise ValueError("resolved manifest does not authenticate the effective schedule")
    if Path(str(manifest.get("source_csv_path", ""))).name != schedule_path.name:
        raise ValueError("effective manifest path/name does not match the schedule artifact")
    if info.get("path") and Path(str(info["path"])).name != schedule_path.name:
        raise ValueError("preflight schedule path does not identify the resolved effective schedule")
    if info.get("manifest_path") and Path(str(info["manifest_path"])).name != manifest_path.name:
        raise ValueError("preflight manifest path does not identify the resolved manifest")
    return schedule_path, manifest_path, preflight


def _class_order(seed: int) -> list[int]:
    return np.random.RandomState(int(seed)).permutation(100).astype(int).tolist()


def _training_pool_counts(preflight: dict[str, Any], dataset: str, seed: int, task_id: int) -> tuple[int, int, int]:
    try:
        counts_payload = preflight[dataset]["train"]["class_counts"]
        counts = np.asarray([int(counts_payload[str(class_id)]) for class_id in range(100)], dtype=np.int64)
        train_shape = preflight[dataset]["train"]["shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"preflight lacks verified training class counts for {dataset}") from exc
    if counts.shape != (100,) or np.any(counts < 0):
        raise ValueError(f"invalid training class-count vector for {dataset}")
    if len(train_shape) != 2 or int(train_shape[0]) != int(counts.sum()):
        raise ValueError(f"training class counts do not reconcile with checksum-verified training row count for {dataset}")
    order = _class_order(seed)
    groups = []
    cursor = 0
    for size in TASK_SIZES:
        groups.append(order[cursor : cursor + size])
        cursor += size
    seen = [class_id for group in groups[: task_id - 1] for class_id in group]
    current = int(counts[groups[task_id - 1]].sum())
    anchor = sum(min(DATASET_ANCHOR_K[dataset], int(counts[class_id])) for class_id in seen)
    generated = len(seen) * 100
    return current, int(anchor), int(generated)


def _schedule_int(row: dict[str, str], name: str, key: tuple[str, int, int], source: str) -> int:
    try:
        value = int(row[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"missing/invalid {name} for {key} in {source}") from exc
    if value < 0:
        raise ValueError(f"negative {name} for {key} in {source}")
    return value


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Effective update-schedule reconciliation",
        "",
        f"- Status: **{report['status']}**",
        f"- Effective schedule SHA-256: `{report['effective_schedule_sha256']}`",
        f"- Validation build: `{report['validation_build_id']}`",
        f"- Effective manifest SHA-256: `{report['effective_manifest_sha256']}`",
        f"- Preflight JSON SHA-256: `{report['preflight_sha256']}`",
        f"- Packaged parent schedule SHA-256: `{report['packaged_schedule_sha256']}`",
        f"- Rows verified: {report['row_count']}",
        f"- Anchor pool corrections: {len(report['anchor_corrections'])}",
        f"- Update budget changes: {len(report['update_budget_changes'])}",
        f"- Evidence basis: `{report['evidence_basis']}`",
        "- Training-data-only confirmation: the report derives support from the checksum-verified training split class-count metadata and seed-specific class order; it does not use test features, test labels, predictions, validation/test metrics, or model performance.",
        "- The packaged schedule is read-only and is not modified by reconciliation.",
        "",
        "## Anchor pool corrections",
        "",
        "| Dataset | Seed | Task | Old anchor count | New anchor count | Reason |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in report["anchor_corrections"]:
        lines.append(
            f"| {item['dataset']} | {item['seed']} | {item['task']} | {item['old_anchor_count']} | {item['new_anchor_count']} | {item['reason']} |"
        )
    lines.extend(["", "## Update-budget changes", "", "| Dataset | Seed | Task | Old updates | New updates | Reason |", "|---|---:|---:|---:|---:|---|"])
    for item in report["update_budget_changes"]:
        lines.append(
            f"| {item['dataset']} | {item['seed']} | {item['task']} | {item['old_updates']} | {item['new_updates']} | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def audit_effective_schedule(
    preflight_path: str | Path,
    package_root: str | Path,
    *,
    expected_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    expected_anchor_changes: int | None = EXPECTED_ANCHOR_CORRECTIONS,
    expected_budget_changes: int | None = EXPECTED_UPDATE_BUDGET_CHANGES,
    discover_manifest: bool = False,
) -> dict[str, Any]:
    """Rebuild row-level reconciliation from immutable source and train metadata."""
    schedule_path, manifest_path, preflight = resolve_effective_artifacts(
        preflight_path, expected_sha256=expected_sha256, discover_manifest=discover_manifest,
    )
    package = Path(package_root).resolve()
    source_path = package / "configs" / "update_budget.csv"
    source_manifest_path = package / "configs" / "update_budget_manifest.json"
    if not source_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("packaged original schedule or its manifest is unavailable; old schedule values cannot be claimed")
    source_manifest = _load_json(source_manifest_path, "packaged schedule manifest")
    parent_hash = sha256_file(source_path)
    if source_manifest.get("source_csv_sha256") != parent_hash:
        raise ValueError("packaged schedule differs from its package manifest; refusing reconciliation")

    schedule_manifest = _load_json(manifest_path, "effective schedule manifest")
    if schedule_manifest.get("status") != "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING":
        raise ValueError("effective schedule manifest has unexpected provenance")
    if schedule_manifest.get("training_reproduction_claimed") is not False:
        raise ValueError("effective schedule manifest must disclaim historical reproduction")
    if schedule_manifest.get("row_count") != SCHEDULE_ROWS:
        raise ValueError("effective schedule manifest does not contain 220 rows")
    if schedule_manifest.get("parent_source_csv_sha256") != parent_hash:
        raise ValueError("effective schedule parent hash does not match the preserved packaged schedule")
    if preflight["schedule"].get("parent_source_sha256") != parent_hash:
        raise ValueError("preflight parent hash does not match the packaged schedule")

    for dataset in ("ember", "az_class"):
        train_meta = preflight.get(dataset, {}).get("train", {})
        identity = schedule_manifest.get("dataset_identities", {}).get(dataset, {})
        for field in ("md5", "sha256"):
            if not train_meta.get(field) or train_meta[field] != identity.get(f"train_{field}"):
                raise ValueError(f"effective schedule does not bind to preflight {dataset} training {field}")

    _, source_rows = _rows(source_path)
    _, effective_rows = _rows(schedule_path)
    expected_keys = {
        (dataset, seed, task)
        for dataset in ("ember", "az_class")
        for seed in range(42, 52)
        for task in range(1, 12)
    }
    if set(source_rows) != expected_keys or set(effective_rows) != expected_keys:
        raise ValueError("packaged and effective schedules must each contain exactly the fixed 220-row matrix")

    anchor_corrections: list[dict[str, Any]] = []
    budget_changes: list[dict[str, Any]] = []
    manifest_adjustments = {
        (str(row["dataset"]), int(row["seed"]), int(row["task"])): row
        for row in schedule_manifest.get("reconciliation", {}).get("adjustments", [])
    }
    for key in sorted(expected_keys):
        dataset, seed, task = key
        old, new = source_rows[key], effective_rows[key]
        actual_current, actual_anchor, actual_generated = _training_pool_counts(preflight, dataset, seed, task)
        current = _schedule_int(new, "current_pool_size", key, "effective schedule")
        anchor = _schedule_int(new, "anchor_pool_size", key, "effective schedule")
        generated = _schedule_int(new, "generated_pool_size", key, "effective schedule")
        full = _schedule_int(new, "full_hybrid_pool_size", key, "effective schedule")
        updates = _schedule_int(new, "proposed_classifier_updates", key, "effective schedule")
        if (current, anchor, generated) != (actual_current, actual_anchor, actual_generated):
            raise ValueError(f"effective pool counts for {key} are not derivable from training split availability")
        if full != current + anchor + generated:
            raise ValueError(f"effective full pool sum is invalid for {key}")
        if new.get("training_reproduction_claimed", "").strip().upper() not in {"NO", "FALSE", "0"}:
            raise ValueError(f"effective row {key} unexpectedly claims historical reproduction")
        if _schedule_int(new, "batch_size", key, "effective schedule") != SCHEDULE_BATCH_SIZE:
            raise ValueError(f"effective row {key} changes the fixed classifier batch size")
        if _schedule_int(new, "classifier_epochs_equivalent", key, "effective schedule") != SCHEDULE_EPOCHS:
            raise ValueError(f"effective row {key} changes the fixed classifier epoch-equivalent")
        formula_updates = SCHEDULE_EPOCHS * math.ceil(full / SCHEDULE_BATCH_SIZE)
        if updates != formula_updates:
            raise ValueError(f"effective update count for {key} does not follow the frozen formula")

        old_anchor = _schedule_int(old, "anchor_pool_size", key, "packaged schedule")
        old_updates = _schedule_int(old, "proposed_classifier_updates", key, "packaged schedule")
        old_full = _schedule_int(old, "full_hybrid_pool_size", key, "packaged schedule")
        old_current = _schedule_int(old, "current_pool_size", key, "packaged schedule")
        old_generated = _schedule_int(old, "generated_pool_size", key, "packaged schedule")
        if (old_current, old_generated) != (actual_current, actual_generated):
            raise ValueError(f"packaged current/generated pools differ from verified protocol for {key}")
        if old_full != old_current + old_anchor + old_generated:
            raise ValueError(f"packaged full-pool sum is invalid for {key}")
        if old_updates != SCHEDULE_EPOCHS * math.ceil(old_full / SCHEDULE_BATCH_SIZE):
            raise ValueError(f"packaged update count formula is invalid for {key}")
        if _schedule_int(new, "source_anchor_pool_size", key, "effective schedule") != old_anchor:
            raise ValueError(f"effective source anchor count does not preserve packaged value for {key}")
        if _schedule_int(new, "source_full_hybrid_pool_size", key, "effective schedule") != old_full:
            raise ValueError(f"effective source total does not preserve packaged value for {key}")
        if _schedule_int(new, "source_proposed_classifier_updates", key, "effective schedule") != old_updates:
            raise ValueError(f"effective source update count does not preserve packaged value for {key}")

        if old_anchor != anchor:
            adjustment = manifest_adjustments.get(key)
            if adjustment is None:
                raise ValueError(f"manifest omitted anchor adjustment for {key}")
            if int(adjustment.get("source_anchor_pool_size", -1)) != old_anchor or int(adjustment.get("effective_anchor_pool_size", -1)) != anchor:
                raise ValueError(f"manifest anchor adjustment values disagree for {key}")
            if new.get("schedule_reconciliation") != "NOMINAL_TO_VERIFIED_SUPPORT":
                raise ValueError(f"anchor correction reason is invalid for {key}")
            anchor_corrections.append({
                "dataset": dataset, "seed": seed, "task": task,
                "old_anchor_count": old_anchor, "new_anchor_count": anchor,
                "training_support_verified": True,
                "reason": f"The packaged nominal per-class quota K={DATASET_ANCHOR_K[dataset]} was replaced by sum(min(K, checksum-verified training support)) for previously seen classes; no performance evidence was consulted.",
            })
        elif new.get("schedule_reconciliation") != "UNCHANGED_VERIFIED_POOL":
            raise ValueError(f"unchanged anchor row has unexpected reason for {key}")

        if old_updates != updates:
            if new.get("schedule_reconciliation") not in {"NOMINAL_TO_VERIFIED_SUPPORT", "UNCHANGED_VERIFIED_POOL"}:
                raise ValueError(f"changed budget row lacks a recognized data-derived anchor reconciliation for {key}")
            budget_changes.append({
                "dataset": dataset, "seed": seed, "task": task,
                "old_updates": old_updates, "new_updates": updates,
                "old_pool_total": old_full, "new_pool_total": full,
                "reason": "Recomputed using 3 * ceil(effective full-hybrid pool / 256) after verified training-support reconciliation; this is an arithmetic consequence of available training examples, not model/test performance.",
            })
    if set(manifest_adjustments) != {tuple((r["dataset"], r["seed"], r["task"])) for r in anchor_corrections}:
        raise ValueError("effective manifest adjustment list does not exactly match CSV anchor corrections")
    declared = schedule_manifest.get("reconciliation", {})
    if declared.get("adjusted_rows") != len(anchor_corrections) or declared.get("changed_update_rows") != len(budget_changes):
        raise ValueError("effective manifest aggregate reconciliation counts disagree with the row audit")
    if expected_anchor_changes is not None and len(anchor_corrections) != expected_anchor_changes:
        raise ValueError(f"expected {expected_anchor_changes} anchor corrections, found {len(anchor_corrections)}")
    if expected_budget_changes is not None and len(budget_changes) != expected_budget_changes:
        raise ValueError(f"expected {expected_budget_changes} changed update budgets, found {len(budget_changes)}")

    return {
        "schema_version": 1,
        "validation_build_id": VALIDATION_BUILD_ID,
        "status": "PASS",
        "artifact_type": "SCHEDULE_RECONCILIATION_ONLY_NO_TRAINING",
        "effective_schedule_path": str(schedule_path),
        "effective_manifest_path": str(manifest_path),
        "effective_manifest_sha256": sha256_file(manifest_path),
        "preflight_path": str(Path(preflight_path).resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "effective_schedule_sha256": sha256_file(schedule_path),
        "packaged_schedule_path": str(source_path),
        "packaged_schedule_sha256": parent_hash,
        "packaged_manifest_sha256": sha256_file(source_manifest_path),
        "row_count": len(effective_rows),
        "anchor_corrections": anchor_corrections,
        "update_budget_changes": budget_changes,
        "evidence_basis": "CHECKSUM_VERIFIED_TRAINING_SPLIT_CLASS_COUNTS_ONLY",
        "performance_evidence_used": False,
        "test_features_or_labels_used_for_reconciliation": False,
        "historical_training_reproduction_claimed": False,
        "packaged_schedule_modified": False,
    }


def write_reconciliation_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    if report.get("status") != "PASS":
        raise ValueError("only a successful row-level reconciliation can be published as PASS")
    root = Path(output_dir)
    digest = str(report["effective_schedule_sha256"])
    preflight_digest = str(report["preflight_sha256"])
    if report.get("validation_build_id") != VALIDATION_BUILD_ID:
        raise ValueError("schedule reconciliation must be generated by the corrected validation build")
    json_path = root / f"schedule_reconciliation_{digest[:12]}_{preflight_digest[:12]}_{VALIDATION_BUILD_ID}.json"
    markdown_path = root / f"schedule_reconciliation_{digest[:12]}_{preflight_digest[:12]}_{VALIDATION_BUILD_ID}.md"
    write_immutable_json(json_path, report)
    write_immutable_bytes(markdown_path, _markdown_report(report).encode("utf-8"))
    return json_path, markdown_path


def validation_code_sha256(package_root: str | Path | None = None) -> str:
    root = Path(package_root).resolve() if package_root else Path(__file__).resolve().parents[1]
    source_root = root / "revision_r1" if (root / "revision_r1").is_dir() else root
    digest = hashlib.sha256()
    for path in sorted(source_root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _workflow_id(schedule_sha256: str, package_root: str | Path | None = None) -> tuple[str, str]:
    if schedule_sha256 != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("validation workflow must use the approved effective schedule SHA-256")
    code_sha = validation_code_sha256(package_root)
    identity = hashlib.sha256(f"{VALIDATION_BUILD_ID}|{schedule_sha256}|{code_sha}".encode()).hexdigest()
    return identity[:16], code_sha


def _validation_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if "validation" not in root.parts:
        raise ValueError(f"all smoke/pilot/report artifacts must be below a validation directory: {root}")
    return root


def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{time.time_ns() % 1_000_000_000:09d}"


def _production_checkpoint_storage_projection(
    measurements: list[dict[str, Any]], *, available_drive_bytes: int | None,
) -> dict[str, Any]:
    """Aggregate per-dataset pilot snapshots into an explicit 100-run Drive gate."""
    if len(measurements) != 2:
        return {
            "status": "NOT_VERIFIABLE", "production_run_count": 100,
            "reason": "both dataset pilot measurements are required",
            "production_storage_gate": "NOT_VERIFIABLE",
        }
    storage = [item.get("projected_storage", {}) for item in measurements]
    base = sum(int(item.get("base_snapshot_scenario_bytes", 0)) for item in storage)
    upper = sum(int(item.get("conservative_full_hybrid_replay_upper_scenario_bytes", 0)) for item in storage)
    transient = max((int(item.get("max_single_checkpoint_payload_bytes", 0)) for item in storage), default=0)
    required = upper + transient + DEFAULT_STORAGE_RESERVE_BYTES
    if available_drive_bytes is None:
        gate = "NOT_VERIFIABLE"
    else:
        gate = "PASS" if available_drive_bytes >= required else "INSUFFICIENT"
    return {
        "status": "ESTIMATE_NOT_FULL_RUN_MEASUREMENT",
        "production_run_count": 100,
        "base_snapshot_scenario_bytes": base,
        "persistent_upper_scenario_bytes": upper,
        "transient_write_peak_bytes": transient,
        "safety_reserve_bytes": DEFAULT_STORAGE_RESERVE_BYTES,
        "required_drive_bytes_including_reserve": required,
        "available_drive_bytes_at_pilot_time": available_drive_bytes,
        "production_storage_gate": gate,
        "production_must_remain_disabled_unless_gate_passes": True,
        "all_values_are_projected_not_measured_full_run_storage": True,
    }


def find_unfinished_manual_reset(
    attempt_root: str | Path, schedule_sha256: str, code_sha256: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Return the sole current-identity reset attempt not marked terminal."""
    root = Path(attempt_root)
    unfinished: list[tuple[Path, dict[str, Any]]] = []
    if not root.is_dir():
        return None
    for pending_path in sorted(root.glob("pending_*.json")):
        pending = _load_json(pending_path, "manual runtime-reset pending record")
        if pending.get("schedule_sha256") != schedule_sha256 or pending.get("code_sha256") != code_sha256:
            raise ValueError(f"manual runtime-reset attempt has a different schedule/code identity: {pending_path}")
        attempt_id = str(pending.get("attempt_id", ""))
        if not attempt_id or Path(attempt_id).name != attempt_id:
            raise ValueError(f"manual runtime-reset record has an invalid attempt ID: {pending_path}")
        if (root / f"completed_{attempt_id}.json").is_file() or (root / f"failed_{attempt_id}.json").is_file():
            continue
        unfinished.append((pending_path, pending))
    if len(unfinished) > 1:
        raise ValueError("multiple unfinished manual runtime-reset attempts exist; resolve them before starting another")
    return unfinished[0] if unfinished else None


def _manual_reset_attempt_id(schedule_sha256: str, code_sha256: str) -> str:
    """One manual-reset fixture namespace per immutable schedule/build identity."""
    return f"{schedule_sha256[:12]}_{code_sha256[:12]}"


def _run_worker(validation_root: Path, workflow_id: str, label: str, arguments: list[str]) -> dict[str, Any]:
    logs = validation_root / "logs"
    log_path = logs / f"{workflow_id}_{label}.log"
    package_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "revision_r1.validation_worker", *arguments]
    checkpoint_roots = [
        Path(arguments[index + 1])
        for index, argument in enumerate(arguments[:-1])
        if argument in {"--drive-checkpoint-root", "--local-checkpoint-root"}
    ]
    completed = _stream_worker_command(command, cwd=package_root, log_path=log_path, label=label, checkpoint_roots=checkpoint_roots)
    result: dict[str, Any] = {"returncode": completed["returncode"], "log_path": str(log_path)}
    lines = [line for line in completed["stdout"].splitlines() if line.strip()]
    if completed["returncode"] == 0 and lines:
        try:
            result.update(json.loads(lines[-1]))
        except json.JSONDecodeError:
            result.update({"status": "FAIL", "error": "worker output did not end in JSON"})
    elif completed["returncode"] != 0:
        if lines:
            try:
                result.update(json.loads(lines[-1]))
            except json.JSONDecodeError:
                result.update({"status": "FAIL", "error": completed["stdout"][-4000:]})
        else:
            result.update({"status": "FAIL", "error": f"worker exited {completed['returncode']}"})
    else:
        result.update({"status": "FAIL", "error": "worker produced no result JSON"})
    return result


def _stream_worker_command(
    command: list[str], *, cwd: Path, log_path: Path, label: str,
    checkpoint_roots: list[Path] | None = None,
    log_limit_bytes: int = 32 * 1024 * 1024,
    heartbeat_seconds: float = 15.0,
) -> dict[str, Any]:
    """Stream subprocess output into one bounded durable log and the notebook cell."""
    existing_bytes = log_path.stat().st_size if log_path.exists() else 0
    remaining_log_bytes = log_limit_bytes - existing_bytes
    if remaining_log_bytes <= 0:
        raise RuntimeError(f"validation log reached its {log_limit_bytes}-byte cap; no worker launched: {log_path}")
    require_path_capacity(log_path, required_bytes=remaining_log_bytes, reserve_bytes=64 * 1024 * 1024)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] starting worker; live log: {log_path}", flush=True)
    start_time = time.monotonic()
    last_heartbeat = start_time
    stdout_bytes = bytearray()
    log_bytes = existing_bytes
    with log_path.open("ab") as log:
        header = f"\n\n=== attempt {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\nCOMMAND: {json.dumps(command)}\n".encode()
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                events = selector.select(timeout=min(5.0, heartbeat_seconds))
                if not events and process.poll() is None:
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        elapsed = int(now - start_time)
                        checkpoint_status = []
                        for checkpoint_root in checkpoint_roots or []:
                            manifests = sorted(checkpoint_root.rglob("generation-*.manifest.json")) if checkpoint_root.exists() else []
                            if manifests:
                                try:
                                    latest = _load_json(manifests[-1], "live worker checkpoint")
                                    checkpoint_status.append(
                                        f"{checkpoint_root.name}: gen={latest.get('generation')} kind={latest.get('kind')} "
                                        f"payload={latest.get('payload_bytes')}B"
                                    )
                                except ValueError:
                                    checkpoint_status.append(f"{checkpoint_root.name}: manifest currently unreadable")
                            else:
                                checkpoint_status.append(f"{checkpoint_root.name}: no verified manifest yet")
                        progress = "; ".join(checkpoint_status) or "worker has not emitted checkpoint path metadata"
                        message = f"[{label}] running {elapsed}s; {progress}\n"
                        print(message, end="", flush=True)
                        log.write(message.encode("utf-8"))
                        log.flush()
                        log_bytes += len(message.encode("utf-8"))
                        last_heartbeat = now
                    continue
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if log_bytes + len(chunk) > log_limit_bytes:
                        process.terminate()
                        process.wait(timeout=10)
                        raise RuntimeError(
                            f"validation worker log exceeded its {log_limit_bytes}-byte cap; "
                            "worker stopped safely and any Drive checkpoint remains resumable"
                        )
                    stdout_bytes.extend(chunk)
                    log.write(chunk)
                    log.flush()
                    log_bytes += len(chunk)
                    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
            returncode = process.wait()
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()
        footer = f"\nEXIT_CODE: {returncode}\n".encode()
        log.write(footer)
        log.flush()
        os.fsync(log.fileno())
    return {"returncode": returncode, "stdout": stdout_bytes.decode("utf-8", errors="replace")}


def _save_phase_report(root: Path, label: str, workflow_id: str, report: dict[str, Any]) -> Path:
    report.setdefault("validation_build_id", VALIDATION_BUILD_ID)
    path = root / "reports" / f"{label}_{workflow_id}_{_timestamp()}.json"
    write_immutable_json(path, report)
    return path


def _assert_schedule_identity(path: str | Path, schedule_sha256: str) -> None:
    if schedule_sha256 != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("validation workflow must use the approved effective schedule SHA-256")
    if sha256_file(path) != schedule_sha256:
        raise ValueError("validation run was given a schedule whose bytes do not match the approved hash")


def _validate_smoke_case_artifacts(
    output_root: str | Path, *, validation_root: Path, dataset: str, arm: str, schedule_sha256: str,
    expected_code_sha256: str | None = None,
) -> dict[str, Any]:
    """Recheck immutable synthetic smoke outputs without launching a worker."""
    from .config import DATASETS
    from .validation import validate_run_artifacts

    output = Path(output_root).expanduser().resolve()
    if not output.is_relative_to((validation_root / "smoke").resolve()):
        raise ValueError(f"smoke artifact directory is outside isolated validation storage: {output}")
    validation = validate_run_artifacts(output, require_production=False)
    result = _load_json(output / "full-result.json", "smoke result")
    config = _load_json(output / "config.json", "smoke configuration")
    if result.get("schedule_sha256") != schedule_sha256 or config.get("update_schedule_sha256") != schedule_sha256:
        raise ValueError("smoke result/config lost the effective schedule binding")
    if result.get("dataset") != dataset or result.get("arm") != arm:
        raise ValueError("smoke result dataset/arm identity mismatch")
    if result.get("namespace") != "SYNTHETIC_NON_REPORTABLE":
        raise ValueError("smoke artifact is not explicitly marked synthetic and non-reportable")
    code_sha = result.get("code_sha256") or config.get("code_sha256")
    if code_sha != config.get("code_sha256"):
        raise ValueError("smoke result/config code identities differ")
    if expected_code_sha256 is not None and code_sha != expected_code_sha256:
        raise ValueError("smoke output code identity does not match the compatibility audit")
    if result.get("classifier_updates") != 11:
        raise ValueError("smoke did not execute one fixed-size classifier update per task")
    sources = json.loads((output / "source_proportions.json").read_text(encoding="utf-8"))
    if len(sources) != 11 or any(
        sum(int(v) for v in row["realized_draws"].values()) != 256 for row in sources
    ):
        raise ValueError("source proportions or full 256-example updates are incomplete")
    if any(row.get("source_first_balancing") is not False for row in sources):
        raise ValueError("smoke used a different source balancing sampler")
    details = json.loads((output / "per_task_details.json").read_text(encoding="utf-8"))
    selections = [
        json.loads(line) for line in (output / "selection_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if arm in {"generated_only", "full_hybrid"} and (
        result.get("generator_steps", 0) <= 0 or result.get("critic_steps", 0) <= 0 or not selections
    ):
        raise ValueError("GAN/replay generation/selection path was not exercised")
    if arm == "derpp_matched" and (
        result.get("derpp_replay_updates", 0) <= 0 or not any(
            row.get("loss_terms", {}).get("replay_ce", 0) > 0
            and row.get("loss_terms", {}).get("logit_mse", 0) > 0 for row in details
        )
    ):
        raise ValueError("DER++ replay CE and stored-logit targets were not both exercised")
    if validation.get("valid") is not True:
        raise ValueError("artifact validator did not accept the smoke result")
    return {
        "artifact_validation": validation, "result": result, "config": config,
        "source_logging_rows": len(sources), "replay_selection_records": len(selections),
        "environment": _load_json(output / "environment.json", "smoke environment"),
        "artifact_manifest_sha256": sha256_file(output / "artifact_manifest.json"),
        "artifact_root": str(output), "code_sha256": code_sha,
    }


def _find_compatible_v4_smoke(
    root: Path, package_root: str | Path | None, schedule_sha256: str,
) -> tuple[Path | None, dict[str, Any], list[dict[str, Any]]] | None:
    package = Path(package_root).resolve() if package_root else Path(__file__).resolve().parents[1]
    compatibility_path = package / f"smoke_reuse_compatibility_{VALIDATION_BUILD_ID}.json"
    if not compatibility_path.is_file():
        compatibility_path = package / "smoke_reuse_compatibility.json"
    if not compatibility_path.is_file():
        return None
    compatibility = _load_json(compatibility_path, "v4 smoke reuse compatibility manifest")
    old_code_sha = compatibility.get("prior_runner_code_sha256")
    if (
        compatibility.get("status") != "PASS"
        or compatibility.get("validation_build_id") != VALIDATION_BUILD_ID
        or compatibility.get("smoke_outputs_reusable") is not True
        or compatibility.get("checkpoint_reuse_compatible") is not False
        or compatibility.get("expected_schedule_sha256") != schedule_sha256
        or not isinstance(old_code_sha, str)
        or len(old_code_sha) != 64
    ):
        return None
    reports = sorted((root / "reports").glob("gpu_smoke_*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    from .config import ARMS, DATASETS
    expected = {(dataset, arm) for dataset in ("ember", "az_class") for arm in ARMS}
    for report_path in reports:
        report = _load_json(report_path, "prior GPU smoke report")
        rows = report.get("completed_cases")
        if (
            report.get("status") != "PASS"
            or report.get("validation_build_id") != compatibility.get("prior_build_id")
            or report.get("schedule_sha256") != schedule_sha256
            or report.get("code_sha256") != old_code_sha
            or not isinstance(rows, list) or len(rows) != 10
            or any(row.get("status") != "PASS" for row in rows)
        ):
            continue
        found: set[tuple[str, str]] = set()
        validated_rows = []
        try:
            for row in rows:
                key = (row["dataset"], row["arm"])
                if key not in expected or key in found or row.get("schedule_sha256") != schedule_sha256:
                    raise ValueError("prior smoke report has duplicate or unexpected case identity")
                evidence = _validate_smoke_case_artifacts(
                    Path(row["artifact_root"]), validation_root=root, dataset=key[0], arm=key[1],
                    schedule_sha256=schedule_sha256, expected_code_sha256=old_code_sha,
                )
                validated_rows.append({**row, **evidence, "provenance": "REUSED_COMPATIBLE_V4_OUTPUT"})
                found.add(key)
            if found != expected:
                continue
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        return report_path, report, validated_rows

    # An interrupted v4 smoke may have no aggregate report at all. Reuse each
    # completed, independently hash-validated case rather than rerunning all 10.
    prior_build_id = compatibility.get("prior_build_id")
    if prior_build_id != "revision_R1_colab_validation_v4":
        return None
    valid_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in expected}
    smoke_root = root / "smoke"
    for result_path in sorted(smoke_root.rglob("full-result.json")) if smoke_root.is_dir() else []:
        output = result_path.parent
        try:
            result = _load_json(result_path, "candidate prior smoke result")
            key = (result.get("dataset"), result.get("arm"))
            if key not in expected:
                continue
            evidence = _validate_smoke_case_artifacts(
                output, validation_root=root, dataset=key[0], arm=key[1],
                schedule_sha256=schedule_sha256, expected_code_sha256=old_code_sha,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        result = evidence["result"]
        valid_by_case[key].append({
            "dataset": key[0], "feature_dim": int(DATASETS[key[0]]["feature_dim"]),
            "arm": key[1], "schedule_sha256": schedule_sha256,
            "worker_status": "REUSED_COMPATIBLE_V4_OUTPUT", "status": "PASS",
            "classifier_updates": result["classifier_updates"],
            "critic_steps": result["critic_steps"], "generator_steps": result["generator_steps"],
            "derpp_replay_updates": result["derpp_replay_updates"],
            "source_logging_rows": evidence["source_logging_rows"],
            "replay_selection_records": evidence["replay_selection_records"],
            "synthetic_namespace": result.get("namespace"),
            "environment": evidence["environment"],
            "artifact_manifest_sha256": evidence["artifact_manifest_sha256"],
            "artifact_root": str(output), "code_sha256": old_code_sha,
            "artifact_validation": evidence["artifact_validation"],
            "provenance": "REUSED_COMPATIBLE_V4_OUTPUT",
        })
    validated_rows = []
    for key in sorted(expected):
        # Multiple candidate outputs make selection ambiguous; let the current
        # validation run create a fresh, unambiguous case instead.
        if len(valid_by_case[key]) == 1:
            validated_rows.append(valid_by_case[key][0])
    if validated_rows:
        return None, {
            "status": "PARTIAL_COMPATIBLE_OUTPUTS",
            "validation_build_id": prior_build_id,
            "schedule_sha256": schedule_sha256,
            "code_sha256": old_code_sha,
            "provenance": "DISCOVERED_COMPLETED_V4_CASE_OUTPUTS",
        }, validated_rows
    return None


def _collect_checkpoint_file_evidence(checkpoint_root: Path) -> list[dict[str, Any]]:
    from .checkpointing import _validate_manifest

    records: list[dict[str, Any]] = []
    manifests = sorted(checkpoint_root.rglob("generation-*.manifest.json")) if checkpoint_root.is_dir() else []
    for manifest_path in manifests:
        manifest = _load_json(manifest_path, "checkpoint manifest before payload cleanup")
        identity = manifest.get("identity", {})
        if identity.get("execution_scope") != "synthetic_validation":
            raise ValueError("validation checkpoint cleanup found a non-synthetic identity")
        _validate_manifest(manifest_path.parent, manifest, identity)
        payload_path = manifest_path.parent / str(manifest["payload_file"])
        records.append({
            "checkpoint_root": str(checkpoint_root.resolve()),
            "manifest_relative_path": manifest_path.relative_to(checkpoint_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "payload_relative_path": payload_path.relative_to(checkpoint_root).as_posix(),
            "payload_sha256": sha256_file(payload_path),
            "payload_bytes": payload_path.stat().st_size,
            "execution_scope": "synthetic_validation",
            "schedule_sha256": identity.get("schedule_sha256"),
            "code_sha256": identity.get("code_sha256"),
            "checkpoint_kind": manifest.get("kind"),
            "generation": manifest.get("generation"),
        })
    return records


def _persist_smoke_case_evidence_and_prune(
    *, validation_root: Path, workflow_id: str, dataset: str, arm: str,
    schedule_sha256: str, code_sha256: str, checkpoint_root: Path,
    artifact_manifest_sha256: str,
    evidence_label: str = "",
) -> dict[str, Any]:
    label = f"_{evidence_label}" if evidence_label else ""
    evidence_path = validation_root / "reports" / f"smoke_case_{workflow_id}{label}_{dataset}_{arm}_checkpoint_evidence.json"
    if evidence_path.is_file():
        evidence = _load_json(evidence_path, "prior smoke checkpoint evidence")
        if (
            evidence.get("status") != "PASS"
            or evidence.get("schedule_sha256") != schedule_sha256
            or evidence.get("code_sha256") != code_sha256
            or evidence.get("artifact_manifest_sha256") != artifact_manifest_sha256
        ):
            raise ValueError("prior smoke checkpoint evidence has a different output/code/schedule identity")
    else:
        evidence = {
            "schema_version": 1, "status": "PASS",
            "artifact_type": "SYNTHETIC_SMOKE_ARTIFACT_VALIDATION",
            "validation_build_id": VALIDATION_BUILD_ID,
            "dataset": dataset, "arm": arm,
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha256,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "checkpoint_evidence": _collect_checkpoint_file_evidence(checkpoint_root),
            "checkpoint_payload_hashes_retain_manifest_and_report_evidence": True,
        }
        write_immutable_json(evidence_path, evidence)
    cleanup = prune_checkpoint_payloads_after_pass(checkpoint_root, evidence_path)
    return {"evidence_path": str(evidence_path), "cleanup": cleanup}


def _adopt_v4_smoke_checkpoint_evidence(
    *, validation_root: Path, workflow_id: str, schedule_sha256: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    output = Path(case["artifact_root"]).resolve()
    relative = output.relative_to((validation_root / "smoke").resolve())
    checkpoint_root = validation_root / "checkpoints" / relative
    evidence_label = "adopted_v4"
    evidence_path = validation_root / "reports" / (
        f"smoke_case_{workflow_id}_{evidence_label}_{case['dataset']}_{case['arm']}_checkpoint_evidence.json"
    )
    if evidence_path.is_file():
        return _persist_smoke_case_evidence_and_prune(
            validation_root=validation_root, workflow_id=workflow_id,
            dataset=case["dataset"], arm=case["arm"],
            schedule_sha256=schedule_sha256, code_sha256=case["code_sha256"],
            checkpoint_root=checkpoint_root,
            artifact_manifest_sha256=case["artifact_manifest_sha256"],
            evidence_label=evidence_label,
        )
    if not any(checkpoint_root.rglob("generation-*.pt")):
        return {"status": "NO_PRIOR_CHECKPOINT_PAYLOADS", "payloads_removed": 0}
    try:
        return _persist_smoke_case_evidence_and_prune(
            validation_root=validation_root, workflow_id=workflow_id,
            dataset=case["dataset"], arm=case["arm"],
            schedule_sha256=schedule_sha256, code_sha256=case["code_sha256"],
            checkpoint_root=checkpoint_root,
            artifact_manifest_sha256=case["artifact_manifest_sha256"],
            evidence_label=evidence_label,
        )
    except Exception as exc:
        # Reuse remains valid because the run artifacts already passed their
        # validator; preserve checkpoint files if cleanup evidence is incomplete.
        return {
            "status": "NOT_PRUNED_FAIL_CLOSED", "payloads_removed": 0,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def run_gpu_smoke(
    *,
    validation_root: str | Path,
    schedule_path: str | Path,
    schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
    local_checkpoint_root: str | Path = "/content/revision_r1_validation_staging",
    run_missing_cases: bool = False,
) -> dict[str, Any]:
    """Adopt compatible cases; launch missing synthetic CUDA cases only by explicit opt-in."""
    root = _validation_root(validation_root)
    _assert_schedule_identity(schedule_path, schedule_sha256)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    from .config import ARMS, DATASETS
    prior = _find_compatible_v4_smoke(root, package_root, schedule_sha256)
    if prior is not None:
        prior_path, prior_report, prior_cases = prior
        if len(prior_cases) == 10:
            cleanup = [
                _adopt_v4_smoke_checkpoint_evidence(
                    validation_root=root, workflow_id=workflow_id,
                    schedule_sha256=schedule_sha256, case=case,
                )
                for case in prior_cases
            ]
            report = {
                "schema_version": 1, "status": "PASS",
                "artifact_type": "SYNTHETIC_GPU_SMOKE_COMPATIBLE_PRIOR_OUTPUTS_NOT_RERUN",
                "validation_build_id": VALIDATION_BUILD_ID,
                "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
                "expected_cases": 10, "completed_cases": prior_cases,
                "checkpoint_payload_cleanup": cleanup,
                "provenance": {
                    "status": "REUSED_COMPATIBLE_V4_OUTPUTS",
                    "prior_report_path": str(prior_path) if prior_path else None,
                    "prior_report_sha256": sha256_file(prior_path) if prior_path else None,
                    "prior_build_id": prior_report.get("validation_build_id"),
                    "prior_code_sha256": prior_report.get("code_sha256"),
                    "completed_cases_discovered_without_aggregate_report": prior_path is None,
                    "current_code_executed_for_smoke": False,
                    "training_algorithm_source_compatibility_audited": True,
                    "v4_checkpoints_are_not_reusable": True,
                },
                "uses_actual_dataset": False, "scientific_settings_tuned": False,
            }
            report_path = _save_phase_report(root, "gpu_smoke", workflow_id, report)
            return {**report, "report_path": str(report_path)}
        prior_case_map = {(case["dataset"], case["arm"]): case for case in prior_cases}
    else:
        prior_report = None
        prior_cases = []
        prior_case_map = {}
    expected_cases = {(dataset, arm) for dataset in ("ember", "az_class") for arm in ARMS}
    missing_cases = sorted(expected_cases - set(prior_case_map))
    if missing_cases and not run_missing_cases:
        cleanup = [
            _adopt_v4_smoke_checkpoint_evidence(
                validation_root=root, workflow_id=workflow_id,
                schedule_sha256=schedule_sha256, case=case,
            )
            for case in prior_cases
        ]
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "artifact_type": "SYNTHETIC_GPU_SMOKE_MISSING_CASES_DISABLED",
            "validation_build_id": VALIDATION_BUILD_ID,
            "reason": "smoke execution for missing cases is disabled by default; no new GPU smoke jobs were launched",
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
            "expected_cases": 10, "completed_cases": prior_cases,
            "reused_compatible_v4_cases": len(prior_cases),
            "missing_cases": [{"dataset": dataset, "arm": arm} for dataset, arm in missing_cases],
            "smoke_execution_enabled": False,
            "checkpoint_payload_cleanup": cleanup,
        }
        report_path = _save_phase_report(root, "gpu_smoke", workflow_id, report)
        return {**report, "report_path": str(report_path)}
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available, gpu_name = False, None
    if not cuda_available:
        cleanup = [
            _adopt_v4_smoke_checkpoint_evidence(
                validation_root=root, workflow_id=workflow_id,
                schedule_sha256=schedule_sha256, case=case,
            )
            for case in prior_cases
        ]
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "artifact_type": "SYNTHETIC_GPU_SMOKE_NOT_PRODUCTION_EVIDENCE",
            "reason": "CUDA/PyTorch GPU runtime is unavailable; no CPU result is promoted to GPU PASS.",
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
            "expected_cases": 10, "completed_cases": prior_cases,
            "reused_compatible_v4_cases": len(prior_cases),
            "missing_cases": [{"dataset": dataset, "arm": arm} for dataset, arm in missing_cases],
            "smoke_execution_enabled": True,
            "checkpoint_payload_cleanup": cleanup,
        }
        report_path = _save_phase_report(root, "gpu_smoke", workflow_id, report)
        return {**report, "report_path": str(report_path)}

    check_validation_storage_capacity(
        local_path=local_checkpoint_root, drive_path=root,
    )
    cases: list[dict[str, Any]] = []
    for dataset in ("ember", "az_class"):
        for arm in ARMS:
            if (dataset, arm) in prior_case_map:
                case = prior_case_map[(dataset, arm)]
                case["checkpoint_payload_cleanup"] = _adopt_v4_smoke_checkpoint_evidence(
                    validation_root=root, workflow_id=workflow_id,
                    schedule_sha256=schedule_sha256, case=case,
                )
                cases.append(case)
                continue
            case_id = hashlib.sha256(f"{workflow_id}|{dataset}|{arm}|seed42".encode()).hexdigest()[:14]
            job_root = root / "smoke" / workflow_id / dataset / arm / case_id
            drive_checkpoints = root / "checkpoints" / workflow_id / "smoke" / dataset / arm / case_id
            local = Path(local_checkpoint_root) / "validation" / workflow_id / "smoke" / dataset / arm / case_id
            worker = _run_worker(root, workflow_id, f"smoke_{dataset}_{arm}", [
                "synthetic", "--mode", "run", "--dataset", dataset, "--arm", arm,
                "--schedule-sha256", schedule_sha256, "--output-dir", str(job_root),
                "--drive-checkpoint-root", str(drive_checkpoints), "--local-checkpoint-root", str(local),
            ])
            case: dict[str, Any] = {
                "dataset": dataset, "feature_dim": int(DATASETS[dataset]["feature_dim"]),
                "arm": arm, "schedule_sha256": schedule_sha256,
                "worker_status": worker.get("status"), "log_path": worker["log_path"],
                "artifact_root": str(job_root),
            }
            try:
                if worker.get("returncode") != 0:
                    raise ValueError(worker.get("error", "worker failed"))
                checked = _validate_smoke_case_artifacts(
                    job_root, validation_root=root, dataset=dataset, arm=arm,
                    schedule_sha256=schedule_sha256, expected_code_sha256=code_sha,
                )
                local_cleanup = _cleanup_local_checkpoint_after_drive_verify(
                    local, drive_checkpoints, schedule_sha256,
                )
                result = checked["result"]
                checkpoint_evidence = _persist_smoke_case_evidence_and_prune(
                    validation_root=root, workflow_id=workflow_id, dataset=dataset,
                    arm=arm, schedule_sha256=schedule_sha256, code_sha256=code_sha,
                    checkpoint_root=drive_checkpoints,
                    artifact_manifest_sha256=checked["artifact_manifest_sha256"],
                )
                case.update({
                    "status": "PASS", "artifact_validation": checked["artifact_validation"],
                    "classifier_updates": result["classifier_updates"],
                    "critic_steps": result["critic_steps"], "generator_steps": result["generator_steps"],
                    "derpp_replay_updates": result["derpp_replay_updates"],
                    "source_logging_rows": checked["source_logging_rows"],
                    "replay_selection_records": checked["replay_selection_records"],
                    "synthetic_namespace": result.get("namespace"),
                    "environment": checked["environment"],
                    "artifact_manifest_sha256": checked["artifact_manifest_sha256"],
                    "local_staging_cleanup": local_cleanup,
                    "checkpoint_payload_cleanup": checkpoint_evidence,
                    "provenance": "CURRENT_V5_RUN",
                })
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                case.update({"status": "FAIL", "error": str(exc)})
            cases.append(case)

    phase_status = "PASS" if len(cases) == 10 and all(row["status"] == "PASS" for row in cases) else "FAIL"
    report = {
        "schema_version": 1, "status": phase_status,
        "artifact_type": "SYNTHETIC_GPU_SMOKE_NOT_PRODUCTION_EVIDENCE",
        "gpu": gpu_name, "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        "expected_cases": 10, "completed_cases": cases,
        "reused_compatible_v4_cases": len(prior_case_map),
        "smoke_execution_enabled": True,
        "covers": ["all five arms", "both exact feature dimensions", "GAN", "classifier", "replay selection", "DER++ CE/logit targets", "metrics", "source proportions", "artifact validation"],
        "uses_actual_dataset": False, "scientific_settings_tuned": False,
    }
    report_path = _save_phase_report(root, "gpu_smoke", workflow_id, report)
    return {**report, "report_path": str(report_path)}


def _launch_synthetic_worker(
    root: Path, workflow_id: str, dataset: str, arm: str, schedule_sha256: str,
    output: Path, drive_checkpoints: Path, local_root: Path,
    *, interruption: str | None = None, checkpoint_sync_policy: str = "all",
    checkpoint_write_policy: str = "all",
    runner_identity_code_sha256: str | None = None, worker_label: str | None = None,
    cpu_test_only: bool = False,
) -> dict[str, Any]:
    mode = "interrupt" if interruption else "run"
    arguments = [
        "synthetic", "--mode", mode, "--dataset", dataset, "--arm", arm,
        "--schedule-sha256", schedule_sha256, "--output-dir", str(output),
        "--drive-checkpoint-root", str(drive_checkpoints), "--local-checkpoint-root", str(local_root),
        "--checkpoint-sync-policy", checkpoint_sync_policy,
        "--checkpoint-write-policy", checkpoint_write_policy,
    ]
    if interruption:
        arguments.extend(["--interruption", interruption])
    if runner_identity_code_sha256:
        arguments.extend(["--runner-identity-code-sha256", runner_identity_code_sha256])
    if cpu_test_only:
        arguments.append("--cpu-test-only")
    result = _run_worker(root, workflow_id, worker_label or f"resume_{dataset}_{arm}_{mode}", arguments)
    if checkpoint_sync_policy == "all":
        _cleanup_local_checkpoint_after_drive_verify(local_root, drive_checkpoints, schedule_sha256)
    return result


def _cleanup_local_checkpoint_after_drive_verify(
    local_root: str | Path, drive_root: str | Path, expected_schedule_sha256: str,
) -> dict[str, Any]:
    """Drop only this worker's ephemeral staging after its Drive copy verifies."""
    local = Path(local_root).expanduser().resolve()
    safe_base = Path(allowed_staging_root).expanduser().resolve() / "validation"
    if local == safe_base or not local.is_relative_to(safe_base):
        raise ValueError(f"refusing to clean local checkpoint path outside validation staging: {local}")
    if not local.exists():
        return {"status": "NO_LOCAL_STAGING", "path": str(local), "bytes_removed": 0}
    if local.is_symlink():
        raise ValueError(f"refusing to clean symlinked local checkpoint staging: {local}")
    recovery_root = Path(drive_root) / "recovery"
    identity = _drive_checkpoint_identity(Path(drive_root), expected_schedule_sha256)
    from .checkpointing import load_latest_valid
    loaded = load_latest_valid(recovery_root, identity)
    if loaded.state.identity != identity:
        raise ValueError("Drive recovery identity differs from local staging cleanup identity")
    bytes_removed = sum(path.stat().st_size for path in local.rglob("*") if path.is_file())
    shutil.rmtree(local)
    return {
        "status": "LOCAL_STAGING_REMOVED_AFTER_DRIVE_VERIFY",
        "path": str(local), "bytes_removed": bytes_removed,
        "latest_drive_generation": loaded.manifest.get("generation"),
        "schedule_sha256": expected_schedule_sha256,
    }


def _drive_checkpoint_identity(root: Path, expected_schedule_sha256: str) -> dict[str, Any]:
    manifests = sorted((root / "recovery").glob("generation-*.manifest.json"))
    if not manifests:
        raise ValueError(f"no Drive-persisted recovery checkpoint generations found: {root}")
    payload = _load_json(manifests[-1], "Drive checkpoint manifest")
    identity = payload.get("identity", {})
    if identity.get("schedule_sha256") != expected_schedule_sha256:
        raise ValueError("Drive checkpoint is not bound to the approved effective schedule")
    if identity.get("execution_scope") != "synthetic_validation":
        raise ValueError("resume checkpoint is not isolated as synthetic validation")
    return identity


def _compare_torch_state(left: Any, right: Any, *, atol: float, rtol: float, path: str = "root") -> tuple[bool, float]:
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return False, float("inf")
        if left.is_floating_point() or right.is_floating_point():
            delta = float((left.detach().cpu() - right.detach().cpu()).abs().max().item()) if left.numel() else 0.0
            return bool(torch.allclose(left.detach().cpu(), right.detach().cpu(), atol=atol, rtol=rtol)), delta
        return bool(torch.equal(left.cpu(), right.cpu())), 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, float("inf")
        maximum = 0.0
        for key in left:
            same, delta = _compare_torch_state(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
            if not same:
                return False, delta
            maximum = max(maximum, delta)
        return True, maximum
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, float("inf")
        maximum = 0.0
        for index, (a, b) in enumerate(zip(left, right)):
            same, delta = _compare_torch_state(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")
            if not same:
                return False, delta
            maximum = max(maximum, delta)
        return True, maximum
    try:
        equal = left == right
        if isinstance(equal, np.ndarray):
            equal = bool(equal.all())
        return bool(equal), 0.0
    except Exception:
        return False, float("inf")


def _compare_task_matrices(
    baseline_matrix: Any,
    resumed_matrix: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> float:
    """Validate and compare the lower-triangular, evaluated task metrics."""
    expected_shape = (11, 11)
    try:
        baseline = np.asarray(baseline_matrix, dtype=object)
        resumed = np.asarray(resumed_matrix, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resume task matrix cannot be represented as a rectangular array: {exc}") from exc

    if baseline.shape != resumed.shape or baseline.shape != expected_shape:
        raise ValueError(
            "resume task matrix shape mismatch: "
            f"baseline={baseline.shape}, resumed={resumed.shape}; expected {expected_shape}"
        )

    # None is the serialized representation of an unevaluated future-task cell.
    # Compare the missing masks first, then validate each against the expected
    # lower-triangular evaluated region (including the diagonal).
    for row in range(expected_shape[0]):
        for column in range(expected_shape[1]):
            baseline_missing = baseline[row, column] is None
            resumed_missing = resumed[row, column] is None
            if baseline_missing != resumed_missing:
                raise ValueError(
                    "resume task-matrix missing-value masks differ first at "
                    f"stage {row + 1}, task {column + 1} (cell[{row},{column}])"
                )

    baseline_values: list[float] = []
    resumed_values: list[float] = []
    evaluated_cells: list[tuple[int, int]] = []
    for row in range(expected_shape[0]):
        for column in range(expected_shape[1]):
            baseline_value = baseline[row, column]
            resumed_value = resumed[row, column]
            if column > row:
                if baseline_value is not None or resumed_value is not None:
                    raise ValueError(
                        "resume task matrix has an unexpected populated future cell at "
                        f"stage {row + 1}, task {column + 1} (cell[{row},{column}])"
                    )
                continue
            if baseline_value is None or resumed_value is None:
                raise ValueError(
                    "resume task matrix has a missing evaluated value at "
                    f"stage {row + 1}, task {column + 1} (cell[{row},{column}])"
                )
            try:
                baseline_float = float(baseline_value)
                resumed_float = float(resumed_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "resume task matrix has a non-numeric evaluated value at "
                    f"stage {row + 1}, task {column + 1} (cell[{row},{column}])"
                ) from exc
            if not math.isfinite(baseline_float) or not math.isfinite(resumed_float):
                raise ValueError(
                    "resume task matrix has a non-finite evaluated metric at "
                    f"stage {row + 1}, task {column + 1} (cell[{row},{column}]): "
                    f"baseline={baseline_float!r}, resumed={resumed_float!r}"
                )
            baseline_values.append(baseline_float)
            resumed_values.append(resumed_float)
            evaluated_cells.append((row, column))

    baseline_array = np.asarray(baseline_values, dtype=float)
    resumed_array = np.asarray(resumed_values, dtype=float)
    finite_deltas = [
        abs(left - right)
        for left, right in zip(baseline_values, resumed_values)
        if math.isfinite(abs(left - right))
    ]
    max_finite_abs_error = max(finite_deltas, default=0.0)
    close = np.isclose(baseline_array, resumed_array, atol=atol, rtol=rtol, equal_nan=False)
    if not bool(np.all(close)):
        first_index = int(np.flatnonzero(~close)[0])
        row, column = evaluated_cells[first_index]
        delta = abs(baseline_values[first_index] - resumed_values[first_index])
        raise ValueError(
            "resume task-matrix metrics differ; "
            f"first difference at stage {row + 1}, task {column + 1} "
            f"(cell[{row},{column}]): baseline={baseline_values[first_index]!r}, "
            f"resumed={resumed_values[first_index]!r}, abs_error={delta!r}, "
            f"max_finite_abs_error={max_finite_abs_error:.12g}; "
            f"tolerance (atol={atol:g}, rtol={rtol:g})"
        )
    return float(max_finite_abs_error)


def compare_resumed_outputs(
    baseline_root: str | Path, resumed_root: str | Path, *,
    baseline_checkpoint_root: str | Path, resumed_checkpoint_root: str | Path,
    cpu_test_only: bool = False,
) -> dict[str, Any]:
    """Compare fresh-process recovery to uninterrupted synthetic execution."""
    baseline, resumed = Path(baseline_root), Path(resumed_root)
    left = _load_json(baseline / "full-result.json", "uninterrupted synthetic result")
    right = _load_json(resumed / "full-result.json", "resumed synthetic result")
    exact_fields = (
        "dataset", "seed", "arm", "schedule_sha256", "classifier_updates", "sampled_examples",
        "generator_steps", "critic_steps", "derpp_replay_updates",
    )
    for field in exact_fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"resume agreement failed for exact result field {field}: {left.get(field)} != {right.get(field)}")
    if left.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("synthetic resume comparison is not bound to the approved effective schedule")

    with np.load(baseline / "predictions.npz", allow_pickle=False) as expected, np.load(resumed / "predictions.npz", allow_pickle=False) as observed:
        if set(expected.files) != set(observed.files):
            raise ValueError("final prediction artifact arrays differ")
        for name in expected.files:
            if not np.array_equal(expected[name], observed[name]):
                raise ValueError(f"synthetic resume final prediction array differs: {name}")
    with np.load(baseline / "task_boundary_predictions.npz", allow_pickle=False) as expected, np.load(resumed / "task_boundary_predictions.npz", allow_pickle=False) as observed:
        if set(expected.files) != set(observed.files):
            raise ValueError("task-boundary prediction array sets differ")
        for name in expected.files:
            if not np.array_equal(expected[name], observed[name]):
                raise ValueError(f"synthetic resume task-boundary prediction differs: {name}")

    left_matrix = _load_json(baseline / "task_matrix.json", "baseline task matrix")["task_matrix"]
    right_matrix = _load_json(resumed / "task_matrix.json", "resumed task matrix")["task_matrix"]
    max_task_matrix_abs_error = _compare_task_matrices(left_matrix, right_matrix, atol=1e-5, rtol=1e-4)
    left_sources = json.loads((baseline / "source_proportions.json").read_text(encoding="utf-8"))
    right_sources = json.loads((resumed / "source_proportions.json").read_text(encoding="utf-8"))
    if left_sources != right_sources:
        raise ValueError("resume source/class sampling proportions or realized draws differ")

    left_trace = [json.loads(line) for line in (baseline / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    right_trace = [json.loads(line) for line in (resumed / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(left_trace) != len(right_trace):
        raise ValueError("resume update-trace lengths differ")
    max_loss_delta = 0.0
    for a, b in zip(left_trace, right_trace):
        for field in ("task_id", "update", "batch_size", "source_draws", "class_draws", "sampler_cursor"):
            if a.get(field) != b.get(field):
                raise ValueError(f"resume update trace differs in exact field {field}")
        delta = abs(float(a["loss"]) - float(b["loss"]))
        max_loss_delta = max(max_loss_delta, delta)
        if not math.isclose(float(a["loss"]), float(b["loss"]), abs_tol=5e-5, rel_tol=1e-4):
            raise ValueError("resume classifier update loss exceeds tolerance (atol=5e-5, rtol=1e-4)")

    import torch
    baseline_models = torch.load(baseline / "final_models.pt", map_location="cpu", weights_only=False)
    resumed_models = torch.load(resumed / "final_models.pt", map_location="cpu", weights_only=False)
    models_equal, max_model_delta = _compare_torch_state(baseline_models, resumed_models, atol=1e-5, rtol=1e-4)
    if not models_equal:
        raise ValueError(f"resume final model state exceeds tolerance; max_abs_delta={max_model_delta}")

    from .checkpointing import _validate_manifest, load_latest_valid

    checkpoint_states = []
    checkpoint_evidence: list[dict[str, Any]] = []
    for label, checkpoint_root in (
        ("uninterrupted", Path(baseline_checkpoint_root)),
        ("resumed", Path(resumed_checkpoint_root)),
    ):
        manifests = sorted((checkpoint_root / "recovery").glob("generation-*.manifest.json"))
        if not manifests:
            raise ValueError(f"{label} Drive checkpoint has no recovery generations")
        latest_manifest = _load_json(manifests[-1], f"{label} latest Drive checkpoint manifest")
        if latest_manifest.get("kind") != "complete":
            raise ValueError(f"{label} Drive checkpoint is not a completed run checkpoint")
        identity = latest_manifest.get("identity", {})
        if identity.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256 or identity.get("execution_scope") != "synthetic_validation":
            raise ValueError(f"{label} Drive checkpoint identity is not the approved synthetic schedule identity")
        loaded = load_latest_valid(checkpoint_root / "recovery", identity)
        if loaded.manifest.get("kind") != "complete" or loaded.state.cursor.phase != "complete":
            raise ValueError(f"{label} latest verified checkpoint is not at the complete phase")
        checkpoint_states.append(loaded.state)
        recovery_root = checkpoint_root / "recovery"
        for manifest_path in manifests:
            manifest = _load_json(manifest_path, f"{label} recovery checkpoint manifest")
            _validate_manifest(recovery_root, manifest, identity)
            payload_path = recovery_root / str(manifest["payload_file"])
            checkpoint_evidence.append({
                "checkpoint_root": str(checkpoint_root.resolve()),
                "manifest_relative_path": manifest_path.relative_to(checkpoint_root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "payload_relative_path": payload_path.relative_to(checkpoint_root).as_posix(),
                "payload_sha256": sha256_file(payload_path),
                "payload_bytes": payload_path.stat().st_size,
                "execution_scope": "synthetic_validation",
                "schedule_sha256": identity.get("schedule_sha256"),
                "code_sha256": identity.get("code_sha256"),
                "checkpoint_kind": manifest.get("kind"),
                "generation": manifest.get("generation"),
            })

    if checkpoint_states[0].identity != checkpoint_states[1].identity:
        raise ValueError("uninterrupted and resumed checkpoint identities differ")
    optimizer_deltas: dict[str, float] = {}
    for field in ("classifier_optimizer_state", "generator_optimizer_state", "critic_optimizer_state"):
        equal, delta = _compare_torch_state(
            getattr(checkpoint_states[0], field), getattr(checkpoint_states[1], field),
            atol=1e-5, rtol=1e-4, path=field,
        )
        if not equal:
            raise ValueError(f"resume checkpoint {field} exceeds tolerance; max_abs_delta={delta}")
        optimizer_deltas[field] = delta

    gan_rng_keys = ("gan_torch_cpu_rng",) if cpu_test_only else ("gan_torch_cpu_rng", "gan_torch_cuda_rng")
    for label, checkpoint_state in zip(("uninterrupted", "resumed"), checkpoint_states):
        if not checkpoint_state.rng_state:
            raise ValueError(f"{label} Drive checkpoint is missing its captured RNG state")
        if not all(key in checkpoint_state.extra for key in gan_rng_keys):
            raise ValueError(f"{label} Drive checkpoint is missing its separate GAN RNG state: {gan_rng_keys}")
    rng_equal, _ = _compare_torch_state(
        checkpoint_states[0].rng_state, checkpoint_states[1].rng_state,
        atol=0.0, rtol=0.0, path="rng_state",
    )
    if not rng_equal:
        raise ValueError("resume checkpoint RNG state differs (exact comparison)")
    gan_rng_equal, _ = _compare_torch_state(
        {key: checkpoint_states[0].extra[key] for key in gan_rng_keys},
        {key: checkpoint_states[1].extra[key] for key in gan_rng_keys},
        atol=0.0, rtol=0.0, path="gan_rng_state",
    )
    if not gan_rng_equal:
        raise ValueError("resume checkpoint separate GAN RNG state differs (exact comparison)")
    baseline_selections = [
        json.loads(line) for line in (baseline / "selection_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    resumed_selections = [
        json.loads(line) for line in (resumed / "selection_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for selection_rows in (baseline_selections, resumed_selections):
        for row in selection_rows:
            # Candidate distances are diagnostic arrays; compare stable content identity/indices.
            row.pop("candidate_l1_distances", None)
    if baseline_selections != resumed_selections:
        raise ValueError("resume candidate selection records or generated replay identities differ")
    return {
        "status": "PASS",
        "exact_counters_and_sampling": True,
        "final_predictions_exact": True,
        "task_matrix_atol": 1e-5, "task_matrix_rtol": 1e-4,
        "max_task_matrix_finite_absolute_error": max_task_matrix_abs_error,
        "classifier_loss_atol": 5e-5, "classifier_loss_rtol": 1e-4,
        "model_state_atol": 1e-5, "model_state_rtol": 1e-4,
        "optimizer_state_atol": 1e-5, "optimizer_state_rtol": 1e-4,
        "max_classifier_loss_absolute_delta": max_loss_delta,
        "max_model_state_absolute_delta": max_model_delta,
        "max_optimizer_state_absolute_deltas": optimizer_deltas,
        "rng_state_exact": True,
        "separate_gan_rng_state_exact": True,
        "separate_gan_rng_devices_compared": list(gan_rng_keys),
        "verified_drive_checkpoint_kinds": ["complete", "complete"],
        "checkpoint_evidence": checkpoint_evidence,
        "replay_selection_records_compared": len(baseline_selections),
        "synthetic_test_predictions_used_only_for_resume_equivalence": True,
        "scientific_settings_tuned": False,
    }


def _output_complete(output: Path) -> bool:
    markers = list((output / "registry" / "runs").glob("*/COMPLETE.json"))
    return len(markers) == 1


def _prior_passed_resume_case(
    root: Path, workflow_id: str, scenario: str,
    schedule_sha256: str, code_sha256: str,
) -> dict[str, Any] | None:
    """Reuse process evidence only when a prior immutable PASS binds the same code/schedule."""
    reports = sorted(
        (root / "reports").glob(f"resume_validation_{workflow_id}_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for report_path in reports:
        report = _load_json(report_path, "prior resume validation report")
        if (
            report.get("status") != "PASS"
            or report.get("schedule_sha256") != schedule_sha256
            or report.get("code_sha256") != code_sha256
        ):
            continue
        for case in report.get("cases", []):
            process_ids = case.get("worker_process_ids", [])
            if (
                case.get("scenario") == scenario
                and case.get("status") == "PASS"
                and case.get("resume_worker_invoked") is True
                and len(process_ids) == 3
                and len(set(process_ids)) == 3
            ):
                return case
    return None


def _resume_case_evidence_path(root: Path, workflow_id: str, scenario: str) -> Path:
    if Path(scenario).name != scenario:
        raise ValueError("resume scenario must be a single safe path component")
    return root / "reports" / f"resume_case_{workflow_id}_{scenario}.json"


def _persist_resume_case_evidence_and_prune(
    *, root: Path, workflow_id: str, schedule_sha256: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    code_sha256 = validation_code_sha256()
    scenario = str(case["scenario"])
    evidence_path = _resume_case_evidence_path(root, workflow_id, scenario)
    if evidence_path.is_file():
        evidence = _load_json(evidence_path, "resume comparison evidence")
        if (
            evidence.get("status") != "PASS"
            or evidence.get("schedule_sha256") != schedule_sha256
            or evidence.get("code_sha256") != code_sha256
            or evidence.get("scenario") != scenario
        ):
            raise ValueError("persisted resume comparison evidence has a different identity")
    else:
        comparison = case.get("comparison", {})
        checkpoints = comparison.get("checkpoint_evidence", [])
        if not checkpoints:
            raise ValueError("successful resume comparison lacks checkpoint hash evidence")
        evidence = {
            "schema_version": 1, "status": "PASS",
            "artifact_type": "SYNTHETIC_FRESH_PROCESS_RESUME_COMPARISON",
            "validation_build_id": VALIDATION_BUILD_ID,
            "scenario": scenario,
            "schedule_sha256": schedule_sha256,
            "code_sha256": code_sha256,
            "case": case,
            "checkpoint_evidence": checkpoints,
            "numerical_tolerances_unchanged": True,
        }
        write_immutable_json(evidence_path, evidence)

    cleanup_results = []
    roots = sorted({str(item["checkpoint_root"]) for item in evidence["checkpoint_evidence"]})
    for checkpoint_root in roots:
        cleanup_results.append(prune_checkpoint_payloads_after_pass(checkpoint_root, evidence_path))
    return {
        "evidence_path": str(evidence_path),
        "cleanup_results": cleanup_results,
        "payloads_removed": sum(int(item.get("payloads_removed", 0)) for item in cleanup_results),
        "manifests_retained": sum(int(item.get("manifests_retained", 0)) for item in cleanup_results),
    }


def _execute_resume_case(
    *, root: Path, workflow_id: str, schedule_sha256: str, scenario: str,
    arm: str, interruption: str, resume_now: bool,
    require_new_resume_process: bool = False,
) -> dict[str, Any]:
    dataset = "ember"
    case_root = root / "resume" / workflow_id / scenario
    baseline_output = case_root / "uninterrupted"
    resumed_output = case_root / "interrupted_then_resumed"
    baseline_drive = root / "checkpoints" / workflow_id / "resume" / scenario / "baseline"
    resumed_drive = root / "checkpoints" / workflow_id / "resume" / scenario / "resumed"
    parent = Path("/content/revision_r1_validation_staging") / "validation" / workflow_id / scenario
    baseline_local = parent / "baseline"
    interrupted_local = parent / "interrupt"
    resumed_local = parent / "resumed"

    case_evidence_path = _resume_case_evidence_path(root, workflow_id, scenario)
    if case_evidence_path.is_file():
        evidence = _load_json(case_evidence_path, "completed resume comparison evidence")
        if (
            evidence.get("status") != "PASS"
            or evidence.get("schedule_sha256") != schedule_sha256
            or evidence.get("code_sha256") != validation_code_sha256()
            or evidence.get("scenario") != scenario
            or not isinstance(evidence.get("case"), dict)
        ):
            raise ValueError("existing resume comparison report cannot be reused for this identity")
        prior_case = dict(evidence["case"])
        prior_case["checkpoint_cleanup"] = _persist_resume_case_evidence_and_prune(
            root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            case=prior_case,
        )
        return prior_case

    if not _output_complete(baseline_output):
        baseline_worker = _launch_synthetic_worker(
            root, workflow_id, dataset, arm, schedule_sha256,
            baseline_output, baseline_drive, baseline_local,
        )
    else:
        baseline_worker = {"status": "ALREADY_VALIDATED", "returncode": 0, "worker_pid": None}
    if baseline_worker.get("returncode") not in {None, 0} or baseline_worker.get("status") not in {"SYNTHETIC_VALIDATED_NON_REPORTABLE", "ALREADY_VALIDATED"}:
        raise RuntimeError(f"uninterrupted resume-control job failed: {baseline_worker}")

    interrupted_status_path = resumed_output / "status.json"
    if _output_complete(resumed_output):
        if require_new_resume_process:
            raise ValueError("manual reset fixture is already complete; refusing to count a skipped resume as a fresh-process PASS")
        interrupted_worker = {"status": "ALREADY_VALIDATED", "returncode": 0, "worker_pid": None}
    elif interrupted_status_path.is_file() and _load_json(interrupted_status_path, "interrupted run status").get("status") == "INTERRUPTED_RESUMABLE":
        interrupted_worker = {"status": "INTERRUPTED_RESUMABLE", "returncode": 0, "worker_pid": None}
    else:
        interrupted_worker = _launch_synthetic_worker(
            root, workflow_id, dataset, arm, schedule_sha256,
            resumed_output, resumed_drive, interrupted_local, interruption=interruption,
        )
    if interrupted_worker.get("status") not in {"INTERRUPTED_RESUMABLE", "ALREADY_VALIDATED"}:
        raise RuntimeError(f"interrupted {scenario} job did not stop resumably: {interrupted_worker}")
    checkpoint_identity = _drive_checkpoint_identity(resumed_drive, schedule_sha256)

    if not resume_now:
        if interrupted_worker.get("status") != "INTERRUPTED_RESUMABLE" or _output_complete(resumed_output):
            raise ValueError("manual runtime-reset preparation did not leave a genuinely incomplete resumable run")
        return {
            "status": "PENDING_RUNTIME_RESET",
            "scenario": scenario, "arm": arm,
            "baseline_output": str(baseline_output), "interrupted_output": str(resumed_output),
            "drive_checkpoint_root": str(resumed_drive),
            "baseline_worker_pid": baseline_worker.get("worker_pid"),
            "interrupt_worker_pid": interrupted_worker.get("worker_pid"),
            "schedule_sha256": schedule_sha256,
            "checkpoint_identity_sha256": hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True).encode()).hexdigest(),
        }

    if not _output_complete(resumed_output):
        resumed_worker = _launch_synthetic_worker(
            root, workflow_id, dataset, arm, schedule_sha256,
            resumed_output, resumed_drive, resumed_local,
        )
    else:
        resumed_worker = {"status": "ALREADY_VALIDATED", "returncode": 0, "worker_pid": None}
    if require_new_resume_process and (
        resumed_worker.get("status") != "SYNTHETIC_VALIDATED_NON_REPORTABLE"
        or resumed_worker.get("worker_pid") is None
    ):
        raise ValueError("manual runtime-reset did not invoke a new worker to resume the incomplete Drive checkpoint")
    if resumed_worker.get("returncode") not in {None, 0} or resumed_worker.get("status") not in {"SYNTHETIC_VALIDATED_NON_REPORTABLE", "ALREADY_VALIDATED"}:
        raise RuntimeError(f"fresh-process resume failed for {scenario}: {resumed_worker}")
    _drive_checkpoint_identity(resumed_drive, schedule_sha256)
    comparison = compare_resumed_outputs(
        baseline_output, resumed_output,
        baseline_checkpoint_root=baseline_drive, resumed_checkpoint_root=resumed_drive,
    )
    process_ids = [value for value in (baseline_worker.get("worker_pid"), interrupted_worker.get("worker_pid"), resumed_worker.get("worker_pid")) if value is not None]
    execution_mode = "FRESH_THREE_WORKER_PROCESSES"
    if require_new_resume_process:
        execution_mode = "FRESH_POST_RESET_RESUME_WORKER; PRE_RESET_WORKERS_RECORDED_IN_PENDING_FIXTURE"
    elif len(process_ids) != 3 or len(set(process_ids)) != 3:
        prior_case = _prior_passed_resume_case(root, workflow_id, scenario, schedule_sha256, validation_code_sha256())
        if prior_case is None:
            raise ValueError("resume artifacts were cached but no prior identity-matched PASS proves three distinct worker processes")
        process_ids = list(prior_case["worker_process_ids"])
        execution_mode = "REVALIDATED_CACHED_ARTIFACTS_WITH_PRIOR_FRESH_PROCESS_PASS"
    return {
        "status": "PASS", "scenario": scenario, "dataset": dataset, "arm": arm,
        "interruption": interruption, "resume_in_fresh_python_process": True,
        "worker_process_ids": process_ids,
        "execution_mode": execution_mode,
        "baseline_output": str(baseline_output), "resumed_output": str(resumed_output),
        "drive_checkpoint_root": str(resumed_drive), "schedule_sha256": schedule_sha256,
        "resume_worker_invoked": True,
        "resume_worker_invoked_this_call": resumed_worker.get("status") == "SYNTHETIC_VALIDATED_NON_REPORTABLE",
        "comparison": comparison,
    }


def _checkpoint_fault_checks(
    canonical_root: Path, validation_root: Path, workflow_id: str, scenario: str,
    *, local_staging_root: str | Path = "/content/revision_r1_validation_staging",
) -> dict[str, Any]:
    from .checkpointing import CheckpointError, _validate_manifest, load_latest_valid
    from .storage import DEFAULT_STORAGE_RESERVE_BYTES, require_path_capacity

    manifests = sorted((canonical_root / "recovery").glob("generation-*.manifest.json"))
    if len(manifests) < 2:
        raise ValueError("corruption-fallback check requires two verified baseline recovery generations")
    latest = _load_json(manifests[-1], "latest recovery manifest")
    identity = latest["identity"]
    selected = manifests[-2:]
    required_bytes = sum(
        int(_load_json(path, "recovery manifest").get("payload_bytes", 0)) for path in selected
    ) + 1024 * 1024
    scratch_parent = Path(local_staging_root).expanduser() / "validation" / "fault_injection"
    require_path_capacity(
        scratch_parent, required_bytes=required_bytes,
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{workflow_id}_{scenario}_", dir=scratch_parent,
    ) as scratch_name:
        scratch = Path(scratch_name)
        for manifest_path in selected:
            manifest = _load_json(manifest_path, "recovery manifest")
            source_root = manifest_path.parent
            _validate_manifest(source_root, manifest, identity)
            shutil.copy2(manifest_path, scratch / manifest_path.name)
            source_payload = source_root / str(manifest["payload_file"])
            shutil.copy2(source_payload, scratch / source_payload.name)
        copied_manifests = sorted(scratch.glob("generation-*.manifest.json"))
        newest = _load_json(copied_manifests[-1], "fault-injection newest manifest")
        payload = scratch / newest["payload_file"]
        with payload.open("ab") as stream:
            stream.write(b"\nvalidation-only-corruption")
            stream.flush()
            os.fsync(stream.fileno())
        recovered = load_latest_valid(scratch, identity)
        older_generation = _load_json(copied_manifests[-2], "fallback manifest")["generation"]
        if recovered.manifest["generation"] != older_generation:
            raise ValueError("corruption fallback did not select the previous valid checkpoint")
        mismatch = dict(identity)
        mismatch["schedule_sha256"] = "0" * 64
        try:
            load_latest_valid(scratch, mismatch)
        except CheckpointError as exc:
            if "identity mismatch" not in str(exc):
                raise
            mismatch_rejected = True
        else:
            raise ValueError("checkpoint loader accepted a different effective schedule identity")
    return {
        "status": "PASS", "canonical_checkpoint_modified": False,
        "fault_injection_copy_removed": True, "fault_injection_copy_bytes_checked": required_bytes,
        "corrupt_generation": newest["generation"],
        "fallback_generation": recovered.manifest["generation"], "identity_mismatch_rejected": mismatch_rejected,
    }


def _duplicate_completion_check(
    *, root: Path, workflow_id: str, dataset: str, arm: str, schedule_sha256: str,
    output: Path, drive_root: Path, checkpoint_sync_policy: str = "all",
    checkpoint_write_policy: str = "all",
    runner_identity_code_sha256: str | None = None, worker_label: str | None = None,
    local_staging_root: str | Path = "/content/revision_r1_validation_staging",
    cpu_test_only: bool = False,
) -> dict[str, Any]:
    registry_runs = list((output / "registry" / "runs").glob("*/registry.json"))
    if len(registry_runs) != 1:
        raise ValueError("duplicate-result test expected exactly one completed registry identity")
    registry_path = registry_runs[0]
    before = _load_json(registry_path, "completed run registry")
    before_count = len(before.get("attempts", []))
    local = Path(local_staging_root) / "validation" / workflow_id / "gan_resume_only" / "duplicate"
    duplicate = _launch_synthetic_worker(
        root, workflow_id, dataset, arm, schedule_sha256, output, drive_root, local,
        checkpoint_sync_policy=checkpoint_sync_policy,
        checkpoint_write_policy=checkpoint_write_policy,
        runner_identity_code_sha256=runner_identity_code_sha256,
        worker_label=worker_label,
        cpu_test_only=cpu_test_only,
    )
    after = _load_json(registry_path, "completed run registry after duplicate invocation")
    after_count = len(after.get("attempts", []))
    if duplicate.get("status") != "ALREADY_VALIDATED" or before_count != after_count:
        raise ValueError("duplicate validation invocation created a second result attempt")
    return {
        "status": "PASS", "runner_status": duplicate["status"],
        "worker_pid": duplicate.get("worker_pid"),
        "attempts_before": before_count, "attempts_after": after_count,
    }


def _workflow_report_bytes(root: Path, workflow_id: str, schedule_sha256: str) -> int:
    report_root = root / "reports"
    paths = list(report_root.glob(f"resume_validation_{workflow_id}_*.json"))
    paths += list(report_root.glob(f"resume_case_{workflow_id}_gan_phase.json"))
    paths += list(report_root.glob(f"gan_resume_cleanup_{workflow_id}.json"))
    paths += list(report_root.glob(f"validation_summary_{schedule_sha256[:12]}_{workflow_id}_*.json"))
    return sum(path.stat().st_size for path in paths if path.is_file() and not path.is_symlink())


def _last_worker_result(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("worker_pid") is not None:
            return item
    return None


def _load_interruption_checkpoint(checkpoint_root: Path, identity: dict[str, Any]) -> Any:
    from .checkpointing import load_latest_valid
    return load_latest_valid(checkpoint_root, identity)


def verify_drive_checkpoint_publication(
    publication: dict[str, Any], *, manifest_path: str | Path,
    payload_path: str | Path, generation: int,
) -> dict[str, Any]:
    """Bind worker publication evidence to exact Drive read-back bytes."""
    manifest = Path(manifest_path)
    payload = Path(payload_path)
    expected = {
        "generation": int(generation),
        "manifest_sha256": sha256_file(manifest),
        "payload_sha256": sha256_file(payload),
        "payload_bytes": payload.stat().st_size,
    }
    status = publication.get("status")
    count = publication.get("published_generation_count", 1 if status == "ALREADY_PERSISTED" else None)
    if status not in {"PERSISTED_AND_READ_BACK_VERIFIED", "ALREADY_PERSISTED"}:
        raise ValueError(f"Drive checkpoint publication status is not verified: {status!r}")
    if count != 1:
        raise ValueError(f"Drive checkpoint publication did not contain exactly one generation: {count!r}")
    for key, value in expected.items():
        if publication.get(key) != value:
            raise ValueError(
                "Drive checkpoint publication evidence differs from read-back "
                f"{key}: worker={publication.get(key)!r}, Drive={value!r}"
            )
    return {"status": "PASS", "generation_count": 1, **expected}


def _ensure_local_checkpoint_owner(
    path: str | Path, *, workspace: Path, workflow_id: str,
    schedule_sha256: str, code_sha256: str, role: str,
) -> Path:
    raw_root = Path(path).expanduser().absolute()
    boundary = workspace.expanduser().absolute()
    current = raw_root
    while current != boundary and boundary in current.parents:
        if current.is_symlink():
            raise ValueError(f"symlinked local checkpoint staging is not allowed: {current}")
        current = current.parent
    if boundary.is_symlink():
        raise ValueError(f"symlinked GAN validation workspace is not allowed: {boundary}")
    root = raw_root.resolve()
    if not root.is_relative_to(boundary.resolve()) or root == boundary.resolve():
        raise ValueError(f"checkpoint staging is outside the owned local workspace: {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "checkpoint_owner.json"
    expected = {
        "validation_build_id": VALIDATION_BUILD_ID, "workflow_id": workflow_id,
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha256,
        "role": role, "artifact_type": "SYNTHETIC_GAN_RESUME_VALIDATION_ONLY",
    }
    if marker.is_file():
        if _load_json(marker, "local checkpoint owner") != expected:
            raise ValueError(f"local checkpoint staging identity differs; refusing reuse: {root}")
    elif any(root.iterdir()):
        raise ValueError(f"unowned files exist in local checkpoint staging; refusing reuse: {root}")
    else:
        write_immutable_json(marker, expected)
    return root


def _replace_local_checkpoint_with_empty_verified_staging(
    path: str | Path, *, workspace: Path, workflow_id: str,
    schedule_sha256: str, code_sha256: str, role: str,
    drive_checkpoint_root: Path, expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Reset only validation-owned local state after its Drive recovery copy verifies."""
    from .checkpointing import load_latest_valid

    root = _ensure_local_checkpoint_owner(
        path, workspace=workspace, workflow_id=workflow_id,
        schedule_sha256=schedule_sha256, code_sha256=code_sha256, role=role,
    )
    recovery = root / "recovery"
    existing = list(recovery.glob("generation-*.manifest.json")) if recovery.is_dir() else []
    if not existing:
        return {"status": "ALREADY_EMPTY", "path": str(root), "bytes_removed": 0}
    identity = _drive_checkpoint_identity(drive_checkpoint_root, schedule_sha256)
    if identity != expected_identity:
        raise ValueError("Drive checkpoint identity changed; refusing to clear local resume state")
    loaded = load_latest_valid(drive_checkpoint_root / "recovery", identity)
    if loaded.state.identity != expected_identity:
        raise ValueError("Drive recovery payload identity differs; refusing to clear local staging")
    for manifest_path in existing:
        manifest = _load_json(manifest_path, "local resumed checkpoint manifest")
        if manifest.get("identity") != expected_identity:
            raise ValueError("local checkpoint identity differs; refusing to clear local staging")
    size = inventory_validation_files(root)["visible_bytes"]
    shutil.rmtree(root)
    root.mkdir(parents=True)
    write_immutable_json(root / "checkpoint_owner.json", {
        "validation_build_id": VALIDATION_BUILD_ID, "workflow_id": workflow_id,
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha256,
        "role": role, "artifact_type": "SYNTHETIC_GAN_RESUME_VALIDATION_ONLY",
    })
    return {"status": "PASS_RESET_OWNED_LOCAL_STATE_FROM_VERIFIED_DRIVE_COPY", "path": str(root), "bytes_removed": size}


def _save_gan_resume_report_and_summary(
    *, root: Path, workflow_id: str, schedule_sha256: str,
    report: dict[str, Any], remaining_report_budget_bytes: int,
) -> dict[str, Any]:
    serialized_size = len(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"))
    summary_slack = 128 * 1024
    free, storage_kind = available_bytes(root)
    required_report_budget = min(int(remaining_report_budget_bytes), serialized_size + summary_slack)
    writer_space_bound = (
        2 * int(remaining_report_budget_bytes) + GAN_REPORT_ATOMIC_OVERHEAD_BYTES
        + GAN_CLEANUP_RECEIPT_WRITE_SPACE_BOUND_BYTES
    )
    if (
        required_report_budget < serialized_size + summary_slack
        or free < writer_space_bound + DEFAULT_STORAGE_RESERVE_BYTES
    ):
        return {
            "report_path": None, "summary": None,
            "report_export_status": "NOT_WRITTEN_STORAGE_RESERVE_PRESERVED",
            "report_bytes_required": serialized_size + summary_slack,
            "report_atomic_write_space_bound_bytes": writer_space_bound,
            "available_bytes": free, "storage_kind": storage_kind,
            "safety_reserve_bytes": DEFAULT_STORAGE_RESERVE_BYTES,
        }
    require_available_bytes(
        root, required_bytes=writer_space_bound,
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=free, storage_kind=storage_kind,
    )
    path = _save_phase_report(root, "resume_validation", workflow_id, report)
    # Summaries are immutable snapshots, not caches. A successful retry must
    # export a fresh snapshot that points at this attempt's new phase report.
    summary = export_validation_summary(validation_root=root, schedule_sha256=schedule_sha256)
    summary["reused_existing_summary"] = False
    return {
        "report_path": str(path), "summary": summary,
        "report_export_status": "PASS", "report_bytes": path.stat().st_size,
        "report_atomic_write_space_bound_bytes": writer_space_bound,
        "storage_kind": storage_kind,
    }


def _find_current_gan_pass(
    root: Path, workflow_id: str, schedule_sha256: str, code_sha256: str,
    *, cpu_test_only: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        (root / "reports").glob(f"resume_validation_{workflow_id}_*.json"),
        key=lambda item: item.stat().st_mtime_ns, reverse=True,
    )
    for path in candidates:
        report = _load_json(path, "current GAN-only resume report")
        gan = report.get("gan_phase", {})
        classifier = report.get("classifier_phase", {})
        expected_status = "CPU_TEST_PASS" if cpu_test_only else "PASS"
        if (
            report.get("status") == expected_status
            and (
                not cpu_test_only
                or report.get("execution_scope") == "LOCAL_CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE"
            )
            and report.get("validation_build_id") == VALIDATION_BUILD_ID
            and report.get("schedule_sha256") == schedule_sha256
            and report.get("code_sha256") == code_sha256
            and classifier.get("status") == "PASS"
            and classifier.get("rerun") is False
            and gan.get("status") == expected_status
        ):
            return path, report
    return None


def run_gan_resume_validation_only(
    *, validation_root: str | Path, schedule_path: str | Path,
    schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
    local_staging_root: str | Path = "/content/revision_r1_validation_staging",
    cpu_test_only: bool = False,
) -> dict[str, Any]:
    """Finish only GAN recovery checks, reusing the exact v5 classifier PASS and checkpoint."""
    root = _validation_root(validation_root)
    if cpu_test_only:
        if schedule_sha256 != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
            raise ValueError("CPU fixture must remain bound to the approved effective schedule identity")
    else:
        _assert_schedule_identity(schedule_path, schedule_sha256)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    package = Path(package_root).resolve() if package_root else Path(__file__).resolve().parents[1]
    inventory = query_drive_validation_inventory(root)
    prior_classifier = validate_prior_classifier_resume_evidence(root)
    compatibility = verify_legacy_runner_source_compatibility(package)
    if compatibility.get("status") != "PASS" or compatibility.get("baseline_code_sha256") != PRIOR_V5_CODE_SHA256:
        raise ValueError(f"v5 checkpoint source compatibility failed; no worker launched: {compatibility}")
    if prior_classifier.get("status") != "PASS" or prior_classifier.get("rerun") is not False:
        raise ValueError("the identity-matched v5 classifier PASS is not reusable; no classifier/GAN worker launched")

    current_pass = _find_current_gan_pass(
        root, workflow_id, schedule_sha256, code_sha, cpu_test_only=cpu_test_only,
    )
    if current_pass is not None:
        report_path, prior_report = current_pass
        # Refresh the immutable snapshot on safe reruns; old summaries remain
        # untouched and cannot mask the latest phase status.
        summary = export_validation_summary(validation_root=root, schedule_sha256=schedule_sha256)
        summary["reused_existing_summary"] = False
        return {
            **prior_report, "report_path": str(report_path), "summary": summary,
            "reused_completed_gan_validation": True,
            "drive_inventory_read_only_refresh": inventory,
        }

    try:
        reference = _inspect_legacy_gan_reference(
            root, schedule_sha256=schedule_sha256, cpu_test_only=cpu_test_only,
        )
    except Exception as exc:
        reference = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}

    staging_root = Path(local_staging_root).expanduser().resolve()
    workspace = staging_root / "validation" / workflow_id / "gan_resume_only"
    workspace_bytes = inventory_validation_files(workspace)["visible_bytes"]
    local_validation_inventory = inventory_validation_files(staging_root / "validation")
    report_root = root / "reports"
    log_root = root / "logs"
    worker_labels = (
        "gan_v6_2_uninterrupted", "gan_v6_2_interrupted", "gan_v6_2_resumed", "gan_v6_2_duplicate",
    )
    current_log_bytes = [
        (log_root / f"{workflow_id}_{label}.log").stat().st_size
        if (log_root / f"{workflow_id}_{label}.log").is_file() else 0
        for label in worker_labels
    ]
    prior_report_bytes = _workflow_report_bytes(root, workflow_id, schedule_sha256)
    remaining_log_bytes = sum(
        max(0, VALIDATION_WORKER_LOG_LIMIT_BYTES - value) for value in current_log_bytes
    )
    each_worker_log_has_capacity = all(value < VALIDATION_WORKER_LOG_LIMIT_BYTES for value in current_log_bytes)
    drive_free = inventory.get("drive", {}).get("available_account_bytes")
    max_payload = int(reference.get("max_checkpoint_payload_bytes", reference.get("checkpoint_payload_bytes", 0)))
    local_free = _available_local_validation_bytes(staging_root)
    if drive_free is None or reference.get("status") != "PASS" or max_payload <= 0:
        storage_plan = {
            "status": "UNVERIFIABLE", "available_drive_bytes": drive_free,
            "max_checkpoint_payload_bound_bytes": max_payload,
            "reason": "Drive quota or verified reference checkpoint size is unavailable",
        }
    else:
        drive_inventory = inventory.get("drive", {})
        storage_plan = plan_gan_resume_storage(
            max_checkpoint_payload_bytes=max_payload,
            available_local_bytes=local_free,
            available_drive_bytes=int(drive_free),
            existing_workspace_bytes=workspace_bytes,
            existing_report_bytes=prior_report_bytes,
            existing_visible_validation_bytes=int(
                inventory.get("visible_validation_files", {}).get("visible_bytes", 0)
            ),
            existing_local_validation_bytes=int(local_validation_inventory.get("visible_bytes", 0)),
            existing_drive_account_usage_bytes=drive_inventory.get("account_usage_bytes"),
            existing_drive_trash_usage_bytes=drive_inventory.get("usage_in_drive_trash_bytes"),
            remaining_worker_log_bytes=remaining_log_bytes,
            worker_log_capacity_passed=each_worker_log_has_capacity,
        )
    storage_plan["existing_workflow_report_bytes"] = prior_report_bytes
    storage_plan["existing_local_validation_file_count"] = local_validation_inventory.get("file_count", 0)
    storage_plan["existing_local_validation_by_top_level"] = local_validation_inventory.get("by_top_level", {})
    storage_plan["existing_validation_drive_file_count"] = inventory.get("visible_validation_files", {}).get("file_count")
    storage_plan["existing_worker_log_bytes"] = dict(zip(worker_labels, current_log_bytes))
    storage_plan["remaining_worker_log_write_budget_bytes"] = remaining_log_bytes
    storage_plan["cumulative_worker_log_write_cap_bytes"] = len(worker_labels) * VALIDATION_WORKER_LOG_LIMIT_BYTES
    print("[GAN-only read-only Drive inventory]", json.dumps(inventory, sort_keys=True), flush=True)
    print("[GAN-only pre-write storage plan]", json.dumps(storage_plan, sort_keys=True), flush=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "artifact_type": "SYNTHETIC_FRESH_PROCESS_GAN_RESUME_NOT_PRODUCTION_EVIDENCE",
        "execution_scope": (
            "LOCAL_CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE"
            if cpu_test_only else "COLAB_CUDA_SYNTHETIC_VALIDATION"
        ),
        "validation_build_id": VALIDATION_BUILD_ID,
        "schedule_sha256": schedule_sha256,
        "code_sha256": code_sha,
        "validation_workflow_id": workflow_id,
        "drive_inventory": inventory,
        "storage_plan": storage_plan,
        "legacy_v5_training_core_compatibility": compatibility,
        "classifier_phase": {
            "status": "PASS", "rerun": False,
            "provenance": prior_classifier.get(
                "evidence_provenance", "REUSED_VERIFIED_V5_CELL10_EVIDENCE",
            ),
            "evidence": prior_classifier,
        },
        "workers_launched": [],
        "prior_gan_failure": {
            "classification": "STORAGE_CAPACITY_FAILURE_BEFORE_NUMERICAL_COMPARISON",
            "numerical_recovery_divergence_established": False,
            "source_report": prior_classifier.get("source_report"),
        },
        "reference_checkpoint": reference,
        "production_matrix_launched": False,
        "smoke_suite_rerun": False,
        "manual_runtime_reset_rerun": False,
        "trash_modified_or_permanently_deleted": False,
    }

    if reference.get("status") != "PASS":
        report["reason"] = "No verified compatible unfinished GAN reference checkpoint was found; no worker launched."
    elif storage_plan.get("status") != "PASS":
        report["reason"] = "Storage preflight is insufficient/unverifiable; GAN recovery comparison remains NOT_RUN and no worker launched."
    elif prior_report_bytes >= GAN_RESUME_REPORT_BUDGET_BYTES:
        report["reason"] = "The cumulative v6.1 resume report/summary write budget is exhausted; no worker launched."
        report["status"] = "NOT_RUN"
    elif not cpu_test_only and _cuda_device_name() is None:
        report["reason"] = "CUDA/PyTorch GPU is unavailable; GAN GPU recovery remains NOT_RUN."
    else:
        used_bytes = prior_report_bytes
        remaining_report_budget = GAN_RESUME_REPORT_BUDGET_BYTES - used_bytes
        _ensure_local_checkpoint_owner(
            workspace / "checkpoints" / "uninterrupted", workspace=workspace,
            workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            code_sha256=code_sha, role="uninterrupted-control",
        )
        _ensure_local_checkpoint_owner(
            workspace / "checkpoints" / "interrupted", workspace=workspace,
            workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            code_sha256=code_sha, role="interrupted-worker",
        )
        _ensure_local_checkpoint_owner(
            workspace / "checkpoints" / "resumed", workspace=workspace,
            workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            code_sha256=code_sha, role="fresh-process-resumed-worker",
        )
        owner = {
            "validation_build_id": VALIDATION_BUILD_ID,
            "workflow_id": workflow_id, "schedule_sha256": schedule_sha256,
            "code_sha256": code_sha, "artifact_type": "SYNTHETIC_GAN_RESUME_VALIDATION_ONLY",
        }
        workspace.mkdir(parents=True, exist_ok=True)
        owner_path = workspace / "workspace_owner.json"
        if owner_path.is_file():
            if _load_json(owner_path, "GAN validation workspace owner") != owner:
                raise ValueError("local GAN validation workspace belongs to another build; no worker launched")
        else:
            write_immutable_json(owner_path, owner)

        baseline_output = workspace / "outputs" / "uninterrupted"
        interrupted_output = workspace / "outputs" / "interrupted_then_resumed"
        baseline_local = workspace / "checkpoints" / "uninterrupted"
        interrupted_local = workspace / "checkpoints" / "interrupted"
        resumed_local = workspace / "checkpoints" / "resumed"
        baseline_drive = root / "checkpoints" / workflow_id / "gan_resume_only" / "no_sync_baseline"
        interrupted_drive = root / "checkpoints" / workflow_id / "gan_resume_only" / "interrupted"
        labels = {
            "baseline": "gan_v6_2_uninterrupted", "interrupted": "gan_v6_2_interrupted",
            "resumed": "gan_v6_2_resumed", "duplicate": "gan_v6_2_duplicate",
        }
        worker_rows: dict[str, Any] = {}
        stage = "baseline_control"
        comparison_performed = False
        try:
            _seed_reference_checkpoint(reference, baseline_local)
            if _output_complete(baseline_output):
                baseline_worker = _last_worker_result(log_root / f"{workflow_id}_{labels['baseline']}.log")
                if baseline_worker is None:
                    raise RuntimeError("completed local GAN baseline has no worker log proving its fresh process")
                baseline_worker = {**baseline_worker, "status": "SYNTHETIC_VALIDATED_NON_REPORTABLE", "returncode": 0, "reused_completed_output": True}
            else:
                baseline_worker = _launch_synthetic_worker(
                    root, workflow_id, "ember", "generated_only", schedule_sha256,
                    baseline_output, baseline_drive, baseline_local,
                    checkpoint_sync_policy="local_only",
                    checkpoint_write_policy="recovery_and_terminal_only",
                    runner_identity_code_sha256=PRIOR_V5_CODE_SHA256,
                    worker_label=labels["baseline"],
                    cpu_test_only=cpu_test_only,
                )
            if baseline_worker.get("returncode") not in {None, 0} or baseline_worker.get("status") not in {"SYNTHETIC_VALIDATED_NON_REPORTABLE", "ALREADY_VALIDATED"}:
                raise RuntimeError(f"uninterrupted local-only GAN control worker failed: {baseline_worker}")
            if baseline_worker.get("worker_pid") is None:
                raise RuntimeError("uninterrupted GAN control worker process identity is unavailable")
            worker_rows["uninterrupted_control"] = baseline_worker

            stage = "checkpoint_fault_checks"
            fault_checks = _checkpoint_fault_checks(
                baseline_local, root, workflow_id, "gan_phase", local_staging_root=staging_root,
            )
            if fault_checks.get("status") != "PASS":
                raise RuntimeError(f"GAN checkpoint corruption/identity fault checks failed: {fault_checks}")

            stage = "interruption_checkpoint"
            interrupted_drive_recovery = interrupted_drive / "recovery"
            existing_drive_manifests = sorted(interrupted_drive_recovery.glob("generation-*.manifest.json"))
            if len(existing_drive_manifests) > 1:
                raise RuntimeError("target GAN Drive branch exceeded its one-generation write bound; refusing additional writes")
            if existing_drive_manifests:
                interrupt_identity = _drive_checkpoint_identity(interrupted_drive, schedule_sha256)
                if interrupt_identity != reference["identity"]:
                    raise ValueError("existing interrupted GAN Drive checkpoint identity differs from the verified v5 branch")
                interrupt_loaded = _load_interruption_checkpoint(interrupted_drive_recovery, interrupt_identity)
                if int(interrupt_loaded.state.resources.get("generator_updates", 0)) <= int(reference["generator_updates_before_interruption"]):
                    raise ValueError("existing Drive interruption checkpoint did not advance a GAN generator update")
                interrupted_worker = _last_worker_result(log_root / f"{workflow_id}_{labels['interrupted']}.log")
                if interrupted_worker is None:
                    raise RuntimeError("existing Drive interruption checkpoint has no live-worker log evidence")
                interrupted_worker = {**interrupted_worker, "returncode": 0, "reused_persisted_interruption": True}
            else:
                _seed_reference_checkpoint(reference, interrupted_local)
                interrupted_worker = _launch_synthetic_worker(
                    root, workflow_id, "ember", "generated_only", schedule_sha256,
                    interrupted_output, interrupted_drive, interrupted_local,
                    interruption="gan_after_next_generator",
                    checkpoint_sync_policy="publish_interruption_once",
                    checkpoint_write_policy="recovery_and_terminal_only",
                    runner_identity_code_sha256=PRIOR_V5_CODE_SHA256,
                    worker_label=labels["interrupted"],
                    cpu_test_only=cpu_test_only,
                )
            if interrupted_worker.get("returncode") not in {None, 0} or interrupted_worker.get("status") != "INTERRUPTED_RESUMABLE":
                raise RuntimeError(f"GAN interrupted worker did not stop resumably: {interrupted_worker}")
            if interrupted_worker.get("worker_pid") is None:
                raise RuntimeError("GAN interruption worker process identity is unavailable")
            if not _output_complete(baseline_output):
                raise RuntimeError("uninterrupted GAN baseline output is incomplete")
            if not existing_drive_manifests:
                publication = interrupted_worker.get("validation_checkpoint_storage", {}).get("interruption_checkpoint_publication", {})
                if publication.get("status") not in {"PERSISTED_AND_READ_BACK_VERIFIED", "ALREADY_PERSISTED"}:
                    raise RuntimeError(f"interrupted GAN checkpoint was not verified on Drive: {publication}")
                if publication.get("published_generation_count", 1) != 1:
                    raise RuntimeError("interrupted GAN worker published more than one Drive checkpoint generation")
            drive_identity = _drive_checkpoint_identity(interrupted_drive, schedule_sha256)
            if drive_identity != reference["identity"]:
                raise ValueError("interrupted Drive checkpoint is not bound to the v5 reference identity")
            interrupted_checkpoint = _load_interruption_checkpoint(interrupted_drive_recovery, drive_identity)
            if int(interrupted_checkpoint.state.resources.get("generator_updates", 0)) <= int(reference["generator_updates_before_interruption"]):
                raise ValueError("Drive-persisted interruption checkpoint did not include a new generator update")
            if len(list(interrupted_drive_recovery.glob("generation-*.manifest.json"))) != 1:
                raise RuntimeError("GAN interruption Drive branch must retain exactly one verified generation")
            interrupted_manifest_path = interrupted_drive_recovery / f"generation-{int(interrupted_checkpoint.manifest['generation']):06d}.manifest.json"
            interrupted_payload_path = interrupted_drive_recovery / str(interrupted_checkpoint.manifest["payload_file"])
            publication = interrupted_worker.get("validation_checkpoint_storage", {}).get("interruption_checkpoint_publication", {})
            publication_match = verify_drive_checkpoint_publication(
                publication,
                manifest_path=interrupted_manifest_path,
                payload_path=interrupted_payload_path,
                generation=int(interrupted_checkpoint.manifest["generation"]),
            )
            worker_rows["interrupted"] = interrupted_worker
            interrupted_local_cleanup = _cleanup_local_checkpoint_staging(
                interrupted_local, workspace=workspace, workflow_id=workflow_id,
                schedule_sha256=schedule_sha256, code_sha256=code_sha,
                role="interrupted-worker",
                allowed_staging_root=staging_root,
            )

            stage = "fresh_process_resume"
            local_reset = _replace_local_checkpoint_with_empty_verified_staging(
                resumed_local, workspace=workspace, workflow_id=workflow_id,
                schedule_sha256=schedule_sha256, code_sha256=code_sha,
                role="fresh-process-resumed-worker", drive_checkpoint_root=interrupted_drive,
                expected_identity=reference["identity"],
            )
            if _output_complete(interrupted_output):
                resumed_worker = _last_worker_result(log_root / f"{workflow_id}_{labels['resumed']}.log")
                if resumed_worker is None:
                    raise RuntimeError("completed resumed GAN output has no worker log proving its fresh-process Drive restore")
                resumed_worker = {**resumed_worker, "status": "SYNTHETIC_VALIDATED_NON_REPORTABLE", "returncode": 0, "reused_completed_output": True}
            else:
                resumed_worker = _launch_synthetic_worker(
                    root, workflow_id, "ember", "generated_only", schedule_sha256,
                    interrupted_output, interrupted_drive, resumed_local,
                    checkpoint_sync_policy="restore_only",
                    checkpoint_write_policy="recovery_and_terminal_only",
                    runner_identity_code_sha256=PRIOR_V5_CODE_SHA256,
                    worker_label=labels["resumed"],
                    cpu_test_only=cpu_test_only,
                )
            if resumed_worker.get("returncode") not in {None, 0} or resumed_worker.get("status") not in {"SYNTHETIC_VALIDATED_NON_REPORTABLE", "ALREADY_VALIDATED"}:
                raise RuntimeError(f"fresh-process GAN resume worker failed: {resumed_worker}")
            if resumed_worker.get("worker_pid") is None:
                raise RuntimeError("fresh-process GAN resume worker process identity is unavailable")
            restore_events = resumed_worker.get("validation_checkpoint_storage", {}).get("drive_restore_events", [])
            restore = next((item for item in restore_events if item.get("local_staging_empty_before_restore")), None)
            if restore is None:
                raise RuntimeError("fresh-process worker did not restore its checkpoint from Drive into empty local staging")
            if not _output_complete(interrupted_output):
                raise RuntimeError("fresh-process GAN resume did not produce a completed output")
            worker_rows["resumed"] = resumed_worker

            stage = "resume_comparison"
            comparison_performed = True
            comparison = compare_resumed_outputs(
                baseline_output, interrupted_output,
                baseline_checkpoint_root=baseline_local,
                resumed_checkpoint_root=resumed_local,
                cpu_test_only=cpu_test_only,
            )
            if comparison.get("status") != "PASS":
                raise ValueError(f"GAN resumed output comparison failed: {comparison}")

            stage = "duplicate_result_check"
            duplicate_check = _duplicate_completion_check(
                root=root, workflow_id=workflow_id, dataset="ember", arm="generated_only",
                schedule_sha256=schedule_sha256, output=interrupted_output,
                drive_root=interrupted_drive, checkpoint_sync_policy="local_only",
                checkpoint_write_policy="recovery_and_terminal_only",
                runner_identity_code_sha256=PRIOR_V5_CODE_SHA256,
                worker_label=labels["duplicate"], local_staging_root=staging_root,
                cpu_test_only=cpu_test_only,
            )
            if duplicate_check.get("status") != "PASS" or duplicate_check.get("attempts_before") != duplicate_check.get("attempts_after"):
                raise RuntimeError(f"GAN duplicate-result prevention failed: {duplicate_check}")
            if duplicate_check.get("worker_pid") is None:
                duplicate_record = _last_worker_result(log_root / f"{workflow_id}_{labels['duplicate']}.log")
                if duplicate_record:
                    duplicate_check["worker_pid"] = duplicate_record.get("worker_pid")
            worker_rows["duplicate_check"] = duplicate_check
            process_ids = [worker_rows[name].get("worker_pid") for name in ("uninterrupted_control", "interrupted", "resumed")]
            if duplicate_check.get("worker_pid") is not None:
                process_ids.append(duplicate_check["worker_pid"])
            if len(process_ids) < 3 or len(set(process_ids)) != len(process_ids):
                raise ValueError(f"GAN validation worker processes were not distinct: {process_ids}")

            checkpoint_write_accounting = {}
            saved_checkpoint_payload_bytes = 0
            saved_checkpoint_manifest_bytes = 0
            suppressed_checkpoint_count = 0
            for worker_key, expected_kind in (
                ("uninterrupted_control", "complete"),
                ("interrupted", "interrupted-resumable"),
                ("resumed", "complete"),
            ):
                checkpoint_storage = worker_rows[worker_key].get("validation_checkpoint_storage", {})
                events = checkpoint_storage.get("checkpoint_write_events", [])
                if checkpoint_storage.get("checkpoint_write_policy") != "recovery_and_terminal_only":
                    raise ValueError(f"{worker_key} did not report the bounded validation checkpoint-write policy")
                if len(events) != 1 or events[0].get("kind") != expected_kind:
                    raise ValueError(
                        f"{worker_key} checkpoint writes were not limited to its required {expected_kind!r} generation: {events}"
                    )
                event_root = Path(events[0].get("root", "")).expanduser().resolve()
                if not event_root.is_relative_to(workspace.resolve()):
                    raise ValueError(f"{worker_key} checkpoint was not written to owned local staging: {event_root}")
                saved_checkpoint_payload_bytes += int(events[0].get("payload_bytes", 0))
                saved_checkpoint_manifest_bytes += int(events[0].get("manifest_bytes", 0))
                suppressed_checkpoint_count += sum(
                    int(value) for value in checkpoint_storage.get("suppressed_checkpoint_kinds", {}).values()
                )
                checkpoint_write_accounting[worker_key] = {
                    "checkpoint_write_policy": checkpoint_storage["checkpoint_write_policy"],
                    "saved_generation_kind": events[0]["kind"],
                    "saved_payload_bytes": int(events[0].get("payload_bytes", 0)),
                    "saved_manifest_bytes": int(events[0].get("manifest_bytes", 0)),
                    "intermediate_saves_suppressed": sum(
                        int(value) for value in checkpoint_storage.get("suppressed_checkpoint_kinds", {}).values()
                    ),
                    "checkpoint_root_is_local_owned_workspace": True,
                }

            final_status = "CPU_TEST_PASS" if cpu_test_only else "PASS"
            report.update({
                "status": final_status,
                "gpu": "CPU_TEST_ONLY" if cpu_test_only else _cuda_device_name(),
                "workers_launched": worker_rows,
                "worker_process_ids": process_ids,
                "fresh_processes_distinct": True,
                "gan_phase": {
                    "status": final_status, "arm": "generated_only", "dataset": "ember",
                    "feature_dimension": 2381,
                    "reference_checkpoint_generation": reference["generation"],
                    "reference_cursor": reference["cursor"],
                    "branching_from_same_reference_state": True,
                    "routine_worker_checkpoints_drive_synced": False,
                    "interruption_drive_checkpoint": {
                        "root": str(interrupted_drive),
                        "generation": int(interrupted_checkpoint.manifest["generation"]),
                        "manifest_sha256": sha256_file(interrupted_manifest_path),
                        "payload_sha256": sha256_file(interrupted_payload_path),
                        "payload_bytes": interrupted_payload_path.stat().st_size,
                        "worker_publication_matches_drive_readback": publication_match,
                        "fresh_process_restored_from_drive": True,
                        "local_staging_empty_before_restore": True,
                        "restore_event": restore,
                    },
                    "fresh_process_restore": {**restore, "status": "PASS", "local_reset": local_reset},
                    "interrupted_local_staging_released_after_drive_verification": interrupted_local_cleanup,
                    "comparison": comparison,
                    "fault_checks": fault_checks,
                    "duplicate_result_check": duplicate_check,
                    "checkpoint_write_accounting": checkpoint_write_accounting,
                    "worker_process_ids": process_ids,
                },
                "validation_write_accounting": {
                    "drive_checkpoint_payload_writes": 1 if not existing_drive_manifests else 0,
                    "drive_checkpoint_payload_bytes_written_upper_bound": reference["max_checkpoint_payload_bytes"] if not existing_drive_manifests else 0,
                    "drive_checkpoint_generations_retained_for_this_case": 1,
                    "checkpoint_write_scope": "only the interruption generation is persisted to Drive; each local worker saves only its required terminal/interruption generation and suppresses intermediate phase snapshots",
                    "bounded_local_worker_checkpoint_payload_bytes_written": saved_checkpoint_payload_bytes,
                    "bounded_local_worker_checkpoint_manifest_bytes_written": saved_checkpoint_manifest_bytes,
                    "intermediate_local_checkpoint_saves_suppressed": suppressed_checkpoint_count,
                    "total_drive_write_volume_bound_bytes": storage_plan.get("total_drive_write_volume_bound_bytes"),
                    "total_local_write_volume_bound_bytes": storage_plan.get("total_local_write_volume_bound_bytes"),
                    "total_validation_write_volume_bound_bytes": storage_plan.get("total_validation_write_volume_bound_bytes"),
                    "drive_worker_log_bytes_after": {
                        label: (log_root / f"{workflow_id}_{label}.log").stat().st_size
                        if (log_root / f"{workflow_id}_{label}.log").is_file() else 0
                        for label in worker_labels
                    },
                    "trash_modified_or_deleted": False,
                    "v5_reference_checkpoint_tree_modified": False,
                },
            })
        except Exception as exc:
            is_storage = (
                type(exc).__name__ == "StorageCapacityError"
                or "StorageCapacityError" in str(exc)
                or "storage" in str(exc).lower()
                or "log reached its" in str(exc).lower()
            )
            terminal_status = _gan_resume_failure_status(
                comparison_performed=comparison_performed, is_storage=is_storage,
            )
            report.update({
                "status": terminal_status,
                "gpu": _cuda_device_name(),
                "failure_stage": stage,
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "failure_classification": "STORAGE_CAPACITY_NOT_NUMERICAL_DIVERGENCE" if is_storage else "VALIDATION_EXECUTION_OR_CHECK_FAILURE",
                "comparison_performed": comparison_performed,
                "workers_launched": worker_rows,
                "gan_phase": {"status": "FAIL" if terminal_status == "FAIL" else "NOT_RUN", "comparison_performed": comparison_performed},
            })

        report["local_workspace_cleanup"] = (
            {"status": "PENDING_UNTIL_COMPACT_EVIDENCE_IS_DURABLE", "path": str(workspace),
             "bytes_expected_to_release": inventory_validation_files(workspace)["visible_bytes"]}
            if report.get("status") == "PASS"
            else {"status": "PRESERVED_FOR_VALIDATION_RESUME_OR_DEBUG", "path": str(workspace)}
        )
        report["classifier_phase"]["rerun"] = False
        report["storage_plan"]["trash_cleanup_performed"] = False
        try:
            drive_free_after = available_bytes(root)[0]
            report["storage_plan"]["available_drive_bytes_after_workers"] = drive_free_after
            post_worker_report_space = int(
                report["storage_plan"].get("report_atomic_write_space_bound_bytes", 0)
            )
            if drive_free_after < post_worker_report_space + DEFAULT_STORAGE_RESERVE_BYTES:
                report["report_export_gate"] = "INSUFFICIENT_SPACE_TO_EXPORT_WITHOUT_BREACHING_RESERVE"
        except Exception as exc:
            report["storage_plan"]["post_worker_quota_recheck_error"] = f"{type(exc).__name__}: {exc}"
    write_budget_remaining = max(0, GAN_RESUME_REPORT_BUDGET_BYTES - prior_report_bytes)
    stored = _save_gan_resume_report_and_summary(
        root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
        report=report, remaining_report_budget_bytes=write_budget_remaining,
    )
    cleanup = report.get("local_workspace_cleanup", {})
    receipt_path = None
    cleanup_error = None
    if report.get("status") == "PASS" and stored.get("report_path"):
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "validation_build_id": VALIDATION_BUILD_ID,
            "validation_workflow_id": workflow_id,
            "schedule_sha256": schedule_sha256,
            "code_sha256": code_sha,
            "resume_report_path": stored["report_path"],
            "resume_report_sha256": sha256_file(stored["report_path"]),
        }
        try:
            cleanup = _cleanup_local_gan_workspace(
                workspace, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
                code_sha256=code_sha, local_staging_root=staging_root,
            )
        except Exception as exc:
            cleanup = {
                "status": "LOCAL_WORKSPACE_CLEANUP_FAILED",
                "path": str(workspace), "error_type": type(exc).__name__, "error": str(exc),
            }
        else:
            receipt["local_workspace_cleanup"] = cleanup
            receipt_path = root / "reports" / f"gan_resume_cleanup_{workflow_id}.json"
            try:
                write_immutable_json(receipt_path, receipt)
            except Exception as exc:
                cleanup_error = {"type": type(exc).__name__, "message": str(exc)}
                receipt_path = None
    return {**report, **stored, "local_workspace_cleanup": cleanup,
            "cleanup_receipt_path": str(receipt_path) if receipt_path else None,
            "cleanup_receipt_export_error": cleanup_error}


def run_resume_validation(
    *, validation_root: str | Path, schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate classifier/GAN recovery in fresh Python workers with Drive checkpoints."""
    root = _validation_root(validation_root)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available, gpu_name = False, None
    if not cuda_available:
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "artifact_type": "SYNTHETIC_FRESH_PROCESS_RESUME_NOT_PRODUCTION_EVIDENCE",
            "reason": "CUDA/PyTorch GPU runtime is unavailable; fresh-process GPU recovery has not been tested.",
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha, "cases": [],
        }
        path = _save_phase_report(root, "resume_validation", workflow_id, report)
        return {**report, "report_path": str(path)}

    check_validation_storage_capacity(
        local_path="/content/revision_r1_validation_staging",
        drive_path=root,
    )

    cases: list[dict[str, Any]] = []
    try:
        classifier_case = _execute_resume_case(
            root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            scenario="classifier_phase", arm="current_only", interruption="classifier_after_update", resume_now=True,
        )
        if "checkpoint_cleanup" not in classifier_case:
            classifier_case["checkpoint_cleanup"] = _persist_resume_case_evidence_and_prune(
                root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
                case=classifier_case,
            )
        cases.append(classifier_case)
        gan_case = _execute_resume_case(
            root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            scenario="gan_phase", arm="generated_only", interruption="gan_after_generator", resume_now=True,
        )
        if "checkpoint_fault_checks" not in gan_case:
            gan_drive = root / "checkpoints" / workflow_id / "resume" / "gan_phase" / "resumed"
            gan_output = root / "resume" / workflow_id / "gan_phase" / "interrupted_then_resumed"
            gan_case["checkpoint_fault_checks"] = _checkpoint_fault_checks(gan_drive, root, workflow_id, "gan_phase")
            gan_case["duplicate_result_check"] = _duplicate_completion_check(
                root=root, workflow_id=workflow_id, dataset="ember", arm="generated_only",
                schedule_sha256=schedule_sha256, output=gan_output, drive_root=gan_drive,
            )
        if "checkpoint_cleanup" not in gan_case:
            gan_case["checkpoint_cleanup"] = _persist_resume_case_evidence_and_prune(
                root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
                case=gan_case,
            )
        cases.append(gan_case)
        statuses = [case["status"] for case in cases]
        status = "PASS" if all(value == "PASS" for value in statuses) else "FAIL"
        report = {
            "schema_version": 1, "status": status,
            "artifact_type": "SYNTHETIC_FRESH_PROCESS_RESUME_NOT_PRODUCTION_EVIDENCE",
            "gpu": gpu_name, "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
            "cases": cases,
            "resume_agreement_tolerances": {
                "classifier_loss": {"atol": 5e-5, "rtol": 1e-4},
                "task_matrix": {
                    "atol": 1e-5, "rtol": 1e-4,
                    "evaluated_cells": "finite lower triangle including diagonal; future upper-triangle cells must be null",
                    "missing_value_masks_must_match": True,
                    "maximum_finite_absolute_error_reported": True,
                },
                "final_model_and_all_optimizer_states_from_verified_drive_checkpoints": {"atol": 1e-5, "rtol": 1e-4},
                "checkpoint_rng_states": "exact",
                "sampling_counts_and_final_prediction_arrays": "exact",
            },
            "includes": ["Drive-persisted classifier phase checkpoint", "Drive-persisted GAN phase checkpoint", "fresh Python processes", "corruption fallback on copied checkpoint only", "schedule identity mismatch rejection", "duplicate completion prevention"],
        }
    except Exception as exc:
        report = {
            "schema_version": 1, "status": "FAIL",
            "artifact_type": "SYNTHETIC_FRESH_PROCESS_RESUME_NOT_PRODUCTION_EVIDENCE",
            "gpu": gpu_name, "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
            "cases": cases, "error_type": type(exc).__name__, "error": str(exc),
        }
    path = _save_phase_report(root, "resume_validation", workflow_id, report)
    return {**report, "report_path": str(path)}


def run_engineering_pilot(
    *, validation_root: str | Path, schedule_path: str | Path,
    data_roots: dict[str, str | Path], cache_root: str | Path,
    schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run short actual-train-split microbenchmarks for both feature dimensions."""
    root = _validation_root(validation_root)
    _assert_schedule_identity(schedule_path, schedule_sha256)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available, gpu_name = False, None
    if not cuda_available:
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "artifact_type": "ACTUAL_DATA_ENGINEERING_PILOT_NOT_SCIENTIFIC_RESULT",
            "reason": "CUDA/PyTorch GPU runtime is unavailable; actual-data engineering measurements were not run.",
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha, "datasets": [],
        }
        path = _save_phase_report(root, "engineering_pilot", workflow_id, report)
        return {**report, "report_path": str(path)}

    check_validation_storage_capacity(
        local_path="/content/revision_r1_validation_staging",
        drive_path=root,
    )

    datasets: list[dict[str, Any]] = []
    for dataset in ("ember", "az_class"):
        if dataset not in data_roots:
            datasets.append({"dataset": dataset, "status": "FAIL", "error": "explicit Drive data root was not supplied"})
            continue
        output = root / "pilot" / workflow_id / dataset
        checkpoint = root / "checkpoints" / workflow_id / "pilot" / dataset
        worker = _run_worker(root, workflow_id, f"pilot_{dataset}", [
            "pilot", "--dataset", dataset, "--data-root", str(data_roots[dataset]),
            "--cache-root", str(cache_root), "--schedule-sha256", schedule_sha256,
            "--effective-schedule", str(schedule_path), "--output-dir", str(output),
            "--checkpoint-dir", str(checkpoint), "--classifier-steps", "5",
        ])
        datasets.append({
            "dataset": dataset, "status": "PASS" if worker.get("returncode") == 0 and worker.get("status") in {"PASS", "ALREADY_VALIDATED"} else "FAIL",
            "log_path": worker.get("log_path"), "result_path": worker.get("result_path"),
            "measurement": worker.get("measurement"), "error": worker.get("error"),
        })
    valid_measurements = [
        item["measurement"] for item in datasets
        if item.get("status") == "PASS" and isinstance(item.get("measurement"), dict)
    ]
    projected_runtime = None
    projected_storage = None
    production_storage_projection = None
    if len(valid_measurements) == 2:
        projected_runtime = {
            name: sum(float(item["runtime_extrapolation"][f"{name}_seconds_estimate"]) for item in valid_measurements)
            for name in ("classifier", "generator", "critic")
        }
        projected_storage = {
            "base_snapshot_scenario_bytes": sum(int(item["projected_storage"]["base_snapshot_scenario_bytes"]) for item in valid_measurements),
            "conservative_full_hybrid_replay_upper_scenario_bytes": sum(int(item["projected_storage"]["conservative_full_hybrid_replay_upper_scenario_bytes"]) for item in valid_measurements),
            "upper_scenario_assumes_all_five_arms_retain_full_hybrid_replay": True,
        }
        drive_free_bytes, _ = available_bytes(root)
        production_storage_projection = _production_checkpoint_storage_projection(
            valid_measurements, available_drive_bytes=drive_free_bytes,
        )
    report = {
        "schema_version": 1,
        "validation_build_id": VALIDATION_BUILD_ID,
        "status": "PASS" if len(datasets) == 2 and all(item["status"] == "PASS" for item in datasets) else "FAIL",
        "artifact_type": "ACTUAL_DATA_ENGINEERING_PILOT_NOT_SCIENTIFIC_RESULT",
        "resume_policy": "completed per-dataset pilot results and verified checkpoints are reused; an interrupted in-progress dataset microbenchmark restarts from its beginning, and partial measurements are never promoted",
        "gpu": gpu_name, "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        "datasets": datasets,
        "100_run_matrix_runtime_estimate_seconds": projected_runtime,
        "100_run_matrix_drive_checkpoint_storage_scenarios": projected_storage,
        "100_run_matrix_production_storage_gate": production_storage_projection,
        "estimates_are_not_measured_full_run_outcomes": True,
        "performance_tuning_from_test_set": False,
        "classifier_update_matching_is_not_total_compute_matching": True,
        "partial_task_runtime_extrapolations_are_estimates": True,
    }
    path = _save_phase_report(root, "engineering_pilot", workflow_id, report)
    return {**report, "report_path": str(path)}


def assert_production_storage_ready(
    *, validation_root: str | Path,
    schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Require a current-build 100-run pilot projection and recheck Drive quota."""
    root = _validation_root(validation_root)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    candidates = sorted(
        (root / "reports").glob(f"engineering_pilot_{workflow_id}_*.json"),
        key=lambda path: path.stat().st_mtime_ns, reverse=True,
    )
    for path in candidates:
        report = _load_json(path, "engineering pilot storage report")
        projection = report.get("100_run_matrix_production_storage_gate")
        if (
            report.get("status") != "PASS"
            or report.get("validation_build_id") != VALIDATION_BUILD_ID
            or report.get("schedule_sha256") != schedule_sha256
            or report.get("code_sha256") != code_sha
            or not isinstance(projection, dict)
            or projection.get("production_storage_gate") != "PASS"
        ):
            continue
        required = int(projection.get("required_drive_bytes_including_reserve", 0))
        free, storage_kind = available_bytes(root)
        if free < required:
            raise RuntimeError(
                "available Drive quota is below the projected 100-run requirement: "
                f"required {required} bytes including reserve, available {free} bytes"
            )
        return {
            "status": "PASS", "pilot_report": str(path),
            "pilot_report_sha256": sha256_file(path),
            "required_drive_bytes_including_reserve": required,
            "available_drive_bytes_rechecked": free,
            "storage_kind": storage_kind,
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        }
    raise RuntimeError(
        "production remains disabled until a PASS engineering pilot for this exact v5 code/schedule "
        "reports a sufficient 100-run checkpoint-storage projection"
    )


def prepare_manual_runtime_reset(
    *, validation_root: str | Path, schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None, kernel_pid: int,
) -> dict[str, Any]:
    """Persist a GAN-phase resume fixture and record the current notebook kernel."""
    root = _validation_root(validation_root)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    attempt_root = root / "manual_runtime_reset" / workflow_id
    unfinished = find_unfinished_manual_reset(attempt_root, schedule_sha256, code_sha)
    if unfinished is not None:
        pending_path, pending = unfinished
        return {"status": "NOT_RUN", "resume_state": "PENDING_RUNTIME_RESET", "pending_path": str(pending_path), "prepared_kernel_pid": pending["prepared_kernel_pid"]}
    attempt_id = _manual_reset_attempt_id(schedule_sha256, code_sha)
    completed_path = attempt_root / f"completed_{attempt_id}.json"
    failed_path = attempt_root / f"failed_{attempt_id}.json"
    terminal_path = completed_path if completed_path.is_file() else failed_path if failed_path.is_file() else None
    if terminal_path is not None:
        prior = _load_json(terminal_path, "manual runtime-reset terminal record")
        return {
            "status": "ALREADY_ATTEMPTED", "resume_state": "TERMINAL_ATTEMPT_REUSED",
            "attempt_id": attempt_id, "terminal_path": str(terminal_path),
            "prior_status": prior.get("status"), "new_checkpoint_tree_created": False,
        }
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if not available:
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "reason": "GPU runtime unavailable; manual reset fixture was not created.",
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        }
        path = _save_phase_report(root, "manual_runtime_reset", workflow_id, report)
        return {**report, "report_path": str(path)}

    scenario = f"manual_runtime_reset_{attempt_id}"
    case = _execute_resume_case(
        root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
        scenario=scenario, arm="generated_only",
        interruption="gan_after_generator", resume_now=False,
    )
    pending = {
        "schema_version": 1, "status": "PENDING_RUNTIME_RESET",
        "attempt_id": attempt_id, "scenario": scenario,
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        "prepared_kernel_pid": int(kernel_pid), "case": case,
        "instructions": "Restart the Colab runtime, remount Drive, verify/extract this same package, rerun preflight/reconciliation, then run the manual resume cell only.",
    }
    attempt_root.mkdir(parents=True, exist_ok=True)
    pending_path = attempt_root / f"pending_{attempt_id}.json"
    write_immutable_json(pending_path, pending)
    report = {
        "schema_version": 1, "status": "NOT_RUN", "resume_state": "PENDING_RUNTIME_RESET",
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        "pending_path": str(pending_path), "prepared_kernel_pid": int(kernel_pid),
        "drive_checkpoint_root": case["drive_checkpoint_root"],
        "checkpoint_identity_sha256": case["checkpoint_identity_sha256"],
    }
    report_path = _save_phase_report(root, "manual_runtime_reset", workflow_id, report)
    return {**report, "report_path": str(report_path)}


def resume_manual_runtime_reset(
    *, validation_root: str | Path, schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None, kernel_pid: int,
) -> dict[str, Any]:
    root = _validation_root(validation_root)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    attempt_root = root / "manual_runtime_reset" / workflow_id
    unfinished = find_unfinished_manual_reset(attempt_root, schedule_sha256, code_sha)
    if unfinished is None:
        raise FileNotFoundError("manual runtime-reset checkpoint is missing; run the prepare cell before restarting")
    pending_path, pending = unfinished
    if int(pending.get("prepared_kernel_pid", -1)) == int(kernel_pid):
        report = {
            "schema_version": 1, "status": "NOT_RUN",
            "reason": "Notebook kernel PID is unchanged. Use Runtime > Restart session, rerun setup/mount/package/preflight/reconciliation cells, then retry this resume cell.",
            "schedule_sha256": schedule_sha256, "prepared_kernel_pid": int(kernel_pid),
        }
        path = _save_phase_report(root, "manual_runtime_reset", workflow_id, report)
        return {**report, "report_path": str(path)}
    try:
        checkpoint_root = Path(pending["case"]["drive_checkpoint_root"])
        checkpoint_identity = _drive_checkpoint_identity(checkpoint_root, schedule_sha256)
        checkpoint_identity_digest = hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True).encode()).hexdigest()
        if checkpoint_identity_digest != pending["case"].get("checkpoint_identity_sha256"):
            raise ValueError("manual runtime-reset Drive checkpoint identity differs from the prepared fixture")
        case = _execute_resume_case(
            root=root, workflow_id=workflow_id, schedule_sha256=schedule_sha256,
            scenario=str(pending["scenario"]), arm="generated_only",
            interruption="gan_after_generator", resume_now=True, require_new_resume_process=True,
        )
    except Exception as exc:
        failed = {
            "status": "FAIL", "attempt_id": pending["attempt_id"],
            "error_type": type(exc).__name__, "error": str(exc),
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        }
        write_immutable_json(attempt_root / f"failed_{pending['attempt_id']}.json", failed)
        path = _save_phase_report(root, "manual_runtime_reset", workflow_id, {
            "schema_version": 1, "status": "FAIL",
            "artifact_type": "MANUAL_COLAB_RUNTIME_RESET_SYNTHETIC_GPU_CHECK",
            **failed,
        })
        return {**failed, "report_path": str(path)}
    report = {
        "schema_version": 1, "status": case["status"],
        "artifact_type": "MANUAL_COLAB_RUNTIME_RESET_SYNTHETIC_GPU_CHECK",
        "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        "prepared_kernel_pid": pending["prepared_kernel_pid"], "resumed_kernel_pid": int(kernel_pid),
        "kernel_pid_changed": True, "case": case,
    }
    process_ids = [
        pending["case"].get("baseline_worker_pid"),
        pending["case"].get("interrupt_worker_pid"),
        *case.get("worker_process_ids", []),
    ]
    process_ids = [int(value) for value in process_ids if value is not None]
    if len(process_ids) < 3 or len(set(process_ids)) != len(process_ids):
        failed = {
            "status": "FAIL", "attempt_id": pending["attempt_id"],
            "error": "manual reset lacked three distinct uninterrupted/interrupted/resumed worker process IDs",
            "worker_process_ids": process_ids,
            "schedule_sha256": schedule_sha256, "code_sha256": code_sha,
        }
        write_immutable_json(attempt_root / f"failed_{pending['attempt_id']}.json", failed)
        path = _save_phase_report(root, "manual_runtime_reset", workflow_id, {
            "schema_version": 1, "artifact_type": "MANUAL_COLAB_RUNTIME_RESET_SYNTHETIC_GPU_CHECK",
            **failed,
        })
        return {**failed, "report_path": str(path)}
    report["worker_process_ids"] = process_ids
    report["three_worker_processes_distinct"] = True
    write_immutable_json(attempt_root / f"completed_{pending['attempt_id']}.json", {
        "status": report["status"], "prepared_kernel_pid": pending["prepared_kernel_pid"],
        "resumed_kernel_pid": int(kernel_pid), "scenario": pending["scenario"],
    })
    path = _save_phase_report(root, "manual_runtime_reset", workflow_id, report)
    return {**report, "report_path": str(path)}


def _latest_report(root: Path, prefix: str, workflow_id: str) -> tuple[str, str | None]:
    candidates = sorted((root / "reports").glob(f"{prefix}_{workflow_id}_*.json"), key=lambda p: p.stat().st_mtime_ns)
    for path in reversed(candidates):
        report = _load_json(path, f"{prefix} report")
        if report.get("validation_build_id") == VALIDATION_BUILD_ID:
            if report.get("execution_scope") == "LOCAL_CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE":
                return "NOT_RUN", str(path)
            return str(report.get("status", "FAIL")), str(path)
    return "NOT_RUN", None


def export_validation_summary(
    *, validation_root: str | Path,
    schedule_sha256: str = EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a compact immutable status snapshot; never infer PASS from CPU checks."""
    root = _validation_root(validation_root)
    workflow_id, code_sha = _workflow_id(schedule_sha256, package_root)
    reconciliation_candidates = sorted(
        (root / "reports").glob(f"schedule_reconciliation_{schedule_sha256[:12]}_*_{VALIDATION_BUILD_ID}.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    reconciliation_ref = None
    reconciliation_status = "NOT_RUN"
    for candidate in reversed(reconciliation_candidates):
        reconciliation = _load_json(candidate, "schedule reconciliation")
        if reconciliation.get("validation_build_id") == VALIDATION_BUILD_ID and reconciliation.get("effective_schedule_sha256") == schedule_sha256:
            reconciliation_ref = str(candidate)
            reconciliation_status = str(reconciliation.get("status", "FAIL"))
            break
    statuses = {
        "schedule_reconciliation": {"status": reconciliation_status, "report": reconciliation_ref},
        "synthetic_gpu_smoke": dict(zip(("status", "report"), _latest_report(root, "gpu_smoke", workflow_id))),
        "fresh_process_resume": dict(zip(("status", "report"), _latest_report(root, "resume_validation", workflow_id))),
        "manual_runtime_reset": dict(zip(("status", "report"), _latest_report(root, "manual_runtime_reset", workflow_id))),
        "actual_data_engineering_pilot": dict(zip(("status", "report"), _latest_report(root, "engineering_pilot", workflow_id))),
        "production_matrix": {"status": "NOT_RUN", "enabled": False, "report": None},
    }
    statuses["manual_runtime_reset"].setdefault("status", "NOT_RUN")
    if statuses["manual_runtime_reset"]["status"] == "NOT_RUN":
        attempt_root = root / "manual_runtime_reset" / workflow_id
        pending_records = sorted(attempt_root.glob("pending_*.json")) if attempt_root.is_dir() else []
        for pending in pending_records:
            record = _load_json(pending, "manual runtime-reset pending record")
            terminal = (attempt_root / f"completed_{record.get('attempt_id')}.json").is_file() or (attempt_root / f"failed_{record.get('attempt_id')}.json").is_file()
            if not terminal:
                statuses["manual_runtime_reset"]["pending_record"] = str(pending)
                break
    phase_states = [value["status"] for value in statuses.values() if isinstance(value, dict)]
    if "FAIL" in phase_states:
        overall = "FAIL"
    elif all(value == "PASS" for value in phase_states[:-1]):
        overall = "PASS"
    else:
        overall = "NOT_RUN"
    report = {
        "schema_version": 1,
        "validation_build_id": VALIDATION_BUILD_ID,
        "overall_status": overall,
        "schedule_sha256": schedule_sha256,
        "validation_code_sha256": code_sha,
        "validation_workflow_id": workflow_id,
        "phases": statuses,
        "scientific_settings_tuned_from_pilot_test_performance": False,
        "production_matrix_launched": False,
        "note": "GPU statuses are produced only by Colab CUDA cells. Local CPU tests never change a GPU phase from NOT_RUN to PASS.",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    path = root / "reports" / f"validation_summary_{schedule_sha256[:12]}_{workflow_id}_{_timestamp()}.json"
    write_immutable_json(path, report)
    return {**report, "summary_path": str(path)}


