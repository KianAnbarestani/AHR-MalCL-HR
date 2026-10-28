"""Production orchestration kept outside the hashed revision_r1 training core."""

from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from revision_r1.config import ARMS, DATASETS, SEEDS, TASK_SIZES, make_run_config
from revision_r1.schedule import PoolCounts, load_update_schedule
from revision_r1.storage import (
    DEFAULT_STORAGE_RESERVE_BYTES,
    StorageCapacityError,
    require_available_bytes,
)
from revision_r1.task_stream import class_order, order_sha256, split_order


PLAN_SCHEMA_VERSION = 1
PLAN_KIND = "REVISION_R1_FROZEN_PRODUCTION_MATRIX"
EXPECTED_EFFECTIVE_SCHEDULE_SHA256 = "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212"
EXPECTED_RUN_COUNT = 100
DATASET_ORDER = ("ember", "az_class")
FIRST_RUN = ("ember", 42, "full_hybrid")
EXPECTED_ANCHOR_CORRECTIONS = 47
EXPECTED_UPDATE_BUDGET_CHANGES = 5
DRIVE_METADATA_SCOPE = "https://www.googleapis.com/auth/drive.metadata.readonly"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
CHECKPOINT_SIZE_MULTIPLIER = 1.5
CHECKPOINT_SIZE_FIXED_OVERHEAD_BYTES = 16 * 1024 * 1024
OUTPUT_BYTES_FLOOR_PER_RUN = 256 * 1024 * 1024
ZIP_ARCHIVE_GROWTH_MARGIN = 0.01
TASK_BOUNDARY_GENERATIONS_PER_RUN = 11
RECOVERY_GENERATIONS_PER_RUN = 2
DRIVE_WRITE_PROBE_BYTES = b"revision_R1 shared-folder capacity probe v1\n"
_DRIVE_WRITE_PROBE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
V62_RECOVERY_BUILD_ID = "revision_R1_colab_validation_v6_2_final_gan_recovery_qa"
V62_RECOVERY_WORKFLOW_ID = "2b4d8f5f1f29a938"
V62_RECOVERY_REPORT_NAME = "resume_validation_2b4d8f5f1f29a938_20261026T142320Z_946699443.json"
V62_RECOVERY_CODE_SHA256 = "adb4ee76f1d7fb82a35ffaf4f38f5ddaa9b7649c36098f6110f47a938f9faf88"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_v62_recovery_checkpoint_bound(report_path: str | Path) -> dict[str, Any]:
    """Read only the user-identified successful v6.2 GPU GAN recovery report.

    Its maximum checkpoint field is already a conservative estimate (the
    validation worker takes the maximum of payload bytes and the checkpoint
    estimator); do not apply that estimator a second time.
    """
    path = Path(report_path).expanduser().resolve()
    if path.name != V62_RECOVERY_REPORT_NAME:
        raise ValueError(f"expected the exact successful v6.2 recovery report {V62_RECOVERY_REPORT_NAME}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read the supplied v6.2 recovery report: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError("v6.2 recovery report must contain a JSON object")
    if (
        report.get("validation_build_id") != V62_RECOVERY_BUILD_ID
        or report.get("validation_workflow_id") != V62_RECOVERY_WORKFLOW_ID
        or report.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or report.get("status") != "PASS"
        or report.get("execution_scope") != "COLAB_CUDA_SYNTHETIC_VALIDATION"
    ):
        raise ValueError("the named v6.2 GPU recovery report identity/status is not the confirmed successful workflow")
    classifier = report.get("classifier_phase")
    gan = report.get("gan_phase")
    if not isinstance(classifier, dict) or classifier.get("status") != "PASS" or classifier.get("rerun") is not False:
        raise ValueError("v6.2 classifier PASS reuse is not verified in the recovery report")
    if not isinstance(gan, dict) or gan.get("status") != "PASS":
        raise ValueError("v6.2 GAN recovery phase is not PASS")
    interruption = gan.get("interruption_drive_checkpoint")
    if not isinstance(interruption, dict):
        raise ValueError("v6.2 Drive interruption checkpoint evidence is missing or malformed")
    publication = interruption.get("worker_publication_matches_drive_readback")
    if not isinstance(publication, dict):
        raise ValueError("v6.2 worker publication/read-back evidence must be a structured PASS record")
    generation_count = publication.get("generation_count")
    if publication.get("status") != "PASS" or type(generation_count) is not int or generation_count != 1:
        raise ValueError("v6.2 worker publication/read-back evidence is not PASS for exactly one generation")
    for field in ("generation", "manifest_sha256", "payload_sha256", "payload_bytes"):
        outer_value = interruption.get(field)
        published_value = publication.get(field)
        if published_value != outer_value:
            raise ValueError(
                f"v6.2 publication/read-back {field} differs from its enclosing interruption evidence"
            )
        if field == "generation" and (type(outer_value) is not int or outer_value <= 0):
            raise ValueError("v6.2 interruption evidence has no positive generation number")
        if field == "payload_bytes" and (type(outer_value) is not int or outer_value <= 0):
            raise ValueError("v6.2 interruption evidence has no positive payload byte count")
        if field.endswith("sha256") and not _is_sha256(outer_value):
            raise ValueError(f"v6.2 interruption evidence has a malformed {field}")
    if (
        interruption.get("fresh_process_restored_from_drive") is not True
        or interruption.get("local_staging_empty_before_restore") is not True
    ):
        raise ValueError("v6.2 Drive publication/read-back/fresh-process evidence is incomplete")
    reference = report.get("reference_checkpoint")
    if not isinstance(reference, dict) or reference.get("status") != "PASS":
        raise ValueError("v6.2 reference checkpoint verification is not PASS")
    identity = reference.get("identity", {})
    if (
        reference.get("code_sha256") != V62_RECOVERY_CODE_SHA256
        or reference.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or reference.get("reference_manifest_verified") is not True
        or reference.get("reference_payload_hash_verified") is not True
        or identity.get("code_sha256") != V62_RECOVERY_CODE_SHA256
        or identity.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or identity.get("execution_scope") != "synthetic_validation"
    ):
        raise ValueError("v6.2 reference checkpoint source, schedule, scope, or integrity evidence is incompatible")
    maximum = reference.get("max_checkpoint_payload_bytes")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("v6.2 report has no positive conservative maximum checkpoint size")
    if not _is_sha256(reference.get("manifest_sha256")) or not _is_sha256(reference.get("payload_sha256")):
        raise ValueError("v6.2 reference manifest/payload hashes are missing or malformed")
    return {
        "max_checkpoint_payload_bytes": maximum,
        "size_is_already_conservative_bound": True,
        "provenance": "v6.2 synthetic GPU recovery report; not a production measurement",
        "build_id": V62_RECOVERY_BUILD_ID,
        "workflow_id": V62_RECOVERY_WORKFLOW_ID,
        "schedule_sha256": EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
        "legacy_reference_code_sha256": V62_RECOVERY_CODE_SHA256,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_generation": int(reference["generation"]),
        "reference_manifest_sha256": reference["manifest_sha256"],
        "reference_payload_sha256": reference["payload_sha256"],
    }


def _refresh_plan_hash(plan: dict[str, Any]) -> dict[str, Any]:
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def training_source_sha256(package_root: str | Path) -> str:
    """Match runner._code_sha256 exactly, over the actual shipped training modules."""
    source_root = Path(package_root) / "revision_r1"
    if not source_root.is_dir():
        raise FileNotFoundError(f"training source directory is missing: {source_root}")
    digest = hashlib.sha256()
    paths = sorted(source_root.glob("*.py"))
    if not paths:
        raise ValueError(f"no training source files under {source_root}")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_dataset_references(dataset_references: dict[str, dict[str, Any]]) -> None:
    if set(dataset_references) != set(DATASET_ORDER):
        raise ValueError("dataset references must contain exactly ember and az_class")
    for dataset in DATASET_ORDER:
        reference = dataset_references[dataset]
        spec = DATASETS[dataset]
        if reference.get("train_md5") != spec["train"]["md5"]:
            raise ValueError(f"{dataset} training MD5 reference differs from the frozen dataset identity")
        if reference.get("test_md5") != spec["test"]["md5"]:
            raise ValueError(f"{dataset} test MD5 reference differs from the frozen dataset identity")
        if reference.get("feature_dim") != spec["feature_dim"]:
            raise ValueError(f"{dataset} feature dimension differs from the frozen dataset identity")
        if list(reference.get("train_shape", [])) != list(spec["train"]["shape"]):
            raise ValueError(f"{dataset} training shape differs from the frozen dataset identity")
        if list(reference.get("test_shape", [])) != list(spec["test"]["shape"]):
            raise ValueError(f"{dataset} test shape differs from the frozen dataset identity")


def _run_order() -> list[tuple[str, int, str]]:
    remaining = [
        (dataset, seed, arm)
        for dataset in DATASET_ORDER
        for seed in SEEDS
        for arm in ARMS
        if (dataset, seed, arm) != FIRST_RUN
    ]
    return [FIRST_RUN, *remaining]


def _expected_protocol() -> dict[str, Any]:
    return {
        "datasets": list(DATASET_ORDER),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "task_sizes": list(TASK_SIZES),
        "classifier_batch_size": 256,
        "sampling": "mixed_pool_inverse_class_frequency_with_replacement",
        "scaler": "taskwise_training_only",
        "replay_coordinates": "raw",
        "matrix_size": EXPECTED_RUN_COUNT,
    }


def build_frozen_run_plan(
    package_root: str | Path,
    *,
    schedule_sha256: str,
    dataset_references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic 100-run manifest; no data or model is loaded."""
    if not _is_sha256(schedule_sha256):
        raise ValueError("effective schedule SHA-256 must be 64 hexadecimal characters")
    _validate_dataset_references(dataset_references)
    source_hash = training_source_sha256(package_root)
    runs = []
    for run_index, (dataset, seed, arm) in enumerate(_run_order(), start=1):
        config = make_run_config(dataset, seed, arm, schedule_sha256)
        order = class_order(seed)
        reference = dataset_references[dataset]
        runs.append({
            "run_index": run_index,
            "run_key": f"{dataset}_seed{seed}_{arm}",
            "dataset": dataset,
            "seed": int(seed),
            "arm": arm,
            "is_first_run": run_index == 1,
            "config": config.payload(),
            "config_sha256": config.sha256(),
            "training_source_sha256": source_hash,
            "schedule_sha256": schedule_sha256,
            "dataset_reference": {
                "train_md5": reference["train_md5"],
                "test_md5": reference["test_md5"],
                "train_shape": list(reference["train_shape"]),
                "test_shape": list(reference["test_shape"]),
                "feature_dim": int(reference["feature_dim"]),
            },
            "class_order": order,
            "class_order_sha256": order_sha256(order),
            "update_budgets_by_task": None,
            "initial_state_sha256": None,
        })
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "training_reproduction_claimed": False,
        "protocol": _expected_protocol(),
        "schedule_sha256": schedule_sha256,
        "training_source_sha256": source_hash,
        "run_count": len(runs),
        "runs": runs,
    }
    _refresh_plan_hash(plan)
    validate_frozen_plan(plan, expected_source_sha256=source_hash)
    return plan


def _expected_run_keys() -> list[tuple[str, int, str]]:
    return _run_order()


def validate_frozen_plan(
    plan: dict[str, Any],
    *,
    expected_source_sha256: str | None = None,
    verify_digest: bool = True,
) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("plan_kind") != PLAN_KIND:
        raise ValueError("unsupported or malformed frozen production plan")
    if plan.get("training_reproduction_claimed") is not False:
        raise ValueError("production plan must explicitly disclaim historical training reproduction")
    if plan.get("protocol") != _expected_protocol():
        raise ValueError("frozen plan scientific protocol differs from revision_R1")
    if plan.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("frozen plan is not bound to the approved effective schedule SHA-256")
    source_hash = plan.get("training_source_sha256")
    if not _is_sha256(source_hash):
        raise ValueError("frozen plan training-source SHA-256 is malformed")
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError("frozen plan training-source identity differs from the actual package")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != EXPECTED_RUN_COUNT or plan.get("run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("production plan must contain exactly 100 run records")
    if not runs or (runs[0].get("dataset"), runs[0].get("seed"), runs[0].get("arm")) != FIRST_RUN:
        raise ValueError("the frozen plan first run must be ember / seed 42 / full_hybrid")
    observed = [(row.get("dataset"), row.get("seed"), row.get("arm")) for row in runs]
    if len(set(observed)) != EXPECTED_RUN_COUNT:
        raise ValueError("production plan contains duplicate run identities")
    if observed != _expected_run_keys():
        raise ValueError("production plan run matrix/order is incomplete or changed")
    for index, row in enumerate(runs, start=1):
        dataset, seed, arm = observed[index - 1]
        config = make_run_config(dataset, seed, arm, plan["schedule_sha256"])
        if row.get("run_index") != index or row.get("run_key") != f"{dataset}_seed{seed}_{arm}":
            raise ValueError(f"run index/key mismatch at plan row {index}")
        if row.get("config") != config.payload() or row.get("config_sha256") != config.sha256():
            raise ValueError(f"run config identity mismatch at plan row {index}")
        if row.get("training_source_sha256") != source_hash or row.get("schedule_sha256") != plan["schedule_sha256"]:
            raise ValueError(f"source or schedule identity mismatch at plan row {index}")
        order = class_order(seed)
        if row.get("class_order") != order or row.get("class_order_sha256") != order_sha256(order):
            raise ValueError(f"class order identity mismatch at plan row {index}")
        if bool(row.get("is_first_run")) is not (index == 1):
            raise ValueError(f"first-run marker mismatch at plan row {index}")
        expected_reference = {
            "train_md5": DATASETS[dataset]["train"]["md5"],
            "test_md5": DATASETS[dataset]["test"]["md5"],
            "train_shape": list(DATASETS[dataset]["train"]["shape"]),
            "test_shape": list(DATASETS[dataset]["test"]["shape"]),
            "feature_dim": int(DATASETS[dataset]["feature_dim"]),
        }
        if row.get("dataset_reference") != expected_reference:
            raise ValueError(f"dataset identity reference mismatch at plan row {index}")
    if verify_digest:
        observed_hash = plan.get("plan_sha256")
        payload = dict(plan)
        payload.pop("plan_sha256", None)
        if observed_hash != _sha256_json(payload):
            raise ValueError("frozen plan content SHA-256 mismatch")


def _preflight_identity(preflight: dict[str, Any], dataset: str) -> dict[str, str]:
    try:
        train = preflight[dataset]["train"]
        test = preflight[dataset]["test"]
        return {
            "train_md5": str(train["md5"]),
            "train_sha256": str(train["sha256"]),
            "test_md5": str(test["md5"]),
            "test_sha256": str(test["sha256"]),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"preflight lacks complete train/test dataset identity for {dataset}") from exc


def _preflight_class_counts(preflight: dict[str, Any], dataset: str) -> dict[int, int]:
    try:
        raw = preflight[dataset]["train"]["class_counts"]
        counts = {int(label): int(count) for label, count in raw.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"preflight lacks valid training class counts for {dataset}") from exc
    if set(counts) != set(range(100)) or any(value < 1 for value in counts.values()):
        raise ValueError(f"preflight training class counts are incomplete for {dataset}")
    return counts


def _pool_matrix_from_preflight(preflight: dict[str, Any]) -> dict[tuple[str, int, int], PoolCounts]:
    matrix = {}
    for dataset in DATASET_ORDER:
        counts = _preflight_class_counts(preflight, dataset)
        quota = int(DATASETS[dataset]["anchor_k"])
        for seed in SEEDS:
            seen: list[int] = []
            for task_id, group in enumerate(split_order(class_order(seed), TASK_SIZES), start=1):
                current = sum(counts[class_id] for class_id in group)
                anchor = sum(min(quota, counts[class_id]) for class_id in seen)
                generated = len(seen) * 100
                matrix[(dataset, seed, task_id)] = PoolCounts(current, anchor, generated)
                seen.extend(group)
    return matrix


def bind_runtime_schedule(
    plan: dict[str, Any],
    effective_schedule_path: str | Path,
    effective_manifest_path: str | Path,
    preflight: dict[str, Any],
    *,
    expected_anchor_corrections: int = EXPECTED_ANCHOR_CORRECTIONS,
    expected_budget_changes: int = EXPECTED_UPDATE_BUDGET_CHANGES,
) -> dict[str, Any]:
    """Bind the delivered plan to the exact Drive schedule and dataset preflight."""
    validate_frozen_plan(plan)
    schedule_path = Path(effective_schedule_path)
    manifest_path = Path(effective_manifest_path)
    if not schedule_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("effective schedule CSV and its manifest must both exist")
    schedule_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    if schedule_hash != plan["schedule_sha256"]:
        raise ValueError(f"effective schedule SHA-256 mismatch: {schedule_hash} != {plan['schedule_sha256']}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"effective schedule manifest is unreadable: {manifest_path}") from exc
    if manifest.get("status") != "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING" or manifest.get("training_reproduction_claimed") is not False:
        raise ValueError("effective schedule manifest has invalid or historical provenance")
    if manifest.get("row_count") != 220 or manifest.get("source_csv_sha256") != schedule_hash:
        raise ValueError("effective schedule manifest row count/hash differs from its CSV")
    if preflight.get("status") != "PREFLIGHT_VALIDATED_NO_TRAINING":
        raise ValueError("dataset preflight is not a validated no-training record")
    preflight_schedule = preflight.get("schedule", {})
    if preflight_schedule.get("sha256") != schedule_hash or preflight_schedule.get("row_count") != 220:
        raise ValueError("preflight schedule identity differs from the effective schedule")
    if preflight_schedule.get("provenance") != "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING":
        raise ValueError("preflight schedule provenance is unexpected")

    dataset_identities = {dataset: _preflight_identity(preflight, dataset) for dataset in DATASET_ORDER}
    if manifest.get("dataset_identities") != dataset_identities:
        raise ValueError("effective schedule manifest dataset identities differ from Drive preflight")
    for dataset in DATASET_ORDER:
        reference = plan["runs"][0]["dataset_reference"] if dataset == "ember" else next(
            row["dataset_reference"] for row in plan["runs"] if row["dataset"] == dataset
        )
        identity = dataset_identities[dataset]
        if identity["train_md5"] != reference["train_md5"] or identity["test_md5"] != reference["test_md5"]:
            raise ValueError(f"{dataset} NPZ checksum identity differs from the frozen plan")

    reconciliation = manifest.get("reconciliation", {})
    adjusted = reconciliation.get("adjusted_rows")
    changed = reconciliation.get("changed_update_rows")
    if adjusted != expected_anchor_corrections or changed != expected_budget_changes:
        raise ValueError(
            "effective schedule reconciliation differs from the verified 47-anchor/5-budget record: "
            f"anchors={adjusted}, budgets={changed}"
        )
    preflight_reconciliation = preflight_schedule.get("reconciliation")
    if preflight_reconciliation is not None:
        if (
            preflight_reconciliation.get("adjusted_rows") != expected_anchor_corrections
            or preflight_reconciliation.get("changed_update_rows") != expected_budget_changes
        ):
            raise ValueError("preflight reconciliation differs from the frozen 47-anchor/5-budget record")

    actual_pools = _pool_matrix_from_preflight(preflight)
    rows = load_update_schedule(
        schedule_path, DATASET_ORDER, SEEDS, TASK_SIZES, actual_pools,
        expected_sha256=schedule_hash,
    )
    bound = json.loads(json.dumps(plan))
    updates_by_run: dict[tuple[str, int], list[int]] = {}
    for row in rows.values():
        updates_by_run.setdefault((row.dataset, row.seed), [0] * len(TASK_SIZES))[row.task - 1] = row.updates
    for run in bound["runs"]:
        run["update_budgets_by_task"] = updates_by_run[(run["dataset"], run["seed"])][:]
        run["effective_schedule_sha256"] = schedule_hash
        run["dataset_identity"] = dataset_identities[run["dataset"]]
    bound["runtime_binding"] = {
        # Paths through Google Drive shortcuts/mount aliases can change between
        # Colab runtimes. Bind content hashes and filenames, not the local
        # mount's resolved absolute path, so a verified resume has a stable plan ID.
        "effective_schedule_path": schedule_path.name,
        "effective_schedule_manifest_path": manifest_path.name,
        "schedule_sha256": schedule_hash,
        "schedule_rows": len(rows),
        "training_reproduction_claimed": False,
        "dataset_identities": dataset_identities,
        "anchor_corrections": adjusted,
        "update_budget_changes": changed,
        "training_source_sha256": plan["training_source_sha256"],
        "npz_payloads_reread_during_schedule_binding": False,
    }
    _refresh_plan_hash(bound)
    return bound


def _drive_file_fields() -> str:
    return (
        "files(id,name,mimeType,parents,owners(emailAddress,displayName),driveId,"
        "shortcutDetails(targetId,targetMimeType),capabilities(canAddChildren),ownedByMe)"
    )


def _drive_query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _get_drive_file(service, file_id: str) -> dict[str, Any]:
    try:
        return service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,parents,owners(emailAddress,displayName),driveId,capabilities(canAddChildren),ownedByMe",
            supportsAllDrives=True,
        ).execute()
    except Exception as exc:
        raise StorageCapacityError(f"cannot resolve configured Drive shortcut target {file_id}: {exc}") from exc


def resolve_drive_folder(service, components: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Resolve only a configured My Drive path; follow each folder shortcut explicitly."""
    names = tuple(str(value) for value in components)
    if not names or any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in names):
        raise ValueError("Drive destination must be a nonempty sequence of safe folder names")
    parent_id = "root"
    parent_metadata: dict[str, Any] = {"id": "root", "name": "My Drive", "owners": [], "capabilities": {"canAddChildren": True}}
    traversed_shortcut = False
    for index, name in enumerate(names):
        query = (
            f"'{parent_id}' in parents and name='{_drive_query_literal(name)}' "
            f"and trashed=false and (mimeType = '{DRIVE_FOLDER_MIME}' "
            f"or mimeType = '{DRIVE_SHORTCUT_MIME}')"
        )
        matches: list[dict[str, Any]] = []
        page_token = None
        try:
            while True:
                response = service.files().list(
                    q=query, pageSize=100, pageToken=page_token,
                    fields=f"nextPageToken,{_drive_file_fields()}",
                    includeItemsFromAllDrives=True, supportsAllDrives=True,
                ).execute()
                matches.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except Exception as exc:
            raise StorageCapacityError(f"cannot inspect configured Drive folder {name!r}: {exc}") from exc
        if len(matches) > 1:
            raise StorageCapacityError(
                f"configured Drive path is ambiguous at {name!r}: found {len(matches)} folders/shortcuts under parent {parent_id}"
            )
        if not matches:
            capabilities = parent_metadata.get("capabilities", {})
            return {
                "folder_id": None,
                "folder_name": names[-1],
                "parent_id": parent_id,
                "owner_email": _owner_email(parent_metadata),
                "drive_id": parent_metadata.get("driveId"),
                "can_add_children": capabilities.get("canAddChildren"),
                "owned_by_me": parent_metadata.get("ownedByMe"),
                "path_exists": False,
                "traversed_shortcut": traversed_shortcut,
                "components": list(names),
                "unresolved_components": list(names[index:]),
            }
        record = matches[0]
        if record.get("mimeType") == DRIVE_SHORTCUT_MIME:
            details = record.get("shortcutDetails", {})
            target_id = details.get("targetId")
            if details.get("targetMimeType") != DRIVE_FOLDER_MIME or not target_id:
                raise StorageCapacityError(f"configured Drive shortcut {name!r} does not point to a folder")
            target = _get_drive_file(service, str(target_id))
            if target.get("mimeType") != DRIVE_FOLDER_MIME:
                raise StorageCapacityError(f"configured Drive shortcut {name!r} target is not a folder")
            record = target
            traversed_shortcut = True
        parent_id = str(record.get("id", ""))
        if not parent_id:
            raise StorageCapacityError(f"Drive metadata omitted folder ID for {name!r}")
        parent_metadata = record
    return {
        "folder_id": parent_id,
        "folder_name": names[-1],
        "parent_id": (parent_metadata.get("parents") or [None])[0],
        "owner_email": _owner_email(parent_metadata),
        "drive_id": parent_metadata.get("driveId"),
        "can_add_children": parent_metadata.get("capabilities", {}).get("canAddChildren"),
        "owned_by_me": parent_metadata.get("ownedByMe"),
        "path_exists": True,
        "traversed_shortcut": traversed_shortcut,
        "components": list(names),
        "unresolved_components": [],
    }


def _owner_email(metadata: dict[str, Any]) -> str | None:
    owners = metadata.get("owners") or []
    if not owners or not isinstance(owners[0], dict):
        return None
    value = owners[0].get("emailAddress")
    return str(value).strip().lower() if value else None


def _drive_api_service():
    try:
        import google.auth
        from googleapiclient.discovery import build

        credentials, _ = google.auth.default(scopes=[DRIVE_METADATA_SCOPE])
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        raise StorageCapacityError(
            "cannot authenticate read-only Google Drive metadata access; mount Drive and authenticate the intended owner account"
        ) from exc


def _logical_drive_path_components(
    path: str | Path, *, drive_root: str | Path = "/content/drive/MyDrive",
) -> tuple[str, tuple[str, ...]]:
    """Resolve configured MyDrive components lexically, without following mount shortcuts."""
    configured = Path(os.path.abspath(str(Path(path).expanduser())))
    logical_root = Path(os.path.abspath(str(Path(drive_root).expanduser())))
    try:
        components = configured.relative_to(logical_root).parts
    except ValueError as exc:
        raise StorageCapacityError(
            f"production Drive path must be configured below {logical_root}: {configured}"
        ) from exc
    if not components or any(value in {"", ".", ".."} for value in components):
        raise StorageCapacityError("production destination must be a child of MyDrive through safe folder components")
    return str(configured), tuple(components)


def _drive_account_snapshot(service) -> dict[str, Any]:
    try:
        account = service.about().get(
            fields="user(emailAddress,displayName),storageQuota(limit,usage,usageInDrive,usageInDriveTrash)"
        ).execute()
    except Exception as exc:
        raise StorageCapacityError(f"cannot read authenticated Google Drive account/quota metadata: {exc}") from exc
    authenticated_user = str(account.get("user", {}).get("emailAddress", "")).strip().lower()
    if not authenticated_user:
        raise StorageCapacityError("Google Drive API did not identify the authenticated user")
    quota = account.get("storageQuota")
    if not isinstance(quota, dict):
        quota = {}
    limit = _optional_nonnegative_int(quota.get("limit"), "quota limit")
    usage = _optional_nonnegative_int(quota.get("usage"), "quota usage")
    if limit is not None and usage is not None:
        if usage > limit:
            raise StorageCapacityError(f"Google Drive usage exceeds quota limit ({usage} > {limit})")
        available = limit - usage
        quota_status = "FINITE_VERIFIED"
        capacity_note = None
    else:
        available = None
        quota_status = "UNKNOWN_CAPACITY"
        capacity_note = (
            "Drive API did not report both a finite limit and total usage. A missing limit may correspond to an "
            "unlimited or pooled-quota account; no numeric free-byte value is inferred."
        )
    return {
        "authenticated_user": authenticated_user,
        "quota": quota,
        "quota_limit_bytes": limit,
        "quota_usage_bytes": usage,
        "usage_in_drive_bytes": _optional_nonnegative_int(quota.get("usageInDrive"), "Drive usage"),
        "usage_in_drive_trash_bytes": _optional_nonnegative_int(quota.get("usageInDriveTrash"), "Drive Trash usage"),
        "available_bytes": available,
        "quota_status": quota_status,
        "quota_capacity_note": capacity_note,
    }


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StorageCapacityError(f"Google Drive returned a nonnumeric {label}") from exc
    if result < 0:
        raise StorageCapacityError(f"Google Drive returned a negative {label}")
    return result


def _lookup_drive_child(service, parent_id: str, name: str) -> dict[str, Any] | None:
    query = (
        f"'{_drive_query_literal(parent_id)}' in parents and name='{_drive_query_literal(name)}' "
        "and trashed=false"
    )
    matches: list[dict[str, Any]] = []
    page_token = None
    try:
        while True:
            response = service.files().list(
                q=query, pageSize=100, pageToken=page_token,
                fields="nextPageToken,files(id,name,mimeType,parents,owners(emailAddress),ownedByMe,quotaBytesUsed,size)",
                includeItemsFromAllDrives=True, supportsAllDrives=True,
            ).execute()
            matches.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        raise StorageCapacityError(f"cannot verify the temporary Drive write-capability probe: {exc}") from exc
    if len(matches) > 1:
        raise StorageCapacityError("Drive returned multiple files for a unique write-capability probe name")
    return matches[0] if matches else None


def _write_capability_probe(
    path: str | Path, location: dict[str, Any], authenticated_user: str, service,
    *, drive_root: str | Path, metadata_lookup=None,
) -> dict[str, Any]:
    """Create/read/remove one tiny file through the mounted destination and verify its Drive owner."""
    configured = Path(path).expanduser()
    target = configured.resolve()
    if target.exists():
        if not target.is_dir() or location.get("path_exists") is not True or not location.get("folder_id"):
            raise StorageCapacityError("write-capability probe cannot bind the mounted path to the resolved Drive folder")
        probe_directory = target
        parent_id = str(location["folder_id"])
        probe_scope = "exact_existing_destination"
    else:
        if location.get("path_exists") is True:
            raise StorageCapacityError("write-capability probe cannot find the API-resolved folder in the mounted Drive path")
        probe_directory = _nearest_existing_directory(configured)
        parent_id = str(location.get("parent_id") or "")
        if not parent_id:
            raise StorageCapacityError("write-capability probe has no resolved existing Drive parent")
        try:
            probe_directory.relative_to(Path(drive_root).expanduser().resolve())
        except ValueError as exc:
            raise StorageCapacityError("write-capability probe's existing parent is outside the mounted My Drive") from exc
        probe_scope = "nearest_existing_parent_for_not_yet_created_destination"

    name = f".revision-r1-capacity-probe-{uuid.uuid4().hex}.bin"
    probe_path = probe_directory / name
    lookup = metadata_lookup or _lookup_drive_child
    try:
        with probe_path.open("xb") as stream:
            stream.write(DRIVE_WRITE_PROBE_BYTES)
            stream.flush()
            os.fsync(stream.fileno())
        if probe_path.read_bytes() != DRIVE_WRITE_PROBE_BYTES:
            raise StorageCapacityError("mounted Drive write-capability probe failed byte-for-byte read-back")
        metadata = lookup(service, parent_id, name)
        if not isinstance(metadata, dict):
            raise StorageCapacityError("Drive API cannot see the mounted write-capability probe in its exact parent")
        file_owner = _owner_email(metadata)
        parents = [str(value) for value in metadata.get("parents", [])]
        quota_bytes = _optional_nonnegative_int(metadata.get("quotaBytesUsed"), "probe file quotaBytesUsed")
        if (
            metadata.get("name") != name
            or parent_id not in parents
            or metadata.get("mimeType") == DRIVE_FOLDER_MIME
            or file_owner != authenticated_user
            or metadata.get("ownedByMe") is not True
            or quota_bytes is None
            or quota_bytes <= 0
        ):
            raise StorageCapacityError(
                "Drive write-capability probe could not verify exact parent, creator ownership, and positive quotaBytesUsed"
            )
        size = _optional_nonnegative_int(metadata.get("size"), "probe file size")
        account_during_probe = _drive_account_snapshot(service)
        if account_during_probe["authenticated_user"] != authenticated_user:
            raise StorageCapacityError("authenticated Google account changed during the write-capability probe")
        return {
            "status": "PASS",
            "probe_name": name,
            "probe_parent_id": parent_id,
            "new_file_owner": file_owner,
            "new_file_owned_by_authenticated_user": True,
            "new_file_quota_bytes": quota_bytes,
            "new_file_size_bytes": size,
            "quota_usage_during_probe_bytes": account_during_probe["quota_usage_bytes"],
            "quota_limit_during_probe_bytes": account_during_probe["quota_limit_bytes"],
            "probe_scope": probe_scope,
            "probe_completed_unix": time.time(),
        }
    except FileExistsError as exc:
        raise StorageCapacityError("unique write-capability probe name unexpectedly already exists; no existing file was modified") from exc
    except OSError as exc:
        raise StorageCapacityError(f"mounted Drive write-capability probe failed at {probe_path}: {exc}") from exc
    finally:
        if probe_path.exists():
            try:
                probe_path.unlink()
            except OSError as exc:
                raise StorageCapacityError(
                    f"could not remove the exact temporary Drive capacity probe {probe_path}; no other file was touched: {exc}"
                ) from exc


def _verified_write_capability(
    path: str | Path, location: dict[str, Any], authenticated_user: str, service,
    *, drive_root: str | Path, metadata_lookup=None,
) -> dict[str, Any]:
    folder_binding = str(location.get("folder_id") or f"{location.get('parent_id')}:{'/'.join(location.get('unresolved_components', []))}")
    cache_key = (authenticated_user, folder_binding)
    cached = _DRIVE_WRITE_PROBE_CACHE.get(cache_key)
    now = time.time()
    if cached:
        return {**cached, "probe_performed_this_check": False, "probe_cache_age_seconds": max(0.0, now - float(cached["probe_completed_unix"]))}
    result = _write_capability_probe(
        path, location, authenticated_user, service,
        drive_root=drive_root, metadata_lookup=metadata_lookup,
    )
    _DRIVE_WRITE_PROBE_CACHE[cache_key] = dict(result)
    return {**result, "probe_performed_this_check": True, "probe_cache_age_seconds": 0.0}


def inspect_drive_destination(
    path: str | Path, *, api_factory=None, drive_root: str | Path = "/content/drive/MyDrive",
    probe_metadata_lookup=None,
) -> dict[str, Any]:
    """Verify exact My Drive path, Editor write access, creator quota attribution, and capacity."""
    configured, components = _logical_drive_path_components(path, drive_root=drive_root)
    # Resolve folder aliases through the Drive API below; filesystem shortcuts
    # may resolve outside MyDrive's lexical mount path while still targeting a
    # user-owned My Drive folder.
    target = Path(path).expanduser().resolve()
    service = (api_factory or _drive_api_service)()
    account_before = _drive_account_snapshot(service)
    authenticated_user = account_before["authenticated_user"]
    location = resolve_drive_folder(service, components)
    if location.get("drive_id"):
        raise StorageCapacityError(
            "configured destination is inside a shared drive; its organization pooled quota cannot be inferred from the authenticated user's My Drive quota"
        )
    if location.get("can_add_children") is not True:
        raise StorageCapacityError("authenticated user lacks verified canAddChildren permission at the production destination")
    probe = _verified_write_capability(
        path, location, authenticated_user, service,
        drive_root=drive_root, metadata_lookup=probe_metadata_lookup,
    )
    account_after = _drive_account_snapshot(service)
    if account_after["authenticated_user"] != authenticated_user:
        raise StorageCapacityError("authenticated Google account changed while verifying the Drive write probe")
    usage_before = account_before.get("quota_usage_bytes")
    usage_during_probe = (
        probe.get("quota_usage_during_probe_bytes")
        if probe.get("probe_performed_this_check") else None
    )
    usage_delta = (
        usage_during_probe - usage_before
        if usage_before is not None and usage_during_probe is not None else None
    )
    return {
        "path": str(target),
        "configured_path": str(configured),
        "path_components": list(components),
        "folder_id": location["folder_id"],
        "parent_id": location["parent_id"],
        "path_exists": location["path_exists"],
        "traversed_shortcut": location["traversed_shortcut"],
        "drive_id": location["drive_id"],
        "authenticated_user": authenticated_user,
        "quota_account_email": authenticated_user,
        "destination_owner": location.get("owner_email"),
        "destination_owned_by_authenticated_user": location.get("owner_email") == authenticated_user,
        "new_file_owner": probe["new_file_owner"],
        "new_file_owned_by_authenticated_user": probe["new_file_owned_by_authenticated_user"],
        "new_file_owner_verified": probe["new_file_owned_by_authenticated_user"],
        "new_file_quota_bytes": probe["new_file_quota_bytes"],
        "new_file_size_bytes": probe.get("new_file_size_bytes"),
        "write_capability_verified": True,
        "write_probe": probe,
        "can_add_children": True,
        "quota_limit_bytes": account_after["quota_limit_bytes"],
        "quota_usage_bytes": account_after["quota_usage_bytes"],
        "usage_in_drive_bytes": account_after["usage_in_drive_bytes"],
        "usage_in_drive_trash_bytes": account_after["usage_in_drive_trash_bytes"],
        "available_bytes": account_after["available_bytes"],
        "quota_status": account_after["quota_status"],
        "quota_metadata_verified": account_after["quota_status"] == "FINITE_VERIFIED",
        "quota_capacity_note": account_after["quota_capacity_note"],
        "quota_usage_before_probe_bytes": usage_before,
        "quota_usage_during_probe_bytes": usage_during_probe,
        "quota_usage_after_probe_cleanup_bytes": account_after.get("quota_usage_bytes"),
        "quota_usage_change_observed_bytes": usage_delta,
        "quota_change_observation": (
            "INCREASE_OBSERVED" if usage_delta is not None and usage_delta > 0
            else "NO_POSITIVE_CHANGE_OBSERVED_OR_API_EVENTUAL_CONSISTENCY"
            if usage_delta is not None
            else "NOT_MEASURED_CACHED_PROOF" if not probe.get("probe_performed_this_check")
            else "UNAVAILABLE"
        ),
        "quota_scope": "authenticated user's Drive API storageQuota; organization pooling semantics are not exposed here",
    }


def _dataset_cache_upper_bytes() -> int:
    """Bound two local NPY caches at float64 features and int64 labels."""
    total = 0
    for dataset in DATASET_ORDER:
        for split in ("train", "test"):
            rows, columns = DATASETS[dataset][split]["shape"]
            total += int(rows) * int(columns) * 8 + int(rows) * 8
    return total


def _dataset_cache_upper_bytes_for(dataset: str) -> int:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    total = 0
    for split in ("train", "test"):
        rows, columns = DATASETS[dataset][split]["shape"]
        total += int(rows) * int(columns) * 8 + int(rows) * 8
    return total


def project_storage(
    plan: dict[str, Any],
    *,
    observed_checkpoint_bytes: int,
    observed_output_bytes: int = 0,
    checkpoint_size_is_already_conservative_bound: bool = False,
    checkpoint_reference_provenance: str | None = None,
    safety_reserve_bytes: int = DEFAULT_STORAGE_RESERVE_BYTES,
) -> dict[str, Any]:
    """Conservative byte envelope; the reference is validation evidence, not production measurement."""
    if int(safety_reserve_bytes) != DEFAULT_STORAGE_RESERVE_BYTES:
        raise ValueError(f"production storage reserve is frozen at {DEFAULT_STORAGE_RESERVE_BYTES} bytes")
    run_count = int(plan.get("run_count", -1))
    if run_count != EXPECTED_RUN_COUNT:
        raise ValueError("storage projection requires the frozen 100-run plan")
    observed_checkpoint_bytes = int(observed_checkpoint_bytes)
    observed_output_bytes = max(0, int(observed_output_bytes))
    if observed_checkpoint_bytes <= 0:
        raise StorageCapacityError("no verified checkpoint-size reference; storage capacity cannot be projected")
    checkpoint_upper = (
        observed_checkpoint_bytes
        if checkpoint_size_is_already_conservative_bound
        else math.ceil(observed_checkpoint_bytes * CHECKPOINT_SIZE_MULTIPLIER) + CHECKPOINT_SIZE_FIXED_OVERHEAD_BYTES
    )
    output_per_run = max(OUTPUT_BYTES_FLOOR_PER_RUN, math.ceil(observed_output_bytes * CHECKPOINT_SIZE_MULTIPLIER))
    checkpoint_slots = TASK_BOUNDARY_GENERATIONS_PER_RUN + RECOVERY_GENERATIONS_PER_RUN
    per_run_retained_checkpoint_bytes = checkpoint_slots * checkpoint_upper
    per_run_drive_retained = per_run_retained_checkpoint_bytes + output_per_run
    checkpoint_atomic = checkpoint_upper
    output_atomic = output_per_run
    safety = DEFAULT_STORAGE_RESERVE_BYTES
    first_run_drive_without_reserve = per_run_drive_retained + checkpoint_atomic + output_atomic
    matrix_drive_retained = run_count * per_run_drive_retained
    matrix_drive_without_reserve = matrix_drive_retained + checkpoint_atomic + output_atomic
    cache_upper = _dataset_cache_upper_bytes()
    first_run_local_without_reserve = cache_upper + per_run_retained_checkpoint_bytes + checkpoint_atomic
    next_run_local_without_reserve = per_run_retained_checkpoint_bytes + checkpoint_atomic
    local_matrix_retained = cache_upper + run_count * per_run_retained_checkpoint_bytes
    local_matrix_peak = cache_upper + per_run_retained_checkpoint_bytes + checkpoint_atomic + safety
    return {
        "status": "ESTIMATE_BOUND_TO_VALIDATION_REFERENCE_NOT_PRODUCTION_MEASUREMENT",
        "run_count": run_count,
        "checkpoint_size_provenance": checkpoint_reference_provenance or (
            "v6.2 GPU synthetic recovery reference; operational size proxy only, not production measurement"
            if checkpoint_size_is_already_conservative_bound
            else "observed payload reference; caller-verified provenance is reported separately"
        ),
        "observed_checkpoint_bytes": observed_checkpoint_bytes,
        "observed_output_bytes": observed_output_bytes,
        "per_run_output_envelope_bytes": output_per_run,
        "checkpoint_payload_upper_bytes": checkpoint_upper,
        "safety_reserve_bytes": safety,
        "checkpoint_layout": {
            "task_boundary_generations_per_run": TASK_BOUNDARY_GENERATIONS_PER_RUN,
            "recovery_generations_per_run": RECOVERY_GENERATIONS_PER_RUN,
            "total_retained_generation_slots_per_run": checkpoint_slots,
            "task_boundary_payloads_pruned": False,
            "recovery_generations_pruned": False,
        },
        "accounting": {
            "checkpoint_growth_formula": (
                "use verified v6.2 max_checkpoint_payload_bytes as-is; it already includes the estimator"
                if checkpoint_size_is_already_conservative_bound
                else "ceil(measured production payload * 1.5) + 16 MiB"
            ),
            "checkpoint_size_reference_already_conservative_bound": checkpoint_size_is_already_conservative_bound,
            "output_envelope_formula": "max(256 MiB, ceil(observed output * 1.5))",
            "all_arms_share_checkpoint_byte_envelope_only": True,
            "runtime_extrapolated_from_full_hybrid": False,
            "local_matrix_policy": "local stage mirrors Drive; runtime reset between runs may discard only ephemeral local copies after Drive verification",
            "production_checkpoint_deletion_or_retention_change": False,
        },
        "checkpoint_write_volume": {
            "matrix_retained_payload_lower_bound_bytes": run_count * per_run_retained_checkpoint_bytes,
            "actual_cumulative_upload_bytes": None,
            "actual_cumulative_upload_status": "MEASURE_DURING_RUNS; phase-boundary and periodic recovery synchronization rewrites are additional",
        },
        "drive": {
            "first_run_retained_bytes": per_run_drive_retained,
            "first_run_temporary_checkpoint_headroom_bytes": checkpoint_atomic,
            "first_run_temporary_output_headroom_bytes": output_atomic,
            "first_run_required_without_reserve_bytes": first_run_drive_without_reserve,
            "first_run_peak_with_reserve_bytes": first_run_drive_without_reserve + safety,
            "matrix_retained_bytes": matrix_drive_retained,
            "matrix_temporary_checkpoint_headroom_bytes": checkpoint_atomic,
            "matrix_temporary_output_headroom_bytes": output_atomic,
            "matrix_required_without_reserve_bytes": matrix_drive_without_reserve,
            "matrix_peak_with_reserve_bytes": matrix_drive_without_reserve + safety,
        },
        "local": {
            "dataset_cache_upper_bytes": cache_upper,
            "first_run_retained_bytes": cache_upper + per_run_retained_checkpoint_bytes,
            "first_run_temporary_checkpoint_headroom_bytes": checkpoint_atomic,
            "first_run_required_without_reserve_bytes": first_run_local_without_reserve,
            "first_run_peak_with_reserve_bytes": first_run_local_without_reserve + safety,
            "next_run_retained_bytes": per_run_retained_checkpoint_bytes,
            "next_run_temporary_checkpoint_headroom_bytes": checkpoint_atomic,
            "next_run_required_without_reserve_bytes": next_run_local_without_reserve,
            "next_run_peak_with_reserve_bytes": next_run_local_without_reserve + safety,
            "matrix_retained_bytes_if_one_runtime_never_resets": local_matrix_retained,
            "matrix_peak_bytes": local_matrix_peak,
            "matrix_execution_is_segmented_across_runtime_resets": True,
            "runtime_reset_effect": "ephemeral local staging is lost at runtime reset; verified Drive checkpoints remain",
        },
    }


def require_run_capacity(
    storage_snapshot: dict[str, Any],
    projection: dict[str, Any],
    *,
    scope: str,
    completed_runs: int = 0,
) -> None:
    """Refuse first/matrix execution unless Drive, local space, and reserve all fit."""
    if projection.get("safety_reserve_bytes") != DEFAULT_STORAGE_RESERVE_BYTES:
        raise StorageCapacityError("storage projection does not preserve the required 512 MiB safety reserve")
    if scope == "first_run":
        drive_key = "first_run_required_without_reserve_bytes"
        local_key = "first_run_required_without_reserve_bytes"
        drive_projection = projection.get("drive", {})
        local_projection = projection.get("local", {})
        description = "first_run"
    elif scope == "matrix":
        drive_key = "matrix_required_without_reserve_bytes"
        local_key = "next_run_required_without_reserve_bytes"
        drive_projection = projection.get("drive", {})
        local_projection = projection.get("local", {})
        description = "matrix (local capacity is checked one complete run at a time)"
        if storage_snapshot.get("production_measurements_validated") is not True:
            raise StorageCapacityError(
                "matrix capacity gate requires validated first-run production checkpoint/output measurements; no matrix run was launched"
            )
        if not 0 <= int(completed_runs) <= EXPECTED_RUN_COUNT:
            raise ValueError("completed run count is outside the frozen matrix")
    else:
        raise ValueError("capacity scope must be 'first_run' or 'matrix'")
    reserve = DEFAULT_STORAGE_RESERVE_BYTES
    drive_remaining_required = None
    if scope == "matrix":
        remaining_runs = EXPECTED_RUN_COUNT - int(completed_runs)
        drive_remaining_required = (
            remaining_runs * int(projection["drive"]["first_run_retained_bytes"])
            + int(projection["drive"]["matrix_temporary_checkpoint_headroom_bytes"])
            + int(projection["drive"]["matrix_temporary_output_headroom_bytes"])
        )
    local_cache_missing = max(
        0,
        int(projection["local"]["dataset_cache_upper_bytes"])
        - max(0, int(storage_snapshot.get("local_existing_dataset_cache_bytes", 0))),
    )
    current_drive_bytes = max(0, int(storage_snapshot.get("current_run_existing_drive_bytes", 0)))
    current_local_bytes = max(0, int(storage_snapshot.get("current_run_existing_local_checkpoint_bytes", 0)))
    for resource, snapshot_key, projection_values, projection_key in (
        ("Drive", "drive_available_bytes", drive_projection, drive_key),
        ("local Colab disk", "local_available_bytes", local_projection, local_key),
    ):
        available = storage_snapshot.get(snapshot_key)
        required = (
            max(0, drive_remaining_required - current_drive_bytes)
            if resource == "Drive" and drive_remaining_required is not None
            else projection_values.get(projection_key)
        )
        if resource == "Drive" and scope == "first_run":
            required = max(0, int(required) - current_drive_bytes)
        if resource == "local Colab disk":
            if scope == "first_run":
                required = max(
                    0, int(required)
                    - max(0, int(storage_snapshot.get("local_existing_dataset_cache_bytes", 0)))
                    - current_local_bytes,
                )
            else:
                required = max(0, int(required) + local_cache_missing - current_local_bytes)
        if available is None or required is None:
            raise StorageCapacityError(f"cannot establish {resource} capacity for {description}; no run was launched")
        try:
            require_available_bytes(
                storage_snapshot.get(f"{resource.lower().replace(' ', '_')}_path", "production destination"),
                required_bytes=int(required), reserve_bytes=reserve,
                available_bytes=int(available), storage_kind=resource,
            )
        except StorageCapacityError as exc:
            raise StorageCapacityError(f"{description} capacity gate stopped before writes: {exc}") from exc


def run_output_root(
    output_root: str | Path, record: dict[str, Any], *, test_only: bool = False,
    runner_version: str = "revision_R1.0.2",
) -> Path:
    """Return a run namespace bound to code, dataset, seed, arm, and config."""
    source = record.get("training_source_sha256")
    config_hash = record.get("config_sha256")
    run_key = str(record.get("run_key", ""))
    if not _is_sha256(source) or not _is_sha256(config_hash) or not re.fullmatch(r"[a-z0-9_]+", run_key):
        raise ValueError("run record is missing safe source/config/run-key identities")
    if not re.fullmatch(r"revision_R1\.\d+\.\d+", runner_version):
        raise ValueError("run namespace requires a safe runner-version identity")
    prefix = Path("SYNTHETIC_NONREPORTABLE_TEST_ONLY") if test_only else Path()
    return (
        Path(output_root).expanduser().resolve() / prefix / "runs" / runner_version
        / source[:16] / str(record["dataset"]) / f"seed_{int(record['seed'])}"
        / str(record["arm"]) / config_hash[:16]
    )


def run_checkpoint_root(
    checkpoint_root: str | Path, record: dict[str, Any], *, test_only: bool = False,
    runner_version: str = "revision_R1.0.2",
) -> Path:
    source = record.get("training_source_sha256")
    config_hash = record.get("config_sha256")
    run_key = str(record.get("run_key", ""))
    if not _is_sha256(source) or not _is_sha256(config_hash) or not re.fullmatch(r"[a-z0-9_]+", run_key):
        raise ValueError("run record is missing safe source/config/run-key identities")
    if not re.fullmatch(r"revision_R1\.\d+\.\d+", runner_version):
        raise ValueError("checkpoint namespace requires a safe runner-version identity")
    prefix = Path("SYNTHETIC_NONREPORTABLE_TEST_ONLY") if test_only else Path()
    return (
        Path(checkpoint_root).expanduser().resolve() / prefix / runner_version
        / source[:16] / str(record["dataset"]) / f"seed_{int(record['seed'])}"
        / str(record["arm"]) / config_hash[:16]
    )


def _utc_iso(unix_time: float | None) -> str | None:
    if unix_time is None:
        return None
    return datetime.fromtimestamp(float(unix_time), tz=timezone.utc).isoformat()


def inspect_lease_readonly(
    path: str | Path, *, now_unix: float | None = None,
    current_host: str | None = None, current_pid: int | None = None,
) -> dict[str, Any]:
    """Read and classify one lease without changing or recovering it."""
    target = Path(path).expanduser().resolve()
    now = time.time() if now_unix is None else float(now_unix)
    host = socket.gethostname() if current_host is None else str(current_host)
    pid = os.getpid() if current_pid is None else int(current_pid)
    report: dict[str, Any] = {
        "path": str(target), "status": "missing", "raw_text": None, "record": None,
        "modified_unix": None, "modified_utc": None, "now_unix": now,
        "now_utc": _utc_iso(now), "owner_process_visibility": "unknown",
        "recovery_may_be_attempted_by_existing_lease_logic": False,
    }
    try:
        raw = target.read_text(encoding="utf-8")
        stat = target.stat()
    except FileNotFoundError:
        return report
    except OSError as exc:
        report.update(status="unreadable", read_error=f"{type(exc).__name__}: {exc}")
        return report
    report.update(raw_text=raw, modified_unix=stat.st_mtime, modified_utc=_utc_iso(stat.st_mtime))
    try:
        record = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        report.update(status="malformed", parse_error=f"{type(exc).__name__}: {exc}")
        return report
    if not isinstance(record, dict):
        report.update(status="malformed", parse_error="lease JSON root is not an object")
        return report
    report["record"] = record
    for field in ("created_at", "heartbeat_at", "expires_at"):
        raw_value = record.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            report[f"{field}_unix"] = value
            report[f"{field}_utc"] = _utc_iso(value)
    try:
        expiry = float(record["expires_at"])
        heartbeat = float(record["heartbeat_at"])
        owner_host = record.get("host")
        owner_pid = int(record["pid"])
        if not math.isfinite(expiry) or not math.isfinite(heartbeat):
            raise ValueError("non-finite heartbeat/expiry")
    except (KeyError, TypeError, ValueError, OverflowError):
        report.update(status="malformed", parse_error="lease lacks finite heartbeat/expiry or valid PID")
        return report
    if owner_host == host:
        try:
            os.kill(owner_pid, 0)
            owner_process = "same_host_pid_alive"
        except ProcessLookupError:
            owner_process = "same_host_pid_not_found"
        except PermissionError:
            owner_process = "same_host_pid_exists_permission_denied"
        except OSError as exc:
            owner_process = f"same_host_pid_probe_error:{type(exc).__name__}"
    else:
        owner_process = "different_runtime_unverifiable"
    report["owner_process_visibility"] = owner_process
    if expiry > now:
        report["status"] = "live"
    elif owner_process in {"same_host_pid_alive", "same_host_pid_exists_permission_denied"}:
        report["status"] = "expired_but_owner_process_may_be_alive"
    else:
        report["status"] = "expired"
        report["recovery_may_be_attempted_by_existing_lease_logic"] = True
    return report


def _diagnostic_run_identity(
    record: dict[str, Any], preflight: dict[str, Any], *, runner_version: str = "revision_R1.0.2",
) -> dict[str, Any]:
    dataset = str(record["dataset"])
    train = preflight[dataset]["train"]
    test = preflight[dataset]["test"]
    source_hash = str(record["training_source_sha256"])
    config_hash = str(record["config_sha256"])
    return {
        "runner_version": runner_version, "config_sha256": config_hash,
        "dataset": {
            "namespace": "PRODUCTION", "dataset": dataset,
            "train_sha256": train["sha256"], "test_sha256": test["sha256"],
            "train_md5": train["md5"], "test_md5": test["md5"],
        },
        "code_sha256": source_hash, "schedule_sha256": str(record["schedule_sha256"]),
        "execution_scope": "production",
        "run_id": _run_id_for(record, synthetic=False),
    }


def _read_json_diagnostic(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "exists": path.is_file(), "mtime_unix": None, "record": None, "error": None}
    if not result["exists"]:
        return result
    try:
        result["mtime_unix"] = path.stat().st_mtime
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON root is not an object")
        result["record"] = value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _hash_file_stream(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_recovery_checkpoint_directory(
    directory: Path, expected_identity: dict[str, Any], *, max_candidates: int = 2,
    progress_callback=None,
) -> dict[str, Any]:
    manifests = []
    unreadable_manifests = []
    try:
        for path in directory.glob("generation-*.manifest.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                unreadable_manifests.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if isinstance(value, dict) and isinstance(value.get("generation"), int):
                manifests.append((int(value["generation"]), path, value))
    except OSError as exc:
        return {"directory": str(directory), "error": f"{type(exc).__name__}: {exc}", "candidates": []}
    manifests.sort(key=lambda item: item[0], reverse=True)
    candidates = []
    for generation, manifest_path, manifest in manifests[:max_candidates]:
        payload_name = manifest.get("payload_file")
        safe = isinstance(payload_name, str) and re.fullmatch(r"generation-\d+\.pt", payload_name)
        payload = directory / payload_name if safe else None
        row = {
            "generation": generation, "manifest_path": str(manifest_path),
            "manifest_sha256": _hash_file_stream(manifest_path),
            "manifest_schema_valid": manifest.get("schema_version") == 1,
            "manifest_generation_matches_filename": manifest_path.name == f"generation-{generation:06d}.manifest.json",
            "manifest_identity_digest_matches": manifest.get("identity_sha256") == hashlib.sha256(_canonical_bytes(expected_identity)).hexdigest(),
            "manifest_identity_matches": manifest.get("identity") == expected_identity,
            "kind": manifest.get("kind"), "created_unix": manifest.get("created_unix"),
            "created_utc": _utc_iso(manifest.get("created_unix")) if isinstance(manifest.get("created_unix"), (int, float)) else None,
            "payload_path": None if payload is None else str(payload),
            "payload_exists": bool(payload and payload.is_file() and not payload.is_symlink()),
            "payload_bytes_expected": manifest.get("payload_bytes"),
            "payload_bytes_observed": payload.stat().st_size if payload and payload.is_file() else None,
            "payload_sha256_expected": manifest.get("payload_sha256"),
            "payload_sha256_observed": None, "payload_hash_matches": False,
        }
        if progress_callback is not None:
            progress_callback({"stage": "checkpoint_hash_start", "path": row["payload_path"], "generation": generation})
        if (
            row["manifest_identity_matches"] and payload is not None and payload.is_file()
            and payload.stat().st_size == manifest.get("payload_bytes")
            and isinstance(manifest.get("payload_sha256"), str)
        ):
            try:
                row["payload_sha256_observed"] = _hash_file_stream(payload)
                row["payload_hash_matches"] = row["payload_sha256_observed"] == manifest["payload_sha256"]
            except OSError as exc:
                row["payload_hash_error"] = f"{type(exc).__name__}: {exc}"
        fully_valid = all(row[key] for key in (
            "manifest_schema_valid", "manifest_generation_matches_filename",
            "manifest_identity_digest_matches", "manifest_identity_matches",
            "payload_exists", "payload_hash_matches",
        ))
        row["integrity_status"] = "PASS_MANIFEST_AND_PAYLOAD_HASH" if fully_valid else "FAIL_OR_UNVERIFIED"
        candidates.append(row)
        if progress_callback is not None:
            progress_callback({"stage": "checkpoint_hash_complete", "generation": generation, "status": row["integrity_status"]})
    return {
        "directory": str(directory), "manifest_generation_count": len(manifests),
        "unreadable_manifests": unreadable_manifests,
        "integrity_method": "manifest identity plus bounded 8 MiB streaming SHA-256; payload is not deserialized",
        "candidates": candidates,
    }


def _first_run_resume_gate(recovery: dict[str, Any], expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Summarize exact v1.3 recovery evidence without loading its payload object."""
    candidates = recovery.get("candidates", [])
    wrong_identity = [row for row in candidates if row.get("manifest_identity_matches") is not True]
    verified = [row for row in candidates
                if row.get("manifest_identity_matches") is True
                and row.get("integrity_status") == "PASS_MANIFEST_AND_PAYLOAD_HASH"]
    if recovery.get("error"):
        status, reason = "BLOCKED_UNREADABLE", str(recovery["error"])
    elif recovery.get("manifest_generation_count", 0) > 2:
        status, reason = "BLOCKED_UNEXPECTED_GENERATION_COUNT", "recovery namespace exceeds the configured two-generation retention"
    elif wrong_identity:
        status, reason = "BLOCKED_IDENTITY_MISMATCH", "a recovery manifest in the immutable namespace has a different run identity"
    elif not verified:
        status, reason = "BLOCKED_NO_VALID_RECOVERY", "no exact-identity generation has a verified payload hash and size"
    else:
        status, reason = "PASS_RESUME_CANDIDATE", "at least one exact-identity checkpoint generation passed manifest and payload hash/size checks; runner deserialization remains required"
    newest_manifest_generation = max((int(row["generation"]) for row in candidates), default=None)
    selected = verified[0] if verified else None
    return {
        "status": status, "reason": reason,
        "automatic_fresh_start_allowed": False,
        "expected_identity": expected_identity,
        "verified_candidates": verified,
        "newest_manifest_generation": newest_manifest_generation,
        "selected_integrity_verified_candidate": selected,
        "runner_deserialization_pending": selected is not None,
        "fallback_from_newest_integrity_verified_candidate": bool(
            selected is not None and newest_manifest_generation is not None
            and int(selected["generation"]) != newest_manifest_generation
        ),
        "checkpoint_cursor_note": "manifest/hash preflight does not deserialize the multi-gigabyte payload; the exact cursor and restored counters come from the runner load. Last progress cursor/counters are informational and may be newer than the last verified Drive snapshot.",
        "manifest_generation_count": recovery.get("manifest_generation_count", 0),
        "recovery_directory": recovery.get("directory"),
        "integrity_method": recovery.get("integrity_method"),
    }


def inspect_first_run_resume_checkpoint(
    plan: dict[str, Any], *, preflight: dict[str, Any], checkpoint_root: str | Path,
    progress_callback=None,
) -> dict[str, Any]:
    """Read-only strict gate for resuming the current v1.3 first run.

    This streams at most the two retained payload hashes. It intentionally does
    not deserialize either checkpoint; the runner remains authoritative for
    loading and fallback at the start of training.
    """
    first = plan["runs"][0]
    if (
        plan.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or plan.get("training_source_sha256") != first.get("training_source_sha256")
        or first.get("run_key") != "ember_seed42_full_hybrid"
        or int(first.get("run_index", 0)) != 1
    ):
        raise ValueError("first-run resume gate received a plan with a different frozen identity")
    expected_identity = _diagnostic_run_identity(first, preflight, runner_version="revision_R1.0.2")
    recovery_dir = run_checkpoint_root(checkpoint_root, first, runner_version="revision_R1.0.2") / "recovery"
    inspection = _inspect_recovery_checkpoint_directory(
        recovery_dir, expected_identity, max_candidates=2, progress_callback=progress_callback,
    )
    return _first_run_resume_gate(inspection, expected_identity)


def _stage_required_resume_manifest(
    drive_recovery: str | Path, local_recovery: str | Path, *,
    expected_identity: dict[str, Any], generation: int, manifest_sha256: str,
) -> Path:
    """Place a tiny local fail-closed marker before Drive-to-local restore.

    The copied manifest is not a checkpoint. If Drive mirroring fails or its
    payload becomes unavailable, the runner sees an existing generation
    manifest and raises instead of interpreting an empty local directory as a
    request for a fresh run.
    """
    source = Path(drive_recovery) / f"generation-{int(generation):06d}.manifest.json"
    target_root = Path(local_recovery)
    target = target_root / source.name
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"required Drive resume manifest became unreadable: {source}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise RuntimeError("required Drive resume manifest changed after the pre-resume identity/hash check")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("required Drive resume manifest became malformed after pre-resume validation") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("generation") != int(generation)
        or manifest.get("identity") != expected_identity
        or manifest.get("identity_sha256") != hashlib.sha256(_canonical_bytes(expected_identity)).hexdigest()
        or manifest.get("payload_file") != f"generation-{int(generation):06d}.pt"
        or type(manifest.get("payload_bytes")) is not int or manifest["payload_bytes"] <= 0
        or not _is_sha256(manifest.get("payload_sha256"))
    ):
        raise RuntimeError("required Drive resume manifest no longer matches the frozen run identity")
    source_payload = Path(drive_recovery) / manifest["payload_file"]
    if not source_payload.is_file() or source_payload.is_symlink() or source_payload.stat().st_size != manifest["payload_bytes"]:
        raise RuntimeError("required Drive resume payload is missing or has an unexpected size")
    target_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw:
            raise RuntimeError(f"local resume guard conflicts with another manifest: {target}")
        return target
    try:
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if target.read_bytes() != raw:
            raise RuntimeError(f"local resume guard conflicts with another manifest: {target}")
    return target


def inspect_first_run_lease_and_progress(
    plan: dict[str, Any], *, preflight: dict[str, Any], output_root: str | Path,
    checkpoint_root: str | Path, legacy_first_run_record: dict[str, Any],
    legacy_build_identity: dict[str, Any], current_runtime: dict[str, Any] | None = None,
    now_unix: float | None = None, progress_callback=None,
) -> dict[str, Any]:
    """Read-only diagnosis for the shared matrix lock and v1.2/current first-run namespaces."""
    if plan.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("lease diagnostic plan schedule differs from the approved effective schedule")
    current_record = plan["runs"][0]
    if current_record.get("run_key") != "ember_seed42_full_hybrid":
        raise ValueError("lease diagnostic expected the frozen first run")
    legacy = dict(legacy_first_run_record)
    if (
        legacy.get("run_key") != current_record.get("run_key")
        or legacy.get("config_sha256") != current_record.get("config_sha256")
        or legacy.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or legacy.get("training_source_sha256") != legacy_build_identity.get("training_source_sha256")
    ):
        raise ValueError("authenticated v1.2 first-run identity does not match current config/schedule")
    output = Path(output_root).expanduser().resolve()
    checkpoints = Path(checkpoint_root).expanduser().resolve()
    matrix_lock = output / ".matrix_execution.lease.json"
    legacy_matrix_registry_path = _matrix_registry_path(output)
    current_matrix_registry_path = _matrix_registry_path(output, plan)
    rows = []
    for label, record in (("v1_2_uploaded_identity", legacy), ("current_package_identity", current_record)):
        identity = _diagnostic_run_identity(
            record, preflight,
            runner_version="revision_R1.0.1" if label == "v1_2_uploaded_identity" else "revision_R1.0.2",
        )
        run_output_base = output
        checkpoint_base = checkpoints
        namespace_version = "revision_R1.0.1" if label == "v1_2_uploaded_identity" else "revision_R1.0.2"
        run_output = run_output_root(run_output_base, record, runner_version=namespace_version)
        run_checkpoints = run_checkpoint_root(checkpoint_base, record, runner_version=namespace_version)
        run_id = identity["run_id"]
        if label == "v1_2_uploaded_identity":
            # v1.2 used flat progress and matrix-registry paths; preserve and
            # inspect those exact historical files without migrating them.
            progress_path = output / "progress" / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
            prep_progress = output / "preparation_progress" / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
        else:
            plan_namespace = str(plan["plan_sha256"])
            progress_path = output / "progress" / plan_namespace / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
            prep_progress = output / "preparation_progress" / plan_namespace / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
        recovery_inspection = _inspect_recovery_checkpoint_directory(
            run_checkpoint_root(checkpoints, record, runner_version=namespace_version) / "recovery", identity,
            progress_callback=progress_callback,
        )
        resume_gate = _first_run_resume_gate(recovery_inspection, identity) if label == "current_package_identity" else None
        rows.append({
            "identity_label": label,
            "run_key": record["run_key"], "config_sha256": record["config_sha256"],
            "training_source_sha256": record["training_source_sha256"],
            "schedule_sha256": record["schedule_sha256"],
            "training_source_matches_current_package": record["training_source_sha256"] == plan["training_source_sha256"],
            "automatic_resume_compatibility": "IDENTITY_MATCH_REQUIRED; old checkpoints are not compatible when source hashes differ",
            "run_output_root": str(run_output), "run_lease": inspect_lease_readonly(run_output / ".run.lease.json", now_unix=now_unix),
            "run_status_json": _read_json_diagnostic(run_output / "status.json"),
            "run_registry_status": _read_json_diagnostic(run_output / "registry" / "runs" / run_id / "registry.json"),
            "progress": _read_json_diagnostic(progress_path),
            "preparation_progress": _read_json_diagnostic(prep_progress),
            "recovery_checkpoints": recovery_inspection,
            "first_run_resume_gate": resume_gate,
            "task_boundary_checkpoint_directory": str(run_checkpoints / "task_boundaries"),
            "task_boundary_manifest_count": sum(1 for _ in (run_checkpoints / "task_boundaries").glob("generation-*.manifest.json")) if (run_checkpoints / "task_boundaries").is_dir() else 0,
        })
    def read_registry(path: Path, run_key: str) -> dict[str, Any] | None:
        try:
            matrix_registry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            return {"path": str(path), "unreadable": f"{type(exc).__name__}: {exc}"}
        if isinstance(matrix_registry, dict) and isinstance(matrix_registry.get("completed_runs"), dict):
            return {
                "path": str(path),
                "completed_run_count": len(matrix_registry["completed_runs"]),
                "first_run_record": matrix_registry["completed_runs"].get(run_key),
                "plan_sha256": matrix_registry.get("plan_sha256"),
                "training_source_sha256": matrix_registry.get("training_source_sha256"),
                "raw_identity_source": "matrix completion registry JSON; not modified",
            }
        return {"path": str(path), "record": matrix_registry, "schema_status": "UNRECOGNIZED_OR_MALFORMED"}

    legacy_matrix_registry = read_registry(legacy_matrix_registry_path, current_record["run_key"])
    matrix_registry = read_registry(current_matrix_registry_path, current_record["run_key"])
    return {
        "schema_version": 1, "diagnostic_mode": "READ_ONLY_NO_LOCK_RECOVERY_NO_TRAINING",
        "created_utc": _utc_iso(time.time() if now_unix is None else now_unix),
        "current_runtime": current_runtime or {},
        "matrix_lease": inspect_lease_readonly(matrix_lock, now_unix=now_unix),
        "matrix_registry": matrix_registry,
        "legacy_v1_2_matrix_registry": legacy_matrix_registry,
        "matrix_registry_paths": {
            "legacy_v1_2": str(legacy_matrix_registry_path),
            "current_package": str(current_matrix_registry_path),
        },
        "run_namespaces": rows,
        "remote_colab_process_visibility": "UNAVAILABLE_FROM_THIS_RUNTIME; inspect Colab Manage Sessions and other open runtimes",
        "historical_oom": {
            "evidence": "Colab cgroup reported OOM kill for PID 44230 (python3), anon-rss 9534112 kB; kernel also said session crashed after using all available RAM",
            "exact_allocation_and_training_phase": "UNKNOWN_FROM_AVAILABLE_EVIDENCE",
            "later_lease_error_proves_prior_worker_never_started": False,
        },
        "safe_recovery_rule": "Do not delete locks. Only the existing RunLease.acquire expiry/guard recovery may act after confirming no other Colab worker; an old-VM PID alone is not proof.",
    }


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw_temp, path)
    finally:
        if os.path.exists(raw_temp):
            os.unlink(raw_temp)


def _append_progress_jsonl(path: Path, document: dict[str, Any]) -> None:
    """Append one compact, identity-bound event without replacing prior attempts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()


def _bundle_identity(bundle) -> dict[str, str]:
    return {
        "namespace": "PRODUCTION", "dataset": bundle.dataset,
        "train_sha256": str(bundle.train_metadata["sha256"]),
        "test_sha256": str(bundle.test_metadata["sha256"]),
        "train_md5": str(bundle.train_metadata["md5"]),
        "test_md5": str(bundle.test_metadata["md5"]),
    }


def load_validated_dataset_bundle(
    data_root: str | Path, dataset: str, cache_root: str | Path, preflight: dict[str, Any],
    *, progress_callback=None,
):
    """Load once through the existing MD5/SHA-256-verified NPZ/cache path."""
    from revision_r1.data import preflight_dataset

    bundle = preflight_dataset(
        data_root, dataset, cache_dir=cache_root, progress_callback=progress_callback,
    )
    from revision_r1.data import require_memory_mapped_production_bundle

    require_memory_mapped_production_bundle(bundle)
    observed = _bundle_identity(bundle)
    expected = {"namespace": "PRODUCTION", "dataset": dataset, **_preflight_identity(preflight, dataset)}
    if observed != expected:
        raise ValueError(f"{dataset} NPZ identity differs from the mounted-Drive preflight record")
    spec = DATASETS[dataset]
    if (
        bundle.feature_dim != int(spec["feature_dim"])
        or list(bundle.train_metadata.get("shape", [])) != list(spec["train"]["shape"])
        or list(bundle.test_metadata.get("shape", [])) != list(spec["test"]["shape"])
    ):
        raise ValueError(f"{dataset} loaded feature shape differs from the frozen dataset reference")
    return bundle


class _SequentialDatasetBundleCache:
    """Retain only the current dataset bundle across the ordered run matrix."""

    def __init__(self, loader):
        self._loader = loader
        self._dataset: str | None = None
        self._bundle = None

    def get(self, dataset: str):
        if self._dataset != dataset:
            bundle = self._loader(dataset)
            self._dataset = dataset
            self._bundle = bundle
        return self._bundle


def _bundle_cache_bytes(bundle, cache_root: str | Path) -> int:
    from revision_r1.data import _cache_paths

    total = 0
    for metadata in (bundle.train_metadata, bundle.test_metadata):
        paths = _cache_paths(
            Path(cache_root).expanduser().resolve(), metadata["md5"],
            metadata["x_key"], metadata["y_key"],
        )
        if not all(path.is_file() and not path.is_symlink() for path in paths):
            raise ValueError("checksum-verified dataset cache is incomplete after preflight load")
        total += sum(path.stat().st_size for path in paths)
    return total


def _nearest_existing_directory(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        raise StorageCapacityError(f"no existing local directory is available for capacity inspection: {path}")
    return candidate


def _preflight_addressed_cache_bytes(
    preflight: dict[str, Any], cache_root: str | Path, *, dataset: str | None = None,
) -> int:
    """Count preflight-named cache files; actual loader still verifies their SHA-256."""
    from revision_r1.data import _cache_paths

    root = Path(cache_root).expanduser().resolve()
    total = 0
    datasets = (dataset,) if dataset is not None else DATASET_ORDER
    for dataset_name in datasets:
        for split in ("train", "test"):
            metadata = preflight.get(dataset_name, {}).get(split, {})
            if not all(metadata.get(key) for key in ("md5", "x_key", "y_key")):
                continue
            paths = _cache_paths(root, metadata["md5"], metadata["x_key"], metadata["y_key"])
            if not all(path.is_file() and not path.is_symlink() for path in paths):
                continue
            try:
                manifest = json.loads(paths[2].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("archive_md5") != metadata["md5"]
                or manifest.get("x_key") != metadata["x_key"]
                or manifest.get("y_key") != metadata["y_key"]
            ):
                continue
            total += sum(path.stat().st_size for path in paths)
    return total


def inspect_production_capacity(
    output_path: str | Path, checkpoint_path: str | Path, local_path: str | Path,
    cache_root: str | Path, *, api_factory=None,
    current_run_output_path: str | Path | None = None,
    current_run_checkpoint_path: str | Path | None = None,
    current_run_local_checkpoint_path: str | Path | None = None,
    expected_run_identity: dict[str, Any] | None = None,
    verified_dataset_cache_bytes: int = 0,
) -> dict[str, Any]:
    """Inspect exact Drive destinations/account quota and current local free space."""
    output_destination = inspect_drive_destination(output_path, api_factory=api_factory)
    checkpoint_destination = inspect_drive_destination(checkpoint_path, api_factory=api_factory)
    same_account = output_destination["quota_account_email"] == checkpoint_destination["quota_account_email"]
    if not same_account:
        raise StorageCapacityError("output and checkpoint roots were inspected under different authenticated quota accounts")
    local_directory = _nearest_existing_directory(local_path)
    local_usage = shutil.disk_usage(local_directory)
    current_drive_bytes = 0
    if current_run_output_path is not None:
        current_drive_bytes += _tree_file_bytes(current_run_output_path)
    if current_run_checkpoint_path is not None:
        current_drive_bytes += _tree_file_bytes(current_run_checkpoint_path, expected_identity=expected_run_identity)
    current_local_bytes = (
        _tree_file_bytes(current_run_local_checkpoint_path, expected_identity=expected_run_identity)
        if current_run_local_checkpoint_path is not None else 0
    )
    output_available = output_destination.get("available_bytes")
    checkpoint_available = checkpoint_destination.get("available_bytes")
    drive_available = (
        min(output_available, checkpoint_available)
        if output_available is not None and checkpoint_available is not None else None
    )
    return {
        "created_unix": time.time(),
        "drive_available_bytes": drive_available,
        "local_available_bytes": int(local_usage.free),
        "local_existing_dataset_cache_bytes": max(0, int(verified_dataset_cache_bytes)),
        "current_run_existing_drive_bytes": current_drive_bytes,
        "current_run_existing_local_checkpoint_bytes": current_local_bytes,
        "destination_owner_verified": bool(output_destination.get("destination_owner") and checkpoint_destination.get("destination_owner")),
        "write_capability_verified": output_destination.get("write_capability_verified") is True and checkpoint_destination.get("write_capability_verified") is True,
        "new_file_owner_verified": output_destination.get("new_file_owned_by_authenticated_user") is True and checkpoint_destination.get("new_file_owned_by_authenticated_user") is True,
        "quota_metadata_verified": output_destination.get("quota_status") == "FINITE_VERIFIED" and checkpoint_destination.get("quota_status") == "FINITE_VERIFIED",
        "quota_status": "FINITE_VERIFIED" if drive_available is not None else "UNKNOWN_CAPACITY",
        "quota_capacity_note": (
            output_destination.get("quota_capacity_note") or checkpoint_destination.get("quota_capacity_note")
            if drive_available is None else None
        ),
        "authenticated_user": output_destination["authenticated_user"],
        "quota_account_email": output_destination["quota_account_email"],
        "destination_owner": output_destination["destination_owner"],
        "output_destination": output_destination,
        "checkpoint_destination": checkpoint_destination,
        "output_path": str(Path(output_path).expanduser().resolve()),
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "local_path": str(Path(local_path).expanduser().resolve()),
        "local_capacity_path_inspected": str(local_directory),
        "production_measurements_validated": False,
    }


def _tree_file_bytes(path: str | Path, *, expected_identity: dict[str, Any] | None = None) -> int:
    """Count one exact run namespace; validate checkpoint manifest identity before crediting space."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        raise StorageCapacityError(f"run storage namespace is not a regular directory: {root}")
    total = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for filename in filenames:
            item = current / filename
            if item.is_symlink() or not item.is_file():
                continue
            if filename.endswith(".manifest.json") and expected_identity is not None:
                try:
                    manifest = json.loads(item.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise StorageCapacityError(f"cannot count unverified checkpoint namespace; manifest is unreadable: {item}") from exc
                if manifest.get("identity") != expected_identity:
                    raise StorageCapacityError(f"cannot credit storage occupied by a different checkpoint identity: {item}")
                payload_name = manifest.get("payload_file")
                if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
                    raise StorageCapacityError(f"checkpoint manifest has unsafe payload path: {item}")
            total += item.stat().st_size
    return total


def _validate_bundle_for_record(record: dict[str, Any], bundle, *, synthetic: bool) -> dict[str, Any]:
    if bundle.dataset != record.get("dataset"):
        raise ValueError("run record and loaded dataset names differ")
    if synthetic:
        if not str(bundle.train_metadata.get("md5", "")).startswith("synthetic"):
            raise ValueError("synthetic test execution requires explicitly synthetic dataset metadata")
        from revision_r1.runner import _dataset_identity

        return _dataset_identity(bundle, synthetic=True)
    identity = _bundle_identity(bundle)
    expected_identity = record.get("dataset_identity", {})
    if not isinstance(expected_identity, dict) or any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("loaded production dataset hashes differ from the bound preflight identity")
    reference = record["dataset_reference"]
    if (
        list(bundle.train_metadata.get("shape", [])) != reference["train_shape"]
        or list(bundle.test_metadata.get("shape", [])) != reference["test_shape"]
        or bundle.feature_dim != int(reference["feature_dim"])
    ):
        raise ValueError("loaded production dataset shape/feature dimension differs from the frozen plan")
    return identity


def _run_id_for(record: dict[str, Any], *, synthetic: bool) -> str:
    suffix = "synthetic_validation" if synthetic else "production"
    return (
        f"{record['dataset']}_seed{int(record['seed'])}_{record['arm']}_"
        f"{record['config_sha256'][:10]}_{record['training_source_sha256'][:8]}_{suffix}"
    )


def _expected_runner_identity(plan: dict[str, Any], record: dict[str, Any], bundle, *, synthetic: bool = False) -> dict[str, Any]:
    from revision_r1.runner import _dataset_identity

    run_id = _run_id_for(record, synthetic=synthetic)
    return {
        "runner_version": "revision_R1.0.2", "config_sha256": record["config_sha256"],
        "dataset": _dataset_identity(bundle, synthetic=synthetic),
        "code_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "execution_scope": "synthetic_validation" if synthetic else "production",
        "run_id": run_id,
    }


def _expected_plan_runner_identity(plan: dict[str, Any], record: dict[str, Any], *, synthetic: bool = False) -> dict[str, Any]:
    if synthetic:
        raise ValueError("synthetic runner identity requires the actual test bundle")
    return {
        "runner_version": "revision_R1.0.2", "config_sha256": record["config_sha256"],
        "dataset": {"namespace": "PRODUCTION", "dataset": record["dataset"], **record["dataset_identity"]},
        "code_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"], "execution_scope": "production",
        "run_id": _run_id_for(record, synthetic=False),
    }


def _matrix_registry_path(root: Path, plan: dict[str, Any] | None = None) -> Path:
    """Use the historical flat registry only for read-only v1.2 inspection.

    New registries are plan-scoped, allowing a changed training-source hash to
    coexist without overwriting or misreading prior completion evidence.
    """
    if plan is None:
        return root / "registry" / "matrix_completion_registry.json"
    plan_sha256 = plan.get("plan_sha256")
    if not _is_sha256(plan_sha256):
        raise ValueError("matrix registry requires a valid frozen plan SHA-256")
    return root / "registry" / str(plan_sha256) / "matrix_completion_registry.json"


def _run_progress_identity(plan: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "run_key": record["run_key"],
        "run_index": int(record["run_index"]),
        "config_sha256": record["config_sha256"],
    }


def load_run_progress(
    path: str | Path, plan: dict[str, Any], record: dict[str, Any],
) -> dict[str, Any] | None:
    """Read an existing run log only when it is bound to the exact frozen run."""
    progress_path = Path(path)
    if not progress_path.is_file():
        return None
    try:
        document = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing progress record is unreadable: {progress_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"existing progress record is not a JSON object: {progress_path}")
    expected = _run_progress_identity(plan, record)
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError(f"existing progress identity differs from the frozen plan/run: {progress_path}")
    for field in ("drive_checkpoint_write_bytes_total", "drive_checkpoint_write_bytes_this_attempt"):
        value = document.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"progress {field} must be a nonnegative integer: {progress_path}")
    if document.get("drive_checkpoint_write_bytes_total", 0) < document.get("drive_checkpoint_write_bytes_this_attempt", 0):
        raise ValueError(f"progress cumulative Drive checkpoint bytes are smaller than this attempt: {progress_path}")
    return document


def _read_matrix_registry(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = _matrix_registry_path(root, plan)
    if not path.exists():
        return {
            "schema_version": 1, "plan_sha256": plan["plan_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "completed_runs": {},
        }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"matrix completion registry is unreadable: {path}") from exc
    expected = (plan["plan_sha256"], plan["schedule_sha256"], plan["training_source_sha256"])
    observed = (document.get("plan_sha256"), document.get("schedule_sha256"), document.get("training_source_sha256"))
    if document.get("schema_version") != 1 or observed != expected or not isinstance(document.get("completed_runs"), dict):
        raise ValueError("matrix completion registry identity differs from the frozen plan")
    by_key = {row["run_key"]: row for row in plan["runs"]}
    synthetic_test_root = root.name == "SYNTHETIC_NONREPORTABLE_TEST_ONLY"
    if set(document["completed_runs"]) - set(by_key):
        raise ValueError("matrix completion registry contains a run outside the frozen 100-run plan")
    for key, entry in document["completed_runs"].items():
        row = by_key[key]
        expected_identity = {
            "run_key": row["run_key"], "run_index": int(row["run_index"]),
            "config_sha256": row["config_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "class_order_sha256": row["class_order_sha256"],
            "dataset_identity": row.get("dataset_identity"),
            "run_id": _run_id_for(row, synthetic=synthetic_test_root),
        }
        if not isinstance(entry, dict) or any(entry.get(name) != value for name, value in expected_identity.items()):
            raise ValueError(f"matrix completion registry identity mismatch for {key}")
        if not _is_sha256(entry.get("artifact_manifest_sha256")):
            raise ValueError(f"matrix completion registry artifact identity is malformed for {key}")
        if entry.get("storage_measurement") is not None and not isinstance(entry.get("storage_measurement"), dict):
            raise ValueError(f"matrix completion registry measurement is malformed for {key}")
    return document


def _validate_completed_run(
    plan: dict[str, Any], record: dict[str, Any], bundle, output: Path, *, synthetic: bool,
) -> dict[str, Any]:
    from revision_r1.registry import RunRegistry
    from revision_r1.validation import file_sha256, validate_run_artifacts

    validation = validate_run_artifacts(output, require_production=not synthetic)
    run_id = _run_id_for(record, synthetic=synthetic)
    registry = RunRegistry(output / "registry")
    state = registry.status(run_id)
    if not registry.is_completed(run_id):
        raise ValueError("runner registry does not contain a hash-validated completion for the frozen identity")
    if bundle is None:
        if synthetic:
            raise ValueError("synthetic completion validation requires its in-memory test bundle")
        expected_dataset = {"namespace": "PRODUCTION", "dataset": record["dataset"], **record["dataset_identity"]}
    else:
        expected_dataset = _validate_bundle_for_record(record, bundle, synthetic=synthetic)
    expected_identity = {
        "runner_version": "revision_R1.0.2", "config_sha256": record["config_sha256"],
        "dataset": expected_dataset, "code_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "execution_scope": "synthetic_validation" if synthetic else "production", "run_id": run_id,
    }
    if state.get("identity") != expected_identity:
        raise ValueError("runner completion registry identity differs from plan/config/code/schedule/dataset")
    try:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "full-result.json").read_text(encoding="utf-8"))
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        class_order_payload = json.loads((output / "class_order.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed run identity metadata is unreadable: {exc}") from exc
    expected_namespace = "SYNTHETIC_NON_REPORTABLE" if synthetic else "PRODUCTION"
    if (
        config.get("config_sha256") != record["config_sha256"]
        or config.get("code_sha256") != plan["training_source_sha256"]
        or config.get("namespace") != expected_namespace
        or summary.get("dataset") != record["dataset"]
        or int(summary.get("seed", -1)) != int(record["seed"])
        or summary.get("arm") != record["arm"]
        or summary.get("schedule_sha256") != plan["schedule_sha256"]
        or summary.get("training_reproduction_claimed") is not False
        or status.get("status") != ("SYNTHETIC_VALIDATED_NON_REPORTABLE" if synthetic else "COMPLETED_VALIDATED")
        or class_order_payload.get("sha256") != record["class_order_sha256"]
    ):
        raise ValueError("run artifacts do not match the frozen plan identity/status")
    manifest_hash = file_sha256(output / "artifact_manifest.json")
    return {"validation": validation, "run_id": run_id, "artifact_manifest_sha256": manifest_hash, "summary": summary}


def _checkpoint_tree_measurement(root: Path, expected_identity: dict[str, Any], *, kind: str, expected_count: int) -> dict[str, Any]:
    directory = root / kind
    manifests = sorted(directory.glob("generation-*.manifest.json"))
    if len(manifests) != expected_count:
        raise ValueError(f"production checkpoint retention mismatch at {directory}: expected {expected_count}, found {len(manifests)}")
    payload_bytes = []
    manifest_bytes = 0
    kinds = []
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"production checkpoint manifest is unreadable: {manifest_path}") from exc
        payload_name = manifest.get("payload_file")
        if (
            manifest.get("identity") != expected_identity
            or not isinstance(payload_name, str) or Path(payload_name).name != payload_name
            or not _is_sha256(manifest.get("payload_sha256"))
        ):
            raise ValueError(f"production checkpoint identity/manifest is invalid: {manifest_path}")
        payload = directory / payload_name
        if not payload.is_file() or payload.stat().st_size != int(manifest.get("payload_bytes", -1)):
            raise ValueError(f"production checkpoint payload is missing or size-mismatched: {payload}")
        payload_bytes.append(payload.stat().st_size)
        manifest_bytes += manifest_path.stat().st_size
        kinds.append(str(manifest.get("kind")))
    if kind == "recovery" and "complete" not in kinds:
        raise ValueError("retained production recovery generations do not include the terminal completion checkpoint")
    if kind == "task_boundaries" and any(value != "task-boundary" for value in kinds):
        raise ValueError("task-boundary checkpoint directory contains a non-boundary generation")
    return {
        "generation_count": len(manifests), "payload_bytes": sum(payload_bytes),
        "max_payload_bytes": max(payload_bytes, default=0), "manifest_bytes": manifest_bytes,
        "generation_kinds": kinds,
    }


def measure_completed_production_run(
    plan: dict[str, Any], record: dict[str, Any], bundle, output: str | Path, checkpoint_root: str | Path,
) -> dict[str, Any]:
    """Measure exact run artifacts/checkpoint retention after validating completion."""
    from revision_r1.registry import RunRegistry

    output_path = Path(output).resolve()
    checkpoint_path = Path(checkpoint_root).resolve()
    validated = _validate_completed_run(plan, record, bundle, output_path, synthetic=False)
    run_state = RunRegistry(output_path / "registry").status(validated["run_id"])
    identity = run_state["identity"]
    recovery = _checkpoint_tree_measurement(checkpoint_path, identity, kind="recovery", expected_count=2)
    boundaries = _checkpoint_tree_measurement(checkpoint_path, identity, kind="task_boundaries", expected_count=11)
    output_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file() and not path.is_symlink())
    try:
        resources = json.loads((output_path / "resource_metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("validated run resource metrics are unreadable") from exc
    summary = validated["summary"]
    if int(summary.get("classifier_updates", -1)) != sum(int(value) for value in record["update_budgets_by_task"]):
        raise ValueError("first-run actual classifier updates do not match the frozen 11-task schedule")
    measurement = {
        "status": "MEASURED_VALIDATED_PRODUCTION_RUN",
        "run_key": record["run_key"], "run_id": validated["run_id"],
        "run_output_bytes": output_bytes,
        "recovery_checkpoints": recovery, "task_boundary_checkpoints": boundaries,
        "max_checkpoint_payload_bytes": max(recovery["max_payload_bytes"], boundaries["max_payload_bytes"]),
        "total_checkpoint_payload_bytes": recovery["payload_bytes"] + boundaries["payload_bytes"],
        "total_checkpoint_manifest_bytes": recovery["manifest_bytes"] + boundaries["manifest_bytes"],
        "classifier_updates": int(summary["classifier_updates"]),
        "generator_updates": int(summary["generator_steps"]),
        "critic_steps": int(summary["critic_steps"]),
        "classifier_seconds": float(resources.get("classifier_seconds", 0.0)),
        "generator_seconds": float(resources.get("generator_seconds", 0.0)),
        "critic_seconds": float(resources.get("critic_seconds", 0.0)),
        "checkpoint_seconds": float(resources.get("checkpoint_seconds", 0.0)),
        "active_seconds": float(resources.get("active_seconds", 0.0)),
        "peak_cuda_allocated_bytes": int(resources.get("peak_cuda_allocated_bytes", 0)),
        "peak_cuda_reserved_bytes": int(resources.get("peak_cuda_reserved_bytes", 0)),
        "training_source_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "measurement_source": "resource_metrics.json and exact identity-bound Drive checkpoint manifests/payload sizes",
    }
    return measurement


def _record_completed_run(
    root: Path, plan: dict[str, Any], record: dict[str, Any], validated: dict[str, Any],
    *, storage_measurement: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    registry = _read_matrix_registry(root, plan)
    identity = {
        "run_key": record["run_key"], "run_index": int(record["run_index"]),
        "config_sha256": record["config_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "class_order_sha256": record["class_order_sha256"],
        "dataset_identity": record.get("dataset_identity"),
        "run_id": validated["run_id"],
        "artifact_manifest_sha256": validated["artifact_manifest_sha256"],
        "storage_measurement": storage_measurement,
    }
    existing = registry["completed_runs"].get(record["run_key"])
    if existing is not None:
        if existing != identity:
            raise ValueError(f"completion registry already contains a different identity for {record['run_key']}")
        return registry, False
    registry["completed_runs"][record["run_key"]] = identity
    _atomic_json(_matrix_registry_path(root, plan), registry)
    return registry, True


class _MatrixLeaseHeartbeat:
    def __init__(self, path: Path, *, interval_seconds: float = 60.0, ttl_seconds: float = 300.0):
        from revision_r1.locking import RunLease

        self.lease = RunLease(path, ttl_seconds=ttl_seconds)
        self.interval_seconds = float(interval_seconds)
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.lease.acquire()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="revision-r1-matrix-lease", daemon=True)
        self._thread.start()
        return self

    def _heartbeat_loop(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.lease.heartbeat()
            except Exception:
                self.lost.set()
                return

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self.lease.release()


def execute_frozen_plan_record(
    plan: dict[str, Any], record: dict[str, Any], bundle, *,
    output_root: str | Path, checkpoint_root: str | Path,
    local_checkpoint_root: str | Path,
    schedule_path: str | Path | None = None,
    synthetic: bool = False, test_only: bool = False,
    production_authorized: bool = False,
    device: str = "cuda", update_counts: list[int] | None = None,
    stop_hook=None, progress_callback=None,
    required_resume_manifest: dict[str, Any] | None = None,
    storage_snapshot: dict[str, Any] | None = None,
    storage_projection: dict[str, Any] | None = None,
    capacity_scope: str = "first_run",
    completed_runs: int = 0,
    matrix_lease: _MatrixLeaseHeartbeat | None = None,
) -> dict[str, Any]:
    """Execute exactly one plan record through the unmodified runner.

    The synthetic path is reserved for local tests under an explicitly marked
    namespace. It cannot create a production completion or report.
    """
    from revision_r1 import runner as runner_module
    from revision_r1.config import make_run_config
    from revision_r1.validation import file_sha256
    from revision_r1.resources import read_host_memory_snapshot
    from scripts.live_progress import PhaseRateEstimator, ProgressThrottle, format_live_progress_event

    if synthetic and not test_only:
        raise ValueError("synthetic runner execution is allowed only in explicit non-reportable test namespaces")
    if not synthetic and not production_authorized:
        raise RuntimeError("production run launch switch is disabled; no run started")
    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(Path(__file__).resolve().parents[1]))
    plan_row = next((row for row in plan["runs"] if row.get("run_index") == record.get("run_index")), None)
    if plan_row is None:
        raise ValueError("requested run index does not exist in the frozen plan")
    if (
        record.get("training_source_sha256") != plan["training_source_sha256"]
        or record.get("schedule_sha256") != plan["schedule_sha256"]
        or record.get("config_sha256") != make_run_config(
            record["dataset"], int(record["seed"]), record["arm"], plan["schedule_sha256"],
        ).sha256()
    ):
        raise ValueError("run record source identity/config/schedule differs from the frozen plan")
    if plan_row != record:
        raise ValueError("requested run record is not byte-identical to the frozen plan row")
    if not synthetic:
        if device != "cuda":
            raise RuntimeError("production execution requires CUDA; no run started")
        if schedule_path is None or hashlib.sha256(Path(schedule_path).read_bytes()).hexdigest() != plan["schedule_sha256"]:
            raise ValueError("production schedule is missing or differs from the frozen schedule; no run started")
        if record.get("effective_schedule_sha256") != plan["schedule_sha256"] or not isinstance(record.get("update_budgets_by_task"), list) or len(record["update_budgets_by_task"]) != 11:
            raise ValueError("run record is not bound to all 11 effective schedule budgets")
        if update_counts is not None:
            raise ValueError("production update counts must be loaded from the hash-verified schedule CSV")
        if storage_snapshot is None or storage_projection is None:
            raise StorageCapacityError("production run requires a current, owner-verified storage snapshot and projection")
        if (
            storage_snapshot.get("write_capability_verified") is not True
            or storage_snapshot.get("new_file_owner_verified") is not True
        ):
            raise StorageCapacityError("production Drive destination write access/new-file owner was not verified; no run started")
        if storage_snapshot.get("quota_status") == "UNKNOWN_CAPACITY":
            raise StorageCapacityError(
                "Drive capacity is unverifiable for the authenticated quota account; "
                + str(storage_snapshot.get("quota_capacity_note") or "no numeric available-space value was returned")
                + "; no run started"
            )
        if (
            storage_snapshot.get("output_path") != str(Path(output_root).expanduser().resolve())
            or storage_snapshot.get("checkpoint_path") != str(Path(checkpoint_root).expanduser().resolve())
        ):
            raise StorageCapacityError("capacity evidence is bound to different Drive output/checkpoint roots; no run started")
        require_run_capacity(
            storage_snapshot, storage_projection, scope=capacity_scope,
            completed_runs=completed_runs,
        )
    _validate_bundle_for_record(record, bundle, synthetic=synthetic)
    config = make_run_config(record["dataset"], int(record["seed"]), record["arm"], plan["schedule_sha256"])
    root = Path(output_root).expanduser().resolve()
    if synthetic and (not test_only or root.name != "SYNTHETIC_NONREPORTABLE_TEST_ONLY"):
        raise ValueError("synthetic output root must be explicitly named SYNTHETIC_NONREPORTABLE_TEST_ONLY")
    output = run_output_root(root, record, test_only=False)
    checkpoints = run_checkpoint_root(checkpoint_root, record, test_only=False)
    local_namespace = run_checkpoint_root(local_checkpoint_root, record, test_only=False)
    if synthetic:
        output = root / "runs" / record["run_key"] / record["training_source_sha256"][:16] / record["config_sha256"][:16]
        checkpoints = root / "checkpoints" / record["run_key"] / record["config_sha256"][:16]
        local_namespace = Path(local_checkpoint_root).expanduser().resolve() / record["run_key"] / record["config_sha256"][:16]

    schedule_for_runner = None if synthetic else schedule_path
    remote_checkpoint = None if synthetic else checkpoints
    if required_resume_manifest is not None:
        if synthetic or remote_checkpoint is None:
            raise ValueError("required Drive resume manifest is only supported for the production runner")
        expected_resume_identity = _expected_runner_identity(plan, record, bundle, synthetic=False)
        staged_manifest = _stage_required_resume_manifest(
            Path(remote_checkpoint) / "recovery", local_namespace / "recovery",
            expected_identity=expected_resume_identity,
            generation=int(required_resume_manifest["generation"]),
            manifest_sha256=str(required_resume_manifest["manifest_sha256"]),
        )
        if not staged_manifest.is_file():
            raise RuntimeError("could not stage the fail-closed first-run resume manifest")
    run_identity_sha256 = _sha256_json({
        "run_key": record["run_key"], "config_sha256": record["config_sha256"],
        "source_sha256": plan["training_source_sha256"], "schedule_sha256": plan["schedule_sha256"],
        "dataset_identity": record.get("dataset_identity"),
    })
    root.mkdir(parents=True, exist_ok=True)
    matrix_lock_path = root / ".matrix_execution.lease.json"
    latest_drive: dict[str, Any] | None = None
    drive_write_bytes = 0
    drive_write_seconds = 0.0
    progress_file = root / "progress" / plan["plan_sha256"] / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
    completion_count = len(_read_matrix_registry(root, plan)["completed_runs"])
    prior_progress = load_run_progress(progress_file, plan, record)
    prior_verified_drive_write_bytes = 0 if prior_progress is None else int(
        prior_progress.get("drive_checkpoint_write_bytes_total", 0)
    )
    callback = progress_callback or (lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    attempt_id = f"{time.time_ns()}-{os.getpid()}"
    attempt_started = time.monotonic()
    progress_throttle = ProgressThrottle(interval_seconds=12.0)
    phase_rate = PhaseRateEstimator(stale_after_seconds=120.0)
    event_log_path = progress_file.parent / "events" / f"{record['run_key']}_{record['config_sha256'][:12]}_{attempt_id}.jsonl"
    last_progress_record: dict[str, Any] | None = None
    progress_observer_overhead_seconds = 0.0

    def latest_local_checkpoint_metadata() -> dict[str, Any] | None:
        recovery = local_namespace / "recovery"
        manifests = sorted(
            recovery.glob("generation-*.manifest.json"),
            key=lambda path: int(re.search(r"generation-(\d+)", path.name).group(1))
            if re.search(r"generation-(\d+)", path.name) else -1,
            reverse=True,
        )
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict) or manifest.get("identity") != _expected_runner_identity(plan, record, bundle):
                continue
            generation = manifest.get("generation")
            payload_name = manifest.get("payload_file")
            payload_bytes = manifest.get("payload_bytes")
            if (
                type(generation) is not int or generation <= 0
                or not isinstance(payload_name, str) or not re.fullmatch(r"generation-\d+\.pt", payload_name)
                or type(payload_bytes) is not int or payload_bytes <= 0
            ):
                continue
            payload = recovery / payload_name
            if not payload.is_file() or payload.stat().st_size != payload_bytes:
                continue
            return {
                "generation": generation, "kind": manifest.get("kind"),
                "path": str(payload), "payload_bytes": payload_bytes,
                "payload_sha256": manifest.get("payload_sha256"),
                "manifest_sha256": file_sha256(manifest_path),
                "publication_status": "RUNNER_SAVE_GENERATION_VALIDATED_AT_PUBLICATION",
            }
        return None

    lease_context = _MatrixLeaseHeartbeat(matrix_lock_path) if matrix_lease is None else contextlib.nullcontext(matrix_lease)
    with lease_context as lease:
        previous_mirror = runner_module.mirror_checkpoint_generations

        def instrumented_mirror(local_path, drive_path, identity, **kwargs):
            nonlocal latest_drive, drive_write_bytes, drive_write_seconds
            started = time.perf_counter()
            copied = previous_mirror(local_path, drive_path, identity, **kwargs)
            elapsed = time.perf_counter() - started
            destination = Path(drive_path).resolve()
            if not synthetic and destination.is_relative_to(checkpoints.resolve()):
                for item in copied:
                    manifest_path = destination / f"generation-{int(item['generation']):06d}.manifest.json"
                    payload_path = destination / str(item["payload_file"])
                    try:
                        readback_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise RuntimeError("Drive checkpoint manifest read-back failed after mirror returned") from exc
                    if (
                        not manifest_path.is_file() or not payload_path.is_file()
                        or readback_manifest != item
                        or payload_path.stat().st_size != int(item["payload_bytes"])
                        or file_sha256(payload_path) != item["payload_sha256"]
                    ):
                        raise RuntimeError("Drive checkpoint read-back failed after mirror returned")
                    latest_drive = {
                        "relative_root": str(destination.relative_to(checkpoints.resolve())),
                        "generation": int(item["generation"]), "kind": item["kind"],
                        "manifest_sha256": file_sha256(manifest_path),
                        "payload_sha256": item["payload_sha256"],
                        "payload_bytes": int(item["payload_bytes"]),
                        "verified_at_unix": time.time(),
                        "write_seconds": elapsed,
                    }
                    drive_write_bytes += int(item["payload_bytes"]) + manifest_path.stat().st_size
                drive_write_seconds += elapsed
            return copied

        runner_module.mirror_checkpoint_generations = instrumented_mirror

        def live_stop_hook(state):
            nonlocal last_progress_record, progress_observer_overhead_seconds
            if lease.lost.is_set():
                return True
            now = time.monotonic()
            task_id = min(11, int(state.cursor.task_index) + 1)
            signature = (int(record["run_index"]), task_id, state.cursor.phase)
            if progress_throttle.should_emit(signature, now):
                observer_started = time.perf_counter()
                memory = read_host_memory_snapshot()
                try:
                    import torch
                    if torch.cuda.is_available():
                        # These allocator queries do not synchronize the CUDA stream.
                        memory["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated())
                        memory["gpu_reserved_bytes"] = int(torch.cuda.memory_reserved())
                        memory["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                        memory["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                except Exception as exc:
                    memory["gpu_memory_status"] = f"unavailable:{type(exc).__name__}"
                attempt_elapsed = max(0.0, time.monotonic() - attempt_started)
                event = {
                    "status": "RUNNING", "attempt_id": attempt_id,
                    **_run_progress_identity(plan, record),
                    "run_key": record["run_key"], "run_index": int(record["run_index"]),
                    "run_count": EXPECTED_RUN_COUNT, "dataset": record["dataset"],
                    "seed": int(record["seed"]), "arm": record["arm"],
                    "config_sha256": record["config_sha256"], "plan_sha256": plan["plan_sha256"],
                    "task_sizes": record["config"].get("task_sizes", []),
                    "class_order": record.get("class_order", []),
                    "update_budgets_by_task": record["update_budgets_by_task"],
                    "last_verified_drive_checkpoint": latest_drive,
                    "latest_local_checkpoint": latest_local_checkpoint_metadata(),
                    "drive_checkpoint_write_bytes_this_attempt": drive_write_bytes,
                    "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                    "drive_checkpoint_write_seconds_this_attempt": drive_write_seconds,
                    "progress_observer_overhead_seconds": progress_observer_overhead_seconds,
                    "now_unix": time.time(),
                }
                matrix = {
                    "completed": completion_count, "total": EXPECTED_RUN_COUNT,
                    "not_yet_validated": max(0, EXPECTED_RUN_COUNT - completion_count),
                    "failed_count": None,
                    "failed_count_status": "not centrally persisted; run-attempt failures remain in per-attempt progress logs",
                    "current_run_index": int(record["run_index"]),
                }
                formatted = format_live_progress_event(
                    event, state, attempt_elapsed_seconds=attempt_elapsed,
                    matrix_counts=matrix, memory_snapshot=memory,
                )
                phase_progress = formatted.get("phase_progress", {})
                eta = phase_rate.estimate(
                    (record["run_key"], task_id, state.cursor.phase),
                    phase_progress.get("completed"), phase_progress.get("total"), now=now,
                )
                formatted["time"]["phase_eta_seconds"] = eta
                formatted["time"]["phase_eta_status"] = "measured_within_current_attempt_and_phase" if eta is not None else "awaiting_observed_progress_within_this_attempt_and_phase"
                update = {
                    "status": "RUNNING", "attempt_id": attempt_id,
                    **_run_progress_identity(plan, record),
                    "run_key": record["run_key"], "run_index": int(record["run_index"]),
                    "run_count": EXPECTED_RUN_COUNT, "dataset": record["dataset"],
                    "seed": int(record["seed"]), "arm": record["arm"],
                    "task": task_id, "task_count": 11, "phase": state.cursor.phase,
                    "cursor": dict(state.cursor.__dict__),
                    "classifier_updates": int(state.resources.get("classifier_updates", 0)),
                    "generator_updates": int(state.resources.get("generator_updates", 0)),
                    "critic_steps": int(state.resources.get("critic_steps", 0)),
                    "elapsed_seconds": float(state.resources.get("tracker", {}).get("active_seconds", 0.0)),
                    "attempt_elapsed_seconds": attempt_elapsed,
                    "last_verified_drive_checkpoint": latest_drive,
                    "drive_checkpoint_write_bytes_this_attempt": drive_write_bytes,
                    "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                    "drive_checkpoint_write_seconds_this_attempt": drive_write_seconds,
                    "validated_completion_count": completion_count,
                    "schedule_sha256": plan["schedule_sha256"],
                    "training_source_sha256": plan["training_source_sha256"],
                    "event_log_path": str(event_log_path),
                    "live_progress": formatted,
                    "matrix_lock_is_best_effort": True,
                }
                _append_progress_jsonl(event_log_path, update)
                _atomic_json(progress_file, update)
                callback(update)
                progress_observer_overhead_seconds += time.perf_counter() - observer_started
                last_progress_record = update
            return bool(stop_hook(state)) if stop_hook is not None else False

        try:
            result = runner_module.run_one(
                config, bundle, output, stop_hook=live_stop_hook,
                update_schedule_path=schedule_for_runner, update_counts=update_counts,
                synthetic=synthetic, device=device,
                drive_checkpoint_root=remote_checkpoint,
                local_checkpoint_root=local_namespace,
            )
        except Exception as exc:
            failed_event = {
                "status": "FAILED", "run_key": record["run_key"],
                **_run_progress_identity(plan, record),
                "attempt_id": attempt_id, "event_log_path": str(event_log_path),
                "attempt_elapsed_seconds": max(0.0, time.monotonic() - attempt_started),
                "progress_observer_overhead_seconds": progress_observer_overhead_seconds,
                "error_type": type(exc).__name__, "error": str(exc),
                "drive_checkpoint_write_bytes_this_attempt": drive_write_bytes,
                "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                "drive_checkpoint_write_measurement_scope": "successful mirror read-back bytes only; partial failed-upload bytes are unknown",
                "last_verified_drive_checkpoint": latest_drive,
                "schedule_sha256": plan["schedule_sha256"],
                "training_source_sha256": plan["training_source_sha256"],
                "last_progress": last_progress_record,
            }
            _append_progress_jsonl(event_log_path, failed_event)
            _atomic_json(progress_file, failed_event)
            callback(failed_event)
            raise
        finally:
            runner_module.mirror_checkpoint_generations = previous_mirror
        if lease.lost.is_set():
            raise RuntimeError("matrix lease heartbeat was lost; this run was stopped at a safe checkpoint boundary")
        if result.get("status") == "INTERRUPTED_RESUMABLE":
            interrupted_event = {
                "status": "INTERRUPTED_RESUMABLE", "run_key": record["run_key"],
                **_run_progress_identity(plan, record),
                "attempt_id": attempt_id, "event_log_path": str(event_log_path),
                "attempt_elapsed_seconds": max(0.0, time.monotonic() - attempt_started),
                "progress_observer_overhead_seconds": progress_observer_overhead_seconds,
                "cursor": result.get("cursor"), "last_verified_drive_checkpoint": latest_drive,
                "drive_checkpoint_write_bytes_this_attempt": drive_write_bytes,
                "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                "schedule_sha256": plan["schedule_sha256"],
                "training_source_sha256": plan["training_source_sha256"],
                "last_progress": last_progress_record,
            }
            _append_progress_jsonl(event_log_path, interrupted_event)
            _atomic_json(progress_file, interrupted_event)
            callback(interrupted_event)
            return {
                **result, "output_path": str(output), "local_checkpoint_root": str(local_namespace),
                "drive_checkpoint_root": None if synthetic else str(checkpoints),
                "run_identity_sha256": run_identity_sha256,
                "drive_checkpoint_write_bytes": drive_write_bytes,
                "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                "drive_checkpoint_write_seconds": drive_write_seconds,
                "completion_count": completion_count,
            }
        validated = _validate_completed_run(plan, record, bundle, output, synthetic=synthetic)
        storage_measurement = None if synthetic else measure_completed_production_run(
            plan, record, bundle, output, checkpoints,
        )
        matrix_registry, added = _record_completed_run(
            root, plan, record, validated, storage_measurement=storage_measurement,
        )
        final_progress = {
            "status": "SYNTHETIC_VALIDATED_NON_REPORTABLE" if synthetic else "COMPLETED_VALIDATED",
            **_run_progress_identity(plan, record),
            "attempt_id": attempt_id, "event_log_path": str(event_log_path),
            "attempt_elapsed_seconds": max(0.0, time.monotonic() - attempt_started),
            "progress_observer_overhead_seconds": progress_observer_overhead_seconds,
            "run_key": record["run_key"], "run_index": int(record["run_index"]),
            "output_path": str(output), "config_sha256": record["config_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"], "class_order_sha256": record["class_order_sha256"],
            "artifact_manifest_sha256": validated["artifact_manifest_sha256"],
            "last_verified_drive_checkpoint": latest_drive,
            "drive_checkpoint_write_bytes": drive_write_bytes,
            "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
            "drive_checkpoint_write_seconds": drive_write_seconds,
            "completion_count": len(matrix_registry["completed_runs"]),
            "completion_registry_added_now": added,
            "storage_measurement": storage_measurement,
        }
        _append_progress_jsonl(event_log_path, final_progress)
        _atomic_json(progress_file, final_progress)
        callback(final_progress)
        return {
            **result, "status": "ALREADY_VALIDATED" if result.get("status") == "ALREADY_VALIDATED" else result.get("status"),
            "output_path": str(output), "local_checkpoint_root": str(local_namespace),
            "drive_checkpoint_root": None if synthetic else str(checkpoints),
            "run_identity_sha256": run_identity_sha256,
            "drive_checkpoint_write_bytes": drive_write_bytes,
            "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
            "drive_checkpoint_write_seconds": drive_write_seconds,
            "completion_count": len(matrix_registry["completed_runs"]),
            "completion_registry_added_now": added,
            "storage_measurement": storage_measurement,
        }


def validate_runtime_bound_plan(
    plan: dict[str, Any], schedule_path: str | Path, schedule_manifest_path: str | Path,
    preflight: dict[str, Any], *,
    expected_anchor_corrections: int = EXPECTED_ANCHOR_CORRECTIONS,
    expected_budget_changes: int = EXPECTED_UPDATE_BUDGET_CHANGES,
) -> dict[str, Any]:
    """Recompute the schedule binding and require byte-identical plan identity."""
    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(Path(__file__).resolve().parents[1]))
    binding = plan.get("runtime_binding")
    if not isinstance(binding, dict) or binding.get("schedule_sha256") != plan["schedule_sha256"]:
        raise ValueError("production plan has no matching runtime schedule binding")
    expected = bind_runtime_schedule(
        plan, schedule_path, schedule_manifest_path, preflight,
        expected_anchor_corrections=expected_anchor_corrections,
        expected_budget_changes=expected_budget_changes,
    )
    if expected["plan_sha256"] != plan.get("plan_sha256"):
        raise ValueError("runtime plan does not reproduce from the mounted schedule, manifest, and preflight")
    for row in plan["runs"]:
        if row.get("effective_schedule_sha256") != plan["schedule_sha256"] or len(row.get("update_budgets_by_task", [])) != 11:
            raise ValueError(f"run {row.get('run_key')} is not bound to all 11 effective update budgets")
    return plan


def execute_first_production_run(
    plan: dict[str, Any], *, preflight: dict[str, Any], schedule_path: str | Path,
    schedule_manifest_path: str | Path, v62_recovery_report_path: str | Path,
    data_roots: dict[str, str | Path], cache_root: str | Path,
    output_root: str | Path, checkpoint_root: str | Path, local_checkpoint_root: str | Path,
    launch_enabled: bool = False, progress_callback=None, api_factory=None,
) -> dict[str, Any]:
    """Run the first counted production run only after explicit enable + gates."""
    if launch_enabled is not True:
        raise RuntimeError("LAUNCH_FIRST_PRODUCTION_RUN is False; no production run started")
    validate_runtime_bound_plan(plan, schedule_path, schedule_manifest_path, preflight)
    reference = load_v62_recovery_checkpoint_bound(v62_recovery_report_path)
    projection = project_storage(
        plan, observed_checkpoint_bytes=reference["max_checkpoint_payload_bytes"],
        checkpoint_size_is_already_conservative_bound=True,
        checkpoint_reference_provenance=reference["provenance"],
    )
    first = plan["runs"][0]
    if (first["dataset"], int(first["seed"]), first["arm"]) != FIRST_RUN:
        raise ValueError("frozen first run is not EMBER / seed 42 / Full Hybrid")
    output_run = run_output_root(output_root, first)
    drive_checkpoint_run = run_checkpoint_root(checkpoint_root, first)
    try:
        validated_existing = _validate_completed_run(plan, first, None, output_run, synthetic=False)
    except Exception:
        status_path = output_run / "registry" / "runs" / _run_id_for(first, synthetic=False) / "registry.json"
        if status_path.is_file():
            try:
                existing_status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_status = {}
            if existing_status.get("status") == "completed":
                raise ValueError("first-run output is registered complete but failed identity/artifact validation; refusing overwrite")
        validated_existing = None
    if validated_existing is not None:
        measurement = measure_completed_production_run(
            plan, first, None, output_run, drive_checkpoint_run,
        )
        matrix_registry, _ = _record_completed_run(
            Path(output_root).expanduser().resolve(), plan, first,
            validated_existing, storage_measurement=measurement,
        )
        return {
            "status": "ALREADY_VALIDATED", "run_key": first["run_key"], "run_index": 1,
            "output_path": str(output_run), "drive_checkpoint_root": str(drive_checkpoint_run),
            "storage_measurement": measurement,
            "completion_count": len(matrix_registry["completed_runs"]),
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
        }
    resume_gate = inspect_first_run_resume_checkpoint(
        plan, preflight=preflight, checkpoint_root=checkpoint_root,
        progress_callback=(
            (lambda row: progress_callback({"stage": "resume_checkpoint_" + str(row.get("stage", "progress")), **row}))
            if progress_callback is not None else None
        ),
    )
    if resume_gate.get("status") != "PASS_RESUME_CANDIDATE":
        raise RuntimeError(
            "first production run is configured as resume-only; refusing a fresh start: "
            + str(resume_gate.get("status")) + ": " + str(resume_gate.get("reason"))
        )
    snapshot = inspect_production_capacity(
        output_root, checkpoint_root, local_checkpoint_root, cache_root, api_factory=api_factory,
        verified_dataset_cache_bytes=_preflight_addressed_cache_bytes(preflight, cache_root),
    )
    missing_cache = max(
        0, int(projection["local"]["dataset_cache_upper_bytes"])
        - int(snapshot["local_existing_dataset_cache_bytes"]),
    )
    require_available_bytes(
        local_checkpoint_root, required_bytes=missing_cache,
        available_bytes=int(snapshot["local_available_bytes"]),
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES, storage_kind="local dataset-cache staging",
    )
    output_root_path = Path(output_root).expanduser().resolve()
    preparation_progress_path = output_root_path / "preparation_progress" / plan["plan_sha256"] / (
        f"{first['run_key']}_{first['config_sha256'][:12]}.json"
    )
    preparation_progress_path.parent.mkdir(parents=True, exist_ok=True)
    preparation_attempt_id = f"{time.time_ns()}-{os.getpid()}"
    preparation_started = time.monotonic()

    def emit_preparation(stage: str, **details) -> None:
        from revision_r1.resources import read_host_memory_snapshot

        event = {
            "schema_version": 1, "stage": stage, "run_key": first["run_key"],
            "attempt_id": preparation_attempt_id,
            "training_source_sha256": plan["training_source_sha256"],
            "config_sha256": first["config_sha256"], "schedule_sha256": plan["schedule_sha256"],
            "updated_unix": time.time(), "updated_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_host": socket.gethostname(), "runtime_pid": os.getpid(),
            "memory": read_host_memory_snapshot(), **details,
            "preparation_elapsed_seconds": max(0.0, time.monotonic() - preparation_started),
        }
        _atomic_json(preparation_progress_path, event)
        if progress_callback is not None:
            progress_callback(event)

    # Acquire before hashing/reading data so a second Colab cell cannot consume
    # a second dataset-sized RAM working set before discovering the matrix lock.
    matrix_lease = _MatrixLeaseHeartbeat(output_root_path / ".matrix_execution.lease.json")
    with matrix_lease:
        emit_preparation("dataset_load_begin", dataset=first["dataset"], cache_root=str(cache_root))
        try:
            from revision_r1.resources import HOST_RAM_RESERVE_BYTES, require_host_memory
            require_host_memory(
                256 << 20, reserve_bytes=HOST_RAM_RESERVE_BYTES,
                phase="validated dataset load and bounded training working set",
            )

            def report_dataset_progress(row: dict[str, Any]) -> None:
                details = dict(row)
                stage = str(details.pop("stage", "progress"))
                emit_preparation("dataset_" + stage, **details)

            bundle = load_validated_dataset_bundle(
                data_roots[first["dataset"]], first["dataset"], cache_root, preflight,
                progress_callback=report_dataset_progress,
            )
            emit_preparation(
                "dataset_load_complete", dataset=first["dataset"],
                train_shape=list(bundle.x_train.shape), test_shape=list(bundle.x_test.shape),
                train_dtype=str(bundle.x_train.dtype), test_dtype=str(bundle.x_test.dtype),
                train_memmap=type(bundle.x_train).__name__ == "memmap",
                test_memmap=type(bundle.x_test).__name__ == "memmap",
            )
        except Exception as exc:
            emit_preparation("dataset_load_failed", error_type=type(exc).__name__, error=str(exc))
            raise
        current_output = run_output_root(output_root, first)
        current_drive_checkpoint = run_checkpoint_root(checkpoint_root, first)
        current_local_checkpoint = run_checkpoint_root(local_checkpoint_root, first)
        expected_identity = _expected_runner_identity(plan, first, bundle)
        snapshot = inspect_production_capacity(
            output_root, checkpoint_root, local_checkpoint_root, cache_root, api_factory=api_factory,
            current_run_output_path=current_output,
            current_run_checkpoint_path=current_drive_checkpoint,
            current_run_local_checkpoint_path=current_local_checkpoint,
            expected_run_identity=expected_identity,
            verified_dataset_cache_bytes=max(
                _preflight_addressed_cache_bytes(preflight, cache_root),
                _bundle_cache_bytes(bundle, cache_root),
            ),
        )
        require_run_capacity(snapshot, projection, scope="first_run")
        result = execute_frozen_plan_record(
            plan, first, bundle, output_root=output_root, checkpoint_root=checkpoint_root,
            local_checkpoint_root=local_checkpoint_root, schedule_path=schedule_path,
            production_authorized=True, device="cuda", progress_callback=progress_callback,
            storage_snapshot=snapshot, storage_projection=projection, capacity_scope="first_run",
            completed_runs=0, matrix_lease=matrix_lease,
            required_resume_manifest=resume_gate["selected_integrity_verified_candidate"],
        )
        if result.get("status") not in {"COMPLETED_VALIDATED", "ALREADY_VALIDATED"}:
            return {**result, "first_run_storage_projection_before_run": projection, "v62_reference": reference}
        measurement = result.get("storage_measurement")
        if not isinstance(measurement, dict) or measurement.get("status") != "MEASURED_VALIDATED_PRODUCTION_RUN":
            raise ValueError("first run completed without validated production resource/checkpoint measurements")
        post_run_projection = project_storage(
            plan,
            observed_checkpoint_bytes=measurement["max_checkpoint_payload_bytes"],
            observed_output_bytes=measurement["run_output_bytes"],
            checkpoint_reference_provenance="measured first production run; applied 1.5x + 16 MiB checkpoint envelope",
        )
        report = {
            "schema_version": 1, "status": "FIRST_COUNTED_PRODUCTION_RUN_VALIDATED",
            "run_key": first["run_key"], "run_index": 1, "run_count": 100,
            "plan_sha256": plan["plan_sha256"], "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"], "v62_storage_reference": reference,
            "actual_measurement": measurement,
            "post_run_storage_projection": post_run_projection,
            "production_matrix_launched": False,
            "training_reproduction_claimed": False,
        }
        report["created_unix"] = time.time()
        report["report_sha256"] = _sha256_json(report)
        report_path = Path(output_root).expanduser().resolve() / "reports" / (
            f"first_run_storage_{plan['plan_sha256'][:12]}_{int(report['created_unix'])}.json"
        )
        _atomic_json(report_path, report)
        return {**result, "first_run_storage_report": str(report_path), "post_run_storage_projection": post_run_projection}


def continue_production_matrix(
    plan: dict[str, Any], *, preflight: dict[str, Any], schedule_path: str | Path,
    schedule_manifest_path: str | Path, data_roots: dict[str, str | Path],
    cache_root: str | Path, output_root: str | Path,
    checkpoint_root: str | Path, local_checkpoint_root: str | Path,
    launch_enabled: bool = False, progress_callback=None, api_factory=None,
) -> dict[str, Any]:
    """Continue/resume records 2–100 sequentially, skipping only validated completions."""
    if launch_enabled is not True:
        raise RuntimeError("LAUNCH_REMAINING_MATRIX is False; no matrix run started")
    validate_runtime_bound_plan(plan, schedule_path, schedule_manifest_path, preflight)
    root = Path(output_root).expanduser().resolve()
    lease = _MatrixLeaseHeartbeat(root / ".matrix_execution.lease.json")
    skipped = 0
    launched_or_resumed = 0
    latest_projection: dict[str, Any] | None = None
    latest_capacity: dict[str, Any] | None = None
    with lease:
        first = plan["runs"][0]
        first_output = run_output_root(root, first)
        first_checkpoint = run_checkpoint_root(checkpoint_root, first)
        try:
            first_validated = _validate_completed_run(plan, first, None, first_output, synthetic=False)
        except Exception as exc:
            raise ValueError("remaining matrix is blocked: first counted production run is not identity-matched and artifact-validated") from exc
        first_measurement = measure_completed_production_run(
            plan, first, None, first_output, first_checkpoint,
        )
        first_registry, _ = _record_completed_run(
            root, plan, first, first_validated, storage_measurement=first_measurement,
        )
        observed_measurements = [first_measurement]
        bundle_cache = _SequentialDatasetBundleCache(
            lambda dataset: load_validated_dataset_bundle(
                data_roots[dataset], dataset, cache_root, preflight,
            )
        )
        for record in plan["runs"][1:]:
            if lease.lost.is_set():
                raise RuntimeError("matrix ownership heartbeat was lost; no new run was started")
            output = run_output_root(root, record)
            checkpoints = run_checkpoint_root(checkpoint_root, record)
            try:
                validated = _validate_completed_run(plan, record, None, output, synthetic=False)
            except FileNotFoundError:
                validated = None
            except ValueError as exc:
                status_path = output / "registry" / "runs" / _run_id_for(record, synthetic=False) / "registry.json"
                if status_path.is_file():
                    try:
                        state = json.loads(status_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        state = {}
                    if state.get("status") == "completed":
                        raise ValueError(f"refusing to skip corrupt or identity-mismatched completion {record['run_key']}: {exc}") from exc
                validated = None
            if validated is not None:
                measurement = measure_completed_production_run(plan, record, None, output, checkpoints)
                matrix_registry, _ = _record_completed_run(
                    root, plan, record, validated, storage_measurement=measurement,
                )
                observed_measurements.append(measurement)
                skipped += 1
                if progress_callback is not None:
                    progress_callback({
                        "status": "ALREADY_VALIDATED", "run_index": record["run_index"],
                        "run_count": 100, "run_key": record["run_key"],
                        "validated_completion_count": len(matrix_registry["completed_runs"]),
                    })
                continue

            max_checkpoint = max(row["max_checkpoint_payload_bytes"] for row in observed_measurements)
            max_output = max(row["run_output_bytes"] for row in observed_measurements)
            latest_projection = project_storage(
                plan, observed_checkpoint_bytes=max_checkpoint, observed_output_bytes=max_output,
                checkpoint_reference_provenance="largest observed validated production run so far; applies conservative growth envelope",
            )
            latest_capacity = inspect_production_capacity(
                output_root, checkpoint_root, local_checkpoint_root, cache_root, api_factory=api_factory,
                current_run_output_path=output,
                current_run_checkpoint_path=checkpoints,
                current_run_local_checkpoint_path=run_checkpoint_root(local_checkpoint_root, record),
                expected_run_identity=_expected_plan_runner_identity(plan, record),
                verified_dataset_cache_bytes=_preflight_addressed_cache_bytes(preflight, cache_root),
            )
            latest_capacity["production_measurements_validated"] = True
            current_registry = _read_matrix_registry(root, plan)
            completed_count = len(current_registry["completed_runs"])
            require_run_capacity(
                latest_capacity, latest_projection, scope="matrix", completed_runs=completed_count,
            )
            bundle = bundle_cache.get(record["dataset"])
            latest_capacity = inspect_production_capacity(
                output_root, checkpoint_root, local_checkpoint_root, cache_root, api_factory=api_factory,
                current_run_output_path=output,
                current_run_checkpoint_path=checkpoints,
                current_run_local_checkpoint_path=run_checkpoint_root(local_checkpoint_root, record),
                expected_run_identity=_expected_runner_identity(plan, record, bundle),
                verified_dataset_cache_bytes=max(
                    _preflight_addressed_cache_bytes(preflight, cache_root),
                    _bundle_cache_bytes(bundle, cache_root),
                ),
            )
            latest_capacity["production_measurements_validated"] = True
            require_run_capacity(
                latest_capacity, latest_projection, scope="matrix", completed_runs=completed_count,
            )
            result = execute_frozen_plan_record(
                plan, record, bundle, output_root=output_root, checkpoint_root=checkpoint_root,
                local_checkpoint_root=local_checkpoint_root, schedule_path=schedule_path,
                production_authorized=True, device="cuda", progress_callback=progress_callback,
                storage_snapshot=latest_capacity, storage_projection=latest_projection,
                capacity_scope="matrix", completed_runs=completed_count, matrix_lease=lease,
            )
            if result.get("status") == "INTERRUPTED_RESUMABLE":
                return {
                    "status": "INTERRUPTED_RESUMABLE", "interrupted_run": record["run_key"],
                    "run_index": record["run_index"], "cursor": result.get("cursor"),
                    "validated_completion_count": completed_count,
                    "schedule_sha256": plan["schedule_sha256"],
                    "training_source_sha256": plan["training_source_sha256"],
                }
            if not isinstance(result.get("storage_measurement"), dict):
                raise ValueError(f"run {record['run_key']} returned without measured checkpoint/output evidence")
            observed_measurements.append(result["storage_measurement"])
            launched_or_resumed += 1
        registry = _read_matrix_registry(root, plan)
        status = "ALL_100_RUNS_VALIDATED" if len(registry["completed_runs"]) == EXPECTED_RUN_COUNT else "MATRIX_INCOMPLETE"
        return {
            "status": status, "run_count": EXPECTED_RUN_COUNT,
            "validated_completion_count": len(registry["completed_runs"]),
            "skipped_identity_validated": skipped,
            "launched_or_resumed": launched_or_resumed,
            "plan_sha256": plan["plan_sha256"], "schedule_sha256": plan["schedule_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "last_storage_projection": latest_projection,
            "last_capacity_snapshot": latest_capacity,
            "production_matrix_launched": launched_or_resumed > 0,
            "training_reproduction_claimed": False,
        }


def export_production_progress(plan: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Export compact registry/log status without rescanning run artifacts."""
    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(Path(__file__).resolve().parents[1]))
    root = Path(output_root).expanduser().resolve()
    registry = _read_matrix_registry(root, plan)
    completed = registry["completed_runs"]
    rows = []
    verified_drive_checkpoint_write_bytes = 0
    progress_records_found = 0
    for record in plan["runs"]:
        entry = completed.get(record["run_key"])
        progress_path = root / "progress" / plan["plan_sha256"] / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
        progress = load_run_progress(progress_path, plan, record)
        if progress is not None:
            progress_records_found += 1
            verified_drive_checkpoint_write_bytes += int(progress.get("drive_checkpoint_write_bytes_total", 0))
        rows.append({
            "run_index": record["run_index"], "run_key": record["run_key"],
            "dataset": record["dataset"], "seed": record["seed"], "arm": record["arm"],
            "status": "REGISTERED_COMPLETION" if entry else "NOT_VALIDATED",
            "validation_freshness": "VALIDATED_AT_REGISTRATION_NOT_RECHECKED" if entry else None,
            "config_sha256": record["config_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "progress_log": None if progress is None else {
                "status": progress.get("status"), "task": progress.get("task"),
                "phase": progress.get("phase"), "cursor": progress.get("cursor"),
                "last_verified_drive_checkpoint": progress.get("last_verified_drive_checkpoint"),
                "drive_checkpoint_write_bytes_total": int(progress.get("drive_checkpoint_write_bytes_total", 0)),
            },
        })
    return {
        "schema_version": 1,
        "status": "ALL_RUNS_REGISTERED_NOT_RECHECKED" if len(completed) == EXPECTED_RUN_COUNT else "IN_PROGRESS",
        "artifact_validation_freshness": "completions were validated when registered; this export does not rehash run outputs",
        "plan_sha256": plan["plan_sha256"], "schedule_sha256": plan["schedule_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "run_count": EXPECTED_RUN_COUNT, "validated_completion_count": len(completed),
        "progress_records_found": progress_records_found,
        "successfully_mirrored_checkpoint_payload_and_manifest_bytes": verified_drive_checkpoint_write_bytes,
        "checkpoint_write_measurement_scope": "successful mirror read-back payload+manifest bytes; partial failed-upload bytes, Drive metadata overhead, result archives, and progress JSON writes excluded",
        "runs": rows, "training_reproduction_claimed": False,
    }


def _validated_export_target(
    output_path: str | Path, storage_snapshot: dict[str, Any], *,
    required_payload_bytes: int, label: str,
) -> Path:
    target = Path(output_path).expanduser().resolve()
    parent = target.parent
    if target.exists() or Path(str(target) + ".manifest.json").exists() or Path(str(target) + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite an existing {label} or sidecar: {target}")
    if (
        storage_snapshot.get("path") != str(parent)
        or storage_snapshot.get("write_capability_verified") is not True
        or storage_snapshot.get("new_file_owner_verified") is not True
        or storage_snapshot.get("quota_metadata_verified") is not True
        or storage_snapshot.get("drive_id") is not None
    ):
        raise StorageCapacityError(f"{label} destination owner/quota evidence is missing or bound to another folder")
    available = storage_snapshot.get("available_bytes")
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise StorageCapacityError(f"cannot establish available Drive space for {label}")
    try:
        require_available_bytes(
            target, required_bytes=int(required_payload_bytes) + 16 * 1024 * 1024,
            reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
            available_bytes=available, storage_kind=f"{label} Drive export",
        )
    except StorageCapacityError as exc:
        raise StorageCapacityError(f"{label} capacity check stopped before archive write: {exc}") from exc
    return target


def _zip_archive_size_upper_bound(uncompressed_source_bytes: int) -> int:
    """Reserve 1% for DEFLATE expansion; target gate adds 16 MiB metadata headroom."""
    if isinstance(uncompressed_source_bytes, bool) or int(uncompressed_source_bytes) < 0:
        raise ValueError("archive source byte count must be a nonnegative integer")
    source_bytes = int(uncompressed_source_bytes)
    return (source_bytes * 101 + 99) // 100


def _write_export_sidecar(target: Path, checksum: str) -> None:
    sidecar = Path(str(target) + ".sha256")
    try:
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(f"{checksum}  {target.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite an existing export checksum sidecar: {sidecar}") from exc


def _validate_registered_completion_for_export(
    plan: dict[str, Any], record: dict[str, Any], output: Path, entry: dict[str, Any],
) -> dict[str, Any]:
    """Check frozen identity/markers cheaply; the archive validator hashes files once."""
    from revision_r1.registry import RunRegistry
    from revision_r1.validation import file_sha256

    run_id = _run_id_for(record, synthetic=False)
    expected_identity = _expected_plan_runner_identity(plan, record)
    run_registry = RunRegistry(output / "registry")
    state = run_registry.status(run_id)
    if state.get("status") != "completed" or state.get("identity") != expected_identity:
        raise ValueError(f"registered runner completion identity differs from frozen plan for {record['run_key']}")
    completion_path = output / "registry" / "runs" / run_id / "COMPLETE.json"
    try:
        marker = json.loads(completion_path.read_text(encoding="utf-8"))
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "full-result.json").read_text(encoding="utf-8"))
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        class_order_payload = json.loads((output / "class_order.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"registered completion metadata is unreadable for {record['run_key']}: {exc}") from exc
    artifacts = state.get("completion_artifacts")
    if (
        marker.get("run_id") != run_id or marker.get("identity") != expected_identity
        or marker.get("artifacts") != artifacts
        or marker.get("attempt_id") != (state.get("attempts") or [{}])[-1].get("attempt_id")
    ):
        raise ValueError(f"runner completion marker differs from registry for {record['run_key']}")
    artifact_manifest = output / "artifact_manifest.json"
    manifest_record = (artifacts or {}).get("artifact_manifest", {})
    manifest_digest = file_sha256(artifact_manifest) if artifact_manifest.is_file() else None
    if (
        not isinstance(entry, dict)
        or entry.get("run_id") != run_id
        or entry.get("artifact_manifest_sha256") != manifest_digest
        or manifest_record.get("sha256") != manifest_digest
    ):
        raise ValueError(f"matrix registry artifact identity differs from current output for {record['run_key']}")
    if (
        config.get("config_sha256") != record["config_sha256"]
        or config.get("code_sha256") != plan["training_source_sha256"]
        or config.get("namespace") != "PRODUCTION"
        or any(config.get(key) != value for key, value in record["config"].items())
        or summary.get("dataset") != record["dataset"]
        or int(summary.get("seed", -1)) != int(record["seed"])
        or summary.get("arm") != record["arm"]
        or summary.get("schedule_sha256") != plan["schedule_sha256"]
        or summary.get("training_reproduction_claimed") is not False
        or status.get("status") != "COMPLETED_VALIDATED"
        or status.get("namespace") != "PRODUCTION"
        or class_order_payload.get("class_order") != record["class_order"]
        or class_order_payload.get("sha256") != record["class_order_sha256"]
    ):
        raise ValueError(f"completed result metadata differs from frozen plan identity for {record['run_key']}")
    return {"run_id": run_id, "artifact_manifest_sha256": manifest_digest}


def export_lightweight_results(
    plan: dict[str, Any], output_root: str | Path, output_zip: str | Path, *,
    storage_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Export registered, revalidated production results without checkpoint payloads."""
    from revision_r1.archives import build_lightweight_results_archive

    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(Path(__file__).resolve().parents[1]))
    root = Path(output_root).expanduser().resolve()
    target_candidate = Path(output_zip).expanduser().resolve()
    if any(target_candidate.is_relative_to(run_output_root(root, row)) for row in plan["runs"]):
        raise ValueError("lightweight results archive must be outside every run-artifact namespace")
    registry = _read_matrix_registry(root, plan)
    if not registry["completed_runs"]:
        raise ValueError("no identity-registered production results are available for lightweight export")
    run_roots = []
    raw_bytes = 0
    for record in plan["runs"]:
        entry = registry["completed_runs"].get(record["run_key"])
        if entry is None:
            continue
        run_root = run_output_root(root, record)
        _validate_registered_completion_for_export(plan, record, run_root, entry)
        run_roots.append(run_root)
        for source in run_root.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            relative_parts = source.relative_to(run_root).parts
            if any(part in {"checkpoints", "registry", "__pycache__"} for part in relative_parts):
                continue
            if source.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors"}:
                raw_bytes += source.stat().st_size
    archive_upper_bound = _zip_archive_size_upper_bound(raw_bytes)
    target = _validated_export_target(
        output_zip, storage_snapshot, required_payload_bytes=archive_upper_bound,
        label="lightweight results",
    )
    archive = build_lightweight_results_archive(run_roots, target)
    _write_export_sidecar(target, archive["sha256"])
    return {
        "status": "EXPORTED_VALIDATED_LIGHTWEIGHT_RESULTS",
        "archive": str(target), "archive_sha256": archive["sha256"], "archive_bytes": archive["bytes"],
        "run_count": len(run_roots), "uncompressed_result_source_bytes": raw_bytes,
        "archive_capacity_upper_bound_bytes": archive_upper_bound,
        "checkpoint_payloads_included": False,
        "sidecar": str(Path(str(target) + ".sha256")),
    }


def export_checkpoint_backup(
    plan: dict[str, Any], checkpoint_root: str | Path, output_zip: str | Path, *,
    storage_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Explicitly archive every identity-verified checkpoint after a capacity gate."""
    from revision_r1.archives import build_checkpoint_archive
    from revision_r1.validation import file_sha256

    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(Path(__file__).resolve().parents[1]))
    root = Path(checkpoint_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    target_candidate = Path(output_zip).expanduser().resolve()
    if target_candidate.is_relative_to(root):
        raise ValueError("checkpoint backup archive must be outside the checkpoint source tree")
    expected_manifests: set[Path] = set()
    total_bytes = 0
    generation_count = 0
    for record in plan["runs"]:
        run_root = run_checkpoint_root(root, record)
        expected_identity = _expected_plan_runner_identity(plan, record)
        if not run_root.is_dir():
            continue
        for manifest_path in run_root.rglob("generation-*.manifest.json"):
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"checkpoint manifest path is not a regular file: {manifest_path}")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"checkpoint manifest is unreadable: {manifest_path}") from exc
            payload_name = manifest.get("payload_file")
            if (
                manifest.get("identity") != expected_identity
                or not isinstance(payload_name, str) or Path(payload_name).name != payload_name
                or not _is_sha256(manifest.get("payload_sha256"))
            ):
                raise ValueError(f"checkpoint manifest identity/path/hash is invalid: {manifest_path}")
            payload_path = manifest_path.parent / payload_name
            if (
                payload_path.is_symlink() or not payload_path.is_file()
                or payload_path.stat().st_size != int(manifest.get("payload_bytes", -1))
                or file_sha256(payload_path) != manifest["payload_sha256"]
            ):
                raise ValueError(f"checkpoint payload failed integrity validation: {payload_path}")
            expected_manifests.add(manifest_path.resolve())
            total_bytes += manifest_path.stat().st_size + payload_path.stat().st_size
            generation_count += 1
    if not expected_manifests:
        raise ValueError("no frozen-plan checkpoint generations are available to archive")
    found_manifests = {
        path.resolve() for path in root.rglob("generation-*.manifest.json")
        if path.is_file() and not path.is_symlink()
    }
    if found_manifests != expected_manifests:
        raise ValueError("checkpoint root contains missing or out-of-plan manifests; refusing a mixed-identity archive")
    archive_upper_bound = _zip_archive_size_upper_bound(total_bytes)
    target = _validated_export_target(
        output_zip, storage_snapshot, required_payload_bytes=archive_upper_bound,
        label="checkpoint backup",
    )
    archive = build_checkpoint_archive(root, target)
    _write_export_sidecar(target, archive["sha256"])
    return {
        "status": "EXPORTED_VALIDATED_CHECKPOINT_BACKUP",
        "archive": str(target), "archive_sha256": archive["sha256"], "archive_bytes": archive["bytes"],
        "generation_count": generation_count, "validated_checkpoint_source_bytes": total_bytes,
        "archive_capacity_upper_bound_bytes": archive_upper_bound,
        "archive_growth_margin_fraction": ZIP_ARCHIVE_GROWTH_MARGIN,
        "sidecar": str(Path(str(target) + ".sha256")),
    }
