"""Production orchestration kept outside the hashed revision_r1 training core."""

from __future__ import annotations

import csv
import contextlib
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from revision_r1.config import ARMS, DATASETS, SEEDS, TASK_SIZES, make_run_config
from revision_r1.compatibility import verify_checkpoint_compatibility
from revision_r1.schedule import PoolCounts, load_update_schedule
from revision_r1.storage import (
    DEFAULT_STORAGE_RESERVE_BYTES,
    StorageCapacityError,
    bind_verified_drive_api_service,
    clear_verified_drive_api_service,
    require_available_bytes,
    verified_drive_api_service,
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
DRIVE_FILE_WRITE_SCOPE = "https://www.googleapis.com/auth/drive"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
CANONICAL_STORAGE_ACCOUNT_EMAIL = "kian.anb777@gmail.com"
CANONICAL_REVISION_FOLDER_ID = "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x"
CHECKPOINT_SIZE_MULTIPLIER = 1.5
CHECKPOINT_SIZE_FIXED_OVERHEAD_BYTES = 16 * 1024 * 1024
OUTPUT_BYTES_FLOOR_PER_RUN = 256 * 1024 * 1024
ZIP_ARCHIVE_GROWTH_MARGIN = 0.01
TASK_BOUNDARY_GENERATIONS_PER_RUN = 11
RECOVERY_GENERATIONS_PER_RUN = 2
RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS = (0.0, 0.25, 0.5, 1.0, 2.0)
DRIVE_WRITE_PROBE_BYTES = b"revision_R1 shared-folder capacity probe v1\n"
DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0, 15.0)
CHECKPOINT_PUBLICATION_RETRY_DELAYS_SECONDS = (1.0, 3.0)
_DRIVE_WRITE_PROBE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_BOUND_DRIVE_API_SERVICE = None
_BOUND_STORAGE_SESSION: dict[str, Any] | None = None
_BOUND_STORAGE_ADAPTER = None
V62_RECOVERY_BUILD_ID = "revision_R1_colab_validation_v6_2_final_gan_recovery_qa"
V62_RECOVERY_WORKFLOW_ID = "2b4d8f5f1f29a938"
V62_RECOVERY_REPORT_NAME = "resume_validation_2b4d8f5f1f29a938_20261026T142320Z_946699443.json"
V62_RECOVERY_CODE_SHA256 = "adb4ee76f1d7fb82a35ffaf4f38f5ddaa9b7649c36098f6110f47a938f9faf88"


class RecoveryJournalPublicationError(RuntimeError):
    """Drive journal identity, publication, or remote read-back failed closed."""

    def __init__(
        self, message: str, *, phase: str, expected: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.expected = expected or {}
        self.observed = observed or {}
        details = {
            "phase": phase, "expected": self.expected, "observed": self.observed,
        }
        super().__init__(f"{message}; journal publication diagnostics={json.dumps(details, sort_keys=True, default=str)}")


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
    """Return the checkpoint identity; enforce the bridge for real runner packages.

    Tiny fixture roots used by plan-integrity tests retain the historical raw
    source-hash behavior. Any directory containing the actual runner must carry
    and pass the explicit compatibility manifest.
    """
    root = Path(package_root).resolve()
    if (root / "revision_r1" / "runner.py").is_file():
        manifest = verify_checkpoint_compatibility(root)
        return str(manifest["baseline_training_source_sha256"])
    source = root / "revision_r1"
    digest = hashlib.sha256()
    modules = sorted(source.glob("*.py")) if source.is_dir() else []
    for module in modules:
        digest.update(module.name.encode())
        digest.update(module.read_bytes())
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


def _ensure_drive_folder_path(service, components: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Create only missing exact path components after verifying parent write capability.

    Colab's mounted Drive view can create a local directory before Drive's API
    exposes it. Finalization journals must be API-readable before checkpoint
    pruning, so provision their exact parent folders through Drive and then
    re-resolve the full path. Existing folders/shortcuts continue to be resolved
    by resolve_drive_folder; no path is guessed from mount symlinks.
    """
    names = tuple(str(value) for value in components)
    current = resolve_drive_folder(service, names)
    if current.get("path_exists") is True and current.get("folder_id"):
        return current
    unresolved = current.get("unresolved_components")
    if not isinstance(unresolved, list) or not unresolved:
        raise StorageCapacityError("Drive folder resolver returned an invalid missing-component record")

    resolved_count = len(names) - len(unresolved)
    expected_parent_id = str(current.get("parent_id") or "")
    if not expected_parent_id or current.get("can_add_children") is not True:
        raise StorageCapacityError(
            "cannot create the exact Drive journal folder: parent ID or canAddChildren permission is unverified"
        )

    for offset, name in enumerate(unresolved):
        prefix = names[: resolved_count + offset + 1]
        prefix_location = resolve_drive_folder(service, prefix)
        if prefix_location.get("path_exists") is True and prefix_location.get("folder_id"):
            expected_parent_id = str(prefix_location["folder_id"])
            continue
        parent_id = str(prefix_location.get("parent_id") or "")
        if (
            not parent_id
            or parent_id != expected_parent_id
            or prefix_location.get("can_add_children") is not True
            or prefix_location.get("unresolved_components") != [name]
        ):
            raise StorageCapacityError(
                f"cannot create Drive journal folder {name!r}: exact parent ancestry/write capability is not verified"
            )
        try:
            created = service.files().create(
                body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
                fields="id,name,mimeType,parents,owners(emailAddress),driveId,capabilities(canAddChildren),ownedByMe",
                supportsAllDrives=True,
            ).execute()
        except Exception as exc:
            # A concurrent external creator is acceptable only if a fresh exact
            # resolution finds one unambiguous folder at the requested prefix.
            after_error = resolve_drive_folder(service, prefix)
            if after_error.get("path_exists") is True and after_error.get("folder_id"):
                expected_parent_id = str(after_error["folder_id"])
                continue
            raise StorageCapacityError(
                f"Drive could not create the exact journal folder {name!r} under verified parent {parent_id}: {exc}"
            ) from exc
        created_id = str(created.get("id") or "") if isinstance(created, dict) else ""
        if (
            not created_id
            or created.get("name") != name
            or created.get("mimeType") != DRIVE_FOLDER_MIME
            or parent_id not in [str(value) for value in created.get("parents", [])]
        ):
            raise StorageCapacityError(
                f"Drive folder creation response did not verify the exact name, type, and parent for {name!r}"
            )
        created_metadata = _get_drive_file(service, created_id)
        if (
            created_metadata.get("id") != created_id
            or created_metadata.get("name") != name
            or created_metadata.get("mimeType") != DRIVE_FOLDER_MIME
            or parent_id not in [str(value) for value in created_metadata.get("parents", [])]
        ):
            raise StorageCapacityError(f"Drive-created journal folder {name!r} failed metadata read-back")
        after_create = resolve_drive_folder(service, prefix)
        if (
            after_create.get("path_exists") is not True
            or str(after_create.get("folder_id")) != created_id
            or str(after_create.get("parent_id")) != parent_id
        ):
            raise StorageCapacityError(
                f"Drive-created journal folder {name!r} did not resolve back to its verified identity"
            )
        expected_parent_id = created_id

    resolved = resolve_drive_folder(service, names)
    if resolved.get("path_exists") is not True or not resolved.get("folder_id"):
        raise StorageCapacityError("Drive journal parent path remains unresolved after exact folder provisioning")
    return resolved


def _owner_email(metadata: dict[str, Any]) -> str | None:
    owners = metadata.get("owners") or []
    if not owners or not isinstance(owners[0], dict):
        return None
    value = owners[0].get("emailAddress")
    return str(value).strip().lower() if value else None


def _drive_api_service(*, scopes: tuple[str, ...] = (DRIVE_METADATA_SCOPE,)):
    if _BOUND_DRIVE_API_SERVICE is not None:
        return _BOUND_DRIVE_API_SERVICE
    try:
        return verified_drive_api_service(expected_account_email=CANONICAL_STORAGE_ACCOUNT_EMAIL)
    except Exception:
        raise StorageCapacityError(
            "canonical Drive API is not bound to the verified main-account OAuth credentials; "
            "ADC fallback is disabled, run Cell A before any Drive operation"
        ) from None


def permanently_delete_recovery_checkpoint(
    path: str | Path, *, api_factory=None, drive_root: str | Path = "/content/drive/MyDrive",
    expected_drive_file_id: str | None = None, expected_parent_id: str | None = None,
) -> bool:
    """Delete one exact, canonical-account-owned recovery file after Drive verification.

    Drive Trash still consumes quota, so recovery rotation uses files.delete rather
    than moving large payloads to Trash. The caller passes paths only from the
    identity-validated recovery directory; this helper re-resolves the exact
    parent, name, and file ID. Only generations owned by the canonical storage
    account may be automatically pruned, even if a shared-folder editor has
    canDelete for a historical file. Mixed-owner legacy artifacts are deferred.

    Returns False when the exact file exists but Drive explicitly denies deletion;
    structural ambiguity or unavailable permission evidence still fails closed.
    """
    if not callable(api_factory):
        raise StorageCapacityError("recovery deletion requires the verified canonical Drive API client")
    if _BOUND_STORAGE_SESSION is not None or _BOUND_DRIVE_API_SERVICE is not None:
        _require_canonical_storage_api_factory(api_factory)
    target = Path(path).expanduser()
    if target.name.startswith(".") or not re.fullmatch(r"generation-\d+\.(?:pt|manifest\.json)", target.name):
        raise StorageCapacityError(f"refusing non-generation Drive checkpoint deletion: {target}")
    parts = target.parts
    required = ("ReVision", "revision_R1", "production", "checkpoints")
    if not any(tuple(parts[index:index + len(required)]) == required for index in range(max(0, len(parts) - len(required)))):
        raise StorageCapacityError(f"refusing checkpoint deletion outside ReVision production/checkpoints: {target}")
    if target.parent.name != "recovery":
        raise StorageCapacityError(f"refusing to prune non-recovery checkpoint artifacts: {target}")
    service = api_factory()
    if _drive_account_snapshot(service)["authenticated_user"] != CANONICAL_STORAGE_ACCOUNT_EMAIL:
        raise StorageCapacityError("recovery deletion API identity is not the canonical storage account")
    _location, parent_id = _checkpoint_drive_location(
        service, target.parent, drive_root=Path(drive_root).expanduser(),
    )
    if expected_parent_id is not None and str(parent_id) != str(expected_parent_id):
        raise StorageCapacityError("recovery folder ID differs from the retention journal; no file was deleted")
    metadata = _lookup_drive_child(service, str(parent_id), target.name)
    if not isinstance(metadata, dict):
        if expected_drive_file_id is not None:
            raise StorageCapacityError(f"journaled Drive recovery file is missing before delete: {target.name}")
        if target.exists():
            raise StorageCapacityError(f"Drive API cannot verify checkpoint file before pruning: {target}")
        return True
    if (
        metadata.get("name") != target.name
        or str(parent_id) not in [str(value) for value in metadata.get("parents", [])]
        or metadata.get("mimeType") == DRIVE_FOLDER_MIME
        or not metadata.get("id")
    ):
        raise StorageCapacityError(
            "refusing checkpoint pruning: exact parent, name, and file identity were not verified"
        )
    if expected_drive_file_id is not None and str(metadata["id"]) != str(expected_drive_file_id):
        raise StorageCapacityError(f"Drive file ID differs from the retention journal for {target.name}; no file was deleted")
    owner = _owner_email(metadata)
    if owner != CANONICAL_STORAGE_ACCOUNT_EMAIL or metadata.get("ownedByMe") is not True:
        # Historical generations may be owned by another account even when
        # Drive grants this editor canDelete. Keep those artifacts untouched;
        # the retention goal is subordinate to preserving mixed-owner history.
        return False
    capabilities = metadata.get("capabilities")
    can_delete = capabilities.get("canDelete") if isinstance(capabilities, dict) else None
    if can_delete is False:
        return False
    if can_delete is not True:
        raise StorageCapacityError(
            "refusing checkpoint pruning: Drive did not verify canDelete for the exact checkpoint file"
        )
    file_id = str(metadata["id"])
    try:
        service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    except Exception as exc:
        raise StorageCapacityError(
            f"could not permanently delete verified recovery file {target.name!r}; older checkpoints remain preserved: {exc}"
        ) from exc
    if _lookup_drive_child(service, str(parent_id), target.name) is not None:
        raise StorageCapacityError(
            f"Drive still reports recovery file {target.name!r} after delete request; no further pruning attempted"
        )
    return True


def verify_canonical_storage_session(
    *, api_factory=None,
    drive_mount_root: str | Path = "/content/drive/MyDrive",
    expected_account_email: str = CANONICAL_STORAGE_ACCOUNT_EMAIL,
    expected_revision_folder_id: str = CANONICAL_REVISION_FOLDER_ID,
    path_bindings: dict[str, str | Path] | None = None,
    required_path_keys: tuple[str, ...] = (),
    conditional_update_probe=None,
    storage_adapter=None,
) -> dict[str, Any]:
    """Bind canonical API storage independently of the Colab DriveFS account.

    The write probe is uploaded, read back, inspected, and deleted using the
    exact explicit-OAuth API service that production storage uses. The mounted
    filesystem is deliberately not used to create, read, or remove this probe;
    it may be authenticated as a different compute account.
    """
    global _BOUND_DRIVE_API_SERVICE, _BOUND_STORAGE_SESSION, _BOUND_STORAGE_ADAPTER
    _BOUND_DRIVE_API_SERVICE = None
    _BOUND_STORAGE_SESSION = None
    _BOUND_STORAGE_ADAPTER = None
    clear_verified_drive_api_service()
    _DRIVE_WRITE_PROBE_CACHE.clear()
    expected_account = str(expected_account_email).strip().lower()
    if not expected_account or not expected_revision_folder_id:
        raise ValueError("canonical Drive account and folder ID must be explicit")
    service = (api_factory or (lambda: _drive_api_service(scopes=(DRIVE_FILE_WRITE_SCOPE,))))()
    account_before = _drive_account_snapshot(service)
    authenticated_user = account_before["authenticated_user"]
    if authenticated_user != expected_account:
        raise StorageCapacityError(
            "storage account mismatch: "
            f"expected {expected_account!r}, observed {authenticated_user!r}; "
            "no production paths or checkpoints were modified"
        )
    revision = resolve_drive_folder(service, ("ReVision",))
    if (
        revision.get("path_exists") is not True
        or str(revision.get("folder_id") or "") != str(expected_revision_folder_id)
    ):
        raise StorageCapacityError(
            "canonical ReVision folder ID mismatch: "
            f"expected {expected_revision_folder_id!r}, observed {revision.get('folder_id')!r}"
        )
    if revision.get("drive_id"):
        raise StorageCapacityError("canonical ReVision resolved inside a shared drive; user My Drive quota is not applicable")
    if revision.get("owner_email") != expected_account:
        raise StorageCapacityError(
            "canonical ReVision folder owner mismatch: "
            f"expected {expected_account!r}, observed {revision.get('owner_email')!r}"
        )
    if revision.get("can_add_children") is not True:
        raise StorageCapacityError("main storage account lacks verified canAddChildren permission on canonical ReVision")

    mount_root = Path(drive_mount_root).expanduser()
    if storage_adapter is None:
        raise StorageCapacityError(
            "canonical storage verification requires the DriveApiStorage adapter; "
            "the DriveFS mount or legacy direct probe cannot authorize durable writes"
        )
    if (
        storage_adapter.service is not service
        or storage_adapter.expected_account_email != expected_account
        or storage_adapter.revision_folder_id != str(expected_revision_folder_id)
        or Path(storage_adapter.logical_revision_root).absolute() != (mount_root / "ReVision").absolute()
    ):
        raise StorageCapacityError(
            "canonical storage adapter is not bound to this verified OAuth service/account/ReVision folder"
        )
    bindings: dict[str, Any] = {}
    for key, raw_path in (path_bindings or {}).items():
        configured, components = _logical_drive_path_components(raw_path, drive_root=mount_root)
        location = resolve_drive_folder(service, components)
        exists = location.get("path_exists") is True
        if key in required_path_keys and not exists:
            raise StorageCapacityError(
                f"required storage folder {key!r} is not visible under the authenticated main account: {configured}"
            )
        bindings[key] = {
            "configured_path": configured,
            "components": list(components),
            "folder_id": location.get("folder_id"),
            "parent_id": location.get("parent_id"),
            "path_exists": exists,
            "traversed_shortcut": location.get("traversed_shortcut") is True,
        }

    probe = _write_capability_probe(
        mount_root / "ReVision",
        revision, authenticated_user, service,
        drive_root=mount_root, delete_service=service,
        storage_adapter=storage_adapter,
    )
    if (
        probe.get("new_file_owner") != expected_account
        or probe.get("new_file_owned_by_authenticated_user") is not True
        or probe.get("probe_parent_id") != expected_revision_folder_id
        or probe.get("permanent_recovery_delete_verified") is not True
    ):
        raise StorageCapacityError(
            "canonical OAuth API probe did not verify main-account ownership, exact ReVision parent, and cleanup capability"
        )
    lock_parent_id = str(
        (bindings.get("output_root") or {}).get("folder_id")
        or revision.get("folder_id")
    )
    from revision_r1.drive_storage import verify_conditional_update_support
    lease_service = getattr(storage_adapter, "lease_service", None) or service
    conditional_update = (conditional_update_probe or verify_conditional_update_support)(
        service, parent_id=lock_parent_id, expected_account_email=expected_account,
        lease_service=lease_service,
    )
    if not isinstance(conditional_update, dict) or conditional_update.get("status") != "PASS":
        raise StorageCapacityError(
            "canonical Drive storage cannot prove stale lease updates are rejected; "
            "production worker launch remains disabled"
        )
    account_after = _drive_account_snapshot(service)
    if account_after["authenticated_user"] != expected_account:
        raise StorageCapacityError("authenticated storage account changed during canonical mount verification")

    evidence = {
        "status": "PASS",
        "storage_account_email": expected_account,
        "storage_quota_account_email": account_after["authenticated_user"],
        "storage_quota_status": account_after["quota_status"],
        "storage_quota_limit_bytes": account_after["quota_limit_bytes"],
        "storage_quota_usage_bytes": account_after["quota_usage_bytes"],
        "storage_available_bytes": account_after["available_bytes"],
        "storage_quota_capacity_note": account_after["quota_capacity_note"],
        "drive_mount_root": str(mount_root),
        "canonical_revision_folder_id": str(expected_revision_folder_id),
        "canonical_revision_owner": revision.get("owner_email"),
        "canonical_revision_traversed_shortcut": revision.get("traversed_shortcut") is True,
        "durable_storage_backend": "explicit_main_account_drive_api_with_local_staging",
        "drivefs_mount_identity_used_for_durable_writes": False,
        "lease_compare_and_swap": conditional_update,
        "probe": probe,
        "path_bindings": bindings,
    }
    bind_verified_drive_api_service(service, expected_account_email=expected_account)
    _BOUND_DRIVE_API_SERVICE = service
    _BOUND_STORAGE_SESSION = evidence
    _BOUND_STORAGE_ADAPTER = storage_adapter
    return evidence


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


def _logical_drive_alias_path(
    resolved_path: str | Path, *, configured_root: str | Path,
    drive_root: str | Path = "/content/drive/MyDrive",
) -> Path:
    """Map a resolved mounted path back through its configured Drive alias safely.

    Checkpoint namespace helpers resolve paths for filesystem identity. For API
    traversal, retain the logical MyDrive spelling (which may be a mounted
    shortcut) and prove it resolves to the exact same target beneath the
    configured checkpoint root.
    """
    logical_root = Path(os.path.abspath(str(Path(configured_root).expanduser())))
    resolved_root = logical_root.resolve()
    resolved_target = Path(resolved_path).expanduser().resolve()
    try:
        relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise StorageCapacityError(
            f"resolved Drive target is outside the configured checkpoint root: {resolved_target}"
        ) from exc

    logical_target = logical_root / relative
    configured, _components = _logical_drive_path_components(logical_target, drive_root=drive_root)
    logical_target = Path(configured)
    if logical_target.resolve() != resolved_target:
        raise StorageCapacityError(
            f"logical Drive alias no longer resolves to the verified mounted target: {logical_target}"
        )
    return logical_target


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
                fields=(
                    "nextPageToken,files(id,name,mimeType,parents,owners(emailAddress),ownedByMe,"
                    "quotaBytesUsed,size,md5Checksum,capabilities(canDelete,canEdit))"
                ),
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


def _checkpoint_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_generation_source(
    local_root: str | Path, identity: dict[str, Any], generation: int,
) -> tuple[dict[str, Any], Path, Path]:
    source = Path(local_root).expanduser()
    manifest_path = source / f"generation-{generation:06d}.manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise StorageCapacityError(f"checkpoint publication source manifest is missing/unsafe: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageCapacityError(f"checkpoint publication source manifest is unreadable: {manifest_path}") from exc
    payload_name = f"generation-{generation:06d}.pt"
    payload_path = source / payload_name
    if (
        not isinstance(manifest, dict)
        or manifest.get("generation") != generation
        or manifest.get("identity") != identity
        or manifest.get("payload_file") != payload_name
        or type(manifest.get("payload_bytes")) is not int
        or manifest["payload_bytes"] <= 0
        or not _is_sha256(manifest.get("payload_sha256"))
        or not payload_path.is_file()
        or payload_path.is_symlink()
        or payload_path.stat().st_size != manifest["payload_bytes"]
    ):
        raise StorageCapacityError(
            f"checkpoint publication source does not match the exact run identity/generation: {manifest_path}"
        )
    return manifest, manifest_path, payload_path


def _checkpoint_drive_location(service, destination: Path, *, drive_root: Path) -> tuple[dict[str, Any], str]:
    configured, components = _logical_drive_path_components(destination, drive_root=drive_root)
    if not components or components[0] != "ReVision":
        raise StorageCapacityError(f"checkpoint publication is outside the canonical ReVision folder: {configured}")
    revision = resolve_drive_folder(service, ("ReVision",))
    if (
        revision.get("path_exists") is not True
        or str(revision.get("folder_id") or "") != CANONICAL_REVISION_FOLDER_ID
        or revision.get("drive_id")
    ):
        raise StorageCapacityError("checkpoint publication could not resolve the exact canonical ReVision folder ID")
    location = resolve_drive_folder(service, components)
    if (
        location.get("path_exists") is not True
        or not location.get("folder_id")
        or location.get("drive_id")
        or location.get("can_add_children") is not True
    ):
        raise StorageCapacityError(f"checkpoint publication folder is missing or not writable by the storage account: {configured}")
    return location, str(location["folder_id"])


def require_checkpoint_publication_capacity(
    local_root: str | Path, destination: str | Path, identity: dict[str, Any], *,
    drive_checkpoint_root: str | Path, drive_mount_root: str | Path = "/content/drive/MyDrive",
    api_factory=None, generation: int,
) -> dict[str, Any]:
    """Recheck canonical-account quota immediately before publishing one large checkpoint."""
    _require_canonical_storage_api_factory(api_factory)
    if type(generation) is not int or generation < 1:
        raise ValueError("Drive checkpoint publication requires one explicit positive generation number")
    source_manifest, manifest_path, _payload_path = _checkpoint_generation_source(
        local_root, identity, generation,
    )
    destination_path = Path(destination).expanduser()
    checkpoint_root = Path(drive_checkpoint_root).expanduser()
    try:
        destination_path.resolve().relative_to(checkpoint_root.resolve())
    except ValueError as exc:
        raise StorageCapacityError("checkpoint publication destination escapes the verified checkpoint root") from exc
    service = api_factory()
    account = _drive_account_snapshot(service)
    if account["authenticated_user"] != CANONICAL_STORAGE_ACCOUNT_EMAIL:
        raise StorageCapacityError("checkpoint quota identity changed; no checkpoint was published")
    _location, parent_id = _checkpoint_drive_location(
        service, destination_path, drive_root=Path(drive_mount_root).expanduser(),
    )
    available = account.get("available_bytes")
    if available is None:
        raise StorageCapacityError(
            "canonical Drive quota is unknown; no checkpoint was published and existing recovery points remain intact"
        )
    manifest_bytes = max(manifest_path.stat().st_size, len(json.dumps(source_manifest, sort_keys=True).encode("utf-8")))
    required = int(source_manifest["payload_bytes"]) + manifest_bytes + CHECKPOINT_SIZE_FIXED_OVERHEAD_BYTES
    try:
        require_available_bytes(
            destination_path, required_bytes=required,
            reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
            available_bytes=int(available), storage_kind="canonical Drive checkpoint publication",
        )
    except StorageCapacityError as exc:
        raise StorageCapacityError(
            f"Drive capacity gate stopped before checkpoint publication: {exc}"
        ) from exc
    return {
        "storage_account_email": account["authenticated_user"],
        "quota_available_bytes_before_write": int(available),
        "folder_id": parent_id,
        "generation": generation,
        "payload_bytes": int(source_manifest["payload_bytes"]),
        "manifest_bytes": manifest_bytes,
        "required_peak_bytes_plus_reserve": required + DEFAULT_STORAGE_RESERVE_BYTES,
        "safety_reserve_bytes": DEFAULT_STORAGE_RESERVE_BYTES,
    }


def verify_drive_checkpoint_generation(
    local_root: str | Path, destination: str | Path, identity: dict[str, Any], *,
    generation: int, drive_mount_root: str | Path = "/content/drive/MyDrive", api_factory=None,
) -> dict[str, Any]:
    """Verify both newly published files through Drive API before recovery pruning."""
    _require_canonical_storage_api_factory(api_factory)
    manifest, manifest_path, payload_path = _checkpoint_generation_source(local_root, identity, generation)
    service = api_factory()
    account = _drive_account_snapshot(service)
    if account["authenticated_user"] != CANONICAL_STORAGE_ACCOUNT_EMAIL:
        raise StorageCapacityError("Drive checkpoint read-back used an unexpected quota account")
    _location, parent_id = _checkpoint_drive_location(
        service, Path(destination).expanduser(), drive_root=Path(drive_mount_root).expanduser(),
    )
    expected_files = (
        (manifest_path, manifest_path.stat().st_size, None),
        (payload_path, int(manifest["payload_bytes"]), manifest["payload_sha256"]),
    )
    verified = []
    for local_path, expected_size, expected_sha256 in expected_files:
        metadata = None
        last_error = None
        for attempt, delay in enumerate((0.0, *DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS)):
            if delay:
                time.sleep(delay)
            try:
                metadata = _lookup_drive_child(service, parent_id, local_path.name)
                last_error = None
            except StorageCapacityError as exc:
                if not _retryable_checkpoint_publication_error(exc):
                    raise
                last_error = exc
            if metadata is not None:
                break
            if last_error is not None and attempt + 1 == len(DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS) + 1:
                raise last_error
        if not isinstance(metadata, dict):
            raise StorageCapacityError(
                f"Drive API could not read back newly published checkpoint file {local_path.name!r} "
                f"from exact parent {parent_id}; no stale recovery generation was pruned"
            )
        owner = _owner_email(metadata)
        parents = [str(value) for value in metadata.get("parents", [])]
        size = _optional_nonnegative_int(metadata.get("size"), "published checkpoint size")
        quota_bytes = _optional_nonnegative_int(metadata.get("quotaBytesUsed"), "published checkpoint quota bytes")
        md5_remote = str(metadata.get("md5Checksum") or "").lower()
        md5_local = _checkpoint_md5(local_path)
        if (
            metadata.get("name") != local_path.name
            or not metadata.get("id")
            or metadata.get("mimeType") == DRIVE_FOLDER_MIME
            or parents != [parent_id]
            or owner != CANONICAL_STORAGE_ACCOUNT_EMAIL
            or metadata.get("ownedByMe") is not True
            or size != expected_size
            or quota_bytes is None or quota_bytes <= 0
            or not md5_remote or md5_remote != md5_local
        ):
            mismatch = StorageCapacityError(
                "Drive checkpoint API read-back failed exact parent, main-account ownership, quota attribution, "
                f"size, or MD5 verification for {local_path.name!r}; no stale recovery generation was pruned"
            )
            # DriveFS can expose a just-published file before its metadata settles.
            # Retry only a clearly incomplete size/hash read-back; parent, owner,
            # quota attribution, IDs, and names are authorization/integrity failures.
            if (
                metadata.get("name") == local_path.name
                and metadata.get("id")
                and metadata.get("mimeType") != DRIVE_FOLDER_MIME
                and parents == [parent_id]
                and owner == CANONICAL_STORAGE_ACCOUNT_EMAIL
                and metadata.get("ownedByMe") is True
                and quota_bytes is not None and quota_bytes > 0
                and (size != expected_size or not md5_remote or md5_remote != md5_local)
            ):
                raise _CheckpointPublicationReadbackPending(
                    f"Drive checkpoint bytes are not yet fully visible for {local_path.name!r}"
                ) from mismatch
            raise mismatch
        if expected_sha256 is not None:
            if local_path.stat().st_size != int(manifest["payload_bytes"]):
                raise StorageCapacityError(
                    f"Drive-mounted checkpoint payload size differs from its manifest: {local_path.name}"
                )
        verified.append({
            "name": local_path.name, "drive_file_id": str(metadata["id"]),
            "parent_id": parent_id, "owner_email": owner, "owned_by_me": True,
            "size_bytes": size, "quota_bytes_used": quota_bytes, "md5_checksum": md5_remote,
            "payload_sha256": expected_sha256,
        })
    return {"status": "PASS", "generation": generation, "folder_id": parent_id, "files": verified}


def _retryable_checkpoint_publication_error(exc: Exception) -> bool:
    if isinstance(exc, _CheckpointPublicationReadbackPending):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    cause = exc.__cause__
    if cause is not None and cause is not exc and _retryable_checkpoint_publication_error(cause):
        return True
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
        if status in {408, 409, 429} or (status is not None and 500 <= status <= 599):
            return True
    if not isinstance(exc, OSError):
        return False
    return exc.errno in {
        errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
        errno.ENETDOWN, errno.ENETUNREACH, errno.ESTALE,
    }


class _CheckpointPublicationReadbackPending(StorageCapacityError):
    """Transient DriveFS/API visibility mismatch after an otherwise valid upload."""


def publish_verified_drive_checkpoint_generation(
    mirror_function, local_root: str | Path, destination: str | Path, identity: dict[str, Any], *,
    drive_checkpoint_root: str | Path, drive_mount_root: str | Path = "/content/drive/MyDrive",
    api_factory=None, generation: int, **mirror_kwargs,
):
    """Publish one checkpoint with live capacity, API ownership read-back and safe retries.

    For a pruning recovery sync, API verification of the newest generation is
    performed inside the deletion callback, before the serializer can delete a
    stale point. Checkpoint tensor serialization itself remains in the frozen
    runner/checkpointing modules and is not changed here.
    """
    original_delete = mirror_kwargs.get("delete_callback")
    pruning = mirror_kwargs.get("prune_destination", True) and mirror_kwargs.get("keep", 2) is not None
    if pruning and not callable(original_delete):
        raise StorageCapacityError("canonical Drive recovery pruning requires the exact-ID delete callback")
    for attempt in range(len(CHECKPOINT_PUBLICATION_RETRY_DELAYS_SECONDS) + 1):
        require_checkpoint_publication_capacity(
            local_root, destination, identity,
            drive_checkpoint_root=drive_checkpoint_root,
            drive_mount_root=drive_mount_root, api_factory=api_factory,
            generation=generation,
        )
        verified_before_pruning = False

        def verified_delete(path):
            nonlocal verified_before_pruning
            if not verified_before_pruning:
                verify_drive_checkpoint_generation(
                    local_root, destination, identity, generation=generation,
                    drive_mount_root=drive_mount_root, api_factory=api_factory,
                )
                verified_before_pruning = True
            return original_delete(path)

        call_kwargs = dict(mirror_kwargs)
        if pruning:
            call_kwargs["delete_callback"] = verified_delete
        try:
            result = mirror_function(local_root, destination, identity, **call_kwargs)
            verify_drive_checkpoint_generation(
                local_root, destination, identity, generation=generation,
                drive_mount_root=drive_mount_root, api_factory=api_factory,
            )
            return result
        except Exception as exc:
            if not _retryable_checkpoint_publication_error(exc) or attempt >= len(CHECKPOINT_PUBLICATION_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(CHECKPOINT_PUBLICATION_RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("bounded checkpoint publication retry loop exited unexpectedly")


def _write_capability_probe(
    path: str | Path, location: dict[str, Any], authenticated_user: str, service,
    *, drive_root: str | Path, metadata_lookup=None, delete_service=None,
    api_create=None, api_read=None, storage_adapter=None,
) -> dict[str, Any]:
    """Probe the same main-account API path used for durable Drive writes.

    ``path`` is retained as a logical destination for diagnostics only. Local
    staging is under the host temporary directory, never under DriveFS.
    """
    configured = Path(path).expanduser()
    if location.get("path_exists") is True:
        if not location.get("folder_id"):
            raise StorageCapacityError("API write-capability probe has no exact existing destination folder ID")
        parent_id = str(location["folder_id"])
        probe_scope = "exact_existing_destination"
    else:
        parent_id = str(location.get("parent_id") or "")
        if not parent_id:
            raise StorageCapacityError("API write-capability probe has no exact resolved Drive parent ID")
        probe_scope = "nearest_existing_parent_for_not_yet_created_destination"

    name = f".revision-r1-capacity-probe-{uuid.uuid4().hex}.bin"
    if storage_adapter is not None:
        configured_path, components = _logical_drive_path_components(path, drive_root=drive_root)
        if location.get("path_exists") is True:
            logical_parent = Path(configured_path)
        else:
            unresolved = location.get("unresolved_components")
            if not isinstance(unresolved, list) or len(unresolved) > len(components):
                raise StorageCapacityError("API probe cannot derive the verified nearest existing logical folder")
            resolved_count = len(components) - len(unresolved)
            logical_parent = Path(drive_root).expanduser().absolute().joinpath(*components[:resolved_count])
        logical_probe = logical_parent / name
        try:
            adapter_result = storage_adapter.probe_write(
                logical_probe, DRIVE_WRITE_PROBE_BYTES, expected_parent_id=parent_id,
            )
        except Exception as exc:
            raise StorageCapacityError(
                f"main-account API adapter probe failed at {logical_probe}: {type(exc).__name__}: {exc}"
            ) from exc
        diagnostics = adapter_result.get("diagnostics")
        observed = diagnostics.get("observed", {}) if isinstance(diagnostics, dict) else {}
        if (
            adapter_result.get("status") != "PASS"
            or adapter_result.get("parent_id") != parent_id
            or adapter_result.get("owner") != authenticated_user
            or adapter_result.get("ownedByMe") is not True
            or adapter_result.get("size") != len(DRIVE_WRITE_PROBE_BYTES)
            or not isinstance(adapter_result.get("quotaBytesUsed"), int)
            or adapter_result.get("quotaBytesUsed", 0) <= 0
            or adapter_result.get("delete_verified") is not True
        ):
            raise StorageCapacityError(
                "main-account API adapter probe did not verify exact owner/parent/quota/read-back/cleanup: "
                + json.dumps(diagnostics or {"observed": observed}, sort_keys=True, default=str)
            )
        return {
            "status": "PASS", "probe_name": name,
            "probe_parent_id": parent_id, "new_file_owner": adapter_result["owner"],
            "new_file_owned_by_authenticated_user": adapter_result["ownedByMe"],
            "new_file_quota_bytes": adapter_result["quotaBytesUsed"],
            "new_file_size_bytes": adapter_result["size"],
            "quota_usage_before_probe_bytes": adapter_result.get("quota_usage_before_bytes"),
            "quota_usage_during_probe_bytes": adapter_result.get("quota_usage_during_bytes"),
            "quota_limit_during_probe_bytes": adapter_result.get("quota_limit_during_bytes"),
            "probe_scope": probe_scope, "probe_completed_unix": time.time(),
            "permanent_recovery_delete_verified": adapter_result["delete_verified"],
            "permanent_delete_probe_file_id": adapter_result["file_id"],
            "permanent_delete_probe_error": None,
            "diagnostics": diagnostics,
            "write_path": "DriveApiStorage.publish_file_download_file_delete_exact",
        }
    lookup = metadata_lookup or _lookup_drive_child
    create = api_create or _drive_api_create_binary_file
    read = api_read or _drive_api_read_bytes
    probe_deleted_by_api = False
    permanent_delete_verified = False
    permanent_delete_error = None
    probe_id = None
    expected = {
        "file_id": "non-empty API-created file ID",
        "parent_id": parent_id,
        "owner": authenticated_user,
        "ownedByMe": True,
        "size": len(DRIVE_WRITE_PROBE_BYTES),
        "quotaBytesUsed": "positive integer",
        "api_visible": True,
    }
    try:
        fd, scratch_name = tempfile.mkstemp(prefix="revision-r1-api-write-probe-", suffix=".bin")
        probe_path = Path(scratch_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(DRIVE_WRITE_PROBE_BYTES)
                stream.flush()
                os.fsync(stream.fileno())
            if probe_path.read_bytes() != DRIVE_WRITE_PROBE_BYTES:
                raise StorageCapacityError("local staging probe failed byte-for-byte verification")
        finally:
            probe_path.unlink(missing_ok=True)

        created = create(service, parent_id, name, DRIVE_WRITE_PROBE_BYTES)
        if not isinstance(created, dict):
            raise StorageCapacityError("main-account Drive API probe create returned no file metadata")
        raw_probe_id = created.get("id")
        if isinstance(raw_probe_id, str) and raw_probe_id:
            probe_id = raw_probe_id

        if probe_id is None:
            raise StorageCapacityError("main-account Drive API probe create returned no exact file ID")
        readback = read(service, probe_id)
        if not isinstance(readback, (bytes, bytearray)) or bytes(readback) != DRIVE_WRITE_PROBE_BYTES:
            raise StorageCapacityError("main-account Drive API probe failed byte-for-byte media read-back")

        metadata = None
        lookup_attempts = 0
        for retry_delay in (0.0, *DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS):
            if retry_delay:
                time.sleep(retry_delay)
            lookup_attempts += 1
            metadata = lookup(service, parent_id, name)
            if metadata is not None:
                break
        if not isinstance(metadata, dict):
            observed = {
                "file_id": probe_id, "parent_id": None, "owner": None,
                "ownedByMe": None, "size": None, "quotaBytesUsed": None,
                "api_visible": False,
            }
            diagnostics = _write_probe_diagnostics(expected, observed, metadata_status="MISSING_OR_STALE_METADATA")
            raise StorageCapacityError(
                "main-account API probe payload was created and read back, but metadata remained missing/stale "
                f"after {lookup_attempts} bounded visibility attempts: {json.dumps(diagnostics, sort_keys=True)}"
            )
        file_owner = _owner_email(metadata)
        parents = [str(value) for value in metadata.get("parents", [])]
        def probe_integer(value):
            if value is None or isinstance(value, bool):
                return None
            try:
                return max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                return None

        quota_bytes = probe_integer(metadata.get("quotaBytesUsed"))
        size = probe_integer(metadata.get("size"))
        observed = {
            "file_id": metadata.get("id"),
            "parent_id": parents[0] if len(parents) == 1 else parents,
            "owner": file_owner,
            "ownedByMe": metadata.get("ownedByMe"),
            "size": size,
            "quotaBytesUsed": quota_bytes,
            "api_visible": True,
        }
        mismatch_fields = []
        if metadata.get("id") and metadata.get("id") != probe_id:
            mismatch_fields.append("file_id")
        if metadata.get("name") and metadata.get("name") != name:
            mismatch_fields.append("name")
        if parents and parent_id not in parents:
            mismatch_fields.append("parent_id")
        if metadata.get("mimeType") == DRIVE_FOLDER_MIME:
            mismatch_fields.append("mimeType")
        if (file_owner is not None and file_owner != authenticated_user) or metadata.get("ownedByMe") is False:
            mismatch_fields.append("owner")
        if size is None or size != len(DRIVE_WRITE_PROBE_BYTES):
            mismatch_fields.append("size")
        if quota_bytes is None or quota_bytes <= 0:
            mismatch_fields.append("quotaBytesUsed")
        diagnostics = _write_probe_diagnostics(expected, observed, metadata_status="VISIBLE")
        missing_fields = [
            field for field, condition in diagnostics["conditions"].items()
            if condition["status"] == "MISSING_OR_STALE"
        ]
        if (file_owner is not None and file_owner != authenticated_user) or metadata.get("ownedByMe") is False:
            diagnostics["classification"] = "CONFIRMED_OWNERSHIP_MISMATCH"
            raise StorageCapacityError(
                "confirmed ownership mismatch for main-account API probe: "
                + json.dumps(diagnostics, sort_keys=True)
            )
        if parents and parent_id not in parents:
            diagnostics["classification"] = "CONFIRMED_PARENT_MISMATCH"
            raise StorageCapacityError(
                "confirmed parent mismatch for main-account API probe: "
                + json.dumps(diagnostics, sort_keys=True)
            )
        if mismatch_fields or missing_fields:
            diagnostics["classification"] = "MISSING_OR_STALE_METADATA"
            diagnostics["mismatched_fields"] = mismatch_fields
            diagnostics["missing_or_stale_fields"] = missing_fields
            raise StorageCapacityError(
                "main-account API probe metadata is incomplete or stale: "
                + json.dumps(diagnostics, sort_keys=True)
            )

        account_during_probe = _drive_account_snapshot(service)
        if account_during_probe["authenticated_user"] != authenticated_user:
            raise StorageCapacityError("authenticated storage account changed during the API write-capability probe")
        if delete_service is None:
            permanent_delete_error = "Drive files.delete capability could not be authorized"
        else:
            try:
                delete_service.files().delete(fileId=probe_id, supportsAllDrives=True).execute()
                for retry_delay in (0.0, *DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS):
                    if retry_delay:
                        time.sleep(retry_delay)
                    if lookup(service, parent_id, name) is None:
                        probe_deleted_by_api = True
                        permanent_delete_verified = True
                        break
                if not permanent_delete_verified:
                    permanent_delete_error = "Drive API still listed the exact probe after files.delete"
            except Exception as exc:
                permanent_delete_error = f"Drive API probe cleanup failed: {type(exc).__name__}"
        if not permanent_delete_verified:
            raise StorageCapacityError(
                "main-account API probe passed write/read-back but exact-ID cleanup was not verified: "
                + json.dumps({
                    "file_id": probe_id, "parent_id": parent_id, "name": name,
                    "api_deleted": probe_deleted_by_api, "error": permanent_delete_error,
                }, sort_keys=True)
            )
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
            "permanent_recovery_delete_verified": permanent_delete_verified,
            "permanent_delete_probe_file_id": probe_id,
            "permanent_delete_probe_error": permanent_delete_error,
            "diagnostics": diagnostics,
            "write_path": "explicit_main_account_drive_api_with_local_staging",
        }
    except FileExistsError as exc:
        raise StorageCapacityError("unique API write-capability probe name unexpectedly already exists; no existing file was modified") from exc
    except OSError as exc:
        raise StorageCapacityError(f"local staging for the API write-capability probe failed: {exc}") from exc
    finally:
        if probe_id is not None and delete_service is not None and not probe_deleted_by_api:
            try:
                delete_service.files().delete(fileId=probe_id, supportsAllDrives=True).execute()
                probe_deleted_by_api = lookup(service, parent_id, name) is None
                permanent_delete_verified = probe_deleted_by_api
                if not permanent_delete_verified:
                    permanent_delete_error = "Drive API still listed the exact probe after files.delete"
            except Exception as exc:
                try:
                    probe_deleted_by_api = lookup(service, parent_id, name) is None
                    permanent_delete_verified = probe_deleted_by_api
                except Exception:
                    probe_deleted_by_api = False
                permanent_delete_error = None if permanent_delete_verified else f"Drive API probe cleanup failed: {type(exc).__name__}"
        if probe_id is not None and not probe_deleted_by_api:
            # Cleanup failures must be surfaced with the exact disposable ID;
            # never fall back to deleting by a broad filename or through DriveFS.
            cleanup = {
                "file_id": probe_id, "parent_id": parent_id, "name": name,
                "api_deleted": False, "error": permanent_delete_error,
            }
            if not sys.exc_info()[0]:
                raise StorageCapacityError(
                    "could not verify cleanup of the exact temporary API probe: "
                    + json.dumps(cleanup, sort_keys=True)
                )


def _write_probe_diagnostics(expected: dict[str, Any], observed: dict[str, Any], *, metadata_status: str) -> dict[str, Any]:
    """Return field-by-field sanitized expectations without credentials/tokens."""
    conditions = {}
    for key in ("file_id", "parent_id", "owner", "ownedByMe", "size", "quotaBytesUsed", "api_visible"):
        value = observed.get(key)
        target = expected.get(key)
        if key == "file_id":
            passed = isinstance(value, str) and bool(value)
        elif key == "quotaBytesUsed":
            passed = type(value) is int and value > 0
        elif key == "api_visible":
            passed = value is True
        elif key == "parent_id" and isinstance(value, list):
            passed = target in value
        elif key == "size":
            passed = type(value) is int and value == target
        else:
            passed = value == target
        conditions[key] = {
            "status": "PASS" if passed else "MISMATCH" if value is not None else "MISSING_OR_STALE",
            "expected": target, "observed": value,
        }
    return {
        "classification": "VERIFIED" if all(item["status"] == "PASS" for item in conditions.values()) else metadata_status,
        "expected_owner": expected.get("owner"),
        "observed_owner": observed.get("owner"),
        "conditions": conditions,
        "expected": dict(expected), "observed": dict(observed),
    }


def _drive_api_create_binary_file(service, parent_id: str, name: str, payload: bytes) -> dict[str, Any]:
    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/octet-stream", resumable=False)
    return service.files().create(
        body={"name": name, "mimeType": "application/octet-stream", "parents": [parent_id]},
        media_body=media, fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed",
        supportsAllDrives=True,
    ).execute()


def _drive_api_read_bytes(service, file_id: str) -> bytes:
    payload = service.files().get(fileId=file_id, alt="media", supportsAllDrives=True).execute()
    if not isinstance(payload, (bytes, bytearray)):
        raise StorageCapacityError(f"Drive API returned non-byte media for exact file ID {file_id!r}")
    return bytes(payload)
def _verified_write_capability(
    path: str | Path, location: dict[str, Any], authenticated_user: str, service,
    *, drive_root: str | Path, metadata_lookup=None, delete_api_factory=None,
    storage_adapter=None,
) -> dict[str, Any]:
    folder_binding = str(location.get("folder_id") or f"{location.get('parent_id')}:{'/'.join(location.get('unresolved_components', []))}")
    cache_key = (authenticated_user, folder_binding)
    now = time.time()
    cached = _DRIVE_WRITE_PROBE_CACHE.get(cache_key)
    if cached:
        return {**cached, "probe_performed_this_check": False, "probe_cache_age_seconds": max(0.0, now - float(cached["probe_completed_unix"]))}
    delete_service = None
    delete_service_error = None
    try:
        if delete_api_factory is not None:
            delete_service = delete_api_factory()
        elif service is not None and not isinstance(service, type) and hasattr(service, "records"):
            # A supplied test/integration service may expose both endpoints.
            delete_service = service
        else:
            delete_service = _drive_api_service(scopes=(DRIVE_FILE_WRITE_SCOPE,))
    except Exception as exc:
        delete_service_error = str(exc)
    result = _write_capability_probe(
        path, location, authenticated_user, service,
        drive_root=drive_root, metadata_lookup=metadata_lookup, delete_service=delete_service,
        storage_adapter=storage_adapter,
    )
    if delete_service_error and not result.get("permanent_recovery_delete_verified"):
        result["permanent_delete_probe_error"] = delete_service_error
    if result.get("permanent_recovery_delete_verified") is True:
        _DRIVE_WRITE_PROBE_CACHE[cache_key] = dict(result)
    return {**result, "probe_performed_this_check": True, "probe_cache_age_seconds": 0.0}


def inspect_drive_destination(
    path: str | Path, *, api_factory=None, drive_root: str | Path = "/content/drive/MyDrive",
    probe_metadata_lookup=None, delete_api_factory=None, storage_adapter=None,
) -> dict[str, Any]:
    """Verify exact My Drive path, Editor write access, creator quota attribution, and capacity."""
    configured, components = _logical_drive_path_components(path, drive_root=drive_root)
    # Resolve folder aliases through the Drive API below; filesystem shortcuts
    # may resolve outside MyDrive's lexical mount path while still targeting a
    # user-owned My Drive folder.
    target = Path(path).expanduser().resolve()
    service = (api_factory or _drive_api_service)()
    if _BOUND_STORAGE_SESSION is not None and service is not _BOUND_DRIVE_API_SERVICE:
        raise StorageCapacityError("Drive operation attempted with a client outside the verified canonical storage session")
    if storage_adapter is None and _BOUND_STORAGE_SESSION is not None:
        storage_adapter = _BOUND_STORAGE_ADAPTER
    if _BOUND_STORAGE_SESSION is not None and storage_adapter is None:
        raise StorageCapacityError(
            "canonical Drive capacity probes require the verified API storage adapter; direct API/DriveFS probe rejected"
        )
    if storage_adapter is not None and (
        storage_adapter.service is not service
        or storage_adapter.expected_account_email != _drive_account_snapshot(service)["authenticated_user"]
        or storage_adapter.revision_folder_id != str(CANONICAL_REVISION_FOLDER_ID)
        or Path(storage_adapter.logical_revision_root).absolute()
        != (Path(drive_root).expanduser().absolute() / "ReVision")
    ):
        raise StorageCapacityError(
            "write-capability probe adapter differs from the verified Drive account/service/ReVision folder"
        )
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
        delete_api_factory=(
            delete_api_factory
            or ((lambda: service) if api_factory is not None else None)
        ),
        storage_adapter=storage_adapter,
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
        "permanent_recovery_delete_verified": probe.get("permanent_recovery_delete_verified") is True,
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
    checkpoint_size_evidence: dict[str, Any] | None = None,
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
            "recovery_generations_pruned": True,
            "recovery_retention_policy": "publish and verify a new generation before pruning; retain two verified Drive generations",
        },
        "checkpoint_size_evidence": checkpoint_size_evidence,
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
            "production_checkpoint_deletion_or_retention_change": True,
            "task_boundary_archives_pruned": False,
            "recovery_generations_pruned_only_after_verified_publication": True,
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
            "next_run_required_without_reserve_bytes": per_run_drive_retained + checkpoint_atomic + output_atomic,
            "next_run_peak_with_reserve_bytes": per_run_drive_retained + checkpoint_atomic + output_atomic + safety,
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
        drive_key = "next_run_required_without_reserve_bytes"
        local_key = "next_run_required_without_reserve_bytes"
        drive_projection = projection.get("drive", {})
        local_projection = projection.get("local", {})
        description = "next matrix run (full-matrix projection is informational only)"
        if storage_snapshot.get("production_measurements_validated") is not True:
            raise StorageCapacityError(
                "matrix capacity gate requires validated first-run production checkpoint/output measurements; no matrix run was launched"
            )
        if not 0 <= int(completed_runs) <= EXPECTED_RUN_COUNT:
            raise ValueError("completed run count is outside the frozen matrix")
    else:
        raise ValueError("capacity scope must be 'first_run' or 'matrix'")
    reserve = DEFAULT_STORAGE_RESERVE_BYTES
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
        required = projection_values.get(projection_key)
        if resource == "Drive":
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
        "lease_acquisition_allowed": False,
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

    # Drive API lease records have an explicit v1 schema and an inactive
    # placeholder/released representation. Use the same validator as the CAS
    # acquisition path so diagnostics cannot call a valid released lease
    # malformed or label corrupt inactive JSON as available.
    if "schema_version" in record or "active" in record:
        from revision_r1.drive_storage import _classify_drive_api_lease_record

        state = _classify_drive_api_lease_record(record, now=now)
        report["record_kind"] = state.get("record_kind")
        if state["status"] == "malformed":
            report.update(status="malformed", parse_error=state["error"])
            return report
        for field in ("created_at", "heartbeat_at", "expires_at", "released_at"):
            value = state.get(field)
            if value is not None:
                report[f"{field}_unix"] = value
                report[f"{field}_utc"] = _utc_iso(value)
        if state["status"] == "inactive":
            report.update(
                status="inactive",
                owner_process_visibility=(
                    "released_owner_not_active" if state["record_kind"] == "released"
                    else "no_active_owner"
                ),
                lease_acquisition_allowed=True,
            )
            return report

        owner_host = state["owner_host"]
        owner_pid = state["owner_pid"]
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
        if state["status"] == "live":
            report["status"] = "live"
        elif owner_process in {"same_host_pid_alive", "same_host_pid_exists_permission_denied"}:
            report["status"] = "expired_but_owner_process_may_be_alive"
        else:
            report.update(
                status="expired",
                recovery_may_be_attempted_by_existing_lease_logic=True,
                lease_acquisition_allowed=True,
            )
        return report

    for field in ("created_at", "heartbeat_at", "expires_at"):
        raw_value = record.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
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
        report["lease_acquisition_allowed"] = True
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


def _checkpoint_payload_size_reference(
    checkpoint_run_root: str | Path,
    *,
    expected_identity: dict[str, Any],
    selected_recovery_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Size the frozen run from matching on-disk checkpoint manifests without loading payloads.

    If supplied, the selected recovery generation must already have passed a
    full streaming SHA-256 check in ``inspect_first_run_resume_checkpoint``.
    Other matching generations (and all generations during Cell C's read-only
    sizing pass) are used only as observed size evidence: manifests, identities,
    payload paths, and file sizes are validated, but multi-gigabyte payloads
    are not rehashed or deserialized just to estimate capacity.
    """
    candidate = selected_recovery_candidate
    if candidate is not None:
        if (
            not isinstance(candidate, dict)
            or candidate.get("integrity_status") != "PASS_MANIFEST_AND_PAYLOAD_HASH"
            or candidate.get("payload_hash_matches") is not True
            or candidate.get("manifest_identity_matches") is not True
        ):
            raise StorageCapacityError("storage projection requires a hash-verified matching recovery checkpoint")
        candidate_bytes = candidate.get("payload_bytes_observed")
        candidate_expected_bytes = candidate.get("payload_bytes_expected")
        candidate_generation = candidate.get("generation")
        if (
            type(candidate_bytes) is not int or candidate_bytes <= 0
            or type(candidate_expected_bytes) is not int or candidate_expected_bytes != candidate_bytes
            or type(candidate_generation) is not int or candidate_generation <= 0
            or not _is_sha256(candidate.get("manifest_sha256"))
            or not _is_sha256(candidate.get("payload_sha256_expected"))
            or candidate.get("payload_sha256_observed") != candidate.get("payload_sha256_expected")
        ):
            raise StorageCapacityError("selected recovery checkpoint lacks a valid generation or observed payload size")
    else:
        candidate_bytes = None
        candidate_generation = None

    root = Path(checkpoint_run_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise StorageCapacityError(f"cannot inspect the exact existing checkpoint namespace: {root}")
    manifests = sorted(root.rglob("generation-*.manifest.json"))
    if not manifests:
        raise StorageCapacityError("matching recovery candidate exists but its checkpoint namespace has no manifests")

    observed = []
    selected_seen = candidate is None
    recovery_manifest_count = 0
    recovery_payload_sizes = []
    for manifest_path in manifests:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise StorageCapacityError(f"checkpoint manifest path is not a regular file: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageCapacityError(f"cannot size checkpoint namespace with an unreadable manifest: {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("identity") != expected_identity:
            raise StorageCapacityError(f"checkpoint-size evidence has a malformed or different run identity: {manifest_path}")
        generation = manifest.get("generation")
        payload_name = manifest.get("payload_file")
        payload_bytes = manifest.get("payload_bytes")
        payload_hash = manifest.get("payload_sha256")
        if (
            type(generation) is not int or generation <= 0
            or manifest_path.name != f"generation-{generation:06d}.manifest.json"
            or not isinstance(payload_name, str) or not re.fullmatch(r"generation-\d+\.pt", payload_name)
            or type(payload_bytes) is not int or payload_bytes <= 0
            or not _is_sha256(payload_hash)
        ):
            raise StorageCapacityError(f"checkpoint-size manifest has invalid generation/payload metadata: {manifest_path}")
        payload_path = manifest_path.parent / payload_name
        if payload_path.is_symlink():
            raise StorageCapacityError(f"checkpoint payload path is a symlink: {payload_path}")
        if payload_path.is_file():
            payload_bytes_observed = payload_path.stat().st_size
        else:
            payload_bytes_observed = 0
        if manifest_path.parent.name == "recovery":
            recovery_manifest_count += 1
            if payload_bytes_observed > 0:
                recovery_payload_sizes.append(payload_bytes_observed)
        if candidate is not None and generation == candidate_generation and manifest_path.parent.name == "recovery":
            manifest_sha = _hash_file_stream(manifest_path)
            if (
                manifest_sha != candidate.get("manifest_sha256")
                or payload_hash != candidate.get("payload_sha256_expected")
                or payload_bytes != candidate_bytes
                or payload_bytes_observed != candidate_bytes
            ):
                raise StorageCapacityError("selected recovery checkpoint changed after its integrity preflight")
            selected_seen = True
        observed.append({
            "generation": generation, "kind": manifest.get("kind"),
            "payload_bytes": payload_bytes_observed,
            "manifest_payload_bytes": payload_bytes,
            # A missing/truncated non-selected generation may be the damaged
            # newest checkpoint that resume will fall back from. Its manifest
            # still records the previous complete payload size; taking the
            # larger of declared and observed bytes prevents a low estimate.
            "payload_bytes_for_projection": max(payload_bytes, payload_bytes_observed),
            "payload_size_matches_manifest": payload_bytes_observed == payload_bytes,
            "directory": str(manifest_path.parent),
        })
    if recovery_manifest_count == 0:
        raise StorageCapacityError("checkpoint namespace has no matching recovery manifest; fresh initialization is not allowed")
    if not recovery_payload_sizes:
        raise StorageCapacityError("checkpoint namespace has recovery manifests but no readable payload size for capacity projection")
    if candidate is not None and not selected_seen:
        raise StorageCapacityError("hash-verified recovery candidate is absent from the exact checkpoint namespace")
    return {
        "max_payload_bytes": max(item["payload_bytes_for_projection"] for item in observed),
        "selected_recovery_payload_bytes": candidate_bytes if candidate is not None else max(recovery_payload_sizes),
        "selected_recovery_generation": candidate_generation,
        "latest_recovery_manifest_generation": max(
            item["generation"] for item in observed
            if Path(item["directory"]).name == "recovery"
        ),
        "matching_manifest_count": len(observed),
        "observed_generations": observed,
        "size_evidence": (
            "matching identity manifests plus max(manifest-declared, observed payload sizes); selected recovery candidate hash verified"
            if candidate is not None else
            "matching identity manifests plus max(manifest-declared, observed payload sizes); read-only sizing only, payload hashes not checked here"
        ),
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
        "manifest_generations": [generation for generation, _path, _manifest in manifests],
        "wrong_identity_manifest_generations": [
            generation for generation, _path, manifest in manifests
            if manifest.get("identity") != expected_identity
        ],
        "uninspected_manifest_generations": [
            generation for generation, _path, _manifest in manifests[max_candidates:]
        ],
        "unreadable_manifests": unreadable_manifests,
        "integrity_method": "manifest identity plus bounded 8 MiB streaming SHA-256; payload is not deserialized",
        "candidates": candidates,
    }


def _first_run_resume_gate(recovery: dict[str, Any], expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Summarize exact v1.3 recovery evidence without loading its payload object."""
    candidates = recovery.get("candidates", [])
    wrong_identity = [row for row in candidates if row.get("manifest_identity_matches") is not True]
    wrong_identity_generations = recovery.get("wrong_identity_manifest_generations", [])
    verified = [row for row in candidates
                if row.get("manifest_identity_matches") is True
                and row.get("integrity_status") == "PASS_MANIFEST_AND_PAYLOAD_HASH"]
    if recovery.get("error"):
        status, reason = "BLOCKED_UNREADABLE", str(recovery["error"])
    elif wrong_identity or wrong_identity_generations:
        status, reason = "BLOCKED_IDENTITY_MISMATCH", "a recovery manifest in the immutable namespace has a different run identity"
    elif not verified:
        status, reason = "BLOCKED_NO_VALID_RECOVERY", "no exact-identity generation has a verified payload hash and size"
    else:
        status, reason = "PASS_RESUME_CANDIDATE", "at least one exact-identity checkpoint generation passed manifest and payload hash/size checks; runner deserialization remains required"
        if recovery.get("manifest_generation_count", 0) > 2:
            reason += "; additional same-identity manifests are reported but do not block resume (Drive may have deferred pruning); runner still validates and selects the checkpoint"
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
        "retention_overflow_count": max(0, int(recovery.get("manifest_generation_count", 0)) - 2),
        "uninspected_manifest_generations": recovery.get("uninspected_manifest_generations", []),
        "wrong_identity_manifest_generations": wrong_identity_generations,
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


def _verified_cache_candidate(
    cache_root: str | Path, dataset: str, split: str, preflight: dict[str, Any],
):
    """Locate one unambiguous cache for a pinned archive identity, without networking."""
    from revision_r1.data import CACHE_FORMAT_VERSION, _cache_paths

    try:
        frozen = DATASETS[dataset][split]
        expected = preflight[dataset][split]
        archive_md5 = str(expected["md5"])
        archive_sha256 = str(expected["sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"preflight lacks {dataset}/{split} archive identity") from exc
    if archive_md5 != frozen["md5"] or len(archive_sha256) != 64:
        raise ValueError(f"preflight {dataset}/{split} hashes differ from the frozen dataset identity")
    root = Path(cache_root).expanduser().resolve()
    candidates = sorted(root.glob(f"{archive_md5}_*_stream_f32_v{CACHE_FORMAT_VERSION}/manifest.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"multiple local cache manifests match {dataset}/{split} archive MD5 {archive_md5}; refusing ambiguous cache"
        )
    manifest_path = candidates[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"local dataset cache manifest is unreadable: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("archive_md5") != archive_md5
        or manifest.get("archive_sha256") != archive_sha256
        or not isinstance(manifest.get("x_key"), str)
        or not isinstance(manifest.get("y_key"), str)
    ):
        raise ValueError(f"local dataset cache manifest has a different source identity: {manifest_path}")
    paths = _cache_paths(root, archive_md5, manifest["x_key"], manifest["y_key"])
    if paths[2] != manifest_path or not all(path.is_file() and not path.is_symlink() for path in paths):
        raise ValueError(f"local dataset cache is incomplete or unsafe: {manifest_path.parent}")
    return paths, manifest


def inspect_dataset_sources(
    data_roots: dict[str, str | Path], cache_root: str | Path, preflight: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Inventory exact Drive NPZ sources or local cache candidates; never downloads."""
    if set(data_roots) != set(DATASET_ORDER):
        raise ValueError("dataset roots must exactly match the frozen dataset names")
    report: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        root = Path(data_roots[dataset]).expanduser()
        splits: dict[str, Any] = {}
        for split in ("train", "test"):
            source = root / DATASETS[dataset][split]["filename"]
            if source.is_symlink():
                raise ValueError(f"dataset source path must not be a symlink: {source}")
            if source.is_file():
                splits[split] = {
                    "source": "CANONICAL_DRIVE_NPZ",
                    "path": str(source),
                    "sha256_verification": "performed_before_dataset_use",
                }
                continue
            candidate = _verified_cache_candidate(cache_root, dataset, split, preflight)
            if candidate is None:
                raise FileNotFoundError(
                    f"{dataset}/{split} is missing from canonical Drive and has no matching local verified-cache candidate; "
                    f"expected Drive file {source}; no external download will be attempted"
                )
            paths, manifest = candidate
            splits[split] = {
                "source": "LOCAL_ARCHIVE_IDENTITY_MATCHED_CACHE",
                "cache_directory": str(paths[0].parent),
                "archive_sha256": manifest["archive_sha256"],
                "cache_content_hashes": "verified immediately before dataset use",
            }
        report[dataset] = {"data_root": str(root), "splits": splits, "external_downloads": "DISABLED"}
    return report


def _load_verified_cache_split(
    cache_root: str | Path, dataset: str, split: str, preflight: dict[str, Any],
):
    import numpy as np
    from revision_r1.data import DatasetBundle, _load_verified_cache

    candidate = _verified_cache_candidate(cache_root, dataset, split, preflight)
    if candidate is None:
        raise FileNotFoundError(f"no checksum-verified local cache candidate exists for {dataset}/{split}")
    paths, cache_manifest = candidate
    spec = DATASETS[dataset][split]
    x, y, manifest = _load_verified_cache(
        paths, str(preflight[dataset][split]["md5"]), manifest_x := cache_manifest["x_key"],
        manifest_y := cache_manifest["y_key"], tuple(spec["shape"]),
        str(preflight[dataset][split]["sha256"]),
    )
    if x is None or y is None or not isinstance(manifest, dict):
        raise ValueError(f"local cache could not be loaded for {dataset}/{split}")
    if x.dtype != np.dtype("float32") or y.dtype != np.dtype("int64"):
        raise ValueError(f"local cache dtype differs from the production coordinate contract for {dataset}/{split}")
    rows_per_chunk = max(1, (8 << 20) // max(4, int(x.shape[1]) * 4))
    for start in range(0, len(x), rows_per_chunk):
        if not np.isfinite(x[start:start + rows_per_chunk]).all():
            raise ValueError(f"local cache contains nonfinite features for {dataset}/{split}")
    labels, counts = np.unique(y, return_counts=True)
    if not np.array_equal(labels, np.arange(100, dtype=np.int64)):
        raise ValueError(f"local cache labels do not contain exactly classes 0..99 for {dataset}/{split}")
    metadata = {
        "path": str(Path(cache_root).expanduser().resolve() / f"{DATASETS[dataset][split]['filename']} (verified local cache)"),
        "filename": DATASETS[dataset][split]["filename"], "split": split,
        "md5": str(preflight[dataset][split]["md5"]),
        "sha256": str(preflight[dataset][split]["sha256"]),
        "x_key": manifest_x, "y_key": manifest_y,
        "shape": list(x.shape), "dtype": str(x.dtype),
        "source_dtype": str(manifest["x_source_dtype"]),
        "label_dtype": str(y.dtype), "source_label_dtype": str(manifest["y_source_dtype"]),
        "class_count": int(labels.size),
        "class_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "cache_enabled": True, "archive_absent_cache_hashes_verified": True,
    }
    return x, y, metadata


def load_validated_dataset_bundle(
    data_root: str | Path, dataset: str, cache_root: str | Path, preflight: dict[str, Any],
    *, progress_callback=None,
):
    """Load pinned Drive NPZs or their checksum-verified local mmap caches."""
    from revision_r1.data import DatasetBundle, load_npz_arrays, require_memory_mapped_production_bundle

    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    root = Path(data_root).expanduser()
    spec = DATASETS[dataset]
    loaded: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for split in ("train", "test"):
        archive = root / spec[split]["filename"]
        if archive.is_symlink():
            raise ValueError(f"dataset archive must not be a symlink: {archive}")
        if archive.is_file():
            arrays = load_npz_arrays(
                archive, split=split, expected_md5=spec[split]["md5"],
                expected_shape=tuple(spec[split]["shape"]), expected_labels=range(100),
                cache_dir=cache_root, progress_callback=progress_callback,
            )
        else:
            arrays = _load_verified_cache_split(cache_root, dataset, split, preflight)
        if (
            arrays[2].get("md5") != preflight[dataset][split].get("md5")
            or arrays[2].get("sha256") != preflight[dataset][split].get("sha256")
        ):
            raise ValueError(f"{dataset}/{split} content identity differs from mounted-Drive preflight")
        loaded[split] = arrays
    train_x, train_y, train_meta = loaded["train"]
    test_x, test_y, test_meta = loaded["test"]
    if train_x.shape[1] != test_x.shape[1] or train_x.shape[1] != int(spec["feature_dim"]):
        raise ValueError(f"{dataset} train/test feature coordinate dimensions differ")
    bundle = DatasetBundle(dataset, train_x, train_y, test_x, test_y, train_meta, test_meta)

    require_memory_mapped_production_bundle(bundle)
    observed = _bundle_identity(bundle)
    expected = {"namespace": "PRODUCTION", "dataset": dataset, **_preflight_identity(preflight, dataset)}
    if observed != expected:
        raise ValueError(f"{dataset} NPZ identity differs from the mounted-Drive preflight record")
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
    storage_adapter=None,
    current_run_output_path: str | Path | None = None,
    current_run_checkpoint_path: str | Path | None = None,
    current_run_local_checkpoint_path: str | Path | None = None,
    expected_run_identity: dict[str, Any] | None = None,
    verified_dataset_cache_bytes: int = 0,
) -> dict[str, Any]:
    """Inspect exact Drive destinations/account quota and current local free space."""
    if _BOUND_STORAGE_SESSION is not None and api_factory is not None:
        if api_factory() is not _BOUND_DRIVE_API_SERVICE:
            raise StorageCapacityError("capacity check attempted with a Drive client outside the verified storage session")
    output_destination = inspect_drive_destination(
        output_path, api_factory=api_factory, storage_adapter=storage_adapter,
    )
    checkpoint_destination = inspect_drive_destination(
        checkpoint_path, api_factory=api_factory, storage_adapter=storage_adapter,
    )
    same_account = output_destination["quota_account_email"] == checkpoint_destination["quota_account_email"]
    if not same_account:
        raise StorageCapacityError("output and checkpoint roots were inspected under different authenticated quota accounts")
    local_directory = _nearest_existing_directory(local_path)
    local_usage = shutil.disk_usage(local_directory)
    current_drive_bytes = 0
    if current_run_output_path is not None:
        current_drive_bytes += (
            storage_adapter.tree_bytes(storage_adapter.logical_path_for_local(current_run_output_path))
            if storage_adapter is not None else _tree_file_bytes(current_run_output_path)
        )
    if current_run_checkpoint_path is not None:
        current_drive_bytes += (
            storage_adapter.tree_bytes(storage_adapter.logical_path_for_local(current_run_checkpoint_path))
            if storage_adapter is not None else
            _tree_file_bytes(current_run_checkpoint_path, expected_identity=expected_run_identity)
        )
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
        "permanent_recovery_delete_verified": output_destination.get("permanent_recovery_delete_verified") is True and checkpoint_destination.get("permanent_recovery_delete_verified") is True,
        "new_file_owner_verified": output_destination.get("new_file_owned_by_authenticated_user") is True and checkpoint_destination.get("new_file_owned_by_authenticated_user") is True,
        "quota_metadata_verified": output_destination.get("quota_status") == "FINITE_VERIFIED" and checkpoint_destination.get("quota_status") == "FINITE_VERIFIED",
        "quota_status": "FINITE_VERIFIED" if drive_available is not None else "UNKNOWN_CAPACITY",
        "quota_capacity_note": (
            output_destination.get("quota_capacity_note") or checkpoint_destination.get("quota_capacity_note")
            if drive_available is None else None
        ),
        "authenticated_user": output_destination["authenticated_user"],
        "quota_account_email": output_destination["quota_account_email"],
        "canonical_storage_identity_verified": bool(
            _BOUND_STORAGE_SESSION is not None
            and _BOUND_STORAGE_SESSION.get("status") == "PASS"
            and _BOUND_STORAGE_SESSION.get("storage_account_email") == CANONICAL_STORAGE_ACCOUNT_EMAIL
            and _BOUND_STORAGE_SESSION.get("canonical_revision_folder_id") == CANONICAL_REVISION_FOLDER_ID
            and output_destination["quota_account_email"] == CANONICAL_STORAGE_ACCOUNT_EMAIL
            and checkpoint_destination["quota_account_email"] == CANONICAL_STORAGE_ACCOUNT_EMAIL
        ),
        "canonical_revision_folder_id": (
            _BOUND_STORAGE_SESSION.get("canonical_revision_folder_id") if _BOUND_STORAGE_SESSION else None
        ),
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
    if len(manifests) < expected_count or (kind != "recovery" and len(manifests) != expected_count):
        raise ValueError(
            f"production checkpoint retention mismatch at {directory}: "
            f"expected {expected_count}, found {len(manifests)}"
        )
    payload_bytes: list[int] = []
    manifest_bytes = 0
    kinds: list[str] = []
    generations: list[int] = []
    orphan_manifests: list[dict[str, Any]] = []
    referenced_payload_names: set[str] = set()
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"production checkpoint manifest is unreadable: {manifest_path}") from exc
        match = re.fullmatch(r"generation-(\d+)\.manifest\.json", manifest_path.name)
        generation = int(match.group(1)) if match else None
        payload_name = manifest.get("payload_file") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or generation is None
            or generation < 1
            or manifest_path.name != f"generation-{generation:06d}.manifest.json"
            or manifest.get("identity") != expected_identity
            or manifest.get("identity_sha256") != _sha256_json(expected_identity)
            or not isinstance(payload_name, str)
            or payload_name != f"generation-{generation:06d}.pt"
            or type(manifest.get("payload_bytes")) is not int
            or manifest["payload_bytes"] <= 0
            or not _is_sha256(manifest.get("payload_sha256"))
            or not isinstance(manifest.get("kind"), str)
        ):
            raise ValueError(f"production checkpoint identity/manifest is invalid: {manifest_path}")
        payload = directory / payload_name
        manifest_size = manifest_path.stat().st_size
        manifest_bytes += manifest_size
        if not payload.is_file():
            if kind != "recovery":
                raise ValueError(
                    f"production checkpoint payload is missing or size-mismatched: {payload}"
                )
            if manifest["kind"] != "recovery":
                raise ValueError(
                    f"manifest-only recovery entry has unexpected kind {manifest['kind']!r}: {manifest_path}"
                )
            orphan_manifests.append({
                "generation": generation,
                "name": manifest_path.name,
                "size_bytes": manifest_size,
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "payload_file": payload_name,
                "payload_bytes_expected": manifest["payload_bytes"],
                "payload_sha256_expected": manifest["payload_sha256"],
            })
            continue
        if payload.stat().st_size != manifest["payload_bytes"]:
            raise ValueError(f"production checkpoint payload is size-mismatched: {payload}")
        payload_bytes.append(payload.stat().st_size)
        generations.append(generation)
        referenced_payload_names.add(payload_name)
        kinds.append(str(manifest.get("kind")))
    if len(generations) != expected_count:
        raise ValueError(
            f"production checkpoint retention mismatch at {directory}: "
            f"expected {expected_count}, found {len(generations)} complete generations"
        )
    actual_payload_names = {path.name for path in directory.glob("generation-*.pt") if path.is_file()}
    if actual_payload_names != referenced_payload_names:
        raise ValueError(
            f"production checkpoint payload inventory differs from manifests at {directory}: "
            f"unreferenced={sorted(actual_payload_names - referenced_payload_names)}, "
            f"missing={sorted(referenced_payload_names - actual_payload_names)}"
        )
    orphan_manifests.sort(key=lambda item: item["generation"])
    if orphan_manifests and (
        kind != "recovery"
        or any(item["generation"] >= min(generations) for item in orphan_manifests)
    ):
        raise ValueError(
            f"manifest-only recovery entries must be older than both retained recovery generations: "
            f"orphans={[item['generation'] for item in orphan_manifests]}, retained={sorted(generations)}"
        )
    if kind == "recovery" and "complete" not in kinds:
        raise ValueError("retained production recovery generations do not include the terminal completion checkpoint")
    if kind == "task_boundaries" and any(value != "task-boundary" for value in kinds):
        raise ValueError("task-boundary checkpoint directory contains a non-boundary generation")
    return {
        "generation_count": len(generations), "generations": sorted(generations),
        "payload_bytes": sum(payload_bytes),
        "max_payload_bytes": max(payload_bytes, default=0), "manifest_bytes": manifest_bytes,
        "generation_kinds": kinds,
        "manifest_only_orphan_generations": [item["generation"] for item in orphan_manifests],
        "manifest_only_orphan_count": len(orphan_manifests),
        "manifest_only_orphan_bytes": sum(item["size_bytes"] for item in orphan_manifests),
        "manifest_only_orphans": orphan_manifests,
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


def _list_drive_folder_files(service, parent_id: str) -> list[dict[str, Any]]:
    query = f"'{_drive_query_literal(parent_id)}' in parents and trashed=false"
    fields = (
        "nextPageToken,files(id,name,mimeType,parents,owners(emailAddress,displayName),ownedByMe,"
        "size,quotaBytesUsed,capabilities(canDelete,canEdit))"
    )
    rows: list[dict[str, Any]] = []
    page_token = None
    try:
        while True:
            response = service.files().list(
                q=query, pageSize=1000, pageToken=page_token, fields=fields,
                includeItemsFromAllDrives=True, supportsAllDrives=True,
            ).execute()
            rows.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return rows
    except Exception as exc:
        raise StorageCapacityError(f"cannot inventory exact Drive recovery folder {parent_id}: {exc}") from exc


def _drive_checkpoint_file_record(
    path: Path, metadata: dict[str, Any], parent_id: str, *, expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    from revision_r1.validation import file_sha256

    owners = metadata.get("owners") or []
    owner_records = [
        {
            "email": str(owner.get("emailAddress", "")).strip().lower() or None,
            "display_name": owner.get("displayName"),
        }
        for owner in owners if isinstance(owner, dict)
    ]
    size = _optional_nonnegative_int(metadata.get("size"), "checkpoint file size")
    if (
        metadata.get("name") != path.name
        or str(parent_id) not in [str(value) for value in metadata.get("parents", [])]
        or not metadata.get("id")
        or metadata.get("mimeType") == DRIVE_FOLDER_MIME
        or size is None
    ):
        raise StorageCapacityError(f"Drive metadata does not prove the exact checkpoint file identity/size: {path.name}")
    if expected_bytes is not None and size != int(expected_bytes):
        raise StorageCapacityError(f"Drive checkpoint size differs from its verified generation record: {path.name}")
    quota_bytes = _optional_nonnegative_int(metadata.get("quotaBytesUsed"), "checkpoint quota bytes")
    local_hash = None
    if path.is_file():
        if path.stat().st_size != size:
            raise StorageCapacityError(f"mounted Drive checkpoint size differs from API metadata: {path.name}")
        local_hash = file_sha256(path)
        if expected_sha256 is not None and local_hash != expected_sha256:
            raise StorageCapacityError(f"mounted Drive checkpoint hash differs from its verified generation record: {path.name}")
    elif expected_sha256 is not None:
        raise StorageCapacityError(f"checkpoint file disappeared from the mounted Drive view: {path.name}")
    return {
        "name": path.name, "drive_file_id": str(metadata["id"]),
        "parent_id": str(parent_id), "mime_type": metadata.get("mimeType"),
        "owners": owner_records, "owned_by_authenticated_user": metadata.get("ownedByMe"),
        "can_delete": (metadata.get("capabilities") or {}).get("canDelete"),
        "can_edit": (metadata.get("capabilities") or {}).get("canEdit"),
        "size_bytes": size, "quota_bytes_used": quota_bytes,
        "local_sha256": local_hash,
    }


def _inspect_verified_recovery_tree(
    recovery: Path, identity: dict[str, Any], service, parent_id: str, *,
    progress_callback=None,
) -> dict[str, list[dict[str, Any]]]:
    from revision_r1.validation import file_sha256

    manifest_paths = sorted(recovery.glob("generation-*.manifest.json"))
    local_generation_files = {
        path.name for path in recovery.iterdir()
        if path.is_file() and path.name.startswith("generation-")
    }
    remote_generation_files: dict[str, list[dict[str, Any]]] = {}
    for metadata in _list_drive_folder_files(service, parent_id):
        name = metadata.get("name")
        if isinstance(name, str) and name.startswith("generation-"):
            remote_generation_files.setdefault(name, []).append(metadata)
    if set(remote_generation_files) != local_generation_files:
        raise StorageCapacityError(
            "mounted recovery generation inventory differs from exact Drive folder metadata; "
            f"Drive-only={sorted(set(remote_generation_files) - local_generation_files)}, "
            f"mount-only={sorted(local_generation_files - set(remote_generation_files))}"
        )
    duplicates = sorted(name for name, rows in remote_generation_files.items() if len(rows) != 1)
    if duplicates:
        raise StorageCapacityError(f"Drive returned duplicate generation filenames in the recovery folder: {duplicates}")
    if not manifest_paths:
        raise ValueError(f"no recovery manifests exist under {recovery}")

    inventory: list[dict[str, Any]] = []
    orphan_manifests: list[dict[str, Any]] = []
    seen_generations: set[int] = set()
    referenced_payloads: set[str] = set()
    for manifest_path in manifest_paths:
        match = re.fullmatch(r"generation-(\d+)\.manifest\.json", manifest_path.name)
        if match is None:
            raise ValueError(f"recovery manifest filename is malformed: {manifest_path.name}")
        generation = int(match.group(1))
        if generation < 1 or generation in seen_generations:
            raise ValueError(f"recovery generation number is invalid or duplicated: {generation}")
        seen_generations.add(generation)
        if manifest_path.name != f"generation-{generation:06d}.manifest.json":
            raise ValueError(f"recovery manifest does not use its canonical generation name: {manifest_path.name}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"recovery manifest is unreadable: {manifest_path}") from exc
        expected_payload_name = f"generation-{generation:06d}.pt"
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or type(manifest.get("generation")) is not int
            or manifest.get("generation") != generation
            or manifest.get("identity") != identity
            or manifest.get("identity_sha256") != _sha256_json(identity)
            or manifest.get("payload_file") != expected_payload_name
            or type(manifest.get("payload_bytes")) is not int
            or manifest["payload_bytes"] <= 0
            or not _is_sha256(manifest.get("payload_sha256"))
            or not isinstance(manifest.get("kind"), str)
        ):
            raise ValueError(f"recovery manifest identity, generation, or payload metadata is invalid: {manifest_path}")
        payload_path = recovery / expected_payload_name
        manifest_hash = file_sha256(manifest_path)
        manifest_metadata = remote_generation_files[manifest_path.name][0]
        manifest_file = _drive_checkpoint_file_record(
            manifest_path, manifest_metadata, parent_id,
            expected_bytes=manifest_path.stat().st_size, expected_sha256=manifest_hash,
        )
        if not payload_path.is_file():
            if expected_payload_name in remote_generation_files:
                raise StorageCapacityError(
                    f"Drive lists recovery payload {expected_payload_name} but it is absent from the mounted folder"
                )
            orphan_manifests.append({
                "generation": generation,
                "kind": manifest["kind"],
                "identity_sha256": manifest["identity_sha256"],
                "payload_file": expected_payload_name,
                "payload_bytes": manifest["payload_bytes"],
                "payload_sha256": manifest["payload_sha256"],
                "manifest_sha256": manifest_hash,
                "manifest_file": manifest_file,
                "integrity_status": "PASS_MANIFEST_ONLY_PAYLOAD_MISSING",
            })
            if progress_callback is not None:
                progress_callback({
                    "stage": "recovery_orphan_manifest_verified",
                    "generation": generation,
                    "manifest_sha256": manifest_hash,
                    "missing_payload": expected_payload_name,
                    "manifest_file": manifest_file,
                })
            continue
        if payload_path.stat().st_size != manifest["payload_bytes"]:
            raise ValueError(f"recovery payload byte count differs from its manifest: {payload_path}")
        if progress_callback is not None:
            progress_callback({"stage": "recovery_payload_hash_start", "generation": generation, "payload_file": expected_payload_name})
        payload_hash = file_sha256(payload_path)
        if payload_hash != manifest["payload_sha256"]:
            raise ValueError(f"recovery payload SHA-256 differs from its manifest: {payload_path}")
        payload_metadata = remote_generation_files[expected_payload_name][0]
        payload_file = _drive_checkpoint_file_record(
            payload_path, payload_metadata, parent_id,
            expected_bytes=manifest["payload_bytes"], expected_sha256=manifest["payload_sha256"],
        )
        manifest_file["local_sha256"] = manifest_hash
        row = {
            "generation": generation, "kind": manifest["kind"],
            "identity": manifest["identity"], "identity_sha256": manifest["identity_sha256"],
            "manifest_sha256": manifest_hash, "payload_sha256": payload_hash,
            "payload_bytes": manifest["payload_bytes"],
            "integrity_status": "PASS_MANIFEST_AND_PAYLOAD_HASH",
            "files": {"manifest": manifest_file, "payload": payload_file},
        }
        inventory.append(row)
        referenced_payloads.add(expected_payload_name)
        if progress_callback is not None:
            progress_callback({
                "stage": "recovery_generation_verified", "generation": generation,
                "kind": manifest["kind"], "manifest_sha256": manifest_hash,
                "payload_sha256": payload_hash, "payload_bytes": manifest["payload_bytes"],
                "files": row["files"],
            })
    payload_names = {
        path.name for path in recovery.glob("generation-*.pt") if path.is_file()
    }
    if payload_names != referenced_payloads:
        raise ValueError(
            "recovery payload inventory contains missing or unreferenced files: "
            f"unreferenced={sorted(payload_names - referenced_payloads)}, "
            f"missing={sorted(referenced_payloads - payload_names)}"
        )
    inventory.sort(key=lambda item: item["generation"])
    orphan_manifests.sort(key=lambda item: item["generation"])
    return {"verified_generations": inventory, "orphan_manifests": orphan_manifests}


def _rotation_deferred_pruning_evidence(
    output_root: Path, plan: dict[str, Any], record: dict[str, Any], stale_generations: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_names = {
        file_record["name"]
        for generation in stale_generations for file_record in generation["files"].values()
    }
    event_root = output_root / "progress" / plan["plan_sha256"] / "events"
    event_paths = sorted(event_root.glob(f"{record['run_key']}_{record['config_sha256'][:12]}_*.jsonl"))
    matched: dict[tuple[int, str], str] = {}
    for event_path in event_paths:
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            checkpoint = event.get("last_verified_drive_checkpoint")
            deferred = checkpoint.get("pruning_deferred_files", []) if isinstance(checkpoint, dict) else []
            for item in deferred if isinstance(deferred, list) else []:
                if not isinstance(item, dict) or item.get("file") not in expected_names:
                    continue
                try:
                    key = (int(item["generation"]), str(item["file"]))
                except (KeyError, TypeError, ValueError):
                    continue
                matched[key] = str(event_path)
    if matched:
        files = [
            {"generation": generation, "file": filename, "event_log": event_path}
            for (generation, filename), event_path in sorted(matched.items())
        ]
        return {
            "status": "CONFIRMED_FROM_RUN_PROGRESS",
            "reason": "checkpoint rotation recorded Drive deletion deferrals for files still present in this recovery inventory",
            "deferred_files_still_present": files,
        }
    return {
        "status": "CODE_PATH_CONFIRMED_RUN_LOG_NOT_CORROBORATED",
        "reason": (
            "checkpoint rotation preserves stale files not owned by the canonical storage account or without "
            "verified Drive canDelete; available per-attempt progress logs did not identify the currently present files"
        ),
        "deferred_files_still_present": [],
    }


def _finalization_journal_digest(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("journal_sha256", None)
    return _sha256_json(payload)


def _journal_transaction_key(document: dict[str, Any]) -> str:
    fields = ("plan_sha256", "run_key", "run_id", "identity_sha256")
    transaction = {key: document.get(key) for key in fields}
    if any(not isinstance(value, str) or not value for value in transaction.values()):
        raise RecoveryJournalPublicationError(
            "retention journal lacks immutable transaction identity",
            phase="transaction-identity", expected={"required_fields": list(fields)},
            observed=transaction,
        )
    return _sha256_json(transaction)


def _journal_retryable_drive_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, TimeoutError, ConnectionError)):
        return True
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return status in {408, 409, 429} or 500 <= status <= 599


def _journal_observed_metadata(metadata: dict[str, Any] | None, payload: bytes | None) -> dict[str, Any]:
    remote_doc = None
    if isinstance(payload, (bytes, bytearray)):
        try:
            value = json.loads(bytes(payload).decode("utf-8"))
            remote_doc = value if isinstance(value, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        size_bytes = _optional_nonnegative_int(metadata.get("size"), "journal observed size") if isinstance(metadata, dict) else None
    except StorageCapacityError:
        size_bytes = metadata.get("size") if isinstance(metadata, dict) else None
    return {
        "file_id": metadata.get("id") if isinstance(metadata, dict) else None,
        "name": metadata.get("name") if isinstance(metadata, dict) else None,
        "parent_ids": metadata.get("parents") if isinstance(metadata, dict) else None,
        "mime_type": metadata.get("mimeType") if isinstance(metadata, dict) else None,
        "size_bytes": size_bytes,
        "md5_checksum": metadata.get("md5Checksum") if isinstance(metadata, dict) else None,
        "sha256": hashlib.sha256(bytes(payload)).hexdigest() if isinstance(payload, (bytes, bytearray)) else None,
        "journal_revision": remote_doc.get("journal_revision") if remote_doc else None,
        "app_properties": metadata.get("appProperties") if isinstance(metadata, dict) else None,
        "drive_version": metadata.get("version") if isinstance(metadata, dict) else None,
        "owners": metadata.get("owners", []) if isinstance(metadata, dict) else [],
    }


def _journal_error(
    message: str, *, phase: str, expected: dict[str, Any],
    metadata: dict[str, Any] | None = None, payload: bytes | None = None,
    observed: dict[str, Any] | None = None,
) -> RecoveryJournalPublicationError:
    actual = _journal_observed_metadata(metadata, payload)
    if observed:
        actual.update(observed)
    return RecoveryJournalPublicationError(
        message, phase=phase, expected=expected, observed=actual,
    )


def _list_drive_journal_children(service, parent_id: str, name: str) -> list[dict[str, Any]]:
    query = (
        f"'{_drive_query_literal(parent_id)}' in parents and name='{_drive_query_literal(name)}' "
        "and trashed=false"
    )
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        rows: list[dict[str, Any]] = []
        page_token = None
        try:
            while True:
                response = service.files().list(
                    q=query, pageSize=100, pageToken=page_token,
                    fields=("nextPageToken,files(id,name,mimeType,parents,size,md5Checksum,version,"
                            "owners(emailAddress,displayName),appProperties,capabilities(canEdit,canDelete))"),
                    includeItemsFromAllDrives=True, supportsAllDrives=True,
                ).execute()
                rows.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    return rows
        except Exception as exc:
            if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                raise RecoveryJournalPublicationError(
                    f"Drive could not list the exact retention-journal parent/name: {exc}",
                    phase="journal-name-resolution",
                    expected={"parent_id": parent_id, "name": name},
                    observed={"retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                ) from exc
    return []


def _get_drive_journal_metadata(service, file_id: str) -> dict[str, Any]:
    last_error = None
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            metadata = service.files().get(
                fileId=file_id,
                fields=("id,name,mimeType,parents,size,md5Checksum,version,owners(emailAddress,displayName),"
                        "appProperties,capabilities(canEdit,canDelete)"),
                supportsAllDrives=True,
            ).execute()
        except Exception as exc:
            last_error = exc
            if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                raise RecoveryJournalPublicationError(
                    f"Drive could not read retention-journal metadata by exact file ID: {exc}",
                    phase="journal-metadata-read", expected={"file_id": file_id},
                    observed={"retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                ) from exc
            continue
        if not isinstance(metadata, dict):
            raise RecoveryJournalPublicationError(
                "Drive returned malformed retention-journal metadata",
                phase="journal-metadata-read", expected={"file_id": file_id},
                observed={"metadata_type": type(metadata).__name__},
            )
        return metadata
    raise RecoveryJournalPublicationError(
        f"Drive metadata read did not converge: {last_error}", phase="journal-metadata-read",
        expected={"file_id": file_id}, observed={"retryable": True},
    )


def _get_drive_journal_payload(service, file_id: str, *, expected: dict[str, Any]) -> bytes:
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            # Use the discovery client's generated media method. Calling
            # files().get(..., alt="media") still builds the JSON-model
            # metadata endpoint in googleapiclient; the API then returns a
            # File resource (alt=json), not the stored journal bytes.
            request = service.files().get_media(
                fileId=file_id, supportsAllDrives=True,
            )
            # Journal verification requires byte-for-byte content, so bypass
            # HttpRequest.execute()'s JSON deserializer and stream raw media.
            from googleapiclient.http import MediaIoBaseDownload

            output = io.BytesIO()
            downloader = MediaIoBaseDownload(output, request)
            done = False
            while not done:
                _progress, done = downloader.next_chunk()
            payload = output.getvalue()
        except Exception as exc:
            if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                raise RecoveryJournalPublicationError(
                    f"Drive could not read retention-journal bytes by exact file ID: {exc}",
                    phase="journal-payload-read", expected=expected,
                    observed={"file_id": file_id, "retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                ) from exc
            continue
        if not isinstance(payload, (bytes, bytearray)):
            raise RecoveryJournalPublicationError(
                "Drive returned non-byte retention-journal content",
                phase="journal-payload-read", expected=expected,
                observed={"file_id": file_id, "payload_type": type(payload).__name__},
            )
        return bytes(payload)
    raise RecoveryJournalPublicationError(
        "Drive journal payload read did not converge", phase="journal-payload-read",
        expected=expected, observed={"file_id": file_id},
    )


def _verify_drive_journal_location(metadata: dict[str, Any], *, file_id: str, parent_id: str, name: str) -> None:
    parent_ids = [str(value) for value in metadata.get("parents", [])]
    if (
        str(metadata.get("id") or "") != file_id
        or metadata.get("name") != name
        or parent_ids != [parent_id]
        or metadata.get("mimeType") in {DRIVE_FOLDER_MIME, DRIVE_SHORTCUT_MIME}
    ):
        raise RecoveryJournalPublicationError(
            "Drive journal file ID, name, type, or parent differs from the exact binding",
            phase="journal-location-verification",
            expected={"file_id": file_id, "parent_id": parent_id, "name": name},
            observed=_journal_observed_metadata(metadata, None),
        )


def _verify_remote_journal_identity(
    remote: dict[str, Any], local: dict[str, Any], *, file_id: str, parent_id: str,
) -> None:
    if (
        remote.get("journal_sha256") != _finalization_journal_digest(remote)
        or _journal_transaction_key(remote) != _journal_transaction_key(local)
        or remote.get("journal_drive_file_id") not in {None, file_id}
        or remote.get("journal_drive_parent_id") not in {None, parent_id}
    ):
        raise RecoveryJournalPublicationError(
            "existing Drive journal bytes are corrupt or belong to a different finalization transaction",
            phase="journal-existing-content-verification",
            expected={"file_id": file_id, "parent_id": parent_id, "transaction_key": _journal_transaction_key(local)},
            observed={
                "remote_transaction_key": _journal_transaction_key(remote),
                "remote_file_id": remote.get("journal_drive_file_id"),
                "remote_parent_id": remote.get("journal_drive_parent_id"),
                "remote_journal_sha256": remote.get("journal_sha256"),
                "calculated_journal_sha256": _finalization_journal_digest(remote),
                "journal_revision": remote.get("journal_revision"),
            },
        )


def _read_valid_remote_journal(
    service, metadata: dict[str, Any], *, document: dict[str, Any], parent_id: str,
) -> tuple[dict[str, Any] | None, bytes, dict[str, Any]]:
    file_id = str(metadata["id"])
    expected = {
        "file_id": file_id, "parent_id": parent_id, "name": metadata.get("name"),
        "transaction_key": _journal_transaction_key(document),
    }
    payload = b""
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        metadata = _get_drive_journal_metadata(service, file_id)
        _verify_drive_journal_location(metadata, file_id=file_id, parent_id=parent_id, name=str(expected["name"]))
        payload = _get_drive_journal_payload(service, file_id, expected=expected)
        marker = (metadata.get("appProperties") or {}).get("revision_r1_finalization_key")
        if not payload and marker == expected["transaction_key"]:
            return None, payload, metadata
        try:
            remote_size = _optional_nonnegative_int(metadata.get("size"), "journal remote size")
        except StorageCapacityError:
            remote_size = None
        md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        metadata_matches = (
            remote_size == len(payload)
            and (not metadata.get("md5Checksum") or metadata.get("md5Checksum") == md5)
        )
        if metadata_matches:
            break
        if attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
            raise _journal_error(
                "Drive journal metadata remained stale/inconsistent with exact API bytes",
                phase="journal-existing-metadata-verification",
                expected={**expected, "size_bytes": len(payload), "md5_checksum": md5},
                metadata=metadata, payload=payload,
            )
    try:
        remote = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _journal_error(
            "existing Drive journal is neither the recognized empty placeholder nor valid JSON",
            phase="journal-existing-content-verification", expected=expected,
            metadata=metadata, payload=payload,
        ) from exc
    if not isinstance(remote, dict):
        raise _journal_error(
            "existing Drive journal is not a JSON object", phase="journal-existing-content-verification",
            expected=expected, metadata=metadata, payload=payload,
        )
    _verify_remote_journal_identity(remote, document, file_id=file_id, parent_id=parent_id)
    return remote, payload, metadata


def _journal_media_upload(payload: bytes):
    try:
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError as exc:
        raise RecoveryJournalPublicationError(
            "google-api-python-client is unavailable for Drive journal upload",
            phase="journal-api-upload", expected={"dependency": "google-api-python-client"},
        ) from exc
    return MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/json", resumable=False)


def _serialize_journal_for_drive(
    document: dict[str, Any], *, file_id: str, parent_id: str, revision: int,
) -> tuple[dict[str, Any], bytes]:
    candidate = dict(document)
    candidate["journal_drive_file_id"] = file_id
    candidate["journal_drive_parent_id"] = parent_id
    candidate["journal_revision"] = revision
    candidate.pop("journal_sha256", None)
    candidate["journal_sha256"] = _finalization_journal_digest(candidate)
    # Match _atomic_json byte-for-byte (it intentionally emits no trailing newline).
    payload = json.dumps(candidate, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    return candidate, payload


def _verify_journal_api_readback(
    service, *, metadata: dict[str, Any], document: dict[str, Any], payload: bytes,
    file_id: str, parent_id: str, name: str, update_response: dict[str, Any] | None,
) -> dict[str, Any]:
    revision = int(document["journal_revision"])
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    update_version = update_response.get("version") if isinstance(update_response, dict) else None
    expected = {
        "file_id": file_id, "parent_id": parent_id, "name": name,
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "md5_checksum": expected_md5, "journal_revision": revision,
        "drive_version": update_version,
    }
    last_metadata, last_payload, last_mismatch = metadata, None, "stale read-back"
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            last_metadata = service.files().get(
                fileId=file_id,
                fields=("id,name,mimeType,parents,size,md5Checksum,version,owners(emailAddress,displayName),appProperties"),
                supportsAllDrives=True,
            ).execute()
            _verify_drive_journal_location(last_metadata, file_id=file_id, parent_id=parent_id, name=name)
            last_payload = _get_drive_journal_payload(
                service, file_id, expected=expected,
            )
        except RecoveryJournalPublicationError:
            raise
        except Exception as exc:
            last_mismatch = f"Drive API read-back request failed: {exc}"
            if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                raise _journal_error(
                    last_mismatch, phase="journal-api-readback", expected=expected,
                    metadata=last_metadata, payload=last_payload,
                    observed={"retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                ) from exc
            continue
        if not isinstance(last_payload, (bytes, bytearray)):
            raise _journal_error(
                "Drive API returned non-byte journal read-back", phase="journal-api-readback",
                expected=expected, metadata=last_metadata,
                observed={"payload_type": type(last_payload).__name__},
            )
        last_payload = bytes(last_payload)
        observed = _journal_observed_metadata(last_metadata, last_payload)
        app_properties = last_metadata.get("appProperties") or {}
        try:
            observed_size = _optional_nonnegative_int(last_metadata.get("size"), "journal size")
            observed_version = int(last_metadata["version"]) if last_metadata.get("version") is not None else None
            expected_version = int(update_version) if update_version is not None else None
        except (StorageCapacityError, TypeError, ValueError):
            observed_size, observed_version, expected_version = None, None, None
        identity_ok = (
            str(last_metadata.get("id") or "") == file_id
            and last_metadata.get("name") == name
            and parent_id in [str(value) for value in last_metadata.get("parents", [])]
            and last_metadata.get("mimeType") not in {DRIVE_FOLDER_MIME, DRIVE_SHORTCUT_MIME}
        )
        revision_ok = (
            app_properties.get("revision_r1_finalization_key") == _journal_transaction_key(document)
            and str(app_properties.get("revision_r1_journal_revision") or "") == str(revision)
            and observed.get("journal_revision") == revision
        )
        bytes_ok = (
            observed_size == len(payload)
            and last_payload == payload
            and (not last_metadata.get("md5Checksum") or last_metadata.get("md5Checksum") == expected_md5)
        )
        version_ok = expected_version is None or observed_version == expected_version
        if identity_ok and revision_ok and bytes_ok and version_ok:
            return last_metadata
        if not identity_ok:
            raise _journal_error(
                "Drive journal read-back has a different ID, name, or parent",
                phase="journal-api-readback-identity", expected=expected,
                metadata=last_metadata, payload=last_payload,
            )
        if expected_version is not None and observed_version is not None and observed_version > expected_version:
            raise _journal_error(
                "Drive journal version advanced unexpectedly during publication",
                phase="journal-api-readback-conflict", expected=expected,
                metadata=last_metadata, payload=last_payload,
            )
        last_mismatch = (
            f"expected_size={len(payload)}, observed_size={observed_size}, "
            f"expected_sha256={expected['sha256']}, observed_sha256={observed.get('sha256')}, "
            f"expected_journal_revision={revision}, observed_journal_revision={observed.get('journal_revision')}"
        )
    raise _journal_error(
        f"Drive journal publication/read-back did not converge after bounded retries: {last_mismatch}",
        phase="journal-api-readback", expected=expected, metadata=last_metadata, payload=last_payload,
    )


def _save_recovery_finalization_journal(
    path: Path, document: dict[str, Any], service, *, drive_root: Path,
) -> int:
    if not isinstance(document, dict):
        raise RecoveryJournalPublicationError(
            "retention journal is not a JSON object", phase="journal-local-validation",
            expected={"document_type": "dict"}, observed={"document_type": type(document).__name__},
        )
    transaction_key = _journal_transaction_key(document)
    baseline = None
    if path.is_file():
        try:
            baseline = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryJournalPublicationError(
                f"existing local retention journal is unreadable and was preserved: {exc}",
                phase="journal-local-baseline-read", expected={"path": str(path)},
            ) from exc
        if (
            not isinstance(baseline, dict)
            or baseline.get("journal_sha256") != _finalization_journal_digest(baseline)
            or _journal_transaction_key(baseline) != transaction_key
        ):
            raise RecoveryJournalPublicationError(
                "existing local retention journal failed checksum or transaction validation; preserved",
                phase="journal-local-baseline-validation",
                expected={"path": str(path), "transaction_key": transaction_key},
                observed={
                    "stored_sha256": baseline.get("journal_sha256") if isinstance(baseline, dict) else None,
                    "calculated_sha256": _finalization_journal_digest(baseline) if isinstance(baseline, dict) else None,
                },
            )
    local_revision = int((baseline or document).get("journal_revision", 0))
    if local_revision < 0:
        raise RecoveryJournalPublicationError(
            "local journal has an invalid negative revision", phase="journal-local-baseline-validation",
            expected={"minimum_revision": 0}, observed={"journal_revision": local_revision},
        )
    quota = _drive_account_snapshot(service)
    available = quota.get("available_bytes")
    if available is None:
        raise StorageCapacityError(
            "Drive quota is unknown; cannot safely persist the recovery-retention journal before pruning"
        )
    rough_payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    required_bytes = max(4096, (len(rough_payload) + 2048) * 2)
    require_available_bytes(
        path, required_bytes=required_bytes, reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=int(available), storage_kind="Drive recovery-finalization journal",
    )
    local_available = shutil.disk_usage(drive_root).free
    require_available_bytes(
        path, required_bytes=required_bytes, reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=local_available, storage_kind="Colab local journal staging",
    )
    # A mounted-path read-back only verifies the Colab mount cache. Before any
    # checkpoint is pruned, prove that Drive itself sees this exact journal and
    # can return its exact bytes. Bind the Drive file/folder IDs into the journal
    # so retries cannot silently switch to a same-name replacement.
    _configured, components = _logical_drive_path_components(path.parent, drive_root=drive_root)
    location = _ensure_drive_folder_path(service, components)
    parent_id = str(location.get("folder_id") or "")
    if location.get("path_exists") is not True or not parent_id:
        raise RecoveryJournalPublicationError(
            "cannot resolve the exact Drive folder for the recovery-retention journal",
            phase="journal-parent-resolution", expected={"path": str(path.parent)},
            observed={"path_exists": location.get("path_exists"), "folder_id": parent_id},
        )

    matches = _list_drive_journal_children(service, parent_id, path.name)
    if len(matches) > 1:
        raise RecoveryJournalPublicationError(
            "duplicate Drive files have the exact retention-journal name; refusing filename-based selection",
            phase="journal-name-ambiguity",
            expected={"parent_id": parent_id, "name": path.name, "bound_file_id": document.get("journal_drive_file_id")},
            observed={"matching_file_ids": [row.get("id") for row in matches]},
        )
    bound_parent = document.get("journal_drive_parent_id")
    if bound_parent and str(bound_parent) != parent_id:
        raise RecoveryJournalPublicationError(
            "journal is bound to a different Drive parent", phase="journal-parent-binding",
            expected={"parent_id": parent_id}, observed={"bound_parent_id": str(bound_parent)},
        )
    bound_id = str(document.get("journal_drive_file_id") or "")
    if bound_id:
        if matches and str(matches[0].get("id") or "") != bound_id:
            raise RecoveryJournalPublicationError(
                "Drive name lookup disagrees with the journal's bound file ID",
                phase="journal-file-id-binding", expected={"file_id": bound_id, "parent_id": parent_id},
                observed={"matching_file_id": matches[0].get("id")},
            )
        file_id = bound_id
        metadata = _get_drive_journal_metadata(service, file_id)
        _verify_drive_journal_location(metadata, file_id=file_id, parent_id=parent_id, name=path.name)
    elif matches:
        file_id = str(matches[0].get("id") or "")
        if not file_id:
            raise RecoveryJournalPublicationError(
                "Drive exact-name journal record has no file ID", phase="journal-file-id-binding",
                expected={"parent_id": parent_id, "name": path.name}, observed={"metadata": matches[0]},
            )
        metadata = _get_drive_journal_metadata(service, file_id)
        _verify_drive_journal_location(metadata, file_id=file_id, parent_id=parent_id, name=path.name)
    else:
        if location.get("can_add_children") is not True:
            raise RecoveryJournalPublicationError(
                "authenticated editor lacks verified child-create permission at the exact journal parent",
                phase="journal-create-permission", expected={"parent_id": parent_id, "can_add_children": True},
                observed={"can_add_children": location.get("can_add_children"), "owner_email": location.get("owner_email")},
            )
        try:
            generated = service.files().generateIds(count=1, space="drive", fields="ids").execute()
            ids = generated.get("ids", []) if isinstance(generated, dict) else []
            file_id = str(ids[0]) if ids else ""
        except Exception as exc:
            raise RecoveryJournalPublicationError(
                f"Drive could not allocate a journal file ID: {exc}", phase="journal-id-allocation",
                expected={"parent_id": parent_id, "name": path.name},
                observed={"retryable": _journal_retryable_drive_error(exc)},
            ) from exc
        if not file_id:
            raise RecoveryJournalPublicationError(
                "Drive returned no preallocated journal file ID", phase="journal-id-allocation",
                expected={"parent_id": parent_id, "name": path.name}, observed={"response": generated},
            )
        create_body = {
            "id": file_id, "name": path.name, "mimeType": "application/json", "parents": [parent_id],
            "appProperties": {
                "revision_r1_finalization_key": transaction_key,
                "revision_r1_journal_revision": "0",
            },
        }
        last_create_error = None
        for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
            if delay:
                time.sleep(delay)
            try:
                service.files().create(
                    body=create_body,
                    fields="id,name,mimeType,parents,size,md5Checksum,version,owners(emailAddress),appProperties",
                    supportsAllDrives=True,
                ).execute()
                break
            except Exception as exc:
                last_create_error = exc
                if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                    if getattr(getattr(exc, "resp", None), "status", None) == 409:
                        conflict = _get_drive_journal_metadata(service, file_id)
                        if (
                            conflict.get("name") == path.name
                            and parent_id in [str(value) for value in conflict.get("parents", [])]
                            and (conflict.get("appProperties") or {}).get("revision_r1_finalization_key") == transaction_key
                        ):
                            break
                    raise RecoveryJournalPublicationError(
                        f"Drive journal create failed: {exc}", phase="journal-create",
                        expected={"file_id": file_id, "parent_id": parent_id, "name": path.name},
                        observed={"retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                    ) from exc
                # The same preallocated Drive ID makes retrying an ambiguous create safe.
        metadata = _get_drive_journal_metadata(service, file_id)
        _verify_drive_journal_location(metadata, file_id=file_id, parent_id=parent_id, name=path.name)
        if (metadata.get("appProperties") or {}).get("revision_r1_finalization_key") != transaction_key:
            raise RecoveryJournalPublicationError(
                f"created Drive journal has the wrong transaction marker (last create error={last_create_error})",
                phase="journal-create-verification",
                expected={"file_id": file_id, "parent_id": parent_id, "transaction_key": transaction_key},
                observed=_journal_observed_metadata(metadata, None),
            )

    remote_document, remote_payload, metadata = _read_valid_remote_journal(
        service, metadata, document=document, parent_id=parent_id,
    )
    pending_placeholder = (
        remote_document is None and not remote_payload
        and (metadata.get("appProperties") or {}).get("revision_r1_finalization_key") == transaction_key
    )
    if remote_document is None and not pending_placeholder:
        raise RecoveryJournalPublicationError(
            "Drive file is not a valid journal or a recognized empty publication placeholder",
            phase="journal-placeholder-verification",
            expected={"file_id": file_id, "parent_id": parent_id, "transaction_key": transaction_key},
            metadata=metadata, payload=remote_payload,
        )

    input_is_unmodified = (
        isinstance(document.get("journal_sha256"), str)
        and document.get("journal_sha256") == _finalization_journal_digest(document)
    )
    remote_revision = int(remote_document.get("journal_revision", 0)) if remote_document else 0
    if baseline is None and remote_document is not None:
        raise RecoveryJournalPublicationError(
            "remote journal exists but the local retry copy is absent; restore it before preparing another state",
            phase="journal-local-remote-conflict",
            expected={"file_id": file_id, "parent_id": parent_id, "transaction_key": transaction_key},
            metadata=metadata, payload=remote_payload,
        )
    if baseline is not None and remote_document is not None:
        baseline_revision = int(baseline.get("journal_revision", 0))
        baseline_hash = baseline.get("journal_sha256")
        if remote_revision > baseline_revision:
            if not input_is_unmodified:
                raise _journal_error(
                    "remote journal revision is newer than a locally modified retry state; preserving both",
                    phase="journal-revision-conflict",
                    expected={"local_revision": baseline_revision, "remote_revision": remote_revision},
                    metadata=metadata, payload=remote_payload,
                )
            document.clear()
            document.update(remote_document)
            _atomic_json(path, remote_document)
            return len(remote_payload)
        legacy_unbound_equivalent = (
            remote_revision == baseline_revision == 0
            and not remote_document.get("journal_drive_file_id")
            and not remote_document.get("journal_drive_parent_id")
            and {
                key: value for key, value in remote_document.items()
                if key not in {"journal_sha256", "journal_drive_file_id", "journal_drive_parent_id", "journal_revision"}
            } == {
                key: value for key, value in baseline.items()
                if key not in {"journal_sha256", "journal_drive_file_id", "journal_drive_parent_id", "journal_revision"}
            }
        )
        if remote_revision == baseline_revision and remote_document.get("journal_sha256") != baseline_hash and not legacy_unbound_equivalent:
            raise _journal_error(
                "Drive and local journals conflict at the same logical revision",
                phase="journal-revision-conflict",
                expected={"local_revision": baseline_revision, "local_sha256": baseline_hash},
                metadata=metadata, payload=remote_payload,
                observed={"remote_revision": remote_revision, "remote_sha256": remote_document.get("journal_sha256")},
            )
        if (
            remote_revision == baseline_revision
            and remote_document.get("journal_sha256") == baseline_hash
            and input_is_unmodified
            and remote_document.get("journal_drive_file_id") == file_id
            and remote_document.get("journal_drive_parent_id") == parent_id
            and (metadata.get("appProperties") or {}).get("revision_r1_finalization_key") == transaction_key
            and str((metadata.get("appProperties") or {}).get("revision_r1_journal_revision") or "") == str(remote_revision)
            and remote_payload == json.dumps(remote_document, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        ):
            return len(remote_payload)

    revision = max(local_revision, remote_revision) + 1
    candidate, serialized = _serialize_journal_for_drive(
        document, file_id=file_id, parent_id=parent_id, revision=revision,
    )
    required_bytes = max(4096, (len(serialized) + 1024) * 2)
    require_available_bytes(
        path, required_bytes=required_bytes, reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=int(available), storage_kind="Drive recovery-finalization journal upload",
    )
    require_available_bytes(
        path, required_bytes=required_bytes, reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=shutil.disk_usage(drive_root).free, storage_kind="Colab local journal staging",
    )
    update_response = None
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            update_response = service.files().update(
                fileId=file_id,
                body={"appProperties": {
                    "revision_r1_finalization_key": transaction_key,
                    "revision_r1_journal_revision": str(revision),
                }},
                media_body=_journal_media_upload(serialized),
                fields="id,name,mimeType,parents,size,md5Checksum,version,owners(emailAddress),appProperties",
                supportsAllDrives=True,
            ).execute()
            break
        except Exception as exc:
            if not _journal_retryable_drive_error(exc) or attempt + 1 == len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                observed_metadata = None
                observed_payload = None
                try:
                    observed_metadata = _get_drive_journal_metadata(service, file_id)
                    observed_payload = _get_drive_journal_payload(
                        service, file_id,
                        expected={"file_id": file_id, "parent_id": parent_id, "journal_revision": revision},
                    )
                except Exception:
                    pass
                if (
                    _journal_retryable_drive_error(exc)
                    and observed_metadata is not None
                    and observed_payload == serialized
                ):
                    # The transport may have failed after Drive committed the
                    # update. Accept only an exact ID-bound API read-back; never
                    # infer success from the request exception alone.
                    try:
                        _verify_journal_api_readback(
                            service, metadata=observed_metadata, document=candidate,
                            payload=serialized, file_id=file_id, parent_id=parent_id,
                            name=path.name, update_response=observed_metadata,
                        )
                    except RecoveryJournalPublicationError:
                        pass
                    else:
                        update_response = observed_metadata
                        break
                raise _journal_error(
                    f"Drive API rejected or did not confirm retention-journal update: {exc}",
                    phase="journal-api-update",
                    expected={
                        "file_id": file_id, "parent_id": parent_id, "size_bytes": len(serialized),
                        "sha256": hashlib.sha256(serialized).hexdigest(), "journal_revision": revision,
                    },
                    metadata=observed_metadata, payload=observed_payload,
                    observed={"retryable": _journal_retryable_drive_error(exc), "attempt": attempt + 1},
                ) from exc
    if not isinstance(update_response, dict):
        raise RecoveryJournalPublicationError(
            "Drive API update returned no metadata", phase="journal-api-update",
            expected={"file_id": file_id, "journal_revision": revision},
            observed={"response_type": type(update_response).__name__},
        )

    verified_metadata = _verify_journal_api_readback(
        service, metadata=metadata, document=candidate, payload=serialized,
        file_id=file_id, parent_id=parent_id, name=path.name, update_response=update_response,
    )
    after_publish = _list_drive_journal_children(service, parent_id, path.name)
    if len(after_publish) > 1 or (after_publish and str(after_publish[0].get("id") or "") != file_id):
        raise RecoveryJournalPublicationError(
            "Drive journal name became ambiguous after publication; checkpoints remain untouched",
            phase="journal-post-publication-ambiguity",
            expected={"file_id": file_id, "parent_id": parent_id, "name": path.name},
            observed={"matching_file_ids": [row.get("id") for row in after_publish]},
        )
    try:
        _atomic_json(path, candidate)
    except OSError as exc:
        raise _journal_error(
            f"Drive journal is verified but local retry-copy write failed: {exc}",
            phase="journal-local-publication-after-drive-commit",
            expected={"file_id": file_id, "parent_id": parent_id, "size_bytes": len(serialized),
                      "sha256": hashlib.sha256(serialized).hexdigest(), "journal_revision": revision},
            metadata=verified_metadata, payload=serialized,
        ) from exc
    document.clear()
    document.update(candidate)
    return len(serialized)


def _validate_recovery_finalization_journal(
    path: Path, plan: dict[str, Any], record: dict[str, Any], identity: dict[str, Any],
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"recovery-retention journal is unreadable: {path}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("plan_sha256") != plan["plan_sha256"]
        or document.get("run_key") != record["run_key"]
        or document.get("run_id") != _run_id_for(record, synthetic=False)
        or document.get("identity") != identity
        or document.get("identity_sha256") != _sha256_json(identity)
        or document.get("journal_sha256") != _finalization_journal_digest(document)
        or not isinstance(document.get("protected_generations"), list)
        or len(document["protected_generations"]) != 2
        or not isinstance(document.get("delete_generations"), list)
        or not isinstance(document.get("delete_orphan_manifests", []), list)
        or not isinstance(document.get("preserved_orphan_manifests", []), list)
        or not isinstance(document.get("orphan_manifest_reconciliation", {}), dict)
        or not document.get("journal_drive_file_id")
        or not document.get("journal_drive_parent_id")
        or type(document.get("journal_revision", 0)) is not int
        or document.get("journal_revision", 0) < 0
        or document.get("status") not in {"PREPARED", "RETENTION_VERIFIED"}
    ):
        raise ValueError(f"recovery-retention journal does not match the frozen run identity: {path}")
    protected = document["protected_generations"]
    if any(type(value) is not int or value < 1 for value in protected) or protected != sorted(set(protected)):
        raise ValueError("recovery-retention journal protected-generation list is invalid")
    for generation in document["delete_generations"]:
        if (
            not isinstance(generation, dict)
            or type(generation.get("generation")) is not int
            or generation["generation"] in protected
            or not isinstance(generation.get("files"), list)
            or {item.get("role") for item in generation["files"] if isinstance(item, dict)} != {"manifest", "payload"}
        ):
            raise ValueError("recovery-retention journal deletion target is invalid")
        number = int(generation["generation"])
        expected_names = {"manifest": f"generation-{number:06d}.manifest.json", "payload": f"generation-{number:06d}.pt"}
        for item in generation["files"]:
            if (
                item.get("name") != expected_names[item["role"]]
                or not item.get("drive_file_id")
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] <= 0
                or not _is_sha256(item.get("local_sha256"))
                or item.get("can_delete") is not True
            ):
                raise ValueError("recovery-retention journal contains unverified deletion metadata")
    orphan_names: set[str] = set()
    for item in document.get("delete_orphan_manifests", []):
        generation = item.get("generation") if isinstance(item, dict) else None
        expected_name = f"generation-{generation:06d}.manifest.json" if type(generation) is int else None
        if (
            not isinstance(item, dict)
            or type(generation) is not int
            or generation < 1
            or generation >= protected[0]
            or item.get("name") != expected_name
            or item.get("name") in orphan_names
            or not item.get("drive_file_id")
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
            or not _is_sha256(item.get("local_sha256"))
            or item.get("can_delete") is not True
            or item.get("identity_sha256") != document.get("identity_sha256")
            or item.get("payload_file") != f"generation-{generation:06d}.pt"
            or type(item.get("payload_bytes")) is not int
            or item["payload_bytes"] <= 0
            or not _is_sha256(item.get("payload_sha256"))
        ):
            raise ValueError("recovery-retention journal contains unverified orphan-manifest deletion metadata")
        orphan_names.add(item["name"])
    preserved_names: set[str] = set()
    for item in document.get("preserved_orphan_manifests", []):
        generation = item.get("generation") if isinstance(item, dict) else None
        expected_name = f"generation-{generation:06d}.manifest.json" if type(generation) is int else None
        if (
            not isinstance(item, dict)
            or type(generation) is not int
            or generation < 1
            or generation >= protected[0]
            or item.get("name") != expected_name
            or item.get("name") in preserved_names
            or not item.get("drive_file_id")
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
            or not _is_sha256(item.get("local_sha256"))
            or item.get("can_delete") is not False
            or item.get("can_edit") is not True
            or item.get("preservation_reason") != "current_user_can_edit_but_cannot_permanently_delete"
            or item.get("identity_sha256") != document.get("identity_sha256")
            or item.get("payload_file") != f"generation-{generation:06d}.pt"
            or type(item.get("payload_bytes")) is not int
            or item["payload_bytes"] <= 0
            or not _is_sha256(item.get("payload_sha256"))
        ):
            raise ValueError("recovery-retention journal contains unverified orphan-manifest preservation metadata")
        planned = next(
            (candidate for candidate in document.get("delete_orphan_manifests", [])
             if candidate.get("name") == item["name"]),
            None,
        )
        if planned is not None and any(
            planned.get(key) != item.get(key)
            for key in (
                "generation", "name", "drive_file_id", "size_bytes", "local_sha256",
                "identity_sha256", "payload_file", "payload_bytes", "payload_sha256",
            )
        ):
            raise ValueError("preserved orphan manifest differs from its prepared deletion record")
        preserved_names.add(item["name"])
    return document


def _restore_remote_recovery_finalization_journal(
    path: Path, plan: dict[str, Any], record: dict[str, Any], identity: dict[str, Any],
    service, *, drive_root: Path, recovery_folder_id: str,
) -> bool:
    """Restore a published finalization transaction when Drive committed before local write.

    This is specifically the retry case where the API accepted the journal bytes
    but the Colab process stopped before materializing its local retry copy.
    Existing local files are never replaced by this recovery path.
    """
    if path.exists():
        return False
    _configured, components = _logical_drive_path_components(path.parent, drive_root=drive_root)
    try:
        location = resolve_drive_folder(service, components)
    except Exception as exc:
        raise RecoveryJournalPublicationError(
            f"could not resolve the exact Drive parent while recovering the finalization journal: {exc}",
            phase="journal-restore-parent-resolution",
            expected={"logical_components": components},
        ) from exc
    if location.get("path_exists") is not True or not location.get("folder_id"):
        return False
    parent_id = str(location["folder_id"])
    matches = _list_drive_journal_children(service, parent_id, path.name)
    if len(matches) > 1:
        raise RecoveryJournalPublicationError(
            "duplicate Drive files have the exact retention-journal name; refusing remote recovery",
            phase="journal-restore-name-ambiguity",
            expected={"parent_id": parent_id, "name": path.name},
            observed={"matching_file_ids": [row.get("id") for row in matches]},
        )
    if not matches:
        return False
    file_id = str(matches[0].get("id") or "")
    if not file_id:
        raise RecoveryJournalPublicationError(
            "Drive retention journal has no stable file ID", phase="journal-restore-file-id",
            expected={"parent_id": parent_id, "name": path.name}, observed={"metadata": matches[0]},
        )
    transaction_document = {
        "plan_sha256": plan["plan_sha256"], "run_key": record["run_key"],
        "run_id": _run_id_for(record, synthetic=False), "identity_sha256": _sha256_json(identity),
    }
    transaction_key = _journal_transaction_key(transaction_document)
    last_metadata: dict[str, Any] | None = None
    last_payload: bytes | None = None
    for attempt, delay in enumerate(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            last_metadata = _get_drive_journal_metadata(service, file_id)
            _verify_drive_journal_location(last_metadata, file_id=file_id, parent_id=parent_id, name=path.name)
            last_payload = _get_drive_journal_payload(
                service, file_id,
                expected={"file_id": file_id, "parent_id": parent_id, "name": path.name},
            )
        except RecoveryJournalPublicationError:
            raise
        except Exception as exc:
            if _journal_retryable_drive_error(exc) and attempt + 1 < len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                continue
            raise RecoveryJournalPublicationError(
                f"could not read the exact Drive finalization journal: {exc}",
                phase="journal-restore-read", expected={"file_id": file_id, "parent_id": parent_id},
                observed={"attempt": attempt + 1, "retryable": _journal_retryable_drive_error(exc)},
            ) from exc
        marker = (last_metadata.get("appProperties") or {}).get("revision_r1_finalization_key")
        if not last_payload and marker == transaction_key:
            return False
        try:
            remote = json.loads(last_payload.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _journal_error(
                "Drive finalization journal is not valid JSON", phase="journal-restore-integrity",
                expected={"file_id": file_id, "parent_id": parent_id},
                metadata=last_metadata, payload=last_payload,
            ) from exc
        if not isinstance(remote, dict):
            raise _journal_error(
                "Drive finalization journal is not a JSON object", phase="journal-restore-integrity",
                expected={"file_id": file_id, "parent_id": parent_id},
                metadata=last_metadata, payload=last_payload,
            )
        try:
            actual_size = _optional_nonnegative_int(last_metadata.get("size"), "journal restore size")
        except StorageCapacityError:
            actual_size = None
        metadata_consistent = (
            actual_size == len(last_payload)
            and (not last_metadata.get("md5Checksum")
                 or last_metadata.get("md5Checksum") == hashlib.md5(last_payload, usedforsecurity=False).hexdigest())
        )
        if not metadata_consistent:
            if attempt + 1 < len(RECOVERY_JOURNAL_READBACK_RETRY_DELAYS_SECONDS):
                continue
            raise _journal_error(
                "Drive journal metadata remained stale or inconsistent while restoring its bytes",
                phase="journal-restore-readback", expected={"size_bytes": len(last_payload)},
                metadata=last_metadata, payload=last_payload,
                observed={"attempt": attempt + 1, "actual_size": actual_size},
            )
        if remote.get("journal_sha256") != _finalization_journal_digest(remote):
            raise _journal_error(
                "Drive finalization journal checksum is invalid", phase="journal-restore-integrity",
                expected={"journal_sha256": remote.get("journal_sha256")},
                metadata=last_metadata, payload=last_payload,
            )
        _verify_remote_journal_identity(remote, transaction_document, file_id=file_id, parent_id=parent_id)
        app_properties = last_metadata.get("appProperties") or {}
        remote_revision = int(remote.get("journal_revision", 0))
        if marker not in (None, transaction_key) or (
            app_properties.get("revision_r1_journal_revision") is not None
            and str(app_properties["revision_r1_journal_revision"]) != str(remote_revision)
        ):
            raise _journal_error(
                "Drive journal application properties do not match its transaction/revision",
                phase="journal-restore-identity", expected={"transaction_key": transaction_key,
                    "journal_revision": remote_revision}, metadata=last_metadata, payload=last_payload,
            )
        recovery_folder = remote.get("recovery_folder")
        if (
            remote.get("schema_version") != 1
            or remote.get("status") not in {"PREPARED", "RETENTION_VERIFIED"}
            or remote.get("identity") != identity
            or not isinstance(recovery_folder, dict)
            or str(recovery_folder.get("folder_id")) != str(recovery_folder_id)
        ):
            raise _journal_error(
                "Drive journal is validly serialized but does not match this completed run/recovery folder",
                phase="journal-restore-run-identity",
                expected={"run_id": transaction_document["run_id"], "recovery_folder_id": recovery_folder_id},
                metadata=last_metadata, payload=last_payload,
            )
        restored = dict(remote)
        if restored.get("journal_drive_file_id") not in (None, file_id):
            raise _journal_error(
                "Drive journal embeds a different file ID", phase="journal-restore-identity",
                expected={"file_id": file_id}, metadata=last_metadata, payload=last_payload,
            )
        if restored.get("journal_drive_parent_id") not in (None, parent_id):
            raise _journal_error(
                "Drive journal embeds a different parent ID", phase="journal-restore-identity",
                expected={"parent_id": parent_id}, metadata=last_metadata, payload=last_payload,
            )
        restored["journal_drive_file_id"] = file_id
        restored["journal_drive_parent_id"] = parent_id
        restored.pop("journal_sha256", None)
        restored["journal_sha256"] = _finalization_journal_digest(restored)
        try:
            _atomic_json(path, restored)
            _validate_recovery_finalization_journal(path, plan, record, identity)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return True
    raise RecoveryJournalPublicationError(
        "Drive finalization journal recovery read-back did not converge",
        phase="journal-restore-readback", expected={"file_id": file_id, "parent_id": parent_id},
        observed=_journal_observed_metadata(last_metadata, last_payload),
    )


def _drive_recovery_folder(service, recovery: Path, drive_root: Path) -> dict[str, Any]:
    _configured, components = _logical_drive_path_components(recovery, drive_root=drive_root)
    location = resolve_drive_folder(service, components)
    if location.get("path_exists") is not True or not location.get("folder_id"):
        raise StorageCapacityError(f"exact Drive recovery folder could not be resolved: {recovery}")
    return {
        "folder_id": str(location["folder_id"]), "components": list(components),
        "owner_email": location.get("owner_email"), "drive_id": location.get("drive_id"),
    }


def _resume_recovery_finalization_deletions(
    recovery: Path, journal_path: Path, journal: dict[str, Any], identity: dict[str, Any],
    service, drive_root: Path, *, api_factory, progress_callback=None,
) -> None:
    from revision_r1.validation import file_sha256

    folder = _drive_recovery_folder(service, recovery, drive_root)
    parent_id = folder["folder_id"]
    journal_folder = journal.get("recovery_folder")
    if (
        not isinstance(journal_folder, dict)
        or str(journal_folder.get("folder_id")) != str(parent_id)
    ):
        raise StorageCapacityError("recovery folder ID differs from the retention journal; no pruning is allowed")
    expected_protected = journal["protected_generations"]
    target_names = {
        item["name"]
        for generation in journal["delete_generations"] for item in generation["files"]
    }
    orphan_deletions = journal.get("delete_orphan_manifests", [])
    target_names.update(item["name"] for item in orphan_deletions)
    orphan_preservations = journal.get("preserved_orphan_manifests", [])
    target_names.update(item["name"] for item in orphan_preservations)
    protected_names = {
        file_record["name"]
        for row in journal["initial_recovery_inventory"]
        if row["generation"] in expected_protected
        for file_record in row["files"].values()
    }
    allowed_names = target_names | protected_names
    remote = _list_drive_folder_files(service, parent_id)
    remote_generation_names = [row.get("name") for row in remote if str(row.get("name", "")).startswith("generation-")]
    local_generation_names = [path.name for path in recovery.iterdir() if path.is_file() and path.name.startswith("generation-")]
    if not set(remote_generation_names).issubset(allowed_names) or not set(local_generation_names).issubset(allowed_names):
        raise StorageCapacityError("recovery folder contains generation files not authorized by the identity-bound retention journal")

    names_to_records = {row.get("name"): row for row in remote if isinstance(row.get("name"), str)}
    if len(names_to_records) != len(remote):
        raise StorageCapacityError("Drive recovery folder contains duplicate names; refusing checkpoint cleanup")

    # Retained generations are never inferred from a journal alone: recheck each full manifest/payload pair.
    for generation in expected_protected:
        row = next((item for item in journal["initial_recovery_inventory"] if item["generation"] == generation), None)
        if row is None:
            raise ValueError("retention journal omits a protected recovery generation")
        manifest_path = recovery / row["files"]["manifest"]["name"]
        payload_path = recovery / row["files"]["payload"]["name"]
        if not manifest_path.is_file() or not payload_path.is_file():
            raise ValueError(f"protected recovery generation {generation} is incomplete; refusing to prune")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("identity") != identity
            or manifest.get("identity_sha256") != _sha256_json(identity)
            or manifest.get("generation") != generation
            or manifest.get("payload_sha256") != row["payload_sha256"]
            or file_sha256(payload_path) != row["payload_sha256"]
            or file_sha256(manifest_path) != row["manifest_sha256"]
        ):
            raise ValueError(f"protected recovery generation {generation} changed since the retention journal was prepared")
        for role in ("manifest", "payload"):
            expected = row["files"][role]
            metadata = names_to_records.get(expected["name"])
            if metadata is None:
                raise StorageCapacityError(f"Drive no longer has protected recovery file {expected['name']}")
            actual = _drive_checkpoint_file_record(
                recovery / expected["name"], metadata, parent_id,
                expected_bytes=expected["size_bytes"], expected_sha256=expected["local_sha256"],
            )
            if actual["drive_file_id"] != expected["drive_file_id"]:
                raise StorageCapacityError(f"Drive file ID changed for protected recovery file {expected['name']}")

    deleted_names = set(journal.get("deleted_files", []))
    for generation in sorted(journal["delete_generations"], key=lambda row: row["generation"]):
        for item in sorted(generation["files"], key=lambda row: 0 if row["role"] == "payload" else 1):
            path = recovery / item["name"]
            metadata = _lookup_drive_child(service, parent_id, item["name"])
            if metadata is None:
                if path.exists():
                    raise StorageCapacityError(f"Drive removed {item['name']} but the mounted view still exposes it; wait for Drive synchronization and retry")
                deleted_names.add(item["name"])
                continue
            actual = _drive_checkpoint_file_record(
                path, metadata, parent_id,
                expected_bytes=item["size_bytes"], expected_sha256=item["local_sha256"] if path.is_file() else None,
            )
            if actual["drive_file_id"] != item["drive_file_id"] or actual["can_delete"] is not True:
                raise StorageCapacityError(f"Drive identity or canDelete changed for stale recovery file {item['name']}; preserved")
            if progress_callback is not None:
                progress_callback({"stage": "recovery_stale_file_delete_start", "generation": generation["generation"], "file": actual})
            try:
                deleted = permanently_delete_recovery_checkpoint(
                    path, api_factory=api_factory, drive_root=drive_root,
                    expected_drive_file_id=item["drive_file_id"], expected_parent_id=parent_id,
                )
            except Exception as exc:
                raise StorageCapacityError(f"Drive delete failure for {item['name']}; retention journal is preserved for a safe retry: {exc}") from exc
            if deleted is not True:
                raise StorageCapacityError(f"Drive no longer grants canDelete for {item['name']}; retention journal is preserved")
            deadline = time.monotonic() + 15.0
            while path.exists() and time.monotonic() < deadline:
                time.sleep(0.25)
            if path.exists():
                raise StorageCapacityError(f"Drive API removed {item['name']} but the mounted path has not synchronized; retry after synchronization")
            deleted_names.add(item["name"])
            journal["deleted_files"] = sorted(deleted_names)
            _save_recovery_finalization_journal(journal_path, journal, service, drive_root=drive_root)
            if progress_callback is not None:
                progress_callback({"stage": "recovery_stale_file_deleted", "generation": generation["generation"], "file": item["name"]})

    # An orphan manifest is not a recovery point. Permanently delete it only
    # when the current account has that exact capability. A shared-folder
    # editor who cannot delete another user's file may preserve the verified
    # manifest in place; it is metadata, not a recovery payload.
    preserved_by_name = {item["name"]: item for item in orphan_preservations}
    orphan_targets = {item["name"]: item for item in orphan_deletions}
    orphan_targets.update(preserved_by_name)
    for item in sorted(orphan_targets.values(), key=lambda row: row["generation"]):
        already_preserved = item["name"] in preserved_by_name
        path = recovery / item["name"]
        payload_path = recovery / item["payload_file"]
        metadata = _lookup_drive_child(service, parent_id, item["name"])
        if metadata is None:
            if already_preserved:
                raise StorageCapacityError(
                    f"journaled preserved orphan manifest {item['name']} disappeared; its state is ambiguous"
                )
            if path.exists():
                raise StorageCapacityError(
                    f"Drive removed orphan manifest {item['name']} but the mounted view still exposes it; wait for synchronization and retry"
                )
            deleted_names.add(item["name"])
            continue
        if payload_path.exists() or item["payload_file"] in names_to_records:
            raise StorageCapacityError(
                f"orphan manifest {item['name']} now has a payload; refusing orphan cleanup"
            )
        if not path.is_file():
            raise StorageCapacityError(
                f"Drive lists orphan manifest {item['name']} but the mounted manifest is unavailable"
            )
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"journaled orphan manifest is unreadable: {path}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("generation") != item["generation"]
            or manifest.get("identity") != identity
            or manifest.get("identity_sha256") != item["identity_sha256"]
            or manifest.get("payload_file") != item["payload_file"]
            or manifest.get("payload_bytes") != item["payload_bytes"]
            or manifest.get("payload_sha256") != item["payload_sha256"]
            or file_sha256(path) != item["local_sha256"]
        ):
            raise ValueError(f"journaled orphan manifest {item['name']} changed since retention was prepared")
        actual = _drive_checkpoint_file_record(
            path, metadata, parent_id,
            expected_bytes=item["size_bytes"], expected_sha256=item["local_sha256"],
        )
        if actual["drive_file_id"] != item["drive_file_id"]:
            raise StorageCapacityError(
                f"Drive identity changed for orphan manifest {item['name']}; preserved"
            )
        if already_preserved:
            if actual["can_edit"] is not True:
                raise StorageCapacityError(
                    f"Drive no longer verifies edit access to preserved orphan manifest {item['name']}"
                )
            continue
        if actual["can_delete"] is False:
            if actual["can_edit"] is not True:
                raise StorageCapacityError(
                    f"Drive cannot delete or edit orphan manifest {item['name']}; preserved without finalizing"
                )
            preserved = {
                **item,
                "can_delete": False,
                "can_edit": True,
                "preservation_reason": "current_user_can_edit_but_cannot_permanently_delete",
            }
            orphan_preservations.append(preserved)
            preserved_by_name[item["name"]] = preserved
            journal["preserved_orphan_manifests"] = sorted(
                orphan_preservations, key=lambda row: row["generation"],
            )
            journal.setdefault("orphan_manifest_reconciliation", {}).update({
                "status": "PRESERVATION_IN_PROGRESS",
                "preservation_reason": preserved["preservation_reason"],
            })
            _save_recovery_finalization_journal(journal_path, journal, service, drive_root=drive_root)
            if progress_callback is not None:
                progress_callback({
                    "stage": "recovery_orphan_manifest_preserved",
                    "generation": item["generation"], "name": item["name"],
                    "drive_file_id": item["drive_file_id"],
                    "size_bytes": item["size_bytes"],
                    "reason": preserved["preservation_reason"],
                })
            continue
        if actual["can_delete"] is not True:
            raise StorageCapacityError(
                f"Drive does not verify canDelete for orphan manifest {item['name']}; preserved"
            )
        if progress_callback is not None:
            progress_callback({"stage": "recovery_orphan_manifest_delete_start", "generation": item["generation"], "file": actual})
        try:
            deleted = permanently_delete_recovery_checkpoint(
                path, api_factory=api_factory, drive_root=drive_root,
                expected_drive_file_id=item["drive_file_id"], expected_parent_id=parent_id,
            )
        except Exception as exc:
            raise StorageCapacityError(
                f"Drive delete failure for orphan manifest {item['name']}; retention journal is preserved for a safe retry: {exc}"
            ) from exc
        if deleted is not True:
            raise StorageCapacityError(
                f"Drive no longer grants canDelete for orphan manifest {item['name']}; retention journal is preserved"
            )
        deadline = time.monotonic() + 15.0
        while path.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        if path.exists():
            raise StorageCapacityError(
                f"Drive API removed orphan manifest {item['name']} but the mounted path has not synchronized; retry after synchronization"
            )
        deleted_names.add(item["name"])
        journal["deleted_files"] = sorted(deleted_names)
        _save_recovery_finalization_journal(journal_path, journal, service, drive_root=drive_root)
        if progress_callback is not None:
            progress_callback({"stage": "recovery_orphan_manifest_deleted", "generation": item["generation"], "file": item["name"]})

    inspected_final = _inspect_verified_recovery_tree(
        recovery, identity, service, parent_id, progress_callback=progress_callback,
    )
    final_inventory = inspected_final["verified_generations"]
    final_orphans = inspected_final["orphan_manifests"]
    preserved_by_name = {item["name"]: item for item in orphan_preservations}
    observed_orphan_names = {item["manifest_file"]["name"] for item in final_orphans}
    if observed_orphan_names != set(preserved_by_name):
        raise StorageCapacityError(
            "recovery manifest-only entries differ from the journaled preserved set: "
            f"observed={sorted(observed_orphan_names)}, journaled={sorted(preserved_by_name)}"
        )
    for orphan in final_orphans:
        expected = preserved_by_name[orphan["manifest_file"]["name"]]
        actual = orphan["manifest_file"]
        if (
            orphan["generation"] != expected["generation"]
            or actual["drive_file_id"] != expected["drive_file_id"]
            or actual["size_bytes"] != expected["size_bytes"]
            or actual["local_sha256"] != expected["local_sha256"]
            or actual["can_edit"] is not True
        ):
            raise StorageCapacityError(
                f"preserved orphan manifest {actual['name']} changed during final retention verification"
            )
    if [row["generation"] for row in final_inventory] != expected_protected:
        raise StorageCapacityError(
            "recovery retention read-back does not match the two journaled generations: "
            f"expected={expected_protected}, found={[row['generation'] for row in final_inventory]}"
        )
    if final_inventory[-1]["kind"] != "complete":
        raise ValueError("newest retained recovery point is not the terminal complete generation")
    journal["deleted_files"] = sorted(deleted_names)
    journal["status"] = "RETENTION_VERIFIED"
    journal["retention_verified_inventory"] = final_inventory
    orphan_reconciliation = journal.setdefault("orphan_manifest_reconciliation", {})
    deleted_orphan_names = sorted(
        set(journal.get("deleted_files", [])) & {row["name"] for row in orphan_deletions}
    )
    preserved_orphan_names = sorted(preserved_by_name)
    if preserved_orphan_names and deleted_orphan_names:
        orphan_reconciliation["status"] = "VERIFIED_MIXED"
    elif preserved_orphan_names:
        orphan_reconciliation["status"] = "VERIFIED_PRESERVED"
    elif deleted_orphan_names:
        orphan_reconciliation["status"] = "VERIFIED_DELETED"
    else:
        orphan_reconciliation["status"] = "NOT_REQUIRED"
    orphan_reconciliation["deleted_manifest_names"] = deleted_orphan_names
    orphan_reconciliation["preserved_manifest_names"] = preserved_orphan_names
    _save_recovery_finalization_journal(journal_path, journal, service, drive_root=drive_root)


def finalize_completed_production_run(
    plan: dict[str, Any], record: dict[str, Any], *,
    output_root: str | Path, checkpoint_root: str | Path,
    progress_callback=None, api_factory=None,
    drive_root: str | Path = "/content/drive/MyDrive",
) -> dict[str, Any]:
    """Finalize an already validated run without loading data or invoking training.

    The operation is restricted to frozen run 1. It verifies every recovery
    manifest/payload and Drive file identity before pruning only stale recovery
    generations. A small identity-bound journal makes an interrupted pair
    deletion retryable. Task-boundary artifacts and run outputs are read-only.
    """
    from revision_r1.registry import RunRegistry
    from revision_r1.validation import file_sha256

    package_root = Path(__file__).resolve().parents[1]
    validate_frozen_plan(plan, expected_source_sha256=training_source_sha256(package_root))
    if not plan.get("runs") or record != plan["runs"][0] or record.get("run_index") != 1:
        raise ValueError("finalization-only recovery is restricted to frozen matrix run 1")
    if (
        (record.get("dataset"), int(record.get("seed", -1)), record.get("arm")) != FIRST_RUN
        or record.get("effective_schedule_sha256") != plan["schedule_sha256"]
        or not isinstance(record.get("update_budgets_by_task"), list)
        or len(record["update_budgets_by_task"]) != 11
    ):
        raise ValueError("run 1 record is not bound to its exact frozen 11-task schedule")

    output_root_path = Path(output_root).expanduser().resolve()
    checkpoint_root_path = Path(checkpoint_root).expanduser()
    drive_root_path = Path(drive_root).expanduser()
    output = run_output_root(output_root_path, record)
    checkpoint_namespace = run_checkpoint_root(checkpoint_root_path, record)
    # run_checkpoint_root() resolves mounted aliases for filesystem identity.
    # Reconstruct and verify the corresponding logical MyDrive path for Drive
    # API traversal instead of passing .shortcut-targets-by-id to the lexical
    # API-path validator.
    recovery = _logical_drive_alias_path(
        checkpoint_namespace / "recovery",
        configured_root=checkpoint_root_path,
        drive_root=drive_root_path,
    )
    journal_path = checkpoint_root_path / "_finalization" / plan["plan_sha256"] / f"{record['run_key']}.json"
    matrix_lease = _MatrixLeaseHeartbeat(output_root_path / ".matrix_execution.lease.json")
    callback = progress_callback or (lambda _event: None)

    with matrix_lease as lease:
        callback({"stage": "completed_output_validation_start", "run_key": record["run_key"]})
        validated = _validate_completed_run(plan, record, None, output, synthetic=False)
        run_state = RunRegistry(output / "registry").status(validated["run_id"])
        identity = run_state.get("identity")
        if not isinstance(identity, dict) or identity.get("execution_scope") != "production":
            raise ValueError("completed runner registry has no exact production checkpoint identity")
        task_boundary_preflight = _checkpoint_tree_measurement(
            checkpoint_namespace, identity, kind="task_boundaries", expected_count=11,
        )
        callback({
            "stage": "task_boundary_checkpoint_preflight_complete",
            "generation_count": task_boundary_preflight["generation_count"],
            "payload_bytes": task_boundary_preflight["payload_bytes"],
            "manifest_bytes": task_boundary_preflight["manifest_bytes"],
            "all_task_boundary_artifacts_preserved": True,
        })
        callback({
            "stage": "completed_output_validated", "run_id": validated["run_id"],
            "artifact_manifest_sha256": validated["artifact_manifest_sha256"],
        })

        service = (api_factory or (lambda: _drive_api_service(scopes=(DRIVE_FILE_WRITE_SCOPE,))))()
        folder = _drive_recovery_folder(service, recovery, drive_root_path)
        restored_remote_journal = _restore_remote_recovery_finalization_journal(
            journal_path, plan, record, identity, service,
            drive_root=drive_root_path, recovery_folder_id=folder["folder_id"],
        )
        if restored_remote_journal:
            callback({
                "stage": "recovery_retention_journal_restored_from_drive",
                "journal_path": str(journal_path), "recovery_folder_id": folder["folder_id"],
            })
        if journal_path.is_file():
            # Re-publish/read back through Drive before trusting the retry log;
            # this also completes ID binding if interrupted between initial write
            # and the first verified API read-back.
            _save_recovery_finalization_journal(journal_path, json.loads(journal_path.read_text(encoding="utf-8")), service, drive_root=drive_root_path)
            journal = _validate_recovery_finalization_journal(journal_path, plan, record, identity)
            initial_inventory = journal["initial_recovery_inventory"]
            delete_generations = journal["delete_generations"]
            delete_orphan_manifests = journal.get("delete_orphan_manifests", [])
            preserved_orphan_manifests = journal.get("preserved_orphan_manifests", [])
            rotation_explanation = journal["rotation_explanation"]
            callback({
                "stage": "recovery_retention_journal_resume", "journal_path": str(journal_path),
                "journal_status": journal["status"], "deleted_files": journal.get("deleted_files", []),
            })
            _resume_recovery_finalization_deletions(
                recovery, journal_path, journal, identity, service, drive_root_path,
                api_factory=lambda: service, progress_callback=callback,
            )
            # The active account may have edit access but lack permanent-delete
            # authority. The resume routine records that verified preservation
            # decision in the journal; reload it before final read-back/reporting.
            journal = _validate_recovery_finalization_journal(
                journal_path, plan, record, identity,
            )
            delete_generations = journal["delete_generations"]
            initial_inventory = journal["initial_recovery_inventory"]
            delete_orphan_manifests = journal.get("delete_orphan_manifests", [])
            preserved_orphan_manifests = journal.get("preserved_orphan_manifests", [])
            rotation_explanation = journal["rotation_explanation"]
        else:
            inspected_initial = _inspect_verified_recovery_tree(
                recovery, identity, service, folder["folder_id"], progress_callback=callback,
            )
            initial_inventory = inspected_initial["verified_generations"]
            orphan_manifests = inspected_initial["orphan_manifests"]
            generations = [row["generation"] for row in initial_inventory]
            callback({
                "stage": "recovery_inventory_complete",
                "generations": generations,
                "generation_count": len(initial_inventory),
                "orphan_manifest_generations": [row["generation"] for row in orphan_manifests],
                "manifest_and_payload_hashes_verified": True,
                "drive_files_verified": True,
            })
            if len(initial_inventory) < RECOVERY_GENERATIONS_PER_RUN:
                raise ValueError(
                    f"cannot finalize: only {len(initial_inventory)} valid recovery generations exist; "
                    f"at least {RECOVERY_GENERATIONS_PER_RUN} are required"
                )
            retained = initial_inventory[-RECOVERY_GENERATIONS_PER_RUN:]
            if retained[-1]["kind"] != "complete":
                raise ValueError("newest verified recovery point is not the terminal complete generation; no pruning performed")
            if any(row["generation"] >= retained[0]["generation"] for row in orphan_manifests):
                raise ValueError(
                    "an incomplete recovery manifest is as new as a required retained recovery point; "
                    "preserving all files because the fallback chain is ambiguous"
                )
            delete_orphan_manifests = []
            preserved_orphan_manifests = []
            for orphan in orphan_manifests:
                manifest_file = orphan["manifest_file"]
                can_delete = manifest_file.get("can_delete")
                can_edit = manifest_file.get("can_edit")
                if can_delete is not True and not (can_delete is False and can_edit is True):
                    raise StorageCapacityError(
                        f"Drive does not verify delete or editor capability for orphan manifest {manifest_file['name']}; "
                        "all recovery files are preserved"
                    )
                orphan_record = {
                    "generation": orphan["generation"],
                    "name": manifest_file["name"],
                    "drive_file_id": manifest_file["drive_file_id"],
                    "size_bytes": manifest_file["size_bytes"],
                    "local_sha256": manifest_file["local_sha256"],
                    "can_delete": can_delete,
                    "identity_sha256": orphan["identity_sha256"],
                    "payload_file": orphan["payload_file"],
                    "payload_bytes": orphan["payload_bytes"],
                    "payload_sha256": orphan["payload_sha256"],
                }
                if can_delete is True:
                    delete_orphan_manifests.append(orphan_record)
                else:
                    preserved_orphan_manifests.append({
                        **orphan_record,
                        "can_edit": True,
                        "preservation_reason": "current_user_can_edit_but_cannot_permanently_delete",
                    })
            delete_generations = initial_inventory[:-RECOVERY_GENERATIONS_PER_RUN]
            rotation_explanation = _rotation_deferred_pruning_evidence(
                output_root_path, plan, record, delete_generations,
            )
            if delete_generations or delete_orphan_manifests or preserved_orphan_manifests:
                for generation in delete_generations:
                    for role in ("payload", "manifest"):
                        item = generation["files"][role]
                        if item.get("can_delete") is not True:
                            raise StorageCapacityError(
                                f"Drive canDelete is not verified true for stale recovery file {item['name']}; "
                                "all checkpoints are preserved"
                            )
                delete_rows = [
                    {
                        "generation": row["generation"],
                        "files": [
                            {"role": role, **row["files"][role]}
                            for role in ("manifest", "payload")
                        ],
                    }
                    for row in delete_generations
                ]
                journal = {
                    "schema_version": 1, "status": "PREPARED",
                    "plan_sha256": plan["plan_sha256"], "run_key": record["run_key"],
                    "run_id": validated["run_id"], "identity": identity,
                    "identity_sha256": _sha256_json(identity),
                    "recovery_folder": folder, "protected_generations": [row["generation"] for row in retained],
                    "initial_recovery_inventory": initial_inventory,
                    "delete_generations": delete_rows,
                    "delete_orphan_manifests": delete_orphan_manifests,
                    "preserved_orphan_manifests": preserved_orphan_manifests,
                    "deleted_files": [],
                    "orphan_manifest_reconciliation": {
                        "status": (
                            "PREPARED_PRESERVATION" if preserved_orphan_manifests
                            else "PREPARED" if delete_orphan_manifests else "NOT_REQUIRED"
                        ),
                        "mechanism": (
                            "checkpoint rotation deletes a stale payload before its manifest; an interrupted or failed "
                            "manifest deletion can leave a manifest-only generation"
                        ) if (delete_orphan_manifests or preserved_orphan_manifests) else None,
                        "specific_cause_established_from_run_logs": False,
                        "reason": (
                            "manifest metadata and current identity were verified, payloads were absent from both "
                            "the mounted recovery folder and exact Drive folder, and two newer verified recovery "
                            "points were protected"
                        ) if (delete_orphan_manifests or preserved_orphan_manifests) else None,
                    },
                    "rotation_explanation": rotation_explanation,
                    "prepared_utc": datetime.now(timezone.utc).isoformat(),
                }
                callback({
                    "stage": "recovery_retention_journal_write_start", "journal_path": str(journal_path),
                    "protected_generations": journal["protected_generations"],
                    "delete_generations": [row["generation"] for row in delete_rows],
                    "delete_orphan_manifests": [row["name"] for row in delete_orphan_manifests],
                    "preserved_orphan_manifests": [row["name"] for row in preserved_orphan_manifests],
                })
                _save_recovery_finalization_journal(
                    journal_path, journal, service, drive_root=drive_root_path,
                )
                journal = _validate_recovery_finalization_journal(journal_path, plan, record, identity)
                _resume_recovery_finalization_deletions(
                    recovery, journal_path, journal, identity, service, drive_root_path,
                    api_factory=lambda: service, progress_callback=callback,
                )
                delete_generations = journal["delete_generations"]
                initial_inventory = journal["initial_recovery_inventory"]
                delete_orphan_manifests = journal.get("delete_orphan_manifests", [])
                preserved_orphan_manifests = journal.get("preserved_orphan_manifests", [])
            else:
                if [row["generation"] for row in retained] != generations:
                    raise RuntimeError("recovery retention selection did not preserve the two latest valid generations")
                if retained[-1]["kind"] != "complete":
                    raise ValueError("newest retained recovery generation is not complete")

        final_folder = _drive_recovery_folder(service, recovery, drive_root_path)
        inspected_final = _inspect_verified_recovery_tree(
            recovery, identity, service, final_folder["folder_id"], progress_callback=callback,
        )
        final_inventory = inspected_final["verified_generations"]
        preserved_by_name = {item["name"]: item for item in preserved_orphan_manifests}
        final_orphan_names = {
            item["manifest_file"]["name"] for item in inspected_final["orphan_manifests"]
        }
        if final_orphan_names != set(preserved_by_name):
            raise StorageCapacityError(
                "manifest-only recovery files differ from the identity-bound preserved set: "
                f"observed={sorted(final_orphan_names)}, expected={sorted(preserved_by_name)}"
            )
        for orphan in inspected_final["orphan_manifests"]:
            expected_orphan = preserved_by_name[orphan["manifest_file"]["name"]]
            actual_orphan = orphan["manifest_file"]
            if (
                orphan["generation"] != expected_orphan["generation"]
                or actual_orphan["drive_file_id"] != expected_orphan["drive_file_id"]
                or actual_orphan["size_bytes"] != expected_orphan["size_bytes"]
                or actual_orphan["local_sha256"] != expected_orphan["local_sha256"]
                or actual_orphan["can_edit"] is not True
            ):
                raise StorageCapacityError(
                    f"preserved orphan manifest {actual_orphan['name']} changed at final read-back"
                )
        retained_generations = [row["generation"] for row in final_inventory]
        if len(final_inventory) != RECOVERY_GENERATIONS_PER_RUN:
            raise ValueError(
                "recovery retention reconciliation did not converge to the required two verified generations: "
                f"found {retained_generations}"
            )
        if final_inventory[-1]["kind"] != "complete":
            raise ValueError("newest retained recovery generation is not the terminal complete checkpoint")
        if lease.lost.is_set():
            raise RuntimeError("matrix lease was lost during finalization; completion registration was not attempted")

        callback({"stage": "completed_run_storage_measurement_start", "run_key": record["run_key"]})
        measurement = measure_completed_production_run(
            plan, record, None, output, checkpoint_namespace,
        )
        journal_size = journal_path.stat().st_size if journal_path.is_file() else 0
        deleted_file_records = [
            item
            for row in delete_generations
            for item in (row.get("files", []) if isinstance(row, dict) else [])
        ]
        deleted_names = set(journal.get("deleted_files", [])) if journal_path.is_file() else set()
        deleted_orphan_manifest_records = [
            item for item in delete_orphan_manifests if item["name"] in deleted_names
        ]
        preserved_orphan_manifest_records = list(preserved_orphan_manifests)
        # The journal is stable after RETENTION_VERIFIED; this compact addition keeps
        # the registration measurement identical across a safe retry.
        if journal_path.is_file():
            journal_digest = json.loads(journal_path.read_text(encoding="utf-8"))["journal_sha256"]
        else:
            journal_digest = None
        measurement["recovery_retention_finalization"] = {
            "retained_generations": retained_generations,
            "deleted_generations": [row["generation"] for row in delete_generations],
            "deleted_file_count": len(deleted_file_records) + len(deleted_orphan_manifest_records),
            "deleted_orphan_manifest_count": len(deleted_orphan_manifest_records),
            "deleted_orphan_manifest_bytes": sum(
                int(item["size_bytes"]) for item in deleted_orphan_manifest_records
            ),
            "preserved_orphan_manifest_count": len(preserved_orphan_manifest_records),
            "preserved_orphan_manifest_bytes": sum(
                int(item["size_bytes"]) for item in preserved_orphan_manifest_records
            ),
            "deleted_payload_bytes": sum(
                int(item["size_bytes"]) for item in deleted_file_records if item.get("role") == "payload"
            ),
            "deleted_manifest_bytes": sum(
                int(item["size_bytes"]) for item in deleted_file_records if item.get("role") == "manifest"
            ) + sum(int(item["size_bytes"]) for item in deleted_orphan_manifest_records),
            "retention_journal_path": str(journal_path) if journal_path.is_file() else None,
            "retention_journal_sha256": journal_digest,
            "retention_journal_bytes": journal_size,
        }
        callback({"stage": "completion_registration_start", "run_key": record["run_key"]})
        registry, registered = _record_completed_run(
            output_root_path, plan, record, validated, storage_measurement=measurement,
        )
        if lease.lost.is_set():
            raise RuntimeError("matrix lease was lost after completion registration; the exact idempotent registry entry remains persisted")
        callback({
            "stage": "completion_registration_complete", "run_key": record["run_key"],
            "newly_registered": registered,
            "validated_completion_count": len(registry["completed_runs"]),
        })
        return {
            "status": "FINALIZED_COMPLETED_RUN", "run_key": record["run_key"],
            "run_id": validated["run_id"], "artifact_manifest_sha256": validated["artifact_manifest_sha256"],
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "recovery_inventory": initial_inventory,
            "recovery_retention": {
                "expected_generation_count": RECOVERY_GENERATIONS_PER_RUN,
                "retained_generations": retained_generations,
                "deleted_generations": [row["generation"] for row in delete_generations],
                "deleted_orphan_manifests": [item["name"] for item in deleted_orphan_manifest_records],
                "preserved_orphan_manifests": [item["name"] for item in preserved_orphan_manifest_records],
                "orphan_manifest_reconciliation": (
                    {
                        **journal.get("orphan_manifest_reconciliation", {"status": "NOT_REQUIRED"}),
                        "deleted_manifest_records": deleted_orphan_manifest_records,
                        # Backward-compatible name used by earlier finalization
                        # reports; these records describe deleted manifests only.
                        "manifest_records": deleted_orphan_manifest_records,
                        "preserved_manifest_records": preserved_orphan_manifest_records,
                    }
                    if journal_path.is_file() else {"status": "NOT_REQUIRED"}
                ),
                "journal_path": str(journal_path) if journal_path.is_file() else None,
                "rotation_explanation": rotation_explanation,
            },
            "task_boundary_checkpoint_count": measurement["task_boundary_checkpoints"]["generation_count"],
            "task_boundary_checkpoint_preflight": task_boundary_preflight,
            "storage_measurement": measurement,
            "completion_registered": registered,
            "validated_completion_count": len(registry["completed_runs"]),
            "note": "finalization only; no dataset was loaded, model initialized, replay generated, or training run launched",
        }


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


def reuse_validated_first_run_measurement(
    plan: dict[str, Any], record: dict[str, Any], output_root: str | Path,
) -> dict[str, Any]:
    """Validate finalized run 1 using the plan-level production results root.

    ``output_root`` is the matrix ``results`` root. Per-run artifacts are
    derived internally; callers must not pass ``run_output_root(...)`` here.
    """
    if not plan.get("runs") or record != plan["runs"][0] or int(record.get("run_index", -1)) != 1:
        raise ValueError("only frozen completed run 1 can be reused without execution")
    root = Path(output_root).expanduser().resolve()
    registry = _read_matrix_registry(root, plan)
    entry = registry["completed_runs"].get(record["run_key"])
    if not isinstance(entry, dict):
        raise ValueError("finalized first run is not present in the existing completion registry; refusing to repair or rerun it")
    validated = _validate_completed_run(
        plan, record, None, run_output_root(root, record), synthetic=False,
    )
    if (
        entry.get("run_id") != validated.get("run_id")
        or entry.get("artifact_manifest_sha256") != validated.get("artifact_manifest_sha256")
    ):
        raise ValueError("registered first-run completion differs from its validated artifact manifest")
    measurement = entry.get("storage_measurement")
    if (
        not isinstance(measurement, dict)
        or measurement.get("status") != "MEASURED_VALIDATED_PRODUCTION_RUN"
        or measurement.get("run_key") != record["run_key"]
        or measurement.get("run_id") != validated.get("run_id")
        or type(measurement.get("max_checkpoint_payload_bytes")) is not int
        or measurement["max_checkpoint_payload_bytes"] <= 0
        or type(measurement.get("run_output_bytes")) is not int
        or measurement["run_output_bytes"] < 0
    ):
        raise ValueError("registered first run lacks its original validated storage measurement; refusing remeasurement/finalization")
    return measurement


class _MatrixLeaseHeartbeat:
    def __init__(self, path: Path, *, interval_seconds: float = 60.0, ttl_seconds: float = 300.0,
                 storage_adapter=None):
        if storage_adapter is None:
            from revision_r1.locking import RunLease
            self.lease = RunLease(path, ttl_seconds=ttl_seconds)
        else:
            from revision_r1.drive_storage import DriveApiLease
            logical_path = storage_adapter.logical_path_for_local(path)
            self.lease = DriveApiLease(
                storage_adapter, logical_path, ttl_seconds=ttl_seconds,
            )
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


def _api_storage_include_nonlease(logical_path: Path, metadata: dict[str, Any]) -> bool:
    del metadata
    return logical_path.name not in {
        ".run.lease.json", ".matrix_execution.lease.json", ".recovering",
    } and not logical_path.name.endswith(".tmp")


def _hydrate_api_folder(storage_adapter, logical_folder: Path, local_folder: Path) -> list[dict[str, Any]]:
    if storage_adapter is None:
        raise StorageCapacityError("production storage adapter is missing; DriveFS fallback is disabled")
    try:
        return storage_adapter.download_tree(
            logical_folder, local_folder, include=_api_storage_include_nonlease,
        )
    except FileNotFoundError:
        return []


def _hydrate_api_run_state(storage_adapter, root: Path, plan: dict[str, Any], record: dict[str, Any],
                           output: Path, checkpoints: Path, *, include_recovery: bool = True) -> None:
    if storage_adapter is None:
        raise StorageCapacityError("production requires explicit OAuth API storage with local staging")
    logical_output = storage_adapter.logical_path_for_local(output)
    _hydrate_api_folder(storage_adapter, logical_output, output)
    # Recovery generations are sufficient to resume. Task-boundary artifacts
    # remain durable on Drive and are not downloaded into local staging for a
    # pending run (they can exceed 13 GB for the already-finalized first run).
    if include_recovery:
        logical_recovery = storage_adapter.logical_path_for_local(checkpoints / "recovery")
        _hydrate_api_folder(storage_adapter, logical_recovery, checkpoints / "recovery")
    _rebind_staged_completion_paths(output)


def _hydrate_api_matrix_registry(storage_adapter, root: Path, plan: dict[str, Any]) -> None:
    if storage_adapter is None:
        raise StorageCapacityError("production matrix registry requires explicit OAuth API storage")
    registry_path = _matrix_registry_path(root, plan)
    try:
        storage_adapter.download_file(storage_adapter.logical_path_for_local(registry_path), registry_path)
    except FileNotFoundError:
        # A missing registry is acceptable only as the first new run; the
        # caller still checks the identity-bound Run 1 completion requirement.
        return


def matrix_registry_evidence(
    storage_adapter, root: str | Path, plan: dict[str, Any],
) -> dict[str, Any]:
    """Report the exact API-backed plan registry after verifying its staged bytes.

    ``root`` is the production results root, not an individual run's output
    directory. Missing or stale registry content is an error; this function
    never creates or repairs a completion registry.
    """
    if storage_adapter is None:
        raise StorageCapacityError("matrix registry evidence requires explicit OAuth API storage")
    results_root = Path(root).expanduser().resolve()
    local_path = _matrix_registry_path(results_root, plan)
    logical_path = storage_adapter.logical_path_for_local(local_path)
    if not local_path.is_file() or local_path.is_symlink():
        raise StorageCapacityError(
            "canonical completion registry is not present in local staging after API hydration; "
            f"plan_sha256={plan.get('plan_sha256')}, logical_path={logical_path}, local_path={local_path}; "
            "no replacement registry was created"
        )
    metadata = storage_adapter.exact_file(logical_path)
    if metadata is None:
        raise StorageCapacityError(
            f"canonical completion registry is not visible by exact Drive path: {logical_path}"
        )
    expected_parent_id = storage_adapter.folder_id(logical_path.parent, create=False)
    observed_parents = [str(value) for value in metadata.get("parents", [])]
    local_bytes = local_path.read_bytes()
    local_size = len(local_bytes)
    local_md5 = hashlib.md5(local_bytes, usedforsecurity=False).hexdigest()
    try:
        remote_size = int(metadata.get("size"))
    except (TypeError, ValueError, OverflowError):
        remote_size = -1
    if (
        not isinstance(metadata.get("id"), str) or not metadata["id"]
        or metadata.get("name") != logical_path.name
        or expected_parent_id not in observed_parents
        or remote_size != local_size
        or metadata.get("md5Checksum") != local_md5
    ):
        expected_metadata = {
            "name": logical_path.name,
            "parent_id": expected_parent_id,
            "size": local_size,
            "md5": local_md5,
        }
        observed_metadata = {
            "file_id": metadata.get("id"),
            "name": metadata.get("name"),
            "parents": observed_parents,
            "size": remote_size,
            "md5": metadata.get("md5Checksum"),
        }
        raise StorageCapacityError(
            "canonical completion registry metadata does not match the staged API bytes; "
            f"expected={json.dumps(expected_metadata, sort_keys=True)}, "
            f"observed={json.dumps(observed_metadata, sort_keys=True)}"
        )
    registry = _read_matrix_registry(results_root, plan)
    completed_keys = [
        row["run_key"] for row in plan["runs"]
        if row["run_key"] in registry["completed_runs"]
    ]
    return {
        "logical_drive_path": str(logical_path),
        "drive_file_id": str(metadata["id"]),
        "drive_parent_id": expected_parent_id,
        "local_staging_path": str(local_path),
        "plan_sha256": plan["plan_sha256"],
        "registry_bytes": local_size,
        "registry_md5": local_md5,
        "completed_run_keys": completed_keys,
    }


def _rebind_staged_completion_paths(run_output: Path) -> None:
    """Rebind historical COMPLETE marker paths only in local staging.

    The marker's artifact bytes/hashes and run identity are preserved. This is
    needed because older releases recorded DriveFS absolute paths, whereas the
    API-backed runtime validates their downloaded copies under local staging.
    These rewritten staging records are never uploaded for an already-complete
    historical run.
    """
    registry_root = run_output / "registry" / "runs"
    if not registry_root.is_dir():
        return
    for marker_path in registry_root.glob("*/COMPLETE.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            registry_path = marker_path.parent / "registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, dict) or registry.get("completion_artifacts") != artifacts:
            continue
        rebound = {}
        valid = True
        for label, item in artifacts.items():
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                valid = False
                break
            local_file = run_output / Path(item["path"]).name
            if (
                not local_file.is_file() or local_file.is_symlink()
                or local_file.stat().st_size != item.get("bytes")
                or _file_sha256_path(local_file) != item.get("sha256")
            ):
                valid = False
                break
            rebound[label] = {**item, "path": str(local_file.resolve())}
        if not valid:
            continue
        marker["artifacts"] = rebound
        registry["completion_artifacts"] = rebound
        _atomic_json(marker_path, marker)
        _atomic_json(registry_path, registry)


def _file_sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _api_publish_file(storage_adapter, local_path: str | Path) -> dict[str, Any]:
    local = Path(local_path).expanduser().resolve()
    if not local.is_file() or local.is_symlink():
        raise StorageCapacityError(f"refusing to publish a missing/symlinked local staging artifact: {local}")
    logical = storage_adapter.logical_path_for_local(local)
    return storage_adapter.publish_file(
        local, logical,
        allow_foreign_owner_update=storage_adapter.is_mutable_control_path(logical),
    )


def _api_publish_checkpoint_generation(storage_adapter, local_directory: Path, identity: dict[str, Any],
                                       generation: int) -> list[dict[str, Any]]:
    manifest_path = local_directory / f"generation-{int(generation):06d}.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageCapacityError(f"checkpoint generation manifest is unreadable before API publication: {manifest_path}") from exc
    payload_name = manifest.get("payload_file") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict) or manifest.get("generation") != int(generation)
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != _sha256_json(identity)
        or payload_name != f"generation-{int(generation):06d}.pt"
    ):
        raise StorageCapacityError(f"checkpoint generation {generation} failed identity/manifest validation before upload")
    payload_path = local_directory / payload_name
    if (
        not payload_path.is_file() or payload_path.is_symlink()
        or payload_path.stat().st_size != manifest.get("payload_bytes")
        or _file_sha256_path(payload_path) != manifest.get("payload_sha256")
    ):
        raise StorageCapacityError(f"checkpoint generation {generation} payload failed local integrity validation")
    # Publish payload first and manifest last. The manifest is the commit record
    # and cannot become visible before its exact payload is verified by Drive.
    payload_result = _api_publish_file(storage_adapter, payload_path)
    manifest_result = _api_publish_file(storage_adapter, manifest_path)
    return [payload_result, manifest_result]


def _api_delete_checkpoint_file(storage_adapter, local_path: str | Path) -> bool:
    local = Path(local_path).expanduser().resolve()
    logical = storage_adapter.logical_path_for_local(local)
    metadata = storage_adapter.exact_file(logical)
    if metadata is None:
        return False
    try:
        storage_adapter.delete_exact(logical, expected_file_id=str(metadata["id"]))
    except Exception as exc:
        raise StorageCapacityError(
            f"API checkpoint retention could not delete the exact verified old file {metadata.get('id')}; local/remote copy preserved"
        ) from exc
    return True


def _publish_api_tree(storage_adapter, local_root: Path) -> list[dict[str, Any]]:
    if storage_adapter is None:
        raise StorageCapacityError("durable publication requires the explicit OAuth API adapter")
    logical_root = storage_adapter.logical_path_for_local(local_root)
    return storage_adapter.publish_tree(
        local_root, logical_root, include=lambda logical, _local: _api_storage_include_nonlease(logical, {}),
    )


def _require_canonical_storage_api_factory(api_factory) -> None:
    if _BOUND_STORAGE_SESSION is None or _BOUND_DRIVE_API_SERVICE is None:
        raise StorageCapacityError("canonical main-account Drive storage session has not been verified; no production run started")
    if (
        _BOUND_STORAGE_SESSION.get("status") != "PASS"
        or _BOUND_STORAGE_SESSION.get("storage_account_email") != CANONICAL_STORAGE_ACCOUNT_EMAIL
        or _BOUND_STORAGE_SESSION.get("canonical_revision_folder_id") != CANONICAL_REVISION_FOLDER_ID
    ):
        raise StorageCapacityError("canonical storage-session identity is invalid; no production run started")
    if api_factory is None or api_factory() is not _BOUND_DRIVE_API_SERVICE:
        raise StorageCapacityError("production operation did not receive the exact verified Drive API client; no run started")


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
    api_factory=None,
    storage_adapter=None,
    logical_output_root: str | Path | None = None,
    logical_checkpoint_root: str | Path | None = None,
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
    if not synthetic:
        _require_canonical_storage_api_factory(api_factory)
        if storage_adapter is None:
            raise StorageCapacityError("production requires main-account API storage plus local staging; DriveFS writes are disabled")
        if storage_adapter.service is not api_factory():
            raise StorageCapacityError("storage adapter is not bound to the exact verified canonical OAuth API service")
        if logical_output_root is None or logical_checkpoint_root is None:
            raise StorageCapacityError("production must bind local staging to explicit logical Drive roots")
    elif storage_adapter is not None and (api_factory is None or storage_adapter.service is not api_factory()):
        raise StorageCapacityError("synthetic API-storage validation is bound to a different Drive API service")
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
        if storage_snapshot.get("canonical_storage_identity_verified") is not True:
            raise StorageCapacityError("production run requires verified canonical main-account storage identity; no run started")
        if (
            storage_snapshot.get("write_capability_verified") is not True
            or storage_snapshot.get("new_file_owner_verified") is not True
        ):
            raise StorageCapacityError("production Drive destination write access/new-file owner was not verified; no run started")
        if storage_snapshot.get("permanent_recovery_delete_verified") is not True:
            raise StorageCapacityError(
                "exact owned Drive files.delete capability was not verified; two-generation recovery retention "
                "cannot be maintained without risking Drive Trash/quota growth; no run started"
            )
        if storage_snapshot.get("quota_status") == "UNKNOWN_CAPACITY":
            raise StorageCapacityError(
                "Drive capacity is unverifiable for the authenticated quota account; "
                + str(storage_snapshot.get("quota_capacity_note") or "no numeric available-space value was returned")
                + "; no run started"
            )
        if storage_adapter is None:
            raise StorageCapacityError("production capacity evidence has no API storage adapter")
        output_binding = (storage_snapshot.get("output_destination") or {}).get("configured_path")
        checkpoint_binding = (storage_snapshot.get("checkpoint_destination") or {}).get("configured_path")
        if (
            output_binding != str(Path(logical_output_root).expanduser().absolute())
            or checkpoint_binding != str(Path(logical_checkpoint_root).expanduser().absolute())
        ):
            raise StorageCapacityError("capacity evidence is bound to different logical Drive roots; no run started")
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

    if storage_adapter is not None:
        _hydrate_api_run_state(storage_adapter, root, plan, record, output, checkpoints)
        # The matrix registry is the authoritative cross-session completion
        # index. Refresh it before counting or registering this run so a fresh
        # local staging directory cannot report an existing completion as new.
        _hydrate_api_matrix_registry(storage_adapter, root, plan)

    schedule_for_runner = None if synthetic else schedule_path
    remote_checkpoint = checkpoints if storage_adapter is not None else None
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
    drive_write_timings = {"upload_seconds": 0.0, "verification_seconds": 0.0, "pruning_seconds": 0.0}
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

    if matrix_lease is None and storage_adapter is not None:
        storage_adapter.ensure_lease_record(storage_adapter.logical_path_for_local(matrix_lock_path))
    lease_context = (
        _MatrixLeaseHeartbeat(matrix_lock_path, storage_adapter=storage_adapter)
        if matrix_lease is None else contextlib.nullcontext(matrix_lease)
    )
    with lease_context as lease:
        previous_mirror = runner_module.mirror_checkpoint_generations
        previous_run_registry = runner_module.RunRegistry
        previous_run_lease = runner_module.RunLease

        if storage_adapter is not None:
            from revision_r1.drive_storage import DriveApiLease

            class ApiRunRegistry(previous_run_registry):
                def begin_attempt(self, run_id, *, identity):
                    result = super().begin_attempt(run_id, identity=identity)
                    _publish_api_tree(storage_adapter, self.root)
                    return result

                def fail_attempt(self, run_id, reason):
                    result = super().fail_attempt(run_id, reason)
                    _publish_api_tree(storage_adapter, self.root)
                    return result

                def complete(self, run_id, artifacts):
                    _publish_api_tree(storage_adapter, self.root.parent)
                    result = super().complete(run_id, artifacts)
                    _publish_api_tree(storage_adapter, self.root)
                    return result

            def api_run_lease(path, *, ttl_seconds=900):
                logical = storage_adapter.logical_path_for_local(path)
                storage_adapter.ensure_lease_record(logical)
                return DriveApiLease(storage_adapter, logical, ttl_seconds=ttl_seconds)

            runner_module.RunRegistry = ApiRunRegistry
            runner_module.RunLease = api_run_lease

        def instrumented_mirror(local_path, drive_path, identity, **kwargs):
            nonlocal latest_drive, drive_write_bytes, drive_write_seconds
            started = time.perf_counter()
            destination = Path(drive_path).resolve()
            if storage_adapter is not None and destination.is_relative_to(checkpoints.resolve()):
                generation = kwargs.get("generation")
                published = False
                publication_results = []

                def publish_before_prune(old_path=None):
                    nonlocal published, publication_results
                    if generation is not None and not published:
                        publication_results = _api_publish_checkpoint_generation(
                            storage_adapter, destination, identity, int(generation),
                        )
                        published = True
                    if old_path is None:
                        return True
                    return _api_delete_checkpoint_file(storage_adapter, old_path)

                call_kwargs = dict(kwargs)
                if call_kwargs.get("prune_destination", True) and call_kwargs.get("keep", 2) is not None:
                    call_kwargs["delete_callback"] = publish_before_prune
                copied = previous_mirror(local_path, drive_path, identity, **call_kwargs)
                if generation is not None and not published:
                    publication_results = _api_publish_checkpoint_generation(
                        storage_adapter, destination, identity, int(generation),
                    )
            else:
                copied = previous_mirror(local_path, drive_path, identity, **kwargs)
            elapsed = time.perf_counter() - started
            if storage_adapter is not None and destination.is_relative_to(checkpoints.resolve()):
                logical_directory = storage_adapter.logical_path_for_local(destination)
                generation = kwargs.get("generation")
                if generation is not None:
                    manifest_path = destination / f"generation-{int(generation):06d}.manifest.json"
                    payload_path = destination / f"generation-{int(generation):06d}.pt"
                    readback_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if len(publication_results) == 2:
                        payload_result, manifest_result = publication_results
                    else:
                        payload_result = storage_adapter.publish_file(payload_path, logical_directory / payload_path.name)
                        manifest_result = storage_adapter.publish_file(manifest_path, logical_directory / manifest_path.name)
                    if readback_manifest.get("identity") != identity:
                        raise StorageCapacityError("local mirrored checkpoint identity changed before API publication")
                    latest_drive = {
                        "relative_root": str(destination.relative_to(checkpoints.resolve())),
                        "generation": int(readback_manifest["generation"]), "kind": readback_manifest["kind"],
                        "manifest_sha256": file_sha256(manifest_path),
                        "payload_sha256": readback_manifest["payload_sha256"],
                        "payload_bytes": int(readback_manifest["payload_bytes"]),
                        "verified_at_unix": time.time(),
                        "write_seconds": elapsed,
                        "upload_seconds": elapsed,
                        "verification_seconds": 0.0,
                        "pruning_seconds": 0.0,
                        "api_publication": [payload_result, manifest_result],
                        "retained_bytes_after_pruning": int(getattr(copied, "retained_bytes", 0) or 0),
                    }
                    drive_write_bytes += int(payload_result.get("write_bytes", 0)) + int(manifest_result.get("write_bytes", 0))
                drive_write_seconds += elapsed
                drive_write_timings["upload_seconds"] += elapsed
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
                    "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
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
                    "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
                    "validated_completion_count": completion_count,
                    "schedule_sha256": plan["schedule_sha256"],
                    "training_source_sha256": plan["training_source_sha256"],
                    "event_log_path": str(event_log_path),
                    "live_progress": formatted,
                    "matrix_lock_is_best_effort": storage_adapter is None,
                    "matrix_lock_backend": (
                        "canonical_drive_v2_etag_if_match" if storage_adapter is not None
                        else "local_filesystem_lock"
                    ),
                }
                _append_progress_jsonl(event_log_path, update)
                _atomic_json(progress_file, update)
                if storage_adapter is not None:
                    _api_publish_file(storage_adapter, event_log_path)
                    _api_publish_file(storage_adapter, progress_file)
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
                drive_delete_callback=(
                    (
                        (lambda path, **kwargs: _api_delete_checkpoint_file(storage_adapter, path))
                        if storage_adapter is not None else None
                    )
                ),
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
                "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
                "last_verified_drive_checkpoint": latest_drive,
                "schedule_sha256": plan["schedule_sha256"],
                "training_source_sha256": plan["training_source_sha256"],
                "last_progress": last_progress_record,
            }
            _append_progress_jsonl(event_log_path, failed_event)
            _atomic_json(progress_file, failed_event)
            if storage_adapter is not None:
                _api_publish_file(storage_adapter, event_log_path)
                _api_publish_file(storage_adapter, progress_file)
            callback(failed_event)
            raise
        finally:
            runner_module.mirror_checkpoint_generations = previous_mirror
            if storage_adapter is not None:
                runner_module.RunRegistry = previous_run_registry
                runner_module.RunLease = previous_run_lease
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
            if storage_adapter is not None:
                _api_publish_file(storage_adapter, event_log_path)
                _api_publish_file(storage_adapter, progress_file)
            callback(interrupted_event)
            return {
                **result, "output_path": str(output), "local_checkpoint_root": str(local_namespace),
                "drive_checkpoint_root": None if synthetic else str(checkpoints),
                "run_identity_sha256": run_identity_sha256,
                "drive_checkpoint_write_bytes": drive_write_bytes,
                "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
                "drive_checkpoint_write_seconds": drive_write_seconds,
                "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
                "completion_count": completion_count,
            }
        validated = _validate_completed_run(plan, record, bundle, output, synthetic=synthetic)
        storage_measurement = None if synthetic else measure_completed_production_run(
            plan, record, bundle, output, checkpoints,
        )
        matrix_registry, added = _record_completed_run(
            root, plan, record, validated, storage_measurement=storage_measurement,
        )
        if storage_adapter is not None:
            _api_publish_file(storage_adapter, _matrix_registry_path(root, plan))
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
            "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
            "completion_count": len(matrix_registry["completed_runs"]),
            "completion_registry_added_now": added,
            "storage_measurement": storage_measurement,
        }
        _append_progress_jsonl(event_log_path, final_progress)
        _atomic_json(progress_file, final_progress)
        if storage_adapter is not None:
            _api_publish_file(storage_adapter, event_log_path)
            _api_publish_file(storage_adapter, progress_file)
            # ApiRunRegistry publishes the immutable run artifacts before
            # writing COMPLETE.json, then publishes COMPLETE.json/registry.json
            # by exact API path. Do not re-upload a hydrated, locally rebound
            # completion marker here: its staging paths are session-specific.
        callback(final_progress)
        return {
            **result, "status": "ALREADY_VALIDATED" if result.get("status") == "ALREADY_VALIDATED" else result.get("status"),
            "output_path": str(output), "local_checkpoint_root": str(local_namespace),
            "drive_checkpoint_root": None if synthetic else str(checkpoints),
            "run_identity_sha256": run_identity_sha256,
            "drive_checkpoint_write_bytes": drive_write_bytes,
            "drive_checkpoint_write_bytes_total": prior_verified_drive_write_bytes + drive_write_bytes,
            "drive_checkpoint_write_seconds": drive_write_seconds,
            "drive_checkpoint_timing_seconds_this_attempt": dict(drive_write_timings),
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
    storage_adapter=None, logical_output_root: str | Path | None = None,
    logical_checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the first counted production run only after explicit enable + gates."""
    if launch_enabled is not True:
        raise RuntimeError("LAUNCH_FIRST_PRODUCTION_RUN is False; no production run started")
    _require_canonical_storage_api_factory(api_factory)
    if storage_adapter is None or storage_adapter.service is not api_factory():
        raise StorageCapacityError("first-run workflow requires the exact verified main-account API storage adapter")
    if logical_output_root is None or logical_checkpoint_root is None:
        raise StorageCapacityError("first-run workflow requires logical Drive roots separate from local staging")
    validate_runtime_bound_plan(plan, schedule_path, schedule_manifest_path, preflight)
    first = plan["runs"][0]
    if (first["dataset"], int(first["seed"]), first["arm"]) != FIRST_RUN:
        raise ValueError("frozen first run is not EMBER / seed 42 / Full Hybrid")
    local_output_root = Path(output_root).expanduser().resolve()
    _hydrate_api_matrix_registry(storage_adapter, local_output_root, plan)
    output_run = run_output_root(local_output_root, first)
    drive_checkpoint_run = run_checkpoint_root(checkpoint_root, first)
    _hydrate_api_run_state(storage_adapter, local_output_root, plan, first,
                           drive_checkpoint_run, include_recovery=False)
    matrix_registry = _read_matrix_registry(local_output_root, plan)
    if first["run_key"] in matrix_registry.get("completed_runs", {}):
        measurement = reuse_validated_first_run_measurement(plan, first, local_output_root)
        return {
            "status": "ALREADY_FINALIZED_COMPLETED_RUN",
            "run_key": first["run_key"],
            "run_index": 1,
            "output_path": str(output_run),
            "drive_checkpoint_root": str(drive_checkpoint_run),
            "storage_measurement": measurement,
            "completion_count": len(matrix_registry["completed_runs"]),
            "training_source_sha256": plan["training_source_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "training_started": False,
            "finalizer_rerun": False,
            "measurement_reused_without_recomputation": True,
        }
    _hydrate_api_run_state(storage_adapter, local_output_root, plan, first,
                           output_run, drive_checkpoint_run, include_recovery=True)
    run_status_path = (
        output_run / "registry" / "runs" / _run_id_for(first, synthetic=False) / "registry.json"
    )
    if run_status_path.is_file():
        try:
            existing_run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_run_status = {}
        if existing_run_status.get("status") == "completed":
            raise ValueError(
                "first-run artifacts are completed but absent from the frozen matrix registry; "
                "refusing to retrain, remeasure, rerun finalization, or register them again"
            )
    reference = load_v62_recovery_checkpoint_bound(v62_recovery_report_path)
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
        _api_publish_file(storage_adapter, _matrix_registry_path(local_output_root, plan))
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
    checkpoint_size_evidence = _checkpoint_payload_size_reference(
        drive_checkpoint_run,
        expected_identity=resume_gate["expected_identity"],
        selected_recovery_candidate=resume_gate["selected_integrity_verified_candidate"],
    )
    observed_output_bytes = _tree_file_bytes(output_run)
    observed_checkpoint_bytes = max(
        int(reference["max_checkpoint_payload_bytes"]),
        int(checkpoint_size_evidence["max_payload_bytes"]),
    )
    projection = project_storage(
        plan,
        observed_checkpoint_bytes=observed_checkpoint_bytes,
        checkpoint_reference_provenance=(
            "maximum of the verified v6.2 operational reference and observed exact-identity production "
            "recovery/task-boundary payloads; production payload envelope applies 1.5x + 16 MiB"
        ),
        checkpoint_size_evidence={
            **checkpoint_size_evidence,
            "v6_2_reference_payload_bytes": int(reference["max_checkpoint_payload_bytes"]),
            "projection_source_bytes": observed_checkpoint_bytes,
            "existing_exact_run_output_bytes": observed_output_bytes,
        },
        observed_output_bytes=observed_output_bytes,
    )
    snapshot = inspect_production_capacity(
        logical_output_root, logical_checkpoint_root, local_checkpoint_root, cache_root,
        api_factory=api_factory, storage_adapter=storage_adapter,
        current_run_output_path=output_run,
        current_run_checkpoint_path=drive_checkpoint_run,
        current_run_local_checkpoint_path=run_checkpoint_root(local_checkpoint_root, first),
        expected_run_identity=resume_gate["expected_identity"],
        verified_dataset_cache_bytes=_preflight_addressed_cache_bytes(preflight, cache_root),
    )
    require_run_capacity(snapshot, projection, scope="first_run")
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
        _api_publish_file(storage_adapter, preparation_progress_path)
        if progress_callback is not None:
            progress_callback(event)

    # Acquire before hashing/reading data so a second Colab cell cannot consume
    # a second dataset-sized RAM working set before discovering the matrix lock.
    matrix_lock_path = output_root_path / ".matrix_execution.lease.json"
    storage_adapter.ensure_lease_record(storage_adapter.logical_path_for_local(matrix_lock_path))
    matrix_lease = _MatrixLeaseHeartbeat(matrix_lock_path, storage_adapter=storage_adapter)
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
            logical_output_root, logical_checkpoint_root, local_checkpoint_root, cache_root,
            api_factory=api_factory, storage_adapter=storage_adapter,
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
            api_factory=api_factory,
            storage_adapter=storage_adapter, logical_output_root=logical_output_root,
            logical_checkpoint_root=logical_checkpoint_root,
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
        _api_publish_file(storage_adapter, report_path)
        return {**result, "first_run_storage_report": str(report_path), "post_run_storage_projection": post_run_projection}


def continue_production_matrix(
    plan: dict[str, Any], *, preflight: dict[str, Any], schedule_path: str | Path,
    schedule_manifest_path: str | Path, data_roots: dict[str, str | Path],
    cache_root: str | Path, output_root: str | Path,
    checkpoint_root: str | Path, local_checkpoint_root: str | Path,
    launch_enabled: bool = False, progress_callback=None, api_factory=None,
    storage_adapter=None, logical_output_root: str | Path | None = None,
    logical_checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """Continue/resume records 2–100 sequentially, skipping only validated completions."""
    if launch_enabled is not True:
        raise RuntimeError("LAUNCH_REMAINING_MATRIX is False; no matrix run started")
    _require_canonical_storage_api_factory(api_factory)
    if storage_adapter is None or storage_adapter.service is not api_factory():
        raise StorageCapacityError("matrix workflow requires the exact verified main-account API storage adapter")
    if logical_output_root is None or logical_checkpoint_root is None:
        raise StorageCapacityError("matrix workflow requires logical Drive roots separate from local staging")
    validate_runtime_bound_plan(plan, schedule_path, schedule_manifest_path, preflight)
    root = Path(output_root).expanduser().resolve()
    _hydrate_api_matrix_registry(storage_adapter, root, plan)
    first = plan["runs"][0]
    _hydrate_api_run_state(
        storage_adapter, root, plan, first, run_output_root(root, first),
        run_checkpoint_root(checkpoint_root, first), include_recovery=False,
    )
    matrix_lock_path = root / ".matrix_execution.lease.json"
    storage_adapter.ensure_lease_record(storage_adapter.logical_path_for_local(matrix_lock_path))
    lease = _MatrixLeaseHeartbeat(matrix_lock_path, storage_adapter=storage_adapter)
    skipped = 0
    launched_or_resumed = 0
    latest_projection: dict[str, Any] | None = None
    latest_capacity: dict[str, Any] | None = None
    with lease:
        # Refresh the small plan registry only after owning the matrix lease.
        # This prevents a worker that waited for the lease from acting on
        # registry bytes staged before the preceding owner completed.
        _hydrate_api_matrix_registry(storage_adapter, root, plan)
        registry_evidence = matrix_registry_evidence(storage_adapter, root, plan)
        registry = _read_matrix_registry(root, plan)
        next_pending_record = next((
            row for row in plan["runs"][1:]
            if row["run_key"] not in registry["completed_runs"]
        ), None)
        first_measurement = reuse_validated_first_run_measurement(plan, first, root)
        if progress_callback is not None:
            progress_callback({
                "status": "MATRIX_REGISTRY_VERIFIED",
                "stage": "matrix_registry_verified",
                "first_run_key": first["run_key"],
                "matrix_registry_evidence": registry_evidence,
                "first_run_status": "ALREADY_FINALIZED_COMPLETED_RUN",
                "next_pending_run_key": (
                    next_pending_record["run_key"] if next_pending_record is not None else None
                ),
            })
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
            _hydrate_api_run_state(
                storage_adapter, root, plan, record, output, checkpoints,
                include_recovery=True,
            )
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
                existing_registry = _read_matrix_registry(root, plan)
                existing_entry = existing_registry["completed_runs"].get(record["run_key"])
                if (
                    not isinstance(existing_entry, dict)
                    or existing_entry.get("artifact_manifest_sha256") != validated["artifact_manifest_sha256"]
                    or not isinstance(existing_entry.get("storage_measurement"), dict)
                ):
                    raise ValueError(
                        f"completed run {record['run_key']} lacks its exact API-published completion measurement; "
                        "refusing to infer checkpoint integrity/size from incomplete local staging"
                    )
                measurement = existing_entry["storage_measurement"]
                matrix_registry = existing_registry
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
                logical_output_root, logical_checkpoint_root, local_checkpoint_root, cache_root,
                api_factory=api_factory, storage_adapter=storage_adapter,
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
                logical_output_root, logical_checkpoint_root, local_checkpoint_root, cache_root,
                api_factory=api_factory, storage_adapter=storage_adapter,
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
                api_factory=api_factory,
                storage_adapter=storage_adapter, logical_output_root=logical_output_root,
                logical_checkpoint_root=logical_checkpoint_root,
            )
            if result.get("status") == "INTERRUPTED_RESUMABLE":
                return {
                    "status": "INTERRUPTED_RESUMABLE", "interrupted_run": record["run_key"],
                    "run_index": record["run_index"], "cursor": result.get("cursor"),
                    "validated_completion_count": completed_count,
                    "schedule_sha256": plan["schedule_sha256"],
                    "training_source_sha256": plan["training_source_sha256"],
                    "first_run_reused": first["run_key"],
                    "next_pending_run_key": record["run_key"],
                    "matrix_registry_evidence_at_start": registry_evidence,
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
            "first_run_reused": first["run_key"],
            "matrix_registry_evidence_at_start": registry_evidence,
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
    storage_adapter=None, logical_output_path: str | Path | None = None,
) -> Path:
    target = Path(output_path).expanduser().resolve()
    parent = target.parent
    if target.exists() or Path(str(target) + ".manifest.json").exists() or Path(str(target) + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite an existing {label} or sidecar: {target}")
    expected_destination = (
        str(Path(logical_output_path).expanduser().absolute())
        if storage_adapter is not None and logical_output_path is not None
        else str(parent)
    )
    observed_destination = (
        storage_snapshot.get("configured_path")
        if storage_adapter is not None else storage_snapshot.get("path")
    )
    if (
        observed_destination != expected_destination
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
    storage_snapshot: dict[str, Any], storage_adapter=None,
    logical_output_path: str | Path | None = None,
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
        label="lightweight results", storage_adapter=storage_adapter,
        logical_output_path=logical_output_path,
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
    storage_snapshot: dict[str, Any], storage_adapter=None,
    logical_output_path: str | Path | None = None,
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
        label="checkpoint backup", storage_adapter=storage_adapter,
        logical_output_path=logical_output_path,
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
