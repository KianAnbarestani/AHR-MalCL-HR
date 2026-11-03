"""Fail-closed storage checks for Colab local staging and mounted Drive."""

from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
from typing import Any


DEFAULT_STORAGE_RESERVE_BYTES = 512 * 1024 * 1024
CHECKPOINT_ESTIMATE_MULTIPLIER = 1.5
CHECKPOINT_ESTIMATE_FIXED_OVERHEAD_BYTES = 16 * 1024 * 1024
DRIVE_METADATA_SCOPE = "https://www.googleapis.com/auth/drive.metadata.readonly"
_CHECKPOINT_TEMP_PATTERN = re.compile(r"\.generation-\d+\.(?:pt|manifest\.json)\..+\.tmp")
_BOUND_DRIVE_API_SERVICE = None
_BOUND_DRIVE_API_ACCOUNT = None


class StorageCapacityError(RuntimeError):
    """Raised before a write when capacity is insufficient or unverifiable."""


def clear_verified_drive_api_service() -> None:
    """Clear the session-bound Drive client before a new storage-auth attempt."""
    global _BOUND_DRIVE_API_SERVICE, _BOUND_DRIVE_API_ACCOUNT
    _BOUND_DRIVE_API_SERVICE = None
    _BOUND_DRIVE_API_ACCOUNT = None


def bind_verified_drive_api_service(service, *, expected_account_email: str) -> str:
    """Bind a Drive service only after its authenticated user is confirmed."""
    global _BOUND_DRIVE_API_SERVICE, _BOUND_DRIVE_API_ACCOUNT
    expected = str(expected_account_email).strip().lower()
    if not expected:
        raise ValueError("expected storage account email must be explicit")
    try:
        account = service.about().get(fields="user(emailAddress)").execute()
    except Exception as exc:
        raise StorageCapacityError(
            "cannot verify the explicit Drive OAuth identity before binding storage operations"
        ) from None
    observed = str((account.get("user") or {}).get("emailAddress") or "").strip().lower()
    if observed != expected:
        clear_verified_drive_api_service()
        raise StorageCapacityError(
            f"storage account mismatch: expected {expected!r}, observed {observed!r}; "
            "no Drive capacity or checkpoint operation was allowed"
        )
    _BOUND_DRIVE_API_SERVICE = service
    _BOUND_DRIVE_API_ACCOUNT = expected
    return expected


def verified_drive_api_service(*, expected_account_email: str | None = None):
    """Return only the service bound by successful canonical-account verification."""
    if _BOUND_DRIVE_API_SERVICE is None or not _BOUND_DRIVE_API_ACCOUNT:
        raise StorageCapacityError(
            "canonical Drive API service has not been bound; run Cell A with the explicit main-account OAuth secrets first"
        )
    if expected_account_email is not None:
        expected = str(expected_account_email).strip().lower()
        if expected != _BOUND_DRIVE_API_ACCOUNT:
            raise StorageCapacityError(
                f"bound Drive service account {_BOUND_DRIVE_API_ACCOUNT!r} does not match required {expected!r}"
            )
    return _BOUND_DRIVE_API_SERVICE


def verified_drive_api_account_email() -> str:
    if _BOUND_DRIVE_API_SERVICE is None or not _BOUND_DRIVE_API_ACCOUNT:
        raise StorageCapacityError(
            "canonical Drive API account has not been bound; authenticate the explicit main-account OAuth first"
        )
    return str(_BOUND_DRIVE_API_ACCOUNT)


def effective_generation_limit(identity: dict[str, Any], requested: int | None) -> int | None:
    """Keep validation recovery bounded while leaving production retention intact."""
    if identity.get("execution_scope") == "synthetic_validation":
        return 2
    return requested


def should_persist_task_boundary_duplicate(identity: dict[str, Any]) -> bool:
    """Production keeps its boundary archive; synthetic recovery uses one stream."""
    return identity.get("execution_scope") != "synthetic_validation"


def cleanup_unpublished_checkpoint_temps(root: str | Path) -> list[str]:
    """Remove only orphaned atomic-write temp files in an exclusively owned run namespace.

    Callers must hold that run's execution lease. Published generation payloads,
    manifests, reports, logs, and unrelated temporary files are never touched.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    removed = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or not _CHECKPOINT_TEMP_PATTERN.fullmatch(path.name):
            continue
        path.unlink()
        removed.append(str(path))
    return removed


def require_available_bytes(
    path: str | Path,
    *,
    required_bytes: int,
    reserve_bytes: int = DEFAULT_STORAGE_RESERVE_BYTES,
    available_bytes: int,
    storage_kind: str,
) -> None:
    required_bytes = int(required_bytes)
    reserve_bytes = int(reserve_bytes)
    available_bytes = int(available_bytes)
    if min(required_bytes, reserve_bytes, available_bytes) < 0:
        raise ValueError("storage byte counts cannot be negative")
    total_required = required_bytes + reserve_bytes
    if available_bytes < total_required:
        raise StorageCapacityError(
            f"insufficient {storage_kind} storage before write to {Path(path)}: "
            f"required {total_required} bytes ({required_bytes} write + "
            f"{reserve_bytes} safety reserve), available {available_bytes} bytes; "
            "no checkpoint payload was written and existing verified recovery points were preserved"
        )


def drive_available_bytes_from_quota(quota: dict[str, Any]) -> int:
    """Return account-wide Drive free bytes, refusing incomplete quota metadata."""
    try:
        limit = int(quota["limit"])
        usage = int(quota["usage"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise StorageCapacityError(
            "cannot verify Google Drive free quota (Drive API omitted a finite limit or usage); "
            "GPU validation is stopped before checkpoint writes"
        ) from exc
    if limit < 0 or usage < 0:
        raise StorageCapacityError("Google Drive returned invalid negative storage quota values")
    if usage > limit:
        raise StorageCapacityError(f"Google Drive usage exceeds quota limit ({usage} > {limit})")
    return limit - usage


def query_drive_available_bytes(*, api_service=None, expected_account_email: str | None = None) -> int:
    """Query quota only through the explicitly bound canonical-user Drive API service."""
    try:
        service = api_service or verified_drive_api_service(expected_account_email=expected_account_email)
        fields = "user(emailAddress),storageQuota(limit,usage)"
        account = service.about().get(fields=fields).execute()
        observed = str((account.get("user") or {}).get("emailAddress") or "").strip().lower()
        expected = (
            str(expected_account_email).strip().lower()
            if expected_account_email is not None else _BOUND_DRIVE_API_ACCOUNT
        )
        if not expected or observed != expected:
            raise StorageCapacityError(
                f"storage account mismatch during quota query: expected {expected!r}, observed {observed!r}"
            )
        quota = account.get("storageQuota", {})
    except Exception as exc:
        if isinstance(exc, StorageCapacityError):
            raise
        raise StorageCapacityError(
            "cannot verify Google Drive free quota through the verified main-account OAuth service; "
            "no ADC fallback is permitted"
        ) from None
    return drive_available_bytes_from_quota(quota)


def is_colab_drive_path(path: str | Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    drive_root = Path("/content/drive").resolve()
    return resolved == drive_root or resolved.is_relative_to(drive_root)


def available_bytes(path: str | Path) -> tuple[int, str]:
    target = Path(path).expanduser()
    if is_colab_drive_path(target):
        return query_drive_available_bytes(), "Google Drive account"
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free), "local filesystem"
    except OSError as exc:
        raise StorageCapacityError(f"cannot determine local free space at {probe}: {exc}") from exc


def require_path_capacity(
    path: str | Path,
    *,
    required_bytes: int,
    reserve_bytes: int = DEFAULT_STORAGE_RESERVE_BYTES,
) -> dict[str, int | str]:
    free, kind = available_bytes(path)
    require_available_bytes(
        path,
        required_bytes=required_bytes,
        reserve_bytes=reserve_bytes,
        available_bytes=free,
        storage_kind=kind,
    )
    return {
        "path": str(Path(path)), "storage_kind": kind,
        "available_bytes": free, "required_bytes": int(required_bytes),
        "reserve_bytes": int(reserve_bytes),
    }


def estimate_checkpoint_bytes(value: Any) -> int:
    """Conservatively estimate a serialized checkpoint without materializing it."""
    seen: set[int] = set()

    def walk(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, (bytes, bytearray, memoryview)):
            return len(item)
        if isinstance(item, str):
            return len(item.encode("utf-8"))
        if isinstance(item, dict):
            return sum(walk(key) + walk(child) for key, child in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sum(walk(child) for child in item)
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            try:
                return int(item.numel()) * int(item.element_size())
            except Exception:
                return 0
        if hasattr(item, "nbytes"):
            try:
                return int(item.nbytes)
            except Exception:
                return 0
        if hasattr(item, "__dict__"):
            return walk(vars(item))
        return 0

    raw = walk(value)
    estimate = math.ceil(raw * CHECKPOINT_ESTIMATE_MULTIPLIER) + CHECKPOINT_ESTIMATE_FIXED_OVERHEAD_BYTES
    return max(CHECKPOINT_ESTIMATE_FIXED_OVERHEAD_BYTES, estimate)

