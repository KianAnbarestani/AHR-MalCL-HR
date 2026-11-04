"""Explicit-OAuth Google Drive storage with local-only staging.

This module never uses Colab DriveFS. Callers provide the already verified
main-account Drive API service and the canonical ReVision folder ID. Existing
files are addressed by exact parent/name resolution and then exact file ID;
ambiguous duplicate names are rejected. Uploads are read back through the API
and checked against size and Drive's server-calculated MD5.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable


class DriveStorageError(RuntimeError):
    """A Drive API storage operation could not be verified safely."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class AmbiguousDrivePathError(DriveStorageError):
    pass


class DriveLeaseError(DriveStorageError):
    pass


TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
READBACK_DELAYS_SECONDS = (0.0, 0.25, 0.75, 1.5)


def _escape_query(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _http_status(error: Exception) -> int | None:
    response = getattr(error, "resp", None)
    raw = getattr(response, "status", None) or getattr(error, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (ValueError, TypeError):
        return None


def _header_value(headers: Any, name: str) -> str | None:
    """Read an HTTP response header without assuming header-name casing."""
    if not hasattr(headers, "items"):
        return None
    for key, value in headers.items():
        if str(key).casefold() == name.casefold() and value is not None:
            return str(value)
    return None


def _execute_with_response_headers(request) -> tuple[Any, dict[str, Any]]:
    """Execute a googleapiclient request while capturing its HTTP response.

    ``googleapiclient.http.HttpRequest`` does not publish the response on a
    ``request.resp`` attribute. Its supported interface is a response callback.
    Capture only the ETag and header names needed for safe lease diagnostics.
    """
    add_callback = getattr(request, "add_response_callback", None)
    if not callable(add_callback):
        raise DriveLeaseError(
            "Drive API request cannot expose response headers through a response callback; "
            "conditional leases cannot be verified safely",
            diagnostics={"observed": {"response_callback_supported": False}},
        )
    captured: dict[str, Any] = {"etag": None, "header_names": []}

    def capture(response) -> None:
        if hasattr(response, "items"):
            captured["header_names"] = sorted(str(key) for key, _value in response.items())
        captured["etag"] = _header_value(response, "etag")

    add_callback(capture)
    result = request.execute()
    return result, captured


def _is_strong_etag(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not value.lstrip().casefold().startswith("w/")


def _retryable(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status = _http_status(error)
    return status in TRANSIENT_HTTP_STATUSES


def _safe_metadata_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    owners = value.get("owners") or []
    owner = owners[0].get("emailAddress") if owners and isinstance(owners[0], dict) else None
    return {
        "id": value.get("id"), "name": value.get("name"),
        "parents": value.get("parents"), "owner": owner,
        "mimeType": value.get("mimeType"),
        "ownedByMe": value.get("ownedByMe"), "size": value.get("size"),
        "quotaBytesUsed": value.get("quotaBytesUsed"),
        "md5Checksum": value.get("md5Checksum"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_media_upload(path: Path, mime_type: str):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), mimetype=mime_type, resumable=True, chunksize=8 << 20)


def _make_media_bytes_upload(payload: bytes, mime_type: str):
    from googleapiclient.http import MediaIoBaseUpload
    import io

    return MediaIoBaseUpload(io.BytesIO(payload), mimetype=mime_type, resumable=False)


def _execute_media_request(request, media_body):
    """Execute large resumable uploads through the API client's chunk loop."""
    resumable = getattr(media_body, "resumable", None)
    if callable(resumable) and resumable():
        if not callable(getattr(request, "next_chunk", None)):
            raise DriveStorageError("Drive client returned no resumable-upload chunk interface")
        response = None
        while response is None:
            _status, response = request.next_chunk(num_retries=0)
        return response
    return request.execute()


def _download_request_to_path(request, target: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        downloader = MediaIoBaseDownload(stream, request, chunksize=8 << 20)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
        stream.flush()
        os.fsync(stream.fileno())


class DriveApiStorage:
    """Path-mapped local staging whose durable writes use one explicit API."""

    def __init__(
        self, *, service, expected_account_email: str, revision_folder_id: str,
        logical_revision_root: str | Path = "/content/drive/MyDrive/ReVision",
        local_staging_root: str | Path = "/content/revision_R1_storage_staging",
        media_upload_factory: Callable[[Path, str], Any] | None = None,
        download_to_path: Callable[[Any, Path], None] | None = None,
        retry_delays: tuple[float, ...] = READBACK_DELAYS_SECONDS,
    ) -> None:
        self.service = service
        self.expected_account_email = str(expected_account_email).strip().lower()
        self.revision_folder_id = str(revision_folder_id)
        self.logical_revision_root = Path(os.path.abspath(str(Path(logical_revision_root).expanduser())))
        self.local_staging_root = Path(local_staging_root).expanduser().resolve()
        self.local_staging_root.mkdir(parents=True, exist_ok=True)
        self.media_upload_factory = media_upload_factory or _make_media_upload
        self.download_to_path = download_to_path or _download_request_to_path
        self.retry_delays = tuple(float(delay) for delay in retry_delays)
        self._root_bindings: list[tuple[Path, Path]] = []
        self._folder_cache: dict[tuple[str, ...], str] = {(): self.revision_folder_id}
        self._verify_account()

    def _verify_account(self) -> None:
        result = self.service.about().get(fields="user(emailAddress)").execute()
        observed = str((result.get("user") or {}).get("emailAddress") or "").strip().lower()
        if observed != self.expected_account_email:
            raise DriveStorageError(
                f"Drive storage API identity mismatch: expected {self.expected_account_email!r}, observed {observed!r}"
            )

    def capacity_snapshot(self) -> dict[str, Any]:
        response = self.service.about().get(fields="user(emailAddress),storageQuota(limit,usage)").execute()
        observed = str((response.get("user") or {}).get("emailAddress") or "").strip().lower()
        if observed != self.expected_account_email:
            raise DriveStorageError("Drive quota identity changed away from the verified storage account")
        quota = response.get("storageQuota") or {}
        try:
            limit = int(quota["limit"]) if quota.get("limit") is not None else None
            usage = int(quota["usage"]) if quota.get("usage") is not None else None
        except (TypeError, ValueError, OverflowError) as exc:
            raise DriveStorageError("Drive returned malformed storageQuota metadata") from exc
        available = max(0, limit - usage) if limit is not None and usage is not None else None
        return {"account": observed, "limit_bytes": limit, "usage_bytes": usage,
                "available_bytes": available,
                "status": "FINITE_VERIFIED" if available is not None else "UNKNOWN_CAPACITY"}

    def require_capacity(self, required_bytes: int, *, reserve_bytes: int = 512 << 20) -> dict[str, Any]:
        snapshot = self.capacity_snapshot()
        available = snapshot["available_bytes"]
        if available is None:
            raise DriveStorageError("canonical Drive free space is unknown; no upload was attempted")
        if available < int(required_bytes) + int(reserve_bytes):
            raise DriveStorageError(
                f"canonical Drive has {available} available bytes; required {int(required_bytes)} "
                f"plus {int(reserve_bytes)} reserve; upload stopped before write"
            )
        return snapshot

    def bind_root(self, logical_root: str | Path, local_root: str | Path) -> None:
        logical = Path(os.path.abspath(str(Path(logical_root).expanduser())))
        local = Path(local_root).expanduser().resolve()
        try:
            relative = logical.relative_to(self.logical_revision_root)
            local.relative_to(self.local_staging_root)
        except ValueError as exc:
            raise DriveStorageError("storage root must stay below its configured logical/local staging root") from exc
        if not relative.parts:
            raise DriveStorageError("bind a named storage subtree below ReVision, not the ReVision root itself")
        self._root_bindings = [item for item in self._root_bindings if item[0] != local]
        self._root_bindings.append((local, logical))
        self._root_bindings.sort(key=lambda item: len(item[0].parts), reverse=True)

    def logical_path_for_local(self, local_path: str | Path) -> Path:
        local = Path(local_path).expanduser().resolve()
        for local_root, logical_root in self._root_bindings:
            try:
                return logical_root / local.relative_to(local_root)
            except ValueError:
                continue
        raise DriveStorageError(f"local path is outside every registered API storage root: {local}")

    def is_mutable_control_path(self, logical_path: str | Path) -> bool:
        """Allow in-place edits only for known application control records.

        Historical records can be owned by an earlier compute account. The
        verified canonical API user may update those exact shared files when
        Drive reports canEdit, without transferring ownership. Scientific
        outputs, checkpoint payloads/manifests, plans, and exports remain
        immutable unless already byte-identical.
        """
        parts = self._relative_parts(logical_path)
        name = parts[-1]
        if name in {".matrix_execution.lease.json", ".run.lease.json"}:
            return True
        if name == "matrix_completion_registry.json":
            return "registry" in parts
        if name == "registry.json":
            return "registry" in parts
        return any(part in {"progress", "preparation_progress"} for part in parts)

    def _relative_parts(self, logical_path: str | Path) -> tuple[str, ...]:
        logical = Path(os.path.abspath(str(Path(logical_path).expanduser())))
        try:
            parts = logical.relative_to(self.logical_revision_root).parts
        except ValueError as exc:
            raise DriveStorageError(f"logical Drive path is outside canonical ReVision: {logical}") from exc
        if any(part in {"", ".", ".."} for part in parts):
            raise DriveStorageError("logical Drive path must stay within canonical ReVision")
        return tuple(parts)

    def _list_children(self, parent_id: str, *, name: str | None = None) -> list[dict[str, Any]]:
        query = f"'{_escape_query(parent_id)}' in parents and trashed=false"
        if name is not None:
            query += f" and name='{_escape_query(name)}'"
        matches: list[dict[str, Any]] = []
        page_token = None
        while True:
            response = self.service.files().list(
                q=query, pageSize=1000, pageToken=page_token,
                fields=("nextPageToken,files(id,name,mimeType,parents,owners(emailAddress),ownedByMe,"
                        "size,quotaBytesUsed,md5Checksum,capabilities(canEdit,canAddChildren,canDelete),version)"),
                includeItemsFromAllDrives=True, supportsAllDrives=True,
            ).execute()
            matches.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return matches

    @staticmethod
    def _one(matches: list[dict[str, Any]], *, label: str) -> dict[str, Any] | None:
        if len(matches) > 1:
            ids = sorted(str(item.get("id")) for item in matches)
            raise AmbiguousDrivePathError(f"multiple Drive objects match exact {label}; IDs={ids}")
        return matches[0] if matches else None

    def folder_id(self, logical_folder: str | Path, *, create: bool = False) -> str:
        parts = self._relative_parts(logical_folder)
        parent = self.revision_folder_id
        traversed: list[str] = []
        for name in parts:
            traversed.append(name)
            key = tuple(traversed)
            if key in self._folder_cache:
                parent = self._folder_cache[key]
                continue
            folder = self._one(
                [item for item in self._list_children(parent, name=name)
                 if item.get("mimeType") == "application/vnd.google-apps.folder"],
                label=f"folder {name!r} under {parent!r}",
            )
            if folder is None:
                if not create:
                    raise FileNotFoundError(f"Drive folder not found: {logical_folder}")
                parent_metadata = self.service.files().get(
                    fileId=parent, fields="id,capabilities(canAddChildren),parents",
                    supportsAllDrives=True,
                ).execute()
                if (parent_metadata.get("capabilities") or {}).get("canAddChildren") is not True:
                    raise DriveStorageError(f"main storage account cannot add children to Drive folder {parent!r}")
                created = self.service.files().create(
                    body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
                    fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,capabilities(canAddChildren)",
                    supportsAllDrives=True,
                ).execute()
                folder = self._one(self._list_children(parent, name=name), label=f"new folder {name!r}")
                if folder is None or folder.get("id") != created.get("id"):
                    raise DriveStorageError(f"Drive did not read back the exact created folder {name!r}")
                owner = self._owner(folder)
                if owner != self.expected_account_email or folder.get("ownedByMe") is not True:
                    raise DriveStorageError(f"new Drive folder {name!r} is not owned by the verified storage account")
            if parent not in [str(value) for value in folder.get("parents", [])]:
                raise DriveStorageError(f"Drive folder {name!r} is not in the exact expected parent {parent!r}")
            self._folder_cache[key] = str(folder["id"])
            parent = str(folder["id"])
        return parent

    @staticmethod
    def _owner(metadata: dict[str, Any]) -> str | None:
        owners = metadata.get("owners") or []
        return str(owners[0].get("emailAddress") or "").strip().lower() if owners else None

    def exact_file(self, logical_file: str | Path) -> dict[str, Any] | None:
        logical = Path(os.path.abspath(str(Path(logical_file).expanduser())))
        parent_id = self.folder_id(logical.parent, create=False)
        return self._one(self._list_children(parent_id, name=logical.name), label=f"file {logical}")

    def list_files(self, logical_folder: str | Path) -> list[dict[str, Any]]:
        """List exact immediate children after resolving canonical folder ancestry."""
        return self._list_children(self.folder_id(logical_folder, create=False))

    def _download_id(self, metadata: dict[str, Any], target: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
        file_id = str(metadata.get("id") or "")
        if not file_id:
            raise DriveStorageError("cannot download Drive content without an exact file ID")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(metadata.get("size") or 0)
        if expected_size <= 0:
            raise DriveStorageError(f"Drive metadata has no positive content size for file ID {file_id}")
        from .storage import DEFAULT_STORAGE_RESERVE_BYTES
        free = shutil.disk_usage(target.parent).free
        if free < expected_size + DEFAULT_STORAGE_RESERVE_BYTES:
            raise DriveStorageError(
                f"local staging has {free} free bytes; downloading file ID {file_id} needs "
                f"{expected_size} plus {DEFAULT_STORAGE_RESERVE_BYTES} reserve"
            )
        if (target.is_file() and not target.is_symlink() and target.stat().st_size == expected_size
                and metadata.get("md5Checksum") and _file_md5(target) == metadata.get("md5Checksum")):
            digest = _file_sha256(target)
            if expected_sha256 is not None and digest != expected_sha256:
                raise DriveStorageError(f"existing local staging file SHA-256 mismatch for Drive file ID {file_id}")
            return {"file_id": file_id, "bytes": expected_size, "sha256": digest,
                    "md5_checksum": metadata.get("md5Checksum"), "path": str(target),
                    "reused_verified_local_copy": True}
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".download", dir=target.parent)
        os.close(fd)
        temporary = Path(temp_name)
        try:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            self.download_to_path(request, temporary)
            actual_size = temporary.stat().st_size
            if expected_size and actual_size != expected_size:
                raise DriveStorageError(f"Drive download size mismatch for {file_id}: {actual_size} != {expected_size}")
            digest = _file_sha256(temporary)
            if expected_sha256 is not None and digest != expected_sha256:
                raise DriveStorageError(f"Drive download SHA-256 mismatch for {file_id}")
            os.replace(temporary, target)
            return {"file_id": file_id, "bytes": actual_size, "sha256": digest,
                    "md5_checksum": metadata.get("md5Checksum"), "path": str(target)}
        finally:
            temporary.unlink(missing_ok=True)

    def download_file(
        self, logical_file: str | Path, local_file: str | Path, *,
        expected_sha256: str | None = None, expected_bytes: int | None = None,
    ) -> dict[str, Any]:
        metadata = self.exact_file(logical_file)
        if metadata is None:
            raise FileNotFoundError(f"Drive file not found: {logical_file}")
        if expected_bytes is not None and int(metadata.get("size") or -1) != int(expected_bytes):
            raise DriveStorageError(f"Drive metadata size differs from expected bytes for {logical_file}")
        return self._download_id(metadata, Path(local_file).expanduser().resolve(), expected_sha256=expected_sha256)

    def publish_file(
        self, local_file: str | Path, logical_file: str | Path, *,
        allow_foreign_owner_update: bool = False,
    ) -> dict[str, Any]:
        local = Path(local_file).expanduser().resolve()
        if not local.is_file() or local.is_symlink():
            raise DriveStorageError(f"refusing to publish missing or symlinked staging file: {local}")
        logical = Path(os.path.abspath(str(Path(logical_file).expanduser())))
        parent_id = self.folder_id(logical.parent, create=True)
        existing = self._one(self._list_children(parent_id, name=logical.name), label=f"file {logical}")
        local_size = local.stat().st_size
        local_md5 = _file_md5(local)
        local_sha = _file_sha256(local)
        existing_owner = None
        existing_owned_by_me = None
        if existing is not None:
            if parent_id not in [str(value) for value in existing.get("parents", [])]:
                raise DriveStorageError(f"existing file ID {existing.get('id')} has a different parent")
            if existing.get("capabilities", {}).get("canEdit") is not True:
                raise DriveStorageError(f"storage account cannot edit exact existing file ID {existing.get('id')}")
            if (int(existing.get("size") or -1) == local_size and existing.get("md5Checksum") == local_md5):
                return {"status": "UNCHANGED_VERIFIED_METADATA", "file_id": existing["id"],
                        "parent_id": parent_id, "bytes": local_size, "sha256": local_sha,
                        "new_file": False, "write_bytes": 0,
                        "file_owner": self._owner(existing),
                        "owned_by_canonical_account": existing.get("ownedByMe") is True,
                        "quota_attribution": "existing_file_unchanged"}
            existing_owner = self._owner(existing)
            existing_owned_by_me = existing.get("ownedByMe")
            canonical_owner = (
                existing_owner == self.expected_account_email and existing_owned_by_me is True
            )
            if not canonical_owner and (
                not allow_foreign_owner_update
                or not self.is_mutable_control_path(logical)
                or not existing_owner
                or type(existing_owned_by_me) is not bool
                or existing_owned_by_me is True
            ):
                raise DriveStorageError(
                    "refusing to replace foreign-owned immutable Drive content or a control file without explicit "
                    "shared-editor policy; preserved without upload; "
                    f"expected_owner={self.expected_account_email}, observed={_safe_metadata_summary(existing)}"
                )
            if not canonical_owner and existing.get("capabilities", {}).get("canEdit") is not True:
                raise DriveStorageError(
                    "canonical storage account lacks verified canEdit for the exact shared control file; "
                    f"observed={_safe_metadata_summary(existing)}"
                )
            # Reserve the full replacement size: Drive may retain an old
            # revision while the new media body is being committed.
            self.require_capacity(local_size)
            file_id = str(existing["id"])
            new_file = False
        else:
            self.require_capacity(local_size)
            generated = self.service.files().generateIds(count=1, space="drive").execute()
            ids = generated.get("ids", [])
            if len(ids) != 1 or not isinstance(ids[0], str):
                raise DriveStorageError("Drive did not return exactly one pre-generated ID for idempotent file creation")
            file_id = ids[0]
            new_file = True
        delays = (0.0, *self.retry_delays)
        response = None
        last_error = None
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            media_body = self.media_upload_factory(local, "application/octet-stream")
            if new_file:
                request = self.service.files().create(
                    body={"id": file_id, "name": logical.name,
                          "mimeType": "application/octet-stream", "parents": [parent_id]},
                    media_body=media_body, media_mime_type="application/octet-stream",
                    supportsAllDrives=True,
                    fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed,md5Checksum,capabilities(canEdit)",
                )
            else:
                request = self.service.files().update(
                    fileId=file_id, media_body=media_body,
                    media_mime_type="application/octet-stream", supportsAllDrives=True,
                    fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed,md5Checksum,capabilities(canEdit)",
                )
            try:
                response = _execute_media_request(request, media_body)
                break
            except Exception as exc:
                last_error = exc
                if not _retryable(exc) or attempt + 1 == len(delays):
                    break
                # If a create reached Drive but its response was lost, look up
                # only the pre-generated ID. A same-name search is never used
                # to choose among ambiguous files.
                if new_file:
                    try:
                        observed = self.service.files().get(
                            fileId=file_id,
                            fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed,md5Checksum",
                            supportsAllDrives=True,
                        ).execute()
                        if (observed.get("name") == logical.name
                                and parent_id in [str(value) for value in observed.get("parents", [])]
                                and int(observed.get("size") or -1) == local_size
                                and observed.get("md5Checksum") == local_md5):
                            response = observed
                            break
                    except Exception as get_error:
                        if _http_status(get_error) != 404 and not _retryable(get_error):
                            last_error = get_error
                            break
        if response is None:
            raise DriveStorageError(
                f"Drive upload failed for exact logical path {logical}: "
                f"{type(last_error).__name__ if last_error else 'no response'}"
            ) from last_error
        if response.get("id") != file_id:
            raise DriveStorageError("Drive upload response changed the preselected exact file ID")
        verified = None
        last_readback_error = None
        expected_readback = {
            "id": file_id, "name": logical.name, "parent_id": parent_id,
            "size": local_size, "md5Checksum": local_md5,
        }
        readback_delays = (0.0, *self.retry_delays)
        for attempt, delay in enumerate(readback_delays):
            if delay:
                time.sleep(delay)
            try:
                verified = self.service.files().get(
                    fileId=file_id,
                    fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed,md5Checksum,capabilities(canEdit)",
                    supportsAllDrives=True,
                ).execute()
            except Exception as exc:
                last_readback_error = exc
                if not _retryable(exc) or attempt + 1 == len(readback_delays):
                    raise DriveStorageError(
                        f"Drive upload metadata read-back failed for exact file ID {file_id}: {type(exc).__name__}"
                    ) from exc
                continue
            base_metadata_matches = (
                isinstance(verified, dict) and verified.get("id") == file_id
                and verified.get("name") == logical.name
                and parent_id in [str(value) for value in verified.get("parents", [])]
                and int(verified.get("size") or -1) == local_size
                and verified.get("md5Checksum") == local_md5
            )
            new_file_identity_visible = (
                not new_file or (
                    self._owner(verified) is not None and type(verified.get("ownedByMe")) is bool
                    and verified.get("quotaBytesUsed") is not None
                )
            ) if isinstance(verified, dict) else False
            if base_metadata_matches and new_file_identity_visible:
                break
            if (isinstance(verified, dict)
                    and verified.get("parents")
                    and parent_id not in [str(value) for value in verified.get("parents", [])]):
                raise DriveStorageError(
                    "confirmed Drive parent mismatch after upload; "
                    f"expected_parent={parent_id}, observed={_safe_metadata_summary(verified)}"
                )
            observed_owner = self._owner(verified) if isinstance(verified, dict) else None
            if new_file and observed_owner is not None and observed_owner != self.expected_account_email:
                raise DriveStorageError(
                    "confirmed new-file ownership mismatch after upload; "
                    f"expected_owner={self.expected_account_email}, observed={_safe_metadata_summary(verified)}",
                    diagnostics={
                        "phase": "new_file_identity_readback",
                        "expected": {
                            "file_id": file_id, "parent_id": parent_id,
                            "owner": self.expected_account_email, "ownedByMe": True,
                            "size": local_size, "quotaBytesUsed": "positive integer",
                            "api_visible": True,
                        },
                        "observed": {
                            "file_id": verified.get("id"),
                            "parent_id": (verified.get("parents") or [None])[0],
                            "owner": self._owner(verified), "ownedByMe": verified.get("ownedByMe"),
                            "size": int(verified.get("size")) if verified.get("size") is not None else None,
                            "quotaBytesUsed": (int(verified.get("quotaBytesUsed"))
                                               if verified.get("quotaBytesUsed") is not None else None),
                            "api_visible": True,
                        },
                    },
                )
        if (not isinstance(verified, dict) or verified.get("id") != file_id
                or verified.get("name") != logical.name
                or parent_id not in [str(value) for value in verified.get("parents", [])]
                or int(verified.get("size") or -1) != local_size
                or verified.get("md5Checksum") != local_md5):
            raise DriveStorageError(
                "Drive upload metadata remained stale or mismatched after bounded read-back attempts; "
                f"expected={expected_readback}, observed={_safe_metadata_summary(verified)}, "
                f"last_error={type(last_readback_error).__name__ if last_readback_error else None}; "
                "local staging file preserved"
            )
        if new_file:
            if self._owner(verified) != self.expected_account_email or verified.get("ownedByMe") is not True:
                raise DriveStorageError(
                    "new Drive file ownership metadata remains incomplete/mismatched after bounded read-back; "
                    f"expected_owner={self.expected_account_email}, observed={_safe_metadata_summary(verified)}",
                    diagnostics={
                        "phase": "new_file_identity_readback",
                        "expected": {
                            "file_id": file_id, "parent_id": parent_id,
                            "owner": self.expected_account_email, "ownedByMe": True,
                            "size": local_size, "quotaBytesUsed": "positive integer",
                            "api_visible": True,
                        },
                        "observed": {
                            "file_id": verified.get("id"),
                            "parent_id": (verified.get("parents") or [None])[0],
                            "owner": self._owner(verified), "ownedByMe": verified.get("ownedByMe"),
                            "size": int(verified.get("size")) if verified.get("size") is not None else None,
                            "quotaBytesUsed": (int(verified.get("quotaBytesUsed"))
                                               if verified.get("quotaBytesUsed") is not None else None),
                            "api_visible": True,
                        },
                    },
                )
            quota_bytes = int(verified.get("quotaBytesUsed") or 0)
            if quota_bytes <= 0:
                raise DriveStorageError(
                    "new Drive file quota metadata remains missing after bounded read-back; "
                    f"expected positive quotaBytesUsed, observed={_safe_metadata_summary(verified)}",
                    diagnostics={
                        "phase": "new_file_quota_readback",
                        "expected": {
                            "file_id": file_id, "parent_id": parent_id,
                            "owner": self.expected_account_email, "ownedByMe": True,
                            "size": local_size, "quotaBytesUsed": "positive integer",
                            "api_visible": True,
                        },
                        "observed": {
                            "file_id": verified.get("id"),
                            "parent_id": (verified.get("parents") or [None])[0],
                            "owner": self._owner(verified), "ownedByMe": verified.get("ownedByMe"),
                            "size": int(verified.get("size")) if verified.get("size") is not None else None,
                            "quotaBytesUsed": quota_bytes,
                            "api_visible": True,
                        },
                    },
                )
            same_name = None
            for attempt, delay in enumerate(readback_delays):
                if delay:
                    time.sleep(delay)
                same_name = self._one(
                    self._list_children(parent_id, name=logical.name), label=f"file {logical}",
                )
                if same_name is not None:
                    break
                if attempt + 1 == len(readback_delays):
                    break
            if same_name is None or same_name.get("id") != file_id:
                raise DriveStorageError(
                    "new Drive file exact-name visibility remained missing/mismatched after bounded retries; "
                    f"expected_file_id={file_id}, expected_parent_id={parent_id}, observed={_safe_metadata_summary(same_name)}",
                    diagnostics={
                        "phase": "exact_name_visibility_readback",
                        "expected": {
                            "file_id": file_id, "parent_id": parent_id,
                            "owner": self.expected_account_email, "ownedByMe": True,
                            "size": local_size, "quotaBytesUsed": "positive integer",
                            "api_visible": True,
                        },
                        "observed": {
                            "file_id": None, "parent_id": None, "owner": None,
                            "ownedByMe": None, "size": None, "quotaBytesUsed": None,
                            "api_visible": False,
                        },
                    },
                )
        return {"status": "PASS", "file_id": file_id, "parent_id": parent_id,
                "bytes": local_size, "sha256": local_sha, "md5_checksum": local_md5,
                "new_file": new_file, "write_bytes": local_size,
                "file_owner": (self._owner(verified) if isinstance(verified, dict) else existing_owner),
                "owned_by_canonical_account": (
                    verified.get("ownedByMe") is True if isinstance(verified, dict) else False
                ),
                "quota_attribution": (
                    "verified_canonical_account" if new_file or existing_owned_by_me is True
                    else "existing_foreign_owner_quota_unverified"
                )}

    def probe_write(
        self, logical_file: str | Path, payload: bytes, *, expected_parent_id: str,
    ) -> dict[str, Any]:
        """Exercise this adapter's actual create/read-back/delete path on one probe.

        The unique probe is first checked absent, created by ``publish_file``,
        downloaded through ``download_file``, and removed only by its exact ID
        after owner/parent/size/checksum verification. This is intentionally the
        same code path used for durable production objects.
        """
        logical = Path(os.path.abspath(str(Path(logical_file).expanduser())))
        parent_id = self.folder_id(logical.parent, create=False)
        if parent_id != str(expected_parent_id):
            raise DriveStorageError(
                "confirmed API probe parent mismatch before upload; "
                f"expected_parent_id={expected_parent_id}, observed_parent_id={parent_id}"
            )
        if self.exact_file(logical) is not None:
            raise DriveStorageError(f"unique API storage probe path already exists; preserved: {logical.name}")
        raw_payload = bytes(payload)
        if not raw_payload:
            raise ValueError("API storage probe payload must be nonempty")
        digest = hashlib.sha256(raw_payload).hexdigest()
        md5 = hashlib.md5(raw_payload, usedforsecurity=False).hexdigest()
        expected = {
            "file_id": "non-empty API-created file ID", "parent_id": parent_id,
            "owner": self.expected_account_email, "ownedByMe": True,
            "size": len(raw_payload), "quotaBytesUsed": "positive integer",
            "api_visible": True,
        }
        probe_dir = self.local_staging_root / ".probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        local = probe_dir / f"{logical.name}.source"
        restored = probe_dir / f"{logical.name}.readback"
        file_id: str | None = None
        created_by_probe = False
        delete_verified = False
        before = self.capacity_snapshot()
        primary_error: Exception | None = None
        failure_observed: dict[str, Any] | None = None
        result: dict[str, Any] | None = None

        def inspect() -> dict[str, Any] | None:
            try:
                row = self.exact_file(logical)
            except Exception:
                return None
            if row is None:
                return None
            parents = [str(value) for value in row.get("parents", [])]
            quota_raw = row.get("quotaBytesUsed")
            size_raw = row.get("size")
            try:
                quota = int(quota_raw) if quota_raw is not None else None
            except (TypeError, ValueError, OverflowError):
                quota = None
            try:
                size = int(size_raw) if size_raw is not None else None
            except (TypeError, ValueError, OverflowError):
                size = None
            return {
                "file_id": row.get("id"),
                "parent_id": parents[0] if len(parents) == 1 else parents,
                "owner": self._owner(row), "ownedByMe": row.get("ownedByMe"),
                "size": size, "quotaBytesUsed": quota, "api_visible": True,
                "name": row.get("name"), "mimeType": row.get("mimeType"),
                "md5Checksum": row.get("md5Checksum"),
                "capabilities": row.get("capabilities"),
            }

        def delete_probe_by_id_after_stale_listing() -> None:
            if not file_id:
                raise DriveStorageError("probe file ID is unavailable for exact-ID cleanup")
            metadata = self.service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,md5Checksum,capabilities(canDelete)",
                supportsAllDrives=True,
            ).execute()
            parents = [str(value) for value in metadata.get("parents", [])]
            ownership_valid = (
                self._owner(metadata) == self.expected_account_email
                and metadata.get("ownedByMe") is True
            )
            probe_delete_allowed = (
                bool(file_id) and metadata.get("id") == file_id
                and (metadata.get("capabilities") or {}).get("canDelete") is True
            )
            if (
                metadata.get("id") != file_id or metadata.get("name") != logical.name
                or parent_id not in parents or metadata.get("mimeType") == "application/vnd.google-apps.folder"
                or int(metadata.get("size") or -1) != len(raw_payload)
                or metadata.get("md5Checksum") != md5
                or not (ownership_valid or probe_delete_allowed)
            ):
                raise DriveStorageError(
                    "stale-listing probe cleanup refused; exact file ID owner/parent/content identity is unverified; "
                    f"observed={_safe_metadata_summary(metadata)}"
                )
            self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
            try:
                self.service.files().get(fileId=file_id, fields="id", supportsAllDrives=True).execute()
            except Exception as exc:
                if _http_status(exc) == 404:
                    return
                raise DriveStorageError(
                    f"exact-ID probe deletion read-back is inconclusive for {file_id}: {type(exc).__name__}"
                ) from exc
            raise DriveStorageError(f"exact-ID probe {file_id} remains visible after deletion")

        try:
            with local.open("xb") as stream:
                stream.write(raw_payload)
                stream.flush()
                os.fsync(stream.fileno())
            publication = self.publish_file(local, logical)
            if publication.get("new_file") is not True:
                raise DriveStorageError("probe path was concurrently occupied; existing file was preserved")
            file_id = str(publication["file_id"])
            expected["file_id"] = file_id
            created_by_probe = True
            observed = inspect()
            if observed is None:
                raise DriveStorageError("probe upload returned but exact API metadata is not visible")
            if (
                observed["file_id"] != file_id
                or observed["parent_id"] != parent_id
                or observed["owner"] != self.expected_account_email
                or observed["ownedByMe"] is not True
                or observed["size"] != len(raw_payload)
                or type(observed["quotaBytesUsed"]) is not int
                or observed["quotaBytesUsed"] <= 0
                or observed["md5Checksum"] != md5
            ):
                raise DriveStorageError("probe metadata differs after the adapter verified upload")
            self.download_file(logical, restored, expected_sha256=digest, expected_bytes=len(raw_payload))
            if restored.read_bytes() != raw_payload:
                raise DriveStorageError("probe bytes differ after adapter download/read-back")
            during = self.capacity_snapshot()
            self.delete_exact(logical, expected_file_id=file_id)
            if self.exact_file(logical) is not None:
                raise DriveStorageError("exact API probe remains visible after verified delete")
            delete_verified = True
            result = {
                "status": "PASS", "file_id": file_id, "parent_id": parent_id,
                "owner": observed["owner"], "ownedByMe": observed["ownedByMe"],
                "size": observed["size"], "quotaBytesUsed": observed["quotaBytesUsed"],
                "quota_usage_before_bytes": before["usage_bytes"],
                "quota_usage_during_bytes": during["usage_bytes"],
                "quota_limit_before_bytes": before["limit_bytes"],
                "quota_limit_during_bytes": during["limit_bytes"],
                "sha256": digest, "md5_checksum": md5,
                "delete_verified": True, "diagnostics": {
                    "classification": "VERIFIED",
                    "expected": expected, "observed": observed,
                },
            }
        except Exception as exc:
            primary_error = exc
            structured = getattr(exc, "diagnostics", None)
            if isinstance(structured, dict):
                candidate_observed = structured.get("observed")
                if isinstance(candidate_observed, dict):
                    failure_observed = candidate_observed
                structured_expected = structured.get("expected") or {}
                candidate_id = structured_expected.get("file_id")
                if isinstance(candidate_id, str) and candidate_id:
                    file_id = candidate_id
                    expected.update(structured_expected)
                    created_by_probe = True
        finally:
            # Cleanup can touch only the unique probe name created by this call,
            # and only when its exact identity/content/owner still verify.
            try:
                current = inspect()
                if current is None and created_by_probe and not delete_verified:
                    delete_probe_by_id_after_stale_listing()
                    delete_verified = True
                elif current is not None and created_by_probe and not delete_verified:
                    exact_probe_content = (
                        current.get("file_id") == file_id
                        and current.get("parent_id") == parent_id
                        and current.get("size") == len(raw_payload)
                        and current.get("md5Checksum") == md5
                    )
                    cleanup_authorized = (
                        current.get("owner") == self.expected_account_email
                        and current.get("ownedByMe") is True
                    ) or (
                        current.get("file_id") == file_id
                        and (current.get("capabilities") or {}).get("canDelete") is True
                    )
                    if exact_probe_content and cleanup_authorized:
                        if current.get("owner") == self.expected_account_email and current.get("ownedByMe") is True:
                            self.delete_exact(logical, expected_file_id=str(current["file_id"]))
                            delete_verified = self.exact_file(logical) is None
                        else:
                            delete_probe_by_id_after_stale_listing()
                            delete_verified = True
                        if not delete_verified and primary_error is None:
                            primary_error = DriveStorageError("probe cleanup delete was not verified by exact API lookup")
                    elif primary_error is None:
                        primary_error = DriveStorageError(
                            "probe cleanup refused because exact owner/parent/content identity is unverified; "
                            f"probe preserved; observed={current}"
                        )
            except Exception as cleanup_exc:
                if primary_error is None:
                    primary_error = DriveStorageError(
                        "probe cleanup failed for its exact verified path; no unrelated file was touched: "
                        f"{type(cleanup_exc).__name__}"
                    )
                else:
                    primary_error = DriveStorageError(
                        f"{primary_error}; exact-probe cleanup also failed: {type(cleanup_exc).__name__}"
                    )
            local.unlink(missing_ok=True)
            restored.unlink(missing_ok=True)

        if primary_error is not None:
            observed = failure_observed or inspect()
            if observed is None:
                observed = {
                    "file_id": file_id, "parent_id": None, "owner": None,
                    "ownedByMe": None, "size": None, "quotaBytesUsed": None,
                    "api_visible": False,
                }
            conditions = {}
            for key, target in expected.items():
                value = observed.get(key)
                if key == "file_id":
                    passed = isinstance(value, str) and bool(value) and (
                        target == "non-empty API-created file ID" or value == target
                    )
                elif key == "quotaBytesUsed":
                    passed = type(value) is int and value > 0
                elif key == "api_visible":
                    passed = value is True
                elif key == "size":
                    passed = type(value) is int and value == target
                elif key == "parent_id" and isinstance(value, list):
                    passed = target in value
                else:
                    passed = value == target
                conditions[key] = {
                    "status": "PASS" if passed else "MISMATCH" if value is not None else "MISSING_OR_STALE",
                    "expected": target, "observed": value,
                }
            if observed.get("owner") not in (None, self.expected_account_email) or observed.get("ownedByMe") is False:
                classification = "CONFIRMED_OWNERSHIP_MISMATCH"
            elif observed.get("parent_id") is not None and parent_id not in (
                observed.get("parent_id") if isinstance(observed.get("parent_id"), list)
                else [observed.get("parent_id")]
            ):
                classification = "CONFIRMED_PARENT_MISMATCH"
            elif file_id is None and observed.get("api_visible") is not True:
                classification = "WRITE_FAILED_OR_API_NOT_VISIBLE"
            else:
                classification = "MISSING_OR_STALE_METADATA"
            diagnostics = {
                "classification": classification, "expected": expected,
                "observed": observed, "conditions": conditions,
                "cause_type": type(primary_error).__name__,
            }
            raise DriveStorageError(
                "canonical API adapter write-capability probe failed; "
                f"diagnostics={json.dumps(diagnostics, sort_keys=True, default=str)}"
            ) from primary_error
        assert result is not None
        return result

    def _walk_download(self, folder_id: str, logical_folder: Path, local_folder: Path,
                       *, include: Callable[[Path, dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
        local_folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in self._list_children(folder_id):
            name = str(item.get("name") or "")
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise DriveStorageError(f"unsafe Drive child name in folder {folder_id!r}")
            logical = logical_folder / name
            local = local_folder / name
            if include is not None and not include(logical, item):
                continue
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                rows.extend(self._walk_download(str(item["id"]), logical, local, include=include))
            elif item.get("mimeType") == "application/vnd.google-apps.shortcut":
                raise DriveStorageError(f"refusing implicit shortcut traversal for storage artifact {logical}")
            else:
                rows.append(self._download_id(item, local))
        return rows

    def download_tree(self, logical_root: str | Path, local_root: str | Path, *, include=None) -> list[dict[str, Any]]:
        logical = Path(os.path.abspath(str(Path(logical_root).expanduser())))
        target = Path(local_root).expanduser().resolve()
        try:
            folder_id = self.folder_id(logical, create=False)
        except FileNotFoundError:
            return []
        required = self.tree_bytes(logical)
        from .storage import DEFAULT_STORAGE_RESERVE_BYTES
        target.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target.parent).free
        if free < required + DEFAULT_STORAGE_RESERVE_BYTES:
            raise DriveStorageError(
                f"local staging has {free} free bytes; API subtree download needs {required} bytes "
                f"plus {DEFAULT_STORAGE_RESERVE_BYTES} reserve; no subtree download was started"
            )
        return self._walk_download(folder_id, logical, target, include=include)

    def tree_bytes(self, logical_root: str | Path) -> int:
        """Count exact remote binary sizes by folder IDs without using DriveFS."""
        try:
            folder_id = self.folder_id(logical_root, create=False)
        except FileNotFoundError:
            return 0
        total = 0
        stack = [folder_id]
        while stack:
            current = stack.pop()
            for item in self._list_children(current):
                mime = item.get("mimeType")
                if mime == "application/vnd.google-apps.folder":
                    stack.append(str(item["id"]))
                elif mime == "application/vnd.google-apps.shortcut":
                    raise DriveStorageError("refusing to count an implicit shortcut in durable artifact storage")
                else:
                    size = item.get("size")
                    if size is None:
                        raise DriveStorageError(f"Drive omitted size for exact artifact ID {item.get('id')}")
                    total += int(size)
        return total

    def publish_tree(self, local_root: str | Path, logical_root: str | Path, *, include=None) -> list[dict[str, Any]]:
        local_base = Path(local_root).expanduser().resolve()
        logical_base = Path(os.path.abspath(str(Path(logical_root).expanduser())))
        if not local_base.is_dir():
            return []
        results = []
        for path in sorted(local_base.rglob("*")):
            if path.is_symlink():
                raise DriveStorageError(f"refusing to publish symlinked staging artifact: {path}")
            if path.is_dir():
                continue
            logical = logical_base / path.relative_to(local_base)
            if include is not None and not include(logical, path):
                continue
            results.append(self.publish_file(
                path, logical,
                allow_foreign_owner_update=self.is_mutable_control_path(logical),
            ))
        return results

    def delete_exact(self, logical_file: str | Path, *, expected_file_id: str) -> dict[str, Any]:
        metadata = self.exact_file(logical_file)
        if metadata is None or str(metadata.get("id")) != str(expected_file_id):
            raise DriveStorageError(f"exact file identity changed before delete for {logical_file}; file preserved")
        if (self._owner(metadata) != self.expected_account_email or metadata.get("ownedByMe") is not True
                or metadata.get("capabilities", {}).get("canDelete") is not True):
            raise DriveStorageError(f"exact file ID {expected_file_id} is not verified as deletable by storage account")
        self.service.files().delete(fileId=expected_file_id, supportsAllDrives=True).execute()
        try:
            self.service.files().get(fileId=expected_file_id, fields="id", supportsAllDrives=True).execute()
        except Exception as exc:
            if _http_status(exc) == 404:
                return {"status": "PASS_DELETED_EXACT_ID", "file_id": expected_file_id}
            raise DriveStorageError(f"delete read-back for exact ID {expected_file_id} is inconclusive") from exc
        raise DriveStorageError(f"Drive still returns exact file ID {expected_file_id} after delete; no further deletion attempted")

    def ensure_lease_record(self, logical_path: str | Path) -> dict[str, Any]:
        """Provision a singleton lease document through API; ambiguity blocks workers."""
        path = Path(os.path.abspath(str(Path(logical_path).expanduser())))
        try:
            existing = self.exact_file(path)
        except FileNotFoundError:
            self.folder_id(path.parent, create=True)
            existing = None
        if existing is not None:
            owner = self._owner(existing)
            owned_by_me = existing.get("ownedByMe")
            if (not owner or type(owned_by_me) is not bool
                    or existing.get("capabilities", {}).get("canEdit") is not True
                    or existing.get("mimeType") == "application/vnd.google-apps.folder"):
                raise DriveLeaseError(
                    "existing canonical lease file has missing owner metadata or is not editable by the verified "
                    f"storage account; no worker may start; observed={_safe_metadata_summary(existing)}"
                )
            return {
                "status": "EXISTING_EXACT_ID" if owned_by_me else "EXISTING_SHARED_EDITOR_EXACT_ID",
                "file_id": str(existing["id"]), "parent_id": existing["parents"][0],
                "file_owner": owner, "owned_by_canonical_account": owned_by_me,
                "quota_attribution": (
                    "canonical_account" if owned_by_me
                    else "existing_owner_quota_not_inferred"
                ),
            }
        local = self.local_staging_root / ".control" / f"lease-{hashlib.sha256(str(path).encode()).hexdigest()[:20]}.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            local.write_text(json.dumps({"schema_version": 1, "active": False, "expires_at": 0}, sort_keys=True) + "\n")
        created = self.publish_file(local, path)
        exact = self.exact_file(path)
        if exact is None or str(exact.get("id")) != str(created.get("file_id")):
            raise DriveLeaseError("API lease record provisioning is ambiguous; no worker may start")
        return {"status": "CREATED_AND_VERIFIED", "file_id": str(exact["id"]), "parent_id": exact["parents"][0]}


class DriveApiLease:
    """CAS-based lease over a single pre-provisioned API file.

    The lease cannot run unless the API supplies a strong response ETag and
    conditional content updates reject stale ETags (HTTP 412). Callers must run
    `verify_conditional_update_support` in the current Colab API session.
    """

    def __init__(self, storage: DriveApiStorage, logical_path: str | Path, *, ttl_seconds: float = 300.0,
                 owner_token: str | None = None):
        import socket
        import uuid

        if ttl_seconds <= 0:
            raise ValueError("lease TTL must be positive")
        self.storage = storage
        self.logical_path = Path(os.path.abspath(str(Path(logical_path).expanduser())))
        self.ttl_seconds = float(ttl_seconds)
        self.owner_token = owner_token or uuid.uuid4().hex
        self.hostname = socket.gethostname()
        self._held = False
        self._file_id: str | None = None
        self._etag: str | None = None

    def _read(self) -> tuple[dict[str, Any], str, dict[str, Any]]:
        metadata = self.storage.exact_file(self.logical_path)
        if metadata is None:
            raise DriveLeaseError("canonical API lease record is not provisioned; refusing a non-atomic create race")
        expected_parent_id = self.storage.folder_id(self.logical_path.parent, create=False)
        if (
            not self.storage._owner(metadata)
            or type(metadata.get("ownedByMe")) is not bool
            or metadata.get("capabilities", {}).get("canEdit") is not True
            or metadata.get("mimeType") == "application/vnd.google-apps.folder"
            or str(metadata.get("name")) != self.logical_path.name
            or expected_parent_id not in [str(value) for value in metadata.get("parents", [])]
        ):
            raise DriveLeaseError(
                "canonical API lease file ownership, edit permission, or exact parent is unverified; "
                f"observed={_safe_metadata_summary(metadata)}"
            )
        file_id = str(metadata["id"])
        request = self.storage.service.files().get(
            fileId=file_id,
            fields=("id,name,mimeType,parents,owners(emailAddress),ownedByMe,"
                    "capabilities(canEdit),size,quotaBytesUsed,version"),
            supportsAllDrives=True,
        )
        resource, response = _execute_with_response_headers(request)
        if (
            resource.get("id") != file_id
            or resource.get("name") != self.logical_path.name
            or not self.storage._owner(resource)
            or type(resource.get("ownedByMe")) is not bool
            or (resource.get("capabilities") or {}).get("canEdit") is not True
            or expected_parent_id not in [str(value) for value in resource.get("parents", [])]
        ):
            raise DriveLeaseError(
                "canonical API lease metadata changed identity or ownership during read; "
                f"expected_file_id={file_id}, expected_parent_id={expected_parent_id}, "
                f"observed={_safe_metadata_summary(resource)}"
            )
        etag = response.get("etag")
        if not _is_strong_etag(etag):
            diagnostics = {
                "failure_class": "etag_unavailable_or_weak",
                "expected": {
                    "file_id": file_id,
                    "parent_id": expected_parent_id,
                    "owner": self.storage.expected_account_email,
                    "strong_etag": True,
                },
                "observed": {
                    "api_visible": isinstance(resource, dict),
                    "file_id": resource.get("id") if isinstance(resource, dict) else None,
                    "parent_ids": resource.get("parents") if isinstance(resource, dict) else None,
                    "owner": self.storage._owner(resource) if isinstance(resource, dict) else None,
                    "ownedByMe": resource.get("ownedByMe") if isinstance(resource, dict) else None,
                    "size": resource.get("size") if isinstance(resource, dict) else None,
                    "quotaBytesUsed": resource.get("quotaBytesUsed") if isinstance(resource, dict) else None,
                    "etag_present": bool(etag),
                    "etag_is_strong": _is_strong_etag(etag),
                    "response_header_names": response.get("header_names", []),
                },
            }
            raise DriveLeaseError(
                "Drive API response did not contain a strong ETag required for safe conditional lease updates; "
                f"diagnostics={json.dumps(diagnostics, sort_keys=True)}",
                diagnostics=diagnostics,
            )
        payload = self.storage.service.files().get_media(fileId=file_id, supportsAllDrives=True).execute()
        if not isinstance(payload, (bytes, bytearray)):
            raise DriveLeaseError("Drive API returned non-byte lease content")
        try:
            document = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DriveLeaseError("canonical API lease record is malformed; preserved") from exc
        if not isinstance(document, dict):
            raise DriveLeaseError("canonical API lease record is not an object; preserved")
        return document, etag, resource

    def _cas_write(self, document: dict[str, Any], etag: str) -> str:
        payload = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        request = self.storage.service.files().update(
            fileId=str(self._file_id),
            media_body=_make_media_bytes_upload(payload, "application/json"),
            media_mime_type="application/json", supportsAllDrives=True,
            fields="id,name,parents,version",
        )
        headers = getattr(request, "headers", None)
        if headers is None or not hasattr(headers, "__setitem__"):
            raise DriveLeaseError("Drive API client cannot send If-Match for a conditional lease update")
        headers["If-Match"] = etag
        try:
            response = request.execute()
        except Exception as exc:
            if _http_status(exc) == 412:
                raise DriveLeaseError("lease compare-and-swap lost to another worker; no run started") from exc
            raise DriveLeaseError(f"conditional Drive lease update failed: {type(exc).__name__}") from exc
        if response.get("id") != self._file_id:
            raise DriveLeaseError("Drive lease update returned a different file ID")
        check, next_etag, _metadata = self._read()
        if check != document:
            raise DriveLeaseError("Drive lease compare-and-swap read-back differs; ownership not assumed")
        return next_etag

    def acquire(self) -> None:
        if self._held:
            raise DriveLeaseError("lease is already held by this object")
        document, etag, metadata = self._read()
        if metadata.get("mimeType") == "application/vnd.google-apps.folder":
            raise DriveLeaseError("lease ID resolves to a folder")
        now = time.time()
        if type(document.get("active")) is not bool:
            raise DriveLeaseError("canonical API lease active field is malformed; preserved")
        try:
            expires_at = float(document.get("expires_at"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise DriveLeaseError("canonical API lease expiry is malformed; preserved") from exc
        if not math.isfinite(expires_at):
            raise DriveLeaseError("canonical API lease expiry is non-finite; preserved")
        active = document["active"] is True and expires_at > now
        if active:
            raise DriveLeaseError("canonical Drive API lease is held by an unexpired owner")
        self._file_id = str(metadata["id"])
        claimed = {
            "schema_version": 1, "active": True, "owner_token": self.owner_token,
            "host": self.hostname, "pid": os.getpid(), "created_at": now,
            "heartbeat_at": now, "expires_at": now + self.ttl_seconds,
        }
        self._etag = self._cas_write(claimed, etag)
        self._held = True

    def heartbeat(self) -> None:
        if not self._held or not self._file_id or not self._etag:
            raise DriveLeaseError("cannot renew a lease not held by this owner")
        current, etag, _metadata = self._read()
        if current.get("owner_token") != self.owner_token or etag != self._etag:
            self._held = False
            raise DriveLeaseError("Drive lease owner/token or ETag changed before heartbeat")
        now = time.time()
        renewed = {**current, "heartbeat_at": now, "expires_at": now + self.ttl_seconds}
        self._etag = self._cas_write(renewed, etag)

    def release(self) -> None:
        if not self._held or not self._file_id or not self._etag:
            raise DriveLeaseError("cannot release a lease not held by this owner")
        current, etag, _metadata = self._read()
        if current.get("owner_token") != self.owner_token or etag != self._etag:
            self._held = False
            raise DriveLeaseError("Drive lease token/ETag changed before release")
        released = {**current, "active": False, "released_at": time.time(), "expires_at": 0}
        self._etag = self._cas_write(released, etag)
        self._held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()


def verify_conditional_update_support(service, *, parent_id: str, expected_account_email: str) -> dict[str, Any]:
    """Live capability probe: a stale If-Match must be rejected with HTTP 412."""
    import uuid

    expected = str(expected_account_email).strip().lower()
    name = f".revision-r1-cas-probe-{uuid.uuid4().hex}.json"
    created_id = None
    cleanup_verified = False
    try:
        ids = service.files().generateIds(count=1, space="drive").execute().get("ids", [])
        if len(ids) != 1:
            raise DriveLeaseError("Drive CAS probe did not receive one unique file ID")
        created_id = str(ids[0])
        payload = b'{"probe":0}\n'
        create = service.files().create(
            body={"id": created_id, "name": name, "mimeType": "application/json", "parents": [parent_id]},
            media_body=_make_media_bytes_upload(payload, "application/json"),
            media_mime_type="application/json", supportsAllDrives=True,
            fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed",
        ).execute()
        if create.get("id") != created_id:
            raise DriveLeaseError("Drive CAS probe create returned a different file ID")
        first_get = service.files().get(
            fileId=created_id,
            fields="id,name,mimeType,parents,owners(emailAddress),ownedByMe,size,quotaBytesUsed",
            supportsAllDrives=True,
        )
        first_metadata, response = _execute_with_response_headers(first_get)
        etag = response.get("etag")
        metadata = first_metadata if isinstance(first_metadata, dict) else {}
        owner = DriveApiStorage._owner(metadata)
        observed_parent_ids = [str(value) for value in metadata.get("parents", [])]
        checks = {
            "api_visible": isinstance(first_metadata, dict),
            "file_id_matches": metadata.get("id") == created_id,
            "parent_matches": parent_id in observed_parent_ids,
            "owner_matches": owner == expected,
            "owned_by_authenticated_user": metadata.get("ownedByMe") is True,
            "strong_etag_present": _is_strong_etag(etag),
        }
        if not all(checks.values()):
            failed_checks = [key for key, passed in checks.items() if not passed]
            if not checks["api_visible"] or not checks["file_id_matches"]:
                failure_class = "probe_metadata_missing_or_stale"
            elif not checks["parent_matches"] or not checks["owner_matches"] or not checks["owned_by_authenticated_user"]:
                failure_class = "probe_identity_or_parent_mismatch"
            else:
                failure_class = "probe_strong_etag_unavailable_or_weak"
            diagnostics = {
                "failure_class": failure_class,
                "expected": {
                    "file_id": created_id,
                    "parent_id": parent_id,
                    "owner": expected,
                    "ownedByMe": True,
                    "strong_etag": True,
                },
                "observed": {
                    "api_visible": checks["api_visible"],
                    "file_id": metadata.get("id"),
                    "parent_ids": observed_parent_ids,
                    "owner": owner,
                    "ownedByMe": metadata.get("ownedByMe"),
                    "size": metadata.get("size"),
                    "quotaBytesUsed": metadata.get("quotaBytesUsed"),
                    "etag_present": bool(etag),
                    "etag_is_strong": _is_strong_etag(etag),
                    "response_header_names": response.get("header_names", []),
                },
                "checks": checks,
            }
            raise DriveLeaseError(
                "Drive CAS probe identity/parent/strong-ETag verification failed; "
                f"failed_checks={failed_checks}; diagnostics={json.dumps(diagnostics, sort_keys=True)}",
                diagnostics=diagnostics,
            )
        first_payload = b'{"probe":1}\n'
        first_update = service.files().update(
            fileId=created_id,
            media_body=_make_media_bytes_upload(first_payload, "application/json"),
            media_mime_type="application/json", supportsAllDrives=True,
        )
        first_update.headers["If-Match"] = etag
        first_update.execute()
        stale_update = service.files().update(
            fileId=created_id,
            media_body=_make_media_bytes_upload(b'{"probe":2}\n', "application/json"),
            media_mime_type="application/json", supportsAllDrives=True,
        )
        stale_update.headers["If-Match"] = etag
        try:
            stale_update.execute()
        except Exception as exc:
            if _http_status(exc) != 412:
                raise DriveLeaseError(f"stale ETag probe returned {type(exc).__name__}, not HTTP 412") from exc
        else:
            raise DriveLeaseError("Drive ignored stale If-Match; API cannot safely coordinate active workers")
        return {"status": "PASS", "contract": "stale_if_match_rejected_http_412",
                "probe_file_id": created_id, "probe_parent_id": parent_id,
                "owner": owner, "etag_present": True,
                "etag_source": "http_response_callback"}
    finally:
        if created_id is not None:
            try:
                service.files().delete(fileId=created_id, supportsAllDrives=True).execute()
                try:
                    service.files().get(fileId=created_id, fields="id", supportsAllDrives=True).execute()
                except Exception as exc:
                    cleanup_verified = _http_status(exc) == 404
                if not cleanup_verified:
                    raise DriveLeaseError(f"CAS probe cleanup is unverified for exact temporary file ID {created_id}")
            except DriveLeaseError:
                raise
            except Exception as exc:
                raise DriveLeaseError(f"CAS probe cleanup failed for exact temporary file ID {created_id}") from exc
