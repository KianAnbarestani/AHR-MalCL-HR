"""Resolve prepared malware datasets from Drive or verified Zenodo downloads.

Downloads are staged only under the caller's local cache root. Existing Drive
files are read-only and can be linked into a mixed local fallback directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from revision_r1.config import DATASETS


ZENODO_RECORD_URL = "https://zenodo.org/records/14537891"
ZENODO_DOI = "10.5281/zenodo.14537891"
ZENODO_DOWNLOADS: dict[str, dict[str, Any]] = {
    "EMBER_Class_train.npz": {
        "url": "https://zenodo.org/api/records/14537891/files/EMBER_Class_train.npz/content",
        "size_bytes": 2_890_138_390,
    },
    "EMBER_Class_test.npz": {
        "url": "https://zenodo.org/api/records/14537891/files/EMBER_Class_test.npz/content",
        "size_bytes": 321_132_222,
    },
    "AZ_Class_Train.npz": {
        "url": "https://zenodo.org/api/records/14537891/files/AZ_Class_Train.npz/content",
        "size_bytes": 2_523_966_748,
    },
    "AZ_Class_Test.npz": {
        "url": "https://zenodo.org/api/records/14537891/files/AZ_Class_Test.npz/content",
        "size_bytes": 280_450_144,
    },
}
DEFAULT_LOCAL_RESERVE_BYTES = 512 << 20
DEFAULT_DOWNLOAD_CHUNK_BYTES = 8 << 20


def _sha256(path: Path, *, chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path, *, chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_identity(preflight: dict[str, Any], dataset: str, split: str, frozen: dict[str, Any]) -> tuple[str, str]:
    try:
        record = preflight[dataset][split]
        expected_md5 = str(record["md5"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"preflight is missing {dataset}/{split} dataset hashes") from exc
    if expected_md5 != frozen.get("md5"):
        raise ValueError(f"preflight {dataset}/{split} MD5 differs from the frozen dataset identity")
    if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
        raise ValueError(f"preflight {dataset}/{split} SHA-256 is invalid")
    return expected_md5, expected_sha256


def _verify_local_file(path: Path, *, expected_size: int, expected_md5: str, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"local dataset cache target is not a regular file: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"cached dataset size mismatch at {path}: {path.stat().st_size} != {expected_size}")
    if _md5(path) != expected_md5:
        raise ValueError(f"cached dataset MD5 mismatch at {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"cached dataset SHA-256 mismatch at {path}")


def _download_file(
    destination: Path,
    *,
    url: str,
    expected_size: int,
    expected_md5: str,
    expected_sha256: str,
    urlopen: Callable[..., Any],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    chunk_bytes: int,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "zenodo.org":
        raise ValueError(f"dataset download URL is not an HTTPS Zenodo URL: {url}")
    temp_path = destination.with_name(f".{destination.name}.revision_r1_download.part")
    if temp_path.is_symlink() or (temp_path.exists() and not temp_path.is_file()):
        raise ValueError(f"unsafe dataset download staging path: {temp_path}")
    request = urllib.request.Request(url, headers={"User-Agent": "revision_R1-dataset-bootstrap/1"})
    digest_md5 = hashlib.md5(usedforsecurity=False)
    digest_sha256 = hashlib.sha256()
    byte_count = 0
    last_progress = 0
    try:
        with urlopen(request, timeout=60) as response:
            status = getattr(response, "status", None)
            if status is not None and int(status) != 200:
                raise OSError(f"Zenodo returned HTTP {status} for {destination.name}")
            header_size = response.headers.get("Content-Length")
            if header_size is not None and int(header_size) != expected_size:
                raise ValueError(
                    f"Zenodo Content-Length for {destination.name} is {header_size}, expected {expected_size}"
                )
            with temp_path.open("wb") as stream:
                while True:
                    block = response.read(chunk_bytes)
                    if not block:
                        break
                    byte_count += len(block)
                    if byte_count > expected_size:
                        raise ValueError(f"Zenodo response exceeded the pinned size for {destination.name}")
                    stream.write(block)
                    digest_md5.update(block)
                    digest_sha256.update(block)
                    if progress_callback is not None and byte_count - last_progress >= 64 << 20:
                        progress_callback({
                            "stage": "dataset_download_progress", "filename": destination.name,
                            "downloaded_bytes": byte_count, "expected_bytes": expected_size,
                            "source": ZENODO_RECORD_URL,
                        })
                        last_progress = byte_count
                stream.flush()
                os.fsync(stream.fileno())
        if byte_count != expected_size:
            raise ValueError(f"Zenodo download size mismatch for {destination.name}: {byte_count} != {expected_size}")
        if digest_md5.hexdigest() != expected_md5:
            raise ValueError(f"Zenodo download MD5 mismatch for {destination.name}")
        if digest_sha256.hexdigest() != expected_sha256:
            raise ValueError(f"Zenodo download SHA-256 mismatch for {destination.name}")
        # Hard-link publication is atomic and refuses to replace an existing
        # target if another cell/process races this setup cell.
        os.link(temp_path, destination)
        temp_path.unlink()
    finally:
        if temp_path.exists() and not temp_path.is_symlink():
            temp_path.unlink()


def prepare_dataset_roots(
    drive_roots: dict[str, str | Path],
    local_root: str | Path,
    preflight: dict[str, Any],
    *,
    dataset_specs: dict[str, dict[str, Any]] | None = None,
    download_catalog: dict[str, dict[str, Any]] | None = None,
    urlopen: Callable[..., Any] | None = None,
    reserve_bytes: int = DEFAULT_LOCAL_RESERVE_BYTES,
    chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Choose Drive roots when complete; otherwise fill a local, verified fallback.

    No Drive file is written, replaced, or deleted. For a partially populated
    dataset, Drive-present files are symlinked into the local fallback while
    missing files are downloaded from the pinned Zenodo record.
    """
    specs = DATASETS if dataset_specs is None else dataset_specs
    catalog = ZENODO_DOWNLOADS if download_catalog is None else download_catalog
    opener = urllib.request.urlopen if urlopen is None else urlopen
    if set(drive_roots) != set(specs):
        raise ValueError("Drive roots must exactly match the configured dataset names")
    if int(reserve_bytes) < 0 or int(chunk_bytes) <= 0:
        raise ValueError("download reserve must be nonnegative and chunk size must be positive")

    local_base = Path(local_root).expanduser().resolve()
    local_fallback_needed = False
    total_required_bytes = 0
    for dataset, dataset_spec in specs.items():
        drive_root = Path(drive_roots[dataset]).expanduser()
        drive_files = {
            split: drive_root / dataset_spec[split]["filename"]
            for split in ("train", "test")
        }
        if all(path.is_file() for path in drive_files.values()):
            continue
        local_fallback_needed = True
        for split in ("train", "test"):
            filename = str(dataset_spec[split]["filename"])
            local_file = local_base / dataset / filename
            if local_file.exists() or local_file.is_symlink() or drive_files[split].is_file():
                continue
            metadata = catalog.get(filename)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("size_bytes"), int):
                raise ValueError(f"no exact size/URL metadata for missing prepared dataset file {filename}")
            total_required_bytes += int(metadata["size_bytes"])

    if local_fallback_needed:
        local_base.mkdir(parents=True, exist_ok=True)
        available_bytes = int(shutil.disk_usage(local_base).free)
        if available_bytes < total_required_bytes + int(reserve_bytes):
            raise OSError(
                "insufficient local Colab storage for missing datasets: "
                f"need {total_required_bytes} download bytes + {int(reserve_bytes)} reserve, "
                f"have {available_bytes} free at {local_base}"
            )

    roots: dict[str, Path] = {}
    report: dict[str, dict[str, Any]] = {}

    for dataset, dataset_spec in specs.items():
        drive_root = Path(drive_roots[dataset]).expanduser()
        drive_files = {
            split: drive_root / dataset_spec[split]["filename"]
            for split in ("train", "test")
        }
        drive_present = {split: path.is_file() for split, path in drive_files.items()}
        if all(drive_present.values()):
            roots[dataset] = drive_root
            report[dataset] = {
                "source": "DRIVE_ONLY", "data_root": str(drive_root),
                "drive_files_present": drive_present, "downloaded_files": [],
                "local_fallback_bytes": 0, "zenodo_record": ZENODO_RECORD_URL,
                "zenodo_doi": ZENODO_DOI,
            }
            continue

        local_base.mkdir(parents=True, exist_ok=True)
        local_dataset_root = local_base / dataset
        local_dataset_root.mkdir(parents=True, exist_ok=True)
        local_files = {
            split: local_dataset_root / dataset_spec[split]["filename"]
            for split in ("train", "test")
        }
        pending: list[tuple[str, Path, dict[str, Any], str, str]] = []
        downloaded: list[str] = []
        sources: dict[str, str] = {}

        for split in ("train", "test"):
            frozen = dataset_spec[split]
            filename = str(frozen["filename"])
            expected_md5, expected_sha256 = _expected_identity(preflight, dataset, split, frozen)
            drive_file = drive_files[split]
            local_file = local_files[split]
            if local_file.exists() or local_file.is_symlink():
                if local_file.is_symlink():
                    if not drive_file.is_file() or local_file.resolve() != drive_file.resolve():
                        raise ValueError(f"unexpected symlink in local dataset cache: {local_file}")
                    sources[split] = "DRIVE_SYMLINK"
                    continue
                metadata = catalog.get(filename)
                if metadata is None:
                    raise ValueError(f"no pinned Zenodo metadata for local dataset file {filename}")
                _verify_local_file(
                    local_file, expected_size=int(metadata["size_bytes"]),
                    expected_md5=expected_md5, expected_sha256=expected_sha256,
                )
                sources[split] = "VERIFIED_LOCAL_CACHE"
                continue
            if drive_file.is_file():
                local_file.symlink_to(drive_file.resolve())
                sources[split] = "DRIVE_SYMLINK"
                continue
            metadata = catalog.get(filename)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("size_bytes"), int):
                raise ValueError(f"no exact size/URL metadata for missing prepared dataset file {filename}")
            pending.append((split, local_file, metadata, expected_md5, expected_sha256))
            sources[split] = "ZENODO_LOCAL_DOWNLOAD"

        for split, destination, metadata, expected_md5, expected_sha256 in pending:
            url = str(metadata.get("url", ""))
            _download_file(
                destination, url=url, expected_size=int(metadata["size_bytes"]),
                expected_md5=expected_md5, expected_sha256=expected_sha256,
                urlopen=opener, progress_callback=progress_callback, chunk_bytes=int(chunk_bytes),
            )
            downloaded.append(destination.name)

        roots[dataset] = local_dataset_root
        if all(value == "ZENODO_LOCAL_DOWNLOAD" or value == "VERIFIED_LOCAL_CACHE" for value in sources.values()):
            source = "LOCAL_ZENODO_FALLBACK"
        else:
            source = "MIXED_DRIVE_AND_ZENODO_LOCAL"
        report[dataset] = {
            "source": source, "data_root": str(local_dataset_root),
            "drive_files_present": drive_present, "file_sources": sources,
            "downloaded_files": downloaded,
            "local_fallback_bytes": sum(
                path.stat().st_size for path in local_files.values()
                if path.is_file() and not path.is_symlink()
            ),
            "download_bytes_this_session": sum(
                int(catalog[name]["size_bytes"]) for name in downloaded
            ),
            "local_free_bytes_before_download": available_bytes,
            "local_reserve_bytes": int(reserve_bytes), "zenodo_record": ZENODO_RECORD_URL,
            "zenodo_doi": ZENODO_DOI,
        }

    return roots, report

