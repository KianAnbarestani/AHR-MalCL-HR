"""Known-reference NPZ validation and single-load dataset access."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from typing import Any, Sequence

import numpy as np

from .config import DATASETS
from .storage import DEFAULT_STORAGE_RESERVE_BYTES, require_path_capacity


CACHE_FORMAT_VERSION = 2
CACHE_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_metadata: dict[str, Any]
    test_metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        return int(self.x_train.shape[1])


def require_memory_mapped_production_bundle(bundle: DatasetBundle) -> None:
    """Refuse production execution unless all four dataset arrays are verified mmap inputs."""
    arrays = {
        "x_train": (bundle.x_train, bundle.train_metadata),
        "y_train": (bundle.y_train, bundle.train_metadata),
        "x_test": (bundle.x_test, bundle.test_metadata),
        "y_test": (bundle.y_test, bundle.test_metadata),
    }
    failures = [
        name for name, (array, metadata) in arrays.items()
        if not isinstance(array, np.memmap) or metadata.get("cache_enabled") is not True
    ]
    if failures:
        raise ValueError(
            "production requires all features/labels to come from the verified streamed mmap cache; "
            f"uncached arrays: {', '.join(failures)}"
        )


def file_digest(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_keys(files: Sequence[str], split: str) -> tuple[str, str]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    x_names = (f"X_{split}", f"x_{split}", "X", "x")
    y_names = (f"y_{split}", f"Y_{split}", "y", "Y")
    x_matches = [name for name in x_names if name in files]
    y_matches = [name for name in y_names if name in files]
    if len(x_matches) != 1 or len(y_matches) != 1:
        raise ValueError(
            f"NPZ must contain one unambiguous feature/label key pair for {split}; "
            f"feature_candidates={x_matches}, label_candidates={y_matches}, keys={list(files)}"
        )
    return x_matches[0], y_matches[0]


def _cache_paths(cache_dir: Path, archive_md5: str, x_key: str, y_key: str) -> tuple[Path, Path, Path]:
    for key in (x_key, y_key):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError(f"unsafe NPZ member key for cache path: {key!r}")
    base = cache_dir / f"{archive_md5}_{x_key}_{y_key}_stream_f32_v{CACHE_FORMAT_VERSION}"
    return base / "x.npy", base / "y.npy", base / "manifest.json"


def _load_verified_cache(
    paths: tuple[Path, Path, Path], archive_md5: str, x_key: str, y_key: str,
    expected_shape: tuple[int, int] | None, expected_archive_sha256: str,
):
    x_path, y_path, manifest_path = paths
    present = [path.is_file() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"incomplete streamed dataset cache exists and was preserved: {manifest_path.parent}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["format_version"] != CACHE_FORMAT_VERSION
            or manifest["archive_md5"] != archive_md5
            or manifest["archive_sha256"] != expected_archive_sha256
            or manifest["x_key"] != x_key or manifest["y_key"] != y_key
            or manifest["x_dtype"] != "float32" or manifest["y_dtype"] != "int64"
        ):
            raise ValueError("local dataset cache identity mismatch")
        if file_digest(x_path, "sha256") != manifest["x_sha256"] or file_digest(y_path, "sha256") != manifest["y_sha256"]:
            raise ValueError("local dataset cache checksum mismatch")
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)
        if (
            list(x.shape) != manifest["x_shape"] or list(y.shape) != manifest["y_shape"]
            or x.dtype != np.dtype("float32") or y.dtype != np.dtype("int64")
            or (expected_shape is not None and tuple(x.shape) != tuple(expected_shape))
        ):
            raise ValueError("local dataset cache shape mismatch")
        return x, y, manifest
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local dataset cache manifest: {manifest_path}") from error


def _npy_header(stream) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(stream)
    shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _npz_array_member(archive: zipfile.ZipFile, key: str) -> tuple[str, tuple[int, ...], bool, np.dtype, int]:
    member = f"{key}.npy"
    try:
        info = archive.getinfo(member)
    except KeyError as exc:
        raise ValueError(f"NPZ member is missing: {member}") from exc
    with archive.open(info, "r") as stream:
        shape, fortran_order, dtype = _npy_header(stream)
        data_offset = stream.tell()
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError(f"object/structured NPZ arrays are not supported: {member}")
    expected_size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if info.file_size != data_offset + expected_size:
        raise ValueError(f"NPZ member byte length does not match its NPY header: {member}")
    return member, shape, fortran_order, dtype, data_offset


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ValueError("truncated NPY payload inside compressed NPZ")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _stream_member_to_npy(
    archive: zipfile.ZipFile, *, member: str, destination: Path,
    expected_shape: tuple[int, ...] | None, target_dtype: np.dtype, kind: str,
    progress_callback=None, chunk_bytes: int = CACHE_CHUNK_BYTES,
) -> tuple[tuple[int, ...], str]:
    with archive.open(member, "r") as stream:
        shape, fortran_order, source_dtype = _npy_header(stream)
        if expected_shape is not None and tuple(shape) != tuple(expected_shape):
            raise ValueError(f"{member} shape {shape} != expected {tuple(expected_shape)}")
        if kind == "features" and (
            not np.issubdtype(source_dtype, np.number)
            or np.issubdtype(source_dtype, np.complexfloating)
        ):
            raise ValueError(f"{member} features must be real numeric values")
        if kind == "labels" and not (
            np.issubdtype(source_dtype, np.integer) or np.issubdtype(source_dtype, np.floating)
        ):
            raise ValueError(f"{member} labels must be integer class IDs")
        total_elements = int(np.prod(shape, dtype=np.int64))
        from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory
        largest_elements = min(total_elements, max(1, int(chunk_bytes) // max(1, source_dtype.itemsize)))
        required_memory = largest_elements * (int(source_dtype.itemsize) + int(target_dtype.itemsize) + 2) + (8 << 20)
        require_host_memory(
            required_memory, reserve_bytes=HOST_RAM_RESERVE_BYTES,
            phase=f"streaming {member} into the verified local cache",
        )
        target = np.lib.format.open_memmap(
            destination, mode="w+", dtype=target_dtype, shape=shape,
            fortran_order=fortran_order and len(shape) > 1,
        )
        order = "F" if fortran_order and len(shape) > 1 else "C"
        flat_target = target.ravel(order=order)
        if not np.shares_memory(flat_target, target):
            raise RuntimeError(f"cannot stream {member} into a bounded contiguous cache")
        source_elements_per_chunk = max(1, int(chunk_bytes) // max(1, source_dtype.itemsize))
        for start in range(0, total_elements, source_elements_per_chunk):
            stop = min(total_elements, start + source_elements_per_chunk)
            payload = _read_exact(stream, (stop - start) * source_dtype.itemsize)
            source = np.frombuffer(payload, dtype=source_dtype, count=stop - start)
            if kind == "features":
                converted = np.asarray(source, dtype=target_dtype)
                if not np.isfinite(converted).all():
                    raise ValueError(f"{member} contains values not finite after the training float32 cast")
            else:
                if np.issubdtype(source_dtype, np.floating):
                    if not np.isfinite(source).all() or not np.equal(source, np.floor(source)).all():
                        raise ValueError(f"{member} labels must be finite integer class IDs")
                converted = np.asarray(source, dtype=target_dtype)
            flat_target[start:stop] = converted
            if progress_callback is not None:
                progress_callback({
                    "stage": "member_stream_progress", "member": member,
                    "bytes_written": int(stop * target_dtype.itemsize),
                    "total_bytes": int(total_elements * target_dtype.itemsize),
                })
        if stream.read(1):
            raise ValueError(f"{member} contains unexpected bytes after the declared NPY data")
        target.flush()
        del flat_target
        del target
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return shape, file_digest(destination, "sha256")


def inspect_npz_array_layout(path: str | Path, *, split: str) -> dict[str, dict[str, Any]]:
    """Inspect NPY headers inside an NPZ without decoding array payloads."""
    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        members = [info.filename[:-4] for info in archive.infolist() if info.filename.endswith(".npy")]
        if len(members) != len(set(members)):
            raise ValueError(f"NPZ contains duplicate array member names: {source}")
        x_key, y_key = _array_keys(members, split)
        result = {}
        for role, key in (("features", x_key), ("labels", y_key)):
            member, shape, fortran_order, dtype, _ = _npz_array_member(archive, key)
            info = archive.getinfo(member)
            result[role] = {
                "key": key, "member": member, "shape": list(shape), "dtype": str(dtype),
                "fortran_order": fortran_order, "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
            }
        return result


def _populate_streamed_cache(
    source_path: Path, paths: tuple[Path, Path, Path], *, archive_md5: str,
    archive_sha256: str, x_key: str, y_key: str,
    expected_shape: tuple[int, int] | None, progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_path, y_path, manifest_path = paths
    final_dir = manifest_path.parent
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise ValueError(f"streamed cache directory exists without a valid manifest; preserved: {final_dir}")
    with zipfile.ZipFile(source_path, "r") as archive:
        x_member = _npz_array_member(archive, x_key)
        y_member = _npz_array_member(archive, y_key)
        x_shape, y_shape = x_member[1], y_member[1]
        if len(x_shape) != 2 or len(y_shape) != 1 or x_shape[0] != y_shape[0]:
            raise ValueError(f"{source_path.name} must contain 2-D features and matching 1-D labels")
        if expected_shape is not None and tuple(x_shape) != tuple(expected_shape):
            raise ValueError(f"{source_path.name} feature shape {x_shape} != expected {tuple(expected_shape)}")
        required = int(np.prod(x_shape, dtype=np.int64)) * 4 + int(y_shape[0]) * 8 + (2 << 20)
        require_path_capacity(final_dir.parent, required_bytes=required, reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES)
        staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.building-", dir=final_dir.parent))
        try:
            stage_x, stage_y = staging / "x.npy", staging / "y.npy"
            if progress_callback is not None:
                progress_callback({"stage": "member_stream_start", "member": x_member[0], "source_dtype": str(x_member[3]), "target_dtype": "float32"})
            written_x_shape, x_hash = _stream_member_to_npy(
                archive, member=x_member[0], destination=stage_x,
                expected_shape=expected_shape, target_dtype=np.dtype("float32"),
                kind="features", progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback({"stage": "member_stream_start", "member": y_member[0], "source_dtype": str(y_member[3]), "target_dtype": "int64"})
            written_y_shape, y_hash = _stream_member_to_npy(
                archive, member=y_member[0], destination=stage_y,
                expected_shape=(x_shape[0],), target_dtype=np.dtype("int64"),
                kind="labels", progress_callback=progress_callback,
            )
            manifest = {
                "format_version": CACHE_FORMAT_VERSION, "cache_policy": "stream_npy_members_float32_features_int64_labels",
                "archive_md5": archive_md5, "archive_sha256": archive_sha256,
                "x_key": x_key, "y_key": y_key,
                "x_shape": list(written_x_shape), "y_shape": list(written_y_shape),
                "x_source_dtype": str(x_member[3]), "y_source_dtype": str(y_member[3]),
                "x_dtype": "float32", "y_dtype": "int64", "x_sha256": x_hash, "y_sha256": y_hash,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
            staged_paths = (stage_x, stage_y, staging / "manifest.json")
            verified = _load_verified_cache(
                staged_paths, archive_md5, x_key, y_key, expected_shape, archive_sha256,
            )
            if verified is None:
                raise RuntimeError("streamed cache verification did not produce a complete cache")
            try:
                os.replace(staging, final_dir)
            except OSError:
                if not final_dir.exists():
                    raise
                existing = _load_verified_cache(
                    paths, archive_md5, x_key, y_key, expected_shape, archive_sha256,
                )
                if existing is None:
                    raise ValueError(f"concurrent streamed cache publication is invalid: {final_dir}")
                verified = existing
            if progress_callback is not None:
                progress_callback({"stage": "streamed_cache_published", "cache_dir": str(final_dir)})
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    loaded = _load_verified_cache(
        paths, archive_md5, x_key, y_key, expected_shape, archive_sha256,
    )
    if loaded is None:
        raise RuntimeError("streamed cache disappeared after publication")
    return loaded


def load_npz_arrays(
    path: str | Path,
    *,
    split: str,
    expected_md5: str,
    expected_shape: tuple[int, int] | None,
    expected_labels: Sequence[int] | None,
    cache_dir: str | Path | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file is missing: {path}")
    if not expected_md5 or len(expected_md5) != 32:
        raise ValueError("a known 32-character dataset MD5 reference is required")
    actual_md5 = file_digest(path, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(f"{path.name} MD5 mismatch: {actual_md5} != {expected_md5}")
    archive_sha256 = file_digest(path, "sha256")
    if progress_callback is not None:
        progress_callback({"stage": "archive_verified", "filename": path.name, "md5": actual_md5, "sha256": archive_sha256})
    if cache_dir is not None:
        with zipfile.ZipFile(path, "r") as archive:
            members = [info.filename[:-4] for info in archive.infolist() if info.filename.endswith(".npy")]
            if len(members) != len(set(members)):
                raise ValueError(f"NPZ contains duplicate array member names: {path}")
            x_key, y_key = _array_keys(members, split)
            cache_paths = _cache_paths(Path(cache_dir), actual_md5, x_key, y_key)
            cached = _load_verified_cache(
                cache_paths, actual_md5, x_key, y_key, expected_shape, archive_sha256,
            )
            if cached is None:
                x, y, manifest = _populate_streamed_cache(
                    path, cache_paths, archive_md5=actual_md5, archive_sha256=archive_sha256,
                    x_key=x_key, y_key=y_key, expected_shape=expected_shape,
                    progress_callback=progress_callback,
                )
            else:
                x, y, manifest = cached
        source_x_dtype = manifest["x_source_dtype"]
        source_y_dtype = manifest["y_source_dtype"]
    else:
        # Small validation fixtures and explicitly uncached analysis callers may
        # use the conventional in-memory path. Production requires verified mmap.
        with np.load(path, allow_pickle=False) as archive:
            x_key, y_key = _array_keys(archive.files, split)
            x, y = archive[x_key], archive[y_key]
        source_x_dtype, source_y_dtype = str(x.dtype), str(y.dtype)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"{path.name} must contain 2-D features and a matching 1-D label array")
    if expected_shape is not None and tuple(x.shape) != tuple(expected_shape):
        raise ValueError(f"{path.name} feature shape {tuple(x.shape)} != expected {tuple(expected_shape)}")
    if not np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError(f"{path.name} features must be real numeric values")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.isfinite(y).all() or not np.array_equal(y, y.astype(np.int64)):
            raise ValueError(f"{path.name} labels must be integer class IDs")
        y = y.astype(np.int64)
    labels, counts = np.unique(y, return_counts=True)
    if expected_labels is not None and not np.array_equal(labels.astype(np.int64), np.asarray(expected_labels, dtype=np.int64)):
        raise ValueError(f"{path.name} labels do not match expected class IDs")
    finite_chunk_rows = max(1, CACHE_CHUNK_BYTES // max(4, int(x.shape[1]) * int(x.dtype.itemsize)))
    for start in range(0, len(x), finite_chunk_rows):
        if not np.isfinite(x[start : start + finite_chunk_rows]).all():
            raise ValueError(f"{path.name} contains non-finite feature values")
    metadata = {
        "path": str(path), "filename": path.name, "split": split,
        "md5": actual_md5, "sha256": file_digest(path, "sha256"),
        "x_key": x_key, "y_key": y_key, "shape": list(x.shape),
        "dtype": str(x.dtype), "source_dtype": str(source_x_dtype),
        "label_dtype": str(y.dtype), "source_label_dtype": str(source_y_dtype),
        "class_count": int(labels.size),
        "class_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
        "cache_enabled": cache_dir is not None,
    }
    return x, y, metadata


def preflight_dataset(
    data_root: str | Path, dataset: str, cache_dir: str | Path | None = None,
    progress_callback=None,
) -> DatasetBundle:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    root = Path(data_root)
    spec = DATASETS[dataset]
    train_x, train_y, train_meta = load_npz_arrays(
        root / spec["train"]["filename"], split="train", expected_md5=spec["train"]["md5"],
        expected_shape=spec["train"]["shape"], expected_labels=range(100), cache_dir=cache_dir,
        progress_callback=progress_callback,
    )
    test_x, test_y, test_meta = load_npz_arrays(
        root / spec["test"]["filename"], split="test", expected_md5=spec["test"]["md5"],
        expected_shape=spec["test"]["shape"], expected_labels=range(100), cache_dir=cache_dir,
        progress_callback=progress_callback,
    )
    if train_x.shape[1] != test_x.shape[1] or train_x.shape[1] != spec["feature_dim"]:
        raise ValueError(f"{dataset} train/test feature coordinate dimensions differ")
    return DatasetBundle(dataset, train_x, train_y, test_x, test_y, train_meta, test_meta)

