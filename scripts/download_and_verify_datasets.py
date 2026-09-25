#!/usr/bin/env python3
"""Download the four prepared Zenodo dataset files and verify exact integrity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

RECORD_ID = "14537891"
FILES = {
    "EMBER_Class_train.npz": {"md5": "27e39d1cb697434107f92a8084128734", "shape": (303331, 2381)},
    "EMBER_Class_test.npz": {"md5": "9a844162d5ccca20987ce520d6355f33", "shape": (33704, 2381)},
    "AZ_Class_Train.npz": {"md5": "644649dfb93f0f0086052a923b657833", "shape": (257023, 2439)},
    "AZ_Class_Test.npz": {"md5": "1e4014ff6eb613845a6bcf38a2461001", "shape": (28559, 2439)},
}


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def arrays_from_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        x_key = next((key for key in ("X_train", "X_test", "x_train", "x_test", "X", "x") if key in archive), None)
        y_key = next((key for key in ("y_train", "y_test", "Y_train", "Y_test", "y", "Y") if key in archive), None)
        if not x_key or not y_key:
            raise ValueError(f"{path}: cannot find feature and label arrays; keys={archive.files}")
        return archive[x_key], archive[y_key]


def verify(path: Path, specification: dict[str, object], include_sha256: bool) -> None:
    observed_md5 = digest(path, "md5")
    expected_md5 = str(specification["md5"])
    if observed_md5 != expected_md5:
        raise ValueError(f"{path}: MD5 mismatch: expected {expected_md5}, observed {observed_md5}")
    features, labels = arrays_from_npz(path)
    if tuple(features.shape) != specification["shape"]:
        raise ValueError(f"{path}: shape mismatch: expected {specification['shape']}, observed {tuple(features.shape)}")
    values = np.unique(labels)
    if values.size != 100 or values.min() != 0 or values.max() != 99:
        raise ValueError(f"{path}: expected exactly labels 0-99; observed {values.min()}-{values.max()} with {values.size} classes")
    sha_suffix = f", sha256={digest(path, 'sha256')}" if include_sha256 else ""
    print(f"verified {path.name}: md5={observed_md5}, shape={tuple(features.shape)}, labels=0-99{sha_suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="Also print computed SHA-256 values.")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, specification in FILES.items():
        target = args.data_dir / name
        if target.exists():
            try:
                verify(target, specification, args.sha256)
                continue
            except ValueError as exc:
                raise SystemExit(f"Refusing to overwrite existing mismatched file. {exc}")
        url = f"https://zenodo.org/records/{RECORD_ID}/files/{name}?download=1"
        temporary = target.with_suffix(target.suffix + ".part")
        if temporary.exists():
            temporary.unlink()
        print(f"downloading {url}")
        urlretrieve(url, temporary)
        try:
            verify(temporary, specification, args.sha256)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
