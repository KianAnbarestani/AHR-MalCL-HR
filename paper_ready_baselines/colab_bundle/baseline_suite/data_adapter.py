#!/usr/bin/env python3
"""Colab-compatible data adapter for AHR-MalCL baseline runs."""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "v2 - Colab-compatible AHR-MalCL data adapter"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "paper_ready_baselines"
RECORD_ID = 14537891

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "ember": {
        "canonical_name": "ember",
        "display_name": "EMBER",
        "aliases": ["ember", "EMBER"],
        "train_file": "EMBER_Class_train.npz",
        "test_file": "EMBER_Class_test.npz",
        "train_md5": "27e39d1cb697434107f92a8084128734",
        "test_md5": "9a844162d5ccca20987ce520d6355f33",
        "expected_feature_dim": 2381,
        "num_classes": 100,
        "initial_classes": 50,
        "increment_classes_per_task": 5,
        "task_ordering": "random",
        "scaler_mode": "incremental",
        "notebook_source": "ahr_malcl_v15_1_final_paper_critic5_one_cell.ipynb",
    },
    "az_class": {
        "canonical_name": "az_class",
        "display_name": "AZ-Class",
        "aliases": ["az", "az_class", "az-class", "AZ-Class", "AZ"],
        "train_file": "AZ_Class_Train.npz",
        "test_file": "AZ_Class_Test.npz",
        "train_md5": "644649dfb93f0f0086052a923b657833",
        "test_md5": "1e4014ff6eb613845a6bcf38a2461001",
        "expected_feature_dim": 1789,
        "num_classes": 100,
        "initial_classes": 50,
        "increment_classes_per_task": 5,
        "task_ordering": "random",
        "scaler_mode": "incremental",
        "notebook_source": "ahr_malcl_v15_1_final_paper_critic5_az_k200_one_cell.ipynb",
    },
}


@dataclass(frozen=True)
class ArrayIndexView:
    """Lazy indexed view over a NumPy array or memmap."""

    base: Any
    indices: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        base_shape = tuple(getattr(self.base, "shape", ()))
        return (int(len(self.indices)), *base_shape[1:])

    @property
    def dtype(self) -> Any:
        return getattr(self.base, "dtype", None)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: Any) -> Any:
        return self.base[self.indices[item]]

    def take_array(self, max_rows: int | None = None) -> np.ndarray:
        idx = self.indices[:max_rows] if max_rows is not None else self.indices
        return np.asarray(self.base[idx])


def canonical_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, cfg in DATASET_CONFIGS.items():
        aliases = {str(alias).lower().replace("-", "_").replace(" ", "_") for alias in cfg["aliases"]}
        if normalized == canonical or normalized in aliases:
            return canonical
    raise ValueError(f"Unknown dataset {dataset_name!r}; expected EMBER or AZ-Class.")


def get_dataset_config(dataset_name: str) -> dict[str, Any]:
    canonical = canonical_dataset_name(dataset_name)
    cfg = dict(DATASET_CONFIGS[canonical])
    cfg["record_id"] = RECORD_ID
    cfg["data_dir_candidates"] = [str(path) for path in data_dir_candidates()]
    return cfg


def data_dir_candidates(extra: Iterable[Path | str] = ()) -> list[Path]:
    env = os.environ.get("AHR_MALCL_DATA_DIR")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path("/content/data"),
            Path("/content/drive/MyDrive/AHR-MalCL/data"),
            Path("/content/drive/MyDrive/AHR-MalCL"),
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT,
        ]
    )
    candidates.extend(Path(item).expanduser() for item in extra)
    seen: set[str] = set()
    out = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def md5_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def find_dataset_files(dataset_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = get_dataset_config(dataset_name)
    if config:
        cfg.update(config.get("dataset_config", {}))
    extra_dirs = []
    for key in ("data_dir", "dataset_root"):
        value = (config or {}).get(key)
        if value:
            extra_dirs.append(value)
    train_name = cfg["train_file"]
    test_name = cfg["test_file"]
    checked: list[str] = []
    for root in data_dir_candidates(extra_dirs):
        train = root / train_name
        test = root / test_name
        checked.append(str(root))
        if train.exists() and test.exists():
            return {"ready": True, "train_npz": train, "test_npz": test, "checked_dirs": checked}
    return {
        "ready": False,
        "reason": f"Missing required files `{train_name}` and `{test_name}`.",
        "train_file": train_name,
        "test_file": test_name,
        "checked_dirs": checked,
    }


def extract_npz_to_npy(npz_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(npz_path, "r") as archive:
        npy_names = [name for name in archive.namelist() if name.endswith(".npy")]
        if not npy_names:
            raise ValueError(f"No .npy arrays found inside {npz_path}")
        for name in npy_names:
            dest = out_dir / f"{npz_path.stem}__{Path(name).name}"
            if not dest.exists() or dest.stat().st_size == 0:
                with archive.open(name) as src, dest.open("wb") as dst:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
            extracted.append(dest)
    return extracted


def pick_Xy_from_extracted_npy(npy_paths: list[Path]) -> tuple[Any, Any]:
    arrays = []
    for path in npy_paths:
        arr = np.load(path, mmap_mode="r")
        arrays.append((path, arr.shape, str(arr.dtype)))
    x_candidates = [(path, shape, dtype) for path, shape, dtype in arrays if len(shape) == 2]
    y_candidates = [(path, shape, dtype) for path, shape, dtype in arrays if len(shape) == 1]
    if not x_candidates or not y_candidates:
        raise ValueError(f"Could not identify X/y arrays. Found: {arrays}")
    x_path, x_shape, _ = max(x_candidates, key=lambda item: int(item[1][0]) * int(item[1][1]))
    n_rows = int(x_shape[0])
    compatible_y = [(path, shape, dtype) for path, shape, dtype in y_candidates if int(shape[0]) == n_rows]
    if not compatible_y:
        raise ValueError(f"No y array compatible with X rows={n_rows}. Found: {arrays}")

    def y_score(item: tuple[Path, tuple[int, ...], str]) -> int:
        return 1 if ("int" in item[2] or "uint" in item[2]) else 0

    y_path, _, _ = max(compatible_y, key=y_score)
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


def load_dataset_memmap(npz_path: Path, cache_dir: Path) -> tuple[Any, Any]:
    return pick_Xy_from_extracted_npy(extract_npz_to_npy(npz_path, cache_dir))


def subsample_indices(n_rows: int, max_n: int | None, seed: int) -> np.ndarray:
    indices = np.arange(int(n_rows), dtype=np.int64)
    if max_n and int(max_n) > 0 and int(max_n) < int(n_rows):
        rng = np.random.RandomState(int(seed))
        indices = rng.choice(indices, size=int(max_n), replace=False)
        indices.sort()
    return indices


def make_tasks(y: Any, first: int, step: int, total: int, seed: int, ordering: str) -> list[list[int]]:
    uniq, counts = np.unique(np.asarray(y), return_counts=True)
    if len(uniq) < total:
        total = len(uniq)
    if ordering == "giant_first":
        order = uniq[np.argsort(-counts)].tolist()
    else:
        rng = np.random.RandomState(int(seed))
        order_arr = uniq.copy()
        rng.shuffle(order_arr)
        order = order_arr.tolist()
    order = [int(value) for value in order[:total]]
    tasks = [order[:first]]
    cursor = first
    while cursor < total:
        tasks.append(order[cursor : cursor + step])
        cursor += step
    return tasks


def get_task_splits(dataset_name: str, seed: int, config: dict[str, Any] | None = None) -> list[list[int]]:
    stream = load_class_incremental_stream(dataset_name, seed, config=config)
    return [list(task["class_ids"]) for task in stream["tasks"]]


def get_feature_dim(dataset_name: str) -> int:
    cfg = get_dataset_config(dataset_name)
    found = find_dataset_files(dataset_name)
    if found.get("ready"):
        cache = Path(found["train_npz"]).parent / f"cache_train_{cfg['canonical_name']}"
        x_train, _ = load_dataset_memmap(Path(found["train_npz"]), cache)
        return int(x_train.shape[1])
    return int(cfg["expected_feature_dim"])


def get_num_classes(dataset_name: str) -> int:
    return int(get_dataset_config(dataset_name)["num_classes"])


def load_class_incremental_stream(
    dataset_name: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = get_dataset_config(dataset_name)
    config = dict(config or {})
    found = find_dataset_files(dataset_name, config=config)
    if not found.get("ready"):
        raise FileNotFoundError(found["reason"] + " Checked: " + ", ".join(found["checked_dirs"]))

    train_npz = Path(found["train_npz"])
    test_npz = Path(found["test_npz"])
    cache_root = Path(config.get("cache_dir") or train_npz.parent)
    canonical = cfg["canonical_name"]
    x_train, y_train = load_dataset_memmap(train_npz, cache_root / f"cache_train_{canonical}")
    x_test, y_test = load_dataset_memmap(test_npz, cache_root / f"cache_test_{canonical}")

    max_train = config.get("max_train_samples")
    max_test = config.get("max_test_samples")
    train_indices = subsample_indices(len(y_train), int(max_train) if max_train else None, seed)
    test_indices = subsample_indices(len(y_test), int(max_test) if max_test else None, seed)
    y_train_sub = np.asarray(y_train[train_indices])
    y_test_sub = np.asarray(y_test[test_indices])

    tasks_classes = make_tasks(
        y_train_sub,
        first=int(config.get("initial_classes", cfg["initial_classes"])),
        step=int(config.get("increment_classes_per_task", cfg["increment_classes_per_task"])),
        total=int(config.get("num_classes", cfg["num_classes"])),
        seed=int(seed),
        ordering=str(config.get("task_ordering", cfg["task_ordering"])),
    )
    class_order = [int(cls) for task in tasks_classes for cls in task]
    label_mapping = {int(cls): idx for idx, cls in enumerate(class_order)}

    tasks = []
    for task_id, class_ids in enumerate(tasks_classes):
        class_arr = np.asarray(class_ids, dtype=np.int64)
        task_train_indices = train_indices[np.isin(y_train_sub, class_arr)]
        task_test_indices = test_indices[np.isin(y_test_sub, class_arr)]
        tasks.append(
            {
                "task_id": int(task_id),
                "class_ids": [int(value) for value in class_ids],
                "X_train": ArrayIndexView(x_train, task_train_indices),
                "y_train": np.asarray(y_train[task_train_indices], dtype=np.int64),
                "X_test": ArrayIndexView(x_test, task_test_indices),
                "y_test": np.asarray(y_test[task_test_indices], dtype=np.int64),
                "train_indices": task_train_indices,
                "test_indices": task_test_indices,
            }
        )

    return {
        "dataset": cfg["canonical_name"],
        "dataset_display_name": cfg["display_name"],
        "seed": int(seed),
        "feature_dim": int(x_train.shape[1]),
        "num_classes": int(len(class_order)),
        "tasks": tasks,
        "label_mapping": label_mapping,
        "class_order": class_order,
        "metadata": {
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "scaler_mode": cfg["scaler_mode"],
            "task_ordering": str(config.get("task_ordering", cfg["task_ordering"])),
            "initial_classes": int(config.get("initial_classes", cfg["initial_classes"])),
            "increment_classes_per_task": int(config.get("increment_classes_per_task", cfg["increment_classes_per_task"])),
            "source_notebook": cfg["notebook_source"],
            "md5": {
                "train_expected": cfg["train_md5"],
                "test_expected": cfg["test_md5"],
            },
        },
    }


def summarize_stream(stream: dict[str, Any]) -> dict[str, Any]:
    labels = []
    has_nonfinite = False
    for task in stream["tasks"]:
        labels.extend(np.asarray(task["y_train"]).tolist())
        labels.extend(np.asarray(task["y_test"]).tolist())
        for key in ("X_train", "X_test"):
            view = task[key]
            sample = view.take_array(max_rows=min(len(view), 256)) if hasattr(view, "take_array") else np.asarray(view[:256])
            if sample.size and not np.isfinite(sample).all():
                has_nonfinite = True
    label_array = np.asarray(labels, dtype=np.int64) if labels else np.asarray([], dtype=np.int64)
    return {
        "dataset": stream["dataset"],
        "seed": int(stream["seed"]),
        "feature_dim": int(stream["feature_dim"]),
        "num_classes": int(stream["num_classes"]),
        "task_count": len(stream["tasks"]),
        "classes_per_task": [len(task["class_ids"]) for task in stream["tasks"]],
        "train_samples_per_task": [len(task["y_train"]) for task in stream["tasks"]],
        "test_samples_per_task": [len(task["y_test"]) for task in stream["tasks"]],
        "label_min": int(label_array.min()) if label_array.size else None,
        "label_max": int(label_array.max()) if label_array.size else None,
        "has_nan_or_inf_in_sample": bool(has_nonfinite),
        "class_order": list(stream["class_order"]),
        "metadata": dict(stream["metadata"]),
    }


def adapter_status(dataset_name: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    datasets = [dataset_name] if dataset_name else ["EMBER", "AZ-Class"]
    checks = {}
    ready = True
    for dataset in datasets:
        found = find_dataset_files(dataset, config=config)
        checks[canonical_dataset_name(dataset)] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in found.items()
        }
        ready = ready and bool(found.get("ready"))
    return {
        "ready": ready,
        "datasets": checks,
        "required_files": {
            name: [cfg["train_file"], cfg["test_file"]] for name, cfg in DATASET_CONFIGS.items()
        },
        "note": "Set AHR_MALCL_DATA_DIR or pass data_dir in config to point at mounted Colab/Drive dataset files.",
    }


def write_todo(dataset_name: str, reason: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_dataset_config(dataset_name)
    path = REPORT_DIR / "data_adapter_todo.md"
    lines = [
        "# Data Adapter TODO",
        "",
        "The data adapter code is present, but the dataset stream could not be loaded in this environment.",
        "",
        f"- Dataset: `{cfg['display_name']}`",
        f"- Reason: {reason}",
        "",
        "## Required Dataset Files",
        "",
        "- `EMBER_Class_train.npz`",
        "- `EMBER_Class_test.npz`",
        "- `AZ_Class_Train.npz`",
        "- `AZ_Class_Test.npz`",
        "",
        "## How To Unblock In Colab",
        "",
        "1. Mount Google Drive.",
        "2. Place the required `.npz` files in `/content/data`, `/content/drive/MyDrive/AHR-MalCL/data`, or set `AHR_MALCL_DATA_DIR`.",
        "3. Rerun `python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42`.",
        "4. Rerun `python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42`.",
        "",
        "No external files were downloaded by this adapter.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(VERSION, flush=True)
    print(adapter_status())
