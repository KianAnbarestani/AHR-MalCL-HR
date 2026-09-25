#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from patched_malcl.models import Classifier, Discriminator, Generator, model_memory_mb, run_topology_validation


OFFICIAL_COMMIT = "2cfb344461f594877e24fd4d19919b6bca38c861"
OFFICIAL_REL = Path("runners/external_code/MalCL_official")
EXPECTED_SEED42_FIRST10 = [83, 53, 70, 45, 44, 39, 22, 80, 10, 0]
SEEDS = list(range(42, 52))
DATASETS = ["EMBER", "AZ-Class"]
FEATURE_DIMS = {"EMBER": 2381, "AZ-Class": 2439}
EXPECTED_SHAPES = {
    "EMBER": {"train": (303331, 2381), "test": (33704, 2381)},
    "AZ-Class": {"train": (257023, 2439), "test": (28559, 2439)},
}
DATASET_FILES = {
    "EMBER": {
        "train": "EMBER_Class_train.npz",
        "test": "EMBER_Class_test.npz",
        "x_train_keys": ["X_train", "x_train", "X", "x"],
        "y_train_keys": ["Y_train", "y_train", "y", "labels"],
        "x_test_keys": ["X_test", "x_test", "X", "x"],
        "y_test_keys": ["Y_test", "y_test", "y", "labels"],
    },
    "AZ-Class": {
        "train": "AZ_Class_Train.npz",
        "test": "AZ_Class_Test.npz",
        "x_train_keys": ["X_train", "x_train", "X", "x"],
        "y_train_keys": ["Y_train", "y_train", "y", "labels"],
        "x_test_keys": ["X_test", "x_test", "X", "x"],
        "y_test_keys": ["Y_test", "y_test", "y", "labels"],
    },
}


MALCL_EPOCHS = int(os.environ.get("MALCL_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("MALCL_BATCH_SIZE", "256"))
FAST_DEBUG = os.environ.get("FAST_DEBUG", "False").lower() == "true"
ALLOW_SUBSAMPLING = os.environ.get("ALLOW_SUBSAMPLING", "False").lower() == "true"
GENERATOR_LOSS = os.environ.get("MALCL_GENERATOR_LOSS", "FML")
SAMPLE_SELECT = os.environ.get("MALCL_SAMPLE_SELECT", "L1_C_Mean")
MALCL_K = int(os.environ.get("MALCL_K", "3"))
LEARNING_RATE = float(os.environ.get("MALCL_LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("MALCL_WEIGHT_DECAY", "0.000001"))
MOMENTUM = float(os.environ.get("MALCL_MOMENTUM", "0.9"))
REPLAY_REFRESH_EVERY_BATCH = os.environ.get("REPLAY_REFRESH_EVERY_BATCH", "True").lower() == "true"
FAST_DEBUG_MAX_BATCHES = int(os.environ.get("FAST_DEBUG_MAX_BATCHES", "2"))
FAST_DEBUG_MAX_ROWS_PER_SPLIT = int(os.environ.get("FAST_DEBUG_MAX_ROWS_PER_SPLIT", "12000"))
PRINT_TRAINING_PROGRESS = os.environ.get("PRINT_TRAINING_PROGRESS", "True").lower() == "true"
PROGRESS_EVERY_N_BATCHES = max(1, int(os.environ.get("PROGRESS_EVERY_N_BATCHES", "25")))
USE_TQDM_PROGRESS = os.environ.get("USE_TQDM_PROGRESS", "True").lower() == "true"
TQDM_MININTERVAL = float(os.environ.get("TQDM_MININTERVAL", "2.0"))


def load_tqdm():
    if not USE_TQDM_PROGRESS:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


TQDM = load_tqdm()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def emit_status(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "configs": base / "configs",
        "validation": base / "validation",
        "raw_outputs": base / "raw_outputs",
        "tables": base / "tables",
        "logs": base / "logs",
        "memory": base / "memory",
        "reports": base / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def optional_imports() -> dict[str, object]:
    modules: dict[str, object] = {}
    for name in ["numpy", "pandas", "sklearn", "sklearn.metrics", "sklearn.preprocessing", "torch"]:
        try:
            modules[name] = __import__(name, fromlist=["*"])
        except Exception as exc:
            modules[name] = exc
    return modules


def validate_environment(mods: dict[str, object]) -> dict[str, object]:
    torch = mods["torch"]
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": getattr(mods["numpy"], "__version__", None),
        "pandas_version": getattr(mods["pandas"], "__version__", None),
        "sklearn_version": getattr(mods["sklearn"], "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": False,
        "gpu_name": None,
        "available_gpu_memory_mb": None,
        "nvidia_smi": run_cmd(["nvidia-smi"], timeout=30),
        "passed": True,
        "errors": [],
    }
    for name, module in mods.items():
        if isinstance(module, Exception):
            env["passed"] = False
            env["errors"].append(f"{name} import failed: {module!r}")
    if not isinstance(torch, Exception):
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                env["available_gpu_memory_mb"] = free_bytes / (1024 * 1024)
                env["total_gpu_memory_mb"] = total_bytes / (1024 * 1024)
            except Exception as exc:
                env["errors"].append(f"torch.cuda.mem_get_info failed: {exc!r}")
        else:
            env["passed"] = False
            env["errors"].append("CUDA is not available; full MalCL baseline requires Colab GPU")
    return env


def dataset_candidates(filename: str) -> list[Path]:
    configured = [
        Path(value) / filename
        for value in (os.environ.get("AHR_DATA_DIR"), os.environ.get("AHR_EXTRA_DATA_DIR"))
        if value
    ]
    return configured + [PROJECT_ROOT / "data" / filename, Path.cwd() / "data" / filename]


def find_dataset_file(filename: str) -> Path | None:
    for path in dataset_candidates(filename):
        if path.exists():
            return path
    return None


def pick_key(npz, candidates: list[str]) -> str | None:
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def validate_datasets(np_mod) -> dict[str, object]:
    report = {"passed": True, "datasets": {}, "errors": []}
    for dataset in DATASETS:
        emit_status(f"DATASET_VALIDATION_DATASET_START dataset={dataset}")
        spec = DATASET_FILES[dataset]
        expected = EXPECTED_SHAPES[dataset]
        item = {"passed": True, "files": {}, "keys": {}, "shapes": {}, "labels": {}}
        train_path = find_dataset_file(spec["train"])
        test_path = find_dataset_file(spec["test"])
        item["files"] = {"train": str(train_path) if train_path else None, "test": str(test_path) if test_path else None}
        if train_path is None or test_path is None:
            item["passed"] = False
            report["passed"] = False
            report["errors"].append(f"{dataset}: missing train/test dataset files")
            report["datasets"][dataset] = item
            continue
        try:
            emit_status(f"DATASET_VALIDATION_LOAD_START dataset={dataset} train={train_path} test={test_path}")
            train = np_mod.load(train_path)
            test = np_mod.load(test_path)
            x_train_key = pick_key(train, spec["x_train_keys"])
            y_train_key = pick_key(train, spec["y_train_keys"])
            x_test_key = pick_key(test, spec["x_test_keys"])
            y_test_key = pick_key(test, spec["y_test_keys"])
            item["keys"] = {
                "x_train": x_train_key,
                "y_train": y_train_key,
                "x_test": x_test_key,
                "y_test": y_test_key,
            }
            if None in item["keys"].values():
                item["passed"] = False
                report["passed"] = False
                report["errors"].append(f"{dataset}: missing expected X/Y keys in npz files")
                report["datasets"][dataset] = item
                continue
            emit_status(
                f"DATASET_VALIDATION_ARRAY_CHECK_START dataset={dataset} "
                f"x_train_key={x_train_key} x_test_key={x_test_key}"
            )
            x_train = train[x_train_key]
            y_train = train[y_train_key]
            x_test = test[x_test_key]
            y_test = test[y_test_key]
            item["shapes"] = {"train": list(x_train.shape), "test": list(x_test.shape)}
            if tuple(x_train.shape) != expected["train"]:
                item["passed"] = False
                report["passed"] = False
                report["errors"].append(f"{dataset}: train shape {tuple(x_train.shape)} != {expected['train']}")
            if tuple(x_test.shape) != expected["test"]:
                item["passed"] = False
                report["passed"] = False
                report["errors"].append(f"{dataset}: test shape {tuple(x_test.shape)} != {expected['test']}")
            labels = np_mod.concatenate([np_mod.asarray(y_train).reshape(-1), np_mod.asarray(y_test).reshape(-1)])
            unique = np_mod.unique(labels)
            item["labels"] = {
                "min": int(labels.min()),
                "max": int(labels.max()),
                "unique_count": int(len(unique)),
                "first10_unique": [int(v) for v in unique[:10]],
            }
            if item["labels"]["min"] != 0 or item["labels"]["max"] != 99 or item["labels"]["unique_count"] != 100:
                item["passed"] = False
                report["passed"] = False
                report["errors"].append(f"{dataset}: label range/classes invalid: {item['labels']}")
            emit_status(
                f"DATASET_VALIDATION_DATASET_DONE dataset={dataset} passed={item['passed']} "
                f"train_shape={item['shapes'].get('train')} test_shape={item['shapes'].get('test')}"
            )
        except Exception as exc:
            item["passed"] = False
            report["passed"] = False
            report["errors"].append(f"{dataset}: dataset validation failed: {type(exc).__name__}: {exc}")
            emit_status(f"DATASET_VALIDATION_DATASET_FAILED dataset={dataset} error={type(exc).__name__}: {exc}")
        report["datasets"][dataset] = item
    return report


def validate_official_malcl() -> dict[str, object]:
    official_root = PROJECT_ROOT / OFFICIAL_REL
    required = [
        "MalCL_torch/models.py",
        "MalCL_torch/train.py",
        "MalCL_torch/sample_selection.py",
        "MalCL_torch/function.py",
        "MalCL_torch/main.py",
        "MalCL_torch/arguments.py",
    ]
    report = {
        "official_root": str(official_root),
        "expected_commit": OFFICIAL_COMMIT,
        "actual_commit": None,
        "commit_source": None,
        "commit_matches": False,
        "required_files": {},
        "passed": True,
        "errors": [],
    }
    if not official_root.exists():
        report["passed"] = False
        report["errors"].append("official MalCL clone is missing")
        return report
    git = run_cmd(["git", "rev-parse", "HEAD"], cwd=official_root)
    if git["returncode"] == 0:
        report["actual_commit"] = git["stdout"]
        report["commit_source"] = "git rev-parse HEAD"
    else:
        marker = official_root / "OFFICIAL_COMMIT.txt"
        if marker.exists():
            report["actual_commit"] = marker.read_text(encoding="utf-8").strip()
            report["commit_source"] = "OFFICIAL_COMMIT.txt"
        else:
            report["actual_commit"] = None
            report["commit_source"] = "missing git metadata and OFFICIAL_COMMIT.txt"
    report["commit_matches"] = report["actual_commit"] == OFFICIAL_COMMIT
    if not report["commit_matches"]:
        report["passed"] = False
        report["errors"].append(f"official commit mismatch: {report['actual_commit']} != {OFFICIAL_COMMIT}")
    for rel_path in required:
        exists = (official_root / rel_path).exists()
        report["required_files"][rel_path] = exists
        if not exists:
            report["passed"] = False
            report["errors"].append(f"missing official file: {rel_path}")
    return report


def build_class_order(np_mod, seed: int) -> list[int]:
    return [int(v) for v in np_mod.random.RandomState(seed).permutation(100).tolist()]


def task_splits(order: list[int]) -> list[list[int]]:
    return [order[:50]] + [order[i : i + 5] for i in range(50, 100, 5)]


def validate_class_orders(np_mod) -> dict[str, object]:
    report = {"passed": True, "rule": "numpy.random.RandomState(seed).permutation(100)", "seeds": {}, "errors": []}
    for seed in SEEDS:
        order = build_class_order(np_mod, seed)
        tasks = task_splits(order)
        item = {
            "seed": seed,
            "class_order": order,
            "first10": order[:10],
            "tasks": tasks,
            "task_sizes": [len(task) for task in tasks],
            "passed": len(order) == 100 and len(set(order)) == 100 and [len(task) for task in tasks] == [50] + [5] * 10,
        }
        if seed == 42 and order[:10] != EXPECTED_SEED42_FIRST10:
            item["passed"] = False
            report["errors"].append(f"seed 42 first10 mismatch: {order[:10]}")
        if not item["passed"]:
            report["passed"] = False
        report["seeds"][str(seed)] = item
    return report


def final_config(class_order_report: dict[str, object], output_root: Path) -> dict[str, object]:
    return {
        "created_at": utc_now(),
        "method": "MalCL",
        "mode": "final_protocol_matched_malcl_baseline",
        "official_malcl_commit": OFFICIAL_COMMIT,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "num_classes": 100,
        "num_tasks": 11,
        "task_sizes": [50] + [5] * 10,
        "feature_dims": FEATURE_DIMS,
        "expected_shapes": EXPECTED_SHAPES,
        "class_order_rule": class_order_report["rule"],
        "class_orders": class_order_report["seeds"],
        "training": {
            "epochs": MALCL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "z_dim": 62,
            "lr": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "generator_loss": GENERATOR_LOSS,
            "sample_select": SAMPLE_SELECT,
            "k": MALCL_K,
            "allow_subsampling": ALLOW_SUBSAMPLING,
            "fast_debug": FAST_DEBUG,
            "replay_refresh_every_batch": REPLAY_REFRESH_EVERY_BATCH,
            "print_training_progress": PRINT_TRAINING_PROGRESS,
            "progress_every_n_batches": PROGRESS_EVERY_N_BATCHES,
            "use_tqdm_progress": USE_TQDM_PROGRESS,
            "tqdm_mininterval": TQDM_MININTERVAL,
            "discriminator_topology": "feature-dimension-parameterized flattened discriminator",
            "official_feature_length_dependent_flattened_topology": True,
        },
        "output_paths": {
            "output_root": str(output_root),
            "raw_outputs": str(output_root / "raw_outputs"),
            "tables": str(output_root / "tables"),
            "memory": str(output_root / "memory"),
            "reports": str(output_root / "reports"),
        },
        "memory_accounting_rules": [
            "parameter memory for live generator/discriminator/classifier",
            "copied past generator/classifier memory for replay",
            "generated replay tensor memory accumulated per run",
            "peak GPU memory from torch.cuda.max_memory_allocated",
        ],
    }


def remap_labels(np_mod, y, order: list[int]):
    mapping = np_mod.zeros(100, dtype=np_mod.int64)
    for new_label, original_label in enumerate(order):
        mapping[int(original_label)] = int(new_label)
    return mapping[np_mod.asarray(y, dtype=np_mod.int64).reshape(-1)]


def load_dataset_arrays(np_mod, dataset: str, order: list[int]) -> dict[str, object]:
    spec = DATASET_FILES[dataset]
    train_path = find_dataset_file(spec["train"])
    test_path = find_dataset_file(spec["test"])
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"{dataset}: dataset files are missing")
    train = np_mod.load(train_path)
    test = np_mod.load(test_path)
    x_train_key = pick_key(train, spec["x_train_keys"])
    y_train_key = pick_key(train, spec["y_train_keys"])
    x_test_key = pick_key(test, spec["x_test_keys"])
    y_test_key = pick_key(test, spec["y_test_keys"])
    x_train = train[x_train_key].astype("float32", copy=False)
    y_train_original = train[y_train_key]
    x_test = test[x_test_key].astype("float32", copy=False)
    y_test_original = test[y_test_key]
    y_train = remap_labels(np_mod, y_train_original, order)
    y_test = remap_labels(np_mod, y_test_original, order)
    if ALLOW_SUBSAMPLING or FAST_DEBUG:
        n_train = min(len(x_train), FAST_DEBUG_MAX_ROWS_PER_SPLIT)
        n_test = min(len(x_test), FAST_DEBUG_MAX_ROWS_PER_SPLIT)
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]
    return {
        "x_train": x_train,
        "y_train": y_train.astype("int64", copy=False),
        "x_test": x_test,
        "y_test": y_test.astype("int64", copy=False),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "x_train_key": x_train_key,
        "y_train_key": y_train_key,
        "x_test_key": x_test_key,
        "y_test_key": y_test_key,
    }


def model_param_memory(generator, discriminator, classifier) -> dict[str, float]:
    return {
        "generator_parameter_memory_mb": model_memory_mb(generator),
        "discriminator_parameter_memory_mb": model_memory_mb(discriminator),
        "classifier_parameter_memory_mb": model_memory_mb(classifier),
    }


def tensor_memory_mb(tensor) -> float:
    return float(tensor.nelement() * tensor.element_size() / (1024 * 1024))


def classifier_prob_loss(torch_mod, probs, labels):
    selected = probs.gather(1, labels.view(-1, 1)).clamp_min(1e-8)
    return -torch_mod.log(selected).mean()


def make_loader(torch_mod, np_mod, x_scaled, y, batch_size: int):
    class_sample_count = np_mod.array([max(1, len(np_mod.where(y == t)[0])) for t in np_mod.unique(y)])
    min_label = int(np_mod.min(np_mod.unique(y)))
    weights = 1.0 / class_sample_count
    samples_weight = np_mod.array([weights[int(label) - min_label] for label in y])
    sampler = torch_mod.utils.data.WeightedRandomSampler(
        torch_mod.from_numpy(samples_weight).float(),
        len(samples_weight),
        replacement=True,
    )
    dataset = torch_mod.utils.data.TensorDataset(
        torch_mod.from_numpy(x_scaled).float(),
        torch_mod.from_numpy(y.astype("int64", copy=False)).long(),
    )
    return torch_mod.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, pin_memory=torch_mod.cuda.is_available())


def compute_class_feature_means(torch_mod, np_mod, classifier, scaler, x_train, y_train, seen_classes, device, batch_size: int):
    classifier.eval()
    means = []
    for cls in seen_classes:
        idx = np_mod.flatnonzero(y_train == cls)
        if len(idx) == 0:
            means.append(None)
            continue
        total = None
        count = 0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = scaler.transform(x_train[batch_idx]).astype("float32", copy=False)
            xb_t = torch_mod.from_numpy(xb).float().to(device)
            with torch_mod.no_grad():
                feat = classifier.get_logits(xb_t).detach()
            summed = feat.sum(dim=0)
            total = summed if total is None else total + summed
            count += feat.shape[0]
        means.append((total / max(1, count)).detach())
    valid = [m for m in means if m is not None]
    if not valid:
        return None
    return torch_mod.stack([m if m is not None else torch_mod.zeros_like(valid[0]) for m in means]).to(device)


def select_replay(torch_mod, classifier, generator, logits_real, prev_classes: int, config: SimpleNamespace, device):
    if logits_real is None or prev_classes <= 0:
        return None, None, 0.0
    generator.eval()
    classifier.eval()
    sample_target = max(prev_classes * config.k, config.batch_size)
    syn_batches = int(math.ceil(sample_target / float(config.batch_size)))
    synthetic_parts = []
    logits_parts = []
    with torch_mod.no_grad():
        for _ in range(syn_batches):
            z = torch_mod.rand((config.batch_size, config.z_dim), device=device)
            syn = generator(z)
            synthetic_parts.append(syn)
            logits_parts.append(classifier.get_logits(syn).detach())
    synthetic = torch_mod.cat(synthetic_parts, dim=0)[:sample_target]
    logits_gen = torch_mod.cat(logits_parts, dim=0)[:sample_target]
    dist = torch_mod.mean(torch_mod.abs(logits_real.unsqueeze(1) - logits_gen.unsqueeze(0)), dim=2)
    selected_indices = []
    selected_labels = []
    masked = dist.detach().clone()
    for cls in range(prev_classes):
        for _ in range(config.k):
            ind = torch_mod.argmin(masked[cls]).item()
            selected_indices.append(ind)
            selected_labels.append(cls)
            masked[:, ind] = float("inf")
    replay = synthetic[selected_indices].detach()
    labels = torch_mod.tensor(selected_labels, dtype=torch_mod.long, device=device)
    return replay, labels, tensor_memory_mb(replay)


def evaluate_seen(torch_mod, np_mod, metrics_mod, classifier, scaler, x_test, y_test, seen_classes: list[int], task_ranges: list[tuple[int, int]], device, batch_size: int):
    classifier.eval()
    y_true_all = []
    y_pred_all = []
    task_accuracies = []
    for start_cls, end_cls in task_ranges:
        idx = np_mod.flatnonzero((y_test >= start_cls) & (y_test < end_cls))
        if len(idx) == 0:
            task_accuracies.append(None)
            continue
        preds = []
        true = []
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = scaler.transform(x_test[batch_idx]).astype("float32", copy=False)
            xb_t = torch_mod.from_numpy(xb).float().to(device)
            with torch_mod.no_grad():
                probs = classifier(xb_t)
                pred = torch_mod.argmax(probs, dim=1).detach().cpu().numpy()
            preds.extend([int(v) for v in pred.tolist()])
            true.extend([int(v) for v in y_test[batch_idx].tolist()])
        acc = float(np_mod.mean(np_mod.asarray(preds) == np_mod.asarray(true)))
        task_accuracies.append(acc)
        y_true_all.extend(true)
        y_pred_all.extend(preds)
    seen_idx = np_mod.flatnonzero(np_mod.isin(y_test, np_mod.asarray(seen_classes, dtype=np_mod.int64)))
    if len(seen_idx) > 0:
        y_true_all = []
        y_pred_all = []
        for start in range(0, len(seen_idx), batch_size):
            batch_idx = seen_idx[start : start + batch_size]
            xb = scaler.transform(x_test[batch_idx]).astype("float32", copy=False)
            xb_t = torch_mod.from_numpy(xb).float().to(device)
            with torch_mod.no_grad():
                probs = classifier(xb_t)
                pred = torch_mod.argmax(probs, dim=1).detach().cpu().numpy()
            y_pred_all.extend([int(v) for v in pred.tolist()])
            y_true_all.extend([int(v) for v in y_test[batch_idx].tolist()])
    return task_accuracies, y_true_all, y_pred_all


def metric_bundle(np_mod, metrics_mod, accuracy_matrix: list[list[float | None]], y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    final_row = accuracy_matrix[-1] if accuracy_matrix else []
    final_valid = [v for v in final_row if v is not None]
    seen_avgs = [float(np_mod.mean([v for v in row if v is not None])) for row in accuracy_matrix if [v for v in row if v is not None]]
    final_taskwise = float(np_mod.mean(final_valid)) if final_valid else None
    mean_seen = float(np_mod.mean(seen_avgs)) if seen_avgs else None
    forgetting_vals = []
    if accuracy_matrix:
        cols = len(final_row)
        for col in range(cols):
            values = [row[col] for row in accuracy_matrix if col < len(row) and row[col] is not None]
            if values:
                forgetting_vals.append(max(values) - values[-1])
    forgetting = float(np_mod.mean(forgetting_vals)) if forgetting_vals else None
    result = {
        "per_task_accuracy": seen_avgs,
        "final_task_accuracies": final_row,
        "final_taskwise_accuracy": final_taskwise,
        "mean_seen_accuracy": mean_seen,
        "forgetting": forgetting,
        "macro_f1": None,
        "weighted_f1": None,
        "balanced_accuracy": None,
        "old_class_accuracy": None,
        "new_class_accuracy": None,
    }
    if y_true and y_pred:
        result["macro_f1"] = float(metrics_mod.f1_score(y_true, y_pred, average="macro", zero_division=0))
        result["weighted_f1"] = float(metrics_mod.f1_score(y_true, y_pred, average="weighted", zero_division=0))
        result["balanced_accuracy"] = float(metrics_mod.balanced_accuracy_score(y_true, y_pred))
        y_true_np = np_mod.asarray(y_true)
        y_pred_np = np_mod.asarray(y_pred)
        old_idx = y_true_np < 50
        new_idx = y_true_np >= 50
        if int(old_idx.sum()) > 0:
            result["old_class_accuracy"] = float(np_mod.mean(y_true_np[old_idx] == y_pred_np[old_idx]))
        if int(new_idx.sum()) > 0:
            result["new_class_accuracy"] = float(np_mod.mean(y_true_np[new_idx] == y_pred_np[new_idx]))
    return result


def valid_existing_run(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required_metrics = ["final_taskwise_accuracy", "mean_seen_accuracy", "forgetting", "macro_f1", "weighted_f1", "balanced_accuracy"]
        return (
            data.get("real_training_executed") is True
            and data.get("synthetic_fallback_used") is False
            and data.get("full_task_completed") is True
            and data.get("full_data_completed") is True
            and data.get("class_order_matched") is True
            and all(data.get(metric) is not None for metric in required_metrics)
            and float(data.get("memory_mb_total_method_side_estimate") or 0.0) > 0.0
        )
    except Exception:
        return False


def run_dataset_seed(np_mod, torch_mod, preprocessing_mod, metrics_mod, dataset: str, seed: int, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_result_path = output_dir / "full-result.json"
    if valid_existing_run(full_result_path):
        print(f"RESUME_SKIP dataset={dataset} seed={seed} existing_valid_result={full_result_path}", flush=True)
        data = json.loads(full_result_path.read_text(encoding="utf-8"))
        data["resume_skipped"] = True
        return data

    warnings: list[str] = []
    errors: list[str] = []
    progress_messages: list[str] = []
    progress_rows: list[dict[str, object]] = []

    def emit(message: str) -> None:
        stamped = f"[{utc_now()}] {message}"
        progress_messages.append(stamped)
        if PRINT_TRAINING_PROGRESS:
            print(stamped, flush=True)

    start_time = time.time()
    real_training_executed = False
    full_task_completed = False
    full_data_completed = False
    order = build_class_order(np_mod, seed)
    tasks_original = task_splits(order)
    task_ranges = [(0, 50)] + [(i, i + 5) for i in range(50, 100, 5)]
    feature_dim = FEATURE_DIMS[dataset]
    config = SimpleNamespace(
        dataset=dataset,
        seed=seed,
        z_dim=62,
        epochs=MALCL_EPOCHS,
        batch_size=BATCH_SIZE,
        k=MALCL_K,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        generator_loss=GENERATOR_LOSS,
        sample_select=SAMPLE_SELECT,
    )

    metrics = {
        "per_task_accuracy": [],
        "final_task_accuracies": [],
        "final_taskwise_accuracy": None,
        "mean_seen_accuracy": None,
        "forgetting": None,
        "macro_f1": None,
        "weighted_f1": None,
        "balanced_accuracy": None,
        "old_class_accuracy": None,
        "new_class_accuracy": None,
    }
    memory = {
        "generator_parameter_memory_mb": 0.0,
        "discriminator_parameter_memory_mb": 0.0,
        "classifier_parameter_memory_mb": 0.0,
        "copied_past_generator_classifier_memory_mb": 0.0,
        "generated_replay_tensor_memory_mb": 0.0,
        "peak_gpu_memory_mb": None,
        "memory_mb_total_method_side_estimate": 0.0,
    }
    status = "failed"

    try:
        emit(
            f"RUN_PREP dataset={dataset} seed={seed} feature_dim={feature_dim} "
            f"epochs={MALCL_EPOCHS} batch_size={BATCH_SIZE} progress_every={PROGRESS_EVERY_N_BATCHES}"
        )
        if FAST_DEBUG:
            warnings.append("FAST_DEBUG=True; run is incomplete and not paper-usable")
        if ALLOW_SUBSAMPLING:
            warnings.append("ALLOW_SUBSAMPLING=True; run is incomplete and not paper-usable")
        if GENERATOR_LOSS != "FML":
            warnings.append(f"Generator loss is {GENERATOR_LOSS}, not official default FML")
        if SAMPLE_SELECT != "L1_C_Mean":
            warnings.append(f"Sample selection is {SAMPLE_SELECT}, not official default L1_C_Mean")

        data = load_dataset_arrays(np_mod, dataset, order)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        emit(
            f"DATA_LOADED dataset={dataset} seed={seed} train_shape={tuple(x_train.shape)} "
            f"test_shape={tuple(x_test.shape)} device={device}"
        )
        torch_mod.manual_seed(seed)
        np_mod.random.seed(seed)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.manual_seed_all(seed)
            torch_mod.cuda.reset_peak_memory_stats()

        generator = Generator(feature_dim=feature_dim, z_dim=62).to(device)
        discriminator = Discriminator(feature_dim=feature_dim).to(device)
        classifier = Classifier(feature_dim=feature_dim, output_dim=50).to(device)
        generator.train()
        discriminator.train()
        classifier.train()
        generator.reinit()
        discriminator.reinit()

        g_optimizer = torch_mod.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
        d_optimizer = torch_mod.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
        c_optimizer = torch_mod.optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        bce = torch_mod.nn.BCELoss()
        scaler = preprocessing_mod.StandardScaler()
        accuracy_matrix: list[list[float | None]] = []
        final_true: list[int] = []
        final_pred: list[int] = []
        logits_real = None
        past_generator = None
        past_classifier = None
        generated_replay_mb = 0.0

        max_batches = FAST_DEBUG_MAX_BATCHES if FAST_DEBUG else None
        task_iter = task_ranges
        if TQDM is not None:
            task_iter = TQDM(
                task_ranges,
                total=len(task_ranges),
                desc=f"{dataset} seed {seed} tasks",
                position=1,
                leave=True,
                dynamic_ncols=True,
                mininterval=TQDM_MININTERVAL,
            )
        for task_id, (start_cls, end_cls) in enumerate(task_iter):
            if task_id > 0:
                classifier.expand_output_layer(50, 5, task_id)
                classifier.to(device)
                c_optimizer = torch_mod.optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

            train_idx = np_mod.flatnonzero((y_train >= start_cls) & (y_train < end_cls))
            if len(train_idx) == 0:
                raise RuntimeError(f"{dataset} seed {seed} task {task_id}: no training samples")
            x_task = x_train[train_idx]
            y_task = y_train[train_idx]
            scaler.partial_fit(x_task)
            x_task_scaled = scaler.transform(x_task).astype("float32", copy=False)
            loader = make_loader(torch_mod, np_mod, x_task_scaled, y_task, BATCH_SIZE)
            total_batches = len(loader)
            effective_batches = min(total_batches, max_batches) if max_batches is not None else total_batches
            task_classes = tasks_original[task_id]
            if TQDM is not None and hasattr(task_iter, "set_postfix"):
                task_iter.set_postfix(task=f"{task_id + 1}/11", samples=len(train_idx), refresh=True)
            emit(
                f"TASK_START dataset={dataset} seed={seed} task={task_id + 1}/11 "
                f"remapped_classes={start_cls}-{end_cls - 1} original_classes={task_classes} "
                f"train_samples={len(train_idx)} epochs={MALCL_EPOCHS} batches_per_epoch={effective_batches}"
            )

            replay_cache = None
            replay_label_cache = None
            if task_id > 0 and not REPLAY_REFRESH_EVERY_BATCH:
                replay_cache, replay_label_cache, replay_mb = select_replay(torch_mod, past_classifier, past_generator, logits_real, start_cls, config, device)
                generated_replay_mb += replay_mb

            epoch_iter = range(MALCL_EPOCHS)
            if TQDM is not None:
                epoch_iter = TQDM(
                    epoch_iter,
                    total=MALCL_EPOCHS,
                    desc=f"{dataset} seed {seed} task {task_id + 1}/11 epochs",
                    position=2,
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=TQDM_MININTERVAL,
                )
            for epoch in epoch_iter:
                epoch_start = time.time()
                last_d_loss = None
                last_g_loss = None
                last_c_loss = None
                if TQDM is not None and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(epoch=f"{epoch + 1}/{MALCL_EPOCHS}", refresh=True)
                emit(
                    f"EPOCH_START dataset={dataset} seed={seed} task={task_id + 1}/11 "
                    f"epoch={epoch + 1}/{MALCL_EPOCHS} batches={effective_batches}"
                )
                batch_iterable = loader if max_batches is None else itertools.islice(loader, max_batches)
                if TQDM is not None:
                    batch_iterable = TQDM(
                        batch_iterable,
                        total=effective_batches,
                        desc=f"{dataset} seed {seed} task {task_id + 1}/11 epoch {epoch + 1}/{MALCL_EPOCHS} batches",
                        position=3,
                        leave=False,
                        dynamic_ncols=True,
                        mininterval=TQDM_MININTERVAL,
                    )
                for batch_id, (xb_cpu, yb_cpu) in enumerate(batch_iterable):
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)

                    if task_id > 0:
                        if REPLAY_REFRESH_EVERY_BATCH:
                            replay, replay_labels, replay_mb = select_replay(torch_mod, past_classifier, past_generator, logits_real, start_cls, config, device)
                            generated_replay_mb += replay_mb
                        else:
                            replay, replay_labels = replay_cache, replay_label_cache
                        if replay is not None and replay_labels is not None and len(replay) > 0:
                            xb_for_classifier = torch_mod.cat([xb, replay], dim=0)
                            yb_for_classifier = torch_mod.cat([yb, replay_labels], dim=0)
                        else:
                            xb_for_classifier = xb
                            yb_for_classifier = yb
                    else:
                        xb_for_classifier = xb
                        yb_for_classifier = yb

                    generator.train()
                    discriminator.train()
                    classifier.train()

                    y_real = torch_mod.ones((xb.shape[0], 1), device=device)
                    y_fake = torch_mod.zeros((xb.shape[0], 1), device=device)
                    z = torch_mod.rand((xb.shape[0], 62), device=device)
                    fake = generator(z)

                    d_optimizer.zero_grad()
                    d_real, _ = discriminator(xb)
                    d_fake, _ = discriminator(fake.detach())
                    d_loss = bce(d_real, y_real) + bce(d_fake, y_fake)
                    d_loss.backward()
                    d_optimizer.step()

                    g_optimizer.zero_grad()
                    z = torch_mod.rand((xb.shape[0], 62), device=device)
                    fake = generator(z)
                    if GENERATOR_LOSS == "FML":
                        _, features_fake = discriminator(fake)
                        _, features_real = discriminator(xb)
                        g_loss = torch_mod.mean(torch_mod.abs(features_real.detach().mean(dim=0) - features_fake.mean(dim=0)))
                    else:
                        d_fake_for_g, _ = discriminator(fake)
                        g_loss = bce(d_fake_for_g, torch_mod.ones_like(d_fake_for_g))
                    g_loss.backward()
                    g_optimizer.step()

                    c_optimizer.zero_grad()
                    probs = classifier(xb_for_classifier)
                    c_loss = classifier_prob_loss(torch_mod, probs, yb_for_classifier)
                    c_loss.backward()
                    c_optimizer.step()
                    real_training_executed = True
                    last_d_loss = float(d_loss.detach().cpu().item())
                    last_g_loss = float(g_loss.detach().cpu().item())
                    last_c_loss = float(c_loss.detach().cpu().item())
                    batch_num = batch_id + 1
                    if batch_num == 1 or batch_num % PROGRESS_EVERY_N_BATCHES == 0 or batch_num == effective_batches:
                        elapsed = time.time() - start_time
                        progress_row = {
                            "timestamp": utc_now(),
                            "dataset": dataset,
                            "seed": seed,
                            "task_id": task_id,
                            "task_number": task_id + 1,
                            "epoch": epoch + 1,
                            "batch": batch_num,
                            "total_batches": effective_batches,
                            "d_loss": last_d_loss,
                            "g_loss": last_g_loss,
                            "c_loss": last_c_loss,
                            "generated_replay_mb": generated_replay_mb,
                            "elapsed_seconds": elapsed,
                        }
                        progress_rows.append(progress_row)
                        if TQDM is not None and hasattr(batch_iterable, "set_postfix"):
                            batch_iterable.set_postfix(
                                d=f"{last_d_loss:.4f}",
                                g=f"{last_g_loss:.4f}",
                                c=f"{last_c_loss:.4f}",
                                replay=f"{generated_replay_mb:.1f}MB",
                                refresh=True,
                            )
                        emit(
                            f"BATCH_PROGRESS dataset={dataset} seed={seed} task={task_id + 1}/11 "
                            f"epoch={epoch + 1}/{MALCL_EPOCHS} batch={batch_num}/{effective_batches} "
                            f"d_loss={last_d_loss:.6f} g_loss={last_g_loss:.6f} c_loss={last_c_loss:.6f} "
                            f"replay_mb={generated_replay_mb:.3f} elapsed_sec={elapsed:.1f}"
                        )

                emit(
                    f"EPOCH_DONE dataset={dataset} seed={seed} task={task_id + 1}/11 "
                    f"epoch={epoch + 1}/{MALCL_EPOCHS} seconds={time.time() - epoch_start:.1f} "
                    f"last_d_loss={last_d_loss} last_g_loss={last_g_loss} last_c_loss={last_c_loss}"
                )
                if FAST_DEBUG:
                    break

            seen_classes = list(range(end_cls))
            current_task_ranges = task_ranges[: task_id + 1]
            task_accs, y_true_seen, y_pred_seen = evaluate_seen(
                torch_mod,
                np_mod,
                metrics_mod,
                classifier,
                scaler,
                x_test,
                y_test,
                seen_classes,
                current_task_ranges,
                device,
                BATCH_SIZE,
            )
            padded = task_accs + [None] * (11 - len(task_accs))
            accuracy_matrix.append(padded)
            final_true = y_true_seen
            final_pred = y_pred_seen
            valid_task_accs = [value for value in task_accs if value is not None]
            seen_avg = float(np_mod.mean(valid_task_accs)) if valid_task_accs else None
            if TQDM is not None and hasattr(task_iter, "set_postfix"):
                task_iter.set_postfix(task=f"{task_id + 1}/11", seen_avg=seen_avg, refresh=True)
            emit(
                f"TASK_EVAL dataset={dataset} seed={seed} task={task_id + 1}/11 "
                f"seen_avg_accuracy={seen_avg} task_accuracies={task_accs}"
            )

            logits_real = compute_class_feature_means(torch_mod, np_mod, classifier, scaler, x_train, y_train, seen_classes, device, BATCH_SIZE)
            past_generator = deepcopy(generator).to(device).eval()
            past_classifier = deepcopy(classifier).to(device).eval()

        metrics = metric_bundle(np_mod, metrics_mod, accuracy_matrix, final_true, final_pred)
        full_task_completed = len(accuracy_matrix) == 11 and not FAST_DEBUG
        full_data_completed = not FAST_DEBUG and not ALLOW_SUBSAMPLING
        param_mem = model_param_memory(generator, discriminator, classifier)
        memory.update(param_mem)
        memory["copied_past_generator_classifier_memory_mb"] = param_mem["generator_parameter_memory_mb"] + param_mem["classifier_parameter_memory_mb"]
        memory["generated_replay_tensor_memory_mb"] = generated_replay_mb
        if torch_mod.cuda.is_available():
            memory["peak_gpu_memory_mb"] = float(torch_mod.cuda.max_memory_allocated() / (1024 * 1024))
        memory["memory_mb_total_method_side_estimate"] = float(
            param_mem["generator_parameter_memory_mb"]
            + param_mem["discriminator_parameter_memory_mb"]
            + param_mem["classifier_parameter_memory_mb"]
            + memory["copied_past_generator_classifier_memory_mb"]
            + generated_replay_mb
        )
        status = "passed" if real_training_executed and metrics["final_taskwise_accuracy"] is not None else "failed"
        if not full_task_completed:
            warnings.append("full task stream did not complete; result is not paper-usable")
        if not full_data_completed:
            warnings.append("full data protocol did not complete; result is not paper-usable")
    except Exception:
        errors.append(traceback.format_exc())
        status = "failed"
    finally:
        if "torch_mod" in locals() and not isinstance(torch_mod, Exception) and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

    runtime_seconds = time.time() - start_time
    emit(
        f"RUN_FINISH dataset={dataset} seed={seed} status={status} "
        f"full_task_completed={full_task_completed} final_taskwise_accuracy={metrics['final_taskwise_accuracy']} "
        f"runtime_seconds={runtime_seconds:.1f}"
    )
    result = {
        "dataset": dataset,
        "seed": seed,
        "method": "MalCL",
        "mode": "final_protocol_matched_malcl_baseline",
        "official_malcl_commit": OFFICIAL_COMMIT,
        "real_training_executed": bool(real_training_executed),
        "synthetic_fallback_used": False,
        "paper_claim_allowed": False,
        "baseline_candidate": True,
        "final_paper_usable": False,
        "class_order_matched": order == build_class_order(np_mod, seed),
        "paired_comparison_ready": order == build_class_order(np_mod, seed),
        "feature_dim": feature_dim,
        "num_classes": 100,
        "num_tasks": 11,
        "task_sizes": [50] + [5] * 10,
        "epochs": MALCL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "full_task_completed": bool(full_task_completed),
        "full_data_completed": bool(full_data_completed),
        "per_task_accuracy": metrics["per_task_accuracy"],
        "final_task_accuracies": metrics["final_task_accuracies"],
        "final_taskwise_accuracy": metrics["final_taskwise_accuracy"],
        "mean_seen_accuracy": metrics["mean_seen_accuracy"],
        "forgetting": metrics["forgetting"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "old_class_accuracy": metrics["old_class_accuracy"],
        "new_class_accuracy": metrics["new_class_accuracy"],
        "runtime_seconds": runtime_seconds,
        "memory_mb_total_method_side_estimate": memory["memory_mb_total_method_side_estimate"],
        "peak_gpu_memory_mb": memory["peak_gpu_memory_mb"],
        "warnings": warnings,
        "errors": errors,
        "status": status,
        "completed_at": utc_now(),
    }

    write_json(full_result_path, result)
    write_json(output_dir / "memory_accounting.json", memory)
    write_json(
        output_dir / "class_order.json",
        {
            "seed": seed,
            "rule": "numpy.random.RandomState(seed).permutation(100)",
            "class_order": order,
            "tasks_original_labels": tasks_original,
            "task_sizes": [50] + [5] * 10,
            "class_order_matched": result["class_order_matched"],
            "paired_comparison_ready": result["paired_comparison_ready"],
        },
    )
    write_json(output_dir / "run_config.json", vars(config))
    with (output_dir / "training_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        progress_fields = [
            "timestamp",
            "dataset",
            "seed",
            "task_id",
            "task_number",
            "epoch",
            "batch",
            "total_batches",
            "d_loss",
            "g_loss",
            "c_loss",
            "generated_replay_mb",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=progress_fields)
        writer.writeheader()
        for row in progress_rows:
            writer.writerow({field: row.get(field, "") for field in progress_fields})
    with (output_dir / "per_task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "seen_average_accuracy", "final_task_accuracy"])
        writer.writeheader()
        final_accs = metrics.get("final_task_accuracies") or []
        for task_id in range(11):
            seen_value = metrics["per_task_accuracy"][task_id] if task_id < len(metrics["per_task_accuracy"]) else ""
            final_value = final_accs[task_id] if task_id < len(final_accs) else ""
            writer.writerow({"task_id": task_id, "seen_average_accuracy": seen_value, "final_task_accuracy": final_value})
    write_text(output_dir / "run_log.txt", "\n".join(["progress:", *progress_messages, "", "warnings:", *warnings, "errors:", *errors]) + "\n")
    return result


def completed_run(data: dict[str, object]) -> bool:
    required_metrics = ["final_taskwise_accuracy", "mean_seen_accuracy", "forgetting", "macro_f1", "weighted_f1", "balanced_accuracy"]
    return (
        data.get("status") == "passed"
        and data.get("real_training_executed") is True
        and data.get("synthetic_fallback_used") is False
        and data.get("full_task_completed") is True
        and data.get("full_data_completed") is True
        and data.get("class_order_matched") is True
        and all(data.get(metric) is not None for metric in required_metrics)
        and float(data.get("memory_mb_total_method_side_estimate") or 0.0) > 0.0
    )


def summarize_runs(dirs: dict[str, Path], validation_report: dict[str, object]) -> dict[str, object]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = dirs["raw_outputs"] / dataset.replace("-", "_") / f"seed_{seed}" / "full-result.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = {"dataset": dataset, "seed": seed, "status": "missing"}
            data["completed_for_decision"] = completed_run(data)
            rows.append(data)

    seed_level_path = dirs["tables"] / "final_malcl_seed_level_results.csv"
    fields = [
        "dataset",
        "seed",
        "status",
        "real_training_executed",
        "full_task_completed",
        "full_data_completed",
        "class_order_matched",
        "paired_comparison_ready",
        "final_taskwise_accuracy",
        "mean_seen_accuracy",
        "forgetting",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "memory_mb_total_method_side_estimate",
        "runtime_seconds",
        "completed_for_decision",
    ]
    with seed_level_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    def mean_std(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        mean_value = sum(values) / len(values)
        if len(values) == 1:
            return mean_value, 0.0
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        return mean_value, math.sqrt(variance)

    summary_rows = []
    metric_names = [
        "final_taskwise_accuracy",
        "mean_seen_accuracy",
        "forgetting",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "memory_mb_total_method_side_estimate",
        "runtime_seconds",
    ]
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset and row.get("completed_for_decision")]
        summary = {"dataset": dataset, "completed_runs": len(dataset_rows), "expected_runs": 10}
        for metric in metric_names:
            values = [float(row[metric]) for row in dataset_rows if row.get(metric) is not None]
            mean_value, std_value = mean_std(values)
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
            summary[f"{metric}_mean_pm_std"] = (
                f"{summary[f'{metric}_mean']:.6f} +/- {summary[f'{metric}_std']:.6f}"
                if summary[f"{metric}_mean"] is not None and summary[f"{metric}_std"] is not None
                else ""
            )
        summary_rows.append(summary)

    dataset_summary_path = dirs["tables"] / "final_malcl_dataset_summary.csv"
    summary_fields = ["dataset", "completed_runs", "expected_runs"]
    for metric in metric_names:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_pm_std"])
    with dataset_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in summary_fields})

    paired_path = dirs["tables"] / "final_malcl_paired_ready_table.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "paired_comparison_ready", "completed_for_decision", "reason"])
        writer.writeheader()
        for row in rows:
            reason = "ready" if row.get("paired_comparison_ready") and row.get("completed_for_decision") else "missing/incomplete/invalid"
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "paired_comparison_ready": row.get("paired_comparison_ready", False),
                    "completed_for_decision": row.get("completed_for_decision", False),
                    "reason": reason,
                }
            )

    memory_path = dirs["memory"] / "final_malcl_memory_summary.csv"
    with memory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "memory_mb_total_method_side_estimate", "peak_gpu_memory_mb", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "memory_mb_total_method_side_estimate": row.get("memory_mb_total_method_side_estimate", ""),
                    "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb", ""),
                    "status": row.get("status", ""),
                }
            )

    completed_count = sum(1 for row in rows if row.get("completed_for_decision"))
    validation_passed = bool(validation_report.get("passed"))
    any_failed = any(row.get("status") == "failed" for row in rows)
    if validation_passed and completed_count == 20 and not any_failed:
        decision = "MALCL_BASELINE_COMPLETE_READY_FOR_REVIEW"
    elif validation_passed and completed_count > 0 and completed_count < 20 and not any_failed:
        decision = "MALCL_BASELINE_PARTIAL_NEEDS_RESUME"
    elif validation_passed and completed_count == 0 and not any_failed:
        decision = "MALCL_BASELINE_PARTIAL_NEEDS_RESUME"
    else:
        decision = "MALCL_BASELINE_NOT_VALID_PATCH_REQUIRED"

    aggregate = {
        "created_at": utc_now(),
        "decision": decision,
        "completed_runs": completed_count,
        "expected_runs": 20,
        "validation_passed": validation_passed,
        "any_failed_runs": any_failed,
        "seed_level_table": str(seed_level_path),
        "dataset_summary_table": str(dataset_summary_path),
        "paired_ready_table": str(paired_path),
        "memory_summary_table": str(memory_path),
        "dataset_summaries": summary_rows,
        "runs": rows,
    }
    return aggregate


def write_reports(dirs: dict[str, Path], validation_report: dict[str, object], aggregate: dict[str, object]) -> None:
    reports = dirs["reports"]
    decision = aggregate["decision"]
    completed = aggregate["completed_runs"]
    expected = aggregate["expected_runs"]

    write_json(dirs["validation"] / "final_malcl_validation_report.json", validation_report)
    write_text(
        reports / "FINAL_MALCL_BASELINE_VALIDATION_REPORT.md",
        f"""# Final MalCL Baseline Validation Report

- Environment validation passed: {validation_report['environment'].get('passed')}
- Dataset validation passed: {validation_report['datasets'].get('passed')}
- Official MalCL validation passed: {validation_report['official_malcl'].get('passed')}
- Topology validation passed: {validation_report['topology'].get('passed')}
- Class-order validation passed: {validation_report['class_orders'].get('passed')}
- Config validation passed: {validation_report['config'].get('passed')}

Overall validation passed: {validation_report.get('passed')}
""",
    )
    write_text(
        reports / "FINAL_MALCL_BASELINE_RESULT_REPORT.md",
        f"""# Final MalCL Baseline Result Report

Decision: `{decision}`

Completed valid runs: {completed}/{expected}

Seed-level table: `tables/final_malcl_seed_level_results.csv`

Dataset summary table: `tables/final_malcl_dataset_summary.csv`
""",
    )
    write_text(
        reports / "FINAL_MALCL_BASELINE_MEMORY_REPORT.md",
        """# Final MalCL Baseline Memory Report

Memory accounting includes live generator, discriminator, classifier, copied past generator/classifier, generated replay tensors, and PyTorch peak GPU allocation when CUDA is available.

See `memory/final_malcl_memory_summary.csv`.
""",
    )
    seed42 = validation_report["class_orders"]["seeds"].get("42", {})
    write_text(
        reports / "FINAL_MALCL_BASELINE_CLASS_ORDER_REPORT.md",
        f"""# Final MalCL Baseline Class-Order Report

Rule: `numpy.random.RandomState(seed).permutation(100)`

Seed 42 first 10 classes: `{seed42.get('first10')}`

Expected seed 42 first 10 classes: `{EXPECTED_SEED42_FIRST10}`

All seeds valid: {validation_report['class_orders'].get('passed')}
""",
    )
    write_text(
        reports / "FINAL_MALCL_BASELINE_FAITHFULNESS_REPORT.md",
        f"""# Final MalCL Baseline Faithfulness Report

Official commit expected: `{OFFICIAL_COMMIT}`

Official commit matched: {validation_report['official_malcl'].get('commit_matches')}

The patched adapter preserves the official generator role, classifier role, discriminator convolutional front-end, feature-matching path, and `z_dim=62`. It parameterizes `feature_dim` so that EMBER and AZ-Class can be evaluated under the same task stream.

The adapted implementation parameterizes `feature_dim` for EMBER and AZ-Class while retaining the official feature-length-dependent flattened discriminator topology. The first discriminator fully connected layer consumes `256 * feature_dim` inputs and produces 1024 outputs. The baseline remains a protocol-matched adapted MalCL baseline rather than an exact official reproduction because the shared continual-learning protocol, feature-dimension handling, classifier expansion, and consolidated runner contain documented adaptations.

Other documented implementation details include a refreshed classifier optimizer after output-layer expansion and explicit reporting of any debug/subsampling switches.

Topology validation status: `{validation_report['topology'].get('status')}`
""",
    )
    write_text(
        reports / "FINAL_MALCL_BASELINE_CLAIM_SAFETY_REPORT.md",
        f"""# Final MalCL Baseline Claim Safety Report

Decision: `{decision}`

Safe only if full validation passes:

- We reproduced a protocol-matched MalCL baseline candidate under the AHR-MalCL evaluation stream.
- Results are subject to documented implementation deviations.

Always unsafe:

- AHR beats MalCL unless later statistical comparison confirms it.
- MalCL is fully reproduced exactly if there are deviations.
- One seed proves superiority.
- Memory fairness is established without matching accounting.

Individual `full-result.json` files intentionally keep `paper_claim_allowed=false`; aggregate validation controls later paper use.
""",
    )
    write_text(
        reports / "FINAL_MALCL_BASELINE_NEXT_DECISION.md",
        f"""# Final MalCL Baseline Next Decision

`{decision}`

Completed runs: {completed}/{expected}

Use `MALCL_BASELINE_PARTIAL_NEEDS_RESUME` by rerunning the same Colab cell if Colab disconnects before all 20 runs complete.
""",
    )


def create_zip(output_root: Path, destination_zip: Path | None) -> Path:
    zip_path = output_root / "final_malcl_baseline_results_colab.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_root.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(output_root))
    if destination_zip is not None:
        destination_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, destination_zip)
        return destination_zip
    return zip_path


def initial_validation(output_root: Path, dirs: dict[str, Path], mods: dict[str, object]) -> dict[str, object]:
    emit_status(f"VALIDATION_START output_root={output_root}")
    emit_status("VALIDATION_ENVIRONMENT_START")
    env = validate_environment(mods)
    emit_status(
        f"VALIDATION_ENVIRONMENT_DONE passed={env.get('passed')} "
        f"cuda={env.get('cuda_available')} gpu={env.get('gpu_name')}"
    )
    np_mod = mods["numpy"]
    dataset_report = {"passed": False, "errors": ["numpy unavailable"], "datasets": {}}
    class_report = {"passed": False, "errors": ["numpy unavailable"], "seeds": {}}
    if not isinstance(np_mod, Exception):
        emit_status("VALIDATION_DATASETS_START")
        dataset_report = validate_datasets(np_mod)
        emit_status(f"VALIDATION_DATASETS_DONE passed={dataset_report.get('passed')}")
        emit_status("VALIDATION_CLASS_ORDERS_START")
        class_report = validate_class_orders(np_mod)
        write_json(dirs["validation"] / "class_orders_42_51.json", class_report)
        emit_status(f"VALIDATION_CLASS_ORDERS_DONE passed={class_report.get('passed')}")
    else:
        emit_status(f"VALIDATION_NUMPY_UNAVAILABLE error={np_mod!r}")
    emit_status("VALIDATION_OFFICIAL_MALCL_START")
    official_report = validate_official_malcl()
    emit_status(
        f"VALIDATION_OFFICIAL_MALCL_DONE passed={official_report.get('passed')} "
        f"commit_matches={official_report.get('commit_matches')}"
    )
    emit_status("VALIDATION_TOPOLOGY_START")
    topology_report = run_topology_validation(feature_dims=(2381, 2439), batch_size=2)
    emit_status(
        f"VALIDATION_TOPOLOGY_DONE passed={topology_report.get('passed')} "
        f"status={topology_report.get('status')}"
    )
    emit_status("VALIDATION_CONFIG_WRITE_START")
    cfg = final_config(class_report, output_root)
    write_json(dirs["configs"] / "final_malcl_config.json", cfg)
    config_report = {
        "passed": True,
        "config_path": str(dirs["configs"] / "final_malcl_config.json"),
        "fast_debug": FAST_DEBUG,
        "allow_subsampling": ALLOW_SUBSAMPLING,
        "epochs": MALCL_EPOCHS,
        "batch_size": BATCH_SIZE,
    }
    validation = {
        "created_at": utc_now(),
        "environment": env,
        "datasets": dataset_report,
        "official_malcl": official_report,
        "topology": topology_report,
        "class_orders": class_report,
        "config": config_report,
    }
    validation["passed"] = all(
        [
            env.get("passed"),
            dataset_report.get("passed"),
            official_report.get("passed"),
            topology_report.get("passed"),
            class_report.get("passed"),
            config_report.get("passed"),
        ]
    )
    emit_status(f"VALIDATION_DONE passed={validation.get('passed')}")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final protocol-matched MalCL baseline in Colab.")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR), help="Output root. Defaults to this final package folder.")
    parser.add_argument("--drive-output-zip", default=os.environ.get("AHR_OUTPUT_ZIP", "final_malcl_baseline_results_colab.zip"))
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    dirs = ensure_dirs(output_root)
    emit_status("RUNNER_IMPORTS_START")
    mods = optional_imports()
    emit_status("RUNNER_IMPORTS_DONE")
    validation_report = initial_validation(output_root, dirs, mods)

    if not validation_report["passed"]:
        aggregate = summarize_runs(dirs, validation_report)
        write_reports(dirs, validation_report, aggregate)
        write_json(output_root / "final_malcl_baseline_summary.json", aggregate)
        zip_dest = Path(args.drive_output_zip) if args.drive_output_zip else None
        zip_path = create_zip(output_root, zip_dest)
        print("FINAL_MALCL_BASELINE_COLAB_RUN_COMPLETE")
        print(f"decision: {aggregate['decision']}")
        print("completed_runs: 0/20")
        print(f"output_zip: {zip_path}")
        return 2

    np_mod = mods["numpy"]
    torch_mod = mods["torch"]
    preprocessing_mod = mods["sklearn.preprocessing"]
    metrics_mod = mods["sklearn.metrics"]

    run_jobs = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    emit_status(f"TRAINING_LOOP_START jobs={len(run_jobs)} epochs={MALCL_EPOCHS} batch_size={BATCH_SIZE}")
    run_iter = run_jobs
    if TQDM is not None:
        run_iter = TQDM(
            run_jobs,
            total=len(run_jobs),
            desc="Final MalCL runs",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=TQDM_MININTERVAL,
        )
    completed_so_far = 0
    for dataset, seed in run_iter:
        run_dir = dirs["raw_outputs"] / dataset.replace("-", "_") / f"seed_{seed}"
        if TQDM is not None and hasattr(run_iter, "set_postfix"):
            run_iter.set_postfix(dataset=dataset, seed=seed, completed=f"{completed_so_far}/20", refresh=True)
        print(f"RUN_START dataset={dataset} seed={seed} output={run_dir}", flush=True)
        result = run_dataset_seed(np_mod, torch_mod, preprocessing_mod, metrics_mod, dataset, seed, run_dir)
        if completed_run(result):
            completed_so_far += 1
        if TQDM is not None and hasattr(run_iter, "set_postfix"):
            run_iter.set_postfix(dataset=dataset, seed=seed, completed=f"{completed_so_far}/20", status=result.get("status"), refresh=True)
        print(f"RUN_DONE dataset={dataset} seed={seed} status={result.get('status')} completed={completed_run(result)}", flush=True)

    aggregate = summarize_runs(dirs, validation_report)
    write_reports(dirs, validation_report, aggregate)
    write_json(output_root / "final_malcl_baseline_summary.json", aggregate)
    zip_dest = Path(args.drive_output_zip) if args.drive_output_zip else None
    zip_path = create_zip(output_root, zip_dest)

    print("FINAL_MALCL_BASELINE_COLAB_RUN_COMPLETE")
    print(f"decision: {aggregate['decision']}")
    print(f"completed_runs: {aggregate['completed_runs']}/20")
    print(f"output_zip: {zip_path}")
    return 0 if aggregate["decision"] in {"MALCL_BASELINE_COMPLETE_READY_FOR_REVIEW", "MALCL_BASELINE_PARTIAL_NEEDS_RESUME"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
