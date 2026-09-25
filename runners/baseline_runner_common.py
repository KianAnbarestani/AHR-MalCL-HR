#!/usr/bin/env python3
"""Shared Colab-compatible training code for external baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


VERSION = "v1 - shared Colab baseline runner core"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_suite import data_adapter

REPORT_DIR = PROJECT_ROOT / "paper_ready_baselines"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "result_external_baselines"


class RunningStandardScaler:
    """No-sklearn incremental standard scaler matching the notebook convention."""

    def __init__(self) -> None:
        self.n = 0
        self.mean_: np.ndarray | None = None
        self.m2_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def partial_fit(self, x: np.ndarray) -> "RunningStandardScaler":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("X must be 2-D")
        if self.mean_ is None:
            self.mean_ = np.zeros(x.shape[1], dtype=np.float64)
            self.m2_ = np.zeros(x.shape[1], dtype=np.float64)
        batch_n = int(x.shape[0])
        if batch_n == 0:
            return self
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_m2 = batch_var * batch_n
        if self.n == 0:
            self.mean_ = batch_mean
            self.m2_ = batch_m2
            self.n = batch_n
            return self
        assert self.m2_ is not None
        delta = batch_mean - self.mean_
        total = self.n + batch_n
        self.mean_ = self.mean_ + delta * batch_n / total
        self.m2_ = self.m2_ + batch_m2 + (delta**2) * self.n * batch_n / total
        self.n = total
        return self

    def finalize(self) -> "RunningStandardScaler":
        if self.mean_ is None or self.m2_ is None:
            raise ValueError("Cannot finalize scaler before fitting data.")
        var = self.m2_ / max(self.n, 1)
        scale = np.sqrt(var)
        scale[scale < 1e-8] = 1.0
        self.mean_ = self.mean_.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been finalized.")
        return ((x.astype(np.float32) - self.mean_) / self.scale_).astype(np.float32)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def import_status() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def import_torch() -> Any:
    import torch

    return torch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        torch = import_torch()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
    except Exception:
        pass


def resolve_seeds(args: argparse.Namespace, config: dict[str, Any]) -> list[int]:
    if args.single_seed is not None:
        return [int(args.single_seed)]
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(seed) for seed in config.get("seeds", [42, 43, 44, 45, 46])]


def baseline_code(kind: str) -> str:
    return {"derpp": "DERPP", "er": "ER", "joint": "Joint"}[kind]


def dataset_dir_name(config: dict[str, Any]) -> str:
    canonical = data_adapter.canonical_dataset_name(config.get("dataset", ""))
    return "EMBER" if canonical == "ember" else "AZ-Class"


def k_dir_name(config: dict[str, Any]) -> str | None:
    k = config.get("memory_budget", {}).get("k_per_class")
    return f"K={k}" if k is not None else None


def resolve_output_base(config: dict[str, Any], kind: str, output_root: str | None) -> Path:
    if output_root:
        root = Path(output_root).expanduser()
        parts = [baseline_code(kind), dataset_dir_name(config)]
        k_dir = k_dir_name(config)
        if k_dir:
            parts.append(k_dir)
        return root.joinpath(*parts)
    raw = config.get("output_directory")
    if raw:
        return (PROJECT_ROOT / str(raw)).resolve() if not Path(str(raw)).is_absolute() else Path(str(raw))
    parts = [baseline_code(kind), dataset_dir_name(config)]
    k_dir = k_dir_name(config)
    if k_dir:
        parts.append(k_dir)
    return DEFAULT_OUTPUT_ROOT.joinpath(*parts)


def output_path_safety(path: Path) -> tuple[bool, str]:
    resolved = path.resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "result").resolve())
        return False, f"Unsafe output path under trusted result/: {resolved}"
    except ValueError:
        pass
    return True, str(resolved)


def device_status(device: str) -> dict[str, Any]:
    status: dict[str, Any] = {"requested": device}
    if not str(device).startswith("cuda"):
        status.update({"available": True, "cuda_available": False, "device_name": "cpu"})
        return status
    try:
        torch = import_torch()
    except Exception as exc:  # noqa: BLE001
        status.update({"available": False, "reason": f"torch import failed: {exc}"})
        return status
    ok = torch.cuda.is_available()
    status.update(
        {
            "available": bool(ok),
            "cuda_available": bool(ok),
            "device_name": torch.cuda.get_device_name(0) if ok else "",
        }
    )
    return status


def fit_scaler_on_view(scaler: RunningStandardScaler, view: Any, chunk: int = 20000) -> RunningStandardScaler:
    for start in range(0, len(view), chunk):
        stop = min(start + chunk, len(view))
        scaler.partial_fit(np.asarray(view[start:stop], dtype=np.float32))
    return scaler.finalize()


def mlp_model(torch: Any, input_dim: int, output_dim: int, hidden: list[int], dropout: float) -> Any:
    layers: list[Any] = []
    prev = int(input_dim)
    for width in hidden:
        layers.append(torch.nn.Linear(prev, int(width)))
        layers.append(torch.nn.LayerNorm(int(width)))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(float(dropout)))
        prev = int(width)
    layers.append(torch.nn.Linear(prev, int(output_dim)))
    return torch.nn.Sequential(*layers)


class TaskTensorDataset:
    def __init__(self, task: dict[str, Any], split: str, scaler: RunningStandardScaler, label_mapping: dict[int, int]) -> None:
        self.x_view = task[f"X_{split}"]
        self.y_orig = np.asarray(task[f"y_{split}"], dtype=np.int64)
        self.scaler = scaler
        self.label_mapping = {int(k): int(v) for k, v in label_mapping.items()}

    def __len__(self) -> int:
        return int(len(self.y_orig))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        x = self.scaler.transform(np.asarray(self.x_view[idx], dtype=np.float32)[None, :])[0]
        y = self.label_mapping[int(self.y_orig[idx])]
        return x, int(y)


class BufferDataset:
    def __init__(self, features: np.ndarray, labels: np.ndarray, logits: np.ndarray | None = None) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.logits = None if logits is None else np.asarray(logits, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int) -> Any:
        if self.logits is None:
            return self.features[idx], int(self.labels[idx])
        return self.features[idx], int(self.labels[idx]), self.logits[idx]


def selected_seen_exemplars(
    stream: dict[str, Any],
    seen_classes: list[int],
    k_per_class: int,
    scaler: RunningStandardScaler,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    features: list[np.ndarray] = []
    labels: list[int] = []
    label_mapping = {int(k): int(v) for k, v in stream["label_mapping"].items()}
    for cls in seen_classes:
        cls_features = []
        for task in stream["tasks"]:
            y = np.asarray(task["y_train"], dtype=np.int64)
            positions = np.flatnonzero(y == int(cls))
            if positions.size:
                cls_features.append(np.asarray(task["X_train"][positions], dtype=np.float32))
        if not cls_features:
            continue
        raw = np.concatenate(cls_features, axis=0)
        count = min(int(k_per_class), raw.shape[0])
        chosen = rng.choice(np.arange(raw.shape[0]), size=count, replace=False)
        transformed = scaler.transform(raw[chosen])
        features.append(transformed)
        labels.extend([label_mapping[int(cls)]] * count)
    if not features:
        return np.empty((0, int(stream["feature_dim"])), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(features, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def logits_for_features(model: Any, features: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    torch = import_torch()
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def mask_logits(logits: Any, allowed_ids: list[int]) -> Any:
    torch = import_torch()
    mask = torch.full_like(logits, -1e9)
    mask[:, allowed_ids] = logits[:, allowed_ids]
    return mask


def evaluate_tasks(model: Any, stream: dict[str, Any], scaler: RunningStandardScaler, task_count: int, device: str) -> dict[str, Any]:
    torch = import_torch()
    label_mapping = {int(k): int(v) for k, v in stream["label_mapping"].items()}
    seen_classes = [cls for task in stream["tasks"][:task_count] for cls in task["class_ids"]]
    seen_ids = [label_mapping[int(cls)] for cls in seen_classes]
    task_accs = []
    all_true: list[int] = []
    all_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for task in stream["tasks"][:task_count]:
            dataset = TaskTensorDataset(task, "test", scaler, label_mapping)
            if len(dataset) == 0:
                task_accs.append(float("nan"))
                continue
            loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device)
                logits = mask_logits(model(xb), seen_ids)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
                all_true.extend([int(x) for x in yb.detach().cpu().numpy().tolist()])
                all_pred.extend([int(x) for x in pred.detach().cpu().numpy().tolist()])
            task_accs.append(correct / total if total else float("nan"))
    valid_accs = [acc for acc in task_accs if not math.isnan(acc)]
    return {
        "task_accuracies": task_accs,
        "seen_accuracy": mean(valid_accs) if valid_accs else float("nan"),
        "y_true": all_true,
        "y_pred": all_pred,
        "seen_ids": seen_ids,
    }


def classification_outputs(y_true: list[int], y_pred: list[int], labels: list[int]) -> tuple[list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        report_rows = []
        for label, values in report_dict.items():
            if isinstance(values, dict):
                row = {"label": label}
                row.update(values)
                report_rows.append(row)
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    except Exception:
        report_rows = []
        matrix = [[0 for _ in labels] for _ in labels]
        index = {label: pos for pos, label in enumerate(labels)}
        for yt, yp in zip(y_true, y_pred):
            if yt in index and yp in index:
                matrix[index[yt]][index[yp]] += 1
    per_class = []
    matrix_arr = np.asarray(matrix, dtype=np.int64)
    for pos, label in enumerate(labels):
        total = int(matrix_arr[pos].sum()) if matrix_arr.size else 0
        correct = int(matrix_arr[pos, pos]) if matrix_arr.size else 0
        per_class.append({"class_id": label, "accuracy": correct / total if total else ""})
    return report_rows, matrix, per_class


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_matrix_csv(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def memory_mb_model(model: Any) -> float:
    return sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 * 1024)


def train_one_seed(kind: str, config: dict[str, Any], seed: int, device: str, output_base: Path) -> dict[str, Any]:
    torch = import_torch()
    set_seed(seed)
    stream = data_adapter.load_class_incremental_stream(config["dataset"], int(seed), config=config)
    model_cfg = config.get("classifier_backbone", {})
    model = mlp_model(
        torch,
        input_dim=int(stream["feature_dim"]),
        output_dim=int(stream["num_classes"]),
        hidden=[int(x) for x in model_cfg.get("hidden_layers", [1024, 512, 256])],
        dropout=float(model_cfg.get("dropout", 0.25)),
    ).to(device)
    opt_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 0.001)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0001)),
    )
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 256))
    k_per_class = int(config.get("memory_budget", {}).get("k_per_class") or 0)
    alpha = float(config.get("derpp", {}).get("alpha_logit_mse", 0.5) if str(config.get("derpp", {}).get("alpha_logit_mse", "")).replace(".", "", 1).isdigit() else 0.5)
    beta = float(config.get("derpp", {}).get("beta_replay_ce", 1.0) if str(config.get("derpp", {}).get("beta_replay_ce", "")).replace(".", "", 1).isdigit() else 1.0)
    criterion = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    scaler = RunningStandardScaler()
    seen_classes: list[int] = []
    accs_seen: list[float] = []
    acc_matrix: list[list[float]] = []
    buffer_features = np.empty((0, int(stream["feature_dim"])), dtype=np.float32)
    buffer_labels = np.empty((0,), dtype=np.int64)
    buffer_logits: np.ndarray | None = None
    start_time = time.time()

    if kind == "joint":
        for task in stream["tasks"]:
            fit_scaler_on_view(scaler, task["X_train"])
        seen_classes = list(stream["class_order"])
        datasets = [TaskTensorDataset(task, "train", scaler, stream["label_mapping"]) for task in stream["tasks"]]
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for _epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        eval_out = evaluate_tasks(model, stream, scaler, len(stream["tasks"]), device)
        accs_seen = [float(eval_out["seen_accuracy"])]
        acc_matrix = [eval_out["task_accuracies"]]
    else:
        for task_index, task in enumerate(stream["tasks"]):
            fit_scaler_on_view(scaler, task["X_train"])
            seen_classes.extend(int(cls) for cls in task["class_ids"])
            current_dataset = TaskTensorDataset(task, "train", scaler, stream["label_mapping"])
            current_loader = torch.utils.data.DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
            replay_dataset = None
            if len(buffer_labels):
                replay_dataset = BufferDataset(buffer_features, buffer_labels, buffer_logits if kind == "derpp" else None)
            for _epoch in range(epochs):
                model.train()
                replay_iter = iter(torch.utils.data.DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)) if replay_dataset else None
                for xb, yb in current_loader:
                    xb = xb.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    if replay_iter is not None:
                        try:
                            replay_batch = next(replay_iter)
                        except StopIteration:
                            replay_iter = iter(torch.utils.data.DataLoader(replay_dataset, batch_size=batch_size, shuffle=True))
                            replay_batch = next(replay_iter)
                        if kind == "derpp":
                            rb_x, rb_y, rb_logits = replay_batch
                            rb_x = rb_x.to(device=device, dtype=torch.float32)
                            rb_y = rb_y.to(device=device)
                            rb_logits = rb_logits.to(device=device, dtype=torch.float32)
                            out = model(rb_x)
                            loss = loss + beta * criterion(out, rb_y) + alpha * mse(out, rb_logits)
                        else:
                            rb_x, rb_y = replay_batch
                            rb_x = rb_x.to(device=device, dtype=torch.float32)
                            rb_y = rb_y.to(device=device)
                            loss = loss + criterion(model(rb_x), rb_y)
                    loss.backward()
                    optimizer.step()
            eval_out = evaluate_tasks(model, stream, scaler, task_index + 1, device)
            accs_seen.append(float(eval_out["seen_accuracy"]))
            acc_matrix.append([float(x) if not math.isnan(x) else None for x in eval_out["task_accuracies"]])
            if k_per_class:
                buffer_features, buffer_labels = selected_seen_exemplars(stream, seen_classes, k_per_class, scaler, seed + task_index)
                buffer_logits = logits_for_features(model, buffer_features, device, batch_size) if kind == "derpp" else None

    final_eval = evaluate_tasks(model, stream, scaler, len(stream["tasks"]), device)
    labels = list(range(int(stream["num_classes"])))
    report_rows, confusion, per_class = classification_outputs(final_eval["y_true"], final_eval["y_pred"], labels)
    seed_dir = output_base / f"seed={int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(seed_dir / "final_classification_report.csv", report_rows)
    write_matrix_csv(seed_dir / "final_confusion_matrix.csv", confusion)
    write_dict_csv(seed_dir / "final_per_class_accuracy.csv", per_class)
    memory_model = memory_mb_model(model)
    memory_features = float(buffer_features.nbytes) / (1024 * 1024)
    memory_labels = float(buffer_labels.nbytes) / (1024 * 1024)
    memory_logits = float(buffer_logits.nbytes) / (1024 * 1024) if buffer_logits is not None else 0.0
    final_task_accs = [acc for acc in final_eval["task_accuracies"] if not math.isnan(acc)]
    result = {
        "method": baseline_code(kind),
        "dataset": stream["dataset"],
        "seed": int(seed),
        "k": k_per_class if kind != "joint" else None,
        "mean_acc_seen": float(mean(accs_seen)) if accs_seen else None,
        "final_taskwise_average_accuracy": float(mean(final_task_accs)) if final_task_accs else None,
        "forgetting": float(max(accs_seen) - accs_seen[-1]) if len(accs_seen) > 1 else 0.0,
        "accs_seen_per_task": accs_seen,
        "acc_matrix_taskwise": acc_matrix,
        "memory_MB": memory_model + memory_features + memory_labels + memory_logits,
        "model_memory_MB": memory_model,
        "exemplar_feature_memory_MB": memory_features,
        "label_memory_MB": memory_labels,
        "stored_logits_memory_MB": memory_logits,
        "offline_oracle_upper_bound": bool(kind == "joint"),
        "full_historical_data_available": bool(kind == "joint"),
        "elapsed_minutes": (time.time() - start_time) / 60.0,
        "config": config,
        "stream_summary": data_adapter.summarize_stream(stream),
    }
    (seed_dir / "full-result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def aggregate_results(results: list[dict[str, Any]], output_base: Path, kind: str, config: dict[str, Any]) -> None:
    metrics = ["mean_acc_seen", "final_taskwise_average_accuracy", "forgetting", "memory_MB", "model_memory_MB"]
    aggregate: dict[str, Any] = {
        "method": baseline_code(kind),
        "dataset": data_adapter.canonical_dataset_name(config["dataset"]),
        "k": config.get("memory_budget", {}).get("k_per_class"),
        "seeds": [row["seed"] for row in results],
        "runs": results,
    }
    for metric in metrics:
        values = [float(row[metric]) for row in results if row.get(metric) is not None]
        aggregate[f"{metric}_mean"] = mean(values) if values else None
        aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    write_dict_csv(output_base / "aggregate.csv", [{k: v for k, v in aggregate.items() if k != "runs"}])


def dry_run(kind: str, args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> int:
    failures = []
    imports = import_status()
    missing = [key for key, ok in imports.items() if not ok]
    if missing:
        failures.append("Missing required imports: " + ", ".join(missing))
    device = device_status(args.device)
    if not device.get("available"):
        failures.append("Requested device unavailable: " + str(device))
    output_base = resolve_output_base(config, kind, args.output_root)
    path_ok, path_msg = output_path_safety(output_base)
    if not path_ok:
        failures.append(path_msg)
    stream_summary = None
    try:
        stream = data_adapter.load_class_incremental_stream(config["dataset"], resolve_seeds(args, config)[0], config=config)
        stream_summary = data_adapter.summarize_stream(stream)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Data stream load failed: {type(exc).__name__}: {exc}")
        data_adapter.write_todo(config.get("dataset", ""), f"{type(exc).__name__}: {exc}")
    print("Config loads: YES")
    print(f"Config path: {config_path}")
    print("Required imports:", imports)
    print("Device:", device)
    print(f"Output path safe: {'YES' if path_ok else 'NO'} - {path_msg}")
    print("Seeds:", resolve_seeds(args, config))
    print("Stream summary:", json.dumps(stream_summary, indent=2, default=str) if stream_summary else "UNAVAILABLE")
    print("No training started: YES")
    if failures:
        print("Dry-run failed:")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("Dry-run passed.")
    return 0


def run(kind: str, args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.dry_run:
        return dry_run(kind, args, config, config_path)
    device = device_status(args.device)
    if not device.get("available"):
        print("Training aborted; requested device unavailable:", device)
        return 2
    output_base = resolve_output_base(config, kind, args.output_root)
    path_ok, path_msg = output_path_safety(output_base)
    if not path_ok:
        print("Training aborted;", path_msg)
        return 2
    results = []
    for seed in resolve_seeds(args, config):
        print(f"Running {baseline_code(kind)} dataset={config['dataset']} seed={seed} device={args.device} output={output_base}")
        results.append(train_one_seed(kind, config, int(seed), args.device, output_base))
    aggregate_results(results, output_base, kind, config)
    print(f"Wrote outputs under {output_base}")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root")
    parser.add_argument("--single-seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser


if __name__ == "__main__":
    print(VERSION, flush=True)
    print("Use run_derpp_baseline.py, run_er_baseline.py, or run_joint_baseline.py.")
