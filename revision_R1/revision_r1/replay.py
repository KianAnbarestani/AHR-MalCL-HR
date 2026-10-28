"""Replay pool transforms and globally mapped logit-target refresh."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .preprocessing import RunningScaler


class IndexedFeatureRows:
    """A lazy row selection over an ndarray/memmap, cast only on requested reads."""

    def __init__(self, source, indices: Sequence[int]):
        if len(source.shape) != 2:
            raise ValueError("indexed feature source must be two-dimensional")
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or (len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= source.shape[0])):
            raise ValueError("indexed feature rows are outside the source array")
        self.shape = (len(self.indices), int(source.shape[1]))
        self.dtype = np.dtype("float32")

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, positions):
        chosen = self.indices[positions]
        return np.asarray(self.source[chosen], dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        result = self[:]
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result


class ConcatenatedFeatureRows:
    """Concatenation view that materializes only requested rows from each source."""

    def __init__(self, segments: Sequence[Any]):
        self.segments = tuple(segments)
        if not self.segments:
            raise ValueError("feature pool requires at least one source")
        dimensions = {int(segment.shape[1]) for segment in self.segments}
        if len(dimensions) != 1:
            raise ValueError("feature source dimensions do not match")
        self.shape = (sum(int(segment.shape[0]) for segment in self.segments), dimensions.pop())
        self.dtype = np.dtype("float32")
        self._offsets = np.cumsum([0, *(int(segment.shape[0]) for segment in self.segments)])

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, positions):
        scalar = np.isscalar(positions)
        if isinstance(positions, slice):
            start, stop, step = positions.indices(len(self))
            chosen = np.arange(start, stop, step, dtype=np.int64)
        else:
            chosen = np.asarray(positions)
            if chosen.dtype == bool:
                if chosen.shape != (len(self),):
                    raise IndexError("boolean feature-pool index has the wrong shape")
                chosen = np.flatnonzero(chosen)
            chosen = chosen.astype(np.int64, copy=False).reshape(-1)
            chosen = np.where(chosen < 0, chosen + len(self), chosen)
        if len(chosen) and (chosen.min() < 0 or chosen.max() >= len(self)):
            raise IndexError("feature-pool row index out of range")
        result = np.empty((len(chosen), self.shape[1]), dtype=np.float32)
        for index, segment in enumerate(self.segments):
            start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
            selected = (chosen >= start) & (chosen < stop)
            if selected.any():
                local = chosen[selected] - start
                result[selected] = np.asarray(segment[local], dtype=np.float32)
        return result[0] if scalar else result

    def __array__(self, dtype=None, copy=None):
        result = self[:]
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result


class StandardizedFeatureRows:
    def __init__(self, raw: ConcatenatedFeatureRows, scaler: RunningScaler):
        self.raw = raw
        self.scaler = scaler
        self.shape = raw.shape
        self.dtype = np.dtype("float32")

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, positions):
        return self.scaler.transform(self.raw[positions])

    def __array__(self, dtype=None, copy=None):
        result = self[:]
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result


@dataclass(frozen=True)
class MixedPool:
    x_raw: Any
    x_standardized: Any
    y_global: np.ndarray
    sources: np.ndarray
    sample_ids: list[str]


def build_mixed_pool(parts: Mapping[str, tuple[np.ndarray, np.ndarray, Sequence[str]]], scaler: RunningScaler) -> MixedPool:
    raw_parts, label_parts, source_parts, identity_parts = [], [], [], []
    for source in ("current", "anchor", "generated"):
        if source not in parts:
            continue
        raw, labels, identities = parts[source]
        if not isinstance(raw, IndexedFeatureRows):
            raw = np.asarray(raw, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        identities = [str(value) for value in identities]
        if raw.ndim != 2 or labels.ndim != 1 or len(raw) != len(labels) or len(identities) != len(labels):
            raise ValueError(f"invalid raw feature/label/identity arrays for source {source}")
        if len(raw) == 0:
            continue
        if not np.isin(labels, np.arange(100)).all():
            raise ValueError(f"source {source} contains a non-global class ID")
        raw_parts.append(raw)
        label_parts.append(labels)
        source_parts.append(np.full(len(labels), source, dtype=object))
        identity_parts.extend(identities)
    if not raw_parts:
        raise ValueError("classifier mixed pool has no active examples")
    dimensions = {value.shape[1] for value in raw_parts}
    if len(dimensions) != 1:
        raise ValueError("active source feature dimensions do not match")
    x_raw = ConcatenatedFeatureRows(raw_parts)
    y_global = np.concatenate(label_parts)
    sources = np.concatenate(source_parts).astype(str)
    x_standardized = StandardizedFeatureRows(x_raw, scaler)
    return MixedPool(x_raw, x_standardized, y_global, sources, identity_parts)


def refresh_stored_logits(model, raw_features: np.ndarray, labels_global: Sequence[int], scaler: RunningScaler, class_ids: Sequence[int], device: str | torch.device = "cpu", batch_size: int = 2048) -> dict:
    raw = raw_features if isinstance(raw_features, (IndexedFeatureRows, ConcatenatedFeatureRows)) else np.asarray(raw_features, dtype=np.float32)
    labels = [int(value) for value in labels_global]
    classes = [int(value) for value in class_ids]
    if len(raw.shape) != 2 or len(raw) != len(labels):
        raise ValueError("stored replay features and labels must be aligned 2-D/1-D arrays")
    if len(set(classes)) != len(classes) or any(value < 0 or value >= 100 for value in classes):
        raise ValueError("stored logit class IDs must be unique global IDs 0..99")
    if any(value not in classes for value in labels):
        raise ValueError("stored replay label is absent from the stored head class mapping")
    if batch_size <= 0:
        raise ValueError("logit refresh batch size must be positive")
    from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory
    require_host_memory(
        min(len(raw), batch_size) * int(raw.shape[1]) * 8 + (16 << 20),
        reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="bounded stored-logit refresh batch",
    )
    was_training = bool(model.training)
    model.eval()
    outputs = []
    target_device = torch.device(device)
    try:
        with torch.no_grad():
            for start in range(0, len(raw), batch_size):
                normalized = scaler.transform(raw[start : start + batch_size])
                tensor = torch.as_tensor(normalized, dtype=torch.float32, device=target_device)
                logits = model(tensor)
                if logits.ndim != 2 or logits.shape[1] != len(classes):
                    raise ValueError("model output columns do not match stored global class IDs")
                outputs.append(logits.detach().cpu().numpy())
    finally:
        model.train(was_training)
    logits = np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(classes)), dtype=np.float32)
    seen_mask = [False] * 100
    for class_id in classes:
        seen_mask[class_id] = True
    return {
        "logits": logits.tolist(), "class_ids": classes, "global_labels": labels,
        "seen_class_mask": seen_mask, "scaler_sha256": scaler.sha256(),
    }

