"""Taskwise training-only scaler with explicit raw-coordinate replay semantics."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import torch


class RunningScaler:
    """Population-standard-deviation scaler incrementally fit on training rows."""

    def __init__(self):
        self.n = 0
        self.mean_: np.ndarray | None = None
        self.M2_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def update(self, raw_train: np.ndarray) -> "RunningScaler":
        raw = np.asarray(raw_train, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError("scaler update expects a 2-D training feature array")
        if not np.isfinite(raw).all():
            raise ValueError("scaler update requires finite training features")
        if self.mean_ is None:
            self.mean_ = np.zeros(raw.shape[1], dtype=np.float64)
            self.M2_ = np.zeros(raw.shape[1], dtype=np.float64)
        if raw.shape[1] != self.mean_.size:
            raise ValueError("feature dimension changed during scaler updates")
        if not len(raw):
            return self
        batch_n = len(raw)
        batch_mean = raw.mean(axis=0)
        batch_M2 = raw.var(axis=0) * batch_n
        if self.n == 0:
            self.mean_ = batch_mean
            self.M2_ = batch_M2
            self.n = batch_n
        else:
            delta = batch_mean - self.mean_
            total = self.n + batch_n
            self.mean_ += delta * batch_n / total
            self.M2_ += batch_M2 + delta**2 * self.n * batch_n / total
            self.n = total
        variance = self.M2_ / max(self.n, 1)
        self.scale_ = np.sqrt(np.maximum(variance, 0.0))
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.mean_ = self.mean_.astype(np.float32)
        self.scale_ = self.scale_.astype(np.float32)
        return self

    def update_indexed(self, source, indices, *, chunk_rows: int = 4096) -> "RunningScaler":
        """Fit population statistics while gathering at most one bounded row chunk."""
        if chunk_rows < 1:
            raise ValueError("scaler chunk_rows must be positive")
        index_values = np.asarray(indices, dtype=np.int64)
        if index_values.ndim != 1:
            raise ValueError("indexed scaler rows must be a one-dimensional index sequence")
        if len(source.shape) != 2:
            raise ValueError("scaler source must be a 2-D training feature array")
        if len(index_values) and (index_values.min() < 0 or index_values.max() >= source.shape[0]):
            raise ValueError("indexed scaler rows are outside the source array")
        feature_dim = int(source.shape[1])
        if self.mean_ is not None and feature_dim != self.mean_.size:
            raise ValueError("feature dimension changed during scaler updates")
        if not len(index_values):
            return self

        running_mean = np.zeros(feature_dim, dtype=np.float64) if self.mean_ is None else self.mean_.astype(np.float64, copy=True)
        running_m2 = np.zeros(feature_dim, dtype=np.float64) if self.M2_ is None else self.M2_.copy()
        running_n = int(self.n)
        for offset in range(0, len(index_values), int(chunk_rows)):
            selected = index_values[offset : offset + int(chunk_rows)]
            from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory
            required_memory = len(selected) * feature_dim * (4 + 8 + 1) + (16 << 20)
            require_host_memory(
                required_memory, reserve_bytes=HOST_RAM_RESERVE_BYTES,
                phase="bounded taskwise scaler update",
            )
            raw32 = np.asarray(source[selected], dtype=np.float32)
            if raw32.ndim != 2 or raw32.shape[1] != feature_dim:
                raise ValueError("indexed scaler source returned an invalid feature block")
            if not np.isfinite(raw32).all():
                raise ValueError("scaler update requires finite training features")
            raw = raw32.astype(np.float64)
            batch_n = len(raw)
            batch_mean = raw.mean(axis=0)
            batch_m2 = raw.var(axis=0) * batch_n
            if running_n == 0:
                running_mean = batch_mean
                running_m2 = batch_m2
                running_n = batch_n
            else:
                delta = batch_mean - running_mean
                total = running_n + batch_n
                running_mean += delta * batch_n / total
                running_m2 += batch_m2 + delta**2 * running_n * batch_n / total
                running_n = total

        self.n = running_n
        self.mean_ = running_mean.astype(np.float32)
        self.M2_ = running_m2
        variance = self.M2_ / max(self.n, 1)
        self.scale_ = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def _ready(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_ is None or self.scale_ is None or self.n <= 0:
            raise RuntimeError("scaler has not received training rows")
        return self.mean_, self.scale_

    def transform(self, raw):
        mean, scale = self._ready()
        if isinstance(raw, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=raw.device, dtype=raw.dtype)
            scale_t = torch.as_tensor(scale, device=raw.device, dtype=raw.dtype)
            if raw.shape[-1] != mean.size:
                raise ValueError("feature dimension does not match scaler")
            return (raw - mean_t) / scale_t
        values = np.asarray(raw, dtype=np.float32)
        if values.shape[-1] != mean.size:
            raise ValueError("feature dimension does not match scaler")
        return ((values - mean) / scale).astype(np.float32)

    def inverse(self, standardized):
        mean, scale = self._ready()
        if isinstance(standardized, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=standardized.device, dtype=standardized.dtype)
            scale_t = torch.as_tensor(scale, device=standardized.device, dtype=standardized.dtype)
            if standardized.shape[-1] != mean.size:
                raise ValueError("feature dimension does not match scaler")
            return standardized * scale_t + mean_t
        values = np.asarray(standardized, dtype=np.float32)
        if values.shape[-1] != mean.size:
            raise ValueError("feature dimension does not match scaler")
        return (values * scale + mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "version": 1, "n": self.n,
            "mean": None if self.mean_ is None else self.mean_.copy(),
            "M2": None if self.M2_ is None else self.M2_.copy(),
            "scale": None if self.scale_ is None else self.scale_.copy(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningScaler":
        if state.get("version") != 1:
            raise ValueError("unsupported scaler state version")
        result = cls()
        result.n = int(state["n"])
        result.mean_ = None if state["mean"] is None else np.asarray(state["mean"], dtype=np.float32).copy()
        result.M2_ = None if state["M2"] is None else np.asarray(state["M2"], dtype=np.float64).copy()
        result.scale_ = None if state["scale"] is None else np.asarray(state["scale"], dtype=np.float32).copy()
        if result.n < 0:
            raise ValueError("scaler sample count cannot be negative")
        if result.mean_ is not None and (result.M2_ is None or result.scale_ is None or result.mean_.shape != result.scale_.shape or result.mean_.shape != result.M2_.shape):
            raise ValueError("inconsistent scaler state dimensions")
        return result

    def sha256(self) -> str:
        mean, scale = self._ready()
        digest = hashlib.sha256(str(self.n).encode())
        digest.update(np.ascontiguousarray(mean).tobytes())
        digest.update(np.ascontiguousarray(self.M2_).tobytes())
        digest.update(np.ascontiguousarray(scale).tobytes())
        return digest.hexdigest()


def to_raw_generator_output(x_standardized, scaler_state: dict):
    return RunningScaler.from_state_dict(deepcopy(scaler_state)).inverse(x_standardized)

