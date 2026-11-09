"""Class-balanced replacement sampling over the whole active mixed pool."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DrawBatch:
    indices: np.ndarray
    labels: np.ndarray
    sources: np.ndarray
    source_counts: dict[str, int]
    class_counts: dict[int, int]


def pool_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("mixed pool labels must be a nonempty 1-D array")
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64)


def expected_source_proportions(labels: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    sources = np.asarray(sources).astype(str)
    if len(labels) != len(sources) or labels.size == 0:
        raise ValueError("labels and source IDs must be nonempty arrays of equal length")
    weights = pool_weights(labels)
    total = float(weights.sum())
    return {
        source: float(weights[sources == source].sum() / total)
        for source in sorted(set(sources.tolist()))
    }


class MixedPoolSampler:
    """Draw fixed full batches from concatenated sources; never select a source first."""

    def __init__(self, labels: np.ndarray, sources: np.ndarray, rng: np.random.Generator):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sources = np.asarray(sources).astype(str)
        if self.labels.ndim != 1 or len(self.labels) == 0 or len(self.labels) != len(self.sources):
            raise ValueError("mixed pool needs equal, nonempty 1-D label/source arrays")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        self.weights = pool_weights(self.labels)
        self.probabilities = self.weights / self.weights.sum()
        self.rng = rng
        self.cursor = 0
        self.realized_source_counts: dict[str, int] = {}
        self.realized_class_counts: dict[int, int] = {}

    def draw_batch(self, batch_size: int = 256) -> DrawBatch:
        if batch_size != 256:
            raise ValueError("revision_R1 classifier batches are fixed at 256 examples")
        indices = self.rng.choice(len(self.labels), size=256, replace=True, p=self.probabilities)
        labels = self.labels[indices].copy()
        sources = self.sources[indices].copy()
        source_counts = {name: int(np.count_nonzero(sources == name)) for name in sorted(set(sources.tolist()))}
        class_counts = {int(name): int(count) for name, count in zip(*np.unique(labels, return_counts=True))}
        for name, count in source_counts.items():
            self.realized_source_counts[name] = self.realized_source_counts.get(name, 0) + count
        for name, count in class_counts.items():
            self.realized_class_counts[name] = self.realized_class_counts.get(name, 0) + count
        self.cursor += 256
        return DrawBatch(indices.astype(np.int64), labels, sources, source_counts, class_counts)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "cursor": self.cursor,
            "realized_source_counts": dict(self.realized_source_counts),
            "realized_class_counts": dict(self.realized_class_counts),
        }

    @classmethod
    def from_state(cls, labels: np.ndarray, sources: np.ndarray, state: dict[str, Any]) -> "MixedPoolSampler":
        if state.get("version") != 1:
            raise ValueError("unsupported sampler-state version")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        sampler = cls(labels, sources, rng)
        sampler.cursor = int(state["cursor"])
        if sampler.cursor < 0 or sampler.cursor % 256:
            raise ValueError("sampler cursor must be a nonnegative number of complete updates")
        sampler.realized_source_counts = {str(k): int(v) for k, v in state["realized_source_counts"].items()}
        sampler.realized_class_counts = {int(k): int(v) for k, v in state["realized_class_counts"].items()}
        if any(v < 0 for v in (*sampler.realized_source_counts.values(), *sampler.realized_class_counts.values())):
            raise ValueError("sampler counters cannot be negative")
        return sampler

