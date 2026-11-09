"""Named, reproducible random streams; GAN randomness cannot advance classifier streams."""

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


def _seed(dataset: str, seed: int, stream: str, arm: str | None = None) -> int:
    token = f"revision_R1|{dataset}|{int(seed)}|{stream}|{arm or 'paired'}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**63 - 1)


@dataclass
class RNGStreams:
    class_order: list[int]
    classifier_init_seed: int
    dropout_seed: int
    sampling: np.random.Generator
    anchor_seed: int
    anchor_selection: np.random.Generator
    gan_seed: int
    gan: np.random.Generator

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_order": list(self.class_order),
            "classifier_init_seed": self.classifier_init_seed,
            "dropout_seed": self.dropout_seed,
            "sampling": self.sampling.bit_generator.state,
            "anchor_seed": self.anchor_seed,
            "anchor_selection": self.anchor_selection.bit_generator.state,
            "gan_seed": self.gan_seed,
            "gan": self.gan.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("class_order", "classifier_init_seed", "dropout_seed", "anchor_seed", "gan_seed"):
            if state[name] != getattr(self, name):
                raise ValueError(f"RNG stream identity mismatch for {name}")
        self.sampling.bit_generator.state = state["sampling"]
        self.anchor_selection.bit_generator.state = state["anchor_selection"]
        self.gan.bit_generator.state = state["gan"]


def make_rng_streams(dataset: str, seed: int, arm: str) -> RNGStreams:
    if dataset not in {"ember", "az_class"}:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if arm not in {"current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched"}:
        raise ValueError(f"unsupported arm {arm!r}")
    # Preserve the already-used V15.5/V16 class-order mapping exactly.
    order = np.random.RandomState(int(seed)).permutation(100).astype(int).tolist()
    init_seed = _seed(dataset, seed, "classifier_init")
    dropout_seed = _seed(dataset, seed, "classifier_dropout")
    sampling_seed = _seed(dataset, seed, "classifier_sampling")
    anchor_seed = _seed(dataset, seed, "anchor_selection")
    gan_seed = _seed(dataset, seed, "gan_operations", arm)
    return RNGStreams(
        class_order=order,
        classifier_init_seed=init_seed,
        dropout_seed=dropout_seed,
        sampling=np.random.default_rng(sampling_seed),
        anchor_seed=anchor_seed,
        anchor_selection=np.random.default_rng(anchor_seed),
        gan_seed=gan_seed,
        gan=np.random.default_rng(gan_seed),
    )

