"""Pilot-only IDs and run matrix; model defaults come from vendor/revision_r1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .stream import WINDOW_IDS
from .vendor.revision_r1.config import (
    BATCH_SIZE,
    CLASSIFIER_EPOCH_EQUIVALENT,
    RunConfig as VerifiedRunConfig,
)

EXPERIMENT_ID = "ahrmalcl_temporal_ember18v2_newfamily_v1"
DEFAULT_OUTPUT_ROOT = Path("temporal_ember_pilot/outputs") / EXPERIMENT_ID
ARMS = ("full_hybrid", "anchor_only", "current_only")
SEEDS = (42, 43, 44)
RUN_IDS = tuple(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS)
ANCHOR_K = 100
GENERATED_K = 100
GAN_CANDIDATES_PER_CLASS = 1000
FEATURE_DIM = 2381
FROZEN_SCHEDULE_SHA256 = "1d7ec52ff9988b6991106da05a9d96329c5179d71d1a934ce1fb431d4bce9170"


@dataclass(frozen=True)
class PilotRunConfig:
    arm: str
    seed: int
    experiment_id: str = EXPERIMENT_ID

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unsupported pilot arm {self.arm!r}")
        if self.seed not in SEEDS:
            raise ValueError(f"pilot seed must be one of {SEEDS}")
        if self.experiment_id != EXPERIMENT_ID:
            raise ValueError("pilot run must use its isolated experiment ID")

    @property
    def run_id(self) -> str:
        return f"{self.arm}_seed{self.seed}"

    @property
    def verified_defaults(self) -> VerifiedRunConfig:
        # This value object supplies the unchanged revision_R1 architecture and
        # optimizer/GAN settings; pilot window budgets are computed separately.
        return VerifiedRunConfig(
            dataset="ember", seed=self.seed,
            arm=self.arm, update_schedule_sha256="0" * 64,
            task_sizes=(50,) + (5,) * 10,
            anchor_k=ANCHOR_K if self.arm in {"full_hybrid", "anchor_only"} else 0,
        )


def validate_run_matrix(configs=None) -> tuple[PilotRunConfig, ...]:
    result = tuple(configs or (PilotRunConfig(arm, seed) for arm in ARMS for seed in SEEDS))
    pairs = [(item.arm, item.seed) for item in result]
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(pairs) != 9 or len(set(pairs)) != 9 or set(pairs) != expected:
        raise ValueError("production matrix must be exactly three arms by seeds 42, 43, and 44")
    if len({item.experiment_id for item in result}) != 1:
        raise ValueError("all pilot runs must share the isolated pilot experiment ID")
    return result
