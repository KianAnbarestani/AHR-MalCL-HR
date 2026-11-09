"""Immutable revision_R1 defaults and known dataset identity references."""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

VERSION = "revision_R1.0.1"
BATCH_SIZE = 256
CLASSIFIER_EPOCH_EQUIVALENT = 3
TASK_SIZES = (50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
SEEDS = tuple(range(42, 52))
ARMS = ("current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched")


def arm_components(arm: str) -> dict[str, bool]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm {arm!r}")
    return {
        "anchors": arm in {"anchor_only", "full_hybrid", "derpp_matched"},
        "generated": arm in {"generated_only", "full_hybrid"},
        "gan": arm in {"generated_only", "full_hybrid"},
        "derpp_loss": arm == "derpp_matched",
    }

# MD5 values and shapes are preserved prepared-data references from V15.5/V16.
# MD5 is a legacy identity fingerprint, not a security primitive; SHA-256 is
# computed and recorded as well.
DATASETS: dict[str, dict[str, Any]] = {
    "ember": {
        "train": {"filename": "EMBER_Class_train.npz", "md5": "27e39d1cb697434107f92a8084128734", "shape": (303331, 2381)},
        "test": {"filename": "EMBER_Class_test.npz", "md5": "9a844162d5ccca20987ce520d6355f33", "shape": (33704, 2381)},
        "anchor_k": 100, "feature_dim": 2381,
    },
    "az_class": {
        "train": {"filename": "AZ_Class_Train.npz", "md5": "644649dfb93f0f0086052a923b657833", "shape": (257023, 2439)},
        "test": {"filename": "AZ_Class_Test.npz", "md5": "1e4014ff6eb613845a6bcf38a2461001", "shape": (28559, 2439)},
        "anchor_k": 200, "feature_dim": 2439,
    },
}


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    seed: int
    arm: str
    update_schedule_sha256: str
    batch_size: int = BATCH_SIZE
    task_sizes: tuple[int, ...] = TASK_SIZES
    classifier_epochs_equivalent: int = CLASSIFIER_EPOCH_EQUIVALENT
    anchor_k: int = 0
    generated_k: int = 100
    generated_candidates_per_class: int = 1000
    classifier_lr: float = 1e-3
    classifier_weight_decay: float = 1e-4
    dropout: float = 0.25
    classifier_hidden: tuple[int, ...] = (1024, 512, 256)
    classifier_embedding_dim: int = 512
    gan_epochs: int = 3
    gan_batch_size: int = 256
    critic_steps: int = 5
    generator_lr: float = 2e-4
    critic_lr: float = 2e-4
    gan_betas: tuple[float, float] = (0.0, 0.9)
    gradient_penalty_weight: float = 10.0
    feature_matching_weight: float = 1.0
    diversity_lambda: float = 0.25
    noise_dim: int = 128
    gradient_clip_norm: float = 10.0
    checkpoint_interval_updates: int = 25
    checkpoint_interval_seconds: int = 60
    drive_sync_interval_seconds: int = 300

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {self.dataset!r}")
        if self.arm not in ARMS:
            raise ValueError(f"unsupported arm {self.arm!r}")
        if self.batch_size != BATCH_SIZE:
            raise ValueError(f"revision_R1 classifier batch size must be {BATCH_SIZE}")
        if self.task_sizes != TASK_SIZES:
            raise ValueError("revision_R1 task sizes are frozen to 50 + ten increments of five")
        if self.seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS}")
        if self.anchor_k not in (0, DATASETS[self.dataset]["anchor_k"]):
            raise ValueError("anchor quota must be zero or the dataset's main K")
        if self.arm in {"anchor_only", "full_hybrid", "derpp_matched"} and self.anchor_k == 0:
            raise ValueError(f"{self.arm} requires the dataset's main anchor quota")
        if self.arm in {"current_only", "generated_only"} and self.anchor_k != 0:
            raise ValueError(f"{self.arm} must not activate real anchors")
        if not self.update_schedule_sha256 or len(self.update_schedule_sha256) != 64:
            raise ValueError("update_schedule_sha256 must be a 64-character SHA-256")
        if self.gan_batch_size != BATCH_SIZE:
            raise ValueError("GAN batch size is fixed at 256")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_sizes"] = list(self.task_sizes)
        data["classifier_hidden"] = list(self.classifier_hidden)
        data["gan_betas"] = list(self.gan_betas)
        data["version"] = VERSION
        data["sampler"] = "mixed_pool_inverse_class_frequency_with_replacement"
        data["sampler_replacement"] = True
        data["remainder_policy"] = "none_full_256_example_updates"
        data["scaler_policy"] = "taskwise_training_only_replay_raw_coordinates"
        return data

    def sha256(self) -> str:
        raw = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def make_run_config(dataset: str, seed: int, arm: str, update_schedule_sha256: str) -> RunConfig:
    anchor_k = DATASETS[dataset]["anchor_k"] if arm in {"anchor_only", "full_hybrid", "derpp_matched"} else 0
    return RunConfig(dataset, int(seed), arm, update_schedule_sha256, anchor_k=anchor_k)

