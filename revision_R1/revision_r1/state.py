"""Serializable phase-level state for separately versioned revision_R1 runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseCursor:
    task_index: int
    phase: str
    item_index: int = 0
    substep_index: int = 0
    update_index: int = 0


@dataclass
class RunState:
    """All mutable state required to resume an update/phase without replaying it."""

    identity: dict[str, Any]
    cursor: PhaseCursor
    classifier_state: dict[str, Any] = field(default_factory=dict)
    classifier_optimizer_state: dict[str, Any] = field(default_factory=dict)
    generator_state: dict[str, Any] = field(default_factory=dict)
    generator_optimizer_state: dict[str, Any] = field(default_factory=dict)
    critic_state: dict[str, Any] = field(default_factory=dict)
    critic_optimizer_state: dict[str, Any] = field(default_factory=dict)
    scaler_state: dict[str, Any] = field(default_factory=dict)
    amp_state: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)
    sampler_state: dict[str, Any] = field(default_factory=dict)
    anchor_replay_state: dict[str, Any] = field(default_factory=dict)
    generated_replay_state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if not isinstance(self.identity, dict) or not self.identity:
            raise ValueError("run state requires a non-empty immutable identity")
        if self.cursor.task_index < 0 or self.cursor.item_index < 0:
            raise ValueError("phase cursor indexes must be non-negative")
        if not self.cursor.phase:
            raise ValueError("phase cursor must name a phase")
        if self.schema_version != 1:
            raise ValueError(f"unsupported run-state schema {self.schema_version}")
