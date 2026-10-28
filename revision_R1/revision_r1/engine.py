"""Small phase-cursor engine; model-specific work is supplied by phase handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .state import PhaseCursor, RunState


class StopRequested(RuntimeError):
    """A cooperative stop was requested at a safe persisted phase boundary."""


def advance_phase_cursor(
    cursor: PhaseCursor,
    *,
    phase_order: Sequence[str],
    phase_limits: Mapping[str, int],
) -> None:
    """Advance exactly one completed unit, transitioning only at its boundary."""
    if cursor.phase == "complete":
        raise ValueError("cannot advance a completed run")
    if cursor.phase not in phase_order:
        raise ValueError(f"phase {cursor.phase!r} is absent from phase order")
    if set(phase_order) != set(phase_limits):
        raise ValueError("phase order and phase limits must name the same phases")
    limit = phase_limits[cursor.phase]
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"phase {cursor.phase!r} must have a positive integer limit")
    cursor.item_index += 1
    if cursor.item_index < limit:
        return
    if cursor.item_index > limit:
        raise ValueError(f"phase {cursor.phase!r} cursor exceeded its limit")
    pos = phase_order.index(cursor.phase)
    cursor.item_index = 0
    cursor.substep_index = 0
    cursor.update_index = 0
    cursor.phase = phase_order[pos + 1] if pos + 1 < len(phase_order) else "complete"


def run_phase_machine(
    state: RunState,
    handlers: Mapping[str, Callable[[RunState], Any]],
    *,
    phase_limits: Mapping[str, int],
    phase_order: Sequence[str],
    stop_hook: Callable[[RunState], bool] | None = None,
    checkpoint_hook: Callable[[RunState], Any] | None = None,
) -> RunState:
    """Run one atomic item at a time, checkpointing only after cursor advance.

    A stop request is honored after the handler's state and cursor have both
    advanced. The next invocation starts at the following unprocessed item.
    """
    state.validate()
    if set(handlers) != set(phase_order):
        raise ValueError("every phase must have exactly one handler")
    if state.cursor.phase not in (*phase_order, "complete"):
        raise ValueError(f"unknown current phase {state.cursor.phase!r}")
    while state.cursor.phase != "complete":
        handlers[state.cursor.phase](state)
        advance_phase_cursor(state.cursor, phase_order=phase_order, phase_limits=phase_limits)
        if checkpoint_hook is not None:
            checkpoint_hook(state)
        if stop_hook is not None and stop_hook(state):
            raise StopRequested(f"cooperative stop at {state.cursor.phase}:{state.cursor.item_index}")
    return state
