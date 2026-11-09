"""Atomic, pilot-only completed-window checkpoints."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any

CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_STATE_KEYS = {
    "stream_cursor", "training_state", "optimizer_state", "scaler_state",
    "anchor_state", "class_map", "gan_state", "rng_state",
}


def _validate_state(state: dict) -> None:
    if not isinstance(state, dict) or not _REQUIRED_STATE_KEYS.issubset(state):
        missing = sorted(_REQUIRED_STATE_KEYS - set(state if isinstance(state, dict) else ()))
        raise ValueError(f"checkpoint state is missing required pilot state: {missing}")
    cursor = state["stream_cursor"]
    if not isinstance(cursor, dict) or int(cursor.get("last_completed_window", -1)) < -1:
        raise ValueError("checkpoint stream cursor is invalid")
    if int(cursor.get("next_window_index", -1)) != int(cursor.get("last_completed_window", -1)) + 1:
        raise ValueError("checkpoint cursor must point immediately after the last completed window")


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> str:
    """Write one full state snapshot atomically and return its file SHA-256."""
    _validate_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"format_version": CHECKPOINT_FORMAT_VERSION, "state": state}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            pickle.dump(envelope, target, protocol=pickle.HIGHEST_PROTOCOL)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as source:
        envelope = pickle.load(source)
    if not isinstance(envelope, dict) or envelope.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or corrupt temporal pilot checkpoint")
    state = envelope.get("state")
    _validate_state(state)
    return state


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_latest_checkpoint(run_dir: str | Path) -> tuple[Path, dict] | None:
    """Load the latest consecutive completed-window checkpoint or reject corruption."""
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob("window_*.pkl"), key=lambda path: int(path.stem.split("_")[-1]))
    if not paths:
        return None
    expected_index = 0
    latest = None
    for path in paths:
        match = re.fullmatch(r"window_(\d+)", path.stem)
        if not match or int(match.group(1)) != expected_index:
            raise ValueError("completed-window checkpoints must be consecutive starting at window 0")
        state = load_checkpoint(path)
        completed = int(state["stream_cursor"]["last_completed_window"])
        if completed != expected_index:
            raise ValueError(f"checkpoint filename/cursor mismatch for {path.name}")
        latest = (path, state)
        expected_index += 1
    return latest
