"""Append-preserving run attempts and hash-validated completion markers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._thread_lock = threading.RLock()

    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
            raise ValueError("run id contains unsafe path characters")
        return self.root / "runs" / run_id

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "registry.json"

    def _read(self, run_id: str) -> dict[str, Any]:
        path = self._state_path(run_id)
        if not path.exists():
            return {"run_id": run_id, "identity": None, "status": "new", "attempts": []}
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def status(self, run_id: str) -> dict[str, Any]:
        with self._thread_lock:
            return self._read(run_id)

    def begin_attempt(self, run_id: str, *, identity: dict[str, Any]) -> dict[str, Any]:
        with self._thread_lock:
            record = self._read(run_id)
            if record["identity"] is not None and record["identity"] != identity:
                raise ValueError("run id is already bound to a different identity")
            if self.is_completed(run_id):
                raise ValueError("validated completed run cannot be silently restarted")
            attempt = {
                "attempt_id": uuid.uuid4().hex,
                "started_at": time.time(),
                "status": "running",
                "failure_reason": None,
            }
            record["identity"] = identity
            record["status"] = "running"
            record["attempts"].append(attempt)
            record["completion_artifacts"] = None
            _atomic_json(self._state_path(run_id), record)
            return attempt

    def fail_attempt(self, run_id: str, reason: str) -> None:
        with self._thread_lock:
            record = self._read(run_id)
            if record["status"] != "running" or not record["attempts"]:
                raise ValueError("there is no active attempt to fail")
            attempt = record["attempts"][-1]
            attempt.update(status="failed", ended_at=time.time(), failure_reason=str(reason))
            record["status"] = "failed"
            _atomic_json(self._state_path(run_id), record)

    def complete(self, run_id: str, artifacts: dict[str, str | Path]) -> bool:
        with self._thread_lock:
            record = self._read(run_id)
            if record["status"] != "running" or not record["attempts"]:
                raise ValueError("only an active attempt can be completed")
            manifest = {}
            for label, raw_path in sorted(artifacts.items()):
                path = Path(raw_path)
                if not label or not path.is_file():
                    raise FileNotFoundError(f"required completion artifact missing: {label} ({path})")
                manifest[label] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _hash_file(path)}
            if not manifest:
                raise ValueError("completion requires at least one validated artifact")
            marker = {
                "run_id": run_id,
                "identity": record["identity"],
                "attempt_id": record["attempts"][-1]["attempt_id"],
                "created_at": time.time(),
                "artifacts": manifest,
            }
            _atomic_json(self._run_dir(run_id) / "COMPLETE.json", marker)
            record["completion_artifacts"] = manifest
            record["status"] = "completed"
            record["attempts"][-1].update(status="completed", ended_at=time.time())
            _atomic_json(self._state_path(run_id), record)
            return self.is_completed(run_id)

    def is_completed(self, run_id: str) -> bool:
        try:
            record = self._read(run_id)
            if record["status"] != "completed" or not record.get("completion_artifacts"):
                return False
            with (self._run_dir(run_id) / "COMPLETE.json").open(encoding="utf-8") as stream:
                marker = json.load(stream)
            if marker.get("run_id") != run_id or marker.get("identity") != record.get("identity"):
                return False
            if marker.get("artifacts") != record["completion_artifacts"]:
                return False
            for item in marker["artifacts"].values():
                path = Path(item["path"])
                if not path.is_file() or path.stat().st_size != item["bytes"] or _hash_file(path) != item["sha256"]:
                    return False
            return bool(marker["artifacts"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False
