"""Single-line console progress renderer and pilot-local progress JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


class ProgressLine:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.previous = None
        self.open = False

    def update(self, run: str, window: str, phase: str, epoch: int, epochs: int, batch: int, batches: int) -> None:
        line = f"run={run} window={window} phase={phase} epoch={epoch}/{epochs} batch={batch}/{batches}"
        if line == self.previous:
            return
        self.stream.write("\r" + line)
        self.stream.flush()
        self.previous = line
        self.open = True

    def finish_phase(self) -> None:
        if self.open:
            self.stream.write("\n")
            self.stream.flush()
        self.previous = None
        self.open = False


class ProgressRegistry:
    """Atomic pilot-only progress file; main-matrix registry is never read."""

    def __init__(self, output_root, experiment_id):
        self.path = Path(output_root) / "progress.json"
        self.experiment_id = str(experiment_id)

    def update(self, run_id, window_id, phase, *, epoch=0, epochs=0, batch=0, batches=0):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "latest": {
                "run_id": str(run_id), "window_id": str(window_id), "phase": str(phase),
                "epoch": int(epoch), "epochs": int(epochs), "batch": int(batch), "batches": int(batches),
            },
        }
        fd, temporary = tempfile.mkstemp(prefix=".progress.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(payload, target, sort_keys=True, separators=(",", ":"))
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
