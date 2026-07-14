#!/usr/bin/env python3
"""MADAR adapter template for exact-protocol AHR-MalCL-HR comparisons.

This file is intentionally a template. Do not train from it until the approved
MADAR source is available locally and the adapter has been validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


VERSION = "v1 - MADAR adapter template"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_ahr_malcl_stream(config: dict[str, Any], seed: int) -> Any:
    """Return the exact EMBER/AZ class-incremental stream for MADAR.

    Expected contract:
    - current-task train features/labels per task
    - seen-task test features/labels per task
    - class order and task metadata
    - scaler state matching the final HR protocol
    """
    raise NotImplementedError("Extract and validate the AHR-MalCL-HR data-stream adapter first.")


def run_madar_with_adapter(config: dict[str, Any]) -> None:
    """Call approved MADAR source code using the exact-protocol stream."""
    raise NotImplementedError("Wire this function to approved local MADAR code after source inspection.")


if __name__ == "__main__":
    print(VERSION, flush=True)
    print("Template only: MADAR source integration is required before training.")
