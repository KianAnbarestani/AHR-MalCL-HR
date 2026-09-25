#!/usr/bin/env python3
"""Recompute every analysis artifact from released canonical input."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "analyze_final_comparison.py"), "--input-root", str(args.input_root), "--output-root", str(args.output_root)],
        check=True,
    )


if __name__ == "__main__":
    main()
