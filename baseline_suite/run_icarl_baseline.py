#!/usr/bin/env python3
"""Optional iCaRL baseline runner scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from optional_baseline_common import run_optional


VERSION = "v1 - iCaRL baseline runner scaffold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or gate the optional iCaRL baseline.")
    parser.add_argument("--config", default="baseline_suite/configs/icarl_ember_k100.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(VERSION, flush=True)
    args = parse_args()
    raise SystemExit(run_optional("iCaRL", Path(args.config), args.dry_run))


if __name__ == "__main__":
    main()
