#!/usr/bin/env python3
"""Colab-compatible canonical ER baseline runner."""

from __future__ import annotations

import argparse

from baseline_runner_common import add_common_args, run
from baseline_config_validation import validate


VERSION = "v1 - Colab-compatible ER baseline runner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run canonical ER under the AHR-MalCL protocol.")
    parser = add_common_args(parser)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(VERSION, flush=True)
    args = parse_args()
    raise SystemExit(validate("er", args) if args.validate_config_only else run("er", args))


if __name__ == "__main__":
    main()
