#!/usr/bin/env python3
"""Colab-compatible Joint/offline upper-bound baseline runner."""

from __future__ import annotations

import argparse

from baseline_runner_common import add_common_args, run


VERSION = "v1 - Colab-compatible Joint baseline runner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run Joint/offline upper bound under the AHR-MalCL protocol.")
    return add_common_args(parser).parse_args()


def main() -> None:
    print(VERSION, flush=True)
    raise SystemExit(run("joint", parse_args()))


if __name__ == "__main__":
    main()
