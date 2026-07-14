#!/usr/bin/env python3
"""Colab-compatible canonical ER baseline runner."""

from __future__ import annotations

import argparse

from baseline_runner_common import add_common_args, run


VERSION = "v1 - Colab-compatible ER baseline runner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run canonical ER under the AHR-MalCL protocol.")
    return add_common_args(parser).parse_args()


def main() -> None:
    print(VERSION, flush=True)
    raise SystemExit(run("er", parse_args()))


if __name__ == "__main__":
    main()
