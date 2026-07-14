#!/usr/bin/env python3
"""Colab-compatible DER++ baseline runner."""

from __future__ import annotations

import argparse

from baseline_runner_common import add_common_args, run


VERSION = "v2 - Colab-compatible DER++ baseline runner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run DER++ under the AHR-MalCL protocol.")
    return add_common_args(parser).parse_args()


def main() -> None:
    print(VERSION, flush=True)
    raise SystemExit(run("derpp", parse_args()))


if __name__ == "__main__":
    main()
