#!/usr/bin/env python3
"""Export a project directory tree to a Markdown file.

Use this when you want to give ChatGPT a compact overview of the repository
structure without copying binary files or large result contents.

Example:
  python3 scripts/export_project_tree.py \
    --root . \
    --output project_directory_tree.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_IGNORE_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
}

DEFAULT_IGNORE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
}


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def should_skip(path: Path, ignore_names: set[str], ignore_suffixes: set[str]) -> bool:
    return path.name in ignore_names or path.suffix.lower() in ignore_suffixes


def render_tree(
    root: Path,
    max_depth: int | None,
    ignore_names: set[str],
    ignore_suffixes: set[str],
) -> list[str]:
    lines = [f"{root.resolve().name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return

        children = [
            child
            for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if not should_skip(child, ignore_names, ignore_suffixes)
        ]

        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            branch = "`-- " if is_last else "|-- "
            next_prefix = "    " if is_last else "|   "

            if child.is_dir():
                lines.append(f"{prefix}{branch}{child.name}/")
                walk(child, prefix + next_prefix, depth + 1)
            else:
                try:
                    size = human_size(child.stat().st_size)
                except OSError:
                    size = "unknown"
                lines.append(f"{prefix}{branch}{child.name} ({size})")

    walk(root, "", 1)
    return lines


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export project directory tree to Markdown.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root to scan.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project_directory_tree.md"),
        help="Markdown output file.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory depth to include. Default: no limit.",
    )
    parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include common binary/image/model files in the tree.",
    )
    parser.add_argument(
        "--ignore-names",
        default=",".join(sorted(DEFAULT_IGNORE_NAMES)),
        help="Comma-separated file or directory names to ignore.",
    )
    parser.add_argument(
        "--ignore-suffixes",
        default=",".join(sorted(DEFAULT_IGNORE_SUFFIXES)),
        help="Comma-separated suffixes to ignore, e.g. .png,.pyc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    ignore_names = parse_csv_set(args.ignore_names)
    ignore_suffixes = set() if args.include_binary else parse_csv_set(args.ignore_suffixes)
    lines = render_tree(root, args.max_depth, ignore_names, ignore_suffixes)

    content = [
        "# Project Directory Tree",
        "",
        f"Root: `{root}`",
        "",
        "```text",
        *lines,
        "```",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
