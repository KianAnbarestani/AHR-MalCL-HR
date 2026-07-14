#!/usr/bin/env python3
"""Create a safe Colab upload bundle for external baseline runs."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


VERSION = "v1 - Colab baseline bundle builder"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "paper_ready_baselines"
BUNDLE_DIR = OUT_ROOT / "colab_bundle"
ZIP_PATH = OUT_ROOT / "colab_baseline_bundle.zip"

REQUIREMENTS = """\
numpy
pandas
scipy
scikit-learn
torch
tqdm
"""


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts or ".venv_stats" in parts or "venv" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def copy_tree_filtered(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if should_skip(path):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def reset_bundle_dir() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


def write_bundle_docs() -> None:
    (BUNDLE_DIR / "requirements_colab.txt").write_text(REQUIREMENTS, encoding="utf-8")
    shutil.copy2(OUT_ROOT / "COLAB_RUN_COMMANDS.md", BUNDLE_DIR / "COLAB_BASELINE_RUN_COMMANDS.md")
    shutil.copy2(OUT_ROOT / "COLAB_README.md", BUNDLE_DIR / "COLAB_README.md")
    note = [
        "# Dataset Note",
        "",
        "Datasets are intentionally not included in this bundle.",
        "",
        "Upload or mount these files separately:",
        "",
        "- `EMBER_Class_train.npz`",
        "- `EMBER_Class_test.npz`",
        "- `AZ_Class_Train.npz`",
        "- `AZ_Class_Test.npz`",
        "",
        "Place them in `/content/data`, `/content/drive/MyDrive/AHR-MalCL/data`, or set `AHR_MALCL_DATA_DIR`.",
        "",
    ]
    (BUNDLE_DIR / "DATASET_NOTE.md").write_text("\n".join(note), encoding="utf-8")


def create_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in BUNDLE_DIR.rglob("*"):
            if path.is_file() and not should_skip(path):
                archive.write(path, path.relative_to(OUT_ROOT))


def run() -> dict[str, str | int]:
    reset_bundle_dir()
    copy_tree_filtered(PROJECT_ROOT / "baseline_suite", BUNDLE_DIR / "baseline_suite")
    copy_tree_filtered(PROJECT_ROOT / "baseline_reference_pack" / "notes", BUNDLE_DIR / "baseline_reference_pack" / "notes")
    copy_tree_filtered(PROJECT_ROOT / "baseline_reference_pack" / "code_links", BUNDLE_DIR / "baseline_reference_pack" / "code_links")
    write_bundle_docs()
    create_zip()
    file_count = sum(1 for path in BUNDLE_DIR.rglob("*") if path.is_file())
    return {
        "bundle_dir": str(BUNDLE_DIR.relative_to(PROJECT_ROOT)),
        "zip_path": str(ZIP_PATH.relative_to(PROJECT_ROOT)),
        "files": file_count,
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print("Created {bundle_dir} with {files} files.".format(**summary))
    print("Created {zip_path}.".format(**summary))


if __name__ == "__main__":
    main()
