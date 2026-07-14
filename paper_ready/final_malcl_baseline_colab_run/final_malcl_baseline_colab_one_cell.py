# FINAL MALCL BASELINE COLAB ONE-CELL RUNNER
# Paste this whole file into one Colab cell and run it with a GPU runtime.

print("FINAL_MALCL_BASELINE_COLAB_RUN_START")

from pathlib import Path
import json
import os
import runpy
import shutil
import sys
import zipfile


# Editable runtime controls. Defaults are protocol-matched; do not change for paper runs.
MALCL_EPOCHS = "3"
MALCL_BATCH_SIZE = "256"
FAST_DEBUG = "False"
ALLOW_SUBSAMPLING = "False"
MALCL_GENERATOR_LOSS = "FML"
MALCL_SAMPLE_SELECT = "L1_C_Mean"
MALCL_K = "3"
REPLAY_REFRESH_EVERY_BATCH = "True"
PRINT_TRAINING_PROGRESS = "True"
PROGRESS_EVERY_N_BATCHES = "25"  # Set to "1" if you want every batch printed.
USE_TQDM_PROGRESS = "True"
TQDM_MININTERVAL = "2.0"


PROJECT_ZIP = Path("/content/drive/MyDrive/final_malcl_baseline_colab_package.zip")
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/data")
CONTENT_DATA_DIR = Path("/content/data")
WORKDIR = Path("/content/AHR-MalCL")
DRIVE_OUTPUT_ZIP = Path("/content/drive/MyDrive/AHR_MalCL/final_malcl_baseline_results_colab.zip")


def mount_drive():
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:
        raise RuntimeError(f"Google Drive mount failed: {exc!r}") from exc


def unzip_project():
    if not PROJECT_ZIP.exists():
        raise FileNotFoundError(f"Project ZIP not found: {PROJECT_ZIP}")
    with zipfile.ZipFile(PROJECT_ZIP, "r") as zf:
        zf.extractall("/content")
    if not WORKDIR.exists():
        candidates = [p for p in Path("/content").glob("**/run_final_malcl_baseline.py") if "final_malcl_baseline_colab_run" in str(p)]
        if not candidates:
            raise FileNotFoundError("Could not detect AHR-MalCL project root after unzip")
        return candidates[0].parents[2]
    return WORKDIR


def ensure_datasets():
    CONTENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        "EMBER_Class_train.npz",
        "EMBER_Class_test.npz",
        "AZ_Class_Train.npz",
        "AZ_Class_Test.npz",
    ]
    for name in required:
        dst = CONTENT_DATA_DIR / name
        src = DRIVE_DATA_DIR / name
        if dst.exists():
            print(f"dataset already in /content/data: {dst}")
            continue
        if not src.exists():
            raise FileNotFoundError(f"Dataset missing from both /content/data and Drive: {name}")
        print(f"copying dataset to /content/data: {src} -> {dst}")
        shutil.copy2(src, dst)


def run_final_baseline(project_root: Path):
    runner = project_root / "paper_ready/final_malcl_baseline_colab_run/run_final_malcl_baseline.py"
    if not runner.exists():
        raise FileNotFoundError(f"Final MalCL runner not found: {runner}")
    DRIVE_OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "MALCL_EPOCHS": MALCL_EPOCHS,
            "MALCL_BATCH_SIZE": MALCL_BATCH_SIZE,
            "FAST_DEBUG": FAST_DEBUG,
            "ALLOW_SUBSAMPLING": ALLOW_SUBSAMPLING,
            "MALCL_GENERATOR_LOSS": MALCL_GENERATOR_LOSS,
            "MALCL_SAMPLE_SELECT": MALCL_SAMPLE_SELECT,
            "MALCL_K": MALCL_K,
            "REPLAY_REFRESH_EVERY_BATCH": REPLAY_REFRESH_EVERY_BATCH,
            "PRINT_TRAINING_PROGRESS": PRINT_TRAINING_PROGRESS,
            "PROGRESS_EVERY_N_BATCHES": PROGRESS_EVERY_N_BATCHES,
            "USE_TQDM_PROGRESS": USE_TQDM_PROGRESS,
            "TQDM_MININTERVAL": TQDM_MININTERVAL,
            "PYTHONUNBUFFERED": "1",
        }
    )
    argv = [
        str(runner),
        "--output-root",
        str(runner.parent),
        "--drive-output-zip",
        str(DRIVE_OUTPUT_ZIP),
    ]
    print("running in-process for notebook progress display:", " ".join([sys.executable, *argv]), flush=True)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    return_code = 0
    try:
        sys.argv = argv
        os.chdir(project_root)
        try:
            runpy.run_path(str(runner), run_name="__main__")
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return_code = exc.code
            elif exc.code in (None, ""):
                return_code = 0
            else:
                print(f"runner exited with non-integer SystemExit code: {exc.code!r}", flush=True)
                return_code = 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    if return_code not in (0, 2):
        raise RuntimeError(f"Final MalCL runner crashed with return code {return_code}")
    if not DRIVE_OUTPUT_ZIP.exists():
        raise FileNotFoundError(f"Expected output ZIP was not created: {DRIVE_OUTPUT_ZIP}")
    summary_path = runner.parent / "final_malcl_baseline_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"decision": "SUMMARY_MISSING", "completed_runs": 0, "expected_runs": 20}
    return return_code, summary


mount_drive()
project_root = unzip_project()
print(f"project_root: {project_root}")
ensure_datasets()
return_code, summary = run_final_baseline(project_root)
print("FINAL_MALCL_BASELINE_COLAB_RUN_COMPLETE")
print(f"decision: {summary.get('decision')}")
print(f"completed_runs: {summary.get('completed_runs')}/{summary.get('expected_runs', 20)}")
print(f"output_zip: {DRIVE_OUTPUT_ZIP}")
print(f"runner_return_code: {return_code}")
