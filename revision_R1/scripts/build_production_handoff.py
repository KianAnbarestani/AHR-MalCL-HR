from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from revision_r1.config import DATASETS
from scripts.production_workflow import (
    EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    build_frozen_run_plan,
    training_source_sha256,
)


PRODUCTION_BUILD_ID = "revision_R1_production_v1_4_1_20261028"
CODE_ZIP_FILENAME = "revision_R1_production_v1_4_1_20261028.zip"
DISTRIBUTED_NOTEBOOK_FILENAME = "Revision_R1_Production_Colab_v1_4_1_20261028.ipynb"
LEGACY_IDENTITY_FILENAME = "legacy_v1_2_first_run_identity.json"
AUTHENTIC_V1_2_ZIP_SHA256 = "2647c6bb51b3e3c28fd3009008f6f616deac5e73f725fd20e66aebbb23097d9c"
AUTHENTIC_V1_2_SOURCE_SHA256 = "bd175dcc27492f848459653536c63aa7bed8421bc16c60d7141fa359004e8660"
EXPECTED_SCHEDULE_SHA256 = "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212"

RELEASE_TESTS = (
    "conftest.py",
    "test_archives.py",
    "test_checkpoint_storage_policy.py",
    "test_cli_gates.py",
    "test_config_data_sampling.py",
    "test_coordinates.py",
    "test_live_progress.py",
    "test_package_import.py",
    "test_production_notebook.py",
    "test_production_workflow.py",
    "test_protocol_math.py",
    "test_registry_lock.py",
    "test_resources.py",
    "test_resume_checkpoint.py",
    "test_runner_dropout_resume.py",
    "test_runner_synthetic.py",
    "test_runtime_source_contracts.py",
    "test_task_stream_config.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell(kind: str, source: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def notebook_document(expected_zip_sha256: str) -> dict:
    if len(expected_zip_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_zip_sha256):
        raise ValueError("the standalone notebook must pin a lowercase SHA-256 for its exact production ZIP")
    cells = [
        _cell("markdown", f"""# revision_R1 production handoff — {PRODUCTION_BUILD_ID}

This notebook runs the frozen 100-run revision_R1 experiment only when the corresponding cell switch is explicitly changed to `True`. The first counted run (`ember`, seed 42, `full_hybrid`) and the remaining 99 runs are separate cells. Setup does not train or run pytest. The protocol and effective schedule are fixed; this is a new revision experiment, not a historical V15.5/V16 reproduction.

Run Cells A → B → C, then run Cell D for a read-only RAM/lease/checkpoint diagnosis and an exact v1.3 resume summary. Do not enable production Cell E until the report is reviewed and no other Colab worker owns the matrix lease. Cell E remains explicitly disabled by default and refuses a fresh first-run start. Cell F continues the matrix; Cell G exports compact results.
"""),
        _cell("code", f'''# Cell A — paths, Drive setup, exact package verification, runtime and source identity.
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, sys, tempfile, zipfile
from google.colab import auth, drive

DRIVE_PATHS = {{
    "package_zip": "/content/drive/MyDrive/ReVision/{CODE_ZIP_FILENAME}",
    "ember_data_root": "/content/drive/MyDrive/AHR_MalCL/datasets",
    "az_data_root": "/content/drive/MyDrive/AHR_MalCL/datasets",
    "preflight_path": "/content/drive/MyDrive/ReVision/revision_R1/results/preflight/preflight.json",
    "v62_recovery_report": "/content/drive/MyDrive/ReVision/revision_R1/results/validation/reports/resume_validation_2b4d8f5f1f29a938_20261026T142320Z_946699443.json",
    "output_root": "/content/drive/MyDrive/ReVision/revision_R1/production/results",
    "checkpoint_root": "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints",
    "dataset_cache_root": "/content/revision_R1_production/cache",
    "local_checkpoint_root": "/content/revision_R1_production/checkpoints",
    "export_root": "/content/drive/MyDrive/ReVision/revision_R1/production/exports",
    "checkpoint_backup_root": "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoint_backups",
}}
EXPECTED_SCHEDULE_SHA256 = "{EXPECTED_SCHEDULE_SHA256}"
EXPECTED_ZIP_SHA256 = "{expected_zip_sha256}"
EXPECTED_BUILD_ID = "{PRODUCTION_BUILD_ID}"
drive.mount("/content/drive", force_remount=False)
auth.authenticate_user()

ZIP_PATH = Path(DRIVE_PATHS["package_zip"]).expanduser()
SIDECAR_PATH = Path(str(ZIP_PATH) + ".sha256")
if not ZIP_PATH.is_file() or not SIDECAR_PATH.is_file():
    raise FileNotFoundError(f"Upload the exact production ZIP and .sha256 sidecar to MyDrive/ReVision: {{ZIP_PATH}}")
SIDECAR_ZIP_SHA256 = SIDECAR_PATH.read_text(encoding="utf-8").split()[0]
if SIDECAR_ZIP_SHA256 != EXPECTED_ZIP_SHA256:
    raise ValueError(f"Uploaded checksum sidecar does not match the notebook-pinned release: {{SIDECAR_ZIP_SHA256}}")
digest = hashlib.sha256()
with ZIP_PATH.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
ACTUAL_ZIP_SHA256 = digest.hexdigest()
if ACTUAL_ZIP_SHA256 != EXPECTED_ZIP_SHA256:
    raise ValueError(f"Production package checksum mismatch: {{ACTUAL_ZIP_SHA256}} != {{EXPECTED_ZIP_SHA256}}")

EXTRACTION_ROOT = Path(f"/content/revision_R1_production_package_{{ACTUAL_ZIP_SHA256[:12]}}")
def verify_extracted(root):
    manifest_path = root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("production_build_id") != EXPECTED_BUILD_ID:
        raise ValueError(f"Wrong production package identity: {{manifest.get('production_build_id')}}")
    for relative, item in manifest["files"].items():
        target = (root / relative).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise FileNotFoundError(f"Missing/unsafe package member: {{relative}}")
        if target.stat().st_size != item["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Package member hash/size mismatch: {{relative}}")
    if manifest.get("effective_schedule_sha256") != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("Package manifest schedule identity differs from the approved effective schedule")
    return manifest

if EXTRACTION_ROOT.exists():
    PACKAGE_MANIFEST = verify_extracted(EXTRACTION_ROOT)
else:
    staging = Path(tempfile.mkdtemp(prefix="revision_R1_production_extract_", dir="/content"))
    try:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            if archive.testzip() is not None:
                raise ValueError("Production ZIP has a corrupt member")
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe ZIP member: {{member.filename}}")
            archive.extractall(staging)
        PACKAGE_MANIFEST = verify_extracted(staging)
        os.replace(staging, EXTRACTION_ROOT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    PACKAGE_MANIFEST = verify_extracted(EXTRACTION_ROOT)

PROJECT = EXTRACTION_ROOT / "revision_R1"
os.chdir(PROJECT)
def is_old_revision_path(entry):
    resolved = Path(entry or ".").resolve()
    return (resolved == EXTRACTION_ROOT.resolve() or resolved == PROJECT.resolve()
            or (resolved.name == "revision_R1" and resolved.parent.name.startswith("revision_R1_production_package_")))
sys.path[:] = [entry for entry in sys.path if not is_old_revision_path(entry)]
sys.path.insert(0, str(PROJECT))
for module_name in list(sys.modules):
    if (module_name == "revision_r1" or module_name.startswith("revision_r1.")
            or module_name == "scripts" or module_name.startswith("scripts.")):
        del sys.modules[module_name]

import platform, torch
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Expected Colab Python 3.12, got {{platform.python_version()}}")
if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
    raise RuntimeError(f"Unsupported PyTorch version: {{torch.__version__}}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. No training started; choose a GPU runtime and rerun setup.")

from scripts.production_workflow import training_source_sha256
import revision_r1.runner as runner_module
import scripts.production_workflow as production_workflow_module
if not Path(runner_module.__file__).resolve().is_relative_to(PROJECT.resolve()):
    raise ImportError(f"Training code was imported outside the verified ZIP: {{runner_module.__file__}}")
if not Path(production_workflow_module.__file__).resolve().is_relative_to(PROJECT.resolve()):
    raise ImportError(f"Production orchestrator was imported outside the verified ZIP: {{production_workflow_module.__file__}}")
ACTUAL_TRAINING_SOURCE_SHA256 = training_source_sha256(PROJECT)
if ACTUAL_TRAINING_SOURCE_SHA256 != runner_module._code_sha256() or ACTUAL_TRAINING_SOURCE_SHA256 != PACKAGE_MANIFEST["training_source_sha256"]:
    raise ValueError("Actual production training-source identity differs from package manifest/runner hash")
print({{"build": EXPECTED_BUILD_ID, "zip_sha256": ACTUAL_ZIP_SHA256,
       "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
       "python": platform.python_version(), "torch": torch.__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}})
'''),
        _cell("code", f'''# Cell B — exact data paths and mounted-Drive preflight/effective-schedule verification.
from revision_r1.config import DATASETS
from revision_r1.validation_workflow import resolve_effective_artifacts

for dataset, root_key in (("ember", "ember_data_root"), ("az_class", "az_data_root")):
    root = Path(DRIVE_PATHS[root_key]).expanduser()
    missing = [DATASETS[dataset][part]["filename"] for part in ("train", "test")
               if not (root / DATASETS[dataset][part]["filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{{dataset}} exact dataset files missing below {{root}}: {{missing}}")

PREFLIGHT_PATH = Path(DRIVE_PATHS["preflight_path"]).expanduser()
if not PREFLIGHT_PATH.is_file():
    raise FileNotFoundError(f"Reuse the passed Drive preflight; not found: {{PREFLIGHT_PATH}}")
EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT = resolve_effective_artifacts(
    PREFLIGHT_PATH, expected_sha256=EXPECTED_SCHEDULE_SHA256, discover_manifest=True)

PACKAGED_SCHEDULE_PATH = PROJECT / "configs" / "update_budget.csv"
PACKAGED_SCHEDULE_MANIFEST_PATH = PROJECT / "configs" / "update_budget_manifest.json"
PACKAGED_SCHEDULE_SHA256 = hashlib.sha256(PACKAGED_SCHEDULE_PATH.read_bytes()).hexdigest()
PACKAGED_SCHEDULE_MANIFEST = json.loads(PACKAGED_SCHEDULE_MANIFEST_PATH.read_text(encoding="utf-8"))
if PACKAGED_SCHEDULE_SHA256 != PACKAGED_SCHEDULE_MANIFEST.get("source_csv_sha256"):
    raise ValueError("The original packaged schedule/manifest pair has been altered")
if PREFLIGHT.get("status") != "PREFLIGHT_VALIDATED_NO_TRAINING" or PREFLIGHT.get("training_reproduction_claimed") is not False:
    raise ValueError("Drive preflight is not a validated no-training schedule/data record")
print("Effective schedule:", EFFECTIVE_SCHEDULE_PATH)
print("Effective schedule SHA-256:", EXPECTED_SCHEDULE_SHA256, "| rows:", PREFLIGHT["schedule"]["row_count"])
print("Matching manifest:", EFFECTIVE_SCHEDULE_MANIFEST_PATH)
print("Original packaged schedule preserved at SHA-256:", PACKAGED_SCHEDULE_SHA256)
print("Exact EMBER/AZ-Class NPZ files exist. NPZ payloads were not reread in this preflight reuse cell.")
'''),
        _cell("code", '''# Cell C — freeze the 100-run plan and check first-run/full-matrix storage readiness.
import json
from scripts.production_workflow import (
    bind_runtime_schedule, build_frozen_run_plan, inspect_production_capacity,
    load_v62_recovery_checkpoint_bound, project_storage, require_run_capacity,
    validate_frozen_plan, validate_runtime_bound_plan,
)
from revision_r1.storage import StorageCapacityError

BASE_PLAN_PATH = PROJECT / "configs" / "production_run_plan_v1.json"
BASE_PLAN = json.loads(BASE_PLAN_PATH.read_text(encoding="utf-8"))
validate_frozen_plan(BASE_PLAN, expected_source_sha256=ACTUAL_TRAINING_SOURCE_SHA256)
RUNTIME_PLAN = bind_runtime_schedule(
    BASE_PLAN, EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT)
validate_runtime_bound_plan(
    RUNTIME_PLAN, EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT)
if RUNTIME_PLAN["run_count"] != 100 or RUNTIME_PLAN["runs"][0]["run_key"] != "ember_seed42_full_hybrid":
    raise ValueError("Runtime-bound matrix/order differs from the frozen first-run protocol")

V62_REFERENCE = load_v62_recovery_checkpoint_bound(DRIVE_PATHS["v62_recovery_report"])
STORAGE_PROJECTION = project_storage(
    RUNTIME_PLAN,
    observed_checkpoint_bytes=V62_REFERENCE["max_checkpoint_payload_bytes"],
    checkpoint_size_is_already_conservative_bound=True,
    checkpoint_reference_provenance=V62_REFERENCE["provenance"],
)
OUTPUT_ROOT = Path(DRIVE_PATHS["output_root"]).expanduser()
CHECKPOINT_ROOT = Path(DRIVE_PATHS["checkpoint_root"]).expanduser()
LOCAL_CHECKPOINT_ROOT = Path(DRIVE_PATHS["local_checkpoint_root"]).expanduser()
CACHE_ROOT = Path(DRIVE_PATHS["dataset_cache_root"]).expanduser()
DATA_ROOTS = {
    "ember": Path(DRIVE_PATHS["ember_data_root"]).expanduser(),
    "az_class": Path(DRIVE_PATHS["az_data_root"]).expanduser(),
}
CAPACITY = inspect_production_capacity(
    OUTPUT_ROOT, CHECKPOINT_ROOT, LOCAL_CHECKPOINT_ROOT, CACHE_ROOT,
    verified_dataset_cache_bytes=0,
)
print("Drive capacity evidence:", json.dumps({
    "quota_account_email": CAPACITY["quota_account_email"],
    "quota_status": CAPACITY["quota_status"],
    "quota_limit_bytes": CAPACITY["output_destination"]["quota_limit_bytes"],
    "quota_usage_bytes": CAPACITY["output_destination"]["quota_usage_bytes"],
    "available_bytes": CAPACITY["drive_available_bytes"],
    "quota_capacity_note": CAPACITY["quota_capacity_note"],
    "output_folder_owner": CAPACITY["output_destination"]["destination_owner"],
    "checkpoint_folder_owner": CAPACITY["checkpoint_destination"]["destination_owner"],
    "new_file_owner_verified": CAPACITY["new_file_owner_verified"],
    "write_capability_verified": CAPACITY["write_capability_verified"],
    "write_probe": CAPACITY["output_destination"]["write_probe"],
}, sort_keys=True, indent=2))
require_run_capacity(CAPACITY, STORAGE_PROJECTION, scope="first_run")

# Persist the runtime-bound plan immutably so a runtime reset resumes the same plan.
PLAN_PATH = OUTPUT_ROOT / "plans" / f"production_run_plan_{RUNTIME_PLAN['plan_sha256']}.json"
PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
PLAN_BYTES = (json.dumps(RUNTIME_PLAN, sort_keys=True, indent=2, allow_nan=False) + "\\n").encode("utf-8")
if PLAN_PATH.exists():
    if PLAN_PATH.read_bytes() != PLAN_BYTES:
        raise ValueError(f"Existing runtime plan has different bytes; preserved without overwrite: {{PLAN_PATH}}")
else:
    try:
        with PLAN_PATH.open("xb") as stream:
            stream.write(PLAN_BYTES)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if PLAN_PATH.read_bytes() != PLAN_BYTES:
            raise ValueError(f"Concurrent runtime plan differs; preserved without overwrite: {{PLAN_PATH}}")

drive_available = CAPACITY["drive_available_bytes"]
matrix_peak = int(STORAGE_PROJECTION["drive"]["matrix_peak_with_reserve_bytes"])
matrix_status = (
    "CAPACITY_UNKNOWN" if drive_available is None
    else "PROJECTED_FIT" if drive_available >= matrix_peak
    else "INSUFFICIENT_FOR_FULL_MATRIX"
)
matrix_deficit = None if drive_available is None else max(0, matrix_peak - drive_available)
print(json.dumps({
    "status": "FIRST_RUN_STORAGE_READY",
    "first_run": RUNTIME_PLAN["runs"][0]["run_key"], "run_count": RUNTIME_PLAN["run_count"],
    "plan_sha256": RUNTIME_PLAN["plan_sha256"], "training_source_sha256": RUNTIME_PLAN["training_source_sha256"],
    "schedule_sha256": RUNTIME_PLAN["schedule_sha256"], "runtime_plan_path": str(PLAN_PATH),
    "quota_account_email": CAPACITY["quota_account_email"],
    "output_folder_owner": CAPACITY["output_destination"]["destination_owner"],
    "checkpoint_folder_owner": CAPACITY["checkpoint_destination"]["destination_owner"],
    "drive_available_bytes": drive_available,
    "first_run_drive_peak_with_reserve_bytes": STORAGE_PROJECTION["drive"]["first_run_peak_with_reserve_bytes"],
    "first_run_local_peak_with_reserve_bytes": STORAGE_PROJECTION["local"]["first_run_peak_with_reserve_bytes"],
    "full_matrix_drive_peak_with_reserve_bytes": matrix_peak,
    "full_matrix_capacity_status_before_first_run": matrix_status,
    "full_matrix_capacity_deficit_bytes": matrix_deficit,
    "matrix_retained_bytes": STORAGE_PROJECTION["drive"]["matrix_retained_bytes"],
    "matrix_checkpoint_payload_lower_bound_bytes": STORAGE_PROJECTION["checkpoint_write_volume"]["matrix_retained_payload_lower_bound_bytes"],
    "safety_reserve_bytes": STORAGE_PROJECTION["safety_reserve_bytes"],
    "checkpoint_policy": "11 task-boundary + 2 recovery generations/run retained; no automatic pruning",
    "checkpoint_size_reference": V62_REFERENCE["provenance"],
    "production_measurements_available": False,
}, indent=2))
if matrix_status != "PROJECTED_FIT":
    print("Cell E remains capacity-gated. The displayed deficit already includes the 512 MiB reserve. Folder ownership is not required: the authenticated account must have verified write access, new-file ownership, and sufficient quota. If capacity is unknown, no numeric free-space value is inferred. Choose the final storage destination before Cell E; moving a completed run is unsupported. No checkpoints are deleted or pruned to force a fit.")
'''),
        _cell("code", '''# Cell D — read-only RAM, lock, progress, and checkpoint diagnosis; NO training or lock recovery.
import json, os, socket, time, uuid
from datetime import datetime, timezone
from revision_r1.data import inspect_npz_array_layout
from revision_r1.resources import read_host_memory_snapshot
from scripts.production_workflow import inspect_first_run_lease_and_progress

LEGACY_V12_IDENTITY_PATH = PROJECT / "configs" / "__LEGACY_IDENTITY_FILENAME__"
LEGACY_V12_IDENTITY = json.loads(LEGACY_V12_IDENTITY_PATH.read_text(encoding="utf-8"))
if LEGACY_V12_IDENTITY.get("zip_sha256") != "__AUTHENTIC_V1_2_ZIP_SHA256__" or LEGACY_V12_IDENTITY.get("training_source_sha256") != "__AUTHENTIC_V1_2_SOURCE_SHA256__":
    raise ValueError("Packaged v1.2 diagnostic baseline identity did not verify")
if not globals().get("COLAB_RUNTIME_INSTANCE_ID"):
    COLAB_RUNTIME_INSTANCE_ID = uuid.uuid4().hex

def _proc_start_ticks():
    try:
        raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        fields_after_comm = raw[raw.rfind(")") + 2:].split()
        return int(fields_after_comm[19])
    except (OSError, ValueError, IndexError):
        return None

def _proc_start_time_utc():
    try:
        raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        start_ticks = int(raw[raw.rfind(")") + 2:].split()[19])
        boot_line = next(line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        boot_unix = int(boot_line.split()[1])
        return datetime.fromtimestamp(boot_unix + start_ticks / os.sysconf("SC_CLK_TCK"), timezone.utc).isoformat()
    except (OSError, ValueError, IndexError, StopIteration, AttributeError):
        return None

RUNTIME_FACTS = {
    "runtime_instance_id": COLAB_RUNTIME_INSTANCE_ID,
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "pid": os.getpid(), "hostname": socket.gethostname(),
    "process_start_ticks": _proc_start_ticks(), "process_start_time_utc": _proc_start_time_utc(),
    "python_executable": sys.executable, "python_version": sys.version,
    "memory": read_host_memory_snapshot(),
    "oom_score": Path("/proc/self/oom_score").read_text().strip() if Path("/proc/self/oom_score").is_file() else None,
    "memory_pressure": Path("/proc/pressure/memory").read_text().strip() if Path("/proc/pressure/memory").is_file() else None,
    "remote_colab_runtime_visibility": "UNAVAILABLE; inspect Colab Manage Sessions and all open runtimes manually",
}

DATASET_LAYOUTS = {}
for _dataset, _root_key in (("ember", "ember_data_root"), ("az_class", "az_data_root")):
    _root = Path(DRIVE_PATHS[_root_key]).expanduser()
    _layout = {}
    for _part in ("train", "test"):
        _npz = _root / DATASETS[_dataset][_part]["filename"]
        _layout[_part] = inspect_npz_array_layout(_npz, split=_part)
    DATASET_LAYOUTS[_dataset] = _layout

from scripts.live_progress import BoundedProgressBuffer
PROGRESS_HANDLE = None
PROGRESS_BUFFER = BoundedProgressBuffer(max_events=5)
def live_progress(record):
    global PROGRESS_HANDLE
    text = PROGRESS_BUFFER.append(record)
    try:
        from IPython.display import display
        if PROGRESS_HANDLE is None:
            PROGRESS_HANDLE = display(text, display_id=True)
        else:
            PROGRESS_HANDLE.update(text)
    except Exception:
        print(text, flush=True)

def _diagnostic_progress(event):
    live_progress(event)

LEASE_RAM_DIAGNOSTIC = inspect_first_run_lease_and_progress(
    RUNTIME_PLAN, preflight=PREFLIGHT, output_root=OUTPUT_ROOT,
    checkpoint_root=CHECKPOINT_ROOT,
    legacy_first_run_record=LEGACY_V12_IDENTITY["first_run_record"],
    legacy_build_identity=LEGACY_V12_IDENTITY,
    current_runtime=RUNTIME_FACTS, progress_callback=_diagnostic_progress,
)
LEASE_RAM_DIAGNOSTIC.update({
    "production_build_id": EXPECTED_BUILD_ID,
    "production_training_source_sha256": RUNTIME_PLAN["training_source_sha256"],
    "schedule_sha256": RUNTIME_PLAN["schedule_sha256"],
    "plan_sha256": RUNTIME_PLAN["plan_sha256"],
    "dataset_npz_layout_headers_only": DATASET_LAYOUTS,
    "historical_oom_correlation": "UNKNOWN: this runtime snapshot cannot reconstruct the earlier failed process allocation.",
})
HISTORICAL_OOM_EVIDENCE = {
    "source": "user-provided earlier Colab kernel log excerpt",
    "cgroup_oom_kill_confirmed": True,
    "pid": 44230,
    "anon_rss_kib": 9534112,
    "total_vm_kib": 23506504,
    "timestamp_utc": None,
    "interpretation": "anon-rss is resident memory; total-vm is virtual address space, not physical RAM",
}
def _pid_matches(value, path="$", parent=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key.lower() in {"pid", "process_id", "worker_pid", "runtime_pid"}:
                try:
                    if int(item) == HISTORICAL_OOM_EVIDENCE["pid"]:
                        found.append({"path": child, "pid": int(item)})
                except (TypeError, ValueError):
                    pass
            found.extend(_pid_matches(item, child, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_pid_matches(item, f"{path}[{index}]", parent))
    return found

_legacy_namespace = LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]
_persisted_records = {}
_pid_record_matches = []
for _record_name in ("run_status_json", "run_registry_status", "progress", "preparation_progress"):
    _record_info = _legacy_namespace[_record_name]
    _record = _record_info.get("record")
    _persisted_records[_record_name] = {
        "path": _record_info.get("path"), "exists": _record_info.get("exists", False),
        "mtime_utc": datetime.fromtimestamp(_record_info["mtime_unix"], timezone.utc).isoformat()
        if isinstance(_record_info.get("mtime_unix"), (int, float)) else None,
        "error": _record_info.get("error"),
    }
    if _record is not None:
        _pid_record_matches.extend({"record": _record_name, **row} for row in _pid_matches(_record))
LEASE_RAM_DIAGNOSTIC["historical_oom_evidence"] = HISTORICAL_OOM_EVIDENCE
LEASE_RAM_DIAGNOSTIC["historical_oom_log_correlation"] = {
    "available_persisted_v1_2_run_records": _persisted_records,
    "matching_pid_records": _pid_record_matches,
    "status": "PID_MATCHED_IN_PERSISTED_RUN_RECORDS" if _pid_record_matches else "NO_PID_MATCH_IN_PERSISTED_RUN_RECORDS",
    "time_correlation": "UNAVAILABLE: supplied kernel-log excerpt has no timestamp; persisted record modification times are listed above",
    "does_not_establish": "whether an earlier attempt entered training or the exact allocation that triggered the cgroup kill",
}
diagnostic_dir = OUTPUT_ROOT / "diagnostics"
diagnostic_dir.mkdir(parents=True, exist_ok=True)
LEASE_DIAGNOSTIC_PATH = diagnostic_dir / f"lease_ram_diagnostic_{RUNTIME_PLAN['plan_sha256'][:12]}_{time.time_ns()}.json"
with LEASE_DIAGNOSTIC_PATH.open("x", encoding="utf-8") as _stream:
    json.dump(LEASE_RAM_DIAGNOSTIC, _stream, sort_keys=True, indent=2, default=str, allow_nan=False)
    _stream.flush()
    os.fsync(_stream.fileno())
print("READ-ONLY DIAGNOSTIC SAVED:", LEASE_DIAGNOSTIC_PATH)
CURRENT_V13_NAMESPACE = next(
    row for row in LEASE_RAM_DIAGNOSTIC["run_namespaces"]
    if row["identity_label"] == "current_package_identity"
)
print("FIRST-RUN RESUME SUMMARY (read-only):")
print(json.dumps({
    "run_key": CURRENT_V13_NAMESPACE["run_key"],
    "source_sha256": CURRENT_V13_NAMESPACE["training_source_sha256"],
    "schedule_sha256": CURRENT_V13_NAMESPACE["schedule_sha256"],
    "config_sha256": CURRENT_V13_NAMESPACE["config_sha256"],
    "registry": CURRENT_V13_NAMESPACE["run_registry_status"],
    "last_progress": CURRENT_V13_NAMESPACE["progress"],
    "resume_gate": CURRENT_V13_NAMESPACE["first_run_resume_gate"],
}, sort_keys=True, indent=2, default=str))
print(json.dumps({
    "matrix_lease": LEASE_RAM_DIAGNOSTIC["matrix_lease"],
    "v1_2_run_lease": LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]["run_lease"],
    "v1_2_run_status": LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]["run_status_json"],
    "v1_2_progress": LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]["progress"],
    "v1_2_recovery_checkpoints": LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]["recovery_checkpoints"],
    "v1_2_task_boundary_manifest_count": LEASE_RAM_DIAGNOSTIC["run_namespaces"][0]["task_boundary_manifest_count"],
    "legacy_v1_2_matrix_registry": LEASE_RAM_DIAGNOSTIC["legacy_v1_2_matrix_registry"],
    "current_v1_3_matrix_registry": LEASE_RAM_DIAGNOSTIC["matrix_registry"],
    "current_v1_3_first_run_resume_gate": CURRENT_V13_NAMESPACE["first_run_resume_gate"],
    "matrix_registry_paths": LEASE_RAM_DIAGNOSTIC["matrix_registry_paths"],
    "runtime_memory": RUNTIME_FACTS["memory"],
    "historical_oom": LEASE_RAM_DIAGNOSTIC["historical_oom"],
    "historical_oom_log_correlation": LEASE_RAM_DIAGNOSTIC["historical_oom_log_correlation"],
}, sort_keys=True, indent=2, default=str))
print("NO lock was recovered/deleted. NO training was launched.")
'''.replace("__LEGACY_IDENTITY_FILENAME__", LEGACY_IDENTITY_FILENAME)
        .replace("__AUTHENTIC_V1_2_ZIP_SHA256__", AUTHENTIC_V1_2_ZIP_SHA256)
        .replace("__AUTHENTIC_V1_2_SOURCE_SHA256__", AUTHENTIC_V1_2_SOURCE_SHA256)),
        _cell("code", '''# Cell E — first counted production run only; gated on explicit lease review.
import json, time
LAUNCH_FIRST_PRODUCTION_RUN = False
ACKNOWLEDGE_NO_OTHER_COLAB_WORKER = False

if not LAUNCH_FIRST_PRODUCTION_RUN:
    print("NOT_LAUNCHED: run Cell D, verify the exact v1.3 first-run recovery gate is PASS_RESUME_CANDIDATE, confirm no other Colab worker, and only then explicitly enable both switches.")
else:
    if ACKNOWLEDGE_NO_OTHER_COLAB_WORKER is not True:
        raise RuntimeError("Production blocked: explicitly confirm in Colab Manage Sessions that no other worker/runtime is active.")
    if not Path(LEASE_DIAGNOSTIC_PATH).is_file() or LEASE_RAM_DIAGNOSTIC.get("diagnostic_mode") != "READ_ONLY_NO_LOCK_RECOVERY_NO_TRAINING":
        raise RuntimeError("Production blocked: run the read-only Cell D diagnostic in this runtime first.")
    if time.time() - float(Path(LEASE_DIAGNOSTIC_PATH).stat().st_mtime) > 1800:
        raise RuntimeError("Production blocked: Cell D diagnostic is older than 30 minutes; rerun Cell D and re-review.")
    _resume_gate = CURRENT_V13_NAMESPACE.get("first_run_resume_gate", {})
    if _resume_gate.get("status") != "PASS_RESUME_CANDIDATE":
        raise RuntimeError("Production is resume-only: Cell D did not verify an exact v1.3 recovery checkpoint; this cell never initializes from scratch. Refusing a fresh start: " + str(_resume_gate.get("reason")))
    _last_recorded_progress = (CURRENT_V13_NAMESPACE.get("progress", {}).get("record") or {})
    print("RESUMING EXISTING RUN — checkpoint preflight passed; runner deserialization is next; no fresh initialization is allowed")
    print(json.dumps({
        "run_key": CURRENT_V13_NAMESPACE["run_key"],
        "dataset": "EMBER", "arm": "Full Hybrid", "seed": 42,
        "config_sha256": CURRENT_V13_NAMESPACE["config_sha256"],
        "training_source_sha256": CURRENT_V13_NAMESPACE["training_source_sha256"],
        "schedule_sha256": CURRENT_V13_NAMESPACE["schedule_sha256"],
        "newest_integrity_checked_drive_candidate": _resume_gate.get("selected_integrity_verified_candidate"),
        "fallback_from_newest_integrity_checked_candidate": _resume_gate.get("fallback_from_newest_integrity_verified_candidate"),
        "runner_checkpoint_deserialization_pending": _resume_gate.get("runner_deserialization_pending"),
        "last_recorded_progress_cursor_informational_only": _last_recorded_progress.get("cursor"),
        "last_recorded_progress_counters_informational_only": {
            key: _last_recorded_progress.get(key) for key in ("classifier_updates", "generator_updates", "critic_steps")
        },
        "last_verified_drive_checkpoint_in_last_progress": _last_recorded_progress.get("last_verified_drive_checkpoint"),
        "cursor_note": _resume_gate.get("checkpoint_cursor_note"),
    }, sort_keys=True, indent=2, default=str))
    fresh_matrix_lease = production_workflow_module.inspect_lease_readonly(OUTPUT_ROOT / ".matrix_execution.lease.json")
    fresh_leases = [fresh_matrix_lease]
    for _namespace in LEASE_RAM_DIAGNOSTIC["run_namespaces"]:
        _lease_path = _namespace["run_lease"]["path"]
        fresh_leases.append(production_workflow_module.inspect_lease_readonly(_lease_path))
    _blocking = [item for item in fresh_leases if item["status"] in {"live", "malformed", "unreadable", "expired_but_owner_process_may_be_alive"}]
    if _blocking:
        raise RuntimeError("Production blocked by unresolved lease status: " + ", ".join(item["status"] + " at " + item["path"] for item in _blocking) + "; preserve it and resolve ownership first.")
    FIRST_RUN_RESULT = production_workflow_module.execute_first_production_run(
        RUNTIME_PLAN, preflight=PREFLIGHT,
        schedule_path=EFFECTIVE_SCHEDULE_PATH,
        schedule_manifest_path=EFFECTIVE_SCHEDULE_MANIFEST_PATH,
        v62_recovery_report_path=DRIVE_PATHS["v62_recovery_report"],
        data_roots=DATA_ROOTS, cache_root=CACHE_ROOT,
        output_root=OUTPUT_ROOT, checkpoint_root=CHECKPOINT_ROOT,
        local_checkpoint_root=LOCAL_CHECKPOINT_ROOT,
        launch_enabled=LAUNCH_FIRST_PRODUCTION_RUN,
        progress_callback=live_progress,
    )
    print(json.dumps(FIRST_RUN_RESULT, sort_keys=True, indent=2, default=str))
'''),
        _cell("code", '''# Cell F — continue/resume runs 2–100; run 1 is validated and excluded by the executor.
import json, time
LAUNCH_REMAINING_MATRIX = False
ACKNOWLEDGE_NO_OTHER_COLAB_WORKER = False

if not LAUNCH_REMAINING_MATRIX:
    print("NOT_LAUNCHED: only after the first run is validated, full-matrix capacity passes, and a fresh read-only lease review confirms no other worker.")
else:
    if ACKNOWLEDGE_NO_OTHER_COLAB_WORKER is not True:
        raise RuntimeError("Matrix continuation blocked: explicitly confirm no other Colab worker/runtime in Manage Sessions.")
    if not Path(LEASE_DIAGNOSTIC_PATH).is_file() or time.time() - Path(LEASE_DIAGNOSTIC_PATH).stat().st_mtime > 1800:
        raise RuntimeError("Matrix continuation blocked: rerun Cell D and review a fresh lease diagnostic.")
    fresh_matrix_lease = production_workflow_module.inspect_lease_readonly(OUTPUT_ROOT / ".matrix_execution.lease.json")
    fresh_leases = [fresh_matrix_lease]
    for _namespace in LEASE_RAM_DIAGNOSTIC["run_namespaces"]:
        fresh_leases.append(production_workflow_module.inspect_lease_readonly(_namespace["run_lease"]["path"]))
    _blocking = [item for item in fresh_leases if item["status"] in {"live", "malformed", "unreadable", "expired_but_owner_process_may_be_alive"}]
    if _blocking:
        raise RuntimeError("Matrix continuation blocked by unresolved lease status: " + ", ".join(item["status"] + " at " + item["path"] for item in _blocking))
    MATRIX_RESULT = production_workflow_module.continue_production_matrix(
        RUNTIME_PLAN, preflight=PREFLIGHT,
        schedule_path=EFFECTIVE_SCHEDULE_PATH,
        schedule_manifest_path=EFFECTIVE_SCHEDULE_MANIFEST_PATH,
        data_roots=DATA_ROOTS, cache_root=CACHE_ROOT,
        output_root=OUTPUT_ROOT, checkpoint_root=CHECKPOINT_ROOT,
        local_checkpoint_root=LOCAL_CHECKPOINT_ROOT,
        launch_enabled=LAUNCH_REMAINING_MATRIX,
        progress_callback=live_progress,
    )
    print(json.dumps(MATRIX_RESULT, sort_keys=True, indent=2, default=str))
'''),
        _cell("code", '''# Cell G — compact progress + lightweight results; full checkpoint backup is separate and OFF by default.
import json, os, time
from scripts.production_workflow import export_production_progress

EXPORT_CHECKPOINT_BACKUP = False
EXPORT_ROOT = Path(DRIVE_PATHS["export_root"]).expanduser()
CHECKPOINT_BACKUP_ROOT = Path(DRIVE_PATHS["checkpoint_backup_root"]).expanduser()
progress_snapshot = production_workflow_module.inspect_drive_destination(EXPORT_ROOT)
from revision_r1.storage import require_available_bytes, DEFAULT_STORAGE_RESERVE_BYTES
require_available_bytes(EXPORT_ROOT, required_bytes=1 << 20,
    reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
    available_bytes=progress_snapshot["available_bytes"], storage_kind="production progress export Drive")
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
PROGRESS_REPORT = export_production_progress(RUNTIME_PLAN, OUTPUT_ROOT)
PROGRESS_PATH = EXPORT_ROOT / f"production_progress_{RUNTIME_PLAN['plan_sha256'][:12]}_{time.time_ns()}.json"
with PROGRESS_PATH.open("x", encoding="utf-8") as stream:
    json.dump(PROGRESS_REPORT, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())
print("Progress export:", PROGRESS_PATH)
print("Registered completions (validated at registration; not rehashed by progress export):",
      PROGRESS_REPORT["validated_completion_count"], "/ 100")

if PROGRESS_REPORT["validated_completion_count"]:
    results_snapshot = production_workflow_module.inspect_drive_destination(EXPORT_ROOT)
    RESULTS_PATH = EXPORT_ROOT / f"revision_R1_lightweight_results_{RUNTIME_PLAN['plan_sha256'][:12]}_{time.time_ns()}.zip"
    RESULTS_EXPORT = production_workflow_module.export_lightweight_results(
        RUNTIME_PLAN, OUTPUT_ROOT, RESULTS_PATH, storage_snapshot=results_snapshot)
    print("Lightweight results only (model/checkpoint payloads excluded):", json.dumps(RESULTS_EXPORT, sort_keys=True))
else:
    print("No registered completion yet; lightweight result archive not created.")

if EXPORT_CHECKPOINT_BACKUP:
    backup_snapshot = production_workflow_module.inspect_drive_destination(CHECKPOINT_BACKUP_ROOT)
    BACKUP_PATH = CHECKPOINT_BACKUP_ROOT / f"revision_R1_checkpoint_backup_{RUNTIME_PLAN['plan_sha256'][:12]}_{time.time_ns()}.zip"
    BACKUP_EXPORT = production_workflow_module.export_checkpoint_backup(
        RUNTIME_PLAN, CHECKPOINT_ROOT, BACKUP_PATH, storage_snapshot=backup_snapshot)
    print("Verified checkpoint archive:", json.dumps(BACKUP_EXPORT, sort_keys=True))
else:
    print("Full checkpoint archive: NOT_REQUESTED (no checkpoint files copied).")
print("No completion files or checkpoints were deleted. Reset loss is bounded by the latest verified Drive synchronization, not guaranteed zero.")
'''),
    ]
    document = {
        "cells": cells,
        "metadata": {
            "colab": {"name": DISTRIBUTED_NOTEBOOK_FILENAME, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for item in cells:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
    return document


def _package_source_paths(revision_root: Path) -> list[Path]:
    files = [*sorted((revision_root / "revision_r1").glob("*.py"))]
    files.extend([
        revision_root / "scripts" / "live_progress.py",
        revision_root / "scripts" / "production_workflow.py",
        revision_root / "scripts" / "build_production_handoff.py",
        revision_root / "PRODUCTION_COLAB_README.md",
        revision_root / "requirements-colab.txt",
        revision_root / "legacy_checkpoint_source_compatibility.json",
        revision_root / "configs" / "update_budget.csv",
        revision_root / "configs" / "update_budget_manifest.json",
    ])
    files.extend(revision_root / "tests" / name for name in RELEASE_TESTS)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"production package source files are missing: {missing}")
    return sorted(set(files))


def _package_sources(revision_root: Path, expected_plan: dict) -> list[Path]:
    files = _package_source_paths(revision_root)
    generated = [revision_root / "configs" / "production_run_plan_v1.json"]
    legacy_identity = revision_root / "configs" / LEGACY_IDENTITY_FILENAME
    if legacy_identity.is_file():
        generated.append(legacy_identity)
    files.extend(generated)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"generated production package files are missing: {missing}")
    observed_plan = json.loads(generated[0].read_text(encoding="utf-8"))
    if observed_plan != expected_plan:
        raise ValueError("generated production run plan changed before packaging")
    if training_source_sha256(revision_root) != expected_plan["training_source_sha256"]:
        raise ValueError("production training-source hash changed before packaging")
    return sorted(set(files))


def _authenticate_v12_baseline(repository_root: Path) -> dict:
    """Recover lock-diagnostic identity only from the verified distributed v1.2 artifact."""
    old_root = repository_root / "revision_R1" / "dist" / "production_v1_2_handoff_20261027"
    zip_path = old_root / "revision_R1_production_v1_2_20261027.zip"
    sidecar_path = Path(str(zip_path) + ".sha256")
    build_path = old_root / "BUILD_MANIFEST.json"
    plan_path = old_root / "production_run_plan_v1.json"
    for required in (zip_path, sidecar_path, build_path, plan_path):
        if not required.is_file():
            raise FileNotFoundError(f"authentic v1.2 baseline artifact is missing: {required}")
    observed_zip_sha = sha256_file(zip_path)
    sidecar_sha = sidecar_path.read_text(encoding="utf-8").split()[0]
    build = json.loads(build_path.read_text(encoding="utf-8"))
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if (
        observed_zip_sha != AUTHENTIC_V1_2_ZIP_SHA256
        or sidecar_sha != AUTHENTIC_V1_2_ZIP_SHA256
        or build.get("production_build_id") != "revision_R1_production_v1_2_20261027"
        or build.get("training_source_sha256") != AUTHENTIC_V1_2_SOURCE_SHA256
        or build.get("code_zip", {}).get("sha256") != AUTHENTIC_V1_2_ZIP_SHA256
        or build.get("plan_manifest_sha256") != sha256_bytes(plan_bytes)
        or plan.get("training_source_sha256") != AUTHENTIC_V1_2_SOURCE_SHA256
        or plan.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
        or plan.get("plan_sha256") != build.get("plan_sha256")
        or not plan.get("runs")
    ):
        raise ValueError("v1.2 ZIP/sidecar/build manifest/plan identity does not match the authenticated baseline")
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("authentic v1.2 distributed ZIP has a corrupt member")
        package_manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        if (
            package_manifest.get("production_build_id") != "revision_R1_production_v1_2_20261027"
            or package_manifest.get("training_source_sha256") != AUTHENTIC_V1_2_SOURCE_SHA256
            or package_manifest.get("effective_schedule_sha256") != EXPECTED_SCHEDULE_SHA256
        ):
            raise ValueError("authentic v1.2 ZIP package manifest identity is inconsistent")
        for relative, item in package_manifest.get("files", {}).items():
            try:
                content = archive.read(relative)
            except KeyError as exc:
                raise ValueError(f"authentic v1.2 ZIP is missing manifest member {relative}") from exc
            if len(content) != item.get("bytes") or sha256_bytes(content) != item.get("sha256"):
                raise ValueError(f"authentic v1.2 ZIP member hash/size mismatch: {relative}")
        core_hash = hashlib.sha256()
        core_hashes = []
        names = sorted(
            name for name in package_manifest["files"]
            if name.startswith("revision_R1/revision_r1/") and name.endswith(".py")
        )
        for name in names:
            content = archive.read(name)
            leaf = PurePosixPath(name).name
            core_hash.update(leaf.encode())
            core_hash.update(content)
            core_hashes.append({"filename": leaf, "sha256": sha256_bytes(content)})
        if core_hash.hexdigest() != AUTHENTIC_V1_2_SOURCE_SHA256:
            raise ValueError("v1.2 training-core hashes do not reproduce its authenticated source identity")
        packaged_plan = json.loads(archive.read("revision_R1/configs/production_run_plan_v1.json"))
        if packaged_plan != plan:
            raise ValueError("v1.2 top-level plan differs from the plan inside the authenticated ZIP")
    first = plan["runs"][0]
    if (
        first.get("run_key") != "ember_seed42_full_hybrid"
        or first.get("training_source_sha256") != AUTHENTIC_V1_2_SOURCE_SHA256
        or first.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
    ):
        raise ValueError("authentic v1.2 first-run identity is not the approved run/schedule")
    return {
        "schema_version": 1,
        "source": "authentic distributed v1.2 ZIP, checksum sidecar, build manifest, and included run plan",
        "production_build_id": "revision_R1_production_v1_2_20261027",
        "zip_filename": zip_path.name, "zip_sha256": observed_zip_sha,
        "training_source_sha256": AUTHENTIC_V1_2_SOURCE_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "first_run_record": first,
        "verified_training_core_file_count": len(core_hashes),
        "verified_training_core_file_hashes": core_hashes,
        "automatic_resume_compatibility": "INSPECT_ONLY; not asserted compatible with a different source hash",
    }


def _base_plan(revision_root: Path) -> dict:
    references = {
        name: {
            "train_md5": DATASETS[name]["train"]["md5"],
            "test_md5": DATASETS[name]["test"]["md5"],
            "feature_dim": int(DATASETS[name]["feature_dim"]),
            "train_shape": list(DATASETS[name]["train"]["shape"]),
            "test_shape": list(DATASETS[name]["test"]["shape"]),
        }
        for name in ("ember", "az_class")
    }
    return build_frozen_run_plan(
        revision_root, schedule_sha256=EXPECTED_SCHEDULE_SHA256,
        dataset_references=references,
    )


def _zip_members(revision_root: Path, files: list[Path], plan: dict) -> tuple[list[tuple[Path, str]], dict]:
    records = {}
    mapped = []
    for source in files:
        relative = source.relative_to(revision_root.parent).as_posix()
        mapped.append((source, relative))
        records[relative] = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    schedule_manifest = json.loads((revision_root / "configs/update_budget_manifest.json").read_text(encoding="utf-8"))
    legacy_identity = json.loads((revision_root / "configs" / LEGACY_IDENTITY_FILENAME).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "production_build_id": PRODUCTION_BUILD_ID,
        "training_reproduction_claimed": False,
        "training_source_sha256": plan["training_source_sha256"],
        "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "packaged_parent_schedule_sha256": schedule_manifest["source_csv_sha256"],
        "run_count": 100,
        "first_run": {"dataset": "ember", "seed": 42, "arm": "full_hybrid"},
        "production_matrix_enabled_by_default": False,
        "distributed_notebook_filename": DISTRIBUTED_NOTEBOOK_FILENAME,
        "legacy_validation_identity_override_used_for_production": False,
        "v62_recovery_reference": {
            "build_id": "revision_R1_colab_validation_v6_2_final_gan_recovery_qa",
            "workflow_id": "2b4d8f5f1f29a938",
            "report_name": "resume_validation_2b4d8f5f1f29a938_20261026T142320Z_946699443.json",
            "used_as_checkpoint_size_proxy_only": True,
            "gpu_recovery_reexecuted_during_build": False,
        },
        "checkpoint_retention": {
            "task_boundary_generations_per_run": 11,
            "recovery_generations_per_run": 2,
            "automatic_pruning": False,
            "runtime_reset_can_lose_work_since_last_verified_drive_sync": True,
        },
        "host_ram_safety": {
            "npz_cache_population": "stream compressed NPZ NPY members to verified memmap cache in 8 MiB chunks",
            "feature_scans_and_training_transforms": "bounded row/chunk processing; no full feature-matrix float64 cast",
            "checkpoint_readback": "preflight host/cgroup RAM headroom; release the first decoded state before a separate published-payload integrity reload",
            "host_ram_reserve_bytes": 1073741824,
            "matrix_lease_acquired_before_dataset_load": True,
            "memory_telemetry": "host available/RSS and cgroup current/limit/peak/events in preparation progress and run resource metrics; Colab diagnostic records process-start UTC",
        },
        "legacy_v1_2_checkpoint_handling": {
            "authentic_v1_2_zip_sha256": legacy_identity["zip_sha256"],
            "authentic_v1_2_training_source_sha256": legacy_identity["training_source_sha256"],
            "v1_2_training_core_file_count_verified": legacy_identity["verified_training_core_file_count"],
            "automatic_resume_from_v1_2_checkpoint": False,
            "reason": "the memory-bounded training implementation has a different source identity; any v1.2 checkpoint is reported and preserved pending explicit compatibility review",
        },
        "lease_diagnostic": {
            "read_only": True, "lock_recovery_performed": False,
            "remote_colab_process_visibility": "unavailable from a different runtime; user must inspect Colab Manage Sessions",
        },
        "included_tests": list(RELEASE_TESTS),
        "files": records,
    }
    return mapped, manifest


def build_handoff(repository_root: str | Path, output_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    revision_root = repository / "revision_R1"
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite a non-empty production release directory: {output}")
        raise FileExistsError(f"refusing to reuse an existing release directory: {output}")
    legacy_identity = _authenticate_v12_baseline(repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    package_root = staging / "revision_R1"
    package_root.mkdir()
    (package_root / "configs").mkdir()
    try:
        for source in _package_source_paths(revision_root):
            destination = package_root / source.relative_to(revision_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        plan = _base_plan(revision_root)
        plan_path = staging / "production_run_plan_v1.json"
        plan_bytes = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        plan_path.write_bytes(plan_bytes)
        shutil.copyfile(plan_path, package_root / "configs" / "production_run_plan_v1.json")
        legacy_identity_path = package_root / "configs" / LEGACY_IDENTITY_FILENAME
        legacy_identity_path.write_text(json.dumps(legacy_identity, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        shutil.copyfile(revision_root / "PRODUCTION_COLAB_README.md", staging / "PRODUCTION_COLAB_README.md")
        shutil.copyfile(revision_root / "PRODUCTION_COLAB_README.md", package_root / "PRODUCTION_COLAB_README.md")

        files = _package_sources(package_root, plan)
        mapped, package_manifest = _zip_members(package_root, files, plan)
        code_zip = staging / CODE_ZIP_FILENAME
        with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source, relative in mapped:
                archive.write(source, relative)
            archive.writestr("PACKAGE_MANIFEST.json", json.dumps(package_manifest, sort_keys=True, indent=2) + "\n")
        zip_digest = sha256_file(code_zip)
        (staging / f"{CODE_ZIP_FILENAME}.sha256").write_text(f"{zip_digest}  {CODE_ZIP_FILENAME}\n", encoding="utf-8")
        notebook_path = staging / DISTRIBUTED_NOTEBOOK_FILENAME
        notebook_bytes = (json.dumps(notebook_document(zip_digest), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        notebook_path.write_bytes(notebook_bytes)
        notebook_digest = sha256_bytes(notebook_bytes)
        (staging / f"{DISTRIBUTED_NOTEBOOK_FILENAME}.sha256").write_text(
            f"{notebook_digest}  {DISTRIBUTED_NOTEBOOK_FILENAME}\n", encoding="utf-8",
        )
        zip_manifest = {
            "production_build_id": PRODUCTION_BUILD_ID,
            "archive_sha256": zip_digest, "archive_bytes": code_zip.stat().st_size,
            "package_manifest_sha256": sha256_bytes(json.dumps(package_manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"),
            "files": package_manifest["files"],
        }
        (staging / f"{CODE_ZIP_FILENAME}.manifest.json").write_text(json.dumps(zip_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        qa_report = """# Production handoff QA\n\nStatus: `PENDING_EXTRACTED_PACKAGE_QA`. Production, Colab CUDA, Drive recovery, and production matrix checks are `NOT_RUN`.\n"""
        (staging / "QA_REPORT.md").write_text(qa_report, encoding="utf-8")
        result = {
            "schema_version": 1, "production_build_id": PRODUCTION_BUILD_ID,
            "production_matrix_launched": False, "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "training_source_sha256": plan["training_source_sha256"],
            "plan_sha256": plan["plan_sha256"], "plan_manifest_sha256": sha256_bytes(plan_bytes),
            "code_zip": {"filename": CODE_ZIP_FILENAME, "sha256": zip_digest, "bytes": code_zip.stat().st_size},
            "notebook": {"filename": DISTRIBUTED_NOTEBOOK_FILENAME, "sha256": notebook_digest},
            "included_tests": list(RELEASE_TESTS), "qa_status": "PENDING_EXTRACTED_PACKAGE_QA",
            "production_gpu_drive_status": "NOT_RUN",
        }
        (staging / "BUILD_MANIFEST.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"release destination appeared during build; preserved without replacement: {output}")
        os.rename(staging, output)
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def finalize_handoff_qa(output_dir: str | Path, evidence: dict) -> dict:
    """Finalize only external release QA metadata after extracted-ZIP tests pass."""
    output = Path(output_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    code_zip = output / CODE_ZIP_FILENAME
    notebook = output / DISTRIBUTED_NOTEBOOK_FILENAME
    if not output.is_dir() or not code_zip.is_file() or not notebook.is_file():
        raise FileNotFoundError("release directory is incomplete; QA metadata was not changed")
    extracted = Path(evidence.get("extracted_package_root", "")).expanduser().resolve()
    extracted_project = extracted / "revision_R1"
    if not extracted_project.is_dir() or extracted_project.is_relative_to(repository):
        raise ValueError("QA must use a clean extracted ZIP outside the repository")
    suite = evidence.get("extracted_package_test_suite")
    scale = evidence.get("scale_memory_test")
    if not isinstance(suite, dict) or suite.get("status") != "PASS":
        raise ValueError("extracted package test suite has not passed")
    if type(suite.get("failed")) is not int or suite["failed"] != 0 or type(suite.get("passed")) is not int or suite["passed"] <= 0:
        raise ValueError("extracted package test counts are absent or include failures")
    suite_cwd = Path(suite.get("working_directory", "")).resolve()
    import_origin = Path(suite.get("revision_r1_import_origin", "")).resolve()
    if suite_cwd != extracted_project or not import_origin.is_relative_to(extracted_project):
        raise ValueError("test command did not run from, and import revision_r1 from, the extracted package")
    if not isinstance(scale, dict) or scale.get("status") != "PASS" or scale.get("working_directory") != str(extracted_project):
        raise ValueError("extracted-package scale-memory validation has not passed")
    zip_sha = sha256_file(code_zip)
    zip_sidecar = Path(str(code_zip) + ".sha256").read_text(encoding="utf-8").split()[0]
    notebook_sha = sha256_file(notebook)
    notebook_sidecar = Path(str(notebook) + ".sha256").read_text(encoding="utf-8").split()[0]
    if zip_sha != zip_sidecar or notebook_sha != notebook_sidecar:
        raise ValueError("release ZIP/notebook checksum sidecar mismatch; QA manifest left unchanged")
    notebook_doc = json.loads(notebook.read_text(encoding="utf-8"))
    notebook_code = "\n".join("".join(cell.get("source", [])) for cell in notebook_doc.get("cells", []) if cell.get("cell_type") == "code")
    if f'EXPECTED_ZIP_SHA256 = "{zip_sha}"' not in notebook_code:
        raise ValueError("notebook package filename/digest does not bind to this exact ZIP")
    with zipfile.ZipFile(code_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("distributed ZIP has a CRC failure")
        package_manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        if package_manifest.get("production_build_id") != PRODUCTION_BUILD_ID:
            raise ValueError("distributed ZIP build identity differs from this release")
        for relative, item in package_manifest.get("files", {}).items():
            content = archive.read(relative)
            if len(content) != item.get("bytes") or sha256_bytes(content) != item.get("sha256"):
                raise ValueError(f"distributed ZIP member does not match manifest: {relative}")
        if package_manifest.get("training_source_sha256") != evidence.get("training_source_sha256"):
            raise ValueError("QA evidence training-core identity differs from the distributed ZIP")
    if sha256_file(code_zip) != evidence.get("code_zip_sha256"):
        raise ValueError("QA evidence was produced for a different ZIP")
    manifest_path = output / "BUILD_MANIFEST.json"
    build = json.loads(manifest_path.read_text(encoding="utf-8"))
    if build.get("production_build_id") != PRODUCTION_BUILD_ID or build.get("code_zip", {}).get("sha256") != zip_sha:
        raise ValueError("external build manifest does not match the ZIP being finalized")
    build["qa_status"] = "LOCAL_EXTRACTED_PACKAGE_AND_SCALE_MEMORY_PASS"
    build["qa_evidence"] = {
        "source": "actual distributed ZIP extracted outside repository; full applicable CPU suite and scale-memory fixture",
        "code_zip_sha256": zip_sha,
        "extracted_package_test_suite": suite,
        "scale_memory_test": scale,
        "production_gpu_drive_status": "NOT_RUN",
        "production_matrix_launched": False,
    }
    qa_report = (
        "# Production handoff QA\n\n"
        "Status: `LOCAL_EXTRACTED_PACKAGE_AND_SCALE_MEMORY_PASS`.\n\n"
        f"- Distributed ZIP SHA-256: `{zip_sha}`\n"
        f"- Notebook SHA-256: `{notebook_sha}`\n"
        f"- Extracted package: `{extracted_project}`\n"
        f"- CPU suite: {suite['passed']} passed, {suite['failed']} failed, {int(suite.get('skipped', 0))} skipped.\n"
        f"- Scale-memory fixture: {scale.get('rows')} rows × {scale.get('feature_dim')} features; peak RSS `{scale.get('peak_rss_bytes')}` bytes.\n"
        "- Colab GPU, Drive lease/recovery, actual datasets, and production runs: `NOT_RUN` (hardware/runtime checks remain).\n"
        "- Production matrix launched: `false`.\n"
    )
    for path, payload in (
        (output / "QA_REPORT.md", qa_report.encode("utf-8")),
        (manifest_path, (json.dumps(build, sort_keys=True, indent=2) + "\n").encode("utf-8")),
    ):
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=output)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return build


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the uniquely versioned revision_R1 production Colab handoff")
    parser.add_argument("--output-dir", type=Path, required=True, help="new, non-existing release directory")
    parser.add_argument("--finalize-qa-evidence", type=Path, help="JSON evidence from extracted-ZIP suite and scale-memory checks")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    if args.finalize_qa_evidence:
        evidence = json.loads(args.finalize_qa_evidence.read_text(encoding="utf-8"))
        result = finalize_handoff_qa(args.output_dir, evidence)
    else:
        result = build_handoff(repository, args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
