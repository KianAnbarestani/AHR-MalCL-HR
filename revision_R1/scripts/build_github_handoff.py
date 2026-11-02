"""Build the integrity manifest and stable GitHub-pinned Colab notebook."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from scripts.build_production_handoff import RELEASE_TESTS, notebook_document as zip_notebook_document
from scripts.production_workflow import EXPECTED_EFFECTIVE_SCHEDULE_SHA256
from revision_r1.compatibility import runtime_source_sha256, verify_checkpoint_compatibility


EXPECTED_TRAINING_SOURCE_SHA256 = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
GITHUB_REPOSITORY = "KianAnbarestani/AHR-MalCL-HR"
GITHUB_TOKEN_SECRET_NAME = "GITHUB_READ_TOKEN"
PACKAGE_MANIFEST_NAME = "PACKAGE_MANIFEST.json"
BUILD_MANIFEST_NAME = "BUILD_MANIFEST.json"
EXTRA_TESTS = (
    "test_checkpoint_compatibility.py",
    "test_github_handoff.py",
    "test_runtime_compatibility.py",
)
EXTRA_FILES = (
    "CHECKPOINT_COMPATIBILITY.md",
    "COMPATIBILITY_REPORT.md",
    "FROZEN_PROTOCOL.md",
    "legacy_checkpoint_source_compatibility.json",
    "PRODUCTION_COLAB_README.md",
    "README_GITHUB_COLAB.md",
    "requirements-colab.txt",
    "requirements-test.txt",
    "checkpoint_compatibility.json",
)
SCRIPT_FILES = (
    "scripts/build_checkpoint_compatibility.py",
    "scripts/build_github_handoff.py",
    "scripts/build_production_handoff.py",
    "scripts/dataset_bootstrap.py",
    "scripts/live_progress.py",
    "scripts/production_workflow.py",
)
CONFIG_FILES = (
    "configs/legacy_v1_2_first_run_identity.json",
    "configs/production_run_plan_v1.json",
    "configs/update_budget.csv",
    "configs/update_budget_manifest.json",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _training_source_sha256(root: Path) -> str:
    return str(verify_checkpoint_compatibility(root)["baseline_training_source_sha256"])


def package_file_paths(package_root: str | Path) -> list[Path]:
    root = Path(package_root).resolve()
    paths = [*sorted((root / "revision_r1").glob("*.py"))]
    paths.extend(root / relative for relative in (*SCRIPT_FILES, *CONFIG_FILES, *EXTRA_FILES))
    paths.extend(root / "tests" / name for name in (*RELEASE_TESTS, *EXTRA_TESTS))
    unique = sorted(set(paths))
    missing = [path.relative_to(root).as_posix() for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GitHub runner source files are missing: {missing}")
    for path in unique:
        relative = path.relative_to(root).as_posix().lower()
        if relative.endswith((".npz", ".pt", ".pth", ".ckpt")):
            raise ValueError(f"data/checkpoint artifacts are not allowed in the runner package: {relative}")
        if any(part in {"results", "checkpoints", "cache", "__pycache__", ".pytest_cache"}
               for part in Path(relative).parts):
            raise ValueError(f"generated/local state is not allowed in the runner package: {relative}")
    return unique


def build_package_manifest(package_root: str | Path) -> dict[str, Any]:
    root = Path(package_root).resolve()
    compatibility = verify_checkpoint_compatibility(root)
    training_hash = _training_source_sha256(root)
    if training_hash != EXPECTED_TRAINING_SOURCE_SHA256:
        raise ValueError(
            "training source differs from the compatible first-run checkpoint identity; "
            f"expected {EXPECTED_TRAINING_SOURCE_SHA256}, observed {training_hash}"
        )
    schedule_path = root / "configs" / "update_budget.csv"
    schedule_manifest_path = root / "configs" / "update_budget_manifest.json"
    schedule_manifest = json.loads(schedule_manifest_path.read_text(encoding="utf-8"))
    packaged_schedule_hash = _sha256(schedule_path.read_bytes())
    if schedule_manifest.get("source_csv_sha256") != packaged_schedule_hash:
        raise ValueError("packaged update-budget CSV and manifest do not match")
    plan = json.loads((root / "configs" / "production_run_plan_v1.json").read_text(encoding="utf-8"))
    if plan.get("training_source_sha256") != training_hash:
        raise ValueError("frozen plan training-source identity differs from the runner modules")
    if plan.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256:
        raise ValueError("frozen plan effective schedule identity differs from the agreed schedule")
    if plan.get("run_count") != 100 or not plan.get("runs") or plan["runs"][0].get("run_key") != "ember_seed42_full_hybrid":
        raise ValueError("frozen production plan/order is not the approved 100-run matrix")

    files = {}
    for path in package_file_paths(root):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path.read_bytes())}
    return {
        "schema_version": 1,
        "release_kind": "GITHUB_PINNED_COLAB_RUNNER",
        "repository": GITHUB_REPOSITORY,
        "training_source_sha256": training_hash,
        "runtime_source_sha256": compatibility["runtime_source_sha256"],
        "checkpoint_compatibility_baseline_commit": compatibility["baseline_commit_sha"],
        "checkpoint_compatibility_manifest_sha256": _sha256((root / "checkpoint_compatibility.json").read_bytes()),
        "expected_effective_schedule_sha256": EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
        "packaged_schedule_sha256": packaged_schedule_hash,
        "plan_sha256": plan["plan_sha256"],
        "run_count": plan["run_count"],
        "first_run_key": plan["runs"][0]["run_key"],
        "protocol": plan["protocol"],
        "credential_policy": "no credential for public repository; private read-only token only via Colab Secrets",
        "production_enabled_by_default": False,
        "files": files,
    }


def write_manifests(package_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(package_root).resolve()
    package_manifest = build_package_manifest(root)
    package_bytes = (json.dumps(package_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (root / PACKAGE_MANIFEST_NAME).write_bytes(package_bytes)
    build_manifest = {
        "schema_version": 1,
        "release_kind": "GITHUB_PINNED_COLAB_RUNNER",
        "package_manifest_sha256": _sha256(package_bytes),
        "training_source_sha256": EXPECTED_TRAINING_SOURCE_SHA256,
        "runtime_source_sha256": package_manifest["runtime_source_sha256"],
        "effective_schedule_sha256": EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
        "file_count": len(package_manifest["files"]),
        "commit_sha_is_bound_by_colab_not_embedded": True,
    }
    (root / BUILD_MANIFEST_NAME).write_text(
        json.dumps(build_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return package_manifest, build_manifest


def stage_clean_package(source_root: str | Path, output_root: str | Path) -> Path:
    """Copy only the reviewed allowlist into a new clean checkout-shaped directory."""
    source = Path(source_root).resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing GitHub package staging directory: {output}")
    output.mkdir(parents=True)
    for path in package_file_paths(source):
        relative = path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_manifests(output)
    return output


def verify_package_integrity(package_root: str | Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Verify the exact tracked package inventory and both committed manifests."""
    root = Path(package_root).resolve()
    package_bytes = (root / PACKAGE_MANIFEST_NAME).read_bytes()
    build_manifest = json.loads((root / BUILD_MANIFEST_NAME).read_text(encoding="utf-8"))
    package_manifest_sha = _sha256(package_bytes)
    package_manifest = json.loads(package_bytes)
    files = package_manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PACKAGE_MANIFEST has no file-integrity map")
    if (package_manifest.get("schema_version") != 1
        or package_manifest.get("release_kind") != "GITHUB_PINNED_COLAB_RUNNER"
        or package_manifest.get("repository", "").lower() != GITHUB_REPOSITORY.lower()
        or package_manifest.get("training_source_sha256") != EXPECTED_TRAINING_SOURCE_SHA256
        or package_manifest.get("runtime_source_sha256") != runtime_source_sha256(root)
        or package_manifest.get("expected_effective_schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or package_manifest.get("production_enabled_by_default") is not False):
        raise ValueError("PACKAGE_MANIFEST release, repository, core, schedule, or production-gate identity is incompatible")
    if (build_manifest.get("package_manifest_sha256") != package_manifest_sha
        or build_manifest.get("training_source_sha256") != EXPECTED_TRAINING_SOURCE_SHA256
        or build_manifest.get("runtime_source_sha256") != package_manifest.get("runtime_source_sha256")
        or build_manifest.get("effective_schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or build_manifest.get("file_count") != len(files)):
        raise ValueError("BUILD_MANIFEST does not bind the current package/core/schedule identities")
    observed = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Symbolic links are not allowed in the runner package: {candidate.relative_to(root)}")
        if candidate.is_file() and candidate.name not in {PACKAGE_MANIFEST_NAME, BUILD_MANIFEST_NAME}:
            observed.add(candidate.relative_to(root).as_posix())
    if observed != set(files):
        raise ValueError(
            f"runner file inventory differs from manifest; unmanifested={sorted(observed - set(files))}, "
            f"missing={sorted(set(files) - observed)}"
        )
    for relative, item in files.items():
        rel = Path(relative)
        target = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or root not in target.parents or not target.is_file():
            raise FileNotFoundError(f"missing or unsafe package member: {relative}")
        payload = target.read_bytes()
        if len(payload) != item.get("bytes") or _sha256(payload) != item.get("sha256"):
            raise ValueError(f"package member hash/size mismatch: {relative}")
    if _training_source_sha256(root) != EXPECTED_TRAINING_SOURCE_SHA256:
        raise ValueError("actual runner training-source hash differs from its compatible checkpoint identity")
    compatibility = verify_checkpoint_compatibility(root)
    if package_manifest.get("checkpoint_compatibility_manifest_sha256") != _sha256((root / "checkpoint_compatibility.json").read_bytes()):
        raise ValueError("checkpoint compatibility manifest differs from the package manifest")
    if package_manifest.get("checkpoint_compatibility_baseline_commit") != compatibility.get("baseline_commit_sha"):
        raise ValueError("checkpoint compatibility baseline commit differs from the package manifest")
    schedule_path = root / "configs" / "update_budget.csv"
    schedule_manifest = json.loads((root / "configs" / "update_budget_manifest.json").read_text(encoding="utf-8"))
    if (_sha256(schedule_path.read_bytes()) != package_manifest.get("packaged_schedule_sha256")
        or schedule_manifest.get("source_csv_sha256") != package_manifest.get("packaged_schedule_sha256")):
        raise ValueError("packaged schedule does not match both integrity manifests")
    plan = json.loads((root / "configs" / "production_run_plan_v1.json").read_text(encoding="utf-8"))
    if (plan.get("plan_sha256") != package_manifest.get("plan_sha256")
        or plan.get("training_source_sha256") != EXPECTED_TRAINING_SOURCE_SHA256
        or plan.get("schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or plan.get("run_count") != package_manifest.get("run_count")
        or not plan.get("runs")
        or plan["runs"][0].get("run_key") != package_manifest.get("first_run_key")
        or plan.get("protocol") != package_manifest.get("protocol")):
        raise ValueError("frozen plan differs from authenticated training/schedule identities")
    return package_manifest, build_manifest, package_manifest_sha


def _github_setup_cell(commit_sha: str) -> str:
    template = r'''# Cell A — Drive/GitHub setup, one pinned detached fetch, and package integrity checks.
from pathlib import Path, PurePosixPath
import hashlib, json, os, platform, shutil, subprocess, sys, tempfile, urllib.error, urllib.request
from google.colab import auth, drive

GITHUB_REPOSITORY = "KianAnbarestani/AHR-MalCL-HR"
PINNED_GIT_COMMIT = "__PINNED_COMMIT__"  # Change only this full SHA for a tested code release.
GITHUB_TOKEN_SECRET_NAME = "GITHUB_READ_TOKEN"  # Used only if the repository is private.
STORAGE_ACCOUNT_EMAIL = "kian.anb777@gmail.com"
CANONICAL_REVISION_FOLDER_ID = "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x"
EXPECTED_TRAINING_SOURCE_SHA256 = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
EXPECTED_SCHEDULE_SHA256 = "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212"
if len(PINNED_GIT_COMMIT) != 40 or any(c not in "0123456789abcdef" for c in PINNED_GIT_COMMIT):
    raise ValueError("PINNED_GIT_COMMIT must be the explicitly selected full lowercase 40-character commit SHA")

DRIVE_PATHS = {
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
}
# The compute account may differ. Select STORAGE_ACCOUNT_EMAIL in both Google prompts.
# If cached ADC selects another account, stop and restart a fresh runtime.
print(f"Canonical Drive storage account: {STORAGE_ACCOUNT_EMAIL}; choose it in the Google auth/mount prompts.")
auth.authenticate_user()
drive.mount("/content/drive", force_remount=True)

GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPOSITORY}"
def _github_metadata(token=None):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(GITHUB_API_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if not token and exc.code == 404:
            return None  # GitHub hides private repositories from unauthenticated callers.
        raise RuntimeError(f"GitHub repository access failed (HTTP {exc.code}); token contents are never displayed") from None
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Could not verify GitHub repository visibility/identity; check network and retry Cell A") from None

REPOSITORY_METADATA = _github_metadata()
GITHUB_TOKEN = None
if REPOSITORY_METADATA is None or REPOSITORY_METADATA.get("private") is True:
    try:
        from google.colab import userdata
        GITHUB_TOKEN = userdata.get(GITHUB_TOKEN_SECRET_NAME)
    except Exception:
        GITHUB_TOKEN = None
    if not GITHUB_TOKEN:
        raise RuntimeError(
            f"This repository is private or not publicly resolvable. Add a fine-grained read-only GitHub token "
            f"to Colab Secrets as {GITHUB_TOKEN_SECRET_NAME}, restrict it to {GITHUB_REPOSITORY} with Contents: Read-only, "
            "and enable notebook access. Do not paste it into a cell."
        )
    try:
        REPOSITORY_METADATA = _github_metadata(GITHUB_TOKEN)
    except Exception:
        GITHUB_TOKEN = None
        raise
if REPOSITORY_METADATA.get("full_name", "").lower() != GITHUB_REPOSITORY.lower():
    GITHUB_TOKEN = None
    raise RuntimeError("GitHub returned a different repository identity than the pinned notebook setting")

CHECKOUT_PARENT = Path("/content/revision_R1_pinned_sources")
CHECKOUT_PATH = CHECKOUT_PARENT / PINNED_GIT_COMMIT
PACKAGE_ROOT = CHECKOUT_PATH / "revision_R1"
def _git(args, *, cwd, env):
    result = subprocess.run(["git", *args], cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        # Raw stderr is intentionally withheld to prevent accidental credential disclosure.
        raise RuntimeError(f"Git operation {args[0]!r} failed (exit {result.returncode}); verify the commit and repository read access")
    return result.stdout.strip()

if CHECKOUT_PATH.exists():
    GITHUB_TOKEN = None  # Existing verified checkout needs no credential for local reuse.
    if not (CHECKOUT_PATH / ".git").is_dir():
        raise RuntimeError(f"Pinned checkout path exists but is not a Git checkout; preserved: {CHECKOUT_PATH}")
    head = _git(["rev-parse", "HEAD"], cwd=CHECKOUT_PATH, env=os.environ.copy())
    if head != PINNED_GIT_COMMIT:
        raise RuntimeError(f"Existing session checkout is a different commit; preserved: {head}")
    if _git(["status", "--porcelain"], cwd=CHECKOUT_PATH, env=os.environ.copy()):
        raise RuntimeError("Pinned session checkout is modified; refusing to reuse a dirty checkout")
else:
    CHECKOUT_PARENT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{PINNED_GIT_COMMIT[:12]}-", dir=CHECKOUT_PARENT))
    askpass_dir = None
    token = GITHUB_TOKEN
    git_env = os.environ.copy()
    git_env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_TRACE": "0", "GIT_CURL_VERBOSE": "0",
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    if token:
        askpass_dir = tempfile.TemporaryDirectory(prefix="revision_r1_git_auth_")
        askpass = Path(askpass_dir.name) / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\nimport os,sys\np=sys.argv[-1].lower()\n"
            "print('x-access-token' if 'username' in p else os.environ.get('GITHUB_READ_TOKEN',''))\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        git_env.update({"GIT_ASKPASS": str(askpass), "GIT_ASKPASS_REQUIRE": "force", "GITHUB_READ_TOKEN": token})
    try:
        _git(["init", "--quiet"], cwd=staging, env=git_env)
        _git(["remote", "add", "origin", f"https://github.com/{GITHUB_REPOSITORY}.git"], cwd=staging, env=git_env)
        _git(["fetch", "--no-tags", "--depth=1", "origin", PINNED_GIT_COMMIT], cwd=staging, env=git_env)
        _git(["checkout", "--detach", "FETCH_HEAD"], cwd=staging, env=git_env)
        fetched = _git(["rev-parse", "HEAD"], cwd=staging, env=git_env)
        if fetched != PINNED_GIT_COMMIT:
            raise RuntimeError("Fetched object did not match PINNED_GIT_COMMIT")
        if _git(["remote", "get-url", "origin"], cwd=staging, env=git_env) != f"https://github.com/{GITHUB_REPOSITORY}.git":
            raise RuntimeError("Git remote is not credential-free; refusing to retain it")
        os.replace(staging, CHECKOUT_PATH)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if askpass_dir is not None:
            askpass_dir.cleanup()
        git_env.pop("GITHUB_READ_TOKEN", None)
        GITHUB_TOKEN = None
        token = None

ACTUAL_GIT_COMMIT = _git(["rev-parse", "HEAD"], cwd=CHECKOUT_PATH, env=os.environ.copy())
if ACTUAL_GIT_COMMIT != PINNED_GIT_COMMIT or _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=CHECKOUT_PATH, env=os.environ.copy()) != "HEAD":
    raise RuntimeError("Pinned source was not checked out at the exact detached commit")

def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

PROJECT = PACKAGE_ROOT
os.chdir(PROJECT)
sys.dont_write_bytecode = True
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != PROJECT.resolve()]
sys.path.insert(0, str(PROJECT))
for module_name in list(sys.modules):
    if module_name == "revision_r1" or module_name.startswith("revision_r1.") or module_name == "scripts" or module_name.startswith("scripts."):
        del sys.modules[module_name]
from scripts.build_github_handoff import verify_package_integrity
PACKAGE_MANIFEST, BUILD_MANIFEST, PACKAGE_MANIFEST_SHA256 = verify_package_integrity(PACKAGE_ROOT)
from scripts.build_production_handoff import PRODUCTION_BUILD_ID as EXPECTED_BUILD_ID
import numpy as np, platform, torch
EXPECTED_PYTHON_VERSION = (3, 12, 13)
EXPECTED_TORCH_VERSION = "2.11.0+cu128"
EXPECTED_CUDA_VERSION = "12.8"
if sys.version_info[:3] != EXPECTED_PYTHON_VERSION:
    raise RuntimeError(f"Expected Colab Python {'.'.join(map(str, EXPECTED_PYTHON_VERSION))}, got {platform.python_version()}")
if np.__version__ != "2.0.2":
    raise RuntimeError(f"Expected pinned NumPy 2.0.2, got {np.__version__}; no packages were changed")
if torch.__version__ != EXPECTED_TORCH_VERSION:
    raise RuntimeError(f"Expected preinstalled CUDA PyTorch {EXPECTED_TORCH_VERSION}, got {torch.__version__}; no packages were changed")
if torch.version.cuda != EXPECTED_CUDA_VERSION:
    raise RuntimeError(f"Expected PyTorch CUDA build {EXPECTED_CUDA_VERSION}, got {torch.version.cuda}; no packages were changed")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. No training started; select a GPU Colab runtime and rerun Cell A.")
from scripts.production_workflow import training_source_sha256
from revision_r1.compatibility import runtime_source_sha256, verify_checkpoint_compatibility
import revision_r1.runner as runner_module
import scripts.production_workflow as production_workflow_module
import google.auth
from googleapiclient.discovery import build
STORAGE_CREDENTIALS, STORAGE_GCLOUD_PROJECT = google.auth.default(
    scopes=["https://www.googleapis.com/auth/drive"]
)
STORAGE_API_SERVICE = build("drive", "v3", credentials=STORAGE_CREDENTIALS, cache_discovery=False)
STORAGE_API_FACTORY = lambda: STORAGE_API_SERVICE
ACTUAL_TRAINING_SOURCE_SHA256 = training_source_sha256(PROJECT)
CHECKPOINT_COMPATIBILITY = verify_checkpoint_compatibility(PROJECT)
ACTUAL_RUNTIME_SOURCE_SHA256 = runtime_source_sha256(PROJECT)
if (ACTUAL_TRAINING_SOURCE_SHA256 != EXPECTED_TRAINING_SOURCE_SHA256
    or runner_module._code_sha256() != EXPECTED_TRAINING_SOURCE_SHA256
    or PACKAGE_MANIFEST.get("training_source_sha256") != ACTUAL_TRAINING_SOURCE_SHA256
    or PACKAGE_MANIFEST.get("runtime_source_sha256") != ACTUAL_RUNTIME_SOURCE_SHA256):
    raise ValueError("Training-core hash is incompatible with the resumable first-run checkpoint; no fallback/restart is allowed")
if not Path(runner_module.__file__).resolve().is_relative_to(PROJECT.resolve()):
    raise ImportError(f"Training runner imported outside the pinned Git checkout: {runner_module.__file__}")
try:
    STORAGE_SESSION_EVIDENCE = production_workflow_module.verify_canonical_storage_session(
        api_factory=STORAGE_API_FACTORY,
        drive_mount_root="/content/drive/MyDrive",
        expected_account_email=STORAGE_ACCOUNT_EMAIL,
        expected_revision_folder_id=CANONICAL_REVISION_FOLDER_ID,
        path_bindings={
            "revision_root": "/content/drive/MyDrive/ReVision",
            "datasets": "/content/drive/MyDrive/AHR_MalCL/datasets",
            "preflight": "/content/drive/MyDrive/ReVision/revision_R1/results/preflight",
            "validation_reports": "/content/drive/MyDrive/ReVision/revision_R1/results/validation/reports",
            "production_results": "/content/drive/MyDrive/ReVision/revision_R1/production/results",
            "production_checkpoints": "/content/drive/MyDrive/ReVision/revision_R1/production/checkpoints",
        },
        required_path_keys=(
            "revision_root", "datasets", "preflight", "validation_reports",
            "production_results", "production_checkpoints",
        ),
    )
except Exception as exc:
    raise RuntimeError(
        "Canonical Drive storage verification failed before any run. Both the mounted filesystem and Drive API "
        f"must use {STORAGE_ACCOUNT_EMAIL} and ReVision folder {CANONICAL_REVISION_FOLDER_ID}; wrong storage account "
        "selection is not recoverable by changing run identities. If cached credentials "
        "selected another account, restart the runtime and select the canonical account when prompted; do not "
        f"proceed or change paths. Cause: {type(exc).__name__}: {exc}"
    ) from exc
print(json.dumps({"repository": GITHUB_REPOSITORY, "repository_private": bool(REPOSITORY_METADATA.get("private")),
                  "selected_git_commit": ACTUAL_GIT_COMMIT, "checkout": str(CHECKOUT_PATH),
                  "detached_checkout": True, "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
                  "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
                  "runtime_source_sha256": ACTUAL_RUNTIME_SOURCE_SHA256,
                  "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
                  "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "storage_account": STORAGE_SESSION_EVIDENCE["storage_account_email"],
                  "storage_quota_account": STORAGE_SESSION_EVIDENCE["storage_quota_account_email"],
                  "canonical_revision_folder_id": STORAGE_SESSION_EVIDENCE["canonical_revision_folder_id"],
                  "mount_and_api_identity_bound": True}, indent=2))
'''
    return template.replace("__PINNED_COMMIT__", commit_sha)


def _github_data_cell() -> str:
    return '''# Cell B — exact canonical Drive datasets; no external downloads.
from revision_r1.validation_workflow import resolve_effective_artifacts
from revision_r1.config import DATASETS
from revision_r1.data import inspect_npz_array_layout
from scripts.production_workflow import inspect_dataset_sources

PREFLIGHT_PATH = Path(DRIVE_PATHS["preflight_path"]).expanduser()
if not PREFLIGHT_PATH.is_file():
    raise FileNotFoundError(f"Passed Drive preflight not found: {PREFLIGHT_PATH}")
EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT = resolve_effective_artifacts(
    PREFLIGHT_PATH, expected_sha256=EXPECTED_SCHEDULE_SHA256, discover_manifest=True,
)
if PREFLIGHT.get("status") != "PREFLIGHT_VALIDATED_NO_TRAINING" or PREFLIGHT.get("training_reproduction_claimed") is not False:
    raise ValueError("Drive preflight is not the validated no-training schedule/data record")
DATA_ROOTS = {
    "ember": Path(DRIVE_PATHS["ember_data_root"]).expanduser(),
    "az_class": Path(DRIVE_PATHS["az_data_root"]).expanduser(),
}
DATASET_SOURCE_REPORT = inspect_dataset_sources(DATA_ROOTS, DRIVE_PATHS["dataset_cache_root"], PREFLIGHT)
DATASET_LAYOUTS = {}
for _dataset, _root in DATA_ROOTS.items():
    _layout = {}
    for _split in ("train", "test"):
        _npz = _root / DATASETS[_dataset][_split]["filename"]
        if _npz.is_file():
            _layout[_split] = inspect_npz_array_layout(_npz, split=_split)
        else:
            _layout[_split] = {"archive": "ABSENT", "source": DATASET_SOURCE_REPORT[_dataset]["splits"][_split]["source"]}
    DATASET_LAYOUTS[_dataset] = _layout

PACKAGED_SCHEDULE_PATH = PROJECT / "configs" / "update_budget.csv"
PACKAGED_SCHEDULE_MANIFEST_PATH = PROJECT / "configs" / "update_budget_manifest.json"
PACKAGED_SCHEDULE_SHA256 = hashlib.sha256(PACKAGED_SCHEDULE_PATH.read_bytes()).hexdigest()
PACKAGED_SCHEDULE_MANIFEST = json.loads(PACKAGED_SCHEDULE_MANIFEST_PATH.read_text(encoding="utf-8"))
if PACKAGED_SCHEDULE_SHA256 != PACKAGED_SCHEDULE_MANIFEST.get("source_csv_sha256"):
    raise ValueError("Original packaged schedule/manifest pair is altered")
print("Effective schedule:", EFFECTIVE_SCHEDULE_PATH)
print("Effective schedule SHA-256:", EXPECTED_SCHEDULE_SHA256, "| rows:", PREFLIGHT["schedule"]["row_count"])
print("Matching manifest:", EFFECTIVE_SCHEDULE_MANIFEST_PATH)
print("Original packaged schedule SHA-256 (unchanged):", PACKAGED_SCHEDULE_SHA256)
print("Canonical dataset sources (missing Drive files require a matching local verified cache):")
print(json.dumps(DATASET_SOURCE_REPORT, sort_keys=True, indent=2))
print("If both canonical Drive files and a matching verified local cache are absent, preflight fails; no external download will be attempted.")
print("External dataset downloads: DISABLED")
'''


def _github_readiness_cell() -> str:
    return '''# Cell C — validate the frozen plan, reuse Run 1's existing measurement, and gate the next pending run.
from scripts.production_workflow import (
    _expected_plan_runner_identity, _preflight_addressed_cache_bytes, _read_matrix_registry,
    bind_runtime_schedule, build_frozen_run_plan, inspect_production_capacity,
    load_v62_recovery_checkpoint_bound, project_storage, require_run_capacity,
    reuse_validated_first_run_measurement, run_checkpoint_root, run_output_root,
    validate_frozen_plan, validate_runtime_bound_plan,
)
from revision_r1.storage import DEFAULT_STORAGE_RESERVE_BYTES, require_available_bytes

BASE_PLAN_PATH = PROJECT / "configs" / "production_run_plan_v1.json"
BASE_PLAN = json.loads(BASE_PLAN_PATH.read_text(encoding="utf-8"))
validate_frozen_plan(BASE_PLAN, expected_source_sha256=ACTUAL_TRAINING_SOURCE_SHA256)
RUNTIME_PLAN = bind_runtime_schedule(BASE_PLAN, EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT)
validate_runtime_bound_plan(RUNTIME_PLAN, EFFECTIVE_SCHEDULE_PATH, EFFECTIVE_SCHEDULE_MANIFEST_PATH, PREFLIGHT)
if RUNTIME_PLAN["run_count"] != 100 or RUNTIME_PLAN["runs"][0]["run_key"] != "ember_seed42_full_hybrid":
    raise ValueError("Runtime-bound matrix/order differs from the frozen protocol")

OUTPUT_ROOT = Path(DRIVE_PATHS["output_root"]).expanduser()
CHECKPOINT_ROOT = Path(DRIVE_PATHS["checkpoint_root"]).expanduser()
LOCAL_CHECKPOINT_ROOT = Path(DRIVE_PATHS["local_checkpoint_root"]).expanduser()
CACHE_ROOT = Path(DRIVE_PATHS["dataset_cache_root"]).expanduser()
FIRST_RUN = RUNTIME_PLAN["runs"][0]
FIRST_RUN_MEASUREMENT = reuse_validated_first_run_measurement(RUNTIME_PLAN, FIRST_RUN, OUTPUT_ROOT)
MATRIX_REGISTRY = _read_matrix_registry(OUTPUT_ROOT.resolve(), RUNTIME_PLAN)
if FIRST_RUN["run_key"] not in MATRIX_REGISTRY["completed_runs"]:
    raise RuntimeError("Run 1 is not registered complete; this release will not launch or finalize it")
COMPLETED_RUN_KEYS = set(MATRIX_REGISTRY["completed_runs"])
NEXT_PENDING_RECORD = next((row for row in RUNTIME_PLAN["runs"][1:] if row["run_key"] not in COMPLETED_RUN_KEYS), None)
V62_REFERENCE = load_v62_recovery_checkpoint_bound(DRIVE_PATHS["v62_recovery_report"])
OBSERVED_CHECKPOINT_PAYLOAD_BYTES = max(
    int(FIRST_RUN_MEASUREMENT["max_checkpoint_payload_bytes"]),
    int(V62_REFERENCE["max_checkpoint_payload_bytes"]),
)
STORAGE_PROJECTION = project_storage(
    RUNTIME_PLAN,
    observed_checkpoint_bytes=OBSERVED_CHECKPOINT_PAYLOAD_BYTES,
    observed_output_bytes=int(FIRST_RUN_MEASUREMENT["run_output_bytes"]),
    checkpoint_reference_provenance=(
        "registered Run 1 measurement reused plus the verified v6.2 operational reference; "
        "planning estimate only, not a new measurement"
    ),
    checkpoint_size_evidence={
        "run_1_registered_measurement_reused": FIRST_RUN_MEASUREMENT,
        "v6_2_reference_payload_bytes": int(V62_REFERENCE["max_checkpoint_payload_bytes"]),
        "projection_source_bytes": OBSERVED_CHECKPOINT_PAYLOAD_BYTES,
    },
)
if NEXT_PENDING_RECORD is None:
    CAPACITY = inspect_production_capacity(
        OUTPUT_ROOT, CHECKPOINT_ROOT, LOCAL_CHECKPOINT_ROOT, CACHE_ROOT,
        api_factory=STORAGE_API_FACTORY,
    )
else:
    NEXT_OUTPUT = run_output_root(OUTPUT_ROOT, NEXT_PENDING_RECORD)
    NEXT_CHECKPOINT = run_checkpoint_root(CHECKPOINT_ROOT, NEXT_PENDING_RECORD)
    NEXT_LOCAL_CHECKPOINT = run_checkpoint_root(LOCAL_CHECKPOINT_ROOT, NEXT_PENDING_RECORD)
    CAPACITY = inspect_production_capacity(
        OUTPUT_ROOT, CHECKPOINT_ROOT, LOCAL_CHECKPOINT_ROOT, CACHE_ROOT,
        api_factory=STORAGE_API_FACTORY,
        current_run_output_path=NEXT_OUTPUT,
        current_run_checkpoint_path=NEXT_CHECKPOINT,
        current_run_local_checkpoint_path=NEXT_LOCAL_CHECKPOINT,
        expected_run_identity=_expected_plan_runner_identity(RUNTIME_PLAN, NEXT_PENDING_RECORD),
        verified_dataset_cache_bytes=_preflight_addressed_cache_bytes(
            PREFLIGHT, CACHE_ROOT, dataset=NEXT_PENDING_RECORD["dataset"],
        ),
    )
    CAPACITY["production_measurements_validated"] = True
    require_run_capacity(
        CAPACITY, STORAGE_PROJECTION, scope="matrix",
        completed_runs=len(MATRIX_REGISTRY["completed_runs"]),
    )
if CAPACITY.get("canonical_storage_identity_verified") is not True:
    raise RuntimeError("Capacity evidence is not bound to the canonical main-account storage session")

# Bind the selected release separately; never include storage/compute account in training identity.
CODE_BINDING_PATH = OUTPUT_ROOT / "runtime_code_binding" / RUNTIME_PLAN["plan_sha256"] / f"{ACTUAL_GIT_COMMIT}.json"
CODE_BINDING = {
    "schema_version": 1, "selected_git_commit": ACTUAL_GIT_COMMIT,
    "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
    "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
    "runtime_source_sha256": ACTUAL_RUNTIME_SOURCE_SHA256,
    "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    "frozen_plan_sha256": RUNTIME_PLAN["plan_sha256"],
    "first_run_key": FIRST_RUN["run_key"], "first_run_config_sha256": FIRST_RUN["config_sha256"],
}
CODE_BINDING_BYTES = (json.dumps(CODE_BINDING, sort_keys=True, indent=2, allow_nan=False) + "\\n").encode("utf-8")
PLAN_PATH = OUTPUT_ROOT / "plans" / f"production_run_plan_{RUNTIME_PLAN['plan_sha256']}.json"
PLAN_BYTES = (json.dumps(RUNTIME_PLAN, sort_keys=True, indent=2, allow_nan=False) + "\\n").encode("utf-8")
for _path, _payload in ((CODE_BINDING_PATH, CODE_BINDING_BYTES), (PLAN_PATH, PLAN_BYTES)):
    if _path.exists() and _path.read_bytes() != _payload:
        raise ValueError(f"Existing immutable plan/binding differs; preserved without overwrite: {_path}")
PENDING_IMMUTABLE_BYTES = sum(len(_payload) for _path, _payload in (
    (CODE_BINDING_PATH, CODE_BINDING_BYTES), (PLAN_PATH, PLAN_BYTES),
) if not _path.exists())
if NEXT_PENDING_RECORD is not None:
    require_available_bytes(
        OUTPUT_ROOT,
        required_bytes=int(STORAGE_PROJECTION["drive"]["next_run_required_without_reserve_bytes"]) + PENDING_IMMUTABLE_BYTES,
        reserve_bytes=DEFAULT_STORAGE_RESERVE_BYTES,
        available_bytes=CAPACITY["drive_available_bytes"],
        storage_kind="canonical Drive next-run and immutable plan capacity",
    )
def _write_immutable_drive_file(_path, _payload):
    _path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _path.open("xb") as _stream:
            _stream.write(_payload)
            _stream.flush()
            os.fsync(_stream.fileno())
    except FileExistsError:
        if _path.read_bytes() != _payload:
            raise ValueError(f"Concurrent immutable file differs; preserved: {_path}")
    if _path.read_bytes() != _payload:
        raise IOError(f"Immutable file read-back differs: {_path}")
_write_immutable_drive_file(CODE_BINDING_PATH, CODE_BINDING_BYTES)
_write_immutable_drive_file(PLAN_PATH, PLAN_BYTES)

drive_available = CAPACITY["drive_available_bytes"]
matrix_peak = int(STORAGE_PROJECTION["drive"]["matrix_peak_with_reserve_bytes"])
matrix_status = "CAPACITY_UNKNOWN" if drive_available is None else (
    "PROJECTED_FIT" if drive_available >= matrix_peak else "INSUFFICIENT_FOR_FULL_MATRIX"
)
print(json.dumps({
    "status": "CANONICAL_STORAGE_READY_NEXT_RUN_GATED",
    "storage_account": CAPACITY["quota_account_email"],
    "canonical_revision_folder_id": CAPACITY["canonical_revision_folder_id"],
    "output_folder_owner": CAPACITY["output_destination"]["destination_owner"],
    "checkpoint_folder_owner": CAPACITY["checkpoint_destination"]["destination_owner"],
    "output_new_file_owner": CAPACITY["output_destination"]["new_file_owner"],
    "checkpoint_new_file_owner": CAPACITY["checkpoint_destination"]["new_file_owner"],
    "quota_status": CAPACITY["quota_status"], "drive_available_bytes": drive_available,
    "next_pending_run": None if NEXT_PENDING_RECORD is None else NEXT_PENDING_RECORD["run_key"],
    "next_run_peak_with_reserve_bytes": STORAGE_PROJECTION["drive"]["next_run_peak_with_reserve_bytes"],
    "full_matrix_projection_status_informational_only": matrix_status,
    "full_matrix_peak_with_reserve_bytes": matrix_peak,
    "completed_run_1": "FINALIZED_COMPLETED_RUN_REUSED_AND_SKIPPED",
    "run_1_measurement_recomputed": False, "run_1_registry_modified": False,
    "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
    "selected_git_commit": ACTUAL_GIT_COMMIT,
    "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    "plan_sha256": RUNTIME_PLAN["plan_sha256"],
    "runtime_plan_path": str(PLAN_PATH), "code_binding_path": str(CODE_BINDING_PATH),
    "checkpoint_policy": "local recovery every 5 minutes; Drive every 15 minutes plus task boundaries/interruption; two verified recovery generations; task boundaries retained separately",
}, sort_keys=True, indent=2))
'''


def _github_completed_run_guard_cell() -> str:
    return '''# Cell E — deliberate no-op: Run 1 is already finalized and registered.
print("ALREADY_FINALIZED_COMPLETED_RUN — Run 1 was identity-validated and its existing storage measurement reused in Cell C.")
print("This cell will not train, load a model, remeasure, re-register, or rerun finalization for Run 1.")
'''


def notebook_document(commit_sha: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("the stable Colab notebook requires a lowercase full 40-character Git commit SHA")
    document = zip_notebook_document("0" * 64)
    document["cells"][0]["source"] = (
        "# revision_R1 production — GitHub-pinned runner\n\n"
        "Cells A–G use one explicitly selected full Git commit per Colab session. Cell A fetches it once and checks out detached into a commit-versioned local directory, then verifies both manifests, the compatible training-core hash, and frozen schedule.\n\n"
        "Storage identity is independent of the Colab compute account: `kian.anb777@gmail.com`, ReVision folder ID `1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x`. Select that account in Cell A's Google authorization prompts. The mounted filesystem and Drive API are checked against the account, exact folder/paths, a tiny write/read/delete probe, new-file ownership, and quota. Wrong cached credentials fail closed; restart the runtime and choose the canonical account.\n\n"
        "The repository is currently public, so no GitHub token is required. If private later, use the fine-grained read-only Colab Secret `GITHUB_READ_TOKEN` (Contents: Read-only, restricted to this repo; enable notebook access). Never paste credentials into cells or outputs.\n\n"
        "Run A → B → C → D. D is read-only. B uses Drive datasets or checksum-verified local caches only; external downloads are disabled. Run 1 is already finalized/registered and is reused without remeasurement; E is a no-op. F remains disabled by default."
    ).splitlines(keepends=True)
    document["cells"][1]["source"] = _github_setup_cell(commit_sha).splitlines(keepends=True)
    document["cells"][2]["source"] = _github_data_cell().splitlines(keepends=True)
    document["cells"][3]["source"] = _github_readiness_cell().splitlines(keepends=True)
    document["cells"][5]["source"] = _github_completed_run_guard_cell().splitlines(keepends=True)
    cell_f = "".join(document["cells"][6]["source"])
    if "progress_callback=live_progress,\n    )" not in cell_f:
        raise ValueError("matrix continuation template changed; cannot bind the canonical Drive API client")
    cell_f = cell_f.replace(
        "progress_callback=live_progress,\n    )",
        "progress_callback=live_progress, api_factory=STORAGE_API_FACTORY,\n    )",
    )
    document["cells"][6]["source"] = cell_f.splitlines(keepends=True)
    cell_g = "".join(document["cells"][7]["source"])
    cell_g = cell_g.replace(
        "inspect_drive_destination(EXPORT_ROOT)",
        "inspect_drive_destination(EXPORT_ROOT, api_factory=STORAGE_API_FACTORY)",
    ).replace(
        "inspect_drive_destination(CHECKPOINT_BACKUP_ROOT)",
        "inspect_drive_destination(CHECKPOINT_BACKUP_ROOT, api_factory=STORAGE_API_FACTORY)",
    )
    document["cells"][7]["source"] = cell_g.splitlines(keepends=True)
    document["cells"] = document["cells"][:8]  # opening markdown, then A–G; no Run 1 finalizer.
    for index, cell in enumerate(document["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    return document


def write_notebook(output_path: str | Path, commit_sha: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing notebook: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(notebook_document(commit_sha), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return path
