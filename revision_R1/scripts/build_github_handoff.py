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


EXPECTED_TRAINING_SOURCE_SHA256 = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
GITHUB_REPOSITORY = "KianAnbarestani/AHR-MalCL-HR"
GITHUB_TOKEN_SECRET_NAME = "GITHUB_READ_TOKEN"
PACKAGE_MANIFEST_NAME = "PACKAGE_MANIFEST.json"
BUILD_MANIFEST_NAME = "BUILD_MANIFEST.json"
EXTRA_TESTS = ("test_github_handoff.py",)
EXTRA_FILES = (
    "COMPATIBILITY_REPORT.md",
    "FROZEN_PROTOCOL.md",
    "legacy_checkpoint_source_compatibility.json",
    "PRODUCTION_COLAB_README.md",
    "README_GITHUB_COLAB.md",
    "requirements-colab.txt",
    "requirements-test.txt",
)
SCRIPT_FILES = (
    "scripts/build_github_handoff.py",
    "scripts/build_production_handoff.py",
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
    digest = hashlib.sha256()
    sources = sorted((root / "revision_r1").glob("*.py"))
    if not sources:
        raise FileNotFoundError(f"no training source modules under {root / 'revision_r1'}")
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        or package_manifest.get("expected_effective_schedule_sha256") != EXPECTED_EFFECTIVE_SCHEDULE_SHA256
        or package_manifest.get("production_enabled_by_default") is not False):
        raise ValueError("PACKAGE_MANIFEST release, repository, core, schedule, or production-gate identity is incompatible")
    if (build_manifest.get("package_manifest_sha256") != package_manifest_sha
        or build_manifest.get("training_source_sha256") != EXPECTED_TRAINING_SOURCE_SHA256
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
drive.mount("/content/drive", force_remount=False)
auth.authenticate_user()

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
import numpy as np, platform, torch
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Expected Colab Python 3.12, got {platform.python_version()}")
if np.__version__ != "2.0.2":
    raise RuntimeError(f"Expected pinned NumPy 2.0.2, got {np.__version__}; no packages were changed")
if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
    raise RuntimeError(f"Unsupported PyTorch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. No training started; select a GPU Colab runtime and rerun Cell A.")
from scripts.production_workflow import training_source_sha256
import revision_r1.runner as runner_module
import scripts.production_workflow as production_workflow_module
ACTUAL_TRAINING_SOURCE_SHA256 = training_source_sha256(PROJECT)
if (ACTUAL_TRAINING_SOURCE_SHA256 != EXPECTED_TRAINING_SOURCE_SHA256
    or runner_module._code_sha256() != EXPECTED_TRAINING_SOURCE_SHA256
    or PACKAGE_MANIFEST.get("training_source_sha256") != ACTUAL_TRAINING_SOURCE_SHA256):
    raise ValueError("Training-core hash is incompatible with the resumable first-run checkpoint; no fallback/restart is allowed")
if not Path(runner_module.__file__).resolve().is_relative_to(PROJECT.resolve()):
    raise ImportError(f"Training runner imported outside the pinned Git checkout: {runner_module.__file__}")
print(json.dumps({"repository": GITHUB_REPOSITORY, "repository_private": bool(REPOSITORY_METADATA.get("private")),
                  "selected_git_commit": ACTUAL_GIT_COMMIT, "checkout": str(CHECKOUT_PATH),
                  "detached_checkout": True, "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
                  "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
                  "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
                  "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0)}, indent=2))
'''
    return template.replace("__PINNED_COMMIT__", commit_sha)


def notebook_document(commit_sha: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("the stable Colab notebook requires a lowercase full 40-character Git commit SHA")
    document = zip_notebook_document("0" * 64)
    document["cells"][0]["source"] = (
        "# revision_R1 production — GitHub-pinned runner\n\n"
        "Cells A–G use one explicitly selected full Git commit per Colab session. Cell A fetches it once and checks out detached into a commit-versioned local directory, then verifies both manifests, the compatible training-core hash, and the frozen schedule. It never auto-runs the latest branch head.\n\n"
        "The repository is currently public, so no GitHub token is required. If it is later made private, use the fine-grained read-only Colab Secret `GITHUB_READ_TOKEN` (Contents: Read-only, restricted to this repository; enable notebook access). Never paste credentials into cells or outputs.\n\n"
        "Run A → B → C → D; D is read-only. Cell C verifies shared-Editor write/quota evidence and binds the selected commit and protocol to the frozen plan. Production Cells E/F remain disabled by default. The Drive paths, checkpoint identities, and resume behavior are unchanged."
    ).splitlines(keepends=True)
    document["cells"][1]["source"] = _github_setup_cell(commit_sha).splitlines(keepends=True)

    cell_c = "".join(document["cells"][3]["source"])
    insertion = '''# Bind this exact Git commit separately from the training-source/checkpoint identity.
CODE_BINDING_PATH = (OUTPUT_ROOT / "runtime_code_binding" / RUNTIME_PLAN["plan_sha256"]
                     / f"{ACTUAL_GIT_COMMIT}.json")
CODE_BINDING = {
    "schema_version": 1,
    "selected_git_commit": ACTUAL_GIT_COMMIT,
    "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
    "training_source_sha256": ACTUAL_TRAINING_SOURCE_SHA256,
    "effective_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    "frozen_plan_sha256": RUNTIME_PLAN["plan_sha256"],
    "first_run_key": RUNTIME_PLAN["runs"][0]["run_key"],
    "first_run_config_sha256": RUNTIME_PLAN["runs"][0]["config_sha256"],
}
CODE_BINDING_BYTES = (json.dumps(CODE_BINDING, sort_keys=True, indent=2, allow_nan=False) + "\\n").encode("utf-8")
CODE_BINDING_PATH.parent.mkdir(parents=True, exist_ok=True)
if CODE_BINDING_PATH.exists():
    if CODE_BINDING_PATH.read_bytes() != CODE_BINDING_BYTES:
        raise ValueError(f"Existing commit/protocol binding differs; preserved without overwrite: {CODE_BINDING_PATH}")
else:
    try:
        with CODE_BINDING_PATH.open("xb") as _binding_stream:
            _binding_stream.write(CODE_BINDING_BYTES)
            _binding_stream.flush()
            os.fsync(_binding_stream.fileno())
    except FileExistsError:
        if CODE_BINDING_PATH.read_bytes() != CODE_BINDING_BYTES:
            raise ValueError(f"Concurrent commit/protocol binding differs; preserved: {CODE_BINDING_PATH}")

'''
    anchor = "# Persist the runtime-bound plan immutably so a runtime reset resumes the same plan.\n"
    if anchor not in cell_c:
        raise ValueError("production Cell C template changed; cannot safely inject the immutable code binding")
    cell_c = cell_c.replace(anchor, insertion + anchor, 1)
    cell_c = cell_c.replace('"checkpoint_folder_owner": CAPACITY["checkpoint_destination"]["destination_owner"],',
        '"checkpoint_folder_owner": CAPACITY["checkpoint_destination"]["destination_owner"],\n'
        '    "output_new_file_owner": CAPACITY["output_destination"]["new_file_owner"],\n'
        '    "checkpoint_new_file_owner": CAPACITY["checkpoint_destination"]["new_file_owner"],\n'
        '    "new_file_owner_verified": CAPACITY["new_file_owner_verified"],')
    document["cells"][3]["source"] = cell_c.splitlines(keepends=True)
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
