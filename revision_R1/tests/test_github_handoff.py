from __future__ import annotations

import hashlib
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from scripts.build_github_handoff import (
    EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
    EXPECTED_TRAINING_SOURCE_SHA256,
    build_package_manifest,
    notebook_document,
    stage_clean_package,
    verify_package_integrity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_github_package_manifest_binds_frozen_core_schedule_and_allowlist(tmp_path):
    manifest = build_package_manifest(ROOT)

    assert manifest["release_kind"] == "GITHUB_PINNED_COLAB_RUNNER"
    assert manifest["training_source_sha256"] == EXPECTED_TRAINING_SOURCE_SHA256
    assert manifest["expected_effective_schedule_sha256"] == EXPECTED_EFFECTIVE_SCHEDULE_SHA256
    assert manifest["run_count"] == 100
    assert manifest["first_run_key"] == "ember_seed42_full_hybrid"
    assert manifest["protocol"]["classifier_batch_size"] == 256
    assert manifest["protocol"]["sampling"] == "mixed_pool_inverse_class_frequency_with_replacement"
    assert manifest["protocol"]["replay_coordinates"] == "raw"
    assert len(manifest["files"]) > 35
    assert manifest["files"]["revision_r1/runner.py"]["sha256"]
    assert "scripts/dataset_bootstrap.py" in manifest["files"]
    assert "scripts/storage_auth.py" in manifest["files"]
    assert "tests/test_dataset_bootstrap.py" in manifest["files"]
    assert "tests/test_storage_auth.py" in manifest["files"]
    assert "requirements-colab.txt" in manifest["files"]
    assert "requirements-test.txt" in manifest["files"]

    packaged_schedule = ROOT / "configs" / "update_budget.csv"
    packaged_schedule_sha = hashlib.sha256(packaged_schedule.read_bytes()).hexdigest()
    schedule_manifest = json.loads((ROOT / "configs" / "update_budget_manifest.json").read_text())
    assert manifest["packaged_schedule_sha256"] == packaged_schedule_sha
    assert packaged_schedule_sha == schedule_manifest["source_csv_sha256"]
    assert manifest["plan_sha256"] == json.loads(
        (ROOT / "configs" / "production_run_plan_v1.json").read_text()
    )["plan_sha256"]

    paths = set(manifest["files"])
    assert "tests/test_checkpoint_compatibility.py" in paths
    assert not any(
        path.lower().endswith((".npz", ".pt", ".pth", ".ckpt"))
        or "/results/" in path
        or "/checkpoints/" in path
        or "cache" in path.lower()
        for path in paths
    )
    clean_package = stage_clean_package(ROOT, tmp_path / "clean-package")
    verified, build_manifest, manifest_sha = verify_package_integrity(clean_package)
    assert verified == manifest
    assert build_manifest["package_manifest_sha256"] == manifest_sha
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stage_clean_package(ROOT, clean_package)


def test_clean_package_verifier_rejects_file_tampering(tmp_path):
    clean_package = stage_clean_package(ROOT, tmp_path / "tampered-package")
    target = clean_package / "tests" / "test_github_handoff.py"
    target.write_bytes(target.read_bytes() + b"# tampered\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_package_integrity(clean_package)


@pytest.mark.parametrize("secret_value,expected_api_calls", [(None, 1), ("sentinel-do-not-leak", 2)])
def test_private_repository_missing_or_invalid_secret_fails_without_token_leak(
    monkeypatch, capsys, secret_value, expected_api_calls,
):
    import urllib.request

    calls = []
    secret_names = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        status = 404 if len(calls) == 1 else 401
        raise urllib.error.HTTPError("https://api.github.com/repos/example/private", status, "denied", {}, None)

    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    fake_colab = types.ModuleType("google.colab")
    fake_colab.auth = types.SimpleNamespace(authenticate_user=lambda: None)
    fake_colab.drive = types.SimpleNamespace(mount=lambda *_args, **_kwargs: None)
    fake_colab.userdata = types.SimpleNamespace(
        get=lambda name: (secret_names.append(name), secret_value)[1]
    )
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    source = "".join(notebook_document("0123456789abcdef0123456789abcdef01234567")["cells"][1]["source"])
    with pytest.raises(RuntimeError) as error:
        exec(compile(source, "notebook-cell-A", "exec"), {})

    captured = capsys.readouterr()
    assert len(calls) == expected_api_calls
    assert secret_names == ["GITHUB_READ_TOKEN"]
    if secret_value:
        assert "HTTP 401" in str(error.value)
        assert "token contents are never displayed" in str(error.value)
    else:
        assert "GITHUB_READ_TOKEN" in str(error.value)
    assert "sentinel-do-not-leak" not in str(error.value)
    assert "sentinel-do-not-leak" not in captured.out
    assert "sentinel-do-not-leak" not in captured.err


def test_stable_notebook_pins_commit_and_keeps_production_gated():
    commit = "0123456789abcdef0123456789abcdef01234567"
    notebook = notebook_document(commit)
    assert len(notebook["cells"]) == 8  # opening markdown, then Cells A through G
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    all_source = "\n".join(cells)

    assert [f"Cell {letter}" in cells[index] for index, letter in enumerate("ABCDEFG", start=1)] == [True] * 7
    assert commit in cells[1]
    assert "auth.authenticate_user()" not in cells[1]
    assert "google.auth.default(" not in cells[1]
    assert "build_main_account_storage_api(" in cells[1]
    assert "drive.mount(" not in cells[1]
    assert "build_main_account_drive_v2_api(" not in cells[1]
    assert "lease_service=STORAGE_LEASE_API_SERVICE" not in cells[1]
    assert "DriveApiStorage" in cells[1]
    assert cells[1].index("verify_package_integrity(PACKAGE_ROOT)") < cells[1].index("build_main_account_storage_api(")
    assert cells[1].index('_git(["fetch"') < cells[1].index("build_main_account_storage_api(")
    assert "credential_source" in cells[1]
    assert "userdata.get" in cells[1]
    assert "kian.anb777@gmail.com" in cells[1]
    assert "1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x" in cells[1]
    assert "verify_canonical_storage_session(" in cells[1]
    assert "STORAGE_API_FACTORY" in cells[1]
    assert '"preflight_drive_path": "/content/drive/MyDrive/ReVision/revision_R1/results/preflight/preflight.json"' in cells[1]
    assert '"preflight_staging_root": "/content/revision_R1_storage_staging/preflight"' in cells[1]
    assert 'Path(DRIVE_PATHS["preflight_drive_path"]).parent' in cells[1]
    assert "read-only" in cells[1].lower()
    assert "storage account" in cells[1].lower()
    assert 'STORAGE_AUTH_EVIDENCE["credential_source"]' in cells[1]
    assert '_git(["checkout", "--detach", "FETCH_HEAD"]' in cells[1]
    assert "GITHUB_READ_TOKEN" in cells[1]
    assert "Authorization" in cells[1]
    assert "token contents are never displayed" in cells[1]
    assert '"GIT_TRACE": "0"' in cells[1]
    assert "if CHECKOUT_PATH.exists():" in cells[1]
    assert "sys.dont_write_bytecode = True" in cells[1]
    assert "git pull" not in cells[1]
    assert "verify_package_integrity(PACKAGE_ROOT)" in cells[1]
    assert "from scripts.build_production_handoff import PRODUCTION_BUILD_ID as EXPECTED_BUILD_ID" in cells[1]
    assert "from scripts.build_production_handoff import PRODUCTION_BUILD_ID as EXPECTED_BUILD_ID" in cells[1]
    assert EXPECTED_TRAINING_SOURCE_SHA256 in cells[1]
    assert EXPECTED_EFFECTIVE_SCHEDULE_SHA256 in cells[1]
    assert "EXPECTED_PYTHON_VERSION = (3, 12, 13)" in cells[1]
    assert "sys.version_info[:3] != EXPECTED_PYTHON_VERSION" in cells[1]
    assert "np.__version__ != \"2.0.2\"" in cells[1]
    assert 'EXPECTED_TORCH_VERSION = "2.11.0+cu128"' in cells[1]
    assert "torch.__version__ != EXPECTED_TORCH_VERSION" in cells[1]
    assert 'EXPECTED_CUDA_VERSION = "12.8"' in cells[1]
    assert "torch.version.cuda != EXPECTED_CUDA_VERSION" in cells[1]
    assert "torch.cuda.is_available()" in cells[1]
    assert "revision_R1_production_v1_4_1_20261028.zip" not in all_source
    assert "revision_R1_production_v1_4_1_20261028.zip.sha256" not in all_source
    assert "runtime_code_binding" in cells[3]
    assert "selected_git_commit" in cells[3]
    assert "output_new_file_owner" in cells[3]
    assert "checkpoint_new_file_owner" in cells[3]
    assert "quota_account_email" in cells[3]
    assert "reuse_validated_first_run_measurement(" in cells[3]
    assert "matrix_registry_evidence(STORAGE_ADAPTER, OUTPUT_ROOT, RUNTIME_PLAN)" in cells[3]
    assert "CELL_C_CANONICAL_COMPLETION_REGISTRY" in cells[3]
    assert "require_run_capacity(CAPACITY, STORAGE_PROJECTION, scope=\"first_run\")" not in cells[3]
    assert "prepare_dataset_roots" not in cells[2]
    assert "inspect_dataset_sources(" in cells[2]
    assert "no external download will be attempted" in cells[2]
    assert "if _npz.is_file():" in cells[2]
    assert "LAUNCH_FIRST_PRODUCTION_RUN" not in cells[5]
    assert "ALREADY_FINALIZED_COMPLETED_RUN" in cells[5]
    assert "execute_first_production_run(" not in cells[5]
    assert "CELL_E_REUSES_CELL_C_REGISTRY_EVIDENCE" in cells[5]
    assert "LAUNCH_REMAINING_MATRIX = False" in cells[6]
    assert "api_factory=STORAGE_API_FACTORY" in cells[6]
    assert "api_factory=STORAGE_API_FACTORY" in cells[7]
    assert "finalize_completed_production_run(" not in all_source
    assert "execute_first_production_run(" not in all_source
    assert "continue_production_matrix(" in cells[6]
    assert "CLEAN_RESTART_RUN2 = True" in cells[6]
    assert "ACKNOWLEDGE_NO_OTHER_COLAB_WORKER = False" in cells[6]
    assert "clean_restart_run2=CLEAN_RESTART_RUN2" in cells[6]
    assert "acknowledge_no_other_worker=ACKNOWLEDGE_NO_OTHER_COLAB_WORKER" in cells[6]
    assert "Run 1 remains finalized/skipped" in cells[6]
    assert "complete_exported_run(" not in cells[6]
    assert "prepare_pending_checkpoint_publication(" not in cells[6]
    assert "resolve_matrix_run_paths" in cells[3]
    assert "replacement_completion_matches_route" in cells[3]
    assert "CELL_C_EFFECTIVE_RUN2_DISPOSITION" in cells[3]
    assert "resolve_matrix_run_paths" in cells[4]
    assert "run2_registry_disposition" in cells[4]
    assert "registered_run_output_path" in cells[7]
    assert "registered_run_checkpoint_path" in cells[7]
    assert "clean_replacement_run2=True" in cells[7]
    assert "EFFECTIVE_COMPLETED_RUN_KEYS" in cells[7]
    assert "STORAGE_ADAPTER.publish_file(" not in cells[4]
    assert "Diagnostic saved locally:" in cells[4]
    assert "ColabProgressDisplay(" in cells[4]
    assert "training_source_sha256" in cells[3]
    assert "selected_git_commit" in cells[3]
    assert "fine-grained read-only" in cells[0]
    assert "GITHUB_READ_TOKEN" in cells[0]
    assert "REVISION_R1_STORAGE_REFRESH_TOKEN" in cells[0]
    assert "REVISION_R1_STORAGE_OAUTH_CLIENT_ID" in cells[0]
    assert "REVISION_R1_STORAGE_OAUTH_CLIENT_SECRET" in cells[0]
    assert "dedicated user-OAuth refresh token" in cells[0]


def test_notebook_builder_uses_saved_drive_notebook_without_temporary_recovery_cells():
    saved_drive_sha256 = "a220a08e0ff472876a479eaded35cd7946bc5d0c35ec245a7203c33a42ff3f9d"
    template_path = ROOT / "notebooks" / "Revision_R1_Production_Colab_Saved_Drive_Base.ipynb"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["metadata"]["revision_r1_saved_drive_source_sha256"] == saved_drive_sha256
    assert len(template["cells"]) == 8

    generated = notebook_document("0123456789abcdef0123456789abcdef01234567")
    cells = ["".join(cell.get("source", [])) for cell in generated["cells"]]
    assert len(cells) == 8
    assert "PINNED_GIT_COMMIT = \"0123456789abcdef0123456789abcdef01234567\"" in cells[1]
    assert "stage_canonical_preflight_artifacts(" in cells[2]
    assert "ColabProgressDisplay(" in cells[4]
    assert "ALREADY_FINALIZED_COMPLETED_RUN" in cells[5]
    assert "continue_production_matrix(" in cells[6]
    assert "COMPLETE_EXPORTED_RUN_ONLY" not in "\n".join(cells)
    assert "_r1_helpers" not in "\n".join(cells)
    assert "revision_R1_saved_drive_source_sha256" not in "\n".join(cells)


def test_cell_b_preflight_uses_api_staging_and_frozen_schedule_digest():
    commit = "0123456789abcdef0123456789abcdef01234567"
    cells = ["".join(cell.get("source", [])) for cell in notebook_document(commit)["cells"]]
    all_source = "\n".join(cells)
    cell_b = cells[2]

    assert "stage_canonical_preflight_artifacts(" in cell_b
    assert 'DRIVE_PATHS["preflight_drive_path"]' in cell_b
    assert 'DRIVE_PATHS["preflight_staging_root"]' in cell_b
    assert 'PREFLIGHT_PATH = Path(PREFLIGHT_STAGING_EVIDENCE["preflight_path"])' in cell_b
    assert "print(json.dumps(PREFLIGHT_STAGING_EVIDENCE, sort_keys=True, indent=2))" in cell_b
    assert 'PREFLIGHT_PATH = Path(DRIVE_PATHS["preflight_drive_path"])' not in cell_b
    assert 'if not PREFLIGHT_PATH.is_file()' in cell_b  # checks only the already API-staged local file
    assert "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212" in cells[1]
    assert "Cells A through C require no DriveFS mount" in cells[0]


def test_generated_cell_f_is_one_disabled_launch_resume_path():
    from scripts.build_github_handoff import _github_matrix_cell
    source = _github_matrix_cell()
    assert "LAUNCH_REMAINING_MATRIX = False" in source
    assert "ACKNOWLEDGE_NO_OTHER_COLAB_WORKER = False" in source
    assert "CLEAN_RESTART_RUN2 = True" in source
    assert "continue_production_matrix(" in source
    assert "clean_restart_run2=CLEAN_RESTART_RUN2" in source
    assert "PREPARE_EXISTING_CHECKPOINT_ONLY" not in source
    assert "complete_exported_run(" not in source
    assert "Run interrupted safely. Re-run Cell D, then rerun this same Cell F" in source


def test_user_edited_cell_d_is_local_only_and_keeps_readable_progress():
    cells = ["".join(cell.get("source", [])) for cell in notebook_document(
        "0123456789abcdef0123456789abcdef01234567"
    )["cells"]]
    diagnostic = cells[4]
    assert "ColabProgressDisplay(" in diagnostic
    assert "live_progress =" in diagnostic
    assert "Diagnostic saved locally:" in diagnostic
    assert "STORAGE_ADAPTER.publish_file(" not in diagnostic
    assert "resolve_matrix_run_paths" in diagnostic
    assert "_next_progress_relative" in diagnostic
    assert "next_run_recovery_checkpoint_metadata_only" in diagnostic


def test_production_readme_has_one_github_execution_order_and_no_zip_setup():
    readme = (ROOT / "PRODUCTION_COLAB_README.md").read_text(encoding="utf-8")

    assert readme.count("## Exact execution order") == 1
    assert "verify the distributed ZIP" not in readme
    assert "Revision_R1_Production_Colab.ipynb" in readme
    assert "kian.anb777@gmail.com" in readme
    assert "Mixed-owner historical generations" in readme
    assert "PyTorch 2.11.0+cu128" in readme
    assert "PyTorch CUDA build 12.8" in readme
