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
    assert '_git(["checkout", "--detach", "FETCH_HEAD"]' in cells[1]
    assert "GITHUB_READ_TOKEN" in cells[1]
    assert "Authorization" in cells[1]
    assert "token contents are never displayed" in cells[1]
    assert '"GIT_TRACE": "0"' in cells[1]
    assert "if CHECKOUT_PATH.exists():" in cells[1]
    assert "sys.dont_write_bytecode = True" in cells[1]
    assert "git pull" not in cells[1]
    assert "verify_package_integrity(PACKAGE_ROOT)" in cells[1]
    assert EXPECTED_TRAINING_SOURCE_SHA256 in cells[1]
    assert EXPECTED_EFFECTIVE_SCHEDULE_SHA256 in cells[1]
    assert "np.__version__ != \"2.0.2\"" in cells[1]
    assert "torch.cuda.is_available()" in cells[1]
    assert "revision_R1_production_v1_4_1_20261028.zip" not in all_source
    assert "revision_R1_production_v1_4_1_20261028.zip.sha256" not in all_source
    assert "runtime_code_binding" in cells[3]
    assert "selected_git_commit" in cells[3]
    assert "output_new_file_owner" in cells[3]
    assert "checkpoint_new_file_owner" in cells[3]
    assert "quota_account_email" in cells[3]
    assert "LAUNCH_FIRST_PRODUCTION_RUN = False" in cells[5]
    assert "LAUNCH_REMAINING_MATRIX = False" in cells[6]
    assert "fine-grained read-only" in cells[0]
    assert "GITHUB_READ_TOKEN" in cells[0]
