from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_production_handoff import (
    CODE_ZIP_FILENAME,
    DISTRIBUTED_NOTEBOOK_FILENAME,
    PRODUCTION_BUILD_ID,
    build_handoff,
    notebook_document,
)
from scripts.production_workflow import (
    bind_runtime_schedule,
    build_frozen_run_plan,
    export_checkpoint_backup,
    export_lightweight_results,
    load_v62_recovery_checkpoint_bound,
    run_checkpoint_root,
    run_output_root,
    _validate_registered_completion_for_export,
)
from revision_r1.registry import RunRegistry
from revision_r1.state import PhaseCursor, RunState
from revision_r1.checkpointing import save_generation
from revision_r1.config import DATASETS, SEEDS
from revision_r1.storage import DEFAULT_STORAGE_RESERVE_BYTES, StorageCapacityError
from test_production_workflow import (
    DATASET_REFERENCES, SCHEDULE_SHA, _preflight,
    _write_effective_schedule, _write_v62_producer_report_fixture,
)
import scripts.production_workflow as production_workflow


def _code_cells(document: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"]


def _bound_plan_for_archive_fixture(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule_dir = tmp_path / "effective"
    schedule_dir.mkdir()
    schedule_path, manifest_path = _write_effective_schedule(schedule_dir, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=schedule_sha,
        dataset_references=DATASET_REFERENCES,
    )
    return bind_runtime_schedule(
        plan, schedule_path, manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )


def test_production_notebook_cells_parse_and_are_ordered_with_launches_disabled():
    pinned_zip_sha256 = "a" * 64
    document = notebook_document(pinned_zip_sha256)
    code_cells = _code_cells(document)
    for source in code_cells:
        ast.parse(source)
    joined = "\n".join(code_cells)

    assert len(code_cells) == 7
    assert [source.splitlines()[0].split("—")[0].strip() for source in code_cells] == [
        "# Cell A", "# Cell B", "# Cell C", "# Cell D", "# Cell E", "# Cell F", "# Cell G",
    ]
    assert "inspect_first_run_lease_and_progress(" in code_cells[3]
    assert "READ-ONLY DIAGNOSTIC SAVED" in code_cells[3]
    assert "NO training was launched" in code_cells[3]
    assert "process_start_time_utc" in code_cells[3]
    assert '"pid": 44230' in code_cells[3]
    assert "FIRST-RUN RESUME SUMMARY" in code_cells[3]
    assert "first_run_resume_gate" in code_cells[3]
    assert "BoundedProgressBuffer(max_events=5)" in code_cells[3]
    assert "PROGRESS_HANDLE.update(text)" in code_cells[3]
    assert "historical_oom_correlation" in code_cells[3]
    assert "NO_PID_MATCH_IN_PERSISTED_RUN_RECORDS" in code_cells[3]
    assert "LAUNCH_FIRST_PRODUCTION_RUN = False" in code_cells[4]
    assert "ACKNOWLEDGE_NO_OTHER_COLAB_WORKER = False" in code_cells[4]
    assert "PASS_RESUME_CANDIDATE" in code_cells[4]
    assert "never initializes from scratch" in code_cells[4]
    assert "ACKNOWLEDGE_V12_EVIDENCE_PRESERVED_AND_START_FRESH" not in joined
    assert "execute_first_production_run(" in code_cells[4]
    assert "continue_production_matrix(" not in code_cells[4]
    assert "LAUNCH_REMAINING_MATRIX = False" in code_cells[5]
    assert "continue_production_matrix(" in code_cells[5]
    assert "execute_first_production_run(" not in code_cells[5]
    assert "EXPORT_CHECKPOINT_BACKUP = False" in code_cells[6]
    assert "export_lightweight_results(" in code_cells[6]
    assert "export_checkpoint_backup(" in code_cells[6]
    assert "if EXPORT_CHECKPOINT_BACKUP:" in code_cells[6]
    assert "from pathlib import Path, PurePosixPath" in code_cells[0]
    assert "PurePosixPath(member.filename)" in code_cells[0]
    assert f'EXPECTED_ZIP_SHA256 = "{pinned_zip_sha256}"' in code_cells[0]
    assert "revision_R1_production_v1_4_1_20261028" in joined
    assert "Revision_R1_Production_Colab_v1_4_1_20261028" not in joined
    assert "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212" in joined
    assert "training_reproduction_claimed" in joined


def test_cell_c_executes_v62_reader_and_interpolates_the_runtime_plan_digest(tmp_path):
    source = next(
        item for item in _code_cells(notebook_document("a" * 64))
        if item.startswith("# Cell C")
    )
    tree = ast.parse(source)
    assignments = {
        target.id: statement.value
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }

    report_path = _write_v62_producer_report_fixture(tmp_path)
    reader_expression = compile(
        ast.Expression(assignments["V62_REFERENCE"]), "<Cell C V62_REFERENCE>", "eval",
    )
    reference = eval(reader_expression, {
        "DRIVE_PATHS": {"v62_recovery_report": str(report_path)},
        "load_v62_recovery_checkpoint_bound": load_v62_recovery_checkpoint_bound,
    })
    assert reference["reference_generation"] == 79
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert reference["reference_manifest_sha256"] == report["reference_checkpoint"]["manifest_sha256"]

    plan_sha = "c" * 64
    plan_expression = compile(
        ast.Expression(assignments["PLAN_PATH"]), "<Cell C PLAN_PATH>", "eval",
    )
    plan_path = eval(plan_expression, {
        "OUTPUT_ROOT": tmp_path,
        "RUNTIME_PLAN": {"plan_sha256": plan_sha},
    })
    assert plan_path == tmp_path / "plans" / f"production_run_plan_{plan_sha}.json"


def test_cell_g_interpolates_plan_digest_and_unique_export_timestamps(tmp_path):
    source = next(
        item for item in _code_cells(notebook_document("a" * 64))
        if item.startswith("# Cell G")
    )
    tree = ast.parse(source)
    assignments = {
        target.id: statement.value
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    plan_sha = "d" * 64
    time = type("FakeTime", (), {"time_ns": staticmethod(lambda: 123456)})
    context = {
        "EXPORT_ROOT": tmp_path / "exports",
        "CHECKPOINT_BACKUP_ROOT": tmp_path / "backups",
        "RUNTIME_PLAN": {"plan_sha256": plan_sha},
        "time": time,
    }
    expected_names = {
        "PROGRESS_PATH": f"production_progress_{plan_sha[:12]}_123456.json",
        "RESULTS_PATH": f"revision_R1_lightweight_results_{plan_sha[:12]}_123456.zip",
        "BACKUP_PATH": f"revision_R1_checkpoint_backup_{plan_sha[:12]}_123456.zip",
    }
    for variable, expected_name in expected_names.items():
        value = eval(compile(ast.Expression(assignments[variable]), f"Cell G {variable}", "eval"), context)
        assert value.name == expected_name


def test_production_handoff_builder_emits_unique_verified_zip_and_plan(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    old_v12 = repository_root / "revision_R1/dist/production_v1_2_handoff_20261027/revision_R1_production_v1_2_20261027.zip"
    if not old_v12.is_file():
        pytest.skip("the authentic v1.2 build-time source archive is intentionally not bundled in the runtime ZIP")
    old_v62 = repository_root / "revision_R1/dist/colab_validation_v6_2_final_gan_recovery_qa/revision_R1_code_v6_2_final_gan_recovery_qa.zip"
    old_hash_before = hashlib.sha256(old_v62.read_bytes()).hexdigest() if old_v62.is_file() else None
    release = tmp_path / "production-release"

    result = build_handoff(repository_root, release)

    code_zip = release / CODE_ZIP_FILENAME
    notebook_path = release / DISTRIBUTED_NOTEBOOK_FILENAME
    plan_path = release / "production_run_plan_v1.json"
    assert code_zip.is_file() and notebook_path.is_file() and plan_path.is_file()
    assert hashlib.sha256(code_zip.read_bytes()).hexdigest() == Path(str(code_zip) + ".sha256").read_text().split()[0]
    assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == Path(str(notebook_path) + ".sha256").read_text().split()[0]
    zip_digest = hashlib.sha256(code_zip.read_bytes()).hexdigest()
    notebook_cells = _code_cells(json.loads(notebook_path.read_text(encoding="utf-8")))
    assert f'EXPECTED_ZIP_SHA256 = "{zip_digest}"' in notebook_cells[0]
    assert result["production_build_id"] == PRODUCTION_BUILD_ID
    assert result["production_matrix_launched"] is False
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["run_count"] == 100 and len(plan["runs"]) == 100
    assert plan["runs"][0]["run_key"] == "ember_seed42_full_hybrid"
    assert plan["training_reproduction_claimed"] is False

    with zipfile.ZipFile(code_zip) as archive:
        names = archive.namelist()
        assert "PACKAGE_MANIFEST.json" in names
        assert "revision_R1/scripts/production_workflow.py" in names
        assert "revision_R1/scripts/live_progress.py" in names
        assert "revision_R1/configs/production_run_plan_v1.json" in names
        assert "revision_R1/configs/legacy_v1_2_first_run_identity.json" in names
        assert "revision_R1/tests/test_live_progress.py" in names
        assert "revision_R1/Revision_R1_Production_Colab_v1_2_20261027.ipynb" not in names
        assert not any(name.lower().endswith((".npz", ".pt", ".pth", ".ckpt")) for name in names)
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert manifest["production_build_id"] == PRODUCTION_BUILD_ID
        assert manifest["effective_schedule_sha256"] == SCHEDULE_SHA
        assert manifest["training_source_sha256"] == plan["training_source_sha256"]
        legacy = json.loads(archive.read("revision_R1/configs/legacy_v1_2_first_run_identity.json"))
        assert legacy["zip_sha256"] == "2647c6bb51b3e3c28fd3009008f6f616deac5e73f725fd20e66aebbb23097d9c"
        assert legacy["training_source_sha256"] == "bd175dcc27492f848459653536c63aa7bed8421bc16c60d7141fa359004e8660"
        assert legacy["verified_training_core_file_count"] == 27
        assert manifest["legacy_v1_2_checkpoint_handling"]["automatic_resume_from_v1_2_checkpoint"] is False
        for relative, item in manifest["files"].items():
            payload = archive.read(relative)
            assert len(payload) == item["bytes"]
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        packaged_schedule = hashlib.sha256(archive.read("revision_R1/configs/update_budget.csv")).hexdigest()
        packaged_schedule_manifest = json.loads(archive.read("revision_R1/configs/update_budget_manifest.json"))
        assert packaged_schedule == packaged_schedule_manifest["source_csv_sha256"]
        assert packaged_schedule != SCHEDULE_SHA

    if old_hash_before is not None:
        assert hashlib.sha256(old_v62.read_bytes()).hexdigest() == old_hash_before
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_handoff(repository_root, release)
    if old_hash_before is not None:
        assert hashlib.sha256(old_v62.read_bytes()).hexdigest() == old_hash_before


def test_checkpoint_backup_checks_capacity_before_publication_and_binds_identity(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule_dir = tmp_path / "effective"
    schedule_dir.mkdir()
    schedule_path, manifest_path = _write_effective_schedule(schedule_dir, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=schedule_sha,
        dataset_references=DATASET_REFERENCES,
    )
    plan = bind_runtime_schedule(
        plan, schedule_path, manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    record = plan["runs"][0]
    checkpoint_root = tmp_path / "checkpoints"
    identity = production_workflow._expected_plan_runner_identity(plan, record)
    state = RunState(identity=identity, cursor=PhaseCursor(task_index=0, phase="complete"))
    save_generation(run_checkpoint_root(checkpoint_root, record) / "recovery", state, identity, "complete")
    export_root = tmp_path / "exports"
    export_root.mkdir()
    archive_path = export_root / "checkpoint_backup.zip"
    snapshot = {
        "path": str(export_root.resolve()), "write_capability_verified": True,
        "new_file_owner_verified": True, "quota_metadata_verified": True,
        "drive_id": None, "available_bytes": 0,
    }

    with pytest.raises(StorageCapacityError, match="checkpoint backup capacity"):
        export_checkpoint_backup(plan, checkpoint_root, archive_path, storage_snapshot=snapshot)
    assert not archive_path.exists()

    checkpoint_source_bytes = sum(
        path.stat().st_size for path in checkpoint_root.rglob("generation-*") if path.is_file()
    )
    snapshot["available_bytes"] = (
        checkpoint_source_bytes + 16 * 1024 * 1024 + DEFAULT_STORAGE_RESERVE_BYTES
    )
    with pytest.raises(StorageCapacityError, match="checkpoint backup capacity"):
        export_checkpoint_backup(plan, checkpoint_root, archive_path, storage_snapshot=snapshot)
    assert not archive_path.exists()

    snapshot["available_bytes"] = 2_000_000_000
    nested_archive = run_checkpoint_root(checkpoint_root, record) / "backup.zip"
    nested_snapshot = {
        **snapshot, "path": str(nested_archive.parent.resolve()),
    }
    with pytest.raises(ValueError, match="outside the checkpoint source tree"):
        export_checkpoint_backup(plan, checkpoint_root, nested_archive, storage_snapshot=nested_snapshot)
    assert not nested_archive.exists()

    backup = export_checkpoint_backup(plan, checkpoint_root, archive_path, storage_snapshot=snapshot)
    assert backup["status"] == "EXPORTED_VALIDATED_CHECKPOINT_BACKUP"
    assert backup["generation_count"] == 1
    assert backup["archive_capacity_upper_bound_bytes"] > backup["validated_checkpoint_source_bytes"]
    assert backup["archive_growth_margin_fraction"] == 0.01
    assert archive_path.is_file()
    assert Path(str(archive_path) + ".sha256").is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert "ARCHIVE_MANIFEST.json" in archive.namelist()
        assert any(name.endswith("generation-000001.pt") for name in archive.namelist())


def test_checkpoint_backup_rejects_identity_mismatch_without_creating_archive(tmp_path, monkeypatch):
    preflight = _preflight()
    schedule_dir = tmp_path / "effective"
    schedule_dir.mkdir()
    schedule_path, manifest_path = _write_effective_schedule(schedule_dir, preflight)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    monkeypatch.setattr(production_workflow, "EXPECTED_EFFECTIVE_SCHEDULE_SHA256", schedule_sha)
    preflight["schedule"]["sha256"] = schedule_sha
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=schedule_sha,
        dataset_references=DATASET_REFERENCES,
    )
    plan = bind_runtime_schedule(
        plan, schedule_path, manifest_path, preflight,
        expected_anchor_corrections=0, expected_budget_changes=0,
    )
    record = plan["runs"][0]
    checkpoint_root = tmp_path / "checkpoints"
    identity = production_workflow._expected_plan_runner_identity(plan, record)
    state = RunState(identity=identity, cursor=PhaseCursor(task_index=0, phase="complete"))
    manifest = save_generation(run_checkpoint_root(checkpoint_root, record) / "recovery", state, identity, "complete")
    manifest_path = next(run_checkpoint_root(checkpoint_root, record).rglob("generation-*.manifest.json"))
    broken = json.loads(manifest_path.read_text())
    broken["identity"]["schedule_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(broken), encoding="utf-8")
    export_root = tmp_path / "exports"
    export_root.mkdir()
    archive_path = export_root / "checkpoint_backup.zip"
    snapshot = {
        "path": str(export_root.resolve()), "write_capability_verified": True,
        "new_file_owner_verified": True, "quota_metadata_verified": True,
        "drive_id": None, "available_bytes": 2_000_000_000,
    }

    with pytest.raises(ValueError, match="identity"):
        export_checkpoint_backup(plan, checkpoint_root, archive_path, storage_snapshot=snapshot)
    assert not archive_path.exists()


def test_lightweight_archive_target_cannot_modify_a_run_artifact_namespace(tmp_path):
    plan = build_frozen_run_plan(
        Path(__file__).resolve().parents[1], schedule_sha256=SCHEDULE_SHA,
        dataset_references=DATASET_REFERENCES,
    )
    output_root = tmp_path / "production-results"
    target = run_output_root(output_root, plan["runs"][0]) / "accidental-results.zip"

    with pytest.raises(ValueError, match="outside every run-artifact namespace"):
        export_lightweight_results(plan, output_root, target, storage_snapshot={})
    assert not target.exists()


def test_lightweight_export_metadata_checks_identity_without_rehashing_full_artifact_tree(tmp_path, monkeypatch):
    plan = _bound_plan_for_archive_fixture(tmp_path, monkeypatch)
    record = plan["runs"][0]
    output_root = tmp_path / "results"
    run_root = run_output_root(output_root, record)
    run_root.mkdir(parents=True)
    artifact_manifest = run_root / "artifact_manifest.json"
    artifact_manifest.write_text("{}", encoding="utf-8")
    run_id = production_workflow._run_id_for(record, synthetic=False)
    identity = production_workflow._expected_plan_runner_identity(plan, record)
    registry = RunRegistry(run_root / "registry")
    registry.begin_attempt(run_id, identity=identity)
    registry.complete(run_id, {"artifact_manifest": artifact_manifest})
    config = {**record["config"], "config_sha256": record["config_sha256"],
              "code_sha256": plan["training_source_sha256"], "namespace": "PRODUCTION"}
    summary = {
        "dataset": record["dataset"], "seed": record["seed"], "arm": record["arm"],
        "schedule_sha256": plan["schedule_sha256"], "training_reproduction_claimed": False,
    }
    status = {"status": "COMPLETED_VALIDATED", "namespace": "PRODUCTION"}
    class_order = {
        "class_order": record["class_order"],
        "sha256": record["class_order_sha256"],
    }
    for filename, value in (("config.json", config), ("full-result.json", summary),
                            ("status.json", status), ("class_order.json", class_order)):
        (run_root / filename).write_text(json.dumps(value), encoding="utf-8")
    entry = {
        "run_id": run_id,
        "artifact_manifest_sha256": hashlib.sha256(artifact_manifest.read_bytes()).hexdigest(),
    }

    checked = _validate_registered_completion_for_export(plan, record, run_root, entry)
    assert checked["run_id"] == run_id

    summary["arm"] = "wrong-arm"
    (run_root / "full-result.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen plan identity"):
        _validate_registered_completion_for_export(plan, record, run_root, entry)
