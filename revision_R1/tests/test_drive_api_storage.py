from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pytest
import numpy as np

from revision_r1 import drive_storage
from revision_r1.drive_storage import (
    AmbiguousDrivePathError,
    DriveApiLease,
    DriveApiStorage,
    DriveLeaseError,
    DriveStorageError,
    verify_conditional_update_support,
)


OWNER = "kian.anb777@gmail.com"


class _HttpError(Exception):
    def __init__(self, status):
        self.resp = type("Response", (), {"status": status})()


class _Request:
    def __init__(self, execute, response_headers=None):
        self._execute = execute
        self.headers = {}
        self._response_headers = response_headers or {}
        self._response_callbacks = []

    def add_response_callback(self, callback):
        self._response_callbacks.append(callback)

    def execute(self):
        result = self._execute(self)
        for callback in self._response_callbacks:
            callback(self._response_headers)
        return result


class _FakeFiles:
    def __init__(self, records):
        self.records = records
        self.next_id = 1
        self.update_calls = 0
        self.fail_next_create = False
        self.fail_create_name = None
        self.fail_create_status = 403
        self.hide_probe_names_from_list = False
        self.owner_override = None
        self.owned_by_me_override = None
        self.response_etag_header_name = "etag"
        self.omit_response_etag = False
        self.response_etag_override = None
        self.metadata_parent_override = None
        self.metadata_file_id_override = None
        self.ignore_if_match = False

    @staticmethod
    def _contents(media_body):
        if isinstance(media_body, Path):
            return media_body.read_bytes()
        if isinstance(media_body, bytes):
            return media_body
        raise AssertionError(f"unexpected fake media type: {type(media_body)}")

    def list(self, *, q, **_kwargs):
        parent = re.search(r"'((?:\\.|[^'])+)' in parents", q).group(1).replace("\\'", "'")
        match = re.search(r"name='((?:\\.|[^'])*)'", q)
        name = match.group(1).replace("\\'", "'") if match else None
        rows = [row for row in self.records.values()
                if parent in row.get("parents", []) and (name is None or row["name"] == name)
                and not row.get("trashed", False)
                and not (self.hide_probe_names_from_list and row["name"].startswith(".revision-r1-capacity-probe-"))]
        return _Request(lambda _request: {"files": rows})

    def generateIds(self, *, count, space):
        assert count == 1 and space == "drive"
        result = f"generated-{self.next_id}"
        self.next_id += 1
        return _Request(lambda _request: {"ids": [result]})

    def create(self, *, body, media_body=None, **_kwargs):
        def execute(_request):
            if body.get("name") == self.fail_create_name:
                self.fail_create_name = None
                raise _HttpError(self.fail_create_status)
            if self.fail_next_create:
                self.fail_next_create = False
                raise _HttpError(503)
            if body.get("id"):
                file_id = body["id"]
            else:
                file_id = f"created-{self.next_id}"
                self.next_id += 1
            payload = self._contents(media_body) if media_body is not None else b""
            record = {
                "id": file_id, "name": body["name"], "mimeType": body["mimeType"],
                "parents": list(body["parents"]), "owners": [{"emailAddress": self.owner_override or OWNER}],
                "ownedByMe": (True if self.owned_by_me_override is None else self.owned_by_me_override),
                "size": str(len(payload)), "quotaBytesUsed": str(len(payload)),
                "md5Checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                "capabilities": {"canEdit": True, "canDelete": True, "canAddChildren": True},
                "content": payload, "etag": '"v1"', "version": "1",
            }
            self.records[file_id] = record
            return {key: value for key, value in record.items() if key != "content"}
        return _Request(execute)

    def update(self, *, fileId, media_body, **_kwargs):
        self.update_calls += 1
        def execute(request):
            record = self.records[fileId]
            if (not self.ignore_if_match and request.headers.get("If-Match") is not None
                    and request.headers["If-Match"] != record["etag"]):
                raise _HttpError(412)
            payload = self._contents(media_body)
            record.update(content=payload, size=str(len(payload)), quotaBytesUsed=str(len(payload)),
                          md5Checksum=hashlib.md5(payload, usedforsecurity=False).hexdigest())
            version = int(record.get("version", "0")) + 1
            record["version"] = str(version)
            record["etag"] = f'"v{version}"'
            return {key: value for key, value in record.items() if key != "content"}
        return _Request(execute)

    def get(self, *, fileId, alt=None, **_kwargs):
        if fileId not in self.records:
            return _Request(lambda _request: (_ for _ in ()).throw(_HttpError(404)))
        record = self.records[fileId]
        if alt == "media":
            return _Request(lambda _request: record["content"])
        response_headers = {} if self.omit_response_etag else {
            self.response_etag_header_name: self.response_etag_override or record["etag"]
        }

        def metadata_response(_request):
            result = {key: value for key, value in record.items() if key != "content"}
            if self.metadata_parent_override is not None:
                result["parents"] = [self.metadata_parent_override]
            if self.metadata_file_id_override is not None:
                result["id"] = self.metadata_file_id_override
            return result

        return _Request(
            metadata_response,
            response_headers,
        )

    def get_media(self, *, fileId, **_kwargs):
        return self.get(fileId=fileId, alt="media")

    def delete(self, *, fileId, **_kwargs):
        return _Request(lambda _request: self.records.pop(fileId, None) or {})


class _FakeService:
    def __init__(self):
        self.records = {
            "revision": {
                "id": "revision", "name": "ReVision", "mimeType": "application/vnd.google-apps.folder",
                "parents": ["root"], "owners": [{"emailAddress": OWNER}], "ownedByMe": True,
                "capabilities": {"canAddChildren": True}, "etag": '"v1"', "version": "1",
            },
            "results": {
                "id": "results", "name": "results", "mimeType": "application/vnd.google-apps.folder",
                "parents": ["revision"], "owners": [{"emailAddress": OWNER}], "ownedByMe": True,
                "capabilities": {"canAddChildren": True}, "etag": '"v1"', "version": "1",
            },
        }
        self._files = _FakeFiles(self.records)

    def files(self):
        return self._files

    def about(self):
        return self

    def get(self, **_kwargs):
        return _Request(lambda _request: {
            "user": {"emailAddress": OWNER},
            "storageQuota": {"limit": str(1 << 40), "usage": "1000"},
        })


class _FakeV2Files:
    def __init__(self, records):
        self.records = records
        self.delegate = _FakeFiles(records)
        self.update_calls = 0
        self.omit_etag = False
        self.omit_resource_etag = False
        self.etag_override = None
        self.metadata_parent_override = None
        self.metadata_file_id_override = None

    def get(self, *, fileId, **_kwargs):
        record = self.records.get(fileId)
        if record is None:
            return _Request(lambda _request: (_ for _ in ()).throw(_HttpError(404)))
        response_etag = self.etag_override or record.get("etag")
        headers = {} if self.omit_etag or response_etag is None else {"ETag": response_etag}

        def execute(_request):
            result = {
                "id": self.metadata_file_id_override or record["id"], "title": record["name"],
                "mimeType": record["mimeType"],
                "parents": [{"id": parent} for parent in (
                    [self.metadata_parent_override] if self.metadata_parent_override
                    else record.get("parents", [])
                )],
                "owners": record.get("owners", []),
                "ownedByMe": record.get("ownedByMe"),
                "editable": True,
                "capabilities": record.get("capabilities", {}),
                "fileSize": record.get("size"),
                "quotaBytesUsed": record.get("quotaBytesUsed"),
                "version": record.get("version"),
            }
            if not self.omit_etag and not self.omit_resource_etag:
                result["etag"] = response_etag
            return result

        return _Request(execute, headers)

    def update(self, *, fileId, media_body, **kwargs):
        self.update_calls += 1
        return self.delegate.update(fileId=fileId, media_body=media_body, **kwargs)


class _FakeV2Service:
    def __init__(self, storage_service):
        self.records = storage_service.records
        self._files = _FakeV2Files(self.records)
        self.email = OWNER

    def files(self):
        return self._files

    def about(self):
        return self

    def get(self, **_kwargs):
        return _Request(lambda _request: {"user": {"emailAddress": self.email}})


def _storage(service, stage_root, monkeypatch, *, lease_service=None):
    monkeypatch.setattr(drive_storage, "_make_media_upload", lambda path, _mime: Path(path))
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    return DriveApiStorage(
        service=service, expected_account_email=OWNER, revision_folder_id="revision",
        lease_service=lease_service,
        logical_revision_root="/drive/MyDrive/ReVision", local_staging_root=stage_root,
        download_to_path=lambda request, path: Path(path).write_bytes(request.execute()),
        retry_delays=(0,),
    )


def test_existing_notebook_adapter_call_discovers_paired_v2_client_from_verified_v3_service(tmp_path, monkeypatch):
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    service._revision_r1_lease_service = lease_service

    # This matches the stable notebook's existing constructor call: no new
    # parameter, secret, or setup-cell edit is required for the v2 CAS client.
    storage = DriveApiStorage(
        service=service, expected_account_email=OWNER, revision_folder_id="revision",
        logical_revision_root="/drive/MyDrive/ReVision", local_staging_root=tmp_path / "local",
        retry_delays=(0,),
    )

    assert storage.lease_service is lease_service


def test_storage_adapter_rejects_paired_v2_lease_client_for_another_account(tmp_path):
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    lease_service.email = "compute@example.org"
    service._revision_r1_lease_service = lease_service

    with pytest.raises(DriveStorageError, match="Drive lease API identity mismatch"):
        DriveApiStorage(
            service=service, expected_account_email=OWNER, revision_folder_id="revision",
            logical_revision_root="/drive/MyDrive/ReVision", local_staging_root=tmp_path / "local",
            retry_delays=(0,),
        )


def test_api_storage_uses_main_account_independent_of_mount_and_roundtrips_by_exact_id(tmp_path, monkeypatch):
    service = _FakeService()
    storage = _storage(service, tmp_path / "local", monkeypatch)
    storage.bind_root("/drive/MyDrive/ReVision/results", tmp_path / "local/results")
    local = tmp_path / "local/results/progress.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"cursor":3}\n')

    evidence = storage.publish_file(local, "/drive/MyDrive/ReVision/results/progress.json")
    restored = tmp_path / "restored/progress.json"
    readback = storage.download_file("/drive/MyDrive/ReVision/results/progress.json", restored)

    assert evidence["status"] == "PASS"
    assert evidence["new_file"] is True
    assert readback["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert restored.read_bytes() == local.read_bytes()
    assert next(row for row in service.records.values() if row["name"] == "progress.json")["owners"][0]["emailAddress"] == OWNER
    # The independent compute/mount identity is deliberately not consulted.
    compute_mount_account = "kiananbarestani62@gmail.com"
    assert compute_mount_account != storage.expected_account_email


def _seed_preflight_drive(service, *, schedule_bytes=None):
    if schedule_bytes is None:
        schedule_bytes = (
            "dataset,seed,task\n"
            + "".join(f"ember,42,{task}\n" for task in range(1, 221))
        ).encode("utf-8")
    folders = [
        ("revision-r1", "revision_R1", "revision"),
        ("results", "results", "revision-r1"),
        ("preflight", "preflight", "results"),
        ("schedules", "schedules", "preflight"),
    ]
    for file_id, name, parent in folders:
        service.records[file_id] = {
            "id": file_id, "name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent], "owners": [{"emailAddress": OWNER}], "ownedByMe": True,
            "capabilities": {"canAddChildren": True, "canDelete": True}, "etag": '"v1"', "version": "1",
        }

    schedule_name = "effective_update_budget_test.csv"
    manifest_name = "effective_update_budget_test.manifest.json"
    schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
    historical_schedule = (
        "/content/drive/.shortcut-targets-by-id/old/ReVision/revision_R1/results/preflight/"
        f"schedules/{schedule_name}"
    )
    historical_manifest = (
        "/content/drive/MyDrive/ReVision/revision_R1/results/preflight/"
        f"schedules/{manifest_name}"
    )
    preflight_payload = json.dumps({
        "status": "PREFLIGHT_VALIDATED_NO_TRAINING",
        "training_reproduction_claimed": False,
        "schedule": {
            "path": historical_schedule, "manifest_path": historical_manifest,
            "sha256": schedule_sha, "row_count": 220,
        },
    }, sort_keys=True).encode("utf-8")
    manifest_payload = json.dumps({
        "source_csv_path": historical_schedule, "source_csv_sha256": schedule_sha,
    }, sort_keys=True).encode("utf-8")

    def add_file(file_id, name, parent, payload, mime):
        service.records[file_id] = {
            "id": file_id, "name": name, "mimeType": mime, "parents": [parent],
            "owners": [{"emailAddress": OWNER}], "ownedByMe": True,
            "size": str(len(payload)), "quotaBytesUsed": str(len(payload)),
            "md5Checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "capabilities": {"canEdit": True, "canDelete": True},
            "content": payload, "etag": '"v1"', "version": "1",
        }

    add_file("preflight-json", "preflight.json", "preflight", preflight_payload, "application/json")
    add_file("effective-csv", schedule_name, "schedules", schedule_bytes, "text/csv")
    add_file("schedule-manifest", manifest_name, "schedules", manifest_payload, "application/json")
    return {
        "schedule_sha": schedule_sha,
        "historical_schedule": historical_schedule,
        "historical_manifest": historical_manifest,
        "schedule_name": schedule_name,
        "manifest_name": manifest_name,
        "preflight_payload": preflight_payload,
        "schedule_payload": schedule_bytes,
        "manifest_payload": manifest_payload,
    }


def test_cell_b_stages_complete_preflight_bundle_through_api_without_drive_mount(tmp_path, monkeypatch, capsys):
    from scripts.build_github_handoff import notebook_document
    from scripts import production_workflow
    from revision_r1.config import DATASETS
    import io
    if not hasattr(np.lib.format, "_read_array_header"):
        # Local NumPy 2.5 moved this unchanged parser; Colab pins NumPy 2.0.2.
        from numpy.lib import _format_impl
        monkeypatch.setattr(np.lib.format, "_read_array_header", _format_impl._read_array_header, raising=False)

    service = _FakeService()
    fixture = _seed_preflight_drive(service)
    original_specs = {dataset: {split: DATASETS[dataset][split] for split in ("train", "test")}
                      for dataset in DATASETS}
    def seed_file(file_id, name, parent, payload, mime="application/json"):
        service.records[file_id] = {
            "id": file_id, "name": name, "mimeType": mime, "parents": [parent],
            "owners": [{"emailAddress": OWNER}], "ownedByMe": True,
            "size": str(len(payload)), "quotaBytesUsed": str(len(payload)),
            "md5Checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "content": payload, "version": "1", "etag": '"v1"',
            "capabilities": {"canEdit": True, "canDelete": True, "canAddChildren": True},
        }
    seed_file("dataset-parent", "AHR_MalCL", "root", b"", "application/vnd.google-apps.folder")
    seed_file("dataset-folder", "datasets", "dataset-parent", b"", "application/vnd.google-apps.folder")
    preflight = json.loads(fixture["preflight_payload"])
    for dataset in DATASETS:
        preflight[dataset] = {}
        for split in ("train", "test"):
            stream = io.BytesIO()
            np.savez_compressed(stream, **{f"X_{split}": np.ones((100, 3), dtype=np.float32),
                                         f"y_{split}": np.arange(100)})
            payload = stream.getvalue()
            spec = dict(DATASETS[dataset][split], shape=[100, 3],
                        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest())
            monkeypatch.setitem(DATASETS[dataset], split, spec)
            preflight[dataset][split] = {"md5": spec["md5"], "sha256": hashlib.sha256(payload).hexdigest()}
            seed_file(f"{dataset}-{split}", spec["filename"], "dataset-folder", payload, "application/zip")
    fixture["preflight_payload"] = json.dumps(preflight, sort_keys=True).encode()
    seed_file("preflight-json", "preflight.json", "preflight", fixture["preflight_payload"])
    seed_file("reference", "reference.json", "results", b'{"fixture":"recovery"}')
    downloads = []
    def download(request, path):
        payload = request.execute()
        downloads.append(payload)
        Path(path).write_bytes(payload)
    stage_root = tmp_path / "storage" / "preflight"
    storage = DriveApiStorage(
        service=service, expected_account_email=OWNER, revision_folder_id="revision",
        logical_revision_root="/content/drive/MyDrive/ReVision", local_staging_root=tmp_path / "storage",
        download_to_path=download, retry_delays=(0,),
    )
    logical_preflight = "/content/drive/MyDrive/ReVision/revision_R1/results/preflight/preflight.json"
    storage.bind_root(Path(logical_preflight).parent, stage_root)
    source = "".join(notebook_document("0123456789abcdef0123456789abcdef01234567")["cells"][2]["source"])
    namespace = {
        "DRIVE_PATHS": {
            "ember_data_root": "/content/drive/MyDrive/AHR_MalCL/datasets",
            "az_data_root": "/content/drive/MyDrive/AHR_MalCL/datasets",
            # Older Cell A used preflight_path; the replacement B must supply
            # the canonical logical defaults without depending on a mount.
            "preflight_path": logical_preflight,
            "v62_recovery_report": "/content/drive/MyDrive/ReVision/revision_R1/results/reference.json",
            "dataset_cache_root": str(tmp_path / "cache"),
        },
        "STORAGE_ADAPTER": storage,
        "EXPECTED_SCHEDULE_SHA256": fixture["schedule_sha"],
        "PROJECT": Path(__file__).resolve().parents[1],
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
    }

    exec(compile(source, "generated-cell-B", "exec"), namespace)
    capsys.readouterr()

    evidence = namespace["PREFLIGHT_STAGING_EVIDENCE"]
    assert evidence["schedule_sha256"] == fixture["schedule_sha"]
    assert evidence["drive_files"]["preflight"]["file_id"] == "preflight-json"
    assert evidence["drive_files"]["schedule"]["file_id"] == "effective-csv"
    assert evidence["drive_files"]["manifest"]["file_id"] == "schedule-manifest"
    assert Path(namespace["PREFLIGHT_PATH"]).read_bytes() == fixture["preflight_payload"]
    assert namespace["EFFECTIVE_SCHEDULE_PATH"].read_bytes() == fixture["schedule_payload"]
    assert namespace["EFFECTIVE_SCHEDULE_MANIFEST_PATH"].read_bytes() == fixture["manifest_payload"]
    assert Path(namespace["EFFECTIVE_SCHEDULE_PATH"]).is_relative_to(stage_root)
    assert all(Path(item["local_path"]).is_file() for item in evidence["drive_files"].values())
    assert fixture["historical_schedule"] not in str(namespace["EFFECTIVE_SCHEDULE_PATH"])
    dataset_downloads = len(downloads)
    assert all(Path(root).is_relative_to(tmp_path) for root in namespace["DATA_ROOTS"].values())
    assert Path(namespace["DRIVE_PATHS"]["v62_recovery_report"]).is_file()
    remote_snapshot = {
        key: (record.get("size"), record.get("md5Checksum"), record.get("content"))
        for key, record in service.records.items()
    }
    first_bundle_key = evidence["bundle_key"]
    exec(compile(source, "generated-cell-B-rerun", "exec"), namespace)
    capsys.readouterr()
    assert namespace["PREFLIGHT_STAGING_EVIDENCE"]["bundle_key"] == first_bundle_key
    # Small preflight records may redownload; NPZs and the operational report
    # must reuse already hash-verified local bytes.
    assert len(downloads) == dataset_downloads + 4
    assert sum(payload.startswith(b"PK") for payload in downloads) == 4
    assert {
        key: (record.get("size"), record.get("md5Checksum"), record.get("content"))
        for key, record in service.records.items()
    } == remote_snapshot
    from revision_r1 import data as data_module
    for dataset, root in namespace["DATA_ROOTS"].items():
        for split in ("train", "test"):
            archive = root / DATASETS[dataset][split]["filename"]
            x, y, _metadata = data_module.load_npz_arrays(
                archive, split=split, expected_md5=DATASETS[dataset][split]["md5"],
                expected_shape=(100, 3), expected_labels=range(100),
                cache_dir=namespace["LOCAL_DATASET_CACHE_ROOT"],
            )
            assert isinstance(x, np.memmap) and isinstance(y, np.memmap)
            del x, y
            archive.unlink()  # Only this test's synthetic local NPZ fixture.
    def forbid_decompression(*_args, **_kwargs):
        raise AssertionError("verified cache must not be decompressed a second time")
    monkeypatch.setattr(data_module, "_populate_streamed_cache", forbid_decompression)
    exec(compile(source, "generated-cell-B-cache-only", "exec"), namespace)
    assert sum(payload.startswith(b"PK") for payload in downloads) == 4
    assert all(row["source"] == "VERIFIED_LOCAL_MMAP_CACHE"
               for row in namespace["DATASET_API_STAGING_EVIDENCE"].values())
    # Test the real generated C with external readiness evidence mocked. The
    # schedule/data downloads and filesystem serialization above remain real.
    for dataset, splits in original_specs.items():
        for split, spec in splits.items():
            monkeypatch.setitem(DATASETS[dataset], split, spec)
    plan = json.loads((namespace["PROJECT"] / "configs/production_run_plan_v1.json").read_text())
    for row in plan["runs"]:
        row["dataset_identity"] = {}
    first = plan["runs"][0]
    measurement = {"max_checkpoint_payload_bytes": 100, "run_output_bytes": 100}
    monkeypatch.setattr(production_workflow, "bind_runtime_schedule", lambda *_args: plan)
    monkeypatch.setattr(production_workflow, "validate_runtime_bound_plan", lambda *_args: None)
    monkeypatch.setattr(production_workflow, "_hydrate_api_matrix_registry", lambda *_args: None)
    monkeypatch.setattr(production_workflow, "_hydrate_api_run_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(production_workflow, "matrix_registry_evidence", lambda *_args: {"completed_run_keys": [first["run_key"]]})
    monkeypatch.setattr(production_workflow, "reuse_validated_first_run_measurement", lambda *_args: measurement)
    monkeypatch.setattr(production_workflow, "_read_matrix_registry", lambda *_args: {"completed_runs": {first["run_key"]: {}}})
    def reference_reader(path):
        assert Path(path).is_relative_to(tmp_path)
        assert json.loads(Path(path).read_bytes()) == {"fixture": "recovery"}
        return {"max_checkpoint_payload_bytes": 100}
    monkeypatch.setattr(production_workflow, "load_v62_recovery_checkpoint_bound", reference_reader)
    monkeypatch.setattr(production_workflow, "_preflight_addressed_cache_bytes", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(production_workflow, "require_run_capacity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(production_workflow, "project_storage", lambda *_args, **_kwargs: {"drive": {
        "matrix_peak_with_reserve_bytes": 100, "next_run_peak_with_reserve_bytes": 100,
        "next_run_required_without_reserve_bytes": 100,
    }})
    monkeypatch.setattr(production_workflow, "inspect_production_capacity", lambda *_args, **_kwargs: {
        "canonical_storage_identity_verified": True, "drive_available_bytes": 1 << 40,
        "quota_account_email": OWNER, "canonical_revision_folder_id": "revision",
        "quota_status": "TEST_FIXTURE", "output_destination": {"destination_owner": OWNER, "new_file_owner": OWNER},
        "checkpoint_destination": {"destination_owner": OWNER, "new_file_owner": OWNER},
    })
    monkeypatch.setattr(drive_storage, "_make_media_upload", lambda path, _mime: Path(path))
    monkeypatch.setattr(storage, "media_upload_factory", lambda path, _mime: Path(path))
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    for key in ("output_root", "checkpoint_root", "export_root", "checkpoint_backup_root"):
        logical = Path("/content/drive/MyDrive/ReVision/revision_R1/production") / key
        namespace["DRIVE_PATHS"][key] = str(logical)
        local_name = {"output_root": "results", "checkpoint_root": "checkpoints",
                      "export_root": "exports", "checkpoint_backup_root": "checkpoint_backups"}[key]
        storage.bind_root(logical, tmp_path / "storage/production" / local_name)
    namespace["DRIVE_PATHS"]["local_checkpoint_root"] = str(tmp_path / "local-checkpoints")
    namespace.update(STORAGE_API_FACTORY=lambda: service, ACTUAL_GIT_COMMIT="a" * 40,
                     ACTUAL_TRAINING_SOURCE_SHA256=plan["training_source_sha256"],
                     ACTUAL_RUNTIME_SOURCE_SHA256="b" * 64, PACKAGE_MANIFEST_SHA256="c" * 64)
    c_source = "".join(notebook_document("a" * 40)["cells"][3]["source"])
    exec(compile(c_source, "generated-cell-C", "exec"), namespace)
    assert namespace["NEXT_PENDING_RECORD"]["run_key"] == "ember_seed42_current_only"
    assert namespace["DATA_ROOTS"]["ember"].is_relative_to(tmp_path)


def test_preflight_staging_rejects_wrong_schedule_hash_missing_or_ambiguous_canonical_file(tmp_path, monkeypatch):
    from scripts.build_github_handoff import stage_canonical_preflight_artifacts

    def make_adapter(service, root):
        return DriveApiStorage(
            service=service, expected_account_email=OWNER, revision_folder_id="revision",
            logical_revision_root="/content/drive/MyDrive/ReVision", local_staging_root=root / "storage",
            download_to_path=lambda request, path: Path(path).write_bytes(request.execute()), retry_delays=(0,),
        )

    logical_preflight = "/content/drive/MyDrive/ReVision/revision_R1/results/preflight/preflight.json"
    service = _FakeService()
    fixture = _seed_preflight_drive(service)
    adapter = make_adapter(service, tmp_path)
    with pytest.raises(ValueError, match="does not authenticate"):
        stage_canonical_preflight_artifacts(
            adapter, logical_preflight, tmp_path / "storage" / "wrong-hash", "0" * 64,
        )

    corrupted_service = _FakeService()
    corrupted_fixture = _seed_preflight_drive(corrupted_service)
    corrupted_record = corrupted_service.records["effective-csv"]
    corrupted_record["content"] = b"X" * len(corrupted_record["content"])
    with pytest.raises(DriveStorageError, match="SHA-256 mismatch"):
        stage_canonical_preflight_artifacts(
            make_adapter(corrupted_service, tmp_path / "corrupted"), logical_preflight,
            tmp_path / "corrupted" / "storage" / "preflight", corrupted_fixture["schedule_sha"],
        )

    missing_service = _FakeService()
    missing_fixture = _seed_preflight_drive(missing_service)
    missing_service.records.pop("effective-csv")
    with pytest.raises(FileNotFoundError, match="effective_update_budget_test.csv"):
        stage_canonical_preflight_artifacts(
            make_adapter(missing_service, tmp_path / "missing"), logical_preflight,
            tmp_path / "missing" / "storage" / "preflight", missing_fixture["schedule_sha"],
        )

    ambiguous_service = _FakeService()
    ambiguous_fixture = _seed_preflight_drive(ambiguous_service)
    duplicate = dict(ambiguous_service.records["effective-csv"], id="duplicate-effective-csv")
    ambiguous_service.records["duplicate-effective-csv"] = duplicate
    with pytest.raises(AmbiguousDrivePathError, match="multiple Drive objects"):
        stage_canonical_preflight_artifacts(
            make_adapter(ambiguous_service, tmp_path / "ambiguous"), logical_preflight,
            tmp_path / "ambiguous" / "storage" / "preflight", ambiguous_fixture["schedule_sha"],
        )


def test_canonical_write_probe_uses_the_same_adapter_create_download_and_exact_delete_path(tmp_path, monkeypatch):
    from scripts import production_workflow

    service = _FakeService()
    storage = _storage(service, tmp_path / "local", monkeypatch)
    location = {"path_exists": True, "folder_id": "revision"}
    evidence = production_workflow._write_capability_probe(
        "/drive/MyDrive/ReVision", location, OWNER, service,
        drive_root="/drive/MyDrive", storage_adapter=storage,
    )

    assert evidence["status"] == "PASS"
    assert evidence["write_path"] == "DriveApiStorage.publish_file_download_file_delete_exact"
    assert evidence["probe_parent_id"] == "revision"
    assert evidence["new_file_owner"] == OWNER
    assert evidence["new_file_owned_by_authenticated_user"] is True
    assert evidence["permanent_recovery_delete_verified"] is True
    assert evidence["diagnostics"]["classification"] == "VERIFIED"
    assert not any(row["name"].startswith(".revision-r1-capacity-probe-") for row in service.records.values())


def test_storage_session_verifier_uses_the_bound_adapter_probe(tmp_path, monkeypatch):
    from scripts import production_workflow
    from revision_r1 import storage

    service = _FakeService()
    lease_service = _FakeV2Service(service)
    adapter = _storage(service, tmp_path / "stage", monkeypatch, lease_service=lease_service)
    captured_cas_services = []
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_ADAPTER", None)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    storage.clear_verified_drive_api_service()
    evidence = production_workflow.verify_canonical_storage_session(
        api_factory=lambda: service, drive_mount_root="/drive/MyDrive",
        expected_account_email=OWNER, expected_revision_folder_id="revision",
        storage_adapter=adapter,
        conditional_update_probe=lambda _service, **kwargs: (
            captured_cas_services.append(kwargs.get("lease_service"))
            or {"status": "PASS", "test_only": True}
        ),
    )

    assert evidence["durable_storage_backend"] == "explicit_main_account_drive_api_with_local_staging"
    assert evidence["probe"]["write_path"] == "DriveApiStorage.publish_file_download_file_delete_exact"
    assert captured_cas_services == [lease_service]
    assert production_workflow._BOUND_STORAGE_ADAPTER is adapter
    assert storage.verified_drive_api_service(expected_account_email=OWNER) is service


def test_adapter_probe_reports_stale_name_visibility_and_cleans_only_its_verified_probe(tmp_path, monkeypatch):
    from scripts import production_workflow
    from revision_r1.storage import StorageCapacityError

    service = _FakeService()
    service.files().hide_probe_names_from_list = True
    storage = _storage(service, tmp_path / "local", monkeypatch)
    with pytest.raises(StorageCapacityError, match="canonical API adapter write-capability probe failed") as caught:
        production_workflow._write_capability_probe(
            "/drive/MyDrive/ReVision", {"path_exists": True, "folder_id": "revision"},
            OWNER, service, drive_root="/drive/MyDrive", storage_adapter=storage,
        )
    diagnostics_text = str(caught.value).split("diagnostics=", 1)[1]
    diagnostics = json.loads(diagnostics_text)
    assert diagnostics["classification"] == "MISSING_OR_STALE_METADATA"
    assert diagnostics["expected"]["file_id"].startswith("generated-")
    assert diagnostics["conditions"]["api_visible"]["status"] == "MISMATCH"
    assert not any(row["name"].startswith(".revision-r1-capacity-probe-") for row in service.records.values())


def test_adapter_probe_reports_confirmed_owner_mismatch_and_removes_only_the_exact_probe(tmp_path, monkeypatch):
    from scripts import production_workflow
    from revision_r1.storage import StorageCapacityError

    service = _FakeService()
    service.files().owner_override = "compute@example.org"
    service.files().owned_by_me_override = False
    storage = _storage(service, tmp_path / "local", monkeypatch)
    with pytest.raises(StorageCapacityError, match="canonical API adapter write-capability probe failed") as caught:
        production_workflow._write_capability_probe(
            "/drive/MyDrive/ReVision", {"path_exists": True, "folder_id": "revision"},
            OWNER, service, drive_root="/drive/MyDrive", storage_adapter=storage,
        )

    diagnostics = json.loads(str(caught.value).split("diagnostics=", 1)[1])
    assert diagnostics["classification"] == "CONFIRMED_OWNERSHIP_MISMATCH"
    assert diagnostics["expected"]["owner"] == OWNER
    assert diagnostics["observed"]["owner"] == "compute@example.org"
    assert diagnostics["conditions"]["ownedByMe"]["status"] == "MISMATCH"
    assert not any(row["name"].startswith(".revision-r1-capacity-probe-") for row in service.records.values())


def test_api_tree_download_visits_each_exact_drive_child_once(tmp_path, monkeypatch):
    service = _FakeService()
    storage = _storage(service, tmp_path / "local", monkeypatch)
    local = tmp_path / "source.json"
    local.write_text('{"only_once":true}\n')
    storage.publish_file(local, "/drive/MyDrive/ReVision/results/one.json")

    rows = storage.download_tree(
        "/drive/MyDrive/ReVision/results", tmp_path / "restored/results",
    )

    assert [row["path"] for row in rows] == [str(tmp_path / "restored/results/one.json")]
    assert (tmp_path / "restored/results/one.json").read_bytes() == local.read_bytes()


def test_api_nested_control_file_resolves_from_a_fresh_storage_adapter(tmp_path, monkeypatch):
    service = _FakeService()
    logical_root = "/drive/MyDrive/ReVision/validation/SYNTHETIC_NONREPORTABLE_TEST_ONLY"
    first = _storage(service, tmp_path / "first", monkeypatch)
    first.bind_root(logical_root, tmp_path / "first/output")
    local = tmp_path / "first/output/registry/plan-hash/matrix_completion_registry.json"
    local.parent.mkdir(parents=True)
    local.write_text('{"completed_runs":{}}\n')
    published = first.publish_file(
        local, f"{logical_root}/registry/plan-hash/matrix_completion_registry.json",
    )

    fresh = _storage(service, tmp_path / "fresh", monkeypatch)
    fresh.bind_root(logical_root, tmp_path / "fresh/output")
    observed = fresh.exact_file(
        f"{logical_root}/registry/plan-hash/matrix_completion_registry.json",
    )

    assert observed is not None
    assert observed["id"] == published["file_id"]


def test_api_storage_rejects_duplicate_names_and_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    service = _FakeService()
    first = {"id": "dup-a", "name": "same.json", "mimeType": "application/json", "parents": ["results"]}
    second = {**first, "id": "dup-b"}
    service.records.update({"dup-a": first, "dup-b": second})
    storage = _storage(service, tmp_path / "local", monkeypatch)
    with pytest.raises(AmbiguousDrivePathError):
        storage.exact_file("/drive/MyDrive/ReVision/results/same.json")

    # A transient create failure is retried against the same generated ID;
    # it must not produce duplicate names.
    service.records.pop("dup-a")
    service.records.pop("dup-b")
    local = tmp_path / "source.json"
    local.write_text('{"x":1}\n')
    service.files().fail_next_create = True
    result = storage.publish_file(local, "/drive/MyDrive/ReVision/results/retry.json")
    matching = [row for row in service.records.values() if row["name"] == "retry.json"]
    assert result["status"] == "PASS"
    assert len(matching) == 1


def test_drive_api_lease_enforces_active_owner_heartbeat_release_and_stale_etag(tmp_path, monkeypatch):
    from scripts.production_workflow import inspect_lease_readonly

    service = _FakeService()
    lease_service = _FakeV2Service(service)
    storage = _storage(service, tmp_path / "local", monkeypatch, lease_service=lease_service)
    local_lock = tmp_path / "local/lease.json"
    local_lock.parent.mkdir(parents=True, exist_ok=True)
    local_lock.write_text(json.dumps({"schema_version": 1, "active": False, "expires_at": 0}) + "\n")
    logical_path = "/drive/MyDrive/ReVision/results/matrix.lease.json"
    storage.publish_file(local_lock, logical_path)

    def inspect_drive_record():
        observed_path = tmp_path / "observed-lease.json"
        observed_path.unlink(missing_ok=True)
        storage.download_file(logical_path, observed_path)
        return inspect_lease_readonly(
            observed_path, now_unix=time.time(), current_host="diagnostic-runtime", current_pid=1,
        )

    assert inspect_drive_record()["status"] == "inactive"

    lease1 = DriveApiLease(storage, logical_path, owner_token="worker-1")
    lease2 = DriveApiLease(storage, logical_path, owner_token="worker-2")
    lease1.acquire()
    assert inspect_drive_record()["status"] == "live"
    with pytest.raises(DriveLeaseError, match="held"):
        lease2.acquire()
    lease1.heartbeat()
    assert inspect_drive_record()["status"] == "live"
    lease1.release()
    released = inspect_drive_record()
    assert released["status"] == "inactive"
    assert released["owner_process_visibility"] == "released_owner_not_active"
    lease2.acquire()
    lease2.release()

    stale_owner = DriveApiLease(storage, logical_path, ttl_seconds=0.02, owner_token="stale-owner")
    replacement_owner = DriveApiLease(storage, logical_path, owner_token="replacement-owner")
    stale_owner.acquire()
    time.sleep(0.04)
    assert inspect_drive_record()["status"] == "expired"
    replacement_owner.acquire()
    with pytest.raises(DriveLeaseError, match="owner/token or ETag changed"):
        stale_owner.heartbeat()
    assert json.loads(service.records[stale_owner._file_id]["content"])["owner_token"] == "replacement-owner"
    replacement_owner.release()
    assert inspect_drive_record()["status"] == "inactive"

    assert lease_service.files().update_calls >= 5
    assert service.files().update_calls == 0, "lease CAS updates must not use the Drive v3 data client"

    cas = verify_conditional_update_support(
        service, parent_id="results", expected_account_email=OWNER, lease_service=lease_service,
    )
    assert cas["status"] == "PASS"
    assert cas["contract"] == "stale_if_match_rejected_http_412"
    assert not any("cas-probe" in row["name"] for row in service.records.values())


@pytest.mark.parametrize("document", [
    {"schema_version": 1, "active": False, "expires_at": 10},
    {"schema_version": 1, "active": False, "expires_at": -1},
    {"schema_version": 1, "active": False, "expires_at": 10 ** 400},
    {"schema_version": 1, "active": False, "expires_at": 0, "owner_token": "partial"},
    {"schema_version": 2, "active": False, "expires_at": 0},
    {"schema_version": 1, "active": True, "expires_at": 2_000_000_000},
])
def test_drive_api_lease_acquisition_rejects_corrupted_or_ambiguous_records(
    tmp_path, monkeypatch, document,
):
    from scripts.production_workflow import inspect_lease_readonly

    service = _FakeService()
    lease_service = _FakeV2Service(service)
    storage = _storage(service, tmp_path / "local", monkeypatch, lease_service=lease_service)
    local_lock = tmp_path / "local/lease.json"
    local_lock.parent.mkdir(parents=True, exist_ok=True)
    local_lock.write_text(json.dumps(document) + "\n", encoding="utf-8")
    storage.publish_file(local_lock, "/drive/MyDrive/ReVision/results/matrix.lease.json")

    diagnostic = inspect_lease_readonly(local_lock)
    assert diagnostic["status"] == "malformed"
    assert diagnostic["lease_acquisition_allowed"] is False

    lease = DriveApiLease(storage, "/drive/MyDrive/ReVision/results/matrix.lease.json", owner_token="new-worker")
    with pytest.raises(DriveLeaseError, match="malformed.*preserved"):
        lease.acquire()

    assert lease_service.files().update_calls == 0
    stored = json.loads(next(row["content"] for row in service.records.values() if row["name"] == "matrix.lease.json"))
    assert stored == document


def test_conditional_update_probe_uses_drive_v2_etag_from_response_callback(monkeypatch):
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    lease_service.files().omit_resource_etag = True

    result = verify_conditional_update_support(
        service, parent_id="results", expected_account_email=OWNER, lease_service=lease_service,
    )

    assert result["status"] == "PASS"
    assert result["contract"] == "stale_if_match_rejected_http_412"
    assert result["etag_source"] == "drive_v2_response_callback"
    assert result["api_version"] == "v2_lease_cas_only"
    assert result["probe_bytes_preserved_after_stale_update"] is True
    assert lease_service.files().update_calls == 2
    assert not any("cas-probe" in row["name"] for row in service.records.values())


def test_conditional_update_probe_rejects_drive_v2_ignoring_stale_if_match_and_cleans_probe(monkeypatch):
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    lease_service.files().delegate.ignore_if_match = True

    with pytest.raises(DriveLeaseError, match="ignored stale If-Match"):
        verify_conditional_update_support(
            service, parent_id="results", expected_account_email=OWNER,
            lease_service=lease_service,
        )

    assert not any("cas-probe" in row["name"] for row in service.records.values())


def test_google_api_http_request_response_headers_use_callback_not_request_resp_attribute():
    from googleapiclient.http import HttpRequest
    from httplib2 import Response

    class _Http:
        def request(self, *_args, **_kwargs):
            return Response({"status": "200", "ETag": '"real-client-v1"'}), b'{"visible":true}'

    request = HttpRequest(
        http=_Http(), postproc=lambda _response, content: json.loads(content),
        uri="https://example.invalid/drive-probe", method="GET",
    )
    assert not hasattr(request, "resp")

    payload, response = drive_storage._execute_with_response_headers(request)

    assert payload == {"visible": True}
    assert response["etag"] == '"real-client-v1"'


@pytest.mark.parametrize("etag", [None, 'W/"weak-v1"'])
def test_conditional_update_probe_rejects_missing_or_weak_etag_with_field_diagnostics(etag, monkeypatch):
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    lease_service.files().omit_etag = etag is None
    if etag is not None:
        lease_service.files().etag_override = etag

    with pytest.raises(DriveLeaseError, match="strong-ETag verification failed") as raised:
        verify_conditional_update_support(
            service, parent_id="results", expected_account_email=OWNER, lease_service=lease_service,
        )

    assert raised.value.diagnostics["checks"]["api_visible"] is True
    assert raised.value.diagnostics["checks"]["parent_matches"] is True
    assert raised.value.diagnostics["checks"]["owner_matches"] is True
    assert raised.value.diagnostics["checks"]["strong_etag_present"] is False
    assert raised.value.diagnostics["failure_class"] == "probe_strong_etag_unavailable_or_weak"
    assert "size" in raised.value.diagnostics["observed"]
    assert "quotaBytesUsed" in raised.value.diagnostics["observed"]
    assert not service.records.get("generated-1")


@pytest.mark.parametrize(
    ("fault", "failure_class", "failed_check"),
    [
        ("owner", "probe_identity_or_parent_mismatch", "owner_matches"),
        ("ownedByMe", "probe_identity_or_parent_mismatch", "owned_by_authenticated_user"),
        ("parent", "probe_identity_or_parent_mismatch", "parent_matches"),
        ("file_id", "probe_metadata_missing_or_stale", "file_id_matches"),
    ],
)
def test_conditional_update_probe_reports_identity_metadata_failures(fault, failure_class, failed_check, monkeypatch):
    monkeypatch.setattr(drive_storage, "_make_media_bytes_upload", lambda payload, _mime: bytes(payload))
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    if fault == "owner":
        service.files().owner_override = "compute@example.org"
    elif fault == "ownedByMe":
        service.files().owned_by_me_override = False
    elif fault == "parent":
        lease_service.files().metadata_parent_override = "not-the-expected-parent"
    elif fault == "file_id":
        lease_service.files().metadata_file_id_override = "not-the-probe-id"

    with pytest.raises(DriveLeaseError) as raised:
        verify_conditional_update_support(
            service, parent_id="results", expected_account_email=OWNER,
            lease_service=lease_service,
        )

    diagnostics = raised.value.diagnostics
    assert diagnostics["failure_class"] == failure_class
    assert diagnostics["checks"][failed_check] is False
    assert diagnostics["observed"]["api_visible"] is True
    assert diagnostics["observed"]["size"] is not None
    assert diagnostics["observed"]["quotaBytesUsed"] is not None


def test_api_storage_rejects_wrong_account_before_any_durable_write(tmp_path, monkeypatch):
    service = _FakeService()
    storage = _storage(service, tmp_path / "local", monkeypatch)
    storage.expected_account_email = "other@example.org"
    with pytest.raises(DriveStorageError, match="identity mismatch"):
        storage._verify_account()
    assert set(service.records) == {"revision", "results"}


def test_api_storage_updates_only_explicitly_mutable_shared_editor_control_in_place(tmp_path, monkeypatch):
    service = _FakeService()
    old_content = b"old owner data\n"
    service.records["foreign-progress"] = {
        "id": "foreign-progress", "name": "progress.json",
        "mimeType": "application/octet-stream", "parents": ["results"],
        "owners": [{"emailAddress": "compute@example.org"}], "ownedByMe": False,
        "size": str(len(old_content)), "quotaBytesUsed": str(len(old_content)),
        "md5Checksum": hashlib.md5(old_content, usedforsecurity=False).hexdigest(),
        "capabilities": {"canEdit": True, "canDelete": True}, "content": old_content,
        "etag": '"v1"', "version": "1",
    }
    storage = _storage(service, tmp_path / "local", monkeypatch)
    progress_parent = storage.folder_id("/drive/MyDrive/ReVision/results/progress", create=True)
    service.records["foreign-progress"]["parents"] = [progress_parent]
    local = tmp_path / "local/progress.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"new canonical progress\n")

    with pytest.raises(DriveStorageError, match="foreign-owned immutable.*preserved"):
        storage.publish_file(local, "/drive/MyDrive/ReVision/results/progress/progress.json")

    assert service.records["foreign-progress"]["content"] == old_content
    assert service.records["foreign-progress"]["version"] == "1"

    result = storage.publish_file(
        local, "/drive/MyDrive/ReVision/results/progress/progress.json",
        allow_foreign_owner_update=True,
    )
    updated = service.records["foreign-progress"]
    assert result["file_id"] == "foreign-progress"
    assert result["quota_attribution"] == "existing_foreign_owner_quota_unverified"
    assert updated["content"] == local.read_bytes()
    assert updated["owners"][0]["emailAddress"] == "compute@example.org"
    assert updated["ownedByMe"] is False


def test_api_storage_allows_canonical_editor_to_cas_existing_foreign_lease_but_rejects_viewer(tmp_path, monkeypatch):
    service = _FakeService()
    service.records["foreign-lease"] = {
        "id": "foreign-lease", "name": "matrix.lease.json",
        "mimeType": "application/octet-stream", "parents": ["results"],
        "owners": [{"emailAddress": "compute@example.org"}], "ownedByMe": False,
        "size": "78", "quotaBytesUsed": "78",
        "content": b'{"schema_version":1,"active":false,"expires_at":0}',
        "capabilities": {"canEdit": True, "canDelete": True},
        "etag": '"v1"', "version": "1",
    }
    storage = _storage(service, tmp_path / "local", monkeypatch)

    evidence = storage.ensure_lease_record("/drive/MyDrive/ReVision/results/matrix.lease.json")
    assert evidence["status"] == "EXISTING_SHARED_EDITOR_EXACT_ID"
    assert evidence["file_id"] == "foreign-lease"
    assert evidence["quota_attribution"] == "existing_owner_quota_not_inferred"
    lease = DriveApiLease(
        storage, "/drive/MyDrive/ReVision/results/matrix.lease.json", owner_token="canonical-worker",
    )
    lease.acquire()
    lease.release()
    assert service.records["foreign-lease"]["owners"][0]["emailAddress"] == "compute@example.org"
    assert json.loads(service.records["foreign-lease"]["content"])["active"] is False

    service.records["foreign-lease"]["capabilities"]["canEdit"] = False
    with pytest.raises(DriveLeaseError, match="not editable.*no worker"):
        storage.ensure_lease_record("/drive/MyDrive/ReVision/results/matrix.lease.json")


def _synthetic_plan_and_bundle():
    from revision_r1.config import DATASETS
    from revision_r1.data import DatasetBundle
    from scripts.production_workflow import EXPECTED_EFFECTIVE_SCHEDULE_SHA256, build_frozen_run_plan

    references = {
        name: {
            "train_md5": DATASETS[name]["train"]["md5"],
            "test_md5": DATASETS[name]["test"]["md5"],
            "feature_dim": DATASETS[name]["feature_dim"],
            "train_shape": DATASETS[name]["train"]["shape"],
            "test_shape": DATASETS[name]["test"]["shape"],
        }
        for name in ("ember", "az_class")
    }
    package_root = Path(__file__).resolve().parents[1]
    plan = build_frozen_run_plan(
        package_root, schedule_sha256=EXPECTED_EFFECTIVE_SCHEDULE_SHA256,
        dataset_references=references,
    )
    rng = np.random.default_rng(77)
    x_train = np.vstack([
        rng.normal(class_id / 20, .2, size=(2, 4)) for class_id in range(100)
    ]).astype(np.float32)
    y_train = np.repeat(np.arange(100, dtype=np.int64), 2)
    x_test = np.vstack([
        rng.normal(class_id / 20, .2, size=(1, 4)) for class_id in range(100)
    ]).astype(np.float32)
    y_test = np.arange(100, dtype=np.int64)
    marker = {"md5": "synthetic-only", "sha256": "synthetic-only"}
    return plan, DatasetBundle("ember", x_train, y_train, x_test, y_test, marker, marker)


def _bind_synthetic_storage(service, stage_root: Path, tmp_path: Path, monkeypatch):
    storage = _storage(service, stage_root, monkeypatch)
    output_root = stage_root / "SYNTHETIC_NONREPORTABLE_TEST_ONLY"
    logical_output = "/drive/MyDrive/ReVision/validation/SYNTHETIC_NONREPORTABLE_TEST_ONLY"
    storage.bind_root(logical_output, output_root)
    return storage, output_root, logical_output


def test_real_runner_api_storage_interrupt_fresh_stage_resume_and_idempotent_completion(tmp_path, monkeypatch):
    import torch
    from revision_r1 import resources
    from scripts.production_workflow import execute_frozen_plan_record, _matrix_registry_path
    import scripts.production_workflow as workflow

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    progress_events = []

    def quiet_progress(event):
        progress_events.append(event)
    published_local_paths = []
    real_api_publish = workflow._api_publish_file

    def capture_api_publish(storage, local_path):
        result = real_api_publish(storage, local_path)
        if Path(local_path).name == "matrix_completion_registry.json":
            published_local_paths.append(str(storage.logical_path_for_local(local_path)))
        return result

    monkeypatch.setattr(workflow, "_api_publish_file", capture_api_publish)
    # This integration test exercises the real runner, serialization, restore,
    # and API publication path. Its purpose is not to test the host-RAM guard;
    # that guard has separate real-snapshot tests. The shared CI host runs close
    # to the immutable 1 GiB reserve after importing CPU PyTorch, so give this
    # synthetic-only fixture deterministic headroom without changing the guard
    # implementation or any production settings.
    real_memory_snapshot = resources.read_host_memory_snapshot

    def synthetic_headroom_snapshot():
        snapshot = real_memory_snapshot()
        snapshot["effective_available_bytes"] = max(
            int(snapshot.get("effective_available_bytes") or 0), 4 << 30,
        )
        return snapshot

    monkeypatch.setattr(resources, "read_host_memory_snapshot", synthetic_headroom_snapshot)
    try:
        service = _FakeService()
        plan, bundle = _synthetic_plan_and_bundle()
        record = plan["runs"][0]
        storage1, output1, logical_output = _bind_synthetic_storage(
            service, tmp_path / "stage-first", tmp_path, monkeypatch,
        )

        def stop_at_resumable_boundary(state):
            return state.cursor.task_index == 1 and state.cursor.phase == "scaler"

        interrupted = execute_frozen_plan_record(
            plan, record, bundle, output_root=output1,
            checkpoint_root=output1 / "checkpoints",
            local_checkpoint_root=tmp_path / "stage-first" / "local-checkpoints",
            synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
            stop_hook=stop_at_resumable_boundary,
            api_factory=lambda: service, storage_adapter=storage1,
            logical_output_root=logical_output,
            logical_checkpoint_root="/drive/MyDrive/ReVision/validation/checkpoints",
            progress_callback=quiet_progress,
        )
        assert interrupted["status"] == "INTERRUPTED_RESUMABLE"
        assert interrupted["completion_count"] == 0
        local_recovery_dir = (
            output1 / "checkpoints" / record["run_key"] / record["config_sha256"][:16] / "recovery"
        )
        logical_recovery_dir = storage1.logical_path_for_local(local_recovery_dir)
        recovery_parent_id = storage1.folder_id(logical_recovery_dir, create=False)
        generation_payloads = [
            row for row in service.records.values()
            if row.get("name", "").startswith("generation-") and row.get("name", "").endswith(".pt")
            and recovery_parent_id in row.get("parents", [])
        ]
        generation_manifests = [
            row for row in service.records.values()
            if row.get("name", "").startswith("generation-") and row.get("name", "").endswith(".manifest.json")
            and recovery_parent_id in row.get("parents", [])
        ]
        assert generation_payloads and generation_manifests
        assert all(row["owners"][0]["emailAddress"] == OWNER for row in generation_payloads)
        assert all(row["content"] for row in generation_payloads)

        highest_remote_generation = max(
            int(json.loads(row["content"]).get("generation", 0))
            for row in generation_manifests
        )
        preserved_remote_payload_ids = {
            row["id"] for row in generation_payloads
            if int(row["name"].split("-")[1].split(".")[0]) <= highest_remote_generation
        }

        # Fresh process/session analogue: no local checkpoints or outputs are
        # reused; both are reconstructed from the exact remote API tree. First
        # force an authorization failure on the next checkpoint publication.
        storage2, output2, _logical_output = _bind_synthetic_storage(
            service, tmp_path / "stage-fresh", tmp_path, monkeypatch,
        )
        service.files().fail_create_name = f"generation-{highest_remote_generation + 1:06d}.pt"
        with pytest.raises(DriveStorageError, match="upload failed"):
            execute_frozen_plan_record(
                plan, record, bundle, output_root=output2,
                checkpoint_root=output2 / "checkpoints",
                local_checkpoint_root=tmp_path / "stage-fresh" / "local-checkpoints",
                synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
                api_factory=lambda: service, storage_adapter=storage2,
                logical_output_root=logical_output,
                logical_checkpoint_root="/drive/MyDrive/ReVision/validation/checkpoints",
                progress_callback=quiet_progress,
            )
        assert preserved_remote_payload_ids <= {
            row["id"] for row in service.records.values()
            if row.get("name", "").startswith("generation-") and row.get("name", "").endswith(".pt")
        }
        assert not any(
            row.get("name") == f"generation-{highest_remote_generation + 1:06d}.manifest.json"
            and recovery_parent_id in row.get("parents", [])
            for row in service.records.values()
        )

        # The failed upload left its local recovery state intact; a third empty
        # staging tree can still restore the last fully published generation.
        storage3, output3, _logical_output = _bind_synthetic_storage(
            service, tmp_path / "stage-resume", tmp_path, monkeypatch,
        )
        resumed = execute_frozen_plan_record(
            plan, record, bundle, output_root=output3,
            checkpoint_root=output3 / "checkpoints",
            local_checkpoint_root=tmp_path / "stage-resume" / "local-checkpoints",
            synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
            api_factory=lambda: service, storage_adapter=storage3,
            logical_output_root=logical_output,
            logical_checkpoint_root="/drive/MyDrive/ReVision/validation/checkpoints",
            progress_callback=quiet_progress,
        )
        assert resumed["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
        assert resumed["completion_count"] == 1
        assert resumed["completion_registry_added_now"] is True
        completion_markers = list(
            (Path(resumed["output_path"]) / "registry" / "runs").glob("*/COMPLETE.json")
        )
        assert len(completion_markers) == 1
        remote_marker = storage3.exact_file(storage3.logical_path_for_local(completion_markers[0]))
        assert remote_marker is not None
        assert remote_marker["size"] == str(completion_markers[0].stat().st_size)
        progress_path = (
            output3 / "progress" / plan["plan_sha256"]
            / f"{record['run_key']}_{record['config_sha256'][:12]}.json"
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert progress["status"] == "SYNTHETIC_VALIDATED_NON_REPORTABLE"
        assert any(
            event.get("matrix_lock_is_best_effort") is False
                and event.get("matrix_lock_backend") == "canonical_drive_v2_etag_if_match"
            for event in progress_events
        )

        # The API-published matrix registry records exactly one completion.
        matrix_rows = [
            row for row in service.records.values()
            if row.get("name") == "matrix_completion_registry.json"
        ]
        assert len(matrix_rows) == 1
        expected_matrix_logical = storage3.logical_path_for_local(
            _matrix_registry_path(output3, plan),
        )
        assert str(expected_matrix_logical) in published_local_paths
        assert storage3.exact_file(expected_matrix_logical) is not None
        matrix_before = json.loads(matrix_rows[0]["content"])
        assert len(matrix_before["completed_runs"]) == 1
        attempts_before = len(next(
            json.loads(row["content"]) for row in service.records.values()
            if row.get("name") == "registry.json" and b"synthetic_validation" in row.get("content", b"")
        )["attempts"])

        storage4, output4, _logical_output = _bind_synthetic_storage(
            service, tmp_path / "stage-rerun", tmp_path, monkeypatch,
        )
        from revision_r1.registry import RunRegistry
        from scripts.production_workflow import (
            _hydrate_api_matrix_registry, _hydrate_api_run_state,
            _read_matrix_registry, _run_id_for,
        )
        rerun_output = (
            output4 / "runs" / record["run_key"]
            / record["training_source_sha256"][:16] / record["config_sha256"][:16]
        )
        rerun_checkpoints = output4 / "checkpoints" / record["run_key"] / record["config_sha256"][:16]
        remote_matrix_path = storage4.logical_path_for_local(_matrix_registry_path(output4, plan))
        assert storage4.exact_file(remote_matrix_path) is not None, (
            "completed matrix registry was not published under its canonical path; "
            f"stage3={expected_matrix_logical}, stage4={remote_matrix_path}, "
            f"published={published_local_paths}"
        )
        _hydrate_api_run_state(
            storage4, output4, plan, record, rerun_output, rerun_checkpoints,
        )
        _hydrate_api_matrix_registry(storage4, output4, plan)
        assert RunRegistry(rerun_output / "registry").is_completed(_run_id_for(record, synthetic=True)), (
            "fresh API hydration did not restore a hash-valid completed run; refusing to execute the rerun"
        )
        assert record["run_key"] in _read_matrix_registry(output4, plan)["completed_runs"]
        repeated = execute_frozen_plan_record(
            plan, record, bundle, output_root=output4,
            checkpoint_root=output4 / "checkpoints",
            local_checkpoint_root=tmp_path / "stage-rerun" / "local-checkpoints",
            synthetic=True, test_only=True, device="cpu", update_counts=[1] * 11,
            api_factory=lambda: service, storage_adapter=storage4,
            logical_output_root=logical_output,
            logical_checkpoint_root="/drive/MyDrive/ReVision/validation/checkpoints",
            progress_callback=quiet_progress,
        )
        # run_one returns the validated artifact status from its completion
        # summary even on its early already-completed path; registry attempt
        # count and completion registration below are the idempotency proof.
        assert repeated["status"] in {
            "ALREADY_VALIDATED", "SYNTHETIC_VALIDATED_NON_REPORTABLE",
        }
        assert repeated["completion_registry_added_now"] is False
        matrix_after = json.loads(matrix_rows[0]["content"])
        assert len(matrix_after["completed_runs"]) == 1
        assert len(next(
            json.loads(row["content"]) for row in service.records.values()
            if row.get("name") == "registry.json" and b"synthetic_validation" in row.get("content", b"")
        )["attempts"]) == attempts_before
    finally:
        torch.set_num_threads(previous_threads)


def test_fresh_api_registry_staging_reuses_run_one_and_cell_f_selects_current_only(
    tmp_path, monkeypatch,
):
    import threading

    from revision_r1.config import DATASETS
    from scripts import production_workflow as workflow

    package_root = Path(__file__).resolve().parents[1]
    dataset_references = {
        name: {
            "train_md5": values["train"]["md5"],
            "test_md5": values["test"]["md5"],
            "feature_dim": values["feature_dim"],
            "train_shape": list(values["train"]["shape"]),
            "test_shape": list(values["test"]["shape"]),
        }
        for name, values in DATASETS.items()
    }
    plan = workflow.build_frozen_run_plan(
        package_root,
        schedule_sha256="7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212",
        dataset_references=dataset_references,
    )
    first = plan["runs"][0]
    second = plan["runs"][1]
    assert first["run_key"] == "ember_seed42_full_hybrid"
    assert second["run_key"] == "ember_seed42_current_only"

    logical_results = Path(
        "/drive/MyDrive/ReVision/revision_R1/production/results"
    )
    logical_checkpoints = Path(
        "/drive/MyDrive/ReVision/revision_R1/production/checkpoints"
    )
    service = _FakeService()
    seed_root = tmp_path / "seed-stage"
    seed_results = seed_root / "production" / "results"
    seed_storage = _storage(service, seed_root, monkeypatch)
    seed_storage.bind_root(logical_results, seed_results)

    run_id = workflow._run_id_for(first, synthetic=False)
    artifact_sha = "a" * 64
    measurement = {
        "status": "MEASURED_VALIDATED_PRODUCTION_RUN",
        "run_key": first["run_key"],
        "run_id": run_id,
        "max_checkpoint_payload_bytes": 1_000,
        "run_output_bytes": 2_000,
    }
    entry = {
        "run_key": first["run_key"],
        "run_index": first["run_index"],
        "config_sha256": first["config_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "class_order_sha256": first["class_order_sha256"],
        "dataset_identity": first.get("dataset_identity"),
        "run_id": run_id,
        "artifact_manifest_sha256": artifact_sha,
        "storage_measurement": measurement,
    }
    registry_document = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule_sha256"],
        "training_source_sha256": plan["training_source_sha256"],
        "completed_runs": {first["run_key"]: entry},
    }
    canonical_registry = workflow._matrix_registry_path(seed_results, plan)
    canonical_registry.parent.mkdir(parents=True)
    canonical_registry.write_text(
        json.dumps(registry_document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    logical_registry = seed_storage.logical_path_for_local(canonical_registry)
    published = seed_storage.publish_file(canonical_registry, logical_registry)
    assert published["status"] == "PASS"
    original_registry_bytes = canonical_registry.read_bytes()

    # A new Colab session has empty local staging and must hydrate the existing
    # canonical API registry before Cells C/E/F inspect completed runs.
    fresh_root = tmp_path / "fresh-stage"
    fresh_results = fresh_root / "production" / "results"
    fresh_checkpoints = fresh_root / "production" / "checkpoints"
    fresh_storage = _storage(service, fresh_root, monkeypatch)
    fresh_storage.bind_root(logical_results, fresh_results)
    fresh_storage.bind_root(logical_checkpoints, fresh_checkpoints)
    assert not fresh_results.exists()
    from revision_r1.storage import StorageCapacityError

    with pytest.raises(StorageCapacityError, match="not present in local staging"):
        workflow.matrix_registry_evidence(fresh_storage, fresh_results, plan)
    assert not workflow._matrix_registry_path(fresh_results, plan).exists()
    workflow._hydrate_api_matrix_registry(fresh_storage, fresh_results, plan)
    staged_registry = workflow._matrix_registry_path(fresh_results, plan)
    assert staged_registry.read_bytes() == original_registry_bytes

    def validated_run(_plan, record, _bundle, _output, *, synthetic):
        assert synthetic is False
        if record["run_key"] == first["run_key"]:
            return {"run_id": run_id, "artifact_manifest_sha256": artifact_sha}
        raise FileNotFoundError(record["run_key"])

    monkeypatch.setattr(workflow, "_validate_completed_run", validated_run)
    # Cell C's finalized-run reuse must consult the plan-level registry root.
    assert workflow.reuse_validated_first_run_measurement(
        plan, first, fresh_results,
    ) == measurement
    registry_evidence = workflow.matrix_registry_evidence(
        fresh_storage, fresh_results, plan,
    )
    assert registry_evidence == {
        "logical_drive_path": str(logical_registry),
        "drive_file_id": published["file_id"],
        "drive_parent_id": fresh_storage.folder_id(logical_registry.parent, create=False),
        "local_staging_path": str(staged_registry),
        "plan_sha256": plan["plan_sha256"],
        "registry_bytes": len(original_registry_bytes),
        "registry_md5": hashlib.md5(original_registry_bytes, usedforsecurity=False).hexdigest(),
        "completed_run_keys": ["ember_seed42_full_hybrid"],
    }

    # Cell E is deliberately read-only and reuses the evidence from Cell C.
    import contextlib
    import io
    from scripts.build_github_handoff import _github_completed_run_guard_cell

    cell_e_output = io.StringIO()
    with contextlib.redirect_stdout(cell_e_output):
        exec(_github_completed_run_guard_cell(), {
            "MATRIX_REGISTRY_EVIDENCE": registry_evidence,
        })
    assert "ALREADY_FINALIZED_COMPLETED_RUN" in cell_e_output.getvalue()
    assert "CELL_E_REUSES_CELL_C_REGISTRY_EVIDENCE" in cell_e_output.getvalue()
    assert published["file_id"] in cell_e_output.getvalue()
    assert "ember_seed42_full_hybrid" in cell_e_output.getvalue()

    # Cell F must use the same root, skip Run 1, and stop at the first pending
    # record. The intercepted runner returns before any model/training work.
    lease_events = []
    lease_state = {"acquired": False}

    class TrackedLease:
        def __init__(self, *_args, **_kwargs):
            self.lost = threading.Event()

        def __enter__(self):
            lease_state["acquired"] = True
            lease_events.append("lease_acquired")
            return self

        def __exit__(self, *_exc):
            lease_state["acquired"] = False
            lease_events.append("lease_released")
            return False

    original_hydrate_registry = workflow._hydrate_api_matrix_registry
    f_hydration_count = 0

    def track_f_registry_hydration(storage_adapter, root, runtime_plan):
        nonlocal f_hydration_count
        original_hydrate_registry(storage_adapter, root, runtime_plan)
        f_hydration_count += 1
        lease_events.append(("registry_hydrated", f_hydration_count, lease_state["acquired"]))
        if f_hydration_count == 1:
            # Simulate stale local staging after the pre-lease fetch. The
            # under-lease fetch must replace it from the canonical API bytes
            # before Run 1 can be reused or the next key selected.
            staged = workflow._matrix_registry_path(Path(root), runtime_plan)
            stale = json.loads(staged.read_text(encoding="utf-8"))
            stale["completed_runs"] = {}
            staged.write_text(json.dumps(stale), encoding="utf-8")
            lease_events.append("prelease_staging_invalidated")

    monkeypatch.setattr(workflow, "_require_canonical_storage_api_factory", lambda _factory: None)
    monkeypatch.setattr(workflow, "validate_runtime_bound_plan", lambda value, *_a, **_k: value)
    monkeypatch.setattr(workflow, "_MatrixLeaseHeartbeat", TrackedLease)
    monkeypatch.setattr(workflow, "_hydrate_api_matrix_registry", track_f_registry_hydration)
    monkeypatch.setattr(fresh_storage, "ensure_lease_record", lambda _path: {"status": "EXISTING_EXACT_ID"})
    monkeypatch.setattr(workflow, "project_storage", lambda *_a, **_k: {"status": "TEST_ONLY"})
    monkeypatch.setattr(
        workflow, "inspect_production_capacity",
        lambda *_a, **_k: {"production_measurements_validated": True},
    )
    monkeypatch.setattr(workflow, "require_run_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(workflow, "_preflight_addressed_cache_bytes", lambda *_a, **_k: 0)
    monkeypatch.setattr(workflow, "_bundle_cache_bytes", lambda *_a, **_k: 0)
    monkeypatch.setattr(workflow, "_expected_plan_runner_identity", lambda *_a, **_k: {})
    monkeypatch.setattr(workflow, "_expected_runner_identity", lambda *_a, **_k: {})
    monkeypatch.setattr(
        workflow, "_SequentialDatasetBundleCache",
        lambda _loader: type("Cache", (), {"get": lambda _self, _name: object()})(),
    )
    runner_calls = []

    def stop_before_training(_plan, record, *_args, **_kwargs):
        runner_calls.append(record["run_key"])
        return {"status": "INTERRUPTED_RESUMABLE", "cursor": None}

    monkeypatch.setattr(workflow, "execute_frozen_plan_record", stop_before_training)
    events = []
    result = workflow.continue_production_matrix(
        plan,
        preflight={},
        schedule_path=tmp_path / "schedule.csv",
        schedule_manifest_path=tmp_path / "schedule.manifest.json",
        data_roots={},
        cache_root=tmp_path / "cache",
        output_root=fresh_results,
        checkpoint_root=fresh_checkpoints,
        local_checkpoint_root=tmp_path / "local-checkpoints",
        launch_enabled=True,
        progress_callback=events.append,
        api_factory=lambda: service,
        storage_adapter=fresh_storage,
        logical_output_root=logical_results,
        logical_checkpoint_root=Path(
            "/drive/MyDrive/ReVision/revision_R1/production/checkpoints"
        ),
    )

    assert result["status"] == "INTERRUPTED_RESUMABLE"
    assert result["interrupted_run"] == "ember_seed42_current_only"
    assert result["run_index"] == 2
    assert runner_calls == ["ember_seed42_current_only"]
    assert result["first_run_reused"] == "ember_seed42_full_hybrid"
    assert result["matrix_registry_evidence_at_start"] == registry_evidence
    assert f_hydration_count == 2
    assert lease_events == [
        ("registry_hydrated", 1, False),
        "prelease_staging_invalidated",
        "lease_acquired",
        ("registry_hydrated", 2, True),
        "lease_released",
    ]
    matrix_registry_event = next(
        event for event in events if event.get("stage") == "matrix_registry_verified"
    )
    assert matrix_registry_event["matrix_registry_evidence"] == registry_evidence
    assert matrix_registry_event["next_pending_run_key"] == "ember_seed42_current_only"
    assert len(workflow._read_matrix_registry(fresh_results, plan)["completed_runs"]) == 1
    assert workflow._read_matrix_registry(fresh_results, plan)["completed_runs"].keys() == {first["run_key"]}
    remote_metadata = fresh_storage.exact_file(logical_registry)
    assert remote_metadata["id"] == published["file_id"]
    assert remote_metadata["version"] == service.records[published["file_id"]]["version"] == "1"
    assert remote_metadata["md5Checksum"] == hashlib.md5(
        original_registry_bytes, usedforsecurity=False,
    ).hexdigest()
    assert fresh_storage.download_file(
        logical_registry, tmp_path / "registry-readback.json",
    )["bytes"] == len(original_registry_bytes)
    assert json.loads((tmp_path / "registry-readback.json").read_bytes()) == registry_document
