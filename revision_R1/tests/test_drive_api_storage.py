from __future__ import annotations

import hashlib
import json
import re
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
    service = _FakeService()
    lease_service = _FakeV2Service(service)
    storage = _storage(service, tmp_path / "local", monkeypatch, lease_service=lease_service)
    local_lock = tmp_path / "local/lease.json"
    local_lock.parent.mkdir(parents=True, exist_ok=True)
    local_lock.write_text(json.dumps({"schema_version": 1, "active": False, "expires_at": 0}) + "\n")
    storage.publish_file(local_lock, "/drive/MyDrive/ReVision/results/matrix.lease.json")

    lease1 = DriveApiLease(storage, "/drive/MyDrive/ReVision/results/matrix.lease.json", owner_token="worker-1")
    lease2 = DriveApiLease(storage, "/drive/MyDrive/ReVision/results/matrix.lease.json", owner_token="worker-2")
    lease1.acquire()
    with pytest.raises(DriveLeaseError, match="held"):
        lease2.acquire()
    lease1.heartbeat()
    lease1.release()
    lease2.acquire()
    lease2.release()
    assert lease_service.files().update_calls >= 5
    assert service.files().update_calls == 0, "lease CAS updates must not use the Drive v3 data client"

    cas = verify_conditional_update_support(
        service, parent_id="results", expected_account_email=OWNER, lease_service=lease_service,
    )
    assert cas["status"] == "PASS"
    assert cas["contract"] == "stale_if_match_rejected_http_412"
    assert not any("cas-probe" in row["name"] for row in service.records.values())


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
