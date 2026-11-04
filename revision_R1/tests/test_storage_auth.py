from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.storage_auth import (
    STORAGE_CLIENT_ID_SECRET,
    STORAGE_CLIENT_SECRET_SECRET,
    STORAGE_REFRESH_TOKEN_SECRET,
    StorageOAuthError,
    build_main_account_storage_api,
    build_main_account_drive_v2_api,
)


MAIN_ACCOUNT = "kian.anb777@gmail.com"


def test_default_drive_builder_isolates_real_http_requests_by_thread(monkeypatch):
    import concurrent.futures
    import threading
    import httplib2
    from google.auth.credentials import AnonymousCredentials
    from googleapiclient import discovery
    from scripts import storage_auth

    captured = {}
    monkeypatch.setattr(discovery, "build", lambda *_a, **kw: captured.update(kw))
    storage_auth._default_drive_service_builder(
        "drive", "v3", credentials=AnonymousCredentials(), cache_discovery=False,
    )
    assert "requestBuilder" in captured, "default shared httplib2 transport is unsafe"
    build_request = captured["requestBuilder"]
    barrier = threading.Barrier(2)
    transports = []
    def request(http, uri, method="GET", body=None, headers=None, **kwargs):
        transports.append((threading.get_ident(), id(http)))
        barrier.wait(timeout=5)
        return httplib2.Response({"status": "200"}), b'{"ok":true}'
    monkeypatch.setattr(httplib2.Http, "request", request)
    def worker():
        ids = []
        for _ in range(2):
            req = build_request(object(), lambda _r, body: json.loads(body), "https://example.invalid/drive")
            ids.append(id(req.http))
            assert req.execute() == {"ok": True}
        return ids
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda _: worker(), range(2)))
    assert a[0] == a[1] and b[0] == b[1]
    assert a[0] != b[0]
    assert len({transport for _thread, transport in transports}) == 2


class _FakeCredentials:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAboutRequest:
    def __init__(self, email):
        self.email = email
        self.fields = None

    def execute(self):
        return {"user": {"emailAddress": self.email, "permissionId": "user-1"}}


class _FakeAbout:
    def __init__(self, email):
        self.request = _FakeAboutRequest(email)

    def get(self, *, fields):
        self.request.fields = fields
        return self.request


class _FakeService:
    def __init__(self, email):
        self.about_resource = _FakeAbout(email)

    def about(self):
        return self.about_resource


def _secrets(refresh_token="refresh-secret", email=None):
    return {
        STORAGE_CLIENT_ID_SECRET: "client-id",
        STORAGE_CLIENT_SECRET_SECRET: "client-secret",
        STORAGE_REFRESH_TOKEN_SECRET: refresh_token,
    }


def test_explicit_storage_oauth_injects_colab_secret_credentials_without_adc():
    requested = []
    service = _FakeService(MAIN_ACCOUNT)
    lease_service = _FakeService(MAIN_ACCOUNT)
    service_build_calls = []
    credential_calls = []

    def secret_getter(name):
        requested.append(name)
        return _secrets()[name]

    def service_builder(api_name, version, *, credentials, cache_discovery):
        assert api_name == "drive"
        assert version in {"v2", "v3"}
        assert credentials.kwargs["refresh_token"] == "refresh-secret"
        assert cache_discovery is False
        service_build_calls.append(version)
        return service if version == "v3" else lease_service

    def credentials_factory(**kwargs):
        credential_calls.append(kwargs)
        return _FakeCredentials(**kwargs)

    observed_service, credentials, evidence = build_main_account_storage_api(
        secret_getter=secret_getter,
        expected_account_email=MAIN_ACCOUNT,
        credentials_factory=credentials_factory,
        drive_service_builder=service_builder,
    )

    assert observed_service is service
    assert observed_service._revision_r1_lease_service is lease_service
    assert credentials.kwargs == {
        "token": None,
        "refresh_token": "refresh-secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": ["https://www.googleapis.com/auth/drive"],
    }
    assert requested == [
        STORAGE_CLIENT_ID_SECRET,
        STORAGE_CLIENT_SECRET_SECRET,
        STORAGE_REFRESH_TOKEN_SECRET,
    ]
    assert credential_calls == [credentials.kwargs]
    assert service_build_calls == ["v3", "v2"]
    assert service.about_resource.request.fields == "user(emailAddress,permissionId)"
    assert evidence == {
        "status": "PASS",
        "credential_source": "explicit_colab_secret_user_oauth",
        "authenticated_user": MAIN_ACCOUNT,
        "lease_api_version": "v2_lease_cas_only",
        "lease_authenticated_user": MAIN_ACCOUNT,
    }


def test_explicit_storage_oauth_rejects_wrong_account_without_leaking_secrets(monkeypatch):
    service = _FakeService("kiananbarestani62@gmail.com")
    monkeypatch.setattr(
        "scripts.storage_auth._default_credentials_factory",
        lambda **kwargs: _FakeCredentials(**kwargs),
    )
    monkeypatch.setattr(
        "scripts.storage_auth._default_drive_service_builder",
        lambda *_args, **_kwargs: service,
    )

    with pytest.raises(StorageOAuthError, match="storage account mismatch") as exc:
        build_main_account_storage_api(
            secret_getter=lambda name: _secrets()[name],
            expected_account_email=MAIN_ACCOUNT,
        )

    assert "kiananbarestani62@gmail.com" in str(exc.value)
    assert "refresh-secret" not in str(exc.value)
    assert "client-secret" not in str(exc.value)


def test_drive_v2_lease_api_reuses_explicit_credentials_and_verifies_canonical_user():
    credentials = _FakeCredentials(refresh_token="canonical-refresh")
    service = _FakeService(MAIN_ACCOUNT)
    builder_calls = []

    def builder(api_name, version, *, credentials, cache_discovery):
        builder_calls.append((api_name, version, credentials, cache_discovery))
        return service

    observed, evidence = build_main_account_drive_v2_api(
        credentials=credentials,
        expected_account_email=MAIN_ACCOUNT,
        drive_service_builder=builder,
    )

    assert observed is service
    assert builder_calls == [("drive", "v2", credentials, False)]
    assert service.about_resource.request.fields == "user(emailAddress,permissionId)"
    assert evidence == {
        "status": "PASS",
        "credential_source": "same_explicit_main_account_oauth_as_drive_v3",
        "api_version": "v2_lease_cas_only",
        "authenticated_user": MAIN_ACCOUNT,
    }


def test_drive_v2_lease_api_rejects_wrong_account_and_hides_transport_errors():
    wrong = _FakeService("compute@example.org")
    with pytest.raises(StorageOAuthError, match="storage account mismatch") as caught:
        build_main_account_drive_v2_api(
            credentials=_FakeCredentials(refresh_token="secret-refresh"),
            expected_account_email=MAIN_ACCOUNT,
            drive_service_builder=lambda *_args, **_kwargs: wrong,
        )
    assert "compute@example.org" in str(caught.value)
    assert "secret-refresh" not in str(caught.value)

    with pytest.raises(StorageOAuthError, match="Drive v2 lease API could not verify") as caught:
        build_main_account_drive_v2_api(
            credentials=_FakeCredentials(refresh_token="secret-refresh"),
            expected_account_email=MAIN_ACCOUNT,
            drive_service_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secret-refresh")),
        )
    assert "secret-refresh" not in str(caught.value)


def test_drive_capacity_helpers_require_and_reuse_verified_storage_service(monkeypatch):
    from revision_r1 import storage

    service = _FakeService(MAIN_ACCOUNT)
    service.about_resource.request.execute = lambda: {
        "user": {"emailAddress": MAIN_ACCOUNT},
        "storageQuota": {"limit": "1000", "usage": "125"},
    }
    storage.clear_verified_drive_api_service()

    with pytest.raises(storage.StorageCapacityError, match="explicit main-account OAuth"):
        storage.query_drive_available_bytes()

    storage.bind_verified_drive_api_service(service, expected_account_email=MAIN_ACCOUNT)
    assert storage.query_drive_available_bytes() == 875
    assert storage.verified_drive_api_service(expected_account_email=MAIN_ACCOUNT) is service
    storage.clear_verified_drive_api_service()


def test_drive_capacity_service_binding_rejects_wrong_account_and_clears_prior_binding():
    from revision_r1 import storage

    storage.clear_verified_drive_api_service()
    with pytest.raises(storage.StorageCapacityError, match="storage account mismatch"):
        storage.bind_verified_drive_api_service(
            _FakeService("kiananbarestani62@gmail.com"),
            expected_account_email=MAIN_ACCOUNT,
        )
    with pytest.raises(storage.StorageCapacityError, match="has not been bound"):
        storage.verified_drive_api_service(expected_account_email=MAIN_ACCOUNT)


def test_production_drive_api_factory_has_no_adc_fallback():
    from revision_r1 import storage
    from scripts import production_workflow

    storage.clear_verified_drive_api_service()
    production_workflow._BOUND_DRIVE_API_SERVICE = None
    with pytest.raises(storage.StorageCapacityError, match="ADC fallback is disabled"):
        production_workflow._drive_api_service()


def test_canonical_storage_verifier_rejects_wrong_oauth_service_before_probe(tmp_path, monkeypatch):
    from revision_r1 import storage
    from scripts import production_workflow

    drive_root = tmp_path / "MyDrive"
    revision_folder = drive_root / "ReVision"
    revision_folder.mkdir(parents=True)
    service = _FakeService("kiananbarestani62@gmail.com")
    storage.clear_verified_drive_api_service()
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    monkeypatch.setattr(production_workflow, "_drive_api_create_binary_file", lambda _svc, _parent, _name, _payload: {"id": "probe-file-id"})
    monkeypatch.setattr(production_workflow, "_drive_api_read_bytes", lambda _svc, _file_id: production_workflow.DRIVE_WRITE_PROBE_BYTES)

    with pytest.raises(storage.StorageCapacityError, match="storage account mismatch"):
        production_workflow.verify_canonical_storage_session(
            api_factory=lambda: service,
            drive_mount_root=drive_root,
            expected_account_email=MAIN_ACCOUNT,
            expected_revision_folder_id="canonical-revision-id",
        )

    assert list(revision_folder.iterdir()) == []
    with pytest.raises(storage.StorageCapacityError, match="has not been bound"):
        storage.verified_drive_api_service(expected_account_email=MAIN_ACCOUNT)


def test_canonical_storage_verifier_binds_oauth_service_for_all_following_drive_calls(tmp_path, monkeypatch):
    from revision_r1 import storage
    from scripts import production_workflow

    class _ProbeFiles:
        def __init__(self, service):
            self.service = service

        def delete(self, *, fileId, supportsAllDrives):
            assert fileId == "probe-file-id"
            assert supportsAllDrives is True

            class _DeleteRequest:
                def execute(inner_self):
                    self.service.deleted = True
                    return {}

            return _DeleteRequest()

    class _MainService(_FakeService):
        def __init__(self):
            super().__init__(MAIN_ACCOUNT)
            self.deleted = False
            self.about_resource.request.execute = lambda: {
                "user": {"emailAddress": MAIN_ACCOUNT, "permissionId": "main-user"},
                "storageQuota": {"limit": "1000", "usage": "125"},
            }

        def files(self):
            return _ProbeFiles(self)

    service = _MainService()
    drive_root = tmp_path / "MyDrive"
    revision_path = drive_root / "ReVision"
    revision_path.mkdir(parents=True)

    class _ProbeAdapter:
        def __init__(self):
            self.service = service
            self.expected_account_email = MAIN_ACCOUNT
            self.revision_folder_id = "canonical-revision-id"
            self.logical_revision_root = revision_path

        def probe_write(self, logical_file, payload, *, expected_parent_id):
            assert Path(logical_file).parent == revision_path
            assert expected_parent_id == self.revision_folder_id
            assert payload == production_workflow.DRIVE_WRITE_PROBE_BYTES
            self.service.deleted = True
            observed = {
                "file_id": "probe-file-id", "parent_id": expected_parent_id,
                "owner": MAIN_ACCOUNT, "ownedByMe": True, "size": len(payload),
                "quotaBytesUsed": len(payload), "api_visible": True,
            }
            return {
                "status": "PASS", "file_id": "probe-file-id", "parent_id": expected_parent_id,
                "owner": MAIN_ACCOUNT, "ownedByMe": True, "size": len(payload),
                "quotaBytesUsed": len(payload), "delete_verified": True,
                "quota_usage_before_bytes": 125, "quota_usage_during_bytes": 125 + len(payload),
                "quota_limit_during_bytes": 1000,
                "diagnostics": {"classification": "VERIFIED", "expected": observed, "observed": observed},
            }
    storage.clear_verified_drive_api_service()
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    monkeypatch.setattr(production_workflow, "_drive_api_create_binary_file", lambda _svc, _parent, _name, _payload: {"id": "probe-file-id"})
    monkeypatch.setattr(production_workflow, "_drive_api_read_bytes", lambda _svc, _file_id: production_workflow.DRIVE_WRITE_PROBE_BYTES)
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _service, _parent, name: (
        None if service.deleted else {
            "id": "probe-file-id", "name": name, "mimeType": "application/octet-stream",
            "parents": ["canonical-revision-id"], "owners": [{"emailAddress": MAIN_ACCOUNT}],
            "ownedByMe": True,
            "quotaBytesUsed": str(len(production_workflow.DRIVE_WRITE_PROBE_BYTES)),
            "size": str(len(production_workflow.DRIVE_WRITE_PROBE_BYTES)),
        }
    ))
    monkeypatch.setattr(production_workflow, "resolve_drive_folder", lambda _service, _components: {
        "path_exists": True, "folder_id": "canonical-revision-id", "owner_email": MAIN_ACCOUNT,
        "can_add_children": True, "drive_id": None,
    })

    evidence = production_workflow.verify_canonical_storage_session(
        api_factory=lambda: service,
        drive_mount_root=drive_root,
        expected_account_email=MAIN_ACCOUNT,
        expected_revision_folder_id="canonical-revision-id",
        storage_adapter=_ProbeAdapter(),
        conditional_update_probe=lambda *_args, **_kwargs: {"status": "PASS", "contract": "test_fake_cas"},
    )

    assert evidence["status"] == "PASS"
    assert service.deleted is True
    assert production_workflow._drive_api_service() is service
    assert storage.verified_drive_api_service(expected_account_email=MAIN_ACCOUNT) is service
    assert storage.query_drive_available_bytes() == 875
    assert list((drive_root / "ReVision").iterdir()) == []


def test_api_write_probe_reports_missing_or_stale_metadata_field_by_field(monkeypatch):
    from scripts import production_workflow
    from revision_r1.storage import StorageCapacityError

    monkeypatch.setattr(production_workflow, "DRIVE_PROBE_VISIBILITY_RETRY_DELAYS_SECONDS", ())
    payload = production_workflow.DRIVE_WRITE_PROBE_BYTES
    with pytest.raises(StorageCapacityError, match="metadata remained missing/stale") as caught:
        production_workflow._write_capability_probe(
            "/content/drive/MyDrive/ReVision",
            {"path_exists": True, "folder_id": "revision-folder-id"},
            MAIN_ACCOUNT, object(), drive_root="/content/drive/MyDrive",
            metadata_lookup=lambda *_args: None,
            api_create=lambda *_args: {"id": "probe-file-id"},
            api_read=lambda *_args: payload,
        )

    diagnostic = json.loads(str(caught.value).split(": ", 1)[1])
    assert diagnostic["classification"] == "MISSING_OR_STALE_METADATA"
    assert diagnostic["expected"]["parent_id"] == "revision-folder-id"
    assert diagnostic["observed"] == {
        "file_id": "probe-file-id", "parent_id": None, "owner": None,
        "ownedByMe": None, "size": None, "quotaBytesUsed": None,
        "api_visible": False,
    }
    assert set(diagnostic["conditions"]) == {
        "file_id", "parent_id", "owner", "ownedByMe", "size", "quotaBytesUsed", "api_visible",
    }
    assert diagnostic["conditions"]["parent_id"]["status"] == "MISSING_OR_STALE"
    assert diagnostic["conditions"]["api_visible"]["status"] == "MISMATCH"


def test_api_write_probe_distinguishes_confirmed_new_file_owner_mismatch(monkeypatch):
    from scripts import production_workflow
    from revision_r1.storage import StorageCapacityError

    payload = production_workflow.DRIVE_WRITE_PROBE_BYTES
    metadata = {
        "id": "probe-file-id", "name": "probe.bin", "mimeType": "application/octet-stream",
        "parents": ["revision-folder-id"], "owners": [{"emailAddress": "compute@example.org"}],
        "ownedByMe": False, "size": str(len(payload)), "quotaBytesUsed": str(len(payload)),
    }
    with pytest.raises(StorageCapacityError, match="confirmed ownership mismatch") as caught:
        production_workflow._write_capability_probe(
            "/content/drive/MyDrive/ReVision",
            {"path_exists": True, "folder_id": "revision-folder-id"},
            MAIN_ACCOUNT, object(), drive_root="/content/drive/MyDrive",
            metadata_lookup=lambda *_args: metadata,
            api_create=lambda *_args: {"id": "probe-file-id"},
            api_read=lambda *_args: payload,
        )

    diagnostic = json.loads(str(caught.value).split(": ", 1)[1])
    assert diagnostic["classification"] == "CONFIRMED_OWNERSHIP_MISMATCH"
    assert diagnostic["expected"]["owner"] == MAIN_ACCOUNT
    assert diagnostic["observed"]["owner"] == "compute@example.org"
    assert diagnostic["conditions"]["owner"]["status"] == "MISMATCH"
    assert diagnostic["conditions"]["ownedByMe"]["status"] == "MISMATCH"


def test_validation_inventory_has_no_adc_fallback(tmp_path):
    from revision_r1 import storage
    from revision_r1.validation_workflow import query_drive_validation_inventory

    storage.clear_verified_drive_api_service()
    with pytest.raises(RuntimeError, match="ADC fallback is disabled"):
        query_drive_validation_inventory(tmp_path)


@pytest.mark.parametrize(
    "missing_name",
    [STORAGE_CLIENT_ID_SECRET, STORAGE_CLIENT_SECRET_SECRET, STORAGE_REFRESH_TOKEN_SECRET],
)
def test_explicit_storage_oauth_requires_each_secret_and_never_echoes_values(missing_name):
    supplied = _secrets()
    supplied.pop(missing_name)

    with pytest.raises(StorageOAuthError, match=missing_name) as exc:
        build_main_account_storage_api(
            secret_getter=lambda name: supplied.get(name),
            expected_account_email=MAIN_ACCOUNT,
        )

    assert "refresh-secret" not in str(exc.value)
    assert "client-secret" not in str(exc.value)
