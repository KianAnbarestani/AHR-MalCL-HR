from __future__ import annotations

import pytest

from scripts.storage_auth import (
    STORAGE_CLIENT_ID_SECRET,
    STORAGE_CLIENT_SECRET_SECRET,
    STORAGE_REFRESH_TOKEN_SECRET,
    StorageOAuthError,
    build_main_account_storage_api,
)


MAIN_ACCOUNT = "kian.anb777@gmail.com"


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
    credential_calls = []

    def secret_getter(name):
        requested.append(name)
        return _secrets()[name]

    def service_builder(api_name, version, *, credentials, cache_discovery):
        assert api_name == "drive"
        assert version == "v3"
        assert credentials.kwargs["refresh_token"] == "refresh-secret"
        assert cache_discovery is False
        return service

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
    assert service.about_resource.request.fields == "user(emailAddress,permissionId)"
    assert evidence == {
        "status": "PASS",
        "credential_source": "explicit_colab_secret_user_oauth",
        "authenticated_user": MAIN_ACCOUNT,
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
    (drive_root / "ReVision").mkdir(parents=True)
    storage.clear_verified_drive_api_service()
    monkeypatch.setattr(production_workflow, "_BOUND_DRIVE_API_SERVICE", None)
    monkeypatch.setattr(production_workflow, "_BOUND_STORAGE_SESSION", None)
    monkeypatch.setattr(production_workflow, "_DRIVE_WRITE_PROBE_CACHE", {})
    monkeypatch.setattr(production_workflow, "_lookup_drive_child", lambda _service, _parent, name: (
        None if service.deleted else {
            "id": "probe-file-id", "name": name, "mimeType": "application/octet-stream",
            "parents": ["canonical-revision-id"], "owners": [{"emailAddress": MAIN_ACCOUNT}],
            "ownedByMe": True, "quotaBytesUsed": "64", "size": "64",
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
    )

    assert evidence["status"] == "PASS"
    assert service.deleted is True
    assert production_workflow._drive_api_service() is service
    assert storage.verified_drive_api_service(expected_account_email=MAIN_ACCOUNT) is service
    assert storage.query_drive_available_bytes() == 875
    assert list((drive_root / "ReVision").iterdir()) == []


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
