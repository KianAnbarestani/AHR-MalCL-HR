"""Explicit main-account OAuth credentials for the canonical Drive API client.

Colab's Application Default Credentials may already contain a valid user
credential for a different Google account. This module deliberately does not
consult ADC: callers inject the canonical user's OAuth client and refresh
token from Colab Secrets, then verify the identity through Drive ``about``
before receiving a usable API service.
"""

from __future__ import annotations

from typing import Any, Callable


STORAGE_CLIENT_ID_SECRET = "REVISION_R1_STORAGE_OAUTH_CLIENT_ID"
STORAGE_CLIENT_SECRET_SECRET = "REVISION_R1_STORAGE_OAUTH_CLIENT_SECRET"
STORAGE_REFRESH_TOKEN_SECRET = "REVISION_R1_STORAGE_REFRESH_TOKEN"
STORAGE_TOKEN_URI = "https://oauth2.googleapis.com/token"
STORAGE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


class StorageOAuthError(RuntimeError):
    """Raised when explicit canonical-user Drive OAuth cannot be established."""


def _default_credentials_factory(**kwargs):
    from google.oauth2.credentials import Credentials

    return Credentials(**kwargs)


def _default_drive_service_builder(api_name: str, version: str, **kwargs):
    from googleapiclient.discovery import build

    return build(api_name, version, **kwargs)


def _read_secret(secret_getter: Callable[[str], Any], name: str) -> str:
    try:
        value = secret_getter(name)
    except Exception as exc:
        raise StorageOAuthError(
            f"Colab Secret {name} is unavailable; add it in Notebook settings → Secrets and enable notebook access"
        ) from None
    if not isinstance(value, str) or not value.strip():
        raise StorageOAuthError(
            f"Colab Secret {name} is missing or empty; add the required storage OAuth secret and enable notebook access"
        )
    return value.strip()


def build_main_account_storage_api(
    *,
    secret_getter: Callable[[str], Any],
    expected_account_email: str,
    credentials_factory: Callable[..., Any] | None = None,
    drive_service_builder: Callable[..., Any] | None = None,
) -> tuple[Any, Any, dict[str, str]]:
    """Build a Drive API service only from explicit OAuth secrets and verify its user.

    The API service is never sourced from ``google.auth.default()``. Its refresh
    token belongs to the account that authorized the OAuth consent flow; Drive
    ``about`` must return the exact configured account before the caller can use
    the service for any further request or write.
    """
    expected = str(expected_account_email).strip().lower()
    if not expected:
        raise ValueError("expected storage account email must be explicit")

    client_id = _read_secret(secret_getter, STORAGE_CLIENT_ID_SECRET)
    client_secret = _read_secret(secret_getter, STORAGE_CLIENT_SECRET_SECRET)
    refresh_token = _read_secret(secret_getter, STORAGE_REFRESH_TOKEN_SECRET)
    credentials_factory = credentials_factory or _default_credentials_factory
    service_builder = drive_service_builder or _default_drive_service_builder

    try:
        credentials = credentials_factory(
            token=None,
            refresh_token=refresh_token,
            token_uri=STORAGE_TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            scopes=[STORAGE_DRIVE_SCOPE],
        )
        service = service_builder(
            "drive", "v3", credentials=credentials, cache_discovery=False,
        )
        account = service.about().get(fields="user(emailAddress,permissionId)").execute()
    except Exception as exc:
        raise StorageOAuthError(
            "explicit Drive OAuth could not verify the canonical storage account; check the Colab Secret values, "
            "OAuth consent, and Drive API access, then rerun Cell A"
        ) from None

    observed = str((account.get("user") or {}).get("emailAddress") or "").strip().lower()
    if observed != expected:
        raise StorageOAuthError(
            f"storage account mismatch: expected {expected!r}, observed {observed!r}; "
            "no production paths or checkpoints were modified"
        )
    return service, credentials, {
        "status": "PASS",
        "credential_source": "explicit_colab_secret_user_oauth",
        "authenticated_user": observed,
    }
