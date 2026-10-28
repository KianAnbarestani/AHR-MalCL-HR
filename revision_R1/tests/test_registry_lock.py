from __future__ import annotations

import json
import time

import pytest

from revision_r1.locking import LeaseError, RunLease
from revision_r1.registry import RunRegistry


def test_lease_refuses_concurrent_owner_renews_and_checks_release_token(tmp_path):
    path = tmp_path / "lease.json"
    first = RunLease(path, ttl_seconds=60, owner_token="owner-1")
    second = RunLease(path, ttl_seconds=60, owner_token="owner-2")
    first.acquire()
    with pytest.raises(LeaseError, match="held"):
        second.acquire()
    before = json.loads(path.read_text())
    first.heartbeat()
    after = json.loads(path.read_text())
    assert after["owner_token"] == "owner-1"
    assert after["expires_at"] >= before["expires_at"]
    with pytest.raises(LeaseError, match="owner token"):
        second.release()
    assert path.exists()
    first.release()
    assert not path.exists()


def test_expired_lease_can_be_recovered_without_losing_new_owner(tmp_path):
    path = tmp_path / "lease.json"
    path.write_text(json.dumps({"owner_token": "dead", "expires_at": time.time() - 10}))
    replacement = RunLease(path, ttl_seconds=60, owner_token="live")
    replacement.acquire()
    assert json.loads(path.read_text())["owner_token"] == "live"
    replacement.release()


def test_registry_retains_attempt_failure_and_skips_only_valid_completion(tmp_path):
    registry = RunRegistry(tmp_path / "registry")
    run_id = "ember-seed42-current-only"
    registry.begin_attempt(run_id, identity={"config": "hash-a"})
    registry.fail_attempt(run_id, "synthetic interruption")
    assert registry.status(run_id)["attempts"][-1]["failure_reason"] == "synthetic interruption"
    assert not registry.is_completed(run_id)

    registry.begin_attempt(run_id, identity={"config": "hash-a"})
    artifact = tmp_path / "result.csv"
    artifact.write_text("seed,accuracy\n42,0.8\n")
    assert registry.complete(run_id, {"result.csv": artifact}) is True
    assert registry.is_completed(run_id)
    artifact.write_text("tampered\n")
    assert not registry.is_completed(run_id)


def test_registry_does_not_accept_missing_completion_artifact(tmp_path):
    registry = RunRegistry(tmp_path / "registry")
    run_id = "az-seed43-derpp"
    registry.begin_attempt(run_id, identity={"config": "hash-b"})
    with pytest.raises((ValueError, FileNotFoundError)):
        registry.complete(run_id, {"missing.csv": tmp_path / "missing.csv"})
    assert not registry.is_completed(run_id)
