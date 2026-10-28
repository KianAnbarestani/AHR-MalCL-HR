from __future__ import annotations

import pytest

from revision_r1.storage import (
    StorageCapacityError,
    cleanup_unpublished_checkpoint_temps,
    drive_available_bytes_from_quota,
    effective_generation_limit,
    require_available_bytes,
    should_persist_task_boundary_duplicate,
)


def test_synthetic_checkpoint_retention_is_capped_at_two_even_if_unbounded_requested():
    identity = {"execution_scope": "synthetic_validation"}

    assert effective_generation_limit(identity, requested=None) == 2
    assert effective_generation_limit(identity, requested=11) == 2
    assert effective_generation_limit(identity, requested=2) == 2


def test_production_checkpoint_retention_policy_is_unchanged():
    identity = {"execution_scope": "production"}

    assert effective_generation_limit(identity, requested=None) is None
    assert effective_generation_limit(identity, requested=11) == 11


def test_synthetic_validation_does_not_create_redundant_task_boundary_checkpoint_stream():
    assert should_persist_task_boundary_duplicate({"execution_scope": "synthetic_validation"}) is False
    assert should_persist_task_boundary_duplicate({"execution_scope": "production"}) is True


def test_checkpoint_temp_cleanup_removes_only_unpublished_checkpoint_temp_files(tmp_path):
    stale_payload = tmp_path / ".generation-000004.pt.abc123.tmp"
    stale_manifest = tmp_path / ".generation-000004.manifest.json.def456.tmp"
    unrelated = tmp_path / ".report.json.xyz.tmp"
    verified_payload = tmp_path / "generation-000003.pt"
    verified_manifest = tmp_path / "generation-000003.manifest.json"
    for path in (stale_payload, stale_manifest, unrelated, verified_payload, verified_manifest):
        path.write_bytes(b"fixture")

    removed = cleanup_unpublished_checkpoint_temps(tmp_path)

    assert set(removed) == {str(stale_payload), str(stale_manifest)}
    assert not stale_payload.exists()
    assert not stale_manifest.exists()
    assert unrelated.exists()
    assert verified_payload.exists()
    assert verified_manifest.exists()


def test_insufficient_storage_is_rejected_before_write_with_required_and_available_bytes(tmp_path):
    with pytest.raises(StorageCapacityError, match=r"required 110 bytes.*available 109 bytes"):
        require_available_bytes(
            tmp_path, required_bytes=100, reserve_bytes=10, available_bytes=109,
            storage_kind="local",
        )


def test_exact_capacity_boundary_is_accepted(tmp_path):
    require_available_bytes(
        tmp_path, required_bytes=100, reserve_bytes=10, available_bytes=110,
        storage_kind="local",
    )


def test_drive_quota_requires_known_limit_and_usage():
    assert drive_available_bytes_from_quota({"limit": "5000", "usage": "1300"}) == 3700

    with pytest.raises(StorageCapacityError, match="cannot verify Google Drive free quota"):
        drive_available_bytes_from_quota({"usage": "1300"})


def test_drive_quota_rejects_inconsistent_usage():
    with pytest.raises(StorageCapacityError, match="usage exceeds quota limit"):
        drive_available_bytes_from_quota({"limit": "100", "usage": "101"})


def test_validation_storage_inventory_reports_drive_trash_and_checkpoint_generations():
    from revision_r1.validation_workflow import summarize_drive_storage_inventory

    report = summarize_drive_storage_inventory(
        {"limit": "10000", "usage": "9000", "usageInDrive": "5100", "usageInDriveTrash": "1700"},
        [
            {"name": "generation-000021.pt", "size": "1200"},
            {"name": "generation-000021.manifest.json", "size": "100"},
            {"name": "notes.txt", "size": "900"},
            {"name": "generation-bad.pt", "size": "80"},
        ],
    )

    assert report["available_account_bytes"] == 1000
    assert report["usage_in_drive_bytes"] == 5100
    assert report["usage_in_drive_trash_bytes"] == 1700
    assert report["trashed_checkpoint_generation_files"] == 2
    assert report["trashed_checkpoint_generation_bytes"] == 1300


def test_gan_resume_storage_plan_bounds_atomic_drive_write_and_local_peak():
    from revision_r1.validation_workflow import plan_gan_resume_storage

    payload_bound = 200 * 1024 * 1024
    plan = plan_gan_resume_storage(
        max_checkpoint_payload_bytes=payload_bound,
        available_local_bytes=3 * 1024**3,
        available_drive_bytes=(
            2 * payload_bound + 64 * 1024 + 4 * 32 * 1024**2
            + 80 * 1024**2 + 64 * 1024 + 512 * 1024**2
        ),
    )

    assert plan["status"] == "PASS"
    assert plan["persistent_drive_checkpoint_generations"] == 1
    assert plan["drive_checkpoint_write_volume_bound_bytes"] == 2 * payload_bound + 64 * 1024
    assert plan["drive_checkpoint_persistent_payload_bytes"] == payload_bound
    assert plan["drive_atomic_checkpoint_peak_bytes"] == 2 * payload_bound
    assert plan["safety_reserve_bytes"] == 512 * 1024**2
    assert plan["expected_local_checkpoint_slots"] == 8


def test_gan_resume_storage_plan_reports_total_validation_footprint_and_write_volume():
    from revision_r1.validation_workflow import plan_gan_resume_storage

    mib = 1024 * 1024
    gib = 1024**3
    plan = plan_gan_resume_storage(
        max_checkpoint_payload_bytes=200 * mib,
        available_local_bytes=4 * gib,
        available_drive_bytes=2 * gib,
        existing_workspace_bytes=100 * mib,
        existing_visible_validation_bytes=14 * gib,
        existing_local_validation_bytes=1 * gib,
        existing_drive_account_usage_bytes=17 * gib,
        existing_drive_trash_usage_bytes=2 * gib,
    )

    assert plan["projected_drive_validation_visible_after_bytes"] == 14 * gib + 336 * mib + 64 * 1024
    assert plan["total_drive_write_volume_bound_bytes"] == 544 * mib + 64 * 1024
    assert plan["total_local_write_volume_bound_bytes"] == 9 * (200 * mib + 64 * 1024) + 256 * mib
    assert plan["total_validation_write_volume_bound_bytes"] == (
        plan["total_drive_write_volume_bound_bytes"] + plan["total_local_write_volume_bound_bytes"]
    )
    assert plan["report_atomic_headroom_bytes"] == 64 * mib + 64 * 1024
    assert plan["projected_drive_validation_visible_peak_bytes"] == 14 * gib + 544 * mib + 128 * 1024 + 64 * mib
    assert plan["projected_drive_account_usage_peak_bytes"] == 17 * gib + 608 * mib + 128 * 1024
    assert plan["projected_drive_account_usage_peak_includes_trash"] is True
    assert plan["projected_local_validation_peak_bytes"] == 1 * gib - 100 * mib + 1856 * mib
    assert plan["safety_reserve_bytes"] == 512 * mib


def test_gan_resume_storage_plan_rejects_workspace_larger_than_declared_local_peak():
    from revision_r1.validation_workflow import plan_gan_resume_storage

    plan = plan_gan_resume_storage(
        max_checkpoint_payload_bytes=200 * 1024**2,
        available_local_bytes=8 * 1024**3,
        available_drive_bytes=8 * 1024**3,
        existing_workspace_bytes=2 * 1024**3,
    )

    assert plan["local_workspace_within_declared_bound"] is False
    assert plan["local_capacity_passed"] is False
    assert plan["status"] == "INSUFFICIENT"


@pytest.mark.parametrize(
    ("comparison_performed", "is_storage", "expected"),
    [
        (False, True, "NOT_RUN"),
        (False, False, "FAIL"),
        (True, True, "FAIL"),
    ],
)
def test_gan_resume_storage_failure_before_comparison_is_not_mislabeled_as_validation_failure(
    comparison_performed, is_storage, expected,
):
    from revision_r1.validation_workflow import _gan_resume_failure_status

    assert _gan_resume_failure_status(
        comparison_performed=comparison_performed, is_storage=is_storage,
    ) == expected


def test_drive_checkpoint_publication_must_match_readback_hashes_and_generation(tmp_path):
    import hashlib

    from revision_r1.validation_workflow import verify_drive_checkpoint_publication

    payload = tmp_path / "generation-000069.pt"
    manifest = tmp_path / "generation-000069.manifest.json"
    payload.write_bytes(b"persisted checkpoint payload")
    manifest.write_bytes(b'{"generation":69}')
    base = {
        "status": "PERSISTED_AND_READ_BACK_VERIFIED",
        "generation": 69,
        "published_generation_count": 1,
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }

    assert verify_drive_checkpoint_publication(base, manifest_path=manifest, payload_path=payload, generation=69)["status"] == "PASS"

    for changes in (
        {"payload_sha256": "0" * 64},
        {"manifest_sha256": "0" * 64},
        {"generation": 68},
        {"payload_bytes": payload.stat().st_size + 1},
        {"published_generation_count": 2},
    ):
        with pytest.raises(ValueError, match="Drive checkpoint publication"):
            verify_drive_checkpoint_publication(
                {**base, **changes}, manifest_path=manifest,
                payload_path=payload, generation=69,
            )


def test_gan_resume_storage_plan_fails_closed_before_workers_when_drive_is_too_low():
    from revision_r1.validation_workflow import plan_gan_resume_storage

    plan = plan_gan_resume_storage(
        max_checkpoint_payload_bytes=200 * 1024 * 1024,
        available_local_bytes=8 * 1024**3,
        available_drive_bytes=700 * 1024**2,
    )

    assert plan["status"] == "INSUFFICIENT"
    assert plan["drive_capacity_passed"] is False
    assert plan["safety_reserve_bytes"] == 512 * 1024**2
