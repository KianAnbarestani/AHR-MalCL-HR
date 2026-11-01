from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from revision_r1.checkpointing import (
    CheckpointError,
    CheckpointSchedule,
    CheckpointPolicy,
    copy_generation_payload,
    load_latest_valid,
    mirror_checkpoint_generations,
    save_generation,
)
from revision_r1.engine import StopRequested, run_phase_machine
from revision_r1.state import PhaseCursor, RunState
import revision_r1.checkpointing as checkpointing


PHASES = ("classifier", "gan_critic", "generation", "selection", "evaluation", "export")


def _new_state(identity: dict[str, str]) -> RunState:
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
    return RunState(
        identity=identity,
        cursor=PhaseCursor(task_index=0, phase="classifier", item_index=0),
        classifier_state=model.state_dict(),
        classifier_optimizer_state=optimizer.state_dict(),
        rng_state={
            "python": random.Random(8).getstate(),
            "numpy": np.random.default_rng(9).bit_generator.state,
            "torch_cpu": torch.Generator().manual_seed(10).get_state(),
        },
        sampler_state={"cursor": 0, "draws": []},
        metrics={"rows": []},
        resources={"classifier_updates": 0, "generator_updates": 0, "critic_steps": 0},
        extra={"phase_trace": []},
    )


def _handlers():
    def classifier(state):
        model = torch.nn.Linear(2, 2)
        model.load_state_dict(state.classifier_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
        optimizer.load_state_dict(state.classifier_optimizer_state)
        generator = torch.Generator()
        generator.set_state(state.rng_state["torch_cpu"])
        x = torch.rand((4, 2), generator=generator)
        target = torch.randint(0, 2, (4,), generator=generator)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x), target).backward()
        optimizer.step()
        state.classifier_state = model.state_dict()
        state.classifier_optimizer_state = optimizer.state_dict()
        state.rng_state["torch_cpu"] = generator.get_state()
        py_rng = random.Random()
        py_rng.setstate(state.rng_state["python"])
        state.extra.setdefault("python_draws", []).append(py_rng.random())
        state.rng_state["python"] = py_rng.getstate()
        np_rng = np.random.default_rng()
        np_rng.bit_generator.state = state.rng_state["numpy"]
        state.extra.setdefault("numpy_draws", []).append(float(np_rng.random()))
        state.rng_state["numpy"] = np_rng.bit_generator.state
        state.resources["classifier_updates"] += 1
        state.sampler_state["draws"].append(int(state.cursor.item_index * 3 + 2))

    def critic(state):
        state.resources["critic_steps"] += 1
        state.extra["critic_cursor"] = state.cursor.item_index + 1

    def generate(state):
        state.extra.setdefault("generated", []).append(float(state.cursor.item_index) / 7)

    def select(state):
        state.extra["selected"] = sorted(state.extra.get("generated", []))[:2]

    def evaluate(state):
        i = state.cursor.item_index
        state.metrics["rows"].append({"task": 0, "eval_index": i, "accuracy": 0.5 + i / 100})

    def export(state):
        state.extra["exported_rows"] = copy.deepcopy(state.metrics["rows"])

    handlers = dict(zip(PHASES, (classifier, critic, generate, select, evaluate, export)))
    def record_phase(phase, handler):
        def wrapped(state):
            state.extra["phase_trace"].append((phase, state.cursor.item_index))
            handler(state)
        return wrapped
    return {phase: record_phase(phase, handler) for phase, handler in handlers.items()}


def _run(state: RunState, *, stop_hook=None, checkpoint=None):
    return run_phase_machine(
        state,
        _handlers(),
        phase_limits={phase: 2 for phase in PHASES},
        phase_order=PHASES,
        stop_hook=stop_hook,
        checkpoint_hook=checkpoint,
    )


def _run_to_end(state: RunState, checkpoint=None) -> RunState:
    return _run(state, checkpoint=checkpoint)


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_tree_equal(lhs, rhs)
    else:
        assert left == right


def test_phase_interruption_resume_matches_uninterrupted_state(tmp_path):
    identity = {"config": "cfg-a", "data": "data-a", "code": "code-a"}
    baseline = _run_to_end(_new_state(identity))

    for interrupt_after in range(1, 12):
        root = tmp_path / f"resume-{interrupt_after}"
        partial = _new_state(identity)
        steps = 0

        def stop_after_step(_state):
            nonlocal steps
            steps += 1
            return steps == interrupt_after

        with pytest.raises(StopRequested):
            _run(partial, stop_hook=stop_after_step)
        save_generation(root, partial, identity, kind="recovery")
        restored = load_latest_valid(root, identity).state
        resumed = _run_to_end(restored)
        assert resumed.cursor == baseline.cursor
        assert resumed.classifier_state["weight"].equal(baseline.classifier_state["weight"])
        assert resumed.classifier_state["bias"].equal(baseline.classifier_state["bias"])
        _assert_tree_equal(resumed.classifier_optimizer_state, baseline.classifier_optimizer_state)
        assert resumed.sampler_state == baseline.sampler_state
        assert resumed.metrics == baseline.metrics
        assert resumed.resources == baseline.resources
        assert resumed.extra == baseline.extra
        _assert_tree_equal(resumed.rng_state, baseline.rng_state)


def test_corrupt_newest_generation_falls_back_to_previous_verified_one(tmp_path):
    identity = {"run": "same"}
    root = tmp_path / "checkpoints"
    first = _new_state(identity)
    save_generation(root, first, identity, kind="task-boundary")
    second = copy.deepcopy(first)
    second.cursor.item_index = 1
    save_generation(root, second, identity, kind="recovery")
    newest = sorted(root.glob("generation-*.pt"))[-1]
    newest.write_bytes(b"truncated")

    loaded = load_latest_valid(root, identity)
    assert loaded.manifest["generation"] == 1
    assert loaded.state.cursor.item_index == 0


def test_two_invalid_generations_fail_closed(tmp_path):
    identity = {"run": "same"}
    root = tmp_path / "checkpoints"
    save_generation(root, _new_state(identity), identity, kind="recovery")
    save_generation(root, _new_state(identity), identity, kind="recovery")
    for path in root.glob("generation-*.pt"):
        path.write_bytes(b"corrupt")
    with pytest.raises(CheckpointError, match="no valid checkpoint"):
        load_latest_valid(root, identity)


def test_checkpoint_identity_mismatch_never_loads_other_run(tmp_path):
    root = tmp_path / "checkpoints"
    save_generation(root, _new_state({"run": "A"}), {"run": "A"}, kind="recovery")
    with pytest.raises(CheckpointError, match="identity mismatch"):
        load_latest_valid(root, {"run": "B"})


def test_save_serializes_once_and_does_not_deserialize_the_full_state(tmp_path, monkeypatch):
    identity = {"run": "single-serialization"}
    root = tmp_path / "checkpoints"
    save_calls = 0
    original_save = torch.save

    def observed_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return original_save(*args, **kwargs)

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("checkpoint creation must not deserialize the full RunState")

    monkeypatch.setattr(torch, "save", observed_save)
    monkeypatch.setattr(checkpointing, "_torch_load", unexpected_load)
    manifest = save_generation(root, _new_state(identity), identity, kind="recovery")
    assert manifest["generation"] == 1
    assert save_calls == 1
    assert manifest["checkpoint_accounting"]["serialization_seconds"] >= 0
    manifest_path = root / "generation-000001.manifest.json"
    accounting = manifest["checkpoint_accounting"]
    assert accounting["local_manifest_write_bytes"] >= manifest_path.stat().st_size
    assert accounting["local_manifest_write_bytes_since_policy_start"] == accounting["local_manifest_write_bytes"]
    assert accounting["recovery_artifact_bytes_retained_after_pruning"] == sum(
        path.stat().st_size for path in root.glob("generation-*") if path.is_file()
    )


def test_orphan_payload_without_manifest_is_not_published(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "generation-000001.pt").write_bytes(b"partial payload")
    with pytest.raises(CheckpointError, match="no valid checkpoint"):
        load_latest_valid(root, {"run": "A"})


def test_latest_valid_manifest_falls_back_without_deserializing_checkpoint_state(tmp_path, monkeypatch):
    identity = {"run": "manifest-fallback"}
    root = tmp_path / "recovery"
    older = save_generation(root, _new_state(identity), identity, kind="recovery")
    newer_state = _new_state(identity)
    newer_state.cursor.item_index = 1
    newer = save_generation(root, newer_state, identity, kind="recovery")
    (root / newer["payload_file"]).write_bytes(b"tampered")
    monkeypatch.setattr(checkpointing, "_torch_load", lambda *_a, **_k: pytest.fail("manifest selection must not deserialize state"))

    selected = checkpointing.latest_valid_manifest(root, identity)

    assert selected["generation"] == older["generation"]


def test_checkpoint_mirror_copies_and_reverifies_two_generations(tmp_path):
    identity = {"run": "mirror"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    for n in range(3):
        state = _new_state(identity)
        state.cursor.item_index = n
        save_generation(local, state, identity, kind="recovery")
    assert len(list(local.glob("generation-*.manifest.json"))) == 2
    copied = mirror_checkpoint_generations(local, drive, identity, keep=2)
    assert len(copied) == 2
    assert load_latest_valid(drive, identity).manifest["generation"] == 3


def test_repeated_checkpoint_mirror_does_not_rewrite_existing_generations(tmp_path, monkeypatch):
    identity = {"run": "idempotent-mirror"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    for n in range(2):
        state = _new_state(identity)
        state.cursor.item_index = n
        save_generation(local, state, identity, kind="recovery")
    first = mirror_checkpoint_generations(local, drive, identity, keep=2)
    assert len(first) == 2

    def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("an already verified generation must not be uploaded again")

    monkeypatch.setattr(checkpointing.shutil, "copyfileobj", unexpected_copy)
    repeated = mirror_checkpoint_generations(local, drive, identity, keep=2)
    assert repeated == []


def test_mirror_rehashes_an_existing_same_size_destination_before_skipping(tmp_path):
    identity = {"run": "idempotent-mirror-detects-corruption"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    manifest = save_generation(local, _new_state(identity), identity, kind="recovery")
    mirror_checkpoint_generations(local, drive, identity, keep=2)
    destination_payload = drive / manifest["payload_file"]
    payload = bytearray(destination_payload.read_bytes())
    payload[-1] ^= 1
    destination_payload.write_bytes(payload)

    repaired = mirror_checkpoint_generations(local, drive, identity, keep=2, generation=manifest["generation"])

    assert [item["generation"] for item in repaired] == [manifest["generation"]]
    assert load_latest_valid(drive, identity).manifest["payload_sha256"] == manifest["payload_sha256"]


def test_newly_serialized_payload_is_not_rehashed_before_drive_copy(tmp_path, monkeypatch):
    identity = {"run": "preverified-publication"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    calls = []
    original_sha256 = checkpointing._sha256

    def observed_hash(path):
        calls.append(Path(path).resolve())
        return original_sha256(Path(path))

    monkeypatch.setattr(checkpointing, "_sha256", observed_hash)
    manifest = save_generation(local, _new_state(identity), identity, kind="recovery")
    assert len(calls) == 1  # the serialization output is hashed once

    copied = mirror_checkpoint_generations(
        local, drive, identity, keep=2, generation=manifest["generation"],
        preverified_manifests=(manifest,),
    )

    assert [item["generation"] for item in copied] == [manifest["generation"]]
    assert calls.count((local / manifest["payload_file"]).resolve()) == 0
    assert calls.count((drive / manifest["payload_file"]).resolve()) == 1


def test_task_boundary_generation_continues_remote_archive_without_copying_it_locally(tmp_path):
    identity = {"run": "boundary-sequence"}
    recovery, remote_boundaries = tmp_path / "recovery", tmp_path / "drive-boundaries"
    first_state = _new_state(identity)
    first_state.cursor.task_index = 0
    first = save_generation(recovery, first_state, identity, kind="recovery")
    boundary_one = copy_generation_payload(
        recovery, first, remote_boundaries, identity, kind="task-boundary", generation=1,
    )
    first_hash = (remote_boundaries / boundary_one["payload_file"]).read_bytes()

    second_state = _new_state(identity)
    second_state.cursor.task_index = 1
    second = save_generation(recovery, second_state, identity, kind="recovery")
    # This mirrors the fresh-runtime case: no previous boundary payloads were
    # downloaded to the local staging directory.
    source_root = tmp_path / "empty-local-boundaries"
    copied_boundary = copy_generation_payload(
        recovery, second, source_root, identity, kind="task-boundary", generation=2,
    )
    assert copied_boundary["checkpoint_accounting"]["local_manifest_write_bytes"] == (
        source_root / "generation-000002.manifest.json"
    ).stat().st_size
    mirrored = mirror_checkpoint_generations(
        source_root, remote_boundaries, identity,
        keep=None, prune_destination=False, generation=copied_boundary["generation"],
    )

    assert [item["generation"] for item in mirrored] == [2]
    assert (remote_boundaries / "generation-000001.pt").read_bytes() == first_hash
    assert (remote_boundaries / "generation-000002.pt").is_file()


def test_incremental_drive_publish_copies_one_new_generation_then_prunes_exact_old_files(tmp_path):
    identity = {"run": "incremental-publish"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    for item_index in range(2):
        state = _new_state(identity)
        state.cursor.item_index = item_index
        save_generation(local, state, identity, kind="recovery")
    mirror_checkpoint_generations(local, drive, identity, keep=2)

    state = _new_state(identity)
    state.cursor.item_index = 2
    new_manifest = save_generation(local, state, identity, kind="recovery")
    deleted = []
    published = mirror_checkpoint_generations(
        local, drive, identity, keep=2, generation=new_manifest["generation"],
        delete_callback=lambda path: (deleted.append(path.name), path.unlink()),
    )

    assert [item["generation"] for item in published] == [3]
    assert published.bytes_written < int(new_manifest["payload_bytes"]) + 4096
    assert set(deleted) == {"generation-000001.pt", "generation-000001.manifest.json"}
    assert [
        json.loads(path.read_text())["generation"]
        for path in sorted(drive.glob("generation-*.manifest.json"))
    ] == [2, 3]
    assert load_latest_valid(drive, identity).manifest["generation"] == 3


def test_failed_drive_copy_preserves_existing_verified_recovery_and_skips_pruning(tmp_path, monkeypatch):
    identity = {"run": "failed-incremental-publish"}
    local, drive = tmp_path / "local", tmp_path / "drive"
    first = save_generation(local, _new_state(identity), identity, kind="recovery")
    mirror_checkpoint_generations(local, drive, identity, keep=2)
    before = load_latest_valid(drive, identity).manifest["payload_sha256"]
    next_state = _new_state(identity)
    next_state.cursor.item_index = 1
    newest = save_generation(local, next_state, identity, kind="recovery")
    deleted = []

    def fail_copy(*_args, **_kwargs):
        raise OSError("simulated destination-full write")

    monkeypatch.setattr(checkpointing.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="destination-full"):
        mirror_checkpoint_generations(
            local, drive, identity, keep=2, generation=newest["generation"],
            delete_callback=lambda path: deleted.append(path),
        )

    assert not deleted
    assert load_latest_valid(drive, identity).manifest["generation"] == first["generation"]
    assert load_latest_valid(drive, identity).manifest["payload_sha256"] == before


def test_task_boundary_checkpoint_generations_are_not_pruned(tmp_path):
    identity = {"run": "boundaries"}
    root = tmp_path / "task-boundaries"
    for task in range(3):
        state = _new_state(identity)
        state.cursor.task_index = task
        save_generation(root, state, identity, kind="task-boundary", retain_generations=None)
    assert len(list(root.glob("generation-*.manifest.json"))) == 3
    copied = mirror_checkpoint_generations(root, tmp_path / "drive-boundaries", identity, keep=None)
    assert len(copied) == 3


def test_synthetic_task_boundary_checkpoint_generations_are_capped_at_two(tmp_path):
    identity = {"run": "synthetic-boundaries", "execution_scope": "synthetic_validation"}
    root = tmp_path / "synthetic-task-boundaries"
    for task in range(5):
        state = _new_state(identity)
        state.cursor.task_index = task
        save_generation(root, state, identity, kind="task-boundary", retain_generations=None)

    manifests = sorted(root.glob("generation-*.manifest.json"))
    assert len(manifests) == 2
    assert [json.loads(path.read_text())["generation"] for path in manifests] == [4, 5]


def test_synthetic_drive_mirror_prunes_destination_even_if_caller_disables_pruning(tmp_path, monkeypatch):
    import revision_r1.checkpointing as checkpointing

    identity = {"run": "synthetic-mirror", "execution_scope": "synthetic_validation"}
    source, drive = tmp_path / "source", tmp_path / "drive"
    unbounded_policy = checkpointing.effective_generation_limit
    monkeypatch.setattr(checkpointing, "effective_generation_limit", lambda _identity, requested: requested)
    for index in range(5):
        state = _new_state(identity)
        state.cursor.item_index = index
        save_generation(source, state, identity, kind="task-boundary", retain_generations=None)

    monkeypatch.setattr(checkpointing, "effective_generation_limit", unbounded_policy)
    copied = mirror_checkpoint_generations(source, drive, identity, keep=None, prune_destination=False)

    assert [record["generation"] for record in copied] == [4, 5]
    manifests = sorted(drive.glob("generation-*.manifest.json"))
    assert [json.loads(path.read_text())["generation"] for path in manifests] == [4, 5]


def test_failed_capacity_check_preserves_previous_verified_recovery(tmp_path, monkeypatch):
    import revision_r1.checkpointing as checkpointing
    from revision_r1.storage import StorageCapacityError

    identity = {"run": "capacity-guard", "execution_scope": "synthetic_validation"}
    root = tmp_path / "capacity-guard"
    save_generation(root, _new_state(identity), identity, kind="initial")
    before = load_latest_valid(root, identity).manifest["payload_sha256"]

    monkeypatch.setattr(
        checkpointing, "require_path_capacity",
        lambda *args, **kwargs: (_ for _ in ()).throw(StorageCapacityError("test low space")),
    )
    with pytest.raises(StorageCapacityError, match="test low space"):
        save_generation(root, _new_state(identity), identity, kind="recovery")

    assert len(list(root.glob("generation-*.manifest.json"))) == 1
    assert load_latest_valid(root, identity).manifest["payload_sha256"] == before


def test_checkpoint_schedule_is_completion_based_and_update_independent():
    clock = [0.0]
    schedule = CheckpointSchedule(local_interval_seconds=300, drive_interval_seconds=900, clock=lambda: clock[0])
    metadata = schedule.metadata(legacy_config_fields={
        "checkpoint_interval_updates": 25,
        "checkpoint_interval_seconds": 60,
        "drive_sync_interval_seconds": 300,
    })
    assert metadata["local_recovery_interval_seconds"] == 300
    assert metadata["drive_publish_interval_seconds"] == 900
    assert metadata["deadline_anchor"] == "previous_checkpoint_cycle_completion"
    assert metadata["legacy_config_fields_preserved_for_run_identity"]["checkpoint_interval_seconds"] == 60
    assert metadata["runtime_policy_overrides_legacy_operational_fields"] is True
    assert not schedule.local_due()
    assert not schedule.drive_due()

    # A checkpoint cycle becomes due, then its serialization and Drive publish
    # take 12 minutes. The cycle completion resets both deadlines.
    clock[0] = 900.0
    assert schedule.local_due()
    assert schedule.drive_due()
    clock[0] = 1620.0
    schedule.mark_local_complete()
    schedule.mark_drive_complete()
    assert not schedule.local_due()
    assert not schedule.drive_due()

    clock[0] = 1919.0
    assert not schedule.local_due()
    assert not schedule.drive_due()
    clock[0] = 1920.0
    assert schedule.local_due()
    assert not schedule.drive_due()
    clock[0] = 2520.0
    assert schedule.drive_due()


def test_legacy_update_count_does_not_trigger_a_checkpoint():
    policy = CheckpointPolicy(update_interval=1, seconds_interval=300)
    assert not policy.should_checkpoint(
        last_checkpoint_update=0, update=10_000,
        last_checkpoint_time=10.0, now=309.9,
    )
    assert policy.should_checkpoint(
        last_checkpoint_update=0, update=10_000,
        last_checkpoint_time=10.0, now=310.0,
    )
