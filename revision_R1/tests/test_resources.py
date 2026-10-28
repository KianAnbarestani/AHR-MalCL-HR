from revision_r1.resources import ResourceTracker
import time
import torch
import pytest


def test_classifier_generator_and_critic_costs_are_kept_separate(tmp_path):
    tracker = ResourceTracker()
    tracker.add_steps("classifier", 4)
    tracker.add_steps("generator", 6)
    tracker.add_steps("critic", 30)
    with tracker.measure("classifier"):
        sum(range(1000))
    snapshot = tracker.snapshot(artifact_root=tmp_path)
    assert snapshot["steps"] == {"classifier": 4, "generator": 6, "critic": 30}
    assert snapshot["seconds"]["classifier"] >= 0
    assert snapshot["artifact_storage_bytes"] == 0


def test_invalid_or_negative_step_accounting_is_rejected():
    tracker = ResourceTracker()
    try:
        tracker.add_steps("evaluation", 1)
    except ValueError as exc:
        assert "classifier/generator/critic" in str(exc)
    else:
        raise AssertionError("evaluation was incorrectly counted as an optimizer step")


def test_timing_boundaries_synchronize_and_checkpoint_overhead_is_separate(monkeypatch):
    sync_calls = 0

    def delayed_sync():
        nonlocal sync_calls
        sync_calls += 1
        time.sleep(0.01)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", delayed_sync)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 20)
    tracker = ResourceTracker()

    with tracker.measure("classifier"):
        time.sleep(0.01)
    with tracker.measure("checkpoint"):
        time.sleep(0.005)

    snapshot = tracker.snapshot()
    assert sync_calls == 4
    assert snapshot["seconds"]["classifier"] >= 0.02
    assert snapshot["seconds"]["checkpoint"] >= 0.015
    assert snapshot["timing_metadata"]["checkpoint_overhead_seconds"] == snapshot["seconds"]["checkpoint"]
    assert snapshot["timing_metadata"]["cuda_synchronization_api_seconds_overlaps_component_seconds"] is True


def test_host_memory_snapshot_reads_proc_and_cgroup_without_assuming_colab_layout(tmp_path):
    from revision_r1.resources import read_host_memory_snapshot

    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    proc.mkdir()
    cgroup.mkdir()
    (proc / "meminfo").write_text("MemTotal: 16777216 kB\nMemAvailable: 4194304 kB\n", encoding="utf-8")
    (proc / "self").mkdir()
    (proc / "self" / "status").write_text("VmRSS:\t524288 kB\nVmHWM:\t786432 kB\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(str(8 * 1024**3), encoding="utf-8")
    (cgroup / "memory.current").write_text(str(3 * 1024**3), encoding="utf-8")
    (cgroup / "memory.peak").write_text(str(4 * 1024**3), encoding="utf-8")
    (cgroup / "memory.events").write_text("low 0\nhigh 1\nmax 2\noom 0\noom_kill 0\n", encoding="utf-8")

    snapshot = read_host_memory_snapshot(proc_root=proc, cgroup_root=cgroup)

    assert snapshot["host_available_bytes"] == 4096 * 1024**2
    assert snapshot["process_rss_bytes"] == 512 * 1024**2
    assert snapshot["process_peak_rss_bytes"] == 768 * 1024**2
    assert snapshot["cgroup_limit_bytes"] == 8 * 1024**3
    assert snapshot["cgroup_current_bytes"] == 3 * 1024**3
    assert snapshot["cgroup_peak_bytes"] == 4 * 1024**3
    assert snapshot["cgroup_events"]["oom_kill"] == 0


