"""Separate classifier, generator, critic, evaluation, memory, and storage accounting."""

from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch


HOST_RAM_RESERVE_BYTES = 1 << 30


def _read_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw in {"", "max"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 uses a huge sentinel for unlimited memory.
    return None if value >= (1 << 60) else value


def _parse_proc_kib(path: Path, keys: tuple[str, ...]) -> dict[str, int | None]:
    result = {key: None for key in keys}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        name, _, rest = line.partition(":")
        if name not in result:
            continue
        fields = rest.split()
        if not fields:
            continue
        try:
            result[name] = int(fields[0]) * 1024
        except ValueError:
            pass
    return result


def _parse_cgroup_events(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return result


def read_host_memory_snapshot(
    *, proc_root: str | Path = "/proc", cgroup_root: str | Path = "/sys/fs/cgroup",
) -> dict:
    """Read bounded host/process and cgroup memory facts without scanning processes."""
    proc = Path(proc_root)
    cgroup = Path(cgroup_root)
    host = _parse_proc_kib(proc / "meminfo", ("MemTotal", "MemAvailable"))
    process = _parse_proc_kib(proc / "self" / "status", ("VmRSS", "VmHWM", "VmSize"))
    if (cgroup / "memory.max").exists():
        limit_path, current_path, peak_path, events_path = (
            cgroup / "memory.max", cgroup / "memory.current", cgroup / "memory.peak", cgroup / "memory.events",
        )
    else:
        legacy = cgroup / "memory"
        limit_path, current_path, peak_path, events_path = (
            legacy / "memory.limit_in_bytes", legacy / "memory.usage_in_bytes",
            legacy / "memory.max_usage_in_bytes", legacy / "memory.failcnt",
        )
    cgroup_current = _read_integer(current_path)
    cgroup_limit = _read_integer(limit_path)
    available_candidates = [value for value in (
        host.get("MemAvailable"),
        None if cgroup_limit is None or cgroup_current is None else max(0, cgroup_limit - cgroup_current),
    ) if value is not None]
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "host_total_bytes": host.get("MemTotal"),
        "host_available_bytes": host.get("MemAvailable"),
        "process_rss_bytes": process.get("VmRSS"),
        "process_peak_rss_bytes": process.get("VmHWM"),
        "process_virtual_bytes": process.get("VmSize"),
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "cgroup_peak_bytes": _read_integer(peak_path),
        "cgroup_available_bytes": None if cgroup_limit is None or cgroup_current is None else max(0, cgroup_limit - cgroup_current),
        "effective_available_bytes": min(available_candidates) if available_candidates else None,
        "cgroup_events": _parse_cgroup_events(events_path),
        "sources": {
            "meminfo": str(proc / "meminfo"), "process_status": str(proc / "self" / "status"),
            "cgroup_limit": str(limit_path), "cgroup_current": str(current_path),
            "cgroup_peak": str(peak_path), "cgroup_events": str(events_path),
        },
    }


def require_host_memory(
    required_bytes: int,
    *,
    reserve_bytes: int = HOST_RAM_RESERVE_BYTES,
    phase: str,
    snapshot: dict | None = None,
) -> dict:
    """Fail before a planned allocation if measured host/cgroup headroom is insufficient."""
    required_bytes, reserve_bytes = int(required_bytes), int(reserve_bytes)
    if required_bytes < 0 or reserve_bytes < 0:
        raise ValueError("host-memory byte counts cannot be negative")
    facts = read_host_memory_snapshot() if snapshot is None else snapshot
    available = facts.get("effective_available_bytes")
    if available is None:
        raise MemoryError(f"cannot verify available host/cgroup RAM before {phase}; refusing the allocation")
    needed = required_bytes + reserve_bytes
    if int(available) < needed:
        raise MemoryError(
            f"insufficient host/cgroup RAM before {phase}: need {needed} bytes "
            f"({required_bytes} planned + {reserve_bytes} reserve), available {int(available)} bytes; "
            "no new model/data allocation was started; preserve checkpoints and restart only after headroom is restored"
        )
    return {"available_bytes": int(available), "required_bytes": required_bytes,
            "reserve_bytes": reserve_bytes, "phase": phase}


class ResourceTracker:
    COMPONENTS = ("classifier", "generator", "critic", "generation", "selection", "evaluation", "data", "checkpoint")

    def __init__(self):
        self.steps = defaultdict(int)
        self.seconds = defaultdict(float)
        self._active_started: dict[str, float] = {}
        self._active_total = 0.0
        self._last_resume: float | None = None
        self.peak_cuda_allocated_bytes = 0
        self.peak_cuda_reserved_bytes = 0
        self.cuda_synchronization_calls = 0
        self.cuda_synchronization_seconds = 0.0
        self.instrumentation_seconds = 0.0

    def resume_active_clock(self) -> None:
        if self._last_resume is None:
            self._last_resume = time.perf_counter()

    def pause_active_clock(self) -> None:
        if self._last_resume is not None:
            self._active_total += time.perf_counter() - self._last_resume
            self._last_resume = None

    @property
    def active_seconds(self) -> float:
        return self._active_total + (time.perf_counter() - self._last_resume if self._last_resume is not None else 0.0)

    @contextlib.contextmanager
    def measure(self, component: str) -> Iterator[None]:
        if component not in self.COMPONENTS:
            raise ValueError(f"unknown resource component {component!r}")
        if component in self._active_started:
            raise RuntimeError(f"nested timing for resource component {component!r}")
        self._active_started[component] = self.start_timer()
        try:
            yield
        finally:
            self.finish_timer(component, self._active_started.pop(component))

    def start_timer(self) -> float:
        """Drain prior CUDA work, then start a synchronized wall-clock boundary."""
        self._synchronize_cuda()
        return time.perf_counter()

    def finish_timer(self, component: str, started: float) -> float:
        """Stop only after queued CUDA work for this component has completed."""
        if component not in self.COMPONENTS:
            raise ValueError(f"unknown resource component {component!r}")
        self._synchronize_cuda()
        elapsed = time.perf_counter() - float(started)
        self.seconds[component] += elapsed
        self._sample_cuda()
        return elapsed

    def _synchronize_cuda(self) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            started = time.perf_counter()
            torch.cuda.synchronize()
            self.cuda_synchronization_seconds += time.perf_counter() - started
            self.cuda_synchronization_calls += 1

    def add_steps(self, component: str, count: int = 1) -> None:
        if component not in ("classifier", "generator", "critic") or count < 0:
            raise ValueError("step counters apply to classifier/generator/critic and must be non-negative")
        self.steps[component] += int(count)

    def state_dict(self) -> dict:
        return {
            "steps": dict(self.steps),
            "seconds": dict(self.seconds),
            "active_seconds": self.active_seconds,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "cuda_synchronization_calls": self.cuda_synchronization_calls,
            "cuda_synchronization_seconds": self.cuda_synchronization_seconds,
            "instrumentation_seconds": self.instrumentation_seconds,
        }

    def load_state_dict(self, state: dict) -> None:
        self.pause_active_clock()
        self.steps.clear()
        self.steps.update({str(k): int(v) for k, v in state.get("steps", {}).items()})
        self.seconds.clear()
        self.seconds.update({str(k): float(v) for k, v in state.get("seconds", {}).items()})
        self._active_total = float(state.get("active_seconds", 0.0))
        self.peak_cuda_allocated_bytes = int(state.get("peak_cuda_allocated_bytes", 0))
        self.peak_cuda_reserved_bytes = int(state.get("peak_cuda_reserved_bytes", 0))
        self.cuda_synchronization_calls = int(state.get("cuda_synchronization_calls", 0))
        self.cuda_synchronization_seconds = float(state.get("cuda_synchronization_seconds", 0.0))
        self.instrumentation_seconds = float(state.get("instrumentation_seconds", 0.0))
        self._last_resume = None

    def _sample_cuda(self) -> None:
        started = time.perf_counter()
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                self.peak_cuda_allocated_bytes = max(self.peak_cuda_allocated_bytes, torch.cuda.max_memory_allocated())
                self.peak_cuda_reserved_bytes = max(self.peak_cuda_reserved_bytes, torch.cuda.max_memory_reserved())
        finally:
            self.instrumentation_seconds += time.perf_counter() - started

    def timing_metadata(self) -> dict:
        return {
            "method": "synchronized perf_counter wall time",
            "classifier_boundary": "forward/loss through optimizer step, synchronized CUDA wall time",
            "optimizer_component_boundary": "after pre-boundary CUDA drain through post-operation CUDA completion",
            "includes_host_launch_and_device_completion": True,
            "checkpoint_overhead_seconds": float(self.seconds.get("checkpoint", 0.0)),
            "resource_instrumentation_seconds": float(self.instrumentation_seconds),
            "resource_instrumentation_scope": "CUDA peak-memory sampling calls only; excludes synchronization, component timers, JSON serialization, and artifact-size scans",
            "cuda_synchronization_calls": int(self.cuda_synchronization_calls),
            "cuda_synchronization_api_seconds": float(self.cuda_synchronization_seconds),
            "cuda_synchronization_api_seconds_overlaps_component_seconds": True,
            "checkpoint_scope": "checkpoint snapshot/capture, serialization, and local/Drive persistence; measured separately from optimizer updates",
        }

    def snapshot(self, *, artifact_root: str | Path | None = None, host_ram_bytes: int | None = None) -> dict:
        storage = None
        if artifact_root is not None:
            storage = sum(path.stat().st_size for path in Path(artifact_root).rglob("*") if path.is_file())
        return {
            "steps": dict(self.steps),
            "seconds": dict(self.seconds),
            "active_seconds": self.active_seconds,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "host_ram_bytes": host_ram_bytes,
            "host_memory": read_host_memory_snapshot(),
            "artifact_storage_bytes": storage,
            "timing_metadata": self.timing_metadata(),
        }
