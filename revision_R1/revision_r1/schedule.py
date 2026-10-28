"""Strict loader for the proposed, non-historical classifier update schedule."""

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PoolCounts:
    current: int
    anchor: int
    generated: int

    @property
    def total(self) -> int:
        return self.current + self.anchor + self.generated


@dataclass(frozen=True)
class BudgetRow:
    dataset: str
    seed: int
    task: int
    pools: PoolCounts
    updates: int
    batch_size: int
    epochs_equivalent: int
    provenance: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(row: Mapping[str, str], field: str, line: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid integer {field!r} on schedule line {line}") from error
    if value < 0:
        raise ValueError(f"negative {field!r} on schedule line {line}")
    return value


def load_update_schedule(
    path: str | Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    task_sizes: Sequence[int],
    actual_pool_counts: Mapping[tuple[str, int, int], PoolCounts | Mapping[str, int]],
    *,
    expected_sha256: str | None = None,
) -> dict[tuple[str, int, int], BudgetRow]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"proposed update schedule not found: {path}")
    observed_hash = file_sha256(path)
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise ValueError(f"update schedule SHA-256 mismatch: {observed_hash} != {expected_sha256}")

    required = {
        "dataset", "seed", "task", "current_pool_size", "anchor_pool_size",
        "generated_pool_size", "full_hybrid_pool_size", "batch_size",
        "classifier_epochs_equivalent", "proposed_classifier_updates",
        "budget_provenance", "training_reproduction_claimed",
    }
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"update schedule missing required columns: {missing}")
        rows: dict[tuple[str, int, int], BudgetRow] = {}
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"malformed schedule row on line {line}")
            dataset = raw["dataset"].strip()
            seed = _integer(raw, "seed", line)
            task = _integer(raw, "task", line)
            key = (dataset, seed, task)
            if key in rows:
                raise ValueError(f"duplicate schedule row {key}")
            pools = PoolCounts(
                _integer(raw, "current_pool_size", line),
                _integer(raw, "anchor_pool_size", line),
                _integer(raw, "generated_pool_size", line),
            )
            full_pool = _integer(raw, "full_hybrid_pool_size", line)
            batch_size = _integer(raw, "batch_size", line)
            epochs = _integer(raw, "classifier_epochs_equivalent", line)
            updates = _integer(raw, "proposed_classifier_updates", line)
            provenance = raw["budget_provenance"].strip()
            if not dataset or not provenance:
                raise ValueError(f"empty dataset/provenance on schedule line {line}")
            if raw["training_reproduction_claimed"].strip().upper() not in {"NO", "FALSE", "0"}:
                raise ValueError("the proposed schedule must not claim historical training reproduction")
            if batch_size != 256:
                raise ValueError(f"classifier batch size must be 256 on schedule line {line}")
            if epochs != 3:
                raise ValueError(f"classifier epoch-equivalent count must be 3 on schedule line {line}")
            if full_pool != pools.total:
                raise ValueError(f"full hybrid pool sum mismatch on schedule line {line}")
            expected_updates = epochs * ((full_pool + batch_size - 1) // batch_size)
            if updates != expected_updates:
                raise ValueError(
                    f"update budget mismatch on schedule line {line}: {updates} != {expected_updates}"
                )
            if key not in actual_pool_counts:
                raise ValueError(f"missing actual pool counts for schedule row {key}")
            actual = actual_pool_counts[key]
            if isinstance(actual, PoolCounts):
                expected_pools = actual
            else:
                expected_pools = PoolCounts(
                    int(actual["current"]), int(actual["anchor"]), int(actual["generated"])
                )
            if pools != expected_pools:
                raise ValueError(f"schedule pool counts do not match actual dataset pools for {key}: {pools} != {expected_pools}")
            rows[key] = BudgetRow(dataset, seed, task, pools, updates, batch_size, epochs, provenance)

    task_count = len(task_sizes)
    expected_keys = {
        (str(dataset), int(seed), task)
        for dataset in datasets for seed in seeds for task in range(1, task_count + 1)
    }
    actual_keys = set(rows)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"schedule matrix keys mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if set(actual_pool_counts) != expected_keys:
        raise ValueError("actual pool-count matrix has missing or extra dataset/seed/task rows")
    return rows


def reconcile_update_schedule(
    source_path: str | Path,
    output_dir: str | Path,
    actual_pool_counts: Mapping[tuple[str, int, int], PoolCounts | Mapping[str, int]],
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    task_sizes: Sequence[int],
    anchor_quotas: Mapping[str, int],
    dataset_identities: Mapping[str, Mapping[str, str]],
) -> tuple[Path, Path, dict[str, int]]:
    """Create an immutable, data-derived schedule without changing its source.

    The Phase 1 source schedule sometimes records the nominal per-class anchor
    cap. The runner stores at most the available examples per class, so the
    effective anchor pool can be smaller. Reconcile only that known nominal
    case; current/generated pool differences and unexplained anchor differences
    fail closed.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    with source_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("source update schedule has no header")
        fieldnames = list(reader.fieldnames)
        required = {
            "dataset", "seed", "task", "current_pool_size", "anchor_pool_size",
            "generated_pool_size", "full_hybrid_pool_size", "batch_size",
            "classifier_epochs_equivalent", "proposed_classifier_updates",
            "budget_provenance", "training_reproduction_claimed",
        }
        if not required.issubset(fieldnames):
            raise ValueError(f"source update schedule missing required columns: {sorted(required - set(fieldnames))}")
        source_rows = list(reader)

    extra_fields = (
        "source_anchor_pool_size", "source_full_hybrid_pool_size",
        "source_proposed_classifier_updates", "schedule_reconciliation",
    )
    output_fields = fieldnames + [name for name in extra_fields if name not in fieldnames]
    expected_keys = {
        (str(dataset), int(seed), task)
        for dataset in datasets for seed in seeds for task in range(1, len(task_sizes) + 1)
    }
    if set(actual_pool_counts) != expected_keys:
        raise ValueError("actual pool-count matrix has missing or extra dataset/seed/task rows")

    observed_keys: set[tuple[str, int, int]] = set()
    effective_rows: list[dict[str, str]] = []
    adjusted_rows = 0
    changed_update_rows = 0
    source_anchor_total = 0
    effective_anchor_total = 0
    adjustment_records: list[dict[str, int | str]] = []

    for line, raw in enumerate(source_rows, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"malformed source schedule row on line {line}")
        dataset = raw["dataset"].strip()
        seed = _integer(raw, "seed", line)
        task = _integer(raw, "task", line)
        key = (dataset, seed, task)
        if key in observed_keys:
            raise ValueError(f"duplicate source schedule row {key}")
        observed_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected source schedule row {key}")
        if dataset not in anchor_quotas:
            raise ValueError(f"no anchor quota configured for {dataset!r}")

        current = _integer(raw, "current_pool_size", line)
        source_anchor = _integer(raw, "anchor_pool_size", line)
        generated = _integer(raw, "generated_pool_size", line)
        source_full = _integer(raw, "full_hybrid_pool_size", line)
        batch_size = _integer(raw, "batch_size", line)
        epochs = _integer(raw, "classifier_epochs_equivalent", line)
        source_updates = _integer(raw, "proposed_classifier_updates", line)
        if raw["training_reproduction_claimed"].strip().upper() not in {"NO", "FALSE", "0"}:
            raise ValueError("the source proposed schedule must not claim historical training reproduction")
        if batch_size != 256 or epochs != 3:
            raise ValueError(f"source schedule must retain batch size 256 and three epoch-equivalents on line {line}")
        if source_full != current + source_anchor + generated:
            raise ValueError(f"source full-hybrid pool sum mismatch on line {line}")
        if source_updates != epochs * ((source_full + batch_size - 1) // batch_size):
            raise ValueError(f"source update formula mismatch on line {line}")

        pool = actual_pool_counts[key]
        if not isinstance(pool, PoolCounts):
            pool = PoolCounts(int(pool["current"]), int(pool["anchor"]), int(pool["generated"]))
        if current != pool.current:
            raise ValueError(f"current pool differs from verified data for {key}: {current} != {pool.current}")
        if generated != pool.generated:
            raise ValueError(f"generated pool differs from protocol for {key}: {generated} != {pool.generated}")

        seen_classes = sum(int(value) for value in task_sizes[: task - 1])
        nominal_anchor = int(anchor_quotas[dataset]) * seen_classes
        if source_anchor not in {nominal_anchor, pool.anchor}:
            raise ValueError(
                f"source anchor pool for {key} is neither nominal nor actual: "
                f"{source_anchor} not in {{{nominal_anchor}, {pool.anchor}}}"
            )

        effective_full = pool.total
        effective_updates = epochs * ((effective_full + batch_size - 1) // batch_size)
        row = dict(raw)
        row["source_anchor_pool_size"] = str(source_anchor)
        row["source_full_hybrid_pool_size"] = str(source_full)
        row["source_proposed_classifier_updates"] = str(source_updates)
        row["schedule_reconciliation"] = (
            "NOMINAL_TO_VERIFIED_SUPPORT" if source_anchor != pool.anchor else "UNCHANGED_VERIFIED_POOL"
        )
        row["anchor_pool_size"] = str(pool.anchor)
        row["full_hybrid_pool_size"] = str(effective_full)
        row["proposed_classifier_updates"] = str(effective_updates)
        effective_rows.append(row)
        source_anchor_total += source_anchor
        effective_anchor_total += pool.anchor
        if source_anchor != pool.anchor:
            adjusted_rows += 1
            adjustment_records.append({
                "dataset": dataset, "seed": seed, "task": task,
                "source_anchor_pool_size": source_anchor,
                "effective_anchor_pool_size": pool.anchor,
                "source_classifier_updates": source_updates,
                "effective_classifier_updates": effective_updates,
            })
        if source_updates != effective_updates:
            changed_update_rows += 1

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        raise ValueError(f"source schedule matrix missing rows: {missing[:5]}")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(effective_rows)
    csv_payload = buffer.getvalue().encode("utf-8")
    effective_hash = hashlib.sha256(csv_payload).hexdigest()
    identity_payload = json.dumps(
        {"parent_schedule_sha256": file_sha256(source_path), "dataset_identities": dataset_identities},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    identity_hash = hashlib.sha256(identity_payload).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_path = output_dir / f"effective_update_budget_{identity_hash}_{effective_hash[:12]}.csv"
    manifest_path = effective_path.with_suffix(".manifest.json")

    def write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"refusing to replace a different immutable schedule artifact: {path}")
            return
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    write_immutable(effective_path, csv_payload)
    summary = {
        "adjusted_rows": adjusted_rows,
        "changed_update_rows": changed_update_rows,
        "source_anchor_total": source_anchor_total,
        "effective_anchor_total": effective_anchor_total,
    }
    manifest = {
        "artifact_id": "revision_R1_verified_pool_classifier_update_budget",
        "status": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
        "training_reproduction_claimed": False,
        "row_count": len(effective_rows),
        "batch_size": 256,
        "classifier_epochs_equivalent": 3,
        "formula": "3 * ceil(full_hybrid_pool_size / 256)",
        "anchor_rule": "per-class min(K, verified training support), selected without replacement",
        "source_csv_path": str(effective_path.resolve()),
        "source_csv_sha256": effective_hash,
        "parent_source_csv_path": str(source_path.resolve()),
        "parent_source_csv_sha256": file_sha256(source_path),
        "dataset_identities": dataset_identities,
        "reconciliation": {**summary, "adjustments": adjustment_records},
    }
    manifest_payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_immutable(manifest_path, manifest_payload)
    return effective_path, manifest_path, summary

