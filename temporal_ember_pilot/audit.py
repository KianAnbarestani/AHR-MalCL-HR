"""CPU-only streaming audit for the official EMBER 2018 v2 archive."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile
import time

from .stream import WINDOWS, build_manifest, manifest_sha256

OFFICIAL_ARCHIVE_SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812"
OFFICIAL_ARCHIVE_NAME = "ember_dataset_2018_2.tar.bz2"
OFFICIAL_SOURCE_URL = "https://github.com/elastic/ember"
TRAIN_MEMBER_RE = re.compile(r"(?:^|/)train_features_[0-9]+\.jsonl$")


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_meta(record: dict) -> tuple[str, int | None, str]:
    appeared = record.get("appeared")
    appeared = str(appeared).strip() if appeared is not None else ""
    try:
        label = int(record.get("label"))
    except (TypeError, ValueError):
        label = None
    family = record.get("avclass")
    return appeared, label, str(family).strip() if family is not None else ""


def _family_month(record: dict) -> str | None:
    appeared = record.get("appeared")
    if not isinstance(appeared, str) or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", appeared.strip()):
        return None
    month = appeared.strip()[:7]
    try:
        year, number = (int(value) for value in month.split("-"))
    except ValueError:
        return None
    return month if year == 2018 and 1 <= number <= 10 else None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(path: Path, manifest: dict) -> None:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def audit_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    expected_sha256: str = OFFICIAL_ARCHIVE_SHA256,
    progress=None,
) -> dict:
    """Verify and stream-audit official train shards without extracting the archive."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_sha = sha256_file(archive_path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")

    scan = Counter()
    dates = Counter()
    candidate_records: dict[str, dict] = {}
    min_appeared = None
    max_appeared = None
    train_members: list[str] = []
    started = time.monotonic()
    db_fd, db_name = tempfile.mkstemp(prefix="ember-audit-", suffix=".sqlite", dir=output_dir)
    os.close(db_fd)
    db_path = Path(db_name)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE seen (sample_sha TEXT PRIMARY KEY, meta TEXT NOT NULL, conflict INTEGER NOT NULL DEFAULT 0)")
    batch = []
    try:
        with tarfile.open(archive_path, mode="r|bz2") as archive:
            for member in archive:
                if not TRAIN_MEMBER_RE.search(member.name):
                    continue
                train_members.append(member.name)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read train member {member.name}")
                for raw_line in source:
                    scan["source_rows"] += 1
                    try:
                        record = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        scan["malformed_json_rows"] += 1
                        continue
                    if not isinstance(record, dict):
                        scan["malformed_record_rows"] += 1
                        continue
                    appeared = record.get("appeared")
                    if not isinstance(appeared, str) or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", appeared.strip()):
                        scan["missing_or_malformed_date"] += 1
                    else:
                        month_text = appeared.strip()[:7]
                        dates[month_text] += 1
                        min_appeared = month_text if min_appeared is None or month_text < min_appeared else min_appeared
                        max_appeared = month_text if max_appeared is None or month_text > max_appeared else max_appeared
                    try:
                        label = int(record.get("label"))
                    except (TypeError, ValueError):
                        label = None
                    family_value = record.get("avclass")
                    family = str(family_value).strip() if family_value is not None else ""
                    if label == 1:
                        scan["malware_rows"] += 1
                        if family:
                            scan["family_labeled_malware_rows"] += 1
                        else:
                            scan["malware_without_family"] += 1
                    sample_id = str(record.get("sha256", "")).lower()
                    repeated_or_conflicting = False
                    if not re.fullmatch(r"[0-9a-f]{64}", sample_id):
                        scan["missing_or_malformed_hash"] += 1
                    else:
                        signature = json.dumps(_identity_meta(record), separators=(",", ":"), ensure_ascii=False)
                        old = connection.execute("SELECT meta, conflict FROM seen WHERE sample_sha=?", (sample_id,)).fetchone()
                        if old is None:
                            connection.execute("INSERT INTO seen(sample_sha,meta) VALUES (?,?)", (sample_id, signature))
                            scan["unique_sample_hashes"] += 1
                        elif old[1]:
                            repeated_or_conflicting = True
                        elif old[0] == signature:
                            scan["exact_duplicate_rows"] += 1
                            repeated_or_conflicting = True
                        else:
                            connection.execute("UPDATE seen SET conflict=1 WHERE sample_sha=?", (sample_id,))
                            scan["conflicting_sample_hashes"] += 1
                            candidate_records.pop(sample_id, None)
                            repeated_or_conflicting = True
                    month = _family_month(record)
                    if (not repeated_or_conflicting and label == 1 and family and month is not None
                            and re.fullmatch(r"[0-9a-f]{64}", sample_id)):
                        candidate_records[sample_id] = {
                            "sha256": sample_id,
                            "appeared": month,
                            "label": 1,
                            "avclass": family,
                        }
                    if scan["source_rows"] % 100_000 == 0 and progress:
                        progress(scan["source_rows"], tuple(train_members))
                connection.commit()
        if len(train_members) != 6:
            raise ValueError(f"expected six official train JSONL members, found {len(train_members)}")
        conflict_rows = connection.execute("SELECT sample_sha FROM seen WHERE conflict=1").fetchall()
        for (sample_id,) in conflict_rows:
            candidate_records.pop(sample_id, None)

        manifest = build_manifest(candidate_records.values())
        manifest["source"] = {
            "name": OFFICIAL_ARCHIVE_NAME,
            "url": OFFICIAL_SOURCE_URL,
            "archive_sha256": actual_sha,
            "train_members": sorted(Path(name).name for name in train_members),
            "test_member_used": False,
        }
        manifest["manifest_sha256"] = manifest_sha256(manifest)
        manifest_path = output_dir / "split_manifest.json.gz"
        _write_manifest(manifest_path, manifest)

        family_support = {}
        for window in WINDOWS:
            per_split = manifest["family_support_by_window"][window["id"]]
            family_support[window["id"]] = {
                "months": list(window["months"]),
                "family_labeled_malware": sum(sum(per_split[split].values()) for split in ("train", "heldout")),
                "unique_families": len(set(per_split["train"]) | set(per_split["heldout"])),
                "train_rows": sum(per_split["train"].values()),
                "heldout_rows": sum(per_split["heldout"].values()),
                "training_family_support": dict(sorted(per_split["train"].items())),
                "heldout_family_support": dict(sorted(per_split["heldout"].items())),
            }
        report = {
            "audit_status": "passed",
            "source": manifest["source"],
            "source_sha256": actual_sha,
            "audit_scope": "six official train_features_*.jsonl members only; official test_features.jsonl excluded",
            "date_field": {"name": "appeared", "meaning": "month the sample was first seen, at month precision"},
            "date_coverage": {
                "minimum_month": min_appeared,
                "maximum_month": max_appeared,
                "distinct_months": len(dates),
                "malformed_or_missing_date_rows": scan["missing_or_malformed_date"],
                "monthly_row_counts": dict(sorted(dates.items())),
            },
            "duplicate_policy": {
                "identity": "EMBER sha256 field",
                "exact_repeats": "collapse once",
                "conflicting_date_label_or_family": "quarantine that identity",
                "feature_vector_duplicates": "retain distinct sha256 identities",
                "exact_duplicate_rows": scan["exact_duplicate_rows"],
                "conflicting_sample_hashes": scan["conflicting_sample_hashes"],
            },
            "source_counts": dict(sorted(scan.items())),
            "family_support_by_window": family_support,
            "labeled_malware_total": sum(v["family_labeled_malware"] for v in family_support.values()),
            "distinct_family_labels_in_windows": len({
                family for value in manifest["family_support_by_window"].values()
                for split in value.values() for family in split
            }),
            "chronological_new_to_learner_family_admission": manifest["admissions"],
            "admission_note": "Pilot admission is based only on training-partition support and is not a real-world emergence-date claim.",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file": manifest_path.name,
            "elapsed_audit_seconds": round(time.monotonic() - started, 3),
        }
        _write_json(output_dir / "audit_report.json", report)
        return report
    finally:
        connection.close()
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
