"""Frozen, seed-independent chronological split and causal class admission."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Iterable, Mapping

WINDOWS = (
    {"id": "2018-01", "months": ("2018-01",)},
    {"id": "2018-02_03", "months": ("2018-02", "2018-03")},
    {"id": "2018-04_05", "months": ("2018-04", "2018-05")},
    {"id": "2018-06_07", "months": ("2018-06", "2018-07")},
    {"id": "2018-08_09", "months": ("2018-08", "2018-09")},
    {"id": "2018-10", "months": ("2018-10",)},
)
WINDOW_IDS = tuple(window["id"] for window in WINDOWS)
MONTH_TO_WINDOW = {month: i for i, window in enumerate(WINDOWS) for month in window["months"]}
DEFAULT_SPLIT_SALT = "AHR-MalCL-temporal-ember-split-v1:"
DATE_FIELD_MEANING = "appeared is the month the sample was first seen, at month precision."


def _month(value: object) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", value.strip()):
        return None
    result = value.strip()[:7]
    year, month = result.split("-")
    if not 1 <= int(month) <= 12:
        return None
    return result


def _split_for(sample_sha256: str, split_salt: str, heldout_cutoff: int) -> str:
    digest = hashlib.sha256((split_salt + sample_sha256).encode("ascii")).digest()
    return "heldout" if digest[0] < heldout_cutoff else "train"


def _metadata(record: Mapping) -> tuple[str, int | None, str | None]:
    sample_id = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sample_id):
        sample_id = ""
    raw_label = record.get("label")
    try:
        label = int(raw_label) if raw_label is not None else None
    except (TypeError, ValueError):
        label = None
    family = record.get("avclass")
    family = str(family).strip() if family is not None else ""
    return sample_id, label, family or None


def build_manifest(
    records: Iterable[Mapping],
    *,
    split_salt: str = DEFAULT_SPLIT_SALT,
    heldout_cutoff: int = 51,
    initial_family_limit: int = 50,
    admission_family_limit: int = 5,
) -> dict:
    """Build an immutable monthwise manifest from EMBER record metadata.

    Conflicting metadata for one sample identity quarantines that identity. Exact
    repeated identities collapse to one row. The function never uses row order
    or model seed in split or admission decisions.
    """
    if not split_salt or not 0 < heldout_cutoff < 256:
        raise ValueError("split salt and held-out byte cutoff must be valid")
    if initial_family_limit < 1 or admission_family_limit < 1:
        raise ValueError("family limits must be positive")
    observed: dict[str, tuple[tuple, dict]] = {}
    conflicts: set[str] = set()
    counts = Counter()
    for record in records:
        counts["source_records"] += 1
        if not isinstance(record, Mapping):
            counts["malformed_records"] += 1
            continue
        sample_id, label, family = _metadata(record)
        month = _month(record.get("appeared"))
        if not sample_id:
            counts["missing_or_malformed_hash"] += 1
            continue
        signature = (month, label, family)
        if sample_id in conflicts:
            continue
        if sample_id in observed:
            if observed[sample_id][0] == signature:
                counts["exact_duplicates_collapsed"] += 1
            else:
                conflicts.add(sample_id)
                observed.pop(sample_id, None)
                counts["conflicting_identities_quarantined"] += 1
            continue
        if month is None:
            counts["missing_or_malformed_date"] += 1
            continue
        if label != 1:
            counts["non_malware_excluded"] += 1
            continue
        if not family:
            counts["missing_family_excluded"] += 1
            continue
        if month not in MONTH_TO_WINDOW:
            counts["outside_window_excluded"] += 1
            continue
        row = {
            "sha256": sample_id,
            "month": month,
            "window_index": MONTH_TO_WINDOW[month],
            "window_id": WINDOWS[MONTH_TO_WINDOW[month]]["id"],
            "family": family,
            "split": _split_for(sample_id, split_salt, heldout_cutoff),
        }
        observed[sample_id] = (signature, row)

    rows = [entry[1] for entry in observed.values() if entry[1]["sha256"] not in conflicts]
    rows.sort(key=lambda row: (row["window_index"], row["month"], row["family"], row["sha256"]))
    support: dict[str, dict[str, dict[str, int]]] = {
        window["id"]: {"train": {}, "heldout": {}} for window in WINDOWS
    }
    for row in rows:
        values = support[row["window_id"]][row["split"]]
        values[row["family"]] = values.get(row["family"], 0) + 1

    admissions: dict[str, list[str]] = {}
    observed_training_families: set[str] = set()
    family_admission_index: dict[str, int] = {}
    for i, window in enumerate(WINDOWS):
        train_support = support[window["id"]]["train"]
        if i == 0:
            candidates = train_support
            limit = initial_family_limit
        else:
            candidates = {family: n for family, n in train_support.items()
                          if family not in observed_training_families}
            limit = admission_family_limit
        chosen = sorted(candidates, key=lambda family: (-candidates[family], family))[:limit]
        admissions[window["id"]] = chosen
        for family in chosen:
            family_admission_index[family] = i
        # All training labels become observed now, even when not selected; this
        # prevents a family from being backfilled at a later cutoff.
        observed_training_families.update(train_support)

    manifest = {
        "schema_version": 1,
        "date_field": {"name": "appeared", "meaning": DATE_FIELD_MEANING},
        "split_policy": {
            "algorithm": "SHA-256 over fixed salt plus EMBER sha256; first digest byte below cutoff is heldout",
            "salt": split_salt,
            "heldout_first_byte_lt": heldout_cutoff,
            "model_seed_independent": True,
        },
        "windows": [{"id": w["id"], "months": list(w["months"])} for w in WINDOWS],
        "window_ids": list(WINDOW_IDS),
        "rows": rows,
        "family_support_by_window": support,
        "admissions": admissions,
        "family_admission_index": family_admission_index,
        "admitted_families": sorted(family_admission_index, key=lambda f: (family_admission_index[f], f)),
        "audit_counts": dict(sorted(counts.items())),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def manifest_sha256(manifest: Mapping) -> str:
    import json

    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def window_partitions(manifest: Mapping, window_id: str) -> dict:
    """Return only current and causally prior IDs; no future IDs are exposed."""
    window_ids = list(manifest["window_ids"])
    if window_id not in window_ids:
        raise KeyError(f"unknown window {window_id!r}")
    index = window_ids.index(window_id)
    admission_index: Mapping[str, int] = manifest["family_admission_index"]
    known_before = {family for family, admitted_at in admission_index.items() if admitted_at < index}
    known_after = known_before | set(manifest["admissions"][window_id])
    rows = manifest["rows"]
    current_rows = [row for row in rows if row["window_index"] == index]
    train_ids = [row["sha256"] for row in current_rows
                 if row["split"] == "train" and row["family"] in known_after]
    scoring_ids = [row["sha256"] for row in current_rows
                   if row["split"] == "heldout" and row["family"] in known_before]
    if index == 0:
        scoring_ids = [row["sha256"] for row in current_rows
                       if row["split"] == "heldout" and row["family"] in known_after]
    # January's top-50 are the initial cohort, so they have no acquisition
    # metric. Acquisition begins for families admitted at the first update.
    new_previous = set(manifest["admissions"][window_ids[index - 1]]) if index > 1 else set()
    acquisition_ids = [row["sha256"] for row in current_rows
                       if row["split"] == "heldout" and row["family"] in new_previous]
    fit_ids = [row["sha256"] for row in rows
               if row["split"] == "train" and row["window_index"] <= index]
    # Preprocessing is fit on all training-partition rows available through the
    # cutoff. It never sees held-out rows or any later window.
    return {
        "window_id": window_id,
        "window_index": index,
        "known_families_before": sorted(known_before),
        "admitted_families_this_window": list(manifest["admissions"][window_id]),
        "known_families_after": sorted(known_after),
        "train_ids": sorted(train_ids),
        "fit_ids": sorted(fit_ids),
        "scaler_fit_ids": sorted(fit_ids),
        "evaluation_ids": sorted(scoring_ids),
        "scoring_ids": sorted(scoring_ids),
        "acquisition_ids": sorted(acquisition_ids),
        "acquisition_families": sorted(new_previous),
        "planned_current_ids": sorted(train_ids),
    }
