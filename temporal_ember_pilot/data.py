"""Causal manifest loading and official EMBER v2 feature materialization."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import hashlib

import numpy as np

from .audit import OFFICIAL_ARCHIVE_SHA256, TRAIN_MEMBER_RE, sha256_file
from .stream import manifest_sha256

FEATURE_DIM = 2381
FEATURE_VERSION = 2
EMBER_FEATURE_COMMIT = "d97a0b523de02f3fe5ea6089d080abacab6ee931"


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        manifest = json.load(source)
    recorded = manifest.get("manifest_sha256")
    if not recorded or manifest_sha256(manifest) != recorded:
        raise ValueError("split manifest hash verification failed")
    if manifest.get("window_ids") != [item["id"] for item in manifest.get("windows", [])]:
        raise ValueError("split manifest window list is inconsistent")
    return manifest


def _official_extractor():
    path = Path(__file__).parent / "vendor" / "ember" / "features.py"
    spec = importlib.util.spec_from_file_location("temporal_ember_official_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load vendored official EMBER feature extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from sklearn.feature_extraction import FeatureHasher as SklearnFeatureHasher

    class LegacyFeatureHasher:
        """Retain the legacy EMBER singleton-string hashing behavior on new sklearn."""

        def __init__(self, *args, **kwargs):
            self.input_type = kwargs.get("input_type", "dict")
            self.hasher = SklearnFeatureHasher(*args, **kwargs)

        def transform(self, samples):
            if self.input_type == "string":
                samples = [list(sample) if isinstance(sample, str) else sample for sample in samples]
            return self.hasher.transform(samples)

    # The vendored EMBER source passes SectionInfo.entry as a bare string. Older
    # sklearn iterated that string as characters; newer sklearn rejects it.
    module.FeatureHasher = LegacyFeatureHasher
    extractor = module.PEFeatureExtractor(feature_version=FEATURE_VERSION, print_feature_warning=False)
    if int(extractor.dim) != FEATURE_DIM:
        raise ValueError(f"EMBER v2 extractor dimension changed: expected {FEATURE_DIM}, got {extractor.dim}")
    return extractor


def materialize_feature_store(
    archive_path: str | Path,
    manifest_or_path: dict | str | Path,
    output_dir: str | Path,
    *,
    progress=None,
) -> dict:
    """Vectorize only manifest sample identities from official train shards.

    All feature vectors use the official EMBER PEFeatureExtractor v2. The
    official held-out member is never read, and no model fitting occurs here.
    """
    archive_path = Path(archive_path)
    manifest = load_manifest(manifest_or_path) if isinstance(manifest_or_path, (str, Path)) else manifest_or_path
    if manifest_sha256(manifest) != manifest.get("manifest_sha256"):
        raise ValueError("provided manifest hash verification failed")
    source_sha = sha256_file(archive_path)
    expected_sha = manifest.get("source", {}).get("archive_sha256", OFFICIAL_ARCHIVE_SHA256)
    if source_sha != expected_sha or source_sha != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError("feature archive does not match the audited official EMBER archive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    row_ids_path = output_dir / "row_ids.json"
    metadata_path = output_dir / "feature_store.json"
    if metadata_path.exists() and feature_path.exists() and row_ids_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("manifest_sha256") == manifest["manifest_sha256"] and metadata.get("archive_sha256") == source_sha:
            if (sha256_file(feature_path) == metadata.get("features_sha256")
                    and sha256_file(row_ids_path) == metadata.get("row_ids_sha256")):
                return metadata

    rows = manifest["rows"]
    sample_ids = [row["sha256"] for row in rows]
    row_index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    if len(row_index) != len(sample_ids):
        raise ValueError("split manifest contains duplicate sample identities")
    temporary_features = output_dir / "features.partial.npy"
    try:
        temporary_features.unlink()
    except FileNotFoundError:
        pass
    matrix = np.lib.format.open_memmap(
        temporary_features, mode="w+", dtype=np.float32, shape=(len(sample_ids), FEATURE_DIM)
    )
    written = np.zeros(len(sample_ids), dtype=np.bool_)
    written_count = 0
    extractor = _official_extractor()
    member_count = 0
    raw_count = 0
    try:
        with tarfile.open(archive_path, mode="r|bz2") as archive:
            for member in archive:
                if not TRAIN_MEMBER_RE.search(member.name):
                    continue
                member_count += 1
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read train member {member.name}")
                for raw_line in source:
                    raw_count += 1
                    try:
                        record = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise ValueError(f"malformed JSON at training record {raw_count}") from error
                    sample_id = str(record.get("sha256", "")).lower()
                    position = row_index.get(sample_id)
                    if position is None:
                        continue
                    if written[position]:
                        row = rows[position]
                        if (str(record.get("appeared", ""))[:7] == row["month"]
                                and int(record.get("label", -1)) == 1
                                and str(record.get("avclass", "")).strip() == row["family"]):
                            continue
                        raise ValueError(f"conflicting source identity during vectorization: {sample_id}")
                    vector = np.asarray(extractor.process_raw_features(record), dtype=np.float32)
                    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
                        raise ValueError(f"invalid EMBER feature vector for {sample_id}")
                    matrix[position] = vector
                    written[position] = True
                    written_count += 1
                    if progress and written_count % 25_000 == 0:
                        progress(written_count, len(sample_ids), raw_count, member_count)
        if member_count != 6:
            raise ValueError(f"expected six official train shards, found {member_count}")
        if not written.all():
            missing = int((~written).sum())
            raise ValueError(f"feature materialization missed {missing} manifest identities")
        matrix.flush()
    except BaseException:
        del matrix
        try:
            temporary_features.unlink()
        except FileNotFoundError:
            pass
        raise
    del matrix
    os.replace(temporary_features, feature_path)
    row_ids_path.write_text(json.dumps(sample_ids, separators=(",", ":")) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": source_sha,
        "feature_source": f"elastic/ember/ember/features.py@{EMBER_FEATURE_COMMIT}",
        "feature_source_sha256": sha256_file(Path(__file__).parent / "vendor" / "ember" / "features.py"),
        "feature_version": FEATURE_VERSION,
        "feature_dimension": FEATURE_DIM,
        "rows": len(sample_ids),
        "dtype": "float32",
        "features_file": feature_path.name,
        "features_sha256": sha256_file(feature_path),
        "row_ids_file": row_ids_path.name,
        "row_ids_sha256": sha256_file(row_ids_path),
        "train_shards_read": member_count,
        "source_records_scanned": raw_count,
        "official_test_shard_used": False,
    }
    fd, temporary = tempfile.mkstemp(prefix=".feature-store.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(metadata, target, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, metadata_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return metadata


class FeatureStore:
    def __init__(self, output_dir: str | Path, manifest: dict):
        self.root = Path(output_dir)
        self.metadata = json.loads((self.root / "feature_store.json").read_text(encoding="utf-8"))
        if self.metadata.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError("feature store and frozen manifest do not match")
        if self.metadata.get("feature_dimension") != FEATURE_DIM or self.metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("feature store is not official EMBER v2")
        self.row_ids = json.loads((self.root / self.metadata["row_ids_file"]).read_text(encoding="utf-8"))
        if sha256_file(self.root / self.metadata["row_ids_file"]) != self.metadata.get("row_ids_sha256"):
            raise ValueError("feature store sample-ID checksum mismatch")
        self.row_index = {sample_id: index for index, sample_id in enumerate(self.row_ids)}
        if len(self.row_index) != self.metadata["rows"]:
            raise ValueError("feature store row index contains duplicate sample identities")
        if self.row_ids != [row["sha256"] for row in manifest["rows"]]:
            raise ValueError("feature store sample order does not match the frozen manifest")
        self.features = np.load(self.root / self.metadata["features_file"], mmap_mode="r")
        if self.features.shape != (self.metadata["rows"], FEATURE_DIM):
            raise ValueError("feature store matrix has an invalid shape")
        if sha256_file(self.root / self.metadata["features_file"]) != self.metadata.get("features_sha256"):
            raise ValueError("feature store checksum mismatch")

    def get(self, sample_ids) -> np.ndarray:
        try:
            positions = [self.row_index[str(sample_id)] for sample_id in sample_ids]
        except KeyError as error:
            raise KeyError(f"sample not materialized in feature store: {error.args[0]}") from error
        return np.asarray(self.features[positions], dtype=np.float32)
