import csv
import hashlib
import json
from importlib import import_module
from importlib.util import find_spec
import os
from pathlib import Path
import resource
import tempfile
import zipfile

import numpy as np
import pytest

SCHEDULE_FIELDS = [
    "dataset", "seed", "task", "current_pool_size", "anchor_pool_size",
    "generated_pool_size", "full_hybrid_pool_size", "batch_size",
    "classifier_epochs_equivalent", "proposed_classifier_updates",
    "budget_provenance", "training_reproduction_claimed",
]


def _api(module_name: str, symbol: str):
    assert find_spec(f"revision_r1.{module_name}") is not None, f"missing revision_r1.{module_name}"
    module = import_module(f"revision_r1.{module_name}")
    assert hasattr(module, symbol), f"missing revision_r1.{module_name}.{symbol}"
    return getattr(module, symbol)


def _schedule_file(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(task: int, current: int, anchor: int, generated: int) -> dict:
    total = current + anchor + generated
    return {
        "dataset": "ember", "seed": 42, "task": task,
        "current_pool_size": current, "anchor_pool_size": anchor,
        "generated_pool_size": generated, "full_hybrid_pool_size": total,
        "batch_size": 256, "classifier_epochs_equivalent": 3,
        "proposed_classifier_updates": 3 * ((total + 255) // 256),
        "budget_provenance": "SYNTHETIC_TEST_FIXTURE",
        "training_reproduction_claimed": "NO",
    }


def test_schedule_accepts_exact_proposed_budget_and_actual_pool_counts(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    load_update_schedule = _api("schedule", "load_update_schedule")
    path = _schedule_file(tmp_path / "budget.csv", [_row(1, 300, 0, 0), _row(2, 10, 200, 100)])
    actual = {
        ("ember", 42, 1): PoolCounts(300, 0, 0),
        ("ember", 42, 2): PoolCounts(10, 200, 100),
    }
    rows = load_update_schedule(path, ["ember"], [42], [2, 1], actual)
    assert rows[("ember", 42, 1)].updates == 6
    assert rows[("ember", 42, 2)].updates == 6


def test_schedule_rejects_wrong_update_count_before_training(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    load_update_schedule = _api("schedule", "load_update_schedule")
    rows = [_row(1, 300, 0, 0), _row(2, 10, 200, 100)]
    rows[0]["proposed_classifier_updates"] = 5
    path = _schedule_file(tmp_path / "budget.csv", rows)
    actual = {("ember", 42, 1): PoolCounts(300, 0, 0), ("ember", 42, 2): PoolCounts(10, 200, 100)}
    with pytest.raises(ValueError, match="update budget"):
        load_update_schedule(path, ["ember"], [42], [2, 1], actual)


def test_schedule_rejects_duplicate_rows_and_reconstructed_pool_mismatch(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    load_update_schedule = _api("schedule", "load_update_schedule")
    first = _row(1, 300, 0, 0)
    second = _row(2, 10, 200, 100)
    path = _schedule_file(tmp_path / "duplicate.csv", [first, first, second])
    actual = {("ember", 42, 1): PoolCounts(300, 0, 0), ("ember", 42, 2): PoolCounts(10, 200, 100)}
    with pytest.raises(ValueError, match="duplicate"):
        load_update_schedule(path, ["ember"], [42], [2, 1], actual)
    path = _schedule_file(tmp_path / "wrong-pool.csv", [first, second])
    actual[("ember", 42, 2)] = PoolCounts(11, 200, 100)
    with pytest.raises(ValueError, match="pool counts"):
        load_update_schedule(path, ["ember"], [42], [2, 1], actual)


def test_schedule_rejects_missing_dataset_seed_task_and_hash_mismatch(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    load_update_schedule = _api("schedule", "load_update_schedule")
    path = _schedule_file(tmp_path / "budget.csv", [_row(1, 300, 0, 0)])
    actual = {("ember", 42, 1): PoolCounts(300, 0, 0), ("ember", 42, 2): PoolCounts(10, 200, 100)}
    with pytest.raises(ValueError, match="missing"):
        load_update_schedule(path, ["ember"], [42], [2, 1], actual)
    with pytest.raises(ValueError, match="SHA-256"):
        load_update_schedule(path, ["ember"], [42], [1], {("ember", 42, 1): PoolCounts(300, 0, 0)}, expected_sha256="0" * 64)


def test_packaged_update_schedule_is_exact_phase1_proposed_artifact():
    package_root = Path(__file__).resolve().parents[1]
    schedule_path = package_root / "configs" / "update_budget.csv"
    manifest_path = package_root / "configs" / "update_budget_manifest.json"
    assert schedule_path.is_file(), "missing copied Phase 1 proposed schedule"
    assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == "e719a3a6673edf98d2178a3d3bb666278788b09f26d92689d7aec75bf0f379ed"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 220
    assert manifest["training_reproduction_claimed"] is False
    assert manifest["status"] == "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING"


def test_production_bundle_requires_verified_memory_mapped_arrays(tmp_path):
    from revision_r1.data import DatasetBundle, require_memory_mapped_production_bundle

    arrays = []
    for name, shape, dtype in (
        ("x_train", (8, 3), np.float32), ("y_train", (8,), np.int64),
        ("x_test", (4, 3), np.float32), ("y_test", (4,), np.int64),
    ):
        array = np.lib.format.open_memmap(tmp_path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        array[...] = 0
        arrays.append(array)
    train_meta = {"md5": "a" * 32, "sha256": "b" * 64, "cache_enabled": True}
    test_meta = {"md5": "c" * 32, "sha256": "d" * 64, "cache_enabled": True}
    bundle = DatasetBundle("ember", *arrays, train_meta, test_meta)

    require_memory_mapped_production_bundle(bundle)
    with pytest.raises(ValueError, match="x_train"):
        require_memory_mapped_production_bundle(
            DatasetBundle("ember", np.asarray(arrays[0]), *arrays[1:], train_meta, test_meta)
        )


@pytest.mark.skipif(
    os.environ.get("REVISION_R1_RUN_SCALE_MEMORY_TEST") != "1",
    reason="set REVISION_R1_RUN_SCALE_MEMORY_TEST=1 for the optional ~1.1 GiB streaming/cache stress fixture",
)
def test_scale_memory_streams_full_feature_dimension_and_updates_scaler_in_chunks():
    from revision_r1.data import file_digest, load_npz_arrays
    from revision_r1.preprocessing import RunningScaler

    rows, columns = 120_000, 2_439
    parent = Path(os.environ.get("REVISION_R1_SCALE_TMPDIR", Path(__file__).resolve().parents[1]))
    parent.mkdir(parents=True, exist_ok=True)
    peak_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    with tempfile.TemporaryDirectory(prefix="revision_r1_scale_qa_", dir=parent) as raw_temp:
        root = Path(raw_temp)
        source = root / "scale_fixture.npz"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            with archive.open("X_train.npy", "w") as stream:
                np.lib.format.write_array_header_1_0(stream, {
                    "descr": np.lib.format.dtype_to_descr(np.dtype("float32")),
                    "fortran_order": False, "shape": (rows, columns),
                })
                column_pattern = np.arange(columns, dtype=np.float32) % 31
                for start in range(0, rows, 512):
                    stop = min(rows, start + 512)
                    row_pattern = (np.arange(start, stop, dtype=np.float32) % 17)[:, None]
                    block = row_pattern + column_pattern[None, :]
                    stream.write(block.tobytes(order="C"))
            with archive.open("y_train.npy", "w") as stream:
                np.lib.format.write_array_header_1_0(stream, {
                    "descr": np.lib.format.dtype_to_descr(np.dtype("int64")),
                    "fortran_order": False, "shape": (rows,),
                })
                for start in range(0, rows, 8192):
                    labels = np.arange(start, min(rows, start + 8192), dtype=np.int64) % 100
                    stream.write(labels.tobytes(order="C"))

        x, y, metadata = load_npz_arrays(
            source, split="train", expected_md5=file_digest(source, "md5"),
            expected_shape=(rows, columns), expected_labels=range(100),
            cache_dir=root / "cache",
        )
        assert isinstance(x, np.memmap) and isinstance(y, np.memmap)
        assert x.dtype == np.float32 and y.dtype == np.int64
        scaler = RunningScaler().update_indexed(x, np.arange(rows, dtype=np.int64), chunk_rows=2048)
        assert scaler.n == rows and np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()
        assert metadata["source_dtype"] == "float32"
        peak_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        peak_bytes = max(peak_before, peak_after)
        assert peak_bytes < 5 * 1024**3, f"scale fixture peak RSS exceeded 5 GiB: {peak_bytes} bytes"
        print(json.dumps({
            "rows": rows, "feature_dim": columns,
            "cache_feature_bytes": rows * columns * 4,
            "cache_array_types": [type(x).__name__, type(y).__name__],
            "source_dtype": metadata["source_dtype"], "peak_rss_bytes": peak_bytes,
            "scaler_rows": scaler.n,
            "method": "streamed compressed NPZ to float32 memmap; indexed training-only scaler in 2048-row chunks",
        }, sort_keys=True))


def test_reconcile_nominal_anchor_pool_to_verified_class_support_without_mutating_source(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    reconcile = _api("schedule", "reconcile_update_schedule")
    source_rows = [_row(1, 300, 0, 0), _row(2, 16230, 10000, 5000)]
    for row in source_rows:
        row["dataset"] = "az_class"
    source_path = _schedule_file(tmp_path / "source.csv", source_rows)
    source_bytes = source_path.read_bytes()
    actual = {
        ("az_class", 42, 1): PoolCounts(300, 0, 0),
        ("az_class", 42, 2): PoolCounts(16230, 9984, 5000),
    }

    effective_path, manifest_path, report = reconcile(
        source_path,
        tmp_path / "derived",
        actual,
        datasets=("az_class",), seeds=(42,), task_sizes=(50, 50),
        anchor_quotas={"az_class": 200},
        dataset_identities={"az_class": {"train_sha256": "known-fixture-sha256"}},
    )

    assert source_path.read_bytes() == source_bytes
    with effective_path.open(newline="", encoding="utf-8") as stream:
        effective_rows = list(csv.DictReader(stream))
    adjusted = effective_rows[1]
    assert adjusted["source_anchor_pool_size"] == "10000"
    assert adjusted["anchor_pool_size"] == "9984"
    assert adjusted["full_hybrid_pool_size"] == "31214"
    assert adjusted["source_proposed_classifier_updates"] == "366"
    assert adjusted["proposed_classifier_updates"] == "366"
    assert adjusted["schedule_reconciliation"] == "NOMINAL_TO_VERIFIED_SUPPORT"
    assert report["adjusted_rows"] == 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_hash = hashlib.sha256(effective_path.read_bytes()).hexdigest()
    assert manifest["source_csv_sha256"] == effective_hash
    assert manifest["parent_source_csv_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert manifest["training_reproduction_claimed"] is False
    load_update_schedule = _api("schedule", "load_update_schedule")
    load_update_schedule(
        effective_path, ("az_class",), (42,), (50, 50), actual,
        expected_sha256=effective_hash,
    )


def test_reconcile_rejects_anchor_count_that_is_neither_nominal_nor_verified(tmp_path):
    PoolCounts = _api("schedule", "PoolCounts")
    reconcile = _api("schedule", "reconcile_update_schedule")
    rows = [_row(1, 300, 0, 0), _row(2, 16230, 9999, 5000)]
    for row in rows:
        row["dataset"] = "az_class"
    source_path = _schedule_file(tmp_path / "source.csv", rows)
    actual = {
        ("az_class", 42, 1): PoolCounts(300, 0, 0),
        ("az_class", 42, 2): PoolCounts(16230, 9984, 5000),
    }

    with pytest.raises(ValueError, match="neither nominal nor actual"):
        reconcile(
            source_path, tmp_path / "derived", actual,
            datasets=("az_class",), seeds=(42,), task_sizes=(50, 50),
            anchor_quotas={"az_class": 200},
            dataset_identities={"az_class": {"train_sha256": "known-fixture-sha256"}},
        )


def test_mixed_pool_weights_balance_classes_without_source_first_quotas():
    pool_weights = _api("sampling", "pool_weights")
    expected_source_proportions = _api("sampling", "expected_source_proportions")
    labels = np.array([0, 0, 0, 1, 1])
    sources = np.array(["current", "current", "anchor", "current", "generated"])
    np.testing.assert_allclose(pool_weights(labels), [1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2])
    proportions = expected_source_proportions(labels, sources)
    assert proportions == pytest.approx({"current": 7 / 12, "anchor": 1 / 6, "generated": 1 / 4})
    class_probabilities = np.bincount(labels, weights=pool_weights(labels)) / pool_weights(labels).sum()
    np.testing.assert_allclose(class_probabilities, [0.5, 0.5])


def test_mixed_pool_sampler_returns_full_batches_and_restores_exact_cursor():
    MixedPoolSampler = _api("sampling", "MixedPoolSampler")
    labels = np.array([0, 0, 0, 1, 1])
    sources = np.array(["current", "current", "anchor", "current", "generated"])
    sampler = MixedPoolSampler(labels, sources, np.random.default_rng(91))
    first = sampler.draw_batch(256)
    state = sampler.state_dict()
    expected = sampler.draw_batch(256)
    resumed = MixedPoolSampler.from_state(labels, sources, state)
    actual = resumed.draw_batch(256)
    assert len(first.indices) == 256
    np.testing.assert_array_equal(actual.indices, expected.indices)
    assert resumed.cursor == 512
    assert set(actual.source_counts) <= {"current", "anchor", "generated"}


def test_rng_streams_are_paired_and_gan_advancement_isolated():
    make_rng_streams = _api("rng", "make_rng_streams")
    anchor = make_rng_streams("ember", 42, "anchor_only")
    derpp = make_rng_streams("ember", 42, "derpp_matched")
    full = make_rng_streams("ember", 42, "full_hybrid")
    assert anchor.class_order[:10] == [83, 53, 70, 45, 44, 39, 22, 80, 10, 0]
    assert anchor.classifier_init_seed == derpp.classifier_init_seed == full.classifier_init_seed
    assert anchor.anchor_seed == derpp.anchor_seed == full.anchor_seed
    assert anchor.gan_seed != full.gan_seed
    baseline = make_rng_streams("ember", 42, "anchor_only").sampling.random(32)
    isolated = make_rng_streams("ember", 42, "anchor_only")
    make_rng_streams("ember", 42, "full_hybrid").gan.random(10000)
    np.testing.assert_array_equal(isolated.sampling.random(32), baseline)


def _write_npz(path: Path, x_key="X_train", y_key="y_train", labels=None):
    if labels is None:
        labels = np.array([0, 1, 0], dtype=np.int64)
    x = np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2)
    np.savez(path, **{x_key: x, y_key: labels})
    return hashlib.md5(path.read_bytes()).hexdigest(), x


def test_npz_preflight_verifies_known_hash_shape_keys_and_labels(tmp_path):
    load_npz_arrays = _api("data", "load_npz_arrays")
    path = tmp_path / "small.npz"
    digest, expected_x = _write_npz(path)
    x, y, metadata = load_npz_arrays(
        path, split="train", expected_md5=digest, expected_shape=(3, 2), expected_labels=(0, 1)
    )
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, [0, 1, 0])
    assert metadata["x_key"] == "X_train"
    assert metadata["y_key"] == "y_train"


def test_npz_preflight_fails_closed_for_hash_shape_labels_and_ambiguous_keys(tmp_path):
    load_npz_arrays = _api("data", "load_npz_arrays")
    path = tmp_path / "small.npz"
    digest, _ = _write_npz(path)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        load_npz_arrays(path, split="train", expected_md5="0" * 32, expected_shape=(3, 2), expected_labels=(0, 1))
    with pytest.raises(ValueError, match="shape"):
        load_npz_arrays(path, split="train", expected_md5=digest, expected_shape=(4, 2), expected_labels=(0, 1))
    with pytest.raises(ValueError, match="labels"):
        load_npz_arrays(path, split="train", expected_md5=digest, expected_shape=(3, 2), expected_labels=(0, 2))
    _write_npz(path, labels=np.array([0, 2, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="labels"):
        load_npz_arrays(path, split="train", expected_md5=hashlib.md5(path.read_bytes()).hexdigest(), expected_shape=(3, 2), expected_labels=(0, 1))
    np.savez(path, X_train=np.ones((3, 2)), y_train=np.array([0, 1, 0]), X=np.ones((3, 2)), y=np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="ambiguous"):
        load_npz_arrays(path, split="train", expected_md5=hashlib.md5(path.read_bytes()).hexdigest(), expected_shape=(3, 2), expected_labels=(0, 1))


def test_npz_cache_streams_compressed_members_without_decoding_full_archive(tmp_path, monkeypatch):
    import numpy.lib.npyio

    load_npz_arrays = _api("data", "load_npz_arrays")
    path = tmp_path / "compressed.npz"
    expected_x = (np.arange(48, dtype=np.float64).reshape(12, 4) / 7).astype(np.float32)
    expected_y = np.arange(12, dtype=np.int64) % 3
    np.savez_compressed(path, X_train=expected_x.astype(np.float64), y_train=expected_y)
    digest = hashlib.md5(path.read_bytes()).hexdigest()

    def forbidden_whole_member_decode(*_args, **_kwargs):
        raise AssertionError("compressed NPZ member was decoded as a full in-memory array")

    monkeypatch.setattr(numpy.lib.npyio.NpzFile, "__getitem__", forbidden_whole_member_decode)
    x, y, metadata = load_npz_arrays(
        path, split="train", expected_md5=digest, expected_shape=(12, 4),
        expected_labels=(0, 1, 2), cache_dir=tmp_path / "cache",
    )

    assert isinstance(x, np.memmap)
    assert isinstance(y, np.memmap)
    assert x.dtype == np.float32
    assert y.dtype == np.int64
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)
    assert metadata["source_dtype"] == "float64"
    assert metadata["dtype"] == "float32"

