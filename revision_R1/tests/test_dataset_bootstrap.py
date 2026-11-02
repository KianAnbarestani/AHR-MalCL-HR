from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from scripts.dataset_bootstrap import ZENODO_DOWNLOADS, prepare_dataset_roots
from scripts.build_production_handoff import notebook_document


def _fixture(tmp_path: Path):
    payloads = {
        "train.npz": b"verified training data",
        "test.npz": b"verified test data",
    }
    specs = {
        "ember": {
            "train": {"filename": "train.npz", "md5": hashlib.md5(payloads["train.npz"]).hexdigest()},
            "test": {"filename": "test.npz", "md5": hashlib.md5(payloads["test.npz"]).hexdigest()},
        }
    }
    preflight = {
        "ember": {
            split: {
                "md5": specs["ember"][split]["md5"],
                "sha256": hashlib.sha256(payloads[specs["ember"][split]["filename"]]).hexdigest(),
            }
            for split in ("train", "test")
        }
    }
    catalog = {
        filename: {
            "url": f"https://zenodo.org/api/records/14537891/files/{filename}/content",
            "size_bytes": len(payload),
        }
        for filename, payload in payloads.items()
    }
    drive_root = tmp_path / "drive" / "datasets"
    drive_root.mkdir(parents=True)
    return payloads, specs, preflight, catalog, drive_root


class _Response:
    status = 200

    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_complete_drive_dataset_is_used_without_network_or_local_copy(tmp_path):
    payloads, specs, preflight, catalog, drive_root = _fixture(tmp_path)
    for filename, payload in payloads.items():
        (drive_root / filename).write_bytes(payload)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("complete Drive datasets must not be downloaded")

    roots, report = prepare_dataset_roots(
        {"ember": drive_root}, tmp_path / "local", preflight,
        dataset_specs=specs, download_catalog=catalog, urlopen=unexpected_network,
    )

    assert roots["ember"] == drive_root
    assert report["ember"]["source"] == "DRIVE_ONLY"
    assert report["ember"]["downloaded_files"] == []
    assert not (tmp_path / "local").exists()


def test_missing_drive_file_downloads_locally_and_preserves_existing_drive_file(tmp_path):
    payloads, specs, preflight, catalog, drive_root = _fixture(tmp_path)
    existing_drive_file = drive_root / "train.npz"
    existing_drive_file.write_bytes(payloads["train.npz"])
    original_drive_bytes = existing_drive_file.read_bytes()
    opened_urls = []

    def open_download(request, timeout):
        assert timeout > 0
        opened_urls.append(request.full_url)
        return _Response(payloads["test.npz"])

    roots, report = prepare_dataset_roots(
        {"ember": drive_root}, tmp_path / "local", preflight,
        dataset_specs=specs, download_catalog=catalog, urlopen=open_download,
        reserve_bytes=0,
    )

    local_train = roots["ember"] / "train.npz"
    local_test = roots["ember"] / "test.npz"
    assert local_train.is_symlink()
    assert local_train.resolve() == existing_drive_file.resolve()
    assert local_test.read_bytes() == payloads["test.npz"]
    assert opened_urls == [catalog["test.npz"]["url"]]
    assert existing_drive_file.read_bytes() == original_drive_bytes
    assert report["ember"]["source"] == "MIXED_DRIVE_AND_ZENODO_LOCAL"
    assert report["ember"]["downloaded_files"] == ["test.npz"]


def test_download_checksum_mismatch_never_publishes_final_file(tmp_path):
    payloads, specs, preflight, catalog, drive_root = _fixture(tmp_path)
    (drive_root / "train.npz").write_bytes(payloads["train.npz"])
    catalog["test.npz"]["size_bytes"] = len(payloads["test.npz"])
    preflight["ember"]["test"]["sha256"] = "0" * 64

    def open_download(_request, timeout):
        return _Response(payloads["test.npz"])

    with pytest.raises(ValueError, match="SHA-256"):
        prepare_dataset_roots(
            {"ember": drive_root}, tmp_path / "local", preflight,
            dataset_specs=specs, download_catalog=catalog, urlopen=open_download,
            reserve_bytes=0,
        )

    assert not (tmp_path / "local" / "ember" / "test.npz").exists()
    assert not list((tmp_path / "local" / "ember").glob("*.part"))


def test_insufficient_local_space_stops_before_network_or_drive_writes(tmp_path, monkeypatch):
    payloads, specs, preflight, catalog, drive_root = _fixture(tmp_path)
    drive_file = drive_root / "train.npz"
    drive_file.write_bytes(payloads["train.npz"])
    from scripts import dataset_bootstrap

    class DiskUsage:
        free = len(payloads["test.npz"]) - 1

    monkeypatch.setattr(dataset_bootstrap.shutil, "disk_usage", lambda _path: DiskUsage())

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("capacity failure must happen before any network write")

    with pytest.raises(OSError, match="space|reserve|capacity|available"):
        prepare_dataset_roots(
            {"ember": drive_root}, tmp_path / "local", preflight,
            dataset_specs=specs, download_catalog=catalog, urlopen=unexpected_network,
            reserve_bytes=0,
        )

    assert drive_file.read_bytes() == payloads["train.npz"]


def test_capacity_for_all_missing_datasets_is_checked_before_first_download(tmp_path, monkeypatch):
    payloads, one_spec, one_preflight, catalog, drive_root = _fixture(tmp_path)
    az_drive_root = tmp_path / "drive" / "az"
    az_drive_root.mkdir()
    preflight = {"ember": one_preflight["ember"], "az_class": one_preflight["ember"]}
    specs = {"ember": one_spec["ember"], "az_class": one_spec["ember"]}
    required_bytes = sum(len(payload) for payload in payloads.values()) * 2
    download_attempts = []

    class DiskUsage:
        free = required_bytes - 1

    from scripts import dataset_bootstrap

    monkeypatch.setattr(dataset_bootstrap.shutil, "disk_usage", lambda _path: DiskUsage())

    def unexpected_network(request, timeout):
        download_attempts.append(request.full_url)
        return _Response(payloads["train.npz"])

    with pytest.raises(OSError, match="insufficient local Colab storage"):
        prepare_dataset_roots(
            {"ember": drive_root, "az_class": az_drive_root}, tmp_path / "local", preflight,
            dataset_specs=specs, download_catalog=catalog, urlopen=unexpected_network,
            reserve_bytes=0,
        )

    assert download_attempts == []
    assert not (tmp_path / "local" / "ember" / "train.npz").exists()
    assert not (tmp_path / "local" / "az_class" / "train.npz").exists()


def test_production_notebook_bootstraps_data_before_storage_and_reports_it_in_cell_d():
    document = notebook_document("a" * 64)
    code_cells = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    ]
    cell_b = next(source for source in code_cells if source.startswith("# Cell B"))
    cell_c = next(source for source in code_cells if source.startswith("# Cell C"))
    cell_d = next(source for source in code_cells if source.startswith("# Cell D"))

    assert "prepare_dataset_roots(" in cell_b
    assert "DATA_ROOTS, DATASET_SOURCE_REPORT" in cell_b
    assert "local_dataset_root" in cell_b
    assert "DATA_ROOTS = {\n    \"ember\": Path(DATA_ROOTS[\"ember\"])" in cell_c
    assert "DATA_ROOTS[_dataset]" in cell_d
    assert '"dataset_source_report": DATASET_SOURCE_REPORT' in cell_d


def test_pinned_zenodo_file_metadata_matches_frozen_dataset_names_and_sizes():
    assert ZENODO_DOWNLOADS == {
        "EMBER_Class_train.npz": {
            "url": "https://zenodo.org/api/records/14537891/files/EMBER_Class_train.npz/content",
            "size_bytes": 2_890_138_390,
        },
        "EMBER_Class_test.npz": {
            "url": "https://zenodo.org/api/records/14537891/files/EMBER_Class_test.npz/content",
            "size_bytes": 321_132_222,
        },
        "AZ_Class_Train.npz": {
            "url": "https://zenodo.org/api/records/14537891/files/AZ_Class_Train.npz/content",
            "size_bytes": 2_523_966_748,
        },
        "AZ_Class_Test.npz": {
            "url": "https://zenodo.org/api/records/14537891/files/AZ_Class_Test.npz/content",
            "size_bytes": 280_450_144,
        },
    }

