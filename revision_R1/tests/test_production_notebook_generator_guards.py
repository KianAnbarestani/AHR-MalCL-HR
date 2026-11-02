from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.build_production_handoff import (
    CODE_ZIP_FILENAME,
    DISTRIBUTED_NOTEBOOK_FILENAME,
    PRODUCTION_BUILD_ID,
    notebook_document,
)
from scripts.production_workflow import training_source_sha256


def _code_cell(document: dict, header: str) -> str:
    for cell in document["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.startswith(header):
                return source
    raise AssertionError(f"missing code cell {header!r}")


def test_release_is_a_new_colab_build_without_changing_frozen_protocol_ids():
    assert PRODUCTION_BUILD_ID == "revision_R1_production_v1_4_8_20261102"
    assert CODE_ZIP_FILENAME == "revision_R1_production_v1_4_8_20261102.zip"
    assert DISTRIBUTED_NOTEBOOK_FILENAME == "Revision_R1_Production_Colab_v1_4_8_20261102.ipynb"
    notebook = notebook_document("a" * 64)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212" in source
    assert training_source_sha256(Path(__file__).resolve().parents[1]) == (
        "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
    )


def test_cell_b_defaults_missing_local_dataset_root_and_preserves_custom_value():
    source = _code_cell(notebook_document("a" * 64), "# Cell B")
    fallback = 'DRIVE_PATHS.setdefault("local_dataset_root", "/content/revision_R1_production/datasets")'
    resolve = 'LOCAL_DATASET_ROOT = Path(DRIVE_PATHS["local_dataset_root"]).expanduser()'
    assert fallback in source
    assert resolve in source
    assert source.index(fallback) < source.index("prepare_dataset_roots(")

    namespace = {"DRIVE_PATHS": {}, "Path": Path}
    exec(fallback + "\n" + resolve, namespace)
    assert namespace["LOCAL_DATASET_ROOT"] == Path("/content/revision_R1_production/datasets")

    custom_path = "/content/custom_revision_cache"
    namespace = {"DRIVE_PATHS": {"local_dataset_root": custom_path}, "Path": Path}
    exec(fallback + "\n" + resolve, namespace)
    assert namespace["LOCAL_DATASET_ROOT"] == Path(custom_path)


def test_cell_d_imports_dataset_metadata_before_use():
    source = _code_cell(notebook_document("a" * 64), "# Cell D")
    import_line = "from revision_r1.config import DATASETS"
    assert import_line in source
    assert source.index(import_line) < source.index("DATASETS[_dataset]")


def test_cell_d_dataset_roots_json_print_executes_with_valid_dictionary_syntax():
    source = _code_cell(notebook_document("a" * 64), "# Cell D")
    tree = ast.parse(source)
    report_call = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "json" or node.func.attr != "dumps":
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        keys = {key.value for key in node.args[0].keys if isinstance(key, ast.Constant)}
        if {"roots", "sources"}.issubset(keys):
            report_call = node
            break
    assert report_call is not None, "Cell D must serialize the dataset roots/source report"

    encoded = eval(
        compile(ast.Expression(report_call), "<Cell D dataset-root JSON>", "eval"),
        {
            "json": json,
            "DATA_ROOTS": {"ember": Path("/drive/EMBER"), "az_class": Path("/drive/AZ")},
            "DATASET_SOURCE_REPORT": {"ember": {"source": "DRIVE_ONLY"}},
        },
    )
    report = json.loads(encoded)
    assert report["roots"] == {"ember": "/drive/EMBER", "az_class": "/drive/AZ"}
    assert report["sources"] == {"ember": {"source": "DRIVE_ONLY"}}
