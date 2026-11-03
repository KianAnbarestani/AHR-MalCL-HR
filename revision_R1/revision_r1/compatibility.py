"""Explicit, fail-closed bridge from the frozen checkpoint identity to runtime code."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_IDENTITY_SHA256 = "1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c"
COMPATIBILITY_MANIFEST_NAME = "checkpoint_compatibility.json"
INFRASTRUCTURE_ONLY_CHANGES = {
    "checkpointing.py", "resources.py", "runner.py", "storage.py", "validation_workflow.py",
}
ADDED_INFRASTRUCTURE_MODULES = {"compatibility.py", "drive_storage.py"}


def _stable_ast_dump(node: ast.AST) -> str:
    """Render empty AST fields consistently across supported Python versions."""
    try:
        # Python 3.13+ defaults to omitting empty fields; 3.12 includes them.
        # Pin the newer behavior explicitly so the stored fingerprint is portable.
        return ast.dump(node, annotate_fields=True, include_attributes=False, show_empty=True)
    except TypeError as exc:
        if "show_empty" not in str(exc):
            raise
        return ast.dump(node, annotate_fields=True, include_attributes=False)


def runtime_module_hashes(package_root: str | Path | None = None) -> dict[str, str]:
    root = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
    source = root / "revision_r1"
    if not source.is_dir():
        raise FileNotFoundError(f"runner source directory is missing: {source}")
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("*.py"))
    }


def runtime_source_sha256(package_root: str | Path | None = None) -> str:
    """Hash the actual runtime modules using the historical runner's algorithm."""
    root = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
    source = root / "revision_r1"
    digest = hashlib.sha256()
    modules = sorted(source.glob("*.py"))
    if not modules:
        raise FileNotFoundError(f"no runner source modules under {source}")
    for path in modules:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_behavior_fingerprint(source: str | bytes) -> str:
    """Hash the classifier/GAN/task phase state-machine blocks in ``run_one``."""
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    tree = ast.parse(text)
    run_one = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_one"),
        None,
    )
    if run_one is None:
        raise ValueError("runner source has no run_one function")

    phase_nodes: list[tuple[str, ast.If]] = []
    for node in ast.walk(run_one):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            not isinstance(test.left, ast.Name) or test.left.id != "phase_before"
            or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq)
            or len(test.comparators) != 1 or not isinstance(test.comparators[0], ast.Constant)
            or not isinstance(test.comparators[0].value, str)
        ):
            continue
        phase_nodes.append((test.comparators[0].value, node))
    if not phase_nodes:
        raise ValueError("runner source has no phase_before training branches")
    for _, phase_node in phase_nodes:
        for parent in ast.walk(phase_node):
            for field_name, value in ast.iter_fields(parent):
                if not isinstance(value, list):
                    continue
                setattr(parent, field_name, [
                    child for child in value
                    if not (
                        isinstance(child, ast.Expr)
                        and isinstance(child.value, ast.Call)
                        and isinstance(child.value.func, ast.Name)
                        and child.value.func.id == "checkpoint"
                    )
                ])
    phase_blocks = [
        (phase, _stable_ast_dump(node))
        for phase, node in phase_nodes
    ]
    payload = json.dumps(phase_blocks, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def verify_checkpoint_compatibility(package_root: str | Path | None = None) -> dict[str, Any]:
    """Verify the package's separate runtime hash and legacy checkpoint identity.

    The historical ``1731…`` value is not asserted to be the hash of the new
    files. It is accepted only when this exact runtime matches the committed
    compatibility manifest and its recorded training-behavior fingerprint.
    Build-time tests bind that fingerprint and unchanged modules to the named
    authentic baseline commit.
    """
    root = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
    path = root / COMPATIBILITY_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"checkpoint compatibility manifest is missing or unreadable: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("checkpoint compatibility manifest schema is invalid")
    if manifest.get("baseline_training_source_sha256") != CHECKPOINT_IDENTITY_SHA256:
        raise RuntimeError("checkpoint compatibility manifest names a different historical checkpoint identity")

    observed_modules = runtime_module_hashes(root)
    if manifest.get("runtime_module_sha256") != observed_modules:
        raise RuntimeError("runtime source modules do not match the checkpoint compatibility manifest")
    observed_runtime_hash = runtime_source_sha256(root)
    if manifest.get("runtime_source_sha256") != observed_runtime_hash:
        raise RuntimeError("runtime source digest differs from checkpoint compatibility manifest")
    observed_fingerprint = training_behavior_fingerprint((root / "revision_r1" / "runner.py").read_bytes())
    if (
        manifest.get("training_behavior_fingerprint_sha256") != observed_fingerprint
        or manifest.get("training_behavior_matches_baseline") is not True
    ):
        raise RuntimeError("training behavior differs from the verified checkpoint-compatible baseline")

    baseline_modules = manifest.get("baseline_module_sha256")
    if not isinstance(baseline_modules, dict):
        raise RuntimeError("checkpoint compatibility manifest lacks baseline source-file hashes")
    changed = {
        name for name, baseline_hash in baseline_modules.items()
        if name not in observed_modules or observed_modules[name] != baseline_hash
    }
    added = set(observed_modules) - set(baseline_modules)
    if changed != INFRASTRUCTURE_ONLY_CHANGES or added != ADDED_INFRASTRUCTURE_MODULES:
        raise RuntimeError(
            "runtime source changes exceed the verified checkpoint-only compatibility scope: "
            f"changed={sorted(changed)}, added={sorted(added)}"
        )
    return manifest
