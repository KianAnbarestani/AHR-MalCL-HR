"""Shared dry-run helpers for optional external baseline templates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


VERSION = "v1 - optional baseline helper"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTES_PATH = PROJECT_ROOT / "paper_ready_baselines" / "optional_baseline_implementation_notes.md"


def import_status() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def adapter_status() -> dict[str, Any]:
    try:
        from baseline_suite import data_adapter
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "reason": f"Could not import baseline_suite.data_adapter: {exc}"}
    return data_adapter.adapter_status()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notes(active_name: str, failures: list[str]) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    failure_lines = [f"- {failure}" for failure in failures] if failures else ["- No blocking issue reported by this wrapper."]
    lines = [
        "# Optional Baseline Implementation Notes",
        "",
        "Optional runners were scaffolded for LwF, EWC, and iCaRL. They are secondary to DER++, MADAR, canonical ER, Joint, and the MalCL-faithfulness decision.",
        "",
        "## Current Shared Blockers",
        "",
        "- The exact data-stream adapter is not yet importable from notebook code.",
        "- The `.venv_stats` environment currently lacks the full training stack if `torch` or `sklearn` are missing in dry-run output.",
        "- No optional-baseline training has been run.",
        "",
        "## Baseline Notes",
        "",
        "- LwF: replay-free distillation; count teacher/previous-model checkpoint memory if retained.",
        "- EWC: regularization baseline; count Fisher/importance tensors and previous parameter snapshot memory.",
        "- iCaRL: exemplar/prototype baseline; count exemplars, labels, and prototype state if material.",
        "",
        f"## Last Dry-Run: {active_name}",
        "",
        *failure_lines,
        "",
    ]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_optional(name: str, config_path: Path, dry_run: bool) -> int:
    cfg = load_config(config_path)
    imports = import_status()
    adapter = adapter_status()
    failures: list[str] = []
    missing = [key for key, ok in imports.items() if not ok]
    if missing:
        failures.append("Missing required imports: " + ", ".join(missing))
    if not adapter.get("ready"):
        failures.append("Data adapter is not ready: " + str(adapter.get("reason", "unknown reason")))
    if cfg.get("ready_to_run") is not True:
        failures.append("Config is template-only: " + str(cfg.get("template_only_reason", "not marked ready")))

    exact_command = f"python baseline_suite/run_{name.lower()}_baseline.py --config {config_path}"
    print(f"{name} config loads: YES")
    print("Required imports:", imports)
    print("Data adapter:", adapter)
    print("No training started: YES")
    print("Exact future training command:")
    print(exact_command)
    write_notes(name, failures)

    if dry_run:
        if failures:
            print(f"{name} dry-run failed; wrote {NOTES_PATH.relative_to(PROJECT_ROOT)}")
            return 2
        print(f"{name} dry-run passed.")
        return 0

    print(f"{name} training is not enabled until optional implementation notes are resolved.")
    return 3


if __name__ == "__main__":
    print(VERSION, flush=True)
    print("Import this helper through run_lwf_baseline.py, run_ewc_baseline.py, or run_icarl_baseline.py.")
