#!/usr/bin/env python3
"""Prepare a safe MADAR external-baseline adapter without cloning or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VERSION = "v1 - MADAR baseline preparation"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "paper_ready_baselines"
CONFIG_DIR = PROJECT_ROOT / "baseline_suite" / "configs"
ADAPTER_TEMPLATE = PROJECT_ROOT / "baseline_suite" / "madar_adapter_template.py"


def local_madar_candidates() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "MADAR",
        PROJECT_ROOT / "madar",
        PROJECT_ROOT / "external" / "MADAR",
        PROJECT_ROOT / "external" / "madar",
        PROJECT_ROOT / "baseline_suite" / "external" / "MADAR",
        PROJECT_ROOT / "baseline_suite" / "external" / "madar",
    ]
    return [path for path in candidates if path.exists()]


def write_setup_instructions(found: list[Path]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found_text = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in found) if found else "- none"
    lines = [
        "# MADAR Setup Instructions",
        "",
        "MADAR training was not run. No repository was cloned or downloaded.",
        "",
        "## Local Source Check",
        "",
        found_text,
        "",
        "## Required Setup After Approval",
        "",
        "1. Place the approved MADAR source under `baseline_suite/external/MADAR/` or another reviewed local path.",
        "2. Map MADAR data loading to the exact AHR-MalCL-HR class stream for EMBER and AZ-Class.",
        "3. Use seeds `[42, 43, 44, 45, 46]`, random ordering, 50 initial classes, then 5 new classes per task.",
        "4. Use budgets `EMBER K=100` and `AZ-Class K=200` for the main external comparison.",
        "5. Emit `full-result.json`, `final_classification_report.csv`, `final_confusion_matrix.csv`, and `final_per_class_accuracy.csv` per seed.",
        "6. Write all outputs under `result_external_baselines/MADAR/...`, never under `result/`.",
        "",
        "## Adapter Requirements",
        "",
        "- EMBER features: adapter-provided tabular features matching the final HR protocol.",
        "- AZ-Class features: adapter-provided tabular features matching the final HR protocol.",
        "- Class stream: same task split and class order as AHR-MalCL-HR.",
        "- Memory budget: count replay features, labels, model memory, and material distribution-selection state.",
        "- Metrics: mean seen accuracy, final taskwise accuracy, forgetting, macro-F1, weighted-F1, balanced accuracy, old/new-class accuracy, memory MB.",
        "",
        "## Current Readiness",
        "",
        f"- Local MADAR source found: {'YES' if found else 'NO'}",
        "- Ready to run MADAR: NO",
        "",
    ]
    path = OUT_DIR / "madar_setup_instructions.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_adapter_template() -> Path:
    template = '''#!/usr/bin/env python3
"""MADAR adapter template for exact-protocol AHR-MalCL-HR comparisons.

This file is intentionally a template. Do not train from it until the approved
MADAR source is available locally and the adapter has been validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


VERSION = "v1 - MADAR adapter template"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_ahr_malcl_stream(config: dict[str, Any], seed: int) -> Any:
    """Return the exact EMBER/AZ class-incremental stream for MADAR.

    Expected contract:
    - current-task train features/labels per task
    - seen-task test features/labels per task
    - class order and task metadata
    - scaler state matching the final HR protocol
    """
    raise NotImplementedError("Extract and validate the AHR-MalCL-HR data-stream adapter first.")


def run_madar_with_adapter(config: dict[str, Any]) -> None:
    """Call approved MADAR source code using the exact-protocol stream."""
    raise NotImplementedError("Wire this function to approved local MADAR code after source inspection.")


if __name__ == "__main__":
    print(VERSION, flush=True)
    print("Template only: MADAR source integration is required before training.")
'''
    ADAPTER_TEMPLATE.write_text(template, encoding="utf-8")
    return ADAPTER_TEMPLATE


def write_madar_configs() -> list[Path]:
    from baseline_suite.write_config_templates import madar

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "madar_ember_k100.json": madar("ember", 100),
        "madar_az_k200.json": madar("az_class", 200),
    }
    written = []
    for filename, payload in payloads.items():
        path = CONFIG_DIR / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def run(dry_run: bool) -> dict[str, Any]:
    found = local_madar_candidates()
    setup_path = write_setup_instructions(found)
    adapter_path = write_adapter_template()
    config_paths = write_madar_configs()
    return {
        "dry_run": dry_run,
        "local_source_found": bool(found),
        "setup_path": str(setup_path.relative_to(PROJECT_ROOT)),
        "adapter_template": str(adapter_path.relative_to(PROJECT_ROOT)),
        "configs": [str(path.relative_to(PROJECT_ROOT)) for path in config_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MADAR adapter notes and templates.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect local paths and write setup notes only.")
    return parser.parse_args()


def main() -> None:
    print(VERSION, flush=True)
    args = parse_args()
    summary = run(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    if not summary["local_source_found"]:
        print("MADAR source is not local; ready to run MADAR: NO")


if __name__ == "__main__":
    main()
