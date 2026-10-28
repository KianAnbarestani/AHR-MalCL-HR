"""Command-line gates for data preflight, run execution, and artifact validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import DATASETS, SEEDS, TASK_SIZES, make_run_config
from .data import preflight_dataset
from .exports import write_json
from .schedule import PoolCounts, file_sha256, load_update_schedule, reconcile_update_schedule
from .task_stream import class_order, split_order
from .validation import validate_run_artifacts


def build_actual_pool_matrix(ember_bundle, az_bundle) -> dict[tuple[str, int, int], PoolCounts]:
    """Derive all proposed-budget pool counts from validated train labels."""
    actual = {}
    for dataset, bundle in (("ember", ember_bundle), ("az_class", az_bundle)):
        counts = np.bincount(bundle.y_train, minlength=100)
        quota = int(DATASETS[dataset]["anchor_k"])
        for seed in SEEDS:
            seen: list[int] = []
            groups = split_order(class_order(seed), TASK_SIZES)
            for task_id, group in enumerate(groups, start=1):
                current = int(sum(counts[class_id] for class_id in group))
                anchors = int(sum(min(quota, int(counts[class_id])) for class_id in seen))
                generated = len(seen) * 100
                actual[(dataset, seed, task_id)] = PoolCounts(current, anchors, generated)
                seen.extend(group)
    return actual


def _verified_schedule_hash(path: str | Path, manifest_path: str | Path) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("training_reproduction_claimed") is not False:
        raise ValueError("update-budget manifest must explicitly disclaim historical training reproduction")
    if manifest.get("status") != "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING":
        raise ValueError("update-budget manifest has an unexpected provenance status")
    if manifest.get("row_count") != 220:
        raise ValueError("update-budget manifest must describe all 220 proposed rows")
    observed = file_sha256(path)
    expected = manifest.get("source_csv_sha256")
    if not expected or observed != expected:
        raise ValueError(f"proposed update schedule SHA-256 mismatch: {observed} != {expected}")
    return observed


def preflight_all(args) -> dict:
    ember = preflight_dataset(args.ember_data_root, "ember", cache_dir=args.cache_root)
    az_class = preflight_dataset(args.az_data_root, "az_class", cache_dir=args.cache_root)
    source_schedule_path = Path(args.schedule)
    source_schedule_hash = _verified_schedule_hash(source_schedule_path, args.schedule_manifest)
    actual_pools = build_actual_pool_matrix(ember, az_class)
    dataset_identities = {
        "ember": {
            "train_md5": ember.train_metadata["md5"],
            "train_sha256": ember.train_metadata["sha256"],
            "test_md5": ember.test_metadata["md5"],
            "test_sha256": ember.test_metadata["sha256"],
        },
        "az_class": {
            "train_md5": az_class.train_metadata["md5"],
            "train_sha256": az_class.train_metadata["sha256"],
            "test_md5": az_class.test_metadata["md5"],
            "test_sha256": az_class.test_metadata["sha256"],
        },
    }
    effective_schedule_path, effective_manifest_path, reconciliation = reconcile_update_schedule(
        source_schedule_path,
        Path(args.output).parent / "schedules",
        actual_pools,
        datasets=("ember", "az_class"), seeds=SEEDS, task_sizes=TASK_SIZES,
        anchor_quotas={dataset: int(spec["anchor_k"]) for dataset, spec in DATASETS.items()},
        dataset_identities=dataset_identities,
    )
    effective_hash = _verified_schedule_hash(effective_schedule_path, effective_manifest_path)
    rows = load_update_schedule(
        effective_schedule_path, ("ember", "az_class"), SEEDS, TASK_SIZES,
        actual_pools, expected_sha256=effective_hash,
    )
    result = {
        "status": "PREFLIGHT_VALIDATED_NO_TRAINING",
        "training_reproduction_claimed": False,
        "ember": {"train": ember.train_metadata, "test": ember.test_metadata},
        "az_class": {"train": az_class.train_metadata, "test": az_class.test_metadata},
        "schedule": {
            "path": str(effective_schedule_path.resolve()),
            "manifest_path": str(effective_manifest_path.resolve()),
            "sha256": effective_hash,
            "row_count": len(rows),
            "provenance": "PROPOSED_BUDGET_ONLY_NOT_HISTORICAL_TRAINING",
            "parent_source_path": str(source_schedule_path.resolve()),
            "parent_source_sha256": source_schedule_hash,
            "reconciliation": reconciliation,
        },
        "launch_gate": "all 220 data-derived proposed rows match checksum-verified dataset pools",
    }
    if args.output:
        write_json(args.output, result)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="revision-r1", description="Separately versioned matched continual-learning runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate both verified datasets and all proposed pool rows")
    preflight.add_argument("--ember-data-root", required=True, help="explicit Drive directory containing EMBER NPZ files")
    preflight.add_argument("--az-data-root", required=True, help="explicit Drive directory containing AZ-Class NPZ files")
    preflight.add_argument("--cache-root", required=True, help="explicit writable NPY cache directory")
    preflight.add_argument("--schedule", required=True)
    preflight.add_argument("--schedule-manifest", required=True)
    preflight.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="launch one production arm after the separate explicit authorization flag")
    run.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--arm", choices=("current_only", "anchor_only", "generated_only", "full_hybrid", "derpp_matched"), required=True)
    run.add_argument("--data-root", required=True, help="explicit Drive directory containing this dataset's NPZ files")
    run.add_argument("--cache-root", required=True)
    run.add_argument("--output-root", required=True, help="explicit persistent Drive output directory")
    run.add_argument("--schedule", required=True)
    run.add_argument("--schedule-manifest", required=True)
    run.add_argument("--drive-checkpoint-root", required=True, help="explicit Drive checkpoint mirror directory")
    run.add_argument("--local-checkpoint-root", default="/content/revision_r1_checkpoints", help="ephemeral Colab checkpoint staging directory; Drive remains the recovery authority")
    run.add_argument("--production-authorized", action="store_true", help="required manual gate; never set by setup/smoke cells")

    validate = subparsers.add_parser("validate", help="verify a completed result artifact set")
    validate.add_argument("--run-root", required=True)
    validate.add_argument("--production", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight_all(args)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
        if args.command == "validate":
            result = validate_run_artifacts(args.run_root, require_production=args.production)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
        if not args.production_authorized:
            parser.error("production launch is separately gated; pass --production-authorized only after explicit approval")
        # Keep inspection/preflight/test entry points usable without importing
        # the CUDA training stack. Training still imports it after authorization.
        from .runner import run_one

        cache = Path(args.cache_root)
        bundle = preflight_dataset(args.data_root, args.dataset, cache_dir=cache)
        schedule_hash = _verified_schedule_hash(args.schedule, args.schedule_manifest)
        config = make_run_config(args.dataset, args.seed, args.arm, schedule_hash)
        run_namespace = (
            Path(config.payload()["version"]) / args.dataset / f"seed_{args.seed}" /
            args.arm / config.sha256()[:16]
        )
        output = Path(args.output_root) / run_namespace
        result = run_one(
            config, bundle, output, update_schedule_path=args.schedule,
            device="cuda",
            drive_checkpoint_root=Path(args.drive_checkpoint_root) / run_namespace,
            local_checkpoint_root=Path(args.local_checkpoint_root) / run_namespace,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"revision_R1 error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
