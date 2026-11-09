# Temporal EMBER pilot

This package is an independent six-window benchmark for the AHR-MalCL revision. It uses the upstream EMBER 2018 v2 data because the available MalCL EMBER feature rows do not have a verified mapping back to sample hashes and observation dates. It does not represent the exact MalCL subset.

The stream is **chronological new-to-learner family admission**. It does not claim that a family's first appearance in these pilot training partitions is its real-world emergence date. The initial January cohort is the top 50 families ranked from January training-partition support only. Each later window admits up to five families found in that window's training partition, without backfilling skipped families.

## Package contents

- `temporal_ember_pilot.ipynb`: Colab workflow with Setup → Audit → Run/Resume → Report.
- `audit_report.json`: CPU audit of the source archive, date coverage, duplicate policy, and per-window family support.
- `split_manifest.json.gz`: frozen, seed-independent sample split and family admissions.
- `frozen_budget_schedule.json`: release-frozen update count for each window; all arms and seeds use the same six counts.
- `verified_source_manifest.json`: source and SHA-256 provenance for the byte-identical verified R1 method modules.
- `runner.py`, `engine.py`, and `checkpoint.py`: standalone chronological runner and pilot-only resume state.
- `report.py`: paired report export and manifest consistency checks.

The package does not contain production outputs or checkpoints. Codex did not launch production training.

## Launch in Colab

1. Download `temporal_ember_pilot.zip`, `temporal_ember_pilot.zip.sha256`, and `temporal_ember_pilot.ipynb` from the handoff. Put all three files in `MyDrive/AHR-MalCL-temporal-ember-pilot/` without renaming them.
2. Open the notebook in Colab. Choose a GPU runtime before enabling training. The initial feature store is about 7.1 GiB; allow room in Drive for the archive, feature store, and checkpoints. Audit alone runs on CPU.
3. Run the notebook's **Setup** step and authorize the Drive mount. Setup verifies the ZIP checksum, extracts the standalone package, and installs the pinned data-vectorization dependencies. The output root is `MyDrive/AHR-MalCL-temporal-ember-pilot/outputs/ahrmalcl_temporal_ember18v2_newfamily_v1/`.
4. Run **Audit**. If needed, it downloads the official archive to the pilot's `data/` folder, checks the published SHA-256, performs the CPU audit, and verifies that the resulting manifest equals the shipped frozen manifest.
5. Review the frozen windows and budget in the printed audit/setup output. To launch the nine production configurations, edit the **Run/Resume** cell to set `ENABLE_TRAINING = True` and set `CONFIRM_EXPERIMENT_ID` to `ahrmalcl_temporal_ember18v2_newfamily_v1`. Keep the run matrix unchanged. The runner creates one checkpoint per completed window and resumes from the latest consecutive checkpoint if the cell is run again.
6. Run **Report** after all nine configurations complete. It writes `report/temporal_report.json`, `report/window_metrics.csv`, `report/new_family_acquisition.csv`, and a compressed frozen-manifest sidecar below the pilot output root. The report keeps January post-fit separate and computes the primary Full Hybrid minus Anchor-only difference with equal weight for each of the five later windows. It preserves negative and mixed differences.

Checkpoints and `progress.json` stay below this pilot output root. They are separate from the active randomized matrix's experiment IDs and storage. The notebook keeps the approximately 7.1 GiB feature memmap in `/content` for faster randomized training reads; it will be rematerialized from the Drive archive after a Colab runtime reset, while completed-window checkpoints persist on Drive. To persist the feature store instead, change `FEATURE_STORE_PATH` to `OUTPUT_ROOT / "feature_store"` and allow the additional Drive storage.

## Frozen protocol at a glance

| Window | Current train rows | Planned anchors | Planned generated rows | Classifier updates per arm/seed |
|---|---:|---:|---:|---:|
| Jan 2018 | 20,145 | 0 | 0 | 237 |
| Feb–Mar 2018 | 26,500 | 4,674 | 5,000 | 426 |
| Apr–May 2018 | 26,416 | 5,147 | 5,500 | 435 |
| Jun–Jul 2018 | 19,817 | 5,273 | 6,000 | 366 |
| Aug–Sep 2018 | 21,745 | 5,466 | 6,500 | 396 |
| Oct 2018 | 23,878 | 5,728 | 7,000 | 429 |

Budgets are frozen from the planned Full Hybrid pool before training. Realized GAN replay counts are logged independently and cannot change the update budget. Full Hybrid and Anchor-only share the same real-anchor policy and budget. Seeds 42, 43, and 44 affect randomness only; the date order and family admissions do not change.

The primary report is the mean, over the five post-January windows with equal window weight, of paired Full Hybrid minus Anchor-only pre-update macro-F1. All three seed-specific means and all five per-window differences are included. The report also includes sample counts, initial-family retention, per-family new-family acquisition counts, and separate GAN compute and wall time. Acquisition below 30 held-out samples for any admitted family is marked exploratory; October additions have no later acquisition evaluation and are marked unavailable.

## Sources and preprocessing note

The [official EMBER README](https://github.com/elastic/ember/blob/master/README.md) identifies the 2018 v2 archive and published SHA-256. The audit uses its six `train_features_*.jsonl` members and excludes `test_features.jsonl`. The `appeared` field is used at month precision as the sample's first-seen month. CoLLAs' [published malware repository](https://github.com/msrocean/continual-learning-malware) and its EMBER family-processing script are preprocessing references only; their randomized split is not reused.

The official EMBER feature extractor is pinned in `vendor/ember/features.py`. The compatibility shim preserves the legacy `FeatureHasher(input_type="string")` behavior for singleton strings on current scikit-learn. It does not infer or alter dates, labels, or family identity.
