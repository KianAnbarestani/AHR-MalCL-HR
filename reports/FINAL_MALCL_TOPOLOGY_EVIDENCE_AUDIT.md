# Final MalCL Topology Evidence Audit

## Scope

This reconciliation audit inspected the completed final MalCL artifact set without rerunning training or changing any raw result. The primary evidence is the twenty native `full-result.json`, `run_config.json`, `class_order.json`, `memory_accounting.json`, and `run_log.txt` files under `paper_ready/final_malcl_baseline_results/raw_outputs/`.

## Final Artifact Completeness

| Dataset | Seeds | Native completed runs | Classifier epochs | GAN epochs | Batch size |
|---|---|---:|---:|---:|---:|
| EMBER | 42--51 | 10 | 3 | 3 | 256 |
| AZ-Class | 42--51 | 10 | 3 | 3 | 256 |

All twenty `full-result.json` records use the required label: `protocol-matched adapted MalCL baseline`.

## Discriminator Footprint Evidence

| Dataset | Feature dimension | Observed discriminator parameter memory (all ten seeds) | Flattened first-FC parameterization | Interpretation |
|---|---:|---:|---|---|
| EMBER | 2381 | 2382.526371 MiB | `Linear(256 * 2381, 1024)` | Consistent with the feature-length-dependent flattened topology |
| AZ-Class | 2439 | 2440.526371 MiB | `Linear(256 * 2439, 1024)` | Consistent with the feature-length-dependent flattened topology |

The values include the discriminator parameter tensors measured in the final-run memory artifacts. They are incompatible with a 256-wide first fully connected layer after global adaptive pooling, whose parameter footprint would be orders of magnitude smaller. V14.1 therefore replaces the erroneous release-only pooled implementation with a feature-dimension-parameterized flattened discriminator. The preserved official MalCL clone remains unchanged.

## Source-Snapshot Limitation

The final raw-output directory contains run logs and per-run configurations but no immutable source archive, source hash, or code snapshot captured at run time. The run timestamps are 2026-07-12 for EMBER and 2026-07-13 for AZ-Class. Consequently, V14.1 does not claim a byte-for-byte recovery of the runtime source. It reconciles the release implementation to the strongest available completed-run evidence: the exact topology-dependent parameter-memory values, final configurations, and protocol records.

## Reconciliation Decision

The corrected release implementation is suitable only as a **protocol-matched adapted MalCL baseline**. It is not labeled an exact official MalCL reproduction because feature-dimension parameterization, classifier/output handling, and protocol integration remain documented adaptations.
