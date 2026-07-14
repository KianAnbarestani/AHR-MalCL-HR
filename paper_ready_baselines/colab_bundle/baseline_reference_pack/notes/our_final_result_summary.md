# AHR-MalCL Result Summary for Baseline Audit

This file summarizes the current trusted results and evidence rules.

## Main Method

Official method:

AHR-MalCL-HR / hybrid_random_buffer

Core idea:

Replay-drift-aware anchored hybrid replay for malware class-incremental learning.

Main mechanism:

* Generated replay using conditional replay generator
* WGAN-GP
* Projection critic
* Feature Matching Loss
* Random real anchor buffer
* Strict incremental scaler
* Incremental class learning
* MLP classifier
* AdamW optimizer

Final main method disables:

* use_diversity_buffer = false
* use_kd = false
* use_proto_align = false

## Official Best-Performance Results

### EMBER

Setting:

* Dataset: EMBER
* K = 100
* Method: AHR-MalCL-HR / hybrid_random_buffer
* Seeds: 5

Results:

* Mean Seen Accuracy: 83.70 ± 0.67%
* Final Taskwise Accuracy: 79.76 ± 1.42%
* Forgetting: 8.38 ± 1.15%
* Macro-F1: 72.68 ± 0.59%
* Weighted-F1: 80.84 ± 0.63%
* Balanced Accuracy: 79.51 ± 0.30%
* Old-Class Accuracy: 79.40 ± 0.59%
* New-Class Accuracy: 82.12 ± 7.91%
* MMD: 0.0025 ± 0.0011
* Wasserstein: 0.0496 ± 0.0099
* Memory: 531.8 MB

### AZ-Class

Setting:

* Dataset: AZ-Class
* K = 200
* Method: AHR-MalCL-HR / hybrid_random_buffer
* Seeds: 5

Results:

* Mean Seen Accuracy: 75.76 ± 1.40%
* Final Taskwise Accuracy: 74.31 ± 1.24%
* Forgetting: 8.43 ± 0.94%
* Macro-F1: 67.48 ± 0.74%
* Weighted-F1: 71.32 ± 0.72%
* Balanced Accuracy: 83.48 ± 0.60%
* Old-Class Accuracy: 70.94 ± 0.16%
* New-Class Accuracy: 73.11 ± 18.36%
* MMD: 0.0013 ± 0.0005
* Wasserstein: 0.0383 ± 0.0052
* Memory: 637.1 MB

## Valid Memory-Efficient AHR-MalCL-HR Results

These come from final ablation hybrid_random_buffer files, not from old memory-efficient folders.

### EMBER K=25

* Mean Seen Accuracy: 79.50 ± 0.75%
* Final Taskwise Accuracy: 74.28 ± 1.96%
* Forgetting: 18.22 ± 1.78%
* Macro-F1: 67.39 ± 0.98%
* Balanced Accuracy: 73.50 ± 0.55%
* Memory: 463.7 MB

### AZ-Class K=100

* Mean Seen Accuracy: 73.36 ± 1.45%
* Final Taskwise Accuracy: 71.29 ± 1.18%
* Forgetting: 12.89 ± 1.06%
* Macro-F1: 64.05 ± 1.25%
* Balanced Accuracy: 79.59 ± 0.34%
* Memory: 544.4 MB

## Internal Baselines Already Available

Existing controlled ablations:

* classifier_real_only
* malcl_like
* wgan_projection_generated_only
* real_buffer_only
* hybrid_random_buffer
* hybrid_diversity_buffer
* plus_kd

Important interpretation:

* classifier_real_only may cover no-replay / fine-tuning if it truly trains incrementally without replay.
* real_buffer_only may cover ER if it is a canonical real replay setup.
* wgan_projection_generated_only covers pure generated replay for our architecture.
* malcl_like is an internal MalCL-like baseline, but it may not be a faithful MalCL reproduction.
* hybrid_random_buffer is the proposed method, not a baseline.
* hybrid_diversity_buffer and plus_kd are ablation variants, not external baseline papers.

## Trusted Statistical Validation Status

* Final main HR table is safe to use: YES
* Memory-efficient HR table is safe to use: YES
* Memory-efficient legacy/config-mismatch rows used as main HR results: NO
* Training rerun needed: NO
* Result files modified: NO
* Critical failures: 0
* Old memory-efficient folders are legacy/full-complexity config mismatch and must not be used as official simplified HR evidence.

## Claims That Are Safe

Strong safe claims:

* Real-memory anchoring strongly improves pure generated replay.
* AHR-MalCL-HR reduces forgetting compared with generated-only replay.
* AHR-MalCL-HR strongly reduces replay drift compared with generated-only replay.
* AHR-MalCL-HR outperforms MalCL-like internal baseline in the current protocol.
* Random real anchoring is stronger than the tested diversity/KD variants in the final ablation settings.

Claims that must be avoided:

* Do not claim replay-drift metrics are statistically correlated with final accuracy, because there are not enough points.
* Do not claim old memory-budget rows prove the final simplified HR method across all K values.
* Do not claim diversity/KD are always bad.
* Do not claim AHR-MalCL-HR improves new-class accuracy in every case.
