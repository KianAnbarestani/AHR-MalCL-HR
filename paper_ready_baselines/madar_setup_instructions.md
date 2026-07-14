# MADAR Setup Instructions

MADAR training was not run. No repository was cloned or downloaded.

## Local Source Check

- none

## Required Setup After Approval

1. Place the approved MADAR source under `baseline_suite/external/MADAR/` or another reviewed local path.
2. Map MADAR data loading to the exact AHR-MalCL-HR class stream for EMBER and AZ-Class.
3. Use seeds `[42, 43, 44, 45, 46]`, random ordering, 50 initial classes, then 5 new classes per task.
4. Use budgets `EMBER K=100` and `AZ-Class K=200` for the main external comparison.
5. Emit `full-result.json`, `final_classification_report.csv`, `final_confusion_matrix.csv`, and `final_per_class_accuracy.csv` per seed.
6. Write all outputs under `result_external_baselines/MADAR/...`, never under `result/`.

## Adapter Requirements

- EMBER features: adapter-provided tabular features matching the final HR protocol.
- AZ-Class features: adapter-provided tabular features matching the final HR protocol.
- Class stream: same task split and class order as AHR-MalCL-HR.
- Memory budget: count replay features, labels, model memory, and material distribution-selection state.
- Metrics: mean seen accuracy, final taskwise accuracy, forgetting, macro-F1, weighted-F1, balanced accuracy, old/new-class accuracy, memory MB.

## Current Readiness

- Local MADAR source found: NO
- Ready to run MADAR: NO
