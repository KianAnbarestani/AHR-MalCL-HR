# V15.5 canonical continual-metrics lock

This lock is for the forthcoming K-sensitivity analysis; it does not launch it.

- Canonical module/function: `paper_ready/AHR_MalCL_HR_reproducibility_release_v15_5/analysis/harmonized_continual_metrics.py:conventional_forgetting`
- Module SHA256: `846d05fdb7735bf2fed8d7baecc04afa43ac940a786824bd280dc2897e2e47a2`
- Matrix orientation: rows are post-training stages; columns are evaluation tasks; upper/future cells are unavailable.
- Units: fractions in code and CSVs; multiply by 100 only for percentage display.
- Forgetting: for N tasks, `(1/(N-1)) * sum(task=0..N-2)[max(stage=task..N-2) A[stage,task] - A[N-1,task]]`.
- BWT: `(1/(N-1)) * sum(task=0..N-2)[A[N-1,task] - A[task,task]]`.
- TM-AIA: arithmetic mean over stages of the arithmetic mean of all seen task accuracies.
- SW-AIA: arithmetic mean over stages of the support-weighted mean of seen task accuracies.
- Final TM: arithmetic mean of the final-stage task accuracies over all N tasks.

BWT and conventional forgetting are intentionally distinct: BWT uses acquisition diagonal; forgetting uses the best pre-final stage.
