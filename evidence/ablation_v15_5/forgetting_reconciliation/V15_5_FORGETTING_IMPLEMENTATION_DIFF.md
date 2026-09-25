# V15.5 forgetting implementation reconciliation

## Result

The authoritative implementation is `analysis/harmonized_continual_metrics.py:conventional_forgetting` (SHA256 `846d05fdb7735bf2fed8d7baecc04afa43ac940a786824bd280dc2897e2e47a2`). It implements the manuscript equation with zero-based `task in range(n-1)` and `stage in range(task, n-1)`: rows are post-training stages, columns are evaluated tasks, and the final row is excluded from the pre-final maximum. Future cells are unavailable (`None`; historical raw `-1` is normalized to unavailable).

The historical reconciliation implementation is `paper_ready/AHR_MalCL_v15_5_vs_v16_RECONCILIATION/reconcile.py:matrix_metrics` (SHA256 `c28ab556a9f64826aea88a06c934ec4228df5fb8c330d0030f18c40aa234c00e`), line 29. It uses `a[u:n-1,u]`, which is the same formula and matches the manuscript.

The final-80 implementation is `paper_ready/AHR_MalCL_V15_5_STANDARD_A_FINAL_80_ANALYSIS/analyze_final_80.py:metrics_from_matrix` (SHA256 `fcf9b107b7d7bfea461b30a0fb94a645801e9410213ff22c2e83fc0fffe5d959`), line 128. Its range `range(j, 11)` includes row 10, the final stage. Thus it computes `max(pre-final, final)-final`, which truncates any negative forgetting contribution to zero. This is the sole identified cause of the small discrepancy.

## Tests of common failure modes

`V15_5_FORGETTING_COMMON_FAILURE_MODE_TESTS.csv` evaluates all 20 Full-Hybrid matrices. Including the final stage reproduces final-80 exactly. The historical formula matches the manuscript exactly. The final stage exclusion is not a transpose issue, a `-1`/null issue, an Nth-task inclusion, diagonal BWT, or rounding artifact. BWT remains a distinct diagonal-minus-final metric.

## Authoritative means

- EMBER Full Hybrid: 9.223520%
- AZ-Class Full Hybrid: 9.159115%
