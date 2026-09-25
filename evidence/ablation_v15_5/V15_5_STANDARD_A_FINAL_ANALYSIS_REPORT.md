# Final 80-row V15.5 Standard-A analysis report

## Validated corpus

- New Standard-A T4 executions: 40/40 found and 40/40 validated; 0 validation failures.
- Historical rows: 40/40 reused exactly from `V15_5_FINAL_REUSE_MANIFEST.csv`; no historical row was rediscovered or substituted.
- Canonical scientific grid: 80/80, with 0 duplicate dataset × variant × seed keys.
- Each cell has 20 runs and each dataset has 40 runs. Every dataset × variant cell contains seeds 42–51.

## Analysis method

All continual metrics were recomputed from the 11×11 task matrices by one harmonized implementation. Summary metrics are fractions with sample SD and N=10 per dataset × variant. Paired tests use first-minus-second raw differences, benefit-aligned differences, deterministic 20,000-resample paired bootstrap (seed 20260821), exact sign-enumerated Wilcoxon p-values, paired t-test p-values, and Holm correction only across the three primary Full-Hybrid contrasts for each dataset × metric.

## Evidence boundary and readiness

Standard-A is component-dependent in optimizer exposure by design, so it combines replay/content effects with the native update-budget changes caused by different pool sizes. It is not fixed-compute causal isolation. V15.5 and V16 are reported separately and never pooled.

The 80-row ablation is complete: 40 newly executed Standard-A rows and 40 validated historical rows were analyzed. The separate 60-row K-sensitivity study is complete and is documented in evidence/k_sensitivity_v15_5. This report does not imply that the K-sensitivity runs are still pending.
