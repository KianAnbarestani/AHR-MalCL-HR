# Statistical Validation Summary

## Evidence Decisions

Final main HR table is safe to use: YES
Memory-efficient HR table is safe to use: YES
Memory-efficient legacy/config-mismatch rows used as main HR results: NO
Training rerun needed: NO
Result files modified: NO

## Final Main Result Summary

- az_class final best performance K=200: mean seen accuracy 75.76 ± 1.40, final taskwise accuracy 74.31 ± 1.24, forgetting 8.43 ± 0.94.
- ember final best performance K=100: mean seen accuracy 83.70 ± 0.67, final taskwise accuracy 79.76 ± 1.42, forgetting 8.38 ± 1.15.

- Final main HR table is safe to use: YES
- Memory-efficient legacy/config-mismatch rows used as main HR results: NO

## Memory-Efficient Simplified HR Summary

- az_class K=100 from ablation hybrid_random_buffer: mean seen accuracy 73.36 ± 1.45, final taskwise accuracy 71.29 ± 1.18, forgetting 12.89 ± 1.06.
- ember K=25 from ablation hybrid-random-buffer: mean seen accuracy 79.50 ± 0.75, final taskwise accuracy 74.28 ± 1.96, forgetting 18.22 ± 1.78.
- Memory-efficient HR table is safe to use: YES
- No rerun is needed for memory-efficient AHR-MalCL-HR results.

## Statistical Test Highlights

- Holm-significant test rows: 122
- Non-significant or uncorrected test rows: 242
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on mean_acc_seen (Holm p=6.303453650320866e-06).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on final_taskwise_average_accuracy (Holm p=3.839425851777974e-06).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on forgetting (Holm p=1.2522072692351384e-05).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on macro_f1 (Holm p=3.619781079548264e-06).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on balanced_accuracy (Holm p=1.1270219225344389e-08).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs classifier_real_only on old_class_accuracy (Holm p=5.23827703001438e-07).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on mean_acc_seen (Holm p=0.0002266881175909).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on final_taskwise_average_accuracy (Holm p=0.0002083770039065).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on forgetting (Holm p=0.0001619784040023).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on macro_f1 (Holm p=6.09881469810288e-05).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on balanced_accuracy (Holm p=1.4374372842463288e-05).
- Significant: az_class K=100 paired_t hybrid_random_buffer vs malcl_like on old_class_accuracy (Holm p=0.0003782246206411).
- Skipped metric-comparison rows: 16; see `stats/skipped_tests.csv`.

## Memory-Budget Elbow Summary

- Memory-budget rows come from older/full-complexity configuration and should be interpreted as budget-sensitivity evidence, not as the official final simplified AHR-MalCL-HR main result.
- az_class: recommended practical K=200; best accuracy K=200; lowest forgetting K=200.
- ember: recommended practical K=100; best accuracy K=100; lowest forgetting K=100.

## Replay-Drift Summary

- ember: mmd_real_generated vs final_taskwise_average_accuracy Pearson=nan, Spearman=nan. note=not enough points for correlation: n=2
- ember: mmd_real_generated vs forgetting Pearson=nan, Spearman=nan. note=not enough points for correlation: n=2
- ember: wasserstein_real_generated vs final_taskwise_average_accuracy Pearson=nan, Spearman=nan. note=not enough points for correlation: n=2
- ember: wasserstein_real_generated vs forgetting Pearson=nan, Spearman=nan. note=not enough points for correlation: n=2
- Replay-drift correlations still have insufficient points and should not be used as correlation claims.

## Validity Audit Summary

- Validity audit rows: 110.
- Critical failures: 0.
- Legacy/config-mismatch run rows: 10.
- Expected exclusion run rows: 1.
- Valid simplified HR memory-efficient table rows: 2.
- No rerun is needed for memory-efficient AHR-MalCL-HR results.
- Legacy memory-efficient result rows kept as supplementary only: 2.

## Excluded Rows

- ember final_memory_efficient ahr_malcl_v15_1_final_paper_critic2_ember_k25 K=25 seed=42: exclude_from_paper_claims (excluded_incomplete) - stray incomplete EMBER critic2 memory-efficient seed file; not part of declared final result set [result/final-run/EMBER/memory-efficient setting/seed=42/full-result.json]

## Supplementary-Only Legacy Rows

- az_class final_memory_efficient hybrid_random_buffer K=100 seed=42: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json;result/final-run/AZ-Class/memory-efficient setting/seed=42/full-result.json]
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=43: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json;result/final-run/AZ-Class/memory-efficient setting/seed=43/full-result.json]
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=44: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json;result/final-run/AZ-Class/memory-efficient setting/seed=44/full-result.json]
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=45: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json;result/final-run/AZ-Class/memory-efficient setting/seed=45/full-result.json]
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=46: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json;result/final-run/AZ-Class/memory-efficient setting/seed=46/full-result.json]
- ember final_memory_efficient hybrid_random_buffer K=25 seed=42: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json]
- ember final_memory_efficient hybrid_random_buffer K=25 seed=43: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json;result/final-run/EMBER/memory-efficient setting/seed=43/full-result.json]
- ember final_memory_efficient hybrid_random_buffer K=25 seed=44: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json;result/final-run/EMBER/memory-efficient setting/seed=44/full-result.json]
- ember final_memory_efficient hybrid_random_buffer K=25 seed=45: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json;result/final-run/EMBER/memory-efficient setting/seed=45/full-result.json]
- ember final_memory_efficient hybrid_random_buffer K=25 seed=46: supplementary_only (legacy_or_config_mismatch) - memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method [result/final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json;result/final-run/EMBER/memory-efficient setting/seed=46/full-result.json]

## Manual-Review Rows Not Used For Main Claims

- az_class other ahr_malcl_v14_5_az_validation_k25 K=25 seed=42: not part of main final, ablation, memory-budget, or replay-drift evidence categories [result/AZ-Class/first run/ahr_malcl_v14_5_az_validation_k25_results.json]
- ember oracle_fidelity ahr_malcl_v14_5_oracle_fidelity_k25 K=25 seed=42: not part of main final, ablation, memory-budget, or replay-drift evidence categories [result/EMBER/oracle-fidelity/ahr_malcl_v14_5_oracle_fidelity_k25_results.json]
- ember ordering_sensitivity ahr_malcl_v14_5_ordering_random K=25 seed=42: not part of main final, ablation, memory-budget, or replay-drift evidence categories [result/EMBER/ordering/ahr_malcl_v14_5_ordering_sensitivity_results.json;result/EMBER/ordering/random/ahr_malcl_v14_5_ordering_random_results.json]
- ember ordering_sensitivity ahr_malcl_v14_5_ordering_giant_first K=25 seed=42: not part of main final, ablation, memory-budget, or replay-drift evidence categories [result/EMBER/ordering/ahr_malcl_v14_5_ordering_sensitivity_results.json]
- ember scaler_sensitivity ahr_malcl_v14_5_scaler_incremental K=25 seed=42: not part of main final, ablation, memory-budget, or replay-drift evidence categories [result/EMBER/scaler-sensitivity/incremental/ahr_malcl_v14_5_scaler_incremental_results.json]

## Warnings / Limitations

- Sanity-check critical failures: 0.
- Sanity-check warnings: 18.
- Expected exclusions: 1.
- Manifest parse errors: 0.
- Missing metrics are expected for some older result files; top records:
  - all_runs_long precision_macro: missing 19 of 110.
  - all_runs_long precision_weighted: missing 19 of 110.
  - all_runs_long recall_macro: missing 19 of 110.
  - all_runs_long recall_weighted: missing 19 of 110.
  - all_runs_long mmd_real_generated: missing 20 of 110.
  - all_runs_long wasserstein_real_generated: missing 20 of 110.
  - all_runs_long feature_center_l2: missing 20 of 110.
  - all_aggregates_long precision_macro: missing 19 of 37.
  - all_aggregates_long precision_weighted: missing 19 of 37.
  - all_aggregates_long recall_macro: missing 19 of 37.
  - all_aggregates_long recall_weighted: missing 19 of 37.
  - all_aggregates_long replay_memory_MB: missing 37 of 37.
- Paired p-values are restricted to ablation rows with matching dataset, setting, K, and seed sets.
- Literature reference numbers are not tested for significance because no raw per-seed values are present.

## Files Generated

- `paper_ready_stats/STATISTICAL_VALIDATION_SUMMARY.md`
- `paper_ready_stats/checks/pairing_matrix.csv`
- `paper_ready_stats/checks/sanity_check_report.csv`
- `paper_ready_stats/checks/sanity_check_report.md`
- `paper_ready_stats/figure_data/ablation_barplot_data.csv`
- `paper_ready_stats/figure_data/accuracy_over_tasks.csv`
- `paper_ready_stats/figure_data/memory_budget_curve.csv`
- `paper_ready_stats/figure_data/old_new_tradeoff.csv`
- `paper_ready_stats/figure_data/replay_drift_curve.csv`
- `paper_ready_stats/figure_data/runtime_memory_data.csv`
- `paper_ready_stats/figures/fig_ablation_forgetting.png`
- `paper_ready_stats/figures/fig_ablation_mean_accuracy.png`
- `paper_ready_stats/figures/fig_accuracy_over_tasks.png`
- `paper_ready_stats/figures/fig_memory_budget_forgetting.png`
- `paper_ready_stats/figures/fig_memory_budget_mean_accuracy.png`
- `paper_ready_stats/figures/fig_old_new_tradeoff.png`
- `paper_ready_stats/figures/fig_replay_drift_mmd.png`
- `paper_ready_stats/figures/fig_replay_drift_wasserstein.png`
- `paper_ready_stats/figures/fig_runtime_memory.png`
- `paper_ready_stats/raw/all_aggregates_long.csv`
- `paper_ready_stats/raw/all_runs_long.csv`
- `paper_ready_stats/raw/file_manifest.csv`
- `paper_ready_stats/raw/missing_metrics_report.csv`
- `paper_ready_stats/stats/effect_sizes.csv`
- `paper_ready_stats/stats/holm_corrected_tests.csv`
- `paper_ready_stats/stats/paired_t_tests.csv`
- `paper_ready_stats/stats/replay_drift_correlations.csv`
- `paper_ready_stats/stats/skipped_tests.csv`
- `paper_ready_stats/stats/wilcoxon_tests.csv`
- `paper_ready_stats/tables/descriptive_stats_ablation.csv`
- `paper_ready_stats/tables/descriptive_stats_all.csv`
- `paper_ready_stats/tables/descriptive_stats_final.csv`
- `paper_ready_stats/tables/descriptive_stats_memory_budget.csv`
- `paper_ready_stats/tables/table_1_final_main_results.csv`
- `paper_ready_stats/tables/table_1_final_main_results.md`
- `paper_ready_stats/tables/table_2_full_metrics.csv`
- `paper_ready_stats/tables/table_2_full_metrics.md`
- `paper_ready_stats/tables/table_3_ablation.csv`
- `paper_ready_stats/tables/table_3_ablation.md`
- `paper_ready_stats/tables/table_4_component_effects.csv`
- `paper_ready_stats/tables/table_4_component_effects.md`
- `paper_ready_stats/tables/table_5_memory_budget.csv`
- `paper_ready_stats/tables/table_5_memory_budget.md`
- `paper_ready_stats/tables/table_6_replay_drift.csv`
- `paper_ready_stats/tables/table_6_replay_drift.md`
- `paper_ready_stats/tables/table_7_runtime_memory.csv`
- `paper_ready_stats/tables/table_7_runtime_memory.md`
- `paper_ready_stats/tables/table_component_effects.csv`
- `paper_ready_stats/tables/table_legacy_memory_efficient_results.csv`
- `paper_ready_stats/tables/table_legacy_memory_efficient_results.md`
- `paper_ready_stats/tables/table_memory_budget.csv`
- `paper_ready_stats/tables/table_memory_budget_elbow.csv`
- `paper_ready_stats/tables/table_memory_efficient_hr_results.csv`
- `paper_ready_stats/tables/table_memory_efficient_hr_results.md`
- `paper_ready_stats/tables/table_replay_drift.csv`
- `paper_ready_stats/tables/table_validity_audit.csv`
- `paper_ready_stats/tables/table_validity_audit.md`

## Next Steps For Paper Writing

- Inspect `checks/sanity_check_report.md` before quoting any result.
- Use `tables/table_validity_audit.md` to confirm which result rows support each paper claim category.
- Use `tables/table_1_final_main_results.md` through `table_7_runtime_memory.md` as the paper table drafts.
- Use `tables/table_memory_efficient_hr_results.md` for simplified HR memory-efficient evidence.
- Use `stats/holm_corrected_tests.csv` and `tables/table_4_component_effects.md` for ablation claims.
- Use `figure_data/*.csv` as the source of truth if figures need journal-specific restyling.
