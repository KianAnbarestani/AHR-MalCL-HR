# AHR-MalCL Statistical Validation Pipeline

This folder contains read-only analysis scripts for the AHR-MalCL statistical validation and result engineering phase. The scripts read existing JSON result files under `result/` and write derived CSV tables, statistical test outputs, figure-ready CSVs, figures, and a final summary under `paper_ready_stats/`.

The scripts do not edit notebooks, rerun training, change model code, or modify anything inside `result/`.

## Inputs Read

- Final best-performance JSONs under `result/final-run/EMBER/best-performance setting/` and `result/final-run/AZ-Class/best-performance setting/`.
- Final memory-efficient JSONs under `result/final-run/EMBER/memory-efficient setting/` and `result/final-run/AZ-Class/memory-efficient setting/`.
- Ablation JSONs under `result/final-run/EMBER/ablations/` and `result/final-run/AZ-Class/ablation/`.
- Memory-budget JSONs under `result/EMBER/memory budget/` and `result/AZ-Class/memory budget /`.
- Replay-drift, oracle-fidelity, ordering-sensitivity, and scaler-sensitivity JSONs discovered by recursive scan where present.

## Outputs Created

All outputs are written under:

```text
paper_ready_stats/
├── raw/
├── checks/
├── tables/
├── stats/
├── figure_data/
├── figures/
└── STATISTICAL_VALIDATION_SUMMARY.md
```

Key outputs:

- `raw/all_runs_long.csv`: one deduplicated row per logical seed run.
- `raw/all_aggregates_long.csv`: aggregate rows extracted from result JSONs.
- `raw/file_manifest.csv`: every JSON scanned, parse status, and any error.
- `raw/missing_metrics_report.csv`: missing metric counts by table and setting.
- `checks/sanity_check_report.md`: seed/config/pairing/memory/numeric sanity checks.
- `checks/pairing_matrix.csv`: seed compatibility matrix for ablation comparisons.
- `tables/descriptive_stats_*.csv`: mean, standard deviation, standard error, CI, median, min, max, and n.
- `stats/paired_t_tests.csv`: paired t-test rows for valid ablation comparisons.
- `stats/wilcoxon_tests.csv`: Wilcoxon signed-rank test rows.
- `stats/effect_sizes.csv`: Cohen's dz, paired differences, improvements, and CIs.
- `stats/holm_corrected_tests.csv`: Holm-Bonferroni corrected p-values.
- `stats/skipped_tests.csv`: comparisons skipped because pairing or metric values were invalid.
- `tables/table_1_final_main_results.*` through `tables/table_7_runtime_memory.*`: paper-ready table drafts.
- `figure_data/*.csv`: source data for all generated figures.
- `figures/*.png`: basic matplotlib figures.
- `STATISTICAL_VALIDATION_SUMMARY.md`: final run summary, highlights, warnings, generated files, and next steps.

## Exact Command Order

From the project root:

```bash
cd /home/kian/projects/shz_uni/paper-works/AHR-MalCL

python3 -m venv .venv_stats
source .venv_stats/bin/activate
pip install -U pip
pip install pandas numpy scipy matplotlib statsmodels

python stat_validation/run_all.py
```

Individual commands:

```bash
python stat_validation/collect_results.py
python stat_validation/sanity_checks.py
python stat_validation/descriptive_stats.py
python stat_validation/stat_tests.py
python stat_validation/memory_budget_analysis.py
python stat_validation/replay_drift_analysis.py
python stat_validation/make_tables.py
python stat_validation/make_figures.py
```

## Safe Rerun Notes

Rerunning the pipeline overwrites only derived files under `paper_ready_stats/`. It never writes into `result/`, never edits notebooks, and never starts or resumes training.

The collector uses `Path.rglob("*.json")`, handles malformed JSON files by recording them in `raw/file_manifest.csv`, and keeps missing metrics as empty/NaN values so downstream reports remain explicit.

## Statistical Rules Enforced

- Paired tests are restricted to ablation comparisons with the same dataset, setting type, K, and seed set.
- The ablation reference is `hybrid_random_buffer`.
- Final best-performance settings are not tested as if they were controlled ablations.
- Literature reference numbers are not given p-values unless raw per-seed values exist.
- Wilcoxon tests that cannot run, including all-zero differences, are written with NaN p-values and an explanation.
