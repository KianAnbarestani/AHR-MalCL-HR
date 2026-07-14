# Baseline Readiness Report

## 1. Files Created

- `baseline_suite/audit_baseline_coverage.py`
- `baseline_suite/write_config_templates.py`
- `baseline_suite/data_adapter.py`
- `baseline_suite/run_derpp_baseline.py`
- `baseline_suite/run_er_baseline.py`
- `baseline_suite/run_joint_baseline.py`
- `baseline_suite/baseline_runner_common.py`
- `baseline_suite/check_data_adapter.py`
- `baseline_suite/make_colab_baseline_bundle.py`
- `baseline_suite/prepare_madar_baseline.py`
- `baseline_suite/madar_adapter_template.py`
- `baseline_suite/run_lwf_baseline.py`
- `baseline_suite/run_ewc_baseline.py`
- `baseline_suite/run_icarl_baseline.py`
- `baseline_suite/integrate_external_baseline_results.py`
- `baseline_suite/configs/*.json` with 16 baseline templates
- `paper_ready_baselines/baseline_coverage_audit.csv`
- `paper_ready_baselines/baseline_coverage_audit.md`
- `paper_ready_baselines/baseline_reference_audit.md`
- `paper_ready_baselines/missing_reference_papers.md`
- `paper_ready_baselines/final_baseline_plan.md`
- `paper_ready_baselines/derpp_runner_todo.md`
- `paper_ready_baselines/madar_setup_instructions.md`
- `paper_ready_baselines/optional_baseline_implementation_notes.md`
- `paper_ready_baselines/external_baseline_summary.csv`
- `paper_ready_baselines/external_baseline_summary.md`
- `paper_ready_baselines/external_baseline_stat_tests.csv`
- `paper_ready_baselines/external_baseline_missing_results.md`
- `paper_ready_baselines/RUN_COMMANDS.md`
- `paper_ready_baselines/COLAB_RUN_COMMANDS.md`
- `paper_ready_baselines/COLAB_README.md`
- `paper_ready_baselines/colab_bundle/`
- `paper_ready_baselines/colab_baseline_bundle.zip`
- `paper_ready_baselines/BASELINE_READINESS_REPORT.md`
- `result_external_baselines/.gitkeep`

## 2. Result Directory Safety

- Files under `result/` modified: NO.
- New external-output root created: `result_external_baselines/`.
- Trusted statistical validation outputs were read only.

## 3. Training Status

- Training was run: NO.
- Only audits, config generation, dry-runs, and missing-result integration were run.

## 4. Mandatory Baselines Already Covered

- Fine-tuning / None: covered by `classifier_real_only`.
- Pure generated replay: covered by `wgan_projection_generated_only`.
- MalCL-like internal baseline: covered by `malcl_like`, but not faithful external MalCL.

## 5. Mandatory Baselines Missing Or Partial

- Joint/offline upper bound: missing.
- Canonical ER: partially covered only; `real_buffer_only` is not clean ER because it includes diversity/KD/prototype behavior.
- DER++: missing.
- MADAR: missing and requires external source.
- Faithful MalCL: needed if the paper requires exact closest generative replay reproduction.

## 6. New GPU Runs Required

- DER++ EMBER K=100.
- DER++ AZ-Class K=200.
- MADAR EMBER K=100 after source approval.
- MADAR AZ-Class K=200 after source approval.
- Canonical ER EMBER K=100 and AZ-Class K=200.
- Joint EMBER and AZ-Class upper bounds.
- Faithful MalCL only after the MalCL-like sufficiency decision.

## 7. External Code Required

- MADAR: YES.
- Faithful MalCL: likely YES unless locally reimplemented from the paper.
- FreeMOCA: YES if used numerically.

## 8. DER++ Readiness

- DER++ runner is ready: NO.
- DER++ dry-run passes: NO.
- Blocking issues: `.venv_stats` lacks `torch` and `sklearn`; local dataset `.npz` files are not present.
- TODO report: `paper_ready_baselines/derpp_runner_todo.md`.

## 9. MADAR Readiness

- MADAR adapter is ready: NO.
- MADAR source found locally: NO.
- MADAR requires external code: YES.
- Setup report: `paper_ready_baselines/madar_setup_instructions.md`.

## 10. Exact Commands To Run Next

```bash
cd /home/kian/projects/shz_uni/paper-works/AHR-MalCL
source .venv_stats/bin/activate
python baseline_suite/audit_baseline_coverage.py
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --dry-run
python baseline_suite/prepare_madar_baseline.py --dry-run
python baseline_suite/integrate_external_baseline_results.py
```

## 11. Assumptions And TODOs

- The trusted statistical validation pipeline remains the source of truth.
- Published paper numbers are literature-only unless reproduced under the exact EMBER/AZ protocol.
- Data loading logic has been extracted into `baseline_suite/data_adapter.py`.
- Mount or upload required dataset `.npz` files before adapter checks can pass.
- Install or activate a Colab training environment with `torch` and `sklearn` before ER, Joint, or DER++ training.
- Obtain approval before cloning or downloading MADAR, MalCL, or FreeMOCA source.

## 12. Final Go/No-Go

- Ready to run DER++: NO.
- Ready to run MADAR: NO.
- Need faithful MalCL reproduction: YES, if `malcl_like` is not accepted as sufficient internal evidence.
- Need Joint baseline: YES.
- Need ER baseline: YES.

## Colab Execution Plan

### Local CPU Tasks

- Maintain the baseline audit and readiness reports.
- Run data-adapter checks when dataset files are available locally.
- Generate configs and Colab bundle files.
- Integrate copied Colab outputs with `baseline_suite/integrate_external_baseline_results.py`.
- Do not run GPU-heavy training locally.

### Colab GPU Tasks

- ER training.
- Joint/offline upper-bound training.
- DER++ training.
- MADAR training only after source integration is explicitly approved.
- Faithful MalCL training only if needed later.

### Dry-Run Commands

```bash
python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42
python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

### One-Seed Smoke Test Commands

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --single-seed 42 --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --device cuda --single-seed 42 --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --device cuda --single-seed 42 --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

### Full Five-Seed Commands

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_az.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

### Output Folder

`/content/drive/MyDrive/AHR-MalCL/result_external_baselines`

### Integration Command

```bash
python baseline_suite/integrate_external_baseline_results.py --input-root result_external_baselines
```

### Colab Go/No-Go

- Ready for local audit: YES.
- Ready for Colab adapter check: YES, after dataset `.npz` files are mounted.
- Ready for Colab ER dry-run: YES, after dataset `.npz` files are mounted and Colab dependencies are installed.
- Ready for Colab Joint dry-run: YES, after dataset `.npz` files are mounted and Colab dependencies are installed.
- Ready for Colab DER++ dry-run: YES, after dataset `.npz` files are mounted and Colab dependencies are installed.
- Ready for GPU training: NO until adapter checks and dry-runs pass in Colab.
