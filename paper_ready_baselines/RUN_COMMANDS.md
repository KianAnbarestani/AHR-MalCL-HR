# External Baseline Run Commands

Do not run training commands until the audit and dry-run confirm which baselines are truly missing and runnable.

## Audit Only

```bash
cd /home/kian/projects/shz_uni/paper-works/AHR-MalCL
source .venv_stats/bin/activate
python baseline_suite/audit_baseline_coverage.py
```

## Reference Audit

```bash
python baseline_suite/audit_baseline_coverage.py
```

## Config Templates

```bash
python baseline_suite/write_config_templates.py
```

## DER++ Dry-Run

```bash
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --dry-run
```

Current dry-run status: fails safely because `torch`, `sklearn`, and the importable data-stream adapter are missing.

## DER++ Training, Only After Dry-Run Passes

```bash
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json
```

## MADAR Preparation

```bash
python baseline_suite/prepare_madar_baseline.py --dry-run
```

Current MADAR status: source code is not local; setup instructions are in `paper_ready_baselines/madar_setup_instructions.md`.

## Optional Baseline Dry-Runs

```bash
python baseline_suite/run_lwf_baseline.py --dry-run
python baseline_suite/run_ewc_baseline.py --dry-run
python baseline_suite/run_icarl_baseline.py --dry-run
```

## Integration

```bash
python baseline_suite/integrate_external_baseline_results.py
```

## Next Required Fix Before Training

```bash
python baseline_suite/data_adapter.py
```

Use this adapter boundary to extract and validate the notebook data loading into `load_class_incremental_stream(config, seed)`.
