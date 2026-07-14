# Colab Baseline README

This package is for running external baselines in Google Colab while keeping local work limited to audit, dry-runs, packaging, and result integration.

## 1. What To Upload Or Mount

- Mount or upload the project folder as `/content/drive/MyDrive/AHR-MalCL`.
- Make sure `baseline_suite/`, `baseline_suite/configs/`, and `paper_ready_baselines/colab_bundle/requirements_colab.txt` are present.
- Dataset files are not bundled. Provide them separately in `/content/data`, `/content/drive/MyDrive/AHR-MalCL/data`, or set `AHR_MALCL_DATA_DIR`.

Required dataset files:

- `EMBER_Class_train.npz`
- `EMBER_Class_test.npz`
- `AZ_Class_Train.npz`
- `AZ_Class_Test.npz`

## 2. Install Dependencies

```bash
cd /content/drive/MyDrive/AHR-MalCL
pip install -r paper_ready_baselines/colab_bundle/requirements_colab.txt
```

## 3. Verify GPU

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
```

## 4. Run Adapter Checks

```bash
python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42
python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42
```

Reports are written to `paper_ready_baselines/data_adapter_check_ember_seed42.md` and `paper_ready_baselines/data_adapter_check_az_seed42.md`.

## 5. Run Dry-Runs

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

Repeat with AZ configs after EMBER dry-runs pass.

## 6. Run One Seed First

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --single-seed 42 --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

Inspect the seed folder before launching all seeds.

## 7. Run All Seeds

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_az.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## 8. Output Locations

Outputs go under:

`/content/drive/MyDrive/AHR-MalCL/result_external_baselines/`

Each seed writes:

- `full-result.json`
- `final_classification_report.csv`
- `final_confusion_matrix.csv`
- `final_per_class_accuracy.csv`

Each baseline setting also writes:

- `aggregate.json`
- `aggregate.csv`

## 9. Copy Outputs Back Locally

After Colab finishes, sync or copy `result_external_baselines/` back into the local project root:

`/home/kian/projects/shz_uni/paper-works/AHR-MalCL/result_external_baselines/`

Do not copy anything into `result/`.

## 10. Integrate Results Locally

```bash
cd /home/kian/projects/shz_uni/paper-works/AHR-MalCL
source .venv_stats/bin/activate
python baseline_suite/integrate_external_baseline_results.py --input-root result_external_baselines
```

Integration outputs are written to `paper_ready_baselines/external_baseline_summary.csv`, `external_baseline_summary.md`, `external_baseline_stat_tests.csv`, and `external_baseline_missing_results.md`.
