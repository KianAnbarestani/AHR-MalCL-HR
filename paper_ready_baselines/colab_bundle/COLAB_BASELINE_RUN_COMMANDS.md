# Colab Baseline Run Commands

Run GPU-heavy training only in Google Colab.

## Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Enter Project And Install Dependencies

```bash
cd /content/drive/MyDrive/AHR-MalCL
pip install -r paper_ready_baselines/colab_bundle/requirements_colab.txt
```

## GPU Check

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
```

## Adapter Checks

```bash
python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42
python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42
```

## DER++ Dry-Runs

```bash
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## ER Dry-Runs

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_az_k200.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## Joint Dry-Runs

```bash
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_az.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## One-Seed Smoke Test

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --single-seed 42 --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## Full Training Commands, Only After Dry-Runs Pass

```bash
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_er_baseline.py --config baseline_suite/configs/er_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines

python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_ember.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_joint_baseline.py --config baseline_suite/configs/joint_az.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines

python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## Local Integration After Copy/Sync Back

```bash
cd /home/kian/projects/shz_uni/paper-works/AHR-MalCL
source .venv_stats/bin/activate
python baseline_suite/integrate_external_baseline_results.py --input-root result_external_baselines
```
