# DER++ Runner TODO

DER++ is now Colab-compatible, but it is not ready for GPU training until dry-run passes in Colab.

## Current Local Blockers

- `.venv_stats` does not include `torch`.
- `.venv_stats` does not include `sklearn`.
- Required dataset files are not present locally:
  - `EMBER_Class_train.npz`
  - `EMBER_Class_test.npz`
  - `AZ_Class_Train.npz`
  - `AZ_Class_Test.npz`

## Current Adapter Status

- `baseline_suite/data_adapter.py` exists.
- It uses the extracted AHR-MalCL-HR dataset-loading, NPZ extraction, subsampling, random task ordering, and class-stream logic.
- It will load streams after the dataset `.npz` files are available in `/content/data`, `/content/drive/MyDrive/AHR-MalCL/data`, or `AHR_MALCL_DATA_DIR`.

## Colab Dry-Runs To Pass Before Training

```bash
python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42
python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --dry-run --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```

## Training Commands After Dry-Runs Pass

```bash
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_ember_k100.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
python baseline_suite/run_derpp_baseline.py --config baseline_suite/configs/derpp_az_k200.json --device cuda --output-root /content/drive/MyDrive/AHR-MalCL/result_external_baselines
```
