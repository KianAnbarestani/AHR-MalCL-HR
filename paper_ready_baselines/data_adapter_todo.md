# Data Adapter TODO

The data adapter code is present, but the dataset stream could not be loaded in this environment.

- Dataset: `EMBER`
- Reason: FileNotFoundError: Missing required files `EMBER_Class_train.npz` and `EMBER_Class_test.npz`. Checked: /content/data, /content/drive/MyDrive/AHR-MalCL/data, /content/drive/MyDrive/AHR-MalCL, /home/kian/projects/shz_uni/paper-works/AHR-MalCL/data, /home/kian/projects/shz_uni/paper-works/AHR-MalCL/datasets, /home/kian/projects/shz_uni/paper-works/AHR-MalCL

## Required Dataset Files

- `EMBER_Class_train.npz`
- `EMBER_Class_test.npz`
- `AZ_Class_Train.npz`
- `AZ_Class_Test.npz`

## How To Unblock In Colab

1. Mount Google Drive.
2. Place the required `.npz` files in `/content/data`, `/content/drive/MyDrive/AHR-MalCL/data`, or set `AHR_MALCL_DATA_DIR`.
3. Rerun `python baseline_suite/check_data_adapter.py --dataset EMBER --seed 42`.
4. Rerun `python baseline_suite/check_data_adapter.py --dataset AZ-Class --seed 42`.

No external files were downloaded by this adapter.
