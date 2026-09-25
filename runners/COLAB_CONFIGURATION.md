# Colab Configuration

The release launcher has no embedded personal Drive or workstation path. In a Colab cell, set the paths for your own runtime before executing `final_malcl_baseline_colab_one_cell.py`:

```python
import os
os.environ.update({
    "AHR_DRIVE_MOUNT": "<your Colab Drive mount path>",
    "AHR_PROJECT_ZIP": "<path to the release project zip>",
    "AHR_DATA_SOURCE_DIR": "<directory containing the four verified NPZ files>",
    "AHR_RUNTIME_DATA_DIR": "<fast runtime data directory>",
    "AHR_WORKDIR": "<runtime project directory>",
    "AHR_OUTPUT_ZIP": "<destination for the output archive>",
    "AHR_DATA_DIR": "<fast runtime data directory>",
})
```

Then run the launcher. The dataset files must first pass `scripts/download_and_verify_datasets.py` or an equivalent published-MD5 verification. Do not put credentials or personal paths into committed code.
