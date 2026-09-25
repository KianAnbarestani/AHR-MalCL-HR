# Target environment lock

- Google Colab Linux, Tesla T4
- Python 3.12
- PyTorch 2.6+ with CUDA (use the Colab preinstalled CUDA build; do not replace it with a CPU wheel)
- NumPy 2.0.2
- tqdm 4.67.1
- pytest 8.3.5

Every executed run serializes the exact Python, PyTorch, CUDA, cuDNN, driver, and GPU values in `environment.json`.

