# Environment Requirements

The final MalCL run evidence records Python 3.12, PyTorch 2.11.0, and CUDA 12.8. Re-running GPU training requires a compatible CUDA-enabled PyTorch build and a GPU with enough memory for the feature-length-dependent flattened discriminator; it is not part of release validation. Analysis and verification require Python 3.10+.

Install minimal verification dependencies with:

```bash
pip install -r requirements-verification.txt
pip install -r requirements-analysis.txt
```

GPU training dependencies are separately listed in `requirements-training.txt`. The released runners use PyTorch, NumPy, scikit-learn, and tqdm; no Gym dependency is required.
