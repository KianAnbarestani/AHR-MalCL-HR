# Final MalCL Previous Evidence Analysis

## Phase 9C Colab Evidence

Source:

```text
paper_ready/phase9c_malcl_real_limited_pilot_outputs_colab
```

Observed:

- Colab environment was usable: `torch_available=true`, `cuda_available=true`, GPU `Tesla T4`.
- All four dataset files were found.
- EMBER limited pilot passed.
- AZ-Class limited pilot passed.
- Real training executed for 2 limited runs.
- Non-null metrics were exported for 2 limited runs.
- Non-zero memory accounting was exported for 2 limited runs.
- The limited Phase 9C output was not paper-claimable and was not paired-comparison ready.

Interpretation:

Phase 9C proved that the Colab data/environment path can execute real limited MalCL-style training on both datasets, but it was not a full protocol-matched MalCL baseline.

## Phase 9D Evidence

Source:

```text
paper_ready/phase9d_malcl_one_seed_pilot
```

Observed:

- Seed-42 class order was recovered and matched AHR/ER/DER++ artifacts.
- Recovered rule: `numpy.random.RandomState(seed).permutation(100)`.
- Seed 42 first 10 classes: `[83, 53, 70, 45, 44, 39, 22, 80, 10, 0]`.
- Task split: 50 initial classes, then 10 tasks of 5.
- Local one-seed full-task execution was skipped because the local machine had no torch, CUDA, or datasets.

Interpretation:

Phase 9D supplied the class-order rule and pairing logic needed by the final runner. It did not produce MalCL results.

## Final Runner Consequence

The final runner in this folder uses Phase 9C only as evidence that Colab can run real training and Phase 9D only for the recovered class-order rule. It does not treat either phase as a paper-ready MalCL baseline.
