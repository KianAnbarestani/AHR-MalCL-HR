# Final MalCL Protocol Matching Check

Decision: `PROTOCOL_MATCHED_WITH_DOCUMENTED_DEVIATIONS`

Recommended label for the run: **protocol-matched adapted MalCL baseline**.

This check inspects protocol and configuration correctness only. It does not analyze memory accounting and does not rerun any full experiment.

## Scope

Inspected locations:

- `paper_ready/final_malcl_baseline_colab_run/`
- `paper_ready/phase9a_malcl_reproduction_design/external_code/MalCL_official/`
- `paper_ready/phase9c_malcl_real_limited_pilot_outputs_colab/`
- `paper_ready/phase9d_malcl_one_seed_pilot/`
- `result_external_baselines/`
- `result/`

## Summary

| Check | Status | Finding |
|---|---:|---|
| Dataset protocol | Pass with runtime validation | The final runner encodes the required EMBER and AZ-Class shapes, feature dimensions, label range, and 100-class checks. Local `.npz` files were not reloaded during this audit; the Colab runner validates them before training. |
| Task protocol | Pass | The final runner uses 11 tasks: 50 classes in task 0 and 5 classes in each task 1-10. Evaluation after each task is on all seen task ranges/classes. |
| Seed protocol | Pass | The final runner uses seeds 42-51 for both datasets, giving 10 seeds per dataset and 20 total runs. |
| Class-order protocol | Pass | The final runner uses `numpy.random.RandomState(seed).permutation(100)`. Seed 42 first 10 classes match `[83, 53, 70, 45, 44, 39, 22, 80, 10, 0]`. |
| Training protocol | Pass after patch | The final runner now defaults to 3 epochs per task through one shared `MALCL_EPOCHS` control. Both GAN and classifier updates are executed inside those epochs. |
| Batch/replay config | Pass | Batch size is 256; generator loss is `FML`; sample selection is `L1_C_Mean`; `k=3`; replay is refreshed every batch. |
| MalCL faithfulness | Documented deviation | The implementation is not an exact official MalCL reproduction. It is an adapted MalCL baseline matched to the malware CL protocol. |
| Prior pilot evidence | Superseded caveats | Phase 9C was a limited/non-paired pilot with wrong class order and 1-epoch/2-task settings. Phase 9D corrected class order and matched AHR ordering for the one-seed pilot. |

## Dataset Protocol

The final runner defines and validates:

- EMBER train/test shapes: `(303331, 2381)` and `(33704, 2381)`.
- AZ-Class train/test shapes: `(257023, 2439)` and `(28559, 2439)`.
- Label minimum 0, label maximum 99.
- Exactly 100 unique classes.

This is encoded in `run_final_malcl_baseline.py` through `FEATURE_DIMS`, `EXPECTED_SHAPES`, and `validate_datasets()`. The local data files were not re-opened in this audit because the large `.npz` files are expected to be supplied on Colab/Drive. The final runner performs the shape and label checks before any training starts, so a mismatched dataset will fail early.

External baseline artifacts inspected under `result_external_baselines/` confirm the same protocol metadata for representative ER runs:

- EMBER uses feature dimension 2381, 100 classes, labels 0-99, and the expected 11-task split.
- AZ-Class uses feature dimension 2439, 100 classes, labels 0-99, and the expected 11-task split.

One stale non-final code caveat remains: `baseline_suite/data_adapter.py` contains an older AZ-Class expected feature dimension value in a helper path. The final MalCL runner and external/Phase 2 artifacts use AZ-Class feature dimension 2439, so this stale value should not be used as authoritative for the final run.

## Task Protocol

The final runner constructs tasks as:

- Task 0: first 50 classes from the seed-specific class order.
- Tasks 1-10: ten increments of 5 classes each.

This exactly yields 11 tasks and 100 total classes. After training each task, the runner calls seen-class evaluation using task ranges through the current task. This matches the AHR/ER/DER++ continual-learning evaluation protocol: evaluation after task `t` is on all classes seen through task `t`.

## Seed Protocol

The final runner defines:

```python
SEEDS = list(range(42, 52))
DATASETS = ["EMBER", "AZ-Class"]
run_jobs = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
```

Therefore:

- EMBER seeds: 42-51.
- AZ-Class seeds: 42-51.
- Runs per dataset: 10.
- Total final runs: 20.

This matches the intended final 10-seed protocol.

## Class-Order Protocol

The final runner uses:

```python
numpy.random.RandomState(seed).permutation(100)
```

and validates seed 42 against the required first 10 classes:

```text
[83, 53, 70, 45, 44, 39, 22, 80, 10, 0]
```

The Phase 9D one-seed pilot also records an exact match to the AHR class order. Phase 9C did not match this order, but Phase 9C was a limited pilot and is not the final runner.

## Training Protocol

The intended paper protocol is 3 epochs per task for both GAN and classifier training. The final runner now defaults to:

```python
MALCL_EPOCHS = int(os.environ.get("MALCL_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("MALCL_BATCH_SIZE", "256"))
```

The Colab one-cell runner also sets:

```python
MALCL_EPOCHS = "3"
MALCL_BATCH_SIZE = "256"
```

The current runner has one shared epoch control, `MALCL_EPOCHS`, not separate `GAN_EPOCHS` and `CLF_EPOCHS`. Within each task epoch, it performs generator/discriminator and classifier updates. Therefore, with `MALCL_EPOCHS=3`, the GAN path and classifier path both receive three training passes per task.

Other final settings:

- Generator loss: `FML`.
- Sample selection: `L1_C_Mean`.
- `k`: 3.
- Replay refresh: every batch.
- Batch size: 256.

This is protocol-matched to the intended training schedule after the 3-epoch patch.

## MalCL Faithfulness

The final code should not be described as an exact official MalCL reproduction.

The correct label is:

> protocol-matched adapted MalCL baseline

Observed deviations from the official MalCL code include:

- Feature dimensions are parameterized to support both EMBER 2381 and AZ-Class 2439. The official code hard-codes 2381.
- The discriminator uses an adaptive pooling head to avoid the official fully connected head that scales as `256 * feature_dim`. This changes the discriminator feature path.
- The classifier is parameterized for task/output expansion and uses explicit softmax dimension handling.
- Replay/sample-selection logic is adapted/reimplemented in the final runner rather than invoked as an unchanged official script.
- The final runner executes a consolidated task loop matched to the AHR/ER/DER++ protocol. The update ordering differs from the official training script.

These deviations do not invalidate the baseline for protocol comparison, but they must be disclosed. The run is an adapted MalCL baseline, not a bit-for-bit official reproduction.

## Final Decision

`PROTOCOL_MATCHED_WITH_DOCUMENTED_DEVIATIONS`

No blocking protocol mismatch was found for the final runner after the 3-epoch correction. The main caveat is implementation faithfulness: the code is adapted from MalCL to fit the shared malware continual-learning protocol and Colab runtime constraints, so the paper should call it a **protocol-matched adapted MalCL baseline**.

## Main Blockers

None blocking for protocol execution.

Documented caveats:

- Not an exact official MalCL reproduction.
- Uses feature-dimension parameterization and a T4-safe discriminator pooling patch.
- Uses adapted replay/sample-selection and a protocol-specific training loop.
- Local large dataset files were not directly reloaded in this audit; the final Colab runner validates them before training.
