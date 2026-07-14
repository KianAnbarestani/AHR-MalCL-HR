# Final MalCL Baseline Colab Run

This folder contains the final one-cell Colab runner for the protocol-matched MalCL baseline.

Main file:

```text
paper_ready/final_malcl_baseline_colab_run/final_malcl_baseline_colab_one_cell.py
```

Colab execution entry point:

```text
paper_ready/final_malcl_baseline_colab_run/run_final_malcl_baseline.py
```

The runner validates:

- official MalCL commit `2cfb344461f594877e24fd4d19919b6bca38c861`
- required official MalCL model/training/sample-selection files
- EMBER and AZ-Class dataset existence, shape, and label range
- Python, numpy, pandas, sklearn, torch, CUDA, GPU, and `nvidia-smi`
- parameterized MalCL generator/discriminator/classifier topology
- class orders for seeds 42-51 using `numpy.random.RandomState(seed).permutation(100)`
- final configuration and output paths

The discriminator uses a T4-memory-safe pooled head while preserving the MalCL discriminator convolutional front-end and real/fake role. This is an adapted protocol-matched MalCL baseline candidate, not an exact official MalCL reproduction.

The full run covers:

- datasets: EMBER and AZ-Class
- seeds: 42,43,44,45,46,47,48,49,50,51
- 100 classes
- 11 tasks
- task sizes: 50, then 10 tasks of 5
- protocol-matched training settings: 3 epochs per task for both the GAN updates and classifier updates, batch size 256, `z_dim=62`, FML generator loss, `L1_C_Mean` sample selection, `k=3`

Training progress is shown directly in Colab by default with `tqdm.auto` progress bars plus text logs:

- `RUN_START` and `RUN_DONE` for each dataset/seed.
- `TASK_START` for each task.
- `EPOCH_START` and `EPOCH_DONE` for each epoch.
- A full 20-run progress bar.
- Task progress bars for each dataset/seed.
- Epoch progress bars for each task.
- Batch progress bars for each epoch, with discriminator, generator, classifier loss, and replay-memory postfix values.
- `BATCH_PROGRESS` text every `PROGRESS_EVERY_N_BATCHES` batches.
- `TASK_EVAL` after each task.

Each run also writes `training_progress.csv` beside `full-result.json`.

The Colab one-cell script expects:

```text
/content/drive/MyDrive/final_malcl_baseline_colab_package.zip
/content/drive/MyDrive/data/EMBER_Class_train.npz
/content/drive/MyDrive/data/EMBER_Class_test.npz
/content/drive/MyDrive/data/AZ_Class_Train.npz
/content/drive/MyDrive/data/AZ_Class_Test.npz
```

It also supports datasets already copied to:

```text
/content/data/
```

Final Colab output ZIP:

```text
/content/drive/MyDrive/AHR_MalCL/final_malcl_baseline_results_colab.zip
```

Decision logic:

- `MALCL_BASELINE_COMPLETE_READY_FOR_REVIEW`: all 20 runs completed with real training, real data, non-null metrics, non-zero memory accounting, matched class order, and passing validation.
- `MALCL_BASELINE_PARTIAL_NEEDS_RESUME`: validation passed but some runs are missing/incomplete; rerun the same Colab cell to resume.
- `MALCL_BASELINE_NOT_VALID_PATCH_REQUIRED`: validation, topology, metrics, memory, or run integrity failed.

The local files in this folder do not claim MalCL results. Paper usability is decided only by the Colab-generated aggregate reports.
