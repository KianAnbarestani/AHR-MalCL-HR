# V15.5 Standard-A GPU execution handoff

This source handoff was assembled to execute the 40 then-missing Standard-A rows. Those rows have since been executed and validated; their analysis and validation records are included in the parent repository. Use these instructions only for a separate rerun. The handoff does not run Full-Hybrid historical reuse rows or the K-sensitivity experiment.

## Requirements

- Google Colab or Linux with one Tesla T4 selected.
- The four prepared dataset files together in DATA_ROOT, with exact hashes and shapes enforced by preflight.
- A persistent Google Drive directory for OUTPUT_ROOT.
- Colab's CUDA-enabled PyTorch. The requirements file installs the pinned non-PyTorch dependencies.

The launcher refuses full or smoke execution when CUDA is unavailable or the detected GPU is not a T4. Every completed task is checkpointed atomically. Re-running the same command resumes from the last completed task.

## Commands

From this directory, verify the prepared datasets:

    python launch_v15_5_standard_a.py --data-root /content/data --output-root /content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A --preflight

Run the isolated smoke checks:

    python launch_v15_5_standard_a.py --data-root /content/data --output-root /content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A --smoke-tests

Run or resume the 40-row Standard-A grid:

    python launch_v15_5_standard_a.py --data-root /content/data --output-root /content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A --run-all --max-hours 10

Inspect or validate a saved output root:

    python launch_v15_5_standard_a.py --output-root /content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A --status
    python launch_v15_5_standard_a.py --data-root /content/data --output-root /content/drive/MyDrive/AHR_MalCL_V15_5_STANDARD_A --validate

Do not edit the registry in the source handoff. At first use it is copied into the output root; all state changes occur in that working copy. Smoke outputs are isolated and never count as scientific runs.

## Output

Runs are stored under OUTPUT_ROOT/runs/<variant>/<dataset>/seed_<seed>. The output registry is V15_5_STANDARD_A_40_RUN_REGISTRY.csv. The launcher records sampler draws, task pool sizes and update counts, task matrices, final metrics, environment identity, checkpoints, checksums, and a VALIDATED_COMPLETE marker. Validation enforces component isolation and a consistent hardware/software signature.
