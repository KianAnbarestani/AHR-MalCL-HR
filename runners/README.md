# Runner inventory

## V15.5 Standard-A handoff

The exact V15.5 Standard-A source used for the 40 newly executed ablation rows is in v15_5_standard_a. See that folder’s README and RUNNER_PROVENANCE.md at the repository root. The handoff covers Current-only, Anchor-only at MAIN-K, and Generated-only rows. It does not rerun historical Full-Hybrid rows or provide a K-sensitivity training runner.

## Protocol-matched comparison runners

- run_er_baseline.py and run_derpp_baseline.py provide the available ER and DER++ runners and protocol checks.
- run_final_malcl_baseline.py and final_malcl_baseline_colab_one_cell.py provide the available adapted-MalCL implementation and launcher.
- patched_malcl/ and external_code/MalCL_official/ preserve the adapted topology and upstream MalCL source, including its separate license.
- baseline_runner_common.py contains shared baseline-runner utilities.

The older AHR one-cell notebooks from the preliminary V15.1 runner were excluded because they are not the exact source used for the V15.5 Standard-A executions. The main comparison’s historical results and matrices remain available in the evidence and task_matrices directories.
