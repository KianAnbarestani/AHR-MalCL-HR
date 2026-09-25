# MalCL task-matrix provenance audit V15.5

Status: **PATH D — preserved historical outputs only**.

Searches covered the repository, Git history and all branches, MalCL pilot and
final experiment directories, V14/V15 release packages, notebooks, runners,
logs, staging scripts, and prior audit reports. The exact 20 historical
`per_task_metrics.csv` files contain the triangular `taskwise_accuracies`
field and are included with SHA-256 hashes. The recovered pilot/final runner
variants write only simpler per-task schemas; no exact runner or
postprocessor that generated the augmented field was found. The 20 normalized
matrices reproduce the CSV values exactly, but runner-to-matrix reproducibility
is not claimed. This is authentic, hashed preserved output-derived evidence,
not a fabricated bridge.
