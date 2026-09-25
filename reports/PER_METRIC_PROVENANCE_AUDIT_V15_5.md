# Per-metric provenance audit V15.5

The V15.5 seed schema separates raw-result, task-matrix, harmonized-definition,
classification-report, resource-measurement, and diagnostic sources. SW-AIA,
TM-AIA, forgetting, BWT, acquisition and old/new-task accuracy come from task
matrices. F1 and balanced accuracy come from raw classification outputs;
memory/GPU/parameters from resource fields; replay diagnostics from final-task
diagnostic outputs. Task matrices are never cited as sources for categories
they do not contain. Coverage and field semantics are listed in the CSV.
