# Anchored Hybrid Replay for Class-Incremental Malware Classification

V15.5 reproducibility release for the study “Anchored Hybrid Replay for Class-Incremental Malware Classification: Evaluating the Stability–Plasticity Trade-off.”

This repository consolidates the current analysis code, reproducibility records, validated outputs, and the V15.5 Standard-A runner source. V15.5 is the evidence boundary; no V16 results are included or pooled.

## What is included

- Analysis, protocol, and release-validation scripts.
- The 80-record main comparison across four methods, two datasets, and seeds 42–51.
- The 80-row four-way ablation and its per-seed analyses: 40 newly executed Standard-A rows validated on Tesla T4 and 40 validated historical rows reused.
- The 60-row quota-sensitivity analysis: 30 newly executed Tesla T4 rows and 30 validated historical rows reused.
- Task matrices, preserved non-sensitive source outputs, result tables, provenance manifests, figures, and validation tests.
- The exact V15.5 Standard-A source handoff under runners/v15_5_standard_a.

The ablation and quota-sensitivity results are provided as validated per-seed analyses and provenance records. The package does not include all original run directories, checkpoints, or raw GPU logs. Some manifest and result fields preserve original run-environment paths as provenance labels. They are not repository-relative paths; verify artifacts through the recorded hashes and packaged historical outputs.

## Reproduce the packaged analyses

Use Python 3.10 or later. From the repository root, install the analysis and verification requirements:

    python -m pip install -r requirements-verification.txt
    python -m pip install -r requirements-analysis.txt

Then run:

    python -m pytest -q
    python scripts/validate_protocol.py
    python scripts/verify_malcl_matrix_provenance.py
    python scripts/build_v15_5_release.py --analysis-only
    python scripts/build_stability_plasticity_figure_v15_5.py
    python scripts/validate_release.py

These commands validate and regenerate analyses from packaged evidence. They do not launch GPU training.

## Training-source boundary

The source tree at runners/v15_5_standard_a is the verified runner used for the 40 newly executed Standard-A ablation rows. Its combined source SHA-256 is:

    e239660453035ba93abe2a1de2517044fd492569f7da7c5714b06d900269f470

That handoff covers Current-only, Anchor-only at MAIN-K, and Generated-only rows. Full-Hybrid rows in that ablation are reused historical results. The handoff does not contain a quota-sensitivity training runner.

The main MalCL per-task matrices are preserved historical outputs. Their exact runner-to-matrix generation path was not recovered, so this release does not claim exact end-to-end training reproduction for every reported method or configuration. See RELEASE_METADATA_V15_5.md and reports/REPRODUCIBILITY_SCOPE_V15_5.md.

## Dataset access

The four prepared NPZ arrays are not included. Obtain them from the source MalCL Zenodo record, DOI 10.5281/zenodo.14537891, and run scripts/download_and_verify_datasets.py to check the published file hashes, shapes, and labels. See DATASETS.md and DATA_AVAILABILITY.md. The MIT license in this repository does not change rights attached to the source datasets.

## File guide

- analysis/, scripts/, tests/: analysis and validation code.
- runners/: available baseline/reference code and the verified V15.5 Standard-A runner.
- historical_outputs/, task_matrices/, analysis_outputs/, results/, evidence/: preserved inputs and derived outputs.
- manifests/, reports/, figures/: provenance records, technical audits, and current V15.5 figures.
- LICENSE and LICENSE_SCOPE.md: software license and its scope.
- RELEASE_QA.md: checks run while preparing this repository.
- THIRD_PARTY_NOTICES.md: third-party source and dataset notices.

This is the code and reproducibility repository, not the IEEE submission package. Manuscript PDFs and sources, author portraits, similarity reports, approval forms, private correspondence, datasets, credentials, and model checkpoints are not included.
