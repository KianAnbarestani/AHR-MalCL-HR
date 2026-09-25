# Release QA

Audit date: 2026-09-25

## Passed checks

- The supplied handoff ZIP SHA-256 matched `257563b31f45bedde131918fbd0c60ce4133d104d284022f633dbc9b9ddfdbd6`; ZIP integrity passed and the archive contained 528 files. After merging the repository's existing ignore protections and updating this QA record, the 527-entry `SHA256SUMS.txt` was regenerated and verified; a synchronized 528-file ZIP was rebuilt, integrity-checked, and byte-compared after extraction.
- The V15.5 Standard-A source hash, computed by hashing each sorted `v15_5_runner/*.py` file and then `launch_v15_5_standard_a.py`, adding each basename followed by its raw bytes, matched `e239660453035ba93abe2a1de2517044fd492569f7da7c5714b06d900269f470`.
- Protocol validation passed: 100 classes, 11 tasks, seeds 42–51, full data, all-seen evaluation.
- MalCL provenance validation verified all 20 preserved CSV-to-matrix chains; the runner/postprocessor limitation remains documented.
- Fourteen direct checks in `tests/v15_5_checks.py` passed.
- Python `compileall` passed for packaged Python sources.
- On a disposable copy, the V15.5 analysis and figure rebuilds completed; all 13 compared analysis-output SHA-256 values remained identical.
- The release archive contains no prepared dataset arrays, model/checkpoint files, credentials/environment files, manuscript sources or manuscript PDFs, author portraits, similarity reports, approval forms, or private correspondence. The three included PDFs are the listed analysis figures.

## Test-suite result and limitations

- The full `python -m pytest -q` run was attempted by `scripts/validate_release.py` and stopped during collection. `tests/test_diagnostic_label_consistency.py` and `tests/test_replay_diagnostic_faithfulness.py` both import `check_notebook_diagnostics`, which is absent from the packaged `tests/v15_5_checks.py`; neither test executed. The helper found in a separate manuscript-side package expects two notebook files that are not part of this release, so those files were not copied in as a workaround.
- Excluding only those two collection-blocked tests, pytest reported **22 passed, 2 skipped**. The skips were the optional PyTorch topology check and the memory-heavy topology check requiring suitable GPU hardware. This partial result is not a full-suite pass.
- PyTorch and CUDA are unavailable in this environment. No GPU test or training run was performed.
- `scripts/validate_release.py` therefore did not finish end-to-end. Its static checks ran before pytest collection; its later validation and deterministic-rebuild steps were run independently on the disposable copy and passed, including protocol, provenance, analysis/figure rebuild, and all 13 output-hash comparisons.

## Preserved source-format findings

- The default staged `git diff --check` reports CR characters at CRLF line endings in packaged CSVs, as well as whitespace already present in the supplied source. With `core.whitespace=cr-at-eol`, 54 trailing-whitespace lines and 10 blank-at-EOF findings remain across 16 files, including the preserved upstream MalCL source and V15.5 runner files. These files are byte-identical to the verified handoff; they were left unchanged to preserve upstream source and the recorded Standard-A hash.

## Scope

These checks validate the packaged code and preserved evidence. They do not claim fresh GPU training, exact recovery of the historical MalCL runner-to-matrix path, or redistribution of source datasets/checkpoints. See `RELEASE_METADATA_V15_5.md` and `reports/REPRODUCIBILITY_SCOPE_V15_5.md` for those boundaries.
