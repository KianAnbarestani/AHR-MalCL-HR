# Temporal EMBER Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an independent six-window upstream EMBER 2018 v2 temporal pilot with exactly nine planned runs, causal splits, resumable checkpoints, compact reports, notebook, ZIP, and checksums.

**Architecture:** Keep all pilot code, data state, checkpoints, and outputs under `temporal_ember_pilot/`. Vendor the verified revision_R1 implementations unchanged and import them from a separate causal temporal runner; build an audit-created frozen manifest before any training. The Colab notebook exposes Setup → Audit → Run/Resume → Report and guards production training off by default.

**Tech Stack:** Python, NumPy, PyTorch, official EMBER v2 feature extraction, SQLite/JSONL audit, Jupyter notebook, ZIP/SHA-256.

**Spec:** `docs/superpowers/specs/2026-11-09-temporal-ember-pilot-design.md`

## Global Constraints

- Use upstream EMBER 2018 v2; official archive SHA-256 is `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`.
- Preserve six windows: Jan; Feb–Mar; Apr–May; Jun–Jul; Aug–Sep; Oct 2018.
- January’s initial 50 families come only from January training data; later admission means chronological new-to-learner family admission, not real-world emergence.
- Evaluate each later held-out window before updating on that window; held-out samples never enter training, replay, scaler fitting, or tuning.
- Freeze classifier updates from the planned Full Hybrid pool using `3 * ceil(pool / 256)` before training; actual GAN counts never alter budgets.
- Run exactly Full Hybrid, Anchor-only, Current-only × seeds 42, 43, 44; never launch production runs from Codex.
- Primary result is the equal-window mean across five post-January paired Full Hybrid minus Anchor-only pre-update macro-F1 differences; January post-fit is separate.
- Training is disabled by default; package state and paths are independent of the main matrix.

## Review Focus

- Malformed or conflicting sample identity/date/family rows must be quarantined and recorded; test with duplicate/conflict fixtures in Task 1.
- A late family label must not be backfilled into an earlier window or admission list; test with an adversarial chronological fixture in Task 1.
- A holdout or future row must not affect earlier scaler/training/replay inputs; test sample and scaler membership in Task 1.
- Mid-window interruption must restore window cursor, model/optimizer/scaler, and all RNG streams; test uninterrupted versus resumed equivalence in Task 2.
- Invalid or partial run state must not be skipped and paired metrics must use identical score IDs; test resume validation and report pairing in Task 3.

---

### Task 1: CPU audit and frozen causal split manifest

**Files:** Create `temporal_ember_pilot/audit.py`, `temporal_ember_pilot/stream.py`, `temporal_ember_pilot/tests/test_leakage.py`, and generated audit/manifest outputs. Do not modify `revision_R1/`.

**Interfaces:** `audit_archive(archive_path, output_dir) -> dict`; `build_manifest(records, split_salt=...) -> dict`; `window_partitions(manifest, window_id) -> dict[str, list[str]]`. Use official `appeared` month and `sha256` identity only.

- [x] Write tests for month-based windows, fixed seed-independent split assignment, conflicting identity quarantine, Jan-only top-50 support, no future-family backfill, and held-out/future exclusion from earlier update/scaler IDs.
- [x] Run `python3 -m unittest discover -s temporal_ember_pilot/tests -p test_leakage.py -v`; expected initial failures before audit/stream implementations.
- [x] Implement bounded streaming CPU audit over the six official train shards, sample identity policy, support tables, and deterministic compressed split manifest with its hash.
- [x] Run the leakage tests and audit the verified archive; expect all named assertions to pass and counts to match the design audit.
- [ ] Commit only Task 1 package files and the task ledger update.

### Task 2: Verified method bundle and separate temporal engine

**Files:** Vendor unchanged method modules under `temporal_ember_pilot/vendor/revision_r1/`; vendor official EMBER `features.py` under `vendor/ember/`; create `config.py`, `data.py`, `engine.py`, `runner.py`, `checkpoint.py`, `progress.py`, and `tests/test_resume.py`.

**Interfaces:** `planned_pool_counts(current_n, prior_family_support) -> dict`; `planned_updates(current_n, prior_family_support) -> int`; `run_window(state, window, *, updates, train_batch, checkpoint_path, finalize_window=None, stop_after_batches=None) -> state`; `save_checkpoint(path, state)` / `load_checkpoint(path) -> state`. `finalize_window` writes model, optimizer, scaler, anchor, class, memory, and metric state before the single atomic checkpoint. Checkpoint includes all global and named RNG streams.

- [x] Write a synthetic interruption test comparing uninterrupted and stop/resumed model state, optimizer state, next RNG draws, and completed-window cursor; assert each window’s planned budget is arm/seed invariant.
- [x] Run `python3 -m unittest discover -s temporal_ember_pilot/tests -p test_resume.py -v`; expected initial failures before the engine/checkpoint existed.
- [x] Copy verified model, preprocessing, sampler, GAN, loss, metric, config, RNG, and progress sources byte-for-byte and record their hashes; implement three-arm causal window updates using only prior and current training rows.
- [x] Run the resume and budget tests; expect identical synthetic state after resume and one checkpoint per completed window.
- [ ] Commit only Task 2 files and the task ledger update.

### Task 3: Paired metrics and report export

**Files:** Create `temporal_ember_pilot/report.py` and `temporal_ember_pilot/tests/test_report.py`.

**Interfaces:** `export_report(manifest, run_records, output_dir) -> dict`; report JSON and CSV include per-window counts/macro-F1, January post-fit, old-family retention, new-family acquisition with support, planned/realized replay, GAN compute/wall time, paired seed deltas, and primary equal-window mean.

- [x] Write synthetic report tests for same scoring IDs across arms, five post-January per-seed differences, equal window weighting, separate January score, sparse-family exploratory flag, and October acquisition unavailable.
- [x] Run `python3 -m unittest discover -s temporal_ember_pilot/tests -p test_report.py -v`; expected initial exporter import failure.
- [x] Implement deterministic compact JSON/CSV export and compressed frozen manifest sidecar; reject incomplete or non-paired primary comparisons.
- [x] Run report tests; expect paired values and sample counts to match hand-computed fixtures.
- [ ] Commit only Task 3 files and the task ledger update.

### Task 4: Notebook, launch package, smoke verification, and release artifacts

**Files:** Create `temporal_ember_pilot/temporal_ember_pilot.ipynb`, `README.md`, `PROVENANCE.json`, `audit_report.json`, packaging/checksum files, and `tests/test_package_smoke.py`.

- [x] Write a synthetic end-to-end smoke test covering manifest input, one window checkpoint, interruption/resume, and report export; assert notebook has four ordered sections and `ENABLE_TRAINING = False`.
- [x] Run the smoke test and notebook/package checks before implementation; expect failures while release artifacts are absent.
- [x] Build the notebook and launch instructions, official archive/source references, audit report, exact nine-run config validation, and provenance. ZIP and SHA-256 files pending. Do not run production training.
- [ ] Run the smoke, full pilot test set, checksum validation, notebook structure check, and inspect package contents for main-matrix paths; expect all checks pass and no production state is included.
- [ ] Perform a final author self-review, record the result in the ledger, and commit only Task 4 pilot files and ledger update.
