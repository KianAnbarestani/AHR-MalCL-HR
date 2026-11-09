# Temporal EMBER pilot design

## Goal and boundaries

Build a standalone `temporal_ember_pilot/` package and matching Colab notebook for a separate, nine-run temporal benchmark. The active `revision_R1` randomized matrix remains the source of verified model, method, configuration, and progress behavior. No main notebook, runner, schedule, checkpoint, progress registry, or Drive path is changed or reused as pilot state.

The pilot is an upstream EMBER 2018 v2 benchmark because the prepared MalCL EMBER NPZ has 303,331 feature rows but no verified row-to-sample/date mapping. Historical MalCL artifacts are preserved outputs without the recovered generation chain needed to establish that mapping. The upstream source is not represented as the exact MalCL subset.

## Source audit

- Official source: Elastic EMBER archive `ember_dataset_2018_2.tar.bz2`, SHA-256 `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`.
- The CPU audit streamed only the six official `train_features_*.jsonl` members; it did not extract or use `test_features.jsonl`. It read 800,000 rows, found 800,000 unique consistent sample hashes, zero duplicate rows, zero conflicting hashes, zero missing hashes, and zero malformed JSON records.
- Use only records with `label == 1`, a non-empty `avclass`, and `appeared` within Jan–Oct 2018. `appeared` is the sample's month first seen, at month precision; no row order, class ID, class count, or ZIP metadata is used as time.
- The source training shards cover 2006-12 through 2018-10. Family-labeled malware support for the ten 2018 months is: Jan 30,723; Feb 29,368; Mar 19,080; Apr 26,061; May 21,530; Jun 23,963; Jul 25,880; Aug 20,590; Sep 35,708; Oct 56,343. These records contain 2,731 distinct family labels across the period. The official test member is excluded.
- Row identity is `sha256` from the EMBER record. Collapse exact repeated identities once; quarantine an identity with conflicting date, label, or family metadata; retain distinct hashes even if their feature vectors match. The audit saw no duplicates/conflicts, but the policy is enforced by the package.
- Published CoLLAs preprocessing is used only as a reference for EMBER v2 features, month field, and `avclass`. Its randomized monthly split is not reused.

## Stream and leakage controls

Freeze six contiguous windows, without inferring sub-month order:

1. Jan 2018 (initial training window)
2. Feb–Mar 2018
3. Apr–May 2018
4. Jun–Jul 2018
5. Aug–Sep 2018
6. Oct 2018

Within each month, assign each sample by a deterministic SHA-256 rule independent of model seed: `SHA256("AHR-MalCL-temporal-ember-split-v1:" + sample_sha256)`, first byte `< 51` is held out, otherwise train. The split manifest is generated and hashed before any training. It records sample-hash, month, split, family, cohort status, and exclusion reason; a compact report embeds the manifest hash and per-window counts.

The benchmark uses **chronological new-to-learner family admission**, not a full-period top-family list. Pilot admission does not claim the family's real-world emergence date:

- January's initial cohort is 50 families selected from January training data only, ranked by support with family name as a deterministic tie-break.
- At each later window, consider only family labels present in that window's training partition and absent from all earlier training partitions. Admit up to five families by current-window training support, with the same tie-break. Then mark every family label from that training partition as observed so an unselected family cannot be backfilled from an earlier window.
- This yields at most 75 modeled classes over the six windows. It uses no future family list, test label, or test sample to select classes. The stream is labeled `chronological new-to-learner family admission`; no fixed-family `temporal drift` mode is needed because the source has ample family labels and later-window arrivals.
- The audit's deterministic prototype selected five new-to-learner labels in each of the five update windows. It found 103, 29, 3, and 10 held-out samples from preceding-window additions in the following windows, respectively, so acquisition is measurable but can have sparse support. Report acquisition with per-family test counts, flag sparse support as exploratory, and report October additions as unavailable because there is no later window.

At every window, the runner evaluates its held-out portion before using any training portion from that window. After labels are available, it updates once using only that window's training rows and past replay. January held-out samples are evaluated after the initial Jan fit and reported separately. No held-out sample enters training, anchors, GAN inputs, preprocessing, or the next checkpoint. New-family acquisition is macro-F1 on families introduced in the preceding training update, evaluated on the next window; old-family retention is macro-F1 on the initial 50 families. For every pre-update comparison, all arms use the exact same held-out sample IDs and the same known-family scoring set. Report total held-out, scored, unknown/unscored counts, and per-family acquisition support.

At each update cutoff, fit/update the verified training-only preprocessing on training rows available through that cutoff. Keep raw anchor rows and transform them under the current fitted preprocessing. The runner receives only the current window plus prior training/replay state; a leakage test will verify sample disjointness, temporal monotonicity, scaler fit membership, and that future partitions are not passed to update functions.

## Methods, runs, and state

Vendor the verified `revision_R1` model, preprocessing, sampler, GAN, metric, configuration, and progress source files byte-for-byte with their source SHA-256 values. Add a small temporal runner that imports these implementations and settings; do not vendor or modify the task-fixed production runner as the pilot runner.

Production matrix, exactly nine runs:

- Arms: `full_hybrid`, `anchor_only`, `current_only`.
- Seeds: 42, 43, 44. Seeds affect initialization, minibatch sampling, and GAN randomness only; split, window order, and family admission are frozen and seed-independent.
- Each method uses the same frozen windows, batch size 256, classifier architecture/settings, and classifier update count at each window. Before training, freeze the update count from the **planned** Full Hybrid pool: planned current rows + planned real anchors + the configured generated quota, using the verified three-epoch-equivalent rule. Do not let actual GAN output counts change the update count by arm or seed. Report planned and realized replay counts separately.
- Full Hybrid and Anchor-only use identical real-anchor selection and a quota of 100 per family. Full Hybrid uses the verified GAN settings, 100 generated examples per known family, and 1,000 candidates per family. GAN compute seconds and whole GAN-stage wall seconds are separate fields. Current-only uses current-window rows only.
- No pilot output, checkpoint, or progress record is written under the main matrix's experiment IDs or storage paths.

Use a separate experiment ID, `ahrmalcl_temporal_ember18v2_newfamily_v1`, with a pilot-only output root, one atomic checkpoint per completed window, and a pilot-only progress JSON. Resume from the last completed window and skip a run only when its final checkpoint and result validate. Progress renders one updating line with run, window, phase, epoch, and batch; phase completion emits one newline.

Each window checkpoint contains the stream cursor, classifier and GAN state, optimizer state, scaler, anchor state, class map, and Python/NumPy/Torch CPU and CUDA random states. Stopping mid-window reloads the prior completed-window checkpoint and replays that window from its saved random state. The synthetic smoke compares uninterrupted with interrupted/resumed execution for final training state, cursor, and next random draws.

## Notebook, report, and validation

The matching notebook has exactly four operator sections: Setup → Audit → Run/Resume → Report. Setup creates a separate pilot Drive/output path and verifies the source archive hash. Audit is CPU-only and freezes the manifest. Training is guarded by `ENABLE_TRAINING = False` and an explicit experiment-ID check. Report exports one compact JSON/CSV summary, a compressed split-manifest sidecar, checksums, per-window macro-F1 and counts, old-family retention, measurable new-family acquisition, and Full Hybrid minus Anchor-only per-window differences for each paired seed. The primary result is the equal-weight mean of the five post-January pre-update macro-F1 differences, Full Hybrid minus Anchor-only, with all three paired-seed values shown. January's post-fit score is separate. Mixed and negative results remain unmodified.

Before handoff, run the requested future-data leakage checks and a synthetic-only end-to-end smoke through checkpoint write, stop/resume state equivalence, and report export. Do not run any production matrix job from Codex. Deliver the package ZIP, standalone notebook, SHA-256 checksum files, audit report, README launch steps, and a separate pilot provenance manifest.

## External references

- EMBER source and archive description: https://github.com/elastic/ember
- EMBER 2018 v2 archive reference: https://github.com/elastic/ember/blob/master/README.md
- CoLLAs preprocessing reference: https://github.com/msrocean/continual-learning-malware
- CoLLAs family preprocessing script: https://raw.githubusercontent.com/msrocean/continual-learning-malware/main/ember_data/EMBER_2018_TASK_CLASS_IL_FAMILY-CoLLAs-2022.py

## Design decision

Use vendored verified model/config/progress source with a separate temporal runner. Reusing the production runner would carry fixed 11-task/100-class assumptions into a different temporal stream; rewriting the methods would lose source provenance. The user selected the separate-runner approach. This document is the pre-implementation design for review.
