# Temporal EMBER pilot audit report

**Status:** passed. **Selection:** upstream EMBER 2018 v2, used as a separate benchmark. The pilot does not claim to use the exact MalCL EMBER subset because no verified mapping from those prepared rows to sample hashes and observation dates was available.

## Source and integrity

- Archive: `ember_dataset_2018_2.tar.bz2` from [the official EMBER download](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2)
  - SHA-256: `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`
- Scope: six official train_features_*.jsonl members only; official test_features.jsonl excluded
- Rows scanned: 800,000; unique sample hashes: 800,000.
- Frozen split manifest SHA-256 (semantic): `00aeef3fda7e15f9a40c0f7143c532f7441231d95fd575bc969b675f470c2a1d`
- Compressed split manifest SHA-256: `5d8daecfee62f8dfa556db9b475428fa2362ac95e64f56d8f71cacd780417875`
- Frozen six-window update schedule SHA-256: `1d7ec52ff9988b6991106da05a9d96329c5179d71d1a934ce1fb431d4bce9170`

## Dates and identity policy

- Date field: `appeared` — month the sample was first seen, at month precision.
- Coverage in source rows: `2006-12` through `2018-10` across 139 distinct months; malformed/missing dates: 0.
- Identity: `EMBER sha256 field`.
- Exact repeats: collapse once; conflicting date/label/family: quarantine that identity; distinct hashes with equal vectors: retain distinct sha256 identities.
- Observed exact duplicate rows: 0; conflicting sample hashes: 0.
- No date is inferred from row order, class IDs, class sizes, or ZIP metadata. The official test member is excluded.

## Six-window family support

| Window | Months | Family-labeled rows | Distinct families | Train rows | Held-out rows |
|---|---|---:|---:|---:|---:|
| 2018-01 | 2018-01 | 30,723 | 913 | 24,485 | 6,238 |
| 2018-02–2018-03 | 2018-02, 2018-03 | 48,448 | 1,310 | 38,805 | 9,643 |
| 2018-04–2018-05 | 2018-04, 2018-05 | 47,591 | 1,198 | 38,068 | 9,523 |
| 2018-06–2018-07 | 2018-06, 2018-07 | 49,843 | 1,193 | 39,934 | 9,909 |
| 2018-08–2018-09 | 2018-08, 2018-09 | 56,298 | 1,481 | 45,136 | 11,162 |
| 2018-10 | 2018-10 | 56,343 | 393 | 45,158 | 11,185 |

The complete per-family training and held-out counts for every window are recorded in [`audit_report.json`](audit_report.json) under `family_support_by_window`. The frozen sample-level assignment and admissions are in [`split_manifest.json.gz`](split_manifest.json.gz).

## Admission interpretation

The stream is **chronological new-to-learner family admission**. A first appearance in the pilot training partitions is not a real-world family emergence date. January’s initial 50 are selected from January training support only. Later windows admit up to five families from the current window’s training partition; unselected labels are not backfilled.

The frozen admission cohort is:

- `2018-01`: installmonster, adposhel, zusy, fareit, emotet, dealply, dotdo, startsurf, high, mira, tiggre, upatre, gamehack, sality, zbot, wannacry, ausiv, filetour, razy, gamarue, wabot, eorezo, tinba, bladabindi, ramnit, installcore, icloader, delf, autoit, vobfus, qqpass, slimware, virut, ircbot, scar, yakes, archsms, poison, wdfload, darkkomet, berbew, ekstak, netsky, gepys, msilperseus, virlock, nymaim, lethic, vtflooder, cosmicduke
- `2018-02_03`: soft32downloader, pluto, gandcrypt, hematite, chapak
- `2018-04_05`: warezov, skypespam, coins, hploki, mokes
- `2018-06_07`: ribaj, yuner, czyk, upantix, resod
- `2018-08_09`: triusor, dfay, gjue, gify, dropware
- `2018-10`: spesr, offergenerator, brambul, cnbtech, diskscanfa

## Feature preprocessing

The official EMBER v2 `PEFeatureExtractor` is pinned at the source commit in [`PROVENANCE.json`](PROVENANCE.json). The published CoLLAs EMBER family preprocessing is a reference for features, month, and family labels; its randomized split is not reused. On current scikit-learn, a compatibility wrapper preserves the older `FeatureHasher(input_type="string")` character-wise behavior for singleton strings passed by upstream EMBER.

References: [official EMBER README](https://github.com/elastic/ember/blob/master/README.md), [CoLLAs malware repository](https://github.com/msrocean/continual-learning-malware), [CoLLAs EMBER family-processing script](https://github.com/msrocean/continual-learning-malware/blob/main/ember_data/EMBER_2018_TASK_CLASS_IL_FAMILY-CoLLAs-2022.py).

**No production training was launched during this audit.**
