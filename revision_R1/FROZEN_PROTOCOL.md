# Frozen revision_R1 production protocol

This directory is the GitHub-delivered runner for the already-approved 100-run revision_R1 matrix. It is not a new experiment design and does not claim historical V15.5/V16 reproduction.

The runtime plan is `configs/production_run_plan_v1.json`. The five arms are `current_only`, `anchor_only`, `generated_only`, `full_hybrid`, and `derpp_matched`; the ten seeds are 42–51; both datasets use the 100-class, 11-task stream; each classifier update consumes 256 examples. Sampling is the agreed class-balanced mixed-pool inverse-frequency sampler with replacement. Replay is stored in raw coordinates and transformed using the taskwise training-only scaler. DER++ keeps its configured loss semantics.

The packaged parent update-budget CSV and its manifest remain unchanged and are integrity-checked. Production uses the separately preflighted effective schedule from the existing Drive preflight directory. Its required SHA-256 is:

`7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`

The first matrix entry is `ember_seed42_full_hybrid`. Its expected training-source identity is `1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`. A reporting-only Git commit may differ while this training-source identity and protocol remain compatible. Any source or protocol mismatch must fail validation before a production launch; the runner must never silently create a new run in place of an incompatible checkpoint.

The full frozen parameter values, per-run configurations, class orders, dataset fingerprints, and plan digest are recorded in the JSON plan. No training output, dataset, checkpoint, runtime credential, or cache belongs in this source package.
