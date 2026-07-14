# Final Baseline Plan

## 1. Baselines Already Covered

- Fine-tuning / None: covered by `classifier_real_only`.
- Pure generated replay: covered by `wgan_projection_generated_only`.
- MalCL-like internal baseline: covered as `malcl_like`, but not faithful external MalCL.
- Real-replay ablation: `real_buffer_only` exists, but it is not clean canonical ER.

## 2. Baselines That Need No New Run

- `classifier_real_only` for None/fine-tuning.
- `wgan_projection_generated_only` for pure generated replay.
- Existing `malcl_like` may be used as an internal baseline only.

## 3. Baselines That Need Only A Small Validation Rerun

- Canonical ER should be a small rerun if the existing notebook toggles can disable diversity/KD/prototype alignment cleanly.
- Joint should be straightforward once the same data-stream adapter is available, but it is still a new exact-protocol run.

## 4. Baselines That Require New Implementation

- DER++: new local runner and GPU runs required.
- Canonical ER: new clean replay setting required.
- Joint/offline upper bound: new exact-protocol upper-bound setting required.
- Optional iCaRL, LwF, and EWC: runners are scaffolded as secondary baselines.

## 5. Baselines That Require External Repository/Code Integration

- MADAR requires external source integration.
- Faithful MalCL likely requires external source integration or a careful paper-level reimplementation.
- FreeMOCA requires external source integration if used numerically.

## 6. Literature-Only Baselines

- FreeMOCA: strong related work and novelty-risk paper unless reproduced.
- GDumb: optional memory-only sanity baseline.
- SI: optional related-work-only unless reviewers ask for it.

## 7. Related Work Only

- Avalanche and Mammoth are implementation references, not competitors.
- Concept-drift and active-learning malware papers cited by MADAR/FreeMOCA should stay related-work-only unless they match the class-incremental family-classification protocol.

## 8. Novelty-Risk Papers

- MalCL: closest generative malware CL work.
- MADAR: closest distribution-aware malware replay work.
- FreeMOCA: recent replay-free malware CL work.

## 9. Minimal Q1 Baseline Suite

- Current covered baselines: None, pure generated replay, MalCL-like internal baseline.
- New exact-protocol baselines to add: DER++, MADAR, canonical ER, and Joint.
- Faithful MalCL is the next add if the paper needs the closest external generative replay reproduction rather than only MalCL-like internal evidence.

## 10. Strong Q1 Baseline Suite

- Minimal suite plus faithful MalCL, iCaRL, LwF, EWC, and FreeMOCA if integration time is available.

## 11. Exact Recommended Next Training Jobs

1. Fix/install the importable data-stream adapter and dependencies required by `baseline_suite/run_derpp_baseline.py`.
2. Run DER++ for EMBER K=100 and AZ-Class K=200 after dry-run passes.
3. Prepare MADAR integration, then run MADAR for EMBER K=100 and AZ-Class K=200 after source approval.
4. Run canonical ER for EMBER K=100 and AZ-Class K=200.
5. Run Joint upper bounds for EMBER and AZ-Class.
6. Decide whether faithful MalCL reproduction is needed after DER++/MADAR results are available.
7. Run iCaRL/LwF/EWC only if GPU time remains or reviewers require them.

## 12. GPU Priority Order

1. DER++.
2. MADAR setup and reproduction.
3. Faithful MalCL if `malcl_like` is judged insufficient for the venue.
4. Joint upper bound.
5. Canonical ER.
6. iCaRL, LwF, EWC.
7. FreeMOCA, GDumb, SI.

## 13. Go/No-Go Recommendation

- Go: use the current audit outputs to plan missing baselines.
- No-go for immediate training: DER++ dry-run must pass first, and MADAR external code must be available and approved.
- Go for paper drafting only if numerical comparison tables clearly separate existing internal ablations from unreproduced literature-only methods.

## Explicit Answers

- Is `classifier_real_only` enough for None? YES. Status: already covered.
- Is `real_buffer_only` enough for ER? NO. Status: partially covered; it is useful ablation evidence but not canonical ER.
- Is `wgan_projection_generated_only` enough for pure generated replay? YES. Status: already covered.
- Is `malcl_like` enough for MalCL? NO for faithful external MalCL. Status: needs faithful reproduction.
- Do we need DER++? YES.
- Do we need MADAR? YES.
- Should iCaRL/LwF/EWC/FreeMOCA be run now or only discussed? Discuss/scaffold now; run after mandatory DER++, MADAR, ER, and Joint unless GPU time is abundant.
- What is the smallest defensible set? Current None + pure generated + MalCL-like, plus new DER++, MADAR, canonical ER, and Joint.
- What is the strongest set? Smallest set plus faithful MalCL, iCaRL, LwF, EWC, and possibly FreeMOCA.
