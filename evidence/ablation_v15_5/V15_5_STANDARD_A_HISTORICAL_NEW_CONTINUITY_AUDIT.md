# Historical-to-new semantic continuity audit

## Classification: SEMANTICALLY_CONTINUOUS

The completed 80-row merge is semantically continuous with the V15.5 native Standard-A protocol. This is not an upgrade of historical provenance: Full Hybrid remains `EXACTLY_COMPATIBLE`, while the reused historical Current-only and Generated-only records remain `COMPATIBLE_WITH_RECONSTRUCTED_EVIDENCE`.

| Aspect | New Standard-A T4 corpus | Historical reused corpus | Assessment |
|---|---|---|---|
| Classifier | MLP [1024, 512, 256], embedding 512, expanding arrival-index head | Preserved/reconstructed V15.5 semantics | Compatible |
| Optimizer lifecycle | AdamW recreation, lr 1e-3, weight decay 1e-4, clipping norm 10 | V15.5 evidence | Compatible |
| Sampler and classifier training | WeightedRandomSampler, batch 256, three epochs | V15.5 native protocol | Compatible |
| Scaling/evaluation | Incremental scaler; task-boundary evaluation | V15.5 evidence | Compatible |
| Generated replay | WGAN-GP/projection critic, 5 critic steps, class quota, carried replay | Historical generated evidence | Compatible with reconstructed evidence |
| Auxiliary losses | KD, prototype alignment, diversity auxiliary disabled | Required V15.5 baseline semantics | Compatible |
| Class order | New literal vectors exactly match `numpy.random.RandomState(seed).permutation(100)` | Literal historical vectors unavailable in some controls | `INFERRED_FROM_SEEDED_PROTOCOL`, not fabricated as exact |

## Resumed new executions

Four new rows record one or more `RESUME next_task=` events. All 40 new rows passed final registry validation, have an 11-row 1–11 task sequence with no duplication, complete finite matrices, valid checksums, and `VALIDATED_COMPLETE`. The compact transferred result corpus does not preserve a separately readable checkpoint payload for every run, so the internal checkpoint tensors cannot be independently deserialized after transfer; continuity is established from the serialized final outputs, task/exposure sequence, resume logs, and final validation markers. No resumed row was excluded merely because it resumed.
