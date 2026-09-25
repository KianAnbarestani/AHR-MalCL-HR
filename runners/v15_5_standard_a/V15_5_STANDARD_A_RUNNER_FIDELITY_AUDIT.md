# V15.5 Standard-A runner fidelity audit

## Scope and source

The executable package was compared directly with the preserved V15.5 final runner:

- `paper_ready/AHR_MalCL_HR_reproducibility_release_v15_5/runners/ahr_final_ember_one_cell.ipynb`
- source SHA256: `f237d6d4ce1186d45aa04d8242740b44f56cd70e514b951af11f1896b01421df`

The AZ-Class notebook was also checked (SHA256 `21706565b6f940db8942be3bff0dec202a9a467d251f8ca3b76f54d5ca53b1ed`). Dataset-specific feature dimensions and MAIN-K are configuration differences, not model-family changes. No V16 source is imported.

## Behavior comparison

| Behavior | Classification | Finding |
|---|---|---|
| Classifier topology | IDENTICAL | Tabular MLP 1024/512/256, each with Linear/LayerNorm/GELU/Dropout(0.25), followed by a 512-unit GELU embedding and dropout. |
| Model initialization order | IDENTICAL | V15.5 constructs generator and critic before the classifier even when GAN use is disabled. The package preserves that RNG-consuming order, then discards inactive GAN models before training. |
| Expanding head | IDENTICAL | Arrival-index head; old weights/biases copied, new weights Xavier-uniform, new biases zero. |
| Classifier optimizer | IDENTICAL | AdamW is recreated for every task, learning rate 1e-3, weight decay 1e-4. No persistent V16 optimizer. |
| Classifier clipping | IDENTICAL | Global gradient norm clipped to 10 before each optimizer step. |
| Classifier AMP | IDENTICAL | Disabled, matching `USE_AMP_CLF=False` in the final V15.5 runner. |
| Class-balanced sampler | IDENTICAL | Native `WeightedRandomSampler`, inverse class frequency, `num_samples=len(pool)`, replacement, batch 256, `drop_last=False`, three epochs. |
| Standard-A exposure | IDENTICAL | Epoch length follows each component-specific pool. No update/draw/compute equalization is introduced. |
| Incremental scaler | IDENTICAL | Chunked population-variance update on current-task rows before training, then current scaler transforms current and carried raw replay. |
| Random anchor update | EQUIVALENT_REIMPLEMENTATION | Strict post-evaluation random selection for arriving classes with `RandomState(12345 + seed + task_index)` and no revisiting old raw data. Iterating only new classes is equivalent because retained old buffers are unchanged and consume no RNG. |
| Generator architecture | IDENTICAL | Conditional embedding, two FC layers, three ConvTranspose1d layers, and historical BatchNorm/GELU path. This is not the V16 MLP GAN. |
| Critic architecture | IDENTICAL | Three-layer 1-D convolutional projection critic with unconditional and class-projection scores. |
| GAN optimizer/lifecycle | IDENTICAL | Adam optimizers recreated per task, lr 2e-4, betas (0.0, 0.9), WGAN-GP lambda 10, five critic steps per generator step, three epochs, norm-10 clipping. |
| Feature matching | IDENTICAL | L1 matching of mean critic features with weight 1.0. This belongs to the historical generated component and is not a classifier diversity auxiliary. |
| Generated replay timing | IDENTICAL | Task t trains on carried replay from t-1; replay from the updated generator is selected only after evaluation for task t+1. No fresh same-stage V16 replay. |
| Generated selection | IDENTICAL | 1,000 candidates/class; L1 distance to class-mean hidden representation plus diversity lambda 0.25; 100 selected/class. |
| Selector representation | IDENTICAL | The preserved classifier's `return_mid=True` returns the 256-dimensional pre-embedding hidden vector even though the classifier additionally contains a 512-dimensional embedding. The package preserves that executable behavior rather than silently changing selection to 512 dimensions. |
| Generated-only mean data | EQUIVALENT_REIMPLEMENTATION | New classes use at most 512 randomly selected current examples; old classes use carried generated replay in strict mode, with seed `seed + 999 + task_index`. No real anchor access. |
| Task ordering | IDENTICAL | `RandomState(seed).permutation(100)`, then 50 classes and ten groups of five. |
| Evaluation timing | IDENTICAL | All seen task groups are evaluated after classifier training and before post-task anchor/replay update. |
| Current-only removals | INTENTIONALLY_REMOVED_COMPONENT | Anchors, generation, critic, feature matching, KD, prototype alignment, and diversity auxiliary are inactive. |
| Clean Anchor-only removals | INTENTIONALLY_REMOVED_COMPONENT | Generated replay/GAN, KD, prototype alignment, and diversity auxiliary are inactive; only current data and random real anchors remain. |
| Generated-only removals | INTENTIONALLY_REMOVED_COMPONENT | Real anchor memory, KD, prototype alignment, and classifier diversity auxiliary are inactive. Historical generated replay/GAN/FML/selection remain. |
| Checkpointing and source accounting | INSTRUMENTATION_ONLY | Atomic task-boundary state/RNG capture and realized sampler source tags do not alter pools, sampling probabilities, model updates, or evaluation order. |

## Resolution

`runner_fidelity: PASS`

There are no unexplained `MISMATCH` classifications for any component used by the three executable variants. The implementation deliberately preserves the historical 256-dimensional selector input exposed by `return_mid=True`; describing that path as a 512-dimensional selection representation would contradict the authoritative executable runner.
