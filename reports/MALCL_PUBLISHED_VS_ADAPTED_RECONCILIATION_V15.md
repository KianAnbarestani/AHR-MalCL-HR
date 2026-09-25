# MalCL Published vs Adapted Reconciliation V15

## Sources checked

- Park et al., AAAI 2025, DOI 10.1609/aaai.v39i1.32047.
- Official repository revision recorded by the project: `2cfb344461f594877e24fd4d19919b6bca38c861`.
- V14.1.2 patched topology tests, run configs, task metrics, memory records, and 20 final raw runs.

## Preserved official elements

The adapted implementation retains a classifier/GAN workflow, feature-matching generator loss, hidden-representation replay selection (`L1_C_Mean`), `k=3`, and the feature-length-dependent flattened discriminator topology. The topology test confirms that no adaptive-pooling layer replaced the flattening path.

## Shared-protocol adaptations

Input feature dimensions and prepared NPZ files; incremental scaler context; 100-class 11-task construction; random matched class orders; classifier/task training loop; three epochs and batch size 256; ten runs; final seen-class evaluation; and release memory instrumentation follow this study. The shared protocol uses prepared EMBER/AZ-Class arrays rather than recreating the official paper's complete experimental pipeline.

## Why published and adapted values differ

The MalCL paper highlights average accuracy under its own class order, preprocessing, training, and reporting definitions. The manuscript's main table reports final taskwise accuracy after Task 11 under a different shared protocol. Average-over-time/average-task accuracy and final taskwise accuracy are not interchangeable. The adaptation also changes feature dimensions, class ordering, run count, classifier/task loop, and training budget. A numerical subtraction between the proposed method and the official published value would therefore be invalid.

## Allowed claim

The proposed method has higher final taskwise accuracy than the **protocol-matched adapted MalCL implementation under the shared evaluation protocol**. The release does not claim an unchanged official reproduction or superiority over the published MalCL system.
