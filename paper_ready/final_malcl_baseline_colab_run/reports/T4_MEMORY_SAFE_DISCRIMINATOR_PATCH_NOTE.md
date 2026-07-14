# T4 Memory-Safe Discriminator Patch Note

The final Colab package uses a protocol-matched adapted MalCL discriminator for T4 execution.

The official MalCL discriminator flattens a `256 x feature_dim` convolutional feature map and feeds it into a 1024-unit fully connected layer. For EMBER and AZ-Class, this creates an impractically large discriminator head for a full 20-run Colab T4 baseline.

The package keeps:

- the MalCL generator role and `z_dim=62`
- the discriminator convolutional front-end
- the discriminator real/fake training role
- the feature-matching generator loss path
- the classifier role and expansion stream
- the AHR-MalCL class-order/task protocol

The package changes:

- the discriminator feature map is pooled with `AdaptiveAvgPool1d(1)`
- the discriminator head receives the pooled 256-dimensional feature vector

Claim-safety implication:

This should be described as a protocol-matched, T4-memory-safe MalCL baseline candidate with documented implementation adaptation. Do not call it an exact official MalCL reproduction.
