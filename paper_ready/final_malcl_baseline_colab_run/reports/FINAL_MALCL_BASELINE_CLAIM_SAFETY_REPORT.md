# Final MalCL Baseline Claim Safety Report

This local package is not a completed MalCL baseline result. It is the final Colab runner package.

Safe only after Colab returns `MALCL_BASELINE_COMPLETE_READY_FOR_REVIEW`:

- We reproduced a protocol-matched MalCL baseline candidate under the AHR-MalCL evaluation stream.
- Results are subject to documented implementation deviations.

Unsafe unless later evidence supports it:

- AHR beats MalCL.
- MalCL is fully reproduced exactly.
- One seed proves superiority.
- Memory fairness is established.
- A partial or debug run is paper-usable.
- The T4-memory-safe discriminator head is an exact official MalCL reproduction.

Each dataset/seed `full-result.json` keeps `paper_claim_allowed=false`. The aggregate decision decides whether the MalCL baseline can be reviewed for paper use.
