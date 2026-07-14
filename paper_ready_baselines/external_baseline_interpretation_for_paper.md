# External Baseline Interpretation For Paper

## Scope

These notes summarize the integrated external baseline evidence from ER, DER++, and Joint runs under `result_external_baselines/`. AHR-MalCL-HR is the main method, ER and DER++ are strong memory/replay baselines, and Joint is an offline upper bound.

All paired differences use `AHR-MalCL-HR - baseline`. For forgetting and memory, lower is better. For accuracy, F1, and balanced accuracy, higher is better.

## Safe Claims

- On EMBER, AHR-MalCL-HR has lower forgetting by mean than ER and DER++, but ER/DER++ have higher mean final accuracy, macro-F1, weighted-F1, and balanced accuracy.
- On AZ-Class, AHR-MalCL-HR has better mean final accuracy, forgetting, and balanced accuracy than ER and DER++.
- No Holm-corrected external superiority claim for AHR-MalCL-HR is currently supported.
- Joint is an offline upper bound, not a fair memory-limited CL baseline.

## Unsafe Claims

- Do not claim AHR-MalCL-HR beats ER/DER++ on EMBER accuracy.
- Do not claim AHR-MalCL-HR is memory-efficient compared with ER/DER++.
- Do not claim AHR-MalCL-HR beats Joint.
- Do not claim universal external superiority.

## Recommended Final Framing

"AHR-MalCL-HR is not uniformly accuracy-dominant against strong replay baselines. Instead, it offers a replay-drift-aware hybrid replay mechanism that improves retention/forgetting behavior and remains competitive with ER and DER++, with stronger evidence on AZ-Class and retention-oriented gains on EMBER."

## Final Recommendation

- MADAR is still recommended if feasible because it is malware-specific.
- If MADAR cannot be reproduced, discuss it as literature-only and clearly state protocol differences.
- MalCL-faithful reproduction is also recommended if implementation time allows; otherwise keep malcl_like as internal approximation only.

## Table Files

- `paper_ready_baselines/table_external_baselines.csv`
- `paper_ready_baselines/table_external_baselines.md`
- `paper_ready_baselines/table_external_baseline_paired_claims.csv`
- `paper_ready_baselines/table_external_baseline_paired_claims.md`
