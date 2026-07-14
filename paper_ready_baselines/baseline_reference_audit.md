# Baseline Reference Audit

The local reference pack was inspected for method role, fair adaptation, and runnability. Published numbers are not treated as comparable unless the exact protocol is reproduced.

| reference | local_file | present | audit_use |
| --- | --- | --- | --- |
| Avalanche | baseline_reference_pack/papers/avalanche_2023.pdf | YES | Framework reference, not a numeric competitor. |
| DER++ | baseline_reference_pack/papers/der_2020.pdf | YES | Mandatory generic replay + stored-logit distillation baseline. |
| EWC | baseline_reference_pack/papers/ewc_2017.pdf | YES | Classic regularization baseline. |
| FreeMOCA | baseline_reference_pack/papers/freemoca_2026.pdf | YES | Recent malware-specific replay-free novelty-risk paper; literature-only unless reproduced. |
| GDumb | baseline_reference_pack/papers/gdumb_2020.pdf | YES | Optional memory-only sanity baseline. |
| iCaRL | baseline_reference_pack/papers/icarl_2017.pdf | YES | Strong class-incremental exemplar/prototype baseline. |
| LwF | baseline_reference_pack/papers/lwf_2016.pdf | YES | Strong replay-free distillation baseline. |
| MADAR | baseline_reference_pack/papers/madar_2025.pdf | YES | Mandatory malware-specific distribution-aware replay comparator; needs external code. |
| MalCL | baseline_reference_pack/papers/malcl_2025.pdf | YES | Closest malware-specific generative replay baseline; decide faithful reproduction need. |

## Notes

- MalCL, MADAR, DER++, iCaRL, LwF, EWC, FreeMOCA, GDumb, and Avalanche PDFs are present locally.
- `baseline_reference_pack/notes/baseline_reference_manifest.md` and `baseline_reference_pack/code_links/baseline_code_links.md` identify code links for method understanding only; no repository was cloned or downloaded.
- No published paper number should enter the main numeric comparison table without exact-protocol reproduction.
- Missing required PDFs: none.
