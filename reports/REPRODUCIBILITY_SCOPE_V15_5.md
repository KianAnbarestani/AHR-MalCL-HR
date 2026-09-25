# Reproducibility scope V15.5

| Component | Evidence status | Verification included |
|---|---|---|
| Main comparison | 80 method–dataset–seed records and 80 task matrices | Metric derivation, support, task order, and provenance checks |
| Four-way ablation | 80 rows; 40 new Standard-A rows validated, 40 historical rows reused | Per-seed analysis, canonical row and hash manifests, paired tests |
| Quota sensitivity | 60 rows; 30 new Tesla T4 rows validated, 30 historical rows reused | Per-seed analysis, canonical row and hash manifests, paired tests |
| ER and DER++ runners | Import/configuration path verified | Import, CLI, and configuration checks |
| Adapted-MalCL task matrices | Preserved historical outputs only (Path D) | Exact source CSV-to-normalized-matrix comparison; runner-to-matrix path unavailable |
| V15.5 Standard-A runner | Exact Python source hash available for the 40 new ablation rows | Source identity and output validation |
| K-sensitivity training runner | Not present in the available project materials | Result analysis and validation artifacts are included |
| Complete end-to-end rerun of every historical training run | Not claimed | Analysis reproduction is supported from packaged evidence |

“Clean-room” describes analysis reproduction from preserved evidence. It does not mean that every historical GPU training run can be reproduced from this repository.
