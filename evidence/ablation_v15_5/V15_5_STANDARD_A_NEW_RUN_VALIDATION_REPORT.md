# New Standard-A T4 run validation

- Expected / found: 40 / 40
- Validated: 40
- Invalid: 0
- All environment records identify Tesla T4.
- Registry status is `VALIDATED` for every new row; individual `status.json` files retain their pre-validation `COMPLETED_UNVALIDATED` state, while `VALIDATED_COMPLETE` and the registry are the final validation evidence.
- Resumed runs: 4. Each was retained after verifying a complete 11-row task sequence, finite matrix, and final validation marker; no resumed run was excluded merely for resuming.
- Validation includes hashes listed in each `checksums.sha256`, canonical class orders, isolation flags, task completion, sampler protocol, KD/prototype disablement, and replay semantics.

## Historical source-identity cross-check

All 40 approved historical rows match their `V15_5_FINAL_REUSE_MANIFEST.csv` source task-matrix SHA256 when hashed in the manifest’s compact raw representation (including historical `-1` future-task sentinel cells). The final canonical index instead hashes the harmonized representation with those future cells converted to null/undefined; this representation change is explicit and does not indicate a source mismatch.
