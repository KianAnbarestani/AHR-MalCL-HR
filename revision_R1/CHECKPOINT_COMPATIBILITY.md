# Checkpoint identity and runtime-source compatibility

The existing production checkpoint identity is
`1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`.
It is the exact source digest produced by commit
`e7bea685827137a1ea48c5886db22518d68a6b06`, which remains the baseline for
the first Full Hybrid run. The checkpoint patch changes the current runtime
source digest; the old digest is not presented as a hash of the new files.

`checkpoint_compatibility.json` binds both identities. At runtime, the package
verifies every current `revision_r1/*.py` hash, the combined runtime digest,
the frozen baseline digest, and the training-phase AST fingerprint. The
build-time compatibility test reads the baseline files directly from the
named Git commit, verifies the baseline's exact historical digest, confirms
that only `checkpointing.py`, `resources.py`, and `runner.py` changed among
existing runner modules, and checks the unchanged training-phase fingerprint.
Only then may the runner keep the existing checkpoint identity. Any other
source/config/data/schedule mismatch fails closed; no checkpoint is rewritten
or silently restarted.

The package manifest reports `training_source_sha256` for the historical
checkpoint identity and `runtime_source_sha256` for the actual new code. The
notebook prints and binds both separately from its selected Git commit.

## Checkpoint policy

- Local recovery serialization: completion-based monotonic timer, every 300
  seconds; task boundaries and graceful interruption are forced.
- Drive recovery publication: completion-based monotonic timer, every 900
  seconds; task boundaries and graceful interruption are forced.
- The frozen RunConfig's older cadence fields are retained unchanged as part
  of the resume identity; the 300/900-second operational override is recorded
  separately in resource metrics and does not alter the config hash.
- A periodic Drive publication copies only the newly-created recovery
  generation. It does not resend the existing recovery pair. Startup verifies
  the newest local generation first and stages only a newer valid Drive
  generation; if the newest Drive generation is damaged, metadata/hash
  fallback selects the preceding valid generation. The selected full state is
  deserialized once.
- Recovery directories retain two hash-verified generations. A new Drive
  generation is staged and verified before the previous generation is
  pruned. Pruning uses exact owned Drive file IDs with permanent deletion
  (not Trash); a failed publish or unverifiable owner prevents pruning. Cell C
  must verify that the authenticated account can permanently delete its own
  probe file before a run can rely on this rotation policy. If the running
  account does not own an older recovery file, the exact-file ownership gate
  refuses to prune it; no unrelated or foreign-owned file is touched.
- The rotation applies only to the `recovery` directory. Existing task-boundary
  artifacts remain separately retained and are never pruned by the recovery
  policy. Existing files already in Drive Trash are not automatically
  permanently deleted; they continue to count against quota until an account
  owner handles them separately.
- Required task-boundary artifacts remain a separate unpruned archive. Their
  generation number continues from Drive manifest metadata after runtime
  restart, without downloading the archive payloads.
- A serialization/publish failure does not initiate a second full-state save.
- Run outputs separate optimizer/training seconds from snapshot capture,
  serialization, upload, verification, pruning, and artifact-copy time; they
  report retained bytes and cumulative successful payload/manifest write bytes
  separately. Historical cumulative traffic before this build is explicitly
  marked unknown and is not inferred from current retained files. Component
  times are nested within total checkpoint wall time and are not additive.

These changes do not modify model/optimizer configuration, data sampling,
RNG streams, fixed update budgets, or the frozen schedule. The notebook must
rerun current package/identity verification and checkpoint discovery before
resuming. It must not create a fresh run if the exact current checkpoint is
missing or incompatible.
