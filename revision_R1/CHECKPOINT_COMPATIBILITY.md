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

## Taskwise checkpoint policy

- The matrix uses one Drive `matrix_progress.json`, with all 100 frozen runs in
  plan order and eleven task entries per run. A task is `done` only after its
  full-state checkpoint has been written and verified in Drive.
- Each new run writes exactly one full-state checkpoint at the end of each
  completed task. The checkpoint includes model, optimizer, RNG, scaler,
  replay, sampler, cursor, and accumulated task state needed for the next task.
- Save order is checkpoint first, then atomic local progress replacement and
  Drive progress publication. A failure before progress publication leaves
  the task unfinished there; a stop during a task repeats only that task.
- All eleven task checkpoints are retained. There are no initial/mid-task
  checkpoints, leases, heartbeats, expiry handling, recovery generations,
  timed publication, or checkpoint rotation in the Colab taskwise path.
- After Task 11, result artifacts publish first and `artifact_manifest.json`
  publishes last. Only after successful result publication is the same progress
  record updated to mark the run complete. Export-only retry resumes from the
  Task 11 checkpoint and performs no training updates.
- Run 1 remains imported from its existing validated completion registry.
  The clean replacement Run 2 uses the `taskwise_v1/<plan_sha256>/` namespace;
  abandoned legacy Run 2 artifacts are neither read as completion nor altered.
- Missing or unloadable recorded checkpoint files fail closed with run, task,
  and logical path. The runner never silently starts a replacement from Task 1.

These changes do not modify model/optimizer configuration, data sampling,
RNG streams, fixed update budgets, or the frozen schedule. The notebook must
rerun current package/identity verification and checkpoint discovery before
resuming. It must not create a fresh run if the exact current checkpoint is
missing or incompatible.
