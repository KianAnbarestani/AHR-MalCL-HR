# revision_R1 production Colab handoff — GitHub-pinned source

The production source is fetched from `KianAnbarestani/AHR-MalCL-HR` at the
explicit full commit SHA in the stable notebook
`My Drive/ReVision/Revision_R1_Production_Colab.ipynb`. The current repository
is public; no GitHub token is needed. If the owner later makes it private,
Cell A accepts a fine-grained read-only token from the Colab Secret
`GITHUB_READ_TOKEN` (Contents: Read-only, restricted to this repository,
with notebook access enabled). The token is never placed in notebook source,
output, a saved Git remote, Drive, or the repository.

Cell A fetches once, checks out the selected commit detached into a
commit-versioned local directory, and verifies `PACKAGE_MANIFEST.json`,
`BUILD_MANIFEST.json`, all listed file hashes, training-source identity
`1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`, and
the required effective schedule SHA-256
`7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`. A
reporting-only commit is bound separately from the training/checkpoint
identity; a changed training core or frozen protocol is rejected rather than
silently starting a new run. The checkout is reused for the entire session.

The Colab account can be any Editor of the existing shared ReVision folder.
Cell C verifies write capability and reports the owner and quota identity of
new files separately. If quota attribution or numeric capacity is not
established, it fails closed; it never substitutes folder-owner quota or
changes sharing/ownership. All Drive paths and existing run/checkpoint
identities remain unchanged.

## Exact execution order

Run Cells A → B → C → D. Cell D is read-only. Cells E and F remain disabled
until the user explicitly enables them after reviewing current lease,
checkpoint, and capacity evidence. E resumes the first
`ember_seed42_full_hybrid` run; F continues the remaining matrix. Cell G
exports lightweight results. See `README_GITHUB_COLAB.md` for the Secret
setup and routine pinned-commit update procedure. No production run has been
launched by this handoff.

## Exact execution order

Run Cells A, B, and C in order. They mount/authenticate Drive, verify the
distributed ZIP and every extracted member, verify that the loaded training
source hash matches the unchanged `revision_r1.runner` identity, locate the
exact 220-row effective schedule and manifest from the existing preflight,
bind all 100 run records to that schedule and preflight, verify the successful
v6.2 recovery report as a checkpoint-size reference, and check Drive owner,
quota, local capacity, and storage projections. The preflight-reuse cell
checks the exact NPZ filenames but does not reread their payloads. The first
dataset load for an actual run verifies the current NPZ hashes and shapes
against preflight before training.

Cell C persists an immutable runtime-bound plan in the configured production
results root. It reports both the first-run requirement and the projected
full-matrix requirement. The v6.2 checkpoint measurement is a conservative
size proxy, not a production measurement. The full-matrix estimate includes
11 retained task-boundary generations and two recovery generations per run,
result/prediction storage floors, atomic-write headroom, dataset-cache space,
and the unchanged 512 MiB reserve. It does not equate retained bytes with
cumulative Drive writes: periodic verified checkpoint synchronization adds
write volume, which is measured separately during production.

Cell D is read-only. It reports the exact matrix/run lease paths and raw JSON,
heartbeat/expiry times, run status, progress, hashes of the newest recovery
checkpoint candidates (streamed in 8 MiB blocks), current process/runtime ID,
PID, process-start UTC time, cgroup/host memory, and NPZ
shape/dtype/uncompressed byte sizes from headers only. It also searches the
persisted v1.2 status/registry/progress/preparation records for PID 44230 and
reports their file timestamps. The supplied kernel excerpt has no event
timestamp; if the PID is absent from those records, the diagnostic says so
explicitly rather than inferring the failed phase. It never removes/recover
locks, loads checkpoints, or starts training. The historical report confirms a
Colab cgroup OOM kill for PID 44230 at 9,534,112 kB anon RSS; it does not
identify the exact allocation. Cell D also displays the exact v1.3 training
identity's first-run resume gate, matching checkpoint generations, registry,
and last progress record. A PID from a previous VM cannot prove a worker
is live or dead. Inspect Colab Manage Sessions and all open notebooks, then
share the diagnostic before enabling Cell E. Drive locking is best-effort and
remote runtime visibility is not available from this notebook.
The checkpoint preflight hashes only the two retained payloads in bounded
8 MiB blocks and never deserializes them. Its cursor/counters come from the
last progress snapshot and are labeled informational because that snapshot
can be newer than the last Drive-synchronized generation. Cell E stages the
matching manifest locally as a fail-closed marker; if Drive restore fails, the
runner raises rather than treating local staging as a fresh-run request. The
runner then deserializes candidates sequentially and owns corruption fallback.

Cell E is the only launch for run 1: `ember / seed 42 / full_hybrid`. Its
`LAUNCH_FIRST_PRODUCTION_RUN` and `ACKNOWLEDGE_NO_OTHER_COLAB_WORKER` switches
both default to `False`. It rechecks the matrix lease immediately before
launch; malformed, unreadable, live, or same-host-live PID records fail
closed. An expired lease is never manually deleted: the existing guarded
`RunLease.acquire` expiry recovery is the only recovery path, and requires the
explicit no-other-worker acknowledgement. The matrix lease is acquired before
dataset hashing/loading and held through training, preventing duplicate
dataset-sized RAM work in parallel cells. Preparation progress is written to
Drive before and throughout dataset loading, so an OOM phase can be diagnosed
after a runtime reset. Before allocation, Cell E re-hashes the exact v1.3
recovery generations. It is resume-only: if the source, config, schedule,
dataset identity, generation set, payload size, or payload hash fails, the
first run stops and never initializes from scratch.

The current release changes only the external orchestrator, notebook, and
progress presentation; the 27-file training-core hash remains exactly
`1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`. The
authentic v1.2 ZIP and all 27 source-file hashes are verified from the
distributed v1.2 artifact during packaging, but v1.2 checkpoints are never
resumed by the current training identity. They are inspected read-only and
remain untouched. A matching v1.3 first-run checkpoint is required; the
notebook no longer offers a “start fresh” acknowledgement for that run. After
an interruption, rerun A–D and then E; the strict checkpoint gate and runner
restore the newest valid matching recovery generation, falling back only to
another matching valid generation.

Cell F handles only runs 2–100 and also defaults to `False`. It requires the
first counted run to be registered and artifact-validated, then it checks the
full remaining-matrix capacity before each new/resumed run. It skips only
matching, revalidated completions. No ETA is extrapolated from Full Hybrid to
other arms. Both launch cells use one bounded updating display with a five-
event summary and a plain-text fallback. Every emitted safe-boundary event is
also appended to a unique attempt JSONL under
`results/progress/<plan>/events/`; the latest-progress snapshot remains in its
existing location. Updates occur immediately at task/phase transitions and
roughly every 12 seconds otherwise. GAN epochs/batches/critic substeps,
scheduled classifier updates, already-computed losses, source draws and pool
proportions, completed-task metrics, component timings, available memory,
checkpoint sync bytes/age, progress observer overhead, per-attempt/cumulative time, and same-phase ETA are
shown only where the runner has evidence. Non-GAN work and unstored training
accuracy are `N/A`; the critic trace stores total critic loss, not a separate
gradient-penalty term, so that term is also `N/A`. The completion registry does
not persist a central failed-run count; the display says so instead of
inventing one. No extra model pass, evaluation, RNG use, or CUDA
synchronization is added for display; ETA is not transferred across arms or
phases.

Cell G writes compact progress and validated lightweight result archives.
Those archives exclude model/checkpoint payloads and are capacity-checked
before writing with a 1% compression-growth margin, 16 MiB archive metadata
headroom, and the unchanged 512 MiB reserve. A full checkpoint archive is separately controlled by
`EXPORT_CHECKPOINT_BACKUP = False`; if enabled, every manifest/payload is
identity- and checksum-verified first, and available Drive capacity plus the
reserve is required. Neither exporter deletes or prunes source artifacts.

## Drive ownership and storage readiness

Production output/checkpoint/export paths default below the authenticated
account's `MyDrive`. A My Drive folder shared with any authenticated Google
account as an Editor is supported; folder-owner email need not match. The
notebook resolves shortcuts, checks exact-folder `canAddChildren`, and
verifies a tiny mounted write is owned by the authenticated account and has
positive `quotaBytesUsed`. Only that uniquely named probe is removed. The
proof is cached for the current Python runtime per account and folder; capabilities and
quota are refreshed on every check.

Capacity comes from the authenticated account's Drive API `storageQuota`, the
quota associated with newly created files. Workspace storage may be pooled,
and the API response does not always identify pooling. If finite `limit` and
`usage` are not both present, capacity is reported as unknown and production
stops without inferring free bytes. A missing limit may represent unlimited
quota, but this workflow does not manufacture a numeric estimate. Existing
shared-folder files remain charged to their owners. Shared drives remain
unsupported because their pooled quota is not measured by this My Drive
check. Output and checkpoint roots must resolve under the same authenticated
quota account, not the same folder owner.

The current quota must be queried in Colab rather than inferred from an older
observation. Four GiB is not sufficient for the complete 100-run retained
checkpoint matrix. A known v6.2 recovery payload was 181,822,939 bytes; using
that only as an illustrative lower-bound input, 13 retained checkpoint
payloads per run plus the 256 MiB/run output floor implies at least about
246 GiB peak for 100 runs after atomic-write headrooms and the 512 MiB
reserve. Cell C uses the verified v6.2 report's actual maximum checkpoint
bound and computes the live projection, which may be larger. Do not enable
Cell E unless live quota is numerically verified and the projection fits.
The displayed deficit includes the 512 MiB reserve. Configure the final
output/checkpoint destination before Cell D; moving completed runs is
unsupported because completion records contain absolute artifact paths.
There is no automatic checkpoint offload/pruning mode in this release.

## RAM/OOM safeguards

Production cache creation streams compressed NPZ members to `.npy` files in
8 MiB blocks and opens the verified arrays as read-only memory maps. Feature
finite checks, taskwise scaler statistics, GAN batches, classifier replay
batches, stored-logit refresh, evaluation, and class-mean extraction use
bounded row batches rather than constructing whole-dataset float64 or
standardized copies. The inverse-frequency mixed-pool replacement sampler,
256-example classifier updates, source proportions, class/task ordering,
optimizer/loss protocol, and schedule are unchanged. A 1 GiB host/cgroup RAM
reserve is required before checkpoint serialization/read-back and before
uncached bulk decoding; low headroom stops before the allocation. This reserve
is separate from the unchanged 512 MiB Drive/local-disk reserve.

During checkpoint publication, the first integrity read-back object is
released before the published payload is deserialized for independent
manifest/hash validation. This prevents two complete in-memory RunState graphs
from overlapping; serialization bytes, hashes, and identity checks are
unchanged.

The loader logs each stage with available host/cgroup RAM, process RSS and
peak RSS, cgroup usage/limit/peak and OOM events. `resource_metrics.json`
records memory samples at task/phase boundaries and final artifact export.
These safeguards reduce avoidable anonymous-array peaks but do not constitute
a measured production memory bound; Colab Cell D and the first production run
remain the hardware validation points.

Local scale testing measured a 1.17 GB float32 memmap feature cache for a
120,000 × 2,439 synthetic fixture and a 1.43 GiB process peak RSS while
streaming compressed NPZ data and fitting the scaler in 2,048-row chunks. This
validates that fixture only; it is not a measurement of real EMBER/AZ data,
Colab cgroup headroom, or a complete production task.

The production checkpoint policy is unchanged: task-boundary snapshots and
two recovery generations are retained on Drive and in per-run local staging.
Local staging is not automatically pruned after a run; a long uninterrupted
session accumulates those local copies, and Cell C / the per-run capacity gate
account for that. A runtime reset clears ephemeral local staging; the next
run restores from the identity-verified Drive checkpoint. A reset can lose
updates since the last verified Drive synchronization (the configured
periodic sync is five minutes, with forced phase/task boundaries); it is not
zero-loss. Drive locking is best-effort, not a transactional distributed
mutex. Keep one notebook session active for this matrix. If local capacity
approaches its gate, stop cleanly and resume after a runtime reset rather than
deleting recovery state.

## Protocol and evidence boundary

The matrix is 2 datasets × 10 seeds/orders (42–51) × 5 arms = 100 runs. Each
has an initial 50-class task and ten five-class increments, fixed classifier
batch size 256, and all 11 update budgets from the exact effective-schedule
hash `7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`.
It preserves mixed-pool inverse-label-frequency replacement sampling,
independent RNG streams, paired initializations/orders, training-only
preprocessing, raw-coordinate replay, and declared DER++ terms. Matching
classifier-update budgets do not imply matching total compute or source
exposure; generator and critic updates/timing are reported separately.

The successful v6.2 GAN-recovery report is reused only as verified recovery
and conservative storage-size evidence. Its classifier PASS is not rerun, and
its synthetic validation identity is never used for production. New runs use
the actual hash of the packaged training core and are never relabeled as
historical reproduction. Production outputs, predictions, resource metrics,
checkpoint manifests, progress, and completion registry remain in the
configured Drive paths. The working-tree source, historical results, and
manuscript are not changed by running the notebook.
