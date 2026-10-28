# revision_R1 compatibility and packaging report

## Scope and provenance

`revision_R1` is a newly specified, separately versioned matched-runner
protocol. It is not V15.5/V16 historical training and cannot reproduce or
relabel their outputs. V15.5 remains the historical baseline; V16 and all
revision_R1 runs have separate namespaces and paths. This package does not
edit manuscripts, Phase 1 analysis, prior source releases, or prior results.
No production GPU run was launched while preparing it.

## Reused conventions verified in the workspace

The V15.5 Standard-A handoff at
`paper_ready/AHR_MalCL_V15_5_STANDARD_A_GPU_HANDOFF/` provides a Colab notebook,
launcher, pinned handoff requirements, source hash manifest, tests, and an
environment lock. The V15.5 release at
`paper_ready/AHR_MalCL_HR_reproducibility_release_v15_5/` separates analysis,
verification, and GPU-training dependencies and retains checksummed analysis
artifacts. The V16 release and validation directories use separate run
registries, provenance manifests, validation exports, and post-hardware-fix
artifacts. revision_R1 adopts those packaging principles: immutable versioned
configuration, hash-identified code/data/schedule, fail-closed preflight,
machine-readable run manifests, independent CPU validation, and explicit
production gating. It does not copy or overwrite their source/result trees.

The V15.5 handoff lock records a Colab Linux/T4 target, Python 3.12, NumPy
2.0.2, tqdm 4.67.1, pytest 8.3.5, and a CUDA PyTorch build (2.6+ in that
handoff). The later V15.5 release records Python 3.12, PyTorch 2.11.0, CUDA
12.8 for its final evidence; the observed Colab interpreter was Python
3.12.13 and PyTorch was `2.11.0+cu128`. The production Cell A now checks those
exact Python, PyTorch, and CUDA versions, along with NumPy 2.0.2 and CUDA
availability, without installing/replacing the platform CUDA build. A future
Colab image mismatch stops before production and requires review and testing.
The targeted continuation does not rerun pytest; focused
storage/workflow/package regressions are checked during package preparation.
PyTorch-dependent tests require a PyTorch runtime. The notebook never installs
a CPU torch wheel or silently replaces a CUDA build.

## Data and schedule compatibility

The archive contains no dataset arrays, credentials, historical checkpoints,
or prior run outputs. Dataset directories are configured explicitly in the
notebook or CLI. The preflight expects the four exact V15.5 prepared NPZ names,
checks their known MD5 and computed SHA-256, array keys/shapes/finite values,
labels 0–99, split integrity, class support, and all 220 proposed update-budget
rows against the actual dataset pools. The schedule is the proposed Phase 1
schedule copied unchanged into `configs/update_budget.csv`; its companion
manifest identifies provenance. It is not an observed historical training
schedule. Preflight preserves that CSV and materializes a separate immutable,
content-hashed effective schedule plus manifest under
`<output_root>/preflight/schedules/`. Only rows whose anchor count equals the
configured nominal quota (`K × prior classes`) may be reduced to the verified
available-support count (`sum(min(K, per-class support))`). Current and
generated pools, unexplained anchor counts, and the full 220-row key matrix
remain strict checks. A later explicitly authorized run must use the derived
schedule path recorded in `preflight.json`, not the package's parent CSV.
Local verification cannot substitute for Colab preflight because those
prepared NPZs are absent from this workspace.

## Runtime and resume behavior

Production requires an available CUDA device and the CLI's separate
`--production-authorized` flag. The notebook's setup, preflight, and import
cells do not set that flag. The production cell is separate, defaults
to disabled, and launches one explicitly selected dataset/seed/arm only.
The runner uses one run namespace per protocol version/configuration digest,
dataset/seed/arm for both output and checkpoint paths. Classifier and GAN recovery checkpoints stage under the
Colab-local directory, mirror verified generations to the configured Drive
folder, and resume from that Drive mirror after a runtime reset. Task-boundary
generations are retained separately. Drive/FUSE locking and rename behavior
remain best-effort, not transactional; the runner uses identity checks and
read-back verification but cannot guarantee distributed locking.

## Exports

Each validated run includes final and task-boundary predictions, class order,
task matrix, selection/source traces, configuration/environment identities,
resource accounting, an artifact hash manifest, and completion registry
marker. The lightweight results exporter explicitly excludes `.pt`, `.pth`,
`.ckpt`, `.safetensors`, `checkpoints/`, and `registry/`. A separate full
checkpoint exporter includes all verified recovery and task-boundary
generations it is given. Each ZIP has an internal file/hash manifest and an
external SHA-256 manifest. No production results or checkpoint archive exists
until a separately authorized run is performed.

## Notebook validation workflow and limits

The current storage-bounded validation package is
`revision_R1_colab_validation_v6_1_gan_storage_bounded`; its training core remains
`revision_R1.0.1`. The preceding v5 runner explicitly sets classifier
train/eval mode by phase, restores helper modes, restores global and named RNG
state after model/head and optimizer reconstruction, and times the full
classifier update with synchronized CUDA boundaries. This v6.1 continuation
does not alter production training logic or repeat that classifier test. It
reuses the exact v5 classifier PASS evidence and selects an unfinished v5 GAN
checkpoint only after validating v5 source hashes, config/data identity,
schedule identity, manifest, and payload hash. Other prior reports remain
preserved and do not satisfy v6.1 gates. V4 checkpoints are not resumed because
their all-source code identity differs.
The build-time compatibility audit separately compares all model/training
modules against the retained v4 ZIP and normalizes the runner diff to the
synthetic-only duplicate-task-boundary policy, the checkpointing diff to
storage capacity/retention/orphan-temp cleanup, and pilot-only storage
projection reporting. If that audit passes, v4 smoke
outputs can be adopted only after schedule/code identity and artifact hashes
are verified; completed outputs can be discovered case-by-case without an old
aggregate report. Missing smoke cases are `NOT_RUN` by default; launching them
requires an explicit notebook opt-in. Neither v4 checkpoint state nor v4 resume
results are reused.

The resume comparator validates the expected 11×11 lower-triangular task
matrix, requires identical null masks, rejects missing/non-finite evaluated
cells and populated future cells, then compares only evaluated cells with the
existing `atol=1e-5`, `rtol=1e-4`. It reports the first differing stage/task
cell and maximum finite absolute error. Prediction arrays, source proportions,
update/loss traces, model states, optimizer states, and captured global and
separate GAN RNG states remain part of the resume comparison.

The notebook reads `preflight/preflight.json` and its effective schedule and
manifest from mounted Drive, checks the four exact NPZ paths and the preflight's
recorded MD5/SHA-256/shape metadata against pinned references, and reconciles
every row against the immutable packaged parent schedule. It trusts the prior
successful checksum-verified preflight rather than re-reading the large NPZ
payloads on every notebook rerun. The
expected effective schedule SHA-256 is
`7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`; the
row-level report records 47 anchor-pool corrections and 5 changed update
budgets with training-availability-only reasons. If the existing preflight is
stale, it is preserved and a timestamped new one is written. Ambiguous
schedule/manifest discovery fails closed.

The notebook's CUDA smoke comprises 10 non-reportable synthetic cases (five
arms × EMBER/AZ-Class feature dimensions). Compatible v4 smoke outputs may be
adopted with their original provenance; the build does not claim that v6.1
re-executed them. Missing/incompatible cases are not launched by default.
The v6.1 targeted continuation runs only the GAN fresh-process interruption /
recovery check. It branches an uninterrupted control and interrupted worker
from one verified unfinished v5 GAN checkpoint. In these validation workers,
redundant intermediate phase snapshots are suppressed; verified starts and the
required terminal/interruption/failure checkpoints remain local. Only the
single interruption generation is copied to Drive, and a fresh process must
read that generation from Drive with empty local staging.
The worker's generation, byte count, and SHA-256 values must match Drive
read-back. The GAN output comparison retains its existing numerical
tolerances. Corruption fallback uses a temporary local copy; identity
rejection and duplicate-result prevention are also checked. The v5 classifier
PASS is reused and not rerun. Manual Colab runtime reset remains optional.
GPU-only phases explicitly report `NOT_RUN` without CUDA; local CPU tests never
stand in for a GPU PASS.

An actual-data engineering pilot reads only checksum-verified training NPZs.
It records classifier, critic, and generator throughput separately, peak GPU
memory, Drive checkpoint payload/write/read-back time, and storage projections.
Completed per-dataset pilot results are reused on rerun; an interrupted
in-progress dataset microbenchmark restarts from its beginning, and partial
measurements are never promoted as completed results.
Any full-matrix runtime projection is a partial-task estimate and excludes
candidate generation/selection, evaluation, data preparation, later-task
architecture/replay effects, and most Drive synchronization. Pilot/test-set
performance is not used to tune scientific settings. Smoke, resume, and pilot
outputs/checkpoints are kept below `results/validation/`, not production run
roots. Synthetic validation retains at most two verified recovery generations
per active worker, does not write duplicate task-boundary checkpoint streams,
and removes checkpoint payloads after successful smoke-artifact validation or
resume comparison while keeping hash manifests/reports. Failed workers retain
their last two verified recovery generations. The manual runtime-reset test is
optional and limited to one fixture per code/schedule identity. Fault-injection
copies are temporary local files; abandoned checkpoint atomic-write temp files
are removed only inside the exclusively leased synthetic run namespace before
its next write. Worker logs stream live and are capped at one
append-only log per case. The read-only inventory queries account quota,
`usageInDriveTrash`, visible validation files, and trashed files whose names
match checkpoint-generation patterns; filename matches are not treated as
proof of validation ownership. Account-level usage and Trash are distinct from
the visible validation-tree total, so any Trash contribution can be confirmed
from Colab's API inventory rather than inferred from filenames. No Trash,
unrelated Drive files, historical reports, or v5 checkpoints are removed.
Before GAN workers launch, the storage plan includes existing visible
validation bytes and total account usage (including Trash), one interrupted
Drive checkpoint, conservative temporary-write overlap, bounded remaining
logs, report/summary atomic-write headroom, the complete owned local workspace
and scratch peak, free space, and the unchanged 512 MiB reserve. It reports
cumulative write volume separately from temporary peak space. An existing
workspace larger than its declared bound, unknown quota, or insufficient
capacity stops before workers. None of these checks reproduce historical
training.

Full matrix size is 2 datasets × 10 seeds/orders × 5 unique arms = 100 runs;
each arm has 11 task-boundary classifier update budgets read from the effective
220-row schedule. Matching classifier updates is not matching total compute;
generator/critic costs must be reported separately.
- Matching classifier updates means equal counted classifier optimizer
  updates, not matched total compute. Generator/critic steps, time, memory,
  and checkpoint storage are logged separately.
