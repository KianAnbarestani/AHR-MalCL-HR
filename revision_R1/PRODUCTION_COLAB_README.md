# revision_R1 production Colab handoff

The single stable notebook in `My Drive/ReVision` is
`Revision_R1_Production_Colab.ipynb`. The runner code is fetched
from the explicitly pinned GitHub commit; the notebook does not depend on a
code ZIP in Drive.

## Exact execution order

Run Cells A → B → C → D. A fetches one detached, pinned commit and verifies
both package manifests, the frozen schedule/training identity, and the
canonical Drive storage identity. B checks the passed preflight and exact
Drive dataset files or verified local caches; uncached files are staged once
through the canonical API, and verified mmap caches are reused.
C checks only the next pending run's incremental capacity plus reserve, shows
the full-matrix projection as informational, and skips the already-finalized
Run 1 after reusing its existing registered measurement. D is read-only.

Cell E is a no-op for the finalized Run 1. Cell F can continue remaining
records only after the user explicitly changes its launch switch and reviews
leases; it skips validated completions and resumes interrupted runs from the
canonical verified checkpoints. Cell G exports compact results. No cell
starts production by default.

## Runtime

Cell A expects Python 3.12.13, NumPy 2.0.2, PyTorch 2.11.0+cu128, PyTorch CUDA build 12.8, and an available GPU. It checks these in place and does not install or replace PyTorch.

## Storage and identity

Compute-account changes do not change storage: durable storage uses the
explicit OAuth account `kian.anb777@gmail.com` and ReVision folder ID
`1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x`. Cell A verifies the OAuth identity with
Drive `about`; the DriveFS mount may be authenticated as a different compute
account and is read-only for datasets/preflight. All durable writes use local
staging plus the verified OAuth Drive API: checkpoints, outputs, progress,
manifests, registries, leases, journals, and exports. Cell A verifies the exact
folder/path, API write/read-back probe, new-file owner/quota metadata, cleanup
capability, and Drive v2 current-session stale-ETag rejection using the same
explicit OAuth credentials as the v3 data API. Wrong OAuth credentials,
unknown quota, Viewer access, ambiguous paths, or an unverified conditional
lease update stop before workers. No sharing changes, ownership transfers, or
service accounts are used; no compute-account email allowlist is required.

The Drive API uses `https://www.googleapis.com/auth/drive` because it must
inspect existing user-owned datasets/checkpoints and write/delete verified
files. Tokens remain in session memory. The current GitHub repository is
public; if it becomes private, use the restricted read-only Colab Secret
`GITHUB_READ_TOKEN`.

Training core remains
`1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`; the
schedule remains
`7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`.
The selected Git commit is bound separately. No datasets, checkpoints,
results, credentials, or caches belong in Git.

Local recovery cadence is five minutes; Drive recovery cadence is fifteen
minutes, plus task boundaries and graceful interruption. Publication is
capacity-checked and verified through the canonical API before pruning.
Two verified recovery generations are retained where owned by the canonical
account; all task-boundary files remain. Mixed-owner historical generations
are preserved even when `canDelete` is true, so protected legacy files may
temporarily exceed the recovery count target. No old research artifacts are
automatically migrated, rewritten, transferred, or deleted.

## Existing-runtime upload recovery and display

Do not reset the VM or delete local checkpoints after an upload failure.
After updating the pin and running the matching A–D, set Cell F's
`PREPARE_EXISTING_CHECKPOINT_ONLY = True`, keep `LAUNCH_REMAINING_MATRIX = False`,
and acknowledge no other worker only after checking Manage sessions.
The standalone `Prepare_Checkpoint_Publication.py` cell is an equivalent option.
It holds the canonical matrix lease, checks existing checkpoint identities and
payload hashes, selects the newest valid local/staged/Drive generation, and
publishes payload before manifest with API read-back. It does not construct a
model, load tensor state, prune files, or initialize an experiment. Review its
generation, exact local/Drive paths, hashes and IDs before separately enabling F.
After preparation, rerun D, set preparation False and only then explicitly
set the launch/acknowledgement switches True if ready to resume training.
The resume cursor is reported as unknown until the runner restores the payload.
Finalized Run 1 remains skipped. Run 2 is not marked complete by publication.

The matching Cell D installs `ColabProgressDisplay`, a single updating raw-text
panel with real line breaks. `DEBUG_LOGGING = False` hides verbose dictionaries;
structured local/API logs remain. UI refresh is seven seconds plus transitions.
Permanent task summaries show only actual evaluations. The renderer performs no
Drive reads, checkpoint scans or GPU synchronization. Local checkpoint metadata
is cached from successful saves; memory is sampled at most every 30 seconds.
Progress-file API persistence is separate, every 60 seconds or transition.
Local observation components and API synchronization time are reported separately.

Resumable uploads use Google's supported `googleapiclient.http.build_http()`
transport, which leaves HTTP 308 acknowledgements to the upload state machine.
Chunk retries are bounded; neither global transport patches nor dependency
downgrades are used. Sanitized exception diagnostics expose observed HTTP status
and allowlisted response headers, never credentials, bodies or session URLs.

## Post-export publication failure: completion only

For the Current-only failure at release `96f202d6`, an empty
`selection_records.jsonl` is legitimate: records are emitted only by generated
replay selection. The file must exist and its exact empty-content hash must
verify. Zero bytes are not missing bytes. Publisher checks owner, exact parent,
type, size and content digest; zero or unavailable per-file quota is reported
separately and is never substituted for content integrity or available capacity.
Account quota/capacity and existing reserves remain enforced independently.

Keep the existing Colab VM and local exports/checkpoints. After fetching the
corrected pinned code with A, use `Complete_Exported_Run.py` with existing C
variables, or the matching F with `COMPLETE_EXPORTED_RUN_ONLY = True` and both
launch/preparation False. Explicitly acknowledge no other worker after checking
Manage sessions. This is not a training restart: the helper verifies the frozen
identity, existing 11-task exports and manifest, hashes two recovery generations
(including the terminal complete checkpoint) and all eleven task-boundary
checkpoints, publishes existing bytes, then uses the normal registry completion
API. It never loads tensor state, constructs a model, runs an update or deletes
an artifact. The failed training attempt is retained; any new publication-only
attempt is explicitly labeled `completion_only`.

Runner completion is published/validated before the canonical matrix registry
is updated. Retries reuse valid completion evidence without adding another
completion; fresh staging rebinds absolute paths locally but never publishes
those rebound historical markers over a valid canonical marker. Missing/corrupt
evidence stops with an error rather than regenerating outputs or training.
After Run 2 is finalized, rerun C and D, keep E's Run 1 no-op, set all F recovery
modes False and explicitly enable F only when ready. Frozen-plan continuation
skips Runs 1 and 2 and selects `ember_seed42_anchor_only` next. Launch switches
remain False in the delivered notebook. Compact recovery evidence is published
under `production/results/completion_recovery/`; old logs and artifacts remain.
