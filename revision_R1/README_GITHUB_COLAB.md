# GitHub-pinned Colab runner

The stable production notebook is stored in the existing shared `My Drive/ReVision` folder. It fetches one explicitly pinned full Git commit at Cell A, verifies the detached checkout and `PACKAGE_MANIFEST.json`, then reuses that versioned local checkout for the rest of the Colab session. It never follows a branch head or fetches per run.

The current repository is public, so the notebook fetches without a GitHub token. If the repository is later made private by its owner (this handoff does not change visibility), Cell A supports a Colab Secret named `GITHUB_READ_TOKEN`. Create a fine-grained personal access token restricted to this repository with **Contents: Read-only** permission, add it in Colab’s **Secrets** panel under that exact name, and enable notebook access for the account running Colab. Do not paste the token into a cell, notebook output, Drive, a Git URL, or a repository file. The fetch helper keeps the credential in process memory and does not write it to the checkout or Git remote.

The Colab account can be any account with Editor access to the existing shared ReVision folder. The workflow resolves the shared destination, verifies the current account’s actual write capability, and reports quota ownership/availability from evidence. A folder owner is not assumed to be the account charged for newly created files. If quota ownership or sufficient available capacity cannot be established, readiness stops safely. No sharing permissions or ownership are changed.

Cell A accepts only the observed handoff runtime: Python 3.12.13, NumPy 2.0.2, preinstalled PyTorch 2.11.0+cu128, and its CUDA 12.8 build with an available GPU. The notebook checks these versions in place and never pip-installs or replaces the CUDA PyTorch package. A Colab image mismatch stops before production and requires a reviewed, tested update.

## Colab execution order

1. **A** — mount the existing Drive, resolve any required GitHub Secret, fetch the notebook-pinned full commit once, check it out detached, verify package/training/schedule identities, and set up the imports.
2. **B** — verify the exact existing EMBER/AZ dataset files and the Drive preflight/effective schedule pair.
3. **C** — validate the frozen plan, Drive/local capacity, shared-folder write/quota evidence, and persist an immutable code/protocol binding for this plan and selected Git commit.
4. **D** — read-only checkpoint/lease/progress diagnostic. Check Colab **Manage sessions** separately; a PID from a previous VM cannot prove a worker is live or dead.
5. **E** — resume the first `ember_seed42_full_hybrid` run. Its launch and acknowledgement switches stay `False` until explicitly reviewed and enabled by the user.
6. **F** — resume the remaining matrix only after the first run is validated. Its switches also stay `False`.
7. **G** — export lightweight progress/results. Checkpoint backup remains separate and disabled by default.

No setup cell starts training. The notebook preserves the existing Drive output/checkpoint paths, run IDs, plan/config identities, integrity checks, RNG/state restore, checkpoint discovery, and lease protections.

## Checkpoint cadence and storage behavior

The production runner writes a local recovery generation every five minutes
of monotonic time measured from the previous checkpoint's completion. It
publishes only that new generation to Drive when the 15-minute publication
deadline is due. Initial state, task boundaries, graceful interruption, and
terminal completion still force the required writes. Task-boundary artifacts
remain a separate retained archive. A long checkpoint cycle resets its timer
at completion, so its own duration cannot make another checkpoint immediately
due.
The frozen RunConfig still contains legacy checkpoint-interval fields because
they are part of the existing configuration/checkpoint identity; they are not
rewritten. The actual runtime override (300-second local / 900-second Drive)
is recorded separately in `resource_metrics.json` as `checkpoint_policy`.

Drive recovery rotation publishes and verifies the new generation before it
prunes older recovery generations, leaving two hash-verified generations. Cell
C verifies exact-folder write capability, quota attribution, and permanent
deletion of the authenticated account's tiny probe file; if safe cleanup
cannot be established, the production gate fails closed. Pruning uses exact
Drive file IDs only for files owned by the current account. It never removes
task-boundary archives, foreign-owned files, or unrelated artifacts. Existing
Drive Trash is not emptied automatically and continues to consume quota.

Before resuming the first production run, the capacity envelope is sized from
the maximum payload in the exact-identity recovery/task-boundary manifests
already on Drive (the selected recovery payload is SHA-256 verified; the other
payloads are size-checked without an extra multi-gigabyte read), combined with
the v6.2 synthetic reference using the larger value and the existing 1.5x +
16 MiB margin. All 11 task-boundary archives remain separate and retained; only
recovery generations rotate. The gate subtracts bytes already occupying this
run's exact namespace and checks the 512 MiB reserve before dataset loading or
training. If remaining Drive/local capacity cannot support the frozen
retention policy, the runner refuses to start; it does not prune boundaries or
reduce the reserve to force a fit.

Checkpoint reporting separates optimizer/training time, serialization,
upload, verification, pruning, and local staging/copy time. It reports
retained storage independently from cumulative successful payload/manifest
write traffic. Legacy write traffic from before this build is labeled unknown.
Subcomponent timings are included within total checkpoint wall time and should
not be summed as disjoint elapsed time.

## Routine code update

Push the tested runner changes to GitHub, record the resulting full 40-character commit SHA, and change only `PINNED_GIT_COMMIT` in the stable notebook. Re-run A–D in a fresh Colab session. A source change that alters the training-core hash from `1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c` or alters the frozen schedule/protocol is rejected before launch. Reporting-only commits can use the existing compatible checkpoints and are recorded separately from the training identity.

Keep dataset files, checkpoints, generated results, secrets, and local caches out of Git. Do not move the shared folder or transfer ownership to make capacity checks pass.
