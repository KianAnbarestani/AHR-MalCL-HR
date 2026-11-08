# GitHub-pinned Colab runner

The single stable production notebook is `My Drive/ReVision/Revision_R1_Production_Colab.ipynb`. It is generated from the latest saved Drive notebook after removing temporary recovery-only cells, and pins exactly one full commit SHA in Cell A. It checks out detached into a commit-versioned local directory, verifies package/build manifests, checkpoint compatibility, the training-source hash, and the frozen schedule, then reuses that checkout for the whole Colab session. It never fetches a branch head automatically or downloads runner code per matrix run.

## Canonical storage account and authentication

The Colab compute account may differ from the durable storage account. Storage is always `kian.anb777@gmail.com`, and the canonical folder is ReVision ID `1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x`.

The former `auth.authenticate_user()` + `google.auth.default()` path was not account-selective: Colab can return immediately when any valid user ADC is already cached. Cell A creates Drive API v3 from a separate user-OAuth refresh token stored in Colab Secrets; no Drive API code falls back to ADC. Drive `about.user.emailAddress` must verify `kian.anb777@gmail.com` before the adapter is constructed. Durable reads/writes use that API identity with local staging; the independently authenticated DriveFS mount is not used for durable writes. Cell A verifies the exact ReVision folder, API path bindings, a temporary write/read-back probe, canonical ownership/quota metadata, and cleanup capability. Conditional lease updates are not part of the taskwise single-worker protocol.

### One-time OAuth and Colab Secret setup

1. In the Google Cloud Console, select or create a project and enable the Google Drive API. Configure the OAuth consent screen for the canonical account. If the app is in `Testing`, add `kian.anb777@gmail.com` as a test user.
2. Create an OAuth client of type **Web application** and add this exact authorized redirect URI: `https://developers.google.com/oauthplayground`.
3. Open [Google OAuth 2.0 Playground](https://developers.google.com/oauthplayground). In its settings, select **Use your own OAuth credentials**, set **Access type** to **Offline**, and set the force prompt to **Select Account Screen**. Enter the OAuth client ID and secret.
4. In Step 1, authorize the exact scope `https://www.googleapis.com/auth/drive`. Select `kian.anb777@gmail.com` and verify the account shown before granting access. In Step 2, exchange the authorization code and copy the resulting refresh token.
5. In Colab’s Secrets pane, create these three secrets and enable notebook access for each:

   - `REVISION_R1_STORAGE_OAUTH_CLIENT_ID`
   - `REVISION_R1_STORAGE_OAUTH_CLIENT_SECRET`
   - `REVISION_R1_STORAGE_REFRESH_TOKEN`

   Do not paste secret values into notebook cells, outputs, Git, Drive files, or reports. Cell A reads them in memory and never prints them. The OAuth Playground uses your own OAuth client; its refresh token does not use the Playground’s default-client 24-hour revocation behavior.

Google documents that an External consent screen left in `Testing` expires non-basic-scope authorizations after seven days, including refresh tokens. For persistent use, move the consent screen to production and complete Google’s verification requirements for the requested Drive scope; otherwise renew the refresh-token secret when it expires, then rerun Cell A. A refresh-token renewal does not change training identity or checkpoint state. See [OAuth Playground](https://developers.google.com/oauthplayground), [Google OAuth refresh-token expiry](https://developers.google.com/identity/protocols/oauth2#expiration), and [Drive API scopes](https://developers.google.com/identity/protocols/oauth2/scopes).

The workflow requests the full Drive scope because it must inspect and resume pre-existing datasets, checkpoints, registries, and outputs. `drive.file` does not automatically grant access to all those historical files. No service account is used. The compute/billing account remains unchanged.

The GitHub repository is public, so no token is currently needed. If the owner later makes it private, add a fine-grained GitHub token as the Colab Secret `GITHUB_READ_TOKEN`, restricted to this repository with Contents: Read-only, and enable notebook access. Never paste the token into a cell or URL.

## Exact execution order

Run Cells A → B → C → D → E for setup/readiness. Cell A checks out one explicitly pinned Git commit and verifies package/build manifests, checkpoint compatibility, the training-source hash, and the schedule. Cell B stages the existing preflight, effective schedule, matching manifest, and configured datasets through the canonical API or reuses verified local caches; external dataset downloads are disabled. Cell C verifies the frozen plan and existing Run 1 registry, skips Run 1, imports its eleven existing task-boundary paths, then loads or creates the single `matrix_progress.json`. Cell D reads that JSON without changing it. Cell E is a no-op confirming Run 1 remains skipped.

Cell F is the only launch/resume cell and defaults to `LAUNCH_MATRIX = False`. After verifying storage capacity and confirming no other Colab worker is active, explicitly set both `LAUNCH_MATRIX` and `ACKNOWLEDGE_SINGLE_WORKER` to `True`. It starts with the first unfinished run and task. No matrix/run lease, heartbeat, expiry, initial/mid-task save, recovery generation, or rotation is used. Cell G is a read-only summary of the same progress JSON.

## Proven checkpoint workflow reused

The first-paper V15.5 Standard-A Colab builder (`runners/v15_5_standard_a/tools/build_colab_notebook.py`) mounted Drive once, pointed the launcher at a fixed output root, and reran the same registry-backed launcher. Its launcher (`launch_v15_5_standard_a.py`) skipped completed rows and marked interrupted/failed attempts; `v15_5_runner/checkpointing.py` wrote a temporary checkpoint then atomically replaced `checkpoint/latest.pt`. `v15_5_runner/training.py` saved after each task and restored the saved `next_task` cursor and RNG before continuing. Revision_R1 reuses those operational ideas—one predictable run namespace, atomic task-boundary recovery, and same-run resume—but retains its own checkpoint format and explicit OAuth API/local-staging backend because the Drive mount can belong to a different account. It does not restore from V15.5 checkpoint files.

### Taskwise launch and resume

Run 1 is imported as complete only from its existing validated registry entry. The other 99 rows in the frozen plan start unfinished. The new replacement for Run 2 writes beneath the separate `taskwise_v1/<plan_sha256>/` namespace; files from the abandoned Run 2 namespace are left untouched.

At each task boundary the runner writes one atomic full-state checkpoint locally, publishes that exact checkpoint through the canonical API, and only then atomically updates/publishes `matrix_progress.json`. The state includes model, optimizer, RNG, scaler, replay, sampler, cursor, and accumulated task outputs required by the runner. If Colab stops during a task, the JSON remains at the prior task and that single task is repeated. If a checkpoint path is recorded but the payload is missing or invalid, the runner stops with run, task, and path; it never silently starts from Task 1. After Task 11, result artifacts are published with the artifact manifest last, then the run is marked complete in the same progress JSON. A result-export/publication failure retries from the Task 11 checkpoint without retraining. Leave the two Cell F switches enabled only while the one authorized worker is running; to resume later, rerun A–E and the same Cell F.

## Frozen identities and checkpoint behavior

The compatible training-core identity remains `1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`. The frozen effective schedule remains `7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`. The selected Git commit is recorded separately. Scientific configurations, task order, seeds, datasets, metrics, and result formats are unchanged. The new taskwise checkpoint namespace is deliberately separate from abandoned legacy Run 2 files.

Dataset bytes are read from the configured canonical Drive source. If an archive is absent, only a complete checksum-verified local cache matching the passed preflight is accepted. No external dataset download or automatic dataset migration is performed.

There is exactly one saved full-state checkpoint per completed task for each new run. It is retained; there is no rotation. `matrix_progress.json` is the single matrix progress record and contains all 100 runs in frozen order, each with eleven task entries (`done`, `checkpoint_path`) and a run completion flag. A `done` task is backed by a successfully saved and API-verified checkpoint. The progress file is published only after the checkpoint.

All existing research files, including abandoned Run 2 outputs and checkpoints, are preserved. New checkpoints/results are written in the taskwise namespace through the verified main-account API. API permission, identity, ambiguity, and quota failures stop execution; the progress JSON is not advanced when a task checkpoint fails to publish.

Drive API writes are charged to the verified storage account and are subject to its available quota. There is no current verified 99-run storage projection for this one-checkpoint-per-task policy; review quota and available space before enabling Cell F. No automatic deletion or pruning is used to make room. Each checkpoint and result file is integrity-verified by the storage adapter. Local staging capacity is checked for the files it stages.

## Runtime and routine updates

Cell A expects the observed Colab image: Python 3.12.13, NumPy 2.0.2, PyTorch 2.11.0+cu128, CUDA 12.8, and an available GPU. It verifies these in place and does not install or replace CUDA PyTorch. A mismatch stops before any run.

For a code-only release, commit and push the tested package to GitHub, then change only `PINNED_GIT_COMMIT` in the same stable Drive notebook to the verified full commit SHA. Run A–D to verify the selected release and storage identity before using the disabled continuation switch. Never upload or switch to an untested working-tree checkout. Keep datasets, checkpoints, results, secrets, and caches out of Git.

Cell B retrieves the canonical preflight JSON, its effective schedule CSV, and matching manifest using the verified main-account Drive API adapter. Stale absolute paths embedded in historical preflight JSON are treated as filenames only and resolved beneath the canonical `revision_R1/results/preflight/schedules` folder. The original Drive files are read-only and are not regenerated or overwritten. `DRIVE_PATHS["preflight_drive_path"]` is the logical Drive path; `DRIVE_PATHS["preflight_staging_root"]` is local staging. Cells A–C require no DriveFS mount.

Dataset sources are the existing configured Drive folders, resolved through the same main-account OAuth API. Cell B first verifies a matching local mmap cache; otherwise it reuses a hash-verified staged NPZ or downloads that exact existing Drive file once, verifying the preflight SHA-256 and frozen MD5. `DATA_ROOTS` and the two dataset roots in `DRIVE_PATHS` point at local API staging; `dataset_drive_roots` preserves the logical source addresses for safe reruns. Cell B inspects only archive headers. The runner builds or reuses its streamed mmap cache when each dataset is first needed, so setup does not decompress all arrays before training. No preflight, schedule, dataset, result, or checkpoint is regenerated by Cell B. External dataset downloads remain disabled.

Drive API publication uses the dedicated OAuth credentials and bounded transport retries in the storage adapter. The taskwise runner does not create or recover leases. Because locking is intentionally absent, do not run two Colab workers against the matrix at the same time.
