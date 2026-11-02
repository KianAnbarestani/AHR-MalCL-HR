# GitHub-pinned Colab runner

The single stable production notebook is `My Drive/ReVision/Revision_R1_Production_Colab_d6c99c94.ipynb`. It fetches exactly the full commit SHA in Cell A, checks out detached into a commit-versioned local directory, verifies package/build manifests, checkpoint compatibility, the training-source hash, and the frozen schedule, then reuses that checkout for the whole Colab session. It never fetches a branch head automatically or downloads runner code per matrix run.

## Canonical storage account and authentication

The Colab compute account may change. The storage account is always `kian.anb777@gmail.com`, and the canonical folder is ReVision ID `1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x`. Cell A calls supported Colab user authentication, then force-mounts Drive. Select the canonical storage account in the Google authorization prompts. The authenticated Drive API identity and the mounted filesystem are independently checked: exact folder ID and descendant paths, a tiny write/read/delete probe in ReVision, probe owner, quota identity, and available quota must agree before any run proceeds. If cached credentials select another account, Cell A fails closed; restart the runtime and select the canonical storage account. Do not change paths, share settings, or ownership to work around the check.

The notebook uses the Drive API scope `https://www.googleapis.com/auth/drive`, because the workflow reads pre-existing datasets/checkpoints, inspects quota/ownership, writes files, and may delete only exact verified recovery files owned by the canonical account. A `drive.file` scope would not cover the historical files it must inspect. No service account is used. Tokens are session-only and never placed in the notebook, output, Git, Drive, or reports. This authentication does not change the Colab compute/billing account; it only chooses the durable storage identity.

The GitHub repository is public, so no token is currently needed. If the owner later makes it private, add a fine-grained GitHub token as the Colab Secret `GITHUB_READ_TOKEN`, restricted to this repository with Contents: Read-only, and enable notebook access. Never paste the token into a cell or URL.

## Exact execution order

Run Cells A → B → C → D. Cell A pins code and verifies the canonical storage session. Cell B verifies the Drive preflight/effective schedule and uses only the existing Drive NPZs or checksum-verified local caches; external dataset downloads are disabled. Cell C validates the frozen plan, reads the canonical completion registry, confirms Run 1 is already finalized and registered, reuses its existing storage measurement without recomputing it, checks only the next pending run's incremental capacity plus reserve, and records an immutable selected-commit/protocol binding. The full-matrix projection is informational. Cell D is read-only and reports lease/progress/checkpoint state.

Cell E is a deliberate no-op for Run 1: it does not load a model, train, measure storage, register completion, or rerun finalization. Cell F is the remaining-matrix continuation; its launch switch defaults to `False`. If explicitly enabled after review, it preserves the single-worker lease, skips identity-validated completions, and resumes interrupted runs only from compatible verified checkpoints. Changing compute account does not change Drive paths or run/checkpoint identities. A live lease is never bypassed because the runtime account changed. Cell G exports lightweight results; checkpoint backup is off by default.

## Frozen identities and checkpoint behavior

The compatible training-core identity remains `1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c`. The frozen effective schedule remains `7d327c9ce84a096201d06f910f88d022f4e056ad32e090a84e19be393747d212`. The selected Git commit is recorded separately and is not added to training identity unless a verified training-core or protocol change requires a different checkpoint identity. Existing paths, configurations, run IDs, optimizer/RNG/sampler state, exact cursor, and checkpoint discovery remain unchanged.

Dataset bytes are read from the configured canonical Drive source. If an archive is absent, only a complete checksum-verified local cache matching the passed preflight is accepted. No external dataset download or automatic dataset migration is performed.

Training and serialization stay on local Colab disk. Local recovery is due approximately every five monotonic minutes; Drive recovery publication is due approximately every fifteen minutes, plus task boundaries and graceful interruption. The timers are completion-based. Drive publishes only the newly due generation, reads it back through the same canonical Drive API identity, checks exact parent/name/owner/quota/size/MD5 and checkpoint integrity, then permits recovery pruning. Two verified recovery generations are retained when the account owns them. The eleven task-boundary artifacts are separate and are not pruned.

Mixed-owner historical files are preserved even if the current account has `canDelete`; no automatic transfer, rewrite, or deletion is performed on them. Such protected legacy generations can temporarily make physical recovery retention exceed two. New artifacts must be owned and quota-charged by the canonical account. Write/read-back failures keep the last verified recovery point intact. API errors retry only when transient; identity, permission, quota, and ambiguous-file errors stop immediately.

Capacity is checked against the canonical storage account—not the compute account—and each next pending run is gated on its incremental peak plus the unchanged reserve. The 100-run projection is displayed as planning information only; it does not require capacity for all future runs before beginning one. Capacity is rechecked before each large checkpoint publication. Unknown quota is never converted to an invented free-space number.

## Runtime and routine updates

Cell A expects the observed Colab image: Python 3.12.13, NumPy 2.0.2, PyTorch 2.11.0+cu128, CUDA 12.8, and an available GPU. It verifies these in place and does not install or replace CUDA PyTorch. A mismatch stops before any run.

For a code-only release, commit and push the tested package to GitHub, then change only `PINNED_GIT_COMMIT` in the same stable Drive notebook to the verified full commit SHA. Run A–D to verify the selected release and storage identity before using the disabled continuation switch. Never upload or switch to an untested working-tree checkout. Keep datasets, checkpoints, results, secrets, and caches out of Git.
