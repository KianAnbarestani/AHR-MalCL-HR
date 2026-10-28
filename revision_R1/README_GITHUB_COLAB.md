# GitHub-pinned Colab runner

The stable production notebook is stored in the existing shared `My Drive/ReVision` folder. It fetches one explicitly pinned full Git commit at Cell A, verifies the detached checkout and `PACKAGE_MANIFEST.json`, then reuses that versioned local checkout for the rest of the Colab session. It never follows a branch head or fetches per run.

The current repository is public, so the notebook fetches without a GitHub token. If the repository is later made private by its owner (this handoff does not change visibility), Cell A supports a Colab Secret named `GITHUB_READ_TOKEN`. Create a fine-grained personal access token restricted to this repository with **Contents: Read-only** permission, add it in Colab’s **Secrets** panel under that exact name, and enable notebook access for the account running Colab. Do not paste the token into a cell, notebook output, Drive, a Git URL, or a repository file. The fetch helper keeps the credential in process memory and does not write it to the checkout or Git remote.

The Colab account can be any account with Editor access to the existing shared ReVision folder. The workflow resolves the shared destination, verifies the current account’s actual write capability, and reports quota ownership/availability from evidence. A folder owner is not assumed to be the account charged for newly created files. If quota ownership or sufficient available capacity cannot be established, readiness stops safely. No sharing permissions or ownership are changed.

## Colab execution order

1. **A** — mount the existing Drive, resolve any required GitHub Secret, fetch the notebook-pinned full commit once, check it out detached, verify package/training/schedule identities, and set up the imports.
2. **B** — verify the exact existing EMBER/AZ dataset files and the Drive preflight/effective schedule pair.
3. **C** — validate the frozen plan, Drive/local capacity, shared-folder write/quota evidence, and persist an immutable code/protocol binding for this plan and selected Git commit.
4. **D** — read-only checkpoint/lease/progress diagnostic. Check Colab **Manage sessions** separately; a PID from a previous VM cannot prove a worker is live or dead.
5. **E** — resume the first `ember_seed42_full_hybrid` run. Its launch and acknowledgement switches stay `False` until explicitly reviewed and enabled by the user.
6. **F** — resume the remaining matrix only after the first run is validated. Its switches also stay `False`.
7. **G** — export lightweight progress/results. Checkpoint backup remains separate and disabled by default.

No setup cell starts training. The notebook preserves the existing Drive output/checkpoint paths, run IDs, plan/config identities, integrity checks, RNG/state restore, checkpoint discovery, and lease protections.

## Routine code update

Push the tested runner changes to GitHub, record the resulting full 40-character commit SHA, and change only `PINNED_GIT_COMMIT` in the stable notebook. Re-run A–D in a fresh Colab session. A source change that alters the training-core hash from `1731ee525e3297e276609040295977e4c02f5aeab8e4ab125dd08fbcb993dc9c` or alters the frozen schedule/protocol is rejected before launch. Reporting-only commits can use the existing compatible checkpoints and are recorded separately from the training identity.

Keep dataset files, checkpoints, generated results, secrets, and local caches out of Git. Do not move the shared folder or transfer ownership to make capacity checks pass.
