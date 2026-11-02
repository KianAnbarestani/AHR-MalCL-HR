# revision_R1 production Colab handoff

The single stable notebook in `My Drive/ReVision` is
`Revision_R1_Production_Colab_d6c99c94.ipynb`. The runner code is fetched
from the explicitly pinned GitHub commit; the notebook does not depend on a
code ZIP in Drive.

## Exact execution order

Run Cells A → B → C → D. A fetches one detached, pinned commit and verifies
both package manifests, the frozen schedule/training identity, and the
canonical Drive storage identity. B checks the passed preflight and exact
Drive dataset paths or verified local caches; it does not download datasets.
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

Compute-account changes do not change storage: use `kian.anb777@gmail.com`
and ReVision folder ID `1tQgfvJNt5MX4cNQbH-4U30hJpcDrBO8x`. Select that
account in Cell A's Google prompts. The code binds the Drive API and mounted
filesystem to that identity using an exact folder/path check and tiny
write/read/delete probe. Wrong cached credentials, unknown quota, Viewer
access, or an unverified destination stop before runs. No sharing changes,
ownership transfers, or service accounts are used. There is no editor allowlist:
durable storage always uses the one canonical account.

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
