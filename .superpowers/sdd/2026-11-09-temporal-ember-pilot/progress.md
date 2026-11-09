# Execution ledger — Temporal EMBER Pilot

- Objective: implement the approved six-window upstream EMBER 2018 v2 temporal pilot without changing the active randomized matrix.
- Constraints: no production jobs from Codex; only the requested nine run definitions; `revision_R1/` is read-only; package outputs independent; user will launch from separate Colab runtime.
- Branch at start: `codex/revision-r1-github-colab-20261028`.
- Task 1: complete (commit b3d2af31; CPU audit, frozen manifest, six-window causal leakage checks).
- Task 2: complete (verified model/config/RNG/logging bundle, separate runner, frozen nine-run budget, completed-window checkpoint/resume).
- Task 3: in progress (paired report exporter and synthetic acceptance cases implemented; packaging remains).
- Task 4: pending.
- Task 1: Ruling: use `python3 -m unittest` instead of the planned `python -m pytest`; this workspace has Python 3.14, no `python` alias, and no pytest installation. The checks use only standard `unittest` assertions, so this avoids changing the shared environment. Cost if wrong: pytest-specific collection behavior would not be covered; no pytest fixtures or markers are used.
Task 1 acceptance: `/tmp/temporal-ember-venv/bin/python -m unittest discover -s temporal_ember_pilot/tests -p test_leakage.py -v` → OK (6 tests).

Task 2: Ruling: use the verified `revision_R1` named sampling/GAN streams and isolated GAN Torch RNG, but omit R1’s randomized class permutation because the approved pilot fixes family order by the frozen chronological admission manifest. The separate schedule is frozen and checksummed in the package before any training; all arms/seeds consume its same update counts. Cost if wrong: a mismatch in stream derivation would change paired stochastic behavior, so vendored source hashes and interruption equivalence are retained for review.
Task 2 acceptance: `/tmp/temporal-ember-venv/bin/python -m unittest discover -s temporal_ember_pilot/tests -p 'test_*.py' -v` → OK (14 tests); `py_compile` for the runner/engine/checkpoint/report modules → OK; CPU model initialization from a real frozen run config → 50-class head and six frozen budgets. No production training launched.
