# Execution ledger — Temporal EMBER Pilot

- Objective: implement the approved six-window upstream EMBER 2018 v2 temporal pilot without changing the active randomized matrix.
- Constraints: no production jobs from Codex; only the requested nine run definitions; `revision_R1/` is read-only; package outputs independent; user will launch from separate Colab runtime.
- Branch at start: `codex/revision-r1-github-colab-20261028`.
- Task 1: pending.
- Task 2: pending.
- Task 3: pending.
- Task 4: pending.
- Task 1: Ruling: use `python3 -m unittest` instead of the planned `python -m pytest`; this workspace has Python 3.14, no `python` alias, and no pytest installation. The checks use only standard `unittest` assertions, so this avoids changing the shared environment. Cost if wrong: pytest-specific collection behavior would not be covered; no pytest fixtures or markers are used.
Task 1: complete (commits 122add4..122add4, tests: python3 -m unittest discover -s temporal_ember_pilot/tests -p test_leakage.py -v → OK)
