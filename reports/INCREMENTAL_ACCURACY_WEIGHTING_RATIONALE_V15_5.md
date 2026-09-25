# Incremental-accuracy weighting rationale V15.5

V15.5 reports both **Sample-Weighted Average Incremental Accuracy (SW-AIA)** and
**Task-Macro Average Incremental Accuracy (TM-AIA)**. SW-AIA gives every test
example equal stage weight, so the 50-class initial task and classes with more
test samples exert greater influence. TM-AIA gives every observed task equal
stage weight and therefore exposes performance on the ten later five-class
increments. Neither is universally correct: SW-AIA is operationally
sample-oriented, whereas TM-AIA is task-balanced and aligns directly with the
stability–plasticity question. Both appear in the main dynamics table; ranking
differences are interpreted as weighting sensitivity rather than error.
