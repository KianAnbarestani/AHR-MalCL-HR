# Optional Baseline Implementation Notes

Optional runners were scaffolded for LwF, EWC, and iCaRL. They are secondary to DER++, MADAR, canonical ER, Joint, and the MalCL-faithfulness decision.

## Current Shared Blockers

- The exact data-stream adapter is not yet importable from notebook code.
- The `.venv_stats` environment currently lacks the full training stack if `torch` or `sklearn` are missing in dry-run output.
- No optional-baseline training has been run.

## Baseline Notes

- LwF: replay-free distillation; count teacher/previous-model checkpoint memory if retained.
- EWC: regularization baseline; count Fisher/importance tensors and previous parameter snapshot memory.
- iCaRL: exemplar/prototype baseline; count exemplars, labels, and prototype state if material.

## Last Dry-Run: iCaRL

- Missing required imports: torch, sklearn
- Data adapter is not ready: No importable class-incremental stream adapter exists yet. The trusted data-loading logic is embedded in notebooks.
- Config is template-only: Data-stream adapter is not yet available as an importable Python module.
