# Baseline Reference Manifest

This file tells Codex what each reference is for.

## Mandatory / Near-Mandatory References

### MalCL

Files:

* papers/malcl_2025.pdf

Links:

* https://arxiv.org/abs/2501.01110
* https://arxiv.org/pdf/2501.01110
* https://github.com/MalwareReplayGAN/MalCL

Purpose:

* Closest malware-specific generative replay baseline.
* Use to decide whether our `malcl_like` baseline is faithful enough.
* If not faithful, recommend a faithful MalCL reproduction.

Expected classification:

* Mandatory external baseline.
* Runnable if code and data protocol can be adapted.

---

### MADAR

Files:

* papers/madar_2025.pdf

Links:

* https://arxiv.org/abs/2502.05760
* https://proceedings.mlr.press/v299/rahman25a.html
* https://raw.githubusercontent.com/mlresearch/v299/main/assets/rahman25a/rahman25a.pdf
* https://github.com/IQSeC-Lab/MADAR

Purpose:

* Strong malware-specific replay baseline.
* Important comparator against real replay and diversity-aware replay.
* Must be considered for Q1 baseline coverage.

Expected classification:

* Mandatory or near-mandatory external baseline.
* Runnable if code can be adapted.

---

### DER / DER++

Files:

* papers/der_2020.pdf

Links:

* https://arxiv.org/abs/2004.07211
* https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf
* https://avalanche-api.continualai.org/en/v0.5.0/generated/avalanche.training.DER.html

Purpose:

* Strong generic replay + stored-logit distillation baseline.
* Reviewer-proof comparison for machine-learning reviewers.

Expected classification:

* Mandatory or near-mandatory runnable baseline.
* Must count stored logits in memory accounting.

---

## Strong Recommended References

### FreeMOCA

Files:

* papers/freemoca_2026.pdf

Links:

* https://arxiv.org/abs/2605.09664
* https://arxiv.org/pdf/2605.09664
* https://github.com/IQSeC-Lab/FreeMOCA

Purpose:

* Recent malware-specific replay-free continual-learning competitor.
* Strong related work and possible external baseline.

Expected classification:

* Strong recommended.
* Literature-only if reproduction is too costly.

---

### iCaRL

Files:

* papers/icarl_2017.pdf

Links:

* https://arxiv.org/abs/1611.07725
* https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf

Purpose:

* Classic class-incremental exemplar/prototype baseline.

Expected classification:

* Strong recommended, not mandatory.

---

### LwF

Files:

* papers/lwf_2016.pdf

Links:

* https://arxiv.org/abs/1606.09282
* https://arxiv.org/pdf/1606.09282
* https://github.com/lizhitwo/LearningWithoutForgetting

Purpose:

* Classic replay-free distillation baseline.

Expected classification:

* Strong recommended, especially if non-replay baselines are needed.

---

### EWC

Files:

* papers/ewc_2017.pdf

Links:

* https://www.pnas.org/doi/10.1073/pnas.1611835114
* https://arxiv.org/abs/1612.00796
* https://arxiv.org/pdf/1612.00796

Purpose:

* Classic regularization baseline.

Expected classification:

* Strong recommended or optional.
* Lower priority than DER++, MADAR, MalCL, and LwF.

---

## Optional References

### GDumb

Files:

* papers/gdumb_2020.pdf

Links:

* https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf
* https://github.com/drimpossible/GDumb

Purpose:

* Optional memory-only sanity baseline.
* Not central to replay-drift hypothesis.

Expected classification:

* Optional.

---

## Framework References

### Avalanche

Files:

* papers/avalanche_2023.pdf

Links:

* https://arxiv.org/abs/2302.01766
* https://arxiv.org/pdf/2302.01766
* https://github.com/ContinualAI/avalanche
* https://avalanche.continualai.org/
* https://github.com/ContinualAI/continual-learning-baselines

Purpose:

* Implementation reference for ER, DER, EWC, LwF, iCaRL, and other CL strategies.
* Use for implementation ideas only.
* Do not force project to depend on Avalanche if simpler custom implementation is safer.

### Mammoth

Links:

* https://github.com/aimagelab/mammoth
* https://aimagelab.github.io/mammoth/

Purpose:

* Alternative implementation reference for DER++, iCaRL, EWC, GDumb, and other CL baselines.
* Use for method details and parameter conventions.
