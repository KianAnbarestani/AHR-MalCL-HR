# Baseline Code Links

This file lists code and implementation references for the external baseline audit.

Important rule:

Published paper numbers must not be copied into our final comparison tables unless the protocol exactly matches our datasets, task stream, memory budget, classifier, seed count, and metrics.

Use these repositories and docs to understand implementation details and to build fair adaptations under our protocol.

---

## Malware-specific baselines

### MalCL

Paper:

* MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification
* Year: 2025
* Venue: AAAI 2025
* Paper:

  * https://arxiv.org/abs/2501.01110
  * https://arxiv.org/pdf/2501.01110
  * https://ojs.aaai.org/index.php/AAAI/article/view/32047

Code:

* https://github.com/MalwareReplayGAN/MalCL

Use:

* Faithful malware-specific generative replay baseline.
* Check whether our internal `malcl_like` is faithful enough.
* If not faithful, create a separate `malcl_faithful` baseline plan.

---

### MADAR

Paper:

* MADAR: Efficient Continual Learning for Malware Analysis with Distribution-Aware / Diversity-Aware Replay
* Year: 2025
* Venue: CAMLIS / PMLR 299
* Paper:

  * https://arxiv.org/abs/2502.05760
  * https://proceedings.mlr.press/v299/rahman25a.html
  * https://raw.githubusercontent.com/mlresearch/v299/main/assets/rahman25a/rahman25a.pdf

Code:

* https://github.com/IQSeC-Lab/MADAR

Use:

* Malware-specific real replay / diversity-aware replay comparator.
* Very important external baseline.
* Should be adapted to the same EMBER/AZ stream and memory accounting if run.

---

### FreeMOCA

Paper:

* FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis
* Year: 2026
* Paper:

  * https://arxiv.org/abs/2605.09664
  * https://arxiv.org/pdf/2605.09664

Code:

* https://github.com/IQSeC-Lab/FreeMOCA

Use:

* Strong recommended baseline.
* Replay-free malware continual-learning competitor.
* If not run, cite as recent related work and explain why reproduction is outside the current controlled replay-drift comparison.

---

## Generic continual-learning baselines

### DER / DER++

Paper:

* Dark Experience for General Continual Learning
* Authors: Pietro Buzzega et al.
* Year: 2020
* Venue: NeurIPS 2020
* Paper:

  * https://arxiv.org/abs/2004.07211
  * https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf

Code / implementation references:

* Avalanche DER docs:

  * https://avalanche-api.continualai.org/en/v0.5.0/generated/avalanche.training.DER.html
* Avalanche training API:

  * https://avalanche-api.continualai.org/en/v0.6.0/training.html
* Mammoth:

  * https://github.com/aimagelab/mammoth
  * https://aimagelab.github.io/mammoth/

Use:

* Mandatory or near-mandatory strong generic replay + stored-logit distillation baseline.
* Must count exemplar memory + stored logits in memory usage.

---

### Experience Replay / ER

Implementation references:

* Avalanche:

  * https://github.com/ContinualAI/avalanche
  * https://avalanche.continualai.org/
* ContinualAI baseline suite:

  * https://github.com/ContinualAI/continual-learning-baselines
* Mammoth:

  * https://github.com/aimagelab/mammoth

Use:

* Mandatory real-replay baseline.
* If our `real_buffer_only` is canonical random/reservoir ER, it can cover ER.
* Otherwise implement canonical ER separately.

---

### iCaRL

Paper:

* iCaRL: Incremental Classifier and Representation Learning
* Authors: Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
* Year: 2017
* Venue: CVPR 2017
* Paper:

  * https://arxiv.org/abs/1611.07725
  * https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf

Implementation references:

* ContinualAI baselines:

  * https://github.com/ContinualAI/continual-learning-baselines
* Mammoth:

  * https://github.com/aimagelab/mammoth

Use:

* Strong recommended class-incremental exemplar/prototype baseline.
* Needs exemplar memory and class means/prototypes.

---

### LwF

Paper:

* Learning without Forgetting
* Authors: Zhizhong Li, Derek Hoiem
* Year: 2016 / TPAMI version 2017-2018
* Paper:

  * https://arxiv.org/abs/1606.09282
  * https://arxiv.org/pdf/1606.09282

Code:

* https://github.com/lizhitwo/LearningWithoutForgetting

Implementation references:

* ContinualAI baselines:

  * https://github.com/ContinualAI/continual-learning-baselines
* Mammoth:

  * https://github.com/aimagelab/mammoth

Use:

* Strong recommended replay-free distillation baseline.
* Does not store raw replay data.
* Uses previous model / teacher outputs.

---

### EWC

Paper:

* Overcoming Catastrophic Forgetting in Neural Networks
* Authors: James Kirkpatrick et al.
* Year: 2017
* Venue: PNAS
* Paper:

  * https://www.pnas.org/doi/10.1073/pnas.1611835114
  * https://arxiv.org/abs/1612.00796
  * https://arxiv.org/pdf/1612.00796

Implementation references:

* Avalanche EWC docs:

  * https://avalanche-api.continualai.org/en/v0.1.0/generated/avalanche.training.EWC.html
* ContinualAI baselines:

  * https://github.com/ContinualAI/continual-learning-baselines
* Mammoth:

  * https://github.com/aimagelab/mammoth

Use:

* Strong recommended / optional canonical regularization baseline.
* Requires Fisher/importance estimation.
* Usually weaker than replay methods in Class-IL but useful for reviewer coverage.

---

### GDumb

Paper:

* GDumb: A Simple Approach that Questions Our Progress in Continual Learning
* Authors: Ameya Prabhu, Philip H. S. Torr, Puneet K. Dokania
* Year: 2020
* Venue: ECCV 2020
* Paper:

  * https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf

Code:

* https://github.com/drimpossible/GDumb

Use:

* Optional.
* Good sanity baseline for memory-only retraining.
* Not central to replay-drift hypothesis.
