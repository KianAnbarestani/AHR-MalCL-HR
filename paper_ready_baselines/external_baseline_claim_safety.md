# External Baseline Claim Safety

Input root: `/home/kian/projects/shz_uni/paper-works/AHR-MalCL/result_external_baselines`.
Trusted AHR-MalCL-HR seed rows found: 10.
External seed-level rows found: 30.
External parse errors: 0.

Interpretation rule: `mean_difference_ahr_minus_baseline = AHR-MalCL-HR - external baseline`. For forgetting, lower is better, so a negative difference favors AHR-MalCL-HR. For accuracy, F1, and balanced accuracy, a positive difference favors AHR-MalCL-HR.

## EMBER

- AHR-MalCL-HR trusted K=100: final taskwise accuracy 0.7976, forgetting 0.0838.
- ER K=100: final taskwise accuracy 0.8110 +/- 0.0228, forgetting 0.1158 +/- 0.0249, seeds 42;43;44;45;46.
- DERPP K=100: final taskwise accuracy 0.8224 +/- 0.0194, forgetting 0.1072 +/- 0.0240, seeds 42;43;44;45;46.
- Joint K=offline: final taskwise accuracy 0.8588 +/- 0.0224, forgetting 0.0000 +/- 0.0000, seeds 42;43;44;45;46.

- ER metrics favoring AHR-MalCL-HR by mean: forgetting.
- ER metrics favoring ER by mean: mean_acc_seen, final_taskwise_average_accuracy, macro_f1, weighted_f1, balanced_accuracy, memory_MB, model_memory_MB.
- DERPP metrics favoring AHR-MalCL-HR by mean: forgetting.
- DERPP metrics favoring DERPP by mean: mean_acc_seen, final_taskwise_average_accuracy, macro_f1, weighted_f1, balanced_accuracy, memory_MB, model_memory_MB.
- Joint metrics favoring AHR-MalCL-HR by mean: none.
- Joint metrics favoring Joint by mean: mean_acc_seen, final_taskwise_average_accuracy, forgetting, macro_f1, weighted_f1, balanced_accuracy, memory_MB, model_memory_MB.

## AZ-Class

- AHR-MalCL-HR trusted K=200: final taskwise accuracy 0.7431, forgetting 0.0843.
- ER K=200: final taskwise accuracy 0.7338 +/- 0.0419, forgetting 0.1568 +/- 0.0524, seeds 42;43;44;45;46.
- DERPP K=200: final taskwise accuracy 0.7305 +/- 0.0716, forgetting 0.1657 +/- 0.0816, seeds 42;43;44;45;46.
- Joint K=offline: final taskwise accuracy 0.8457 +/- 0.0217, forgetting 0.0000 +/- 0.0000, seeds 42;43;44;45;46.

- ER metrics favoring AHR-MalCL-HR by mean: final_taskwise_average_accuracy, forgetting, balanced_accuracy.
- ER metrics favoring ER by mean: mean_acc_seen, macro_f1, weighted_f1, memory_MB, model_memory_MB.
- DERPP metrics favoring AHR-MalCL-HR by mean: final_taskwise_average_accuracy, forgetting, balanced_accuracy.
- DERPP metrics favoring DERPP by mean: mean_acc_seen, macro_f1, weighted_f1, memory_MB, model_memory_MB.
- Joint metrics favoring AHR-MalCL-HR by mean: balanced_accuracy.
- Joint metrics favoring Joint by mean: mean_acc_seen, final_taskwise_average_accuracy, forgetting, macro_f1, weighted_f1, memory_MB, model_memory_MB.

## Holm-Significant Differences

- EMBER ER memory_MB: Holm p=0.000000; favored ER; mean diff=429.023643.
- EMBER ER model_memory_MB: Holm p=0.000000; favored ER; mean diff=338.119408.
- EMBER DERPP memory_MB: Holm p=0.000000; favored DERPP; mean diff=425.208946.
- EMBER DERPP model_memory_MB: Holm p=0.000000; favored DERPP; mean diff=338.119408.
- EMBER Joint forgetting: Holm p=0.002585; favored Joint; mean diff=0.083846.
- EMBER Joint macro_f1: Holm p=0.000140; favored Joint; mean diff=-0.090495.
- EMBER Joint weighted_f1: Holm p=0.000220; favored Joint; mean diff=-0.084003.
- EMBER Joint memory_MB: Holm p=0.000000; favored Joint; mean diff=519.927879.
- EMBER Joint model_memory_MB: Holm p=0.000000; favored Joint; mean diff=338.119408.
- AZ-Class ER memory_MB: Holm p=0.000000; favored ER; mean diff=439.118004.
- AZ-Class ER model_memory_MB: Holm p=0.000000; favored ER; mean diff=346.001244.
- AZ-Class DERPP memory_MB: Holm p=0.000000; favored DERPP; mean diff=431.503105.
- AZ-Class DERPP model_memory_MB: Holm p=0.000000; favored DERPP; mean diff=346.001244.
- AZ-Class Joint final_taskwise_average_accuracy: Holm p=0.012639; favored Joint; mean diff=-0.102531.
- AZ-Class Joint forgetting: Holm p=0.001181; favored Joint; mean diff=0.084346.
- AZ-Class Joint macro_f1: Holm p=0.000140; favored Joint; mean diff=-0.118001.
- AZ-Class Joint weighted_f1: Holm p=0.000094; favored Joint; mean diff=-0.116488.
- AZ-Class Joint memory_MB: Holm p=0.000000; favored Joint; mean diff=624.997681.
- AZ-Class Joint model_memory_MB: Holm p=0.000000; favored Joint; mean diff=346.001244.

## Safe Claims

- No external superiority claim is Holm-significant for AHR-MalCL-HR yet.
- Joint should be described as an offline upper bound, not a memory-limited continual-learning method.

## Unsafe Claims

- Do not claim AHR-MalCL-HR beats ER on EMBER mean_acc_seen; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on EMBER final_taskwise_average_accuracy; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on EMBER macro_f1; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on EMBER weighted_f1; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on EMBER balanced_accuracy; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats DERPP on EMBER mean_acc_seen; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on EMBER final_taskwise_average_accuracy; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on EMBER macro_f1; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on EMBER weighted_f1; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on EMBER balanced_accuracy; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats Joint on EMBER mean_acc_seen; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on EMBER final_taskwise_average_accuracy; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on EMBER forgetting; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on EMBER macro_f1; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on EMBER weighted_f1; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on EMBER balanced_accuracy; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats ER on AZ-Class mean_acc_seen; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on AZ-Class macro_f1; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats ER on AZ-Class weighted_f1; the paired mean favors ER.
- Do not claim AHR-MalCL-HR beats DERPP on AZ-Class mean_acc_seen; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on AZ-Class macro_f1; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats DERPP on AZ-Class weighted_f1; the paired mean favors DERPP.
- Do not claim AHR-MalCL-HR beats Joint on AZ-Class mean_acc_seen; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on AZ-Class final_taskwise_average_accuracy; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on AZ-Class forgetting; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on AZ-Class macro_f1; the paired mean favors Joint.
- Do not claim AHR-MalCL-HR beats Joint on AZ-Class weighted_f1; the paired mean favors Joint.

## MADAR

- MADAR is still recommended as an external reference baseline unless a comparable MADAR seed-level result pack is integrated and audited.
