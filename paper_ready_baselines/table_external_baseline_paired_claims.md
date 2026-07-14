# External Baseline Paired Claim Table

| Dataset | Baseline | Metric | Unit | Direction | AHR-MalCL-HR mean | Baseline mean | Difference (AHR - baseline) | Winner by mean | Holm p (paired t) | Holm significant | Claim status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | ER | Final Taskwise Accuracy | percentage points | higher is better | 79.76 | 81.10 | -1.33 | ER | 1.0000 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| EMBER | ER | Forgetting | percentage points | lower is better | 8.38 | 11.58 | -3.19 | AHR-MalCL-HR | 0.5848 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| EMBER | ER | Macro-F1 | percentage points | higher is better | 72.68 | 75.61 | -2.93 | ER | 0.3021 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| EMBER | ER | Weighted-F1 | percentage points | higher is better | 80.84 | 81.91 | -1.07 | ER | 1.0000 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| EMBER | ER | Balanced Accuracy | percentage points | higher is better | 79.51 | 80.47 | -0.96 | ER | 1.0000 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| EMBER | ER | Memory MB | MB | lower is better | 531.8 | 102.8 | 429.0 | ER | 0.00e+00 | yes | Unsafe to claim memory efficiency; mean memory is lower for ER. |
| EMBER | DER++ | Final Taskwise Accuracy | percentage points | higher is better | 79.76 | 82.24 | -2.47 | DER++ | 0.1618 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| EMBER | DER++ | Forgetting | percentage points | lower is better | 8.38 | 10.72 | -2.33 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| EMBER | DER++ | Macro-F1 | percentage points | higher is better | 72.68 | 76.00 | -3.32 | DER++ | 0.2798 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| EMBER | DER++ | Weighted-F1 | percentage points | higher is better | 80.84 | 82.61 | -1.77 | DER++ | 1.0000 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| EMBER | DER++ | Balanced Accuracy | percentage points | higher is better | 79.51 | 81.25 | -1.74 | DER++ | 0.3202 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| EMBER | DER++ | Memory MB | MB | lower is better | 531.8 | 106.6 | 425.2 | DER++ | 0.00e+00 | yes | Unsafe to claim memory efficiency; mean memory is lower for DER++. |
| EMBER | Joint | Final Taskwise Accuracy | percentage points | higher is better | 79.76 | 85.88 | -6.12 | Joint | 0.1618 | no | Offline upper-bound comparison; mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| EMBER | Joint | Forgetting | percentage points | lower is better | 8.38 | 0.00 | 8.38 | Joint | 0.0026 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| EMBER | Joint | Macro-F1 | percentage points | higher is better | 72.68 | 81.73 | -9.05 | Joint | 1.40e-04 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| EMBER | Joint | Weighted-F1 | percentage points | higher is better | 80.84 | 89.24 | -8.40 | Joint | 2.20e-04 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| EMBER | Joint | Balanced Accuracy | percentage points | higher is better | 79.51 | 81.32 | -1.81 | Joint | 0.1762 | no | Offline upper-bound comparison; mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| EMBER | Joint | Memory MB | MB | lower is better | 531.8 | 11.9 | 519.9 | Joint | 0.00e+00 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| AZ-Class | ER | Final Taskwise Accuracy | percentage points | higher is better | 74.31 | 73.38 | 0.94 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | ER | Forgetting | percentage points | lower is better | 8.43 | 15.68 | -7.24 | AHR-MalCL-HR | 0.7594 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | ER | Macro-F1 | percentage points | higher is better | 67.48 | 70.24 | -2.76 | ER | 1.0000 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| AZ-Class | ER | Weighted-F1 | percentage points | higher is better | 71.32 | 71.55 | -0.22 | ER | 1.0000 | no | Mean favors ER; do not claim AHR-MalCL-HR superiority. |
| AZ-Class | ER | Balanced Accuracy | percentage points | higher is better | 83.48 | 81.00 | 2.47 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | ER | Memory MB | MB | lower is better | 637.1 | 198.0 | 439.1 | ER | 0.00e+00 | yes | Unsafe to claim memory efficiency; mean memory is lower for ER. |
| AZ-Class | DER++ | Final Taskwise Accuracy | percentage points | higher is better | 74.31 | 73.05 | 1.27 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | DER++ | Forgetting | percentage points | lower is better | 8.43 | 16.57 | -8.13 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | DER++ | Macro-F1 | percentage points | higher is better | 67.48 | 70.20 | -2.72 | DER++ | 1.0000 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| AZ-Class | DER++ | Weighted-F1 | percentage points | higher is better | 71.32 | 71.64 | -0.32 | DER++ | 1.0000 | no | Mean favors DER++; do not claim AHR-MalCL-HR superiority. |
| AZ-Class | DER++ | Balanced Accuracy | percentage points | higher is better | 83.48 | 80.60 | 2.87 | AHR-MalCL-HR | 1.0000 | no | Mean favors AHR-MalCL-HR, but not Holm-significant externally. |
| AZ-Class | DER++ | Memory MB | MB | lower is better | 637.1 | 205.6 | 431.5 | DER++ | 0.00e+00 | yes | Unsafe to claim memory efficiency; mean memory is lower for DER++. |
| AZ-Class | Joint | Final Taskwise Accuracy | percentage points | higher is better | 74.31 | 84.57 | -10.25 | Joint | 0.0126 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| AZ-Class | Joint | Forgetting | percentage points | lower is better | 8.43 | 0.00 | 8.43 | Joint | 0.0012 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| AZ-Class | Joint | Macro-F1 | percentage points | higher is better | 67.48 | 79.28 | -11.80 | Joint | 1.40e-04 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| AZ-Class | Joint | Weighted-F1 | percentage points | higher is better | 71.32 | 82.97 | -11.65 | Joint | 9.43e-05 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
| AZ-Class | Joint | Balanced Accuracy | percentage points | higher is better | 83.48 | 80.25 | 3.23 | AHR-MalCL-HR | 0.1468 | no | Offline upper-bound comparison; mean favors AHR-MalCL-HR on this metric, but Joint is not a fair memory-limited CL baseline. |
| AZ-Class | Joint | Memory MB | MB | lower is better | 637.1 | 12.1 | 625.0 | Joint | 0.00e+00 | yes | Offline upper-bound comparison; Holm-significant mean favors Joint. Do not claim AHR-MalCL-HR beats Joint. |
