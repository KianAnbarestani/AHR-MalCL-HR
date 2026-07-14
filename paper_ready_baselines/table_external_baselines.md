# External Baseline Comparison Table

| Dataset | Method | K / setting | Final Taskwise Accuracy | Forgetting | Macro-F1 | Weighted-F1 | Balanced Accuracy | Memory MB | Result type / role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | AHR-MalCL-HR | K=100 | 79.76 ± 1.42 | 8.38 ± 1.15 | 72.68 ± 0.59 | 80.84 ± 0.63 | 79.51 ± 0.30 | 531.8 ± 0.0 | main method |
| EMBER | ER | K=100 | 81.10 ± 2.28 | 11.58 ± 2.49 | 75.61 ± 1.41 | 81.91 ± 2.74 | 80.47 ± 0.81 | 102.8 ± 0.0 | strong memory/replay baseline |
| EMBER | DER++ | K=100 | 82.24 ± 1.94 | 10.72 ± 2.40 | 76.00 ± 1.37 | 82.61 ± 2.63 | 81.25 ± 0.95 | 106.6 ± 0.0 | strong memory/replay baseline |
| EMBER | Joint | offline upper bound | 85.88 ± 2.24 | 0.00 ± 0.00 | 81.73 ± 0.60 | 89.24 ± 0.19 | 81.32 ± 0.78 | 11.9 ± 0.0 | offline upper bound |
| AZ-Class | AHR-MalCL-HR | K=200 | 74.31 ± 1.24 | 8.43 ± 0.94 | 67.48 ± 0.74 | 71.32 ± 0.72 | 83.48 ± 0.60 | 637.1 ± 0.0 | main method |
| AZ-Class | ER | K=200 | 73.38 ± 4.19 | 15.68 ± 5.24 | 70.24 ± 2.23 | 71.55 ± 2.29 | 81.00 ± 3.11 | 198.0 ± 0.0 | strong memory/replay baseline |
| AZ-Class | DER++ | K=200 | 73.05 ± 7.16 | 16.57 ± 8.16 | 70.20 ± 2.44 | 71.64 ± 3.62 | 80.60 ± 4.74 | 205.6 ± 0.0 | strong memory/replay baseline |
| AZ-Class | Joint | offline upper bound | 84.57 ± 2.17 | 0.00 ± 0.00 | 79.28 ± 0.24 | 82.97 ± 0.15 | 80.25 ± 0.95 | 12.1 ± 0.0 | offline upper bound |
