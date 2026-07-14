| Dataset | K | Effect | Metric | Method A | Method B | Mean A | Mean B | A - B | Improvement % | Cohen dz | Holm p | Significant 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| az_class | 100 | Generated replay effect | Mean Seen Accuracy | wgan_projection_generated_only | classifier_real_only | 32.13 | 13.78 | 18.35 | 133.18 | 4.4834 | 0.0367 | True |
| az_class | 100 | Generated replay effect | Final Taskwise Accuracy | wgan_projection_generated_only | classifier_real_only | 28.02 | 8.95 | 19.07 | 213.15 | 2.1469 | 0.3371 | False |
| az_class | 100 | Generated replay effect | Forgetting | wgan_projection_generated_only | classifier_real_only | 73.13 | 96.29 | -23.16 | 24.05 | -2.5445 | 0.1932 | False |
| az_class | 100 | Generated replay effect | Macro-F1 | wgan_projection_generated_only | classifier_real_only | 18.18 | 0.63 | 17.55 | 2765.91 | 3.2590 | 0.0999 | False |
| az_class | 100 | Generated replay effect | Balanced Accuracy | wgan_projection_generated_only | classifier_real_only | 22.45 | 5.11 | 17.33 | 339.00 | 2.8013 | 0.1524 | False |
| az_class | 100 | Generated replay effect | Old-Class Accuracy | wgan_projection_generated_only | classifier_real_only | 12.04 | 0.03 | 12.01 | 43786.74 | 2.3331 | 0.2576 | False |
| az_class | 100 | Generated replay effect | New-Class Accuracy | wgan_projection_generated_only | classifier_real_only | 95.17 | 97.83 | -2.66 | -2.72 | -0.6106 | 1.0000 | False |
| az_class | 100 | Real anchoring effect | Mean Seen Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 73.36 | 32.13 | 41.23 | 128.31 | 10.3216 | 0.0031 | True |
| az_class | 100 | Real anchoring effect | Final Taskwise Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 71.29 | 28.02 | 43.28 | 154.47 | 5.0894 | 0.0287 | True |
| az_class | 100 | Real anchoring effect | Forgetting | hybrid_random_buffer | wgan_projection_generated_only | 12.89 | 73.13 | -60.24 | 82.37 | -6.0362 | 0.0173 | True |
| az_class | 100 | Real anchoring effect | Macro-F1 | hybrid_random_buffer | wgan_projection_generated_only | 64.05 | 18.18 | 45.87 | 252.24 | 8.1261 | 0.0069 | True |
| az_class | 100 | Real anchoring effect | Balanced Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 79.59 | 22.45 | 57.15 | 254.58 | 9.4399 | 0.0040 | True |
| az_class | 100 | Real anchoring effect | Old-Class Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 67.43 | 12.04 | 55.40 | 460.26 | 9.9387 | 0.0036 | True |
| az_class | 100 | Real anchoring effect | New-Class Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 73.89 | 95.17 | -21.28 | -22.36 | -1.3185 | 1.0000 | False |
| az_class | 100 | Real anchoring effect | MMD | hybrid_random_buffer | wgan_projection_generated_only | 0.0026 | 0.1134 | -0.1108 | 97.74 | -4.4471 | 0.0368 | True |
| az_class | 100 | Real anchoring effect | Wasserstein | hybrid_random_buffer | wgan_projection_generated_only | 0.0532 | 0.3264 | -0.2732 | 83.70 | -5.6293 | 0.0213 | True |
| az_class | 100 | Hybrid vs real-only | Mean Seen Accuracy | hybrid_random_buffer | real_buffer_only | 73.36 | 68.27 | 5.09 | 7.46 | 6.3324 | 0.0146 | True |
| az_class | 100 | Hybrid vs real-only | Final Taskwise Accuracy | hybrid_random_buffer | real_buffer_only | 71.29 | 61.82 | 9.47 | 15.32 | 4.6274 | 0.0351 | True |
| az_class | 100 | Hybrid vs real-only | Forgetting | hybrid_random_buffer | real_buffer_only | 12.89 | 18.30 | -5.41 | 29.57 | -1.7403 | 0.6179 | False |
| az_class | 100 | Hybrid vs real-only | Macro-F1 | hybrid_random_buffer | real_buffer_only | 64.05 | 55.23 | 8.82 | 15.97 | 4.5513 | 0.0362 | True |
| az_class | 100 | Hybrid vs real-only | Balanced Accuracy | hybrid_random_buffer | real_buffer_only | 79.59 | 75.16 | 4.43 | 5.89 | 5.5861 | 0.0213 | True |
| az_class | 100 | Hybrid vs real-only | Old-Class Accuracy | hybrid_random_buffer | real_buffer_only | 67.43 | 57.38 | 10.05 | 17.51 | 4.9538 | 0.0295 | True |
| az_class | 100 | Hybrid vs real-only | New-Class Accuracy | hybrid_random_buffer | real_buffer_only | 73.89 | 69.71 | 4.17 | 5.99 | 1.1024 | 1.0000 | False |
| az_class | 100 | Diversity effect | Mean Seen Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 67.11 | 73.36 | -6.25 | -8.52 | -7.9471 | 0.0072 | True |
| az_class | 100 | Diversity effect | Final Taskwise Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 63.42 | 71.29 | -7.87 | -11.04 | -2.8503 | 0.1492 | False |
| az_class | 100 | Diversity effect | Forgetting | hybrid_diversity_buffer | hybrid_random_buffer | 21.00 | 12.89 | 8.11 | -62.87 | 2.7428 | 0.1540 | False |
| az_class | 100 | Diversity effect | Macro-F1 | hybrid_diversity_buffer | hybrid_random_buffer | 55.43 | 64.05 | -8.62 | -13.46 | -9.8225 | 0.0037 | True |
| az_class | 100 | Diversity effect | Balanced Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 74.57 | 79.59 | -5.02 | -6.31 | -8.0956 | 0.0069 | True |
| az_class | 100 | Diversity effect | Old-Class Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 56.79 | 67.43 | -10.64 | -15.78 | -9.1253 | 0.0046 | True |
| az_class | 100 | Diversity effect | New-Class Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 75.19 | 73.89 | 1.31 | 1.77 | 0.3510 | 1.0000 | False |
| az_class | 100 | Diversity effect | MMD | hybrid_diversity_buffer | hybrid_random_buffer | 0.0034 | 0.0026 | 0.0008 | -30.98 | 0.6815 | 1.0000 | False |
| az_class | 100 | Diversity effect | Wasserstein | hybrid_diversity_buffer | hybrid_random_buffer | 0.0556 | 0.0532 | 0.0024 | -4.51 | 0.2617 | 1.0000 | False |
| az_class | 100 | KD effect | Mean Seen Accuracy | plus_kd | hybrid_random_buffer | 68.84 | 73.36 | -4.52 | -6.16 | -4.6559 | 0.0351 | True |
| az_class | 100 | KD effect | Final Taskwise Accuracy | plus_kd | hybrid_random_buffer | 64.54 | 71.29 | -6.75 | -9.47 | -3.6215 | 0.0758 | False |
| az_class | 100 | KD effect | Forgetting | plus_kd | hybrid_random_buffer | 19.71 | 12.89 | 6.81 | -52.87 | 2.7857 | 0.1524 | False |
| az_class | 100 | KD effect | Macro-F1 | plus_kd | hybrid_random_buffer | 56.52 | 64.05 | -7.53 | -11.75 | -5.0976 | 0.0287 | True |
| az_class | 100 | KD effect | Balanced Accuracy | plus_kd | hybrid_random_buffer | 76.10 | 79.59 | -3.49 | -4.38 | -4.4193 | 0.0368 | True |
| az_class | 100 | KD effect | Old-Class Accuracy | plus_kd | hybrid_random_buffer | 58.47 | 67.43 | -8.97 | -13.30 | -4.9694 | 0.0295 | True |
| az_class | 100 | KD effect | New-Class Accuracy | plus_kd | hybrid_random_buffer | 72.37 | 73.89 | -1.51 | -2.05 | -1.2005 | 1.0000 | False |
| az_class | 100 | KD effect | MMD | plus_kd | hybrid_random_buffer | 0.0028 | 0.0026 | 0.0003 | -10.70 | 0.2327 | 1.0000 | False |
| az_class | 100 | KD effect | Wasserstein | plus_kd | hybrid_random_buffer | 0.0418 | 0.0532 | -0.0114 | 21.47 | -1.3233 | 1.0000 | False |
| ember | 25 | Generated replay effect | Mean Seen Accuracy | wgan_projection_generated_only | classifier_real_only | 47.28 | 13.34 | 33.94 | 254.31 | 7.2291 | 0.0097 | True |
| ember | 25 | Generated replay effect | Final Taskwise Accuracy | wgan_projection_generated_only | classifier_real_only | 42.05 | 8.89 | 33.15 | 372.81 | 6.8257 | 0.0117 | True |
| ember | 25 | Generated replay effect | Forgetting | wgan_projection_generated_only | classifier_real_only | 58.40 | 96.66 | -38.27 | 39.59 | -7.3760 | 0.0090 | True |
| ember | 25 | Generated replay effect | Macro-F1 | wgan_projection_generated_only | classifier_real_only | 25.21 | 0.71 | 24.50 | 3456.20 | 7.9223 | 0.0072 | True |
| ember | 25 | Generated replay effect | Balanced Accuracy | wgan_projection_generated_only | classifier_real_only | 29.18 | 4.88 | 24.30 | 497.95 | 8.6250 | 0.0056 | True |
| ember | 25 | Generated replay effect | Old-Class Accuracy | wgan_projection_generated_only | classifier_real_only | 20.45 | 0.00 | 20.45 |  | 5.9402 | 0.0180 | True |
| ember | 25 | Generated replay effect | New-Class Accuracy | wgan_projection_generated_only | classifier_real_only | 94.46 | 97.83 | -3.36 | -3.44 | -1.0557 | 1.0000 | False |
| ember | 25 | Real anchoring effect | Mean Seen Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 79.50 | 47.28 | 32.22 | 68.15 | 6.4607 | 0.0137 | True |
| ember | 25 | Real anchoring effect | Final Taskwise Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 74.28 | 42.05 | 32.23 | 76.64 | 6.6977 | 0.0125 | True |
| ember | 25 | Real anchoring effect | Forgetting | hybrid_random_buffer | wgan_projection_generated_only | 18.22 | 58.40 | -40.17 | 68.80 | -7.9048 | 0.0072 | True |
| ember | 25 | Real anchoring effect | Macro-F1 | hybrid_random_buffer | wgan_projection_generated_only | 67.39 | 25.21 | 42.18 | 167.31 | 15.4479 | 0.0007 | True |
| ember | 25 | Real anchoring effect | Balanced Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 73.50 | 29.18 | 44.32 | 151.92 | 17.0589 | 0.0004 | True |
| ember | 25 | Real anchoring effect | Old-Class Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 72.25 | 20.45 | 51.79 | 253.21 | 18.2903 | 0.0003 | True |
| ember | 25 | Real anchoring effect | New-Class Accuracy | hybrid_random_buffer | wgan_projection_generated_only | 87.35 | 94.46 | -7.12 | -7.53 | -1.7408 | 0.6179 | False |
| ember | 25 | Real anchoring effect | MMD | hybrid_random_buffer | wgan_projection_generated_only | 0.0025 | 0.0802 | -0.0777 | 96.92 | -5.0551 | 0.0287 | True |
| ember | 25 | Real anchoring effect | Wasserstein | hybrid_random_buffer | wgan_projection_generated_only | 0.0541 | 0.2707 | -0.2167 | 80.03 | -12.5956 | 0.0015 | True |
| ember | 25 | Hybrid vs real-only | Mean Seen Accuracy | hybrid_random_buffer | real_buffer_only | 79.50 | 70.86 | 8.64 | 12.20 | 9.6859 | 0.0037 | True |
| ember | 25 | Hybrid vs real-only | Final Taskwise Accuracy | hybrid_random_buffer | real_buffer_only | 74.28 | 59.38 | 14.90 | 25.09 | 7.8376 | 0.0073 | True |
| ember | 25 | Hybrid vs real-only | Forgetting | hybrid_random_buffer | real_buffer_only | 18.22 | 29.38 | -11.16 | 37.98 | -5.6279 | 0.0213 | True |
| ember | 25 | Hybrid vs real-only | Macro-F1 | hybrid_random_buffer | real_buffer_only | 67.39 | 47.47 | 19.92 | 41.97 | 12.4750 | 0.0015 | True |
| ember | 25 | Hybrid vs real-only | Balanced Accuracy | hybrid_random_buffer | real_buffer_only | 73.50 | 58.65 | 14.85 | 25.32 | 6.8619 | 0.0117 | True |
| ember | 25 | Hybrid vs real-only | Old-Class Accuracy | hybrid_random_buffer | real_buffer_only | 72.25 | 54.22 | 18.03 | 33.25 | 7.5161 | 0.0085 | True |
| ember | 25 | Hybrid vs real-only | New-Class Accuracy | hybrid_random_buffer | real_buffer_only | 87.35 | 81.68 | 5.67 | 6.94 | 4.7424 | 0.0336 | True |
| ember | 25 | Diversity effect | Mean Seen Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 71.87 | 79.50 | -7.63 | -9.60 | -5.8005 | 0.0194 | True |
| ember | 25 | Diversity effect | Final Taskwise Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 63.46 | 74.28 | -10.82 | -14.56 | -6.6284 | 0.0128 | True |
| ember | 25 | Diversity effect | Forgetting | hybrid_diversity_buffer | hybrid_random_buffer | 28.17 | 18.22 | 9.95 | -54.59 | 5.2158 | 0.0269 | True |
| ember | 25 | Diversity effect | Macro-F1 | hybrid_diversity_buffer | hybrid_random_buffer | 49.04 | 67.39 | -18.35 | -27.23 | -8.3781 | 0.0062 | True |
| ember | 25 | Diversity effect | Balanced Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 59.82 | 73.50 | -13.68 | -18.61 | -8.4367 | 0.0061 | True |
| ember | 25 | Diversity effect | Old-Class Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 55.76 | 72.25 | -16.49 | -22.82 | -3.4621 | 0.0870 | False |
| ember | 25 | Diversity effect | New-Class Accuracy | hybrid_diversity_buffer | hybrid_random_buffer | 86.66 | 87.35 | -0.69 | -0.79 | -0.4899 | 1.0000 | False |
| ember | 25 | Diversity effect | MMD | hybrid_diversity_buffer | hybrid_random_buffer | 0.0043 | 0.0025 | 0.0018 | -73.43 | 0.9150 | 1.0000 | False |
| ember | 25 | Diversity effect | Wasserstein | hybrid_diversity_buffer | hybrid_random_buffer | 0.0677 | 0.0541 | 0.0136 | -25.22 | 2.0648 | 0.3763 | False |
| ember | 25 | KD effect | Mean Seen Accuracy | plus_kd | hybrid_random_buffer | 73.87 | 79.50 | -5.63 | -7.08 | -3.0250 | 0.1246 | False |
| ember | 25 | KD effect | Final Taskwise Accuracy | plus_kd | hybrid_random_buffer | 67.15 | 74.28 | -7.12 | -9.59 | -4.5024 | 0.0367 | True |
| ember | 25 | KD effect | Forgetting | plus_kd | hybrid_random_buffer | 25.00 | 18.22 | 6.78 | -37.20 | 3.4009 | 0.0899 | False |
| ember | 25 | KD effect | Macro-F1 | plus_kd | hybrid_random_buffer | 51.86 | 67.39 | -15.53 | -23.05 | -9.8492 | 0.0037 | True |
| ember | 25 | KD effect | Balanced Accuracy | plus_kd | hybrid_random_buffer | 63.14 | 73.50 | -10.37 | -14.10 | -9.8505 | 0.0037 | True |
| ember | 25 | KD effect | Old-Class Accuracy | plus_kd | hybrid_random_buffer | 58.50 | 72.25 | -13.75 | -19.03 | -3.1742 | 0.1082 | False |
| ember | 25 | KD effect | New-Class Accuracy | plus_kd | hybrid_random_buffer | 86.82 | 87.35 | -0.52 | -0.60 | -0.3419 | 1.0000 | False |
| ember | 25 | KD effect | MMD | plus_kd | hybrid_random_buffer | 0.0087 | 0.0025 | 0.0063 | -253.56 | 1.6836 | 0.6179 | False |
| ember | 25 | KD effect | Wasserstein | plus_kd | hybrid_random_buffer | 0.0684 | 0.0541 | 0.0143 | -26.41 | 1.1461 | 1.0000 | False |
