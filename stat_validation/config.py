#!/usr/bin/env python3
"""Central configuration for the AHR-MalCL statistical validation pipeline."""

from __future__ import annotations

from pathlib import Path


VERSION = "v1 - statistical validation config"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result"
OUTPUT_ROOT = PROJECT_ROOT / "paper_ready_stats"

RAW_DIR = OUTPUT_ROOT / "raw"
CHECKS_DIR = OUTPUT_ROOT / "checks"
TABLES_DIR = OUTPUT_ROOT / "tables"
STATS_DIR = OUTPUT_ROOT / "stats"
FIGURE_DATA_DIR = OUTPUT_ROOT / "figure_data"
FIGURES_DIR = OUTPUT_ROOT / "figures"

OUTPUT_DIRS = [
    RAW_DIR,
    CHECKS_DIR,
    TABLES_DIR,
    STATS_DIR,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
]

EXPECTED_SEEDS = [42, 43, 44, 45, 46]

METRICS = [
    "mean_acc_seen",
    "min_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "backward_transfer",
    "bwt",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "plasticity_stability_gap",
    "memory_MB",
    "replay_memory_MB",
    "generated_replay_MB",
    "real_buffer_MB",
    "model_memory_MB",
    "gpu_peak_memory_MB",
    "mmd_real_generated",
    "wasserstein_real_generated",
    "feature_center_l2",
    "elapsed_minutes",
]

PRIMARY_METRICS = [
    "mean_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "mmd_real_generated",
    "wasserstein_real_generated",
]

DESCRIPTIVE_METRICS = [
    "mean_acc_seen",
    "min_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "plasticity_stability_gap",
    "mmd_real_generated",
    "wasserstein_real_generated",
    "memory_MB",
    "gpu_peak_memory_MB",
    "elapsed_minutes",
]

MEMORY_BUDGET_METRICS = [
    "mean_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "memory_MB",
    "real_buffer_MB",
    "generated_replay_MB",
    "mmd_real_generated",
    "wasserstein_real_generated",
]

LOWER_IS_BETTER = {
    "forgetting",
    "mmd_real_generated",
    "wasserstein_real_generated",
    "memory_MB",
    "gpu_peak_memory_MB",
    "elapsed_minutes",
}

METRIC_ALIASES = {
    "mean_acc_seen": [
        "mean_acc_seen",
        "mean_of_seen_mean_acc",
        "mean_accuracy",
        "final_accuracy",
        "final_acc",
        "accuracy",
    ],
    "min_acc_seen": [
        "min_acc_seen",
        "mean_of_seen_min_acc",
        "min_accuracy",
        "min_acc",
    ],
    "final_taskwise_average_accuracy": [
        "final_taskwise_average_accuracy",
        "final_taskwise_average_accuracy_mean",
        "final_average_accuracy_taskwise",
        "final_taskwise_accuracy",
    ],
    "forgetting": ["forgetting", "forgetting_mean"],
    "backward_transfer": ["backward_transfer", "bwt", "bwt_mean"],
    "bwt": ["bwt", "backward_transfer", "bwt_mean"],
    "precision_macro": ["precision_macro", "precision_macro_mean"],
    "precision_weighted": ["precision_weighted", "precision_weighted_mean"],
    "recall_macro": ["recall_macro", "recall_macro_mean"],
    "recall_weighted": ["recall_weighted", "recall_weighted_mean"],
    "macro_f1": ["macro_f1", "macro_f1_mean", "f1_macro"],
    "weighted_f1": ["weighted_f1", "weighted_f1_mean", "f1_weighted"],
    "balanced_accuracy": ["balanced_accuracy", "balanced_accuracy_mean"],
    "old_class_accuracy": ["old_class_accuracy", "old_class_accuracy_mean"],
    "new_class_accuracy": ["new_class_accuracy", "new_class_accuracy_mean"],
    "plasticity_stability_gap": [
        "plasticity_stability_gap",
        "plasticity_stability_gap_mean",
    ],
    "memory_MB": ["memory_MB", "memory_MB_mean"],
    "replay_memory_MB": ["replay_memory_MB", "replay_memory_MB_mean"],
    "generated_replay_MB": ["generated_replay_MB", "generated_replay_MB_mean"],
    "real_buffer_MB": ["real_buffer_MB", "real_buffer_MB_mean"],
    "model_memory_MB": ["model_memory_MB", "model_memory_MB_mean"],
    "gpu_peak_memory_MB": ["gpu_peak_memory_MB", "gpu_peak_memory_MB_mean"],
    "mmd_real_generated": ["mmd_real_generated", "mmd_real_generated_mean"],
    "wasserstein_real_generated": [
        "wasserstein_real_generated",
        "wasserstein_real_generated_mean",
    ],
    "feature_center_l2": [
        "feature_center_l2",
        "feature_center_l2_mean",
        "class_centroid_l2",
        "class_centroid_l2_mean",
    ],
    "elapsed_minutes": ["elapsed_minutes", "elapsed_minutes_mean"],
}

METHOD_ALIASES = {
    "classifier_real_only": [
        "classifier_real_only",
        "classifier real only",
        "classifier-real-only",
    ],
    "malcl_like": ["malcl_like", "malcl-like", "malcl like"],
    "wgan_projection_generated_only": [
        "wgan_projection_generated_only",
        "wgan-projection-generated-only",
        "wgan projection generated only",
    ],
    "real_buffer_only": [
        "real_buffer_only",
        "real-buffer -only",
        "real-buffer-only",
        "real buffer only",
    ],
    "hybrid_random_buffer": [
        "hybrid_random_buffer",
        "hybrid-random-buffer",
        "hybrid random buffer",
        "ahr_malcl_hr_hybrid_random_buffer",
        "final_paper_critic5",
        "memory_k",
        "az_memory_k",
    ],
    "hybrid_diversity_buffer": [
        "hybrid_diversity_buffer",
        "hybrid-diversity-buffer",
        "hybrid diversity buffer",
    ],
    "plus_kd": ["plus_kd", "plus-kd", "plus kd"],
    "full_ahr_malcl": [
        "full_ahr_malcl",
        "full-ahr-malcl",
        "full ahr malcl",
    ],
}

DATASET_ALIASES = {
    "ember": ["ember"],
    "az_class": ["az-class", "az_class", "az class", "az"],
}

FINAL_RESULT_PATHS = {
    "ember_best_performance": RESULT_ROOT
    / "final-run/EMBER/best-performance setting/K=100/ahr_malcl_hr_hybrid_random_buffer_ember_k100_results.json",
    "az_class_best_performance": RESULT_ROOT
    / "final-run/AZ-Class/best-performance setting/K=200/ahr_malcl_hr_hybrid_random_buffer_az_k200_results.json",
    "ember_memory_efficient": RESULT_ROOT
    / "final-run/EMBER/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_ember_k25_results.json",
    "az_class_memory_efficient": RESULT_ROOT
    / "final-run/AZ-Class/memory-efficient setting/ahr_malcl_v15_1_final_paper_critic5_az_k100_results.json",
}

ABLATION_ROOT_PATHS = [
    RESULT_ROOT / "final-run/EMBER/ablations",
    RESULT_ROOT / "final-run/AZ-Class/ablation",
]

MEMORY_BUDGET_PATHS = [
    RESULT_ROOT / "EMBER/memory budget",
    RESULT_ROOT / "AZ-Class/memory budget ",
]

MAIN_METHOD = "AHR-MalCL-HR / hybrid_random_buffer"
MAIN_METHOD_CANONICAL = "hybrid_random_buffer"
FINAL_BEST_K = {"ember": 100, "az_class": 200}
FINAL_MEMORY_EFFICIENT_K = {"ember": 25, "az_class": 100}

VALIDITY_VALUES = [
    "valid_main_hr",
    "valid_memory_efficient_hr",
    "valid_ablation",
    "valid_memory_budget",
    "valid_replay_drift",
    "legacy_or_config_mismatch",
    "excluded_incomplete",
    "unknown_review_needed",
]

PAPER_USE_CATEGORIES = [
    "main_final_table",
    "memory_efficient_hr_table",
    "ablation_table",
    "memory_budget_table",
    "replay_drift_table",
    "supplementary_only",
    "exclude_from_paper_claims",
    "manual_review_needed",
]


def main() -> None:
    print(VERSION, flush=True)
    print(f"project_root={PROJECT_ROOT}")
    print(f"result_root={RESULT_ROOT}")
    print(f"output_root={OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
