#!/usr/bin/env python3
"""Write JSON config templates for exact-protocol external baselines."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


VERSION = "v1 - baseline config template writer"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "baseline_suite" / "configs"

SEEDS = [42, 43, 44, 45, 46]
METRICS = [
    "mean_acc_seen",
    "final_taskwise_average_accuracy",
    "forgetting",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "old_class_accuracy",
    "new_class_accuracy",
    "final_classification_report",
    "final_confusion_matrix",
    "final_per_class_accuracy",
    "memory_MB",
    "elapsed_minutes",
]


def protocol(dataset: str, k_per_class: int | None, baseline_name: str, output_dir: str) -> dict[str, Any]:
    dataset_label = "EMBER" if dataset == "ember" else "AZ-Class"
    return {
        "schema_version": "v1",
        "dataset": dataset,
        "dataset_label": dataset_label,
        "baseline_name": baseline_name,
        "seeds": SEEDS,
        "class_stream_source": "same class stream as AHR-MalCL-HR final runs; importable adapter required",
        "task_ordering": {
            "mode": "random",
            "initial_classes": 50,
            "increment_classes_per_task": 5,
            "total_classes": 100,
            "task_count": 11,
            "seed_dependent": True,
        },
        "memory_budget": {
            "k_per_class": k_per_class,
            "unit": "examples per class" if k_per_class is not None else "offline upper bound",
            "official_comparison_budget": True if k_per_class is not None else False,
        },
        "classifier_backbone": {
            "type": "mlp",
            "hidden_layers": [1024, 512, 256],
            "dropout": 0.25,
            "embedding_dim": 512,
            "input_features": "adapter-provided feature dimension",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "momentum": 0.9,
        },
        "batch_size": 256,
        "test_batch_size": 2048,
        "epochs": 3,
        "metrics": METRICS,
        "output_directory": output_dir,
        "memory_accounting_rule": {
            "count_model_memory": True,
            "count_exemplar_features": False,
            "count_labels": False,
            "count_stored_logits": False,
            "count_generator": False,
            "count_checkpoints": False,
            "count_fisher_or_importance_state": False,
            "notes": "All future numeric comparisons must use the same accounting convention as paper_ready_stats.",
        },
        "requires_external_code": False,
        "ready_to_run": False,
        "template_only_reason": "Data-stream adapter is not yet available as an importable Python module.",
    }


def with_memory_accounting(cfg: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = deepcopy(cfg)
    out["memory_accounting_rule"].update(updates)
    return out


def derpp(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "DER++", f"result_external_baselines/DERPP/{label}/K={k}/")
    cfg.update(
        {
            "derpp": {
                "buffer_update": "reservoir",
                "store_logits": True,
                "loss_terms": ["current_task_cross_entropy", "replay_logit_mse", "replay_cross_entropy"],
                "alpha_logit_mse": "tune_or_default_0.5",
                "beta_replay_ce": "tune_or_default_1.0",
            }
        }
    )
    return with_memory_accounting(
        cfg,
        count_exemplar_features=True,
        count_labels=True,
        count_stored_logits=True,
        notes="Count exemplar feature tensor, labels, stored logits, and model memory.",
    )


def madar(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "MADAR", f"result_external_baselines/MADAR/{label}/K={k}/")
    cfg.update(
        {
            "requires_external_code": True,
            "madar": {
                "selection": "distribution-aware replay",
                "budgeting": "paper-faithful ratio/uniform variant to be selected after source inspection",
                "family_metadata_required": True,
            },
            "template_only_reason": "MADAR source is not installed locally; adapter required after approval.",
        }
    )
    return with_memory_accounting(
        cfg,
        count_exemplar_features=True,
        count_labels=True,
        notes="Count replay features, labels, model memory, and any material MADAR selection state.",
    )


def er(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "Experience Replay", f"result_external_baselines/ER/{label}/K={k}/")
    cfg.update(
        {
            "er": {
                "buffer_update": "reservoir_or_uniform_random",
                "use_generated_replay": False,
                "use_diversity_selection": False,
                "use_kd": False,
                "use_proto_align": False,
            }
        }
    )
    return with_memory_accounting(
        cfg,
        count_exemplar_features=True,
        count_labels=True,
        notes="Canonical ER must count real replay features, labels, and model memory only.",
    )


def joint(dataset: str) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, None, "Joint offline upper bound", f"result_external_baselines/Joint/{label}/")
    cfg.update(
        {
            "joint": {
                "training_data": "all classes available under the same train/test split",
                "continual_protocol": "offline upper bound, not memory-limited continual learner",
            }
        }
    )
    return cfg


def malcl_faithful(dataset: str) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, 100 if dataset == "ember" else 200, "MalCL faithful", f"result_external_baselines/MalCL-faithful/{label}/")
    cfg.update(
        {
            "requires_external_code": True,
            "malcl": {
                "gan_mode": "paper-faithful standard GAN",
                "feature_matching_loss": True,
                "replay_selection": "paper-faithful selection after source inspection",
                "classifier_backbone_note": "If MLP replaces paper CNN for tabular protocol, disclose as fair adaptation.",
            },
            "template_only_reason": "Faithful MalCL source or paper-level implementation still required.",
        }
    )
    return with_memory_accounting(
        cfg,
        count_generator=True,
        count_exemplar_features=True,
        count_labels=True,
        notes="Count generator/discriminator if retained, replay samples, labels, and classifier memory.",
    )


def lwf(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "LwF", f"result_external_baselines/LwF/{label}/K={k}/")
    cfg.update({"lwf": {"teacher": "previous task model", "distill_on": "current task data", "temperature": "tune_or_default_2.0"}})
    return with_memory_accounting(
        cfg,
        count_checkpoints=True,
        notes="Count teacher/previous-model checkpoint memory if retained for training.",
    )


def ewc(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "EWC", f"result_external_baselines/EWC/{label}/K={k}/")
    cfg.update({"ewc": {"fisher_estimation": "diagonal", "lambda": "tune_or_default", "task_boundary_required": True}})
    return with_memory_accounting(
        cfg,
        count_fisher_or_importance_state=True,
        count_checkpoints=True,
        notes="Count Fisher/importance tensors and previous-parameter snapshot memory.",
    )


def icarl(dataset: str, k: int) -> dict[str, Any]:
    label = "EMBER" if dataset == "ember" else "AZ-Class"
    cfg = protocol(dataset, k, "iCaRL", f"result_external_baselines/iCaRL/{label}/K={k}/")
    cfg.update(
        {
            "icarl": {
                "exemplar_selection": "herding",
                "classifier": "nearest_mean_of_exemplars",
                "distillation": True,
            }
        }
    )
    return with_memory_accounting(
        cfg,
        count_exemplar_features=True,
        count_labels=True,
        notes="Count exemplar features, labels, prototype/class-mean state if material, and model memory.",
    )


def configs() -> dict[str, dict[str, Any]]:
    return {
        "derpp_ember_k100.json": derpp("ember", 100),
        "derpp_az_k200.json": derpp("az_class", 200),
        "madar_ember_k100.json": madar("ember", 100),
        "madar_az_k200.json": madar("az_class", 200),
        "er_ember_k100.json": er("ember", 100),
        "er_az_k200.json": er("az_class", 200),
        "joint_ember.json": joint("ember"),
        "joint_az.json": joint("az_class"),
        "malcl_faithful_ember.json": malcl_faithful("ember"),
        "malcl_faithful_az.json": malcl_faithful("az_class"),
        "lwf_ember_k100.json": lwf("ember", 100),
        "lwf_az_k200.json": lwf("az_class", 200),
        "ewc_ember_k100.json": ewc("ember", 100),
        "ewc_az_k200.json": ewc("az_class", 200),
        "icarl_ember_k100.json": icarl("ember", 100),
        "icarl_az_k200.json": icarl("az_class", 200),
    }


def run() -> dict[str, int]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for filename, payload in configs().items():
        path = CONFIG_DIR / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written += 1
    return {"configs_written": written}


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print("Wrote {configs_written} config templates.".format(**summary))


if __name__ == "__main__":
    main()
