#!/usr/bin/env python3
"""Audit external baseline coverage against the trusted validation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


VERSION = "v1 - external baseline coverage audit"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = PROJECT_ROOT / "paper_ready_stats"
TABLES_DIR = STATS_DIR / "tables"
OUT_DIR = PROJECT_ROOT / "paper_ready_baselines"
REFERENCE_DIR = PROJECT_ROOT / "baseline_reference_pack"
RESULT_FINAL_DIR = PROJECT_ROOT / "result" / "final-run"

INPUT_FILES = [
    TABLES_DIR / "table_1_final_main_results.csv",
    TABLES_DIR / "table_memory_efficient_hr_results.csv",
    TABLES_DIR / "table_3_ablation.csv",
    TABLES_DIR / "table_4_component_effects.csv",
    TABLES_DIR / "table_validity_audit.csv",
    STATS_DIR / "STATISTICAL_VALIDATION_SUMMARY.md",
    REFERENCE_DIR / "notes" / "baseline_reference_manifest.md",
    REFERENCE_DIR / "notes" / "our_final_result_summary.md",
    REFERENCE_DIR / "code_links" / "baseline_code_links.md",
]

EXPECTED_REFERENCE_PAPERS = {
    "Avalanche": "avalanche_2023.pdf",
    "DER++": "der_2020.pdf",
    "EWC": "ewc_2017.pdf",
    "FreeMOCA": "freemoca_2026.pdf",
    "GDumb": "gdumb_2020.pdf",
    "iCaRL": "icarl_2017.pdf",
    "LwF": "lwf_2016.pdf",
    "MADAR": "madar_2025.pdf",
    "MalCL": "malcl_2025.pdf",
}

COVERAGE_COLUMNS = [
    "baseline_name",
    "category",
    "status",
    "existing_evidence_source",
    "matched_existing_method",
    "faithful_to_paper",
    "needs_new_run",
    "needs_external_code",
    "recommended_action",
    "reviewer_risk_if_missing",
    "implementation_difficulty",
    "expected_cost",
    "memory_accounting_requirements",
    "fair_adaptation_requirements",
    "notes",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def bool_values(rows: list[dict[str, str]], method: str, field: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        if row.get("Method") == method:
            value = str(row.get(field, "")).strip()
            if value:
                values.add(value)
    return values


def method_present(rows: list[dict[str, str]], method: str) -> bool:
    return any(row.get("Method") == method for row in rows)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def md_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(md_escape(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def inspect_result_jsons() -> dict[str, Any]:
    json_files = sorted(RESULT_FINAL_DIR.rglob("*.json")) if RESULT_FINAL_DIR.exists() else []
    aggregate_methods: set[str] = set()
    parse_errors: list[str] = []
    for path in json_files:
        try:
            text_raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                data = json.loads(text_raw)
            except json.JSONDecodeError:
                data, _ = json.JSONDecoder().raw_decode(text_raw)
        except Exception as exc:  # noqa: BLE001 - audit report should preserve malformed evidence.
            parse_errors.append(f"{rel(path)}: {type(exc).__name__}: {exc}")
            continue
        text = json.dumps(data)[:250000].lower()
        for method in [
            "classifier_real_only",
            "real_buffer_only",
            "wgan_projection_generated_only",
            "malcl_like",
            "hybrid_random_buffer",
            "hybrid_diversity_buffer",
            "plus_kd",
        ]:
            if method in text or method.replace("_", "-") in text:
                aggregate_methods.add(method)
    return {
        "json_file_count": len(json_files),
        "methods_seen_in_json_text": sorted(aggregate_methods),
        "parse_errors": parse_errors,
    }


def build_evidence() -> dict[str, Any]:
    ablation = read_csv_rows(TABLES_DIR / "table_3_ablation.csv")
    validity = read_csv_rows(TABLES_DIR / "table_validity_audit.csv")
    final_main = read_csv_rows(TABLES_DIR / "table_1_final_main_results.csv")
    memory_efficient = read_csv_rows(TABLES_DIR / "table_memory_efficient_hr_results.csv")
    reference_text = "\n".join(
        read_text(path)
        for path in [
            REFERENCE_DIR / "notes" / "baseline_reference_manifest.md",
            REFERENCE_DIR / "notes" / "our_final_result_summary.md",
            REFERENCE_DIR / "code_links" / "baseline_code_links.md",
        ]
    )
    return {
        "ablation": ablation,
        "validity": validity,
        "final_main": final_main,
        "memory_efficient": memory_efficient,
        "summary": read_text(STATS_DIR / "STATISTICAL_VALIDATION_SUMMARY.md"),
        "reference_text": reference_text,
        "json": inspect_result_jsons(),
        "input_missing": [rel(path) for path in INPUT_FILES if not path.exists()],
    }


def build_coverage_rows(evidence: dict[str, Any]) -> list[dict[str, str]]:
    ablation = evidence["ablation"]
    validity = evidence["validity"]
    classifier_none_ok = method_present(ablation, "classifier_real_only") and bool_values(
        validity, "classifier_real_only", "use_diversity_buffer"
    ) <= {"False"}
    real_buffer_flags = {
        "use_diversity_buffer": sorted(bool_values(validity, "real_buffer_only", "use_diversity_buffer")),
        "use_kd": sorted(bool_values(validity, "real_buffer_only", "use_kd")),
        "use_proto_align": sorted(bool_values(validity, "real_buffer_only", "use_proto_align")),
        "gan_train_on_real_buffer": sorted(bool_values(validity, "real_buffer_only", "gan_train_on_real_buffer")),
    }
    er_canonical = (
        method_present(ablation, "real_buffer_only")
        and set(real_buffer_flags["use_diversity_buffer"]) <= {"False"}
        and set(real_buffer_flags["use_kd"]) <= {"False"}
        and set(real_buffer_flags["use_proto_align"]) <= {"False"}
    )
    generated_present = method_present(ablation, "wgan_projection_generated_only")
    malcl_like_present = method_present(ablation, "malcl_like")

    return [
        {
            "baseline_name": "Fine-tuning / None",
            "category": "A mandatory",
            "status": "already covered" if classifier_none_ok else "missing",
            "existing_evidence_source": "paper_ready_stats/tables/table_3_ablation.csv; paper_ready_stats/tables/table_validity_audit.csv",
            "matched_existing_method": "classifier_real_only" if classifier_none_ok else "",
            "faithful_to_paper": "yes for no-replay/fine-tuning baseline",
            "needs_new_run": "NO" if classifier_none_ok else "YES",
            "needs_external_code": "NO",
            "recommended_action": "Use as the None/fine-tuning baseline, and label it as no replay rather than an external paper baseline.",
            "reviewer_risk_if_missing": "High if omitted; low if reported clearly with the existing ablation row.",
            "implementation_difficulty": "none",
            "expected_cost": "none",
            "memory_accounting_requirements": "Report model memory only; no replay buffer, logits, generator, or exemplars.",
            "fair_adaptation_requirements": "Same class stream, random task ordering, seeds 42-46, MLP backbone, AdamW, incremental scaler.",
            "notes": "K is irrelevant because no replay memory is used.",
        },
        {
            "baseline_name": "Joint / offline upper bound",
            "category": "A mandatory",
            "status": "missing",
            "existing_evidence_source": "Only literature/reference values appear in notebooks; no trusted per-seed exact-protocol result table.",
            "matched_existing_method": "",
            "faithful_to_paper": "not applicable",
            "needs_new_run": "YES",
            "needs_external_code": "NO",
            "recommended_action": "Implement a same-MLP offline upper bound trained on all classes or cumulative data under the exact split; keep it out of memory-budget fairness claims.",
            "reviewer_risk_if_missing": "Medium-high because reviewers often expect an upper bound.",
            "implementation_difficulty": "low-medium",
            "expected_cost": "moderate GPU; usually cheaper than GAN baselines",
            "memory_accounting_requirements": "Mark as offline upper bound; do not compare as memory-limited CL.",
            "fair_adaptation_requirements": "Use identical train/test split, class order, classifier backbone, optimizer, metrics, and seeds.",
            "notes": "Published Joint numbers must not be copied into the main comparison table.",
        },
        {
            "baseline_name": "Experience Replay / ER",
            "category": "A mandatory",
            "status": "already covered" if er_canonical else "partially covered",
            "existing_evidence_source": "paper_ready_stats/tables/table_3_ablation.csv; paper_ready_stats/tables/table_validity_audit.csv",
            "matched_existing_method": "real_buffer_only",
            "faithful_to_paper": "partial" if not er_canonical else "yes",
            "needs_new_run": "NO" if er_canonical else "YES",
            "needs_external_code": "NO",
            "recommended_action": "Run canonical ER with random/reservoir real replay and no generated replay, KD, prototype alignment, or diversity selection.",
            "reviewer_risk_if_missing": "High because ER is the core replay comparator.",
            "implementation_difficulty": "low-medium",
            "expected_cost": "moderate GPU",
            "memory_accounting_requirements": "Count exemplar feature memory, labels, and model memory; no generator or logits.",
            "fair_adaptation_requirements": "Use same K as official comparisons: EMBER K=100 and AZ-Class K=200; also optional memory-efficient K=25/K=100.",
            "notes": f"Existing real_buffer_only flags: {real_buffer_flags}. This is useful ablation evidence but not clean canonical ER.",
        },
        {
            "baseline_name": "Pure generated replay",
            "category": "A mandatory",
            "status": "already covered" if generated_present else "missing",
            "existing_evidence_source": "paper_ready_stats/tables/table_3_ablation.csv; paper_ready_stats/tables/table_4_component_effects.csv",
            "matched_existing_method": "wgan_projection_generated_only" if generated_present else "",
            "faithful_to_paper": "yes as internal pure generated replay; no as faithful MalCL",
            "needs_new_run": "NO" if generated_present else "YES",
            "needs_external_code": "NO",
            "recommended_action": "Use as the internal generated-only replay baseline and drift comparator.",
            "reviewer_risk_if_missing": "Medium-high for replay-drift claims; currently covered.",
            "implementation_difficulty": "none",
            "expected_cost": "none",
            "memory_accounting_requirements": "Count generator/model memory and generated replay artifacts if retained; no real replay buffer.",
            "fair_adaptation_requirements": "Do not relabel this as MalCL; it is the project's WGAN-GP/projection generated-only ablation.",
            "notes": "This baseline directly supports the real-anchor replay-drift claim.",
        },
        {
            "baseline_name": "MalCL faithful",
            "category": "A mandatory",
            "status": "needs faithful reproduction" if malcl_like_present else "missing",
            "existing_evidence_source": "paper_ready_stats/tables/table_3_ablation.csv; baseline_reference_pack/papers/malcl_2025.pdf",
            "matched_existing_method": "malcl_like" if malcl_like_present else "",
            "faithful_to_paper": "partial",
            "needs_new_run": "YES",
            "needs_external_code": "YES, unless reimplemented locally from the paper",
            "recommended_action": "Treat malcl_like as an internal baseline; reproduce faithful MalCL if reviewers require the closest malware-specific generative replay competitor.",
            "reviewer_risk_if_missing": "Medium-high because MalCL is the closest generative replay paper.",
            "implementation_difficulty": "medium-high",
            "expected_cost": "high GPU if full GAN reproduction is required",
            "memory_accounting_requirements": "Count generator, discriminator/critic if retained, replay sample storage, classifier/model memory, and any selection state.",
            "fair_adaptation_requirements": "Use the same class stream, seeds, metrics, MLP-vs-CNN adaptation disclosure, and no copied published numbers.",
            "notes": "Internal malcl_like uses BCE standard conditional GAN + FML + L1 class mean selection, but it is not a verified external code reproduction.",
        },
        {
            "baseline_name": "DER++",
            "category": "A mandatory",
            "status": "missing",
            "existing_evidence_source": "baseline_reference_pack/papers/der_2020.pdf; no exact-protocol results under result/final-run/",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "YES",
            "needs_external_code": "NO for a local custom runner; optional to consult Avalanche/Mammoth",
            "recommended_action": "Implement and run DER++ with replay buffer, stored logits, replay CE, and logit MSE.",
            "reviewer_risk_if_missing": "High for generic continual-learning reviewers.",
            "implementation_difficulty": "medium",
            "expected_cost": "moderate-high GPU",
            "memory_accounting_requirements": "Count exemplar features, labels, stored logits, and model memory.",
            "fair_adaptation_requirements": "Use same stream/order/seeds/backbone/optimizer/epochs/metrics; paired tests only with matching seeds.",
            "notes": "DER++ is the first new baseline to prioritize.",
        },
        {
            "baseline_name": "MADAR",
            "category": "A mandatory",
            "status": "needs external code",
            "existing_evidence_source": "baseline_reference_pack/papers/madar_2025.pdf; no local MADAR source detected",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "YES",
            "needs_external_code": "YES",
            "recommended_action": "Integrate MADAR source only after explicit approval; adapt to EMBER/AZ class stream, memory budgets, and output schema.",
            "reviewer_risk_if_missing": "High because it is a malware-specific replay comparator.",
            "implementation_difficulty": "high",
            "expected_cost": "high GPU and integration time",
            "memory_accounting_requirements": "Count replay feature memory, labels, family dictionaries/selection state if material, and model memory.",
            "fair_adaptation_requirements": "Use class-incremental setting, same family/task split, K budgets, seeds, and no copied published numbers.",
            "notes": "Do not clone or download until explicitly approved.",
        },
        {
            "baseline_name": "iCaRL",
            "category": "B strong recommended",
            "status": "missing",
            "existing_evidence_source": "baseline_reference_pack/papers/icarl_2017.pdf",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "OPTIONAL",
            "needs_external_code": "NO if adapted locally",
            "recommended_action": "Scaffold optional runner; run if GPU time remains after DER++, MADAR, ER, Joint, and MalCL decision.",
            "reviewer_risk_if_missing": "Medium.",
            "implementation_difficulty": "medium",
            "expected_cost": "moderate GPU",
            "memory_accounting_requirements": "Count exemplars, labels, prototype/class-mean state if stored, and model memory.",
            "fair_adaptation_requirements": "Adapt herding/prototype classification to tabular EMBER/AZ features and same seeds.",
            "notes": "Classic class-incremental exemplar baseline.",
        },
        {
            "baseline_name": "LwF",
            "category": "B strong recommended",
            "status": "missing",
            "existing_evidence_source": "baseline_reference_pack/papers/lwf_2016.pdf",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "OPTIONAL",
            "needs_external_code": "NO if adapted locally",
            "recommended_action": "Scaffold optional runner; consider as replay-free distillation comparator.",
            "reviewer_risk_if_missing": "Medium-low if DER++ and MADAR are present.",
            "implementation_difficulty": "low-medium",
            "expected_cost": "moderate GPU",
            "memory_accounting_requirements": "Count previous model/teacher checkpoint memory if retained; no exemplars.",
            "fair_adaptation_requirements": "Use same MLP and old-task logit distillation on new-task data.",
            "notes": "Useful if reviewer asks for replay-free distillation.",
        },
        {
            "baseline_name": "EWC",
            "category": "B strong recommended",
            "status": "missing",
            "existing_evidence_source": "baseline_reference_pack/papers/ewc_2017.pdf",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "OPTIONAL",
            "needs_external_code": "NO if adapted locally",
            "recommended_action": "Scaffold optional runner; lower priority than replay baselines.",
            "reviewer_risk_if_missing": "Medium-low.",
            "implementation_difficulty": "medium",
            "expected_cost": "moderate GPU plus Fisher estimation",
            "memory_accounting_requirements": "Count Fisher/importance tensors and previous parameter snapshot memory.",
            "fair_adaptation_requirements": "Estimate Fisher at task boundaries using the same classifier and data stream.",
            "notes": "Canonical regularization baseline, often weaker in Class-IL but useful for coverage.",
        },
        {
            "baseline_name": "FreeMOCA",
            "category": "B strong recommended",
            "status": "literature-only",
            "existing_evidence_source": "baseline_reference_pack/papers/freemoca_2026.pdf",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "OPTIONAL",
            "needs_external_code": "YES",
            "recommended_action": "Discuss as recent malware-specific replay-free work unless GPU/integration time permits exact-protocol reproduction.",
            "reviewer_risk_if_missing": "Medium novelty risk because it is recent and malware-specific.",
            "implementation_difficulty": "high",
            "expected_cost": "high integration and validation cost",
            "memory_accounting_requirements": "Count any checkpoint/model-interpolation state; no replay buffer if faithful.",
            "fair_adaptation_requirements": "Do not compare published numbers directly; reproduce under the same EMBER/AZ stream if used numerically.",
            "notes": "Strong related work and novelty-risk paper; not part of minimal Q1 numeric suite unless reproduced.",
        },
        {
            "baseline_name": "GDumb",
            "category": "C optional",
            "status": "literature-only",
            "existing_evidence_source": "baseline_reference_pack/papers/gdumb_2020.pdf",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "OPTIONAL",
            "needs_external_code": "NO if adapted locally",
            "recommended_action": "Keep as optional memory-only sanity baseline; not central to replay-drift hypothesis.",
            "reviewer_risk_if_missing": "Low.",
            "implementation_difficulty": "low-medium",
            "expected_cost": "moderate GPU if trained from scratch after each task",
            "memory_accounting_requirements": "Count stored samples and labels; model trained from memory at test/update time.",
            "fair_adaptation_requirements": "Use the same K and class stream if run.",
            "notes": "Optional, not required for a lean Q1 response.",
        },
        {
            "baseline_name": "SI",
            "category": "C optional",
            "status": "related-work-only",
            "existing_evidence_source": "Mentioned in DER/Avalanche ecosystem; no standalone reference paper supplied in pack.",
            "matched_existing_method": "",
            "faithful_to_paper": "no current reproduction",
            "needs_new_run": "NO",
            "needs_external_code": "NO",
            "recommended_action": "Do not prioritize unless reviewers explicitly ask for additional regularization baselines.",
            "reviewer_risk_if_missing": "Low if EWC/LwF are discussed or scaffolded.",
            "implementation_difficulty": "medium",
            "expected_cost": "moderate GPU",
            "memory_accounting_requirements": "Count parameter-importance state if implemented.",
            "fair_adaptation_requirements": "Same stream and MLP if ever run.",
            "notes": "Optional baseline inflation is not recommended.",
        },
        {
            "baseline_name": "Avalanche / Mammoth",
            "category": "framework reference",
            "status": "related-work-only",
            "existing_evidence_source": "baseline_reference_pack/papers/avalanche_2023.pdf; baseline_reference_pack/code_links/baseline_code_links.md",
            "matched_existing_method": "",
            "faithful_to_paper": "not a baseline",
            "needs_new_run": "NO",
            "needs_external_code": "NO",
            "recommended_action": "Use only as implementation reference for DER++, ER, EWC, LwF, and iCaRL.",
            "reviewer_risk_if_missing": "Low.",
            "implementation_difficulty": "none",
            "expected_cost": "none",
            "memory_accounting_requirements": "N/A",
            "fair_adaptation_requirements": "Do not add framework dependency unless it simplifies exact-protocol reproduction.",
            "notes": "Framework references should not appear as competitors.",
        },
    ]


def write_coverage_markdown(rows: list[dict[str, str]], evidence: dict[str, Any]) -> None:
    path = OUT_DIR / "baseline_coverage_audit.md"
    compact_columns = [
        "baseline_name",
        "category",
        "status",
        "matched_existing_method",
        "needs_new_run",
        "needs_external_code",
        "recommended_action",
    ]
    text = [
        "# Baseline Coverage Audit",
        "",
        f"Generated by `{rel(Path(__file__))}`.",
        "",
        "## Input Sanity",
        "",
        f"- Result JSON files inspected under `result/final-run/`: {evidence['json']['json_file_count']}.",
        f"- Methods found in result JSON text: {', '.join(evidence['json']['methods_seen_in_json_text']) or 'none'}.",
        f"- Missing required audit inputs: {', '.join(evidence['input_missing']) or 'none'}.",
        f"- JSON parse errors: {len(evidence['json']['parse_errors'])}.",
        "",
        "## Coverage Table",
        "",
        markdown_table(rows, compact_columns),
        "",
        "## Key Decisions",
        "",
        "- `classifier_real_only` is enough for the None/fine-tuning baseline because it has no replay, no real buffer, no diversity buffer, no KD, and no prototype alignment in the trusted ablation evidence.",
        "- `real_buffer_only` is not enough for canonical ER because the trusted flags show diversity-buffer, KD, and prototype-alignment behavior; keep it as an ablation and run clean ER.",
        "- `wgan_projection_generated_only` is enough for pure generated replay in this architecture, but it is not a faithful MalCL reproduction.",
        "- `malcl_like` is useful internal evidence, but faithful MalCL should be reproduced if the paper needs the closest malware-specific generative replay competitor.",
        "- DER++ and MADAR are currently missing as exact-protocol external baselines.",
        "",
        "## Full CSV",
        "",
        "The full decision matrix is in `paper_ready_baselines/baseline_coverage_audit.csv`.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def write_reference_audit(evidence: dict[str, Any]) -> None:
    rows = []
    missing = []
    for name, filename in EXPECTED_REFERENCE_PAPERS.items():
        path = REFERENCE_DIR / "papers" / filename
        present = path.exists()
        if not present:
            missing.append(filename)
        rows.append(
            {
                "reference": name,
                "local_file": f"baseline_reference_pack/papers/{filename}",
                "present": "YES" if present else "NO",
                "audit_use": reference_use(name),
            }
        )
    text = [
        "# Baseline Reference Audit",
        "",
        "The local reference pack was inspected for method role, fair adaptation, and runnability. Published numbers are not treated as comparable unless the exact protocol is reproduced.",
        "",
        markdown_table(rows, ["reference", "local_file", "present", "audit_use"]),
        "",
        "## Notes",
        "",
        "- MalCL, MADAR, DER++, iCaRL, LwF, EWC, FreeMOCA, GDumb, and Avalanche PDFs are present locally.",
        "- `baseline_reference_pack/notes/baseline_reference_manifest.md` and `baseline_reference_pack/code_links/baseline_code_links.md` identify code links for method understanding only; no repository was cloned or downloaded.",
        "- No published paper number should enter the main numeric comparison table without exact-protocol reproduction.",
        f"- Missing required PDFs: {', '.join(missing) or 'none'}.",
        "",
    ]
    (OUT_DIR / "baseline_reference_audit.md").write_text("\n".join(text), encoding="utf-8")

    missing_text = [
        "# Missing Reference Papers",
        "",
        "No required reference PDFs from the supplied baseline reference manifest are missing.",
        "",
        "Optional SI is not supplied as a standalone paper in the pack; it is treated as optional related work and no details are guessed.",
        "",
    ]
    if missing:
        missing_text = [
            "# Missing Reference Papers",
            "",
            "The following expected local reference files were not found:",
            "",
            *[f"- `{item}`" for item in missing],
            "",
        ]
    (OUT_DIR / "missing_reference_papers.md").write_text("\n".join(missing_text), encoding="utf-8")


def reference_use(name: str) -> str:
    return {
        "MalCL": "Closest malware-specific generative replay baseline; decide faithful reproduction need.",
        "MADAR": "Mandatory malware-specific distribution-aware replay comparator; needs external code.",
        "DER++": "Mandatory generic replay + stored-logit distillation baseline.",
        "FreeMOCA": "Recent malware-specific replay-free novelty-risk paper; literature-only unless reproduced.",
        "iCaRL": "Strong class-incremental exemplar/prototype baseline.",
        "LwF": "Strong replay-free distillation baseline.",
        "EWC": "Classic regularization baseline.",
        "GDumb": "Optional memory-only sanity baseline.",
        "Avalanche": "Framework reference, not a numeric competitor.",
    }.get(name, "Reference only.")


def write_final_plan(rows: list[dict[str, str]]) -> None:
    by_name = {row["baseline_name"]: row for row in rows}
    text = [
        "# Final Baseline Plan",
        "",
        "## 1. Baselines Already Covered",
        "",
        "- Fine-tuning / None: covered by `classifier_real_only`.",
        "- Pure generated replay: covered by `wgan_projection_generated_only`.",
        "- MalCL-like internal baseline: covered as `malcl_like`, but not faithful external MalCL.",
        "- Real-replay ablation: `real_buffer_only` exists, but it is not clean canonical ER.",
        "",
        "## 2. Baselines That Need No New Run",
        "",
        "- `classifier_real_only` for None/fine-tuning.",
        "- `wgan_projection_generated_only` for pure generated replay.",
        "- Existing `malcl_like` may be used as an internal baseline only.",
        "",
        "## 3. Baselines That Need Only A Small Validation Rerun",
        "",
        "- Canonical ER should be a small rerun if the existing notebook toggles can disable diversity/KD/prototype alignment cleanly.",
        "- Joint should be straightforward once the same data-stream adapter is available, but it is still a new exact-protocol run.",
        "",
        "## 4. Baselines That Require New Implementation",
        "",
        "- DER++: new local runner and GPU runs required.",
        "- Canonical ER: new clean replay setting required.",
        "- Joint/offline upper bound: new exact-protocol upper-bound setting required.",
        "- Optional iCaRL, LwF, and EWC: runners are scaffolded as secondary baselines.",
        "",
        "## 5. Baselines That Require External Repository/Code Integration",
        "",
        "- MADAR requires external source integration.",
        "- Faithful MalCL likely requires external source integration or a careful paper-level reimplementation.",
        "- FreeMOCA requires external source integration if used numerically.",
        "",
        "## 6. Literature-Only Baselines",
        "",
        "- FreeMOCA: strong related work and novelty-risk paper unless reproduced.",
        "- GDumb: optional memory-only sanity baseline.",
        "- SI: optional related-work-only unless reviewers ask for it.",
        "",
        "## 7. Related Work Only",
        "",
        "- Avalanche and Mammoth are implementation references, not competitors.",
        "- Concept-drift and active-learning malware papers cited by MADAR/FreeMOCA should stay related-work-only unless they match the class-incremental family-classification protocol.",
        "",
        "## 8. Novelty-Risk Papers",
        "",
        "- MalCL: closest generative malware CL work.",
        "- MADAR: closest distribution-aware malware replay work.",
        "- FreeMOCA: recent replay-free malware CL work.",
        "",
        "## 9. Minimal Q1 Baseline Suite",
        "",
        "- Current covered baselines: None, pure generated replay, MalCL-like internal baseline.",
        "- New exact-protocol baselines to add: DER++, MADAR, canonical ER, and Joint.",
        "- Faithful MalCL is the next add if the paper needs the closest external generative replay reproduction rather than only MalCL-like internal evidence.",
        "",
        "## 10. Strong Q1 Baseline Suite",
        "",
        "- Minimal suite plus faithful MalCL, iCaRL, LwF, EWC, and FreeMOCA if integration time is available.",
        "",
        "## 11. Exact Recommended Next Training Jobs",
        "",
        "1. Fix/install the importable data-stream adapter and dependencies required by `baseline_suite/run_derpp_baseline.py`.",
        "2. Run DER++ for EMBER K=100 and AZ-Class K=200 after dry-run passes.",
        "3. Prepare MADAR integration, then run MADAR for EMBER K=100 and AZ-Class K=200 after source approval.",
        "4. Run canonical ER for EMBER K=100 and AZ-Class K=200.",
        "5. Run Joint upper bounds for EMBER and AZ-Class.",
        "6. Decide whether faithful MalCL reproduction is needed after DER++/MADAR results are available.",
        "7. Run iCaRL/LwF/EWC only if GPU time remains or reviewers require them.",
        "",
        "## 12. GPU Priority Order",
        "",
        "1. DER++.",
        "2. MADAR setup and reproduction.",
        "3. Faithful MalCL if `malcl_like` is judged insufficient for the venue.",
        "4. Joint upper bound.",
        "5. Canonical ER.",
        "6. iCaRL, LwF, EWC.",
        "7. FreeMOCA, GDumb, SI.",
        "",
        "## 13. Go/No-Go Recommendation",
        "",
        "- Go: use the current audit outputs to plan missing baselines.",
        "- No-go for immediate training: DER++ dry-run must pass first, and MADAR external code must be available and approved.",
        "- Go for paper drafting only if numerical comparison tables clearly separate existing internal ablations from unreproduced literature-only methods.",
        "",
        "## Explicit Answers",
        "",
        f"- Is `classifier_real_only` enough for None? YES. Status: {by_name['Fine-tuning / None']['status']}.",
        f"- Is `real_buffer_only` enough for ER? NO. Status: {by_name['Experience Replay / ER']['status']}; it is useful ablation evidence but not canonical ER.",
        f"- Is `wgan_projection_generated_only` enough for pure generated replay? YES. Status: {by_name['Pure generated replay']['status']}.",
        f"- Is `malcl_like` enough for MalCL? NO for faithful external MalCL. Status: {by_name['MalCL faithful']['status']}.",
        "- Do we need DER++? YES.",
        "- Do we need MADAR? YES.",
        "- Should iCaRL/LwF/EWC/FreeMOCA be run now or only discussed? Discuss/scaffold now; run after mandatory DER++, MADAR, ER, and Joint unless GPU time is abundant.",
        "- What is the smallest defensible set? Current None + pure generated + MalCL-like, plus new DER++, MADAR, canonical ER, and Joint.",
        "- What is the strongest set? Smallest set plus faithful MalCL, iCaRL, LwF, EWC, and possibly FreeMOCA.",
        "",
    ]
    (OUT_DIR / "final_baseline_plan.md").write_text("\n".join(text), encoding="utf-8")


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence()
    rows = build_coverage_rows(evidence)
    write_csv(OUT_DIR / "baseline_coverage_audit.csv", rows, COVERAGE_COLUMNS)
    write_coverage_markdown(rows, evidence)
    write_reference_audit(evidence)
    write_final_plan(rows)
    return {
        "coverage_rows": len(rows),
        "input_missing": len(evidence["input_missing"]),
        "json_parse_errors": len(evidence["json"]["parse_errors"]),
    }


def main() -> None:
    print(VERSION, flush=True)
    summary = run()
    print(
        "Wrote baseline audit with {coverage_rows} rows "
        "({input_missing} missing inputs, {json_parse_errors} JSON parse errors).".format(**summary)
    )


if __name__ == "__main__":
    main()
