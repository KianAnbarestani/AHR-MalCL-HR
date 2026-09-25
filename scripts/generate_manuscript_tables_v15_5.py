#!/usr/bin/env python3
"""Generate all changed V15.5 manuscript tables from released CSVs."""
from __future__ import annotations
import argparse,csv
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABEL={"AHR-MalCL-HR (hybrid_random_buffer)":"Anchored HR","ER (standard)":"ER","DER++ (standard)":"DER++",
       "protocol-matched adapted MalCL baseline":"adapted MalCL"}
ORDER=list(LABEL)
def read(p): return list(csv.DictReader(p.open()))
def fmt(mean,sd,bold=False):
    x=f"{float(mean):.2f} $\\pm$ {float(sd):.2f}"
    return "\\textbf{"+x+"}" if bold else x
def main():
    a=argparse.ArgumentParser(); a.add_argument("--source-dir",type=Path,required=True); x=a.parse_args()
    inc=read(ROOT/"analysis_outputs/INCREMENTAL_ACCURACY_SUMMARY_V15_5.csv")
    old=read(ROOT/"results/final_all_methods_dataset_summary.csv")
    I={(r["dataset"],r["method"],r["metric"]):r for r in inc}
    O={(r["dataset"],r["method"],r["metric"]):r for r in old}
    dynamics=[]
    for ds in ("EMBER","AZ-Class"):
        best={m:max(float(I[(ds,k,m)]["mean"]) for k in ORDER) for m in ("final_task_macro_accuracy","tm_aia","sw_aia")}
        low=min(float(O[(ds,k,"forgetting")]["mean"]) for k in ORDER)
        for j,k in enumerate(ORDER):
            vals=[]
            for m in ("final_task_macro_accuracy","tm_aia","sw_aia"):
                r=I[(ds,k,m)]; vals.append(fmt(r["mean"],r["sd"],abs(float(r["mean"])-best[m])<1e-10))
            f=O[(ds,k,"forgetting")]; vals.append(fmt(f["mean"],f["sample_sd"],abs(float(f["mean"])-low)<1e-10))
            dynamics.append(f"{ds if j==0 else ''} & {LABEL[k]} & "+" & ".join(vals)+r" \\")
        dynamics.append(r"\midrule")
    dynamics=dynamics[:-1]
    classification=[]
    for ds in ("EMBER","AZ-Class"):
        metrics=("macro_f1","weighted_f1","balanced_accuracy")
        best={m:max(float(O[(ds,k,m)]["mean"]) for k in ORDER) for m in metrics}
        for j,k in enumerate(ORDER):
            vals=[fmt(O[(ds,k,m)]["mean"],O[(ds,k,m)]["sample_sd"],abs(float(O[(ds,k,m)]["mean"])-best[m])<1e-10) for m in metrics]
            classification.append(f"{ds if j==0 else ''} & {LABEL[k]} & "+" & ".join(vals)+r" \\")
        classification.append(r"\midrule")
    classification=classification[:-1]
    tex=r"""% Generated from V15.5 CSVs by scripts/generate_manuscript_tables_v15_5.py
\begin{table*}[t]
\centering
\caption{Continual-learning dynamics across ten matched seeds. Final TM Accuracy and TM-AIA weight task blocks equally; SW-AIA weights test examples equally within each stage. Forgetting is the common max-pre-final-minus-final definition. Values are mean $\pm$ sample SD in percentage points.}
\label{tab:main_results}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Dataset & Method & Final TM Accuracy & TM-AIA & SW-AIA & Forgetting \\
\midrule
"""+ "\n".join(dynamics)+r"""
\bottomrule
\end{tabular}}
\end{table*}

\begin{table*}[t]
\centering
\caption{Final classification summaries across the same ten matched seeds. These metrics come from original classification outputs, not task matrices. Values are mean $\pm$ sample SD in percentage points.}
\label{tab:classification_results}
\scriptsize
\begin{tabular}{llrrr}
\toprule
Dataset & Method & Macro-F1 & Weighted-F1 & Balanced Accuracy \\
\midrule
"""+ "\n".join(classification)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (x.source_dir/"tables/table_1_final_main_results.tex").write_text(tex)
    paired=read(ROOT/"analysis_outputs/INCREMENTAL_ACCURACY_PAIRED_TESTS_V15_5.csv")
    lines=[]
    for r in paired:
        if r["metric"] not in {"sw_aia","tm_aia"}: continue
        lines.append(f"{r['dataset']} & {r['metric'].replace('_','-').upper()} & {LABEL[r['comparator']]} & "
                     f"{float(r['paired_mean_difference']):.2f} & {float(r['wilcoxon_p']):.4f} & "
                     f"{float(r['holm_adjusted_wilcoxon_p']):.4f} \\\\")
    ptex=r"""% Generated from INCREMENTAL_ACCURACY_PAIRED_TESTS_V15_5.csv
\begin{table*}[t]\centering\caption{Paired Anchored-HR contrasts for the two incremental-accuracy weightings. Differences are Anchored HR minus comparator; Holm correction is within each dataset--metric family of three planned contrasts.}\label{tab:incremental_paired}
\scriptsize\begin{tabular}{lllrrr}\toprule
Dataset & Metric & Comparator & Mean difference & Wilcoxon $p$ & Holm $p$\\\midrule
"""+ "\n".join(lines)+r"""
\bottomrule\end{tabular}\end{table*}
"""
    (x.source_dir/"tables/table_2_incremental_paired_tests.tex").write_text(ptex)
    per=read(ROOT/"analysis_outputs/INCREMENTAL_ACCURACY_PER_SEED_V15_5.csv")
    sl=[r"""% Generated from INCREMENTAL_ACCURACY_PER_SEED_V15_5.csv
\begin{longtable}{lllrr}\caption{Per-seed incremental-accuracy values (\%).}\label{tab:s4_incremental}\\
\toprule Dataset & Method & Seed & SW-AIA & TM-AIA\\\midrule\endfirsthead
\toprule Dataset & Method & Seed & SW-AIA & TM-AIA\\\midrule\endhead"""]
    for r in per: sl.append(f"{r['dataset']} & {LABEL[r['method']]} & {r['seed']} & {float(r['sw_aia']):.2f} & {float(r['tm_aia']):.2f} \\\\")
    sl.append(r"\bottomrule\end{longtable}")
    (x.source_dir/"supplementary/table_s4_incremental_accuracy.tex").write_text("\n".join(sl))
    print("generated V15.5 dynamics, classification, paired-test, and supplementary tables")
if __name__=="__main__": main()
