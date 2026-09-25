#!/usr/bin/env python3
"""Deterministically render TM-AIA versus conventional forgetting."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"analysis_outputs/STABILITY_PLASTICITY_ANALYSIS_V15_5.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"figures")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(a.input.open())); groups=defaultdict(list)
    for r in rows: groups[(r["dataset"],r["method"])].append(r)
    source=[]
    for (dataset,method),g in sorted(groups.items()):
        tm=np.array([float(x["tm_aia"]) for x in g]); sw=np.array([float(x["sw_aia"]) for x in g])
        fg=np.array([float(x["forgetting"]) for x in g]); acq=np.array([float(x["task_acquisition"]) for x in g])
        source.append({"dataset":dataset,"method":method,"n":len(g),"tm_aia_mean":tm.mean(),"tm_aia_sd":tm.std(ddof=1),
                       "sw_aia_mean":sw.mean(),"sw_aia_sd":sw.std(ddof=1),"forgetting_mean":fg.mean(),
                       "forgetting_sd":fg.std(ddof=1),"task_acquisition_mean":acq.mean(),"task_acquisition_sd":acq.std(ddof=1)})
    fields=list(source[0])
    with (a.output_dir/"stability_plasticity_v15_5.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(source)
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.2),constrained_layout=True)
    markers=["o","s","^","D"]
    for ax,dataset in zip(axes,("EMBER","AZ-Class")):
        for marker,r in zip(markers,[x for x in source if x["dataset"]==dataset]):
            public={"AHR-MalCL-HR (hybrid_random_buffer)":"Anchored HR",
                    "ER (standard)":"ER","DER++ (standard)":"DER++",
                    "protocol-matched adapted MalCL baseline":"adapted MalCL"}[r["method"]]
            ax.errorbar(r["tm_aia_mean"],r["forgetting_mean"],xerr=r["tm_aia_sd"],yerr=r["forgetting_sd"],
                        marker=marker,capsize=3,linestyle="none",label=public)
        ax.set_title(dataset); ax.set_xlabel("TM-AIA (%)"); ax.set_ylabel("Conventional forgetting (%)")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle("Task-balanced incremental accuracy and forgetting (mean ± SD, 10 seeds)")
    fig.savefig(a.output_dir/"stability_plasticity_v15_5.pdf",metadata={"Creator":"V15.5 deterministic analysis"})
    fig.savefig(a.output_dir/"stability_plasticity_v15_5.png",dpi=220)
    plt.close(fig)
    print("wrote V15.5 stability figure and source CSV")
if __name__=="__main__": main()
