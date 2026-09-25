#!/usr/bin/env python3
"""Authoritative V15.5 clean-room analysis validation."""
from __future__ import annotations
import hashlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=[
 "analysis_outputs/INCREMENTAL_ACCURACY_PER_SEED_V15_5.csv",
 "analysis_outputs/INCREMENTAL_ACCURACY_SUMMARY_V15_5.csv",
 "analysis_outputs/STAGE_LEVEL_SW_TM_ACCURACY_V15_5.csv",
 "analysis_outputs/INCREMENTAL_ACCURACY_PAIRED_TESTS_V15_5.csv",
 "analysis_outputs/INCREMENTAL_ACCURACY_HOLM_V15_5.csv",
 "analysis_outputs/STABILITY_PLASTICITY_ANALYSIS_V15_5.csv",
 "analysis_outputs/STABILITY_PLASTICITY_SUMMARY_V15_5.csv",
 "analysis_outputs/TASK_SUPPORTS_PER_METHOD_V15_5.csv",
 "results/final_all_methods_seed_level_v15_5.csv",
 "reports/MALCL_MATRIX_PROVENANCE_CHAIN_V15_5.csv",
 "reports/SOURCE_LEVEL_CLASS_ORDER_AUDIT_V15_5.csv",
 "reports/PER_METRIC_PROVENANCE_AUDIT_V15_5.csv",
 "figures/stability_plasticity_v15_5.csv",
]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,cwd,env): subprocess.run(cmd,cwd=cwd,env=env,check=True)
def static(root):
    assert len(list((root/"task_matrices").glob("*.json")))==80
    assert len(list((root/"historical_outputs/adapted_malcl").glob("*/seed_*/per_task_metrics.csv")))==20
    assert (root/"RELEASE_METADATA_V15_5.md").is_file()
    forbidden={"__pycache__",".pytest_cache",".mypy_cache",".ruff_cache",".ipynb_checkpoints"}
    assert not [p for p in root.rglob("*") if p.name in forbidden or p.suffix in {".pyc",".pyo"}]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md",".csv",".json",".py",".txt",".sh",".tex"}:
            private="/"+"home"+"/"+"kian"
            assert private not in p.read_text(errors="ignore"),p
def main():
    static(ROOT)
    expected={x:digest(ROOT/x) for x in OUTPUTS}
    with tempfile.TemporaryDirectory(prefix="v15_5_analysis_reproduction_") as td:
        dest=Path(td)/"release"
        shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc","*.pyo","generated_v15_4","generated_v15_5"))
        env=dict(os.environ); env["MPLCONFIGDIR"]=str(Path(td)/"mpl")
        run([sys.executable,"-m","pytest","-q"],dest,env)
        run([sys.executable,"scripts/validate_protocol.py"],dest,env)
        run([sys.executable,"scripts/verify_malcl_matrix_provenance.py"],dest,env)
        run([sys.executable,"scripts/build_v15_5_release.py","--analysis-only"],dest,env)
        run([sys.executable,"scripts/build_stability_plasticity_figure_v15_5.py"],dest,env)
        for x,h in expected.items(): assert digest(dest/x)==h,(x,digest(dest/x),h)
        for name in ("__pycache__",".pytest_cache",".mypy_cache",".ruff_cache",".ipynb_checkpoints"):
            for p in sorted(dest.rglob(name),reverse=True): shutil.rmtree(p)
        for p in list(dest.rglob("*.pyc"))+list(dest.rglob("*.pyo")): p.unlink()
        static(dest)
    print("V15.5 clean-room analysis reproduction passed deterministically")
if __name__=="__main__": main()
