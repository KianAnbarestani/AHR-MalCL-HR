"""Reusable pytest-native V15.5 validation checks."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path: str):
    return list(csv.DictReader((ROOT / path).open()))


def check_runner_import():
    p=ROOT/"runners/baseline_runner_common.py"
    spec=importlib.util.spec_from_file_location("baseline_runner_common",p)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert callable(module.add_common_args) and callable(module.run)


def check_runner_cli():
    for runner,config in (("run_er_baseline.py","er_az_k200.json"),("run_derpp_baseline.py","derpp_az_k200.json")):
        cmd=[sys.executable,str(ROOT/"runners"/runner)]
        assert subprocess.run(cmd+["--help"],capture_output=True).returncode==0
        assert subprocess.run(cmd+["--config",str(ROOT/"baseline_suite/configs"/config),"--validate-config-only"],capture_output=True).returncode==0


def derived(matrix,support):
    sw=[]; tm=[]
    for i,row in enumerate(matrix):
        x=np.asarray(row[:i+1],float); w=np.asarray(support[:i+1],float)
        sw.append(float(np.average(x,weights=w)*100)); tm.append(float(x.mean()*100))
    return np.mean(sw),np.mean(tm),sw[-1],tm[-1]


def check_formula():
    m=[[.8,None,None],[.7,.6,None],[.9,.5,.4]]
    sw,tm,fs,ft=derived(m,[100,10,1])
    assert np.isclose(sw,(80+(70*100+60*10)/110+(90*100+50*10+40)/111)/3)
    assert np.isclose(tm,(80+65+60)/3)
    assert np.isclose(fs,(90*100+50*10+40)/111)
    assert np.isclose(ft,60)
    assert not np.isclose(sw,tm)


def check_all_incremental():
    expected={(r["dataset"],r["method"],int(r["seed"])):r for r in rows("analysis_outputs/INCREMENTAL_ACCURACY_PER_SEED_V15_5.csv")}
    support={}
    for r in rows("analysis_outputs/TASK_TEST_SUPPORTS_V15_4.csv"):
        support.setdefault((r["dataset"],int(r["seed"])),[]).append(int(r["test_support"]))
    assert len(expected)==80
    for p in (ROOT/"task_matrices").glob("*.json"):
        d=json.loads(p.read_text()); key=(d["dataset"],d["method"],int(d["seed"]))
        got=derived(d["task_level_accuracies"],support[(d["dataset"],int(d["seed"]))])
        r=expected[key]
        assert np.allclose(got,[float(r[k]) for k in ("sw_aia","tm_aia","final_sample_weighted_seen_accuracy","final_task_macro_accuracy")])


def check_forgetting():
    expected={(r["dataset"],r["method"],int(r["seed"])):float(r["forgetting"]) for r in rows("analysis_outputs/HARMONIZED_TASK_METRICS_PER_SEED_V15_4.csv")}
    for p in (ROOT/"task_matrices").glob("*.json"):
        d=json.loads(p.read_text()); m=d["task_level_accuracies"]; vals=[]
        for u in range(10):
            vals.append(max(float(m[t][u]) for t in range(u,10))-float(m[10][u]))
        assert np.isclose(np.mean(vals)*100,expected[(d["dataset"],d["method"],int(d["seed"]))],atol=1e-8)


def check_matrices():
    paths=list((ROOT/"task_matrices").glob("*.json")); assert len(paths)==80
    for p in paths:
        d=json.loads(p.read_text()); m=d["task_level_accuracies"]
        assert len(m)==11 and all(len(r)==11 for r in m)
        assert all(r[j] is None for i,r in enumerate(m) for j in range(i+1,11))


def check_supports():
    r=rows("analysis_outputs/TASK_SUPPORTS_PER_METHOD_V15_5.csv")
    assert len(r)==80 and all(x["schedule_valid"]=="true" and x["matches_v15_4_supports"]=="true" for x in r)
    for x in r:
        s=json.loads(x["task_supports"]); assert len(s)==11 and sum(s)==int(x["full_test_support"])


def check_orders():
    r=rows("reports/SOURCE_LEVEL_CLASS_ORDER_AUDIT_V15_5.csv")
    assert len(r)==20 and all(x["all_equal"]=="true" and x["mismatch_position"]=="" for x in r)


def check_malcl_provenance():
    r=rows("reports/MALCL_MATRIX_PROVENANCE_CHAIN_V15_5.csv"); assert len(r)==20
    for x in r:
        raw=ROOT/x["raw_output"]; norm=ROOT/x["normalized_matrix"]
        assert raw.is_file() and norm.is_file()
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==x["raw_output_sha256"]
        assert hashlib.sha256(norm.read_bytes()).hexdigest()==x["normalized_matrix_sha256"]
        source=[ast.literal_eval(y["taskwise_accuracies"]) for y in csv.DictReader(raw.open())]
        matrix=json.loads(norm.read_text())["task_level_accuracies"]
        assert all(np.allclose(source[i],matrix[i][:i+1]) for i in range(11))
        assert x["chain_complete"]=="false" and x["historical_runner"]=="NOT_RECOVERED"


def check_per_metric_provenance():
    seed=rows("results/final_all_methods_seed_level_v15_5.csv"); assert len(seed)==80
    required={"raw_result_source","raw_result_sha256","task_matrix_source","task_matrix_sha256",
              "harmonized_metric_source","harmonized_metric_definition_version",
              "classification_report_source","resource_measurement_source","diagnostic_source"}
    assert required<=set(seed[0])
    for r in seed:
        assert (ROOT/r["raw_result_source"]).is_file()
        assert (ROOT/r["task_matrix_source"]).is_file()
        assert hashlib.sha256((ROOT/r["raw_result_source"]).read_bytes()).hexdigest()==r["raw_result_sha256"]


def check_stability():
    r=rows("analysis_outputs/STABILITY_PLASTICITY_ANALYSIS_V15_5.csv")
    assert len(r)==80 and all(x["definition_version"]=="V15.5" for x in r)
    fig=rows("figures/stability_plasticity_v15_5.csv"); assert len(fig)==8
    grouped={(x["dataset"],x["method"],x["metric"]):x for x in rows("analysis_outputs/STABILITY_PLASTICITY_SUMMARY_V15_5.csv")}
    for x in fig:
        assert np.isclose(float(x["tm_aia_mean"]),float(grouped[(x["dataset"],x["method"],"tm_aia")]["mean"]))
        assert np.isclose(float(x["forgetting_mean"]),float(grouped[(x["dataset"],x["method"],"forgetting")]["mean"]))


def check_manifests():
    for p in (ROOT/"manifests").glob("analysis_manifest_v15_5.json"):
        d=json.loads(p.read_text())
        def walk(x):
            if isinstance(x,dict):
                for v in x.values(): yield from walk(v)
            elif isinstance(x,list):
                for v in x: yield from walk(v)
            elif isinstance(x,str) and Path(x).suffix.lower() in {".csv",".json",".pdf",".png",".tex",".md"}: yield x
        for x in walk(d): assert (p.parent/x).exists() or (ROOT/x).exists(),(p,x)
    submission=ROOT.parent/"AHR_MalCL_HR_v15_5_submission_package"
    if submission.exists():
        p=submission/"submission_manifest_v15_5.json"; d=json.loads(p.read_text())
        def walk_submission(x):
            if isinstance(x,dict):
                for v in x.values(): yield from walk_submission(v)
            elif isinstance(x,list):
                for v in x: yield from walk_submission(v)
            elif isinstance(x,str) and Path(x).suffix.lower() in {".csv",".json",".pdf",".zip"}: yield x
        for x in walk_submission(d): assert (submission/x).exists(),(p,x)


def check_version():
    assert (ROOT/"RELEASE_METADATA_V15_5.md").is_file()
    assert "V15.5" in (ROOT/"README.md").read_text()


def check_seed_corpus():
    r=rows("results/final_all_methods_seed_level_v15_5.csv")
    assert len(r)==80
    assert {int(x["seed"]) for x in r}==set(range(42,52))
    assert len({(x["method"],x["dataset"],x["seed"]) for x in r})==80
