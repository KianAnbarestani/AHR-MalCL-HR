from collections import Counter
from pathlib import Path
import csv

from v15_5_runner.registry import FIELDS,action_for_status,read,recover_stale_running,update,validate_registry,write

ROOT=Path(__file__).resolve().parents[1]

def rows(name):return list(csv.DictReader((ROOT/name).open()))

def test_registry_has_exactly_40_unique_rows():
    r=rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv');assert validate_registry(r);assert len(r)==40
    assert len({(x['dataset'],x['seed'],x['variant'],x['K']) for x in r})==40

def test_registry_is_directly_equal_to_authoritative_gap():
    r=rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv');g=rows('V15_5_FINAL_NEW_RUN_GAP.csv')
    cols=('run_id','dataset','seed','variant','K')
    assert [{k:x[k] for k in cols} for x in r]==[{k:x[k] for k in cols} for x in g]

def test_registry_schema():assert list(rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv')[0])==FIELDS
def test_variant_counts():assert Counter(x['variant'] for x in rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv'))=={'anchor_only':20,'current_only':10,'generated_only':10}

def test_anchor_seeds_and_k():
    r=[x for x in rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv') if x['variant']=='anchor_only']
    for dataset,k in [('ember','100'),('az_class','200')]:assert sorted(int(x['seed']) for x in r if x['dataset']==dataset)==list(range(42,52)) and all(x['K']==k for x in r if x['dataset']==dataset)

def test_control_seeds_and_k():
    r=rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv')
    for dataset in ('ember','az_class'):
        assert sorted(int(x['seed']) for x in r if x['dataset']==dataset and x['variant']=='current_only')==list(range(47,52))
        assert sorted(int(x['seed']) for x in r if x['dataset']==dataset and x['variant']=='generated_only')==list(range(47,52))
    assert all(x['K']=='NA' for x in r if x['variant']=='current_only')
    assert all(x['K']=='100' for x in r if x['variant']=='generated_only')

def test_reuse_manifest_has_40_read_only_rows():assert len(rows('V15_5_FINAL_REUSE_MANIFEST.csv'))==40
def test_status_actions():
    assert action_for_status('VALIDATED')=='skip';assert action_for_status('COMPLETED_UNVALIDATED')=='validate'
    assert action_for_status('INTERRUPTED')=='run';assert action_for_status('NOT_STARTED')=='run'
def test_failed_requires_explicit_retry():assert action_for_status('FAILED')=='skip' and action_for_status('FAILED',True)=='run'
def test_template_starts_clean():assert all(x['status']=='NOT_STARTED' and x['attempt']=='0' for x in rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv'))
def test_atomic_status_update_and_stale_recovery(tmp_path):
    p=tmp_path/'registry.csv';r=rows('V15_5_STANDARD_A_40_RUN_REGISTRY.csv');write(p,r);run_id=r[0]['run_id'];update(p,run_id,status='RUNNING',attempt=1);recover_stale_running(p);got=read(p)[0]
    assert got['status']=='INTERRUPTED' and 'stale RUNNING' in got['failure_reason'] and got['attempt']=='1'
