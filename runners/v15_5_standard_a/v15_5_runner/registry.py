from pathlib import Path
from datetime import datetime,timezone
import csv, json, os

FIELDS=['run_id','dataset','seed','variant','K','status','attempt','start_time','end_time','host','gpu','output_dir','config_sha256','class_order_sha256','checkpoint','result_sha256','validation_status','failure_reason']
VALID={'NOT_STARTED','RUNNING','INTERRUPTED','FAILED','COMPLETED_UNVALIDATED','VALIDATED'}
def read(path):
    with open(path) as f:return list(csv.DictReader(f))
def write(path,rows):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with open(tmp,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def now():return datetime.now(timezone.utc).isoformat()
def update(path,run_id,**changes):
    rows=read(path);found=False
    for r in rows:
        if r['run_id']==run_id:r.update({k:str(v) for k,v in changes.items()});found=True
    if not found:raise KeyError(run_id)
    write(path,rows)
def validate_registry(rows):
    if len(rows)!=40:raise ValueError(f'registry must have 40 rows, got {len(rows)}')
    keys=[(r['dataset'],r['seed'],r['variant'],r['K']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate registry combinations')
    if any(r['status'] not in VALID for r in rows):raise ValueError('invalid registry status')
    from collections import Counter
    counts=Counter(r['variant'] for r in rows)
    if counts!={'current_only':10,'anchor_only':20,'generated_only':10}:raise ValueError(f'wrong variant breakdown: {dict(counts)}')
    for r in rows:
        seed=int(r['seed']);dataset=r['dataset'];variant=r['variant'];k=r['K']
        required=range(42,52) if variant=='anchor_only' else range(47,52)
        if dataset not in ('ember','az_class') or seed not in required:raise ValueError(f'unexpected registry row: {r["run_id"]}')
        expected_k=(('100' if dataset=='ember' else '200') if variant=='anchor_only' else ('100' if variant=='generated_only' else 'NA'))
        if k!=expected_k:raise ValueError(f'wrong K in {r["run_id"]}: {k}')
    return True

def action_for_status(status,retry_failed=False):
    if status=='VALIDATED':return 'skip'
    if status=='COMPLETED_UNVALIDATED':return 'validate'
    if status=='FAILED' and not retry_failed:return 'skip'
    if status in ('NOT_STARTED','INTERRUPTED') or (status=='FAILED' and retry_failed):return 'run'
    if status=='RUNNING':return 'recover'
    raise ValueError(status)
def recover_stale_running(path):
    rows=read(path);changed=False
    for r in rows:
        if r['status']=='RUNNING':r['status']='INTERRUPTED';r['failure_reason']='stale RUNNING recovered on launcher startup';changed=True
    if changed:write(path,rows)
