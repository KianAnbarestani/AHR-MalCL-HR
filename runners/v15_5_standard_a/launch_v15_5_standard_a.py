#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, os, shutil, socket, sys, time, traceback

ROOT=Path(__file__).resolve().parent
TEMPLATE=ROOT/'V15_5_STANDARD_A_40_RUN_REGISTRY.csv'

def sha256(p):h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def registry_path(output):return Path(output)/'V15_5_STANDARD_A_40_RUN_REGISTRY.csv'
def ensure_registry(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);p=registry_path(out)
    if not p.exists():shutil.copy2(TEMPLATE,p)
    return p
def load_torch(require_cuda=False):
    import torch
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required for smoke/full execution')
    if require_cuda and 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError(f'Tesla T4 is required; detected {torch.cuda.get_device_name(0)}')
    return torch
def preflight(args):
    torch=load_torch(False);from v15_5_runner.data import preflight_dataset;from v15_5_runner.registry import read,validate_registry
    out=Path(args.output_root);cache=out/'_dataset_cache';datasets={d:preflight_dataset(args.data_root,d,cache,True) for d in ('ember','az_class')};rows=read(TEMPLATE);validate_registry(rows)
    env={'python':sys.version.split()[0],'pytorch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    result={'environment':env,'datasets':datasets,'registry_rows':len(rows),'planned_runs':40,'duplicates':len(rows)-len({(r['dataset'],r['seed'],r['variant'],r['K']) for r in rows}),
      'variants':dict(Counter(r['variant'] for r in rows)),'package_lock_sha256':sha256(ROOT/'requirements.txt')}
    print(json.dumps(result,indent=2));return result
def status(args):
    from v15_5_runner.registry import read
    p=ensure_registry(args.output_root);rows=read(p);c=Counter(r['status'] for r in rows);print('[V15.5 Standard-A] registry_rows:',len(rows));print('[V15.5 Standard-A] status:',json.dumps(dict(c),sort_keys=True));print('[V15.5 Standard-A] validated:',c['VALIDATED'],'/40');
    for r in rows:
        if r['status'] in ('RUNNING','INTERRUPTED','FAILED'):print(r['status'],r['run_id'],r['failure_reason'])
def smoke(args,do_preflight=True):
    if do_preflight:preflight(args)
    torch=load_torch(True);from v15_5_runner.config import make_config;from v15_5_runner.training import run_training,model_sha
    code=model_sha(ROOT);base=Path(args.output_root)/'smoke';root=base/code[:12];root.mkdir(parents=True,exist_ok=True)
    for variant in ('current_only','anchor_only','generated_only'):
        cfg=make_config('ember',42,variant,100 if variant in ('anchor_only','generated_only') else 0);d=root/variant
        print('[SMOKE]',variant,'phase 1/interrupt boundary');run_training(torch,cfg,args.data_root,d,ROOT,device='cuda',smoke=True,max_tasks=1,cache_root=Path(args.output_root)/'_dataset_cache')
        print('[SMOKE]',variant,'phase 2/resume');run_training(torch,cfg,args.data_root,d,ROOT,device='cuda',smoke=True,max_tasks=2,cache_root=Path(args.output_root)/'_dataset_cache')
        exp=list(csv.DictReader(open(d/'replay_exposure.csv')))
        if variant=='current_only' and any(int(r[x]) for r in exp for x in ('anchor_unique','generated_unique','generator_steps','critic_steps')):raise RuntimeError('current smoke isolation failure')
        if variant=='anchor_only' and (int(exp[1]['anchor_unique'])<=0 or int(exp[1]['generated_unique']) or int(exp[1]['generator_steps']) or int(exp[1]['critic_steps'])):raise RuntimeError('anchor smoke isolation failure')
        if variant=='generated_only' and (int(exp[1]['anchor_unique']) or int(exp[1]['generated_unique'])<=0 or int(exp[1]['generator_steps'])<=0 or int(exp[1]['critic_steps'])<=0):raise RuntimeError('generated smoke isolation failure')
        (d/'SMOKE_VALIDATED').write_text('PASS\n');print('[SMOKE VALIDATED]',variant)
    marker={'status':'PASS','code_sha256':code,'variants':['current_only','anchor_only','generated_only']}
    (base/'SMOKE_SUITE_VALIDATED.json').write_text(json.dumps(marker,indent=2)+'\n')
def validate_one(torch,row,out,reg):
    from v15_5_runner.validation import validate_run;from v15_5_runner.registry import update,now
    d=out/row['output_dir']
    try:
        result=validate_run(d,row);update(reg,row['run_id'],status='VALIDATED',validation_status='VALIDATED',end_time=now(),result_sha256=result['result_sha256'],failure_reason='');return True
    except Exception as e:update(reg,row['run_id'],status='FAILED',validation_status='FAILED',end_time=now(),failure_reason=f'validation: {type(e).__name__}: {e}');return False
def validate_all(args):
    preflight(args);torch=load_torch(False);from v15_5_runner.registry import read,update
    out=Path(args.output_root);reg=ensure_registry(out);rows=read(reg);updated=failed=0
    for r in rows:
        if r['status'] in ('COMPLETED_UNVALIDATED','VALIDATED'):
            if validate_one(torch,r,out,reg):updated+=1
            else:failed+=1
    validated=[r for r in read(reg) if r['status']=='VALIDATED']
    signatures={}
    for r in validated:
        env=json.loads((out/r['output_dir']/'environment.json').read_text());sig=tuple(str(env.get(k)) for k in ('python','pytorch','cuda','cudnn','driver','gpu'))
        signatures.setdefault(sig,[]).append(r['run_id'])
    if len(signatures)>1:
        failed+=sum(len(v) for v in signatures.values())
        reason='environment mismatch across validated Standard-A runs'
        for ids in signatures.values():
            for run_id in ids:update(reg,run_id,status='FAILED',validation_status='FAILED',failure_reason=reason)
    print('[VALIDATE] validated:',updated,'failed:',failed);return 1 if failed else 0
def run_all(args):
    preflight(args);torch=load_torch(True);from v15_5_runner.registry import read,update,now,recover_stale_running,action_for_status;from v15_5_runner.config import make_config;from v15_5_runner.data import class_order;from v15_5_runner.training import run_training,TimeBudgetReached,model_sha
    out=Path(args.output_root);reg=ensure_registry(out);recover_stale_running(reg)
    smoke_marker=out/'smoke/SMOKE_SUITE_VALIDATED.json';smoke_ok=False
    if smoke_marker.exists():
        try:smoke_ok=json.loads(smoke_marker.read_text()).get('code_sha256')==model_sha(ROOT)
        except Exception:smoke_ok=False
    if not smoke_ok:smoke(args,do_preflight=False)
    deadline=time.time()+args.max_hours*3600 if args.max_hours else None
    for row in read(reg):
        action=action_for_status(row['status'],args.retry_failed)
        if action=='skip':continue
        if action=='validate':validate_one(torch,row,out,reg);continue
        if deadline and time.time()>=deadline:print('[V15.5] max-hours reached before next run');break
        cfg=make_config(row['dataset'],int(row['seed']),row['variant'],0 if row['K']=='NA' else int(row['K']));run_dir=out/row['output_dir'];run_dir.mkdir(parents=True,exist_ok=True)
        order=class_order(int(row['seed']));order_hash=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
        gpu=torch.cuda.get_device_name(0);attempt=int(row['attempt'] or 0)+1;update(reg,row['run_id'],status='RUNNING',attempt=attempt,start_time=now(),end_time='',host=socket.gethostname(),gpu=gpu,config_sha256=cfg.sha256(),class_order_sha256=order_hash,checkpoint=str(run_dir/'checkpoint/latest.pt'),failure_reason='')
        log=open(run_dir/'run.log','a',buffering=1)
        def emit(x):print(x);print(x,file=log)
        try:
            emit(f"RUN {row['run_id']} attempt={attempt}");run_training(torch,cfg,args.data_root,run_dir,ROOT,device='cuda',deadline=deadline,progress=emit,cache_root=out/'_dataset_cache');update(reg,row['run_id'],status='COMPLETED_UNVALIDATED',end_time=now());log.close();current=next(x for x in read(reg) if x['run_id']==row['run_id']);validate_one(torch,current,out,reg)
        except TimeBudgetReached as e:update(reg,row['run_id'],status='INTERRUPTED',end_time=now(),failure_reason=str(e));log.close();break
        except KeyboardInterrupt:update(reg,row['run_id'],status='INTERRUPTED',end_time=now(),failure_reason='KeyboardInterrupt/task checkpoint retained');log.close();raise
        except Exception as e:
            traceback.print_exc(file=log);update(reg,row['run_id'],status='FAILED',end_time=now(),failure_reason=f'{type(e).__name__}: {e}');log.close()
    status(args)
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root');p.add_argument('--output-root',required=True);g=p.add_mutually_exclusive_group(required=True)
    for x in ('preflight','smoke-tests','run-all','status','validate'):g.add_argument('--'+x,action='store_true')
    p.add_argument('--max-hours',type=float,default=0);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if (a.preflight or a.smoke_tests or a.run_all or a.validate) and not a.data_root:p.error('--data-root is required for this action')
    if a.preflight:preflight(a)
    elif a.smoke_tests:smoke(a)
    elif a.run_all:run_all(a)
    elif a.status:status(a)
    elif a.validate:raise SystemExit(validate_all(a))
if __name__=='__main__':main()
