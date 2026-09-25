from pathlib import Path
import csv, hashlib, json, math, os
import numpy as np
from .metrics import continual_metrics

REQUIRED=['config.json','environment.json','class_order.json','task_supports.json','run.log','status.json','task_matrix.json','per_task_metrics.csv','full-result.json','classification_report.csv','replay_exposure.csv','resource_metrics.json','classifier_pool_by_task.csv','sampler_schedule_by_task.csv','optimizer_steps_by_task.csv']
def sha256(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()
def write_checksums(run_dir):
    run_dir=Path(run_dir);lines=[]
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name not in ('checksums.sha256','VALIDATED_COMPLETE'):lines.append(f'{sha256(p)}  {p.name}')
    (run_dir/'checksums.sha256').write_text('\n'.join(lines)+'\n')
def validate_run(run_dir,expected=None):
    d=Path(run_dir);missing=[x for x in REQUIRED if not (d/x).is_file()]
    if missing:raise ValueError(f'missing outputs: {missing}')
    cfg=json.loads((d/'config.json').read_text());matrix=np.asarray(json.loads((d/'task_matrix.json').read_text())['task_matrix'],float)
    if expected:
        for key in ('dataset','variant'):
            if str(cfg[key])!=str(expected[key]):raise ValueError(f'config {key} does not match registry')
        if int(cfg['seed'])!=int(expected['seed']):raise ValueError('config seed does not match registry')
        if expected['variant']=='anchor_only' and int(cfg['anchor_k'])!=int(expected['K']):raise ValueError('anchor K does not match registry')
        if expected['variant']=='generated_only' and int(cfg['generated_k'])!=int(expected['K']):raise ValueError('generated quota does not match registry')
        if expected['variant']=='current_only' and expected['K']!='NA':raise ValueError('current-only must be K-independent')
        if expected.get('config_sha256') and cfg.get('config_sha256')!=expected['config_sha256']:raise ValueError('config hash does not match registry')
        order=json.loads((d/'class_order.json').read_text())
        if expected.get('class_order_sha256') and order.get('sha256')!=expected['class_order_sha256']:raise ValueError('class-order hash does not match registry')
    if matrix.shape!=(11,11):raise ValueError('task matrix is not 11x11')
    for i in range(11):
        if not np.all(np.isfinite(matrix[i,:i+1])):raise ValueError(f'nonfinite task matrix row {i+1}')
    supports=json.loads((d/'task_supports.json').read_text())['task_supports']
    if len(supports)!=11 or any(int(x)<=0 for x in supports):raise ValueError('invalid task supports')
    got=continual_metrics(matrix,supports);stored=json.loads((d/'full-result.json').read_text())
    if int(stored.get('tasks_completed',-1))!=11:raise ValueError('full result does not report 11 tasks')
    for k,v in got.items():
        if abs(float(stored[k])-v)>2e-8:raise ValueError(f'metric mismatch {k}')
    with open(d/'replay_exposure.csv') as f:exp=list(csv.DictReader(f))
    if len(exp)!=11:raise ValueError('exposure rows != 11')
    pools=list(csv.DictReader((d/'classifier_pool_by_task.csv').open()));samplers=list(csv.DictReader((d/'sampler_schedule_by_task.csv').open()));optimizers=list(csv.DictReader((d/'optimizer_steps_by_task.csv').open()))
    if not (len(pools)==len(samplers)==len(optimizers)==11):raise ValueError('instrumentation rows != 11')
    for i,r in enumerate(exp):
        a=int(r['anchor_unique']);g=int(r['generated_unique']);gs=int(r['generator_steps']);cs=int(r['critic_steps'])
        pool=int(r['classifier_pool_size']);epochs=int(r['classifier_epochs']);batch=int(samplers[i]['batch_size']);expected_steps=epochs*math.ceil(pool/batch)
        if pool<=0 or int(pools[i]['total'])!=pool or int(pools[i]['current'])!=int(r['current_unique']) or int(pools[i]['anchor'])!=a or int(pools[i]['generated'])!=g:raise ValueError('classifier pool accounting mismatch')
        if int(r['sampler_num_samples'])!=pool or int(samplers[i]['num_samples'])!=pool or samplers[i]['sampler']!='WeightedRandomSampler' or samplers[i]['replacement']!='True':raise ValueError('sampler schedule mismatch')
        if int(r['actual_optimizer_steps'])!=expected_steps or int(optimizers[i]['actual_steps'])!=expected_steps:raise ValueError('optimizer-step schedule mismatch')
        draws=sum(int(r[x]) for x in ('current_realized_draws','anchor_realized_draws','generated_realized_draws'))
        if draws!=epochs*pool or int(r['total_sampled_indices'])!=draws or r['draw_provenance']!='SERIALIZED_ACTUAL':raise ValueError('realized draw accounting mismatch')
        class_draws=json.loads(r['realized_class_draws_json'])
        if sum(int(x) for x in class_draws.values())!=draws:raise ValueError('realized class draw accounting mismatch')
        expected_draws=sum(float(r[x]) for x in ('current_expected_draws','anchor_expected_draws','generated_expected_draws'))
        if abs(expected_draws-draws)>1e-5 or r['expectation_provenance']!='ANALYTIC_EXPECTATION':raise ValueError('analytical expectation accounting mismatch')
        if cfg['variant']=='current_only' and any((a,g,gs,cs)):raise ValueError('current-only isolation violation')
        if cfg['variant']=='anchor_only' and (g or gs or cs or (i>0 and a<=0)):raise ValueError('anchor-only isolation violation')
        if cfg['variant']=='generated_only' and (a or (i>0 and g<=0) or gs<=0 or cs<=0):raise ValueError('generated-only isolation violation')
        if cfg['variant']=='generated_only' and cs!=int(cfg['critic_steps'])*gs:raise ValueError('historical five-to-one critic schedule violation')
    if cfg.get('kd_enabled') or cfg.get('prototype_alignment_enabled') or cfg.get('diversity_auxiliary_enabled'):raise ValueError('forbidden auxiliary enabled')
    if (cfg.get('classifier_weight_decay'),cfg.get('classifier_epochs'),cfg.get('batch_size'),cfg.get('gradient_clip_norm'))!=(1e-4,3,256,10.0):raise ValueError('historical classifier protocol mismatch')
    env=json.loads((d/'environment.json').read_text())
    if 'T4' not in str(env.get('gpu')):raise ValueError('run was not executed on target Tesla T4')
    order=json.loads((d/'class_order.json').read_text());expected_order=np.random.RandomState(int(cfg['seed'])).permutation(100).astype(int).tolist()
    if order.get('class_order')!=expected_order:raise ValueError('class order does not match canonical seeded permutation')
    report=list(csv.DictReader((d/'classification_report.csv').open()))
    if len(report)!=100:raise ValueError('classification report rows != 100')
    write_checksums(d);(d/'VALIDATED_COMPLETE').write_text('VALIDATED\n');return {'metrics':got,'result_sha256':sha256(d/'full-result.json')}
