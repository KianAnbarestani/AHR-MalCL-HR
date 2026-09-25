import csv,json,math
from pathlib import Path
import numpy as np
import pytest

from v15_5_runner.config import make_config
from v15_5_runner.metrics import continual_metrics
from v15_5_runner.validation import REQUIRED,validate_run

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def fabricate_current_run(d):
    d.mkdir();cfg=make_config('ember',47,'current_only',0);(d/'config.json').write_text(json.dumps({**cfg.payload(),'config_sha256':cfg.sha256()}))
    (d/'environment.json').write_text(json.dumps({'gpu':'Tesla T4','python':'3.12','pytorch':'2.6','cuda':'12.4','cudnn':'x','driver':'x'}))
    order=np.random.RandomState(47).permutation(100).astype(int).tolist();(d/'class_order.json').write_text(json.dumps({'class_order':order,'sha256':'x'}));(d/'task_supports.json').write_text(json.dumps({'task_supports':[1]*11}))
    matrix=np.full((11,11),np.nan)
    for i in range(11):matrix[i,:i+1]=.8
    (d/'task_matrix.json').write_text(json.dumps({'task_matrix':matrix.tolist()}));metrics=continual_metrics(matrix,[1]*11)
    (d/'full-result.json').write_text(json.dumps({**metrics,'tasks_completed':11}));(d/'status.json').write_text(json.dumps({'status':'COMPLETED_UNVALIDATED'}));(d/'run.log').write_text('synthetic\n');(d/'resource_metrics.json').write_text('{}')
    write_csv(d/'per_task_metrics.csv',[{'task_id':i+1,'accuracy':.8} for i in range(11)]);write_csv(d/'classification_report.csv',[{'class':i,'f1':.8} for i in range(100)])
    exp=[];pools=[];samplers=[];opts=[]
    for i in range(11):
        pool=256+i;steps=3*math.ceil(pool/256);exp.append({'task_id':i+1,'current_unique':pool,'anchor_unique':0,'generated_unique':0,'classifier_pool_size':pool,'sampler_num_samples':pool,'batches_per_epoch':math.ceil(pool/256),'classifier_epochs':3,'actual_optimizer_steps':steps,'total_sampled_indices':3*pool,'current_realized_draws':3*pool,'anchor_realized_draws':0,'generated_realized_draws':0,'realized_class_draws_json':json.dumps({'0':3*pool}),'draw_provenance':'SERIALIZED_ACTUAL','current_expected_draws':3*pool,'anchor_expected_draws':0,'generated_expected_draws':0,'expected_class_draws_json':json.dumps({'0':3*pool}),'expectation_provenance':'ANALYTIC_EXPECTATION','generator_steps':0,'critic_steps':0})
        pools.append({'task_id':i+1,'current':pool,'anchor':0,'generated':0,'total':pool});samplers.append({'task_id':i+1,'sampler':'WeightedRandomSampler','replacement':True,'num_samples':pool,'drop_last':False,'batch_size':256,'expected_batches_per_epoch':math.ceil(pool/256)});opts.append({'task_id':i+1,'optimizer':'AdamW','lifecycle':'recreated_this_task','epochs':3,'actual_steps':steps,'gradient_clip_norm':10})
    write_csv(d/'replay_exposure.csv',exp);write_csv(d/'classifier_pool_by_task.csv',pools);write_csv(d/'sampler_schedule_by_task.csv',samplers);write_csv(d/'optimizer_steps_by_task.csv',opts)

def test_complete_output_schema_and_metric_recomputation(tmp_path):
    d=tmp_path/'run';fabricate_current_run(d);result=validate_run(d)
    assert (d/'VALIDATED_COMPLETE').is_file() and (d/'checksums.sha256').is_file() and result['metrics']['tm_aia']==pytest.approx(.8)

def test_component_violation_fails_validation(tmp_path):
    d=tmp_path/'run';fabricate_current_run(d);rows=list(csv.DictReader((d/'replay_exposure.csv').open()));rows[1]['anchor_unique']='1';write_csv(d/'replay_exposure.csv',rows)
    with pytest.raises(ValueError,match='mismatch|isolation'):validate_run(d)
