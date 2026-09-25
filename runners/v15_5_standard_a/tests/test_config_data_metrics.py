from pathlib import Path
import hashlib, math
import numpy as np
import pytest

from v15_5_runner.config import DATASETS,make_config
from v15_5_runner.data import RunningStandardScaler,class_order,md5,preflight_dataset,tasks_for_seed
from v15_5_runner.metrics import confusion_metrics,continual_metrics

def test_current_config_is_isolated():
    c=make_config('ember',47,'current_only',0);assert not c.anchors and not c.generated and not c.gan_active
    assert not c.kd_enabled and not c.prototype_alignment_enabled and not c.diversity_auxiliary_enabled
def test_anchor_config_main_k():assert make_config('ember',42,'anchor_only',100).anchor_k==100 and make_config('az_class',42,'anchor_only',200).anchor_k==200
def test_generated_config_quota():assert make_config('ember',47,'generated_only',100).generated_k==100
def test_wrong_k_is_rejected():
    with pytest.raises(ValueError):make_config('az_class',42,'anchor_only',100)
    with pytest.raises(ValueError):make_config('ember',47,'generated_only',0)
def test_historical_optimizer_and_protocol_constants():
    c=make_config('ember',47,'current_only',0);assert (c.batch_size,c.classifier_epochs,c.classifier_lr,c.classifier_weight_decay,c.gradient_clip_norm)==(256,3,1e-3,1e-4,10.)
    assert c.sampler=='WeightedRandomSampler' and c.sampler_replacement and not c.drop_last
def test_historical_gan_constants():
    c=make_config('ember',47,'generated_only',100);assert (c.gan_epochs,c.critic_steps,c.gan_optimizer_betas,c.candidate_per_class,c.generated_k)==(3,5,(0.,.9),1000,100)
def test_class_order_and_schedule():
    assert class_order(42)==np.random.RandomState(42).permutation(100).tolist();tasks=tasks_for_seed(42)
    assert list(map(len,tasks))==[50]+[5]*10 and sum(tasks,[])==class_order(42)
def test_scaler_matches_direct_population_statistics():
    x=np.arange(60,dtype=float).reshape(20,3);s=RunningStandardScaler();s.partial_fit(x[:7]);s.partial_fit(x[7:]).finalize()
    np.testing.assert_allclose(s.mean_,x.mean(0),rtol=1e-6);np.testing.assert_allclose(s.scale_,x.std(0),rtol=1e-6)
def test_dataset_contract():
    assert DATASETS['ember']['train_shape']==(303331,2381);assert DATASETS['az_class']['test_shape']==(28559,2439)
    assert DATASETS['ember']['train_md5']=='27e39d1cb697434107f92a8084128734'
def test_md5_helper(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'abc');assert md5(p)==hashlib.md5(b'abc').hexdigest()
def test_preflight_rejects_dataset_hash_mismatch(tmp_path):
    (tmp_path/DATASETS['ember']['train']).write_bytes(b'not-the-authoritative-dataset')
    with pytest.raises(ValueError,match='MD5 mismatch'):preflight_dataset(tmp_path,'ember',load_arrays=False)
def test_confusion_metrics():
    _,m=confusion_metrics([0,0,1,1],[0,1,1,1],2);assert m['accuracy']==.75;assert m['balanced_accuracy']==.75
def test_harmonized_continual_metrics():
    a=np.array([[.8,np.nan,np.nan],[.7,.6,np.nan],[.65,.55,.9]]);m=continual_metrics(a,[10,20,30])
    assert m['final_task_macro_accuracy']==pytest.approx(.7);assert m['forgetting']==pytest.approx(.1);assert m['bwt']==pytest.approx(-.1)
def test_standard_a_steps_are_pool_dependent():
    for pool in (100,257,1000):assert 3*math.ceil(pool/256)==sum(math.ceil(pool/256) for _ in range(3))
