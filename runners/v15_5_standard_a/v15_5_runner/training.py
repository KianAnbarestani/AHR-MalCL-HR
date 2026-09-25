from pathlib import Path
from datetime import datetime,timezone
import csv, hashlib, json, math, os, platform, random, socket, time
import numpy as np

from .config import DATASETS
from .data import (IndexDataset,ReplayDataset,TaggedConcat,RunningStandardScaler,build_balanced_sampler,class_order,
                   fit_scaler,load_npz_memmap,scaler_sha256,tasks_for_seed)
from .metrics import confusion_metrics,continual_metrics
from .checkpointing import load as load_checkpoint,save as save_checkpoint,rng_state,restore_rng

class TimeBudgetReached(RuntimeError):pass

def atomic_json(path,obj):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=True,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)));os.replace(tmp,p)
def csv_write(path,rows,fields=None):
    rows=list(rows);fields=fields or list(rows[0]);p=Path(path);tmp=p.with_suffix('.tmp')
    with open(tmp,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def seed_all(torch,seed):
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(int(seed))
    if torch.cuda.is_available():torch.cuda.set_device(0);torch.cuda.manual_seed(int(seed))
    # Preserved historical single-GPU cuDNN behavior.
    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True
def environment(torch,package_root):
    lock=Path(package_root)/'requirements.txt'
    return {'python':platform.python_version(),'pytorch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
      'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
      'driver':os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null').read().strip(),'host':socket.gethostname(),'platform':platform.platform(),
      'package_lock':'requirements.txt','package_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest()}
def model_sha(package_root):
    h=hashlib.sha256()
    root=Path(package_root)
    for p in sorted((root/'v15_5_runner').glob('*.py'))+[root/'launch_v15_5_standard_a.py']:
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def classifier_train(torch,clf,loader,cfg,device,source_counts,class_counts):
    from .models import make_classifier_optimizer
    opt=make_classifier_optimizer(torch,clf,cfg);steps=0;clf.train()
    for _ in range(cfg.classifier_epochs):
        for xb,yb,sources in loader:
            for label in yb.tolist():class_counts[int(label)]=class_counts.get(int(label),0)+1
            xb=xb.to(device).float();yb=yb.to(device).long();logits=clf(xb);loss=torch.nn.functional.cross_entropy(logits,yb)
            if not torch.isfinite(xb).all() or not torch.isfinite(logits).all() or not torch.isfinite(loss):raise ValueError('non-finite classifier batch/logits/loss')
            if yb.numel()==0 or int(yb.min())<0 or int(yb.max())>=clf.num_classes:raise ValueError('classifier label outside expanding-head range')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(clf.parameters(),cfg.gradient_clip_norm);opt.step();steps+=1
            for src in sources:source_counts[str(src)]+=1
    return steps,opt.state_dict()

def _predict(torch,clf,X,y,indices,scaler,c2id,batch_size,device):
    from torch.utils.data import DataLoader
    ds=IndexDataset(X,y,indices,scaler,c2id);dl=DataLoader(ds,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0);ys=[];ps=[];clf.eval()
    with torch.no_grad():
        for xb,yb,_ in dl:ys.extend(yb.numpy().tolist());ps.extend(clf(xb.to(device)).argmax(1).cpu().numpy().tolist())
    return ys,ps

def _checkpoint_payload(torch,cfg,next_task,clf,G,D,scaler,anchors,replay_X,replay_y,matrix,task_rows,exposure_rows,pool_rows,sampler_rows,opt_rows,optimizer_states,elapsed):
    return {'checkpoint_version':'V15.5-Standard-A-task-boundary','config_hash':cfg.sha256(),'next_task':next_task,'classifier_state':clf.state_dict(),
      'generator_state':None if G is None else G.state_dict(),'critic_state':None if D is None else D.state_dict(),'scaler_state':scaler.state(),'anchors':anchors,
      'replay_X_raw':replay_X,'replay_y_orig':replay_y,'task_matrix':matrix,'task_rows':task_rows,'exposure_rows':exposure_rows,'pool_rows':pool_rows,
      'sampler_rows':sampler_rows,'optimizer_rows':opt_rows,'optimizer_states':optimizer_states,'elapsed_seconds':elapsed,'rng_state':rng_state(torch)}

def assert_component_isolation(cfg,task_index,anchor_n,generated_n,source_counts,gan_stats):
    gs=int(gan_stats['generator_steps']);cs=int(gan_stats['critic_steps'])
    if cfg.variant=='current_only' and any((anchor_n,generated_n,source_counts['anchor'],source_counts['generated'],gs,cs)):
        raise RuntimeError('Current-only component isolation violation')
    if cfg.variant=='anchor_only' and (generated_n or source_counts['generated'] or gs or cs or (task_index>0 and (anchor_n<=0 or source_counts['anchor']<=0))):
        raise RuntimeError('Anchor-only component isolation violation')
    if cfg.variant=='generated_only' and (anchor_n or source_counts['anchor'] or (task_index>0 and (generated_n<=0 or source_counts['generated']<=0)) or gs<=0 or cs<=0):
        raise RuntimeError('Generated-only component isolation violation')
    if cfg.kd_enabled or cfg.prototype_alignment_enabled or cfg.diversity_auxiliary_enabled:
        raise RuntimeError('Forbidden auxiliary component enabled')

def run_training(torch,cfg,data_root,run_dir,package_root,device='cuda',deadline=None,smoke=False,max_tasks=None,progress=print,cache_root=None):
    from torch.utils.data import DataLoader
    from .models import V155Classifier,V155ConditionalGenerator,V155ProjectionCritic
    from .replay import train_wgan_gp,class_mean_mid,select_generated,update_random_anchors
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True);ckpt=run_dir/'checkpoint/latest.pt';start=time.time();seed_all(torch,cfg.seed)
    cache=Path(cache_root) if cache_root is not None else Path(run_dir).parents[3]/'_dataset_cache';spec=DATASETS[cfg.dataset]
    Xtr,ytr=load_npz_memmap(Path(data_root)/spec['train'],cache);Xte,yte=load_npz_memmap(Path(data_root)/spec['test'],cache)
    tasks=tasks_for_seed(cfg.seed);task_limit=max_tasks or len(tasks);order=class_order(cfg.seed)
    byclass={c:np.where(np.asarray(ytr)==c)[0].astype(np.int64) for c in range(100)};testclass={c:np.where(np.asarray(yte)==c)[0].astype(np.int64) for c in range(100)}
    if smoke:
        task_limit=min(task_limit or 2,2);tasks=[tasks[0][:3],tasks[1][:2]];order=tasks[0]+tasks[1]
        byclass={c:v[:min(8,len(v))] for c,v in byclass.items()};testclass={c:v[:min(4,len(v))] for c,v in testclass.items()}
    class_hash=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
    feat_dim=int(Xtr.shape[1]);device=torch.device(device)
    # V15.5 constructed G and D before the classifier even in disabled-GAN ablations.
    # Retaining that initialization order preserves the historical classifier RNG stream.
    pre_G=V155ConditionalGenerator(feat_dim,100,cfg.z_dim).to(device);pre_D=V155ProjectionCritic(feat_dim,100).to(device)
    clf=V155Classifier(feat_dim,len(tasks[0]),hidden=cfg.classifier_hidden,emb_dim=cfg.classifier_embedding_dim,dropout=cfg.classifier_dropout).to(device)
    G,D=(pre_G,pre_D) if cfg.generated else (None,None)
    if not cfg.generated:del pre_G,pre_D
    scaler=RunningStandardScaler();anchors={};replay_X=replay_y=None;matrix=np.full((len(tasks),len(tasks)),np.nan);task_rows=[];exposure_rows=[];pool_rows=[];sampler_rows=[];opt_rows=[];optimizer_states={};next_task=0;elapsed_before=0.
    if ckpt.exists():
        p=load_checkpoint(torch,ckpt,map_location=device,restore_random=False);assert p['config_hash']==cfg.sha256();next_task=int(p['next_task']);clf=V155Classifier(feat_dim,50 if next_task==0 else len(sum(tasks[:next_task],[])),hidden=cfg.classifier_hidden,emb_dim=cfg.classifier_embedding_dim,dropout=cfg.classifier_dropout).to(device);clf.load_state_dict(p['classifier_state'])
        if G is not None:G.load_state_dict(p['generator_state']);D.load_state_dict(p['critic_state'])
        scaler=RunningStandardScaler.from_state(p['scaler_state']);anchors=p['anchors'];replay_X=p['replay_X_raw'];replay_y=p['replay_y_orig'];matrix=np.asarray(p['task_matrix']);task_rows=p['task_rows'];exposure_rows=p['exposure_rows'];pool_rows=p['pool_rows'];sampler_rows=p['sampler_rows'];opt_rows=p['optimizer_rows'];optimizer_states=p['optimizer_states'];elapsed_before=float(p.get('elapsed_seconds',0));restore_rng(torch,p['rng_state']);progress(f'RESUME next_task={next_task+1}')
    supports=[]
    for group in tasks:supports.append(sum(len(testclass[c]) for c in group))
    for ti,new_classes in enumerate(tasks[:task_limit]):
        if ti<next_task:continue
        seen=sum(tasks[:ti+1],[]);c2id={c:i for i,c in enumerate(seen)};idx_new=np.concatenate([byclass[c] for c in new_classes]);fit_scaler(scaler,Xtr,idx_new)
        if clf.num_classes<len(seen):clf.expand(len(seen))
        current_ds=IndexDataset(Xtr,ytr,idx_new,scaler,c2id,'current');parts=[current_ds];labels=[np.asarray([c2id[int(ytr[i])] for i in idx_new])];label_sources=['current']
        anchor_n=gen_n=0
        if cfg.anchors and anchors:
            idx=np.concatenate([np.asarray(anchors[c],dtype=np.int64) for c in seen if c in anchors]);anchor_ds=IndexDataset(Xtr,ytr,idx,scaler,c2id,'anchor');parts.append(anchor_ds);labels.append(np.asarray([c2id[int(ytr[i])] for i in idx]));label_sources.append('anchor');anchor_n=len(idx)
        if cfg.generated and replay_X is not None:
            gen_ds=ReplayDataset(replay_X,replay_y,scaler,c2id,'generated');parts.append(gen_ds);labels.append(np.asarray([c2id[int(y)] for y in replay_y]));label_sources.append('generated');gen_n=len(gen_ds)
        gan_stats={'generator_steps':0,'critic_steps':0,'generator_optimizer_state':None,'critic_optimizer_state':None}
        if cfg.generated:
            gan_parts=[current_ds]
            if replay_X is not None:gan_parts.append(ReplayDataset(replay_X,replay_y,scaler,c2id,'generated'))
            gan_ds=TaggedConcat(gan_parts);gan_loader=DataLoader(gan_ds,batch_size=(16 if smoke else cfg.batch_size),shuffle=True,drop_last=False,num_workers=0)
            if smoke:
                # Preserve algorithms while bounding the smoke workload to one batch/epoch.
                gan_loader=[next(iter(gan_loader))];smoke_cfg=type('SmokeCfg',(),dict(cfg.payload(),gan_epochs=1,critic_steps=1,batch_size=16,candidate_per_class=20,generated_k=4))();gan_stats=train_wgan_gp(torch,G,D,gan_loader,smoke_cfg,device)
            else:gan_stats=train_wgan_gp(torch,G,D,gan_loader,cfg,device,progress)
        train_ds=TaggedConcat(parts);all_labels=np.concatenate(labels);sampler=build_balanced_sampler(all_labels);batch=16 if smoke else cfg.batch_size;loader=DataLoader(train_ds,batch_size=batch,sampler=sampler,drop_last=False,num_workers=0)
        source_counts={'current':0,'anchor':0,'generated':0};class_counts={};steps,opt_state=classifier_train(torch,clf,loader,(type('SmokeCfg',(),dict(cfg.payload(),classifier_epochs=1,classifier_lr=cfg.classifier_lr,classifier_weight_decay=cfg.classifier_weight_decay,gradient_clip_norm=10.))() if smoke else cfg),device,source_counts,class_counts)
        assert_component_isolation(cfg,ti,anchor_n,gen_n,source_counts,gan_stats)
        # Evaluation occurs before post-task memory/replay update.
        y_all=[];p_all=[]
        for tj,group in enumerate(tasks[:ti+1]):
            idx=np.concatenate([testclass[c] for c in group]);yy,pp=_predict(torch,clf,Xte,yte,idx,scaler,c2id,cfg.test_batch_size,device);matrix[ti,tj]=sum(a==b for a,b in zip(yy,pp))/max(len(yy),1);y_all+=yy;p_all+=pp
        cm,met=confusion_metrics(y_all,p_all,len(seen));task_rows.append({'task_id':ti+1,'classes_seen':len(seen),'seen_accuracy':met['accuracy'],'macro_f1':met['macro_f1'],'weighted_f1':met['weighted_f1'],'balanced_accuracy':met['balanced_accuracy'],'scaler_sha256':scaler_sha256(scaler)})
        # Historical post-evaluation updates.
        if cfg.anchors:anchors=update_random_anchors(anchors,byclass,new_classes,cfg.anchor_k,cfg.buffer_refresh_seed+cfg.seed+ti)
        if cfg.generated:
            mean_parts=[]
            mean_rng=np.random.RandomState(cfg.seed+999+ti)
            for c in seen:
                if c in new_classes:
                    idx=np.asarray(byclass[c],dtype=np.int64)
                    if len(idx)>cfg.mean_real_per_class:idx=mean_rng.choice(idx,size=cfg.mean_real_per_class,replace=False)
                    mean_parts.append(IndexDataset(Xtr,ytr,idx,scaler,c2id,'current'))
                elif replay_y is not None:
                    pos=np.where(np.asarray(replay_y)==c)[0]
                    if len(pos)>cfg.mean_real_per_class:pos=mean_rng.choice(pos,size=cfg.mean_real_per_class,replace=False)
                    mean_parts.append(ReplayDataset(np.asarray(replay_X)[pos],np.asarray(replay_y)[pos],scaler,c2id,'generated'))
            mean_loader=DataLoader(TaggedConcat(mean_parts),batch_size=cfg.test_batch_size,shuffle=False,drop_last=False,num_workers=0);means=class_mean_mid(clf,mean_loader,len(seen),device)
            selcfg=cfg
            if smoke:selcfg=type('SmokeCfg',(),dict(cfg.payload(),candidate_per_class=20,generated_k=4))()
            Xstd,yids=select_generated(G,clf,means,len(seen),selcfg,device);replay_X=Xstd*scaler.scale_+scaler.mean_;inv={i:c for c,i in c2id.items()};replay_y=np.asarray([inv[int(x)] for x in yids],dtype=np.int64)
        expected_batches=math.ceil(len(train_ds)/batch);epochs=1 if smoke else cfg.classifier_epochs;totals=np.bincount(all_labels,minlength=len(seen)).astype(float);present=totals>0;n_present=int(present.sum());expected={}
        for source in ('current','anchor','generated'):
            src=np.concatenate([arr for arr,name in zip(labels,label_sources) if name==source]) if source in label_sources else np.empty(0,dtype=int);sc=np.bincount(src,minlength=len(seen)).astype(float);expected[source]=float(epochs*len(train_ds)/n_present*np.sum(np.divide(sc,totals,out=np.zeros_like(sc),where=present)))
        expected_class={str(c):float(epochs*len(train_ds)/n_present) for c in np.where(present)[0].tolist()}
        exposure_rows.append({'task_id':ti+1,'current_unique':len(current_ds),'anchor_unique':anchor_n,'generated_unique':gen_n,'classifier_pool_size':len(train_ds),'sampler_num_samples':len(train_ds),'batches_per_epoch':expected_batches,'classifier_epochs':epochs,'actual_optimizer_steps':steps,'total_sampled_indices':sum(source_counts.values()),'current_realized_draws':source_counts['current'],'anchor_realized_draws':source_counts['anchor'],'generated_realized_draws':source_counts['generated'],'realized_class_draws_json':json.dumps(class_counts,sort_keys=True),'draw_provenance':'SERIALIZED_ACTUAL','current_expected_draws':expected['current'],'anchor_expected_draws':expected['anchor'],'generated_expected_draws':expected['generated'],'expected_class_draws_json':json.dumps(expected_class,sort_keys=True),'expectation_provenance':'ANALYTIC_EXPECTATION','generator_steps':gan_stats['generator_steps'],'critic_steps':gan_stats['critic_steps']})
        pool_rows.append({'task_id':ti+1,'current':len(current_ds),'anchor':anchor_n,'generated':gen_n,'total':len(train_ds)})
        sampler_rows.append({'task_id':ti+1,'sampler':'WeightedRandomSampler','replacement':True,'num_samples':len(train_ds),'drop_last':False,'batch_size':batch,'expected_batches_per_epoch':expected_batches})
        opt_rows.append({'task_id':ti+1,'optimizer':'AdamW','lifecycle':'recreated_this_task','epochs':epochs,'actual_steps':steps,'gradient_clip_norm':cfg.gradient_clip_norm})
        optimizer_states={'classifier':opt_state,'generator':gan_stats['generator_optimizer_state'],'critic':gan_stats['critic_optimizer_state']}
        elapsed=elapsed_before+time.time()-start;payload=_checkpoint_payload(torch,cfg,ti+1,clf,G,D,scaler,anchors,replay_X,replay_y,matrix,task_rows,exposure_rows,pool_rows,sampler_rows,opt_rows,optimizer_states,elapsed);save_checkpoint(torch,ckpt,payload);progress(f'task {ti+1}/{task_limit} steps={steps} pool={len(train_ds)}')
        if deadline and time.time()>=deadline:raise TimeBudgetReached('max-hours reached after safe task checkpoint')
    if task_limit<len(tasks):return {'partial':True,'next_task':task_limit}
    # Final artifacts.
    final_indices=np.concatenate([testclass[c] for c in order]);final_c2id={c:i for i,c in enumerate(order)};yy,pp=_predict(torch,clf,Xte,yte,final_indices,scaler,final_c2id,cfg.test_batch_size,device);cm,finalmet=confusion_metrics(yy,pp,100);cl=continual_metrics(matrix,supports)
    atomic_json(run_dir/'config.json',{**cfg.payload(),'config_sha256':cfg.sha256(),'code_sha256':model_sha(package_root)});atomic_json(run_dir/'environment.json',environment(torch,package_root));atomic_json(run_dir/'class_order.json',{'class_order':order,'sha256':class_hash});atomic_json(run_dir/'task_supports.json',{'task_supports':supports});atomic_json(run_dir/'task_matrix.json',{'task_matrix':matrix.tolist()})
    csv_write(run_dir/'per_task_metrics.csv',task_rows);csv_write(run_dir/'replay_exposure.csv',exposure_rows);csv_write(run_dir/'classifier_pool_by_task.csv',pool_rows);csv_write(run_dir/'sampler_schedule_by_task.csv',sampler_rows);csv_write(run_dir/'optimizer_steps_by_task.csv',opt_rows)
    report=[]
    for i,c in enumerate(order):report.append({'arrival_id':i,'class_label':c,'precision':finalmet['per_class_precision'][i],'recall':finalmet['per_class_recall'][i],'f1':finalmet['per_class_f1'][i],'support':int(finalmet['support'][i])})
    csv_write(run_dir/'classification_report.csv',report);elapsed=elapsed_before+time.time()-start;atomic_json(run_dir/'resource_metrics.json',{'elapsed_seconds':elapsed,'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
    atomic_json(run_dir/'full-result.json',{'method':'V15.5 Standard-A','variant':cfg.variant,'dataset':cfg.dataset,'seed':cfg.seed,'tasks_completed':11,**cl,'final_metrics':{k:finalmet[k] for k in ('accuracy','macro_f1','weighted_f1','balanced_accuracy')},'elapsed_seconds':elapsed})
    atomic_json(run_dir/'status.json',{'status':'COMPLETED_UNVALIDATED','tasks_completed':11,'elapsed_seconds':elapsed});return {'partial':False,**cl}
