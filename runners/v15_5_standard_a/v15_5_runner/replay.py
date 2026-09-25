import numpy as np
import torch
import torch.nn.functional as F

def gradient_penalty(critic,real,fake,y):
    alpha=torch.rand(real.size(0),1,device=real.device);inter=(alpha*real.float()+(1-alpha)*fake.float()).requires_grad_(True);score,_=critic(inter,y)
    grad=torch.autograd.grad(score,inter,torch.ones_like(score),create_graph=True,retain_graph=True,only_inputs=True)[0].view(real.size(0),-1)
    return ((grad.norm(2,dim=1)-1.)**2).mean()

def train_wgan_gp(torch_module,G,D,loader,cfg,device,progress=None):
    """Historical per-task optimizer recreation with betas=(0,.9)."""
    G.train();D.train();opt_g=torch_module.optim.Adam(G.parameters(),lr=cfg.generator_lr,betas=cfg.gan_optimizer_betas);opt_d=torch_module.optim.Adam(D.parameters(),lr=cfg.critic_lr,betas=cfg.gan_optimizer_betas)
    gs=cs=0
    for epoch in range(cfg.gan_epochs):
        for batch in loader:
            xb,yb=batch[0].to(device).float(),batch[1].to(device).long()
            if xb.size(0)<2:continue
            for _ in range(cfg.critic_steps):
                z=torch_module.randn(xb.size(0),cfg.z_dim,device=device)
                with torch_module.no_grad():fake=G(z,yb)
                sr,_=D(xb,yb);sf,_=D(fake,yb);loss_d=sf.mean()-sr.mean()+cfg.gp_lambda*gradient_penalty(D,xb,fake,yb)
                opt_d.zero_grad(set_to_none=True);loss_d.backward();torch_module.nn.utils.clip_grad_norm_(D.parameters(),cfg.gradient_clip_norm);opt_d.step();cs+=1
            z=torch_module.randn(xb.size(0),cfg.z_dim,device=device);fake=G(z,yb);sf,ff=D(fake,yb);loss_g=-sf.mean()
            with torch_module.no_grad():_,fr=D(xb,yb)
            loss_g=loss_g+cfg.feature_matching_weight*F.l1_loss(ff.mean(0),fr.mean(0));opt_g.zero_grad(set_to_none=True);loss_g.backward();torch_module.nn.utils.clip_grad_norm_(G.parameters(),cfg.gradient_clip_norm);opt_g.step();gs+=1
        if progress:progress(f'GAN epoch {epoch+1}/{cfg.gan_epochs}')
    return {'generator_steps':gs,'critic_steps':cs,'generator_optimizer_state':opt_g.state_dict(),'critic_optimizer_state':opt_d.state_dict()}

@torch.no_grad()
def class_mean_mid(clf,loader,C,device):
    clf.eval();sums=None;counts=torch.zeros(C,device=device)
    for batch in loader:
        x,y=batch[0].to(device),batch[1].to(device);_,mid=clf(x,return_mid=True)
        if sums is None:sums=torch.zeros(C,mid.size(1),device=device)
        sums.index_add_(0,y,mid.float());counts.index_add_(0,y,torch.ones_like(y,dtype=torch.float32))
    return sums/counts[:,None].clamp_min(1.)

@torch.no_grad()
def diverse_indices(x,dist,k,lam=.25):
    if x.size(0)<=k:return torch.arange(x.size(0),device=x.device)
    nd=(dist-dist.min())/(dist.max()-dist.min()).clamp_min(1e-8);chosen=[int(torch.argmin(nd).item())];feat=F.normalize(x.float(),dim=1);mind=torch.cdist(feat,feat[chosen]).squeeze(1)
    while len(chosen)<k:
        coverage=(mind-mind.min())/(mind.max()-mind.min()).clamp_min(1e-8);score=nd-lam*coverage;score[chosen]=float('inf');i=int(torch.argmin(score).item());chosen.append(i);mind=torch.minimum(mind,torch.cdist(feat,feat[[i]]).squeeze(1))
    return torch.tensor(chosen,device=x.device)

@torch.no_grad()
def select_generated(G,clf,means,C,cfg,device):
    G.eval();clf.eval();xs=[];ys=[]
    for c in range(C):
        cand=[];dist=[];remaining=cfg.candidate_per_class
        while remaining:
            n=min(512,remaining);remaining-=n;y=torch.full((n,),c,device=device,dtype=torch.long);x=G(torch.randn(n,cfg.z_dim,device=device),y);_,mid=clf(x,return_mid=True)
            cand.append(x);dist.append(torch.sum(torch.abs(mid-means[c]),dim=1))
        x=torch.cat(cand);d=torch.cat(dist);idx=diverse_indices(x,d,cfg.generated_k,cfg.diversity_lambda);xs.append(x[idx].cpu().float().numpy());ys.append(np.full(len(idx),c,dtype=np.int64))
    return np.concatenate(xs).astype(np.float32),np.concatenate(ys)

def update_random_anchors(buffer,class_to_indices,new_classes,K,seed):
    rng=np.random.RandomState(seed);out={int(k):list(v) for k,v in buffer.items()}
    for label in new_classes:
        arr=np.asarray(class_to_indices[int(label)],dtype=np.int64);selected=arr.tolist() if len(arr)<=K else rng.choice(arr,size=K,replace=False).astype(int).tolist();out[int(label)]=selected
    return out
