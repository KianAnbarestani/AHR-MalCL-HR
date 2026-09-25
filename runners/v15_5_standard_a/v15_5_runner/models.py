"""Models transcribed from the preserved V15.5 historical runner, not V16."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class V155Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden=(1024,512,256), emb_dim=512, dropout=.25):
        super().__init__(); self.num_classes=int(num_classes); self.feat_dim=int(feat_dim); self.emb_dim=int(emb_dim)
        layers=[];prev=self.feat_dim
        for h in hidden:
            layers += [nn.Linear(prev,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(float(dropout))];prev=h
        self.mlp_features=nn.Sequential(*layers);self.mid_dim=int(hidden[-1])
        self.fc=nn.Linear(self.mid_dim,self.emb_dim);self.drop_fc=nn.Dropout(float(dropout));self.out=nn.Linear(self.emb_dim,self.num_classes)
    def expand(self,new_classes):
        new_classes=int(new_classes)
        if new_classes<=self.num_classes:return
        old=self.out;self.out=nn.Linear(old.in_features,new_classes).to(old.weight.device)
        with torch.no_grad():
            self.out.weight[:self.num_classes].copy_(old.weight);self.out.bias[:self.num_classes].copy_(old.bias)
            nn.init.xavier_uniform_(self.out.weight[self.num_classes:]);nn.init.zeros_(self.out.bias[self.num_classes:])
        self.num_classes=new_classes
    def forward(self,x,return_mid=False):
        mid=self.mlp_features(x.float());emb=self.drop_fc(F.gelu(self.fc(mid)));logits=self.out(emb)
        # Historical selector consumes the pre-embedding `mid` representation.
        return (logits,mid) if return_mid else logits

class V155ConditionalGenerator(nn.Module):
    def __init__(self,feat_dim,num_classes=100,z_dim=128,emb_dim=64,g_hidden=256):
        super().__init__();self.feat_dim=int(feat_dim);self.emb=nn.Embedding(int(num_classes),emb_dim)
        self.base_len=int(math.ceil(self.feat_dim/8));self.g_hidden=g_hidden
        self.fc1=nn.Linear(z_dim+emb_dim,1024);self.fc2=nn.Linear(1024,g_hidden*self.base_len)
        self.deconv1=nn.ConvTranspose1d(g_hidden,128,4,2,1);self.deconv2=nn.ConvTranspose1d(128,64,4,2,1);self.deconv3=nn.ConvTranspose1d(64,1,4,2,1)
        self.bn1=nn.BatchNorm1d(128);self.bn2=nn.BatchNorm1d(64)
    def forward(self,z,y):
        x=torch.cat([z,self.emb(y.view(-1).long())],1);x=F.gelu(self.fc1(x));x=F.gelu(self.fc2(x));x=x.view(z.size(0),self.g_hidden,self.base_len)
        x=F.gelu(self.bn1(self.deconv1(x)));x=F.gelu(self.bn2(self.deconv2(x)));x=self.deconv3(x)[:,0,:]
        return x[:,:self.feat_dim] if x.size(1)>=self.feat_dim else F.pad(x,(0,self.feat_dim-x.size(1)))

class V155ProjectionCritic(nn.Module):
    def __init__(self,feat_dim,num_classes=100,hidden_dim=256):
        super().__init__();self.conv1=nn.Conv1d(1,64,5,2,2);self.conv2=nn.Conv1d(64,128,5,2,2);self.conv3=nn.Conv1d(128,128,5,2,2)
        with torch.no_grad():
            h=torch.zeros(2,1,int(feat_dim));h=F.leaky_relu(self.conv1(h),.2);h=F.leaky_relu(self.conv2(h),.2);h=F.leaky_relu(self.conv3(h),.2);flat=h.view(2,-1).size(1)
        self.fc=nn.Linear(flat,hidden_dim);self.uncond=nn.Linear(hidden_dim,1);self.class_emb=nn.Embedding(int(num_classes),hidden_dim)
    def forward(self,x,y):
        h=x.float().unsqueeze(1);h=F.leaky_relu(self.conv1(h),.2);h=F.leaky_relu(self.conv2(h),.2);h=F.leaky_relu(self.conv3(h),.2);h=h.view(h.size(0),-1);feat=F.leaky_relu(self.fc(h),.2)
        return (self.uncond(feat)+(feat*self.class_emb(y.view(-1).long())).sum(1,keepdim=True)).squeeze(1),feat

def make_classifier_optimizer(torch_module, model, cfg):
    """Called once per task: historical optimizer recreation semantics."""
    return torch_module.optim.AdamW(model.parameters(),lr=cfg.classifier_lr,weight_decay=cfg.classifier_weight_decay)

