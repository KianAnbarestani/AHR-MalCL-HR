import hashlib, json, os, zipfile
from pathlib import Path
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # Lets CPU-only packaging tests exercise hashes/scaler/task logic.
    torch=None
    class Dataset: pass
    WeightedRandomSampler=None
from .config import DATASETS

def md5(path,chunk=1<<20):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''):h.update(b)
    return h.hexdigest()

def extract_npz_to_npy(npz_path,cache_dir):
    cache=Path(cache_dir)/md5(npz_path);cache.mkdir(parents=True,exist_ok=True);out=[]
    with zipfile.ZipFile(npz_path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            dst=cache/Path(name).name
            if not dst.exists() or dst.stat().st_size!=z.getinfo(name).file_size:
                tmp=dst.with_suffix('.tmp')
                with z.open(name) as src,open(tmp,'wb') as f:
                    while True:
                        b=src.read(1<<20)
                        if not b:break
                        f.write(b)
                os.replace(tmp,dst)
            out.append(dst)
    return out

def load_npz_memmap(path,cache_dir):
    arrays=[]
    for p in extract_npz_to_npy(path,cache_dir):
        a=np.load(p,mmap_mode='r');arrays.append((p,a))
    Xs=[x for x in arrays if x[1].ndim==2];ys=[x for x in arrays if x[1].ndim==1]
    if not Xs or not ys:raise ValueError(f'Cannot identify X/y arrays in {path}')
    xp,X=max(Xs,key=lambda z:z[1].shape[0]*z[1].shape[1]);matches=[z for z in ys if z[1].shape[0]==X.shape[0]]
    if not matches:raise ValueError(f'No label vector matching X in {path}')
    yp,y=max(matches,key=lambda z:int(np.issubdtype(z[1].dtype,np.integer)))
    return X,y

def preflight_dataset(data_root,dataset,cache_root=None,load_arrays=True):
    spec=DATASETS[dataset];root=Path(data_root);result={}
    for split in ('train','test'):
        p=root/spec[split]
        if not p.is_file():raise FileNotFoundError(p)
        got=md5(p);exp=spec[split+'_md5']
        if got!=exp:raise ValueError(f'{p.name} MD5 mismatch: {got} != {exp}')
        result[split]={'file':p.name,'md5':got}
        if load_arrays:
            X,y=load_npz_memmap(p,Path(cache_root or root/'.v155_cache'))
            if tuple(X.shape)!=tuple(spec[split+'_shape']):raise ValueError(f'{p.name} shape {X.shape} != {spec[split+"_shape"]}')
            labels=np.unique(y)
            if not np.array_equal(labels,np.arange(100)):raise ValueError(f'{p.name} labels are not exactly 0..99')
            result[split]['shape']=list(X.shape)
    return result

class RunningStandardScaler:
    def __init__(self):self.n=0;self.mean_=None;self.M2_=None;self.scale_=None
    def partial_fit(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2:raise ValueError('X must be 2-D')
        if self.mean_ is None:self.mean_=np.zeros(X.shape[1]);self.M2_=np.zeros(X.shape[1])
        bn=X.shape[0]
        if not bn:return self
        bm=X.mean(0);bM2=X.var(0)*bn
        if self.n==0:self.mean_=bm;self.M2_=bM2;self.n=bn;return self
        delta=bm-self.mean_;total=self.n+bn;self.mean_=self.mean_+delta*bn/total;self.M2_=self.M2_+bM2+(delta**2)*self.n*bn/total;self.n=total;return self
    def finalize(self):
        self.scale_=np.sqrt(self.M2_/max(self.n,1));self.scale_[self.scale_<1e-8]=1.;self.mean_=self.mean_.astype(np.float32);self.scale_=self.scale_.astype(np.float32);return self
    def transform(self,X):return ((np.asarray(X,dtype=np.float32)-self.mean_)/self.scale_).astype(np.float32)
    def state(self):return {'n':self.n,'mean':self.mean_,'M2':self.M2_,'scale':self.scale_}
    @classmethod
    def from_state(cls,s):o=cls();o.n=s['n'];o.mean_=s['mean'];o.M2_=s['M2'];o.scale_=s['scale'];return o

def fit_scaler(scaler,X,indices,chunk=20000):
    for start in range(0,len(indices),chunk):scaler.partial_fit(np.asarray(X[indices[start:start+chunk]],dtype=np.float32))
    return scaler.finalize()

def class_order(seed):return np.random.RandomState(int(seed)).permutation(100).astype(int).tolist()
def tasks_for_seed(seed):
    order=class_order(seed);return [order[:50]]+[order[i:i+5] for i in range(50,100,5)]

class IndexDataset(Dataset):
    def __init__(self,X,y,indices,scaler,c2id,source='current'):
        self.X=X;self.y=y;self.indices=np.asarray(indices,dtype=np.int64);self.scaler=scaler;self.c2id=c2id;self.source=source
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        idx=int(self.indices[i]);x=self.scaler.transform(np.asarray(self.X[idx],dtype=np.float32)[None])[0];label=self.c2id[int(self.y[idx])]
        return torch.from_numpy(x),torch.tensor(label,dtype=torch.long),self.source

class ReplayDataset(Dataset):
    def __init__(self,X,y,scaler,c2id,source='generated'):
        self.X=X;self.y=np.asarray(y,dtype=np.int64);self.scaler=scaler;self.c2id=c2id;self.source=source
    def __len__(self):return 0 if self.X is None else len(self.y)
    def __getitem__(self,i):
        x=self.scaler.transform(np.asarray(self.X[i],dtype=np.float32)[None])[0];return torch.from_numpy(x),torch.tensor(self.c2id[int(self.y[i])],dtype=torch.long),self.source

class TaggedConcat(Dataset):
    def __init__(self,parts):self.parts=list(parts);self.ends=np.cumsum([len(x) for x in parts])
    def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0
    def __getitem__(self,i):
        p=int(np.searchsorted(self.ends,i,side='right'));start=0 if p==0 else int(self.ends[p-1]);return self.parts[p][i-start]

def build_balanced_sampler(labels,torch_generator=None):
    if torch is None:raise RuntimeError('PyTorch is required to construct the historical sampler')
    y=np.asarray(labels,dtype=np.int64);counts=np.bincount(y).astype(np.float64);counts[counts==0]=1.;weights=1./counts[y]
    return WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=len(y),replacement=True,generator=torch_generator)

def scaler_sha256(scaler):
    h=hashlib.sha256();h.update(str(scaler.n).encode())
    for x in (scaler.mean_,scaler.M2_,scaler.scale_):
        if x is not None:h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()
