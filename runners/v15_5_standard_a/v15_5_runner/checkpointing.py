from pathlib import Path
import os, random
import numpy as np

def rng_state(torch):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().cpu(),
      'torch_cuda':([x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])}
def restore_rng(torch,state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'].to(dtype=torch.uint8,device='cpu'))
    if torch.cuda.is_available() and state.get('torch_cuda'):torch.cuda.set_rng_state_all([x.to(dtype=torch.uint8,device='cpu') for x in state['torch_cuda']])
def save(torch,path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def load(torch,path,map_location='cpu',restore_random=True):
    try:p=torch.load(path,map_location=map_location,weights_only=False)
    except TypeError:p=torch.load(path,map_location=map_location)
    if restore_random:restore_rng(torch,p['rng_state'])
    return p

