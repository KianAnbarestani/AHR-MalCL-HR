import pickle,random
import numpy as np
from v15_5_runner.checkpointing import load,rng_state,save

class Cuda:
    def is_available(self):return False
class FakeTensor:
    def __init__(self,x):self.data=np.asarray(x,dtype=np.uint8)
    def cpu(self):return self
    def to(self,**kwargs):return self
    def __array__(self,dtype=None):return np.asarray(self.data,dtype=dtype)
class FakeTorch:
    uint8=np.uint8
    def __init__(self):self.cuda=Cuda();self.state=np.arange(8,dtype=np.uint8)
    def get_rng_state(self):return FakeTensor(self.state.copy())
    def set_rng_state(self,x):self.state=np.asarray(x,dtype=np.uint8).copy()
    def save(self,x,p):pickle.dump(x,open(p,'wb'))
    def load(self,p,**kwargs):return pickle.load(open(p,'rb'))

def test_rng_restore(tmp_path):
    t=FakeTorch();random.seed(7);np.random.seed(7);payload={'rng_state':rng_state(t)};save(t,tmp_path/'c.pt',payload)
    expected=(random.random(),np.random.rand());t.state[:]=99;random.seed(9);np.random.seed(9);load(t,tmp_path/'c.pt')
    assert (random.random(),np.random.rand())==expected and np.array_equal(t.state,np.arange(8,dtype=np.uint8))

def test_synthetic_task_boundary_resume_equivalence(tmp_path):
    def step(w):return w+np.random.normal(size=w.shape)
    random.seed(5);np.random.seed(5);t=FakeTorch();w=np.zeros(4);w=step(w);p={'w':w.copy(),'rng_state':rng_state(t)};save(t,tmp_path/'c.pt',p);uninterrupted=step(w)
    random.seed(99);np.random.seed(99);q=load(t,tmp_path/'c.pt');resumed=step(q['w'])
    np.testing.assert_array_equal(uninterrupted,resumed)
