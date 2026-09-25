import importlib.util,pytest

has_torch=importlib.util.find_spec('torch') is not None
pytestmark=pytest.mark.skipif(not has_torch,reason='PyTorch is intentionally not installed on this non-GPU packaging host')

def test_classifier_forward_and_expansion():
    import torch
    from v15_5_runner.models import V155Classifier
    m=V155Classifier(12,5);x=torch.randn(3,12);logits,mid=m(x,True);assert logits.shape==(3,5) and mid.shape==(3,256);old=m.out.weight.detach().clone();m.expand(7);assert torch.equal(old,m.out.weight[:5])
def test_optimizer_is_recreated():
    import torch
    from v15_5_runner.config import make_config
    from v15_5_runner.models import V155Classifier,make_classifier_optimizer
    m=V155Classifier(12,5);c=make_config('ember',47,'current_only',0);assert make_classifier_optimizer(torch,m,c) is not make_classifier_optimizer(torch,m,c)
def test_weighted_sampler_semantics():
    from v15_5_runner.data import build_balanced_sampler
    s=build_balanced_sampler([0,0,0,1]);assert s.num_samples==4 and s.replacement
def test_gan_shapes():
    import torch
    from v15_5_runner.models import V155ConditionalGenerator,V155ProjectionCritic
    g=V155ConditionalGenerator(31,100);d=V155ProjectionCritic(31,100);y=torch.tensor([1,2]);x=g(torch.randn(2,128),y);score,feat=d(x,y);assert x.shape==(2,31) and score.shape==(2,) and feat.shape==(2,256)
def test_gradient_clip_symbol_is_callable():
    import torch;assert callable(torch.nn.utils.clip_grad_norm_)
