from pathlib import Path
import ast,json

ROOT=Path(__file__).resolve().parents[1]
def src(name):return (ROOT/name).read_text()

def test_all_python_parses():
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
def test_no_v16_import_or_runner_semantics():
    paths=[ROOT/'launch_v15_5_standard_a.py',*sorted((ROOT/'v15_5_runner').glob('*.py'))]
    text='\n'.join(p.read_text().lower() for p in paths)
    forbidden=('v16'+'_runner','fixed '+'masked','source-'+'proportional')
    assert not any(token in text for token in forbidden)
def test_classifier_topology_is_v155():
    text=src('v15_5_runner/models.py');assert 'hidden=(1024,512,256)' in text and 'nn.LayerNorm' in text and 'nn.GELU' in text and 'emb_dim=512' in text
def test_expanding_head_initialization_is_historical():
    text=src('v15_5_runner/models.py');assert 'nn.init.xavier_uniform_' in text and 'nn.init.zeros_' in text and '.copy_(old.weight)' in text
def test_optimizer_recreated_and_clipped():
    text=src('v15_5_runner/training.py');assert 'make_classifier_optimizer' in text and 'clip_grad_norm_' in text
def test_native_weighted_sampler_source():
    text=src('v15_5_runner/data.py');assert 'WeightedRandomSampler' in text and 'num_samples=len(y)' in text and 'replacement=True' in text
def test_gan_architecture_and_betas_source():
    m=src('v15_5_runner/models.py');r=src('v15_5_runner/replay.py');assert 'ConvTranspose1d' in m and 'V155ProjectionCritic' in m and 'betas=cfg.gan_optimizer_betas' in r
def test_carried_post_task_generated_replay_source():
    text=src('v15_5_runner/training.py');assert text.index('# Evaluation occurs before post-task memory/replay update.')<text.index('Xstd,yids=select_generated')
def test_component_assertions_are_executable():assert 'assert_component_isolation(cfg,ti' in src('v15_5_runner/training.py')
def test_resume_restores_rng_after_model_reconstruction():
    text=src('v15_5_runner/training.py');assert text.index("restore_random=False")<text.index("restore_rng(torch,p['rng_state'])")
def test_notebook_is_valid_launcher_only_json():
    p=ROOT/'V15_5_STANDARD_A_GPU_EXECUTION_COLAB.ipynb'
    if p.exists():
        n=json.loads(p.read_text());code='\n'.join(''.join(c.get('source',[])) for c in n['cells'] if c['cell_type']=='code')
        assert 'launch_v15_5_standard_a.py' in code and 'class V155Classifier' not in code
