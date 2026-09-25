from dataclasses import asdict, dataclass
import hashlib, json

DATASETS = {
    "ember": {
        "train": "EMBER_Class_train.npz", "test": "EMBER_Class_test.npz",
        "train_md5": "27e39d1cb697434107f92a8084128734",
        "test_md5": "9a844162d5ccca20987ce520d6355f33",
        "train_shape": (303331, 2381), "test_shape": (33704, 2381),
        "feature_dim": 2381, "anchor_k": 100,
    },
    "az_class": {
        "train": "AZ_Class_Train.npz", "test": "AZ_Class_Test.npz",
        "train_md5": "644649dfb93f0f0086052a923b657833",
        "test_md5": "1e4014ff6eb613845a6bcf38a2461001",
        "train_shape": (257023, 2439), "test_shape": (28559, 2439),
        "feature_dim": 2439, "anchor_k": 200,
    },
}

@dataclass(frozen=True)
class RunConfig:
    dataset: str
    seed: int
    variant: str
    anchor_k: int
    generated_k: int = 100
    candidate_per_class: int = 1000
    batch_size: int = 256
    test_batch_size: int = 2048
    classifier_epochs: int = 3
    gan_epochs: int = 3
    critic_steps: int = 5
    classifier_lr: float = 1e-3
    classifier_weight_decay: float = 1e-4
    generator_lr: float = 2e-4
    critic_lr: float = 2e-4
    gradient_clip_norm: float = 10.0
    gp_lambda: float = 10.0
    feature_matching_weight: float = 1.0
    diversity_lambda: float = 0.25
    z_dim: int = 128
    classifier_hidden: tuple = (1024, 512, 256)
    classifier_embedding_dim: int = 512
    classifier_dropout: float = 0.25
    task_sizes: tuple = (50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    scaler_mode: str = "incremental"
    sampler: str = "WeightedRandomSampler"
    sampler_replacement: bool = True
    drop_last: bool = False
    optimizer_lifecycle: str = "recreate_per_task"
    gan_optimizer_betas: tuple = (0.0, 0.9)
    generated_replay_timing: str = "carry_G_t_to_task_t_plus_1"
    selection: str = "l1_class_mean_diverse"
    mean_real_per_class: int = 512
    buffer_refresh_seed: int = 12345
    kd_enabled: bool = False
    prototype_alignment_enabled: bool = False
    diversity_auxiliary_enabled: bool = False
    standard: str = "PROTOCOL_MATCHED_STANDARD_A"

    @property
    def anchors(self): return self.variant == "anchor_only"
    @property
    def generated(self): return self.variant == "generated_only"
    @property
    def gan_active(self): return self.generated
    @property
    def feature_dim(self): return DATASETS[self.dataset]["feature_dim"]
    def payload(self):
        d=asdict(self);d.update({"anchors":self.anchors,"generated":self.generated,"gan_active":self.gan_active,"feature_dim":self.feature_dim});return d
    def sha256(self): return hashlib.sha256(json.dumps(self.payload(),sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()

def make_config(dataset, seed, variant, k):
    if dataset not in DATASETS: raise ValueError(dataset)
    if variant not in ("current_only", "anchor_only", "generated_only"): raise ValueError(variant)
    anchor_k = DATASETS[dataset]["anchor_k"] if variant == "anchor_only" else 0
    if variant == "anchor_only" and int(k) != anchor_k: raise ValueError(f"{dataset} anchor-only requires K={anchor_k}")
    expected_k = anchor_k if variant == "anchor_only" else (100 if variant == "generated_only" else 0)
    if str(k) not in ({str(expected_k)} if expected_k else {"0", "NA", "None", ""}):
        raise ValueError(f"{variant} requires registry K={expected_k or 'NA'}")
    cfg=RunConfig(dataset=dataset,seed=int(seed),variant=variant,anchor_k=anchor_k)
    assert not cfg.kd_enabled and not cfg.prototype_alignment_enabled and not cfg.diversity_auxiliary_enabled
    return cfg
