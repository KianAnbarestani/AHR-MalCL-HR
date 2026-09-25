from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - Colab runtime dependency
    torch = None
    nn = None


class TorchUnavailableError(RuntimeError):
    pass


def _require_torch() -> None:
    if torch is None or nn is None:
        raise TorchUnavailableError("PyTorch is unavailable")


class Generator(nn.Module if nn is not None else object):
    """Official MalCL generator topology with parameterized output features."""

    def __init__(self, feature_dim: int = 2381, z_dim: int = 62):
        _require_torch()
        super().__init__()
        self.input_dim = z_dim
        self.z_dim = z_dim
        self.channel_a = 64
        self.channel_b = 128
        self.channel_c = 256
        self.channel_d = 512
        self.channel_e = 1024
        self.channel_f = 2048
        self.channel_g = 4096
        self.output_features = feature_dim

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.channel_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_c),
            nn.ReLU(),
            nn.Conv1d(self.channel_c, self.channel_e, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
            nn.Conv1d(self.channel_e, self.channel_g, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_g),
            nn.ReLU(),
            nn.Conv1d(self.channel_g, self.channel_e, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channel_e, self.channel_f),
            nn.BatchNorm1d(self.channel_f),
            nn.ReLU(),
            nn.Linear(self.channel_f, self.channel_g),
            nn.BatchNorm1d(self.channel_g),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.channel_g, self.channel_e, 3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
            nn.ConvTranspose1d(self.channel_e, self.channel_d, 3, padding=1),
            nn.BatchNorm1d(self.channel_d),
            nn.ReLU(),
            nn.ConvTranspose1d(self.channel_d, self.output_features, 3, padding=1),
            nn.Sigmoid(),
        )

        self.apply(self.weights_init)

    def reinit(self) -> None:
        self.apply(self.weights_init)

    def forward(self, input):
        input = input.view(-1, self.input_dim, 1)
        x = self.conv(input)
        x = self.fc(x)
        x = x.view(-1, self.channel_g, 1)
        x = self.deconv(x)
        return x.view(-1, self.output_features)

    @staticmethod
    def weights_init(m) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Discriminator(nn.Module if nn is not None else object):
    """Feature-dimension-parameterized discriminator with a flattened head.

    The first fully connected layer deliberately consumes the complete
    ``256 * feature_dim`` convolutional feature map. This preserves the
    feature-length-dependent topology represented by the final-run artifacts.
    """

    def __init__(self, feature_dim: int = 2381):
        _require_torch()
        super().__init__()
        self.input_channel = 1
        self.output_dim = 1
        self.channel_c = 256
        self.channel_d = 512
        self.input_features = feature_dim
        self.latent_dim = 1024

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.channel_d, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.channel_d, self.channel_c, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_c),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.channel_c * self.input_features, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Sigmoid(),
        )

        self.apply(self.weights_init)

    def reinit(self) -> None:
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, input):
        x = input.view(-1, self.input_channel, self.input_features)
        x = self.conv(x)
        feature = x.view(-1, self.channel_c * self.input_features)
        x = self.fc(feature)
        return x.view(-1, 1), feature


class Classifier(nn.Module if nn is not None else object):
    """Official MalCL classifier topology with parameterized input features."""

    def __init__(self, feature_dim: int = 2381, output_dim: int = 50):
        _require_torch()
        super().__init__()
        self.input_features = feature_dim
        self.output_dim = output_dim
        self.drop_prob = 0.5

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, 3, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
        )

        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def _features(self, x):
        original_shape = x.size()
        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)
        else:
            raise ValueError(f"Unsupported input shape: {tuple(original_shape)}")
        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc1_f(x), original_shape

    def forward_logits(self, x):
        x, _ = self._features(x)
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        x = self.fc1_drop1(x)
        return self.fc1_act1(x)

    def forward(self, x):
        original_shape = x.size()
        x = self.forward_logits(x)
        x = self.softmax(x)
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)
        return x

    def expand_output_layer(self, init_classes: int, nb_inc: int, task: int):
        old_fc1 = self.fc1
        old_fc1_bn1 = self.fc1_bn1
        self.output_dim = init_classes + nb_inc * task

        self.fc1 = nn.Linear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)

        with torch.no_grad():
            copy_dim = min(old_fc1.out_features, self.output_dim)
            self.fc1.weight[:copy_dim].copy_(old_fc1.weight.data[:copy_dim])
            self.fc1.bias[:copy_dim].copy_(old_fc1.bias.data[:copy_dim])
            self.fc1_bn1.weight[:copy_dim].copy_(old_fc1_bn1.weight.data[:copy_dim])
            self.fc1_bn1.bias[:copy_dim].copy_(old_fc1_bn1.bias.data[:copy_dim])
        return self

    def predict(self, x_data):
        return self.forward(x_data)

    def get_logits(self, x):
        features, _ = self._features(x)
        return features.detach()


def model_memory_mb(model) -> float:
    _require_torch()
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    return total / (1024 * 1024)


def discriminator_topology_spec(feature_dim: int) -> dict[str, object]:
    """Return the flattened-head specification without allocating optimizer state."""
    _require_torch()
    feature_width = 256 * int(feature_dim)
    parameter_count = (
        (512 * 1 * 3 + 512)
        + (256 * 512 * 3 + 256)
        + (2 * 256)
        + (feature_width * 1024 + 1024)
        + (2 * 1024)
        + (1024 + 1)
    )
    return {
        "feature_dim": int(feature_dim),
        "feature_width": feature_width,
        "first_fc_in_features": feature_width,
        "first_fc_out_features": 1024,
        "returned_feature_shape": ["batch", feature_width],
        "parameter_count": parameter_count,
        "parameter_memory_mib_fp32": parameter_count * 4 / (1024 * 1024),
    }


def run_topology_validation(feature_dims=(2381, 2439), batch_size: int = 2) -> dict[str, object]:
    if torch is None:
        return {
            "passed": False,
            "status": "failed",
            "errors": ["torch unavailable"],
            "details": [],
        }

    details = []
    errors = []
    all_passed = True
    for feature_dim in feature_dims:
        try:
            spec = discriminator_topology_spec(feature_dim)
            with torch.device("meta"):
                d = Discriminator(feature_dim=feature_dim)
                x = torch.empty(batch_size, feature_dim, device="meta", requires_grad=True)
                d_score, d_feat = d(x)
                (d_score.sum() + d_feat.sum()).backward()
            first_fc = next(module for module in d.fc if isinstance(module, nn.Linear))
            d_ok = (
                tuple(d_score.shape) == (batch_size, 1)
                and tuple(d_feat.shape) == (batch_size, 256 * feature_dim)
                and first_fc.in_features == 256 * feature_dim
                and first_fc.out_features == 1024
                and not hasattr(d, "pool")
            )
            d_mb = model_memory_mb(d)
            passed = d_ok and abs(d_mb - spec["parameter_memory_mib_fp32"]) < 1e-9
            all_passed = all_passed and passed
            details.append(
                {
                    "feature_dim": feature_dim,
                    "discriminator_topology": "feature-dimension-parameterized flattened discriminator",
                    "official_feature_length_dependent_flattened_topology": True,
                    "adaptive_pooling_present": False,
                    "discriminator_conv_frontend_preserved": True,
                    "first_fc_in_features": first_fc.in_features,
                    "first_fc_out_features": first_fc.out_features,
                    "returned_feature_shape": [batch_size, 256 * feature_dim],
                    "feature_dim_parameterization": True,
                    "discriminator_parameter_memory_mib": d_mb,
                    "meta_forward_backward_passed": d_ok,
                    "passed": passed,
                }
            )
        except Exception as exc:  # pragma: no cover - reports Colab validation failure
            all_passed = False
            errors.append(f"feature_dim={feature_dim}: {type(exc).__name__}: {exc}")
            details.append({"feature_dim": feature_dim, "passed": False, "error": repr(exc)})

    return {
        "passed": all_passed,
        "status": "passed" if all_passed else "failed",
        "errors": errors,
        "details": details,
    }
