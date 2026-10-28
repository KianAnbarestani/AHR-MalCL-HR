"""V15.5-compatible expanding MLP and paired initialization helpers."""

import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F


class ExpandingClassifier(nn.Module):
    def __init__(self, feature_dim: int, initial_classes: int = 50, hidden=(1024, 512, 256), embedding_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(initial_classes)
        self.hidden_sizes = tuple(int(x) for x in hidden)
        self.embedding_dim = int(embedding_dim)
        if self.feature_dim <= 0 or self.num_classes <= 0 or not self.hidden_sizes or min(self.hidden_sizes) <= 0:
            raise ValueError("classifier dimensions must be positive")
        layers = []
        previous = self.feature_dim
        for width in self.hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(float(dropout))))
            previous = width
        self.mlp_features = nn.Sequential(*layers)
        self.fc = nn.Linear(previous, self.embedding_dim)
        self.drop_fc = nn.Dropout(float(dropout))
        self.out = nn.Linear(self.embedding_dim, self.num_classes)

    def expand(self, new_class_count: int, seed: int) -> None:
        new_class_count = int(new_class_count)
        if new_class_count < self.num_classes:
            raise ValueError("classifier head cannot shrink")
        if new_class_count == self.num_classes:
            return
        old = self.out
        expanded = nn.Linear(old.in_features, new_class_count).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            expanded.weight[: self.num_classes].copy_(old.weight)
            expanded.bias[: self.num_classes].copy_(old.bias)
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            fan_in, fan_out = old.in_features, new_class_count - self.num_classes
            bound = math.sqrt(6.0 / max(fan_in + fan_out, 1))
            fresh = torch.empty((new_class_count - self.num_classes, fan_in), dtype=torch.float32)
            fresh.uniform_(-bound, bound, generator=generator)
            expanded.weight[self.num_classes :].copy_(fresh.to(device=old.weight.device, dtype=old.weight.dtype))
            expanded.bias[self.num_classes :].zero_()
        self.out = expanded
        self.num_classes = new_class_count

    def forward(self, x: torch.Tensor, return_mid: bool = False):
        mid = self.mlp_features(x.float())
        embedding = self.drop_fc(F.gelu(self.fc(mid)))
        logits = self.out(embedding)
        return (logits, mid) if return_mid else logits


def build_classifier(feature_dim: int, seed: int, initial_classes: int = 50, hidden=(1024, 512, 256), embedding_dim: int = 512, dropout: float = 0.25, device: str | torch.device = "cpu") -> ExpandingClassifier:
    target = torch.device(device)
    cuda_devices = [target.index or torch.cuda.current_device()] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if target.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        model = ExpandingClassifier(feature_dim, initial_classes, hidden, embedding_dim, dropout)
    return model.to(target)


def initial_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class V155ConditionalGenerator(nn.Module):
    """Conditional convolutional generator transcribed from the V15.5 runner."""

    def __init__(self, feature_dim: int, num_classes: int = 100, z_dim: int = 128, embedding_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.z_dim = int(z_dim)
        self.embedding = nn.Embedding(int(num_classes), int(embedding_dim))
        self.base_len = int(math.ceil(self.feature_dim / 8))
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(self.z_dim + int(embedding_dim), 1024)
        self.fc2 = nn.Linear(1024, self.hidden * self.base_len)
        self.deconv1 = nn.ConvTranspose1d(self.hidden, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose1d(64, 1, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([noise, self.embedding(labels.view(-1).long())], dim=1)
        value = F.gelu(self.fc1(joined))
        value = F.gelu(self.fc2(value)).view(noise.size(0), self.hidden, self.base_len)
        value = F.gelu(self.bn1(self.deconv1(value)))
        value = F.gelu(self.bn2(self.deconv2(value)))
        value = self.deconv3(value)[:, 0, :]
        return value[:, : self.feature_dim] if value.size(1) >= self.feature_dim else F.pad(value, (0, self.feature_dim - value.size(1)))


class V155ProjectionCritic(nn.Module):
    """Projection critic transcribed from the V15.5 Standard-A runner."""

    def __init__(self, feature_dim: int, num_classes: int = 100, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, 2, 2)
        self.conv2 = nn.Conv1d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv1d(128, 128, 5, 2, 2)
        with torch.no_grad():
            sample = torch.zeros(2, 1, int(feature_dim))
            sample = F.leaky_relu(self.conv1(sample), 0.2)
            sample = F.leaky_relu(self.conv2(sample), 0.2)
            sample = F.leaky_relu(self.conv3(sample), 0.2)
            flat_dim = sample.reshape(2, -1).shape[1]
        self.fc = nn.Linear(flat_dim, int(hidden_dim))
        self.unconditional = nn.Linear(int(hidden_dim), 1)
        self.class_embedding = nn.Embedding(int(num_classes), int(hidden_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = features.float().unsqueeze(1)
        value = F.leaky_relu(self.conv1(value), 0.2)
        value = F.leaky_relu(self.conv2(value), 0.2)
        value = F.leaky_relu(self.conv3(value), 0.2)
        value = value.reshape(value.size(0), -1)
        representation = F.leaky_relu(self.fc(value), 0.2)
        score = self.unconditional(representation) + (
            representation * self.class_embedding(labels.view(-1).long())
        ).sum(1, keepdim=True)
        return score.squeeze(1), representation

