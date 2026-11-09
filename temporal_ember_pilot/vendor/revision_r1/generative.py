"""V15.5-style WGAN-GP updates and L1-mean/diverse replay selection."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolated = (alpha * real.float() + (1.0 - alpha) * fake.float()).requires_grad_(True)
    score, _ = critic(interpolated, labels)
    gradient = torch.autograd.grad(
        score, interpolated, torch.ones_like(score), create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0].reshape(real.size(0), -1)
    return ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()


def critic_update(generator, critic, real, labels, optimizer, *, z_dim: int, gp_weight: float, clip_norm: float) -> float:
    generator.train()
    critic.train()
    noise = torch.randn(real.size(0), z_dim, device=real.device)
    with torch.no_grad():
        fake = generator(noise, labels)
    real_score, _ = critic(real, labels)
    fake_score, _ = critic(fake, labels)
    loss = fake_score.mean() - real_score.mean() + gp_weight * gradient_penalty(critic, real, fake, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def generator_update(generator, critic, real, labels, optimizer, *, z_dim: int, feature_matching_weight: float, clip_norm: float) -> float:
    generator.train()
    critic.train()
    noise = torch.randn(real.size(0), z_dim, device=real.device)
    fake = generator(noise, labels)
    fake_score, fake_features = critic(fake, labels)
    loss = -fake_score.mean()
    with torch.no_grad():
        _, real_features = critic(real, labels)
    loss = loss + feature_matching_weight * F.l1_loss(fake_features.mean(0), real_features.mean(0))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def class_mean_mid(model, raw_features: np.ndarray, labels: np.ndarray, scaler, class_ids, device: str) -> dict[int, torch.Tensor]:
    from .resources import HOST_RAM_RESERVE_BYTES, require_host_memory
    require_host_memory(
        len(raw_features) * int(raw_features.shape[1]) * 8 + (8 << 20),
        reserve_bytes=HOST_RAM_RESERVE_BYTES,
        phase="bounded generated-replay class-mean batch",
    )
    was_training = model.training
    model.eval()
    try:
        transformed = scaler.transform(raw_features)
        tensor = torch.as_tensor(transformed, dtype=torch.float32, device=device)
        ids = torch.as_tensor(labels, dtype=torch.long, device=device)
        _, middle = model(tensor, return_mid=True)
        result = {}
        for class_id in class_ids:
            selected = ids == int(class_id)
            if selected.any():
                result[int(class_id)] = middle[selected].mean(0)
        return result
    finally:
        model.train(was_training)


@torch.no_grad()
def diverse_indices(features: torch.Tensor, distances: torch.Tensor, k: int, diversity_lambda: float = 0.25) -> torch.Tensor:
    if k <= 0:
        raise ValueError("selection quota must be positive")
    if features.ndim != 2 or distances.shape != (len(features),) or len(features) == 0:
        raise ValueError("candidate features and distances must be aligned and nonempty")
    if len(features) <= k:
        return torch.arange(len(features), device=features.device)
    normalized_distance = (distances - distances.min()) / (distances.max() - distances.min()).clamp_min(1e-8)
    normalized_features = F.normalize(features.float(), dim=1)
    chosen = [int(torch.argmin(normalized_distance).item())]
    minimum_distance = torch.cdist(normalized_features, normalized_features[chosen]).squeeze(1)
    while len(chosen) < k:
        coverage = (minimum_distance - minimum_distance.min()) / (minimum_distance.max() - minimum_distance.min()).clamp_min(1e-8)
        score = normalized_distance - float(diversity_lambda) * coverage
        score[chosen] = float("inf")
        selected = int(torch.argmin(score).item())
        chosen.append(selected)
        minimum_distance = torch.minimum(
            minimum_distance,
            torch.cdist(normalized_features, normalized_features[[selected]]).squeeze(1),
        )
    return torch.as_tensor(chosen, dtype=torch.long, device=features.device)


@torch.no_grad()
def generate_class_candidates(generator, classifier, mean, class_id: int, *, candidates_per_class: int, z_dim: int, device: str, chunk_size: int = 512):
    if candidates_per_class < 1 or chunk_size < 1:
        raise ValueError("generation candidate quota must be positive")
    generator_was_training = generator.training
    classifier_was_training = classifier.training
    generator.eval()
    classifier.eval()
    try:
        candidates, distances = [], []
        remaining = int(candidates_per_class)
        while remaining:
            size = min(int(chunk_size), remaining)
            remaining -= size
            labels = torch.full((size,), int(class_id), device=device, dtype=torch.long)
            noise = torch.randn(size, z_dim, device=device)
            generated = generator(noise, labels)
            _, middle = classifier(generated, return_mid=True)
            candidates.append(generated)
            distances.append(torch.abs(middle - mean).sum(1))
        return torch.cat(candidates), torch.cat(distances)
    finally:
        generator.train(generator_was_training)
        classifier.train(classifier_was_training)


@torch.no_grad()
def select_candidate_features(candidates: torch.Tensor, distances: torch.Tensor, *, class_id: int, quota: int, diversity_lambda: float = 0.25):
    if quota < 1:
        raise ValueError("selection quota must be positive")
    selected = diverse_indices(candidates, distances, min(int(quota), len(candidates)), diversity_lambda)
    all_candidates = candidates.detach().cpu().contiguous().numpy().astype(np.float32)
    selected_features = all_candidates[selected.detach().cpu().numpy()]
    record = {
        "global_class_id": int(class_id),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "candidate_l1_distances": distances.detach().cpu().tolist(),
        "selected_candidate_indices": selected.detach().cpu().tolist(),
        "candidate_features_sha256": hashlib.sha256(all_candidates.tobytes()).hexdigest(),
        "selected_features_sha256": hashlib.sha256(selected_features.tobytes()).hexdigest(),
    }
    return selected_features, record


@torch.no_grad()
def select_generated(
    generator, classifier, means, class_ids, *, candidates_per_class: int,
    quota: int, z_dim: int, diversity_lambda: float, device: str,
    chunk_size: int = 512, return_records: bool = False,
):
    if candidates_per_class < 1 or quota < 1 or chunk_size < 1:
        raise ValueError("generation candidate and selection quotas must be positive")
    feature_parts, label_parts, records = [], [], []
    for class_id in class_ids:
        if int(class_id) not in means:
            raise ValueError(f"no classifier class mean for global class {class_id}")
        candidates, distances = generate_class_candidates(
            generator, classifier, means[int(class_id)], int(class_id),
            candidates_per_class=candidates_per_class, z_dim=z_dim,
            device=device, chunk_size=chunk_size,
        )
        selected_features, record = select_candidate_features(
            candidates, distances, class_id=int(class_id), quota=quota,
            diversity_lambda=diversity_lambda,
        )
        feature_parts.append(selected_features)
        label_parts.append(np.full(len(selected_features), int(class_id), dtype=np.int64))
        records.append(record)
    result = (np.concatenate(feature_parts), np.concatenate(label_parts))
    return (*result, records) if return_records else result
