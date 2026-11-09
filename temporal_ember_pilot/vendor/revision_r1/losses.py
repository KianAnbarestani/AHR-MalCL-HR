"""Classifier objectives with explicit global-class and DER++ target semantics."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    current_ce: torch.Tensor
    replay_ce: torch.Tensor
    logit_mse: torch.Tensor


def global_to_head_indices(global_labels, head_class_ids):
    mapping = {int(class_id): index for index, class_id in enumerate(head_class_ids)}
    try:
        return torch.as_tensor([mapping[int(x)] for x in torch.as_tensor(global_labels).cpu().tolist()], dtype=torch.long)
    except KeyError as error:
        raise ValueError(f"global label {error.args[0]} is not present in classifier head") from error


def _masked_logits(logits: torch.Tensor, head_class_ids, seen_mask: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(head_class_ids, dtype=torch.long, device=logits.device)
    seen = torch.as_tensor(seen_mask, dtype=torch.bool, device=logits.device)
    if seen.shape != (100,) or ids.ndim != 1 or len(ids) != logits.shape[1]:
        raise ValueError("seen mask must cover global IDs 0..99 and head IDs must match logits columns")
    if torch.any(ids < 0) or torch.any(ids >= 100) or len(torch.unique(ids)) != len(ids):
        raise ValueError("classifier head class IDs must be unique global IDs 0..99")
    return logits.masked_fill(~seen[ids].unsqueeze(0), -torch.inf)


def _mapped_targets(labels, head_class_ids, device):
    ids = [int(value) for value in head_class_ids]
    mapping = {class_id: index for index, class_id in enumerate(ids)}
    values = torch.as_tensor(labels, dtype=torch.long).cpu().tolist()
    try:
        mapped = [mapping[int(label)] for label in values]
    except KeyError as error:
        raise ValueError(f"label {error.args[0]} is not mapped to the seen classifier head") from error
    return torch.as_tensor(mapped, dtype=torch.long, device=device)


def pooled_ce_loss(logits: torch.Tensor, labels, head_class_ids, seen_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("classifier CE requires a nonempty [batch, classes] logit tensor")
    target = _mapped_targets(labels, head_class_ids, logits.device)
    if len(target) != len(logits):
        raise ValueError("labels and logits batch sizes differ")
    masked = _masked_logits(logits, head_class_ids, seen_mask)
    return F.cross_entropy(masked, target)


def derpp_loss(
    logits: torch.Tensor,
    labels,
    source_ids,
    stored_logits: torch.Tensor | None,
    stored_class_ids,
    head_class_ids,
    seen_mask: torch.Tensor,
    stored_mask: torch.Tensor | None = None,
    beta: float = 1.0,
    alpha: float = 0.5,
) -> LossBreakdown:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("DER++ requires a nonempty [batch, classes] logit tensor")
    if beta != 1.0 or alpha != 0.5:
        raise ValueError("revision_R1 DER++ fixes beta=1 and alpha=0.5")
    sources = [str(value) for value in source_ids]
    if len(sources) != len(logits) or any(source not in {"current", "anchor"} for source in sources):
        raise ValueError("DER++ matched source IDs must be current or anchor, one per example")
    current_mask = torch.as_tensor([source == "current" for source in sources], dtype=torch.bool, device=logits.device)
    replay_mask = ~current_mask
    mapped = _mapped_targets(labels, head_class_ids, logits.device)
    if len(mapped) != len(logits):
        raise ValueError("labels and logits batch sizes differ")
    masked = _masked_logits(logits, head_class_ids, seen_mask)
    zero = logits.sum() * 0.0
    current_ce = F.cross_entropy(masked[current_mask], mapped[current_mask]) if current_mask.any() else zero
    replay_ce = F.cross_entropy(masked[replay_mask], mapped[replay_mask]) if replay_mask.any() else zero
    logit_mse = zero
    if replay_mask.any():
        if stored_logits is None:
            raise ValueError("stored logits are required for every DER++ replay batch")
        targets = torch.as_tensor(stored_logits, dtype=logits.dtype, device=logits.device)
        stored_ids = [int(value) for value in stored_class_ids]
        if targets.ndim != 2 or targets.shape != (len(logits), len(stored_ids)):
            raise ValueError("stored logits shape must equal [batch, number of stored class IDs]")
        if len(set(stored_ids)) != len(stored_ids) or any(class_id < 0 or class_id >= 100 for class_id in stored_ids):
            raise ValueError("stored class IDs must be unique global IDs 0..99")
        available = torch.ones_like(targets, dtype=torch.bool) if stored_mask is None else torch.as_tensor(stored_mask, dtype=torch.bool, device=logits.device)
        if available.shape != targets.shape:
            raise ValueError("stored target mask shape must match stored logits")
        head_ids = [int(value) for value in head_class_ids]
        id_to_stored = {class_id: index for index, class_id in enumerate(stored_ids)}
        common = [class_id for class_id in head_ids if class_id in id_to_stored]
        if not common:
            raise ValueError("stored logits have no class IDs mapped to the current seen head")
        head_columns = torch.as_tensor([head_ids.index(value) for value in common], dtype=torch.long, device=logits.device)
        stored_columns = torch.as_tensor([id_to_stored[value] for value in common], dtype=torch.long, device=logits.device)
        valid = available[replay_mask][:, stored_columns]
        seen = torch.as_tensor(seen_mask, dtype=torch.bool, device=logits.device)
        valid &= seen[torch.as_tensor(common, dtype=torch.long, device=logits.device)].unsqueeze(0)
        if not valid.any():
            raise ValueError("DER++ replay batch has no stored targets for currently seen classes")
        prediction = masked[replay_mask][:, head_columns]
        target_values = targets[replay_mask][:, stored_columns]
        logit_mse = ((prediction - target_values).square() * valid).sum() / valid.sum()
    total = current_ce + beta * replay_ce + alpha * logit_mse
    return LossBreakdown(total, current_ce, replay_ce, logit_mse)


def classification_loss(arm: str, logits: torch.Tensor, labels, source_ids, stored_logits, stored_class_ids, head_class_ids, seen_mask, stored_mask=None) -> torch.Tensor:
    if arm == "derpp_matched":
        return derpp_loss(logits, labels, source_ids, stored_logits, stored_class_ids, head_class_ids, seen_mask, stored_mask).total
    if arm not in {"current_only", "anchor_only", "er_matched", "generated_only", "full_hybrid"}:
        raise ValueError(f"unsupported classifier arm {arm!r}")
    return pooled_ce_loss(logits, labels, head_class_ids, seen_mask)

