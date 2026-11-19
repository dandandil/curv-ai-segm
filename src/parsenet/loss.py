"""Лоссы ParSeNet: `EmbeddingLoss.triplet_loss` и `primitive_loss` из `src/segment_loss.py`.

Дословно, кроме источника случайности: вместо глобального `np.random` —
переданный генератор, чтобы в воркерах и между прогонами выборка была
воспроизводимой.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def triplet_loss(output: Tensor, labels: np.ndarray, margin: float, max_pairs: int,
                 samples_per_segment: int, rng: np.random.Generator) -> Tensor:
    """`output` — `(B, D, N)`, `labels` — `(B, N)`."""
    batch_size, _, n = output.shape
    output = F.normalize(output.permute(0, 2, 1), p=2, dim=2)
    loss_diff = output.new_zeros(1)
    only_one = 0
    for i in range(batch_size):
        p = labels[i]
        unique = np.unique(p)
        count = min(n // unique.shape[0] + 1, samples_per_segment)
        sampled = {l: output[i, rng.choice(np.where(p == l)[0], count, replace=True), :] for l in unique}
        keys = sorted(sampled)
        if len(keys) == 1:
            only_one += 1
            continue
        loss_shape = output.new_zeros(1)
        normalization = 0
        for _ in range(min(max_pairs, len(keys) * len(keys))):
            k1, k2 = rng.choice(len(keys)), rng.choice(len(keys))
            if k1 == k2:
                continue
            normalization += 1
            anchor = sampled[keys[k1]].unsqueeze(1)
            pos = sampled[keys[k1]].unsqueeze(0)
            neg = sampled[keys[k2]].unsqueeze(0)
            constraint = F.relu(torch.sum((anchor - pos) ** 2, 2) - torch.sum((anchor - neg) ** 2, 2) + margin)
            loss = torch.sum(constraint) - constraint.trace()
            satisfied = (torch.sum(constraint > 0) + 1.0).float()
            loss_shape = loss_shape + loss / satisfied.detach()
        loss_diff = loss_diff + loss_shape / (normalization + 1e-8)
    return loss_diff / (batch_size - only_one + 1e-8)


def primitive_loss(log_prob: Tensor, types: Tensor) -> Tensor:
    return F.nll_loss(log_prob, types)
