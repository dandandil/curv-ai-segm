"""Mean-shift ParSeNet: `src/mean_shift.py` и `src/guard.py` авторов.

Дословно, кроме устройства (берётся у тензора, а не жёстко cuda) и случайной
выборки для ширины ядра (через генератор torch, а не `np.random`).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def guard_exp(x: Tensor, max_value: float = 75, min_value: float = -75) -> Tensor:
    return torch.exp(torch.clamp(x, max=max_value, min=min_value))


def guard_sqrt(x: Tensor, minimum: float = 1e-5) -> Tensor:
    return torch.sqrt(torch.clamp(x, min=minimum))


def compute_bandwidth(x: Tensor, num_samples: int, quantile: float, generator=None) -> Tensor:
    perm = torch.randperm(x.shape[0], device=x.device, generator=generator)
    x = x[perm[:num_samples]]
    dist = 2 - 2 * x @ x.t()
    k = int(quantile * min(num_samples, x.shape[0]))
    top_k = torch.topk(dist, k=max(k, 1), dim=1, largest=False)[0]
    return torch.mean(guard_sqrt(top_k[:, -1], 1e-6))


def shift(x: Tensor, b: Tensor, iterations: int) -> Tensor:
    new_x = x.clone()
    for _ in range(iterations):
        dist = 2.0 - 2.0 * new_x @ x.t()
        k = guard_exp(-dist / (b ** 2) / 2)
        d = 1 / torch.sum(k, 1, keepdim=True)
        new_x = new_x + ((k @ x) * d - new_x)
        new_x = new_x / torch.norm(new_x, dim=1, p=2, keepdim=True)
    return new_x


def nms(centers: Tensor, x: Tensor, b: Tensor) -> Tensor:
    membership = torch.min(2.0 - 2.0 * centers @ x.t(), 0)[1]
    uniques, counts = np.unique(membership.cpu().numpy(), return_counts=True)
    num_mem = torch.zeros(x.shape[0], device=x.device)
    num_mem[torch.from_numpy(uniques).to(x.device)] = torch.from_numpy(counts.astype(np.float32)).to(x.device)
    neighbors = ((2.0 - 2.0 * centers @ centers.t()) < b).float()
    ids = torch.unique(torch.max(neighbors[torch.from_numpy(uniques).to(x.device)] * num_mem.reshape(1, -1), 1)[1])
    return torch.max(centers[ids] @ x.t(), 0)[1]


def guard_mean_shift(embedding: Tensor, quantile: float, iterations: int, num_samples: int,
                     floor: float, max_clusters: int, generator=None) -> Tensor:
    """Метка на точку. Кластеров больше max_clusters — квантиль удваивается."""
    while True:
        with torch.no_grad():
            b = torch.clamp(compute_bandwidth(embedding, num_samples, quantile, generator), min=floor)
            labels = nms(shift(embedding, b, iterations), embedding, b)
        if torch.unique(labels).shape[0] > max_clusters:
            quantile *= 2
        else:
            return labels
