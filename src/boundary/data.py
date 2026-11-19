"""Признаки стыков из B-Rep, которых нет в кэше: касание граней."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from dataset.cache import PARTS


@lru_cache(maxsize=None)
def tangent_pairs(uid: str) -> np.ndarray:
    """Пары слитых граней `(K, 2)`, `a < b`, у которых стык касательный.

    Метки в кэше — номера после слияния швов, смежность в brep.json — по
    исходным граням. Пара слитых граней касательная, если касательны все
    исходные пары между ними: хоть один острый участок — стык острый.
    """
    brep = json.loads((PARTS / uid / "brep.json").read_text())
    merge = brep["merge_map"]
    sharp: dict[tuple[int, int], bool] = {}
    for adj in brep["adjacency"]:
        a, b = sorted((merge[adj["a"]], merge[adj["b"]]))
        if a != b:
            sharp[(a, b)] = sharp.get((a, b), False) or not adj["tangential"]
    pairs = [p for p, s in sharp.items() if not s]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def tangent_mask(labels: Tensor, edges: Tensor, pairs: Tensor) -> Tensor:
    """Стыковые рёбра `(E,)`, лежащие между касательными гранями."""
    a, b = labels[edges[:, 0]], labels[edges[:, 1]]
    joint = a != b
    if pairs.numel() == 0:
        return torch.zeros_like(joint)
    size = int(max(int(labels.max()), int(pairs.max()))) + 1
    key = torch.minimum(a, b) * size + torch.maximum(a, b)
    return joint & torch.isin(key, pairs[:, 0] * size + pairs[:, 1])


def long_range_pairs(idx: list[Tensor], per_layer: int, generator: torch.Generator | None = None) -> Tensor:
    """Дальние пары `(P, 2)`, `i < j`, для отталкивающих связей Mutex Watershed.

    По `per_layer` случайных соседей на треугольник из колец 2-го и 3-го слоёв
    энкодера — те же окрестности, что видит сеть, новые на каждом шаге. Сам
    треугольник (им добиты недобравшие соседей) и повторы выброшены.
    """
    pairs = []
    for layer in idx[1:]:
        n, k = layer.shape
        pick = torch.randint(0, k, (n, per_layer), device=layer.device, generator=generator)
        j = layer.gather(1, pick).reshape(-1)
        i = torch.arange(n, device=layer.device).repeat_interleave(per_layer)
        keep = i != j
        pairs.append(torch.stack([torch.minimum(i, j), torch.maximum(i, j)], 1)[keep])
    return torch.unique(torch.cat(pairs), dim=0)
