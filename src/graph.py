"""Соседство по рёбрам меша: разрежённые кольца вокруг каждого треугольника.

ParSeNet ищет соседей динамическим kNN в пространстве признаков, мы — обходом
по смежности треугольников (README, «Соседство: 40 соседей по рёбрам меша»).
У мешевого графа рецептивное поле растёт линейно по слоям, поэтому дальние
слои берут соседей не подряд, а из прорежённых колец: соседей столько же, а
собраны они из более широкого набора.

Кольца считаются обходом в ширину сразу от всех треугольников, на устройстве
модели. Кольцо `r` получается из кольца `r-1` вычитанием колец `r-1` и `r-2`:
сосед грани из кольца `r-1` лежит ровно в одном из этих трёх.

Фронт усекается до `cap` элементов на треугольник. Без усечения он растёт с
номером кольца, и на дальних слоях память ушла бы в гигабайты. Цена — дальнее
кольцо становится выборкой, а не полным множеством, и вычитание предыдущих
колец перестаёт быть точным: элемент кольца `r-1`, не попавший в усечённый
фронт, может пройти как элемент кольца `r`. Из кольца всё равно берётся малая
доля соседей, так что на состав окрестности это влияет слабо.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Соседей у треугольника не больше трёх — по числу рёбер.
MAX_ADJACENT = 3


def face_neighbors(adjacency: np.ndarray, n_faces: int, device) -> Tensor:
    """Смежность треугольников как таблица `(n_faces, 3)`, недостающие -1.

    Граничные и вырожденные треугольники имеют меньше трёх соседей, поэтому
    таблица дополняется -1, а не замыкается на себя: пустое место должно быть
    отличимо, иначе обход считал бы его настоящим соседом.
    """
    a = torch.as_tensor(np.array(adjacency), dtype=torch.long, device=device)
    src = torch.cat([a[:, 0], a[:, 1]])
    dst = torch.cat([a[:, 1], a[:, 0]])
    order = torch.argsort(src, stable=True)
    src, dst = src[order], dst[order]

    counts = torch.bincount(src, minlength=n_faces)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(src.numel(), device=device) - starts[src]

    neigh = torch.full((n_faces, MAX_ADJACENT), -1, dtype=torch.long, device=device)
    keep = slot < MAX_ADJACENT
    neigh[src[keep], slot[keep]] = dst[keep]
    return neigh


# Порог числа элементов фронта (треугольники × cap × соседи по ребру), выше которого обход
# колец идёт порциями по CHUNK_ELEMENTS: у конфига с кольцами до 84 (cap 288) пик памяти на
# 60 тысячах треугольников был 4.6 ГиБ. Конфиги с cap 96 под порог не попадают.
CHUNK_ABOVE = 20_000_000
CHUNK_ELEMENTS = 10_000_000


def ring_neighbors(neigh: Tensor, rings, k, cap: int, per_ring: int,
                   generator: torch.Generator | None = None) -> list[Tensor]:
    """По таблице соседей — по одному набору соседей `(n_faces, k)` на слой.

    `rings` — по тройке `(первое, последнее, шаг)` на слой: слой берёт кольца
    `первое, первое+шаг, ...` не дальше `последнего`. Шаг больше единицы и есть
    прорежение.

    `k` — число соседей: одно на все слои или своё на каждый. Узкому первому
    слою нужно меньше соседей: в кольцах 1-2 их всего около двенадцати, и
    требование сорока добирало бы повторами.

    Место под фронт выделяется по размеру кольца — `per_ring * ring`, но не
    больше `cap`. Постоянный потолок на всех кольцах резервировал под кольцо 1,
    где элементов три, столько же места, сколько под кольцо 25, и обход из-за
    этого стоил вдвое дороже.
    """
    n = neigh.shape[0]
    rows = torch.arange(n, device=neigh.device)
    # Широкий фронт (cap в сотни) на всех треугольниках сразу занимает гигабайты: строки
    # обхода независимы, поэтому большой обход идёт порциями. Ниже порога — как было,
    # одним куском, с той же последовательностью случайных выборок.
    if n * cap * MAX_ADJACENT > CHUNK_ABOVE:
        step = max(1, CHUNK_ELEMENTS // (cap * MAX_ADJACENT))
        parts = [_ring_rows(neigh, rows[s:s + step], rings, k, cap, per_ring, generator) for s in range(0, n, step)]
        return [torch.cat(layer, 0) for layer in zip(*parts)]
    return _ring_rows(neigh, rows, rings, k, cap, per_ring, generator)


def _ring_rows(neigh: Tensor, self_idx: Tensor, rings, k, cap: int, per_ring: int,
               generator: torch.Generator | None) -> list[Tensor]:
    """`ring_neighbors` для строк `self_idx`."""
    n = self_idx.shape[0]
    prev = cur = self_idx[:, None]
    pools: list[list[Tensor]] = [[] for _ in rings]

    # Дальше последнего кольца, которое кто-то берёт, обход не идёт: при шаге
    # прорежения верхняя граница диапазона может быть недостижимой.
    last_used = max(first + (last - first) // step * step for first, last, step in rings)
    for ring in range(1, last_used + 1):
        room = min(cap, max(MAX_ADJACENT, per_ring * ring))
        cand = neigh[cur.clamp(min=0)].masked_fill((cur < 0)[:, :, None], -1)
        cand = _drop(_drop(cand.reshape(n, -1), cur), prev)
        prev, cur = cur, _sample(_dedup(cand), room, generator)
        for pool, (first, last, step) in zip(pools, rings):
            if first <= ring <= last and (ring - first) % step == 0:
                pool.append(cur)

    # Недобравшим соседей ставится собственный индекс: max по соседям от
    # повтора не меняется, а -1 индексировать нечем.
    ks = [k] * len(rings) if isinstance(k, int) else list(k)
    return [_sample(torch.cat(p, 1), kk, generator, fill=self_idx) for p, kk in zip(pools, ks)]


def _drop(cand: Tensor, ref: Tensor) -> Tensor:
    """Убрать из `cand` то, что уже есть в `ref` (по строкам)."""
    ref = ref.sort(dim=1).values
    pos = torch.searchsorted(ref, cand.clamp(min=0)).clamp(max=ref.shape[1] - 1)
    return cand.masked_fill(ref.gather(1, pos) == cand, -1)


def _dedup(x: Tensor) -> Tensor:
    """Повторы внутри строки — в -1. Один и тот же сосед достижим по нескольким
    рёбрам, и без этого он занял бы несколько мест в окрестности."""
    x = x.sort(dim=1, descending=True).values
    dup = torch.zeros_like(x, dtype=torch.bool)
    dup[:, 1:] = x[:, 1:] == x[:, :-1]
    return x.masked_fill(dup & (x >= 0), -1)


def _sample(x: Tensor, m: int, generator, fill: Tensor | None = None) -> Tensor:
    """Случайные `m` заполненных мест из каждой строки."""
    key = torch.rand(x.shape, device=x.device, generator=generator)
    key = key.masked_fill(x < 0, 2.0)
    out = x.gather(1, key.topk(min(m, x.shape[1]), dim=1, largest=False).indices)
    if out.shape[1] < m:
        pad = out.new_full((x.shape[0], m - out.shape[1]), -1)
        out = torch.cat([out, pad], 1)
    if fill is not None:
        out = torch.where(out < 0, fill[:, None].expand_as(out), out)
    return out
