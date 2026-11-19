"""Нарезка по графу меша: водораздел по силе границы и слияние регионов.

Сила границы на ребре — `1 − P(одна грань)` от головы границ.

1. Водораздел: регионы — бассейны поля силы границы глубже `depth`. Каждый
   треугольник получает метку, разрез замкнут по построению.
2. Агломерация: соседние регионы сливаются по средней силе границы вдоль всей
   общей границы, пока она ниже `merge`. Вето — регионы с далёкими средними
   эмбеддингами не сливаются даже при слабой границе.
3. Регионы мельче `min_region` — соседу с самой длинной общей границей.

Вместо 1-2 (`method: multicut`) — multicut: решается, какие рёбра меша граница,
а граф границ не имеет висячих вершин по построению. Грани разделяются по сумме
уверенности вдоль всего стыка, а не по самому слабому ребру, как перевал
водораздела (README, «Нарезка multicut»).
"""
from __future__ import annotations

import heapq

import numpy as np

from src.fitting import absorb_small

from .config import PartitionCfg

PROB_EPS = 1e-6             # сила границы обрезается от 0 и 1, чтобы логарифм шансов был конечен


def _find(parent: list[int], i: int) -> int:
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def watershed(n: int, edges: np.ndarray, strength: np.ndarray, depth: float) -> np.ndarray:
    """Регионы — бассейны поля силы границы, залитые снизу вверх.

    Рёбра проходятся от слабых к сильным; на перевале между двумя разными
    бассейнами они остаются раздельными, если он поднялся над дном каждого не
    меньше чем на `depth`, и сливаются иначе. Абсолютного порога «здесь
    заведомо внутренность грани» нет: узкое скругление, где поле нигде не
    опускается низко, всё равно получает свой регион, лишь бы внутри него было
    ниже, чем на стыке.
    """
    floor = np.full(n, np.inf)
    np.minimum.at(floor, edges[:, 0], strength)
    np.minimum.at(floor, edges[:, 1], strength)
    parent, bottom = list(range(n)), floor.tolist()
    for k in np.argsort(strength, kind="stable"):
        a, b = _find(parent, int(edges[k, 0])), _find(parent, int(edges[k, 1]))
        if a == b or strength[k] - max(bottom[a], bottom[b]) >= depth:
            continue
        parent[b] = a
        bottom[a] = min(bottom[a], bottom[b])
    roots = np.array([_find(parent, i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def mutex_watershed(n: int, edges: np.ndarray, w_attr: np.ndarray, pairs: np.ndarray,
                    w_rep: np.ndarray) -> np.ndarray:
    """Mutex Watershed (Wolf et al., TPAMI 2020, Algorithm 2, connect_all = False).

    Притягивающие связи — рёбра меша с весом P(одна грань), отталкивающие — дальние
    пары с весом P(разные). Все связи по убыванию веса: притягивающая сливает два
    кластера, если между ними нет запрета; отталкивающая ставит запрет, если они
    ещё не слиты. Порогов нет: число регионов задают сами веса.
    """
    links = np.concatenate([edges, pairs]).astype(np.int64)
    weight = np.concatenate([w_attr, w_rep])
    attract = np.r_[np.ones(len(edges), bool), np.zeros(len(pairs), bool)]
    parent = list(range(n))
    mutex: list[set[int]] = [set() for _ in range(n)]
    for k in np.argsort(-weight, kind="stable"):
        a, b = _find(parent, int(links[k, 0])), _find(parent, int(links[k, 1]))
        if a == b:
            continue
        if not attract[k]:
            mutex[a].add(b)
            mutex[b].add(a)
        elif b not in mutex[a]:
            if len(mutex[a]) < len(mutex[b]):
                a, b = b, a
            parent[b] = a
            for c in mutex[b]:                      # запреты поглощённого кластера — новому корню
                mutex[c].discard(b)
                mutex[c].add(a)
            mutex[a] |= mutex[b]
            mutex[b] = set()
    roots = np.array([_find(parent, i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def multicut(n: int, edges: np.ndarray, strength: np.ndarray, beta: float,
             embedding: np.ndarray | None, veto: float | None) -> np.ndarray:
    """Multicut жадным стягиванием рёбер, GAEC (Keuper et al., ICCV 2015).

    Вес ребра — логарифм шансов «одна грань» со сдвигом `beta` (Beier et al.,
    ICCV 2015): `log((1 − s)/s) + log((1 − β)/β)`, положительный тянет
    треугольники в одну грань. Стягивается пара регионов с наибольшим
    суммарным весом общей границы, пока он положителен. Вето — как в
    `agglomerate`: регионы с далёкими средними эмбеддингами не стягиваются.
    """
    s = np.clip(strength, PROB_EPS, 1 - PROB_EPS)
    w = np.log((1 - s) / s) + np.log((1 - beta) / beta)
    nbr: list[dict[int, float]] = [{} for _ in range(n)]
    for (a, b), x in zip(edges.tolist(), w.tolist()):
        if a != b:
            nbr[a][b] = nbr[b][a] = nbr[a].get(b, 0.0) + x
    heap = [(-x, a, b) for a in range(n) for b, x in nbr[a].items() if a < b and x > 0]
    heapq.heapify(heap)
    sums = None
    if embedding is not None and veto is not None:
        sums = np.array(embedding, dtype=np.float64)
    parent = list(range(n))
    while heap:
        x, a, b = heapq.heappop(heap)
        if nbr[a].get(b) != -x:                           # запись устарела
            continue
        if sums is not None:
            cos = sums[a] @ sums[b] / (np.linalg.norm(sums[a]) * np.linalg.norm(sums[b]) + 1e-12)
            if 1.0 - cos > veto:
                continue
        if len(nbr[a]) < len(nbr[b]):                     # меньший список соседей вливается в больший
            a, b = b, a
        del nbr[a][b], nbr[b][a]
        for c, y in nbr[b].items():
            del nbr[c][b]
            v = nbr[a][c] = nbr[c][a] = nbr[a].get(c, 0.0) + y
            if v > 0:
                heapq.heappush(heap, (-v, min(a, c), max(a, c)))
        nbr[b] = {}
        if sums is not None:
            sums[a] += sums[b]
        parent[b] = a
    roots = np.array([_find(parent, i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def agglomerate(labels: np.ndarray, edges: np.ndarray, strength: np.ndarray, merge: float,
                embedding: np.ndarray | None, veto: float | None) -> np.ndarray:
    """Жадное слияние по средней силе общей границы (правило average из GASP)."""
    r = int(labels.max()) + 1
    la, lb = labels[edges[:, 0]], labels[edges[:, 1]]
    cross = la != lb
    lo, hi = np.minimum(la, lb)[cross], np.maximum(la, lb)[cross]
    keys, inverse = np.unique(lo.astype(np.int64) * r + hi, return_inverse=True)
    total = np.bincount(inverse, weights=strength[cross])
    count = np.bincount(inverse).astype(float)
    # Одна запись [сумма, число рёбер] на пару, общая для обеих сторон.
    border: list[dict[int, list[float]]] = [{} for _ in range(r)]
    for key, s, c in zip(keys, total, count):
        a, b = divmod(int(key), r)
        border[a][b] = border[b][a] = [s, c]

    sums = None
    if embedding is not None and veto is not None:
        sums = np.zeros((r, embedding.shape[1]))
        np.add.at(sums, labels, embedding)

    heap = [(rec[0] / rec[1], a, b) for a in range(r) for b, rec in border[a].items() if a < b]
    heapq.heapify(heap)
    parent = list(range(r))
    while heap:
        mean, a, b = heapq.heappop(heap)
        if mean >= merge:
            break
        rec = border[a].get(b)
        if rec is None or rec[0] / rec[1] != mean:        # запись устарела
            continue
        if sums is not None:
            cos = sums[a] @ sums[b] / (np.linalg.norm(sums[a]) * np.linalg.norm(sums[b]) + 1e-12)
            if 1.0 - cos > veto:
                continue
        del border[a][b], border[b][a]
        for c, (s, k) in list(border[b].items()):
            del border[c][b]
            joined = border[a].get(c)
            if joined is None:
                joined = border[a][c] = border[c][a] = [0.0, 0.0]
            joined[0] += s
            joined[1] += k
            heapq.heappush(heap, (joined[0] / joined[1], min(a, c), max(a, c)))
        border[b] = {}
        if sums is not None:
            sums[a] += sums[b]
        parent[b] = a
    roots = np.array([_find(parent, i) for i in range(r)])
    return np.unique(roots[labels], return_inverse=True)[1]


def partition(n: int, edges: np.ndarray, strength: np.ndarray, cfg: PartitionCfg,
              embedding: np.ndarray | None = None) -> np.ndarray:
    """Метка региона на треугольник. `embedding` — нормированный `(N, D)`; без
    него вето не применяется."""
    edges = np.asarray(edges, dtype=np.int64)
    if cfg.method == "multicut":
        labels = multicut(n, edges, strength, cfg.beta, embedding, cfg.veto)
    elif cfg.method == "watershed":
        labels = watershed(n, edges, strength, cfg.depth)
        labels = agglomerate(labels, edges, strength, cfg.merge, embedding, cfg.veto)
    else:
        raise ValueError(f"нарезка {cfg.method!r}: есть watershed и multicut")
    labels = absorb_small(labels, edges, cfg.min_region)
    return np.unique(labels, return_inverse=True)[1]
