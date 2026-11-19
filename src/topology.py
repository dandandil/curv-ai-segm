"""Границы кластеров на меше — каркас будущего B-Rep.

Граница между двумя кластерами разбивается на цепочки рёбер меша. Цепочка
кончается в узле: вершине, где сходятся три кластера и больше, или где
граница пары обрывается. Цепочка без узлов — замкнутая петля. Узел станет
вершиной B-Rep, цепочка — ребром, пара кластеров цепочки — пересекаемыми
поверхностями.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Chain:
    pair: tuple[int, int]
    vertices: np.ndarray             # вершины меша по порядку
    ends: tuple[int, int] | None     # узлы на концах; None — замкнутая петля


@dataclass
class Boundaries:
    chains: list[Chain]
    nodes: list[np.ndarray]          # вершины меша, стянутые в узел
    node_labels: list[frozenset]


def _find(parent: dict, v: int) -> int:
    while parent[v] != v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v


def boundaries(mesh: trimesh.Trimesh, labels: np.ndarray, min_chain_edges: int) -> Boundaries:
    """Цепочки и узлы границ кластеров.

    Цепочки короче `min_chain_edges` рёбер стягиваются: на нарезке сети стык
    трёх граней часто выглядит как два узла через ребро-другое, и без
    стягивания из них вышло бы ребро B-Rep нулевой длины.
    """
    adj, edges = mesh.face_adjacency, mesh.face_adjacency_edges
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    cut = la != lb
    graphs = defaultdict(lambda: defaultdict(set))
    for a, b, u, v in zip(np.minimum(la, lb)[cut], np.maximum(la, lb)[cut], *edges[cut].T):
        graphs[(int(a), int(b))][int(u)].add(int(v))
        graphs[(int(a), int(b))][int(v)].add(int(u))

    vertex_faces = mesh.vertex_faces
    around = {}
    for g in graphs.values():
        for v in g:
            if v not in around:
                f = vertex_faces[v]
                around[v] = frozenset(labels[f[f >= 0]].tolist())

    raw = []
    for pair, g in graphs.items():
        # Вершина с двумя метками и чётной степенью — место, где граница тонкой
        # грани касается сама себя; через неё проходим насквозь. Резать там
        # значило дробить одну петлю на десятки кусков [измерено: 32566_7bfc936b_5
        # по разметке — 218 участков на 19 граней, 56 из 95 узлов такие].
        def is_node(v):
            return len(around[v]) >= 3 or len(g[v]) % 2 == 1
        used = set()

        def walk(start, nxt):
            path, prev, cur = [start], start, nxt
            used.add(frozenset((start, nxt)))
            while True:
                path.append(cur)
                if is_node(cur) or cur == start:
                    return path
                step = [x for x in g[cur] if frozenset((cur, x)) not in used]
                if not step:
                    return path
                prev, cur = cur, step[0]
                used.add(frozenset((prev, cur)))

        for s in [v for v in g if is_node(v)]:
            for nxt in g[s]:
                if frozenset((s, nxt)) not in used:
                    raw.append((pair, walk(s, nxt), False))
        for s in g:
            for nxt in g[s]:
                if frozenset((s, nxt)) not in used:
                    raw.append((pair, walk(s, nxt)[:-1], True))

    parent = {}
    for pair, path, closed in raw:
        if not closed:
            for v in (path[0], path[-1]):
                parent.setdefault(v, v)
    kept = []
    for pair, path, closed in raw:
        if len(path) - 1 < min_chain_edges:
            if not closed:
                parent[_find(parent, path[0])] = _find(parent, path[-1])
            continue
        kept.append((pair, path, closed))

    roots = sorted({_find(parent, v) for v in parent})
    index = {r: k for k, r in enumerate(roots)}
    members = defaultdict(list)
    for v in parent:
        members[index[_find(parent, v)]].append(v)
    nodes = [np.array(members[k]) for k in range(len(roots))]
    node_labels = [frozenset().union(*(around[v] for v in m)) for m in nodes]

    chains = [Chain(pair, np.array(path),
                    None if closed else (index[_find(parent, path[0])], index[_find(parent, path[-1])]))
              for pair, path, closed in kept]
    return Boundaries(chains, nodes, node_labels)


def drop_chainless(mesh: trimesh.Trimesh, labels: np.ndarray, min_chain_edges: int):
    """Кластеры без единого участка границы отдать соседу, пока такие есть.

    У узкой грани шириной в ребро-два все участки границы бывают короче
    `min_chain_edges`, и `boundaries` стягивает их в узлы: у грани не остаётся
    рёбер, построить её нельзя, и на её месте дыра в оболочке [22228_1c82530b_5
    по разметке: грань в 38 треугольников, остальные 48 рёбер построены]. Без
    неё соседние поверхности пересекаются между собой и место закрывают.
    Сосед — с самой длинной общей границей. Возвращает метки, границы и
    отданные кластеры.
    """
    adj = mesh.face_adjacency
    labels = labels.copy()
    dropped = []
    while True:
        bounds = boundaries(mesh, labels, min_chain_edges)
        have = {l for ch in bounds.chains for l in ch.pair}
        lonely = [int(l) for l in np.unique(labels) if l not in have]
        if len(np.unique(labels)) <= 1 or not lonely:
            return labels, bounds, dropped
        moved = False
        for l in lonely:
            la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
            nb = np.r_[lb[(la == l) & (lb != l)], la[(lb == l) & (la != l)]]
            if len(nb):
                labels[labels == l] = np.bincount(nb).argmax()
                dropped.append(l)
                moved = True
        if not moved:
            return labels, bounds, dropped


def touching(mesh: trimesh.Trimesh, labels: np.ndarray) -> set[tuple[int, int]]:
    """Пары кластеров с общим ребром меша — соседи для соотношений. Цепочки
    короче `min_chain_edges` стягиваются в узлы, и пара пропадает из цепочек:
    на тонкой пластине граница стенки со скруглением идёт поперёк толщины в 2-4
    ребра, касание стенки и скругления не находилось, их линии на пластине
    расходились на микроны, и контур разреза не замыкался [117695_0901cb06_0
    по разметке: 4 касательные пары из 6, пластина 0.27 мм, ребро 0.07 мм]."""
    la, lb = labels[mesh.face_adjacency[:, 0]], labels[mesh.face_adjacency[:, 1]]
    cut = la != lb
    return {(int(min(a, b)), int(max(a, b))) for a, b in zip(la[cut], lb[cut])}


# Связный кусок рабочего меша мельче этой доли треугольников — осколок ремеша, а
# не часть тела. [22228_1c82530b_5: куски из 36 и 2 треугольников, 0.83 x 0.015
# мм, с отрицательным объёмом, на тонком месте детали; кластеры на них не
# получают ни соседей, ни рёбер]
MIN_COMPONENT_SHARE = 0.01


def main_parts(mesh: trimesh.Trimesh, *per_face: np.ndarray):
    """Меш без осколков и те же массивы на треугольник, обрезанные под него."""
    parts = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    big = [p for p in parts if len(p) >= MIN_COMPONENT_SHARE * len(mesh.faces)]
    # Меш из одних мелких кусков (больше сотни равных) — осколков не отличить, берётся как есть.
    if not big:
        return (mesh, *per_face)
    keep = np.concatenate(big)
    if len(keep) == len(mesh.faces):
        return (mesh, *per_face)
    keep = np.sort(keep)
    return (mesh.submesh([keep], append=True), *(a[keep] for a in per_face))
