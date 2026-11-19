"""Переподгонка с перенарезкой (README, «Переподгонка с перенарезкой»).

Кластеры нарезки обходятся от меньших к большим. Примитив типа, за который голосует
сеть, оценивается по пятну вокруг самой глубокой точки кластера, и от неё растёт
область, как подгонка по клику в ScanRuler (github.com/CNCKitchen/scanruler,
`src/core/fit/regionGrow.ts`; схема и числа оттуда, код свой): треугольник берётся,
если он в полосе шума у примитива и его нормаль согласна с примитивом, примитив
переподгоняется каждый круг с отсечением выбросов, переходная полоса скруглённого
края в конце срезается. Выросшее закрепляется и бо́льшим кластерам уже не достаётся,
поэтому мелкие грани не съедаются, лишние разрезы дорастают до целой грани, а у
слипшегося кластера примитив забирает только свою часть. Не доставшиеся никому
связные куски с телом проходят тот же путь ещё раз; остальное (полосы вдоль
скруглённых сканом рёбер, куски, не подогнанные и со второго раза) раздаётся
соседним областям — ближайшему примитиву.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import fitting as FT
from . import surfaces as S
from .model import PRIMITIVE_TYPES

# Пятна вокруг точки старта по очереди, в треугольниках: следующее — если на предыдущем
# примитив не подогнался (мало или слишком плоско). У ScanRuler 1500/6000/24000 вершин;
# треугольников в замкнутом меше вдвое больше вершин.
SEED_PATCHES = (3000, 12000, 48000)
CLIP_SIGMAS = 3.0           # переподгонка без точек дальше стольких σ
CLIP_ROUNDS = 3
BAND_SIGMAS = 3.5           # полоса роста — столько σ, не меньше пола
PLANE_FLOOR_REL = 2e-4      # пол полосы плоскости — доля диагонали детали
ROUND_FLOOR_REL = 0.004     # пол полосы круглых — доля радиуса (тора — трубки), у конуса пола нет
RADIUS_PARAM = {"sphere": 3, "cylinder": 6, "torus": 7}
# Нормаль треугольника и примитива расходятся не больше: у круглых шум нормалей больше.
NORMAL_DEG = {"plane": 25.0, "cylinder": 32.0, "cone": 32.0, "torus": 32.0, "sphere": 38.0}
GROW_ROUNDS = 30
GROW_CAP = 3                # за круг область вырастает не больше чем во столько раз
GROW_CAP_MIN = 6000
GROW_SETTLED = 0.002        # рост остановился: прирост меньше этой доли (и не больше 3 треугольников)
GROW_MIN_ROUNDS = 3
FIT_SUBSAMPLE = 60000       # переподгонка по прореженным точкам, область — вся
MIN_REGION_TRIANGLES = 60
# Срезание переходной полосы: кольцо у края области срезается, пока медианный наклон его
# нормалей к примитиву больше вдвое медианного по области (и хотя бы на 0.5°), не больше
# 6 колец и не меньше половины области.
PEEL_TILT_FACTOR = 2.0
PEEL_TILT_FLOOR_DEG = 0.5
PEEL_MAX_RINGS = 6
# Кусок, не доставшийся никому, подгоняется второй раз, если в нём не меньше `min_region`
# треугольников дальше стольких колец от взятого: у половины слипшегося кластера тело есть,
# у полосы вдоль скруглённого ребра (ширина 1-3 треугольника) — нет. Выбрано.
BAND_RINGS = 2
NO_SURFACE_COST = 1e9       # область без примитива берёт ничьё, только если рядом больше никого


def _neighbors(adjacency: np.ndarray, n: int) -> csr_matrix:
    a, b = adjacency[:, 0], adjacency[:, 1]
    return csr_matrix((np.ones(2 * len(a), bool), (np.r_[a, b], np.r_[b, a])), shape=(n, n))


class _Mesh:
    def __init__(self, vertices, faces, adjacency, cfg):
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces)
        self.adjacency = np.asarray(adjacency, np.int64)
        self.centre, self.normal, self.area = FT.triangle_geometry(self.vertices, self.faces)
        self.graph = _neighbors(self.adjacency, len(self.faces))
        self.cfg = cfg
        self.tol = FT.rms_tolerance(self.area, cfg)
        self.diag = float(np.linalg.norm(np.ptp(self.vertices, axis=0)))

    def points(self, tri: np.ndarray):
        """Точки, нормали и веса вершин `tri` без полосы у края, как у подгонки кластеров."""
        lab = np.full(len(self.faces), -1)
        lab[tri] = 0
        excluded = FT.boundary_vertices(self.faces, self.adjacency, lab, self.cfg.boundary_rings,
                                        len(self.vertices))
        return FT.cluster_points(0, lab, excluded, self.vertices, self.faces, self.normal, self.area,
                                 self.cfg.min_fit_points)

    def band(self, surface: S.Surface, sigma: float) -> float:
        if surface.kind == "plane":
            floor = PLANE_FLOOR_REL * self.diag
        elif surface.kind in RADIUS_PARAM:
            floor = ROUND_FLOOR_REL * abs(float(surface.params[RADIUS_PARAM[surface.kind]]))
        else:
            floor = 0.0
        return max(BAND_SIGMAS * sigma, floor)

    def tilt_cos(self, surface: S.Surface, tri: np.ndarray) -> np.ndarray:
        """|cos| между нормалью треугольника и примитива у его центра."""
        g = S.normals(surface.kind, surface.params, self.centre[tri])
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
        return np.abs((g * self.normal[tri]).sum(1))

    def accept(self, surface: S.Surface, tri: np.ndarray, band: float) -> np.ndarray:
        """Треугольники `tri` на примитиве: среднее знаковое расстояние вершин в полосе, нормаль согласна."""
        d = FT.distance(surface.kind, surface.params, self.vertices[self.faces[tri]].reshape(-1, 3))
        near = np.abs(d.reshape(-1, 3).mean(1)) <= band
        return near & (self.tilt_cos(surface, tri) >= np.cos(np.radians(NORMAL_DEG[surface.kind])))


def _fit_clipped(kind: str | None, pts, nrm, w, fixed: bool) -> tuple[S.Surface, float] | None:
    """Примитив с отсечением точек дальше `CLIP_SIGMAS` σ и σ оставшихся. `fixed` — только тип
    `kind`; иначе `kind` — голос сети, и при неудаче тип выбирает `fit_cluster` перебором."""
    surface = FT.fit_kind(kind, pts, nrm, w) if fixed else FT.fit_cluster(pts, nrm, w, kind)
    if surface is None or not np.isfinite(surface.rms):
        return None
    sigma = surface.rms
    for _ in range(CLIP_ROUNDS):
        keep = np.abs(FT.distance(surface.kind, surface.params, pts)) <= CLIP_SIGMAS * sigma
        if keep.all() or keep.sum() < 7:
            break
        refit = FT.fit_kind(surface.kind, pts[keep], nrm[keep], w[keep])
        if refit is None or not np.isfinite(refit.rms):
            break
        surface, sigma = refit, refit.rms
    return surface, sigma


def _refit(mesh: _Mesh, kind: str, region: np.ndarray) -> tuple[S.Surface, float] | None:
    pts, nrm, w = mesh.points(region)
    if len(pts) > FIT_SUBSAMPLE:
        idx = np.linspace(0, len(pts) - 1, FIT_SUBSAMPLE).astype(int)
        pts, nrm, w = pts[idx], nrm[idx], w[idx]
    return _fit_clipped(kind, pts, nrm, w, fixed=True)


def _patch(mesh: _Mesh, seed: int, inside: np.ndarray, limit: int) -> np.ndarray:
    """До `limit` треугольников вокруг `seed` обходом в ширину по треугольникам `inside`."""
    taken = np.zeros(len(inside), bool)
    taken[seed] = True
    front, count = np.array([seed]), 1
    while len(front) and count < limit:
        cand = np.unique(mesh.graph[front].indices)
        front = cand[inside[cand] & ~taken[cand]][:limit - count]
        taken[front] = True
        count += len(front)
    return np.flatnonzero(taken)


def _deepest(mesh: _Mesh, tri: np.ndarray) -> int:
    """Треугольник кластера, дальше всех по кольцам от его границы: эрозия по подграфу кластера."""
    sub = mesh.graph[tri][:, tri]
    left = np.asarray(sub.sum(1)).ravel() == np.asarray(mesh.graph[tri].sum(1)).ravel()   # все соседи свои
    last = np.arange(len(tri))
    while left.any():
        last = np.flatnonzero(left)
        eroded = left & ~((sub @ (~left).astype(np.int32)) > 0)
        if eroded.sum() == len(last):
            break                               # границы нет (островок целиком один кластер)
        left = eroded
    return int(tri[last[0]])


def _grow(mesh: _Mesh, seed: int, free: np.ndarray, surface: S.Surface, sigma: float, count: int):
    """Рост от `seed` с переподгонкой каждый круг; область, примитив и σ, None — не вырос."""
    prev = max(count, 50)
    region = None
    for round_ in range(GROW_ROUNDS):
        band = mesh.band(surface, sigma)
        cap = max(GROW_CAP_MIN, GROW_CAP * prev)
        taken = np.zeros(len(free), bool)
        taken[seed] = True
        front, size = np.array([seed]), 1
        while len(front) and size < cap:
            cand = np.unique(mesh.graph[front].indices)
            cand = cand[free[cand] & ~taken[cand]]
            front = cand[mesh.accept(surface, cand, band)][:cap - size] if len(cand) else cand
            taken[front] = True
            size += len(front)
        region = np.flatnonzero(taken)
        if len(region) < 10:
            return None
        fit = _refit(mesh, surface.kind, region)
        if fit is None:
            return None
        surface, sigma = fit
        growth = abs(len(region) - prev)
        if size < cap and round_ >= GROW_MIN_ROUNDS and growth <= max(3, GROW_SETTLED * prev):
            break
        prev = len(region)
    return _peel(mesh, region, surface), surface, sigma


def _peel(mesh: _Mesh, region: np.ndarray, surface: S.Surface) -> np.ndarray:
    """Срезать с края области переходную полосу скруглённого сканом ребра: её кольца ещё в
    полосе шума, но нормали отворачиваются от примитива сильнее шума на самой грани."""
    if len(region) < MIN_REGION_TRIANGLES:
        return region
    tilt = np.arccos(np.clip(mesh.tilt_cos(surface, region), 0, 1))
    noise = float(np.median(tilt))
    limit = max(PEEL_TILT_FACTOR * noise, noise + np.radians(PEEL_TILT_FLOOR_DEG))
    keep_at_least = max(MIN_REGION_TRIANGLES, len(region) // 2)
    member = np.zeros(len(mesh.faces), bool)
    member[region] = True
    tilt_of = np.zeros(len(mesh.faces))
    tilt_of[region] = tilt
    for _ in range(PEEL_MAX_RINGS):
        outside = (mesh.graph @ (~member).astype(np.int32)) > 0
        ring = np.flatnonzero(member & outside)
        if not len(ring) or member.sum() - len(ring) < keep_at_least or np.median(tilt_of[ring]) < limit:
            break
        member[ring] = False
    return np.flatnonzero(member)


def _largest_component(mesh: _Mesh, tri: np.ndarray) -> np.ndarray:
    sub = mesh.graph[tri][:, tri]
    comp = connected_components(sub, directed=False)[1]
    return tri[comp == np.bincount(comp).argmax()]


def _claim(mesh: _Mesh, tri: np.ndarray, free: np.ndarray, votes: np.ndarray) -> tuple[np.ndarray, S.Surface | None]:
    """Область кластера `tri` после переподгонки и её примитив; None — не подогнался."""
    v = votes[tri]
    v = v[v >= 0]
    vote = PRIMITIVE_TYPES[np.bincount(v).argmax()] if len(v) else None
    seed = _deepest(mesh, tri)
    inside = np.zeros(len(mesh.faces), bool)
    inside[tri] = True
    for limit in SEED_PATCHES:
        patch = _patch(mesh, seed, inside, limit)
        fit = _fit_clipped(vote, *mesh.points(patch), fixed=False)
        if fit is not None and fit[1] <= mesh.tol:
            grown = _grow(mesh, seed, free, fit[0], fit[1], len(patch))
            if grown is not None:
                region, surface, sigma = grown
                if len(region) >= min(MIN_REGION_TRIANGLES, len(tri)) and sigma <= mesh.tol:
                    return _largest_component(mesh, region), surface
        if len(patch) >= len(tri):
            break                               # больше пятна кластер не даст
    return tri, None


def _flood(mesh: _Mesh, out: np.ndarray, surfaces: dict[int, S.Surface]) -> None:
    """Ничьи треугольники — соседним областям заливкой по приоритету (seeded region growing,
    Adams & Bischof 1994): первым берётся тот, что ближе к примитиву соседней области."""
    def cost(t, label):
        s = surfaces.get(label)
        if s is None:
            return NO_SURFACE_COST
        return float(np.abs(FT.distance(s.kind, s.params, mesh.vertices[mesh.faces[t]])).max())

    free = out < 0
    heap = []
    for x, y in (mesh.adjacency.T, mesh.adjacency[:, ::-1].T):
        m = free[x] & ~free[y]
        heap += [(cost(t, l), int(t), int(l)) for t, l in zip(x[m], out[y[m]])]
    heapq.heapify(heap)
    while heap:
        _, t, label = heapq.heappop(heap)
        if out[t] >= 0:
            continue
        out[t] = label
        for u in mesh.graph[t].indices:
            if out[u] < 0:
                heapq.heappush(heap, (cost(u, label), int(u), label))


def refit_partition(vertices: np.ndarray, faces: np.ndarray, adjacency: np.ndarray, labels: np.ndarray,
                    votes: np.ndarray, cfg, min_region: int) -> tuple[np.ndarray, dict]:
    """Метки после переподгонки и сводка. `votes` — тип на треугольник индексом в
    `PRIMITIVE_TYPES`, -1 — нет голоса; `cfg` — `CFG.brep` (допуск, точки подгонки)."""
    mesh = _Mesh(vertices, faces, adjacency, cfg)
    n = len(mesh.faces)
    out = np.full(n, -1)
    surfaces, kept = {}, 0

    def run(groups, second=False):
        # во второй раз не подогнанный кусок ничего не берёт: это полоса у рёбер, на которую не
        # легла ни одна из соседних граней [val:1: такие куски, закреплённые хотя бы телом вне
        # полосы, лежали в 0-5 кольцах от истинных стыков и резали 6 граней], её раздаст заливка
        nonlocal kept
        for tri in sorted(groups, key=len):
            tri = tri[out[tri] < 0]
            if not len(tri):
                continue                        # целиком досталось меньшему кластеру
            region, surface = _claim(mesh, tri, out < 0, votes)
            if surface is None and second:
                continue
            label = int(out.max()) + 1
            out[region] = label
            if surface is None:
                kept += 1
            else:
                surfaces[label] = surface

    def rest_groups():
        rest = np.flatnonzero(out < 0)
        if not len(rest):
            return []
        comp = connected_components(mesh.graph[rest][:, rest], directed=False)[1]
        order = np.argsort(comp, kind="stable")
        return np.split(rest[order], np.cumsum(np.bincount(comp))[:-1])

    order = np.argsort(labels, kind="stable")
    run(np.split(order, np.cumsum(np.bincount(labels))[:-1]))
    # Второй раз — только куски с телом: полоса вдоль скруглённого сканом ребра не ложится ни
    # на один примитив, и подгонять её отдельно — плодить кластеры-полоски.
    band = out >= 0
    for _ in range(BAND_RINGS):
        band |= (mesh.graph @ band.astype(np.int32)) > 0
    groups = [tri for tri in rest_groups() if (~band[tri]).sum() >= min_region]
    run(groups, second=True)
    _flood(mesh, out, surfaces)
    for tri in rest_groups():                    # не касается ни одной области — как есть
        out[tri] = int(out.max()) + 1
        kept += 1
    out = FT.absorb_small(out, mesh.adjacency, min_region)
    out = np.unique(out, return_inverse=True)[1]
    return out, {"clusters_before": int(labels.max()) + 1, "clusters_after": int(out.max()) + 1,
                 "fitted": len(surfaces), "kept": kept, "leftovers": len(groups), "tolerance": mesh.tol}
