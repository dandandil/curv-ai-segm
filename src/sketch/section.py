"""Сечение плоскостью: скан-меш -> цепочки отрезков, точный STEP -> кривые и стыки
(README, «Эскиз из сечения»).

Отрезок цепочки — след одного треугольника меша; соседние отрезки делят ребро меша, его
точка пересечения с плоскостью — стык между ними. Точное сечение — `BRepAlgoAPI_Section`:
рёбра с типом кривой и гранью-предком; граница — вершина, где сменилась грань (после
слияния швов, `merge_map`), острый или касание — по углу касательных.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

CURVE_TYPES = ("line", "arc", "ellipse", "spline")
_OCC_TYPE = {GeomAbs_CurveType.GeomAbs_Line: 0, GeomAbs_CurveType.GeomAbs_Circle: 1,
             GeomAbs_CurveType.GeomAbs_Ellipse: 2}


def plane_frame(normal: np.ndarray) -> np.ndarray:
    """`(3, 3)`: строки e1, e2, n — правая система с n по нормали плоскости."""
    n = normal / np.linalg.norm(normal)
    a = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n])


@dataclass
class Chain:
    """Цепочка отрезков скана в плоскости. `points` — стыки `(K, 2)` в системе плоскости
    (у замкнутой K = числу отрезков, отрезок i — от точки i к i + 1 по кругу; у открытой
    K = отрезков + 1), `normals` — нормали треугольников в системе (e1, e2, n) `(M, 3)`,
    `tri` — номера треугольников `(M,)`. Обход — материал слева."""
    points: np.ndarray
    normals: np.ndarray
    tri: np.ndarray
    closed: bool

    @property
    def n(self) -> int:
        return len(self.tri)

    def segment_ends(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.points[:self.n]
        b = np.roll(self.points, -1, 0)[:self.n] if self.closed else self.points[1:]
        return a, b


def mesh_section(v: np.ndarray, f: np.ndarray, normals: np.ndarray, origin: np.ndarray,
                 frame: np.ndarray) -> list[Chain]:
    """Цепочки сечения меша плоскостью (origin, frame[2]). Вершины на самой плоскости сдвигаются
    на её положительную сторону, чтобы каждый пересечённый треугольник дал ровно один отрезок."""
    d = (v - origin) @ frame[2]
    scale = np.abs(d).max() + 1e-12
    d = np.where(np.abs(d) < 1e-9 * scale, 1e-9 * scale, d)
    side = d[f] > 0
    cross = np.flatnonzero(side.any(1) & ~side.all(1))
    if not len(cross):
        return []
    ft = f[cross]
    # два ребра со сменой знака у каждого треугольника, по порядку обхода треугольника
    ea = ft[:, [0, 1, 2]]
    eb = ft[:, [1, 2, 0]]
    change = side[cross][:, [0, 1, 2]] != side[cross][:, [1, 2, 0]]
    ka, kb = ea[change].reshape(-1, 2), eb[change].reshape(-1, 2)
    lo, hi = np.minimum(ka, kb), np.maximum(ka, kb)
    key = lo * len(v) + hi                                          # (T, 2) ключи рёбер
    t = (d[lo] / (d[lo] - d[hi]))[..., None]
    p3 = v[lo] + (v[hi] - v[lo]) * t                                # (T, 2, 3)
    p2 = p3 @ frame[:2].T                                           # (T, 2, 2)

    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    owners = [[] for _ in range(len(uniq))]
    for i, (a, b) in enumerate(inv):
        owners[a].append(i)
        owners[b].append(i)
    point = np.zeros((len(uniq), 2))
    point[inv[:, 0]] = p2[:, 0]
    point[inv[:, 1]] = p2[:, 1]

    used = np.zeros(len(cross), bool)

    def walk(start: int, via: int) -> tuple[list[int], list[int], bool]:
        """От треугольника `start` через ребро-точку `via` (вход) до конца или замыкания."""
        tris, pts = [], [via]
        i, enter = start, via
        while True:
            used[i] = True
            tris.append(i)
            out = inv[i, 1] if inv[i, 0] == enter else inv[i, 0]
            nxt = [j for j in owners[out] if j != i]
            if not nxt:
                pts.append(out)
                return tris, pts, False
            j = nxt[0]
            if j == start:
                return tris, pts, True
            if used[j]:                                             # немногообразное ребро
                pts.append(out)
                return tris, pts, False
            pts.append(out)
            i, enter = j, out

    chains = []
    ends = [k for k in range(len(uniq)) if len(owners[k]) == 1]     # концы открытых цепочек
    for k in ends:
        i = owners[k][0]
        if not used[i]:
            chains.append(walk(i, k))
    for i in range(len(cross)):
        if not used[i]:
            chains.append(walk(i, inv[i, 0]))

    out = []
    for tris, pts, closed in chains:
        tri = cross[np.array(tris)]
        nrm = normals[tri] @ frame.T
        c = Chain(point[np.array(pts)], nrm, tri, closed)
        a, b = c.segment_ends()
        tang = b - a
        left = np.stack([-tang[:, 1], tang[:, 0]], 1)
        # наружная нормаль смотрит вправо от обхода, если материал слева
        if (nrm[:, :2] * left).sum() > 0:
            c = Chain(c.points[::-1].copy(), nrm[::-1].copy(), tri[::-1].copy(), closed)
            if closed:                                              # отрезок i снова от точки i к i + 1
                c.points = np.roll(c.points, -1, 0)
        out.append(c)
    return out


@dataclass
class ExactEdge:
    kind: int                    # номер в CURVE_TYPES
    face: int                    # слитая грань-предок, -1 — не найдена
    samples: np.ndarray          # (S, 2) точки по кривой
    ends: np.ndarray             # (2, 2) начало и конец
    tangents: np.ndarray         # (2, 2) единичные касательные в концах, по направлению ребра


def face_index(shape) -> dict:
    out, exp, i = {}, TopExp_Explorer(shape, TopAbs_FACE), 0
    while exp.More():
        out.setdefault(TopoDS.Face_s(exp.Current()).TShape(), i)
        i += 1
        exp.Next()
    return out


def exact_section(shape, origin: np.ndarray, frame: np.ndarray, merge_map: np.ndarray | None,
                  face_index: dict, spacing: float) -> list[ExactEdge]:
    """Рёбра точного сечения; точки по кривой с шагом `spacing` (не меньше 16 на ребро).
    Эллипс с равными осями — окружность: OCC так отдаёт сечение цилиндра поперёк оси."""
    sec = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, frame[2]))), False)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        raise ValueError("сечение не построено")
    out = []
    exp = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        exp.Next()
        c = BRepAdaptor_Curve(e)
        u0, u1 = c.FirstParameter(), c.LastParameter()
        if e.Orientation() == TopAbs_REVERSED:
            u0, u1 = u1, u0
        length = GCPnts_AbscissaPoint.Length_s(c)
        us = np.linspace(u0, u1, int(np.clip(length / spacing, 16, 20000)))
        pts = np.array([[c.Value(u).X(), c.Value(u).Y(), c.Value(u).Z()] for u in us])
        tans = []
        for u in (u0, u1):
            p, vec = gp_Pnt(), gp_Vec()
            c.D1(u, p, vec)
            tans.append(np.array([vec.X(), vec.Y(), vec.Z()]) * (1 if u1 >= u0 else -1))
        tans = np.array(tans) @ frame[:2].T
        tans /= np.maximum(np.linalg.norm(tans, axis=1, keepdims=True), 1e-12)
        kind = _OCC_TYPE.get(c.GetType(), 3)
        if kind == 2:
            el = c.Ellipse()
            if el.MajorRadius() - el.MinorRadius() < 1e-6 * el.MajorRadius():
                kind = 1
        anc = TopoDS_Shape()
        face = -1
        if sec.HasAncestorFaceOn1(e, anc):
            fi = face_index.get(anc.TShape(), -1)
            face = int(merge_map[fi]) if (merge_map is not None and fi >= 0) else fi
        p2 = pts @ frame[:2].T
        out.append(ExactEdge(kind, face, p2, p2[[0, -1]], tans))
    return out


@dataclass
class ExactVertex:
    point: np.ndarray            # (2,)
    boundary: bool               # сменилась грань или тип кривой
    tangent: bool                # угол касательных меньше порога
    degree: int


def _from_vertex(samples: np.ndarray, at_end: bool, step: float) -> np.ndarray:
    """Точки ребра на длине 0, step, 2·step от его конца (интерполяция по дуге)."""
    pts = samples[::-1] if at_end else samples
    acc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.minimum([0, step, 2 * step], acc[-1])
    return np.c_[np.interp(t, acc, pts[:, 0]), np.interp(t, acc, pts[:, 1])]


def _curvature(p: np.ndarray) -> float:
    """Кривизна со знаком по трём точкам (знак — поворот при обходе p0 -> p2)."""
    a, b = p[1] - p[0], p[2] - p[1]
    den = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(p[2] - p[0])
    return float(2 * (a[0] * b[1] - a[1] * b[0]) / den) if den > 1e-12 else 0.0


def curvature_jump(a: ExactEdge, at_end_a: bool, b: ExactEdge, at_end_b: bool, step: float) -> float:
    """|Δκ| в вершине при обходе по `a` к вершине и дальше по `b`; у отрезка κ = 0."""
    ka = 0.0 if a.kind == 0 else _curvature(_from_vertex(a.samples, at_end_a, step)[::-1])
    kb = 0.0 if b.kind == 0 else _curvature(_from_vertex(b.samples, at_end_b, step))
    return abs(ka - kb)


def exact_vertices(edges: list[ExactEdge], tol: float, tangent_deg: float, seam_curvature: float = 0.0,
                   step: float = 0.0, seam_deg: float = 0.0, spline_deg: float = 0.0) -> list[ExactVertex]:
    """Вершины точного сечения: концы рёбер, совпавшие в пределах `tol`. Вершина с углом
    касательных меньше `seam_deg` и скачком кривизны меньше `seam_curvature` — невидимый шов,
    не граница. Стык двух сплайнов — граница, только если угол больше `spline_deg`."""
    ends = np.array([e.ends[k] for e in edges for k in (0, 1)]).reshape(-1, 2)
    if not len(ends):
        return []
    group = -np.ones(len(ends), int)
    reps = []
    for i, p in enumerate(ends):
        for g, r in enumerate(reps):
            if np.linalg.norm(p - r) < tol:
                group[i] = g
                break
        else:
            group[i] = len(reps)
            reps.append(p)
    cos_tol = np.cos(np.radians(tangent_deg))
    cos_seam = np.cos(np.radians(seam_deg))
    cos_spline = np.cos(np.radians(spline_deg))
    out = []
    for g, r in enumerate(reps):
        idx = np.flatnonzero(group == g)
        if len(idx) != 2:
            out.append(ExactVertex(r, True, False, len(idx)))
            continue
        (ea, ka), (eb, kb) = [(i // 2, i % 2) for i in idx]
        a, b = edges[ea], edges[eb]
        # направления «наружу из вершины» вдоль каждого ребра; касание — когда они противоположны
        da = a.tangents[ka] * (1 if ka == 0 else -1)
        db = b.tangents[kb] * (1 if kb == 0 else -1)
        tangent = float(-(da @ db)) > cos_tol
        boundary = not (a.face == b.face and a.face >= 0 and a.kind == b.kind)
        if boundary and float(-(da @ db)) > cos_seam and seam_curvature > 0:
            boundary = curvature_jump(a, ka == 1, b, kb == 1, step) >= seam_curvature
        if boundary and a.kind == 3 and b.kind == 3 and float(-(da @ db)) > cos_spline:
            boundary = False
        out.append(ExactVertex(r, boundary, tangent, 2))
    return out


def merge_short(c: Chain, min_len: float) -> Chain:
    """Сливает отрезки короче `min_len` со следующим: стык между ними выбрасывается, нормаль —
    средняя по длине, треугольник — первый."""
    a, b = c.segment_ends()
    seg = np.linalg.norm(b - a, axis=1)
    keep, acc = [], 0.0
    # стык i (начало отрезка i) остаётся, если накопленная длина с прошлого оставленного ≥ min_len
    for i in range(c.n):
        if i == 0 or acc >= min_len:
            keep.append(i)
            acc = 0.0
        acc += seg[i]
    if c.closed and len(keep) > 1 and acc < min_len:
        keep.pop()                     # последний кусок короток — сливается с первым по кругу
    keep = np.array(keep)
    group = np.searchsorted(keep, np.arange(c.n), side="right") - 1
    if c.closed:
        group[:keep[0]] = len(keep) - 1
    w = seg[:, None]
    nrm = np.zeros((len(keep), 3))
    np.add.at(nrm, group, c.normals * w)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    pts = c.points[keep] if c.closed else np.vstack([c.points[keep], c.points[-1:]])
    return Chain(pts, nrm, c.tri[keep], c.closed)


def label_chain(c: Chain, edges: list[ExactEdge], verts: list[ExactVertex], tol: float,
                gap: int = 0, turn_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Метки цепочки по точному сечению.

    Тип отрезка — тип ближайшего точного ребра у середины, -1 — дальше `tol`. Стык (точка i —
    между отрезками i − 1 и i) — 1, если к нему ближе всего видимая граничная вершина (не дальше
    `tol`), соседние с ним стыки — -1 (вершина лежит между стыками, оба годятся), прочие 0; концы
    открытой цепочки -1. Касание — 1/0 на граничных стыках, -1 на прочих. Стык меньше разрешения
    скана — -2 (не требовать): другой размеченный не дальше `gap` точек — оба; острый, у которого
    скан поворачивает меньше `turn_deg` (направления ±3 отрезка)."""
    from scipy.spatial import cKDTree
    a, b = c.segment_ends()
    seg_type = -np.ones(c.n, np.int8)
    if edges:
        samples = np.concatenate([e.samples for e in edges])
        kinds = np.concatenate([np.full(len(e.samples), e.kind) for e in edges])
        dist, idx = cKDTree(samples).query((a + b) / 2)
        seg_type = np.where(dist <= tol, kinds[idx], -1).astype(np.int8)
    k = len(c.points)
    boundary = np.zeros(k, np.int8)
    tangent = -np.ones(k, np.int8)
    valid = np.ones(k, bool)
    if not c.closed:
        valid[[0, -1]] = False
    tree = cKDTree(c.points)
    for v in verts:
        if not v.boundary:
            continue
        dist, j = tree.query(v.point)
        if dist > tol or not valid[j]:
            continue
        boundary[j] = 1
        tangent[j] = int(v.tangent)
        for nb in ((j - 1) % k, (j + 1) % k):
            if boundary[nb] == 0:
                boundary[nb] = -1
    boundary[~valid] = -1
    marked = np.flatnonzero(boundary == 1)
    drop = set()
    for i, j in enumerate(marked):
        for o in marked[i + 1:]:
            d = o - j
            if (min(d, k - d) if c.closed else d) <= gap:
                drop.update((int(j), int(o)))
        if tangent[j] == 0 and turn_deg > 0:
            t = _scan_turn(c.points, int(j), c.closed)
            if t is not None and abs(t) < turn_deg:
                drop.add(int(j))
    for j in drop:
        boundary[j], tangent[j] = -2, -1
    return seg_type, boundary, tangent


def _scan_turn(points: np.ndarray, j: int, closed: bool, k: int = 3) -> float | None:
    """Поворот скана в точке j по направлениям k отрезков до и после, градусы."""
    n = len(points)
    i0, i1 = j - k, j + k
    if closed:
        i0, i1 = i0 % n, i1 % n
    elif i0 < 0 or i1 >= n:
        return None
    a, b = points[j] - points[i0], points[i1] - points[j]
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def sample_planes(rng, v: np.ndarray, faces: list[dict], count: int, oblique_frac: float,
                  margin: float, coplanar: float) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Плоскости сечения: (точка, система, наклонная ли). Параллельная — к случайной плоской
    грани, нормаль внутрь тела (туда, где протяжённость больше), глубина по протяжённости без
    краёв; ближе `coplanar` к другой параллельной плоской грани — новая попытка."""
    planes = [f["params"] for f in faces if f["type"] == "plane"]
    center = (v.min(0) + v.max(0)) / 2
    out = []
    for _ in range(count * 4):
        if len(out) == count:
            break
        if not planes or rng.random() < oblique_frac:
            n = rng.normal(size=3)
            frame = plane_frame(n)
            ext = (v - center) @ frame[2]
            origin = center + frame[2] * rng.uniform(ext.min(), ext.max()) * (1 - 2 * margin)
            out.append((origin, frame, True))
            continue
        p = planes[rng.integers(len(planes))]
        n = np.asarray(p["normal"], float)
        o = np.asarray(p["origin"], float)
        ext = (v - o) @ n
        if -ext.min() > ext.max():
            n, ext = -n, -ext
        depth = rng.uniform(margin, 1 - margin) * ext.max()
        offs = [(np.asarray(q["origin"], float) - o) @ n for q in planes
                if abs(abs(np.dot(q["normal"], n)) - 1) < 1e-9]
        if any(abs(depth - d) < coplanar for d in offs):
            continue
        frame = plane_frame(n)
        out.append((o + n * depth, frame, False))
    return out
