"""Описание сегментации для агента: сводка и примитив на кластер, границы как геометрия, связи кластеров.

Номера кластеров — как у сети: без поглощения мелких и слияний `fitting.fit_clusters`
(решено пользователем 2026-11-02, README «MCP-сервер сегментатора»); что стоит слить, видно в
`same_surface`. Всё считается сразу после нарезки и отдаётся словарями, поля — как в моделях
ответа `scripts/mcp_server.py`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from src import fitting as FT
from src import surfaces as S
from src import topology as TP
from src.model import PRIMITIVE_TYPES
from src.surfaces import Surface

# Допуск Douglas–Peucker для полилиний границ долей ребра рабочего меша: точнее половины ребра
# граница на меше не бывает (README, «Разметка»), мельче — остаётся зигзаг по рёбрам [измерено:
# test.stl, ребро 0.3 мм — допуск 0.02 мм: 3.8 точки на мм, 0.15 мм: 1.3; решено пользователем].
POLYLINE_EDGE_SHARE = 0.5
# Допуск связей — столько медианных RMS подгонки кластеров детали, не строже `rms_tol_mm`:
# правило 3σ конфига, но σ измерена на самой детали. На реальном скане 0.04 мм не находил ни
# одной связи [измерено: test.stl, медиана RMS 0.046, p75 0.10 мм; решено пользователем].
RELATION_SIGMAS = 3.0
# Точек вдоль цепочки для угла между гранями: угол меняется вдоль границы плавно, больше не нужно.
ANGLE_SAMPLES = 16
DIGITS = 4


@dataclass
class Fit:
    """Примитив кластера и точки, по которым он подогнан (вершины без полосы у границы)."""

    surface: Surface | None
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    sign: float             # +1, если нормаль поверхности со знаком OCC смотрит наружу, как нормали меша
    dist: np.ndarray        # |расстояние| точек до поверхности


def _r(a):
    return np.round(np.asarray(a, dtype=np.float64), DIGITS).tolist()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cut_edges(mesh, labels: np.ndarray, strength: np.ndarray) -> dict:
    """Рёбра рабочего меша между кластерами: пара `(a, b)`, a < b, треугольник со стороны a и b,
    вершины ребра, длина, сила границы."""
    adj = np.asarray(mesh.face_adjacency)
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    cut = la != lb
    swap = la[cut] > lb[cut]
    f = adj[cut]
    fa = np.where(swap, f[:, 1], f[:, 0])
    fb = np.where(swap, f[:, 0], f[:, 1])
    uv = np.asarray(mesh.face_adjacency_edges)[cut]
    v = np.asarray(mesh.vertices)
    return {"pair": np.stack([labels[fa], labels[fb]], 1), "fa": fa, "fb": fb, "uv": uv,
            "length": np.linalg.norm(v[uv[:, 0]] - v[uv[:, 1]], axis=1), "strength": strength[cut]}


def summarize(mesh, labels: np.ndarray, types: np.ndarray, cluster_types: dict[int, str], edges: dict) -> list[dict]:
    """Сводка по кластерам: тип, площадь, центр, габарит, средняя нормаль, соседи по границе."""
    k = int(labels.max()) + 1
    area = mesh.area_faces
    cl_area = np.bincount(labels, area, k)
    centroid = np.stack([np.bincount(labels, area * mesh.triangles_center[:, i], k) for i in range(3)], 1)
    normal = np.stack([np.bincount(labels, area * mesh.face_normals[:, i], k) for i in range(3)], 1)
    cl_type = np.array([PRIMITIVE_TYPES.index(cluster_types[c]) for c in range(k)])
    agree = np.bincount(labels, area * (types == cl_type[labels]), k)
    lo = np.full((k, 3), np.inf)
    hi = np.full((k, 3), -np.inf)
    np.minimum.at(lo, labels, mesh.triangles.min(1))
    np.maximum.at(hi, labels, mesh.triangles.max(1))

    uniq, inv = np.unique(edges["pair"], axis=0, return_inverse=True)
    shared = np.bincount(inv, edges["length"])
    strength = np.bincount(inv, edges["length"] * edges["strength"]) / shared
    neighbours = [[] for _ in range(k)]
    for (i, j), s, w in zip(uniq.tolist(), shared, strength):
        neighbours[i].append({"cluster": j, "boundary_mm": round(float(s), DIGITS), "strength": round(float(w), DIGITS)})
        neighbours[j].append({"cluster": i, "boundary_mm": round(float(s), DIGITS), "strength": round(float(w), DIGITS)})

    faces = np.bincount(labels, minlength=k)
    return [{
        "cluster": c,
        "type": cluster_types[c],
        "type_share": round(float(agree[c] / cl_area[c]), DIGITS),
        "faces": int(faces[c]),
        "area_mm2": round(float(cl_area[c]), DIGITS),
        "centroid": _r(centroid[c] / cl_area[c]),
        "bbox_min": _r(lo[c]),
        "bbox_max": _r(hi[c]),
        "mean_normal": _r(normal[c] / cl_area[c]),
        "neighbours": sorted(neighbours[c], key=lambda n: -n["boundary_mm"]),
    } for c in range(k)]


def fit_surfaces(mesh, labels: np.ndarray, cluster_types: dict[int, str], cfg) -> tuple[list[Fit], float]:
    """Примитив на каждый кластер, как в `fitting.fit_clusters`, но без поглощения и слияний.
    Возвращает подгонки и допуск RMS `cfg` в мм."""
    vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency)
    _, normal, area = FT.triangle_geometry(vertices, faces)
    tol = FT.rms_tolerance(area, cfg)
    excluded = FT.boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, len(vertices))
    fits = []
    for c in range(int(labels.max()) + 1):
        p, n, w = FT.cluster_points(c, labels, excluded, vertices, faces, normal, area, cfg.min_fit_points)
        s = FT.fit_cluster(p, n, w, cluster_types[c])
        if s is None:
            fits.append(Fit(None, p, n, w, 1.0, np.full(len(p), np.inf)))
            continue
        d = S.distance(s.kind, torch.from_numpy(s.params), torch.from_numpy(p)).numpy()
        sign = 1.0 if np.sum(S.normals(s.kind, s.params, p) * n) >= 0 else -1.0
        fits.append(Fit(s, p, n, w, sign, np.abs(d)))
    return fits, tol


def part_tolerance(fits: list[Fit], floor: float) -> float:
    """Допуск связей и «в допуске» для детали: `RELATION_SIGMAS` медианных RMS, не меньше `floor`."""
    rms = [f.surface.rms for f in fits if f.surface is not None]
    return max(floor, RELATION_SIGMAS * float(np.median(rms))) if rms else floor


def surface_params(f: Fit, tol: float) -> dict | None:
    """Параметры примитива в читаемом виде: плоскость — внешняя нормаль и точка, тела вращения — ось
    через точку против центра кластера и протяжённость кластера вдоль оси. `tol` — допуск детали
    `part_tolerance`."""
    s = f.surface
    if s is None:
        return None
    p = s.params
    out = {"kind": s.kind, "rms_mm": round(float(s.rms), DIGITS),
           "inlier_share": round(float(f.weights[f.dist <= tol].sum() / f.weights.sum()), DIGITS),
           "within_tolerance": bool(s.rms <= tol)}
    c = np.average(f.points, axis=0, weights=f.weights)
    if s.kind == "plane":
        scale = np.linalg.norm(p[:3]) * f.sign
        n, d = p[:3] / scale, p[3] / scale
        return out | {"normal": _r(n), "offset": round(float(d), DIGITS), "point": _r(c - (n @ c + d) * n)}
    if s.kind == "sphere":
        return out | {"center": _r(p[:3]), "radius": round(abs(float(p[3])), DIGITS)}
    a = _unit(p[3:6])
    z = (f.points - p[:3]) @ a
    if s.kind == "cone":
        if np.median(z) < 0:          # ось — от вершины в сторону раскрытия
            a, z = -a, -z
        return out | {"apex": _r(p[:3]), "axis": _r(a), "half_angle_deg": round(float(np.degrees(abs(p[6]))), DIGITS),
                      "axial_range": _r([z.min(), z.max()])}
    if s.kind == "torus":
        return out | {"center": _r(p[:3]), "axis": _r(a), "major_radius": round(abs(float(p[6])), DIGITS),
                      "minor_radius": round(abs(float(p[7])), DIGITS), "axial_range": _r([z.min(), z.max()])}
    t = (c - p[:3]) @ a
    return out | {"axis_point": _r(p[:3] + t * a), "axis": _r(a), "radius": round(abs(float(p[6])), DIGITS),
                  "axial_range": _r([z.min() - t, z.max() - t])}


def _oriented_normals(f: Fit, x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if f.surface is None:
        return fallback
    return f.sign * _unit(S.normals(f.surface.kind, f.surface.params, x))


def simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas–Peucker: индексы точек, которые остаются."""
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg, rel = points[j] - points[i], points[i + 1:j] - points[i]
        length = np.linalg.norm(seg)
        d = np.linalg.norm(rel if length < 1e-12 else np.cross(rel, seg / length), axis=1)
        m = int(d.argmax())
        if d[m] > tol:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return np.flatnonzero(keep)


def boundaries(mesh, labels: np.ndarray, fits: list[Fit], edges: dict, cfg, edge_mm: float) -> tuple[list[dict], list[list[float]]]:
    """Цепочки границ (`topology.boundaries`) полилиниями: угол между внешними нормалями подогнанных
    поверхностей вдоль цепочки и вид ребра — выпуклое, вогнутое или касательное (угол меньше
    `parallel_deg`). Выпуклость — по треугольникам по обе стороны: у выпуклого ребра центр соседнего
    лежит под касательной плоскостью своего. Возвращает цепочки и центры узлов."""
    v = np.asarray(mesh.vertices, np.float64)
    centers, face_n = mesh.triangles_center, mesh.face_normals
    b = TP.boundaries(mesh, labels, cfg.min_chain_edges)
    node_xyz = [v[m].mean(0) for m in b.nodes]
    by_pair = defaultdict(list)
    for e, (i, j) in enumerate(edges["pair"].tolist()):
        by_pair[(i, j)].append(e)

    chains = []
    for ch in b.chains:
        a, c = ch.pair
        e = np.array(by_pair[(a, c)])
        e = e[np.isin(edges["uv"][e], ch.vertices).all(1)]
        fa, fc = edges["fa"][e], edges["fb"][e]
        na = _oriented_normals(fits[a], centers[fa], face_n[fa])
        nc = _oriented_normals(fits[c], centers[fc], face_n[fc])
        gap = centers[fc] - centers[fa]
        bulge = np.mean(np.sum(gap * na, 1) - np.sum(gap * nc, 1))

        pts = v[ch.vertices]
        if ch.ends is None:
            pts = np.vstack([pts, pts[:1]])
        else:
            pts[[0, -1]] = node_xyz[ch.ends[0]], node_xyz[ch.ends[1]]
        s = pts[np.linspace(0, len(pts) - 1, min(ANGLE_SAMPLES, len(pts))).round().astype(int)]
        cos = np.sum(_oriented_normals(fits[a], s, np.tile(na.mean(0), (len(s), 1)))
                     * _oriented_normals(fits[c], s, np.tile(nc.mean(0), (len(s), 1))), 1)
        angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
        kind = "tangent" if angle < cfg.parallel_deg else ("convex" if bulge < 0 else "concave")
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        w = edges["length"][e]
        chains.append({
            "clusters": [a, c],
            "closed": ch.ends is None,
            "nodes": None if ch.ends is None else list(ch.ends),
            "length_mm": round(length, DIGITS),
            "normal_angle_deg": round(angle, 2),
            "edge": kind,
            "strength": round(float((w * edges["strength"][e]).sum() / w.sum()), DIGITS) if len(e) else None,
            "points": _r(pts[simplify(pts, POLYLINE_EDGE_SHARE * edge_mm)]),
        })
    return chains, [_r(x) for x in node_xyz]


def _groups(n: int, linked) -> list[list[int]]:
    """Связные компоненты по попарному отношению `linked(i, j)`; только компоненты из двух и больше."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if linked(i, j):
                parent[find(i)] = find(j)
    out = defaultdict(list)
    for i in range(n):
        out[find(i)].append(i)
    return [g for g in out.values() if len(g) > 1]


def relations(clusters: list[dict], fits: list[Fit], tol: float, cfg) -> dict:
    """Связи кластеров по всей детали, по подогнанным примитивам.

    Направления: нормали плоскостей и оси тел вращения жадно (от большей площади) собираются в
    группы с углом меньше `parallel_deg`; у группы — торцы (плоскости, нормаль вдоль), тела вращения
    (ось вдоль) и стенки (плоскости, нормаль поперёк): кандидаты в направление вытягивания и ось
    вращения. Копланарность — общая группа и смещение вдоль направления в пределах допуска RMS
    подгонки; соосность и касание — правило `constraints.detect`: расхождение долей радиуса
    `tangent_rel`, не строже допуска. `same_surface` — соседи одного типа, общий примитив которых
    держит допуск на каждой части отдельно (критерий `fitting.merge_splits`). `tol` — допуск детали
    `part_tolerance` или заданный агентом."""
    sin_eps = np.sin(np.radians(cfg.parallel_deg))
    surf = {c: surface_params(f, tol) for c, f in enumerate(fits)}
    area = {c["cluster"]: c["area_mm2"] for c in clusters}
    planes = [c for c, s in surf.items() if s and s["kind"] == "plane"]
    axial = [c for c, s in surf.items() if s and s["kind"] in ("cylinder", "cone", "torus")]

    def direction(c):
        return np.array(surf[c]["normal"] if surf[c]["kind"] == "plane" else surf[c]["axis"])

    groups = []
    for c in sorted(planes + axial, key=lambda c: -area[c]):
        d = direction(c)
        for g in groups:
            if np.linalg.norm(np.cross(g["sum"] / np.linalg.norm(g["sum"]), d)) < sin_eps:
                g["sum"] += area[c] * (d if g["sum"] @ d >= 0 else -d)
                g["members"].append(c)
                break
        else:
            groups.append({"sum": area[c] * d, "members": [c]})

    directions, coplanar, coaxial = [], [], []
    for g in groups:
        d = g["sum"] / np.linalg.norm(g["sum"])
        caps = [c for c in g["members"] if c in planes]
        rot = [c for c in g["members"] if c in axial]
        walls = [c for c in planes if abs(direction(c) @ d) < sin_eps]
        offset = {c: float(d @ np.array(surf[c]["point"])) for c in caps}
        for grp in _groups(len(caps), lambda i, j: abs(offset[caps[i]] - offset[caps[j]]) <= tol):
            members = sorted((caps[i] for i in grp), key=lambda c: -area[c])
            coplanar.append({"clusters": members, "normal": _r(d), "offset": round(float(np.mean([offset[c] for c in members])), DIGITS)})

        def point(c):
            s = surf[c]
            return np.array(s.get("axis_point") or s.get("center") or s["apex"])

        def size(c):
            s = surf[c]
            return s.get("radius") or s.get("minor_radius") or 0.0

        def same_axis(i, j):
            gap = np.linalg.norm(np.cross(point(rot[i]) - point(rot[j]), d))
            sizes = [size(rot[k]) for k in (i, j) if size(rot[k])]
            return gap < max(cfg.tangent_rel * min(sizes, default=0.0), tol)

        for grp in _groups(len(rot), same_axis):
            members = sorted((rot[i] for i in grp), key=lambda c: -area[c])
            coaxial.append({"clusters": members, "axis": _r(d), "axis_point": _r(point(members[0]))})
        if len(caps) + len(rot) + len(walls) > 1:
            directions.append({
                "direction": _r(d),
                "caps": [{"cluster": c, "offset": round(offset[c], DIGITS)} for c in sorted(caps, key=lambda c: offset[c])],
                "axial": sorted(rot, key=lambda c: -area[c]),
                "walls": sorted(walls, key=lambda c: -area[c]),
                "area_mm2": round(sum(area[c] for c in g["members"] + walls), DIGITS),
            })
    directions.sort(key=lambda g: -g["area_mm2"])

    cyl = [c for c, s in surf.items() if s and s["kind"] == "cylinder"]
    radii = [{"radius": round(float(np.mean([surf[cyl[i]]["radius"] for i in grp])), DIGITS),
              "clusters": sorted(cyl[i] for i in grp)}
             for grp in _groups(len(cyl), lambda i, j: abs(surf[cyl[i]]["radius"] - surf[cyl[j]]["radius"]) <= tol)]

    same = []
    for c in clusters:
        a = c["cluster"]
        for n in c["neighbours"]:
            b = n["cluster"]
            fa, fb = fits[a].surface, fits[b].surface
            if b < a or fa is None or fb is None or fa.kind != fb.kind:
                continue
            union = FT.fit_kind(fa.kind, *(np.concatenate([x, y]) for x, y in
                                           zip((fits[a].points, fits[a].normals, fits[a].weights),
                                               (fits[b].points, fits[b].normals, fits[b].weights))))
            if union is None:
                continue
            rms = max(S.rms(union.kind, union.params, f.points, f.weights) for f in (fits[a], fits[b]))
            if rms <= tol:
                same.append({"clusters": [a, b], "kind": fa.kind, "rms_mm": round(float(rms), DIGITS),
                             "boundary_strength": n["strength"]})
    return {"tolerance_mm": round(float(tol), DIGITS), "parallel_deg": cfg.parallel_deg, "directions": directions,
            "coplanar": coplanar, "coaxial": coaxial, "equal_radius": radii, "same_surface": same}
