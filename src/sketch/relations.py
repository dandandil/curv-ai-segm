"""Эскиз сечения из точной топологии и метки ограничений по точному сечению (README, «Эскиз из
сечения», пункт 5 схемы; шаг 1 — решатель с оракулом, вариант (а)).

Эскиз: стыки — размеченные (`boundary == 1`), типы кусков — по меткам отрезков, кривые подогнаны
по точкам скана (`build_pieces`). Каждому куску сопоставляются рёбра точного сечения, ближайшие к
его точкам; по ним — точные параметры (прямая, окружность) и метки: параллельно, перпендикулярно,
коллинеарно, равная длина, равный радиус, концентрично — с допуском, у пар любых кусков сечения;
касание — у соседей на размеченном касательном стыке, если оно точное и по геометрии (отрезок–дуга:
расстояние от центра до прямой равно радиусу; дуга–дуга: окружности касаются): метка стыка
«касательный» — угол меньше 10°, а решатель понимает касание как угол 0, и излом в 5° ломал
перпендикулярность соседей. Равная длина — только у отрезков со стыками на обоих концах (у конца
открытой цепочки скан обрывается, длина куска не равна длине ребра). Кусок, у которого точные
рёбра другого типа или ложатся на прямую / окружность хуже допуска, и отрезок с неподвижными
обоими концами (соседи — эллипс, сплайн) меток не получают.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .curves import build_pieces, sample_curve
from .data import chain_slices
from .solver import Curve, Relation, Sketch, bulge_of

# Допуски меток по точной геометрии CAD: угол — синус 1e-3 (0.06°), длины — доля диагонали.
# Точное сечение даёт эти величины до 1e-6, так что допуск отделяет замысел от совпадения, а не
# от шума. Выбрано, на подтверждение.
SIN_TOL = 1e-3
LEN_TOL = 1e-4
# Доля точек куска, ближайших к ребру, чтобы ребро считалось его.
EDGE_SHARE = 0.2
KIND = {0: "line", 1: "arc"}


@dataclass
class Exact:
    kind: str                    # line | arc
    point: np.ndarray            # line: точка; arc: центр
    direction: np.ndarray        # line: единичное направление
    radius: float
    length: float                # line: длина по точным точкам
    samples: np.ndarray


def _fit_exact(kind: str, s: np.ndarray, tol: float) -> Exact | None:
    if kind == "line":
        c = s.mean(0)
        d = np.linalg.svd(s - c)[2][0]
        off = np.abs((s - c) @ np.array([-d[1], d[0]])).max()
        t = (s - c) @ d
        return Exact(kind, c, d, 0.0, float(np.ptp(t)), s) if off < tol else None
    a = np.c_[2 * s, np.ones(len(s))]
    sol = np.linalg.lstsq(a, (s ** 2).sum(1), rcond=None)[0]
    cen = sol[:2]
    r = float(np.sqrt(max(sol[2] + cen @ cen, 0.0)))
    off = np.abs(np.linalg.norm(s - cen, axis=1) - r).max()
    return Exact(kind, cen, np.zeros(2), r, 0.0, s) if off < tol else None


def section_sketch(z, sec: int) -> tuple[Sketch, list[Exact | None], list[Relation], list[tuple]] | None:
    """Эскиз сечения `sec` файла деталей с размеченными стыками и типами, точные параметры кусков,
    метки ограничений и кривые подгонки без решателя (по куску)."""
    diag = float(z["diag"])
    eo = np.r_[0, np.cumsum(z["edge_samples"])]
    edges = np.flatnonzero(z["edge_sec"] == sec)
    if not len(edges):
        return None
    samples = [z["samples"][eo[i]:eo[i + 1]].astype(np.float64) for i in edges]
    tree = cKDTree(np.concatenate(samples))
    owner = np.concatenate([np.full(len(s), k) for k, s in enumerate(samples)])

    vertices, fixed, curves, exact, fits = [], [], [], [], []
    ends = set()                                   # стыки-концы открытых цепочек
    relations = []
    for ps, ss, closed, s in chain_slices(z):
        if s != sec:
            continue
        pts = z["points"][ps].astype(np.float64)
        st, bd, tg = z["seg_type"][ss], z["boundary"][ps], z["tangent"][ps]
        n = len(st)
        g = np.flatnonzero(bd == 1)
        g = g[g < (n if closed else len(pts) - 1)]
        if not closed:
            g = g[g > 0]
        one_hot = np.full((n, 4), 0.25)
        one_hot[st >= 0] = 0
        one_hot[st >= 0, st[st >= 0]] = 1
        pieces, jpos = build_pieces(pts, closed, g, tg[g] == 0, one_hot)
        # стыки цепочки: размеченные (уточнённые), у открытой — ещё концы
        vid = {int(j): len(vertices) + k for k, j in enumerate(g)}
        vertices += list(jpos)
        fixed += [False] * len(g)
        if not closed:
            for end in (0, len(pts) - 1):
                vid[end] = len(vertices)
                ends.add(len(vertices))
                vertices.append(pts[end])
                fixed.append(False)
        first = len(curves)
        for pc in pieces:
            idx = (np.r_[np.arange(pc.start, len(pts)), np.arange(0, pc.end + 1)]
                   if closed and pc.end <= pc.start else np.arange(pc.start, pc.end + 1))
            p = pts[idx]
            kind = pc.curve[0]
            v0, v1 = vid.get(pc.start, -1), vid.get(pc.end, -1)
            if kind == "line":
                cv = Curve("line", v0, v1, points=p)
            elif kind == "arc" and v0 != v1:
                mid = sample_curve(pc.curve, 3)[1]
                cv = Curve("arc", v0, v1, np.array([bulge_of(vertices[v0], vertices[v1], mid)]), points=p)
            elif kind == "circle":
                cv = Curve("circle", -1, -1, np.r_[pc.curve[1], pc.curve[2]], points=p)
            else:
                cv = Curve("fixed", v0, v1, points=p)
                for v in (v0, v1):
                    if v >= 0:
                        fixed[v] = True
            curves.append(cv)
            fits.append(pc.curve)
            # точные рёбра куска
            own = owner[tree.query(p[1:-1] if len(p) > 2 else p)[1]]
            share = np.bincount(own, minlength=len(samples)) / len(own)
            mine = np.flatnonzero(share >= EDGE_SHARE)
            kinds = {int(z["edge_kind"][edges[k]]) for k in mine}
            want = {"line": 0, "arc": 1, "circle": 1}.get(cv.kind)
            ex = None
            if want is not None and kinds == {want}:
                ex = _fit_exact("line" if want == 0 else "arc", np.concatenate([samples[k] for k in mine]), LEN_TOL * diag)
            exact.append(ex)
        # касание на размеченном касательном стыке соседей (кусок k кончается в стыке, k + 1 начинается)
        m = len(pieces)
        for k in range(m if closed else m - 1):
            a, b = first + k, first + (k + 1) % m
            j = pieces[k].end
            if a != b and tg[j] == 1 and curves[a].kind in ("line", "arc") and curves[b].kind in ("line", "arc") \
                    and exact_tangent(exact[a], exact[b], LEN_TOL * diag):
                relations.append(Relation("tangent", a, b))
    if not curves:
        return None
    fixed = np.array(fixed, bool)
    for i, c in enumerate(curves):
        if c.kind == "line" and fixed[c.v0] and fixed[c.v1]:
            exact[i] = None
    relations = [r for r in relations if exact[r.a] is not None and exact[r.b] is not None]
    relations += label_pairs(exact, curves, diag, ends)
    return Sketch(np.array(vertices).reshape(-1, 2), fixed, curves), exact, relations, fits


def exact_tangent(a: Exact | None, b: Exact | None, tol: float) -> bool:
    """Касание по точной геометрии: отрезок–дуга — расстояние от центра до прямой равно радиусу,
    дуга–дуга — расстояние между центрами равно сумме или разности радиусов."""
    if a is None or b is None:
        return False
    if a.kind == "line" and b.kind == "line":
        return False
    if a.kind == "arc" and b.kind == "arc":
        d = np.linalg.norm(a.point - b.point)
        return min(abs(d - a.radius - b.radius), abs(d - abs(a.radius - b.radius))) < tol
    line, arc = (a, b) if a.kind == "line" else (b, a)
    off = arc.point - line.point
    return abs(abs(line.direction[0] * off[1] - line.direction[1] * off[0]) - arc.radius) < tol


def label_pairs(exact: list[Exact | None], curves: list[Curve], diag: float, ends: set[int]) -> list[Relation]:
    """Метки ограничений у пар кусков по точным параметрам."""
    out = []
    tol = LEN_TOL * diag
    for a in range(len(exact)):
        for b in range(a + 1, len(exact)):
            ea, eb = exact[a], exact[b]
            if ea is None or eb is None:
                continue
            if ea.kind == eb.kind == "line" and curves[a].kind == curves[b].kind == "line":
                sin = abs(ea.direction[0] * eb.direction[1] - ea.direction[1] * eb.direction[0])
                if sin < SIN_TOL:
                    off = eb.point - ea.point
                    if abs(ea.direction[0] * off[1] - ea.direction[1] * off[0]) < tol:
                        out.append(Relation("collinear", a, b))
                    else:
                        out.append(Relation("parallel", a, b))
                elif abs(ea.direction @ eb.direction) < SIN_TOL:
                    out.append(Relation("perpendicular", a, b))
                inner = not ({curves[a].v0, curves[a].v1, curves[b].v0, curves[b].v1} & ends)
                if inner and abs(ea.length - eb.length) < tol:
                    out.append(Relation("equal_length", a, b))
            elif ea.kind == eb.kind == "arc" and curves[a].kind in ("arc", "circle") and curves[b].kind in ("arc", "circle"):
                if np.linalg.norm(ea.point - eb.point) < tol:
                    out.append(Relation("concentric", a, b))
                if abs(ea.radius - eb.radius) < tol:
                    out.append(Relation("equal_radius", a, b))
    return out
