"""Решатель эскиза: кривые к точкам сечения при точных ограничениях (README, «Эскиз из сечения»,
пункт 4 схемы). 2D-аналог совместной подгонки `src/constraints.py`: Гаусс-Ньютон в нуль-
пространстве линеаризованных условий с демпфированием Левенберга-Марквардта.

Переменные — точки стыков (общие у соседних кривых, так что цепочка не рвётся), у дуги — прогиб
b = tg(θ/4) (θ — угол дуги со знаком, b > 0 — против часовой от начала к концу), у окружности
без стыков — центр и радиус. Эллипс и сплайн пока неподвижны: их концы — постоянные.
Невязка отрезка — до прямой, дуги — до окружности, так что конец открытой цепочки и касательный
стык могут скользить вдоль кривой без цены; их держит слабая привязка стыка к начальному
положению (вес — одна точка данных).
Ограничения: касание на стыке, параллельно, перпендикулярно, коллинеарно, равная длина,
равный радиус, концентрично.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.func import jacfwd

# Точек на кривую в решателе, не больше: на эскизе в 20 кривых решение — доли секунды.
SOLVE_POINTS = 100
# Вес привязки стыка к начальному положению. При весе одной точки стык двух почти параллельных
# отрезков (излом в несколько градусов) улетал на десятки процентов диагонали: их пересечение
# плохо обусловлено. [200 деталей val, решатель без ограничений, p90 / p99 отклонения: вес 1 —
# 7.4 / 29‰, 10 — 6.8 / 26‰, 100 — 6.5 / 22‰; кривые по отдельности — 6.4 / 21‰.] Выбрано.
ANCHOR_WEIGHT = 100.0
RELATIONS = ("tangent", "parallel", "perpendicular", "collinear", "equal_length", "equal_radius", "concentric")


@dataclass
class Curve:
    kind: str                    # line | arc | circle | fixed
    v0: int = -1                 # номера стыков начала и конца (у окружности -1)
    v1: int = -1
    extra: np.ndarray = field(default_factory=lambda: np.zeros(0))   # arc: [b]; circle: [cx, cy, r]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


@dataclass(frozen=True)
class Relation:
    name: str
    a: int
    b: int                       # у касания: b начинается в стыке, где кончается a


def arc_center(p0: Tensor, p1: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Центр и радиус дуги от p0 к p1 с прогибом b."""
    d = p1 - p0
    length = torch.sqrt((d ** 2).sum().clamp_min(1e-24))
    n = torch.stack([-d[1], d[0]]) / length
    c = (p0 + p1) / 2 + n * (length / 2) * (1 - b ** 2) / (2 * b)
    return c, length * (1 + b ** 2) / (4 * b.abs())


def bulge_of(p0: np.ndarray, p1: np.ndarray, mid: np.ndarray) -> float:
    """Прогиб дуги через p0, mid, p1: при b > 0 дуга справа от хорды p0 -> p1."""
    d = p1 - p0
    length = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / max(length, 1e-12)
    b = -2 * float((mid - (p0 + p1) / 2) @ n) / max(length, 1e-12)
    return b if abs(b) > 1e-6 else 1e-6


class Sketch:
    """Эскиз: стыки `vertices (V, 2)`, у каких стыков положение постоянно, кривые."""

    def __init__(self, vertices: np.ndarray, fixed: np.ndarray, curves: list[Curve]):
        self.vertices, self.fixed, self.curves = vertices.astype(np.float64), fixed.astype(bool), curves
        self.anchors = self.vertices.copy()             # наблюдаемые положения стыков

    # раскладка переменных: свободные стыки, затем extra кривых по порядку
    def _layout(self):
        free = np.flatnonzero(~self.fixed)
        slot = {int(v): 2 * k for k, v in enumerate(free)}
        off, extra = 2 * len(free), {}
        for i, c in enumerate(self.curves):
            if c.kind in ("arc", "circle"):
                extra[i] = slice(off, off + len(c.extra))
                off += len(c.extra)
        return slot, extra, off


def _problem(sketch: Sketch, relations: list[Relation]):
    """Переменные в нормированной системе, невязка, условия и запись решения обратно в эскиз."""
    curves = sketch.curves
    movable = [i for i, c in enumerate(curves) if c.kind != "fixed" and len(c.points)]
    pts_all = np.concatenate([c.points for c in curves if len(c.points)] + [sketch.vertices])
    origin = pts_all.mean(0)
    scale = float(np.linalg.norm(pts_all - origin, axis=1).max()) or 1.0
    rng = np.random.default_rng(0)
    pts = {}
    for i in movable:
        p = curves[i].points
        if len(p) > SOLVE_POINTS:
            p = p[np.sort(rng.choice(len(p), SOLVE_POINTS, replace=False))]
        pts[i] = torch.from_numpy((p - origin) / scale)

    # переменные в нормированной системе
    slot, extra, n = sketch._layout()
    const = torch.from_numpy((sketch.vertices - origin) / scale)
    anchor_ids = sorted(slot)
    anchor = torch.from_numpy((sketch.anchors[anchor_ids] - origin) / scale) if anchor_ids else None
    x0 = np.zeros(n)
    for v, s in slot.items():
        x0[s:s + 2] = (sketch.vertices[v] - origin) / scale
    for i, s in extra.items():
        c = curves[i]
        x0[s] = np.r_[(c.extra[:2] - origin) / scale, c.extra[2] / scale] if c.kind == "circle" else c.extra

    def vertex(x: Tensor, v: int) -> Tensor:
        s = slot.get(v)
        return const[v] if s is None else x[s:s + 2]

    def bulge(x: Tensor, i: int) -> Tensor:
        return x[extra[i]][0]

    def centre(x: Tensor, i: int) -> tuple[Tensor, Tensor]:
        c = curves[i]
        if c.kind == "circle":
            q = x[extra[i]]
            return q[:2], q[2]
        return arc_center(vertex(x, c.v0), vertex(x, c.v1), bulge(x, i))

    def tangent(x: Tensor, i: int, at: int) -> Tensor:
        c = curves[i]
        if c.kind == "line":
            d = vertex(x, c.v1) - vertex(x, c.v0)
        else:
            cen, _ = centre(x, i)
            r = vertex(x, at) - cen
            # обход дуги: при b > 0 против часовой — касательная повёрнута от радиуса на +90°
            d = torch.stack([-r[1], r[0]]) * torch.sign(bulge(x, i))
        return d / torch.sqrt((d ** 2).sum().clamp_min(1e-24))

    def unit(x: Tensor, i: int) -> Tensor:
        return tangent(x, i, curves[i].v0)

    def length(x: Tensor, i: int) -> Tensor:
        d = vertex(x, curves[i].v1) - vertex(x, curves[i].v0)
        return torch.sqrt((d ** 2).sum().clamp_min(1e-24))

    def cross(a: Tensor, b: Tensor) -> Tensor:
        return a[0] * b[1] - a[1] * b[0]

    def residual(x: Tensor) -> Tensor:
        out = []
        for i in movable:
            p, c = pts[i], curves[i]
            if c.kind == "line":
                a, b = vertex(x, c.v0), vertex(x, c.v1)
                d = b - a
                out.append(((p[:, 0] - a[0]) * d[1] - (p[:, 1] - a[1]) * d[0]) / torch.sqrt((d ** 2).sum().clamp_min(1e-24)))
            else:
                cen, r = centre(x, i)
                out.append(torch.sqrt(((p - cen) ** 2).sum(1).clamp_min(1e-24)) - r)
        if anchor_ids:
            v = torch.stack([vertex(x, k) for k in anchor_ids])
            out.append(np.sqrt(ANCHOR_WEIGHT) * (v - anchor).reshape(-1))
        return torch.cat(out) if out else x.new_zeros(0)

    def equations(x: Tensor) -> Tensor:
        parts = [x.new_zeros(0)]
        for rel in relations:
            a, b = rel.a, rel.b
            if rel.name == "tangent":
                at = curves[a].v1
                parts.append(cross(tangent(x, a, at), tangent(x, b, at)).reshape(1))
            elif rel.name == "parallel":
                parts.append(cross(unit(x, a), unit(x, b)).reshape(1))
            elif rel.name == "perpendicular":
                parts.append((unit(x, a) @ unit(x, b)).reshape(1))
            elif rel.name == "collinear":
                parts.append(torch.stack([cross(unit(x, a), unit(x, b)),
                                          cross(unit(x, a), vertex(x, curves[b].v0) - vertex(x, curves[a].v0))]))
            elif rel.name == "equal_length":
                parts.append((length(x, a) - length(x, b)).reshape(1))
            elif rel.name == "equal_radius":
                parts.append((centre(x, a)[1] - centre(x, b)[1]).reshape(1))
            elif rel.name == "concentric":
                parts.append(centre(x, a)[0] - centre(x, b)[0])
            else:
                raise ValueError(rel.name)
        return torch.cat(parts)

    def finish(x: np.ndarray) -> None:
        for v, s in slot.items():
            sketch.vertices[v] = x[s:s + 2] * scale + origin
        for i, s in extra.items():
            c = curves[i]
            c.extra = np.r_[x[s][:2] * scale + origin, x[s][2] * scale] if c.kind == "circle" else x[s].copy()

    rho = float(sum(len(v) for v in pts.values()))
    return torch.from_numpy(x0), residual, equations, finish, rho


def solve(sketch: Sketch, relations: list[Relation], iterations: int = 50) -> tuple[Sketch, float]:
    """Подгонка эскиза к точкам кривых при условиях. Возвращает эскиз (тот же объект, изменённый)
    и наибольшее нарушение условий в нормированной системе (доли размаха точек)."""
    x, residual, equations, finish, rho = _problem(sketch, relations)

    def merit(x: Tensor) -> float:
        r = residual(x)
        return float(0.5 * r @ r + rho * equations(x).abs().sum())

    mu = 1e-6
    for _ in range(iterations if len(x) else 0):
        r, g = residual(x), equations(x)
        J = jacfwd(residual)(x)
        if len(g):
            A = jacfwd(equations)(x)
            U, sv, Vh = torch.linalg.svd(A)
            k = int((sv > sv.max() * 1e-9).sum()) if len(sv) and sv.max() > 0 else 0
            d0 = -Vh[:k].T @ ((U[:, :k].T @ g) / sv[:k])
            Z = Vh[k:].T
        else:
            d0, Z = torch.zeros_like(x), torch.eye(len(x), dtype=x.dtype)
        JZ = J @ Z
        H = JZ.T @ JZ
        rhs = -JZ.T @ (r + J @ d0)
        eye = torch.eye(len(H), dtype=H.dtype) * float(H.diagonal().max().clamp_min(1e-12)) if len(H) else H
        current = merit(x)
        while mu < 1e10:
            step = d0 + (Z @ torch.linalg.solve(H + mu * eye, rhs) if len(H) else 0)
            new = merit(x + step)
            if new < current:
                x = x + step
                mu = max(mu / 4, 1e-12)
                break
            mu *= 4
        else:
            break
        if float(step.norm()) < 1e-13 or current - new < 1e-10 * current:
            break

    violation = float(equations(x).abs().max()) if relations else 0.0
    finish(x.numpy())
    return sketch, violation


# Ранг условий — по сингулярным числам якобиана, относительный порог; нарушение, при котором
# условие считается выполненным, — доли размаха точек. Выбрано.
RANK_TOL = 1e-6
SATISFIED = 1e-5
# Ограничение отвергается и тогда, когда невязка эскиза (данные и привязка) выросла больше чем
# во столько раз: выполненное условие ценой ухода от точек — не замысел (README, «Эскиз из
# сечения», пункт 7). Выбрано, на подтверждение.
GROWTH = 1.5


def _cost(sketch: Sketch) -> float:
    x, residual, _, _, _ = _problem(sketch, [])
    r = residual(x)
    return float(r @ r)


def _rank(sketch: Sketch, relations: list[Relation]) -> int:
    if not relations:
        return 0
    x, _, equations, _, _ = _problem(sketch, relations)
    sv = torch.linalg.svdvals(jacfwd(equations)(x))
    return int((sv > max(float(sv.max()), 1e-12) * RANK_TOL).sum())


def prune(relations: list[Relation]) -> list[Relation]:
    """Убирает ограничения, следующие из оставленных комбинаторно: параллельность (с ней —
    коллинеарность), коллинеарность, равная длина, равный радиус, концентричность — отношения
    эквивалентности, из пар группы нужны только связи, объединяющие разные части; перпендикулярность
    — одна на пару групп параллельных. Порядок сохраняется. Без этого на сечении с 50 отверстиями
    ~1200 пар «равный радиус» и «концентрично» шли в проверку ранга по одной."""
    parent: dict = {}

    def find(key):
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(name, a, b) -> bool:
        ra, rb = find((name, a)), find((name, b))
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    out, perp = [], set()
    for r in relations:
        if r.name == "perpendicular":
            key = frozenset((find(("parallel", r.a)), find(("parallel", r.b))))
            if key not in perp:
                perp.add(key)
                out.append(r)
        elif r.name == "collinear":
            fresh = union("collinear", r.a, r.b)
            union("parallel", r.a, r.b)
            if fresh:
                out.append(r)
        elif r.name in ("parallel", "equal_length", "equal_radius", "concentric"):
            if union(r.name, r.a, r.b):
                out.append(r)
        else:
            out.append(r)
    return out


def solve_greedy(sketch: Sketch, relations: list[Relation], iterations: int = 30):
    """Ограничения по одному в данном порядке (у сети — по убыванию p; README, «Эскиз из сечения»,
    пункт 7): следующее из принятых (ранг условий в текущем решении не растёт) пропускается,
    невыполнимое вместе с принятыми или уводящее эскиз от точек (невязка больше GROWTH раз)
    отвергается. Избыточные условия решать все сразу нельзя: на
    решении они зависимы, а вдали от него линеаризация считает их независимыми, и шаг не сходится
    (прямоугольник с четырьмя перпендикулярностями застревал на нарушении 0.04).
    Возвращает эскиз, принятые, избыточные, отвергнутые."""
    solve(sketch, [], iterations)
    cost = _cost(sketch)
    kept = prune(relations)
    accepted, redundant, rejected = [], [r for r in relations if r not in set(kept)], []
    relations = kept
    rank = 0
    for r in relations:
        new_rank = _rank(sketch, accepted + [r])
        if new_rank == rank:
            redundant.append(r)
            continue
        trial = copy.deepcopy(sketch)
        _, violation = solve(trial, accepted + [r], iterations)
        new_cost = _cost(trial)
        if violation < SATISFIED and new_cost <= GROWTH * cost:
            sketch, rank, cost = trial, new_rank, new_cost
            accepted.append(r)
        else:
            rejected.append(r)
    return sketch, accepted, redundant, rejected


def curve_tuple(sketch: Sketch, i: int) -> tuple:
    """Кривая в формате `src/sketch/curves.py` (для `sample_curve`)."""
    c = sketch.curves[i]
    if c.kind == "line":
        return ("line", sketch.vertices[c.v0], sketch.vertices[c.v1])
    if c.kind == "circle":
        return ("circle", c.extra[:2], float(c.extra[2]))
    if c.kind == "arc":
        p0, p1 = sketch.vertices[c.v0], sketch.vertices[c.v1]
        cen, r = arc_center(torch.from_numpy(p0), torch.from_numpy(p1), torch.tensor(float(c.extra[0]), dtype=torch.float64))
        mid_side = float(c.extra[0]) < 0                 # b < 0 — дуга слева от хорды
        return ("arc", cen.numpy(), float(r), p0, p1, mid_side)
    return ("none",)
