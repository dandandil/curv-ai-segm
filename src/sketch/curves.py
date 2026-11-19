"""Стыки и кривые по выходу 2D сети (README, «Эскиз из сечения», схема решена 2026-10-25).

Водораздел по цепочке: стык — локальный максимум вероятности границы не ниже порога в окне ±w
стыков (по кругу у замкнутой). Между стыками — кусок с типом по среднему вероятностей его
отрезков. Кривые строятся через зафиксированные стыки: отрезок — хорда; дуга — окружность через
оба конца, центр на серединном перпендикуляре подбирается по точкам куска (одна степень
свободы). Острый стык двух отрезков уточняется пересечением прямых, подогнанных по внутренним
точкам кусков: пик точен только до отрезка скана. Эллипс — коника через оба конца, подогнанная
по точкам куска (алгебраически, концы — точные линейные ограничения); не вышел эллипсом — дуга.
Сплайн — кубический B-сплайн по длине хорды с закреплёнными концами, узлов по числу точек.
Замкнутая цепочка без стыков — окружность, эллипс или сплайн из точки 0 в неё же. Касание на
стыках здесь не навязывается: это дело решателя ограничений.

Страховка от пропущенного стыка (`build_pieces_checked`): кусок, точки которого отходят от его
кривой дальше допуска, режется — в смене предсказанного типа, ближайшей к худшей точке, или в самой
худшей точке, — и кривые строятся заново. Без неё пропущенный стык отрезок–полуокружность склеивает
их в кусок с типом «отрезок», и хорда ложится поперёк контура.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

# Сплайн: внутренний узел на SPLINE_POINTS_PER_KNOT точек куска, не больше SPLINE_MAX_KNOTS —
# гибкости хватает на гладкую кривую, шум скана не повторяет. Выбрано, на подтверждение.
SPLINE_POINTS_PER_KNOT = 10
SPLINE_MAX_KNOTS = 8
# Эллипс с отношением осей больше этого или осью длиннее ELLIPSE_MAX_SPAN хорд — вырожденная
# подгонка (почти прямой кусок), строится дуга. Выбрано, на подтверждение.
ELLIPSE_MAX_RATIO = 20.0
ELLIPSE_MAX_SPAN = 20.0
# Кусок короче этого (отрезков) с каждой стороны разреза не режется. Выбрано.
SPLIT_MIN_SEGMENTS = 3


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def peaks(p: np.ndarray, closed: bool, threshold: float, window: int) -> np.ndarray:
    """Номера стыков-пиков в массиве `p` стыков цепочки."""
    n = len(p)
    if not n:
        return np.zeros(0, int)
    idx = np.arange(n)
    best = np.ones(n, bool)
    for d in range(1, window + 1):
        for s in (d, -d):
            j = (idx + s) % n if closed else np.clip(idx + s, 0, n - 1)
            other = p[j]
            # при равенстве побеждает меньший номер
            best &= (p > other) | ((p == other) & (idx <= j))
    return np.flatnonzero(best & (p >= threshold))


@dataclass
class Piece:
    start: int                   # номер точки-стыка начала в points цепочки
    end: int                     # номер точки конца (по кругу у замкнутой)
    kind: int
    curve: tuple                 # ("line", a, b) | ("arc", center, radius, a, b, ccw) | ("circle", c, r)
    #                              | ("ellipse", center, axes, radii, a, b, ccw) | ("ellipse_closed", center, axes, radii)
    #                              | ("spline", knots, coefs, degree) | ("none",)


def _circle_fit(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Окружность по точкам (Kåsa)."""
    a = np.c_[2 * p, np.ones(len(p))]
    sol = np.linalg.lstsq(a, (p ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def _arc_through(a: np.ndarray, b: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Окружность через a и b по точкам `pts`. Свободная величина — угол дуги θ ∈ (0, 2π) со стороны
    хорды, где лежат точки: центр на серединном перпендикуляре в (L/2) / tg(θ/2) от середины."""
    m, chord = (a + b) / 2, b - a
    L = np.linalg.norm(chord)
    perp = np.array([-chord[1], chord[0]]) / max(L, 1e-12)
    side = 1.0 if (pts - m).mean(0) @ perp >= 0 else -1.0

    def center(theta):
        return m - side * perp * (L / 2) / np.tan(theta / 2)

    def cost(theta):
        c = center(theta)
        return float(((np.linalg.norm(pts - c, axis=1) - np.linalg.norm(a - c)) ** 2).sum())

    r = minimize_scalar(cost, bounds=(1e-4, 2 * np.pi - 1e-4), method="bounded")
    c = center(r.x)
    return c, float(np.linalg.norm(a - c))


def _conic_rows(x: np.ndarray) -> np.ndarray:
    return np.c_[x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2, x[:, 0], x[:, 1], np.ones(len(x))]


def _ellipse_fit(pts: np.ndarray, through: np.ndarray | None = None):
    """Эллипс по точкам: алгебраическая коника, `through` — точки, через которые она проходит точно.
    Возвращает (центр, оси столбцами, полуоси) или None, если коника не эллипс."""
    every = pts if through is None else np.vstack([through, pts])
    m = every.mean(0)
    s = max(np.abs(every - m).max(), 1e-12)
    rows = _conic_rows((pts - m) / s)
    if through is None:
        theta = np.linalg.svd(rows)[2][-1]
    else:
        null = np.linalg.svd(_conic_rows((through - m) / s))[2][len(through):].T
        theta = null @ np.linalg.svd(rows @ null)[2][-1]
    a, b, c, d, e, f = theta
    if b * b - 4 * a * c >= 0:
        return None
    q = np.array([[a, b / 2], [b / 2, c]])
    cen = np.linalg.solve(2 * q, [-d, -e])
    f0 = cen @ q @ cen + d * cen[0] + e * cen[1] + f
    lam, axes = np.linalg.eigh(q)
    if np.any(-f0 / lam <= 0):
        return None
    radii = np.sqrt(-f0 / lam) * s
    return cen * s + m, axes, radii


def _ellipse_ok(fit, span: float) -> bool:
    return fit is not None and fit[2].max() / fit[2].min() < ELLIPSE_MAX_RATIO and fit[2].max() < ELLIPSE_MAX_SPAN * span


def _spline_through(a: np.ndarray, b: np.ndarray, inner: np.ndarray, degree: int = 3) -> tuple:
    """Кубический B-сплайн через a и b (первый и последний коэффициенты), по длине хорды."""
    p = np.vstack([a, inner, b])
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u /= max(u[-1], 1e-12)
    m = int(np.clip(len(inner) // SPLINE_POINTS_PER_KNOT, 0, SPLINE_MAX_KNOTS))
    knots = np.r_[np.zeros(degree + 1), np.linspace(0, 1, m + 2)[1:-1], np.ones(degree + 1)]
    basis = BSpline.design_matrix(u, knots, degree, extrapolate=True).toarray()   # иначе строка u = 1 нулевая
    rhs = p - np.outer(basis[:, 0], a) - np.outer(basis[:, -1], b)
    mid = np.linalg.lstsq(basis[:, 1:-1], rhs, rcond=None)[0]
    return ("spline", knots, np.vstack([a, mid, b]), degree)


def _line_fit(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[0]


def _intersect(c1, d1, c2, d2) -> np.ndarray | None:
    m = np.c_[d1, -d2]
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    t = np.linalg.solve(m, c2 - c1)
    return c1 + t[0] * d1


def build_pieces(points: np.ndarray, closed: bool, junctions: np.ndarray, sharp: np.ndarray,
                 type_prob: np.ndarray) -> tuple[list[Piece], np.ndarray]:
    """Куски цепочки и кривые. `junctions` — номера точек-стыков по возрастанию, `sharp` — острый ли,
    `type_prob` — `(отрезков, 4)`. Возвращает куски и точки стыков после уточнения."""
    n_seg = len(type_prob)
    pts = points.astype(np.float64)
    if closed and not len(junctions):
        kind = int(type_prob.mean(0).argmax())
        curve = ("none",)
        if kind == 1:
            curve = ("circle", *_circle_fit(pts))
        elif kind == 2:
            fit = _ellipse_fit(pts)
            curve = ("ellipse_closed", *fit) if _ellipse_ok(fit, np.ptp(pts, 0).max()) else ("circle", *_circle_fit(pts))
        elif kind == 3:
            curve = _spline_through(pts[0], pts[0], pts[1:])
        return [Piece(0, 0, kind, curve)], np.zeros((0, 2))
    if closed:
        bounds = list(zip(junctions, np.roll(junctions, -1)))
    else:
        cuts = [0, *junctions.tolist(), len(pts) - 1]
        bounds = list(zip(cuts[:-1], cuts[1:]))

    def span(s, e):
        """Номера точек от s до e включительно (по кругу)."""
        if closed and e <= s:
            return np.r_[np.arange(s, len(pts)), np.arange(0, e + 1)]
        return np.arange(s, e + 1)

    def segs(s, e):
        if closed and e <= s:
            return np.r_[np.arange(s, n_seg), np.arange(0, e)]
        return np.arange(s, e)

    kinds = [int(type_prob[segs(s, e)].mean(0).argmax()) if len(segs(s, e)) else 0 for s, e in bounds]
    jpos = {int(j): pts[j].copy() for j in junctions}
    # острый стык двух отрезков — пересечение прямых по внутренним точкам кусков
    for k, j in enumerate(junctions):
        if not sharp[k]:
            continue
        left = [b for b in range(len(bounds)) if bounds[b][1] == j]
        right = [b for b in range(len(bounds)) if bounds[b][0] == j]
        if not left or not right or kinds[left[0]] != 0 or kinds[right[0]] != 0:
            continue
        pl, pr = pts[span(*bounds[left[0]])][1:-1], pts[span(*bounds[right[0]])][1:-1]
        if len(pl) < 2 or len(pr) < 2:
            continue
        x = _intersect(*_line_fit(pl), *_line_fit(pr))
        if x is not None and np.linalg.norm(x - pts[j]) < 5 * np.linalg.norm(pts[j] - pts[(j + 1) % len(pts)]) + 1e-9:
            jpos[int(j)] = x
    out = []
    for (s, e), kind in zip(bounds, kinds):
        a = jpos.get(int(s), pts[s])
        b = jpos.get(int(e), pts[e])
        inner = pts[span(s, e)][1:-1]
        ccw = len(inner) > 0 and _cross(b - a, inner[len(inner) // 2] - a) > 0
        fit = _ellipse_fit(inner, np.vstack([a, b])) if kind == 2 and len(inner) >= 5 else None
        if kind in (1, 2) and np.linalg.norm(b - a) < 1e-9 * max(np.ptp(pts, 0).max(), 1e-12):
            curve = ("circle", *_circle_fit(pts[span(s, e)]))          # петля с одним стыком
        elif kind == 0 or not len(inner):
            curve = ("line", a, b)
        elif kind == 2 and _ellipse_ok(fit, np.linalg.norm(b - a)):
            curve = ("ellipse", *fit, a, b, ccw)
        elif kind in (1, 2):
            curve = ("arc", *_arc_through(a, b, inner), a, b, ccw)
        else:
            curve = _spline_through(a, b, inner)
        out.append(Piece(int(s), int(e), kind, curve))
    return out, np.array([jpos[int(j)] for j in junctions]).reshape(-1, 2)


def sample_curve(curve: tuple, k: int = 32) -> np.ndarray | None:
    if curve[0] == "line":
        t = np.linspace(0, 1, k)[:, None]
        return curve[1] * (1 - t) + curve[2] * t
    if curve[0] == "circle":
        a = np.linspace(0, 2 * np.pi, k, endpoint=False)
        return curve[1] + curve[2] * np.c_[np.cos(a), np.sin(a)]
    if curve[0] == "ellipse_closed":
        _, c, axes, radii = curve
        a = np.linspace(0, 2 * np.pi, k, endpoint=False)
        return c + (np.c_[radii[0] * np.cos(a), radii[1] * np.sin(a)]) @ axes.T
    if curve[0] in ("arc", "ellipse") and min(curve[2] if curve[0] == "arc" else curve[3].min(), 1.0) <= 1e-12:
        return None
    if curve[0] in ("arc", "ellipse"):
        if curve[0] == "arc":
            _, c, r, a, b, ccw = curve
            axes, radii = np.eye(2), np.array([r, r])
        else:
            _, c, axes, radii, a, b, ccw = curve

        def at(ts):
            return c + np.c_[radii[0] * np.cos(ts), radii[1] * np.sin(ts)] @ axes.T

        def angle(p):
            q = (p - c) @ axes / radii
            return np.arctan2(q[1], q[0])

        t0, t1 = angle(a), angle(b)
        dt = (t1 - t0) % (2 * np.pi)
        cand = [np.linspace(t0, t0 + dt, k), np.linspace(t0, t0 + dt - 2 * np.pi, k)]
        for ts in cand:
            # дуга с той стороны хорды, где лежат точки куска
            if (_cross(b - a, at(ts[k // 2:k // 2 + 1])[0] - a) > 0) == ccw:
                return at(ts)
        return at(cand[0])
    if curve[0] == "spline":
        _, knots, coefs, degree = curve
        return BSpline(knots, coefs, degree)(np.linspace(0, 1, k))
    return None


def _piece_index(pc: Piece, n: int, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Номера точек и отрезков куска (по кругу у замкнутой)."""
    if closed and pc.end <= pc.start:
        return np.r_[np.arange(pc.start, n), np.arange(0, pc.end + 1)], np.r_[np.arange(pc.start, n), np.arange(0, pc.end)]
    return np.arange(pc.start, pc.end + 1), np.arange(pc.start, pc.end)


def build_pieces_checked(points: np.ndarray, closed: bool, junctions: np.ndarray, sharp: np.ndarray,
                         type_prob: np.ndarray, tol: float, max_splits: int = 8) -> tuple[list[Piece], np.ndarray, np.ndarray, np.ndarray]:
    """`build_pieces` с разрезанием кусков, отходящих от своей кривой дальше `tol`. Новый стык —
    касательный (пропускаются в основном касательные). Возвращает куски, точки стыков, номера
    стыков и острый ли (с добавленными)."""
    junctions, sharp = np.asarray(junctions, int), np.asarray(sharp, bool)
    pts = points.astype(np.float64)
    types = type_prob.argmax(1)
    for _ in range(max_splits + 1):
        pieces, jpos = build_pieces(points, closed, junctions, sharp, type_prob)
        worst = None
        for pc in pieces:
            s = sample_curve(pc.curve, 256)
            idx, seg = _piece_index(pc, len(pts), closed)
            if s is None or len(seg) < 2 * SPLIT_MIN_SEGMENTS:
                continue
            d = np.min(np.linalg.norm(pts[idx][:, None] - s[None], axis=2), axis=1)
            if d.max() > tol and (worst is None or d.max() > worst[0]):
                worst = (d.max(), idx, seg, d)
        if worst is None:
            return pieces, jpos, junctions, sharp
        _, idx, seg, d = worst
        # разрез — во внутренней точке не ближе SPLIT_MIN_SEGMENTS к концам куска; смена типа между
        # отрезками at-1 и at
        inner = np.arange(SPLIT_MIN_SEGMENTS, len(idx) - SPLIT_MIN_SEGMENTS)
        change = inner[types[seg[inner - 1]] != types[seg[inner]]]
        at = inner[np.argmax(d[inner])]
        if len(change):
            at = change[np.argmin(np.abs(change - at))]
        order = np.argsort(np.r_[junctions, idx[at]], kind="stable")
        junctions = np.r_[junctions, idx[at]][order]
        sharp = np.r_[sharp, False][order]
    return pieces, jpos, junctions, sharp
