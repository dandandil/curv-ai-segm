"""Эскиз выдавливания проекцией стенок на опору (README, «Грани операции и эскиз проекцией»).

Стенка выдавливания перпендикулярна опоре: вдоль направления выдавливания она проецируется в свою
кривую профиля, точки всей высоты ложатся на одну кривую. Стенка — одна кривая, тип — по типу грани;
стенки собираются в петли по близости концов; стык соседних стенок — касательный (середина между
концами, касание — из графа) или острый (пересечение продлённых кривых: на месте снятого позже
скругления ребра между стенками разрыв).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree

from .curves import _arc_through, _circle_fit, _cross, _intersect, _line_fit, _spline_through, sample_curve

# Тип кривой стенки по типу грани (номера CURVE_TYPES): плоскость — отрезок, цилиндр и конус —
# дуга (конус — выдавливание с уклоном, проекция — полоса, берётся средняя окружность), прочие — сплайн.
WALL_KIND = {"plane": 0, "cylinder": 1, "cone": 1}
SPLINE = 3
# Клетка прореживания проекции / диагональ: ~0.25 мм на теле Fusion (диагональ 20-35 мм), порядка
# ребра скана 0.3 мм — в клетке поперёк полосы проекции одна-две точки. Выбрано.
CELL_REL = 0.01
# Соседей на точку в графе для упорядочивания. Выбрано.
ORDER_K = 8
# Концы соседних стенок дальше этого / диагональ — не стык (другая петля или пропущенная стенка);
# разрыв от снятого скругления ребра — до радиуса скругления. Выбрано.
JOIN_REL = 0.08
# Виды стыка: касание и острый — соседние по графу стенки; разрыв — не соседние (между ними грань не
# этой операции, обычно позднее скругление ребра); свободный — конец открытой цепочки.
# вставлен — открытая цепочка замкнута отрезком между свободными концами (стенку там целиком
# срезала поздняя операция; эскиз там неизвестен, а ядру нужен замкнутый контур).
JOINTS = ("касание", "острый", "разрыв", "свободный", "замкнутая", "вставлен")
# Слияние кусков одной кривой эскиза (стенку разрезала поздняя операция): прямые — угол меньше
# MERGE_DEG и расстояние до прямой меньше MERGE_REL · D, дуги — центры и радиусы ближе MERGE_REL · D. Выбрано.
MERGE_DEG = 2.0
MERGE_REL = 0.005
# Направление по стенкам берётся, только если они задают его однозначно — второе собственное
# значение больше первого в DIRECTION_RATIO раз — и оно не дальше DIRECTION_DEG от нормали опоры;
# иначе нормаль опоры (у двух параллельных плоскостей направление в их плоскости решал бы шум
# нормалей скана: без проверки средний угол к точному 15° против 0.8° у нормали опоры). Выбрано.
DIRECTION_RATIO = 3.0
DIRECTION_DEG = 10.0
# Отбрасывание выбросов при подгонке стенки: дальше этого числа MAD от кривой. Выбрано.
OUTLIER_MAD = 3.0


@dataclass
class Wall:
    node: int
    kind: int
    points: np.ndarray           # (K, 2) по порядку вдоль кривой (прореженные), для порядка и концов
    closed: bool
    cloud: np.ndarray            # (P, 2) все спроецированные вершины стенки, для подгонки
    fixed: tuple | None = None   # кривая из уточнённой поверхности: ("line", c, d) | ("circle", c, r)


@dataclass
class SketchCurve:
    node: int
    kind: int
    curve: tuple                 # как Piece.curve в curves.py
    ends: tuple[str, str]        # стык в начале и в конце: JOINTS


def project(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Точки `(P, 3)` вдоль `direction` на плоскость (origin, frame[2]) -> `(P, 2)`: координаты по
    frame[:2] без сдвига начала, как у `exact_section`."""
    t = (points - origin) @ frame[2] / (direction @ frame[2])
    return (points - np.outer(t, direction)) @ frame[:2].T


def order_points(q: np.ndarray, cell: float) -> tuple[np.ndarray, bool]:
    """Прореживание по клеткам (центр масс клетки) и порядок вдоль кривой: самый длинный путь в
    минимальном остовном дереве kNN-графа. Замкнута, если концы пути соседствуют."""
    key = np.floor(q / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    c = np.zeros((inv.max() + 1, 2))
    np.add.at(c, inv, q)
    c /= np.bincount(inv)[:, None]
    if len(c) < 3:
        return c, False
    k = min(ORDER_K, len(c) - 1)
    d, j = cKDTree(c).query(c, k + 1)
    rows = np.repeat(np.arange(len(c)), k)
    g = coo_matrix((d[:, 1:].ravel() + 1e-12, (rows, j[:, 1:].ravel())), shape=(len(c), len(c)))
    tree = minimum_spanning_tree(g)
    tree = tree + tree.T

    def farthest(s):
        order, pred = breadth_first_order(tree, s, directed=False)
        dist = np.zeros(len(c))
        for v in order[1:]:
            dist[v] = dist[pred[v]] + np.linalg.norm(c[v] - c[pred[v]])
        return int(order[np.argmax(dist[order])]), pred

    a, _ = farthest(0)
    b, pred = farthest(a)
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    p = c[path]
    length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    closed = np.linalg.norm(p[0] - p[-1]) < 3 * cell and length > 10 * cell
    return p, bool(closed)


def extrusion_direction(v: np.ndarray, f: np.ndarray, tri_node: np.ndarray, nodes: list[int],
                        support_normal: np.ndarray) -> np.ndarray:
    """Направление выдавливания: перпендикулярное нормалям треугольников стенок (наименьший собственный
    вектор Σ a·n·nᵀ), если они его задают (DIRECTION_RATIO, DIRECTION_DEG), иначе нормаль опоры; знак —
    как у нормали опоры. Нормаль опоры по подгонке скана у маленькой или шумной опоры наклонена, и
    проекция высокой стенки расплывается [точный меш, 300 деталей val: с точным направлением профиль
    верен 75.1% против 72.1%]."""
    s = support_normal / np.linalg.norm(support_normal)
    tri = v[f[np.isin(tri_node, nodes)]]
    if not len(tri):
        return s
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    u = nrm / np.maximum(area, 1e-30)[:, None]
    lam, vec = np.linalg.eigh((u * area[:, None]).T @ u / max(area.sum(), 1e-30))
    d = vec[:, 0] if vec[:, 0] @ s >= 0 else -vec[:, 0]
    if lam[1] > DIRECTION_RATIO * lam[0] and d @ s > np.cos(np.radians(DIRECTION_DEG)):
        return d
    return s


def one_side(v: np.ndarray, f: np.ndarray, kind: int, origin: np.ndarray, direction: np.ndarray,
             frame: np.ndarray) -> np.ndarray:
    """Маска треугольников стенки с одной стороны: у тонкой стенки метку грани получают и треугольники
    другой её стороны, их нормали смотрят обратно. Знак — нормаль треугольника в плоскости против
    главного направления нормалей (плоскость) или радиуса от подогнанного центра (дуга); сторона —
    большинство по площади. У сплайна не делится."""
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    n2 = nrm @ frame[:2].T / np.maximum(area, 1e-30)[:, None]
    c2 = project(tri.mean(1), origin, direction, frame)
    if kind == 0:
        ref = np.linalg.svd(n2 * area[:, None], full_matrices=False)[2][0]
        sign = n2 @ ref
    elif kind == 1 and len(c2) >= 3:
        u = c2 - _circle_fit(c2)[0]
        sign = (n2 * u).sum(1)
    else:
        return np.ones(len(f), bool)
    pos = area[sign > 0].sum() >= area[sign <= 0].sum()
    return sign > 0 if pos else sign <= 0


def walls_of(v: np.ndarray, f: np.ndarray, tri_node: np.ndarray, nodes: list[int], types: dict[int, str],
             origin: np.ndarray, direction: np.ndarray, frame: np.ndarray, cell: float,
             fixed: dict[int, tuple] | None = None) -> list[Wall]:
    """Стенки по узлам; `fixed` — кривые стенок из уточнённых поверхностей (выпрямление детали),
    вместо подгонки по проекции."""
    out = []
    for m in nodes:
        fm = f[tri_node == m]
        kind = WALL_KIND.get(types[m], SPLINE)
        if len(fm):
            fm = fm[one_side(v, fm, kind, origin, direction, frame)]
        vid = np.unique(fm)
        if len(vid) < 3:
            continue
        q = project(v[vid], origin, direction, frame)
        p, closed = order_points(q, cell)
        if len(p) >= 2:
            out.append(Wall(m, kind, p, closed, q, (fixed or {}).get(m)))
    return out


def _same_curve(a: Wall, b: Wall, tol: float) -> bool:
    if a.kind != b.kind or a.kind == SPLINE or a.closed or b.closed:
        return False
    ca, cb = _support_curve(a), _support_curve(b)
    if a.kind == 0:
        return (abs(float(ca[2] @ cb[2])) > np.cos(np.radians(MERGE_DEG))
                and abs(float(_cross2(ca[2], (cb[1] - ca[1])[None])[0])) < tol)
    return np.linalg.norm(ca[1] - cb[1]) < tol and abs(ca[2] - cb[2]) < tol


def merge_pieces(walls: list[Wall], cell: float, tol: float) -> list[Wall]:
    """Сливает куски одной кривой (коллинеарные прямые, дуги одной окружности), если между их
    ближайшими концами нет конца другой стенки: иначе профиль идёт через неё (паз в своём профиле)."""
    walls = list(walls)
    merged = True
    while merged:
        merged = False
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                a, b = walls[i], walls[j]
                if not _same_curve(a, b, tol):
                    continue
                ends_a, ends_b = a.points[[0, -1]], b.points[[0, -1]]
                d = np.linalg.norm(ends_a[:, None] - ends_b[None], axis=-1)
                ia, ib = np.unravel_index(d.argmin(), d.shape)
                gap = d[ia, ib]
                others = [w.points[[0, -1]] for k, w in enumerate(walls) if k not in (i, j)]
                if others:
                    o = np.concatenate(others)
                    near = np.minimum(np.linalg.norm(o - ends_a[ia], axis=1), np.linalg.norm(o - ends_b[ib], axis=1))
                    if (near < gap).any():
                        continue
                pts, closed = order_points(np.vstack([a.points, b.points]), cell)
                walls[i] = Wall(a.node, a.kind, pts, closed, np.vstack([a.cloud, b.cloud]), a.fixed)
                walls.pop(j)
                merged = True
                break
            if merged:
                break
    return walls


def _flip(w: Wall) -> Wall:
    return Wall(w.node, w.kind, w.points[::-1], w.closed, w.cloud, w.fixed)


def chain_walls(walls: list[Wall], join: float, adjacent: set[tuple[int, int]]) -> list[tuple[list[Wall], bool]]:
    """Петли из стенок: жадно в обе стороны от первой, следующая — соседняя по графу меша с
    ближайшим концом, нет такой — ближайшая по концу (разрыв); замкнутые стенки — петли из одной.
    Только по концам сборка путалась в профилях со ступеньками, где рядом много концов; только
    вперёд — контур, начатый со стенки в середине, распадался на две цепочки."""
    loops = [([w], True) for w in walls if w.closed]
    free = [w for w in walls if not w.closed]

    def grow(chain: list[Wall]) -> None:
        while free:
            end = chain[-1].points[-1]
            d = [(min(np.linalg.norm(w.points[0] - end), np.linalg.norm(w.points[-1] - end)), i) for i, w in enumerate(free)]
            near = [(x, i) for x, i in d if tuple(sorted((chain[-1].node, free[i].node))) in adjacent and x <= join]
            dist, i = min(near or d)
            if dist > join:
                return
            w = free.pop(i)
            chain.append(_flip(w) if np.linalg.norm(w.points[-1] - end) < np.linalg.norm(w.points[0] - end) else w)

    while free:
        chain = [free.pop(0)]
        grow(chain)
        if np.linalg.norm(chain[-1].points[-1] - chain[0].points[0]) > join:
            chain = [_flip(w) for w in chain[::-1]]
            grow(chain)
        loops.append((chain, np.linalg.norm(chain[-1].points[-1] - chain[0].points[0]) <= join))
    return loops


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[0] * b[:, 1] - a[1] * b[:, 0]


def _robust(fit, dist, p: np.ndarray):
    """Подгонка, затем повторно без точек дальше OUTLIER_MAD медианных отклонений."""
    model = fit(p)
    d = dist(model, p)
    keep = d <= np.median(d) + OUTLIER_MAD * np.median(np.abs(d - np.median(d))) + 1e-12
    return fit(p[keep]) if keep.sum() >= 3 else model


def _line(p: np.ndarray) -> tuple:
    c, d = _robust(_line_fit, lambda m, x: np.abs(_cross2(m[1], x - m[0])), p)
    return ("line", c, d)


def _circle(p: np.ndarray) -> tuple:
    c, r = _robust(_circle_fit, lambda m, x: np.abs(np.linalg.norm(x - m[0], axis=1) - m[1]), p)
    return ("circle", c, r)


def _support_curve(w: Wall) -> tuple:
    """Продлеваемая кривая стенки для пересечения: ("line", c, d) или ("circle", c, r)."""
    if w.fixed is not None:
        return w.fixed
    if w.kind == 0:
        return _line(w.cloud)
    if w.kind == 1 and len(w.cloud) >= 3:
        return _circle(w.cloud)
    return ("line", *_line_fit(w.points[-3:] if len(w.points) >= 3 else w.points))


def _on(curve: tuple, x: np.ndarray) -> np.ndarray:
    """Ближайшая к `x` точка кривой."""
    if curve[0] == "line":
        return curve[1] + ((x - curve[1]) @ curve[2]) * curve[2]
    u = x - curve[1]
    return curve[1] + curve[2] * u / max(np.linalg.norm(u), 1e-12)


def _crossings(a: tuple, b: tuple) -> list[np.ndarray]:
    if a[0] == "circle" and b[0] == "line":
        a, b = b, a
    if a[0] == "line" and b[0] == "line":
        x = _intersect(a[1], a[2], b[1], b[2])
        return [] if x is None else [x]
    if a[0] == "line":
        c, d, (o, r) = a[1], a[2], b[1:]
        w = c - o
        disc = (w @ d) ** 2 - (w @ w - r * r)
        return [] if disc < 0 else [c + t * d for t in (-(w @ d) - np.sqrt(disc), -(w @ d) + np.sqrt(disc))]
    (c1, r1), (c2, r2) = a[1:], b[1:]
    d = np.linalg.norm(c2 - c1)
    if d < 1e-12 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    x = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    h = np.sqrt(max(r1 * r1 - x * x, 0))
    u = (c2 - c1) / d
    return [c1 + x * u + s * h * np.array([-u[1], u[0]]) for s in (-1, 1)]


def junction(a: Wall, b: Wall, tangent: bool, join: float, crossings: bool = True) -> np.ndarray:
    """Стык конца `a` и начала `b`: касательный — середина концов, острый — ближайшее к ней пересечение
    продлённых кривых (не дальше `join`, иначе середина); `crossings` False — всегда середина."""
    mid = (a.points[-1] + b.points[0]) / 2
    if tangent or not crossings:
        return mid
    ca, cb = _support_curve(a), _support_curve(b)
    cand = [x for x in _crossings(ca, cb) if np.linalg.norm(x - mid) <= join]
    if cand:
        return min(cand, key=lambda x: np.linalg.norm(x - mid))
    return (_on(ca, a.points[-1]) + _on(cb, b.points[0])) / 2 if a.kind < SPLINE and b.kind < SPLINE else mid


def wall_curve(w: Wall, a: np.ndarray | None, b: np.ndarray | None) -> tuple:
    """Кривая стенки через стыки `a`, `b`; None — свободный конец открытой цепочки (конец пути на
    подогнанной кривой)."""
    inner = w.points[1:-1]
    if w.kind < SPLINE:
        base = _support_curve(w)
        a = _on(base, w.points[0]) if a is None else a
        b = _on(base, w.points[-1]) if b is None else b
    else:
        a = w.points[0] if a is None else a
        b = w.points[-1] if b is None else b
    if w.kind == 0 or not len(inner):
        return ("line", a, b)
    if w.kind == 1:
        ccw = _cross(b - a, inner[len(inner) // 2] - a) > 0
        if w.fixed is not None and w.fixed[0] == "circle":
            return ("arc", w.fixed[1], w.fixed[2], a, b, ccw)
        return ("arc", *_arc_through(a, b, w.cloud), a, b, ccw)
    return _spline_through(a, b, inner)


def closed_curve(w: Wall) -> tuple:
    if w.fixed is not None and w.fixed[0] == "circle":
        return w.fixed
    if w.kind == 1:
        return _circle(w.cloud)
    return _spline_through(w.points[0], w.points[0], w.points[1:])


def sketch_from_walls(walls: list[Wall], tangent_pairs: set[tuple[int, int]], adjacent: set[tuple[int, int]],
                      join: float, cell: float, merge_tol: float, close_open: bool = False,
                      crossings: bool = True) -> list[list[SketchCurve]]:
    """Петли эскиза: на стенку одна кривая через стыки с соседями; одна стенка с близкими концами —
    замкнутая кривая (у отверстия путь по точкам оставляет щель в пару клеток)."""
    out = []
    for chain, closed in chain_walls(merge_pieces(walls, cell, merge_tol), join, adjacent):
        if len(chain) == 1 and closed:
            out.append([SketchCurve(chain[0].node, chain[0].kind, closed_curve(chain[0]), ("замкнутая", "замкнутая"))])
            continue
        n = len(chain)
        pairs = range(n) if closed else range(n - 1)
        joints, kinds = {}, {}
        for i in pairs:
            a, b = chain[i], chain[(i + 1) % n]
            key = tuple(sorted((a.node, b.node)))
            kinds[i] = "касание" if key in tangent_pairs else "острый" if key in adjacent else "разрыв"
            joints[i] = junction(a, b, kinds[i] == "касание", join, crossings)
        loop = []
        for i, w in enumerate(chain):
            prev = (i - 1) % n if closed or i > 0 else -1
            free = "свободный" if not close_open else "вставлен"
            ends = (kinds.get(prev, free), kinds.get(i, free))
            loop.append(SketchCurve(w.node, w.kind, wall_curve(w, joints.get(prev), joints.get(i)), ends))
        if not closed and close_open:
            a, b = sample_curve(loop[-1].curve, 2), sample_curve(loop[0].curve, 2)
            if a is not None and b is not None:
                loop.append(SketchCurve(-1, 0, ("line", a[-1], b[0]), ("вставлен", "вставлен")))
        out.append(loop)
    return out


def self_intersects(loop: list[SketchCurve], samples: int = 24) -> bool:
    """Пересекает ли петля сама себя: отрезки ломаной по точкам её кривых, кроме соседних."""
    pts = [s[:-1] for c in loop if (s := sample_curve(c.curve, samples)) is not None]
    if not pts:
        return False
    p = np.vstack(pts)
    a, b = p, np.roll(p, -1, axis=0)
    n = len(p)
    d = b - a
    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if not len(j):
            continue
        r, s = d[i], d[j]
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        q = a[j] - a[i]
        ok = np.abs(den) > 1e-15
        t = np.where(ok, (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (q[:, 0] * r[1] - q[:, 1] * r[0]) / np.where(ok, den, 1), -1)
        if np.any(ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)):
            return True
    return False
