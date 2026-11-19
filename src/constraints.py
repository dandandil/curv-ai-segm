"""Соотношения соседних поверхностей и совместная подгонка с ними.

Касание, найденное аналитически по параметрам, надо сделать точным: у почти
касательной пары пересечение численно вырождено — две близкие прямые или
пустота. Поверхности компоненты подгоняются к своим точкам совместно, при
точных условиях (README, «Реконструкция B-Rep»): Гаусс-Ньютон в нуль-
пространстве линеаризованных условий, с демпфированием Левенберга-Марквардта.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.func import jacfwd

from . import surfaces as S
from .surfaces import AXIAL, Surface

# Соотношения, при которых поверхности не пересекаются, а касаются вдоль кривой.
TANGENT = ("plane_offset", "cylinders_tangent", "torus_cylinder_tangent",
           "torus_cone_tangent", "equal_radius",
           "torus_cylinder_elbow", "plane_cone_tangent")
# Порядок типов в паре: у каждого соотношения первым идёт младший.
_RANK = {k: i for i, k in enumerate(S.KINDS)}
# Точек на поверхность в совместной подгонке, не больше: на 18 тысячах точек компоненты из 26
# поверхностей одно решение стоило ~3 с (README, «Ревью подгонки»). RMS проверяется по всем.
SOLVE_POINTS = 300


@dataclass(frozen=True)
class Relation:
    name: str
    a: int
    b: int
    signs: tuple[float, ...] = ()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def detect(surfaces: dict[int, Surface], pairs, parallel_deg: float,
           tangent_rel: float, floor: float) -> list[Relation]:
    """Соотношения между соседними поверхностями по их параметрам.

    Параллельность — по углу, касание и соосность — по расхождению расстояния:
    долей радиуса, но не строже `floor` в миллиметрах (допуск RMS подгонки).
    Без пола касание мелких скруглений не находилось: у тора с r 0.25-0.57 мм
    смещение подгонки 0.02-0.03 мм, это 7-12% радиуса [измерено:
    32566_7bfc936b_5 по разметке, 4 из 6 пар тор–цилиндр]. Смотрятся только
    соседи: соотношения несмежных граней на пересечение не влияют.
    """
    sin_eps = np.sin(np.radians(parallel_deg))

    def near(mismatch: float, radius: float) -> bool:
        return abs(mismatch) < max(tangent_rel * radius, floor)

    out = []
    for i, j in pairs:
        if i not in surfaces or j not in surfaces:
            continue
        if _RANK[surfaces[i].kind] > _RANK[surfaces[j].kind]:
            i, j = j, i
        out += _pair(i, surfaces[i], j, surfaces[j], sin_eps, near)
    return out


def _pair(i: int, s: Surface, j: int, t: Surface, sin_eps: float, near) -> list[Relation]:
    p, q = s.params, t.params
    out = []
    if s.kind == "plane" and t.kind in AXIAL:
        n, a = _unit(p[:3]), _unit(q[3:6])
        offset = n @ q[:3] + p[3] / np.linalg.norm(p[:3])
        if t.kind == "cylinder" and abs(n @ a) < sin_eps:
            out.append(Relation("axis_in_plane", i, j))
            if near(abs(offset) - q[6], q[6]):
                out.append(Relation("plane_offset", i, j, (np.sign(offset),)))
        elif np.linalg.norm(np.cross(n, a)) < sin_eps:
            out.append(Relation("axis_along_normal", i, j))
            if t.kind == "torus" and near(abs(offset) - q[7], q[7]):
                out.append(Relation("plane_offset", i, j, (np.sign(offset),)))
        elif t.kind == "cone":
            # конус касается плоскости по образующей: ось наклонена к плоскости
            # на полуугол, и вершина лежит на плоскости
            tilt = np.arcsin(min(1.0, abs(n @ a)))
            apex = n @ q[:3] + p[3] / np.linalg.norm(p[:3])
            if abs(tilt - q[6]) < np.arcsin(sin_eps) and near(apex, 0.0):
                out.append(Relation("plane_cone_tangent", i, j, (np.sign(n @ a),)))
    elif s.kind == "sphere" and t.kind == "cylinder":
        a, r = _unit(q[3:6]), q[6]
        if near(np.linalg.norm(np.cross(p[:3] - q[:3], a)), r):
            out.append(Relation("coaxial", j, i))
            if near(p[3] - r, r):
                out.append(Relation("equal_radius", i, j))
    elif s.kind in AXIAL and t.kind in AXIAL:
        a = _unit(p[3:6])
        if np.linalg.norm(np.cross(a, _unit(q[3:6]))) >= sin_eps:
            if (s.kind, t.kind) == ("cylinder", "torus") and abs(a @ _unit(q[3:6])) < sin_eps:
                out += _elbow(i, s, j, t, near)
            return out
        out.append(Relation("axes_parallel", i, j))
        gap = np.linalg.norm(np.cross(q[:3] - p[:3], a))
        sizes = [x.params[6] for x in (s, t) if x.kind != "cone"]
        # у пары конусов радиуса нет — расхождение сравнивается с полом допуска
        if near(gap, min(sizes) if sizes else 0.0):
            out.append(Relation("coaxial", i, j))
            out += _coaxial_tangent(i, s, j, t, near)
        elif (s.kind, t.kind) == ("cylinder", "cylinder"):
            for signs in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)):
                if near(gap - signs[0] * p[6] - signs[1] * q[6], min(p[6], q[6])):
                    out.append(Relation("cylinders_tangent", i, j, signs))
                    break
    return out


def _elbow(i: int, s: Surface, j: int, t: Surface, near) -> list[Relation]:
    """Колено трубы: тор-изгиб переходит в прямой цилиндр. Оси перпендикулярны,
    ось цилиндра лежит в средней плоскости тора и касается его осевой
    окружности, радиус цилиндра равен малому радиусу тора; касание — по
    окружности [измерено, 90 деталей val по разметке: 30 из 122 касательных
    стыков без ребра — цилиндр и тор с осями под 89.7°]."""
    p, q = s.params, t.params
    a, axis = _unit(p[3:6]), _unit(q[3:6])
    in_plane = (p[:3] - q[:3]) @ axis
    reach = np.linalg.norm(np.cross(q[:3] - p[:3], a))
    if near(in_plane, q[7]) and near(reach - q[6], q[7]) and near(p[6] - q[7], q[7]):
        return [Relation("torus_cylinder_elbow", i, j)]
    return []


def _coaxial_tangent(i: int, s: Surface, j: int, t: Surface, near) -> list[Relation]:
    """Касание соосных тел вращения вдоль окружности."""
    p, q = s.params, t.params                      # t — тор, если тор есть
    if t.kind != "torus":
        return []
    big, small = q[6], q[7]
    if s.kind == "cylinder":
        gap = big - p[6]
        if near(abs(gap) - small, small):
            return [Relation("torus_cylinder_tangent", j, i, (np.sign(gap),))]
    if s.kind == "cone":
        z = (q[:3] - p[:3]) @ _unit(p[3:6])
        h = big * np.cos(p[6]) - z * np.sin(p[6])
        if near(abs(h) - small, small):
            return [Relation("torus_cone_tangent", j, i, (np.sign(h),))]
    return []


def equations(rel: Relation, kinds: dict[int, str], pa: Tensor, pb: Tensor) -> Tensor:
    """Условие соотношения как вектор, равный нулю при точном выполнении."""
    name = rel.name
    if name == "axis_in_plane":
        return (pa[:3] @ pb[3:6]).reshape(1)
    if name in ("axis_along_normal", "axes_parallel"):
        return torch.linalg.cross(pa[:3] if name == "axis_along_normal" else pa[3:6], pb[3:6])
    if name == "coaxial":
        return torch.linalg.cross(pb[:3] - pa[:3], pa[3:6])
    if name == "plane_offset":
        radius = pb[6] if kinds[rel.b] == "cylinder" else pb[7]
        return (pa[:3] @ pb[:3] + pa[3] - rel.signs[0] * radius).reshape(1)
    if name == "cylinders_tangent":
        gap = torch.sqrt((torch.linalg.cross(pb[:3] - pa[:3], pa[3:6]) ** 2).sum().clamp_min(1e-24))
        return (gap - rel.signs[0] * pa[6] - rel.signs[1] * pb[6]).reshape(1)
    if name == "torus_cylinder_tangent":
        return (rel.signs[0] * (pa[6] - pb[6]) - pa[7]).reshape(1)
    if name == "torus_cone_tangent":
        z = (pa[:3] - pb[:3]) @ pb[3:6]
        return (pa[6] * torch.cos(pb[6]) - z * torch.sin(pb[6]) - rel.signs[0] * pa[7]).reshape(1)
    if name == "torus_cylinder_elbow":
        a = pa[3:6] / torch.linalg.norm(pa[3:6])
        axis = pb[3:6] / torch.linalg.norm(pb[3:6])
        reach = torch.sqrt((torch.linalg.cross(pb[:3] - pa[:3], a) ** 2).sum().clamp_min(1e-24))
        return torch.stack([a @ axis, (pa[:3] - pb[:3]) @ axis, reach - pb[6], pa[6] - pb[7]])
    if name == "plane_cone_tangent":
        n = pa[:3] / torch.linalg.norm(pa[:3])
        a = pb[3:6] / torch.linalg.norm(pb[3:6])
        return torch.stack([n @ a - rel.signs[0] * torch.sin(pb[6]), n @ pb[:3] + pa[3]])
    if name == "equal_radius":
        return (pa[3] - pb[6]).reshape(1)
    raise ValueError(name)


def solve(surfaces: dict[int, Surface], relations: list[Relation],
          samples: dict[int, tuple[np.ndarray, np.ndarray]], iterations: int):
    """Совместная подгонка поверхностей к точкам при точных условиях.

    Шаг: `δ = δ₀ + Z·y`, где `δ₀` — поправка минимальной нормы, выполняющая
    линеаризованные условия, `Z` — базис их нуль-пространства, а `y` —
    демпфированный Гаусс-Ньютон по невязке внутри него. Ранг якобиана условий
    берётся по сингулярным числам, так что избыточные условия (две
    параллельности, говорящие одно и то же) систему не вырождают.

    Возвращает параметры в миллиметрах и максимальное нарушение условий в
    нормированной системе.
    """
    ids = sorted({r.a for r in relations} | {r.b for r in relations})
    kinds = {i: surfaces[i].kind for i in ids}
    samples = {i: _thin(*samples[i]) for i in ids}
    points = np.concatenate([samples[i][0] for i in ids])
    origin = points.mean(0)
    scale = float(np.linalg.norm(points - origin, axis=1).max())

    slices, start, p0 = {}, 0, []
    for i in ids:
        q = S.to_frame(kinds[i], surfaces[i].params, origin, scale)
        slices[i] = slice(start, start + len(q))
        start += len(q)
        p0.append(q)
    x = {i: torch.from_numpy((samples[i][0] - origin) / scale) for i in ids}
    w = {i: torch.from_numpy(np.sqrt(samples[i][1])) for i in ids}
    # Вес нарушения условий в функции качества шага: множители Лагранжа
    # порядка суммарного веса точек, штраф должен быть не слабее их.
    rho = float(sum(samples[i][1].sum() for i in ids))

    def residual(p: Tensor) -> Tensor:
        return torch.cat([w[i] * S.distance(kinds[i], p[slices[i]], x[i]) for i in ids])

    def constraints(p: Tensor) -> Tensor:
        parts = [g.reshape(1) for i in ids for g in S.gauge(kinds[i], p[slices[i]])]
        parts += [equations(r, kinds, p[slices[r.a]], p[slices[r.b]]) for r in relations]
        return torch.cat(parts)

    def merit(p: Tensor) -> float:
        r = residual(p)
        return float(0.5 * r @ r + rho * constraints(p).abs().sum())

    p = torch.from_numpy(np.concatenate(p0))
    mu = 1e-6
    for _ in range(iterations):
        r, g = residual(p), constraints(p)
        J, A = jacfwd(residual)(p), jacfwd(constraints)(p)
        U, sv, Vh = torch.linalg.svd(A)
        k = int((sv > sv.max() * 1e-9).sum())
        d0 = -Vh[:k].T @ ((U[:, :k].T @ g) / sv[:k])
        Z = Vh[k:].T
        JZ = J @ Z
        H = JZ.T @ JZ
        rhs = -JZ.T @ (r + J @ d0)
        eye = torch.eye(len(H), dtype=H.dtype) * float(H.diagonal().max().clamp_min(1e-12))
        current = merit(p)
        while mu < 1e10:
            step = d0 + Z @ torch.linalg.solve(H + mu * eye, rhs)
            new = merit(p + step)
            if new < current:
                p = p + step
                mu = max(mu / 4, 1e-12)
                break
            mu *= 4
        else:
            break
        # Сошлось: качество шага почти не меняется. Прежний критерий «шаг меньше 1e-13» не
        # срабатывал, и решение почти всегда шло все 50 итераций (README, «Ревью подгонки»).
        if float(step.norm()) < 1e-13 or current - new < 1e-10 * current:
            break

    violation = float(constraints(p).abs().max())
    out = {i: S.from_frame(kinds[i], p[slices[i]].numpy(), origin, scale) for i in ids}
    return out, violation


def _thin(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Не больше `SOLVE_POINTS` точек, выбор воспроизводим; суммарный вес сохраняется, чтобы
    штраф за нарушение условий оставался соразмерен невязке."""
    if len(points) <= SOLVE_POINTS:
        return points, weights
    keep = np.random.default_rng(0).choice(len(points), SOLVE_POINTS, replace=False)
    return points[keep], weights[keep] * weights.sum() / weights[keep].sum()


def mismatch(rel: Relation, surfaces: dict[int, Surface], scale: float) -> float:
    """Насколько соотношение нарушено на независимых подгонках: длины — долей `scale`,
    направления — синусом угла."""
    ka, kb = surfaces[rel.a].kind, surfaces[rel.b].kind
    pa = torch.from_numpy(S.to_frame(ka, surfaces[rel.a].params, np.zeros(3), scale))
    pb = torch.from_numpy(S.to_frame(kb, surfaces[rel.b].params, np.zeros(3), scale))
    return float(equations(rel, {rel.a: ka, rel.b: kb}, pa, pb).abs().max())


def components(relations: list[Relation]) -> list[set[int]]:
    """Компоненты связности графа соотношений, обходом в глубину."""
    graph = defaultdict(set)
    for r in relations:
        graph[r.a].add(r.b)
        graph[r.b].add(r.a)
    seen, out = set(), []
    for start in graph:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v not in comp:
                comp.add(v)
                stack.extend(graph[v] - comp)
        seen |= comp
        out.append(comp)
    return out


def reconcile(surfaces: dict[int, Surface], relations: list[Relation],
              samples: dict[int, tuple[np.ndarray, np.ndarray]], tol: float, cfg):
    """Наложить соотношения по компонентам; неподтверждённые отбросить.

    Соотношение подтверждается самой подгонкой: если с ним RMS поверхности
    выросла больше чем в `rms_growth` раз и вышла за допуск, либо условия
    несовместны, у худшей поверхности отбрасывается одно соотношение — сильнее всех
    нарушенное на независимых подгонках, — и компонента решается заново. Раньше
    отбрасывались все соотношения худшей поверхности, и точные касания гибли заодно с
    одним неверным условием [измерено, 90 деталей val по разметке: 32 из 122
    касательных стыков без ребра — касание найдено и отброшено при зазоре 0.013 мм и
    угле 0.15° у пары цилиндр–плоскость]; точное касание нарушено слабо и не
    выбирается. Перебор «решить без каждого по очереди» стоил десятков решений на
    отбрасывание и не заканчивался за 15 минут (README, «Ревью подгонки»).
    """
    surfaces = dict(surfaces)
    before = {i: S.rms(s.kind, s.params, *samples[i]) for i, s in surfaces.items()}
    all_points = np.concatenate([samples[i][0] for i in samples]) if samples else np.zeros((1, 3))
    scale = float(np.linalg.norm(all_points - all_points.mean(0), axis=1).max()) or 1.0
    error = {(r.name, r.a, r.b): mismatch(r, surfaces, scale) for r in relations}
    accepted, dropped = [], []

    def attempt(rels):
        """Решение и мера нарушения: наибольший избыток RMS, ∞ — условия несовместны."""
        params, violation = solve(surfaces, rels, samples, cfg.solver_iterations)
        after = {i: S.rms(surfaces[i].kind, p, *samples[i]) for i, p in params.items()}
        excess = {i: after[i] / max(cfg.rms_growth * before[i], tol) for i in params}
        score = max(excess.values()) if violation < 1e-9 else float("inf")
        return params, after, excess, score

    for comp in components(relations):
        rels = [r for r in relations if r.a in comp]
        while rels:
            params, after, excess, score = attempt(rels)
            if score <= 1:
                for i, p in params.items():
                    surfaces[i] = Surface(surfaces[i].kind, p, after[i])
                accepted += rels
                break
            worst = max(excess, key=excess.get)
            own = [r for r in rels if worst in (r.a, r.b)]
            culprit = max(own, key=lambda r: error[(r.name, r.a, r.b)])
            dropped.append(culprit)
            rels = [r for r in rels if r is not culprit]
    return surfaces, accepted, dropped


def settle(surfaces: dict[int, Surface], pairs, samples: dict[int, tuple[np.ndarray, np.ndarray]],
           tol: float, cfg, rounds: int = 3, short=()):
    """Поиск соотношений и совместная подгонка по кругу, пока находятся новые.

    Часть соотношений видна только после того, как выполнены другие: у
    короткого конического пояса ось подгонялась с наклоном 3°, вершина
    отстояла от оси соседнего цилиндра на 0.53 мм, и соосность не находилась;
    после выравнивания осей расхождение — 0.01 мм [24474_2e83b11c_0 по сети].
    Каждый круг решается от исходных поверхностей со всеми найденными
    соотношениями, чтобы проверка роста RMS сравнивала с исходной подгонкой.

    У пар из `short` (граница короче цепочки) соотношения берутся, только если
    среди них есть касание: такая граница нужна ради касания стенки со
    скруглением поперёк тонкой пластины, а одна параллельность по границе в
    1-4 ребра сводила подгонку, и тело распадалось на оболочки
    [115257_0e775b19_11 по разметке: axis_along_normal и axes_parallel].
    """
    pairs = list(pairs)
    short = {tuple(sorted(p)) for p in short}

    def find(surf):
        out = detect(surf, pairs, cfg.parallel_deg, cfg.tangent_rel, floor=tol)
        touch = {tuple(sorted((r.a, r.b))) for r in out if r.name in TANGENT}
        return [r for r in out if tuple(sorted((r.a, r.b))) not in short or tuple(sorted((r.a, r.b))) in touch]

    found = find(surfaces)
    for _ in range(rounds):
        fitted, accepted, dropped = reconcile(surfaces, found, samples, tol, cfg)
        known = {(r.name, r.a, r.b) for r in found}
        new = [r for r in find(fitted) if (r.name, r.a, r.b) not in known]
        if not new:
            return fitted, accepted, dropped
        found = found + new
    return fitted, accepted, dropped
