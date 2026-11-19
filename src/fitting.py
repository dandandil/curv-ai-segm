"""Подгонка примитивов к кластерам: стартовые параметры для совместной подгонки.

Старт — `~/scan2cad/colouring_test/fits.py` (конус — своя `fit_cone`), затем
все параметры уточняются вместе по взвешенному точному расстоянию
(`refine`): `fits.py` весов не знает, а ось цилиндра берёт из нормалей и не
уточняет. Здесь же выбор точек, выбор типа и слияние лишних разрезов.

Точки — вершины меша, а не центроиды треугольников: при тесселяции на
поверхности лежат вершины, а центроид — внутри хорды. [измерено: 20 деталей
валидации, медиана |d| до истинной поверхности из brep.json, вершины против
центроидов — тесселяция OCC, цилиндр: 0 против 0.0006 мм, тор: 0 против
0.0013; mesh.ply, из которого строится подложка: 0.0001 против 0.0012 и
0.0007 против 0.0030; рабочий меш noisy: 0.0117 против 0.0135 и 0.030 против
0.033 — там смещение центроида тонет в шуме, но остаётся систематическим]
"""
from __future__ import annotations

import fits
import numpy as np
from scipy.optimize import least_squares

from . import surfaces as S
from .model import PRIMITIVE_TYPES
from .surfaces import Surface

def fit_cone(P: np.ndarray, N: np.ndarray):
    """Конус: вершина, ось и полуугол уточняются вместе по точному расстоянию.

    `fits.fit_cone` брал ось как у цилиндра — собственный вектор `NᵀN` без
    центрирования, — а у конуса нормали образуют с осью постоянный, не прямой
    угол: ось — собственный вектор ковариации нормалей. И ось потом не
    уточнялась. На фасках это давало RMS в миллиметры [измерено, 53 истинные
    конические грани val: RMS конуса ~5.8 мм при RMS сферы 0.014]. Здесь два
    старта — вершина как общая точка касательных плоскостей (крутой конус,
    фаска) и цилиндр с малым углом (пологий, литейный уклон), — оба
    уточняются `least_squares`, берётся лучший. Формат результата — как у
    `fits`: словарь с apex, axis, half_angle в градусах.
    """
    centred = N - N.mean(0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, 0]
    if (N @ axis).mean() > 0:
        axis = -axis            # нормали наружу смотрят против оси вдоль раствора
    theta0 = float(np.arcsin(np.clip(abs((N @ axis).mean()), 1e-3, 0.99)))

    def residual(x):
        a = x[3:6] / np.linalg.norm(x[3:6])
        u = P - x[:3]
        z = u @ a
        rho = np.linalg.norm(u - np.outer(z, a), axis=1)
        return rho * np.cos(x[6]) - z * np.sin(x[6])

    starts = []
    try:
        apex, *_ = np.linalg.lstsq(N, (N * P).sum(1), rcond=None)
        starts.append(np.r_[apex, axis, theta0])
    except np.linalg.LinAlgError:
        pass
    cyl, _ = fits.fit_cylinder(P, N)
    if cyl is not None:
        c = np.asarray(cyl["point"] if "point" in cyl else cyl["center"], float)
        r = float(cyl["radius"])
        z = (P - c) @ axis
        # радиус растёт вдоль оси: вершина — там, где он падает до нуля
        apex = c + (z.mean() - r / np.tan(theta0)) * axis
        starts.append(np.r_[apex, axis, theta0])
    best = None
    for x0 in starts:
        try:
            sol = least_squares(residual, x0, method="lm", max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if np.isfinite(rms) and (best is None or rms < best[1]):
            best = (sol.x, rms)
    if best is None:
        return None, np.inf
    x, rms = best
    a = x[3:6] / np.linalg.norm(x[3:6])
    # отрицательный угол остаток по π переводит в (π/2, π), ось разворачивается ниже, как в `refine`
    theta = float(x[6]) % np.pi
    if theta > np.pi / 2:                   # тот же конус с осью в другую сторону
        a, theta = -a, np.pi - theta
    if not 1e-3 < theta < np.pi / 2 - 1e-3 or np.median((P - x[:3]) @ a) <= 0:
        return None, np.inf
    return dict(kind="конус", apex=x[:3], axis=a, half_angle=np.degrees(theta), rms=rms), rms


FITTERS = {"plane": fits.fit_plane, "cylinder": fits.fit_cylinder, "sphere": fits.fit_sphere,
           "cone": fit_cone, "torus": fits.fit_torus}
# От простого к сложному, как `fits.ORDER`: цилиндр не выдаётся за плоскость.
ORDER = ("plane", "cylinder", "sphere", "cone", "torus")
# Во сколько раз следующий тип должен уменьшить RMS, чтобы заменить текущий
# (`fit_cluster`, когда голоса сети нет или его тип не подогнался). [измерено, 2596 истинных граней val без сплайнов, доля
# верного типа при голосе сети, верном на 88% граней: множитель 0.7 — 0.763,
# 0.5 — 0.837, 0.4 — 0.867, 0.3 — 0.881; без голоса 0.745 / 0.803 / 0.819 / 0.827;
# простой минимум RMS — 0.514 со сплайнами]
KIND_RMS_RATIO = 0.3
# Соседние плоскости с углом меньше этого — кандидаты в одну грань
# (`merge_coplanar`). Решено пользователем (5°; по замеру у половины отказов
# плоскость–плоскость угол меньше 1°, медиана 1.3°).
COPLANAR_DEG = 5.0
# ...и сливаются, если точки каждой лежат на плоскости другой с RMS не больше
# стольких допусков подгонки: иначе это ступенька. Выбрано.
COPLANAR_RMS = 3.0



def triangle_geometry(vertices: np.ndarray, faces: np.ndarray):
    """Центроиды, единичные нормали и площади треугольников."""
    tri = vertices[faces].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    return tri.mean(1), cross / np.maximum(2 * area, 1e-30)[:, None], area


def rms_tolerance(area: np.ndarray, cfg) -> float:
    """Допуск RMS в миллиметрах: `rms_tol_mm`, если задан (шум сканера от размера детали не
    зависит), иначе долей характерной длины детали, как в конфигах до 2026-10-10."""
    if cfg.rms_tol_mm > 0:
        return cfg.rms_tol_mm
    return cfg.rms_tol_rel * float(np.sqrt(area.sum()))


def near_boundary(adjacency: np.ndarray, labels: np.ndarray, rings: int) -> np.ndarray:
    """Треугольники не дальше `rings` колец от чужого кластера."""
    a, b = adjacency[:, 0], adjacency[:, 1]
    cut = labels[a] != labels[b]
    band = np.zeros(len(labels), bool)
    band[a[cut]] = band[b[cut]] = True
    for _ in range(rings - 1):
        grown = band.copy()
        grown[a[band[b]]] = grown[b[band[a]]] = True
        band = grown
    return band


def boundary_vertices(faces: np.ndarray, adjacency: np.ndarray, labels: np.ndarray,
                      rings: int, n_vertices: int) -> np.ndarray:
    """Вершины треугольников полосы у границы и вершины, общие для разных
    кластеров: последние бывают и без общего ребра, углом треугольника."""
    per_corner = np.repeat(labels, 3)
    lo = np.full(n_vertices, np.iinfo(np.int64).max)
    hi = np.full(n_vertices, np.iinfo(np.int64).min)
    np.minimum.at(lo, faces.ravel(), per_corner)
    np.maximum.at(hi, faces.ravel(), per_corner)
    out = lo != hi
    out[faces[near_boundary(adjacency, labels, rings)].ravel()] = True
    return out


def absorb_small(labels: np.ndarray, adjacency: np.ndarray, min_triangles: int) -> np.ndarray:
    """Кластеры мельче порога отдать соседу с самой длинной общей границей."""
    labels = labels.copy()
    a, b = adjacency[:, 0], adjacency[:, 1]
    isolated = set()
    while True:
        counts = np.bincount(labels)
        small = [l for l in np.argsort(counts) if 0 < counts[l] < min_triangles and l not in isolated]
        if not small:
            return labels
        l = small[0]
        nb = np.r_[labels[b][labels[a] == l], labels[a][labels[b] == l]]
        nb = nb[nb != l]
        if len(nb) == 0:
            isolated.add(l)
            continue
        labels[labels == l] = np.bincount(nb).argmax()


def cluster_points(label: int, labels: np.ndarray, excluded: np.ndarray, vertices: np.ndarray,
                   faces: np.ndarray, normal: np.ndarray, area: np.ndarray, min_points: int):
    """Вершины кластера: точки, нормали и веса.

    Нормаль и вес вершины собираются только по треугольникам её кластера, иначе
    на границе смешались бы нормали двух граней. Вес — треть площади смежных
    треугольников: рабочий меш адаптивный, и без веса скругления перетягивали
    бы подгонку на себя. Вершины у границы исключены, если без них остаётся
    хотя бы `min_points`.
    """
    tri = faces[labels == label]
    acc_n = np.zeros((len(vertices), 3))
    acc_a = np.zeros(len(vertices))
    tri_normal, tri_area = normal[labels == label], area[labels == label]
    for k in range(3):
        np.add.at(acc_n, tri[:, k], tri_normal * tri_area[:, None])
        np.add.at(acc_a, tri[:, k], tri_area / 3)
    idx = np.unique(tri.ravel())
    inside = idx[~excluded[idx]]
    if len(inside) >= min_points:
        idx = inside
    n = acc_n[idx] / np.maximum(np.linalg.norm(acc_n[idx], axis=1, keepdims=True), 1e-30)
    w = acc_a[idx]
    return vertices[idx].astype(np.float64), n, w / w.mean()


def distance(kind: str, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Знаковое расстояние до поверхности — как `surfaces.distance`, на numpy:
    уточнение зовёт его сотни раз, и torch здесь только мешал бы."""
    if kind == "plane":
        return x @ (p[:3] / np.linalg.norm(p[:3])) + p[3]
    if kind == "sphere":
        return np.linalg.norm(x - p[:3], axis=1) - p[3]
    a = p[3:6] / np.linalg.norm(p[3:6])
    u = x - p[:3]
    z = u @ a
    rho = np.linalg.norm(u - np.outer(z, a), axis=1)
    if kind == "cylinder":
        return rho - p[6]
    if kind == "cone":
        return rho * np.cos(p[6]) - z * np.sin(p[6])
    return np.sqrt((rho - p[6]) ** 2 + z ** 2) - p[7]


def refine(kind: str, params: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Все параметры вместе, по взвешенному точному расстоянию; старт — `fits`.

    Направления нормируются внутри расстояния, поэтому их длина и сдвиг точки
    вдоль оси цилиндра свободны; после решения они приводятся к виду
    `surfaces`: единичные направления, у цилиндра `a·c = 0`.
    """
    sw = np.sqrt(weights)
    method = "lm" if len(points) > len(params) else "trf"
    sol = least_squares(lambda q: sw * distance(kind, q, points), params, method=method, max_nfev=100)
    q = sol.x.copy()
    if kind == "plane":
        n = np.linalg.norm(q[:3])
        q[:3] /= n
    elif kind in S.AXIAL:
        q[3:6] /= np.linalg.norm(q[3:6])
        if kind == "cylinder":
            q[:3] -= (q[:3] @ q[3:6]) * q[3:6]
            q[6] = abs(q[6])
        if kind == "torus":
            q[6:8] = np.abs(q[6:8])
        if kind == "cone":
            # тот же конус: угол в (0, 90°], ось — в сторону раскрытия к точкам
            theta = q[6] % np.pi
            if theta > np.pi / 2:
                q[3:6], theta = -q[3:6], np.pi - theta
            q[6] = theta
            if np.median((points - q[:3]) @ q[3:6]) < 0:
                q[3:6] = -q[3:6]
    elif kind == "sphere":
        q[3] = abs(q[3])
    return q


def _cone_limit(params: np.ndarray) -> str | None:
    """Конус на пределе (`surfaces.CONE_LIMIT_DEG`): почти без раствора —
    цилиндр, почти плоский — плоскость поперёк оси. Подгонка конусом на таких
    гранях даёт вершину в километрах или угол за 90°, на которых не строится
    поверхность OCC [24474_2e83b11c_0 по разметке]."""
    theta = abs(np.degrees(params[6])) % 180
    theta = min(theta, 180 - theta)
    if theta < S.CONE_LIMIT_DEG:
        return "cylinder"
    if theta > 90 - S.CONE_LIMIT_DEG:
        return "plane"
    return None


def fit_kind(kind: str, points: np.ndarray, normals: np.ndarray, weights: np.ndarray) -> Surface | None:
    """Старт `fits.py`, уточнение `refine` и взвешенная RMS результата — та же,
    что потом сравнивается при совместной подгонке. Конус на пределе
    подгоняется цилиндром или плоскостью."""
    try:
        fit, _ = FITTERS[kind](points, normals)
        if fit is None:
            return None
        params = S.from_fit(kind, fit)
        rms = S.rms(kind, params, points, weights)
        refined = refine(kind, params, points, weights)
        rms_refined = S.rms(kind, refined, points, weights)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    if np.isfinite(rms_refined) and rms_refined < rms:
        params, rms = refined, rms_refined
    if kind == "cone" and (limit := _cone_limit(params)) is not None:
        return fit_kind(limit, points, normals, weights)
    return Surface(kind, params, rms)


def fit_cluster(points: np.ndarray, normals: np.ndarray, weights: np.ndarray,
                vote: str | None) -> Surface | None:
    """Тип поверхности — тот, за который голосует сеть (у разметки — истинный).

    Если голоса нет (у разметки так помечены сплайны) или подгонка этого типа
    не сошлась, типы перебираются от простого к сложному, и следующий
    заменяет текущий, только если его взвешенная RMS меньше `KIND_RMS_RATIO`
    от текущей: у более сложного типа больше параметров, и на шуме его RMS
    почти всегда чуть ниже.
    """
    if vote in FITTERS:
        surface = fit_kind(vote, points, normals, weights)
        if surface is not None and np.isfinite(surface.rms):
            return surface
    best = None
    for kind in ORDER:
        surface = fit_kind(kind, points, normals, weights)
        if surface is None or not np.isfinite(surface.rms):
            continue
        if best is None or surface.rms < KIND_RMS_RATIO * best.rms:
            best = surface
    return best


def cluster_pairs(labels: np.ndarray, adjacency: np.ndarray) -> set[tuple[int, int]]:
    la, lb = labels[adjacency[:, 0]], labels[adjacency[:, 1]]
    cut = la != lb
    return set(map(tuple, np.unique(np.sort(np.c_[la[cut], lb[cut]], axis=1), axis=0).tolist()))


def fit_clusters(vertices: np.ndarray, faces: np.ndarray, adjacency: np.ndarray,
                 labels: np.ndarray, votes: np.ndarray, cfg):
    """Кластеры -> поверхности.

    `votes` — тип на треугольник индексом в `PRIMITIVE_TYPES`, -1 неизвестен.
    Возвращает метки после поглощения мелких и слияния, поверхности по метке
    и сводку.
    """
    _, normal, area = triangle_geometry(vertices, faces)
    tol = rms_tolerance(area, cfg)
    labels = absorb_small(labels, adjacency, cfg.min_triangles)
    excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, len(vertices))

    surfaces = {}
    for l in np.unique(labels):
        pts = cluster_points(l, labels, excluded, vertices, faces, normal, area, cfg.min_fit_points)
        v = votes[labels == l]
        v = v[v >= 0]
        vote = PRIMITIVE_TYPES[np.bincount(v).argmax()] if len(v) else None
        fitted = fit_cluster(*pts, vote)
        if fitted is not None:
            surfaces[int(l)] = fitted
    n_clusters = len(np.unique(labels))

    labels, merges = merge_splits(labels, surfaces, adjacency, vertices, faces, normal, area, tol, cfg)
    labels, coplanar = merge_coplanar(labels, surfaces, adjacency, vertices, faces, normal, area, tol, cfg)
    return labels, surfaces, {"clusters": n_clusters, "merges": merges, "coplanar": coplanar, "tolerance": tol,
                              "unfitted": sorted(set(np.unique(labels).tolist()) - set(surfaces)),
                              "approximate": sorted(l for l, s in surfaces.items() if s.rms > tol)}


def merge_splits(labels, surfaces, adjacency, vertices, faces, normal, area, tol, cfg):
    """Лишние разрезы: соседей одного типа слить, если общий примитив в допуске.

    Жадно, от меньшей невязки к большей. Ложное слияние необратимо, поэтому
    допуск держать обязана каждая часть по отдельности: в RMS объединения
    мелкая часть почти не весит, и отклонение её одной было бы не видно.

    На разметке это тоже сливает грани, и правильно: соседние грани B-Rep на
    одной плоскости, разделённые историей построения [121701_05cb78e4_0 — 10
    слияний, у слитых граней смещения плоскостей расходятся не больше 0.03 мм].
    """
    labels = labels.copy()
    n_vertices = len(vertices)

    def points(l, lab, excluded):
        return cluster_points(l, lab, excluded, vertices, faces, normal, area, cfg.min_fit_points)

    excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, n_vertices)
    cache, merges = {}, 0
    while True:
        best = None
        for a, b in cluster_pairs(labels, adjacency):
            sa, sb = surfaces.get(a), surfaces.get(b)
            if sa is None or sb is None or sa.kind != sb.kind or sa.kind not in FITTERS:
                continue
            if (a, b) not in cache:
                merged = np.where(labels == b, a, labels)
                union = points(a, merged, boundary_vertices(faces, adjacency, merged,
                                                            cfg.boundary_rings, n_vertices))
                fitted = fit_kind(sa.kind, *union)
                if fitted is not None:
                    fitted.rms = max(S.rms(fitted.kind, fitted.params, p, w)
                                     for p, _, w in (points(a, labels, excluded), points(b, labels, excluded)))
                cache[(a, b)] = fitted
            fitted = cache[(a, b)]
            if fitted is not None and fitted.rms <= tol and (best is None or fitted.rms < best[0].rms):
                best = (fitted, a, b)
        if best is None:
            return labels, merges
        fitted, a, b = best
        labels[labels == b] = a
        excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, n_vertices)
        p, _, w = points(a, labels, excluded)
        surfaces[a] = Surface(fitted.kind, fitted.params, S.rms(fitted.kind, fitted.params, p, w))
        del surfaces[b]
        cache = {k: v for k, v in cache.items() if a not in k and b not in k}
        merges += 1


def merge_coplanar(labels, surfaces, adjacency, vertices, faces, normal, area, tol, cfg):
    """Соседние плоскости с углом меньше `COPLANAR_DEG`, лежащие одна на другой,
    слить в одну грань.

    Прямая пересечения почти параллельных плоскостей уходит на сотни рёбер от
    их общей границы, и ребро отбраковывается [измерено, 120 деталей val по
    разметке: у половины отказов «кривая далеко» на парах плоскость–плоскость
    угол меньше 1°, медиана промаха 13 рёбер, p75 — 166; в основном одна
    плоскость, разрезанная историей построения]. `merge_splits` их не сливал:
    он требует, чтобы общая плоскость держала допуск на каждой части. «Лежат
    одна на другой» — точки каждого кластера на плоскости соседа с RMS не
    больше `COPLANAR_RMS` допусков; иначе это ступенька, а не одна грань.
    Жадно, от меньшего угла к большему; после слияния плоскость подгоняется
    заново по всем точкам.
    """
    labels = labels.copy()
    excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, len(vertices))

    def pts(l):
        p, n, w = cluster_points(l, labels, excluded, vertices, faces, normal, area, cfg.min_fit_points)
        return p, n, w

    merged = 0
    while True:
        best = None
        for a, b in cluster_pairs(labels, adjacency):
            sa, sb = surfaces.get(a), surfaces.get(b)
            if sa is None or sb is None or sa.kind != "plane" or sb.kind != "plane":
                continue
            na, nb = sa.params[:3] / np.linalg.norm(sa.params[:3]), sb.params[:3] / np.linalg.norm(sb.params[:3])
            angle = float(np.degrees(np.arccos(min(1.0, abs(na @ nb)))))
            if angle >= COPLANAR_DEG:
                continue
            (pa, _, wa), (pb, _, wb) = pts(a), pts(b)
            if max(S.rms("plane", sa.params, pb, wb), S.rms("plane", sb.params, pa, wa)) > COPLANAR_RMS * tol:
                continue
            if best is None or angle < best[0]:
                best = (angle, a, b)
        if best is None:
            return labels, merged
        _, a, b = best
        labels[labels == b] = a
        del surfaces[b]
        excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, len(vertices))
        refit = fit_kind("plane", *pts(a))
        if refit is not None:
            surfaces[a] = refit
        merged += 1


def samples(labels: np.ndarray, adjacency: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
            surfaces: dict, cfg) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Точки и веса совместной подгонки — те же вершины, что и при подгонке."""
    _, normal, area = triangle_geometry(vertices, faces)
    excluded = boundary_vertices(faces, adjacency, labels, cfg.boundary_rings, len(vertices))
    out = {}
    for l in surfaces:
        p, _, w = cluster_points(l, labels, excluded, vertices, faces, normal, area, cfg.min_fit_points)
        out[l] = (p, w)
    return out
