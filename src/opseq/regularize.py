"""Выпрямление детали перед ядром (README, «Выпрямление: отношения пар перед ядром»).

Почти параллельные плоскости по подгонке скана делаются параллельными, почти перпендикулярные —
перпендикулярными и так далее (решено пользователем 2026-10-26 вместо отрезания лишнего у
недошедших выдавливаний; отношения — голова графформера, решено 2026-10-22). Отношения пока —
метки по точной геометрии STEP (`relations` графа). Совместная подгонка `src/constraints.py` на
десятках условий по всем парам уводила плоскости на 2 мм [100243_9fb796fe_0: одна
перпендикулярность поверх групп параллельных — RMS в 3.9 раза больше], поэтому защёлка по шагам:
направления групп параллельных -> перпендикулярность между группами (крупные закрепляются первыми)
-> смещения плоскостей и оси цилиндров заново по точкам при закреплённых направлениях ->
компланарность, соосность, равные радиусы — общими параметрами групп.
"""
from __future__ import annotations

import numpy as np
from trimesh import Trimesh

from src import fitting as F
from src import surfaces as S
from src.config import CFG
from src.surfaces import AXIAL, Surface

from .data import RELATIONS

DIRECTED = ("plane", *AXIAL)
# Две закреплённые перпендикулярные группы почти параллельны — их векторное произведение не берётся. Выбрано.
MIN_SIN = 0.1


def fit_nodes(g: dict, v: np.ndarray, f: np.ndarray, tri_node: np.ndarray, types: dict[int, str]
              ) -> tuple[dict[int, Surface], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Подгонка узлов графа (ключи — номера узлов графа) по вершинам скана без полосы у границы, как
    в `build_graph`; сплайны и неудачные подгонки пропускаются."""
    _, normal, area = F.triangle_geometry(v, f)
    excluded = F.boundary_vertices(f, Trimesh(v, f, process=False).face_adjacency, tri_node,
                                   CFG.brep.boundary_rings, len(v))
    surfaces, samples = {}, {}
    for i, m in enumerate(g["ids"]):
        kind = types.get(int(m))
        if kind not in DIRECTED + ("sphere",):
            continue
        pts, nrm, w = F.cluster_points(int(m), tri_node, excluded, v, f, normal, area, CFG.brep.min_fit_points)
        if len(pts) < 3:
            continue
        s = F.fit_kind(kind, pts, nrm, w)
        if s is not None and s.kind == kind and np.isfinite(s.rms):
            surfaces[i], samples[i] = s, (pts, w)
    return surfaces, samples


def _groups(ids: list[int], linked) -> dict[int, int]:
    """Компоненты по отношению: узел -> представитель (наименьший номер)."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in ids:
        for b in ids:
            if a < b and linked(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def _direction(s: Surface) -> np.ndarray:
    d = s.params[:3] if s.kind == "plane" else s.params[3:6]
    return d / np.linalg.norm(d)


def _circle(p2: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Взвешенная окружность (Kåsa) по точкам в плоскости."""
    sw = np.sqrt(w)
    sol = np.linalg.lstsq(np.c_[2 * p2, np.ones(len(p2))] * sw[:, None], (p2 ** 2).sum(1) * sw, rcond=None)[0]
    return sol[:2], float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 0.0)))


def _coverage(p2: np.ndarray, c: np.ndarray) -> float:
    """Угловой охват точек вокруг центра, доля окружности (по 72 секторам)."""
    ang = np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0])
    return len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 72))) / 72


def _axis_frame(a: np.ndarray) -> np.ndarray:
    e = np.eye(3)[np.argmin(np.abs(a))]
    e1 = np.cross(a, e)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(a, e1)])


def regularize(g: dict, v: np.ndarray, f: np.ndarray, tri_node: np.ndarray, types: dict[int, str]
               ) -> tuple[dict[int, Surface], dict]:
    """Уточнённые поверхности узлов графа и сводка (сколько групп и насколько выросла RMS)."""
    surfaces, samples = fit_nodes(g, v, f, tri_node, types)
    labels = g["relations"]
    k = {r: RELATIONS.index(r) for r in RELATIONS}
    ids = sorted(i for i, s in surfaces.items() if s.kind in DIRECTED)
    weight = {i: float(samples[i][1].sum()) for i in surfaces}
    u = {i: _direction(surfaces[i]) for i in ids}

    # 1. направления групп параллельных: взвешенное среднее с выравниванием знака по нормалям
    # плоскостей группы, оси — только в группе без плоскостей: у короткой дуги (полоса скругления)
    # ось по скану бывает повёрнута на 83-88° от точной при малой RMS [100243_9fb796fe_1]
    par = _groups(ids, lambda a, b: labels[a, b, k["parallel"]])
    reps = sorted(set(par.values()), key=lambda r: -sum(weight[i] for i in ids if par[i] == r))
    gdir = {}
    for r in reps:
        mem = [i for i in ids if par[i] == r]
        use = [i for i in mem if surfaces[i].kind == "plane"] or mem
        ref = max(use, key=weight.get)
        acc = sum(weight[i] * np.sign(u[i] @ u[ref] or 1) * u[i] for i in use)
        gdir[r] = acc / np.linalg.norm(acc)
    # 2. перпендикулярность между группами: крупные закрепляются первыми
    perp = {(par[a], par[b]) for a in ids for b in ids if labels[a, b, k["perpendicular"]] and par[a] != par[b]}
    fixed = []
    for r in reps:
        ps = [gdir[q] for q in fixed if (r, q) in perp or (q, r) in perp]
        d = gdir[r]
        if len(ps) >= 2:
            c = np.cross(ps[0], ps[1])
            if np.linalg.norm(c) > MIN_SIN:
                d = c * np.sign(c @ d or 1)
            else:
                d = d - (d @ ps[0]) * ps[0]
        elif ps:
            d = d - (d @ ps[0]) * ps[0]
        gdir[r] = d / np.linalg.norm(d)
        fixed.append(r)
    for i in ids:
        u[i] = gdir[par[i]] * np.sign(gdir[par[i]] @ u[i] or 1)

    # 3. плоскости: смещение по точкам, у компланарных — общее; цилиндры: ось закреплена,
    # центр и радиус — окружностью поперёк оси, у соосных — общая ось
    out = dict(surfaces)
    planes = [i for i in ids if surfaces[i].kind == "plane"]
    cop = _groups(planes, lambda a, b: labels[a, b, k["coplanar"]])
    for r in set(cop.values()):
        mem = [i for i in planes if cop[i] == r]
        n = u[r]
        x = np.concatenate([samples[i][0] for i in mem])
        w = np.concatenate([samples[i][1] for i in mem])
        off = -float((x @ n * w).sum() / w.sum())
        for i in mem:
            s = np.sign(u[i] @ n or 1)
            out[i] = Surface("plane", np.r_[n * s, off * s], 0.0)
    cyl = [i for i in ids if surfaces[i].kind == "cylinder"]
    coax = _groups(cyl, lambda a, b: labels[a, b, k["coaxial"]])
    for r in set(coax.values()):
        mem = [i for i in cyl if coax[i] == r]
        a = u[r]
        fr = _axis_frame(a)
        if len(mem) == 1:
            # одиночный: центр и радиус заново окружностью поперёк новой оси — старый центр лежал на
            # старой оси, у дуги она могла быть почти поперёк
            c2, r_i = _circle(samples[r][0] @ fr.T, samples[r][1])
            out[r] = Surface("cylinder", np.r_[c2 @ fr, a * np.sign(u[r] @ a or 1), r_i], 0.0)
            continue
        # центр — у члена с наибольшим охватом дуги: у короткой дуги (полоса скругления) центр
        # почти не определён и уводил ось всей группы (среднее центров, общая подгонка Kåsa —
        # RMS цилиндров до 7 мм на 100243_9fb796fe_1)
        own = {i: _circle(samples[i][0] @ fr.T, samples[i][1])[0] for i in mem}
        best = max(mem, key=lambda i: _coverage(samples[i][0] @ fr.T, own[i]))
        c2 = own[best]
        for i in mem:
            r_i = float(np.average(np.linalg.norm(samples[i][0] @ fr.T - c2, axis=1), weights=samples[i][1]))
            out[i] = Surface("cylinder", np.r_[c2 @ fr, a * np.sign(u[i] @ a or 1), r_i], 0.0)
    # конусы и торы не выпрямляются: смена оси без сдвига вершины и центра раздувала RMS (p90 в
    # 11-150 раз), а для эскизов выдавливания они почти не нужны
    # 4. равные радиусы цилиндров — радиус члена с наибольшим охватом дуги (среднее портила
    # короткая дуга с плохим радиусом)
    eq = _groups(cyl, lambda a, b: labels[a, b, k["equal_radius"]])
    for r in set(eq.values()):
        mem = [i for i in cyl if eq[i] == r]
        if len(mem) > 1:
            def cover(i):
                fr = _axis_frame(out[i].params[3:6])
                return _coverage(samples[i][0] @ fr.T, out[i].params[:3] @ fr.T)
            rad = out[max(mem, key=cover)].params[6]
            for i in mem:
                out[i] = Surface("cylinder", np.r_[out[i].params[:6], rad], 0.0)

    growth = []
    for i, s in out.items():
        rms = S.rms(s.kind, s.params, *samples[i])
        out[i] = Surface(s.kind, s.params, rms)
        growth.append(rms / max(surfaces[i].rms, 1e-12))
    summary = {"групп параллельных": len(reps), "перпендикулярных пар групп": len({tuple(sorted(q)) for q in perp}),
               "рост RMS, медиана / максимум": [float(np.median(growth)), float(np.max(growth))] if growth else None}
    return out, summary
