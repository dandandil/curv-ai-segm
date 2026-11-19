"""Сборка B-Rep из подогнанных поверхностей и границ кластеров, экспорт STEP.

Узел границы -> вершина B-Rep: ближайшая общая точка сходящихся поверхностей.
Цепочка между узлами -> ребро на кривой пересечения двух поверхностей; у
касательной пары кривая строится аналитически, у остальных — `GeomAPI_IntSS`,
и из его ветвей берётся ближайшая к границе на меше. Рёбра кластера -> контуры
грани, грани с общими рёбрами -> оболочка, замкнутая оболочка -> тело.

Если кривая не находится или уходит от границы на меше дальше допуска, или
соседний кластер без примитива, ребро строится по самой границе меша,
спроецированной на поверхность: так грань сохраняет замкнутый контур, а дыра
остаётся только на месте непостроенного соседа.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeVertex)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Geom import (Geom_Circle, Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Line,
                      Geom_Plane, Geom_SphericalSurface, Geom_ToroidalSurface, Geom_TrimmedCurve)
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Shell, TopoDS_Wire
from OCP.TopTools import TopTools_HSequenceOfShape

from . import constraints as C
from . import fitting as F
from . import surfaces as S
from .surfaces import Surface
from .topology import Boundaries

# Точность совпадения в OCC, мм: Precision::Confusion.
CONFUSION = 1e-7
# Соотношения, при которых обе поверхности — тела вращения вокруг общей оси
# (плоскость поперёк оси тоже), и пересекаются они по окружностям.
REVOLVING = ("coaxial", "axis_along_normal")
# Сколько точек границы сверять с кривой.
CHECK_POINTS = 32
# Допуск расстояния от линии касания до границы на меше, в радиусах скругления
# (наименьший радиус пары). Линия касания точная по построению, а граница
# кластеров на сглаженном меше съезжает внутрь скругления на долю радиуса, и
# проверка в рёбрах меша отбраковывала верные рёбра [измерено, 120 деталей val
# по разметке: 104 отказа «кривая далеко» на касательных стыках, промах
# p25/50/90 — 0.19 / 0.22 / 0.28 радиуса, в рёбрах меша медиана 3.2]. Выбрано с
# запасом над p90.
TANGENT_TOL_RADII = 0.5
# Участок границы между параллельными плоскостями (`fitting.COPLANAR_DEG`) не
# длиннее стольких рёбер меша стягивается в узел: это конец ступеньки, где две
# параллельные плоскости касаются, а общего ребра у них нет [120 деталей val по
# разметке: 14 отказов «кривой нет» на парах плоскость–плоскость с углом 0°].
# Длинный не трогается: его стягивание исказило бы соседние рёбра. Выбрано.
PARALLEL_CHAIN_EDGES = 10
# Запас допуска вершины сверх её расстояния до кривой: без него округление в
# OCC оставляет вершину на самой границе допуска.
VERTEX_TOL_MARGIN = 1.01
# Угловой прогиб тесселяции для просмотра, рад: грубее датасетных 0.2, картинке
# точность признаков не нужна.
VIEW_ANGULAR = 0.5


def _pnt(x) -> gp_Pnt:
    return gp_Pnt(*map(float, x))


def _dir(x) -> gp_Dir:
    return gp_Dir(*map(float, x))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def geom_surface(s: Surface, points: np.ndarray):
    """Бесконечная поверхность OCC. Конусу нужны точки: опорная окружность
    ставится посреди кластера, а не в вершине, где поверхность вырождена."""
    p = s.params
    if s.kind == "plane":
        n = _unit(p[:3])
        return Geom_Plane(gp_Ax3(_pnt(-p[3] / np.linalg.norm(p[:3]) * n), _dir(n)))
    if s.kind == "sphere":
        return Geom_SphericalSurface(gp_Ax3(_pnt(p[:3]), gp_Dir(0, 0, 1)), float(p[3]))
    a = _unit(p[3:6])
    if s.kind == "cylinder":
        return Geom_CylindricalSurface(gp_Ax3(_pnt(p[:3]), _dir(a)), float(p[6]))
    if s.kind == "torus":
        return Geom_ToroidalSurface(gp_Ax3(_pnt(p[:3]), _dir(a)), float(p[6]), float(p[7]))
    theta = float(p[6])
    if theta < 0:
        a, theta = -a, -theta
    z0 = float(np.median((points - p[:3]) @ a))
    if z0 <= 0 or not 0 < theta < np.pi / 2:
        return None
    return Geom_ConicalSurface(gp_Ax3(_pnt(p[:3] + z0 * a), _dir(a)), theta, z0 * np.tan(theta))


def _parallel_planes(surfaces: dict[int, Surface], a: int, b: int) -> bool:
    if a not in surfaces or b not in surfaces or surfaces[a].kind != "plane" or surfaces[b].kind != "plane":
        return False
    na, nb = (_unit(surfaces[l].params[:3]) for l in (a, b))
    return float(np.degrees(np.arccos(min(1.0, abs(na @ nb))))) < F.COPLANAR_DEG


def _radius(*pair: Surface) -> float:
    """Наименьший радиус кривизны пары: цилиндр, малый радиус тора, сфера."""
    radii = [s.params[6] for s in pair if s.kind == "cylinder"] + \
            [s.params[7] for s in pair if s.kind == "torus"] + [s.params[3] for s in pair if s.kind == "sphere"]
    return float(min(radii)) if radii else 0.0


def contact_curve(rel: C.Relation, surfaces: dict[int, Surface]):
    """Линия касания пары, по точным после совместной подгонки параметрам."""
    p, q = surfaces[rel.a].params, surfaces[rel.b].params
    if rel.name == "plane_offset":
        n = _unit(p[:3])
        foot = q[:3] - (n @ q[:3] + p[3] / np.linalg.norm(p[:3])) * n
        a = _unit(q[3:6])
        if surfaces[rel.b].kind == "cylinder":
            return Geom_Line(_pnt(foot), _dir(a))
        return Geom_Circle(gp_Ax2(_pnt(foot), _dir(a)), float(q[6]))
    if rel.name == "cylinders_tangent":
        a = _unit(p[3:6])
        u = q[:3] - p[:3]
        u = _unit(u - (u @ a) * a)
        return Geom_Line(_pnt(p[:3] + rel.signs[0] * p[6] * u), _dir(a))
    if rel.name == "torus_cylinder_tangent":
        return Geom_Circle(gp_Ax2(_pnt(p[:3]), _dir(_unit(p[3:6]))), float(q[6]))
    if rel.name == "torus_cone_tangent":
        a, theta = _unit(q[3:6]), q[6]
        g = ((p[:3] - q[:3]) @ a) * np.cos(theta) + p[6] * np.sin(theta)
        return Geom_Circle(gp_Ax2(_pnt(q[:3] + g * np.cos(theta) * a), _dir(a)), float(g * np.sin(theta)))
    if rel.name == "equal_radius":
        return Geom_Circle(gp_Ax2(_pnt(p[:3]), _dir(_unit(q[3:6]))), float(q[6]))
    if rel.name == "torus_cylinder_elbow":
        # окружность цилиндра там, где его ось касается осевой окружности тора
        a = _unit(p[3:6])
        foot = p[:3] + ((q[:3] - p[:3]) @ a) * a
        return Geom_Circle(gp_Ax2(_pnt(foot), _dir(a)), float(p[6]))
    if rel.name == "plane_cone_tangent":
        # образующая конуса в плоскости: через вершину, по проекции оси на плоскость
        n, a = _unit(p[:3]), _unit(q[3:6])
        return Geom_Line(_pnt(q[:3]), _dir(_unit(a - (a @ n) * n)))
    raise ValueError(rel.name)


def coaxial_circle(rel: C.Relation, surfaces: dict[int, Surface], poly: np.ndarray):
    """Пересечение соосных тел вращения (или тела с плоскостью поперёк оси) —
    окружность; ближайшая к границе на меше.

    `GeomAPI_IntSS` режет такие окружности на дуги по швам поверхностей, и ни
    одна дуга границу целиком не покрывает [измерено: сфера–цилиндр на
    32566_7bfc936b_5 — четыре сплайна, ближайший в 1.7 мм при допуске 0.84].
    Здесь точка пересечения ищется от вершины границы, а окружность задаётся
    её осевой координатой и расстоянием до оси.
    """
    ref = surfaces[rel.b if rel.name == "axis_along_normal" else rel.a].params
    origin, axis = ref[:3], _unit(ref[3:6])
    pair = [surfaces[rel.a], surfaces[rel.b]]
    x = S.common_point(pair, poly[len(poly) // 2])
    if max(abs(S.rms(s.kind, s.params, x[None], np.ones(1))) for s in pair) > CONFUSION:
        return None
    z = (x - origin) @ axis
    radius = float(np.linalg.norm(x - origin - z * axis))
    if radius < CONFUSION:
        return None
    return Geom_Circle(gp_Ax2(_pnt(origin + z * axis), _dir(axis)), radius)


def _untrimmed(curve):
    """`GeomAPI_IntSS` отдаёт окружность и прямую обрезанными. Концы рёбер
    ставятся по вершинам сами, а обрезка прячет периодичность: полная
    окружность в ней не периодична и не замкнута."""
    return curve.BasisCurve() if isinstance(curve, Geom_TrimmedCurve) else curve


def _project(curve, x: np.ndarray) -> tuple[float, float]:
    """Параметр и расстояние ближайшей точки кривой."""
    proj = GeomAPI_ProjectPointOnCurve(_pnt(x), curve)
    if proj.NbPoints() == 0:
        return float("nan"), float("inf")
    return proj.LowerDistanceParameter(), proj.LowerDistance()


def _mean_distance(curve, poly: np.ndarray) -> float:
    idx = np.linspace(0, len(poly) - 1, min(len(poly), CHECK_POINTS)).astype(int)
    return float(np.mean([_project(curve, x)[1] for x in poly[idx]]))


def _attach(vertex, distance: float) -> None:
    """Допуск вершины должен покрывать её расстояние до кривой ребра."""
    tol = max(distance * VERTEX_TOL_MARGIN, CONFUSION, BRep_Tool.Tolerance_s(vertex))
    BRep_Builder().UpdateVertex(vertex, tol)


def _span_edge(curve, poly: np.ndarray):
    """Ребро по всей кривой, на которую ложится граница: от наименьшего
    параметра её точек до наибольшего.

    Нужно там, где у участка границы нет двух разных узлов, а кривая не
    периодична: линия касания плоскости с цилиндром, ветвь гиперболы,
    сплайн из пересечения. Раньше такой участок ребра не получал, и грань не
    собиралась [измерено, 20 деталей val по разметке: 46 из 50 непостроенных
    рёбер — именно эти случаи].
    """
    idx = np.linspace(0, len(poly) - 1, min(len(poly), CHECK_POINTS)).astype(int)
    ts = np.array([_project(curve, x)[0] for x in poly[idx]])
    ts = ts[np.isfinite(ts)]
    if len(ts) < 2 or ts.max() - ts.min() < CONFUSION:
        return None
    return BRepBuilderAPI_MakeEdge(curve, float(ts.min()), float(ts.max()))


def curve_edge(curve, poly: np.ndarray, ends, nodes):
    """Ребро на кривой между двумя вершинами или замкнутое, если концов нет.

    На периодической кривой между двумя точками две дуги; берётся та, на
    которую проецируется середина границы меша.
    """
    if ends is None:
        if curve.IsPeriodic():
            t0 = curve.FirstParameter()
            return BRepBuilderAPI_MakeEdge(curve, t0, t0 + curve.Period())
        if curve.IsClosed():
            return BRepBuilderAPI_MakeEdge(curve, curve.FirstParameter(), curve.LastParameter())
        return _span_edge(curve, poly)
    (x0, v0), (x1, v1) = nodes[ends[0]], nodes[ends[1]]
    t0, d0 = _project(curve, x0)
    t1, d1 = _project(curve, x1)
    # середина средней хорды, а не средняя вершина: у участка из двух вершин
    # средняя совпала бы с концом, и дуга выбиралась бы наугад
    tm, _ = _project(curve, poly[(len(poly) - 1) // 2: len(poly) // 2 + 1].mean(0))
    if not np.isfinite([t0, t1, tm]).all():
        return None
    if curve.IsPeriodic():
        period = curve.Period()
        t1 = t0 + period if ends[0] == ends[1] else t0 + (t1 - t0) % period
        if t0 + (tm - t0) % period > t1:
            t0, t1, v0, v1, d0, d1 = t1 - period, t0, v1, v0, d1, d0
    elif t0 > t1:
        t0, t1, v0, v1, d0, d1 = t1, t0, v1, v0, d1, d0
    if t1 - t0 < CONFUSION:
        # Оба конца в одном узле на непериодической кривой: длину ребра задаёт
        # сама граница на меше, а не параметры концов.
        return _span_edge(curve, poly)
    _attach(v0, d0)
    _attach(v1, d1)
    return BRepBuilderAPI_MakeEdge(curve, v0, v1, t0, t1)


def build(mesh: trimesh.Trimesh, labels: np.ndarray, surfaces: dict[int, Surface],
          relations: list[C.Relation], bounds: Boundaries, cfg, misses: list | None = None):
    """Поверхности и границы -> форма OCC и сводка по шагам сборки.

    Ребро строится только как настоящая кривая: пересечение поверхностей,
    линия касания или окружность у соосной пары. Если кривая не вышла или
    отошла от границы на меше дальше `edge_tol_edges`, ребро не строится вовсе.
    `misses` — если задан, сюда пишется каждый отказ: причина, типы пары и
    промах кривой в медианных рёбрах меша (для разбора, где теряются рёбра).
    """
    vertices = np.asarray(mesh.vertices, np.float64)
    centroid, normal, _ = F.triangle_geometry(vertices, np.asarray(mesh.faces))
    edge_len = float(np.median(mesh.edges_unique_length))
    edge_tol = cfg.edge_tol_edges * edge_len
    joint_tol = cfg.joint_tol_rel * float(np.linalg.norm(mesh.extents))
    report = Counter()

    geoms, flip = {}, {}
    for l, s in surfaces.items():
        m = labels == l
        geoms[l] = geom_surface(s, centroid[m])
        flip[l] = float((S.normals(s.kind, s.params, centroid[m]) * normal[m]).sum()) < 0
    surfaces = {l: s for l, s in surfaces.items() if geoms[l] is not None}

    nodes = []
    for members, labs in zip(bounds.nodes, bounds.node_labels):
        fitted = [surfaces[l] for l in labs if l in surfaces]
        x0 = vertices[members].mean(0)
        x = S.common_point(fitted, x0) if fitted else x0
        nodes.append((x, BRepBuilderAPI_MakeVertex(_pnt(x)).Vertex()))

    # Узлы, сошедшиеся в одну точку, — одна вершина, а короткий участок между
    # ними вырожден: ребро нулевой длины OCC не строит [22211_b1ee53f0_14 по
    # сети — 18 рёбер с BRepBuilderAPI_DifferentPointsOnClosedCurve].
    parent = list(range(len(nodes)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    degenerate = set()
    for k, ch in enumerate(bounds.chains):
        if ch.ends is None or ch.ends[0] == ch.ends[1]:
            continue
        if _parallel_planes(surfaces, *ch.pair) and len(ch.vertices) - 1 <= PARALLEL_CHAIN_EDGES:
            # конец сходящей на нет ступеньки: параллельные плоскости касаются по
            # короткому участку, общего ребра у них нет — участок стягивается в узел
            parent[root(ch.ends[0])] = root(ch.ends[1])
            degenerate.add(k)
            continue
        if np.linalg.norm(nodes[ch.ends[0]][0] - nodes[ch.ends[1]][0]) < joint_tol:
            parent[root(ch.ends[0])] = root(ch.ends[1])
            poly = vertices[ch.vertices]
            if np.linalg.norm(np.diff(poly, axis=0), axis=1).sum() < edge_tol:
                degenerate.add(k)
    report["вырожденных участков"] = len(degenerate)

    by_pair = defaultdict(dict)
    for r in relations:
        by_pair[frozenset((r.a, r.b))][r.name] = r
    edges = {}
    for k, ch in enumerate(bounds.chains):
        if k in degenerate:
            continue
        a, b = ch.pair
        poly = vertices[ch.vertices]
        ends = None if ch.ends is None else (root(ch.ends[0]), root(ch.ends[1]))
        maker, how = None, None
        if a not in surfaces or b not in surfaces:
            report["ребро: у соседа нет примитива"] += 1
        else:
            rels = by_pair[frozenset(ch.pair)]
            tangent = [r for name, r in rels.items() if name in C.TANGENT]
            revolving = [r for name, r in rels.items() if name in REVOLVING]
            candidates, how = [], "пересечение"
            if tangent:
                candidates, how = [contact_curve(tangent[0], surfaces)], "касание"
            elif revolving:
                candidates = [c for c in [coaxial_circle(revolving[0], surfaces, poly)] if c is not None]
            if not candidates:
                inter = GeomAPI_IntSS(geoms[a], geoms[b], CONFUSION)
                candidates = [_untrimmed(inter.Line(i)) for i in range(1, inter.NbLines() + 1)] \
                    if inter.IsDone() else []
            scored = sorted(((_mean_distance(c, poly), i) for i, c in enumerate(candidates)))
            why = None
            if not scored:
                why = "кривой нет"
            elif scored[0][0] > (max(edge_tol, TANGENT_TOL_RADII * _radius(surfaces[a], surfaces[b]))
                                 if how == "касание" else edge_tol):
                why = "кривая далеко от границы"
            else:
                maker = curve_edge(candidates[scored[0][1]], poly, ends, nodes)
                if maker is None or not maker.IsDone():
                    why = "кривая есть, ребро не вышло"
            if why:
                report[f"ребро: {why}"] += 1
                if misses is not None:
                    misses.append({"why": why, "kinds": sorted([surfaces[a].kind, surfaces[b].kind]),
                                   "pair": [int(a), int(b)], "chain": k,
                                   "how": how, "relations": sorted(rels),
                                   "miss_edges": scored[0][0] / edge_len if scored else None,
                                   "chain_edges": len(poly) - 1, "closed": ch.ends is None})
        if maker is None or not maker.IsDone():
            # Ребро по ломаной границы меша не строится: оно повторяет зубцы
            # нарезки и даёт грань, которой в детали нет. Лучше дыра в
            # оболочке — она видна в отчёте и в метрике, а кривая по мешу
            # молча выдаёт себя за геометрию.
            report["рёбер не построено"] += 1
            continue
        edges[k] = maker.Edge()
        report[f"рёбер {how}"] += 1

    chains_of = defaultdict(list)
    for k, ch in enumerate(bounds.chains):
        if k not in degenerate:
            for l in ch.pair:
                chains_of[l].append(k)
    faces = []
    for l in surfaces:
        face = _face(geoms[l], [edges.get(k) for k in chains_of[l]], joint_tol)
        if face is None:
            report["граней не построено"] += 1
            if misses is not None:
                missing = sum(edges.get(k) is None for k in chains_of[l])
                misses.append({"why": "грань не собралась", "label": int(l), "kind": surfaces[l].kind,
                               "edges": len(chains_of[l]), "missing_edges": missing})
            continue
        if flip[l]:
            face.Reverse()
        faces.append(face)
        report["граней"] += 1
    if not faces:
        return None, dict(report)

    # Сшивка не нужна: вершины и рёбра у соседних граней — одни и те же объекты.
    # BRepBuilderAPI_Sewing на нарезке сети уходил в минуты, ища совпадения,
    # которые и так известны. Исправление (кривые на поверхностях, швы,
    # ориентация) — по оболочке целиком: поштучный ShapeFix_Face резал общее
    # ребро в одной грани, не трогая соседнюю [измерено: 52576_6a9d504d_3 по
    # разметке — до исправления 68 рёбер, все общие; после — 80, из них 20
    # одиночных, оболочка распалась на 9].
    builder, shell = BRep_Builder(), TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    fix = ShapeFix_Shape(shell)
    fix.SetPrecision(CONFUSION)
    fix.SetMaxTolerance(joint_tol)
    fix.Perform()
    shape = fix.Shape()
    shells = _subshapes(shape, TopAbs_SHELL)
    report["оболочек"] = len(shells)
    report["тело"] = False
    if len(shells) == 1 and BRep_Tool.IsClosed_s(shells[0]):
        fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shells[0])).Solid())
        fix.Perform()
        shape = fix.Solid()
        report["тело"] = True
    report["валидна"] = BRepCheck_Analyzer(shape).IsValid()
    return shape, dict(report)


def _face(surface, edges: list, tol: float):
    """Грань на поверхности по её рёбрам; None, если контур не замкнулся."""
    if not edges or any(e is None for e in edges):
        return None
    wires = _wires(edges)
    if wires is None:
        seq, found = TopTools_HSequenceOfShape(), TopTools_HSequenceOfShape()
        for e in edges:
            seq.Append(e)
        ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(seq, tol, True, found)
        wires = [TopoDS.Wire_s(found.Value(i)) for i in range(1, found.Length() + 1)]
    if not wires or not all(BRep_Tool.IsClosed_s(w) for w in wires):
        return None
    if not isinstance(surface, Geom_Plane):
        # У пояса на теле вращения нет внешнего и внутреннего контура — нужны
        # оба круга как есть; ориентацию правит ShapeFix по всей оболочке.
        maker = BRepBuilderAPI_MakeFace(surface, wires[0], True)
        if not maker.IsDone():
            return None
        for w in wires[1:]:
            # второй круг пояса обходится против первого; какой обход нужен,
            # по контурам не видно — берётся тот, с которым грань валидна
            # [24474_2e83b11c_0 по сети: конический пояс из двух кругов не строился]
            for candidate in (w, TopoDS.Wire_s(w.Reversed())):
                trial = BRepBuilderAPI_MakeFace(maker.Face())
                trial.Add(candidate)
                if trial.IsDone() and BRepCheck_Analyzer(trial.Face()).IsValid():
                    maker = trial
                    break
            else:
                trial = BRepBuilderAPI_MakeFace(maker.Face())
                trial.Add(w)
                maker = trial
        return maker.Face() if maker.IsDone() else None
    # Внешний контур — охватывающий наибольшую площадь, а не первый собранный:
    # у грани с отверстиями первой бывала окружность отверстия, и грань
    # строилась диском внутри него [22228_1c82530b_5 по разметке: две грани с
    # отрицательной площадью, оболочка распадалась на 3].
    def span(w):
        trial = BRepBuilderAPI_MakeFace(surface, w, True)
        return abs(_area(trial.Face())) if trial.IsDone() else -1.0

    wires.sort(key=span, reverse=True)
    maker = BRepBuilderAPI_MakeFace(surface, wires[0], True)
    if not maker.IsDone():
        return None
    outer = _area(maker.Face())
    for w in wires[1:]:
        # отверстие обходится против внешнего контура: иначе вычитается не оно
        trial = BRepBuilderAPI_MakeFace(maker.Face())
        trial.Add(w)
        if trial.IsDone() and abs(_area(trial.Face())) < abs(outer):
            maker = trial
        else:
            trial = BRepBuilderAPI_MakeFace(maker.Face())
            trial.Add(TopoDS.Wire_s(w.Reversed()))
            maker = trial
        outer = _area(maker.Face())
    face = maker.Face() if maker.IsDone() else None
    if face is not None and _area(face) < 0:
        face.Reverse()
    return face


def _wires(edges: list) -> list | None:
    """Рёбра -> замкнутые контуры по общим вершинам, без копирования рёбер.

    `ConnectEdgesToWires` возвращает часть рёбер копиями, и общее ребро двух
    граней перестаёт быть общим: оболочка не сшивается [22228_1c82530b_5 по
    разметке: в 4 гранях из 21 подменено 6 рёбер, ровно 6 свободных рёбер в
    оболочке]. Здесь вершины соседних рёбер — одни и те же объекты из узлов
    границы, и контур складывается из самих рёбер. None — если по общим
    вершинам контуры не замкнулись (ребро на незамкнутой кривой без узлов).
    """
    rest = [TopoDS.Edge_s(e) for e in edges]
    ends = [(TopExp.FirstVertex_s(e, True), TopExp.LastVertex_s(e, True)) for e in rest]

    def shared(i, vx):
        return any(j != i and (vx.IsSame(a) or vx.IsSame(b)) for j, (a, b) in enumerate(ends))

    # Обрывок границы у тройного стыка: концы не попали в узлы угла, ни одна
    # вершина не общая с другими рёбрами грани. Он ничего не ограничивает, а у
    # соседней грани то же ребро отбрасывается так же [22228_1c82530b_5: рёбра
    # 0.02-0.2 мм у углов в 4 гранях из 21].
    rest = [e for i, (e, (a, b)) in enumerate(zip(rest, ends)) if a.IsSame(b) or shared(i, a) or shared(i, b)]
    out = []
    while rest:
        edge = rest.pop(0)
        start, end = TopExp.FirstVertex_s(edge, True), TopExp.LastVertex_s(edge, True)
        chain = [edge]
        while not end.IsSame(start):
            for i, e in enumerate(rest):
                first, last = TopExp.FirstVertex_s(e, True), TopExp.LastVertex_s(e, True)
                if first.IsSame(end):
                    chain.append(e)
                    end = last
                elif last.IsSame(end):
                    chain.append(TopoDS.Edge_s(e.Reversed()))
                    end = first
                else:
                    continue
                rest.pop(i)
                break
            else:
                return None
        builder, wire = BRep_Builder(), TopoDS_Wire()
        builder.MakeWire(wire)
        for e in chain:
            builder.Add(wire, e)
        wire.Closed(True)
        out.append(wire)
    return out


def _area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _subshapes(shape: TopoDS_Shape, kind) -> list:
    out, exp = [], TopExp_Explorer(shape, kind)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def write_step(shape: TopoDS_Shape, path: str | Path) -> bool:
    # Статистику переноса OCC пишет из C++ прямо в дескриптор stdout, мимо
    # Message, и она засыпает вывод ячейки — глушится на уровне дескриптора.
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
            return writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def tessellate(shape: TopoDS_Shape, deflection: float) -> trimesh.Trimesh:
    """Форма OCC -> меш, чтобы посмотреть результат тем же рендером."""
    BRepMesh_IncrementalMesh(shape, deflection, False, VIEW_ANGULAR, True)
    verts, tris = [], []
    for face in _subshapes(shape, TopAbs_FACE):
        face = TopoDS.Face_s(face)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf, offset = loc.Transformation(), len(verts)
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        reverse = face.Orientation() == TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = (j - 1 + offset for j in tri.Triangle(i).Get())
            tris.append((a, c, b) if reverse else (a, b, c))
    return trimesh.Trimesh(np.array(verts), np.array(tris), process=False)
