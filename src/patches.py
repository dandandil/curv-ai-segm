"""Сборка тела выбором кусков поверхностей (по мотивам Point2CAD и Mesh2Brep).

1. Каждая поверхность — кусок вокруг своего кластера с запасом `MARGIN_EDGES`
   рёбер меша: прямоугольник параметров по проекциям точек кластера.
2. Все куски режутся друг другом одной общей булевой операцией
   (`BOPAlgo_Builder`): куски локальны, поэтому режут друг друга в основном
   соседи, а рёбра у соседних кусков общие.
3. Покрытие: какая часть площади каждого получившегося кусочка лежит у
   треугольников своего кластера.
4. Выбор кусочков — целочисленная задача (`scipy.optimize.milp`): у каждого
   ребра 0 или 2 выбранных кусочка, цель — пустая площадь минус покрытая.
   Края продлённых кусков принадлежат одному кусочку и отрезаются сами.
5. Выбранные кусочки — оболочка, оболочка — тело.

Замкнутость задаётся условием задачи, а не тем, что сошлись все рёбра по
отдельности (README, «Сборка выбором кусков»).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Vec
from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shell
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ListOfShape
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from . import fitting as FT
from .brep import CONFUSION, _subshapes, contact_curve, geom_surface
from .constraints import TANGENT
from .surfaces import Surface

# Запас куска за свой кластер, в медианных рёбрах меша. Края двух кусков,
# кончающихся ровно на общей кривой, касаются, а не пересекаются, и общая
# булева операция их не режет; запас делает пересечение настоящим. Выбрано.
MARGIN_EDGES = 10.0
# Точек кластера для прямоугольника параметров. Выбрано.
UV_POINTS = 300
# Кусочек покрыт своим кластером там, где до треугольника этого кластера ближе
# max(COVER_TOL · допуск RMS, COVER_EDGES рёбер меша). Выбрано.
COVER_TOL = 3.0
COVER_EDGES = 2.0
# Прогиб тесселяции кусочков для покрытия, в медианных рёбрах меша. Выбрано.
DEFLECTION_EDGES = 1.0
# Плоскости с углом меньше этого и смещением не больше `SAME_PLANE_TOL`
# допусков RMS — одна плоскость: куски почти совпадающих плоскостей
# перекрываются, и общая булева операция дробит их в щепки [22228_1c82530b_5:
# три не соседние грани на одной плоскости со смещениями 0.570-0.602 мм, от
# большой выбран кусочек нулевой площади]. Выбрано.
SAME_PLANE_DEG = 1.0
SAME_PLANE_TOL = 2.0
# Вес площади «в воздухе» против покрытой своим кластером в цели выбора. Выбрано.
AIR_WEIGHT = 1.0
# Сшивка поверхностей в запасном режиме: края ближе стольких допусков подгонки
# склеиваются. Совпадает с порогом покрытия `COVER_TOL`. Выбрано.
SEW_TOL = 3.0
# Запас тесной рамки, по которой в запасном режиме подрезаются кусочки, не
# дорезанные соседом, в рёбрах меша. Совпадает с `COVER_EDGES`: дальше этого от
# кластера покрытия не бывает. Выбрано.
TRIM_EDGES = 2.0
DEBUG = False
DEBUG_LABELS = ()


def _span(values: np.ndarray, period: float | None, pad: float, lo: float, hi: float) -> tuple[float, float]:
    """Отрезок параметра, покрывающий значения, с запасом `pad`. У замкнутого
    направления — дополнение наибольшего разрыва; если разрыв меньше двух
    запасов, кусок берётся на весь период."""
    if period is None:
        return max(values.min() - pad, lo), min(values.max() + pad, hi)
    v = np.sort(np.mod(values - lo, period))
    gaps = np.diff(np.r_[v, v[0] + period])
    k = int(np.argmax(gaps))
    if gaps[k] <= 2 * pad:
        return lo, lo + period
    start = v[(k + 1) % len(v)]
    width = (v[k] - start) % period
    return lo + start - pad, lo + start + width + pad


def _scale(geom, u: float, v: float) -> tuple[float, float]:
    """Длина на единицу параметра по u и v в точке (u, v)."""
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    geom.D1(u, v, p, du, dv)
    return max(du.Magnitude(), CONFUSION), max(dv.Magnitude(), CONFUSION)


def patch(geom, points: np.ndarray, margin: float):
    """Кусок поверхности вокруг точек кластера с запасом `margin` мм."""
    uv = []
    for x in points:
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, x)), geom)
        if proj.NbPoints():
            uv.append(proj.LowerDistanceParameters())
    if len(uv) < 3:
        return None
    uv = np.array(uv)
    u_lo, u_hi, v_lo, v_hi = geom.Bounds()
    su, sv = _scale(geom, *np.median(uv, axis=0))
    pu = geom.UPeriod() if geom.IsUPeriodic() else None
    pv = geom.VPeriod() if geom.IsVPeriodic() else None
    u0, u1 = _span(uv[:, 0], pu, margin / su, u_lo if pu is None else 0.0, u_hi if pu is None else pu)
    v0, v1 = _span(uv[:, 1], pv, margin / sv, v_lo if pv is None else 0.0, v_hi if pv is None else pv)
    maker = BRepBuilderAPI_MakeFace(geom, u0, u1, v0, v1, CONFUSION)
    return maker.Face() if maker.IsDone() else None


def _samples(face, deflection: float):
    """Центры, площади и нормали (с ориентацией грани) треугольников тесселяции."""
    BRepMesh_IncrementalMesh(face, deflection, False, 0.5, True)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    trsf = loc.Transformation()
    nodes = np.array([tri.Node(i).Transformed(trsf).Coord() for i in range(1, tri.NbNodes() + 1)])
    idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
    a, b, c = nodes[idx[:, 0]], nodes[idx[:, 1]], nodes[idx[:, 2]]
    cross = np.cross(b - a, c - a)
    size = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(size, 1e-30)[:, None]
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    return (a + b + c) / 3, size / 2, normal


def plane_groups(surfaces: dict[int, Surface], labels: np.ndarray, mesh: trimesh.Trimesh, tol: float) -> dict[int, int]:
    """Метка -> представитель группы почти совпадающих плоскостей; плоскость
    группы подгоняется заново по вершинам всех её кластеров."""
    planes = [l for l, s in surfaces.items() if s.kind == "plane"]
    parent = {l: l for l in surfaces}

    def root(l):
        while parent[l] != l:
            l = parent[l]
        return l

    for i, a in enumerate(planes):
        for b in planes[i + 1:]:
            na = surfaces[a].params[:3] / np.linalg.norm(surfaces[a].params[:3])
            nb = surfaces[b].params[:3] / np.linalg.norm(surfaces[b].params[:3])
            if np.degrees(np.arccos(min(1.0, abs(na @ nb)))) >= SAME_PLANE_DEG:
                continue
            pb = mesh.vertices[np.unique(mesh.faces[labels == b])]
            pa = mesh.vertices[np.unique(mesh.faces[labels == a])]
            off = max(np.sqrt(np.mean(FT.distance("plane", surfaces[a].params, pb) ** 2)),
                      np.sqrt(np.mean(FT.distance("plane", surfaces[b].params, pa) ** 2)))
            if off <= SAME_PLANE_TOL * tol:
                parent[root(b)] = root(a)
    return {l: root(l) for l in surfaces}


def _contacts(args, mesh, labels, group, surfaces, relations, margin: float) -> int:
    """Линии касания — рёбрами в общую булеву операцию.

    Касательные поверхности общая булева операция не пересекает, и куски
    скругления и его соседа оставались неразрезанными: у кусочка свободный
    край, его нельзя взять, тело не замыкается [3 из 4 разобранных незамкнутых
    деталей val: диск с ободом, скругления-торы без единого кусочка]. После
    совместной подгонки касание точное, линия касания известна
    (`brep.contact_curve`) и лежит на обеих поверхностях — ребро по ней режет
    оба куска. Окружность берётся целиком, прямая — по протяжённости обоих
    кластеров с запасом.
    """
    count = 0
    for rel in relations:
        if rel.name not in TANGENT or rel.a not in surfaces or rel.b not in surfaces:
            continue
        curve = contact_curve(rel, surfaces)
        if curve.IsPeriodic():
            maker = BRepBuilderAPI_MakeEdge(curve)
        else:
            pts = mesh.vertices[np.unique(mesh.faces[np.isin(labels, (rel.a, rel.b))])]
            ts = []
            for x in pts[:: max(1, len(pts) // 200)]:
                proj = GeomAPI_ProjectPointOnCurve(gp_Pnt(*map(float, x)), curve)
                if proj.NbPoints():
                    ts.append(proj.LowerDistanceParameter())
            if len(ts) < 2:
                continue
            maker = BRepBuilderAPI_MakeEdge(curve, min(ts) - margin, max(ts) + margin)
        if maker.IsDone():
            args.Append(maker.Edge())
            count += 1
    return count


def _explain(mesh, labels, group, surfaces, pieces, owner, cover, total, lone, sense, edges, picked, sources):
    """Отладка: у запрещённых покрытых кусочков — чей кусок должен был дорезать
    свободный край и дотягивается ли он; поверхности без кусочка."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    patch_of = {l: f for f, l in sources}
    good = (cover > 0.5 * total) & (cover > 1e-3 * mesh.area)
    for k in sorted(lone):
        if not good[k]:
            continue
        for i, by_piece in sense.items():
            if list(by_piece) != [k] or by_piece[k] == 0:
                continue
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(edges.FindKey(i)))
            mid = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()))
            x = np.array(mid.Coord())
            _, d, tri = trimesh.proximity.closest_point(mesh, x[None])
            nb = group.get(int(labels[tri[0]]), -1)
            reach = None
            if nb in patch_of and nb != owner[k]:
                dist = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(mid).Vertex(), patch_of[nb])
                reach = round(dist.Value(), 3) if dist.IsDone() else None
            print(f"  ЗАПРЕЩЁН кусочек {k} пов. {owner[k]} ({surfaces[owner[k]].kind}), покрыт {cover[k]:.2f}/{total[k]:.2f}: "
                  f"свободный край у кластера {nb} ({surfaces[nb].kind if nb in surfaces else '-'}), "
                  f"край от меша {d[0]:.3f} мм, кусок соседа от края {reach} мм, "
                  f"концы {np.round(curve.Value(curve.FirstParameter()).Coord(), 3)} {np.round(curve.Value(curve.LastParameter()).Coord(), 3)}")
    own = np.array(owner)
    for l in sorted(set(owner)):
        ks = np.flatnonzero(own == l)
        if cover[ks].sum() > 1e-3 * mesh.area and not any(k in picked for k in ks):
            print(f"  БЕЗ КУСОЧКА пов. {l} ({surfaces[l].kind}): покрыто {cover[ks].sum():.2f}, кусочков {len(ks)}, "
                  f"из них запрещено {sum(k in lone for k in ks)}, хорошо покрытых {int(good[ks].sum())}")


def build(mesh: trimesh.Trimesh, labels: np.ndarray, surfaces: dict[int, Surface], relations: list, tol: float):
    """Поверхности после совместной подгонки -> тело и сводка."""
    report = {}
    edge_len = float(np.median(mesh.edges_unique_length))
    margin = MARGIN_EDGES * edge_len
    centroid = mesh.triangles_center
    rng = np.random.default_rng(0)
    group = plane_groups(surfaces, labels, mesh, tol)
    members = {}
    for l, g in group.items():
        members.setdefault(g, []).append(l)
    report["почти совпадающих плоскостей объединено"] = sum(len(m) - 1 for m in members.values())
    args, sources, tight = TopTools_ListOfShape(), [], {}
    for l, ls in members.items():
        mask = np.isin(labels, ls)
        s = surfaces[l]
        if len(ls) > 1:
            verts = np.unique(mesh.faces[mask])
            refit = FT.fit_kind("plane", mesh.vertices[verts].astype(np.float64),
                                np.zeros((len(verts), 3)), np.ones(len(verts)))
            s = refit or s
        geom = geom_surface(s, centroid[mask])
        if geom is None:
            continue
        verts = np.unique(mesh.faces[mask])
        # граница кластера — вся: по ней протяжённость куска; случайная выборка
        # внутренних пропускала края большой грани на шаг больше запаса, и кусок
        # выходил короче грани [22228_1c82530b_5: свободный край у полностью
        # покрытого кусочка плоскости в 44 мм²]
        rim = np.unique(mesh.face_adjacency_edges[mask[mesh.face_adjacency[:, 0]] != mask[mesh.face_adjacency[:, 1]]])
        inner = np.setdiff1d(verts, rim)
        pick = np.r_[rim, rng.choice(inner, min(UV_POINTS, len(inner)), replace=False)] if len(inner) else rim
        pts = mesh.vertices[pick]
        face = patch(geom, pts, margin)
        if face is not None:
            args.Append(face)
            sources.append((face, l))
            tight[l] = patch(geom, pts, TRIM_EDGES * edge_len)
    report["кусков"] = len(sources)
    report["линий касания"] = _contacts(args, mesh, labels, group, surfaces, relations, margin)

    fuse = BOPAlgo_Builder()
    fuse.SetArguments(args)
    fuse.SetRunParallel(True)
    fuse.Perform()
    pieces, owner = [], []
    for face, l in sources:
        for piece in list(fuse.Modified(face)) or ([] if fuse.IsDeleted(face) else [face]):
            pieces.append(TopoDS.Face_s(piece))
            owner.append(l)
    report["кусочков"] = len(pieces)

    # покрытие кусочков своим кластером
    near = max(COVER_TOL * tol, COVER_EDGES * edge_len)
    pts, area, normal, which = [], [], [], []
    for k, piece in enumerate(pieces):
        c, a, nrm = _samples(piece, DEFLECTION_EDGES * edge_len)
        pts.append(c)
        area.append(a)
        normal.append(nrm)
        which.append(np.full(len(a), k))
    pts, area, normal, which = (np.concatenate(x) for x in (pts, area, normal, which))
    _, dist, tri = trimesh.proximity.closest_point(mesh, pts)
    own = np.array(owner)[which]
    mapped = np.array([group.get(int(x), -1) for x in labels[tri]])
    covered = (dist < near) & (mapped == own)
    # сторона: нормаль кусочка (как он есть) против нормали меша наружу
    outward = (normal * mesh.face_normals[tri] * np.sign(mesh.volume)).sum(1) > 0
    total = np.bincount(which, weights=area, minlength=len(pieces))
    cover_fwd = np.bincount(which, weights=area * covered * outward, minlength=len(pieces))
    cover_rev = np.bincount(which, weights=area * covered * ~outward, minlength=len(pieces))
    cover = cover_fwd + cover_rev
    # «в воздухе» — дальше порога от любого треугольника меша: продолжение
    # поверхности за пределы детали. Площадь на меше, но чужого кластера, не
    # штрафуется: это полоска между краем кластера и настоящим ребром, где
    # граница кластеров размыта, и для замыкания она нужна
    air = np.bincount(which, weights=area * (dist >= near), minlength=len(pieces))

    # рёбра и направление, в котором каждый кусочек их обходит
    edges = TopTools_IndexedMapOfShape()
    sense = defaultdict(lambda: defaultdict(int))
    for k, piece in enumerate(pieces):
        for e in _subshapes(piece, TopAbs_EDGE):
            if BRep_Tool.Degenerated_s(TopoDS.Edge_s(e)) or e.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
                continue
            sense[edges.Add(e)][k] += 1 if e.Orientation() == TopAbs_FORWARD else -1
    incidence = []
    for i, by_piece in sense.items():
        faces = sorted(by_piece)
        # шов замкнутой поверхности: одна грань обходит его в обе стороны
        if len(faces) == 1 and by_piece[faces[0]] == 0:
            continue
        incidence.append({k: (1 if d > 0 else -1) for k, d in by_piece.items()})

    # задача выбора. Переменные: p_k — кусочек как есть, q_k — наизнанку,
    # y_e — ребро используется. Ребро: выбранных граней 0 или 2, и обходят они
    # его в противоположные стороны (граница выбранной ориентированной
    # поверхности пуста) — иначе стенка отверстия с дисками плоскостей
    # замыкалась в «пробку». Кусочек с одиночным ребром — край продлённого
    # куска — не берётся.
    n = len(pieces)
    multi = [f for f in incidence if len(f) >= 2]
    lone = {k for f in incidence if len(f) == 1 for k in f}
    m = len(multi)
    width = 2 * n + m
    A = lil_matrix((2 * m + n, width))
    for e, faces in enumerate(multi):
        for k, d in faces.items():
            A[e, k] = A[e, n + k] = 1.0
            A[m + e, k], A[m + e, n + k] = d, -d
        A[e, 2 * n + e] = -2.0
    for k in range(n):
        A[2 * m + k, k] = A[2 * m + k, n + k] = 1.0
    lower = np.r_[np.zeros(2 * m), np.zeros(n)]
    upper = np.r_[np.zeros(2 * m), np.ones(n)]
    scale = mesh.area
    c = np.r_[(AIR_WEIGHT * air - cover_fwd + cover_rev) / scale,
              (AIR_WEIGHT * air - cover_rev + cover_fwd) / scale, np.zeros(m)]
    allowed = np.where([k in lone for k in range(n)], 0.0, 1.0)
    result = milp(c, constraints=LinearConstraint(A.tocsr(), lower, upper), integrality=np.ones(width),
                  bounds=Bounds(np.zeros(width), np.r_[allowed, allowed, np.ones(m)]), options={"time_limit": 60})
    good = (cover > 0.5 * total) & (cover > 1e-3 * mesh.area)
    if result.x is None:
        report["выбор"] = f"не решён: {result.message}"
        return _surfaces(pieces, [], good, cover_fwd, cover_rev, lone, owner, tight, SEW_TOL * tol, report)
    chosen = [(k, False) for k in range(n) if result.x[k] > 0.5] + \
             [(k, True) for k in range(n) if result.x[n + k] > 0.5]
    report["кусочков выбрано"] = len(chosen)
    report["решатель"] = f"{result.status}; цель {result.fun:.4f}"
    picked = {k for k, _ in chosen}
    report["покрытых кусочков запрещено"] = int(sum(good[k] for k in lone))
    report["покрытых кусочков не выбрано"] = int(sum(good[k] and k not in picked and k not in lone for k in range(n)))
    report["доля покрытой площади вне тела"] = float(sum(cover[k] for k in range(n) if k not in picked) / max(cover.sum(), 1e-30))
    by_owner = defaultdict(float)
    for k in picked:
        by_owner[owner[k]] += cover[k]
    report["поверхностей без кусочка"] = int(sum(1 for l in set(owner) if by_owner[l] == 0 and cover[np.array(owner) == l].sum() > 0))
    if DEBUG:
        _explain(mesh, labels, group, surfaces, pieces, owner, cover, total, lone, sense, edges, picked, sources)

    builder, shell = BRep_Builder(), TopoDS_Shell()
    builder.MakeShell(shell)
    for k, flip in chosen:
        builder.Add(shell, pieces[k].Reversed() if flip else pieces[k])
    fix = ShapeFix_Shell(shell)
    fix.Perform()
    shells = _subshapes(fix.Shape(), TopAbs_SHELL)
    report["оболочек"] = len(shells)
    if DEBUG:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        for sh in shells:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(sh, props)
            fs = _subshapes(sh, TopAbs_FACE)
            index = TopTools_IndexedMapOfShape()
            for piece in pieces:
                index.Add(piece)
            ks = [index.FindIndex(f) - 1 for f in fs]
            print(f"  оболочка: граней {len(fs)}, площадь {props.Mass():.2f}, замкнута {BRep_Tool.IsClosed_s(sh)}, "
                  f"поверхности {sorted({owner[k] for k in ks if k >= 0})}, покрыто {sum(cover[k] for k in ks if k >= 0):.2f}, "
                  f"в воздухе {sum(air[k] for k in ks if k >= 0):.2f}")
    report["тело"] = False
    if len(shells) == 1 and BRep_Tool.IsClosed_s(shells[0]):
        solid = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shells[0])).Solid())
        solid.Perform()
        shape = solid.Solid()
        report["тело"] = True
        report["граней"] = len(_subshapes(shape, TopAbs_FACE))
        report["валидна"] = BRepCheck_Analyzer(shape).IsValid()
        return shape, report
    return _surfaces(pieces, chosen, good, cover_fwd, cover_rev, lone, owner, tight, SEW_TOL * tol, report)


def _surfaces(pieces, chosen, good, cover_fwd, cover_rev, lone, owner, tight, sew_tol: float, report: dict):
    """Запасной режим, когда тело не замкнулось: обрезанные поверхности для
    импорта в CAD. Берутся выбранные кусочки и все покрытые своим кластером.
    Где сосед не дорезал (у кусочка одиночный край), кусочек подрезается по
    тесной рамке кластера с запасом `TRIM_EDGES`: на запасе куска край торчал
    на 10 рёбер, вне CAD было 27-63% площади [38192_f9c4b82a_0,
    48832_22363e53_49 по разметке]. Ориентация — по стороне, с которой
    покрытие больше. Сшиваются там, где края сходятся."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    taken = dict(chosen)
    for k in np.flatnonzero(good):
        taken.setdefault(int(k), bool(cover_rev[k] > cover_fwd[k]))
    report["режим"] = "поверхности"
    report["кусочков в поверхностях"] = len(taken)
    if not taken:
        return None, report
    sew = BRepBuilderAPI_Sewing(sew_tol)
    trimmed = 0
    for k, flip in taken.items():
        faces = [pieces[k]]
        if k in lone and tight.get(owner[k]) is not None:
            common = BRepAlgoAPI_Common(pieces[k], tight[owner[k]])
            if common.IsDone():
                faces = [TopoDS.Face_s(f) for f in _subshapes(common.Shape(), TopAbs_FACE)] or faces
                trimmed += 1
        for f in faces:
            sew.Add(f.Reversed() if flip else f)
    report["подрезано по рамке"] = trimmed
    sew.Perform()
    shape = sew.SewedShape()
    report["оболочек после сшивки"] = len(_subshapes(shape, TopAbs_SHELL))
    report["свободных рёбер"] = sew.NbFreeEdges()
    report["граней"] = len(_subshapes(shape, TopAbs_FACE))
    report["валидна"] = BRepCheck_Analyzer(shape).IsValid()
    return shape, report
