"""Сборка тела разбиением пространства подогнанными поверхностями.

Поверхности, продлённые на весь габарит детали, режут его на ячейки
(`BOPAlgo_MakerVolume`). Граница истинного тела лежит на этих поверхностях,
поэтому тело — объединение ячеек, и каждая ячейка целиком внутри или снаружи.
Ячейка берётся, если её внутренние точки лежат внутри меша. Оболочка замкнута
по построению: дыр, как у сборки по рёбрам (`brep.build`), не бывает, а
неточность подгонки уходит в форму, а не в топологию (README, «Сборка
разбиением»).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import trimesh
from OCP.BOPAlgo import BOPAlgo_GlueEnum, BOPAlgo_MakerVolume
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GeomAbs import GeomAbs_Plane
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Shell, ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ListOfShape

from . import surfaces as S
from .brep import CONFUSION, _subshapes, geom_surface
from .surfaces import Surface

# Запас габарита, долей наибольшей стороны: поверхности режут коробку чуть
# больше детали, чтобы грань детали не совпала с гранью коробки.
BOX_PAD = 0.05
# Голос ячейки — по равномерным точкам коробки с тем же вектором знаков
# (`select`); сколько их нужно ячейке, чтобы решать по ним. Раньше голосовали
# точки самой ячейки: центр масс и вершины, стянутые к нему, — и у ячейки,
# обнимающей деталь, вершины лежат на детали, так что голос был «внутри»
# [измерено, 42021_2a7183d7_0: взяты ячейки, у которых по 400 равномерным
# точкам внутри меша 0-2% объёма, объём тела 3x меша]; а равномерные точки,
# отобранные классификатором OCC, на 5 тысячах ячеек не укладывались в 10
# минут. 400 тысяч — в среднем 80 точек на ячейку при 5 тысячах ячеек;
# проверка «внутри меша» на них — доли секунды с embree.
GLOBAL_POINTS = 400_000
MIN_GROUP = 30
# Свои точки ячейки — для её вектора знаков и голоса тонкой ячейки: центр масс,
# вершины, стянутые к нему вдвое, затем случайные в габарите; у выпуклой
# ячейки — выпуклые комбинации вершин с параметром Дирихле меньше единицы,
# чтобы точки расходились к рёбрам, а не стягивались к центру.
CELL_POINTS = 12
SHRINK = 0.5
RANDOM_TRIES = 60
DIRICHLET = 0.5
# Кластер мельче стольких треугольников отдаётся соседу. Меньше, чем у сборки
# по рёбрам (`brep.min_triangles`): здесь потерянная поверхность дороже лишней,
# а трёх вершин хватает плоскости.
MIN_TRIANGLES = 3
# Потолок веса точки в голосе ячейки долей √площади детали, ~0.05 мм на
# медианной: порядок отклонения рабочего меша от CAD (p95 0.0016 по медиане
# деталей). Глубже — точка уверенно внутри, её голос не должен расти дальше.
VOTE_CLIP_REL = 0.002
# Допуск сверки объёма между шагами сборки: шаги не должны менять форму, и
# расхождение больше этого — потерянный или лишний кусок.
VOLUME_CHECK_REL = 1e-3
# Связный кусок взятых ячеек мельче этой доли объёма — щепка, см. `build`.
MIN_PIECE_REL = 0.01


def _volume_props(shape) -> GProp_GProps:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props


def bounded_face(geom, kind: str, corners: np.ndarray):
    """Кусок поверхности, покрывающий коробку с углами `corners`. Сфера и тор
    замкнуты и берутся целиком; у конуса — только полость со стороны точек."""
    if kind in ("sphere", "torus"):
        return BRepBuilderAPI_MakeFace(geom, CONFUSION).Face()
    ax = geom.Position()
    rel = corners - np.array(ax.Location().Coord())
    if kind == "plane":
        u, v = rel @ np.array(ax.XDirection().Coord()), rel @ np.array(ax.YDirection().Coord())
        return BRepBuilderAPI_MakeFace(geom, u.min(), u.max(), v.min(), v.max(), CONFUSION).Face()
    z = rel @ np.array(ax.Direction().Coord())
    if kind == "cylinder":
        return BRepBuilderAPI_MakeFace(geom, 0, 2 * np.pi, z.min(), z.max(), CONFUSION).Face()
    theta, r0 = geom.SemiAngle(), geom.RefRadius()
    apex = -r0 / np.sin(theta)
    v0, v1 = z.min() / np.cos(theta), z.max() / np.cos(theta)
    # вершина конуса — особая точка поверхности, разбиение на ней вырождается
    v0 = max(v0, apex + 1e-3 * (v1 - apex))
    if v1 <= v0:
        return None
    return BRepBuilderAPI_MakeFace(geom, 0, 2 * np.pi, v0, v1, CONFUSION).Face()


def box(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = mesh.bounds
    pad = BOX_PAD * float((hi - lo).max())
    return lo - pad, hi + pad


def cells(mesh: trimesh.Trimesh, labels: np.ndarray, surfaces: dict[int, Surface]) -> tuple[list, list[int]]:
    """Ячейки разбиения коробки детали поверхностями и поверхности, которые в
    разбиение вошли (у конуса без своей полости его нет)."""
    lo, hi = box(mesh)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    args = TopTools_ListOfShape()
    for face in _subshapes(BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape(), TopAbs_FACE):
        args.Append(face)
    centroid = mesh.triangles_center
    used = []
    for l, s in surfaces.items():
        geom = geom_surface(s, centroid[labels == l])
        face = bounded_face(geom, s.kind, corners) if geom is not None else None
        if face is not None:
            args.Append(face)
            used.append(l)
    maker = BOPAlgo_MakerVolume()
    maker.SetArguments(args)
    maker.SetRunParallel(True)
    maker.SetIntersect(True)
    maker.Perform()
    return _subshapes(maker.Shape(), TopAbs_SOLID), used


def _planar(cell) -> bool:
    return all(BRepAdaptor_Surface(TopoDS.Face_s(f)).GetType() == GeomAbs_Plane
               for f in _subshapes(cell, TopAbs_FACE))


def interior_points(cell, rng: np.random.Generator) -> np.ndarray:
    """Точки внутри ячейки.

    Ячейка из одних плоскостей выпукла (пересечение полупространств), и
    случайные выпуклые комбинации её вершин лежат внутри без проверки. Иначе
    точки равномерны в габарите и отбираются классификатором OCC: он и есть
    почти всё время сборки [измерено, 56996_1fb4fc17_0, 5495 ячеек: разбиение
    26 с, отбор точек классификатором — больше 10 минут].
    """
    vertices = np.array([BRep_Tool.Pnt_s(TopoDS.Vertex_s(v)).Coord() for v in _subshapes(cell, TopAbs_VERTEX)])
    if len(vertices) >= 4 and _planar(cell):
        return rng.dirichlet(np.full(len(vertices), DIRICHLET), size=CELL_POINTS) @ vertices
    centre = np.array(_volume_props(cell).CentreOfMass().Coord())
    candidates = [centre[None], centre + SHRINK * (vertices - centre)] if len(vertices) else [centre[None]]
    classifier = BRepClass3d_SolidClassifier(cell)
    out = []

    def take(points):
        for p in points:
            classifier.Perform(gp_Pnt(*p), CONFUSION)
            if classifier.State() == TopAbs_IN:
                out.append(p)
                if len(out) >= CELL_POINTS:
                    return

    take(np.concatenate(candidates))
    bounds = Bnd_Box()
    BRepBndLib.Add_s(cell, bounds)
    for tries in (RANDOM_TRIES, 10 * RANDOM_TRIES):
        # вторая, долгая попытка — только если точек нет вовсе: без них ячейка
        # выпадает, а это дыра в теле
        if len(out) >= CELL_POINTS or (tries > RANDOM_TRIES and out):
            break
        take(rng.uniform(bounds.CornerMin().Coord(), bounds.CornerMax().Coord(), size=(tries, 3)))
    return np.array(out).reshape(-1, 3)


def signs(surfaces: list[Surface], points: np.ndarray) -> np.ndarray:
    """Сторона каждой поверхности для каждой точки, упакованная в байты."""
    x = torch.from_numpy(points)
    side = np.stack([S.distance(s.kind, torch.from_numpy(s.params), x).numpy() > 0 for s in surfaces], axis=1)
    return np.packbits(side, axis=1)


def select(cell_list: list, mesh: trimesh.Trimesh, surfaces: list[Surface], seed: int = 0) -> list:
    """Ячейки внутри меша.

    Ячейку задаёт сторона, по которую от каждой поверхности она лежит, —
    вектор знаков. Доля объёма ячейки внутри меша оценивается по
    `GLOBAL_POINTS` равномерным точкам коробки с тем же вектором знаков в
    габарите ячейки:
    равномерно, по всему объёму и без классификатора OCC на каждую точку. От
    самой ячейки нужна лишь одна внутренняя точка. Если общих точек с её
    вектором меньше `MIN_GROUP` (тонкая ячейка), голосуют её собственные
    точки, взвешенные расстоянием до меша с потолком `VOTE_CLIP_REL`: у
    поверхности меша шум, и такие точки решают слабее.
    """
    rng = np.random.default_rng(seed)
    lo, hi = box(mesh)
    cloud = rng.uniform(lo, hi, size=(GLOBAL_POINTS, 3))
    keys, group = np.unique(signs(surfaces, cloud), axis=0, return_inverse=True)
    inside = mesh.contains(cloud)
    order = np.argsort(group.ravel(), kind="stable")
    starts = np.searchsorted(group.ravel()[order], np.arange(len(keys) + 1))
    lookup = {k.tobytes(): i for i, k in enumerate(keys)}

    points = [interior_points(c, rng) for c in cell_list]
    keep, own = [], []
    for c, p in zip(cell_list, points):
        if not len(p):
            continue
        g = lookup.get(signs(surfaces, p[:1])[0].tobytes())
        mine = np.empty(0, int)
        if g is not None:
            # один вектор знаков бывает у нескольких несвязных областей (вне
            # цилиндра между двумя плоскостями по его бокам), поэтому из
            # точек группы берутся только лежащие в габарите самой ячейки
            mine = order[starts[g]:starts[g + 1]]
            bounds = Bnd_Box()
            BRepBndLib.Add_s(c, bounds)
            a, b = np.array(bounds.CornerMin().Coord()), np.array(bounds.CornerMax().Coord())
            mine = mine[((cloud[mine] >= a) & (cloud[mine] <= b)).all(1)]
        if len(mine) >= MIN_GROUP:
            if inside[mine].mean() > 0.5:
                keep.append(c)
        else:
            own.append((c, p))
    if own:
        flat = np.concatenate([p for _, p in own])
        owner = np.concatenate([np.full(len(p), i) for i, (_, p) in enumerate(own)])
        clip = VOTE_CLIP_REL * float(np.sqrt(mesh.area))
        signed = np.clip(trimesh.proximity.signed_distance(mesh, flat), -clip, clip)   # > 0 внутри
        score = np.bincount(owner, weights=signed, minlength=len(own))
        keep += [c for (c, _), sc in zip(own, score) if sc > 0]
    return keep


def components(keep: list) -> tuple[list[list[int]], list]:
    """Связность взятых ячеек через общие грани и грани границы тела.

    Ячейки разбиения делят грани (один и тот же `TShape`), поэтому граница
    объединения — грани, принадлежащие ровно одной взятой ячейке. Булева
    операция для этого не нужна, а `BRepAlgoAPI_Fuse` на сотнях ячеек падал
    (README, «Сборка разбиением»).
    """
    index = TopTools_IndexedMapOfShape()
    owners, oriented = defaultdict(list), {}
    for i, cell in enumerate(keep):
        for face in _subshapes(cell, TopAbs_FACE):
            k = index.Add(face)
            owners[k].append(i)
            oriented.setdefault(k, face)
    parent = list(range(len(keep)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for cs in owners.values():
        for c in cs[1:]:
            parent[root(c)] = root(cs[0])
    groups = defaultdict(list)
    for i in range(len(keep)):
        groups[root(i)].append(i)
    boundary = [(k, oriented[k], cs[0]) for k, cs in owners.items() if len(cs) == 1]
    return list(groups.values()), boundary


def solid_from_faces(faces: list):
    """Грани границы -> тело: оболочки по связности, ориентация наружу."""
    builder, shell = BRep_Builder(), TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    fix = ShapeFix_Shell(shell)
    fix.SetPrecision(CONFUSION)
    fix.Perform()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    for s in _subshapes(fix.Shape(), TopAbs_SHELL):
        builder.Add(solid, s)
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    return fix.Solid()


def _repaired(shape):
    """Форма, если она валидна, иначе после `ShapeFix_Shape`; None — не вышло.
    Обычно ломаются кривые на поверхностях после слияния граней, а их
    исправление форму не меняет [измерено: 61198_0c99f50a_10 — 362 -> 95
    граней, 110880_b36dfcfb_0 — 618 -> 106]."""
    if shape.IsNull():
        return None
    if not BRepCheck_Analyzer(shape).IsValid():
        fix = ShapeFix_Shape(shape)
        fix.Perform()
        shape = fix.Shape()
    return shape if not shape.IsNull() and BRepCheck_Analyzer(shape).IsValid() else None


def _same(shape, solids: int, volume: float) -> bool:
    """Шаг сборки сохранил число тел и объём. BRepCheck этого не ловит: слияние
    с `Glue` на 44321_4940c526_2 дало валидное тело нулевого объёма."""
    return (len(_subshapes(shape, TopAbs_SOLID)) == solids
            and abs(_volume_props(shape).Mass() - volume) <= VOLUME_CHECK_REL * abs(volume))


def unify(shape):
    """Слить куски одной поверхности в одну грань.

    `UnifySameDomain` на части деталей бросает исключение, портит форму или
    теряет тело [измерено: 44321_4940c526_2 — 183 невалидные грани вместо 72,
    24168_a0ede173_148 — без тела, при наборе ячеек, отличном на одну]. Такой
    результат не берётся: форма остаётся с поделёнными гранями — она та же.
    """
    solids, volume = len(_subshapes(shape, TopAbs_SOLID)), _volume_props(shape).Mass()
    for edges in (True, False):
        try:
            u = ShapeUpgrade_UnifySameDomain(shape, edges, True, True)
            u.Build()
            out = _repaired(u.Shape())
        except Exception:  # noqa: BLE001 — Standard_Failure из OCC
            continue
        if out is not None and _same(out, solids, volume):
            return out
    return _repaired(shape) or shape


def glued(keep: list):
    """Объединение ячеек булевой операцией с `Glue`: ячейки разбиения делят
    грани, и общий случай пересечения не нужен. Без `Glue` `BRepAlgoAPI_Fuse`
    на сотнях ячеек падал с segfault."""
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(keep[0])
    for c in keep[1:]:
        tools.Append(c)
    fuse = BRepAlgoAPI_Fuse()
    fuse.SetArguments(args)
    fuse.SetTools(tools)
    fuse.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
    fuse.SetRunParallel(True)
    fuse.Build()
    return fuse.Shape()


def assemble(keep: list):
    """Взятые ячейки -> тело: склейкой, а если она потеряла объём или тело
    невалидно — гранями границы (`components`). Оба пути проверяются по объёму:
    склейка ошибается молча, а у граней границы бывают рёбра на трёх-четырёх
    гранях, когда ячейки касаются только по ребру [измерено: 24168_a0ede173_148,
    44321_4940c526_2 — оболочка распадалась на 8]."""
    volume = sum(_volume_props(c).Mass() for c in keep)
    if len(keep) == 1:
        return keep[0]
    candidates = []
    for make in (glued, lambda k: solid_from_faces([f for _, f, _ in components(k)[1]])):
        try:
            raw = make(keep)
        except Exception:  # noqa: BLE001 — Standard_Failure из OCC
            continue
        shape = _repaired(raw)
        if shape is not None and _same(shape, 1, volume):
            return shape
        best = shape if shape is not None else raw
        if not best.IsNull() and _subshapes(best, TopAbs_SOLID):
            candidates.append((shape is None, abs(_volume_props(best).Mass() - volume), best))
    # Строгую проверку не прошёл ни один путь: лучше тело с изъяном, видным в
    # сводке («валидна», объём), чем никакого — без него нечего ни показать,
    # ни записать в STEP [скан 1.stl, одна из нарезок сети: ячеек внутри 888, тела нет].
    return min(candidates, key=lambda c: c[:2])[2] if candidates else None


def build(mesh: trimesh.Trimesh, labels: np.ndarray, surfaces: dict[int, Surface]):
    """Поверхности -> тело и сводка.

    Связные куски взятых ячеек мельче `MIN_PIECE_REL` объёма отбрасываются:
    это ячейки-щепки у шумной границы меша, ошибочно взятые, а деталь корпуса —
    одно тело.
    """
    report = {}
    parts, used = cells(mesh, labels, surfaces)
    report["ячеек"] = len(parts)
    keep = select(parts, mesh, [surfaces[l] for l in used])
    report["ячеек внутри"] = len(keep)
    if not keep:
        return None, report
    groups, _ = components(keep)
    volume = np.array([_volume_props(c).Mass() for c in keep])
    sizes = np.array([volume[g].sum() for g in groups])
    big = [g for g, s in zip(groups, sizes) if s >= MIN_PIECE_REL * sizes.sum()]
    report["отброшено щепок"] = len(groups) - len(big)
    keep = [keep[i] for g in big for i in g]
    shape = assemble(keep)
    if shape is None:
        return None, report
    shape = unify(shape)
    solids = _subshapes(shape, TopAbs_SOLID)
    if not solids:
        return None, report
    report["тел"] = len(big)
    report["тело"] = len(solids) >= 1
    report["граней"] = len(_subshapes(shape, TopAbs_FACE))
    report["валидна"] = BRepCheck_Analyzer(shape).IsValid()
    return shape, report
