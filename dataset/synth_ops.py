"""Синтетические детали для opseq (README, «Генератор деталей»): проект реконструкции + случайные операции.

Основа — выдавливания проекта реконструкции (тела инструментов из эскизов, `fusion_recon.tools`). Поверх —
отверстия (по одному и массивами), карманы, бобышки, уступы от кромки, скругления и фаски группами рёбер.
Эскиз добавки строится внутри плоской грани тела в её осях (x — вдоль самой длинной прямой кромки):
контур грани -> 2D многоугольник (shapely) -> область с полями -> форма, которая в неё помещается.
Добавка принимается, если тело валидно и одно, объём изменился заметно, нет граней-щелей и стенок тоньше
MIN_WALL_MM. Каждое тело выдавливания строится нами, поэтому стенки и торцы размечаются так же, как у
реконструкции (`fusion_recon.face_roles`); полосы скруглений и фасок — по поверхностям, которые вернуло ядро.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Lin, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from dataset.fusion_recon import ReconError, Tool, face_roles, face_samples, make_tool
from dataset.step_mesh import bbox_diag, iter_faces

# Поля от кромок грани и наименьшая стенка после добавки, мм (детали 20-30 мм по диагонали). Выбрано.
MARGIN_MM = 0.5
MIN_WALL_MM = 0.3
# Добавка должна менять объём хотя бы на эту долю; грань меньше MIN_FACE_MM2 — щель. Выбрано.
MIN_VOLUME_CHANGE = 1e-3
MIN_FACE_MM2 = 0.1
# Вырез стартует чуть над гранью, бобышка — чуть под ней: без совпадающих плоскостей булевы надёжнее.
OVERLAP_MM = 0.02
ATTEMPTS = 20
# Тип добавки: вероятности. Выбрано.
KINDS = {"hole": 0.35, "pocket": 0.2, "boss": 0.1, "notch": 0.1, "fillet": 0.15, "chamfer": 0.1}
# Скругление: радиус — доля ширины самой узкой соседней грани. Выбрано.
FILLET_REL = (0.1, 0.4)
# Ребро скругляется, только если обе соседние грани не уже этого, мм: одна узкая грань в группе ужимала
# радиус всем рёбрам до долей мм. Выбрано.
MIN_FILLET_FACE_MM = 1.0
# Рёбра соседних граней под углом меньше этого — касательные, их не скругляют. Выбрано.
TANGENT_DEG = 10.0


class GenError(Exception):
    pass


@dataclass
class Op:
    name: str
    kind: str                     # ExtrudeFeature | FilletFeature | ChamferFeature
    boolean: str | None
    source: str                   # recon | synth
    tool: Tool | None = None      # у выдавливания
    surfaces: list = field(default_factory=list)   # у скругления и фаски: поверхности полос (Geom_Surface)
    params: dict = field(default_factory=dict)


# ---------- геометрия тела ----------

def volume(s) -> float:
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, p)
    return abs(p.Mass())


def area(s) -> float:
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(s, p)
    return abs(p.Mass())


def count(shape, kind) -> int:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Extent()


def boolean(body, tool, kind: str):
    op = {"add": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "intersect": BRepAlgoAPI_Common}[kind](body, tool)
    return op.Shape() if op.IsDone() else None


def unhealthy(new, old, by_volume: bool = True) -> str | None:
    """Почему новое тело не годится; None — годится. `by_volume` — добавка должна заметно менять объём
    (у скругления и фаски объём меняется меньше 0.1% и это нормально — там проверяется, что граней стало больше)."""
    if new is None or new.IsNull() or not BRepCheck_Analyzer(new).IsValid():
        return "невалидное тело"
    if count(new, TopAbs_SOLID) != 1:
        return "тел не одно"
    if by_volume:
        v0, v1 = volume(old), volume(new)
        if abs(v1 - v0) < MIN_VOLUME_CHANGE * v0:
            return "объём почти не изменился"
    elif count(new, TopAbs_FACE) <= count(old, TopAbs_FACE):
        return "граней не прибавилось"
    old_small = sum(area(f) < MIN_FACE_MM2 for f in iter_faces(old))
    if sum(area(f) < MIN_FACE_MM2 for f in iter_faces(new)) > old_small:
        return "грань-щель"
    return None


def thickness(body, points: np.ndarray, inward: np.ndarray) -> np.ndarray:
    """Расстояние от точек вдоль `inward` до следующего пересечения с телом; inf — не пересекло."""
    it = IntCurvesFace_ShapeIntersector()
    it.Load(body, 1e-6)
    out = np.full(len(points), np.inf)
    for i, p in enumerate(points):
        q = p + inward * 1e-4
        it.Perform(gp_Lin(gp_Pnt(*map(float, q)), gp_Dir(*map(float, inward))), 1e-5, 1e9)
        if it.IsDone() and it.NbPnt():
            out[i] = min(it.WParameter(k) for k in range(1, it.NbPnt() + 1))
    return out


def thin_new_walls(new, tool, rng) -> bool:
    """Стенки тоньше MIN_WALL_MM у граней нового тела возле инструмента (луч внутрь от точек)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box_ = Bnd_Box()
    BRepBndLib.Add_s(tool, box_)
    box_.Enlarge(MIN_WALL_MM * 2)
    lo, hi = np.array(box_.Get()[:3]), np.array(box_.Get()[3:])
    pts, nrm = [], []
    for face, p in zip(iter_faces(new), face_samples(new, 0.05, 40)):
        if not len(p):
            continue
        near = np.all((p >= lo) & (p <= hi), axis=1)
        if not near.any():
            continue
        p = p[near]
        surf = BRep_Tool.Surface_s(face)
        flip = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        for x in p[: 8]:
            pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, x)), surf)
            if not pr.NbPoints():
                continue
            u, v = pr.LowerDistanceParameters()
            from OCP.BRepLProp import BRepLProp_SLProps
            props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-6)
            if not props.IsNormalDefined():
                continue
            n = props.Normal()
            pts.append(x)
            nrm.append(-flip * np.array([n.X(), n.Y(), n.Z()]))
    if not pts:
        return False
    th = np.array([thickness(new, np.array([p]), d)[0] for p, d in zip(pts, nrm)])
    return bool(np.any(th < MIN_WALL_MM))


# ---------- плоские грани и их 2D ----------

@dataclass
class Planar:
    face: object
    frame: np.ndarray          # начало, x, y, z (z — наружная нормаль)
    poly: Polygon | MultiPolygon
    area: float


def _longest_line(face) -> np.ndarray | None:
    best, d = 0.0, None
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        if c.GetType() == GeomAbs_Line:
            length = GCPnts_AbscissaPoint.Length_s(c)
            if length > best:
                v = c.Line().Direction()
                best, d = length, np.array([v.X(), v.Y(), v.Z()])
        exp.Next()
    return d


def planar_faces(body, min_area: float) -> list[Planar]:
    BRepMesh_IncrementalMesh(body, 0.02, False, 0.3, True)
    out = []
    for face in iter_faces(body):
        s = BRepAdaptor_Surface(face)
        if s.GetType() != GeomAbs_Plane:
            continue
        a = area(face)
        if a < min_area:
            continue
        pl = s.Plane()
        z = np.array([pl.Axis().Direction().X(), pl.Axis().Direction().Y(), pl.Axis().Direction().Z()])
        if face.Orientation() == TopAbs_REVERSED:
            z = -z
        o = np.array([pl.Location().X(), pl.Location().Y(), pl.Location().Z()])
        x = _longest_line(face)
        if x is None or abs(x @ z) > 0.99:
            x = np.eye(3)[np.argmin(np.abs(z))]
        x = x - (x @ z) * z
        x /= np.linalg.norm(x)
        frame = np.stack([o, x, np.cross(z, x), z])
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        v = np.array([[q.X(), q.Y(), q.Z()] for q in (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))])
        f = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
        v2 = (v - o) @ frame[1:3].T
        poly = unary_union([Polygon(v2[t]) for t in f]).buffer(0)
        out.append(Planar(face, frame, poly, a))
    return out


def inradius(poly) -> float:
    lo, hi = 0.0, float(np.sqrt(poly.area))
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if not poly.buffer(-mid).is_empty else (lo, mid)
    return lo


def random_point(poly, rng, tries: int = 200) -> np.ndarray | None:
    x0, y0, x1, y1 = poly.bounds
    for _ in range(tries):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if poly.contains(Point(p)):
            return p
    return None


# ---------- добавки-выдавливания: профиль из библиотеки по привязке ----------

@dataclass
class Style:
    """Размеры, общие для детали: отверстия на одной детали чаще одного диаметра."""
    hole_sizes: list = field(default_factory=list)


def circle_edges(pf: Planar) -> list[tuple[np.ndarray, float]]:
    """Круглые кромки грани (полные окружности и дуги): центр в осях грани и радиус."""
    from OCP.GeomAbs import GeomAbs_Circle
    out = []
    exp = TopExp_Explorer(pf.face, TopAbs_EDGE)
    while exp.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        if c.GetType() == GeomAbs_Circle:
            ci = c.Circle()
            p = np.array([ci.Location().X(), ci.Location().Y(), ci.Location().Z()])
            out.append(((p - pf.frame[0]) @ pf.frame[1:3].T, ci.Radius()))
        exp.Next()
    uniq = []
    for c, r in out:
        if not any(np.linalg.norm(c - c2) < 1e-6 and abs(r - r2) < 1e-6 for c2, r2 in uniq):
            uniq.append((c, r))
    return uniq


def _free(region, c) -> float:
    return float(region.boundary.distance(Point(c))) if region.contains(Point(c)) else 0.0


def anchors(pf: Planar, region, rng) -> list[tuple[str, list, float, list]]:
    """Привязки на грани: (имя, центры копий, наибольший радиус копии, углы поворота копий)."""
    out = []
    c = np.array(region.centroid.coords[0])
    if not region.contains(Point(c)):
        c = np.array(region.representative_point().coords[0])
    out.append(("центр", [c], _free(region, c), [0.0]))
    x0, y0, x1, y1 = region.bounds
    w, h = x1 - x0, y1 - y0
    for cc, r in circle_edges(pf):
        if region.contains(Point(cc)):
            out.append(("соосно", [cc], _free(region, cc), [0.0]))
        # окружность копий вокруг круглой кромки: между ней и краем
        n = int(rng.choice([3, 4, 5, 6, 8]))
        for rr in (r * rng.uniform(1.4, 2.2), r * rng.uniform(0.45, 0.75)):
            pts = [cc + rr * np.array([np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)]) for k in range(n)]
            free = min(_free(region, p) for p in pts)
            free = min(free, rr * np.sin(np.pi / n) - MARGIN_MM / 2)
            if free > 0.3:
                out.append(("по окружности", pts, free, [2 * np.pi * k / n for k in range(n)]))
    d = rng.uniform(0.15, 0.3) * min(w, h)
    corners = [np.array(p) for p in ((x0 + d, y0 + d), (x1 - d, y0 + d), (x1 - d, y1 - d), (x0 + d, y1 - d))]
    if all(region.contains(Point(p)) for p in corners):
        free = min(_free(region, p) for p in corners)
        out.append(("у углов", corners, min(free, d), [0.0] * 4))
        long_x = w >= h
        two = [corners[0], corners[1]] if long_x else [corners[0], corners[3]]
        two = [p + (np.array([0, (h - 2 * d) / 2]) if long_x else np.array([(w - 2 * d) / 2, 0])) for p in two]
        if all(region.contains(Point(p)) for p in two):
            out.append(("пара вдоль длинной стороны", two, min(_free(region, p) for p in two), [0.0, 0.0]))
    a = rng.uniform(0.2, 0.35) * (w if w >= h else h)
    ax = np.array([1.0, 0.0]) if w >= h else np.array([0.0, 1.0])
    pair = [c - a * ax, c + a * ax]
    if all(region.contains(Point(p)) for p in pair):
        out.append(("симметричная пара", pair, min(min(_free(region, p) for p in pair), a - MARGIN_MM / 2), [0.0, 0.0]))
    return [x for x in out if x[2] > 0.3]


NICE = (0.3, 0.4, 0.5, 0.6, 0.7)
# Тип профиля по добавке. Выбрано.
PROFILE_KINDS = {"hole": {"round": 0.75, "rect": 0.1, "other": 0.15}, "pocket": {"round": 0.25, "rect": 0.35, "other": 0.4},
                 "boss": {"round": 0.55, "rect": 0.3, "other": 0.15}, "notch": {"rect": 0.7, "other": 0.3}}
# Бобышка — только в центр или соосно (README, «Генератор деталей»: случайные бобышки выглядели неестественно).
BOSS_ANCHORS = ("центр", "соосно")


def _profile(library, kind: str, rng):
    ks = PROFILE_KINDS[kind]
    k = rng.choice(list(ks), p=np.array(list(ks.values())))
    pool = library[k]
    return pool[rng.integers(len(pool))]


def _notch_place(pf: Planar, region, prof, rng):
    """Уступ от кромки: профиль по середине прямого участка контура, поперёк него, наполовину за краем."""
    ext = pf.poly.exterior if isinstance(pf.poly, Polygon) else max(pf.poly.geoms, key=lambda g: g.area).exterior
    pts = np.array(ext.coords)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    i = int(rng.choice(len(seg), p=seg / seg.sum()))
    a, b = pts[i], pts[i + 1]
    m, t = (a + b) / 2, (b - a) / max(seg[i], 1e-9)
    s = seg[i] * rng.choice((0.15, 0.2, 0.25, 0.3))
    from dataset.profile_library import place
    loops, poly = place(prof, s, float(np.arctan2(t[1], t[0])), m)
    inside = poly.intersection(pf.poly).area
    if not 0.2 * poly.area < inside < 0.8 * poly.area:
        return None
    return [(loops, poly)]


def extrude_op(body, kind: str, name: str, rng, diag: float, stats: Counter, library, style: Style):
    from dataset.fusion_recon import profile_face
    from dataset.profile_library import place
    faces = planar_faces(body, min_area=(4 * MARGIN_MM) ** 2)
    if not faces:
        stats[kind + ": нет плоских граней"] += 1
        return None
    w = np.array([f.area for f in faces])
    pf = faces[rng.choice(len(faces), p=w / w.sum())]
    region = pf.poly.buffer(-MARGIN_MM)
    if region.is_empty:
        stats[kind + ": грань узкая"] += 1
        return None
    prof = _profile(library, kind, rng)
    anchor = "кромка"
    if kind == "notch":
        shapes = _notch_place(pf, region, prof, rng)
    else:
        cands = anchors(pf, region, rng)
        if kind == "boss":
            cands = [x for x in cands if x[0] in BOSS_ANCHORS]
        if not cands:
            stats[kind + ": нет привязки"] += 1
            return None
        anchor, centres, free, angles = cands[rng.integers(len(cands))]
        s = free * float(rng.choice(NICE))
        if kind == "hole" and style.hole_sizes and rng.random() < 0.5:
            same = [x for x in style.hole_sizes if x <= free * 0.95]
            s = same[rng.integers(len(same))] if same else s
        base_angle = float(rng.choice([0.0, np.pi / 2])) if prof.kind != "round" else 0.0
        shapes = [place(prof, s, base_angle + a, c) for c, a in zip(centres, angles)]
        if not all(region.contains(p) for _, p in shapes):
            shapes = None
        elif len(shapes) > 1 and unary_union([p for _, p in shapes]).area < 0.999 * sum(p.area for _, p in shapes):
            shapes = None
        if shapes and kind == "hole":
            style.hole_sizes.append(s)
    if not shapes:
        stats[kind + ": не поместилось"] += 1
        return None
    o, x, y, z = pf.frame
    sk_faces = [f for loops, _ in shapes for f in iter_faces(profile_face(loops, pf.frame, 1.0))]
    probe = [np.array(p.representative_point().coords[0]) for _, p in shapes]
    pts3 = np.array([o + q[0] * x + q[1] * y for q in probe])
    if kind == "boss":
        size = min(np.sqrt(p.area) for _, p in shapes)
        h = min(size * rng.uniform(0.4, 1.0), 0.25 * diag)
        tool = make_tool(name, "add", pf.frame, -OVERLAP_MM, h, sk_faces)
        tool.h0 = 0.0          # для разметки — номинальный торец в плоскости эскиза, без запаса для булевой
        params = {"height": h}
    else:
        th = thickness(body, pts3, -z)
        th = th[np.isfinite(th)]
        if not len(th):
            stats[kind + ": нет толщины"] += 1
            return None
        through = kind != "pocket" and rng.random() < 0.5
        depth = float(th.max() * 1.05 + 0.1) if through else float(th.min() * rng.choice((0.25, 1 / 3, 0.5, 2 / 3)))
        if not through and th.min() - depth < MIN_WALL_MM:
            stats[kind + ": тонко под эскизом"] += 1
            return None
        tool = make_tool(name, "cut", pf.frame, -depth, OVERLAP_MM, sk_faces)
        tool.h1 = 0.0
        params = {"depth": depth, "through": through}
    new = boolean(body, tool.shape, tool.boolean)
    why = unhealthy(new, body) or ("тонкая стенка" if thin_new_walls(new, tool.shape, rng) else None)
    if why:
        stats[f"{kind}: {why}"] += 1
        return None
    params |= {"shape": kind, "copies": len(shapes), "anchor": anchor, "profile": prof.kind}
    return new, Op(name, "ExtrudeFeature", tool.boolean, "synth", tool=tool, params=params)


# ---------- скругления и фаски группами рёбер ----------

def _edge_faces(body):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(body, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        fs = list(m.FindFromIndex(i))
        if len(fs) == 2:
            out.append((TopoDS.Edge_s(m.FindKey(i)), TopoDS.Face_s(fs[0]), TopoDS.Face_s(fs[1])))
    return out


def _normal_at(face, p: np.ndarray) -> np.ndarray | None:
    from OCP.BRepLProp import BRepLProp_SLProps
    pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, p)), BRep_Tool.Surface_s(face))
    if not pr.NbPoints():
        return None
    u, v = pr.LowerDistanceParameters()
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    n = np.array([n.X(), n.Y(), n.Z()])
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _width(face) -> float:
    """Ширина грани: площадь / самая длинная кромка."""
    best = 1e-9
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        best = max(best, GCPnts_AbscissaPoint.Length_s(BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))))
        exp.Next()
    return area(face) / best


def edge_groups(body, last_tool: Tool | None, rng) -> list[list]:
    """Группы рёбер «по замыслу»: кромка последней добавки, рёбра вдоль одной оси, контур одной грани."""
    info = []
    for e, fa, fb in _edge_faces(body):
        c = BRepAdaptor_Curve(e)
        mid = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        p = np.array([mid.X(), mid.Y(), mid.Z()])
        na, nb = _normal_at(fa, p), _normal_at(fb, p)
        if na is None or nb is None or np.degrees(np.arccos(np.clip(na @ nb, -1, 1))) < TANGENT_DEG:
            continue
        d = None
        if c.GetType() == GeomAbs_Line:
            v = c.Line().Direction()
            d = np.array([v.X(), v.Y(), v.Z()])
        info.append((e, fa, fb, p, d))
    if not info:
        return []
    groups = []
    if last_tool is not None:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        near = [x for x in info if BRepExtrema_DistShapeShape(x[0], last_tool.shape).Value() < 1e-4]
        if near:
            groups.append(near)
    lines = [x for x in info if x[4] is not None]
    if lines:
        ax = lines[rng.integers(len(lines))][4]
        groups.append([x for x in lines if abs(x[4] @ ax) > 0.999])
    fset = {}
    for x in info:
        for f in (x[1], x[2]):
            fset.setdefault(f.__hash__(), (f, []))[1].append(x)
    f, es = list(fset.values())[rng.integers(len(fset))]
    groups.append(es)
    return [g for g in groups if g]


def fillet_op(body, kind: str, name: str, rng, last_tool, stats: Counter):
    groups = edge_groups(body, last_tool, rng)
    if not groups:
        stats[kind + ": нет рёбер"] += 1
        return None
    group = groups[rng.integers(len(groups))]
    widths = [min(_width(x[1]), _width(x[2])) for x in group]
    group = [x for x, w in zip(group, widths) if w >= MIN_FILLET_FACE_MM]
    if not group:
        stats[kind + ": соседние грани узкие"] += 1
        return None
    if len(group) > 24:
        group = [group[i] for i in rng.choice(len(group), 24, replace=False)]
    width = min(min(_width(x[1]), _width(x[2])) for x in group)
    size = width * rng.uniform(*FILLET_REL)
    asym = kind == "chamfer" and rng.random() < 0.3
    ratio = rng.uniform(0.5, 0.9) if asym else 1.0
    why = "мелко"
    for _ in range(3):
        if size < 0.1:
            break
        mk = BRepFilletAPI_MakeFillet(body) if kind == "fillet" else BRepFilletAPI_MakeChamfer(body)
        try:
            for e, fa, _, _, _ in group:
                if kind == "fillet":
                    mk.Add(size, e)
                else:
                    mk.Add(size, size * ratio, e, fa)
            mk.Build()
            new = mk.Shape() if mk.IsDone() else None
        except Exception:  # noqa: BLE001 — Standard_Failure из OCC
            new = None
        why = unhealthy(new, body, by_volume=False)
        if why is None:
            # полосы — грани нового тела, которых не было в старом и которые не получены изменением старых: так
            # в операцию попадают и угловые сопряжения, которые OCC достраивает там, где скругления встречаются
            kept = set()
            for f in iter_faces(body):
                kept.add(f.__hash__())
                kept.update(g.__hash__() for g in mk.Modified(f))
            surfaces = [BRep_Tool.Surface_s(f) for f in iter_faces(new) if f.__hash__() not in kept]
            op = Op(name, "FilletFeature" if kind == "fillet" else "ChamferFeature", None, "synth", surfaces=surfaces,
                    params={"size": size, "ratio": ratio, "edges": len(group)})
            return new, op
        size /= 2
    stats[f"{kind}: {why}"] += 1
    return None


# ---------- деталь целиком ----------

def generate(base_tools: list[Tool], rng, target_faces: int, max_ops: int, library: dict, stats: Counter | None = None):
    """Тело и история: выдавливания основы + добавки, пока граней меньше `target_faces` и добавок меньше
    `max_ops`. `library` — профили по типу (`profile_library`), `stats` — счётчик попыток и отказов."""
    stats = Counter() if stats is None else stats
    body, ops = None, []
    for t in base_tools:
        body = t.shape if body is None else boolean(body, t.shape, t.boolean)
        ops.append(Op(t.name, "ExtrudeFeature", t.boolean, "recon", tool=t))
    if body is None or count(body, TopAbs_SOLID) != 1:
        raise GenError("основа не одно тело")
    diag = bbox_diag(body)
    kinds, probs = list(KINDS), np.array(list(KINDS.values()))
    added, style = 0, Style()
    for step in range(max_ops * ATTEMPTS):
        if added >= max_ops or count(body, TopAbs_FACE) >= target_faces:
            break
        kind = kinds[rng.choice(len(kinds), p=probs / probs.sum())]
        name = f"Synth{added + 1}"
        last = next((o.tool for o in reversed(ops) if o.tool is not None and o.source == "synth"), None)
        stats[kind + ": попыток"] += 1
        try:
            res = fillet_op(body, kind, name, rng, last, stats) if kind in ("fillet", "chamfer") else \
                extrude_op(body, kind, name, rng, diag, stats, library, style)
        except (GenError, ReconError) as e:
            stats[f"{kind}: {e}"] += 1
            res = None
        if res is None:
            continue
        stats[kind + ": принято"] += 1
        body, op = res
        ops.append(op)
        added += 1
    return body, ops, stats


def strip_roles(final, ops: list[Op], diag: float, tol_rel: float = 5e-4) -> list[list[tuple[int, str]]]:
    """Грань итогового тела -> [(номер операции, "strip")] для скруглений и фасок: точки грани лежат на
    поверхности полосы, которую вернуло ядро (не меньше 90% точек)."""
    tol = tol_rel * diag
    samples = face_samples(final, tol, 60)
    out = [[] for _ in samples]
    for k, op in enumerate(ops):
        if not op.surfaces:
            continue
        for i, p in enumerate(samples):
            if not len(p):
                continue
            on = np.zeros(len(p), bool)
            for s in op.surfaces:
                for j, x in enumerate(p):
                    if on[j]:
                        continue
                    pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, x)), s)
                    on[j] = pr.NbPoints() > 0 and pr.LowerDistance() <= tol
            if on.mean() >= 0.9:
                out[i].append((k, "strip"))
    return out


def label(final, ops: list[Op]) -> list[list[tuple[int, str]]]:
    diag = bbox_diag(final)
    ext = [k for k, o in enumerate(ops) if o.tool is not None]
    roles = face_roles(final, [ops[k].tool for k in ext], diag)
    roles = [[(ext[k], r) for k, r in fr] for fr in roles]
    for i, sr in enumerate(strip_roles(final, ops, diag)):
        roles[i] += sr
    return roles
