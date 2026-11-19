"""Исполнение операций ядром OCC (README, «Ядро: исполнение операций»).

Эскиз выдавливания (петли `src/sketch/projection.py`, координаты по frame[:2] без сдвига начала) ->
контур -> грань на плоскости опоры -> выдавливание -> булева с телом; скругление и фаска — по рёбрам
между гранями тела, сопоставленными гранями цели. Грани тела сопоставляются с узлами цели по
точкам скана: точка узла -> ближайший треугольник тесселяции тела -> его грань.
"""
from __future__ import annotations

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import Geom_BSplineCurve
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
from scipy.spatial import cKDTree

from src.arrangement import unify
from src.sketch.curves import sample_curve

BOOLEAN_OPS = {"add": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "intersect": BRepAlgoAPI_Common}
# Короче этого / диагональ кривая эскиза не строится (вырожденный кусок между близкими стыками);
# объём выдавливания меньше MIN_VOLUME_REL · D³ — вырожденное. Выбрано.
MIN_EDGE_REL = 1e-5
MIN_VOLUME_REL = 1e-9


class KernelError(Exception):
    pass


def _pnt(p2: np.ndarray, frame: np.ndarray, height: float) -> gp_Pnt:
    p = p2[0] * frame[0] + p2[1] * frame[1] + height * frame[2]
    return gp_Pnt(*map(float, p))


def edge_of(curve: tuple, frame: np.ndarray, height: float, min_len: float):
    """Кривая эскиза (как `Piece.curve`) -> ребро OCC; None — вырожденная."""
    kind = curve[0]
    if kind == "line":
        if np.linalg.norm(curve[2] - curve[1]) < min_len:
            return None
        return BRepBuilderAPI_MakeEdge(_pnt(curve[1], frame, height), _pnt(curve[2], frame, height)).Edge()
    if kind == "circle":
        c = gp_Circ(gp_Ax2(_pnt(curve[1], frame, height), gp_Dir(*map(float, frame[2]))), float(curve[2]))
        return BRepBuilderAPI_MakeEdge(c).Edge()
    if kind == "arc":
        a, b = curve[3], curve[4]
        if np.linalg.norm(b - a) < min_len:
            return None
        mid = sample_curve(curve, 3)
        if mid is None:
            return None
        arc = GC_MakeArcOfCircle(_pnt(a, frame, height), _pnt(mid[1], frame, height), _pnt(b, frame, height))
        if not arc.IsDone():
            return BRepBuilderAPI_MakeEdge(_pnt(a, frame, height), _pnt(b, frame, height)).Edge()
        return BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
    if kind == "spline":
        _, knots, coefs, degree = curve
        if np.linalg.norm(coefs[-1] - coefs[0]) < min_len and len(coefs) <= degree + 1:
            return None
        poles = TColgp_Array1OfPnt(1, len(coefs))
        for i, c in enumerate(coefs, 1):
            poles.SetValue(i, _pnt(c, frame, height))
        values, mults = np.unique(np.round(knots, 12), return_counts=True)
        kv, km = TColStd_Array1OfReal(1, len(values)), TColStd_Array1OfInteger(1, len(values))
        for i, (x, m) in enumerate(zip(values, mults), 1):
            kv.SetValue(i, float(x))
            km.SetValue(i, int(m))
        return BRepBuilderAPI_MakeEdge(Geom_BSplineCurve(poles, kv, km, int(degree))).Edge()
    raise KernelError(f"кривая {kind} не поддержана")


def _area(shape) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return abs(props.Mass())


def loop_face(loop: list, frame: np.ndarray, height: float, min_len: float):
    wire = BRepBuilderAPI_MakeWire()
    for c in loop:
        e = edge_of(c.curve, frame, height, min_len)
        if e is not None:
            wire.Add(e)
    if not wire.IsDone():
        raise KernelError("контур не замкнулся в провод")
    face = BRepBuilderAPI_MakeFace(wire.Wire(), True)
    if not face.IsDone():
        raise KernelError("грань эскиза не построена")
    return face.Face()


def profile_face(loops: list, frame: np.ndarray, height: float, min_len: float):
    """Грань профиля по правилу чётности: петли по убыванию площади, петля внутри набранного —
    вырезается, иначе добавляется."""
    faces = sorted((loop_face(lp, frame, height, min_len) for lp in loops), key=_area, reverse=True)
    out = faces[0]
    for f in faces[1:]:
        inside = _area(BRepAlgoAPI_Common(out, f).Shape()) > 0.5 * _area(f)
        out = (BRepAlgoAPI_Cut if inside else BRepAlgoAPI_Fuse)(out, f).Shape()
    return out


def _volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def extrude(profile, direction: np.ndarray, length: float, min_volume: float):
    out = BRepPrimAPI_MakePrism(profile, gp_Vec(*map(float, direction * length))).Shape()
    if _volume(out) < min_volume:
        raise KernelError("выдавливание без объёма")
    return out


def _valid(shape) -> bool:
    return shape is not None and not shape.IsNull() and BRepCheck_Analyzer(shape).IsValid()


def boolean(body, tool, kind: str):
    if body is None:
        if kind != "add":
            raise KernelError("первая операция — не добавление")
        return tool
    op = BOOLEAN_OPS[kind](body, tool)
    if not op.IsDone():
        raise KernelError(f"булева {kind} не прошла")
    out = unify(op.Shape())
    if not _valid(out):
        raise KernelError(f"булева {kind}: невалидное тело")
    return out


def faces_of(shape) -> list:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return [TopoDS.Face_s(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def tessellate_faces(shape, deflection: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Меш тела и номер грани (в `faces_of`) на треугольник."""
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    verts, tris, owner = [], [], []
    for k, face in enumerate(faces_of(shape)):
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
            owner.append(k)
    return trimesh.Trimesh(np.array(verts), np.array(tris), process=False), np.array(owner)


class BodySamples:
    """Точки по площади тела с гранью-владельцем и дерево по ним: расстояния и сопоставление граней
    без rtree (шаг точек ~0.07 мм на теле Fusion при 200 тысячах — меньше допусков)."""

    def __init__(self, shape, deflection: float, count: int, seed: int = 0):
        mesh, owner = tessellate_faces(shape, deflection)
        if not len(mesh.faces):
            raise KernelError("тело без тесселяции")
        self.points, tri = trimesh.sample.sample_surface(mesh, count, seed=seed)
        self.owner = owner[tri]
        self.tree = cKDTree(self.points)

    def match(self, points: dict[int, np.ndarray], tol: float) -> dict[int, tuple[int, float]]:
        """Узел цели -> (грань тела большинства его точек, доля точек не дальше `tol` от тела)."""
        out = {}
        for node, p in points.items():
            dist, i = self.tree.query(p)
            out[node] = (int(np.bincount(self.owner[i]).argmax()), float((dist <= tol).mean()))
        return out

    def distances(self, scan: np.ndarray, scan_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Скан -> тело (вершины скана) и тело -> скан (точки тела до точек по площади скана: до
        вершин скана с ребром 0.3 мм выходило ~0.1-0.15 мм даром)."""
        return self.tree.query(scan)[0], cKDTree(scan_samples).query(self.points)[0]


def common_edges(shape, fa, fb) -> list:
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        owners = list(m.FindFromIndex(i))
        if any(o.IsSame(fa) for o in owners) and any(o.IsSame(fb) for o in owners):
            out.append(TopoDS.Edge_s(m.FindKey(i)))
    return out


def fillet(body, edges: list[tuple], kind: str):
    """`edges` — (ребро, радиус) у скругления, (ребро, d1, d2, грань со стороны d1) у фаски."""
    mk = BRepFilletAPI_MakeFillet(body) if kind == "fillet" else BRepFilletAPI_MakeChamfer(body)
    for e in edges:
        if kind == "fillet":
            mk.Add(float(e[1]), e[0])
        else:
            mk.Add(float(e[1]), float(e[2]), e[0], e[3])
    try:
        mk.Build()
    except Exception as err:  # noqa: BLE001 — Standard_Failure из OCC
        raise KernelError(f"{kind}: {err}") from None
    if not mk.IsDone():
        raise KernelError(f"{kind} не построено")
    out = unify(mk.Shape())
    if not _valid(out):
        raise KernelError(f"{kind}: невалидное тело")
    return out
