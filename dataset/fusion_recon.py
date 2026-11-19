"""Выдавливания Fusion 360 Gallery Reconstruction (r1.0.1): тела инструментов из профилей эскизов.

Проект — JSON с эскизами (кривые профилей в системе эскиза, матрица эскиза) и выдавливаниями
(профили, протяжённость расстоянием, булева) + STEP итогового тела и после каждого выдавливания.
Единицы JSON — сантиметры, STEP читается OCC в миллиметрах.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_Transform)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GC import GC_MakeArcOfCircle, GC_MakeArcOfEllipse
from OCP.Geom import Geom_BSplineCurve
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Elips, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

from dataset.step_mesh import iter_faces

CM = 10.0          # JSON в см, STEP через OCC — в мм
ROOT = "r1.0.1/reconstruction/"


class ReconError(Exception):
    pass


@dataclass
class Tool:
    name: str
    boolean: str                 # add | cut | intersect
    frame: np.ndarray            # (4, 3): начало, оси x, y, z эскиза (мир, мм)
    h0: float                    # протяжённость по z эскиза, мм
    h1: float
    edges: list                  # рёбра профилей OCC (мир) — по ним стенки при разметке
    shape: object                # тело инструмента OCC (мир)
    faces: list                  # грани профилей в плоскости эскиза (мир)


def _v(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], float)


def _frame(sketch: dict, unit: float) -> np.ndarray:
    t = sketch["transform"]
    return np.stack([_v(t["origin"]) * unit, _v(t["x_axis"]), _v(t["y_axis"]), _v(t["z_axis"])])


def to_world(p: np.ndarray, frame: np.ndarray, unit: float) -> np.ndarray:
    """Точки системы эскиза (единицы JSON) -> мир (мм)."""
    return frame[0] + (p * unit) @ frame[1:]


def _pnt(p: np.ndarray, frame: np.ndarray, unit: float) -> gp_Pnt:
    return gp_Pnt(*map(float, to_world(p, frame, unit)))


def _dir(v: np.ndarray, frame: np.ndarray) -> gp_Dir:
    return gp_Dir(*map(float, v @ frame[1:]))


def edge(c: dict, frame: np.ndarray, unit: float = CM):
    kind = c["type"]
    if kind == "Line3D":
        return BRepBuilderAPI_MakeEdge(_pnt(_v(c["start_point"]), frame, unit), _pnt(_v(c["end_point"]), frame, unit)).Edge()
    if kind == "Circle3D":
        circ = gp_Circ(gp_Ax2(_pnt(_v(c["center_point"]), frame, unit), _dir(_v(c["normal"]), frame)), c["radius"] * unit)
        return BRepBuilderAPI_MakeEdge(circ).Edge()
    if kind == "Arc3D":
        ctr, n = _v(c["center_point"]), _v(c["normal"])
        circ = gp_Circ(gp_Ax2(_pnt(ctr, frame, unit), _dir(n, frame)), c["radius"] * unit)
        arc = GC_MakeArcOfCircle(circ, _pnt(_v(c["start_point"]), frame, unit), _pnt(_v(c["end_point"]), frame, unit), True)
        return BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
    if kind in ("Ellipse3D", "EllipticalArc3D"):
        ctr, n, major = _v(c["center_point"]), _v(c["normal"]), _v(c["major_axis"])
        ax = gp_Ax2(_pnt(ctr, frame, unit), _dir(n, frame), _dir(major, frame))
        el = gp_Elips(ax, c["major_axis_radius"] * unit, c["minor_axis_radius"] * unit)
        if kind == "Ellipse3D":
            return BRepBuilderAPI_MakeEdge(el).Edge()
        arc = GC_MakeArcOfEllipse(el, float(c["start_parameter"]), float(c["end_parameter"]), True)
        return BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
    if kind == "NurbsCurve3D":
        pts = [_v(p) for p in c["control_points"]]
        poles = TColgp_Array1OfPnt(1, len(pts))
        for i, p in enumerate(pts, 1):
            poles.SetValue(i, _pnt(p, frame, unit))
        values, mults = np.unique(np.round(np.asarray(c["knots"], float), 12), return_counts=True)
        kv, km = TColStd_Array1OfReal(1, len(values)), TColStd_Array1OfInteger(1, len(values))
        for i, (x, m) in enumerate(zip(values, mults), 1):
            kv.SetValue(i, float(x))
            km.SetValue(i, int(m))
        if c.get("rational"):
            w = TColStd_Array1OfReal(1, len(pts))
            for i, x in enumerate(c["weights"], 1):
                w.SetValue(i, float(x))
            curve = Geom_BSplineCurve(poles, w, kv, km, int(c["degree"]), bool(c.get("periodic")))
        else:
            curve = Geom_BSplineCurve(poles, kv, km, int(c["degree"]), bool(c.get("periodic")))
        return BRepBuilderAPI_MakeEdge(curve).Edge()
    raise ReconError(f"кривая {kind}")


def profile_face(loops: list, frame: np.ndarray, unit: float = CM):
    wires = []
    for outer, curves in loops:
        w = BRepBuilderAPI_MakeWire()
        for c in curves:
            w.Add(edge(c, frame, unit))
        if not w.IsDone():
            raise ReconError("петля профиля не замкнулась")
        wires.append((outer, w.Wire()))
    outer = [w for o, w in wires if o]
    if len(outer) != 1:
        raise ReconError(f"внешних петель {len(outer)}")
    # отверстия вычитаются гранями: добавленная в грань петля с обходом не в ту сторону заливалась
    face = BRepBuilderAPI_MakeFace(outer[0], True).Shape()
    for o, w in wires:
        if not o:
            face = BRepAlgoAPI_Cut(face, BRepBuilderAPI_MakeFace(w, True).Shape()).Shape()
    return face


def extent(e: dict, unit: float = CM) -> tuple[float, float]:
    """(h0, h1) по нормали эскиза, мм."""
    if e["start_extent"]["type"] != "ProfilePlaneStartDefinition":
        raise ReconError("начало со смещением")
    for k in ("extent_one", "extent_two"):
        if k in e and abs(e[k].get("taper_angle", {}).get("value", 0)) > 1e-9:
            raise ReconError("уклон")
    d1 = e["extent_one"]["distance"]["value"] * unit
    t = e["extent_type"]
    if t == "OneSideFeatureExtentType":
        return (0.0, d1) if d1 >= 0 else (d1, 0.0)
    if t == "SymmetricFeatureExtentType":
        half = d1 / 2 if e["extent_one"].get("is_full_length") else d1
        return -abs(half), abs(half)
    if t == "TwoSidesFeatureExtentType":
        d2 = e["extent_two"]["distance"]["value"] * unit
        return -d2, d1
    raise ReconError(f"протяжённость {t}")


BOOLEAN = {"NewBodyFeatureOperation": "add", "JoinFeatureOperation": "add", "CutFeatureOperation": "cut",
           "IntersectFeatureOperation": "intersect"}


def make_tool(name: str, boolean: str, frame: np.ndarray, h0: float, h1: float, faces: list) -> Tool:
    """Тело инструмента: грани профилей в плоскости эскиза, вытянутые от h0 до h1 по его нормали."""
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    n = frame[3]
    shapes, edges = [], []
    for face in faces:
        tr = gp_Trsf()
        tr.SetTranslation(gp_Vec(*map(float, n * h0)))
        moved = BRepBuilderAPI_Transform(face, tr, True).Shape()
        shapes.append(BRepPrimAPI_MakePrism(moved, gp_Vec(*map(float, n * (h1 - h0)))).Shape())
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edges.append(TopoDS.Edge_s(exp.Current()))
            exp.Next()
    shape = shapes[0]
    for x in shapes[1:]:
        shape = BRepAlgoAPI_Fuse(shape, x).Shape()
    return Tool(name, boolean, frame, h0, h1, edges, shape, faces)


def tools(design: dict, scale: float = 1.0) -> list[Tool]:
    """Инструменты выдавливаний проекта; `scale` — множитель масштаба детали (`recon_scale.py`)."""
    unit = CM * scale
    ents = design["entities"]
    out = []
    for item in design["timeline"]:
        e = ents[item["entity"]]
        if e["type"] != "ExtrudeFeature":
            continue
        h0, h1 = extent(e, unit)
        faces, frame = [], None
        for ref in e["profiles"]:
            sk = ents[ref["sketch"]]
            fr = _frame(sk, unit)
            if frame is not None and not np.allclose(fr, frame, atol=1e-6):
                raise ReconError("профили из разных эскизов")
            frame = fr
            loops = [(lp["is_outer"], lp["profile_curves"]) for lp in sk["profiles"][ref["profile"]]["loops"]]
            faces += list(iter_faces(profile_face(loops, fr, unit)))
        out.append(make_tool(e["name"], BOOLEAN[e["operation"]], frame, h0, h1, faces))
    return out


def load(z: zipfile.ZipFile, design: str) -> dict:
    return json.loads(z.read(f"{ROOT}{design}.json"))


# Точка грани итогового тела лежит на стенке инструмента, если в системе эскиза она на кривой профиля в
# пределах протяжённости, на торце — если на плоскости начала или конца внутри профиля. Грань размечена,
# если так лежат не меньше ROLE_SHARE её точек; роль получает каждый инструмент с долей точек не меньше
# ROLE_PART — грань бывает сложена из кусков двух операций (стенка, нижняя половина которой от одного
# выдавливания, верхняя — от другого, или общий торец двух выдавливаний из одной плоскости). Допуск —
# ROLE_TOL_REL · D, шаг точек кривых профиля — его половина. Выбрано.
ROLE_SHARE = 0.9
ROLE_PART = 0.1
ROLE_TOL_REL = 5e-4
FACE_POINTS = 200


def _curve_points(shape, step: float) -> np.ndarray:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        u = GCPnts_UniformAbscissa(c, step)
        ts = [u.Parameter(i) for i in range(1, u.NbPoints() + 1)] if u.IsDone() else \
            list(np.linspace(c.FirstParameter(), c.LastParameter(), 64))
        out += [(p.X(), p.Y(), p.Z()) for p in map(c.Value, ts)]
        exp.Next()
    return np.array(out)


def face_samples(shape, deflection: float, count: int) -> list[np.ndarray]:
    """Точки по площади каждой грани (порядок `TopExp_Explorer`, как в `step_mesh.iter_faces`). `deflection` —
    допуск, с которым точки потом сравниваются с поверхностями; сетка вчетверо мельче, иначе точки на хордах
    изогнутой грани отходят от неё почти на допуск (полоса скругления теряла до четверти точек)."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(shape, deflection / 4, False, 0.5, True)
    out = []
    for face in iter_faces(shape):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            out.append(np.zeros((0, 3)))
            continue
        trsf = loc.Transformation()
        v = np.array([[q.X(), q.Y(), q.Z()] for q in (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))])
        f = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
        out.append(trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), count, seed=0)[0])
    return out


def _inside(t: Tool, q: np.ndarray, tol: float) -> np.ndarray:
    """Точки (система эскиза, мм) внутри профилей инструмента — классификатором OCC по граням профиля."""
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    out = np.zeros(len(q), bool)
    for i, (x, y, _) in enumerate(q):
        p = gp_Pnt(*map(float, t.frame[0] + x * t.frame[1] + y * t.frame[2]))
        out[i] = any(BRepClass_FaceClassifier(f, p, tol).State() in (TopAbs_IN, TopAbs_ON) for f in t.faces)
    return out


def face_roles(final, tool_list: list[Tool], diag: float) -> list[list[tuple[int, str]]]:
    """Грань итогового тела -> [(номер инструмента, "side" | "start" | "end")]; пустой список — грань
    не легла на инструменты (не размечена)."""
    from scipy.spatial import cKDTree
    tol = ROLE_TOL_REL * diag
    samples = face_samples(final, tol, FACE_POINTS)
    geo = []
    for t in tool_list:
        curve_w = np.concatenate([_curve_points(e, tol / 2) for e in t.edges])
        geo.append(cKDTree(((curve_w - t.frame[0]) @ t.frame[1:].T)[:, :2]))
    roles = []
    for p in samples:
        if not len(p):
            roles.append([])
            continue
        masks = {}
        for k, (t, tree) in enumerate(zip(tool_list, geo)):
            q = (p - t.frame[0]) @ t.frame[1:].T
            z = q[:, 2]
            side = (tree.query(q[:, :2])[0] <= tol) & (z >= t.h0 - tol) & (z <= t.h1 + tol)
            masks[(k, "side")] = side
            for name, h in (("start", t.h0), ("end", t.h1)):
                at = (np.abs(z - h) <= tol) & ~side
                if at.mean() >= ROLE_PART:
                    at[at] = _inside(t, q[at], tol)
                masks[(k, name)] = at
        covered = np.logical_or.reduce(list(masks.values()))
        roles.append([key for key, m in masks.items() if m.mean() >= ROLE_PART] if covered.mean() >= ROLE_SHARE else [])
    return roles
