"""STEP -> меш с разметкой граней: своя тесселяция вместо корпуса step-dataset.

Корпус step-dataset тесселирован с допуском 5e-4 от диагонали и угловым
0.2 рад. Для буфера идентификаторов SAM-2 этого хватает, для кривизны — нет: на
криволинейных гранях шаг излома 0.32 мм по медиане при радиусе струи 0.23 мм,
в окрестность струи попадает 0.6 излома, и струя меряет фасетки вместо
поверхности — на цилиндрах это полосы с шагом в фасетку (README, «Тесселяция»).
Здесь тот же разбор STEP с более мелким допуском.

Формат выхода совпадает с корпусом step-dataset, чтобы остальной код не менялся:

    mesh.ply      сваренная тесселяция, вырожденные треугольники выброшены
    face_id.npy   номер грани на треугольник ПОСЛЕ слияния швов
    brep.json     тип и параметры каждой исходной грани, смежность с касанием,
                  merge_map: исходная грань -> слитая; если включено в конфиге —
                  halfwidth_mm слитых граней и правило узких граней (narrow, label_map)

Порядок граней — обход TopAbs_FACE, тот же, что у step-dataset: номера граней в
brep.json совпадают, и новые детали сверяются со старыми грань в грань.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.IMeshTools import IMeshTools_Parameters
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from src.config import CFG

# Только у этих пяти есть параметры, которые можно сравнить при слиянии швов.
# Остальное (B-сплайн, поверхность вращения, эквидистанта) записывается по
# имени перечисления и не сливается.
SURFACE_TYPES = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "plane",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "cylinder",
    GeomAbs_SurfaceType.GeomAbs_Cone: "cone",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "sphere",
    GeomAbs_SurfaceType.GeomAbs_Torus: "torus",
}
NORMAL_TOL = 1e-7
# Точная полуширина: точек на ребре не больше стольких (ребро в метр при шаге 0.05 мм);
# треугольник делится решёткой с шагом края, но не мельче чем на HALFWIDTH_SUBDIV частей по
# стороне, и точек на грань не больше FACE_MAX_POINTS — у широкой грани точность не нужна.
EDGE_MAX_POINTS = 20000
HALFWIDTH_SUBDIV = 64
FACE_MAX_POINTS = 400_000


def read_step(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"STEP не читается: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def bbox_diag(shape) -> float:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))


def iter_faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield TopoDS.Face_s(exp.Current())
        exp.Next()


def _xyz(p) -> list[float]:
    return [p.X(), p.Y(), p.Z()]


def tessellate(shape, diag: float, linear_rel: float, angular: float, min_size_rel: float
               ) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Меш, номер исходной грани на треугольник и число треугольников на грань.

    Допуски — доли диагонали детали: абсолютный одинаково бессмыслен на штифте
    в 2 мм и на кронштейне в 300 мм.
    """
    params = IMeshTools_Parameters()
    params.Deflection = diag * linear_rel
    params.Angle = angular
    params.MinSize = diag * min_size_rel
    params.Relative = False
    params.InParallel = False
    BRepMesh_IncrementalMesh(shape, params)
    vertices, triangles, face_id, counts = [], [], [], []
    offset = 0
    for fid, face in enumerate(iter_faces(shape)):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            counts.append(0)
            continue
        trsf = loc.Transformation()
        nodes = np.array([_xyz(tri.Node(i).Transformed(trsf))
                          for i in range(1, tri.NbNodes() + 1)], dtype=np.float64)
        idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)],
                       dtype=np.int64) - 1
        # Грань REVERSED ограничивает тело с другой стороны: без разворота обхода
        # её нормали смотрели бы внутрь детали.
        if face.Orientation() == TopAbs_REVERSED:
            idx = idx[:, [0, 2, 1]]
        vertices.append(nodes)
        triangles.append(idx + offset)
        face_id.append(np.full(len(idx), fid, dtype=np.int32))
        counts.append(len(idx))
        offset += len(nodes)
    if not triangles:
        raise ValueError("тесселяция не дала ни одного треугольника")

    mesh = trimesh.Trimesh(np.vstack(vertices), np.vstack(triangles), process=False)
    # Грани тесселируются по отдельности, и на общих рёбрах вершины дублируются.
    # Без сварки подложка и перенос разметки увидели бы шов как край поверхности.
    mesh.merge_vertices()
    ids = np.concatenate(face_id)

    # Треугольники, схлопнутые на вырожденном ребре B-Rep (полюс сферы, вершина
    # конуса): после сварки у них нулевая площадь и ребро нулевой длины, на
    # которых ломается поиск ближайшей точки. Площади они не несут.
    f = mesh.faces
    degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    if degenerate.any():
        mesh.update_faces(~degenerate)
        ids = ids[~degenerate]
    mesh.remove_unreferenced_vertices()
    tri_counts = np.bincount(ids, minlength=len(counts)).astype(np.int32)
    return mesh, ids, tri_counts


def _surface_params(face, kind: str) -> dict:
    """Параметры поверхности в мировых координатах. Оси и нормали — как их задаёт
    поверхность, без разворота для REVERSED: слияние сравнивает направления с
    точностью до знака."""
    s = BRepAdaptor_Surface(face)
    if kind == "plane":
        pln = s.Plane()
        n, loc = pln.Axis().Direction(), pln.Location()
        return {"normal": _xyz(n), "origin": _xyz(loc),
                "offset": n.X() * loc.X() + n.Y() * loc.Y() + n.Z() * loc.Z()}
    if kind == "cylinder":
        c = s.Cylinder()
        return {"axis": _xyz(c.Axis().Direction()), "location": _xyz(c.Axis().Location()),
                "radius": c.Radius()}
    if kind == "cone":
        c = s.Cone()
        return {"axis": _xyz(c.Axis().Direction()), "apex": _xyz(c.Apex()),
                "semi_angle": c.SemiAngle(), "ref_radius": c.RefRadius()}
    if kind == "sphere":
        c = s.Sphere()
        return {"center": _xyz(c.Location()), "radius": c.Radius()}
    if kind == "torus":
        c = s.Torus()
        return {"axis": _xyz(c.Axis().Direction()), "center": _xyz(c.Axis().Location()),
                "major_radius": c.MajorRadius(), "minor_radius": c.MinorRadius()}
    return {}


def _face_area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _normal_at(face, u: float, v: float) -> np.ndarray | None:
    """Внешняя нормаль тела в точке (u, v) грани."""
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, NORMAL_TOL)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    v3 = np.array([n.X(), n.Y(), n.Z()])
    return -v3 if face.Orientation() == TopAbs_REVERSED else v3


def _side_props(edge, face, ts) -> list:
    """Внешняя нормаль тела и главные кривизны грани (большая, меньшая; знак — по внешней
    нормали) в точках ребра с параметрами `ts`; None, где не определены."""
    c2d = BRepAdaptor_Curve2d(edge, face)
    surf = BRepAdaptor_Surface(face)
    rev = face.Orientation() == TopAbs_REVERSED
    out = []
    for t in ts:
        uv = c2d.Value(float(t))
        props = BRepLProp_SLProps(surf, uv.X(), uv.Y(), 2, NORMAL_TOL)
        if not (props.IsNormalDefined() and props.IsCurvatureDefined()):
            out.append(None)
            continue
        n = np.array(_xyz(props.Normal()))
        k = np.array([props.MaxCurvature(), props.MinCurvature()])
        out.append((-n, -k[::-1]) if rev else (n, k))
    return out


def _seam_jump(edge, face_a, face_b, n_samples: int) -> tuple[float, float] | None:
    """Наибольший угол между нормалями, градусы, и наибольший скачок главных кривизн, мм⁻¹,
    вдоль общего ребра двух граней — для слияния сплайновых швов."""
    c = BRepAdaptor_Curve(edge)
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), n_samples + 2)[1:-1]
    angles, jumps = [], []
    for sa, sb in zip(_side_props(edge, face_a, ts), _side_props(edge, face_b, ts)):
        if sa is None or sb is None:
            continue
        angles.append(np.degrees(np.arccos(np.clip(np.dot(sa[0], sb[0]), -1.0, 1.0))))
        jumps.append(np.abs(sa[1] - sb[1]).max())
    return (float(max(angles)), float(max(jumps))) if angles else None


def _edge_angle(edge, face_a, face_b, n_samples: int) -> float | None:
    """Средний угол между нормалями двух граней вдоль общего ребра, градусы.
    Концы ребра пропускаются: там сходится третья грань и нормаль часто не
    определена."""
    per_face = []
    for face in (face_a, face_b):
        c2d = BRepAdaptor_Curve2d(edge, face)
        ts = np.linspace(c2d.FirstParameter(), c2d.LastParameter(), n_samples + 2)[1:-1]
        per_face.append([_normal_at(face, c2d.Value(float(t)).X(), c2d.Value(float(t)).Y())
                         for t in ts])
    dots = [float(np.clip(np.dot(a, b), -1.0, 1.0))
            for a, b in zip(*per_face) if a is not None and b is not None]
    return float(np.degrees(np.arccos(np.mean(dots)))) if dots else None


def build_brep(shape, diag: float, tri_counts: np.ndarray,
               tangency_samples: int, tangency_deg: float) -> dict:
    """Описание граней и граф смежности с признаком касания."""
    faces = list(iter_faces(shape))
    index = {f.TShape(): i for i, f in enumerate(faces)}
    records = []
    for fid, face in enumerate(faces):
        surf = BRepAdaptor_Surface(face).GetType()
        kind = SURFACE_TYPES.get(surf, str(surf).split(".")[-1].lower())
        records.append({"id": fid, "type": kind, "mergeable": kind in SURFACE_TYPES.values(),
                        "reversed": face.Orientation() == TopAbs_REVERSED,
                        "area": _face_area(face),
                        "n_triangles": int(tri_counts[fid]) if fid < len(tri_counts) else 0,
                        "params": _surface_params(face, kind)})

    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    pairs: dict[tuple[int, int], list[float]] = {}
    seams: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for i in range(1, edge_map.Extent() + 1):
        adjacent = list(edge_map.FindFromIndex(i))
        if len(adjacent) != 2:
            continue
        fa, fb = TopoDS.Face_s(adjacent[0]), TopoDS.Face_s(adjacent[1])
        a, b = index[fa.TShape()], index[fb.TShape()]
        if a == b:                      # шов периодической грани на самой себе
            continue
        edge = TopoDS.Edge_s(edge_map.FindKey(i))
        angle = _edge_angle(edge, fa, fb, tangency_samples)
        if angle is not None:
            pairs.setdefault((min(a, b), max(a, b)), []).append(angle)
        if not records[a]["mergeable"] and not records[b]["mergeable"]:
            jump = _seam_jump(edge, fa, fb, tangency_samples)
            if jump is not None:
                seams.setdefault((min(a, b), max(a, b)), []).append(jump)
    adjacency = []
    for (a, b), ang in sorted(pairs.items()):
        adj = {"a": a, "b": b, "angle_deg": float(min(ang)), "n_edges": len(ang),
               "tangential": bool(min(ang) < tangency_deg)}
        if (a, b) in seams:
            adj["seam_angle_deg"] = max(s[0] for s in seams[(a, b)])
            adj["seam_curv_jump"] = max(s[1] for s in seams[(a, b)])
        adjacency.append(adj)
    return {"n_faces": len(faces), "bbox_diag": diag, "faces": records,
            "adjacency": adjacency, "merge_map": None}


def _aligned(u, v, cos_tol: float) -> tuple[bool, float]:
    """Одна ли ось у двух направлений — с точностью до знака: соседние половины
    одной поверхности часто несут противоположные направления."""
    d = float(np.dot(u, v))
    return abs(d) >= cos_tol, float(np.sign(d) or 1.0)


def same_surface(fa: dict, fb: dict, diag: float, angle_deg: float,
                 length_rel: float, radius_rel: float) -> bool:
    """Одна ли это аналитическая поверхность. Касание критерием не является:
    скругление касается плоскости, но это два разных примитива."""
    if fa["type"] != fb["type"] or not fa["mergeable"]:
        return False
    cos_tol = float(np.cos(np.radians(angle_deg)))
    len_tol = length_rel * diag
    pa, pb = fa["params"], fb["params"]

    def rad_close(ra, rb):
        return abs(ra - rb) <= radius_rel * max(ra, rb, 1e-12)

    if fa["type"] == "plane":
        ok, sign = _aligned(pa["normal"], pb["normal"], cos_tol)
        return ok and abs(pa["offset"] - sign * pb["offset"]) <= len_tol
    if fa["type"] == "cylinder":
        ok, _ = _aligned(pa["axis"], pb["axis"], cos_tol)
        if not (ok and rad_close(pa["radius"], pb["radius"])):
            return False
        axis = np.asarray(pa["axis"], float)
        delta = np.subtract(pb["location"], pa["location"])
        return float(np.linalg.norm(delta - np.dot(delta, axis) * axis)) <= len_tol
    if fa["type"] == "cone":
        ok, _ = _aligned(pa["axis"], pb["axis"], cos_tol)
        return (ok and np.linalg.norm(np.subtract(pa["apex"], pb["apex"])) <= len_tol
                and abs(pa["semi_angle"] - pb["semi_angle"]) <= np.radians(angle_deg))
    if fa["type"] == "sphere":
        return (np.linalg.norm(np.subtract(pa["center"], pb["center"])) <= len_tol
                and rad_close(pa["radius"], pb["radius"]))
    if fa["type"] == "torus":
        ok, _ = _aligned(pa["axis"], pb["axis"], cos_tol)
        rmax = max(pa["major_radius"], pb["major_radius"], 1e-12)
        return (ok and np.linalg.norm(np.subtract(pa["center"], pb["center"])) <= len_tol
                and abs(pa["major_radius"] - pb["major_radius"]) <= radius_rel * rmax
                and abs(pa["minor_radius"] - pb["minor_radius"]) <= radius_rel * rmax)
    return False


def smooth_seam(fa: dict, fb: dict, adj: dict, angle_deg: float, curv_jump: float) -> bool:
    """Шов между неаналитическими гранями, которого на меше не видно: нормали совпадают в
    пределах угла слияния, и главные кривизны не скачут больше `curv_jump`, мм⁻¹ (README,
    «Корпус ABC: план сборки»)."""
    return (not fa["mergeable"] and not fb["mergeable"] and "seam_curv_jump" in adj
            and adj["seam_angle_deg"] < angle_deg and adj["seam_curv_jump"] < curv_jump)


def merge_faces(brep: dict, angle_deg: float, length_rel: float, radius_rel: float,
                seam_curv_jump: float = 0.0) -> np.ndarray:
    """merge_map: исходная грань -> номер слитой, подряд с нуля. Union-find по
    смежности; порядок обхода на итоговое разбиение не влияет."""
    parent = list(range(brep["n_faces"]))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for adj in brep["adjacency"]:
        fa, fb = brep["faces"][adj["a"]], brep["faces"][adj["b"]]
        if (same_surface(fa, fb, brep["bbox_diag"], angle_deg, length_rel, radius_rel)
                or smooth_seam(fa, fb, adj, angle_deg, seam_curv_jump)):
            ra, rb = find(adj["a"]), find(adj["b"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(brep["n_faces"])])
    return np.unique(roots, return_inverse=True)[1].astype(np.int32)


def _boundary_samples(shape, merge_map: np.ndarray, spacing: float) -> list[dict]:
    """Точки на рёбрах между разными слитыми гранями (и на свободных рёбрах) с шагом
    `spacing`, мм. По ребру: точки, параметры, слитые грани по обе стороны и сами грани."""
    faces = list(iter_faces(shape))
    index = {f.TShape(): i for i, f in enumerate(faces)}
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    out = []
    for i in range(1, edge_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        adjacent = [TopoDS.Face_s(f) for f in edge_map.FindFromIndex(i)]
        labels = [int(merge_map[index[f.TShape()]]) for f in adjacent]
        if len(adjacent) == 2 and labels[0] == labels[1]:
            continue                    # шов внутри слитой грани
        curve = BRepAdaptor_Curve(edge)
        length = GCPnts_AbscissaPoint.Length_s(curve)
        n = int(np.clip(length / spacing, 2, EDGE_MAX_POINTS))
        ts = np.linspace(curve.FirstParameter(), curve.LastParameter(), n)
        points = np.array([_xyz(curve.Value(float(t))) for t in ts])
        out.append({"edge": edge, "ts": ts, "points": points, "labels": labels[:2],
                    "faces": adjacent[:2]})
    return out


def _interior_points(mesh: trimesh.Trimesh, tri: np.ndarray, spacing: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Точки внутри треугольников грани: барицентрическая решётка с шагом около `spacing`.
    Тесселяция кладёт на узкую полосу длинные треугольники во всю ширину, и решётка
    реже шага промахнулась бы мимо её середины."""
    corners = mesh.vertices[mesh.faces[tri]]
    longest = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(1)
    steps = np.clip(np.ceil(longest / spacing), 1, HALFWIDTH_SUBDIV).astype(int)
    per_tri = (steps + 1) * (steps + 2) // 2
    if per_tri.sum() > FACE_MAX_POINTS:
        keep = rng.permutation(len(tri))[:max(1, int(len(tri) * FACE_MAX_POINTS / per_tri.sum()))]
        corners, steps = corners[keep], steps[keep]
    out = []
    for s in np.unique(steps):
        u = np.array([(i, j) for i in range(s + 1) for j in range(s + 1 - i)], float) / s
        bary = np.c_[u, 1 - u.sum(1)]
        out.append(np.einsum("bk,tkd->tbd", bary, corners[steps == s]).reshape(-1, 3))
    return np.vstack(out)


def face_halfwidth(mesh: trimesh.Trimesh, labels: np.ndarray, samples: list[dict], spacing: float,
                   n: int) -> np.ndarray:
    """Полуширина слитых граней, мм: радиус вписанного круга — наибольшее расстояние от
    точек грани до её края (README, «Узкие грани: ширина в мм и правило разметки»).
    Грань без края (замкнутая) или без треугольников — -1."""
    edge_points: list[list[np.ndarray]] = [[] for _ in range(n)]
    for s in samples:
        for label in set(s["labels"]):
            if label < n:
                edge_points[label].append(s["points"])
    rng = np.random.default_rng(0)
    halfwidth = np.full(n, -1.0)
    for label in range(n):
        tri = np.flatnonzero(labels == label)
        if not len(tri) or not edge_points[label]:
            continue
        tree = cKDTree(np.vstack(edge_points[label]))
        halfwidth[label] = float(tree.query(_interior_points(mesh, tri, spacing, rng))[0].max())
    return halfwidth


def classify_narrow(samples: list[dict], halfwidth: np.ndarray, max_mm: float, step_deg: float,
                    turn_deg: float) -> tuple[list[dict], np.ndarray]:
    """Правило узких граней (решено пользователем 2026-10-09): грань с полушириной до `max_mm`
    между двумя соседями — ступенька, если внешние нормали соседей друг напротив друга почти
    сонаправлены (грань и оба соседа — одна грань), разворот, если почти противоположны
    (грань остаётся), иначе ребро (грань делится между соседями при загрузке).

    Соседи — два с наибольшей длиной общего края; нормали сравниваются в парах ближайших
    точек двух краёв, не дальше четырёх полуширин друг от друга, угол — медиана по парам.
    label_map: слитая грань -> итоговая метка, -1 — отдать соседям.
    """
    n = len(halfwidth)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    narrow, absorb = [], np.zeros(n, bool)
    for label in np.flatnonzero((halfwidth >= 0) & (halfwidth <= max_mm)):
        sides: dict[int, list] = {}
        for s in samples:
            if label not in s["labels"] or len(s["labels"]) != 2 or s["labels"][0] == s["labels"][1]:
                continue
            k = s["labels"].index(label)
            sides.setdefault(s["labels"][1 - k], []).append((s, s["faces"][1 - k]))
        length = {nb: sum(len(s["ts"]) for s, _ in v) for nb, v in sides.items()}
        record = {"label": int(label), "halfwidth": float(halfwidth[label]), "rule": "edge",
                  "angle_deg": None, "neighbors": sorted(length, key=length.get, reverse=True)[:2]}
        if len(record["neighbors"]) == 2:
            pts, nrm = [], []
            for nb in record["neighbors"]:
                p, v = [], []
                for s, face in sides[nb]:
                    for point, props in zip(s["points"], _side_props(s["edge"], face, s["ts"])):
                        if props is not None:
                            p.append(point)
                            v.append(props[0])
                pts.append(np.array(p).reshape(-1, 3))
                nrm.append(np.array(v).reshape(-1, 3))
            if len(pts[0]) and len(pts[1]):
                dist, j = cKDTree(pts[1]).query(pts[0])
                near = dist <= 4 * halfwidth[label] + 1e-9
                if near.any():
                    cos = np.clip(np.einsum("ij,ij->i", nrm[0][near], nrm[1][j[near]]), -1.0, 1.0)
                    record["angle_deg"] = float(np.degrees(np.arccos(np.median(cos))))
        angle = record["angle_deg"]
        if angle is not None and angle < step_deg:
            record["rule"] = "step"
            for nb in record["neighbors"]:
                ra, rb = find(int(label)), find(int(nb))
                parent[max(ra, rb)] = min(ra, rb)
        elif angle is not None and angle > 180 - turn_deg:
            record["rule"] = "turn"
        else:
            absorb[label] = True
        narrow.append(record)
    label_map = np.array([find(i) for i in range(n)])
    roots = np.array([find(i) for i in range(n)])
    # Ребро, втянутое в ступеньку соседа, остаётся в её группе.
    label_map[absorb & (np.bincount(roots, minlength=n)[roots] == 1)] = -1
    return narrow, label_map


def build_part(step_path: Path, out_dir: Path) -> dict:
    """Одна деталь целиком, допуски из `configs/default.yaml`. Пишет во временную папку и переименовывает в конце:
    оборванная сборка не оставит полуготовой детали, которую примут за готовую."""
    shape = read_step(step_path)
    diag = bbox_diag(shape)
    t = CFG.tessellation
    mesh, fid, counts = tessellate(shape, diag, t.linear_rel, t.angular, t.min_size_rel)
    brep = build_brep(shape, diag, counts, t.tangency_samples, t.tangency_deg)
    merge_map = merge_faces(brep, t.merge_angle_deg, t.merge_length_rel, t.merge_radius_rel, t.seam_curv_jump)
    brep["merge_map"] = merge_map.tolist()
    if t.halfwidth_spacing_mm > 0:
        samples = _boundary_samples(shape, merge_map, t.halfwidth_spacing_mm)
        halfwidth = face_halfwidth(mesh, merge_map[fid], samples, t.halfwidth_spacing_mm, int(merge_map.max()) + 1)
        brep["halfwidth_mm"] = halfwidth.round(4).tolist()
        if t.narrow_halfwidth_mm > 0:
            narrow, label_map = classify_narrow(samples, halfwidth, t.narrow_halfwidth_mm,
                                                t.narrow_step_deg, t.narrow_turn_deg)
            brep["narrow"] = narrow
            brep["label_map"] = label_map.tolist()

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mesh.export(str(tmp / "mesh.ply"))
    np.save(tmp / "face_id.npy", merge_map[fid])
    (tmp / "brep.json").write_text(json.dumps(brep, indent=1))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return {"n_triangles": int(len(mesh.faces)), "n_faces": brep["n_faces"],
            "n_merged": int(merge_map.max()) + 1, "watertight": bool(mesh.is_watertight)}
