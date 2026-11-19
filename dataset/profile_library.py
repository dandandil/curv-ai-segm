"""Библиотека профилей эскизов реконструкции для генератора (README, «Генератор деталей»).

Профиль — петли кривых JSON одного профиля эскиза, нормированные: центр многоугольника — в начале, наибольший
радиус — 1. Хранятся кривые (для ядра) и многоугольник (для проверок «помещается ли»).

    python -m dataset.profile_library --zip <r1.0.1.zip> --out <profiles.pkl>
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import zipfile
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Профили сложнее этого не берутся: на детали в 20-30 мм они выродились бы в мелочь. Выбрано.
MAX_CURVES = 40
MAX_LOOPS = 6


@dataclass
class Profile:
    loops: list                 # [(is_outer, [кривая JSON, ...])], нормированные
    poly: Polygon
    kind: str                   # round | rect | other
    curves: int


POINTS = ("start_point", "end_point", "center_point")
VECTORS = ("normal", "major_axis", "reference_vector")
RADII = ("radius", "major_axis_radius", "minor_axis_radius")


def transform_curve(c: dict, s: float, a: float, t: np.ndarray) -> dict:
    """Подобие в плоскости эскиза: x -> s·R(a)·x + t."""
    ca, sa = np.cos(a), np.sin(a)
    out = copy.deepcopy(c)

    def P(p):
        x, y = p["x"], p["y"]
        p["x"], p["y"] = float(s * (ca * x - sa * y) + t[0]), float(s * (sa * x + ca * y) + t[1])

    def V(v):
        x, y = v["x"], v["y"]
        v["x"], v["y"] = float(ca * x - sa * y), float(sa * x + ca * y)

    for k in POINTS:
        if k in out:
            P(out[k])
    for p in out.get("control_points", []):
        P(p)
    for k in VECTORS:
        if k in out:
            V(out[k])
    for k in RADII:
        if k in out:
            out[k] = float(out[k] * s)
    return out


def place(p: Profile, s: float, a: float, t) -> tuple[list, Polygon]:
    t = np.asarray(t, float)
    loops = [(o, [transform_curve(c, s, a, t) for c in cs]) for o, cs in p.loops]
    poly = affinity.translate(affinity.scale(affinity.rotate(p.poly, a, origin=(0, 0), use_radians=True), s, s, origin=(0, 0)), *t)
    return loops, poly


# Прогиб тесселяции профиля — доля его размера: абсолютный 1e-3 у профилей в десятки сантиметров давал миллионы
# треугольников. Выбрано.
DEFLECTION_REL = 2e-3


def _polygon(loops: list) -> Polygon | None:
    """Многоугольник профиля: грань OCC в системе эскиза -> тесселяция -> объединение треугольников."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    from dataset.fusion_recon import profile_face
    from dataset.step_mesh import iter_faces
    frame = np.stack([np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]])
    face = profile_face(loops, frame, 1.0)
    box = Bnd_Box()
    BRepBndLib.Add_s(face, box)
    x0, y0, _, x1, y1, _ = box.Get()
    BRepMesh_IncrementalMesh(face, DEFLECTION_REL * float(np.hypot(x1 - x0, y1 - y0)), False, 0.3, True)
    tris = []
    for f in iter_faces(face):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        v = np.array([[q.X(), q.Y()] for q in (tri.Node(i) for i in range(1, tri.NbNodes() + 1))])
        tris += [Polygon(v[np.array(tri.Triangle(i).Get()) - 1]) for i in range(1, tri.NbTriangles() + 1)]
    if not tris:
        return None
    return unary_union(tris).buffer(0)


def _design_profiles(args: tuple[str, str]) -> list[Profile]:
    zip_path, name = args
    j = json.loads(zipfile.ZipFile(zip_path).read(name))
    out = []
    for e in j["entities"].values():
        if e["type"] != "Sketch" or "profiles" not in e:
            continue
        for prof in e["profiles"].values():
            loops = [(lp["is_outer"], lp["profile_curves"]) for lp in prof["loops"]]
            nc = sum(len(cs) for _, cs in loops)
            if nc > MAX_CURVES or len(loops) > MAX_LOOPS or sum(o for o, _ in loops) != 1:
                continue
            try:
                poly = _polygon(loops)
            except Exception:  # noqa: BLE001 — OCC или незамкнутая петля
                continue
            if poly is None or poly.is_empty or not isinstance(poly, Polygon) or poly.area <= 0:
                continue
            c = np.array(poly.centroid.coords[0])
            r = float(max(np.linalg.norm(np.array(poly.exterior.coords) - c, axis=1)))
            if r <= 0:
                continue
            loops_n = [(o, [transform_curve(cv, 1 / r, 0.0, -c / r) for cv in cs]) for o, cs in loops]
            poly_n = affinity.scale(affinity.translate(poly, *(-c)), 1 / r, 1 / r, origin=(0, 0))
            outer = next(cs for o, cs in loops if o)
            kinds = {cv["type"] for cv in outer}
            kind = "round" if kinds == {"Circle3D"} else "rect" if kinds == {"Line3D"} and len(outer) == 4 else "other"
            out.append(Profile(loops_n, poly_n, kind, nc))
    return out


def build(zip_path: str, limit: int | None = None, workers: int = 8) -> list[Profile]:
    from multiprocessing import Pool
    z = zipfile.ZipFile(zip_path)
    names = sorted(n for n in z.namelist() if n.endswith(".json") and n.count("/") == 2)[:limit]
    out, seen = [], set()
    with Pool(workers) as pool:
        for k, profs in enumerate(pool.imap_unordered(_design_profiles, [(zip_path, n) for n in names], chunksize=8), 1):
            for p in profs:
                key = (p.curves, len(p.loops), round(p.poly.area, 3), round(p.poly.length, 2))
                if key not in seen:
                    seen.add(key)
                    out.append(p)
            if k % 1000 == 0:
                print(f"{k}/{len(names)} профилей {len(out)}", flush=True)
    return out


class _Unpickler(pickle.Unpickler):
    # библиотека собрана через `python -m`: класс записан как __main__.Profile
    def find_class(self, module, name):
        return Profile if name == "Profile" else super().find_class(module, name)


def load(path) -> dict[str, list[Profile]]:
    """Профили библиотеки по типу (round | rect | other)."""
    out = {}
    with open(path, "rb") as f:
        for p in _Unpickler(f).load():
            out.setdefault(p.kind, []).append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    lib = build(args.zip, args.limit, args.workers)
    with open(args.out, "wb") as f:
        pickle.dump(lib, f)
    from collections import Counter
    print(len(lib), Counter(p.kind for p in lib))


if __name__ == "__main__":
    main()
