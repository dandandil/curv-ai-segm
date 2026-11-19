"""Метки шагов opseq по телам Fusion 360 Segmentation: `timeline_info` + точный меш этапа 1 -> opseq.json.

    CURV_PARTS=<каталог деталей> python dataset/fusion_opseq.py --zip <s2.0.1.zip> --split <uids> --workers 20

Узлы — истинные грани после правила узких граней (`label_map`), как в базовой линии (README,
«Тело последовательностью операций»). Грань Fusion i = исходная грань i нашего STEP (порядок
TopAbs_FACE; сверяется по мешу Fusion у каждого тела, несовпавшие тела отбрасываются).

Пишет `<деталь>/opseq.json`:
- nodes: тип, плоскость (нормаль наружу по мешу), операции, которым принадлежит узел, и роль;
- loops: петли узла — циклы соседей вдоль границы его участка на точном меше;
- ops: по операциям истории — тип, булева, боковые узлы; у выдавливания и вращения — варианты
  опоры (свой плоский торец или грань-хозяин) с петлями профиля, «до k» или глубиной и
  направлением; у скругления и фаски — полосы и скругляемые пары;
- deps: пары (раньше, позже) для свободного порядка — добавление и вырез, чьи узлы соседствуют
  (приближение пересечения объёмов: объёмов инструментов в Fusion Seg нет);
- skip: почему тело не годится (пусто — годится).
Целевые множества шага (что допустимо при данном состоянии), глубины сечения и углы вращения
считаются при загрузке из этих меток и скана.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import collections
import json
import sys
import time
import traceback
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset import remesh as R  # noqa: E402

FILLETS = ("FilletFeature", "ChamferFeature")
BOOLEAN = {"NewBodyFeatureOperation": "add", "JoinFeatureOperation": "add",
           "CutFeatureOperation": "cut", "IntersectFeatureOperation": "intersect"}
ROTATIONAL = ("cylinder", "cone", "sphere", "torus")
PARALLEL_COS = 0.995       # нормаль хозяина вдоль направления выдавливания
ORDER_AGREE = 0.95         # доля треугольников меша Fusion, попавших на ту же грань у нас
TOL_REL = 1e-3             # допуск по положению / диагональ детали
PAIR_SHARE = 0.2           # доля периметра полосы у каждой из двух скругляемых граней

_zip = None
NARROW = os.environ.get("OPSEQ_NO_NARROW") is None     # для замера: узлы без правила узких граней


def _z(path: str) -> zipfile.ZipFile:
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(path)
    return _zip


def _fusion_mesh(z: zipfile.ZipFile, uid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, tri = [], []
    for line in z.read(f"s2.0.1/meshes/{uid}.obj").decode().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            tri.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    fidx = np.array(z.read(f"s2.0.1/meshes/{uid}.fidx").split(), dtype=int)
    return np.array(v), np.array(tri), fidx


def _fill_absorbed(lab: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Треугольники граней, ушедших в ребро (label −1), — соседнему узлу, волной по смежности."""
    lab = lab.copy()
    for _ in range(64):
        miss = lab < 0
        if not miss.any():
            break
        a, b = adj[:, 0], adj[:, 1]
        take = miss[a] & ~miss[b]
        lab[a[take]] = lab[b[take]]
        take = miss[b] & ~miss[a]
        lab[b[take]] = lab[a[take]]
    return lab


def _edge_tri(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Направленное ребро -> треугольник, в котором оно так обходится."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    t = np.tile(np.arange(len(faces)), 3)
    return dict(zip(map(tuple, e.tolist()), t.tolist()))


def _loops(mesh: trimesh.Trimesh, lab: np.ndarray, node: int,
           edge_tri: dict[tuple[int, int], int]) -> tuple[list[list[int]], list[float]]:
    """Петли узла: циклы граничных рёбер его участка; петля — соседи по порядку обхода (без повторов подряд).
    Вторым — площадь, охваченная петлёй (модуль векторной площади), чтобы отличить наружную."""
    faces = mesh.faces
    mine = lab == node
    # ребро -> (треугольник участка, сосед по другую сторону)
    e_owner: dict[tuple[int, int], int] = {}
    for t in np.flatnonzero(mine):
        a, b, c = faces[t]
        for u, v in ((a, b), (b, c), (c, a)):
            e_owner[(int(u), int(v))] = t
    nxt: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for (u, v), t in e_owner.items():
        if (v, u) in e_owner:
            continue                     # внутреннее ребро участка
        nxt[u].append((v, t))
    used: set[tuple[int, int]] = set()
    loops, areas = [], []
    verts = mesh.vertices
    for start, outs in list(nxt.items()):
        for v0, _ in outs:
            if (start, v0) in used:
                continue
            seq, path, u, v = [], [], start, v0
            for _ in range(len(e_owner) + 1):
                used.add((u, v))
                path.append(u)
                other = edge_tri.get((v, u))
                seq.append(int(lab[other]) if other is not None else -1)
                cand = [w for w, _ in nxt.get(v, []) if (v, w) not in used]
                if not cand:
                    break
                u, v = v, cand[0]
            run = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
            if len(run) > 1 and run[0] == run[-1]:
                run = run[:-1]
            loops.append(run)
            p = verts[path]
            areas.append(float(np.linalg.norm(0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(0))))
    return loops, areas


def node_geometry(mesh: trimesh.Trimesh, lab: np.ndarray, nodes: list[int]) -> dict:
    """Площадь, нормаль и центр узлов; длины общих границ и соседи; петли узлов с охваченными площадями."""
    area = np.bincount(lab[lab >= 0], weights=mesh.area_faces[lab >= 0], minlength=max(nodes) + 1)
    nrm = np.zeros((max(nodes) + 1, 3))
    np.add.at(nrm, lab[lab >= 0], mesh.face_normals[lab >= 0] * mesh.area_faces[lab >= 0, None])
    cen = np.zeros((max(nodes) + 1, 3))
    np.add.at(cen, lab[lab >= 0], mesh.triangles_center[lab >= 0] * mesh.area_faces[lab >= 0, None])
    cen /= np.maximum(area, 1e-12)[:, None]

    # смежность узлов по рёбрам точного меша, с длиной общей границы
    fa = mesh.face_adjacency
    la, lb = lab[fa[:, 0]], lab[fa[:, 1]]
    el = np.linalg.norm(np.diff(mesh.vertices[mesh.face_adjacency_edges], axis=1)[:, 0], axis=1)
    border = collections.Counter()
    for a, b, length in zip(la, lb, el):
        if a != b and a >= 0 and b >= 0:
            border[(int(a), int(b))] += float(length)
            border[(int(b), int(a))] += float(length)
    nbrs = collections.defaultdict(set)
    for a, b in border:
        nbrs[a].add(b)
    et = _edge_tri(np.asarray(mesh.faces))
    loops, loop_area = {}, {}
    for n in nodes:
        loops[n], loop_area[n] = _loops(mesh, lab, n, et)
    return {"area": area, "normal": nrm, "center": cen, "border": border, "nbrs": nbrs,
            "loops": loops, "loop_area": loop_area}


def node_info_of(n: int, types: set, ops: list, roles: list, g: dict) -> dict:
    nn = g["normal"][n] / (np.linalg.norm(g["normal"][n]) + 1e-12)
    planar = types == {"plane"}
    return {"type": next(iter(types)) if len(types) == 1 else "mixed", "types": sorted(types),
            "planar": planar,
            "normal": nn.tolist() if planar else None,
            "offset": float(nn @ g["center"][n]) if planar else None,
            "area": float(g["area"][n]), "ops": ops, "roles": [list(r) for r in roles]}


def profile_loops(loops: list[list[int]], sides, fillet_nodes) -> list[int]:
    """Петли опоры, вдоль которых в основном идут боковые узлы операции (полосы скруглений не мешают)."""
    out = []
    for li, lp in enumerate(loops):
        ns = [x for x in lp if x >= 0]
        if ns and sum(x in sides or x in fillet_nodes for x in ns) >= 0.5 * len(ns) and any(x in sides for x in ns):
            out.append(li)
    return out


def label_body(uid: str, zip_path: str) -> dict:
    z = _z(zip_path)
    tl = json.loads(z.read(f"s2.0.1/timeline_info/{uid}.json"))
    feats, tfaces = tl["features"], tl["body"]["faces"]
    part = D.PARTS / uid
    brep = json.loads((part / "brep.json").read_text())
    mm = np.asarray(brep["merge_map"])
    lmap = np.asarray(brep.get("label_map") if NARROW and brep.get("label_map") else list(range(int(mm.max()) + 1)))
    mesh = trimesh.load(str(part / "mesh.ply"), process=False)
    fid = np.load(part / "face_id.npy")
    out = {"uid": uid, "skip": []}
    if len(mm) != len(tfaces):
        out["skip"].append(f"граней у нас {len(mm)}, в timeline {len(tfaces)}")
        return out
    v, tri, ffi = _fusion_mesh(z, uid)
    s = np.linalg.norm(mesh.extents) / max(np.linalg.norm(np.ptp(v, 0)), 1e-12)
    agree = float(np.mean(fid[R.closest_faces(mesh, v[tri].mean(1) * s)] == mm[ffi]))
    out["order_agree"] = agree
    if agree < ORDER_AGREE:
        out["skip"].append(f"порядок граней не совпал ({agree:.2f})")
        return out

    if mesh.volume < 0:
        mesh.invert()
    diag = float(np.linalg.norm(mesh.extents))
    tol = TOL_REL * diag
    lab = _fill_absorbed(lmap[fid], mesh.face_adjacency)
    nodes = sorted(int(x) for x in np.unique(lab) if x >= 0)
    if not nodes:
        out["skip"].append("все грани ушли в рёбра (правило узких граней)")
        return out
    # узел -> исходные грани, их операции и роли
    orig_of = collections.defaultdict(list)
    for i, m in enumerate(mm):
        if lmap[m] >= 0:
            orig_of[int(lmap[m])].append(i)
    g = node_geometry(mesh, lab, nodes)
    border, nbrs = g["border"], g["nbrs"]
    node_info = {}
    for n in nodes:
        types = {brep["faces"][i]["type"] for i in orig_of[n]}
        ops = sorted({tfaces[i]["feature"] for i in orig_of[n]})
        roles = sorted({(tfaces[i]["feature"], tfaces[i].get("location_in_feature", "Face")) for i in orig_of[n]})
        node_info[n] = node_info_of(n, types, ops, roles, g)
    if any(len(node_info[n]["ops"]) > 1 for n in nodes):
        out["mixed_nodes"] = sum(len(node_info[n]["ops"]) > 1 for n in nodes)

    def nodes_of(u, roles=None):
        return sorted({n for n in nodes for (f, r) in node_info[n]["roles"]
                       if f == u and (roles is None or r in roles)})

    fillet_nodes = {n for n in nodes if any(feats[f]["type"] in FILLETS for f in node_info[n]["ops"])}
    loops, loop_area = g["loops"], g["loop_area"]
    tri_dir = mesh.face_normals

    ops_out = {}
    for u, f in sorted(feats.items(), key=lambda kv: kv[1]["timeline_index"]):
        mine = nodes_of(u)
        if not mine:
            continue
        op = {"name": f["name"], "type": f["type"], "timeline_index": f["timeline_index"],
              "boolean": BOOLEAN.get(f.get("operation")), "nodes": mine}
        if f["type"] in FILLETS:
            strips = []
            for n in mine:
                per = sum(border[(n, m)] for m in nbrs[n])
                cand = sorted(((border[(n, m)], m) for m in nbrs[n] if m not in mine), reverse=True)
                pair = [m for length, m in cand[:2] if length >= PAIR_SHARE * per]
                strips.append({"node": n, "pair": pair if len(pair) == 2 else None})
            op["strips"] = strips
        elif f["type"] == "ExtrudeFeature":
            sides = nodes_of(u, ("SideFace",))
            ends = [n for n in nodes_of(u, ("StartFace", "EndFace")) if node_info[n]["planar"]]
            op["sides"] = sides
            sn = tri_dir[np.isin(lab, sides)]
            d = None
            if ends:
                d = np.asarray(node_info[ends[0]]["normal"])
            elif len(sn) >= 3:
                _, sv, vt = np.linalg.svd(sn, full_matrices=False)
                if sv[-1] < 0.05 * sv[0] and sv[-2] > 0.1 * sv[0]:
                    d = vt[-1]
            if d is None:
                op["supports"] = []
                op["why_no_support"] = "направление не определено (1-2 боковые грани?)"
            else:
                # протяжённость вдоль d — по боковым узлам только этой операции: слитый с чужими
                # гранями узел шире операции
                pure = [n for n in sides if node_info[n]["ops"] == [u]] or sides
                pts = mesh.vertices[np.unique(mesh.faces[np.isin(lab, pure)])] if pure else np.zeros((0, 3))
                lo, hi = (float((pts @ d).min()), float((pts @ d).max())) if len(pts) else (None, None)
                off = {}
                for m in set(ends) | {m for n in sides for m in nbrs[n]} | \
                        {k for n in sides for m in nbrs[n] if m in fillet_nodes for k in nbrs[m]}:
                    if node_info[m]["planar"] and (m in ends or (m not in mine and
                                                   abs(np.asarray(node_info[m]["normal"]) @ d) > PARALLEL_COS)):
                        off[m] = float(np.asarray(node_info[m]["normal"]) @ d) * node_info[m]["offset"]
                if lo is None and len(ends) == 2:
                    lo, hi = sorted(off[m] for m in ends)
                caps = []
                for m, o in sorted(off.items()):
                    if lo is not None and abs(o - lo) < tol:
                        caps.append((m, "lo"))
                    elif lo is not None and abs(o - hi) < tol:
                        caps.append((m, "hi"))
                    elif m in ends:
                        # свой торец — опора всегда; конец, если протяжённость не мерится
                        caps.append((m, None if lo is None else ("lo" if abs(o - lo) < abs(o - hi) else "hi")))
                add = op["boolean"] == "add"

                # плоскость на уровне торца, на которой профиль операции не лежит, — не опора и не «до»
                profs = {m: profile_loops(loops[m], sides, fillet_nodes) for m, _ in caps}
                for m in ends:
                    if m in profs and not profs[m] and loops[m]:
                        # боковые грани приписаны более ранней операции (слились с её гранями): профиль —
                        # наружная петля своего торца; внутренние могли дать поздние вырезы
                        profs[m] = [int(np.argmax(loop_area[m]))]
                        op["outer_loop_fallback"] = True
                caps = [(m, at) for m, at in caps if m in ends or profs[m]]
                sup = []
                for m, at in caps:
                    own = m in ends
                    far = None
                    if at is not None:
                        far_at = "hi" if at == "lo" else "lo"
                        far = [k for k, a2 in caps if a2 == far_at]
                        if not far:
                            # дальний конец на неплоской чужой грани у боковых («до поверхности» кривой)
                            ext = hi if far_at == "hi" else lo
                            for k in sorted({k for n in sides for k in nbrs[n]}):
                                if k in mine or k in fillet_nodes or node_info[k]["planar"]:
                                    continue
                                pk = mesh.vertices[np.unique(mesh.faces[lab == k])] @ d
                                if (pk.max() >= ext - tol) if far_at == "hi" else (pk.min() <= ext + tol):
                                    far.append(k)
                        far = far or None
                    # направление относительно наружной нормали опоры: свой торец добавления — материал
                    # за ним (−1), дно выреза — вырез перед ним (+1); на хозяине наоборот
                    sign = (-1 if add else 1) if own else (1 if add else -1)
                    prof = profs[m]
                    sup.append({"node": m, "own": own, "loops": prof, "to": far,
                                "depth_mm": (hi - lo) if lo is not None and not far else None, "dir": sign})
                op["supports"] = sup
                if not sup:
                    op["why_no_support"] = "нет плоского торца и хозяина на концах боковых граней"
        elif f["type"] == "RevolveFeature":
            sides = nodes_of(u, ("SideFace",))
            op["sides"] = sides
            rot = [n for n in sides if any(t in ROTATIONAL for t in node_info[n]["types"])]
            ends = [n for n in nodes_of(u, ("StartFace", "EndFace")) if node_info[n]["planar"]]
            op["supports"] = [{"node": n} for n in rot]
            op["ends"] = ends
            if not rot:
                op["why_no_support"] = "нет поверхности вращения"
        ops_out[u] = op

    # зависимости для свободного порядка: добавление и вырез, чьи узлы соседствуют
    solid = [u for u, o in ops_out.items() if o["type"] in ("ExtrudeFeature", "RevolveFeature")]
    deps = []
    for i, a in enumerate(solid):
        for b in solid[i + 1:]:
            ba, bb = ops_out[a]["boolean"], ops_out[b]["boolean"]
            if (ba == "add") == (bb == "add"):
                continue
            if any(m in nbrs[n] for n in ops_out[a]["nodes"] for m in ops_out[b]["nodes"]):
                deps.append([a, b])        # a раньше b в истории (solid идёт по timeline_index)
    bad = [o["name"] for o in ops_out.values() if "why_no_support" in o]
    if bad:
        out["skip"].append("нет опоры: " + ", ".join(bad))
    out.update({"diag_mm": diag, "nodes": {str(n): node_info[n] for n in nodes},
                "loops": {str(n): loops[n] for n in nodes},
                "loop_area": {str(n): loop_area[n] for n in nodes}, "ops": ops_out, "deps": deps})
    return out


def _one(args):
    uid, zip_path = args
    try:
        res = label_body(uid, zip_path)
        (D.PARTS / uid / ("opseq.json" if NARROW else "opseq.nonarrow.json")).write_text(json.dumps(res))
        return uid, res["skip"], None
    except Exception:
        return uid, None, traceback.format_exc(limit=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    uids = [u for u in Path(args.split).read_text().split() if (D.PARTS / u / "brep.json").exists()]
    if args.limit:
        uids = uids[:args.limit]
    t0, skips, errors = time.perf_counter(), collections.Counter(), 0
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for n, (uid, skip, err) in enumerate(pool.imap_unordered(_one, [(u, args.zip) for u in uids], chunksize=4), 1):
            if err:
                errors += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                for s in skip:
                    skips[s.split(":")[0].split("(")[0].strip()] += 1
                skips["годно" if not skip else "отсеяно"] += 1
            if n % 1000 == 0 or n == len(uids):
                print(f"{n}/{len(uids)}  ошибок {errors}  {dict(skips)}  "
                      f"{(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
