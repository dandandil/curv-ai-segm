"""Метки шагов opseq по деталям реконструкции Fusion и генератора: инструменты + точный меш -> opseq.json.

    python dataset/recon_opseq.py --workers 12 [--synth]

Формат — как у `dataset/fusion_opseq.py`, но всё берётся из инструментов, а не угадывается по мешу (README,
«Данные v2»): роли граней — `fusion_recon.face_roles`, направление — нормаль эскиза, торцы — плоские грани в
плоскостях h0 и h1 (`recon_caps.cap_faces`). Любое выдавливание пишется односторонним: опора — грань в
плоскости одного торца (своя или хозяин), «до» — грани в плоскости другого, иначе глубина. Выдавливание без
граней в обоих торцах — `skip`. Зависимости порядка — по пересечению объёмов инструментов. Скругления и
фаски генератора (`synth_ops`) — полосы с парами скругляемых граней, как в `fusion_opseq`.
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

from dataset.fusion_opseq import PAIR_SHARE, _fill_absorbed, node_geometry, node_info_of, profile_loops  # noqa: E402
from dataset.fusion_recon import face_roles, load, tools  # noqa: E402
from dataset.step_mesh import bbox_diag, read_step  # noqa: E402
from scripts.opseq.recon_caps import DATA, R, ZIP, cap_faces  # noqa: E402

SYNTH = DATA / "synth"

ROLES = {"side": "SideFace", "start": "StartFace", "end": "EndFace", "strip": "Face"}
FILLETS = ("FilletFeature", "ChamferFeature")
CAP_TOL_REL = 5e-4         # как в recon_caps
OVERLAP_REL = 1e-6         # объём пересечения инструментов / D³, с которого порядок важен

NARROW = os.environ.get("OPSEQ_NO_NARROW") is None


def _overlap(a, b, diag: float) -> bool:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(BRepAlgoAPI_Common(a, b).Shape(), props)
    return props.Mass() > OVERLAP_REL * diag ** 3


def label_body(uid: str, scale: float) -> dict:
    ts = tools(load(zipfile.ZipFile(ZIP), uid), scale)
    final = read_step(R / "step" / f"{uid}.stp")
    ops = [(t.name, "ExtrudeFeature", t.boolean, t) for t in ts]
    return label_part(uid, history(final, ops, face_roles(final, ts, bbox_diag(final))), R / "parts" / uid)


def history(final, ops: list[tuple], face_ops: list) -> dict:
    """Всё, для чего нужны тела инструментов, — чтобы метки считались без них, после сборки детали.
    ops — [(имя, тип, булева, Tool | None)] по истории; face_ops — исходная грань -> [(номер операции, роль)]."""
    diag = bbox_diag(final)
    ext = [k for k, o in enumerate(ops) if o[3] is not None]
    ends = cap_faces(final, [ops[k][3] for k in ext], CAP_TOL_REL * diag)
    tools_out = {str(k): {"z": ops[k][3].frame[3].tolist(), "h0": float(ops[k][3].h0), "h1": float(ops[k][3].h1),
                          "ends": e} for k, e in zip(ext, ends)}
    # порядок важен только у добавления и выреза с пересекающимися инструментами; пересечение — со всеми
    # (скругления и фаски упорядочивает загрузчик, по соседству)
    order = []
    for i, a in enumerate(ext):
        for b in ext[i + 1:]:
            ba, bb = ops[a][2], ops[b][2]
            if ba == bb and ba != "intersect":
                continue
            if "intersect" in (ba, bb) or _overlap(ops[a][3].shape, ops[b][3].shape, diag):
                order.append([a, b])
    return {"ops": [list(o[:3]) for o in ops], "face_ops": [[list(x) for x in f] for f in face_ops],
            "tools": tools_out, "order": order}


def label_part(uid: str, hist: dict, part: Path) -> dict:
    ops, face_ops, tools_h = hist["ops"], hist["face_ops"], hist["tools"]
    brep = json.loads((part / "brep.json").read_text())
    mm = np.asarray(brep["merge_map"])
    lmap = np.asarray(brep.get("label_map") if NARROW and brep.get("label_map") else list(range(int(mm.max()) + 1)))
    out = {"uid": uid, "skip": []}
    if len(mm) != len(face_ops):
        out["skip"].append(f"граней у меша {len(mm)}, у STEP {len(face_ops)}")
        return out
    names = [o[0] for o in ops]
    keys = names if len(set(names)) == len(names) else [f"{k}:{n}" for k, n in enumerate(names)]

    mesh = trimesh.load(str(part / "mesh.ply"), process=False)
    if mesh.volume < 0:
        mesh.invert()
    lab = _fill_absorbed(lmap[np.load(part / "face_id.npy")], mesh.face_adjacency)
    nodes = sorted(int(x) for x in np.unique(lab) if x >= 0)
    if not nodes:
        out["skip"].append("все грани ушли в рёбра (правило узких граней)")
        return out
    node_of = {i: int(lmap[m]) for i, m in enumerate(mm) if lmap[m] >= 0}
    orig_of = collections.defaultdict(list)
    for i, n in node_of.items():
        orig_of[n].append(i)
    g = node_geometry(mesh, lab, nodes)
    loops, loop_area = g["loops"], g["loop_area"]

    node_info = {}
    for n in nodes:
        types = {brep["faces"][i]["type"] for i in orig_of[n]}
        roles = sorted({(keys[k], ROLES[r]) for i in orig_of[n] for k, r in face_ops[i]})
        node_info[n] = node_info_of(n, types, sorted({u for u, _ in roles}), roles, g)
    blank = [n for n in nodes if not node_info[n]["ops"]]
    if blank:
        out["skip"].append(f"неразмеченные грани ({len(blank)})")
    if any(len(node_info[n]["ops"]) > 1 for n in nodes):
        out["mixed_nodes"] = sum(len(node_info[n]["ops"]) > 1 for n in nodes)

    def nodes_of(u, roles=None):
        return sorted({n for n in nodes for (f, r) in node_info[n]["roles"]
                       if f == u and (roles is None or r in roles)})

    def cap_nodes(end):
        own = {node_of[i] for i in end["own"] if i in node_of}
        host = {node_of[i] for i in end["host"] if i in node_of} - own
        return [(m, m in own) for m in sorted(own | host) if m in node_info and node_info[m]["planar"]]

    border, nbrs = g["border"], g["nbrs"]
    ops_out = {}
    for k, ((name, kind, boolean), u) in enumerate(zip(ops, keys)):
        mine = nodes_of(u)
        if not mine:
            continue
        if kind in FILLETS:
            strips = []
            for n in mine:
                per = sum(border[(n, m)] for m in nbrs[n])
                cand = sorted(((border[(n, m)], m) for m in nbrs[n] if m not in mine), reverse=True)
                pair = [m for length, m in cand[:2] if length >= PAIR_SHARE * per]
                strips.append({"node": n, "pair": pair if len(pair) == 2 else None})
            ops_out[u] = {"name": name, "type": kind, "timeline_index": k, "boolean": None, "nodes": mine,
                          "strips": strips}
            continue
        sides = nodes_of(u, ("SideFace",))
        op = {"name": name, "type": kind, "timeline_index": k, "boolean": boolean, "nodes": mine, "sides": sides}
        t = tools_h[str(k)]
        at = [cap_nodes(e) for e in t["ends"]]
        sup = []
        for e in (0, 1):
            toward = np.asarray(t["z"]) * (1 if e == 0 else -1)          # от торца внутрь инструмента
            far = [m for m, _ in at[1 - e]]
            for m, own in at[e]:
                prof = profile_loops(loops[m], sides, set())
                if own and not prof and loops[m]:
                    prof = [int(np.argmax(loop_area[m]))]
                    op["outer_loop_fallback"] = True
                to = [x for x in far if x != m] or None
                # dir: +1 — инструмент по наружной нормали опоры (как в fusion_opseq)
                sup.append({"node": m, "own": own, "loops": prof, "to": to,
                            "depth_mm": None if to else t["h1"] - t["h0"],
                            "dir": 1 if np.asarray(node_info[m]["normal"]) @ toward > 0 else -1, "end": e})
        op["supports"] = sup
        if not sup:
            op["why_no_support"] = "нет граней в плоскостях обоих торцов"
        ops_out[u] = op

    deps = [[keys[a], keys[b]] for a, b in hist["order"] if keys[a] in ops_out and keys[b] in ops_out]
    bad = [o["name"] for o in ops_out.values() if "why_no_support" in o]
    if bad:
        out["skip"].append("нет опоры: " + ", ".join(bad))
    out.update({"diag_mm": float(np.linalg.norm(mesh.extents)), "nodes": {str(n): node_info[n] for n in nodes},
                "loops": {str(n): loops[n] for n in nodes},
                "loop_area": {str(n): loop_area[n] for n in nodes}, "ops": ops_out, "deps": deps})
    return out


def _one(args):
    uid, scale, synth = args
    try:
        if synth:
            part = SYNTH / "parts" / uid
            res = label_part(uid, json.loads((SYNTH / "hist" / f"{uid}.json").read_text()), part)
        else:
            part = R / "parts" / uid
            res = label_body(uid, scale)
        (part / ("opseq.json" if NARROW else "opseq.nonarrow.json")).write_text(json.dumps(res))
        return uid, res["skip"], None
    except Exception:
        return uid, None, traceback.format_exc(limit=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--synth", action="store_true", help="детали генератора: synth/hist + synth/parts")
    args = ap.parse_args()
    if args.synth:
        uids = sorted(p.stem for p in (SYNTH / "hist").glob("*.json") if (SYNTH / "parts" / p.stem / "brep.json").exists())
        jobs = [(u, None, True) for u in uids]
    else:
        scale = json.loads((R / "scale.json").read_text())
        uids = [u for u in (R / "uids.txt").read_text().split() if (R / "parts" / u / "brep.json").exists()]
        jobs = [(u, scale[u], False) for u in uids]
    if args.limit:
        jobs = jobs[:args.limit]
    t0, skips, errors = time.perf_counter(), collections.Counter(), 0
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for n, (uid, skip, err) in enumerate(pool.imap_unordered(_one, jobs, chunksize=4), 1):
            if err:
                errors += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                for s in skip:
                    skips[s.split(":")[0].split("(")[0].strip()] += 1
                skips["годно" if not skip else "отсеяно"] += 1
            if n % 500 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}  ошибок {errors}  {dict(skips)}  "
                      f"{(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
