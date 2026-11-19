"""Сечения деталей для 2D сети: цепочки отрезков скана и метки по точному сечению STEP.

    python scripts/sketch/build_sections.py --split <root>/uids_all.txt --workers 20

На деталь `planes_per_part` плоскостей (`configs/sketch.yaml`, `src/sketch/section.py`,
`sample_planes`); пишет `<root>/sections/<uid>.npz`. Сечение отбрасывается, если точное
вырождено (вершина не с двумя рёбрами — плоскость через вершину или ребро B-Rep) или меньше
половины отрезков скана легли на точное сечение. Готовые детали пропускаются.
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
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset.step_mesh import read_step  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.section import (exact_section, exact_vertices, face_index, label_chain,  # noqa: E402
                                merge_short, mesh_section, sample_planes)

DATA = Path(SCFG.data.root)
# Доля отрезков скана с меткой типа, ниже которой сечение отбрасывается (скан и точное
# разошлись: полость, которую сканер не видит, вся петля). Выбрано.
MIN_LABELED = 0.5


def build(uid: str, cfg=SCFG.data) -> tuple[dict | None, collections.Counter]:
    stats = collections.Counter()
    z = np.load(DATA / "scan" / f"{uid}.npz")
    v, f = z["vertices"].astype(np.float64), z["faces"].astype(np.int64)
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    brep = json.loads((DATA / "parts" / uid / "brep.json").read_text())
    merge = np.asarray(brep["merge_map"]) if brep["merge_map"] is not None else None
    diag = float(brep["bbox_diag"])
    shape = read_step(D.step_path(uid))
    index = face_index(shape)
    rng = np.random.default_rng(zlib.crc32(uid.encode()))
    planes = sample_planes(rng, v, brep["faces"], cfg.planes_per_part, cfg.oblique_frac,
                           cfg.depth_margin, cfg.coplanar_rel * diag)
    tol = cfg.label_rel * diag
    acc = collections.defaultdict(list)
    n_sec = 0
    for origin, frame, oblique in planes:
        edges = exact_section(shape, origin, frame, merge, index, tol / 5)
        verts = exact_vertices(edges, cfg.vertex_rel * diag, cfg.tangent_deg,
                               cfg.seam_curvature / diag, cfg.seam_step_rel * diag, cfg.seam_angle_deg,
                               cfg.spline_join_deg)
        if not edges or any(x.degree != 2 for x in verts):
            stats["вырождено"] += 1
            continue
        chains = [merge_short(c, cfg.min_seg_rel * diag) for c in mesh_section(v, f, nrm, origin, frame)]
        chains = [c for c in chains if c.n >= cfg.min_chain]
        if not chains:
            stats["пусто на скане"] += 1
            continue
        labels = [label_chain(c, edges, verts, tol, cfg.unresolved_gap, cfg.unresolved_turn_deg) for c in chains]
        seg_type = np.concatenate([t for t, _, _ in labels])
        if (seg_type >= 0).mean() < MIN_LABELED:
            stats["не совпало"] += 1
            continue
        stats["сечений"] += 1
        stats["наклонных"] += int(oblique)
        for c, (t, b, g) in zip(chains, labels):
            acc["chain_sec"].append(n_sec)
            acc["chain_closed"].append(c.closed)
            acc["chain_points"].append(len(c.points))
            acc["chain_segments"].append(c.n)
            acc["points"].append(c.points.astype(np.float32))
            acc["normals"].append(c.normals.astype(np.float16))
            acc["tri"].append(c.tri.astype(np.int32))
            acc["seg_type"].append(t)
            acc["boundary"].append(b)
            acc["tangent"].append(g)
        for e in edges:
            acc["edge_sec"].append(n_sec)
            acc["edge_kind"].append(e.kind)
            acc["edge_face"].append(e.face)
            acc["edge_samples"].append(len(e.samples))
            acc["samples"].append(e.samples.astype(np.float32))
        for x in verts:
            acc["vert_sec"].append(n_sec)
            acc["vert_point"].append(x.point.astype(np.float32))
            acc["vert_boundary"].append(x.boundary)
            acc["vert_tangent"].append(x.tangent)
        acc["sec_origin"].append(origin)
        acc["sec_frame"].append(frame)
        acc["sec_oblique"].append(oblique)
        n_sec += 1
    if not n_sec:
        return None, stats
    out = {"diag": np.float64(diag)}
    for k, val in acc.items():
        if k in ("points", "normals", "tri", "seg_type", "boundary", "tangent", "samples"):
            out[k] = np.concatenate(val)
        else:
            out[k] = np.asarray(val)
    return out, stats


def _one(uid: str) -> tuple[str, collections.Counter, str | None]:
    try:
        if not (DATA / "scan" / f"{uid}.npz").exists():
            return uid, collections.Counter({"нет скана": 1}), None
        g, stats = build(uid)
        if g is None:
            stats["деталь без сечений"] += 1
            return uid, stats, None
        tmp = DATA / "sections" / f"{uid}.tmp.npz"
        np.savez_compressed(tmp, **g)
        tmp.rename(DATA / "sections" / f"{uid}.npz")
        stats["деталей"] += 1
        return uid, stats, None
    except Exception:
        return uid, collections.Counter(), traceback.format_exc(limit=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="остановиться, когда на диске меньше")
    args = ap.parse_args()
    (DATA / "sections").mkdir(exist_ok=True)
    uids = [u for u in Path(args.split).read_text().split() if not (DATA / "sections" / f"{u}.npz").exists()]
    if args.limit:
        uids = uids[:args.limit]
    t0, counts = time.perf_counter(), collections.Counter()
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for n, (uid, stats, err) in enumerate(pool.imap_unordered(_one, uids, chunksize=4), 1):
            if err:
                counts["ошибка"] += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            counts.update(stats)
            if n % 1000 == 0 or n == len(uids):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(uids)}  {dict(counts)}  осталось {(len(uids) - n) * dt / n / 60:.0f} мин", flush=True)
            if n % 100 == 0:
                st = os.statvfs(DATA)
                if st.f_bavail * st.f_frsize / 2 ** 30 < args.min_free_gb:
                    print(f"=== место кончилось: меньше {args.min_free_gb:g} ГБ", flush=True)
                    pool.terminate()
                    break
    print(f"итого: {dict(counts)}, {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
