"""Эскизы операций выдавливания (шаг A сквозной цепочки: операции истинные, эскиз — 2D сеть).

    python scripts/sketch/op_sketches.py --run runs/section_lean14 --parts 300 --split val

На операцию: опора — плоская грань из меток (первая плоская с петлёй), плоскость сечения —
плоскость опоры, сдвинутая на половину выдавливания внутрь (глубина или расстояние до грани «до»;
у глубины сторона — та, где в сечении больше отрезков боковых граней), подальше от скруглений на
краях. Сечение скана целиком -> 2D сеть -> стыки -> кривые (`build_pieces_checked`). Профиль —
отрезки, чьи треугольники лежат на боковых гранях (петля опоры); точный профиль — рёбра точного
сечения STEP той же плоскостью на тех же гранях. Метрики по профилю: стыки (острые — по
расстоянию), тип отрезка, профиль верен целиком (все стыки найдены, лишних нет), отклонение кривых
от точного профиля (и по размеченным стыкам — оракул). Картинки — `outputs/op_sketches/*.png`.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sketch"))

from evaluate_section import match  # noqa: E402

from dataset import cache as D  # noqa: E402
from dataset.step_mesh import read_step  # noqa: E402
from src.opseq.data import split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import build_pieces, build_pieces_checked, peaks, sample_curve  # noqa: E402
from src.sketch.data import collate, segment_features  # noqa: E402
from src.sketch.model import SectionNet  # noqa: E402
from src.sketch.section import (CURVE_TYPES, exact_section, exact_vertices, face_index, label_chain,  # noqa: E402
                                merge_short, mesh_section, plane_frame)

DATA = Path(SCFG.data.root)
OUT = ROOT / "outputs" / "op_sketches"
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")


def op_planes(uid: str) -> list[dict]:
    """Сечения операций детали: всё для сети и оценки, без сети (в процессах пула)."""
    lab = json.loads((DATA / "parts" / uid / "opseq.nonarrow.json").read_text())
    if lab.get("skip"):
        return []
    brep = json.loads((DATA / "parts" / uid / "brep.json").read_text())
    faces = brep["faces"]
    merge = np.asarray(brep["merge_map"]) if brep["merge_map"] is not None else None
    diag = float(brep["bbox_diag"])
    z = np.load(DATA / "scan" / f"{uid}.npz")
    v, f, tri_face = z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["labels"]
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    cfg = SCFG.data
    shape, index = None, None
    out = []
    for name, op in lab["ops"].items():
        if op["type"] != "ExtrudeFeature":
            continue
        sup = next((s for s in op.get("supports", []) if faces[s["node"]]["type"] == "plane" and s.get("loops")), None)
        if sup is None:
            continue
        p = faces[sup["node"]]["params"]
        n, o = np.asarray(p["normal"], float), np.asarray(p["origin"], float)
        side = sorted({x for li in sup["loops"] for x in lab["loops"][str(sup["node"])][li]})
        if sup.get("to"):
            q = faces[sup["to"][0]]
            if q["type"] != "plane":
                continue
            extent = [float((np.asarray(q["params"]["origin"], float) - o) @ n)]
        elif sup.get("depth_mm"):
            extent = [sup["depth_mm"], -sup["depth_mm"]]
        else:
            continue
        frame = plane_frame(n)
        best = None
        for e in extent:
            origin = o + n * e / 2
            chains = [merge_short(c, cfg.min_seg_rel * diag) for c in mesh_section(v, f, nrm, origin, frame)]
            chains = [c for c in chains if c.n >= cfg.min_chain]
            prof = sum(int(np.isin(tri_face[c.tri], side).sum()) for c in chains)
            if best is None or prof > best[0]:
                best = (prof, origin, chains)
        prof, origin, chains = best
        if prof < cfg.min_chain:
            continue
        if shape is None:
            shape = read_step(D.step_path(uid))
            index = face_index(shape)
        tol = cfg.label_rel * diag
        edges = exact_section(shape, origin, frame, merge, index, tol / 5)
        verts = exact_vertices(edges, cfg.vertex_rel * diag, cfg.tangent_deg, cfg.seam_curvature / diag,
                               cfg.seam_step_rel * diag, cfg.seam_angle_deg, cfg.spline_join_deg)
        if not edges or any(x.degree != 2 for x in verts):
            continue
        side_m = {int(merge[x]) if merge is not None else x for x in side}
        exact_prof = [e.samples for e in edges if e.face in side_m]
        if not exact_prof:
            continue
        rec = {"uid": uid, "op": op["name"], "diag": diag, "origin": origin, "frame": frame,
               "exact": np.concatenate(exact_prof), "exact_all": [e.samples for e in edges], "chains": []}
        for c in chains:
            st, bd, tg = label_chain(c, edges, verts, tol, cfg.unresolved_gap, cfg.unresolved_turn_deg)
            rec["chains"].append({"points": c.points, "normals": c.normals, "closed": c.closed, "type": st,
                                  "boundary": bd, "tangent": tg, "profile": np.isin(tri_face[c.tri], side)})
        out.append(rec)
    return out


def evaluate(rec: dict, model, device, hits: Counter, dev: dict, draw: Path | None) -> None:
    diag = rec["diag"]
    item = {"uid": rec["uid"], "diag": diag, "chains": []}
    for c in rec["chains"]:
        item["chains"].append({"feat": segment_features(c["points"], c["normals"], c["closed"], diag),
                               "type": c["type"].astype(np.int64), "boundary": c["boundary"].astype(np.int64),
                               "tangent": c["tangent"].astype(np.int64), "closed": c["closed"],
                               "points": c["points"], "sec": 0})
    batch = collate([item])
    with torch.no_grad():
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        tl, bnd, tan = model(b)
    tprob = torch.softmax(tl, 1).cpu().numpy()
    pb, pt = torch.sigmoid(bnd).cpu().numpy(), torch.sigmoid(tan).cpu().numpy()
    jc, jp = batch["junction_chain"].numpy(), batch["junction_point"].numpy()
    tree = cKDTree(rec["exact"])
    cfg = SCFG.train
    profile_ok = True
    drawn = []
    for ci, c in enumerate(rec["chains"]):
        prof = c["profile"]
        if not prof.any():
            continue
        js = np.flatnonzero(jc == ci)
        s0, n = int(batch["start"][ci]), len(prof)
        closed = c["closed"]
        # стык в профиле — оба соседних отрезка на боковых гранях
        jpt = jp[js]
        inside = prof[(jpt - 1) % n] & prof[jpt % n]
        pk = peaks(pb[js], closed, cfg.peak_threshold, cfg.peak_window)
        pk_in = np.array([q for q in pk if inside[q]], int)
        bd, tg = c["boundary"][jpt], c["tangent"][jpt]
        gold = {int(g) for g in np.flatnonzero((bd == 1) & inside)}
        xy = c["points"][jpt]
        a, fp, fn, pairs = match(pk_in, gold, len(js), closed, xy, {g for g in gold if tg[g] == 0},
                                 np.flatnonzero(bd == -2), cfg.match_rel * diag)
        hits["стыков"] += len(gold)
        hits["найдено"] += a
        hits["лишних"] += fp
        hits["пиков"] += len(pk_in)
        profile_ok &= fp == 0 and fn == 0
        lab = c["type"][prof]
        pred = tprob[s0:s0 + n].argmax(1)[prof]
        hits["отрезков профиля с меткой"] += int((lab >= 0).sum())
        hits["тип верен"] += int((pred[lab >= 0] == lab[lab >= 0]).sum())
        # кривые: сеть и оракул (размеченные стыки, типы по меткам)
        pieces, _, _, _ = build_pieces_checked(c["points"], closed, jpt[pk], pt[js[pk]] <= 0.5,
                                               tprob[s0:s0 + n], cfg.split_rel * diag)
        g_all = np.flatnonzero(bd == 1)
        one_hot = np.full((n, 4), 0.25)
        one_hot[c["type"] >= 0] = 0
        one_hot[c["type"] >= 0, c["type"][c["type"] >= 0]] = 1
        oracle, _ = build_pieces(c["points"], closed, jpt[g_all], tg[g_all] == 0, one_hot)
        for name, ps in (("сеть", pieces), ("оракул", oracle)):
            for pc in ps:
                seg = (np.r_[np.arange(pc.start, n), np.arange(0, pc.end)] if closed and pc.end <= pc.start
                       else np.arange(pc.start, pc.end))
                if not len(seg) or prof[seg].mean() < 0.5:
                    continue
                s = sample_curve(pc.curve, 64)
                if s is not None:
                    dev[name].append(float(tree.query(s)[0].max() / diag * 1000))
                if name == "сеть":
                    drawn.append(pc)
    hits["операций"] += 1
    hits["профиль верен"] += profile_ok
    if draw is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        for s in rec["exact_all"]:
            ax.plot(s[:, 0], s[:, 1], "-", color="0.85", lw=4, zorder=1)
        ax.plot(rec["exact"][:, 0], rec["exact"][:, 1], ".", color="k", ms=1, zorder=2)
        for c in rec["chains"]:
            p = c["points"]
            q = p[c["profile"]] if len(p) == len(c["profile"]) else p[:len(c["profile"])][c["profile"]]
            ax.scatter(q[:, 0], q[:, 1], s=4, c="0.4", zorder=2)
        for pc in drawn:
            s = sample_curve(pc.curve, 64)
            if s is not None:
                ax.plot(s[:, 0], s[:, 1], "-", color=COLORS[pc.kind], lw=1.5, zorder=3)
        ax.set_aspect("equal")
        ax.set_title(f"{rec['uid']} {rec['op']}: профиль {'верен' if profile_ok else 'с ошибками'}")
        fig.tight_layout()
        fig.savefig(draw, dpi=80)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--parts", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--draw", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    uids = sorted(p.name for p in (DATA / "parts").iterdir() if (DATA / "scan" / f"{p.name}.npz").exists())
    uids = split_uids(uids)[args.split][:args.parts]
    device = torch.device("cuda")
    state = torch.load(ROOT / args.run / "best.pt", map_location=device)["model"]
    model = SectionNet().to(device)
    model.load_state_dict(state)
    model.eval()
    hits, dev = Counter(), defaultdict(list)
    drawn = 0
    with Pool(args.workers) as pool:
        for k, recs in enumerate(pool.imap_unordered(op_planes, uids, chunksize=2), 1):
            for rec in recs:
                path = OUT / f"{rec['uid']}_{rec['op']}.png" if drawn < args.draw else None
                drawn += path is not None
                evaluate(rec, model, device, hits, dev, path)
            if k % 50 == 0:
                print(f"{k}/{len(uids)}", file=sys.stderr, flush=True)
    out = {"деталей": len(uids), "операций": hits["операций"],
           "профиль верен целиком": hits["профиль верен"] / max(hits["операций"], 1),
           "стыки профиля: полнота": hits["найдено"] / max(hits["стыков"], 1),
           "стыки профиля: точность": (hits["пиков"] - hits["лишних"]) / max(hits["пиков"], 1),
           "тип отрезка профиля": hits["тип верен"] / max(hits["отрезков профиля с меткой"], 1)}
    for k, v in dev.items():
        v = np.array(v)
        out[f"отклонение кривых профиля, ‰ D ({k})"] = {"медиана": round(float(np.median(v)), 2),
                                                       "p90": round(float(np.percentile(v, 90)), 2),
                                                       "доля > 5‰": round(float((v > 5).mean()), 3), "кривых": len(v)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
