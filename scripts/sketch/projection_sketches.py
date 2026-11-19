"""Эскизы выдавливаний проекцией стенок на опору против точного профиля STEP (README, «Грани операции и
эскиз проекцией»).

    python scripts/sketch/projection_sketches.py --parts 300 --split val

Операции и опоры — как в `op_sketches.py` (первая плоская опора с петлёй и протяжённостью). Стенки —
истинные (`sides` меток), направление проекции — нормаль опоры по подгонке скана (граф opseq), система
плоскости — точная нормаль. Точный профиль — рёбра точных сечений на стенках на `DEPTHS` долях
протяжённости (стенку, срезанную поздней операцией, видно хоть на одной). Метрики: стенка воспроизведена
— есть кривая её типа и точные точки стенки не дальше `TOL_REL` от неё; профиль верен — воспроизведены
все стенки, у которых есть точное ребро; отклонение — от точных точек до кривой своей стенки и от
кривой до точного профиля, ‰ D. Картинки — `outputs/projection_sketches/*.png`.
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
import trimesh
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset.step_mesh import read_step  # noqa: E402
from src.opseq.config import OCFG  # noqa: E402
from src.opseq.data import CORPORA, _exact_geometry, load_graph, split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import sample_curve  # noqa: E402
from src.sketch.projection import (CELL_REL, JOIN_REL, JOINTS, MERGE_REL, extrusion_direction,  # noqa: E402
                                   sketch_from_walls, walls_of)
from src.sketch.section import CURVE_TYPES, exact_section, face_index, plane_frame  # noqa: E402

# корпус — OPSEQ_CORPUS, по умолчанию прежний сканер (на нём сделаны замеры в README)
CORPUS = CORPORA[os.environ.get("OPSEQ_CORPUS", "seg_old")]
DATA = CORPUS.root
OUT = ROOT / "outputs" / "projection_sketches"
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
JOINT_COLORS = {"касание": "tab:green", "острый": "k", "разрыв": "tab:red", "свободный": "tab:pink", "замкнутая": "0.5"}
# Доли протяжённости для точных сечений. Выбрано.
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
# Наклонная стенка: по точной геометрии отклонение от направления выдавливания больше 10° (уклон или
# стенка не этой опоры). Выбрано.
TILT_COS = float(np.sin(np.radians(10)))
# Допуск «стенка воспроизведена» / диагональ (как «доля > 5‰» в op_sketches). Выбрано.
TOL_REL = 0.005


def op_records(job: tuple[str, bool]) -> list[dict]:
    """Записи операций детали; `exact_mesh` — вместо скана точный меш этапа 1 (`mesh.ply`, метки —
    слитые грани по `face_id.npy`): что теряет сам построитель, без шума сканера."""
    uid, exact_mesh = job
    lab = json.loads((DATA / "parts" / uid / OCFG.data.labels).read_text())
    if lab.get("skip"):
        return []
    g = load_graph(CORPUS.graph_path(uid))
    ids = [int(x) for x in g["ids"]]
    pos = {m: i for i, m in enumerate(ids)}
    brep = json.loads((DATA / "parts" / uid / "brep.json").read_text())
    merge = np.asarray(brep["merge_map"]) if brep["merge_map"] is not None else None
    exact = _exact_geometry(brep, g["ids"])
    diag = float(brep["bbox_diag"])
    if exact_mesh:
        mesh = trimesh.load(str(DATA / "parts" / uid / "mesh.ply"), process=False)
        v, f = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
        fid = np.load(DATA / "parts" / uid / "face_id.npy")
        tri_face = merge[fid] if merge is not None else fid
    else:
        z = np.load(CORPUS.scan_path(uid))
        v, f, tri_face = z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["labels"]
    types = {int(k): x["type"] for k, x in lab["nodes"].items()}
    tangent = {tuple(sorted((ids[a], ids[b]))) for (a, b), t in zip(g["edges"], g["tangent"]) if t}
    adjacent = {tuple(sorted((ids[a], ids[b]))) for a, b in g["edges"]}
    shape, index, out, hits = None, None, [], Counter()
    for op in lab["ops"].values():
        if op["type"] != "ExtrudeFeature":
            continue
        sup = next((s for s in op.get("supports", []) if s["node"] in pos and g["plane"][pos[s["node"]]]
                    and types[s["node"]] == "plane" and s.get("loops")), None)
        if sup is None:
            continue
        s = pos[sup["node"]]
        n = exact["dir"][s]
        fit = g["dir"][s].astype(np.float64)
        n = n if n @ fit >= 0 else -n
        o = exact["point"][s]
        if sup.get("to"):
            k = pos.get(sup["to"][0])
            if k is None or not exact["plane"][k]:
                continue
            extent = float((exact["point"][k] - o) @ n)
        elif sup.get("depth_mm"):
            extent = sup["depth_mm"] * float(sup["dir"])
        else:
            continue
        sides = [m for m in op["sides"] if m in pos]
        if not sides:
            continue
        frame = plane_frame(n)
        if shape is None:
            shape = read_step(D.step_path(uid))
            index = face_index(shape)
        gold = defaultdict(list)
        for t in DEPTHS:
            for e in exact_section(shape, o + n * extent * t, frame, merge, index, SCFG.data.label_rel * diag / 5):
                if e.face in sides:
                    gold[e.face].append((e.kind, e.samples))
        direction = extrusion_direction(v, f, tri_face, sides, fit)
        hits["угол направления к точному, °"] += float(np.degrees(np.arccos(min(abs(direction @ n), 1))))
        hits["угол нормали опоры к точному, °"] += float(np.degrees(np.arccos(min(abs(fit @ n), 1))))
        walls = walls_of(v, f, tri_face, sides, types, o, direction, frame, CELL_REL * diag)
        # наклон стенки по точной геометрии: |cos| нормали плоскости или sin угла оси с нормалью опоры
        tilt = {}
        for m in sides:
            i = pos[m]
            if exact["plane"][i]:
                tilt[m] = abs(float(exact["dir"][i] @ n))
            elif exact["axial"][i]:
                tilt[m] = float(np.sqrt(max(1 - float(exact["dir"][i] @ n) ** 2, 0)))
        loops = sketch_from_walls(walls, tangent, adjacent, JOIN_REL * diag, CELL_REL * diag, MERGE_REL * diag)
        out.append({"uid": uid, "op": op["name"], "diag": diag, "gold": dict(gold), "loops": loops,
                    "walls": walls, "sides": sides, "tilt": tilt, "hits": dict(hits)})
        hits = Counter()
    return out


def evaluate(rec: dict, hits: Counter, dev: dict, draw: Path | None, focus: str | None = None) -> bool:
    """Метрики в `hits` и `dev`, картинка в `draw`; возвращает, был ли плохой стык вида `focus`."""
    diag, gold = rec["diag"], rec["gold"]
    curves = {c.node: c for loop in rec["loops"] for c in loop}
    every = np.concatenate([s for parts in gold.values() for _, s in parts]) if gold else np.zeros((0, 2))
    tree = cKDTree(every) if len(every) else None
    ok, bad_focus = bool(gold), False
    hits.update(rec["hits"])
    for m, parts in gold.items():
        kind = Counter(k for k, _ in parts).most_common(1)[0][0]
        hits["стенок"] += 1
        c = curves.get(m)
        s = sample_curve(c.curve, 256) if c is not None else None
        good = False
        if s is not None:
            d = cKDTree(s).query(np.concatenate([p for _, p in parts]))[0].max() / diag
            dev["точное -> кривая"].append(d * 1000)
            # по участкам кривой: середина (20-80% длины) — форма и сдвиг скана, концы — стыки по видам
            pts = np.concatenate([p for _, p in parts])
            dist, at = cKDTree(s).query(pts)
            u = at / (len(s) - 1)
            for name, sel, joint in (("середина", (u > 0.2) & (u < 0.8), None), ("начало", u <= 0.2, c.ends[0]),
                                     ("конец", u >= 0.8, c.ends[1])):
                if sel.any():
                    x = dist[sel].max() / diag
                    key = "середина" if joint is None else joint
                    hits[f"участок {key}"] += 1
                    hits[f"участок {key} > допуска"] += x > TOL_REL
                    bad_focus |= key == focus and x > TOL_REL
            hits["тип верен"] += c.kind == kind
            good = c.kind == kind and d <= TOL_REL
            if tree is not None:
                dev["кривая -> точное"].append(tree.query(s)[0].max() / diag * 1000)
        hits["стенка воспроизведена"] += good
        hits[f"воспроизведена ({CURVE_TYPES[kind]})"] += good
        hits[f"стенок ({CURVE_TYPES[kind]})"] += 1
        if rec["tilt"].get(m, 0) > TILT_COS:
            hits["наклонных стенок"] += 1
            hits["наклонных воспроизведено"] += good
        ok &= good
    hits["операций"] += 1
    hits["профиль верен"] += ok
    hits["стенок без точного ребра"] += len(set(rec["sides"]) - set(gold))
    if draw is not None and (focus is None or bad_focus):
        fig, ax = plt.subplots(figsize=(8, 8))
        for parts in gold.values():
            for _, p in parts:
                ax.plot(p[:, 0], p[:, 1], "-", color="0.8", lw=5, zorder=1)
        for w in rec["walls"]:
            ax.plot(w.points[:, 0], w.points[:, 1], ".", color="0.3", ms=2, zorder=2)
        for loop in rec["loops"]:
            for c in loop:
                s = sample_curve(c.curve, 64)
                if s is not None:
                    ax.plot(s[:, 0], s[:, 1], "-", color=COLORS[c.kind], lw=1.5, zorder=3)
                    for end, kind in zip(s[[0, -1]], c.ends):
                        ax.plot(*end, "o", color=JOINT_COLORS[kind], ms=6, zorder=4)
        ax.set_aspect("equal")
        ax.set_title(f"{rec['uid']} {rec['op']}: профиль {'верен' if ok else 'с ошибками'}\n"
                     "стыки: " + ", ".join(f"{k} — {c}" for k, c in JOINT_COLORS.items()), fontsize=9)
        fig.tight_layout()
        fig.savefig(draw, dpi=80)
        plt.close(fig)
    return bad_focus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--draw", type=int, default=12)
    ap.add_argument("--exact-mesh", action="store_true", help="точный меш этапа 1 вместо скана")
    ap.add_argument("--focus", choices=("середина", *JOINTS), help="рисовать только операции с плохим стыком этого вида")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    uids = CORPUS.uids()
    uids = split_uids(uids)[args.split][:args.parts]
    hits, dev, drawn = Counter(), defaultdict(list), 0
    with Pool(args.workers) as pool:
        for k, recs in enumerate(pool.imap_unordered(op_records, [(u, args.exact_mesh) for u in uids], chunksize=2), 1):
            for rec in recs:
                path = OUT / f"{args.focus or 'все'}_{rec['uid']}_{rec['op']}.png" if drawn < args.draw else None
                hit = evaluate(rec, hits, dev, path, args.focus)
                drawn += path is not None and (args.focus is None or hit)
            if k % 50 == 0:
                print(f"{k}/{len(uids)}", file=sys.stderr, flush=True)
    out = {"деталей": len(uids), "операций": hits["операций"],
           "профиль верен целиком": hits["профиль верен"] / max(hits["операций"], 1),
           "стенка воспроизведена": hits["стенка воспроизведена"] / max(hits["стенок"], 1),
           "тип верен": hits["тип верен"] / max(hits["стенок"], 1),
           "по типам": {t: hits[f"воспроизведена ({t})"] / hits[f"стенок ({t})"] for t in CURVE_TYPES if hits[f"стенок ({t})"]},
           "стенок без точного ребра": hits["стенок без точного ребра"],
           "средний угол к точному направлению, °": {k: round(hits[k] / max(hits["операций"], 1), 3)
                                                    for k in ("угол направления к точному, °", "угол нормали опоры к точному, °")},
           "участки > 5‰ (доля, число)": {k: (round(hits[f"участок {k} > допуска"] / hits[f"участок {k}"], 3), hits[f"участок {k}"])
                                          for k in ("середина", *JOINTS) if hits[f"участок {k}"]},
           "наклонных стенок (доля, воспроизведено)": (hits["наклонных стенок"] / max(hits["стенок"], 1),
                                                     hits["наклонных воспроизведено"] / max(hits["наклонных стенок"], 1))}
    for name, x in dev.items():
        x = np.array(x)
        out[f"отклонение {name}, ‰ D"] = {"медиана": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2),
                                          "доля > 5‰": round(float((x > 5).mean()), 3), "стенок": len(x)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
