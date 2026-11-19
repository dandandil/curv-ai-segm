"""Оценка 2D сети сечения на val или test: стыки водоразделом, топология цепочки, кривые через стыки.

    python scripts/sketch/evaluate_section.py --run runs/section_base --split val

Стык-пик засчитан, если он не дальше одного стыка от размеченного (метка ставится на ближайший
к точной вершине стык, соседний годится так же). Цепочка верна по топологии — все размеченные
стыки найдены и лишних нет; по типам — ещё и тип каждого куска совпал с большинством меток его
отрезков. Отклонение кривой — наибольшее расстояние от точек кривой куска до точного сечения,
/ диагональ; то же по идеальным стыкам и типам (оракул) — потолок построения кривых.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sketch"))

from train_section import loader_of, section_uids  # noqa: E402

from src.opseq.data import split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import build_pieces, build_pieces_checked, peaks, sample_curve  # noqa: E402
from src.sketch.data import Parts  # noqa: E402
from src.sketch.model import SectionNet  # noqa: E402
from src.sketch.section import CURVE_TYPES  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7)


def match(pred, gold, n: int, closed: bool, xy=None, loose=(), ignore=(), tol: float = 0.0
          ) -> tuple[int, int, int, dict]:
    """TP, FP, FN и пары пик -> метка (номера стыков цепочки). Метка из `loose` (острые) — пик не
    дальше `tol` по `xy` (точки стыков), прочие — в ±1 стыке. Пик у метки «не требовать» (`ignore`)
    не лишний."""
    loose, used, pairs = set(loose), set(), {}

    def steps(p, g):
        d = abs(int(p) - int(g))
        return min(d, n - d) if closed else d

    def near(p, g):
        return (xy is not None and float(np.linalg.norm(xy[p] - xy[g])) <= tol) if g in loose else steps(p, g) <= 1

    for p in pred:
        cand = [g for g in gold if g not in used and near(p, g)]
        if cand:
            g = min(cand, key=lambda q: steps(p, q))
            pairs[int(p)] = int(g)
            used.add(g)
    fp = sum(1 for p in pred if int(p) not in pairs and not any(steps(p, q) <= 1 or
             (xy is not None and float(np.linalg.norm(xy[p] - xy[q])) <= tol) for q in ignore))
    tp = len(pairs)
    return tp, fp, len(gold) - tp, pairs


def deviation(curve, tree, diag) -> float | None:
    s = sample_curve(curve)
    if s is None or tree is None:
        return None
    return float(tree.query(s)[0].max() / diag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-split", action="store_true", help="без разрезания кусков по отклонению")
    args = ap.parse_args()
    cfg = SCFG.train
    device = torch.device("cuda")
    model = SectionNet().to(device)
    model.load_state_dict(torch.load(ROOT / args.run / args.checkpoint, map_location=device)["model"])
    model.eval()
    uids = split_uids(section_uids())[args.split]
    if args.limit:
        uids = uids[:args.limit]
    loader = loader_of(Parts(uids, False), False, args.workers)

    hits = Counter()
    dev = defaultdict(list)
    exact_cache: dict = {}

    def exact_tree(uid, sec):
        if uid not in exact_cache:
            exact_cache.clear()
            z = np.load(Path(SCFG.data.root) / "sections" / f"{uid}.npz")
            eo = np.r_[0, np.cumsum(z["edge_samples"])]
            per = defaultdict(list)
            for i, s in enumerate(z["edge_sec"]):
                per[int(s)].append(z["samples"][eo[i]:eo[i + 1]])
            exact_cache[uid] = {s: cKDTree(np.concatenate(v)) for s, v in per.items()}
        return exact_cache[uid].get(sec)

    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            type_logits, bnd, tan = model(b)
            tprob = torch.softmax(type_logits, 1).cpu().numpy()
            pb = torch.sigmoid(bnd).cpu().numpy()
            pt = torch.sigmoid(tan).cpu().numpy()
            jc = batch["junction_chain"].numpy()
            jp = batch["junction_point"].numpy()
            lab_b, lab_t = batch["boundary"].numpy(), batch["tangent"].numpy()
            seg_t = batch["type"].numpy()
            order = np.argsort(jc, kind="stable")
            bounds = np.searchsorted(jc[order], np.arange(len(batch["chains"]) + 1))
            for ci, meta in enumerate(batch["chains"]):
                js = order[bounds[ci]:bounds[ci + 1]]
                s0 = int(batch["start"][ci])
                n_seg = meta["segments"]
                tp_ = tprob[s0:s0 + n_seg]
                st = seg_t[s0:s0 + n_seg]
                closed = meta["closed"]
                gold = set(np.flatnonzero(lab_b[js] == 1).tolist())
                xy = meta["points"][jp[js]]
                sharp_gold = {g for g in gold if lab_t[js[g]] == 0}
                ignore = np.flatnonzero(lab_b[js] == -2)
                mtol = cfg.match_rel * meta["diag"]
                hits["цепочек"] += 1
                tree = exact_tree(meta["uid"], meta["sec"])
                diag = meta["diag"]
                for thr in THRESHOLDS:
                    pk = peaks(pb[js], closed, thr, cfg.peak_window)
                    tp, fp, fn, pairs = match(pk, gold, len(js), closed, xy, sharp_gold, ignore, mtol)
                    hits[f"{thr}:tp"] += tp
                    hits[f"{thr}:fp"] += fp
                    hits[f"{thr}:fn"] += fn
                    ok = fp == 0 and fn == 0
                    hits[f"{thr}:топология"] += ok
                    if thr != cfg.peak_threshold:
                        continue
                    found = set(pairs.values())
                    for g in gold:
                        kind = "касательных" if lab_t[js[g]] == 1 else "острых"
                        hits[f"{kind}:всего"] += 1
                        hits[f"{kind}:найдено"] += g in found
                    for p, g in pairs.items():
                        if lab_t[js[g]] >= 0:
                            hits["касание:всего"] += 1
                            hits["касание:верно"] += int((pt[js[p]] > 0.5) == (lab_t[js[g]] == 1))
                    pts_idx = jp[js[pk]]
                    sharp = pt[js[pk]] <= 0.5
                    if args.no_split:
                        pieces, _ = build_pieces(meta["points"], closed, pts_idx, sharp, tp_)
                    else:
                        pieces, _, pts_idx, _ = build_pieces_checked(meta["points"], closed, pts_idx, sharp, tp_,
                                                                     cfg.split_rel * diag)
                        # стыки после разрезов — обратно в номера стыков цепочки
                        of_point = {int(q): i for i, q in enumerate(jp[js])}
                        pk2 = np.array([of_point[int(q)] for q in pts_idx if int(q) in of_point], int)
                        tp2, fp2, fn2, _ = match(pk2, gold, len(js), closed, xy, sharp_gold, ignore, mtol)
                        hits["разрез:tp"] += tp2
                        hits["разрез:fp"] += fp2
                        hits["разрез:fn"] += fn2
                        hits["разрез:топология"] += fp2 == 0 and fn2 == 0
                        hits["разрезов"] += len(pk2) - len(pk)
                        ok = fp2 == 0 and fn2 == 0
                    if ok:
                        good = True
                        for pc in pieces:
                            seg = (np.r_[np.arange(pc.start, n_seg), np.arange(0, pc.end)]
                                   if closed and pc.end <= pc.start else np.arange(pc.start, pc.end))
                            lab = st[seg][st[seg] >= 0]
                            if len(lab) and np.bincount(lab, minlength=4).argmax() != pc.kind:
                                good = False
                        hits["топология и типы"] += good
                    for pc in pieces:
                        hits[f"кусков {CURVE_TYPES[pc.kind]}"] += 1
                        d = deviation(pc.curve, tree, diag)
                        if d is not None:
                            dev["сеть"].append(d)
                    # оракул: размеченные стыки, касание и типы
                    g_idx = np.array(sorted(gold), int)
                    one_hot = np.zeros_like(tp_)
                    one_hot[st >= 0, st[st >= 0]] = 1
                    one_hot[st < 0] = 0.25
                    pieces_o, _ = build_pieces(meta["points"], closed, jp[js[g_idx]], lab_t[js[g_idx]] == 0, one_hot)
                    for pc in pieces_o:
                        d = deviation(pc.curve, tree, diag)
                        if d is not None:
                            dev["оракул"].append(d)

    out = {"split": args.split, "цепочек": hits["цепочек"]}
    for thr in THRESHOLDS:
        tp, fp, fn = hits[f"{thr}:tp"], hits[f"{thr}:fp"], hits[f"{thr}:fn"]
        out[f"порог {thr}"] = {"точность": tp / max(tp + fp, 1), "полнота": tp / max(tp + fn, 1),
                               "цепочка верна по топологии": hits[f"{thr}:топология"] / max(hits["цепочек"], 1)}
    if hits["разрез:tp"] + hits["разрез:fn"]:
        tp, fp, fn = hits["разрез:tp"], hits["разрез:fp"], hits["разрез:fn"]
        out["после разрезов"] = {"точность": tp / max(tp + fp, 1), "полнота": tp / max(tp + fn, 1),
                                 "цепочка верна по топологии": hits["разрез:топология"] / max(hits["цепочек"], 1),
                                 "разрезов": hits["разрезов"]}
    out["цепочка верна по топологии и типам"] = hits["топология и типы"] / max(hits["цепочек"], 1)
    for kind in ("острых", "касательных"):
        out[f"полнота {kind} стыков"] = hits[f"{kind}:найдено"] / max(hits[f"{kind}:всего"], 1)
    out["касание на найденных стыках"] = hits["касание:верно"] / max(hits["касание:всего"], 1)
    out["кусков"] = {k[7:]: v for k, v in hits.items() if k.startswith("кусков ")}
    for k, v in dev.items():
        v = np.array(v) * 1000
        out[f"отклонение кривой, ‰ диагонали ({k})"] = {"медиана": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                                        "кривых": len(v)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
