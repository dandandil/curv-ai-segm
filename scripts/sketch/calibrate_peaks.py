"""Калибровка водораздела: порог и окно пиков, отдельный порог для стыков, которые сеть считает
касательными (голова касания > 0.5). Сеть прогоняется один раз, дальше — перебор по сетке.

    python scripts/sketch/calibrate_peaks.py --run runs/section_base --split val
    python scripts/sketch/calibrate_peaks.py --run runs/section_base --split test --only 0.5,0.3,2

Критерий — доля цепочек, верных по топологии (все размеченные стыки найдены, лишних нет;
сопоставление как в `evaluate_section.py`). Плюс гистограмма лучшей вероятности в ±1 стыке
у размеченных стыков, острых и касательных: видно, ниже порога пик или его нет.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sketch"))

from evaluate_section import match  # noqa: E402
from train_section import loader_of, section_uids  # noqa: E402

from src.opseq.data import split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import peaks  # noqa: E402
from src.sketch.data import Parts  # noqa: E402
from src.sketch.model import SectionNet  # noqa: E402

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WINDOWS = (1, 2, 3, 5)
BINS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def chains_of(run: str, split: str, workers: int) -> list[tuple]:
    """(вероятность границы, вероятность касания, метки границы, метки касания, замкнута, точки
    стыков, допуск острых) на цепочку."""
    device = torch.device("cuda")
    model = SectionNet().to(device)
    model.load_state_dict(torch.load(ROOT / run / "best.pt", map_location=device)["model"])
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader_of(Parts(split_uids(section_uids())[split], False), False, workers):
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            _, bnd, tan = model(b)
            pb, pt = torch.sigmoid(bnd).cpu().numpy(), torch.sigmoid(tan).cpu().numpy()
            jc = batch["junction_chain"].numpy()
            lb, lt = batch["boundary"].numpy(), batch["tangent"].numpy()
            order = np.argsort(jc, kind="stable")
            bounds = np.searchsorted(jc[order], np.arange(len(batch["chains"]) + 1))
            for ci, meta in enumerate(batch["chains"]):
                js = order[bounds[ci]:bounds[ci + 1]]
                out.append((pb[js], pt[js], lb[js], lt[js], meta["closed"],
                            meta["points"][batch["junction_point"].numpy()[js]], SCFG.train.match_rel * meta["diag"]))
    return out


def score(chains, thr_s: float, thr_t: float, window: int) -> dict:
    tp = fp = fn = ok = 0
    found = {"острых": [0, 0], "касательных": [0, 0]}
    for pb, pt, lb, lt, closed, xy, tol in chains:
        pk = peaks(pb, closed, 0.0, window)
        pk = pk[pb[pk] >= np.where(pt[pk] > 0.5, thr_t, thr_s)]
        gold = set(np.flatnonzero(lb == 1).tolist())
        a, b, c, pairs = match(pk, gold, len(pb), closed, xy, {g for g in gold if lt[g] == 0},
                               np.flatnonzero(lb == -2), tol)
        tp, fp, fn, ok = tp + a, fp + b, fn + c, ok + (b == 0 and c == 0)
        hit = set(pairs.values())
        for g in gold:
            f = found["касательных" if lt[g] == 1 else "острых"]
            f[0] += g in hit
            f[1] += 1
    return {"порог острых": thr_s, "порог касательных": thr_t, "окно": window,
            "топология": ok / len(chains), "точность": tp / max(tp + fp, 1), "полнота": tp / max(tp + fn, 1),
            **{f"полнота {k}": v[0] / max(v[1], 1) for k, v in found.items()}}


def histogram(chains) -> dict:
    best = {"острых": [], "касательных": []}
    for pb, _, lb, lt, closed, _, _ in chains:
        n = len(pb)
        for g in np.flatnonzero(lb == 1):
            near = [(g + d) % n if closed else g + d for d in (-1, 0, 1)]
            near = [j for j in near if 0 <= j < n]
            best["касательных" if lt[g] == 1 else "острых"].append(pb[near].max())
    return {k: {f"{BINS[i]:.1f}-{min(BINS[i + 1], 1):.1f}": round(float(c), 4)
                for i, c in enumerate(np.histogram(v, BINS)[0] / max(len(v), 1))} for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", help="порог острых,порог касательных,окно — без перебора")
    args = ap.parse_args()
    chains = chains_of(args.run, args.split, args.workers)
    out = {"split": args.split, "цепочек": len(chains), "лучшая вероятность у размеченных стыков": histogram(chains)}
    if args.only:
        s, t, w = args.only.split(",")
        out["результат"] = score(chains, float(s), float(t), int(w))
    else:
        grid = [score(chains, s, t, w) for s, t, w in itertools.product(THRESHOLDS, THRESHOLDS, WINDOWS)]
        grid.sort(key=lambda r: -r["топология"])
        out["лучшие"] = grid[:10]
        out["единый порог"] = sorted((r for r in grid if r["порог острых"] == r["порог касательных"]),
                                     key=lambda r: -r["топология"])[:5]
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
