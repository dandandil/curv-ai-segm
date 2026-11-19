"""Решатель эскиза с оракулом ограничений (шаг 1 перед Graphormer эскиза, вариант (а): топология и
типы — из разметки, геометрия — по точкам скана).

    python scripts/sketch/oracle_solver.py --parts 200 --split val

Опыты на каждом сечении: подгонка кусков по отдельности (`build_pieces`, как сейчас), решатель
без ограничений (общие стыки), + касание на стыках (то, что даёт 2D сеть), + все истинные
ограничения. Отклонение — наибольшее расстояние от кривой до точных рёбер её куска, ‰ диагонали;
только отрезки, дуги и окружности с точными параметрами. Плюс частоты меток.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sketch"))

from train_section import section_uids  # noqa: E402

from src.opseq.data import split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import sample_curve  # noqa: E402
from src.sketch.relations import section_sketch  # noqa: E402
from src.sketch.solver import RELATIONS, curve_tuple, solve, solve_greedy  # noqa: E402

EXPERIMENTS = ("по отдельности", "решатель", "+ касание", "+ все ограничения")
# Сечения крупнее не решаются: три сечения одной детали по ~100 кривых и ~2000 меток держали
# весь прогон. Выбрано.
MAX_CURVES = 60
ABLATE = False                   # --ablate: касание + по одному виду ограничений


def deviation(curve: tuple, exact, diag: float) -> float | None:
    s = sample_curve(curve, 64)
    if s is None or exact is None:
        return None
    d = np.linalg.norm(s[:, None] - exact.samples[None, ::max(1, len(exact.samples) // 400)], axis=2).min(1)
    return float(d.max() / diag * 1000)


def one(uid: str) -> tuple[dict, Counter, Counter]:
    torch.set_num_threads(1)
    z = np.load(Path(SCFG.data.root) / "sections" / f"{uid}.npz")
    diag = float(z["diag"])
    dev, labels, stats = defaultdict(list), Counter(), Counter()
    for sec in range(len(z["sec_origin"])):
        got = section_sketch(z, sec)
        if got is None:
            continue
        sketch, exact, relations, fits = got
        if len(sketch.curves) > MAX_CURVES:
            stats["пропущено сечений: кривых больше MAX_CURVES"] += 1
            continue
        stats["сечений"] += 1
        labels.update(r.name for r in relations)
        movable = [i for i, c in enumerate(sketch.curves) if c.kind in ("line", "arc", "circle") and exact[i] is not None]
        stats["кусков с точными параметрами"] += len(movable)
        stats["кусков"] += len(sketch.curves)
        for i in movable:
            d = deviation(fits[i], exact[i], diag)
            if d is not None:
                dev[EXPERIMENTS[0]].append(d)
        tangent = [r for r in relations if r.name == "tangent"]
        runs = list(zip(EXPERIMENTS[1:], ([], tangent, relations)))
        if ABLATE:
            runs = [(f"+ касание + {k}", tangent + [r for r in relations if r.name == k]) for k in RELATIONS[1:]]
        for name, rels in runs:
            try:
                solved, accepted, redundant, rejected = solve_greedy(copy.deepcopy(sketch), rels)
            except Exception:                                   # вырожденный эскиз
                stats[f"сбой: {name}"] += 1
                continue
            stats[f"принято: {name}"] += len(accepted)
            stats[f"избыточных: {name}"] += len(redundant)
            stats[f"отвергнуто: {name}"] += len(rejected)
            for i in movable:
                d = deviation(curve_tuple(solved, i), exact[i], diag)
                if d is not None:
                    dev[name].append(d)
    return dict(dev), labels, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=200)
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ablate", action="store_true", help="касание + по одному виду ограничений")
    args = ap.parse_args()
    global ABLATE
    ABLATE = args.ablate
    uids = split_uids(section_uids())[args.split][:args.parts]
    dev, labels, stats = defaultdict(list), Counter(), Counter()
    with Pool(args.workers) as pool:
        for n, (d, lab, st) in enumerate(pool.imap_unordered(one, uids, chunksize=2), 1):
            if n % 20 == 0:
                print(f"{n}/{len(uids)}", file=sys.stderr, flush=True)
            for k, v in d.items():
                dev[k] += v
            labels.update(lab)
            stats.update(st)
    out = {"деталей": len(uids), **stats, "меток": {r: labels[r] for r in RELATIONS}}
    for name in sorted(dev, key=lambda k: (k not in EXPERIMENTS, k)):
        v = np.array(dev[name])
        if len(v):
            out[name] = {"кривых": len(v), "медиана ‰": round(float(np.median(v)), 2),
                         "p90 ‰": round(float(np.percentile(v, 90)), 2), "p99 ‰": round(float(np.percentile(v, 99)), 2),
                         "доля > 5‰": round(float((v > 5).mean()), 4)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
