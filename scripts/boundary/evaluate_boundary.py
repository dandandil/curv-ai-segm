"""Оценка варианта D: mean-shift и нарезка по графу, с вето и без.

    python scripts/boundary/evaluate_boundary.py --checkpoint runs/boundary_ab/last.pt --limit 100
    python scripts/boundary/evaluate_boundary.py --checkpoint ... --depth 0.1 --merge 0.6 --veto none

Пороги нарезки берутся из configs/boundary.yaml, аргументы их переопределяют —
для подбора на валидации без правки конфига. Пороги в треугольниках (`min_region`,
`--min-face`) по умолчанию — под разрешение конфига (`src.runs.thresholds`), разметка
(`--narrow-rule`, `--absorb-blends-mm`) — та же, с которой прогон обучался.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "boundary"))

import train_boundary as bt  # noqa: E402
from src import cluster as C  # noqa: E402
from src import metrics as M  # noqa: E402
from src.boundary.config import BCFG  # noqa: E402
from src.boundary.model import BoundarySegNet, head_config  # noqa: E402
from src.config import CFG  # noqa: E402
from src.data import CachedParts, input_names, part_meta, split_uids  # noqa: E402
from src.model import PRIMITIVE_TYPES  # noqa: E402
from src.runs import thresholds  # noqa: E402

METHODS = ("seg", "graph", "graphv")


@torch.no_grad()
def evaluate_part(model, item, device, generator, min_face: int, cfg) -> dict:
    batch = bt.build_batch(item, device, generator)
    emb, logits, edge = model(batch["x"], batch["idx"], batch["edges"])
    labels, types = batch["labels"], batch["types"]
    big = (torch.bincount(labels)[labels] >= min_face) & batch["core"]   # метрики — по ядру куска
    counts = torch.bincount(types[types >= 0], minlength=len(PRIMITIVE_TYPES))
    row = {"uid": item["uid"], "dominant_type": PRIMITIVE_TYPES[int(counts.argmax())],
           **{f"edge_{k}": v for k, v in bt.core_edge_scores(edge, batch).items()},
           **M.type_scores(logits[0], types, labels)}
    t = time.perf_counter()
    pred, _ = C.segment(emb[0], CFG.cluster.quantile, CFG.cluster.iterations, CFG.cluster.bandwidth_samples,
                        CFG.cluster.bandwidth_floor, CFG.cluster.chunk, CFG.cluster.max_clusters,
                        width=CFG.cluster.bandwidth)
    results = {"seg": (pred, time.perf_counter() - t)}
    for name, veto in (("graph", False), ("graphv", True)):
        t = time.perf_counter()
        results[name] = (bt.graph_labels(emb, edge, batch, cfg, veto), time.perf_counter() - t)
    for name, (lab, dt) in results.items():
        row.update({f"{name}_{k}": v for k, v in M.segment_scores(labels[big], lab[big]).items()})
        row[f"{name}_seconds"] = dt
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-face", type=int, default=thresholds()[0])
    ap.add_argument("--depth", type=float)
    ap.add_argument("--merge", type=float)
    ap.add_argument("--veto", help="число или none")
    ap.add_argument("--min-region", type=int)
    ap.add_argument("--narrow-rule", action="store_true", help="правило узких граней, как у прогона")
    ap.add_argument("--absorb-blends-mm", type=float, default=0.0,
                    help="скругления и фаски до стольких мм — соседним граням, как у прогона")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = thresholds()[1]
    overrides = {"depth": args.depth, "merge": args.merge, "min_region": args.min_region}
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    if args.veto is not None:
        cfg = dataclasses.replace(cfg, veto=None if args.veto == "none" else float(args.veto))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=device)
    model = BoundarySegNet(state["channels"], CFG.model.emb_size, head_config(BCFG.model, state["model"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    uids = split_uids(args.split, args.variant)[:args.limit]
    # Прогон без струи (--no-jet) узнаётся по числу каналов чекпоинта.
    names = part_meta(uids[0], args.variant)["feature_names"]
    no_jet = state["channels"] == len(input_names(names, True)) < len(names)
    parts = CachedParts(uids, args.variant, absorb_mm=args.absorb_blends_mm, narrow=args.narrow_rule,
                        no_jet=no_jet)
    rows = [evaluate_part(model, parts[i], device, generator, args.min_face, cfg) for i in range(len(parts))]
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    s = {k: float(np.nanmean([r[k] for r in rows])) for k in numeric}
    print(f"деталей {len(rows)}, эпоха чекпоинта {state['epoch']}, нарезка {cfg}")
    print(f"  стыки: точность {s['edge_precision']:.3f}, полнота {s['edge_recall']:.3f}, F1 {s['edge_f1']:.3f}, "
          f"полнота на касательных {s['edge_tangent_recall']:.3f}")
    for name, title in zip(METHODS, ("mean-shift", "граф", "граф с вето")):
        print(f"  {title:<12} IoU {s[f'{name}_mean_iou']:.3f}, потеряно {s[f'{name}_missed_frac']:.3f}, "
              f"протечек {s[f'{name}_leaks']:.2f}, разрезов {s[f'{name}_splits']:.2f}, "
              f"кластеров {s[f'{name}_clusters']:.1f} при {s[f'{name}_faces']:.1f} гранях, "
              f"{s[f'{name}_seconds'] * 1000:.0f} мс")
    if args.out:
        args.out.write_text(json.dumps({"config": dataclasses.asdict(cfg), "summary": s, "parts": rows},
                                       ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
