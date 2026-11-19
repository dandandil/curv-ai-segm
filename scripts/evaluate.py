"""Оценка обученной модели: mean-shift и метрики сегментации.

    python scripts/evaluate.py --checkpoint runs/base/last.pt --split val --limit 100

Кластеризация запускается только здесь: в обучении её нет (README,
«Кластеризация»), и мерить по ней каждую эпоху слишком дорого.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cluster as C  # noqa: E402
from src import graph as G  # noqa: E402
from src import metrics as M  # noqa: E402
from src.config import CFG  # noqa: E402
from src.data import CachedParts, ring_indices, split_uids  # noqa: E402
from src.model import PRIMITIVE_TYPES, PrimitiveSegNet  # noqa: E402


@torch.no_grad()
def evaluate_part(model, item: dict, device, generator, min_face: int) -> dict:
    """Один прогон: эмбеддинг -> кластеры -> метрики по одной детали."""
    x = torch.from_numpy(item["features"]).to(device).unsqueeze(0)
    neigh = G.face_neighbors(item["adjacency"], item["n_faces"], device)
    idx = ring_indices(item, device, generator)
    embedding, logits = model(x, idx)

    labels = torch.from_numpy(item["labels"]).to(device)
    big = torch.bincount(labels)[labels] >= min_face
    types = torch.from_numpy(item["types"]).to(device)
    pred, width = C.segment(embedding[0], CFG.cluster.quantile, CFG.cluster.iterations,
                            CFG.cluster.bandwidth_samples, CFG.cluster.bandwidth_floor,
                            CFG.cluster.chunk, CFG.cluster.max_clusters,
                            width=CFG.cluster.bandwidth)

    dist = M.boundary_distance(neigh, labels)
    counts = torch.bincount(types[types >= 0], minlength=len(PRIMITIVE_TYPES))
    return {
        "uid": item["uid"],
        "clusters": int(pred.unique().numel()),
        "bandwidth": float(width),
        "dominant_type": PRIMITIVE_TYPES[int(counts.argmax())],
        # Грани мельче min_face треугольников не восстановит никакая
        # кластеризация — из метрик нарезки они выброшены, как и в train.py.
        **M.segment_scores(labels[big], pred[big]),
        **M.type_scores(logits[0], types, labels),
        **M.boundary_profile(labels, pred, dist),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-face", type=int, default=M.MIN_FACE,
                    help="грани мельче стольких треугольников из метрик нарезки выбрасываются")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=device)
    if state.get("log_curvature", False) != CFG.model.log_curvature:
        raise SystemExit("чекпоинт обучен с другим преобразованием входа "
                         f"(log_curvature={state.get('log_curvature', False)}), "
                         "а конфиг задаёт другое — признаки не совпадут")
    model = PrimitiveSegNet(state["channels"], CFG.model.emb_size).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    uids = split_uids(args.split, args.variant)[:args.limit]
    parts = CachedParts(uids, args.variant)
    rows = [evaluate_part(model, parts[i], device, generator, args.min_face) for i in range(len(parts))]
    if not rows:
        raise SystemExit(f"в сплите {args.split!r} нет собранных деталей")

    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in numeric}
    print(f"деталей {len(rows)}, эпоха чекпоинта {state['epoch']}")
    print(f"  протечек на деталь {summary['leaks']:.2f}, лишних разрезов {summary['splits']:.2f}"
          f" при {summary['faces']:.1f} гранях")
    print(f"  IoU {summary['mean_iou']:.3f}, граней без кластера {summary['missed_frac']:.3f},"
          f" кластеров {summary['clusters']:.1f}, ширина ядра {summary['bandwidth']:.3f}")
    print(f"  тип: треугольники {summary['type_acc_tri']:.3f}, грани {summary['type_acc_face']:.3f}")
    print("  ошибка по расстоянию до границы: "
          + "  ".join(f"{r}:{summary[f'wrong@{r}']:.3f}" for r in range(9) if f"wrong@{r}" in summary))
    for key in ("leaks", "splits", "mean_iou"):
        print(f"  {key} по преобладающему типу: "
              + "  ".join(f"{t} {v:.2f}" for t, v in M.by_type(rows, key).items()))

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "parts": rows},
                                       ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
