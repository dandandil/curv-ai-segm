"""Оценка эталона ParSeNet: как `generate_predictions.py` авторов.

    python scripts/parsenet/evaluate_parsenet.py --checkpoint runs/parsenet/clean/best.pt --limit 100
    python scripts/parsenet/evaluate_parsenet.py --pretrained ~/reference/point2cad_weights/parsenet.pth --limit 100

`--pretrained` — готовые веса авторов (обучены на ABC, выложены в репозитории
Point2CAD): голова типа на 10 классов их нумерации, поэтому точность типа не
считается.

Форма целиком (10 000 точек) с их нормировкой и шумом вдоль нормали,
нормированный эмбеддинг, mean-shift с квантилем 0.015 и 50 итерациями, S-IoU
авторов. Рядом — наши метрики по точкам (`src/metrics.segment_scores`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import segment_scores  # noqa: E402
from src.model import PRIMITIVE_TYPES  # noqa: E402
from src.parsenet import data as PD  # noqa: E402
from src.parsenet.config import PCFG  # noqa: E402
from src.parsenet.mean_shift import guard_mean_shift  # noqa: E402
from src.parsenet.metrics import segment_iou  # noqa: E402
from src.parsenet.model import ParSeNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, help="чекпоинт нашего обучения")
    ap.add_argument("--pretrained", type=Path, help="state_dict авторов (parsenet.pth из Point2CAD)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if (args.checkpoint is None) == (args.pretrained is None):
        raise SystemExit("нужен ровно один из --checkpoint и --pretrained")
    device = torch.device("cuda")
    if args.pretrained:
        raw = torch.load(args.pretrained, map_location=device)
        # DataParallel добавлял префикс module.; encoder.bn1..bn5 у авторов — дубли
        # нормировок внутри свёрток и две неиспользуемые.
        weights = {k.removeprefix("module."): v for k, v in raw.items()}
        weights = {k: v for k, v in weights.items() if not re.match(r"encoder\.bn\d\.", k)}
        n_types = weights["mlp_prim_prob2.weight"].shape[0]
        source = f"готовые веса {args.pretrained.name}"
    else:
        state = torch.load(args.checkpoint, map_location=device)
        weights, n_types = state["model"], len(PRIMITIVE_TYPES)
        source = f"эпоха {state['epoch']}"
    model = ParSeNet(PCFG.model.emb_size, n_types, PCFG.model.neighbors).to(device)
    model.load_state_dict(weights)
    model.eval()
    split = PD.PointSplit(args.split)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    c = PCFG.cluster
    rows = []
    for i in range(min(len(split), args.limit or len(split))):
        points, normals, labels, prim = PD.shape(split, i, rng, PCFG.data.normal_noise)
        x = torch.from_numpy(np.concatenate([points, normals], 1)).t().unsqueeze(0).to(device)
        with torch.no_grad():
            emb, log_prob = model(x)
        z = F.normalize(emb[0].t(), p=2, dim=1)
        pred = guard_mean_shift(z, c.quantile, c.iterations, c.bandwidth_samples, c.bandwidth_floor,
                                c.max_clusters, generator).cpu().numpy()
        ours = segment_scores(torch.from_numpy(labels), torch.from_numpy(pred))
        type_acc = (float("nan") if args.pretrained
                    else float((log_prob[0].argmax(0).cpu().numpy() == prim).mean()))
        rows.append({"uid": split.uids[i], "s_iou": segment_iou(labels, pred), **ours, "type_acc": type_acc})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}: S-IoU {np.mean([r['s_iou'] for r in rows]):.3f}", flush=True)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    s = {k: float(np.mean([r[k] for r in rows])) for k in numeric}
    print(f"деталей {len(rows)}, {source}: S-IoU ParSeNet {s['s_iou']:.3f}, наш IoU {s['mean_iou']:.3f}, "
          f"протечек {s['leaks']:.2f}, разрезов {s['splits']:.2f}, кластеров {s['clusters']:.1f} при {s['faces']:.1f} гранях, "
          f"тип по точкам {s['type_acc']:.3f}")
    if args.out:
        args.out.write_text(json.dumps({"summary": s, "parts": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
