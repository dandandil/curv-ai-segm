"""Нарезка сети по сплиту в файлы: вход для `evaluate_brep.py --input`.

    CURV_CONFIG=configs/aug60.yaml python scripts/segment_split.py --run aug60 --split test \
        --out outputs/seg/aug60_test --device cpu

На каждую деталь `<uid>.npz` с кластером и типом на треугольник рабочего меша —
то же, что ноутбук `inference.ipynb`, разделы 1-4. Пороги нарезки — под
разрешение конфига (`src.runs.thresholds`).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.cache import part_dir  # noqa: E402
from src import runs  # noqa: E402
from src.boundary import partition as PT  # noqa: E402
from src.boundary.model import boundary_strength  # noqa: E402
from src.data import CachedParts, ring_indices, split_uids  # noqa: E402
from src.config import CONFIG_PATH  # noqa: E402
from src.inference import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", default="0/1", help="i/n: каждая n-я деталь, начиная с i")
    args = ap.parse_args()

    # Конфиг читается из CURV_CONFIG при импорте, веса — по --run: разойдясь, они молча дали бы
    # чужой кэш, кольца и пороги нарезки.
    if runs.run_config(args.run) != CONFIG_PATH:
        raise SystemExit(f"у прогона {args.run} конфиг {runs.run_config(args.run).name}, а загружен "
                         f"{CONFIG_PATH.name}: запустите с CURV_CONFIG={runs.run_config(args.run).relative_to(ROOT)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(ROOT / "runs" / args.run / "best.pt", device)
    _, partition = runs.thresholds()
    i, n = map(int, args.shard.split("/"))
    uids = [u for u in split_uids(args.split, args.variant)
            if (part_dir(u, args.variant) / "work.npz").exists()][i::n]
    for k, uid in enumerate(uids):
        if (out / f"{uid}.npz").exists():
            continue
        t = time.perf_counter()
        item = CachedParts([uid], args.variant)[0]
        generator = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            x = torch.from_numpy(item["features"]).to(device).unsqueeze(0)
            edges = torch.from_numpy(np.asarray(item["adjacency"], np.int64)).to(device)
            idx = ring_indices(item, device, generator)
            embedding, logits, edge = model(x, idx, edges)
        z = torch.nn.functional.normalize(embedding[0].t(), dim=1).double().cpu().numpy()
        labels = PT.partition(item["n_faces"], np.asarray(item["adjacency"]), boundary_strength(edge),
                              partition, z)
        # через временный файл: оборванная запись иначе считалась бы готовой деталью
        tmp = out / f"{uid}.tmp.npz"
        np.savez_compressed(tmp, labels=labels.astype(np.int64),
                            types=logits[0].argmax(0).cpu().numpy().astype(np.int64))
        os.replace(tmp, out / f"{uid}.npz")
        print(f"{k + 1}/{len(uids)} {uid} {time.perf_counter() - t:.1f} с, кластеров {labels.max() + 1}",
              flush=True)


if __name__ == "__main__":
    main()
