"""Предрасчёт колец соседства в кэш.

    python scripts/build_rings.py
    python scripts/build_rings.py --limit 20 --overwrite

Кольца от весов не зависят, а на лету стоили 172 мс на деталь — почти столько
же, сколько сам шаг сети (212 мс). Считаются один раз на GPU и кладутся рядом с
кэшем детали в `rings.npz` вместе с подписью конфига: если кольца в конфиге
поменяются, файл перестанет совпадать по подписи и молча не будет прочитан.

Цена — выборка соседей внутри кольца фиксируется. На лету она была своя на
каждой эпохе и работала как слабая аугментация.
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

from dataset.cache import CACHE, CONFIG_TAG, SPLITS, part_dir as cache_dir  # noqa: E402
from src import graph as G  # noqa: E402
from src.config import CFG  # noqa: E402
from src.data import RINGS_FILE, rings_signature  # noqa: E402

# Индексы хранятся в uint16, если меш не больше 65535 треугольников, иначе в int32.
MAX_INDEX = np.iinfo(np.uint16).max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", help="только детали этих сплитов")
    ap.add_argument("--shard", type=int, default=0, help="номер машины при сборке на нескольких")
    ap.add_argument("--shards", type=int, default=1, help="сколько машин собирают вместе")
    args = ap.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    signature = rings_signature()
    scope = f"{args.variant}.{CONFIG_TAG}" if CONFIG_TAG else args.variant
    uids = sorted(p.parent.parent.name for p in CACHE.glob(f"*/{scope}/meta.json"))
    if args.splits:
        want = {u for s in args.splits for u in (SPLITS / f"{s}.txt").read_text().split()}
        uids = [u for u in uids if u in want]
    uids = uids[:args.limit][args.shard::args.shards]
    print(f"кольца в {RINGS_FILE}, деталей {len(uids)}", flush=True)

    built = skipped = 0
    t0 = time.perf_counter()
    # У кэша кусков (configs/tile.yaml) кольца — в каталоге каждого куска t<i>.
    dirs = [(u, d) for u in uids for d in (sorted(cache_dir(u, args.variant).glob("t[0-9]*"))
                                           or [cache_dir(u, args.variant)])]
    print(f"каталогов с мешем {len(dirs)}", flush=True)
    for i, (uid, here) in enumerate(dirs):
        out = here / RINGS_FILE
        if out.exists() and not args.overwrite:
            with np.load(out) as z:
                if str(z["signature"]) == signature:
                    skipped += 1
                    continue
        with np.load(here / "work.npz") as z:
            adjacency, n_faces = z["adjacency"], len(z["labels"])
        # куски с каймой бывают до 200 тысяч — им int32 (README, «Ядро 60 тысяч»)
        dtype = np.uint16 if n_faces <= MAX_INDEX else np.int32
        neigh = G.face_neighbors(adjacency, n_faces, device)
        idx = G.ring_neighbors(neigh, CFG.model.rings, CFG.model.neighbors,
                               CFG.model.frontier_cap, CFG.model.frontier_per_ring, generator)
        # Слои хранятся по отдельности: у них может быть разное число соседей.
        rings = {f"l{i}": t.to(torch.int32).cpu().numpy().astype(dtype) for i, t in enumerate(idx)}
        # Запись через временный файл: оборванный на полпути файл был бы прочитан.
        tmp = here / (RINGS_FILE.removesuffix(".npz") + ".tmp.npz")
        np.savez(tmp, signature=np.array(signature), **rings)
        os.replace(tmp, out)
        built += 1
        if (i + 1) % 500 == 0 or i + 1 == len(dirs):
            dt = time.perf_counter() - t0
            print(f"{i + 1}/{len(dirs)}  собрано {built}  пропущено {skipped}  "
                  f"осталось {(len(dirs) - i - 1) * dt / max(built, 1) / 60:.0f} мин", flush=True)
    print(f"итого: {built} собрано, {skipped} уже были, {(time.perf_counter() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
