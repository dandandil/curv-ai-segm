"""Графы поверхностей тел opseq по сканам: признаки и точки узлов, соседство, метки отношений.

    CURV_CONFIG=configs/tile.yaml python scripts/opseq/build_graphs.py --corpus seg [--split <uids>] --workers 20

Берутся годные тела корпуса (`skip` пуст в метках `data.labels` из configs/opseq.yaml) со сканом; пишет
`<корень>/<graphs>/<uid>.npz` (`src/opseq/data.py`, `build_graph`). Готовые пропускаются, `--force` —
пересобрать. Конфиг —
tile: сканы сделаны с ним, подгонка берёт из него полосу у границы и минимум точек.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import collections
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.opseq.config import OCFG  # noqa: E402
from src.opseq.data import CORPORA, build_graph  # noqa: E402

C = None                 # корпус; задаётся в main до запуска пула


def _one(uid: str) -> tuple[str, str | None, str | None]:
    try:
        labels = C.labels(uid)
        if labels["skip"]:
            return uid, "отсеяно в метках", None
        if not C.scan_path(uid).exists():
            return uid, "нет скана", None
        g = build_graph(uid, labels, C)
        if g is None:
            return uid, "узлов больше max_nodes", None
        tmp = C.root / C.graphs / f"{uid}.tmp.npz"
        np.savez_compressed(tmp, **g)
        tmp.rename(C.graph_path(uid))
        return uid, None, None
    except Exception:
        return uid, None, traceback.format_exc(limit=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--split", help="файл со списком uid; по умолчанию — все детали корпуса со сканом")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="пересобрать готовые")
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="остановиться, когда на диске меньше")
    args = ap.parse_args()
    global C
    C = CORPORA[args.corpus]
    (C.root / C.graphs).mkdir(exist_ok=True)
    pool_uids = Path(args.split).read_text().split() if args.split else \
        sorted(p.stem for p in (C.root / C.scan).glob("*.npz"))
    uids = [u for u in pool_uids if args.force or not C.graph_path(u).exists()]
    if args.limit:
        uids = uids[:args.limit]
    t0, counts = time.perf_counter(), collections.Counter()
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for n, (uid, skip, err) in enumerate(pool.imap_unordered(_one, uids, chunksize=4), 1):
            if err:
                counts["ошибка"] += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                counts[skip or "собрано"] += 1
            if n % 1000 == 0 or n == len(uids):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(uids)}  {dict(counts)}  осталось {(len(uids) - n) * dt / n / 60:.0f} мин", flush=True)
            if n % 100 == 0:
                st = os.statvfs(C.root)
                if st.f_bavail * st.f_frsize / 2 ** 30 < args.min_free_gb:
                    print(f"=== место кончилось: меньше {args.min_free_gb:g} ГБ", flush=True)
                    pool.terminate()
                    break
    print(f"итого: {dict(counts)}, {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
