"""Синтетические детали (README, «Данные v2»): основа из реконструкции + добавки `synth_ops` -> STEP и история.

    python scripts/opseq/synth_build.py --count 20000 --workers 12 [--start 0]

Деталь i: зерно i, основа — годная деталь реконструкции (метки без `skip`), uid `<дизайн основы>_syn<i>`, поэтому
разбиение `split_uids` по дизайну у синтетики то же, что у её основы. Пишет `synth/step/<uid>.stp` (симлинк в
`dataset/data/raw_step` для `build_parts`) и `synth/hist/<uid>.json` — `recon_opseq.history` по перечитанному
STEP плюс основа и параметры добавок. Каждая деталь — в своём процессе с таймаутом: скругления OCC иногда виснут.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
import zipfile
from collections import Counter
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.opseq.recon_caps import DATA, R, ZIP  # noqa: E402

S = DATA / "synth"
RAW = ROOT / "dataset" / "data" / "raw_step"
LABELS = "opseq.nonarrow.json"
# Как в пробе (outputs/synth_probe.py): граней до 30-80, добавок до 3-15. Выбрано.
TARGET_FACES = (30, 81)
MAX_OPS = (3, 16)
TIMEOUT_S = 300


def bases() -> list[str]:
    out = []
    for u in (R / "uids.txt").read_text().split():
        p = R / "parts" / u / LABELS
        if p.exists() and not json.loads(p.read_text())["skip"]:
            out.append(u)
    return out


def _write_step(shape, path: Path) -> None:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP не записан")


def build_one(i: int, pool: list[str]) -> dict:
    from dataset import profile_library, synth_ops as G
    from dataset.fusion_recon import load, tools
    from dataset.recon_opseq import history
    from dataset.step_mesh import read_step
    rng = np.random.default_rng(i)
    base = pool[rng.integers(len(pool))]
    scale = json.loads((R / "scale.json").read_text())[base]
    lib = profile_library.load(S / "profiles.pkl")
    stats = Counter()
    body, ops, _ = G.generate(tools(load(zipfile.ZipFile(ZIP), base), scale), rng,
                              target_faces=int(rng.integers(*TARGET_FACES)), max_ops=int(rng.integers(*MAX_OPS)),
                              library=lib, stats=stats)
    uid = f"{base.rsplit('_', 1)[0]}_syn{i:05d}"
    step = S / "step" / f"{uid}.stp"
    _write_step(body, step)
    final = read_step(step)
    hist = history(final, [(o.name, o.kind, o.boolean, o.tool) for o in ops], G.label(final, ops))
    hist |= {"base": base, "params": [o.params | {"source": o.source} for o in ops]}
    (S / "hist" / f"{uid}.json").write_text(json.dumps(hist, default=float))
    link = RAW / f"{uid}.stp"
    if not link.exists():
        link.symlink_to(step)
    return {"uid": uid, "added": sum(o.source == "synth" for o in ops), "faces": len(hist["face_ops"])}


def _isolated(args: tuple[int, float]) -> tuple[int, dict | None, str | None]:
    i, timeout = args
    try:
        r = subprocess.run([sys.executable, __file__, "--one", str(i)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return i, None, f"таймаут {timeout:.0f} с"
    lines = r.stdout.strip().splitlines()
    if r.returncode or not lines:
        return i, None, (r.stderr or "")[-800:] or f"код {r.returncode}"
    return i, json.loads(lines[-1]), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one is not None:
        pool = (S / "bases.txt").read_text().split()
        print(json.dumps(build_one(args.one, pool), ensure_ascii=False), flush=True)
        return
    for d in (S / "step", S / "hist"):
        d.mkdir(parents=True, exist_ok=True)
    if not (S / "bases.txt").exists():
        (S / "bases.txt").write_text("\n".join(bases()) + "\n")
    done = {p.stem.rsplit("_syn", 1)[1] for p in (S / "hist").glob("*_syn*.json")}
    jobs = [(i, args.timeout) for i in range(args.start, args.start + args.count) if f"{i:05d}" not in done]
    t0, ok, failed, added = time.perf_counter(), 0, Counter(), Counter()
    with ThreadPool(args.workers) as pool, open(S / "built.jsonl", "a") as log:
        for n, (i, info, err) in enumerate(pool.imap_unordered(_isolated, jobs), 1):
            if err:
                failed[err.strip().splitlines()[-1][:80]] += 1
            else:
                ok += 1
                added[min(info["added"], 10)] += 1
                log.write(json.dumps(info, ensure_ascii=False) + "\n")
                log.flush()
            if n % 200 == 0 or n == len(jobs):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(jobs)}  готово {ok}  ошибок {sum(failed.values())}  "
                      f"{dt / n * args.workers:.1f} с/деталь  осталось {dt / n * (len(jobs) - n) / 60:.0f} мин", flush=True)
    print("добавок:", dict(sorted(added.items())))
    print("ошибки:", dict(failed.most_common(10)))


if __name__ == "__main__":
    main()
