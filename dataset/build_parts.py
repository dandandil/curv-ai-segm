"""Собрать детали из STEP своей тесселяцией: data/raw_step -> data/parts_fine.

    python dataset/build_parts.py --workers 20
    python dataset/build_parts.py --limit 50 --compare

Берутся все uid из сплитов или из списка `--split`. Готовые детали (есть brep.json) пропускаются,
поэтому оборванную сборку можно просто перезапустить. `--compare` сверяет
типы, слияние и смежность с корпусом step-dataset в data/parts: номера граней
у обоих — порядок обхода TopAbs_FACE, так что сверка идёт грань в грань.
"""
from __future__ import annotations

import os

# Один процесс на ядро: сторонние пулы потоков внутри воркера только мешают.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset import step_mesh as M  # noqa: E402

OLD_PARTS = D.DATA / "parts"

# Сплайновая деталь в 3.35 млн треугольников: на кэше это десятки ГБ на воркер
# (README, «Тесселяция»).
EXCLUDED = {"104995_a625a7c5_61"}


def compare(uid: str) -> list[str]:
    """Расхождения с корпусом step-dataset; пустой список — совпало."""
    old_path = OLD_PARTS / uid / "brep.json"
    if not old_path.exists():
        return []
    new = json.loads((D.PARTS / uid / "brep.json").read_text())
    old = json.loads(old_path.read_text())
    issues = []
    if [f["type"] for f in new["faces"]] != [f["type"] for f in old["faces"]]:
        issues.append("типы")
    if new["merge_map"] != old["merge_map"]:
        issues.append("слияние")
    pair = lambda b: {(a["a"], a["b"]): a["tangential"] for a in b["adjacency"]}  # noqa: E731
    if pair(new) != pair(old):
        issues.append("смежность")
    return issues


def _one(args: tuple[str, bool, bool]) -> tuple[str, dict | None, str | None]:
    uid, overwrite, check = args
    out = D.PARTS / uid
    try:
        info = None
        if overwrite or not (out / "brep.json").exists():
            info = M.build_part(D.step_path(uid), out)
        if check:
            info = dict(info or {}, issues=compare(uid))
        return uid, info, None
    except Exception:
        return uid, None, traceback.format_exc(limit=3)


def _one_isolated(args: tuple[str, bool, bool, float]) -> tuple[str, dict | None, str | None]:
    """Деталь в отдельном процессе с таймаутом: на части моделей ABC OpenCASCADE падает
    (segfault) или виснет, и пул ждёт потерянную деталь вечно — первый проход 11 600 моделей
    ABC встал на последних сотнях."""
    uid, overwrite, check, timeout = args
    cmd = ([sys.executable, __file__, "--one", uid] + (["--overwrite"] if overwrite else [])
           + (["--compare"] if check else []))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return uid, None, f"таймаут {timeout:.0f} с"
    lines = r.stdout.strip().splitlines()
    if r.returncode or not lines:
        return uid, None, (r.stderr or "")[-1500:] or f"код {r.returncode}"
    return tuple(json.loads(lines[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, help="только первые N деталей, для проверки")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare", action="store_true", help="сверить с корпусом step-dataset")
    ap.add_argument("--split", help="файл со списком uid, порядок сохраняется; по умолчанию все сплиты Fusion")
    ap.add_argument("--timeout", type=float, help="таймаут на деталь, с: каждая — в своём процессе")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(_one((args.one, args.overwrite, args.compare))), flush=True)
        return

    if args.split:
        uids = [u for u in Path(args.split).read_text().split() if u not in EXCLUDED]
    else:
        uids = sorted({u for s in sorted(D.SPLITS.glob("*.txt")) if not s.name.startswith(D.ABC_PREFIX)
                       for u in s.read_text().split()} - EXCLUDED)
    if args.limit:
        uids = uids[:args.limit]
    D.PARTS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    done = failed = open_mesh = 0
    mismatch: dict[str, list[str]] = {}
    if args.timeout:
        # готовые детали при сверке тоже идут в процесс: собирать их он не станет
        jobs = [(u, args.overwrite, args.compare, args.timeout) for u in uids
                if args.overwrite or args.compare or not (D.PARTS / u / "brep.json").exists()]
        pool, run = ThreadPool(args.workers), _one_isolated
    else:
        jobs = [(u, args.overwrite, args.compare) for u in uids]
        pool, run = Pool(args.workers, maxtasksperchild=200), _one
    with pool:
        for uid, info, err in pool.imap_unordered(run, jobs, chunksize=1 if args.timeout else 2):
            if err:
                failed += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                done += 1
                if info and info.get("watertight") is False:
                    open_mesh += 1
                if info and info.get("issues"):
                    mismatch[uid] = info["issues"]
            n = done + failed
            if n % 200 == 0 or n == len(jobs):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(jobs)}  готово {done}  ошибок {failed}  "
                      f"{dt / n:.2f} с/деталь  осталось {(len(jobs) - n) * dt / n / 60:.0f} мин",
                      flush=True)
    print(f"\nитого: {done} собрано, {failed} с ошибкой, незамкнутых среди пересобранных "
          f"{open_mesh}, {(time.perf_counter() - t0) / 60:.1f} мин")
    if args.compare:
        print(f"расхождений с корпусом: {len(mismatch)}")
        for uid, issues in sorted(mismatch.items())[:30]:
            print(f"  {uid}: {', '.join(issues)}")


if __name__ == "__main__":
    main()
