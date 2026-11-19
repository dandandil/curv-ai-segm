"""Собрать кэш обучения по корпусу.

    python dataset/build_cache.py --variant clean --workers 24
    python dataset/build_cache.py --variant noisy --limit 20 --overwrite

Каждая деталь считается независимо, поэтому пересборка после правки — это
удаление нужных папок в data/cache и повторный запуск: уже готовые детали
пропускаются.

Каждая деталь — в своём процессе: падение в C++ (open3d, OpenCASCADE, pymeshlab) убивает
только его, а пул процессов после гибели сборщика ждал бы потерянную деталь вечно
(README, «Сборка кэша: деталь в своём процессе»).
"""
from __future__ import annotations

import os

# Воркеров столько же, сколько ядер, поэтому каждый обязан быть однопоточным:
# иначе BLAS внутри струи и cKDTree с workers=-1 расходятся на все ядра из
# каждого процесса разом. Ставится до импорта numpy, позже уже не действует;
# процесс детали наследует переменные.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import functools
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402

# Сколько ждать одну деталь, с: деталь скана — минуты, зависшая не держит место сборщика.
TIMEOUT_S = 3600
# Как часто смотреть память процесса детали, с.
POLL_S = 1.0
# Отказ, который ошибкой не считается: остановка по месту на диске.
STOPPED = "остановлена"


def part_seed(uid: str, base: int) -> int:
    """Семя шума детали от её uid: шум не зависит ни от списка, ни от машины, которой деталь
    досталась. Номер в списке совпадал у k-й детали каждого шарда, и при пересоздании списка
    шум деталей менялся."""
    return base + int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16)


def tree_rss_gb(pid: int) -> float:
    """Физическая память процесса и его потомков (gmsh), ГБ."""
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/status") as f:
                total += next((int(line.split()[1]) for line in f if line.startswith("VmRSS:")), 0)
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    stack += [int(c) for c in f.read().split()]
        except OSError:                       # процесс уже вышел
            continue
    return total / 2 ** 20


def build_one(job: tuple[str, str, int, bool], stop: threading.Event, mem_gb: float | None,
              timeout: float) -> tuple[str, str | None]:
    """Деталь в своём процессе. Потолок — физическая память всего дерева процессов, а не
    адресное пространство: потоки open3d резервируют его гигабайтами, и при 3 ГБ настоящей
    памяти сборщик упирался в потолок 12 ГБ (README, «Сборка кэша: деталь в своём процессе»)."""
    uid, variant, seed, overwrite = job
    if stop.is_set():
        return uid, STOPPED
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", uid, "--variant", variant, "--seed", str(seed)]
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd + (["--overwrite"] if overwrite else []), stdout=out, stderr=err,
                                text=True, start_new_session=True)
        start, why = time.monotonic(), None
        while proc.poll() is None:
            if stop.is_set():
                why = STOPPED
            elif mem_gb and tree_rss_gb(proc.pid) > mem_gb:
                why = f"память больше {mem_gb:g} ГБ"
            elif time.monotonic() - start > timeout:
                why = f"таймаут {timeout:.0f} с"
            if why:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                return uid, why
            time.sleep(POLL_S)
        out.seek(0)
        err.seek(0)
        lines = out.read().strip().splitlines()
        if proc.returncode or not lines:
            return uid, f"процесс упал, код {proc.returncode}\n{err.read()[-1500:]}"
        return uid, json.loads(lines[-1])["error"]


def build_in_process(job: tuple[str, str, int, bool]) -> tuple[str, str | None]:
    """Деталь в процессе пула, без своего: у тесселяции CAD (`tile.source: cad`) нет gmsh и open3d, от
    которых изоляция, а запуск процесса на деталь стоил 2.3 с при сотых долях на саму деталь [проба
    2026-11-01]."""
    uid, variant, seed, overwrite = job
    try:
        D.build_part(uid, variant=variant, seed=seed, overwrite=overwrite)
        return uid, None
    except Exception:
        return uid, traceback.format_exc(limit=3)


def _one(uid: str, variant: str, seed: int, overwrite: bool) -> None:
    """Режим процесса детали: собрать её и напечатать ошибку последней строкой."""
    try:
        D.build_part(uid, variant=variant, seed=seed, overwrite=overwrite)
        error = None
    except Exception:
        error = traceback.format_exc(limit=3)
    print(json.dumps({"error": error}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="clean", type=D.check_variant,
                    help=f"{', '.join(D.VARIANTS)} или {D.FIXED_PREFIX}<σ размытия, мм> для оценки по уровням шума")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="только первые N деталей, для проверки")
    ap.add_argument("--seed", type=int, default=0, help="семя шума; у детали к нему прибавляется хеш её uid")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split", help="файл со списком uid, по умолчанию все детали")
    ap.add_argument("--shard", type=int, default=0, help="номер машины при сборке на нескольких")
    ap.add_argument("--shards", type=int, default=1, help="сколько машин собирают вместе")
    ap.add_argument("--mem-limit-gb", type=float,
                    help="потолок физической памяти на деталь, ГБ: раздувшаяся деталь снимается и пропускается")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="сколько ждать одну деталь, с")
    ap.add_argument("--min-free-gb", type=float, default=0.0,
                    help="остановиться, когда свободного места на диске кэша станет меньше")
    ap.add_argument("--reserve-mb", type=float, default=0.0,
                    help="сколько места держать сверх порога на каждую деталь с кэшем, но без колец")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        _one(args.one, args.variant, args.seed, args.overwrite)
        return

    if args.split:
        uids = Path(args.split).read_text().split()
    else:
        uids = sorted(p.name for p in D.PARTS.iterdir() if (p / "mesh.ply").exists())
    if args.limit:
        uids = uids[:args.limit]
    jobs = [(u, args.variant, part_seed(u, args.seed), args.overwrite) for u in uids][args.shard::args.shards]

    t0 = time.perf_counter()
    done = failed = 0

    scope = D.part_dir("_", args.variant).name

    def ring_debt_gb() -> float:
        # Кольца пишутся позже (build_rings.py), поэтому их место держится заранее на каждую
        # деталь, у которой кэш уже есть, а колец ещё нет — по диску, а не по этому запуску.
        if not args.reserve_mb:
            return 0.0
        built = sum(1 for _ in D.CACHE.glob(f"*/{scope}/meta.json"))
        ringed = sum(1 for _ in D.CACHE.glob(f"*/{scope}/{D.RINGS_FILE}"))
        return max(built - ringed, 0) * args.reserve_mb / 1024

    def disk_full() -> bool:
        # Пользователь: на каждой машине не меньше 10 ГБ свободными (2026-10-09).
        return shutil.disk_usage(D.CACHE).free / 2 ** 30 - ring_debt_gb() < args.min_free_gb

    stop = threading.Event()
    if D.CAD_SOURCE:
        pool, run = Pool(args.workers, maxtasksperchild=500), build_in_process
    else:
        pool = ThreadPool(args.workers)
        run = functools.partial(build_one, stop=stop, mem_gb=args.mem_limit_gb, timeout=args.timeout)
    with pool:
        for uid, err in pool.imap_unordered(run, jobs, chunksize=1 if not D.CAD_SOURCE else 8):
            if err == STOPPED:
                continue
            if err:
                failed += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                done += 1
            if args.min_free_gb and not stop.is_set() and disk_full():
                print(f"=== место кончилось: свободно {shutil.disk_usage(D.CACHE).free / 2 ** 30:.1f} ГБ, "
                      f"под кольца держится {ring_debt_gb():.1f} ГБ — остановка", flush=True)
                stop.set()
            n = done + failed
            if n % 50 == 0 or n == len(jobs):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(jobs)}  готово {done}  ошибок {failed}  "
                      f"{dt / n:.1f} с/деталь  осталось {(len(jobs) - n) * dt / n / 60:.0f} мин",
                      flush=True)
    print(f"\nитого: {done} собрано, {failed} с ошибкой, "
          f"{(time.perf_counter() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
