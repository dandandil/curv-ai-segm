"""Сканы тел с историей для opseq: деталь этапа 1 -> виртуальный сканер -> упрощение до ребра tile -> метки граней.
С `--substrate` — вместо сканера сетка gmsh прямо по STEP с ребром SUBSTRATE_EDGE_MM и прежняя модель шума
(`cache.apply_noise`), метки — грань STEP у треугольника gmsh (README, «Базовая линия: шум на подложке из STEP»).

    CURV_PARTS=<каталог деталей> CURV_CONFIG=configs/tile.yaml \
        python scripts/opseq/build_scans.py --split <uids> --out <каталог> --workers 20

Пишет `<out>/<uid>.npz` (вершины, треугольники, номер грани B-Rep на треугольник — до `label_map`,
правило узких граней применяется при загрузке) и `<out>/<uid>.json` (параметры скана, время).
Готовые тела пропускаются; тело — в своём процессе с потолком памяти и таймаутом, как
`dataset/build_cache.py`. Семя — как у кэша: `part_seed(uid, seed)`, поток `SCAN_STREAM`.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dataset import build_cache as BC  # noqa: E402
from dataset import cache as D  # noqa: E402

TIMEOUT_S = 900.0
# Подложка из STEP (решено пользователем 2026-10-28): одна сетка gmsh без тесселяции и ремеша, ребро 0.1 мм
# (0.02 на всех телах — ~3.6 ТБ), размытие σ 0.06-0.15 мм вместо 0.06-0.30 у сегментатора: на 0.3
# грани полушириной до 0.75 мм терялись в 61-66% (README, «Подложка из CAD»).
SUBSTRATE_EDGE_MM = 0.1
SUBSTRATE_BLUR_MM = (0.06, 0.15)


def build(uid: str, out: Path, seed: int) -> None:
    import trimesh

    from dataset import remesh as R
    from dataset import scan as SC
    from src.config import CFG

    part = D.PARTS / uid
    src = trimesh.load(str(part / "mesh.ply"), process=False)
    face_id = np.load(part / "face_id.npy")
    diag = float(np.linalg.norm(src.extents))
    if diag > D.MAX_DIAG_MM:
        raise RuntimeError(f"деталь {diag:.0f} мм по диагонали больше {D.MAX_DIAG_MM:.0f}")
    t0 = time.time()
    rng = np.random.default_rng([BC.part_seed(uid, seed), D.SCAN_STREAM])
    scanned, p = SC.scan(src, SC.sample_params(rng), rng)
    t1 = time.time()
    edge_mm = CFG.tile.edge_mm
    work = R.decimate(scanned, int(scanned.area / (np.sqrt(3) / 4 * edge_mm ** 2)))
    t2 = time.time()
    labels = R.transfer_labels(src, face_id, work)
    t3 = time.time()
    n_faces = int(face_id.max()) + 1
    present = np.unique(labels[labels >= 0])
    tmp = out / f"{uid}.tmp.npz"
    np.savez_compressed(tmp, vertices=np.asarray(work.vertices, np.float32),
                        faces=np.asarray(work.faces, np.int32), labels=labels.astype(np.int32))
    meta = {"uid": uid, "seed": seed, "tessellation": D.TESSELLATION, "edge_mm": edge_mm,
            "noise": {"model": "scan", **asdict(p)}, "diag_mm": diag,
            "scan_faces": int(len(scanned.faces)), "work_faces": int(len(work.faces)),
            "median_edge_mm": float(np.median(work.edges_unique_length)),
            "brep_faces": n_faces, "lost_faces": sorted(set(range(n_faces)) - set(present.tolist())),
            "unlabeled_frac": float(np.mean(labels < 0)),
            "time_s": {"scan": t1 - t0, "decimate": t2 - t1, "labels": t3 - t2}}
    os.replace(tmp, out / f"{uid}.npz")
    (out / f"{uid}.json").write_text(json.dumps(meta))      # последним: по нему тело считается готовым


def build_substrate(uid: str, out: Path, seed: int) -> None:
    import trimesh

    from dataset import step_substrate as S

    part = D.PARTS / uid
    src = trimesh.load(str(part / "mesh.ply"), process=False)
    diag = float(np.linalg.norm(src.extents))
    if diag > D.MAX_DIAG_MM:
        raise RuntimeError(f"деталь {diag:.0f} мм по диагонали больше {D.MAX_DIAG_MM:.0f}")
    brep = json.loads((part / "brep.json").read_text())
    merge = np.asarray(brep["merge_map"]) if brep["merge_map"] is not None else None
    t0 = time.time()
    sub, owner = S.substrate(D.step_path(uid), SUBSTRATE_EDGE_MM, src, with_faces=True)
    t1 = time.time()
    s = BC.part_seed(uid, seed)
    params = D.noise_params(s, blur_sigma_mm=SUBSTRATE_BLUR_MM)
    noisy = D.apply_noise(sub, s, params)
    t2 = time.time()
    labels = merge[owner] if merge is not None else owner
    n_faces = int(labels.max()) + 1 if merge is None else int(merge.max()) + 1
    tmp = out / f"{uid}.tmp.npz"
    np.savez_compressed(tmp, vertices=np.asarray(noisy.vertices, np.float32),
                        faces=np.asarray(noisy.faces, np.int32), labels=labels.astype(np.int32))
    meta = {"uid": uid, "seed": seed, "source": "step-gmsh", "edge_mm": SUBSTRATE_EDGE_MM,
            "noise": {"model": "substrate", **params}, "diag_mm": diag, "work_faces": int(len(noisy.faces)),
            "median_edge_mm": float(np.median(noisy.edges_unique_length)), "brep_faces": n_faces,
            "lost_faces": sorted(set(range(n_faces)) - set(np.unique(labels).tolist())), "unlabeled_frac": 0.0,
            "time_s": {"mesh": t1 - t0, "noise": t2 - t1}}
    os.replace(tmp, out / f"{uid}.npz")
    (out / f"{uid}.json").write_text(json.dumps(meta))


def run_one(uid: str, out: Path, seed: int, mem_gb: float, timeout: float, substrate: bool = False) -> tuple[str, str | None]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", uid, "--out", str(out), "--seed", str(seed)]
    cmd += ["--substrate"] if substrate else []
    with tempfile.TemporaryFile("w+") as so, tempfile.TemporaryFile("w+") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, text=True, start_new_session=True)
        start, why = time.monotonic(), None
        while proc.poll() is None:
            if BC.tree_rss_gb(proc.pid) > mem_gb:
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
            time.sleep(BC.POLL_S)
        so.seek(0)
        se.seek(0)
        lines = so.read().strip().splitlines()
        if proc.returncode or not lines:
            return uid, f"процесс упал, код {proc.returncode}\n{se.read()[-1500:]}"
        return uid, json.loads(lines[-1])["error"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="файл со списком uid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mem-limit-gb", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="остановиться, когда на диске вывода меньше")
    ap.add_argument("--substrate", action="store_true", help="сетка gmsh по STEP и прежний шум вместо сканера")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = Path(args.out)
    if args.one:
        try:
            (build_substrate if args.substrate else build)(args.one, out, args.seed)
            error = None
        except Exception:
            error = traceback.format_exc(limit=3)
        print(json.dumps({"error": error}), flush=True)
        return

    if not args.split:
        ap.error("нужен --split")
    out.mkdir(parents=True, exist_ok=True)
    uids = [u for u in Path(args.split).read_text().split()
            if (D.PARTS / u / "brep.json").exists() and not (out / f"{u}.json").exists()]
    if args.limit:
        uids = uids[:args.limit]
    stop = threading.Event()

    def job(u):
        if stop.is_set():
            return u, BC.STOPPED
        if os.statvfs(out).f_bavail * os.statvfs(out).f_frsize / 2 ** 30 < args.min_free_gb:
            stop.set()
            return u, BC.STOPPED
        return run_one(u, out, args.seed, args.mem_limit_gb, args.timeout, args.substrate)

    t0, done, failed = time.perf_counter(), 0, 0
    with ThreadPool(args.workers) as pool:
        for n, (uid, err) in enumerate(pool.imap_unordered(job, uids), 1):
            if err == BC.STOPPED:
                continue
            if err:
                failed += 1
                print(f"[ошибка] {uid}: {err}", file=sys.stderr, flush=True)
            else:
                done += 1
            if n % 500 == 0 or n == len(uids):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(uids)}  готово {done}  ошибок {failed}  осталось {(len(uids) - n) * dt / n / 60:.0f} мин",
                      flush=True)
    if stop.is_set():
        print(f"=== место кончилось: на диске вывода меньше {args.min_free_gb:g} ГБ", flush=True)
    print(f"итого: {done} собрано, {failed} с ошибкой, {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
