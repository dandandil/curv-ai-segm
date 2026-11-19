"""Детали для сегментации CAD-тесселяций: STEP ABC -> тесселяция с допусками экспорта STL -> parts_cad.

    python dataset/build_parts_cad.py --split runs/cad_tess/probe_uids.txt --out <папка> --workers 20 --timeout 600

Тесселяция идёт в сеть как есть (README, «Сегментация CAD-тесселяций»): только склейка совпадающих вершин и
выброс треугольников нулевой площади после неё, как в `step_mesh.tessellate`. Допуски у каждой детали свои, как
у разных настроек экспорта STL, зерно — от uid. Правило узких граней не применяется, поэтому полуширина граней
не считается (решено пользователем 2026-11-01).

    mesh.ply      вершины float32, как в STL
    face_id.npy   номер грани на треугольник после слияния швов
    brep.json     как у build_parts.py, без halfwidth_mm, narrow и label_map
    tess.json     допуски тесселяции детали
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
import traceback
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset import step_mesh as M  # noqa: E402
from src.config import CFG  # noqa: E402

PARTS_CAD = D.DATA / "parts_cad"
# Пресеты экспорта STL: Onshape Coarse / Medium / Fine — 0.24 / 0.12 / 0.06 мм и 12.5 / 6.25 / 2.5°,
# SolidWorks Fine ~0.05 мм и 10° (README, «Сегментация CAD-тесселяций»). Линейный лог-равномерно, угловой
# равномерно; решено пользователем 2026-11-01.
LINEAR_MM = (0.01, 0.25)
ANGULAR_DEG = (2.5, 15.0)


def tolerances(uid: str) -> tuple[float, float]:
    """Линейный допуск, мм, и угловой, градусы, — у детали всегда одни и те же."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little"))
    linear = float(np.exp(rng.uniform(np.log(LINEAR_MM[0]), np.log(LINEAR_MM[1]))))
    return linear, float(rng.uniform(*ANGULAR_DEG))


def build_part(uid: str, out_dir: Path) -> dict:
    """Пишет во временную папку и переименовывает в конце, как `step_mesh.build_part`."""
    seconds = {}
    t0 = time.perf_counter()
    shape = M.read_step(D.step_path(uid))
    diag = M.bbox_diag(shape)
    linear, angular = tolerances(uid)
    # MinSize 0 — значение OCC по умолчанию (0.1 линейного допуска).
    mesh, fid, counts = M.tessellate(shape, diag, linear / diag, np.radians(angular), 0.0)
    seconds["tessellate"] = time.perf_counter() - t0

    t = CFG.tessellation
    t0 = time.perf_counter()
    brep = M.build_brep(shape, diag, counts, t.tangency_samples, t.tangency_deg)
    merge_map = M.merge_faces(brep, t.merge_angle_deg, t.merge_length_rel, t.merge_radius_rel, t.seam_curv_jump)
    brep["merge_map"] = merge_map.tolist()
    seconds["brep"] = time.perf_counter() - t0
    labels = merge_map[fid]
    n_merged = int(merge_map.max()) + 1

    mesh.vertices = mesh.vertices.astype(np.float32).astype(np.float64)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mesh.export(str(tmp / "mesh.ply"), vertex_normal=False)
    np.save(tmp / "face_id.npy", labels)
    (tmp / "brep.json").write_text(json.dumps(brep, indent=1))
    (tmp / "tess.json").write_text(json.dumps({"linear_mm": linear, "angular_deg": angular}))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)

    edges = mesh.edges_unique_length
    tri_per_face = np.bincount(labels, minlength=n_merged)
    return {"n_triangles": int(len(mesh.faces)), "n_faces": brep["n_faces"], "n_merged": n_merged,
            "watertight": bool(mesh.is_watertight), "diag_mm": diag, "linear_mm": linear, "angular_deg": angular,
            "edge_mm": [float(q) for q in np.percentile(edges, [0, 1, 50, 99, 100])],
            "faces_le4_tri": int((tri_per_face <= 4).sum()),
            "bytes": sum(f.stat().st_size for f in out_dir.iterdir()), "seconds": seconds}


def _one(uid: str, out: Path) -> tuple[str, dict | None, str | None]:
    try:
        return uid, build_part(uid, out / uid), None
    except Exception:
        return uid, None, traceback.format_exc(limit=3)


def _batch(uids: list[str], out: Path, timeout: float) -> None:
    """Детали подряд в одном процессе: импорт стоит 2.6 с, деталь в среднем 0.42 с [проба]. Зависшую деталь
    убивает SIGALRM вместе с процессом — обработчика нет, поэтому он срабатывает и внутри кода OCC."""
    for uid in uids:
        signal.alarm(int(timeout))
        print(json.dumps(_one(uid, out)), flush=True)
        signal.alarm(0)


def _isolated(args: tuple[list[str], Path, float]) -> list[tuple[str, dict | None, str | None]]:
    """Пачка в своём процессе: OpenCASCADE на части моделей ABC падает или виснет (build_parts.py). Если процесс
    умер, недоделанные детали пачки идут по одной, и сбой остаётся на одной детали."""
    uids, out, timeout = args
    cmd = [sys.executable, __file__, "--one", *uids, "--out", str(out), "--timeout", str(timeout)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    results = [tuple(json.loads(line)) for line in r.stdout.splitlines() if line.startswith("[")]
    rest = uids[len(results):]
    if not rest:
        return results
    if len(uids) == 1:
        err = f"таймаут {timeout:.0f} с" if r.returncode == -signal.SIGALRM else \
            (r.stderr or "")[-1500:] or f"код {r.returncode}"
        return [(uids[0], None, err)]
    return results + [res for u in rest for res in _isolated(([u], out, timeout))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="файл со списком uid")
    ap.add_argument("--out", type=Path, default=PARTS_CAD)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=600.0, help="на деталь, с")
    ap.add_argument("--batch", type=int, default=50, help="деталей на процесс")
    ap.add_argument("--log", type=Path, help="jsonl: строка на деталь")
    ap.add_argument("--one", nargs="+", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        _batch(args.one, args.out, args.timeout)
        return

    # готовые детали пропускаются: оборванную сборку можно перезапустить
    uids = [u for u in Path(args.split).read_text().split() if not (args.out / u / "brep.json").exists()]
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a") if args.log else None
    t0 = time.perf_counter()
    done = failed = 0
    jobs = [(uids[i:i + args.batch], args.out, args.timeout) for i in range(0, len(uids), args.batch)]
    with ThreadPool(args.workers) as pool:
        for uid, info, err in (res for batch in pool.imap_unordered(_isolated, jobs) for res in batch):
            if err:
                failed += 1
                print(f"[ошибка] {uid}\n{err}", file=sys.stderr, flush=True)
            else:
                done += 1
            if log:
                log.write(json.dumps({"uid": uid, "info": info, "error": err and err[-300:]}) + "\n")
                log.flush()
            n = done + failed
            if n % 200 == 0 or n == len(uids):
                dt = time.perf_counter() - t0
                print(f"{n}/{len(uids)}  готово {done}  ошибок {failed}  {dt / n:.2f} с/деталь  "
                      f"осталось {(len(uids) - n) * dt / n / 60:.0f} мин", flush=True)
    print(f"\nитого: {done} собрано, {failed} с ошибкой, {(time.perf_counter() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
