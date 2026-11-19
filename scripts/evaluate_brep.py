"""Сборка B-Rep по корпусу: доля деталей, собранных в правильное тело.

    CURV_CONFIG=configs/aug60.yaml python scripts/evaluate_brep.py --split test --input truth

Деталь собрана, если оболочка замкнулась в тело, BRepCheck его принимает, объём
совпадает с объёмом меша и поверхность тела не отходит от меша (README,
«Реконструкция B-Rep», «Оценка по корпусу»). Каждая деталь — в своём процессе с
таймаутом: OCC на вырожденной геометрии может уйти в бесконечный цикл.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Форма сверяется с истинной CAD (тесселяция STEP): p95 расстояний в обе
# стороны долей √площади детали. Порог 0.005 (~0.12 мм на медианной детали) —
# p90-p95 той же величины у самого рабочего меша [измерено, 150 деталей test:
# 0.0044 и 0.0052], то есть тело не хуже входа почти на всех деталях.
DEVIATION_Q = 0.95
DEVIATION_REL = 0.005
# Объём сверяется и с рабочим мешем, и с CAD, засчитывается лучшее: у самого
# входа объём расходится с CAD на 0.9% по медиане и 5.7% на p90 (размытие на
# тонких деталях) [измерено, 150 деталей test]. Тело, повторившее вход, и тело,
# повторившее CAD при раздутом входе, оба верны; потерянный или лишний кусок
# расходится с обоими. Порог ловит именно куски.
VOLUME_REL = 0.02
SAMPLES = 5000
# Касание точное, если расхождение условия касания меньше этого, мм: совместная
# подгонка выполняет условия до 1e-9 в нормированной системе, то есть до
# 1e-8 мм на детали в десятки миллиметров.
TANGENT_EXACT_MM = 1e-5


def reconstruct(mesh, labels, votes, cfg, builder: str):
    """Кластеры -> форма и сводка; те же шаги, что в ноутбуке, раздел 9."""
    from src import arrangement as A
    from src import brep as B
    from src import constraints as CS
    from src import fitting as FT
    from src import patches as P
    from src import reconstruct as R
    from src import topology as TP

    mesh, labels, votes = TP.main_parts(mesh, labels, votes)
    v, f, adj = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces), np.asarray(mesh.face_adjacency)
    if builder == "cells":
        cfg = dataclasses.replace(cfg, min_triangles=A.MIN_TRIANGLES)
    seg, surfaces, info = FT.fit_clusters(v, f, adj, labels, votes, cfg)
    summary = dict(clusters=info["clusters"], merges=info["merges"], unfitted=len(info["unfitted"]),
                   approximate=len(info["approximate"]), surfaces=len(surfaces))
    # соотношения (касание, соосность, параллельность) и совместная подгонка при
    # точных условиях — у обеих сборок: без неё касание выходит лишь приближённым
    # [измерено, 3 детали теста по разметке, точных касаний из истинных:
    # 0/10, 0/18, 0/8 без неё, 10/10, 16/18, 8/8 с ней]
    seg, bounds, chainless = TP.drop_chainless(mesh, seg, cfg.min_chain_edges)
    surfaces = {l: s for l, s in surfaces.items() if l not in chainless}
    summary["без рёбер, отдано соседу"] = len(chainless)
    samples = FT.samples(seg, adj, v, f, surfaces, cfg)
    chained = {ch.pair for ch in bounds.chains}
    touching = TP.touching(mesh, seg)
    surfaces, relations, dropped = CS.settle(surfaces, chained | touching, samples,
                                             info["tolerance"], cfg, short=touching - chained)
    summary.update(relations=len(relations), dropped=len(dropped))
    if builder == "cells":
        shape, report = R.solid(mesh, seg, surfaces, relations, bounds, cfg)
    elif builder == "patches":
        shape, report = P.build(mesh, seg, surfaces, relations, info["tolerance"])
    else:
        shape, report = B.build(mesh, seg, surfaces, relations, bounds, cfg)
    return shape, {**report, **summary}, seg, surfaces


def tangency(uid: str, truth: np.ndarray, seg: np.ndarray, surfaces: dict) -> dict:
    """Сколько истинно касательных пар граней (`brep.json`) вышли точно
    касательными. Грань -> кластер, покрывающий большую часть её
    треугольников; пара считается, если оба кластера разные, с аналитическими
    поверхностями и `constraints` такую пару вообще распознаёт (плоскость,
    цилиндр, тор, конус, сфера в своих сочетаниях)."""
    from src import constraints as CS

    brep = json.loads(part_file(uid, "brep.json").read_text())
    merge_map = brep["merge_map"] or list(range(brep["n_faces"]))
    cluster = {}
    for t in np.unique(truth):
        cluster[int(t)] = int(np.bincount(seg[truth == t]).argmax())
    total = exact = 0
    for adj in brep["adjacency"]:
        if not adj["tangential"]:
            continue
        a, b = (cluster.get(merge_map[adj[k]]) for k in ("a", "b"))
        if a is None or b is None or a == b or a not in surfaces or b not in surfaces:
            continue
        if not {surfaces[a].kind, surfaces[b].kind} <= set(CS._RANK) or {surfaces[a].kind, surfaces[b].kind} == {"plane"}:
            continue
        loose = CS.detect(surfaces, [(a, b)], 5.0, 0.2, floor=0.0)
        if not any(r.name in CS.TANGENT for r in loose):
            continue                # и приближённо не касательны: пару подгонка не узнала
        total += 1
        tight = CS.detect(surfaces, [(a, b)], 1e-4, 1e-9, floor=TANGENT_EXACT_MM)
        exact += any(r.name in CS.TANGENT for r in tight)
    return {"tangent_pairs": total, "tangent_exact": exact}


def fidelity(shape, mesh, prefix: str = "") -> dict:
    """Объём и расхождение поверхностей тела и меша в обе стороны."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    import trimesh
    from src import brep as B

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    scale = float(np.sqrt(mesh.area))
    result = B.tessellate(shape, 1e-3 * scale)
    if len(result.faces) == 0:
        return {f"{prefix}volume_err": float("inf"), f"{prefix}dev_mesh": float("inf"),
                f"{prefix}dev_shape": float("inf")}
    rng = np.random.default_rng(0)
    # расстояние до поверхности, а не до точек второй выборки: у той пол —
    # шаг выборки, ~0.007 √площади при 20 тысячах точек
    da = trimesh.proximity.closest_point(result, mesh.sample(SAMPLES, seed=rng))[1]
    db = trimesh.proximity.closest_point(mesh, result.sample(SAMPLES, seed=rng))[1]
    return {f"{prefix}volume_err": abs(abs(props.Mass()) - abs(mesh.volume)) / abs(mesh.volume),
            f"{prefix}dev_mesh": float(np.quantile(da, DEVIATION_Q)) / scale,
            f"{prefix}dev_shape": float(np.quantile(db, DEVIATION_Q)) / scale}


PARTS = ROOT / "dataset" / "data" / "parts"
PARTS_FINE = ROOT / "dataset" / "data" / "parts_fine"


def part_file(uid: str, name: str) -> Path:
    """Файл детали корпуса step-dataset; у деталей ABC его нет — тогда из этапа 1 (`parts_fine`,
    там тесселяция CAD — `mesh.ply`)."""
    path = PARTS / uid / name
    if path.exists():
        return path
    return PARTS_FINE / uid / ("mesh.ply" if name == "mesh_full.ply" else name)


def run_part(uid: str, source: str, variant: str, builder: str, queue) -> None:
    import trimesh
    from dataset.cache import load_part
    from src.config import CFG
    from src.data import face_types

    import torch

    # процессов по числу ядер, и у каждого свой пул потоков torch на все ядра
    # перегружал машину вдесятеро [load average 66 на 24 ядрах]
    torch.set_num_threads(1)
    t = time.perf_counter()
    work = load_part(uid, variant)["work"]
    mesh = trimesh.Trimesh(work["vertices"], work["faces"], process=False)
    if source == "truth":
        labels = work["labels"].astype(np.int64)
        votes = face_types(uid, int(labels.max()) + 1)[labels]
    else:
        seg = np.load(source / f"{uid}.npz")
        labels, votes = seg["labels"], seg["types"]
    out = {"uid": uid}
    try:
        shape, report, seg, surfaces = reconstruct(mesh, labels, votes, CFG.brep, builder)
        out.update(report)
        from src import topology as TP
        # разметка — на том же меше без осколков, что и сборка
        out.update(tangency(uid, TP.main_parts(mesh, work["labels"].astype(np.int64))[1], seg, surfaces))
        if shape is not None and report.get("тело"):
            out.update(fidelity(shape, mesh))
            # истинная геометрия — тесселяция STEP детали: с ней сравнивается
            # форма без шума рабочего меша
            out.update(fidelity(shape, trimesh.load(part_file(uid, "mesh_full.ply"), process=False), "cad_"))
            out["true_faces"] = int(work["labels"].max() + 1)
    except Exception as e:  # noqa: BLE001 — одна деталь не роняет оценку
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = time.perf_counter() - t
    queue.put(out)


def passed(r: dict) -> bool:
    return bool(r.get("тело") and r.get("валидна") and r.get("тел", 1) == 1 and min(r.get("volume_err", 1), r.get("cad_volume_err", 1)) <= VOLUME_REL
                and max(r.get("cad_dev_mesh", 1), r.get("cad_dev_shape", 1)) <= DEVIATION_REL)


def main() -> None:
    from src.data import split_uids

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--input", default="truth", help="truth или каталог с нарезкой сети <uid>.npz")
    ap.add_argument("--builder", default="edges", choices=("cells", "edges", "patches"),
                    help="сборка разбиением (src/arrangement.py) или по рёбрам (src/brep.py)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--uids", default=None, help="файл со списком деталей вместо всего сплита")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="дописать в --out, пропустив готовые детали")
    args = ap.parse_args()

    from dataset.cache import part_dir
    # кэш поделён между машинами: здесь только детали, чей рабочий меш лежит локально
    uids = [u for u in split_uids(args.split, args.variant)
            if (part_dir(u, args.variant) / "work.npz").exists()][:args.limit]
    if args.uids:
        wanted = set(Path(args.uids).read_text().split())
        uids = [u for u in uids if u in wanted]
    source = args.input if args.input == "truth" else Path(args.input)
    if source != "truth":
        uids = [u for u in uids if (source / f"{u}.npz").exists()]
    done = []
    if args.resume and Path(args.out).exists():
        done = [json.loads(line) for line in open(args.out)]
        uids = [u for u in uids if u not in {r["uid"] for r in done}]
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    running, results, pending = {}, done, list(uids)
    with open(args.out, "a" if args.resume else "w") as log:
        while pending or running:
            while pending and len(running) < args.workers:
                uid = pending.pop(0)
                p = ctx.Process(target=run_part, args=(uid, source, args.variant, args.builder, queue))
                p.start()
                running[uid] = (p, time.time())
            try:
                r = queue.get(timeout=1)
            except Exception:  # noqa: BLE001 — пустая очередь
                r = None
            now = time.time()
            for uid, (p, start) in list(running.items()):
                if r is not None and uid == r["uid"]:
                    continue                # результат уже на руках, сколько бы он ни шёл
                if p.exitcode == 0:
                    continue                # закончил, результат ещё в очереди
                if p.exitcode is not None:
                    r_dead = {"uid": uid, "error": f"процесс упал, код {p.exitcode}"}
                    results.append(r_dead)
                    log.write(json.dumps(r_dead, ensure_ascii=False) + "\n")
                    del running[uid]
                elif now - start > args.timeout:
                    p.kill()
                    p.join()
                    r_dead = {"uid": uid, "error": "таймаут"}
                    results.append(r_dead)
                    log.write(json.dumps(r_dead, ensure_ascii=False) + "\n")
                    del running[uid]
            # результат, успевший в очередь перед снятием по таймауту, уже записан как таймаут
            if r is not None and r["uid"] in running:
                r["ok"] = passed(r)
                results.append(r)
                log.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
                log.flush()
                p, _ = running.pop(r["uid"])
                p.join()
                print(f"{len(results)}/{len(uids)} {r['uid']} ok={r['ok']} тело={r.get('тело')} "
                      f"валидна={r.get('валидна')} {r.get('error', '')}", flush=True)
    summary(results)


def summary(results: list[dict]) -> None:
    n = len(results)
    c = Counter()
    for r in results:
        c["тело"] += bool(r.get("тело"))
        c["тело валидно"] += bool(r.get("тело") and r.get("валидна"))
        c["собрана"] += passed(r)
        c["ошибка"] += "error" in r
    c["одно тело, валидно"] = sum(bool(r.get("тело") and r.get("валидна") and r.get("тел", 1) == 1) for r in results)
    for k, v in c.items():
        print(f"{k}: {v}/{n} = {v / max(n, 1):.1%}")


if __name__ == "__main__":
    main()
