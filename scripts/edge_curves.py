"""Рёбра B-Rep корпуса по типу кривой: сколько их, какой длины и между какими поверхностями.

Считается прямо по STEP, до слияния швов: этап 1 тип кривой ребра не сохраняет. Выборка
случайная, по `--n` деталей из всех сплитов корпуса; итог — dataset/data/edge_curves.<корпус>.json,
его читает notebooks/leaks.ipynb.

    python scripts/edge_curves.py --corpus fusion --n 2000 --workers 12
    python scripts/edge_curves.py --corpus abc --n 2000 --workers 12
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Чанк деталей на подпроцесс: OpenCASCADE на части ABC падает или виснет, и терять
# из-за этого стоит один чанк, а не всю выборку.
CHUNK = 25
CHUNK_TIMEOUT = 900


def part_edges(uid: str) -> dict:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    from dataset.cache import step_path
    from dataset.step_mesh import SURFACE_TYPES, bbox_diag, read_step

    shape = read_step(step_path(uid))
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)
    rows = []
    for i in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge_s(emap.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        curve = BRepAdaptor_Curve(edge)
        kind = str(curve.GetType()).split(".")[-1].removeprefix("GeomAbs_").lower()
        faces, surfs = [], []
        for f in emap.FindFromIndex(i):
            if not any(f.IsSame(g) for g in faces):
                faces.append(f)
                s = BRepAdaptor_Surface(TopoDS.Face_s(f)).GetType()
                surfs.append(SURFACE_TYPES.get(s, str(s).split(".")[-1].lower()))
        rows.append([kind, GCPnts_AbscissaPoint.Length_s(curve), sorted(surfs)])
    return {"uid": uid, "diag": bbox_diag(shape), "edges": rows}


def run_chunk(uids: list[str]) -> list[dict]:
    try:
        stdout = subprocess.run([sys.executable, __file__, "--chunk", *uids], capture_output=True, text=True,
                                timeout=CHUNK_TIMEOUT).stdout
    except subprocess.TimeoutExpired as ex:
        # детали, посчитанные до зависшей, не теряются; при таймауте вывод приходит байтами
        stdout = ex.stdout.decode() if isinstance(ex.stdout, bytes) else ex.stdout or ""
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{") and line.endswith("}")]


def corpus_uids(corpus: str) -> list[str]:
    from dataset.cache import SPLITS, step_path
    names = ("abc_train", "abc_val", "abc_test") if corpus == "abc" else ("train", "val", "test")
    uids = [u for n in names for u in (SPLITS / f"{n}.txt").read_text().split()]
    have = []
    for u in uids:
        try:
            if step_path(u).exists():
                have.append(u)
        except FileNotFoundError:
            pass
    return have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("fusion", "abc"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", nargs="+", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.chunk:
        for uid in args.chunk:
            try:
                print(json.dumps(part_edges(uid)), flush=True)
            except Exception as e:
                print(f"[ошибка] {uid}: {e}", file=sys.stderr)
        return

    uids = corpus_uids(args.corpus)
    sample = random.Random(args.seed).sample(uids, min(args.n, len(uids)))
    chunks = [sample[i:i + CHUNK] for i in range(0, len(sample), CHUNK)]
    parts = []
    with ThreadPool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(run_chunk, chunks), 1):
            parts += res
            print(f"{k}/{len(chunks)} чанков, деталей {len(parts)}", flush=True)
    out = ROOT / "dataset" / "data" / f"edge_curves.{args.corpus}.json"
    out.write_text(json.dumps({"corpus": args.corpus, "seed": args.seed, "of": len(uids), "parts": parts}))
    print(f"итого: {len(parts)} из {len(sample)} деталей, рёбер {sum(len(p['edges']) for p in parts)} -> {out}")


if __name__ == "__main__":
    main()
