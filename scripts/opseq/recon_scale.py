"""Масштаб деталей реконструкции к диагонали DIAG_MM (решено пользователем 2026-10-28): у нынешнего датасета
Fusion 20-30 мм, у реконструкции — настоящие размеры (медиана 84 мм, p90 523 мм), и при ребре подложки
0.1 мм деталь в 500 мм дала бы ~60 млн треугольников. Диагональ — равномерно из DIAG_MM по семени проекта.
Переписывает <out>/step/<проект>.stp, множитель — в <out>/scale.json (тела инструментов из JSON — с ним же).

    python scripts/opseq/recon_scale.py --out <каталог> --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DIAG_MM = (20.0, 30.0)


def scale_one(args: tuple[str, str]) -> tuple[str, float]:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from dataset.step_mesh import bbox_diag, read_step
    uid, out = args
    path = Path(out) / "step" / f"{uid}.stp"
    shape = read_step(path)
    seed = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
    target = float(np.random.default_rng(seed).uniform(*DIAG_MM))
    s = target / bbox_diag(shape)
    tr = gp_Trsf()
    tr.SetScale(gp_Pnt(0, 0, 0), s)
    scaled = BRepBuilderAPI_Transform(shape, tr, True).Shape()
    w = STEPControl_Writer()
    w.Transfer(scaled, STEPControl_AsIs)
    tmp = path.with_suffix(".tmp.stp")
    if w.Write(str(tmp)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP не записан: {uid}")
    tmp.replace(path)
    return uid, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    done = json.loads((out / "scale.json").read_text()) if (out / "scale.json").exists() else {}
    uids = [u for u in (out / "uids.txt").read_text().split() if u not in done]
    with Pool(args.workers) as pool:
        for k, (uid, s) in enumerate(pool.imap_unordered(scale_one, [(u, str(out)) for u in uids], chunksize=8), 1):
            done[uid] = s
            if k % 500 == 0:
                (out / "scale.json").write_text(json.dumps(done))
                print(f"{k}/{len(uids)}", flush=True)
    (out / "scale.json").write_text(json.dumps(done))
    print("готово", len(done))


if __name__ == "__main__":
    main()
