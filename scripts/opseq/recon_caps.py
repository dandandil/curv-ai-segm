"""Торцы выдавливаний реконструкции: лежит ли в плоскости торца плоская грань итогового тела (README, «Опора
выдавливания: приведение к одностороннему»). Пишет recon/caps.jsonl; деталь, у которой хоть одно выдавливание
без грани в обоих торцах, — в recon/no_cap.txt, из обучения исключается.

    python -m scripts.opseq.recon_caps --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# на PC-108-9 DATA смонтирован в /run/media/user
DATA = Path(os.environ.get("OPSEQ_DATA", "/run/media/daniil/DATA/data_kaisarov"))
ZIP = DATA / "fusion360_recon/r1.0.1.zip"
R = DATA / "recon"
# Доля точек грани в плоскости торца, чтобы грань считалась лежащей в ней; доля точек над профилем — «свой».
IN_PLANE = 0.9
OWN = 0.5


def cap_faces(final, ts: list, tol: float) -> list[list[dict]]:
    """Инструмент -> два торца (h0, h1): номера плоских граней тела в плоскости торца, свои и хозяева."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from dataset.fusion_recon import face_samples, _inside
    from dataset.step_mesh import iter_faces
    planes = [(i, p) for i, (f, p) in enumerate(zip(iter_faces(final), face_samples(final, tol, 60)))
              if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane and len(p)]
    out = []
    for t in ts:
        ends = []
        for h in (t.h0, t.h1):
            own, host = [], []
            for i, p in planes:
                q = (p - t.frame[0]) @ t.frame[1:].T
                at = np.abs(q[:, 2] - h) <= tol
                if at.mean() > IN_PLANE:
                    (own if _inside(t, q[at][:20], tol).mean() > OWN else host).append(i)
            ends.append({"h": float(h), "own": own, "host": host})
        out.append(ends)
    return out


def caps(uid: str, scale: float) -> dict:
    from dataset.fusion_recon import load, tools
    from dataset.step_mesh import read_step, bbox_diag
    ts = tools(load(zipfile.ZipFile(ZIP), uid), scale)
    final = read_step(R / "step" / f"{uid}.stp")
    ends = cap_faces(final, ts, 5e-4 * bbox_diag(final))
    return {"uid": uid, "tools": [{"name": t.name, "boolean": t.boolean, "ends": e} for t, e in zip(ts, ends)]}


def _run(a):
    try:
        return caps(*a)
    except Exception as e:  # noqa: BLE001
        return {"uid": a[0], "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    scale = json.loads((R / "scale.json").read_text())
    uids = (R / "uids.txt").read_text().split()
    bad = []
    with Pool(args.workers) as pool, open(R / "caps.jsonl", "w") as f:
        for k, r in enumerate(pool.imap_unordered(_run, [(u, scale[u]) for u in uids], chunksize=4), 1):
            f.write(json.dumps(r) + "\n")
            if "error" in r or any(not any(e["own"] or e["host"] for e in t["ends"]) for t in r["tools"]):
                bad.append(r["uid"])
            if k % 500 == 0:
                print(k, len(uids), "исключено", len(bad), flush=True)
    (R / "no_cap.txt").write_text("\n".join(sorted(bad)) + "\n")
    print("исключено", len(bad), "из", len(uids))


if __name__ == "__main__":
    main()
