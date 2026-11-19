"""Отбор проектов реконструкции для opseq: однотельные, тела выдавливаний из эскизов сходятся с итоговым STEP
(симметричная разность объёмов меньше VOLUME_TOL итогового). STEP -> <out>/step/<проект>.stp, ссылки в
dataset/data/raw_step, список — <out>/uids.txt, причины отказа — <out>/prep.jsonl.

    python scripts/opseq/recon_prep.py --zip <r1.0.1.zip> --out <каталог> --workers 4
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.fusion_recon import ROOT as ZROOT, ReconError, load, tools  # noqa: E402

# Тела выдавливаний считаются совпавшими с итоговым при симметричной разности меньше этой доли объёма. Выбрано.
VOLUME_TOL = 1e-3
_zip, _out = None, None


def _init(zip_path: str, out: str) -> None:
    global _zip, _out
    _zip, _out = zipfile.ZipFile(zip_path), Path(out)


def _volume(s) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, p)
    return abs(p.Mass())


def check(design: str) -> dict:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from dataset.step_mesh import read_step
    j = load(_zip, design)
    if j["properties"].get("body_count", 1) != 1:
        return {"uid": design, "skip": "тел больше одного"}
    try:
        ts = tools(j)
    except ReconError as e:
        return {"uid": design, "skip": str(e)}
    except Exception as e:  # noqa: BLE001 — OCC
        return {"uid": design, "skip": f"OCC: {type(e).__name__}"}
    body = None
    for t in ts:
        body = t.shape if body is None else \
            {"add": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "intersect": BRepAlgoAPI_Common}[t.boolean](body, t.shape).Shape()
    step = _out / "step" / f"{design}.stp"
    step.write_bytes(_zip.read(f"{ZROOT}{design}.step"))
    final = read_step(step)
    v = _volume(final)
    diff = (_volume(BRepAlgoAPI_Cut(final, body).Shape()) + _volume(BRepAlgoAPI_Cut(body, final).Shape())) / max(v, 1e-12)
    if diff > VOLUME_TOL:
        step.unlink()
        return {"uid": design, "skip": "тела выдавливаний не сходятся с итоговым", "diff": diff}
    return {"uid": design, "ops": len(ts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "step").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    designs = sorted(n[len(ZROOT):-5] for n in z.namelist() if n.endswith(".json") and n.count("/") == 2)
    raw = ROOT / "dataset" / "data" / "raw_step"
    ok = []
    with Pool(args.workers, _init, (args.zip, str(out))) as pool, open(out / "prep.jsonl", "w") as log:
        for k, r in enumerate(pool.imap_unordered(check, designs, chunksize=4), 1):
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            if "skip" not in r:
                ok.append(r["uid"])
                link = raw / f"{r['uid']}.stp"
                if not link.exists():
                    link.symlink_to(out / "step" / f"{r['uid']}.stp")
            if k % 500 == 0:
                print(f"{k}/{len(designs)} годных {len(ok)}", flush=True)
    (out / "uids.txt").write_text("\n".join(sorted(ok)) + "\n")
    print("годных", len(ok), "из", len(designs))


if __name__ == "__main__":
    main()
