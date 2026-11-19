"""Облака точек для эталонного ParSeNet: по 10 000 точек с нормалями и метками.

    python scripts/parsenet/build_points.py --workers 10

Точки берутся равномерно по площади с чистого меша `parts_fine`, без шума и
без ремеша — это условия статьи ParSeNet (их облака насэмплированы с CAD-мешей
ABC). Нормаль точки — нормаль её треугольника, метка — грань после слияния
швов, тип — как в `src/data.face_types`, но нестандартные поверхности
(вращения, выдавливания, эквидистанты) отнесены к сплайну: так делал ParSeNet.

На сплит — `points.npy (N, 10000, 3)`, `normals.npy`, `labels.npy`, `prim.npy` и
`uids.txt`; читаются через memmap. Семя сэмплирования — от uid, пересборка
детерминирована.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.cache import PARTS, SPLITS  # noqa: E402
from src.data import TYPE_INDEX  # noqa: E402

OUT = ROOT / "dataset" / "data" / "parsenet_points"
N_POINTS = 10000
SPLINE = TYPE_INDEX["bspline"]


def sample(uid: str):
    part = PARTS / uid
    mesh = trimesh.load(str(part / "mesh.ply"), process=False)
    face_id = np.load(part / "face_id.npy")
    brep = json.loads((part / "brep.json").read_text())
    rng = np.random.default_rng(zlib.crc32(uid.encode()))

    area = mesh.area_faces
    tri = rng.choice(len(area), N_POINTS, p=area / area.sum())
    u, v = rng.random((2, N_POINTS))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = mesh.triangles[tri]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])

    types = np.full(int(face_id.max()) + 1, SPLINE, np.int8)
    for face in brep["faces"]:
        label = brep["merge_map"][face["id"]]
        if label < len(types):
            types[label] = TYPE_INDEX.get(face["type"], SPLINE)
    labels = face_id[tri]
    return (points.astype(np.float32), mesh.face_normals[tri].astype(np.float32),
            labels.astype(np.int16), types[labels])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        uids = [u for u in (SPLITS / f"{split}.txt").read_text().split() if (PARTS / u / "mesh.ply").exists()]
        tmp = OUT / f"{split}.tmp"
        tmp.mkdir(exist_ok=True)
        arrays = {
            "points": np.lib.format.open_memmap(tmp / "points.npy", "w+", np.float32, (len(uids), N_POINTS, 3)),
            "normals": np.lib.format.open_memmap(tmp / "normals.npy", "w+", np.float32, (len(uids), N_POINTS, 3)),
            "labels": np.lib.format.open_memmap(tmp / "labels.npy", "w+", np.int16, (len(uids), N_POINTS)),
            "prim": np.lib.format.open_memmap(tmp / "prim.npy", "w+", np.int8, (len(uids), N_POINTS)),
        }
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap(sample, uids, chunksize=8)):
                for key, value in zip(("points", "normals", "labels", "prim"), result):
                    arrays[key][i] = value
                if (i + 1) % 2000 == 0:
                    print(f"{split}: {i + 1}/{len(uids)}", flush=True)
        for value in arrays.values():
            value.flush()
        (tmp / "uids.txt").write_text("\n".join(uids) + "\n")
        final = OUT / split
        if final.exists():
            for f in final.iterdir():
                f.unlink()
            final.rmdir()
        tmp.rename(final)
        print(f"{split}: {len(uids)} деталей готово", flush=True)


if __name__ == "__main__":
    main()
