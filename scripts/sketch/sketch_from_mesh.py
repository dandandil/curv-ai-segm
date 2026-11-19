"""Эскиз по сечению меша плоскостью: цепочки скана -> 2D сеть -> стыки водоразделом -> кривые
(README, «Эскиз из сечения»).

    python scripts/sketch/sketch_from_mesh.py --mesh part.stl --run runs/section_xy \\
        --origin 0 0 5 --normal 0 0 1 --out outputs/sketch_demo/part
    python scripts/sketch/sketch_from_mesh.py --mesh part.stl --run runs/section_xy --axes --out ...

`--axes` — три плоскости через центр по осям ориентированного габарита (для проверки без выбора
плоскости). Пишет `<out>_<k>.json` (кривые в системе плоскости: origin, e1, e2 — точка
x·e1 + y·e2 + origin) и `<out>_<k>.png`. Ширина сети берётся из чекпоинта.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sketch.config import SCFG  # noqa: E402
from src.sketch.curves import build_pieces_checked, peaks, sample_curve  # noqa: E402
from src.sketch.data import collate, segment_features  # noqa: E402
from src.sketch.model import SectionNet  # noqa: E402
from src.sketch.section import CURVE_TYPES, merge_short, mesh_section, plane_frame  # noqa: E402

COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")


def load_model(run: str, device) -> SectionNet:
    state = torch.load(ROOT / run / "best.pt", map_location=device)["model"]
    cfg = dataclasses.replace(SCFG.model, channels=state["inp.weight"].shape[0])
    model = SectionNet(cfg).to(device)
    model.load_state_dict(state)
    return model.eval()


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def sketch(model, v, f, nrm, origin, normal, diag, device) -> tuple[np.ndarray, list[dict]]:
    """Система плоскости и цепочки с кусками и стыками."""
    frame = plane_frame(normal)
    d = SCFG.data
    chains = [merge_short(c, d.min_seg_rel * diag) for c in mesh_section(v, f, nrm, origin, frame)]
    chains = [c for c in chains if c.n >= d.min_chain]
    if not chains:
        return frame, []
    item = {"uid": "", "diag": diag, "chains": []}
    for c in chains:
        k = len(c.points)
        item["chains"].append({"feat": segment_features(c.points, c.normals, c.closed, diag),
                               "type": -np.ones(c.n, np.int64), "boundary": -np.ones(k, np.int64),
                               "tangent": -np.ones(k, np.int64), "closed": c.closed, "points": c.points, "sec": 0})
    batch = collate([item])
    with torch.no_grad():
        b = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
        type_logits, bnd, tan = model(b)
    tprob = torch.softmax(type_logits, 1).cpu().numpy()
    pb, pt = torch.sigmoid(bnd).cpu().numpy(), torch.sigmoid(tan).cpu().numpy()
    jc, jp = batch["junction_chain"].numpy(), batch["junction_point"].numpy()
    out = []
    for ci, c in enumerate(chains):
        js = np.flatnonzero(jc == ci)
        s0 = int(batch["start"][ci])
        pk = peaks(pb[js], c.closed, SCFG.train.peak_threshold, SCFG.train.peak_window)
        pieces, jpos, _, sharp = build_pieces_checked(c.points, c.closed, jp[js[pk]], pt[js[pk]] <= 0.5,
                                                      tprob[s0:s0 + c.n], SCFG.train.split_rel * diag)
        out.append({"closed": c.closed, "points": c.points,
                    "junctions": [{"point": p, "tangent": not s} for p, s in zip(jpos, sharp)],
                    "pieces": [{"type": CURVE_TYPES[pc.kind], "curve": pc.curve} for pc in pieces]})
    return frame, out


def draw(path: Path, chains: list[dict], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for c in chains:
        p = np.vstack([c["points"], c["points"][:1]]) if c["closed"] else c["points"]
        ax.plot(p[:, 0], p[:, 1], "-", color="0.8", lw=4, zorder=1)
        for pc in c["pieces"]:
            s = sample_curve(pc["curve"], 64)
            if s is not None:
                ax.plot(s[:, 0], s[:, 1], "-", color=COLORS[CURVE_TYPES.index(pc["type"])], lw=1.5, zorder=2)
        for j in c["junctions"]:
            if j["tangent"]:
                ax.scatter(*j["point"], marker="o", facecolors="none", edgecolors="red", s=50, zorder=3)
            else:
                ax.scatter(*j["point"], marker="x", c="red", s=50, zorder=3)
    for k, name in enumerate(CURVE_TYPES):
        ax.plot([], [], color=COLORS[k], lw=2, label=name)
    ax.plot([], [], color="0.8", lw=4, label="скан")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="префикс файлов результата")
    ap.add_argument("--origin", type=float, nargs=3)
    ap.add_argument("--normal", type=float, nargs=3)
    ap.add_argument("--axes", action="store_true", help="три плоскости через центр по осям габарита")
    args = ap.parse_args()
    mesh = trimesh.load(args.mesh, force="mesh")
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    nrm = np.asarray(mesh.face_normals, float)
    diag = float(np.linalg.norm(np.ptp(v, 0)))
    if args.axes:
        to_world = np.asarray(mesh.bounding_box_oriented.primitive.transform)
        planes = [(to_world[:3, 3], to_world[:3, k]) for k in range(3)]
    elif args.origin and args.normal:
        planes = [(np.array(args.origin), np.array(args.normal))]
    else:
        ap.error("нужны --origin и --normal или --axes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.run, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, (origin, normal) in enumerate(planes):
        frame, chains = sketch(model, v, f, nrm, origin, normal, diag, device)
        rec = {"mesh": args.mesh, "run": args.run, "origin": origin, "e1": frame[0], "e2": frame[1],
               "normal": frame[2], "chains": chains}
        Path(f"{out}_{k}.json").write_text(json.dumps(jsonable(rec), ensure_ascii=False))
        kinds = [pc["type"] for c in chains for pc in c["pieces"]]
        draw(Path(f"{out}_{k}.png"), chains, f"{Path(args.mesh).name}, плоскость {k}: {len(chains)} цепочек, "
             f"{len(kinds)} кусков")
        print(f"плоскость {k}: цепочек {len(chains)}, кусков {len(kinds)} " +
              str({t: kinds.count(t) for t in CURVE_TYPES if t in kinds}), flush=True)


if __name__ == "__main__":
    main()
