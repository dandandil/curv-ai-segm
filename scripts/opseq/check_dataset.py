"""Проверка датасета opseq (тела Fusion): полнота, согласованность меток и сканов, сводка.

    CURV_PARTS=<каталог деталей> python scripts/opseq/check_dataset.py --uids <список> --scan <каталог сканов>

Ошибки (печатаются с uid) — то, чего быть не должно: нет файла, номер узла вне графа, петля вне
списка петель узла, зависимость на несуществующую операцию, метка скана вне граней детали.
Отсев (`skip` в opseq.json) — не ошибка, а свойство тела; сводка считает его отдельно.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402

SCAN: Path


def check(uid: str) -> dict:
    errs, st = [], collections.Counter()
    part = D.PARTS / uid
    brep = json.loads((part / "brep.json").read_text())
    n_merged = int(max(brep["merge_map"])) + 1
    lmap = np.asarray(brep["label_map"])
    for name in ("opseq.json", "opseq.nonarrow.json"):
        p = part / name
        if not p.exists():
            errs.append(f"нет {name}")
            continue
        d = json.loads(p.read_text())
        key = "narrow" if name == "opseq.json" else "nonarrow"
        if d["skip"]:
            st[f"{key}:отсеяно"] += 1
            for s in d["skip"]:
                st[f"{key}:отсев:{s.split(':')[0].split('(')[0].strip()}"] += 1
            if "nodes" not in d:
                continue
        else:
            st[f"{key}:годно"] += 1
        nodes = {int(n) for n in d["nodes"]}
        for n, lps in d["loops"].items():
            for lp in lps:
                if any(x >= 0 and x not in nodes for x in lp):
                    errs.append(f"{key}: петля узла {n} ссылается на узел вне графа")
        for u, o in d["ops"].items():
            if any(n not in nodes for n in o["nodes"]):
                errs.append(f"{key}: {o['name']} — узел вне графа")
            for s in o.get("supports", []):
                if s["node"] not in nodes:
                    errs.append(f"{key}: {o['name']} — опора вне графа")
                    continue
                for li in s.get("loops", []):
                    if li >= len(d["loops"][str(s["node"])]):
                        errs.append(f"{key}: {o['name']} — номер петли вне списка")
                for k in s.get("to") or []:
                    if k not in nodes:
                        errs.append(f"{key}: {o['name']} — «до» вне графа")
                if "dir" in s and s["dir"] not in (-1, 1):
                    errs.append(f"{key}: {o['name']} — направление {s['dir']}")
            for s in o.get("strips", []):
                for k in s["pair"] or []:
                    if k not in nodes:
                        errs.append(f"{key}: {o['name']} — пара скругления вне графа")
            if d["skip"]:
                continue
            t = o["type"]
            st[f"{key}:оп:{t}"] += 1
            if t == "ExtrudeFeature":
                sup = o["supports"]
                st[f"{key}:выд:опор"] += len(sup)
                st[f"{key}:выд:с петлями"] += any(s["loops"] for s in sup)
                st[f"{key}:выд:до k"] += any(s["to"] for s in sup)
                st[f"{key}:выд:глубина"] += any(s["depth_mm"] for s in sup)
                st[f"{key}:выд:ни до, ни глубины"] += all(not s["to"] and not s["depth_mm"] for s in sup)
            if t in ("FilletFeature", "ChamferFeature"):
                st[f"{key}:полос"] += len(o["strips"])
                st[f"{key}:полос с парой"] += sum(s["pair"] is not None for s in o["strips"])
        ids = set(d["ops"])
        for a, b in d["deps"]:
            if a not in ids or b not in ids:
                errs.append(f"{key}: зависимость на несуществующую операцию")
        if not d["skip"]:
            st[f"{key}:зависимостей"] += len(d["deps"])
    # скан
    npz, js = SCAN / f"{uid}.npz", SCAN / f"{uid}.json"
    if not (npz.exists() and js.exists()):
        errs.append("нет скана")
    else:
        z = np.load(npz)
        m = json.loads(js.read_text())
        lab = z["labels"]
        if len(lab) != len(z["faces"]) or z["faces"].max() >= len(z["vertices"]):
            errs.append("скан: размеры не сходятся")
        if lab.max() >= n_merged:
            errs.append("скан: метка вне граней детали")
        if not np.isfinite(z["vertices"]).all():
            errs.append("скан: нечисловые вершины")
        st["скан:тел"] += 1
        st["скан:треугольников"] += len(lab)
        st["скан:без метки, тел > 1%"] += m["unlabeled_frac"] > 0.01
        lost = set(m["lost_faces"])
        st["скан:потеряно граней"] += len(lost)
        st["скан:граней"] += m["brep_faces"]
        # потерянные узлы, на которые указывают опоры годных тел
        dn = json.loads((part / "opseq.json").read_text())
        if not dn["skip"]:
            lost_nodes = {int(lmap[i]) for i in lost} - {int(lmap[i]) for i in range(n_merged) if i not in lost}
            if lost_nodes:
                st["narrow:годных тел с потерянным узлом"] += 1
                bad = sum(s["node"] in lost_nodes for o in dn["ops"].values() for s in o.get("supports", []))
                st["narrow:опор на потерянных узлах"] += bad
    return {"uid": uid, "errs": errs, "st": st}


def main() -> None:
    global SCAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    SCAN = Path(args.scan)
    uids = Path(args.uids).read_text().split()
    total, errs = collections.Counter(), []
    with Pool(args.workers, initializer=_init, initargs=(args.scan,)) as pool:
        for r in pool.imap_unordered(_check, uids, chunksize=16):
            total.update(r["st"])
            errs += [(r["uid"], e) for e in r["errs"]]
    print(f"тел {len(uids)}, тел с ошибками {len({u for u, _ in errs})}")
    for e, c in collections.Counter(e for _, e in errs).most_common(20):
        print(f"  {c:6d}  {e}   напр. {next(u for u, x in errs if x == e)}")
    for k in sorted(total):
        print(f"{k:45s} {total[k]}")


def _init(scan):
    global SCAN
    SCAN = Path(scan)


def _check(uid):
    try:
        return check(uid)
    except Exception as e:  # noqa: BLE001
        return {"uid": uid, "errs": [f"исключение {type(e).__name__}: {str(e)[:80]}"], "st": collections.Counter()}


if __name__ == "__main__":
    main()
