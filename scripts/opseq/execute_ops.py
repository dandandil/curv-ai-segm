"""Исполнение истинных операций тела ядром OCC (README, «Ядро: исполнение операций»).

    python scripts/opseq/execute_ops.py --parts 200 --split val

Операции — из меток, в порядке истории; эскиз выдавливания — проекция истинных стенок на опору
(`src/sketch/projection.py`), геометрия (плоскости, «до k», радиусы) — по подгонке скана из графа
opseq, глубина без поверхности — из меток. Тела с вращением пропускаются (проекции в (r, z) нет).
Метрики: операции исполнены, расстояния скан -> тело и тело -> скан / D, доля узлов цели,
объяснённых телом (не меньше EXPLAIN_SHARE точек узла не дальше EXPLAIN_REL от тела), тело верно —
всё исполнено, объяснены все узлы, у тела нет лишнего (p95 тело -> скан не больше EXPLAIN_REL).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool, TimeoutError
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.opseq.config import OCFG  # noqa: E402
from src.opseq.data import CORPORA, load_graph, split_uids  # noqa: E402
from src.opseq.kernel import (MIN_EDGE_REL, MIN_VOLUME_REL, BodySamples, KernelError, boolean, common_edges, extrude,  # noqa: E402
                              faces_of, fillet, profile_face)
from src.sketch.projection import (CELL_REL, JOIN_REL, MERGE_REL, extrusion_direction, self_intersects,  # noqa: E402
                                   sketch_from_walls, walls_of)
from src.opseq.regularize import MIN_SIN, fit_nodes, regularize  # noqa: E402
from src.sketch.curves import _circle_fit, _intersect, _line_fit  # noqa: E402
from src.sketch.section import plane_frame  # noqa: E402
from src.surfaces import AXIAL  # noqa: E402

# корпус — OPSEQ_CORPUS, по умолчанию прежний сканер (на нём сделаны замеры в README)
CORPUS = CORPORA[os.environ.get("OPSEQ_CORPUS", "seg_old")]
DATA = CORPUS.root
# Узел объяснён: не меньше EXPLAIN_SHARE его точек скана не дальше EXPLAIN_REL · D от тела. Допуск
# 0.01 D (~0.25 мм) — с запасом на сдвиг скана (до 0.13 мм на части деталей). Выбрано.
EXPLAIN_REL = 0.01
EXPLAIN_SHARE = 0.9
# Точек скана на узел для сопоставления граней; точек по площади тела. Выбрано.
NODE_POINTS = 300
BODY_SAMPLES = 200_000
# Прогиб тесселяции тела / D. Выбрано.
DEFLECTION_REL = 0.002
# Стенка с отклонением от перпендикуляра к эскизу больше этого — кривая по проекции. Выбрано.
FIXED_TILT_DEG = 5.0
# Потолок времени на тело, с. Выбрано.
TIMEOUT = 300
OPS = ("ExtrudeFeature", "FilletFeature", "ChamferFeature")
# В сечении у ребра полосы соседи фаски — прямые, полоса скругления — дуга: RMS больше этого / D —
# сечение не то (например, ось цилиндра вдоль ребра). Выбрано.
SECTION_RMS_REL = 0.004


class Part:
    def __init__(self, uid: str, straighten: bool = False):
        self.uid = uid
        self.lab = json.loads((DATA / "parts" / uid / OCFG.data.labels).read_text())
        self.g = load_graph(CORPUS.graph_path(uid))
        self.pos = {int(m): i for i, m in enumerate(self.g["ids"])}
        self.diag = float(self.g["diag"])
        z = np.load(CORPUS.scan_path(uid))
        self.v, self.f, self.tri = z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["labels"]
        self.types = {int(k): x["type"] for k, x in self.lab["nodes"].items()}
        ids = [int(x) for x in self.g["ids"]]
        self.tangent = {tuple(sorted((ids[a], ids[b]))) for (a, b), t in zip(self.g["edges"], self.g["tangent"]) if t}
        self.adjacent = {tuple(sorted((ids[a], ids[b]))) for a, b in self.g["edges"]}
        self.inserted = self.extent_from_walls = self.crossed = self.crossed_left = 0
        # выпрямление: уточнённые поверхности по номеру узла метки
        self.refined, self.straightened, self.fits = {}, None, None
        if straighten:
            surf, self.straightened = regularize(self.g, self.v, self.f, self.tri, self.types)
            self.refined = {ids[i]: s for i, s in surf.items()}
        rng = np.random.default_rng(0)
        self.node_points = {}
        for m in self.pos:
            vid = np.unique(self.f[self.tri == m])
            if len(vid):
                self.node_points[m] = self.v[rng.choice(vid, min(NODE_POINTS, len(vid)), replace=False)]

    def plane(self, m: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Нормаль (наружу по мешу) и точка плоскости узла: уточнённая, если есть, иначе из графа."""
        i = self.pos.get(m)
        if i is None or not self.g["plane"][i]:
            return None
        n, a = self.g["dir"][i].astype(np.float64), self.g["anchor"][i].astype(np.float64)
        s = self.refined.get(m)
        if s is not None and s.kind == "plane":
            u = s.params[:3] / np.linalg.norm(s.params[:3])
            off = s.params[3] / np.linalg.norm(s.params[:3])
            a = a - (u @ a + off) * u
            n = u if u @ n >= 0 else -u
        return n, a

    def surface(self, m: int):
        """Поверхность узла: уточнённая, иначе подгонка по скану (`fit_nodes`, один раз на деталь)."""
        if self.fits is None:
            ids = [int(x) for x in self.g["ids"]]
            self.fits = self.refined or {ids[i]: s for i, s in
                                         fit_nodes(self.g, self.v, self.f, self.tri, self.types)[0].items()}
        return self.fits.get(m)

    def points_of(self, m: int) -> np.ndarray:
        return self.v[np.unique(self.f[self.tri == m])]

    def sections(self, p: int, q: int, c: int) -> list:
        """Сечения у ребра полосы `c` между `p` и `q`, где соседи и полоса — кривые на плоскости: у прямого
        ребра между плоскостями — поперёк ребра, у круглого — полуплоскость (r, z) через ось одной из
        поверхностей вращения среди трёх, направление — её ось или нормаль соседней плоскости. Какое
        верно — решает невязка: у узкого цилиндра или короткой дуги ось бывает увернута (невязка
        плоскости в (r, z) до 8‰ D [120221_e2eaa34b_1, 133589_40359674_2])."""
        out = []
        pp, pq = self.plane(p), self.plane(q)
        if pp is not None and pq is not None:
            u = np.cross(pp[0], pq[0])
            if np.linalg.norm(u) > MIN_SIN:
                fr = plane_frame(u)[:2]
                out.append(lambda x: x @ fr.T)
        normals = [x[0] for x in (pp, pq) if x is not None]
        for s in map(self.surface, (p, q, c)):
            if s is None or s.kind not in AXIAL:
                continue
            for a in [s.params[3:6] / np.linalg.norm(s.params[3:6]), *normals]:
                def rz(x, o=s.params[:3], a=a):
                    y = x - o
                    z = y @ a
                    return np.c_[np.linalg.norm(y - z[:, None] * a, axis=1), z]
                out.append(rz)
        return out

    def fixed_curves(self, sides: list[int], d: np.ndarray, frame: np.ndarray, height: float) -> dict[int, tuple]:
        """Кривые эскиза из уточнённых стенок: плоскость стенки ∩ плоскость эскиза, цилиндр с осью
        вдоль d — окружность. Стенка, не перпендикулярная эскизу (уклон), — по проекции."""
        out = {}
        for m in sides:
            s = self.refined.get(m)
            if s is None:
                continue
            p = s.params
            if s.kind == "plane":
                nw = p[:3] / np.linalg.norm(p[:3])
                if abs(nw @ d) > np.sin(np.radians(FIXED_TILT_DEG)):
                    continue
                a = np.stack([nw, d])
                x0 = a.T @ np.linalg.solve(a @ a.T, [-p[3] / np.linalg.norm(p[:3]), height])
                u = frame[:2] @ np.cross(nw, d)
                out[m] = ("line", frame[:2] @ x0, u / np.linalg.norm(u))
            elif s.kind == "cylinder":
                ax = p[3:6] / np.linalg.norm(p[3:6])
                if abs(ax @ d) < np.cos(np.radians(FIXED_TILT_DEG)):
                    continue
                c = p[:3] + ax * (height - d @ p[:3]) / (d @ ax)
                out[m] = ("circle", frame[:2] @ c, float(p[6]))
        return out

    def extrude_op(self, body, op: dict):
        sup = next((s for s in op.get("supports", []) if self.plane(s["node"]) is not None), None)
        if sup is None:
            raise KernelError("нет опоры-плоскости")
        n_s, anchor = self.plane(sup["node"])
        sides = [m for m in op["sides"] if m in self.pos]
        if not sides:
            # боковые грани слились с гранями ранней операции и приписаны ей (outer_loop_fallback в
            # метках): стенки — соседи опоры по её петлям, проекция та же
            sides = sorted({m for li in sup.get("loops") or [] for m in self.lab["loops"][str(sup["node"])][li]
                            if m in self.pos})
        d = n_s if sup["node"] in self.refined else extrusion_direction(self.v, self.f, self.tri, sides, n_s)
        frame = plane_frame(d)
        walls = walls_of(self.v, self.f, self.tri, sides, self.types, anchor, d, frame, CELL_REL * self.diag,
                         self.fixed_curves(sides, d, frame, float(anchor @ d)))
        args = (walls, self.tangent, self.adjacent, JOIN_REL * self.diag, CELL_REL * self.diag, MERGE_REL * self.diag)
        loops = sketch_from_walls(*args, close_open=True)
        if not loops:
            raise KernelError("эскиз пуст")
        if any(self_intersects(lp) for lp in loops):
            # перекручено далёкими пересечениями продлённых кривых: острые стыки — середины концов
            self.crossed += 1
            loops = sketch_from_walls(*args, close_open=True, crossings=False)
            if any(self_intersects(lp) for lp in loops):
                self.crossed_left += 1
        self.inserted += sum(c.node < 0 for lp in loops for c in lp)
        to = self.plane(sup["to"][0]) if sup.get("to") else None
        if to is not None:
            extent = float((to[1] - anchor) @ d)
        elif sup.get("depth_mm") and not sup.get("to"):
            extent = float(sup["depth_mm"]) * float(sup["dir"])
        else:
            # «до» неплоской грани или протяжённость не размечена: до самой далёкой от опоры точки
            # стенок (сторона — где их больше по площади точек); «до поверхности» с отрезанием — потом
            h = (self.v[np.unique(self.f[np.isin(self.tri, sides)])] - anchor) @ d
            extent = float(h.max() if (h > 0).sum() >= (h < 0).sum() else h.min())
            self.extent_from_walls += 1
        profile = profile_face(loops, frame, float(anchor @ d), MIN_EDGE_REL * self.diag)
        return boolean(body, extrude(profile, d, extent, MIN_VOLUME_REL * self.diag ** 3), op["boolean"] or "add")

    def fillet_op(self, body, op: dict):
        if body is None:
            raise KernelError("скругление до тела")
        kind = "fillet" if op["type"] == "FilletFeature" else "chamfer"
        samples = BodySamples(body, DEFLECTION_REL * self.diag, BODY_SAMPLES // 4)
        faces = faces_of(body)
        edges = []
        for s in op.get("strips", []):
            if s["pair"] is None or not all(k in self.node_points for k in s["pair"]) or s["node"] not in self.pos:
                continue
            p, q = s["pair"]
            match = samples.match({p: self.node_points[p], q: self.node_points[q]}, EXPLAIN_REL * self.diag)
            fp, fq = faces[match[p][0]], faces[match[q][0]]
            common = common_edges(body, fp, fq)
            i = self.pos[s["node"]]
            st = self.g["static"][i]
            if kind == "fillet":
                kind_node = self.types[s["node"]]
                r = {"cylinder": st[8], "sphere": st[9], "torus": st[11]}.get(kind_node, 0) * self.diag
                if not st[6] or r <= 0:
                    r = self.fillet_radius(p, q, s["node"])
                edges += [(e, r) for e in common]
            else:
                d1, d2 = self.chamfer_widths(p, q, s["node"])
                edges += [(e, d1, d2, fp) for e in common]
        if kind == "fillet":
            # полоса без радиуса и в сечении (сплайн между неплоскими соседями) — медиана радиусов
            # операции: у скругления Fusion он чаще один на всю операцию
            known = [e[1] for e in edges if e[1] is not None]
            if any(e[1] is None for e in edges):
                if not known:
                    raise KernelError("радиус скругления не подогнан")
                edges = [(e[0], e[1] if e[1] is not None else float(np.median(known))) for e in edges]
        if not edges:
            raise KernelError("нет рёбер для скругления")
        return fillet(body, edges, kind)

    def fillet_radius(self, p: int, q: int, c: int) -> float | None:
        """Радиус полосы, у которой подгонка не прошла (короткая дуга, сплайн): окружность по её точкам
        в сечении у ребра — ось там известна по соседям. None — сечения нет или точки не на дуге."""
        best = None
        for cut in self.sections(p, q, c):
            x = cut(self.points_of(c))
            center, r = _circle_fit(x)
            rms = np.sqrt(np.mean((np.linalg.norm(x - center, axis=1) - r) ** 2))
            if rms <= SECTION_RMS_REL * self.diag and (best is None or rms < best[0]):
                best = rms, r
        return None if best is None else best[1]

    def chamfer_widths(self, p: int, q: int, c: int) -> tuple[float, float]:
        """Ширины фаски (README, «параметры фаски»): ребро — пересечение продлённых соседей, d1 и d2 — от
        него до пересечения фаски с каждым соседом. Между тремя плоскостями — по подогнанным
        плоскостям, иначе (фаска-конус на круглом ребре) — прямыми по точкам в сечении у ребра."""
        planes = [self.plane(m) for m in (p, q, c)]
        if all(x is not None for x in planes):
            (np_, op_), (nq, oq), (nc, oc) = planes
            u = np.cross(np_, nq)
            u /= max(np.linalg.norm(u), 1e-12)

            def point(n1, o1, n2, o2):
                a = np.stack([n1, n2, u])
                return np.linalg.solve(a, [n1 @ o1, n2 @ o2, u @ oc])

            x = point(np_, op_, nq, oq)
            return float(np.linalg.norm(point(np_, op_, nc, oc) - x)), float(np.linalg.norm(point(nq, oq, nc, oc) - x))
        best = None
        for cut in self.sections(p, q, c):
            lines, worst = [], 0.0
            for m in (p, q, c):
                y = cut(self.points_of(m))
                o, d = _line_fit(y)
                worst = max(worst, float(np.sqrt(np.mean(((y - o) @ np.array([-d[1], d[0]])) ** 2))))
                lines.append((o, d))
            if best is None or worst < best[0]:
                best = worst, lines
        if best is None:
            raise KernelError("нет сечения у ребра полосы")
        if best[0] > SECTION_RMS_REL * self.diag:
            raise KernelError("фаска: в сечении не прямые")
        lines = best[1]
        x, xp, xq = _intersect(*lines[0], *lines[1]), _intersect(*lines[0], *lines[2]), _intersect(*lines[1], *lines[2])
        if x is None or xp is None or xq is None:
            raise KernelError("фаска: прямые сечения параллельны")
        return float(np.linalg.norm(xp - x)), float(np.linalg.norm(xq - x))

def run_ops(part: Part, ops: list[dict]):
    """Операции по порядку; упавшая пропускается. Возвращает тело (или None) и журнал."""
    body, log = None, []
    for op in ops:
        if op["type"] not in OPS:
            continue
        try:
            body = part.extrude_op(body, op) if op["type"] == "ExtrudeFeature" else part.fillet_op(body, op)
            log.append([op["name"], op["type"], "ok"])
        except KernelError as err:
            log.append([op["name"], op["type"], str(err)])
        except Exception as err:  # noqa: BLE001 — Standard_Failure из OCC
            log.append([op["name"], op["type"], f"OCC: {type(err).__name__}"])
    return body, log


def execute(job: tuple[str, bool]) -> dict:
    uid, straighten = job
    t0 = time.perf_counter()
    out = {"uid": uid}
    try:
        part = Part(uid, straighten)
        out["straightened"] = part.straightened
        ops = sorted(part.lab["ops"].values(), key=lambda o: o["timeline_index"])
        if any(o["type"] == "RevolveFeature" for o in ops):
            return {**out, "skip": "вращение"}
        body, log = run_ops(part, ops)
        out["ops"] = log
        out["inserted"], out["extent_from_walls"] = part.inserted, part.extent_from_walls
        out["self_intersecting"] = [part.crossed, part.crossed_left]
        if body is None:
            return {**out, "built": False}
        try:
            samples = BodySamples(body, DEFLECTION_REL * part.diag, BODY_SAMPLES)
        except KernelError:
            return {**out, "built": False}
        scan_samples = trimesh.sample.sample_surface(trimesh.Trimesh(part.v, part.f, process=False), BODY_SAMPLES,
                                                     seed=0)[0]
        s2b, b2s = samples.distances(part.v, scan_samples)
        match = samples.match(part.node_points, EXPLAIN_REL * part.diag)
        explained = float(np.mean([fr >= EXPLAIN_SHARE for _, fr in match.values()]))
        out.update({"built": True, "scan_to_body": [float(np.median(s2b) / part.diag), float(np.percentile(s2b, 95) / part.diag)],
                    "body_to_scan": [float(np.median(b2s) / part.diag), float(np.percentile(b2s, 95) / part.diag)],
                    "explained": explained})
        out["correct"] = (all(x[2] == "ok" for x in log) and explained == 1.0
                          and out["body_to_scan"][1] <= EXPLAIN_REL)
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc(limit=3)
    out["seconds"] = time.perf_counter() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=200)
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="outputs/execute_ops/val.jsonl")
    ap.add_argument("--straighten", action="store_true", help="выпрямление: отношения пар в совместной подгонке")
    args = ap.parse_args()
    uids = CORPUS.uids()
    uids = split_uids(uids)[args.split][:args.parts]
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    res = []
    with Pool(args.workers, maxtasksperchild=20) as pool, open(path, "w") as fh:
        jobs = [(u, pool.apply_async(execute, ((u, args.straighten),))) for u in uids]
        for k, (u, job) in enumerate(jobs, 1):
            try:
                r = job.get(timeout=TIMEOUT)
            except TimeoutError:
                r = {"uid": u, "error": "время вышло"}
            res.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if k % 20 == 0:
                print(f"{k}/{len(uids)}", file=sys.stderr, flush=True)
        pool.terminate()
    done = [r for r in res if "skip" not in r and "error" not in r]
    ops = Counter()
    for r in done:
        for name, kind, status in r.get("ops", []):
            ops[(kind, "ok" if status == "ok" else status)] += 1
    built = [r for r in done if r.get("built")]
    summary = {"тел": len(res), "пропущено (вращение)": sum("skip" in r for r in res),
               "ошибок": sum("error" in r for r in res), "исполнено": len(done),
               "тело построено": len(built) / max(len(done), 1),
               "все операции прошли": sum(all(x[2] == "ok" for x in r["ops"]) for r in done) / max(len(done), 1),
               "тело верно": sum(bool(r.get("correct")) for r in done) / max(len(done), 1),
               "эскизов с самопересечением / осталось после середин": np.sum([r.get("self_intersecting", [0, 0]) for r in done], 0).tolist() if done else None,
               "рост RMS при выпрямлении, медиана медиан / максимум": (
                   [float(np.median([r["straightened"]["рост RMS, медиана / максимум"][0] for r in done if r.get("straightened")])),
                    float(np.max([r["straightened"]["рост RMS, медиана / максимум"][1] for r in done if r.get("straightened")]))]
                   if any(r.get("straightened") for r in done) else None),
               "узлов объяснено (среднее)": float(np.mean([r["explained"] for r in built])) if built else 0,
               "скан -> тело, медиана / p95 ‰ D": np.round(np.median([r["scan_to_body"] for r in built], 0) * 1000, 2).tolist() if built else None,
               "тело -> скан, медиана / p95 ‰ D": np.round(np.median([r["body_to_scan"] for r in built], 0) * 1000, 2).tolist() if built else None,
               "операции": {f"{k[0]}: {k[1]}": v for k, v in sorted(ops.items(), key=lambda kv: -kv[1])}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
