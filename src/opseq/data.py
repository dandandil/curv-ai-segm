"""Граф поверхностей детали: узлы, пары, метки шагов; батч тел для обучения.

Базовая линия (README, «Тело последовательностью операций»): узел — истинная грань тела, геометрия
узла — подгонка и точки по её треугольникам на скане, тип — истинный (one-hot). Граф строится один
раз (`build_graph`, `scripts/opseq/build_graphs.py`) и лежит в `<корень корпуса>/<graphs>/<uid>.npz`; метки
шагов (`opseq.nonarrow.json`) переводятся в номера узлов при загрузке (`Body`). Тело в датасете — ключ
`<корпус>/<uid>` (корпуса — `data.corpora` конфига, README «Данные v2»).

Граф на тело один: скругления и фаски — обычные операции цикла (решено пользователем 2026-10-25),
полосы остаются узлами.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from src import fitting as F
from src.config import CFG
from src.model import PRIMITIVE_TYPES
from src.surfaces import AXIAL

from .config import OCFG

BOOLEANS = ("add", "cut", "intersect")
OP_TYPES = ("ExtrudeFeature", "RevolveFeature", "FilletFeature", "ChamferFeature")
FILLET_OPS = ("FilletFeature", "ChamferFeature")
# Выбор головы типа шага: операция или «готово».
STEP_CLASSES = OP_TYPES + ("done",)
# Класс грани для текущей операции: стенка — боковая грань (у скругления и фаски — полоса), торец —
# прочие её грани.
FACE_CLASSES = ("нет", "стенка", "торец")
RELATIONS = ("parallel", "perpendicular", "coplanar", "coaxial", "equal_radius")
# Роль грани в теле (вспомогательная голова до цикла, решено пользователем 2026-10-30): флаги, а не класс —
# узел, слитый из граней разных операций, бывает стенкой одной и торцом другой.
ROLES = ("стенка", "торец", "полоса")
# Опоры вращения — поверхности вращения (как ROTATIONAL в dataset/fusion_opseq.py); выдавливания — плоскость.
REVOLVE_SUPPORTS = ("cylinder", "cone", "sphere", "torus")
STATIC_DIM = 20


@dataclass(frozen=True)
class Corpus:
    name: str
    root: Path
    scan: str
    graphs: str

    def scan_path(self, uid: str) -> Path:
        return self.root / self.scan / f"{uid}.npz"

    def graph_path(self, uid: str) -> Path:
        return self.root / self.graphs / f"{uid}.npz"

    def labels(self, uid: str, cfg=OCFG.data) -> dict:
        return json.loads((self.root / "parts" / uid / cfg.labels).read_text())

    def uids(self) -> list[str]:
        """Тела с собранным графом."""
        return sorted(p.name[:-4] for p in (self.root / self.graphs).glob("*.npz") if not p.name.endswith(".tmp.npz"))


CORPORA = {c["name"]: Corpus(c["name"], Path(c["root"]), c["scan"], c["graphs"]) for c in OCFG.data.corpora}


def corpus_of(key: str) -> tuple[Corpus, str]:
    name, uid = key.split("/", 1)
    return CORPORA[name], uid


def graph_keys(names=OCFG.data.train_corpora) -> list[str]:
    return [f"{n}/{u}" for n in names for u in CORPORA[n].uids()]
STATE_DIM = 1 + len(OP_TYPES) + len(BOOLEANS) + 1
PAIR_DIM = 7


def node_type(info: dict) -> str:
    kind = info["types"][0]
    return kind if kind in PRIMITIVE_TYPES else "bspline"


# ---------- граф по скану ----------

def _frame(points: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    """Строки — оси собственной системы узла: z — нормаль плоскости или ось, если подогнаны,
    иначе ось наименьшего разброса; x, y — главные оси точек в плоскости, перпендикулярной z."""
    q = points - points.mean(0)
    if z is None:
        z = np.linalg.svd(q, full_matrices=False)[2][-1]
    p = q - np.outer(q @ z, z)
    x = np.linalg.svd(p, full_matrices=False)[2][0] if len(p) > 1 else np.eye(3)[np.argmin(np.abs(z))]
    x = x - (x @ z) * z
    x /= max(np.linalg.norm(x), 1e-12)
    return np.stack([x, np.cross(z, x), z])


def _sample(tri: np.ndarray, area: np.ndarray, normal: np.ndarray, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """`k` точек по площади треугольников узла и нормали их треугольников."""
    t = rng.choice(len(tri), k, p=area / area.sum())
    u = rng.random((k, 2))
    u = np.where(u.sum(1, keepdims=True) > 1, 1 - u, u)
    p = tri[t, 0] + u[:, :1] * (tri[t, 1] - tri[t, 0]) + u[:, 1:] * (tri[t, 2] - tri[t, 0])
    return p, normal[t]


def _exact_geometry(brep: dict, ids: np.ndarray) -> dict:
    """Точная геометрия узлов из STEP: направление (нормаль, ось), точка, радиус."""
    first = {}
    for i, m in enumerate(brep["merge_map"]):
        first.setdefault(m, i)
    n = len(ids)
    out = {"dir": np.zeros((n, 3)), "point": np.zeros((n, 3)), "radius": np.full(n, np.nan),
           "plane": np.zeros(n, bool), "axial": np.zeros(n, bool)}
    for i, m in enumerate(ids):
        face = brep["faces"][first[int(m)]]
        kind, p = face["type"], face["params"]
        if kind == "plane":
            out["dir"][i], out["point"][i], out["plane"][i] = p["normal"], p["origin"], True
        elif kind in AXIAL:
            out["dir"][i], out["axial"][i] = p["axis"], True
            out["point"][i] = p["apex"] if kind == "cone" else p.get("location", p.get("center"))
            out["radius"][i] = p.get("radius", p.get("minor_radius", np.nan))
        elif kind == "sphere":
            out["point"][i], out["radius"][i] = p["center"], p["radius"]
    norm = np.linalg.norm(out["dir"], axis=1, keepdims=True)
    out["dir"] = np.where(norm > 0, out["dir"] / np.maximum(norm, 1e-12), 0)
    return out


def relation_labels(brep: dict, ids: np.ndarray, diag: float, cfg=OCFG.data) -> np.ndarray:
    """Метки отношений пар `(N, N, len(RELATIONS))` по точной геометрии STEP."""
    g = _exact_geometry(brep, ids)
    d, pt = g["dir"], g["point"]
    has_dir = np.linalg.norm(d, axis=1) > 0
    cos = d @ d.T
    sin = np.sqrt(np.clip(1 - cos ** 2, 0, None))
    both = has_dir[:, None] & has_dir[None]
    parallel = both & (sin < cfg.relation_sin)
    tol = cfg.relation_rel * diag
    off = pt[None, :, :] - pt[:, None, :]                       # [i, j] = точка j − точка i
    along = np.einsum("ijk,ik->ij", off, d)
    to_axis = np.linalg.norm(off - along[..., None] * d[:, None, :], axis=-1)
    plane, axial = g["plane"], g["axial"]
    r = g["radius"]
    out = np.stack([
        parallel,
        both & (np.abs(cos) < cfg.relation_sin),
        parallel & plane[:, None] & plane[None] & (np.abs(along) < tol),
        parallel & axial[:, None] & axial[None] & (to_axis < tol),
        np.isfinite(r)[:, None] & np.isfinite(r)[None] & (np.abs(r[:, None] - r[None]) < tol),
    ], -1)
    out[np.arange(len(ids)), np.arange(len(ids))] = False
    return out.astype(np.uint8)


def _tangent_merged(brep: dict) -> set[tuple[int, int]]:
    """Пары слитых граней, у которых весь стык касательный (как `src.boundary.data.tangent_pairs`)."""
    merge = brep["merge_map"]
    sharp: dict[tuple[int, int], bool] = {}
    for adj in brep["adjacency"]:
        a, b = sorted((merge[adj["a"]], merge[adj["b"]]))
        if a != b:
            sharp[(a, b)] = sharp.get((a, b), False) or not adj["tangential"]
    return {p for p, s in sharp.items() if not s}


def build_graph(uid: str, labels: dict, corpus: Corpus, cfg=OCFG.data) -> dict | None:
    """Граф тела по скану: признаки и точки узлов, геометрия для признаков пар, соседство, метки
    отношений. None — узлов больше `max_nodes`."""
    z = np.load(corpus.scan_path(uid))
    v, f, lab = z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["labels"].astype(np.int64)
    brep = json.loads((corpus.root / "parts" / uid / "brep.json").read_text())
    ids = np.array(sorted(set(np.unique(lab[lab >= 0]).tolist()) & {int(k) for k in labels["nodes"]}))
    if len(ids) > cfg.max_nodes:
        return None
    lookup = np.full(int(max(lab.max(), ids.max())) + 1, -1)
    lookup[ids] = np.arange(len(ids))
    node = np.where(lab >= 0, lookup[np.maximum(lab, 0)], -1)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    mesh = trimesh.Trimesh(v, f, process=False)
    adjacency = mesh.face_adjacency
    center, normal, area = F.triangle_geometry(v, f)
    excluded = F.boundary_vertices(f, adjacency, lab, CFG.brep.boundary_rings, len(v))
    rng = np.random.default_rng(zlib.crc32(uid.encode()))

    n = len(ids)
    types = np.zeros(n, np.int64)
    static = np.zeros((n, STATIC_DIM), np.float32)
    points = np.zeros((n, cfg.points, 6), np.float16)
    geo = {"dir": np.zeros((n, 3)), "anchor": np.zeros((n, 3)), "fit": np.zeros(n, bool),
           "plane": np.zeros(n, bool), "axial": np.zeros(n, bool)}
    for i, m in enumerate(ids):
        info = labels["nodes"][str(m)]
        kind = node_type(info)
        types[i] = PRIMITIVE_TYPES.index(kind)
        t = node == i
        pts, nrm, w = F.cluster_points(m, lab, excluded, v, f, normal, area, CFG.brep.min_fit_points)
        surf = F.fit_kind(kind, pts, nrm, w) if kind != "bspline" and len(pts) >= 3 else None
        ok = surf is not None and np.isfinite(surf.rms)
        a = area[t]
        c = (center[t] * a[:, None]).sum(0) / a.sum()
        mean_n = (normal[t] * a[:, None]).sum(0)
        d, anchor = None, c
        radii = np.zeros(6)
        if ok:
            # конус на пределе подгоняется цилиндром или плоскостью: параметры — по типу подгонки
            p, fk = surf.params, surf.kind
            if fk == "plane":
                d = p[:3] / np.linalg.norm(p[:3])
                d = d if d @ mean_n >= 0 else -d                # наружу, по нормалям меша
            elif fk in AXIAL:
                d, anchor = p[3:6] / np.linalg.norm(p[3:6]), p[:3]
            else:
                anchor = p[:3]
            if fk == "cylinder":
                radii[0] = p[6] / diag
            elif fk == "sphere":
                radii[1] = p[3] / diag
            elif fk == "torus":
                radii[2:4] = p[6] / diag, p[7] / diag
            elif fk == "cone":
                radii[4:6] = np.sin(p[6]), np.cos(p[6])
        tri = v[f[t]]
        R = _frame(tri.reshape(-1, 3), d)
        q, qn = _sample(tri, a, normal[t], cfg.points, rng)
        points[i] = np.c_[(q - c) @ R.T / diag, qn @ R.T]
        ext = np.ptp(tri.reshape(-1, 3) @ R.T, axis=0) / diag
        static[i, :6] = np.eye(6)[types[i]]
        static[i, 6:8] = ok, (surf.rms / diag * 100 if ok else 0)
        static[i, 8:14] = radii
        static[i, 14] = a.sum() / diag ** 2
        static[i, 16] = np.log1p(len(labels["loops"][str(m)]))
        static[i, 17:20] = ext
        if d is not None:
            geo["dir"][i] = d
        geo["anchor"][i], geo["fit"][i] = anchor, ok
        geo["plane"][i], geo["axial"][i] = ok and surf.kind == "plane", ok and surf.kind in AXIAL

    # соседство узлов по рёбрам скана: длина стыка и выпуклость (знак угла по нормалям меша)
    ta, tb = adjacency[:, 0], adjacency[:, 1]
    na, nb = node[ta], node[tb]
    cut = (na != nb) & (na >= 0) & (nb >= 0)
    ta, tb, na, nb = ta[cut], tb[cut], na[cut], nb[cut]
    el = np.linalg.norm(np.diff(v[mesh.face_adjacency_edges[cut]], axis=1)[:, 0], axis=1)
    convex = np.where(((center[tb] - center[ta]) * normal[ta]).sum(1) < 0, 1.0, -1.0)
    lo, hi = np.minimum(na, nb), np.maximum(na, nb)
    key = lo * n + hi
    uniq, inv = np.unique(key, return_inverse=True)
    border = np.bincount(inv, weights=el)
    convexity = np.bincount(inv, weights=el * convex) / np.maximum(border, 1e-12)
    edges = np.stack([uniq // n, uniq % n], 1)
    static[:, 15] = (np.bincount(edges[:, 0], border, n) + np.bincount(edges[:, 1], border, n)) / diag
    tangent_pairs = _tangent_merged(brep)
    tangent = np.array([(int(ids[a]), int(ids[b])) in tangent_pairs for a, b in edges], bool)
    return {"ids": ids, "diag": np.float64(diag), "types": types, "static": static, "points": points,
            "dir": geo["dir"].astype(np.float32), "anchor": geo["anchor"].astype(np.float32),
            "fit": geo["fit"], "plane": geo["plane"], "axial": geo["axial"],
            "edges": edges.astype(np.int64), "convexity": convexity.astype(np.float32), "tangent": tangent,
            "relations": relation_labels(brep, ids, diag, cfg)}


# ---------- признаки пар ----------

def spd(n: int, edges: np.ndarray, max_spd: int) -> np.ndarray:
    """Кратчайший путь по графу; дальше `max_spd` и недостижимые — `max_spd + 1`."""
    if len(edges) == 0:
        out = np.full((n, n), max_spd + 1)
    else:
        a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        out = shortest_path(a, directed=False, unweighted=True)
        out = np.where(np.isfinite(out), np.minimum(out, max_spd + 1), max_spd + 1)
    np.fill_diagonal(out, 0)
    return out.astype(np.int64)


def pair_features(g: dict, edges: np.ndarray, tangent: np.ndarray, convexity: np.ndarray) -> np.ndarray:
    """`(N, N, PAIR_DIM)`: касание и выпуклость у соседей; |cos| между нормалями или осями; знак cos у
    пары плоскостей; расстояние между плоскостями и между осями / диагональ; маска «у обоих есть
    подгонка». Без подгонки геометрия обнуляется (README, «поправка к вниманию»)."""
    n = len(g["ids"])
    d, x = g["dir"].astype(np.float64), g["anchor"].astype(np.float64)
    diag = float(g["diag"])
    fit = g["fit"][:, None] & g["fit"][None]
    has_dir = np.linalg.norm(d, axis=1) > 0
    both_dir = has_dir[:, None] & has_dir[None]
    planes = g["plane"][:, None] & g["plane"][None]
    axes = g["axial"][:, None] & g["axial"][None]
    cos = d @ d.T
    off = x[None, :, :] - x[:, None, :]
    along_i = np.einsum("ijk,ik->ij", off, d)
    plane_dist = 0.5 * (np.abs(along_i) + np.abs(along_i.T)) / diag
    cross = np.cross(d[:, None, :], d[None, :, :])
    cn = np.linalg.norm(cross, axis=-1)
    skew = np.abs(np.einsum("ijk,ijk->ij", off, cross)) / np.maximum(cn, 1e-12)
    to_axis = np.linalg.norm(off - along_i[..., None] * d[:, None, :], axis=-1)
    axis_dist = np.where(cn > 1e-6, skew, to_axis) / diag
    out = np.zeros((n, n, PAIR_DIM), np.float32)
    a, b = edges[:, 0], edges[:, 1]
    out[a, b, 0] = out[b, a, 0] = tangent
    out[a, b, 1] = out[b, a, 1] = convexity
    out[..., 2] = np.abs(cos) * both_dir
    out[..., 3] = np.sign(cos) * planes
    out[..., 4] = plane_dist * planes
    out[..., 5] = axis_dist * axes
    out[..., 6] = fit
    idx = np.arange(n)
    out[idx, idx, 3:6] = 0
    return out


# ---------- метки шагов ----------

@dataclass
class Support:
    node: int                         # опора, у скругления и фаски — его полоса («затравка»)
    to: frozenset[int]                # узлы «до k»
    depth: float | None               # глубина / диагональ при «без поверхности»
    direction: int                    # ±1 относительно наружной нормали опоры


@dataclass
class Op:
    key: str
    kind: int                         # номер в OP_TYPES
    boolean: int
    nodes: frozenset[int]
    faces: np.ndarray                 # (N,) класс грани для этой операции, FACE_CLASSES
    supports: list[Support]


class Body:
    """Тело с метками в номерах узлов графа. Скругления и фаски — обычные операции цикла (решено
    пользователем 2026-10-25): их полосы — грани операции, затравка указателя — любая полоса; узлы,
    потерянные сканером, выброшены из меток (README, решено 2026-10-24)."""

    def __init__(self, g: dict, labels: dict, cfg=OCFG.data):
        self.uid = str(labels["uid"])
        self.g = g
        ids = g["ids"]
        self.n = n = len(ids)
        idx = {int(m): i for i, m in enumerate(ids)}
        types = g["types"]
        self.ops: dict[str, Op] = {}
        for u, o in labels["ops"].items():
            if o["type"] not in OP_TYPES:
                continue
            nodes = frozenset(idx[x] for x in o["nodes"] if x in idx)
            if not nodes:
                continue
            kind = OP_TYPES.index(o["type"])
            sides = {idx[x] for x in o.get("sides", o["nodes"]) if x in idx}
            faces = np.zeros(n, np.int64)
            faces[list(nodes)] = FACE_CLASSES.index("торец")
            faces[list(sides & nodes)] = FACE_CLASSES.index("стенка")
            sups = {}
            if o["type"] in FILLET_OPS:
                sups = {i: Support(i, frozenset(), None, 1) for i in sorted(nodes)}
            for s in o.get("supports", []):
                i = idx.get(s["node"])
                if i is None or i in sups:
                    continue
                to = frozenset(idx[k] for k in s.get("to") or [] if k in idx and idx[k] != i)
                depth = s["depth_mm"] / float(g["diag"]) if s.get("depth_mm") is not None and not s["to"] else None
                sups[i] = Support(i, to, depth, int(s.get("dir", 1)))
            self.ops[u] = Op(u, kind, BOOLEANS.index(o["boolean"] or "add"), nodes, faces, list(sups.values()))
        self.preds = {u: set() for u in self.ops}
        for a, b in labels["deps"]:
            if a in self.ops and b in self.ops:
                self.preds[b].add(a)
        # скругления и фаски: с любой операцией, чьи узлы соседствуют, — в порядке истории (как правило
        # добавления и выреза в dataset/fusion_opseq.py)
        nbrs = [set() for _ in range(n)]
        for a, b in g["edges"]:
            nbrs[a].add(int(b))
            nbrs[b].add(int(a))
        order = {u: labels["ops"][u]["timeline_index"] for u in self.ops}
        keys = sorted(self.ops, key=order.get)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if OP_TYPES[self.ops[a].kind] not in FILLET_OPS and OP_TYPES[self.ops[b].kind] not in FILLET_OPS:
                    continue
                na = self.ops[a].nodes
                if any(m in na or nbrs[m] & na for m in self.ops[b].nodes):
                    self.preds[b].add(a)
        plane, axial = PRIMITIVE_TYPES.index("plane"), [PRIMITIVE_TYPES.index(t) for t in REVOLVE_SUPPORTS]
        self.support_masks = np.stack([types == plane, np.isin(types, axial), np.ones(n, bool), np.ones(n, bool)])
        # вспомогательные цели до цикла: роль узла и «оба узла — грани одной операции»
        self.roles = np.zeros((n, len(ROLES)), np.float32)
        self.same_op = np.zeros((n, n), bool)
        for op in self.ops.values():
            nodes = sorted(op.nodes)
            if OP_TYPES[op.kind] in FILLET_OPS:
                self.roles[nodes, 2] = 1
            else:
                self.roles[np.flatnonzero(op.faces == FACE_CLASSES.index("стенка")), 0] = 1
                self.roles[np.flatnonzero(op.faces == FACE_CLASSES.index("торец")), 1] = 1
            self.same_op[np.ix_(nodes, nodes)] = True

    def available(self, done: list[str]) -> list[str]:
        return [u for u in self.ops if u not in done and self.preds[u] <= set(done)]

    def state(self, done: list[str]) -> np.ndarray:
        """Состояние узлов `(N, STATE_DIM)` по меткам: объяснён, тип и булева объяснившей операции,
        1 / (1 + сколько шагов назад)."""
        out = np.zeros((self.n, STATE_DIM), np.float32)
        for k, u in enumerate(done):
            op = self.ops[u]
            nodes = list(op.nodes)
            out[nodes] = 0
            out[nodes, 0] = 1
            out[nodes, 1 + op.kind] = 1
            out[nodes, 1 + len(OP_TYPES) + op.boolean] = 1
            out[nodes, -1] = 1 / (len(done) - k)
        return out

    def entries(self, done: list[str]) -> list[tuple[str, Support]]:
        """Допустимые сейчас пары (операция, опора): операция доступна, опора допустимого для её типа узла."""
        return [(u, s) for u in self.available(done) for s in self.ops[u].supports
                if self.support_masks[self.ops[u].kind][s.node]]


def full_graph(body: Body, max_spd: int) -> tuple[np.ndarray, np.ndarray]:
    g = body.g
    return pair_features(g, g["edges"], g["tangent"], g["convexity"]), spd(body.n, g["edges"], max_spd)


# ---------- датасет ----------

def design(key: str) -> str:
    """Дизайн по uid или ключу `<корпус>/<uid>`: номера дизайнов у Fusion Seg и реконструкции общие, синтетика
    названа по дизайну основы — один дизайн во всех корпусах попадает в одну часть разбиения."""
    return key.rsplit("/", 1)[-1].rsplit("_", 1)[0]


def split_uids(uids: list[str], cfg=OCFG.data) -> dict[str, list[str]]:
    """Разделение по дизайнам (README, «данные»): доля по хэшу дизайна, не зависит от состава списка."""
    out = {"train": [], "val": [], "test": []}
    for u in uids:
        h = int(hashlib.md5(design(u).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["test" if h < cfg.test_frac else "val" if h < cfg.test_frac + cfg.val_frac else "train"].append(u)
    return out


def load_graph(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class Bodies(torch.utils.data.Dataset):
    def __init__(self, keys: list[str], cfg=OCFG.data):
        self.cfg = cfg
        self.uids = [k for k in keys if corpus_of(k)[0].graph_path(corpus_of(k)[1]).exists()]
        self.nodes = node_counts(self.uids)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        c, u = corpus_of(self.uids[i])
        body = Body(load_graph(c.graph_path(u)), c.labels(u, self.cfg), self.cfg)
        pf, sp = full_graph(body, self.cfg.max_spd)
        return {"body": body, "pair": pf, "spd": sp}


def node_counts(keys: list[str]) -> np.ndarray:
    """Число узлов тел; кэш `<графы корпуса>/nodes.json` дополняется новыми телами."""
    out = {}
    for name in sorted({k.split("/", 1)[0] for k in keys}):
        c = CORPORA[name]
        path = c.root / c.graphs / "nodes.json"
        known = json.loads(path.read_text()) if path.exists() else {}
        uids = [k.split("/", 1)[1] for k in keys if k.startswith(name + "/")]
        new = [u for u in uids if u not in known]
        for u in new:
            with np.load(c.graph_path(u)) as z:
                known[u] = len(z["ids"])
        if new:
            path.write_text(json.dumps(known))
        out |= {f"{name}/{u}": known[u] for u in uids}
    return np.array([out[k] for k in keys], np.int64)


class BudgetBatches(torch.utils.data.Sampler):
    """Батчи до `batch` тел, но с `B · N_max² <= pairs`. Тела близкого размера идут вместе
    (сортировка внутри окна из 50 батчей), порядок батчей случайный и новый на каждой эпохе."""

    def __init__(self, nodes: np.ndarray, batch: int, pairs: int, shuffle: bool, seed: int = 0):
        self.nodes, self.batch, self.pairs, self.shuffle, self.seed, self.epoch = nodes, batch, pairs, shuffle, seed, 0
        self.size = len(self._batches(0))

    def _batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng((self.seed, epoch))
        order = rng.permutation(len(self.nodes)) if self.shuffle else np.arange(len(self.nodes))
        out = []
        for s in range(0, len(order), 50 * self.batch):
            chunk = order[s:s + 50 * self.batch]
            cur, top = [], 0
            for i in chunk[np.argsort(self.nodes[chunk], kind="stable")]:
                m = max(top, int(self.nodes[i]))
                if cur and (len(cur) == self.batch or (len(cur) + 1) * m * m > self.pairs):
                    out.append(cur)
                    cur, m = [], int(self.nodes[i])
                cur.append(int(i))
                top = m
            if cur:
                out.append(cur)
        return [out[k] for k in rng.permutation(len(out))] if self.shuffle else out

    def __iter__(self):
        batches = self._batches(self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return self.size


def collate(items: list[dict]) -> dict:
    """Узлы дополняются до наибольшего в батче; `mask` — настоящие узлы."""
    b = len(items)
    n = max(it["body"].n for it in items)
    out = {"bodies": [it["body"] for it in items], "mask": torch.zeros(b, n, dtype=torch.bool)}
    shapes = {"static": (STATIC_DIM,), "points": items[0]["body"].g["points"].shape[1:]}
    for k, shape in shapes.items():
        out[k] = torch.zeros((b, n, *shape))
    out["pair"] = torch.zeros(b, n, n, PAIR_DIM)
    out["spd"] = torch.zeros(b, n, n, dtype=torch.long)
    out["relations"] = torch.zeros(b, n, n, len(RELATIONS))
    out["roles"] = torch.zeros(b, n, len(ROLES))
    out["same_op"] = torch.zeros(b, n, n)
    for j, it in enumerate(items):
        body = it["body"]
        m = body.n
        out["mask"][j, :m] = True
        out["static"][j, :m] = torch.from_numpy(body.g["static"])
        out["points"][j, :m] = torch.from_numpy(body.g["points"].astype(np.float32))
        out["relations"][j, :m, :m] = torch.from_numpy(body.g["relations"].astype(np.float32))
        out["roles"][j, :m] = torch.from_numpy(body.roles)
        out["same_op"][j, :m, :m] = torch.from_numpy(body.same_op.astype(np.float32))
        for k in ("pair", "spd"):
            out[k][j, :m, :m] = torch.from_numpy(it[k])
    return out
