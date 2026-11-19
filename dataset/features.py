"""Признаки на треугольник рабочего меша: инварианты струи Monge 4-й степени.

Подгонка струи — копия из прототипа `~/scan2cad/colouring_test` в `jet.py`, с
соседями снаружи: окрестность здесь строится по связности подложки, а не k
ближайшими по расстоянию (README, «Окрестность струи»).

Струя считается на подложке, а не на рабочем меше. Причина измерена: на
рабочем меше радиус окрестности ограничен снизу его же шагом: при 10 тысячах
треугольников (шаг ~0.4 мм) он не опускается ниже 0.9 мм, окрестность залезает
на соседние грани, и плоскость читается как поверхность радиуса 1 мм вместо
нуля. Со счётом по подложке та же плоскость даёт 0.02. При 30 тысячах (шаг
~0.21 мм) разрыв меньше, но по разбросу внутри грани подложка всё равно
выигрывает вдвое.

Масштаб задаётся радиусом окрестности как долей корня из площади детали, а не
числом соседей: число соседей — величина производная от плотности сетки.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.config import CFG

from . import jet as J

_C = CFG.features

# Значения и их обоснование — в configs/default.yaml, секция features.
SCALES_REL = _C.scales_rel
MIN_K = _C.min_k
TARGET_K = _C.target_k
BALL_GROW = _C.ball_grow
BALL_GROW_STEPS = _C.ball_grow_steps
BALL_CHUNK = _C.ball_chunk
CHUNK_BUDGET = _C.chunk_budget


def face_frames(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Центроиды и единичные нормали граней — по одной строке на КАЖДУЮ грань.

    Нормаль грани в норме не вырождается, но на осколках после ремеша
    встречается нулевая. Такие грани нельзя выбрасывать: тогда матрица
    признаков перестаёт соответствовать граням меша по индексам, и разъезжается
    это молча. Вместо этого нулевая нормаль заменяется на нормаль соседа по
    ребру, а если и та вырождена — на ось Z; третьим значением возвращается
    маска подменённых.
    """
    p = mesh.triangles.mean(axis=1)
    n = np.asarray(mesh.face_normals, float).copy()
    ln = np.linalg.norm(n, axis=1)
    bad = ln <= 1e-9
    if bad.any():
        fixed = np.zeros(len(n), bool)
        for a, b in mesh.face_adjacency:
            for u, v in ((a, b), (b, a)):
                if bad[u] and not bad[v] and not fixed[u]:
                    n[u] = n[v]
                    fixed[u] = True
        n[bad & ~fixed] = (0.0, 0.0, 1.0)
        ln = np.linalg.norm(n, axis=1)
    return p, n / ln[:, None], bad


def adjacency_table(mesh: trimesh.Trimesh) -> np.ndarray:
    """Смежность треугольников таблицей `(n, w)`, недостающие -1. Ширина больше
    трёх только у неманифолдных рёбер."""
    n = len(mesh.faces)
    adj = mesh.face_adjacency
    if not len(adj):
        return np.full((n, 1), -1, np.int64)
    a = np.r_[adj[:, 0], adj[:, 1]]
    b = np.r_[adj[:, 1], adj[:, 0]]
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    pos = np.arange(len(a)) - np.searchsorted(a, a)
    table = np.full((n, pos.max() + 1), -1, np.int64)
    table[a, pos] = b
    return table


def _not_in(keys: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Элементы `keys`, которых нет в `ref`; оба отсортированы и без повторов."""
    if not len(ref) or not len(keys):
        return keys
    i = np.searchsorted(ref, keys).clip(max=len(ref) - 1)
    return keys[ref[i] != keys]


def surface_ball(table: np.ndarray, points: np.ndarray, centers: np.ndarray,
                 h: float, chunk: int = BALL_CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Пары (номер центра в `centers`, треугольник): часть шара радиуса `h`,
    связная с центром по смежности. Обратная сторона стенки достижима только
    через край, а край вне шара, поэтому в окрестность она не попадает.

    Обход в ширину слоями, для всех центров пачки разом. Пара кодируется одним
    числом `номер центра · n + треугольник`. Пройденные хранить не нужно: в
    неориентированном графе сосед узла слоя t лежит в слое t-1, t или t+1,
    поэтому новый слой — это соседи текущего минус два последних слоя.
    """
    n = len(points)
    h2 = h * h
    # Пачка мелкая, чтобы ключ влез в int32: сортировка вдвое быстрее.
    dtype = np.int32 if chunk * n < 2 ** 31 else np.int64
    table = table.astype(dtype, copy=False)
    rows_out, ids_out = [], []
    for s in range(0, len(centers), chunk):
        c = centers[s:s + chunk]
        layer = np.arange(len(c), dtype=dtype) * dtype(n) + c.astype(dtype)
        prev = layer[:0]
        parts = [layer]
        while len(layer):
            r = np.repeat(layer // dtype(n), table.shape[1])
            nb = table[layer % dtype(n)].ravel()
            ok = nb >= 0
            r, nb = r[ok], nb[ok]
            dd = points[nb] - points[c[r]]
            ok = np.einsum("ij,ij->i", dd, dd) <= h2
            keys = r[ok] * dtype(n) + nb[ok]
            keys.sort()
            if len(keys):
                keys = keys[np.r_[True, keys[1:] != keys[:-1]]]
            keys = _not_in(_not_in(keys, layer), prev)
            prev, layer = layer, keys
            parts.append(keys)
        keys = np.concatenate(parts)
        rows_out.append((keys // dtype(n)).astype(np.int64) + s)
        ids_out.append((keys % dtype(n)).astype(np.int64))
    return np.concatenate(rows_out), np.concatenate(ids_out)


# Ключ пары — 53 бита хеша, равномерен на [0, 2^53).
_KEY_SPAN = float(2 ** 53)
# Запас порога отбора: сколько пар держать на центр сверх нужных, в долях нужного.
_KEEP_MARGIN = 1.5


def _pair_key(rows: np.ndarray, ids: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Псевдослучайный ключ пары (центр, сосед), сам центр — -1. Хеш, а не
    генератор: выборка соседей не зависит от того, какие ещё центры считаются в
    том же вызове."""
    x = (centers[rows].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
         ^ ids.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    x ^= x >> np.uint64(31)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(29)
    key = (x >> np.uint64(11)).astype(np.float64)
    key[ids == centers[rows]] = -1.0
    return key


def _first_neighbors(rows: np.ndarray, ids: np.ndarray, centers: np.ndarray, counts: np.ndarray,
                     need: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Соседи по центрам, внутри — сам центр первым, дальше по ключу пары; у каждого
    центра в его блоке не меньше `need` первых по ключу. Возвращает соседей и
    начало блока каждого центра.

    Сортировать все пары незачем: в подгонку идут первые `target_k` (80), а в шаре
    крупного масштаба их до тысячи, и полная сортировка стоила половину времени
    струи. Ключ равномерен, поэтому пары с ключом выше порога, рассчитанного по
    размеру шара с запасом, в первые `need` попасть не могут — их отбрасывают до
    сортировки. Центру, у которого под порог попало меньше `need`, берутся все пары:
    выбор совпадает с полной сортировкой до бита.
    """
    key = _pair_key(rows, ids, centers)
    frac = np.minimum(1.0, (_KEEP_MARGIN * need + 16) / np.maximum(counts, 1))
    keep = key < frac[rows] * _KEY_SPAN
    short = np.bincount(rows[keep], minlength=len(counts)) < need
    keep |= short[rows]
    rows, ids, key = rows[keep], ids[keep], key[keep]
    order = np.lexsort((key, rows))
    kept = np.bincount(rows, minlength=len(counts))
    return ids[order], np.r_[0, np.cumsum(kept)[:-1]]


def jet_at(points: np.ndarray, normals: np.ndarray, table: np.ndarray,
           centers: np.ndarray, h: float, target_k: int = TARGET_K
           ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Инварианты струи в `centers` с окрестностью радиуса `h` по поверхности.

    В шаре меньше `MIN_K` точек — радиус растёт: у струи 15 коэффициентов.
    Больше `target_k` в подгонку не идёт — их избыток стоит времени, а не
    точности. Центры с одинаковым числом соседей подгоняются одной пачкой,
    так что соседи не повторяются.

    Вторым значением — что вышло: медиана и минимум соседей, доля центров с
    расширенным шаром и признак промаха (соседей меньше `MIN_K` и после
    расширения).
    """
    rows, ids = surface_ball(table, points, centers, h)
    counts = np.bincount(rows, minlength=len(centers))
    radius = np.full(len(centers), h)
    for _ in range(BALL_GROW_STEPS):
        low = np.flatnonzero(counts < MIN_K)
        if not len(low):
            break
        # Мало соседей у центра — было мало и на всех прошлых шагах, поэтому
        # радиус у всех таких центров один.
        radius[low] *= BALL_GROW
        keep = counts[rows] >= MIN_K
        r2, i2 = surface_ball(table, points, centers[low], float(radius[low[0]]))
        rows, ids = np.r_[rows[keep], low[r2]], np.r_[ids[keep], i2]
        counts = np.bincount(rows, minlength=len(centers))
    k_used = np.minimum(counts, target_k)
    ids, start = _first_neighbors(rows, ids, centers, counts, k_used)
    out: dict[str, np.ndarray] = {}
    for k in np.unique(k_used):
        sel = np.flatnonzero(k_used == k)
        idx = ids[start[sel][:, None] + np.arange(k)[None, :]]
        coeffs, _, _ = J.monge_jet4(points, normals, centers[sel], idx,
                                    chunk=max(500, CHUNK_BUDGET // int(k)))
        for key, val in J.stage_scalars(coeffs).items():
            out.setdefault(key, np.zeros(len(centers)))[sel] = val
    info = {"target": float(h), "median_neighbors": int(np.median(counts)),
            "min_neighbors": int(counts.min()), "grown": float(np.mean(radius > h)),
            "missed": bool(counts.min() < MIN_K)}
    return out, info


def _principal(s: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Главные кривизны со знаком из H и K. Знак несёт вогнутость, поэтому
    берутся именно они, а не ‖S‖."""
    disc = np.sqrt(np.maximum(s["H"] ** 2 - s["K"], 0.0))
    return s["H"] + disc, s["H"] - disc


# Масштабы крупнее этого считаются на прорежённой подложке: в шар 0.05·L на подложке в
# 500 тысяч попадает ~4000 точек, перебор шара — сотни миллионов пар на деталь, а в подгонку
# всё равно идут TARGET_K. Подложка прорежается так, чтобы на самом крупном масштабе в шаре
# было столько же точек, сколько на COARSE_ABOVE на полной.
COARSE_ABOVE = 0.02


def _coarse_frames(substrate: trimesh.Trimesh, frames: list, ratio: float):
    """Точки, нормали, смежность прорежённой подложки и перенос на центроиды целевых мешей."""
    from . import remesh as R
    coarse = R.decimate(substrate, max(1000, int(len(substrate.faces) * ratio ** 2)))
    cp, cn, _ = face_frames(coarse)
    nearest = [cKDTree(cp).query(f[0], k=1)[1] for f in frames]
    centers = np.unique(np.concatenate(nearest))
    return cp, cn, adjacency_table(coarse), centers, [np.searchsorted(centers, near) for near in nearest]


def compute_many(substrate: trimesh.Trimesh, targets: list[trimesh.Trimesh],
                 scales: tuple[float, ...] = SCALES_REL,
                 ) -> tuple[list[np.ndarray], list[str], list[dict[str, float]]]:
    """Признаки для нескольких мешей от одной подложки.

    Струя считается на подложке ОДИН раз на масштаб и переносится на центроиды
    каждого целевого меша по ближайшему: несколько мешей получают
    согласованные значения, а струя не считается дважды.

    Каналы: координаты центроида, нормаль, и на каждый масштаб — главные
    кривизны со знаком, их норма ‖S‖ и норма ковариантной производной второй
    фундаментальной формы. Обезразмеривание характерной длиной: `kappa * L`,
    `‖∇II‖ * L^2`.

    Характерная длина L = sqrt(площади подложки) — ею задаётся и радиус
    окрестности, и обезразмеривание кривизны, чтобы обе величины жили в одной
    единице длины. Начало координат общее для всех целевых мешей. Центрировать
    каждый меш по своему центроиду нельзя: у адаптивного меша больше
    треугольников на кривых участках, среднее смещается, и две ветви увидели бы
    деталь в разных системах координат (измерено расхождение 0.04 в единицах
    канала при размахе ±0.9).

    Третьим значением возвращается по словарю на масштаб: сколько соседей
    вышло, у скольких центров расширялся шар и был ли промах.
    """
    ps, ns, _ = face_frames(substrate)
    tree = cKDTree(ps)
    frames = [face_frames(m) for m in targets]
    nearest = [tree.query(f[0], k=1)[1] for f in frames]
    # Струя нужна только в точках, откуда значения переносятся (~19% подложки);
    # соседей они берут из всей подложки, так что значения те же, что при счёте
    # везде.
    centers = np.unique(np.concatenate(nearest))
    where = [np.searchsorted(centers, near) for near in nearest]
    table = adjacency_table(substrate)
    origin = ps.mean(axis=0)                        # общее начало координат
    scale = float(np.sqrt(substrate.area))          # характерная длина детали

    cols: list[list[np.ndarray]] = [[] for _ in targets]
    names: list[str] = []
    for i, (pw, nw, _) in enumerate(frames):
        cols[i] += [(pw - origin) / scale, nw]
    names += ["cx", "cy", "cz", "nx", "ny", "nz"]

    info: list[dict[str, float]] = []
    coarse = None
    for rel in scales:
        if rel > COARSE_ABOVE:
            if coarse is None:
                coarse = _coarse_frames(substrate, frames, COARSE_ABOVE / max(scales))
            points, normals, tab, ctr, at = coarse
        else:
            points, normals, tab, ctr, at = ps, ns, table, centers, where
        s, meta = jet_at(points, normals, tab, ctr, rel * scale)
        meta["rel"] = float(rel)
        info.append(meta)
        k1, k2 = _principal(s)
        # ‖S‖ выводится из пары, но первый слой линеен и квадратный корень
        # выразить не может. Плюс пара упорядочена, и около сферы, где κ₁ ≈ κ₂,
        # порядок определяется шумом — норма от этого свободна.
        for val, nm, power in ((k1, "k1", 1), (k2, "k2", 1),
                               (np.hypot(k1, k2), "S", 1), (s["d2"], "dII", 2)):
            for i, idx in enumerate(at):
                cols[i].append((val[idx] * scale ** power)[:, None])
            names.append(f"{nm}@{rel}")
    return [np.concatenate(c, axis=1) for c in cols], names, info


def compute(substrate: trimesh.Trimesh, work: trimesh.Trimesh,
            scales: tuple[float, ...] = SCALES_REL) -> tuple[np.ndarray, list[str]]:
    """Признаки для одного меша. Обёртка над `compute_many`."""
    mats, names, _ = compute_many(substrate, [work], scales)
    return mats[0], names
