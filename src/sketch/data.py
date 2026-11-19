"""Данные 2D сети сечения: детали с сечениями (`scripts/sketch/build_sections.py`), батч — плоская
склейка цепочек без паддинга.

Признаки отрезка — как вход 3D сети: координаты середины (от среднего точек цепочки, доля
диагонали × COORD_SCALE) и нормаль треугольника скана в системе плоскости (nx, ny — в плоскости,
nz — наклон поверхности детали к плоскости сечения: единственное, что сеть знает о 3D). Форму
свёртка видит по разностям с соседями. На обучении — случайный поворот сечения в плоскости, как
`rotate` 3D сети (решено пользователем 2026-10-25).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import SCFG

SEG_DIM = 5
# Координаты середины отрезка — доля диагонали × COORD_SCALE: ±5 по детали, соседи ~0.05. Выбрано.
COORD_SCALE = 10.0


def chain_slices(z) -> list[tuple[slice, slice, bool, int]]:
    """(точки, отрезки, замкнута, номер сечения) на цепочку файла деталей."""
    po = np.r_[0, np.cumsum(z["chain_points"])]
    so = np.r_[0, np.cumsum(z["chain_segments"])]
    return [(slice(po[i], po[i + 1]), slice(so[i], so[i + 1]), bool(z["chain_closed"][i]), int(z["chain_sec"][i]))
            for i in range(len(po) - 1)]


def segment_features(points: np.ndarray, normals: np.ndarray, closed: bool, diag: float) -> np.ndarray:
    """`(отрезков, 5)`: середина отрезка (x, y) и нормаль (nx, ny, nz)."""
    n = len(normals)
    b = np.roll(points, -1, 0)[:n] if closed else points[1:]
    mid = ((points[:n] + b) / 2 - points.mean(0)) / diag * COORD_SCALE
    return np.concatenate([mid, normals.astype(np.float64)], 1).astype(np.float32)


def crop(points, normals, seg_type, boundary, tangent, closed, max_len, rng):
    """Окно из `max_len` отрезков со случайного начала: замкнутая цепочка становится открытой."""
    n = len(normals)
    if n <= max_len:
        return points, normals, seg_type, boundary, tangent, closed
    s = int(rng.integers(n if closed else n - max_len + 1))
    seg = (np.arange(max_len) + s) % n
    pts = (np.arange(max_len + 1) + s) % len(points)
    b = boundary[pts].copy()
    b[[0, -1]] = -1
    return points[pts], normals[seg], seg_type[seg], b, tangent[pts], False


def rotate(points: np.ndarray, normals: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Поворот сечения в его плоскости: точки и плоская часть нормалей."""
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]], np.float32)
    nrm = normals.astype(np.float32).copy()
    nrm[:, :2] = nrm[:, :2] @ r.T
    return (points @ r.T).astype(points.dtype), nrm


class Parts(torch.utils.data.Dataset):
    """Деталь — все её цепочки; на обучении длинные режутся окном."""

    def __init__(self, uids: list[str], train: bool, cfg=SCFG.data):
        self.root, self.cfg, self.train = Path(cfg.root) / "sections", cfg, train
        self.uids = [u for u in uids if (self.root / f"{u}.npz").exists()]
        self.segments = segment_counts(self.uids, self.root)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        z = np.load(self.root / f"{self.uids[i]}.npz")
        diag = float(z["diag"])
        rng = np.random.default_rng() if self.train else None
        angle = rng.uniform(0, 2 * np.pi) if self.train else 0.0
        chains = []
        for ps, ss, closed, sec in chain_slices(z):
            item = (z["points"][ps], z["normals"][ss], z["seg_type"][ss], z["boundary"][ps], z["tangent"][ps], closed)
            if self.train:
                item = crop(*item, self.cfg.max_len, rng)
            pts, nrm, st, bd, tg, cl = item
            if self.train:
                pts, nrm = rotate(pts, nrm, angle)
            chains.append({"feat": segment_features(pts, nrm, cl, diag), "type": st.astype(np.int64),
                           "boundary": bd.astype(np.int64), "tangent": tg.astype(np.int64), "closed": cl,
                           "points": pts, "sec": sec})
        return {"uid": self.uids[i], "diag": diag, "chains": chains}


def segment_counts(uids: list[str], root: Path) -> np.ndarray:
    """Отрезков на деталь, кэш в `<root>/segments.json` (дописывается)."""
    path = root / "segments.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [u for u in uids if u not in cache]
    for u in missing:
        cache[u] = int(np.load(root / f"{u}.npz")["chain_segments"].sum())
    if missing:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")             # свой файл: два обучения пишут кэш разом
        tmp.write_text(json.dumps(cache))
        tmp.rename(path)
    return np.array([cache[u] for u in uids])


class SegmentBatches(torch.utils.data.Sampler):
    """Детали в батч, пока сумма отрезков не больше `budget` (деталь больше бюджета — одна)."""

    def __init__(self, segments: np.ndarray, budget: int, shuffle: bool, seed: int = 0):
        self.segments, self.budget, self.shuffle, self.seed, self.epoch = segments, budget, shuffle, seed, 0

    def _batches(self) -> list[list[int]]:
        order = (np.random.default_rng(self.seed + self.epoch).permutation(len(self.segments)) if self.shuffle
                 else np.arange(len(self.segments)))
        out, cur, total = [], [], 0
        for i in order:
            s = int(self.segments[i])
            if cur and total + s > self.budget:
                out.append(cur)
                cur, total = [], 0
            cur.append(int(i))
            total += s
        if cur:
            out.append(cur)
        return out

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())


def collate(items: list[dict]) -> dict:
    """Плоская склейка цепочек. Соседи отрезка на расстоянии d — по `neighbors` (по кругу у
    замкнутых, -1 за концом открытой). Стык j цепочки — между отрезками `junction_prev` и
    `junction_next`; у открытой концы не стыки."""
    feats, types, starts, lengths, closed = [], [], [], [], []
    j_prev, j_next, j_bd, j_tg, j_chain, j_point = [], [], [], [], [], []
    chains_meta = []
    off = 0
    for it in items:
        for c in it["chains"]:
            n = len(c["feat"])
            feats.append(c["feat"])
            types.append(c["type"])
            starts.append(off)
            lengths.append(n)
            closed.append(c["closed"])
            k = len(c["points"])
            if c["closed"]:
                pts = np.arange(n)
                prv, nxt = (pts - 1) % n, pts
            else:
                pts = np.arange(1, n)
                prv, nxt = pts - 1, pts
            j_prev.append(prv + off)
            j_next.append(nxt + off)
            j_bd.append(c["boundary"][pts])
            j_tg.append(c["tangent"][pts])
            j_chain.append(np.full(len(pts), len(chains_meta)))
            j_point.append(pts)
            chains_meta.append({"uid": it["uid"], "diag": it["diag"], "sec": c["sec"], "points": c["points"],
                                "closed": c["closed"], "segments": n, "junctions": k})
            off += n
    return {"feat": torch.from_numpy(np.concatenate(feats)), "type": torch.from_numpy(np.concatenate(types)),
            "start": torch.tensor(starts), "length": torch.tensor(lengths), "closed": torch.tensor(closed),
            "junction_prev": torch.from_numpy(np.concatenate(j_prev)),
            "junction_next": torch.from_numpy(np.concatenate(j_next)),
            "boundary": torch.from_numpy(np.concatenate(j_bd)), "tangent": torch.from_numpy(np.concatenate(j_tg)),
            "junction_chain": torch.from_numpy(np.concatenate(j_chain)),
            "junction_point": torch.from_numpy(np.concatenate(j_point)), "chains": chains_meta}


def neighbors(start: torch.Tensor, length: torch.Tensor, closed: torch.Tensor, d: int) -> torch.Tensor:
    """`(T,)` номер отрезка на `d` дальше по цепочке (d < 0 — назад); -1 за концом открытой."""
    chain = torch.repeat_interleave(torch.arange(len(length), device=length.device), length)
    s, n, c = start[chain], length[chain], closed[chain]
    i = torch.arange(len(chain), device=length.device) - s + d
    wrapped = torch.remainder(i, n)
    return torch.where(c, s + wrapped, torch.where((i >= 0) & (i < n), s + i, -1))
