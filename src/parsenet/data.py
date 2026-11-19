"""Облака точек для ParSeNet: чтение, аугментации и нормировка — как `src/dataset_segments.py`
и `src/augment_utils.py`.

Дословно, включая особенность: аугментация поворотом крутит точки, но не нормали.
Случайность — через переданный генератор.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ROOT

POINTS = ROOT / "dataset" / "data" / "parsenet_points"
EPS = np.finfo(np.float32).eps


class PointSplit:
    """Сплит через memmap: `points`, `normals`, `labels`, `prim`, `uids`."""

    def __init__(self, split: str, root: Path = POINTS):
        d = root / split
        self.points = np.load(d / "points.npy", mmap_mode="r")
        self.normals = np.load(d / "normals.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        self.prim = np.load(d / "prim.npy", mmap_mode="r")
        self.uids = (d / "uids.txt").read_text().split()

    def __len__(self) -> int:
        return len(self.uids)


def rotate_point_cloud(batch, rng):
    out = np.zeros(batch.shape, dtype=np.float32)
    for k in range(batch.shape[0]):
        a = rng.uniform() * 2 * np.pi
        r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        out[k] = np.dot(batch[k].reshape((-1, 3)), r)
    return out


def rotate_perturbation_point_cloud(batch, rng, angle_sigma=0.06, angle_clip=0.30):
    out = np.zeros(batch.shape, dtype=np.float32)
    for k in range(batch.shape[0]):
        a = np.clip(angle_sigma * rng.standard_normal(3), -angle_clip, angle_clip)
        rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
        ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
        rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
        out[k] = np.dot(batch[k].reshape((-1, 3)), rz @ ry @ rx)
    return out


def jitter_point_cloud(batch, rng, sigma=0.01, clip=0.05):
    return (batch + np.clip(sigma * rng.standard_normal(batch.shape), -clip, clip)).astype(np.float32)


def shift_point_cloud(batch, rng, shift_range=0.1):
    return (batch + rng.uniform(-shift_range, shift_range, (batch.shape[0], 1, 3))).astype(np.float32)


def random_scale_point_cloud(batch, rng, low=0.8, high=1.2):
    return (batch * rng.uniform(low, high, (batch.shape[0], 1, 1))).astype(np.float32)


AUGMENT = (rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud,
           random_scale_point_cloud, rotate_point_cloud)


def rotation_matrix_a_to_b(a, b):
    cos = np.dot(a, b)
    sin = np.linalg.norm(np.cross(b, a))
    u = a
    v = b - np.dot(a, b) * a
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(b, a)
    w = w / (np.linalg.norm(w) + EPS)
    f = np.stack([u, v, w], 1)
    g = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    try:
        return f @ g @ np.linalg.inv(f)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def align_and_scale(points, normals):
    """Малая главная ось — вдоль x, наибольший габарит — 1 (`align_canonical`)."""
    s, u = np.linalg.eig(points.T @ points)
    # Матрица симметрична, собственные значения вещественны: мнимая часть нулевая.
    s, u = s.real, u.real
    r = rotation_matrix_a_to_b(u[:, np.argmin(s)], np.array([1, 0, 0]))
    points = (r @ points.T).T
    normals = (r @ normals.T).T
    extent = np.max(points, 0) - np.min(points, 0)
    return (points / (np.max(extent) + EPS)).astype(np.float32), normals.astype(np.float32)


def batch(split: PointSplit, idx: np.ndarray, rng: np.random.Generator, n_sub: int,
          augment: bool, noise: float):
    """Батч обучения или валидации: `get_train`/`get_val` плюс подвыборка из train_parsenet.py."""
    points = np.array(split.points[idx], dtype=np.float32)
    points -= points.mean(1, keepdims=True)
    normals = np.array(split.normals[idx], dtype=np.float32)
    if augment:
        points = AUGMENT[rng.integers(len(AUGMENT))](points, rng)
    shift = np.clip(rng.standard_normal((1, points.shape[1], 1)) * noise, -noise, noise)
    points = (points + normals * shift).astype(np.float32)
    for j in range(len(idx)):
        points[j], normals[j] = align_and_scale(points[j], normals[j])
    keep = rng.permutation(points.shape[1])[:n_sub]
    labels = np.array(split.labels[idx][:, keep], dtype=np.int64)
    prim = np.array(split.prim[idx][:, keep], dtype=np.int64)
    return points[:, keep], normals[:, keep], labels, prim


def shape(split: PointSplit, i: int, rng: np.random.Generator, noise: float):
    """Форма для теста целиком: `normalize_points` из dataset_segments.py."""
    points = np.array(split.points[i], dtype=np.float32)
    normals = np.array(split.normals[i], dtype=np.float32)
    points = points - points.mean(0, keepdims=True)
    points = points + normals * np.clip(rng.standard_normal((points.shape[0], 1)) * noise, -noise, noise)
    points, normals = align_and_scale(points, normals)
    return points, normals, np.array(split.labels[i], dtype=np.int64), np.array(split.prim[i], dtype=np.int64)
