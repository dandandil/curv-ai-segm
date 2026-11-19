"""Аналитические поверхности: параметры, знаковое расстояние, проекции.

Параметры — плоский вектор float64, раскладка по типу:

    plane     n(3) d            n·x + d
    sphere    c(3) r            ‖x − c‖ − r
    cylinder  c(3) a(3) r       ρ − r
    cone      v(3) a(3) θ       ρ cos θ − z sin θ
    torus     c(3) a(3) R r     √((ρ − R)² + z²) − r

`z` и `ρ` — осевая и радиальная координаты относительно точки на оси (у конуса —
вершины) и оси `a`. Знак расстояния совпадает с естественной нормалью
поверхности OpenCASCADE того же типа, поэтому ориентация грани решается одним
скалярным произведением с нормалями меша.

Внутри расстояния направления нормируются, а их длина закрепляется условием
`gauge`: иначе у задачи было бы непрерывное семейство одинаковых решений и
система KKT вырождалась бы. По той же причине у цилиндра точка на оси
закреплена условием `a·c = 0`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.func import jacfwd

KINDS = ("plane", "sphere", "cylinder", "cone", "torus")
# Полуугол конуса, ближе которого к 0° или 90° это цилиндр или плоскость:
# так считают и подгонка (`fitting.fit_kind`), и разметка (`data.face_types`).
# Выбрано по полууглам истинных конусов [53 конические грани val после
# `fitting.fit_cone`: от 1° до 89.3°, в основном 40-50° (фаски) и 1-4°
# (уклоны); на 24474_2e83b11c_0 профили вращения, записанные в STEP конусами, —
# 0.07-0.47° и 87.1-89.6°: по форме цилиндры и плоские кольца].
CONE_LIMIT_DEG = 1.0
AXIAL = ("cylinder", "cone", "torus")


@dataclass
class Surface:
    kind: str
    params: np.ndarray
    rms: float


def _unit(v: Tensor) -> Tensor:
    return v / v.norm()


def _norm(v: Tensor) -> Tensor:
    # под корнем клип: производная нормы в нуле не определена
    return torch.sqrt((v ** 2).sum(-1).clamp_min(1e-24))


def distance(kind: str, p: Tensor, x: Tensor) -> Tensor:
    """Знаковое расстояние точек `(N, 3)` до поверхности."""
    if kind == "plane":
        return x @ _unit(p[:3]) + p[3]
    if kind == "sphere":
        return _norm(x - p[:3]) - p[3]
    a = _unit(p[3:6])
    u = x - p[:3]
    z = u @ a
    rho = _norm(u - z[:, None] * a)
    if kind == "cylinder":
        return rho - p[6]
    if kind == "cone":
        return rho * torch.cos(p[6]) - z * torch.sin(p[6])
    if kind == "torus":
        return torch.sqrt((rho - p[6]) ** 2 + z ** 2) - p[7]
    raise ValueError(kind)


def gauge(kind: str, p: Tensor) -> list[Tensor]:
    """Условия, снимающие непрерывную неоднозначность параметров."""
    if kind == "plane":
        return [p[:3] @ p[:3] - 1]
    if kind == "sphere":
        return []
    out = [p[3:6] @ p[3:6] - 1]
    if kind == "cylinder":
        out.append(p[:3] @ p[3:6])
    return out


def to_frame(kind: str, p: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    """Параметры в систему, где точки центрированы и отнесены к размеру.

    В миллиметрах якобиан смешивал бы длины порядка десятков с безразмерными
    направлениями, и обусловленность системы зависела бы от размера детали.
    """
    q = p.astype(np.float64).copy()
    if kind == "plane":
        q[3] = (p[3] + p[:3] @ origin) / scale
        return q
    q[:3] = (p[:3] - origin) / scale
    if kind in ("sphere", "cylinder"):
        q[-1] /= scale
    if kind == "torus":
        q[6:8] /= scale
    return q


def from_frame(kind: str, q: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    p = q.copy()
    if kind == "plane":
        p[3] = q[3] * scale - q[:3] @ origin
        return p
    p[:3] = q[:3] * scale + origin
    if kind in ("sphere", "cylinder"):
        p[-1] *= scale
    if kind == "torus":
        p[6:8] *= scale
    return p


def from_fit(kind: str, fit: dict) -> np.ndarray:
    """Результат `fits.py` -> вектор параметров."""
    if kind == "plane":
        n = np.asarray(fit["normal"], float)
        return np.r_[n, -n @ fit["point"]]
    if kind == "sphere":
        return np.r_[fit["center"], fit["radius"]]
    if kind == "cylinder":
        return np.r_[fit["center"], fit["axis"], fit["radius"]]
    if kind == "cone":
        return np.r_[fit["apex"], fit["axis"], np.radians(fit["half_angle"])]
    return np.r_[fit["center"], fit["axis"], fit["major"], fit["minor"]]


def rms(kind: str, p: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """Взвешенная RMS невязки."""
    d = distance(kind, torch.from_numpy(p), torch.from_numpy(x)).numpy()
    return float(np.sqrt((w * d ** 2).sum() / w.sum()))


def normals(kind: str, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Градиент расстояния в точках — нормаль поверхности со знаком OCC."""
    xt = torch.from_numpy(x).requires_grad_(True)
    distance(kind, torch.from_numpy(p), xt).sum().backward()
    return xt.grad.numpy()


def common_point(surfaces: list[Surface], x0: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Ближайшая к `x0` точка, где сходятся поверхности, по наименьшим квадратам.

    Шаг — псевдообратная якобиана: при одной-двух поверхностях задача
    недоопределена, и шаг минимальной нормы оставляет точку у старта, то есть
    проецирует её на поверхность или на кривую пересечения.
    """
    ps = [(s.kind, torch.from_numpy(s.params)) for s in surfaces]

    def residual(x: Tensor) -> Tensor:
        return torch.cat([distance(k, p, x[None]) for k, p in ps])

    x = torch.from_numpy(np.asarray(x0, np.float64))
    for _ in range(iterations):
        step = -torch.linalg.pinv(jacfwd(residual)(x)) @ residual(x)
        x = x + step
        if float(step.norm()) < 1e-12:
            break
    return x.numpy()
