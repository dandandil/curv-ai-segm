"""Оффскрин-рендер меша с раскраской по значению.

Рендерим настоящий меш, а не облако точек: плотность и форма треугольников —
это то, что мы и хотим видеть глазами, а точками они не читаются.

Требует `PYOPENGL_PLATFORM=egl` в окружении — так рендерится всё в этом
проекте, дисплея на машине нет.
"""
from __future__ import annotations

import os

import numpy as np
import trimesh

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402  (после установки платформы)

YFOV = np.pi / 4.0
# Свет: цвет на картинке должен совпадать с цветом в легенде, поэтому
# направленный свет слабый, а сцена почти равномерная. [измерено на сфере
# одного цвета, медиана пикселей против заданного #1f77b4: при фоне 0.45 и
# направленном 2.5 — отклонение 12%, при 0.8 и 0.6 — 6%, при равномерном
# свете — 1%.]
AMBIENT = 0.8
DIRECTIONAL = 0.6
# Показатель sRGB-кривой в приближении степенной функцией.
SRGB_GAMMA = 2.2


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Поза камеры pyrender: смотрит вдоль своей -Z."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = np.cross(right, fwd)
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


def face_colors(values: np.ndarray, faces: np.ndarray, cmap: str = "viridis",
                lo: float = 2.0, hi: float = 98.0,
                per_face: bool = False) -> np.ndarray:
    """RGBA на треугольник из значений на вершинах (или сразу на треугольниках).

    Хвосты обрезаются по перцентилям: одна вершина на остром ребре иначе
    забирает всю шкалу и остальное выглядит одноцветным.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    v = np.asarray(values, float)
    v = v if per_face else v[faces].mean(axis=1)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    norm = mcolors.Normalize(vmin=a, vmax=max(b, a + 1e-12))
    return (mpl.colormaps[cmap](norm(v)) * 255).astype(np.uint8)


def _linear(colors: np.ndarray) -> np.ndarray:
    """Цвет в линейное пространство: pyrender считает освещение в нём, а на
    выходе кодирует в sRGB. Без этого гамма применяется один раз лишний, и
    #1f77b4 выходит на картинке как #62b4da — цвет перестаёт совпадать с
    легендой."""
    c = np.asarray(colors, float) / 255.0
    c[..., :3] **= SRGB_GAMMA
    return (c * 255).astype(np.uint8)


def render(mesh: trimesh.Trimesh, colors: np.ndarray | None = None,
           size: int = 760, azim: float = 40.0, elev: float = 22.0,
           margin: float = 1.15) -> np.ndarray:
    """Один кадр меша. `colors` — RGBA на треугольник, как из `face_colors`."""
    m = mesh.copy()
    if colors is not None:
        m.visual.face_colors = _linear(colors)
    ctr = m.vertices.mean(axis=0)
    radius = float(np.linalg.norm(m.vertices - ctr, axis=1).max())
    dist = margin * radius / np.tan(YFOV / 2.0)
    a, e = np.radians(azim), np.radians(elev)
    eye = ctr + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pose = _look_at(eye, ctr)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[AMBIENT] * 3)
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    scene.add(pyrender.PerspectiveCamera(yfov=YFOV), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=DIRECTIONAL), pose=pose)
    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        return renderer.render(scene)[0]
    finally:
        renderer.delete()
