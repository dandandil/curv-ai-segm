"""Тело из поверхностей: по рёбрам с соседями, а если не вышло — разбиением.

По рёбрам (`brep.build`) каждая грань ограничена только кривыми с соседними
кластерами: касательный стык — аналитической линией касания, соосная пара —
окружностью, остальные — ветвью пересечения, ближайшей к общей границе на
меше. Грани чистые, касания точные. Но если хоть одно ребро не построилось,
оболочка не замыкается. Тогда — разбиение (`arrangement.build`): оно замыкает
тело всегда, ценой граней, поделённых следами чужих поверхностей (README,
«Сборка разбиением»).
"""
from __future__ import annotations

import numpy as np
import trimesh

from . import arrangement as A
from . import brep as B
from .surfaces import Surface
from .topology import Boundaries

# Тело по рёбрам принимается, если его объём расходится с мешем не больше этого:
# тот же порог, что у оценки «собрана» (scripts/evaluate_brep.py).
VOLUME_REL = 0.02


def solid(mesh: trimesh.Trimesh, labels: np.ndarray, surfaces: dict[int, Surface], relations: list,
          bounds: Boundaries, cfg):
    """Поверхности после совместной подгонки -> тело и сводка с полем «способ»."""
    shape, report = B.build(mesh, labels, surfaces, relations, bounds, cfg)
    if shape is not None and report.get("тело") and report.get("валидна"):
        volume = abs(A._volume_props(shape).Mass())
        if abs(volume - abs(mesh.volume)) <= VOLUME_REL * abs(mesh.volume):
            return shape, {"способ": "по рёбрам", **report}
    edges = {f"по рёбрам: {k}": v for k, v in report.items()}
    shape, report = A.build(mesh, labels, surfaces)
    return shape, {"способ": "разбиением", **report, **edges}
