"""Дымовой тест сборки B-Rep на синтетике с известным ответом.

Проверяется то, что ломается молча: касание после совместной подгонки точное,
а не приблизительное; параметры при этом не уезжают от данных; куб и цилиндр
с истинной нарезкой собираются в замкнутое валидное тело.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import brep as B  # noqa: E402
from src import constraints as C  # noqa: E402
from src import fitting as F  # noqa: E402
from src import topology as T  # noqa: E402
from src.config import CFG  # noqa: E402
from src.surfaces import Surface  # noqa: E402

RADIUS = 2.0


def test_fillet_becomes_exactly_tangent():
    """Плоскость y=0 и цилиндр-скругление над ней: старт с ошибкой 3% по
    радиусу, после совместной подгонки касание точное, радиус — по данным."""
    rng = np.random.default_rng(0)
    plane_pts = np.c_[rng.uniform(-5, 5, 400), np.zeros(400), rng.uniform(0, 10, 400)]
    phi = rng.uniform(-np.pi, -np.pi / 2, 300)
    cyl_pts = np.c_[RADIUS * np.cos(phi), RADIUS + RADIUS * np.sin(phi), rng.uniform(0, 10, 300)]
    plane_pts[:, 1] += rng.normal(0, 1e-3, 400)

    surfaces = {0: Surface("plane", np.array([0.01, 1.0, 0.0, 0.02]), 0.0),
                1: Surface("cylinder", np.array([0.05, 2.1, 0.0, 0.0, 0.01, 1.0, 2.06]), 0.0)}
    samples = {0: (plane_pts, np.ones(400)), 1: (cyl_pts, np.ones(300))}
    rels = C.detect(surfaces, {(0, 1)}, CFG.brep.parallel_deg, CFG.brep.tangent_rel, 0.05)
    assert {r.name for r in rels} == {"axis_in_plane", "plane_offset"}

    out, _, dropped = C.reconcile(surfaces, rels, samples, 0.05, CFG.brep)
    assert not dropped
    n, d = out[0].params[:3], out[0].params[3]
    c, a, r = out[1].params[:3], out[1].params[3:6], out[1].params[6]
    assert abs(np.linalg.norm(n) - 1) < 1e-9 and abs(n @ a) < 1e-9
    assert abs(abs(n @ c + d) - r) < 1e-9
    assert abs(r - RADIUS) < 0.01


def _build(mesh, labels):
    adjacency = np.asarray(mesh.face_adjacency)
    votes = np.full(len(labels), -1)
    labels, surfaces, _ = F.fit_clusters(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                         adjacency, labels, votes, CFG.brep)
    bounds = T.boundaries(mesh, labels, CFG.brep.min_chain_edges)
    tol = F.rms_tolerance(F.triangle_geometry(mesh.vertices, mesh.faces)[2], CFG.brep)
    rels = C.detect(surfaces, {ch.pair for ch in bounds.chains},
                    CFG.brep.parallel_deg, CFG.brep.tangent_rel, tol)
    samples = F.samples(labels, adjacency, np.asarray(mesh.vertices), np.asarray(mesh.faces),
                        surfaces, CFG.brep)
    surfaces, accepted, _ = C.reconcile(surfaces, rels, samples, tol, CFG.brep)
    return B.build(mesh, labels, surfaces, accepted, bounds, CFG.brep)


@pytest.mark.parametrize("name", ["box", "cylinder"])
def test_closed_solid(name, tmp_path):
    if name == "box":
        mesh = trimesh.creation.box((20, 14, 8)).subdivide_to_size(1.0)
    else:
        mesh = trimesh.creation.cylinder(radius=6, height=10, sections=96).subdivide_to_size(1.0)
    mesh.merge_vertices()
    if name == "box":
        labels = np.unique(np.round(mesh.face_normals, 3), axis=0, return_inverse=True)[1].ravel()
    else:
        labels = np.select([mesh.face_normals[:, 2] > 0.9, mesh.face_normals[:, 2] < -0.9], [0, 1], 2)
    shape, report = _build(mesh, labels)
    assert report["тело"] and report["валидна"], report
    assert B.write_step(shape, tmp_path / "part.step")
