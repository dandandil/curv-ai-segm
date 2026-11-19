"""Эскиз: сечение меша, слияние коротких отрезков, соседи по цепочке, пики и кривые."""
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sketch.curves import build_pieces, build_pieces_checked, peaks, sample_curve  # noqa: E402
from src.sketch.data import neighbors, rotate, segment_features  # noqa: E402
from src.sketch.section import (Chain, ExactEdge, ExactVertex, exact_vertices, label_chain, merge_short,  # noqa: E402
                                mesh_section, plane_frame)


def test_mesh_section_of_box_is_one_ccw_rectangle():
    box = trimesh.creation.box((4.0, 2.0, 2.0))
    v, f = box.vertices, box.faces
    nrm = box.face_normals
    frame = plane_frame(np.array([0.0, 0.0, 1.0]))
    chains = mesh_section(v, f, nrm, np.zeros(3), frame)
    assert len(chains) == 1 and chains[0].closed
    c = chains[0]
    a, b = c.segment_ends()
    assert np.isclose(np.linalg.norm(b - a, axis=1).sum(), 12.0)
    # материал слева: обход против часовой, площадь со знаком > 0
    p = c.points
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    assert area > 0 and np.isclose(area, 8.0)


def test_merge_short_keeps_length_and_closure():
    pts = np.array([[0, 0], [1, 0], [1.01, 0], [2, 0], [2, 1], [0, 1]], float)
    c = Chain(pts, np.tile([0, -1, 0.0], (6, 1)), np.arange(6), True)
    m = merge_short(c, 0.05)
    assert m.n == 5 and m.closed
    a, b = m.segment_ends()
    assert np.isclose(np.linalg.norm(b - a, axis=1).sum(), 6.0)


def test_neighbors_wrap_closed_and_stop_open():
    start, length = torch.tensor([0, 4]), torch.tensor([4, 3])
    closed = torch.tensor([True, False])
    assert neighbors(start, length, closed, 1).tolist() == [1, 2, 3, 0, 5, 6, -1]
    assert neighbors(start, length, closed, -2).tolist() == [2, 3, 0, 1, -1, -1, 4]


def test_segment_features_are_midpoints_and_normals():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    nrm = np.tile([0, -1, 0.0], (4, 1))
    f = segment_features(square, nrm, True, 1.0)
    assert f.shape == (4, 5)
    assert np.allclose(f[:, :2] / 10, [[0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0]])
    assert np.allclose(f[:, 2:], nrm)

def test_peaks_nms_and_threshold():
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.6, 0.1])
    assert peaks(p, True, 0.5, 1).tolist() == [1, 4]
    assert peaks(p, True, 0.7, 2).tolist() == [1]


def test_pieces_rectangle_with_noise_and_arc():
    rng = np.random.default_rng(0)
    # прямоугольник 4×2, по 10 точек на сторону, острые стыки уточняются пересечением прямых
    corners = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], float)
    pts = np.concatenate([np.linspace(corners[i], corners[(i + 1) % 4], 10, endpoint=False) for i in range(4)])
    pts += rng.normal(scale=0.005, size=pts.shape)
    tp = np.tile([1.0, 0, 0, 0], (40, 1))
    pieces, jpos = build_pieces(pts, True, np.array([0, 10, 20, 30]), np.ones(4, bool), tp)
    assert len(pieces) == 4 and all(pc.curve[0] == "line" for pc in pieces)
    assert np.abs(jpos - corners).max() < 0.03
    # полуокружность радиуса 1 через стыки на концах
    t = np.linspace(0, np.pi, 30)
    arc = np.c_[np.cos(t), np.sin(t)]
    pieces, _ = build_pieces(arc, False, np.zeros(0, int), np.zeros(0, bool), np.tile([0, 1.0, 0, 0], (29, 1)))
    c, r = pieces[0].curve[1], pieces[0].curve[2]
    assert np.allclose(c, 0, atol=1e-6) and np.isclose(r, 1, atol=1e-6)
    s = sample_curve(pieces[0].curve)
    assert s[:, 1].min() > -1e-6                                       # дуга сверху, где точки


def _edge(kind, face, pts):
    pts = np.asarray(pts, float)
    d = np.diff(pts[[0, 1, -2, -1]], axis=0)[[0, 2]]
    return ExactEdge(kind, face, pts, pts[[0, -1]], d / np.linalg.norm(d, axis=1, keepdims=True))


def test_invisible_seam_is_not_boundary():
    line = _edge(0, 0, np.linspace([-1, 0], [0, 0], 50))
    flat = _edge(3, 1, np.linspace([0, 0], [1, 0], 50) + [0, 1e-4])          # почти прямой сплайн
    t = np.linspace(-np.pi / 2, 0, 50)
    arc = _edge(1, 1, np.c_[0.1 * np.cos(t), 0.1 + 0.1 * np.sin(t)])         # скругление r = 0.1
    v = exact_vertices([line, flat], 1e-3, 10.0, 1.0, 0.01, 1.0)
    assert [x.boundary for x in v if x.degree == 2] == [False]
    v = exact_vertices([line, arc], 1e-3, 10.0, 1.0, 0.01, 1.0)
    assert [x.boundary for x in v if x.degree == 2] == [True]
    kink = _edge(0, 1, np.linspace([0, 0], [np.cos(0.09), np.sin(0.09)], 50))  # излом ~5°: касательный, не шов
    v = exact_vertices([line, kink], 1e-3, 10.0, 1.0, 0.01, 1.0)
    assert [(x.boundary, x.tangent) for x in v if x.degree == 2] == [(True, True)]


def test_spline_join_below_angle_is_not_boundary():
    first = _edge(3, 0, np.linspace([-1, 0], [0, 0], 50))
    for deg, boundary in ((15, False), (30, True)):
        a = np.radians(deg)
        second = _edge(3, 1, np.linspace([0, 0], [np.cos(a), np.sin(a)], 50))
        v = exact_vertices([first, second], 1e-3, 10.0, 1.0, 0.01, 1.0, 20.0)
        assert [x.boundary for x in v if x.degree == 2] == [boundary]


def test_rotation_turns_coordinates_and_normals_together():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(12, 2)).astype(np.float32)
    nrm = rng.normal(size=(12, 3)).astype(np.float32)
    f0 = segment_features(pts, nrm, True, 3.0)
    f1 = segment_features(*rotate(pts, nrm, 0.7), True, 3.0)
    c, s = np.cos(0.7), np.sin(0.7)
    r = np.array([[c, -s], [s, c]])
    assert np.allclose(f0[:, :2] @ r.T, f1[:, :2], atol=1e-4)
    assert np.allclose(f0[:, 2:4] @ r.T, f1[:, 2:4], atol=1e-4) and np.allclose(f0[:, 4], f1[:, 4])

def test_pieces_ellipse_and_spline_through_ends():
    rng = np.random.default_rng(2)
    # четверть эллипса 3×1 с шумом, концы зафиксированы
    t = np.linspace(0, np.pi / 2, 40)
    pts = np.c_[3 * np.cos(t), np.sin(t)] + rng.normal(scale=0.003, size=(40, 2))
    pts[[0, -1]] = [[3, 0], [0, 1]]
    pieces, _ = build_pieces(pts, False, np.zeros(0, int), np.zeros(0, bool), np.tile([0, 0, 1.0, 0], (39, 1)))
    c = pieces[0].curve
    assert c[0] == "ellipse" and np.allclose(sorted(c[3]), [1, 3], atol=0.1)
    s = sample_curve(c)
    assert np.allclose(s[[0, -1]], [[3, 0], [0, 1]], atol=1e-6)
    assert np.abs(np.hypot(s[:, 0] / 3, s[:, 1]) - 1).max() < 0.02
    # синусоида — сплайн, проходит через концы, близко к точкам
    x = np.linspace(0, 2 * np.pi, 120)                                 # 8 узлов
    pts = np.c_[x, np.sin(x)]
    pieces, _ = build_pieces(pts, False, np.zeros(0, int), np.zeros(0, bool), np.tile([0, 0, 0, 1.0], (119, 1)))
    s = sample_curve(pieces[0].curve, 200)
    assert np.allclose(s[[0, -1]], pts[[0, -1]], atol=1e-9)
    assert np.abs(s[:, 1] - np.sin(s[:, 0])).max() < 0.01


def test_checked_pieces_split_missed_junction():
    # отрезок длины 4 и полуокружность r = 1 без стыка между ними: тип по большинству — отрезок
    line = np.c_[np.linspace(-4, 0, 40, endpoint=False), np.full(40, -1.0)]
    t = np.linspace(-np.pi / 2, np.pi / 2, 30)
    pts = np.vstack([line, np.c_[np.cos(t), np.sin(t)]])
    tp = np.r_[np.tile([1.0, 0, 0, 0], (40, 1)), np.tile([0, 1.0, 0, 0], (29, 1))]
    pieces, _ = build_pieces(pts, False, np.zeros(0, int), np.zeros(0, bool), tp)
    assert len(pieces) == 1
    pieces, _, js, _ = build_pieces_checked(pts, False, np.zeros(0, int), np.zeros(0, bool), tp, 0.05)
    assert [pc.curve[0] for pc in pieces] == ["line", "arc"] and abs(int(js[0]) - 40) <= 1


def _fillet_sketch(rng, noise=0.01):
    """Отрезок (-3,-1)->(0,-1), дуга r = 1 вокруг 0 против часовой до (1,0), отрезок (1,0)->(1,3)."""
    from src.sketch.solver import Curve, Sketch, bulge_of
    l1 = np.c_[np.linspace(-3, 0, 30), np.full(30, -1.0)]
    t = np.linspace(-np.pi / 2, 0, 20)
    arc = np.c_[np.cos(t), np.sin(t)]
    l2 = np.c_[np.full(30, 1.0), np.linspace(0, 3, 30)]
    pts = [p + rng.normal(scale=noise, size=p.shape) for p in (l1, arc, l2)]
    v = np.array([[-3, -1], [0.01, -0.99], [0.99, 0.01], [1, 3]], float)       # стыки сдвинуты, как пик сети на отрезок
    curves = [Curve("line", 0, 1, points=pts[0]),
              Curve("arc", 1, 2, np.array([bulge_of(v[1], v[2], pts[1][10])]), points=pts[1]),
              Curve("line", 2, 3, points=pts[2])]
    return Sketch(v, np.zeros(4, bool), curves)


def test_solver_tangent_fillet():
    from src.sketch.solver import Relation, curve_tuple, solve
    sk, violation = solve(_fillet_sketch(np.random.default_rng(0)),
                          [Relation("tangent", 0, 1), Relation("tangent", 1, 2), Relation("perpendicular", 0, 2)])
    assert violation < 1e-8
    _, c, r, *_ = curve_tuple(sk, 1)
    assert np.allclose(c, 0, atol=0.02) and abs(r - 1) < 0.02
    # касание: радиус в стыке перпендикулярен отрезку
    assert abs((sk.vertices[1] - c) @ (sk.vertices[1] - sk.vertices[0])) < 1e-6
    assert abs((sk.vertices[2] - c) @ (sk.vertices[3] - sk.vertices[2])) < 1e-6


def test_solver_bulge_sign_matches_sampled_arc():
    from src.sketch.solver import curve_tuple, solve
    sk, _ = solve(_fillet_sketch(np.random.default_rng(1), 0.0), [])
    s = sample_curve(curve_tuple(sk, 1), 32)
    assert np.abs(np.linalg.norm(s, axis=1) - 1).max() < 0.015 and s[16, 0] > 0 and s[16, 1] < 0


def test_unresolved_junctions_are_not_required():
    # квадрат 4×4 по 10 точек на сторону; на нижней стороне ступенька из двух близких стыков
    corners = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    pts = np.concatenate([np.linspace(corners[i], corners[(i + 1) % 4], 10, endpoint=False) for i in range(4)])
    c = Chain(pts, np.tile([0, -1, 0.0], (40, 1)), np.arange(40), True)
    verts = [ExactVertex(p, True, False, 2) for p in corners] + \
            [ExactVertex(pts[4], True, False, 2), ExactVertex(pts[6], True, False, 2)]
    _, bd, _ = label_chain(c, [], verts, 0.01, 3, 10.0)
    assert set(np.flatnonzero(bd == 1)) == {0, 10, 20, 30}          # углы
    assert bd[4] == -2 and bd[6] == -2                                # ступенька на прямом: не требовать
