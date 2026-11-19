"""Своя тесселяция STEP: слияние швов и перевод типов через merge_map."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dataset import cache as D
from dataset import step_mesh as M


def _plane(fid, normal, offset):
    return {"id": fid, "type": "plane", "mergeable": True,
            "params": {"normal": normal, "origin": [0, 0, 0], "offset": offset}}


def test_merge_plane_halves_with_opposite_normals():
    # Две половины одной плоскости с противоположными нормалями сливаются,
    # параллельная плоскость рядом — нет, даже будучи соседом.
    faces = [_plane(0, [0, 0, 1], 5.0), _plane(1, [0, 0, -1], -5.0), _plane(2, [0, 0, 1], 5.5)]
    adjacency = [{"a": 0, "b": 1}, {"a": 1, "b": 2}]
    brep = {"n_faces": 3, "bbox_diag": 100.0, "faces": faces, "adjacency": adjacency}
    assert M.merge_faces(brep, 1.0, 1e-4, 1e-3).tolist() == [0, 0, 1]


def test_bspline_not_merged():
    faces = [{"id": i, "type": "geomabs_bsplinesurface", "mergeable": False, "params": {}}
             for i in range(2)]
    brep = {"n_faces": 2, "bbox_diag": 1.0, "faces": faces, "adjacency": [{"a": 0, "b": 1}]}
    assert M.merge_faces(brep, 1.0, 1e-4, 1e-3).tolist() == [0, 1]


def test_real_part_labels_consistent(tmp_path):
    step = sorted(D.RAW_STEP.glob("*.stp"))[:1] if D.RAW_STEP.exists() else []
    if not step:
        pytest.skip("нет STEP корпуса")
    info = M.build_part(step[0], tmp_path / "part")
    brep = json.loads((tmp_path / "part" / "brep.json").read_text())
    fid = np.load(tmp_path / "part" / "face_id.npy")
    assert len(fid) == info["n_triangles"]
    assert fid.max() + 1 == info["n_merged"] == max(brep["merge_map"]) + 1
    # Сливаются только грани одного типа.
    by_label = {}
    for face in brep["faces"]:
        by_label.setdefault(brep["merge_map"][face["id"]], set()).add(face["type"])
    assert all(len(t) == 1 for t in by_label.values())


def test_step_substrate_matches_tessellation():
    """Подложка gmsh из STEP: заданное число треугольников, замкнута, без игл, нормали
    наружу — как у исходной тесселяции."""
    pytest.importorskip("gmsh")
    import trimesh
    from dataset import remesh as R
    from dataset import step_substrate as S
    uid = "40052_42bdc982_4"                       # 35 граней, на пробе gmsh строил её за 2 с
    step = D.RAW_STEP / f"{uid}.stp"
    if not step.exists():
        pytest.skip("нет STEP корпуса")
    src = trimesh.load(str(D.PARTS / uid / "mesh.ply"), process=False)
    faces = 100_000
    edge = R.target_edge(float(src.area), faces)
    sub = S.substrate(step, edge, src)
    assert 0.8 * faces < len(sub.faces) < 1.25 * faces
    assert sub.is_watertight
    v, f = sub.vertices, sub.faces
    a, b = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert np.mean((cos > np.cos(np.radians(1))) | (cos < np.cos(np.radians(169)))) < 0.005
    assert sub.volume > 0 and abs(sub.volume - src.volume) < 0.01 * src.volume


def test_smooth_spline_seam_merged():
    # Невидимый шов (нормали и кривизна непрерывны) сливается, шов со скачком кривизны — нет;
    # по умолчанию (порог 0) слияние сплайнов выключено.
    faces = [{"id": i, "type": "geomabs_bsplinesurface", "mergeable": False, "params": {}} for i in range(3)]
    adjacency = [{"a": 0, "b": 1, "seam_angle_deg": 0.2, "seam_curv_jump": 0.01},
                 {"a": 1, "b": 2, "seam_angle_deg": 0.2, "seam_curv_jump": 0.3}]
    brep = {"n_faces": 3, "bbox_diag": 1.0, "faces": faces, "adjacency": adjacency}
    assert M.merge_faces(brep, 1.0, 1e-4, 1e-3, seam_curv_jump=0.05).tolist() == [0, 0, 1]
    assert M.merge_faces(brep, 1.0, 1e-4, 1e-3).tolist() == [0, 1, 2]


def _narrow_rules(shape, max_mm=0.5):
    diag = M.bbox_diag(shape)
    mesh, fid, counts = M.tessellate(shape, diag, 5e-5, 0.2, 2e-3)
    merge_map = np.arange(len(counts))
    samples = M._boundary_samples(shape, merge_map, 0.05)
    halfwidth = M.face_halfwidth(mesh, merge_map[fid], samples, 0.05, len(counts))
    narrow, label_map = M.classify_narrow(samples, halfwidth, max_mm, 30.0, 30.0)
    return halfwidth, {r["label"]: r["rule"] for r in narrow}, label_map


def test_halfwidth_and_turn_on_thin_plate():
    # Брусок 10×2×1 мм: полуширины граней 0.5 / 0.5 / 1.0 мм; торцы 2×1 и бока 10×1 — узкие
    # между противоположными гранями, то есть разворот: остаются гранями.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    halfwidth, rules, label_map = _narrow_rules(BRepPrimAPI_MakeBox(10.0, 2.0, 1.0).Shape(), max_mm=0.52)
    assert np.allclose(sorted(halfwidth), [0.5] * 4 + [1.0] * 2, atol=0.03)
    assert set(rules.values()) == {"turn"} and len(rules) == 4
    assert (label_map >= 0).all()


def test_step_merges_levels():
    # Плита 10×10×2 и полка 10×5 на 0.3 мм выше: подступенок — узкая грань между
    # сонаправленными верхами, то есть ступенька; она и оба верха — одна метка.
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.gp import gp_Pnt
    fused = BRepAlgoAPI_Fuse(BRepPrimAPI_MakeBox(10.0, 10.0, 2.0).Shape(),
                             BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10.0, 5.0, 2.3).Shape()).Shape()
    unify = ShapeUpgrade_UnifySameDomain(fused)
    unify.Build()
    halfwidth, rules, label_map = _narrow_rules(unify.Shape())
    steps = [label for label, rule in rules.items() if rule == "step"]
    assert len(steps) == 1 and abs(halfwidth[steps[0]] - 0.15) < 0.03
    assert (label_map == label_map[steps[0]]).sum() == 3
