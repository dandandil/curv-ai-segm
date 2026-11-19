"""Вариант D: цель на ребро, голова границ, нарезка по графу."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.boundary import partition as P  # noqa: E402
from src.boundary.config import BCFG  # noqa: E402
from src.boundary.data import long_range_pairs, tangent_mask  # noqa: E402
from src.boundary.losses import edge_targets, repulsive_loss  # noqa: E402
from src.boundary.model import BoundarySegNet, head_config, load_backbone  # noqa: E402
from src.config import CFG  # noqa: E402
from src.model import PrimitiveSegNet  # noqa: E402


def grid(n: int = 16) -> trimesh.Trimesh:
    """Квадрат n x n из пар треугольников."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    vertices = np.c_[i.ravel(), j.ravel(), np.zeros(i.size)].astype(float)
    vid = lambda a, b: a * (n + 1) + b  # noqa: E731
    faces = []
    for a in range(n):
        for b in range(n):
            faces += [[vid(a, b), vid(a + 1, b), vid(a + 1, b + 1)], [vid(a, b), vid(a + 1, b + 1), vid(a, b + 1)]]
    return trimesh.Trimesh(vertices, np.array(faces), process=False)


def test_edge_targets_skip_band_near_joint():
    labels = torch.tensor([0, 0, 0, 1, 1])
    ambiguous = torch.tensor([False, False, True, True, False])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    target, weight = edge_targets(labels, ambiguous, edges, 30.0)
    assert target.tolist() == [1.0, 1.0, 0.0, 1.0]
    # (0,1) внутри грани, (1,2) и (3,4) в полосе у стыка, (2,3) стык: 1 к 1
    assert weight.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_tangent_joints_weighted():
    labels = torch.tensor([0, 1, 2, 2])
    ambiguous = torch.tensor([True, True, True, True])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    tangent = tangent_mask(labels, edges, torch.tensor([[1, 2]]))
    assert tangent.tolist() == [False, True, False]         # (2,3) внутри грани, не стык
    _, plain = edge_targets(labels, ambiguous, edges, 30.0)
    _, weighted = edge_targets(labels, ambiguous, edges, 30.0, tangent, 3.0)
    assert weighted[1] == 3 * plain[1] and weighted[0] == plain[0] and weighted[2] == plain[2]


def test_face_balance_weighs_faces_equally():
    # Полоса 16 x 16: широкая грань 0 (строки 0-11), узкие 1 (12-13) и 2 (14-15).
    mesh = grid()
    rows = mesh.triangles_center[:, 0].astype(int)
    labels = torch.from_numpy(np.select([rows < 12, rows < 14], [0, 1], 2))
    edges = torch.from_numpy(mesh.face_adjacency.astype(np.int64))
    ambiguous = torch.zeros_like(labels, dtype=torch.bool)
    _, weight = edge_targets(labels, ambiguous, edges, 30.0, face_balance=True)
    a, b = edges[:, 0], edges[:, 1]
    same = labels[a] == labels[b]
    # Широкая грань весит столько же, сколько узкая, хотя рёбер у неё впятеро больше.
    for f in range(3):
        assert abs(float(weight[same & (labels[a] == f)].sum()) - 0.5) < 1e-6
    # Стык каждой грани — ещё половина, стыковое ребро получает долю от обеих сторон.
    assert abs(float(weight[~same].sum()) - 1.5) < 1e-6


def test_head_is_symmetric_and_keeps_backbone():
    torch.manual_seed(0)
    mesh = grid(8)
    n = len(mesh.faces)
    idx = [torch.randint(0, n, (n, CFG.model.neighbors)) for _ in range(3)]
    x = torch.randn(1, 10, n)
    x[0, 3:6] = torch.nn.functional.normalize(x[0, 3:6], dim=0)
    edges = torch.as_tensor(mesh.face_adjacency, dtype=torch.long)

    base = PrimitiveSegNet(10, CFG.model.emb_size).eval()
    model = BoundarySegNet(10, CFG.model.emb_size, BCFG.model).eval()
    missing = load_backbone(model, base.state_dict())
    assert missing and all(k.startswith(("edge_", "pair.")) for k in missing)
    with torch.no_grad():
        emb_base, types_base = base(x, idx)
        emb, types, forward = model(x, idx, edges)
        _, _, backward = model(x, idx, edges.flip(1))
    assert torch.allclose(forward, backward, atol=1e-6)
    assert torch.allclose(emb, emb_base, atol=1e-5) and torch.allclose(types, types_base, atol=1e-5)


def test_repulsive_head_symmetric_and_optional():
    torch.manual_seed(0)
    mesh = grid(8)
    n = len(mesh.faces)
    idx = [torch.randint(0, n, (n, CFG.model.neighbors)) for _ in range(3)]
    x = torch.randn(1, 10, n)
    x[0, 3:6] = torch.nn.functional.normalize(x[0, 3:6], dim=0)
    edges = torch.as_tensor(mesh.face_adjacency, dtype=torch.long)
    pairs = long_range_pairs(idx, 4, torch.Generator().manual_seed(0))
    assert (pairs[:, 0] < pairs[:, 1]).all() and len(torch.unique(pairs, dim=0)) == len(pairs)

    head = dataclasses.replace(BCFG.model, repulsive=True)
    model = BoundarySegNet(10, CFG.model.emb_size, head).eval()
    # Веса основной архитектуры подходят, обе новые головы остаются случайными.
    missing = load_backbone(model, PrimitiveSegNet(10, CFG.model.emb_size).state_dict())
    assert any(k.startswith("rep_") for k in missing)
    assert head_config(BCFG.model, model.state_dict()).repulsive
    assert not head_config(BCFG.model, BoundarySegNet(10, CFG.model.emb_size, BCFG.model).state_dict()).repulsive
    with torch.no_grad():
        assert len(model(x, idx, edges)) == 3                  # без пар — прежние три выхода
        *_, forward = model(x, idx, edges, pairs)
        *_, backward = model(x, idx, edges, pairs.flip(1))
    assert forward.shape == (len(pairs),) and torch.allclose(forward, backward, atol=1e-6)


def test_repulsive_loss_balances_classes():
    labels = torch.tensor([0, 0, 0, 1])
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 3]])     # три «одна грань», одна «разные»
    logits = torch.zeros(4)
    # При нулевом логите каждая пара стоит log 2, и классы делят вес пополам.
    assert abs(float(repulsive_loss(logits, labels, pairs)) - float(np.log(2))) < 1e-6
    confident_wrong = torch.tensor([0.0, 0.0, 0.0, 5.0])        # уверенно «одна грань» на разных
    assert float(repulsive_loss(confident_wrong, labels, pairs)) > 2.0


def test_mutex_watershed_splits_by_repulsion():
    """Две половины квадрата со слабым стыком: притяжение через стык всё же выше нуля.
    Без отталкивания всё сливается; с отталкиванием через стык — две грани."""
    mesh = grid(8)
    left = mesh.triangles_center[:, 0] < 4
    edges = mesh.face_adjacency.astype(np.int64)
    across = left[edges[:, 0]] != left[edges[:, 1]]
    w_attr = np.where(across, 0.4, 0.9)
    n = len(mesh.faces)
    none = P.mutex_watershed(n, edges, w_attr, np.zeros((0, 2), np.int64), np.zeros(0))
    assert none.max() == 0
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 400), rng.integers(0, n, 400)
    far = (i != j) & (left[i] != left[j])
    pairs = np.c_[i, j][far]
    split = P.mutex_watershed(n, edges, w_attr, pairs, np.full(len(pairs), 0.7))
    assert split.max() == 1 and (split[left] == split[left][0]).all() and (split[~left] != split[left][0]).all()


# Пороги устройства нарезки, а не рабочие: рабочие подбираются на валидации и
# могут отключать защиты, которые здесь проверяются.
STRUCTURE = dataclasses.replace(BCFG.partition, depth=0.2, merge=0.5, veto=None)


def test_partition_survives_gap_in_boundary():
    """Две грани; на одном стыковом ребре голова почти промолчала — не склеиваются.

    Предел: ребро, где она промолчала совсем, то есть слабее внутренностей
    обеих граней, водораздел проходит насквозь. Бассейны сливаются при перевале
    ниже depth над их дном, а такое ребро и есть дно.
    """
    mesh = grid(16)
    edges = np.asarray(mesh.face_adjacency)
    labels = (mesh.triangles_center[:, 0] > 8).astype(int)
    joint = np.flatnonzero(labels[edges[:, 0]] != labels[edges[:, 1]])
    strength = np.zeros(len(edges))
    strength[joint] = 1.0
    strength[joint[len(joint) // 2]] = 0.3
    out = P.partition(len(mesh.faces), edges, strength, STRUCTURE)
    assert len(np.unique(out)) == 2
    assert max(np.mean(out == labels), np.mean(out != labels)) > 0.99


def test_partition_keeps_narrow_face_without_quiet_interior():
    """Узкая грань, внутри которой сила границы нигде не мала, остаётся своим
    регионом: важен перепад до стыка, а не абсолютная величина."""
    mesh = grid(16)
    edges = np.asarray(mesh.face_adjacency)
    cx = mesh.triangles_center[:, 0]
    labels = np.select([cx < 7, cx > 9], [0, 2], default=1)
    inner_narrow = (labels[edges[:, 0]] == 1) & (labels[edges[:, 1]] == 1)
    strength = np.where(labels[edges[:, 0]] != labels[edges[:, 1]], 0.9, 0.0)
    strength[inner_narrow] = 0.5
    out = P.partition(len(mesh.faces), edges, strength, STRUCTURE)
    assert len(np.unique(out)) == 3
    assert len(np.unique(out[labels == 1])) == 1


def test_partition_merges_weak_ridge():
    """Слабый гребень внутри грани даёт лишний бассейн, агломерация его сливает."""
    mesh = grid(16)
    edges = np.asarray(mesh.face_adjacency)
    cx = mesh.triangles_center[:, 0]
    labels = (cx > 8).astype(int)
    strength = np.where(labels[edges[:, 0]] != labels[edges[:, 1]], 1.0, 0.0)
    left = np.floor(cx[edges[:, 0]]) != np.floor(cx[edges[:, 1]])
    ridge = left & (np.minimum(cx[edges[:, 0]], cx[edges[:, 1]]) < 4) & (np.maximum(cx[edges[:, 0]], cx[edges[:, 1]]) > 4)
    strength[ridge] = 0.3
    cfg = STRUCTURE
    assert len(np.unique(P.watershed(len(mesh.faces), edges, strength, cfg.depth))) > 2
    out = P.partition(len(mesh.faces), edges, strength, cfg)
    assert len(np.unique(out)) == 2


def test_face_halfwidth_is_distance_from_middle_to_edge():
    """Полуширина — от самой дальней от краёв точки грани до края, в кольцах."""
    from src.metrics import face_halfwidth
    edges = np.array([[i, i + 1] for i in range(29)])            # цепочка треугольников
    labels = np.array([0] * 10 + [1] * 6 + [2] * 14)
    assert face_halfwidth(edges, labels).tolist() == [9, 2, 13]
    assert face_halfwidth(edges, labels, cap=9).tolist() == [9, 2, 9]


def test_contrast_loss_rewards_watershed_gap():
    """Две грани в ряд по 4 треугольника, стык — ребро 3-4.

    Внутренние рёбра, чьи треугольники не касаются чужой грани: 0-1, 1-2 у первой
    и 5-6, 6-7 у второй.
    """
    from src.boundary.losses import contrast_loss
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edges = torch.tensor([[i, i + 1] for i in range(7)])

    def logits(first, second, seam):
        s = torch.tensor([first, first, first, seam, second, second, second])
        return torch.log((1 - s) / s)                   # сила s <=> логит «одна грань»

    good = contrast_loss(logits(0.1, 0.05, 0.9), labels, edges, margin=0.1, tau=0.01)
    assert float(good) < 1e-6                          # перепад 0.8 больше запаса

    flat = logits(0.8, 0.7, 0.85).requires_grad_(True)
    bad = contrast_loss(flat, labels, edges, margin=0.2, tau=0.01)
    assert float(bad) > 0.1                            # перепад 0.05 при запасе 0.2
    bad.backward()
    # штраф тянет силу на стыке вверх (логит вниз) и дно более мелкой грани вниз
    assert flat.grad[3] > 0 and flat.grad[0] < 0
    assert flat.grad[5] == 0                           # дно второй грани ниже и не мешает
