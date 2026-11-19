"""Дымовой тест сети: граф соседства, форма выходов, лоссы, кластеризация.

Проверяются инварианты, которые ломаются молча: соседи из нужных колец, а не
откуда придётся; выход по числу треугольников, а не по числу вершин; лосс,
который на схлопнутом эмбеддинге равен margin, а на разведённом — нулю.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cluster as C  # noqa: E402
from src import graph as G  # noqa: E402
from src import metrics as M  # noqa: E402
from src.config import CFG  # noqa: E402
from src.data import (compress_curvature, load_rings, random_rotation,  # noqa: E402
                      rings_signature, rotate_features)
from src.losses import triplet_loss, type_loss  # noqa: E402
from src.model import PRIMITIVE_TYPES, PrimitiveSegNet  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FACES = 2000


@pytest.fixture(scope="module")
def grid():
    """Регулярная триангуляция квадрата: у неё кольца известны заранее."""
    side = int(np.sqrt(N_FACES // 2))
    idx = np.arange(side * side).reshape(side, side)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    import trimesh
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    mesh = trimesh.Trimesh(np.c_[x.ravel(), y.ravel(), np.zeros(side * side)], faces,
                           process=False)
    return mesh


def test_face_neighbors_within_three(grid):
    neigh = G.face_neighbors(grid.face_adjacency, len(grid.faces), DEVICE)
    assert neigh.shape == (len(grid.faces), 3)
    assert int((neigh >= 0).sum(1).max()) <= 3
    # Смежность симметрична: если a сосед b, то b сосед a.
    for a in range(0, len(grid.faces), 37):
        for b in neigh[a][neigh[a] >= 0].tolist():
            assert a in neigh[b].tolist()


def test_rings_are_at_stated_distance(grid):
    """Соседи слоя должны лежать в заявленных кольцах — это единственное, что
    отличает прорежение от обычной окрестности."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(grid.faces)
    adj = grid.face_adjacency
    neigh = G.face_neighbors(adj, n, DEVICE)
    rings = ((1, 3, 1), (4, 8, 2))
    idx = G.ring_neighbors(neigh, rings, 8, CFG.model.frontier_cap,
                           CFG.model.frontier_per_ring,
                           torch.Generator(device=DEVICE).manual_seed(0))

    a = coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]],
                                            np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    src = np.arange(0, n, 23)
    dist = dijkstra(a, indices=src, unweighted=True, limit=12)
    for layer, (first, last, step) in enumerate(rings):
        got = idx[layer][src].cpu().numpy()
        d = dist[np.arange(len(src))[:, None], got]
        # Собственный индекс подставляется, когда соседей не хватило.
        expected = set(range(first, last + 1, step)) | {0}
        assert np.isin(d, list(expected)).mean() > 0.98


def test_forward_shapes(grid):
    n = len(grid.faces)
    neigh = G.face_neighbors(grid.face_adjacency, n, DEVICE)
    idx = G.ring_neighbors(neigh, CFG.model.rings, CFG.model.neighbors,
                           CFG.model.frontier_cap, CFG.model.frontier_per_ring)
    model = PrimitiveSegNet(14, CFG.model.emb_size).to(DEVICE)
    emb, logits = model(torch.randn(1, 14, n, device=DEVICE), idx)
    assert emb.shape == (1, CFG.model.emb_size, n)
    assert logits.shape == (1, len(PRIMITIVE_TYPES), n)


def test_triplet_zero_when_separated():
    """Разведённые грани дают ноль, схлопнутые — margin: иначе лосс меряет не то."""
    labels = torch.arange(4, device=DEVICE).repeat_interleave(50)
    apart = torch.zeros(8, 200, device=DEVICE)
    apart[labels, torch.arange(200, device=DEVICE)] = 1.0
    assert float(triplet_loss(apart, labels, CFG.train.margin)) == pytest.approx(0.0)

    together = torch.ones(8, 200, device=DEVICE)
    assert float(triplet_loss(together, labels, CFG.train.margin)) == pytest.approx(
        CFG.train.margin, rel=1e-3)


def test_type_loss_ignores_unknown():
    logits = torch.zeros(len(PRIMITIVE_TYPES), 10, device=DEVICE)
    types = torch.full((10,), -1, dtype=torch.long, device=DEVICE)
    types[0] = 0
    assert float(type_loss(logits, types)) == pytest.approx(np.log(len(PRIMITIVE_TYPES)), rel=1e-4)


def test_mean_shift_recovers_clusters():
    torch.manual_seed(0)
    centers = torch.nn.functional.normalize(torch.randn(4, 32, device=DEVICE), dim=1)
    true = torch.randint(4, (3000,), device=DEVICE)
    emb = (centers[true] + 0.02 * torch.randn(3000, 32, device=DEVICE)).t()
    for width in (CFG.cluster.bandwidth, None):     # фиксированная ширина и квантиль
        pred, used = C.segment(emb, CFG.cluster.quantile, CFG.cluster.iterations,
                               CFG.cluster.bandwidth_samples, CFG.cluster.bandwidth_floor,
                               CFG.cluster.chunk, CFG.cluster.max_clusters, width=width)
        assert M.segment_scores(true, pred)["mean_iou"] > 0.9
        if width is not None:
            assert float(used) == pytest.approx(width)


def test_metrics_count_leaks_and_splits():
    true = torch.tensor([0] * 10 + [1] * 10, device=DEVICE)
    merged = torch.zeros(20, dtype=torch.long, device=DEVICE)
    assert M.segment_scores(true, merged)["leaks"] == 1.0
    assert M.segment_scores(true, merged)["splits"] == 0.0

    cut = torch.tensor([0] * 5 + [1] * 5 + [2] * 10, device=DEVICE)
    assert M.segment_scores(true, cut)["splits"] == 1.0
    assert M.segment_scores(true, cut)["leaks"] == 0.0


def test_bandwidth_survives_widened_quantile():
    """`segment` удваивает квантиль, пока кластеров не станет меньше потолка,
    и на шестом удвоении он переваливает за единицу — соседей столько не
    бывает."""
    z = torch.nn.functional.normalize(torch.randn(500, 8, device=DEVICE), dim=1)
    assert float(C.bandwidth(z, quantile=1.6, n_samples=400, floor=0.003)) > 0


def test_unknown_type_is_not_scored():
    """Грань с неизвестным типом не должна приносить очки: приравняй её к
    нулевому классу — и предсказание «плоскость» засчиталось бы верным."""
    true = torch.zeros(20, dtype=torch.long, device=DEVICE)
    types = torch.full((20,), -1, dtype=torch.long, device=DEVICE)
    logits = torch.zeros(len(PRIMITIVE_TYPES), 20, device=DEVICE)
    logits[0] = 5.0
    assert np.isnan(M.type_scores(logits, types, true)["type_acc_face"])

    types[:] = 1
    assert M.type_scores(logits, types, true)["type_acc_face"] == 0.0


def test_overlap_skips_unlabeled():
    true = torch.tensor([-1, 0, 1, 1], device=DEVICE)
    pred = torch.tensor([0, 0, 1, 1], device=DEVICE)
    m = M.overlap(true, pred)
    assert m.sum() == 3.0 and m.shape == (2, 2)


def test_total_merge_is_counted_as_leaks():
    """Все грани в одном кластере — это N - 1 протечка, а не ноль. Прежняя
    метрика здесь молчала, потому что каждая грань занимала малую долю кластера."""
    true = torch.arange(20, device=DEVICE).repeat_interleave(50)
    one = torch.zeros_like(true)
    scores = M.segment_scores(true, one)
    assert scores["leaks"] == 19.0
    assert scores["splits"] == 0.0
    assert scores["clusters"] == 1


def test_compress_curvature_keeps_geometry_channels():
    names = ["cx", "cy", "cz", "nx", "ny", "nz", "k1@0.01", "dII@0.01"]
    x = np.array([[0.3], [-0.2], [0.1], [0.0], [0.6], [0.8], [-1e4], [2e8]], dtype=np.float32)
    y = compress_curvature(x, names)
    assert np.array_equal(y[:6], x[:6])
    assert y[6, 0] < 0 and abs(y[6, 0]) < 10
    assert 0 < y[7, 0] < 25


def test_ring_cache_ignored_on_signature_mismatch(tmp_path):
    """Кольца со стороны, собранные при другом конфиге, молча не читаются.

    Слои хранятся раздельно (`l0`, `l1`, ...): число соседей у них может
    различаться, узкому первому слою их нужно меньше. Прежний формат одним
    массивом читается по-прежнему — кэш на корпусе пересобирать незачем.
    """
    layers = {f"l{i}": np.zeros((10, 4), np.uint16) for i in range(3)}
    np.savez(tmp_path / "rings.npz", signature=np.array("чужой конфиг"), **layers)
    assert load_rings(tmp_path) is None
    np.savez(tmp_path / "rings.npz", signature=np.array(rings_signature()), **layers)
    assert [r.shape for r in load_rings(tmp_path)] == [(10, 4)] * 3
    np.savez(tmp_path / "rings.npz", rings=np.zeros((3, 10, 4), np.uint16),
             signature=np.array(rings_signature()))
    assert [r.shape for r in load_rings(tmp_path)] == [(10, 4)] * 3


def test_rotation_turns_geometry_and_keeps_curvature():
    """Поворот детали — это поворот координат и нормали; кривизны инвариантны,
    поэтому пересчитывать геометрию не нужно."""
    names = ["cx", "cy", "cz", "nx", "ny", "nz", "k1@0.01", "S@0.01", "dII@0.01"]
    x = np.random.default_rng(0).normal(size=(len(names), 64)).astype(np.float32)
    rot = random_rotation(np.random.default_rng(1))
    y = rotate_features(x, names, rot)

    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)      # поворот, не отражение
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
    assert np.array_equal(y[6:], x[6:])                            # кривизны не тронуты
    assert np.allclose(y[:3], rot @ x[:3], atol=1e-5)              # координаты повернулись
    assert np.allclose(np.linalg.norm(y[3:6], axis=0),             # нормаль осталась единичной
                       np.linalg.norm(x[3:6], axis=0), atol=1e-5)
