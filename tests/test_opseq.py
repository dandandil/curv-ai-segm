"""opseq: лоссы по множествам, батчи по бюджету, признаки пар."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.opseq.data import BudgetBatches, pair_features, spd  # noqa: E402
from src.opseq.losses import set_nll  # noqa: E402


def test_set_nll_sums_probabilities():
    logits = torch.tensor([0.0, 1.0, 2.0, float("-inf")])
    p = torch.softmax(logits, 0)
    assert torch.isclose(set_nll(logits, [1, 2]), -torch.log(p[1] + p[2]))
    assert set_nll(logits, [0, 1, 2]).abs() < 1e-6


def test_budget_batches_cover_all_and_respect_budget():
    nodes = np.random.default_rng(0).integers(2, 200, 500)
    sampler = BudgetBatches(nodes, batch=16, pairs=20000, shuffle=True)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(500))
    assert all(len(b) <= 16 and (len(b) == 1 or len(b) * nodes[b].max() ** 2 <= 20000) for b in batches)
    assert list(sampler) != batches                     # новая эпоха — новый порядок


def test_pair_features_and_spd():
    # две параллельные плоскости на расстоянии 2 и цилиндр с осью вдоль их нормали
    g = {"ids": np.arange(3), "diag": 10.0,
         "dir": np.array([[0, 0, 1], [0, 0, -1], [0, 0, 1]], np.float32),
         "anchor": np.array([[0, 0, 0], [0, 0, 2], [5, 0, 0]], np.float32),
         "fit": np.ones(3, bool), "plane": np.array([True, True, False]), "axial": np.array([False, False, True])}
    edges = np.array([[0, 2], [1, 2]])
    pf = pair_features(g, edges, np.array([False, True]), np.array([1.0, -1.0], np.float32))
    assert np.isclose(pf[0, 1, 2], 1) and pf[0, 1, 3] == -1 and np.isclose(pf[0, 1, 4], 0.2)
    assert pf[1, 2, 0] == 1 and pf[2, 1, 0] == 1 and pf[0, 2, 1] == 1 and pf[0, 1, 0] == 0
    assert np.isclose(pf[0, 2, 2], 1) and pf[0, 2, 3] == 0
    d = spd(3, edges, max_spd=8)
    assert d[0, 1] == 2 and d[0, 2] == 1 and d[1, 1] == 0
    assert spd(3, edges[:1], max_spd=8)[0, 1] == 9
