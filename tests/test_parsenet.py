"""Перенос ParSeNet: kNN, выравнивание, формы выходов, лосс."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parsenet import data as PD  # noqa: E402
from src.parsenet.loss import triplet_loss  # noqa: E402
from src.parsenet.mean_shift import guard_mean_shift  # noqa: E402
from src.parsenet.metrics import segment_iou  # noqa: E402
from src.parsenet.model import ParSeNet, knn  # noqa: E402


def test_knn_matches_brute_force():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 50)
    idx = knn(x, 5)
    d = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))
    assert torch.equal(idx.sort(-1).values, d.topk(5, largest=False).indices.sort(-1).values)


def test_align_puts_minor_axis_on_x():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3)) * np.array([5.0, 3.0, 0.2])
    pts = pts @ np.linalg.qr(rng.standard_normal((3, 3)))[0].T
    out, _ = PD.align_and_scale(pts - pts.mean(0), np.zeros_like(pts))
    spread = out.std(0)
    assert spread.argmin() == 0 and np.isclose((out.max(0) - out.min(0)).max(), 1.0, atol=1e-5)


def test_forward_and_loss():
    torch.manual_seed(0)
    model = ParSeNet(128, 6, 16)
    points = torch.randn(2, 6, 300)
    emb, log_prob = model(points)
    assert emb.shape == (2, 128, 300) and log_prob.shape == (2, 6, 300)
    labels = np.random.default_rng(0).integers(0, 4, (2, 300))
    loss = triplet_loss(emb, labels, 1.0, 25, 30, np.random.default_rng(0))
    loss.sum().backward()
    assert torch.isfinite(loss).all() and model.encoder.conv1[0].weight.grad is not None


def test_segment_iou_penalizes_oversegmentation():
    target = np.array([0] * 10 + [1] * 10)
    assert np.isclose(segment_iou(target, target + 5), 1.0)
    split = np.array([0] * 10 + [1] * 5 + [2] * 5)
    # грань 0 — 1.0, грань 1 — лучший кусок 0.5, лишний кусок в паре с пустой гранью — 0
    assert np.isclose(segment_iou(target, split), (1.0 + 0.5 + 0.0) / 3)


def test_mean_shift_finds_two_blobs():
    g = torch.Generator().manual_seed(0)
    a = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0]) + 0.01 * torch.randn(100, 3, generator=g), dim=1)
    b = torch.nn.functional.normalize(torch.tensor([0, 1.0, 0]) + 0.01 * torch.randn(100, 3, generator=g), dim=1)
    labels = guard_mean_shift(torch.cat([a, b]), 0.015, 50, 5000, 0.003, 49, g)
    assert len(torch.unique(labels)) == 2 and len(torch.unique(labels[:100])) == 1
