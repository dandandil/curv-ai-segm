"""EdgeConv без тензора пар совпадает с исходной формой через graph_feature."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import EdgeConv, graph_feature  # noqa: E402


def test_edgeconv_without_pairs_matches_reference():
    torch.manual_seed(0)
    layer = EdgeConv(10, 64, groups=2).double()
    with torch.no_grad():                               # γ разного знака: проверяется и ветка с min
        layer.block[1].weight.uniform_(-1.0, 1.0)
        layer.block[1].bias.uniform_(-1.0, 1.0)
    n, k = 400, 12
    layer.CHUNK = 64                                   # несколько порций и неполная последняя
    idx = torch.randint(0, n, (n, k))
    x = torch.randn(1, 10, n, dtype=torch.float64)
    weight = torch.randn(1, 64, n, dtype=torch.float64)

    results = []
    for forward in (lambda t: layer.block(graph_feature(t, idx)).max(dim=-1).values,
                    lambda t: layer(t, idx)):
        t = x.clone().requires_grad_(True)
        layer.zero_grad()
        out = forward(t)
        (out * weight).sum().backward()
        results.append((out.detach(), t.grad, layer.block[0].weight.grad.clone(),
                        layer.block[1].weight.grad.clone(), layer.block[1].bias.grad.clone()))
    for a, b in zip(*results):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-10)
