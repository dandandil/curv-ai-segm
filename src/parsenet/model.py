"""Сеть ParSeNet: DGCNN с динамическим kNN, головы эмбеддинга и типа.

Перенос `src/PointNet.py` (классы `DGCNNEncoderGn`, `PrimitivesEmbeddingDGCNGn`,
режим 5 — точки и нормали). Изменения технические: устройство берётся у
входа, лосс считается снаружи forward, ветки других режимов убраны. Числа
каналов, GroupNorm, k и порядок слоёв — дословно. Dropout у них фактически
выключен (`self.drop = 0.0`), поэтому здесь его нет.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.no_grad()
def knn(x: Tensor, k: int) -> Tensor:
    """`(B, C, N)` -> индексы k ближайших `(B, N, k)` по евклиду в пространстве признаков."""
    out = []
    for b in range(x.shape[0]):
        xb = x[b:b + 1]
        inner = -2 * torch.matmul(xb.transpose(2, 1), xb)
        xx = torch.sum(xb ** 2, dim=1, keepdim=True)
        out.append(-xx - inner - xx.transpose(2, 1))
    return torch.cat(out, 0).topk(k=k, dim=-1)[1]


@torch.no_grad()
def knn_points_normals(x: Tensor, k: int) -> Tensor:
    """Первый слой: расстояние по точкам, умноженное на (1 + расхождение нормалей)."""
    out = []
    for b in range(x.shape[0]):
        p, n = x[b:b + 1, 0:3], x[b:b + 1, 3:6]
        inner = 2 * torch.matmul(p.transpose(2, 1), p)
        xx = torch.sum(p ** 2, dim=1, keepdim=True)
        p_dist = xx - inner + xx.transpose(2, 1)
        n_dist = 2 - 2 * torch.matmul(n.transpose(2, 1), n)
        out.append(-(p_dist * (1 + n_dist)))
    return torch.cat(out, 0).topk(k=k, dim=-1)[1]


def graph_feature(x: Tensor, k: int, with_normals: bool = False) -> Tensor:
    """`(B, C, N)` -> `(B, 2C, N, k)`: `[сосед − центр, центр]`."""
    b, c, n = x.shape
    idx = knn_points_normals(x, k) if with_normals else knn(x, k)
    idx = (idx + torch.arange(b, device=x.device).view(-1, 1, 1) * n).view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(b * n, -1)[idx, :].view(b, n, k, c)
    x = x.view(b, n, 1, c).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 6, k: int = 80):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, 1, bias=False), nn.GroupNorm(2, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, 1, bias=False), nn.GroupNorm(2, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, 1, bias=False), nn.GroupNorm(2, 128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.conv1(graph_feature(x, self.k, with_normals=True)).max(dim=-1)[0]
        x2 = self.conv2(graph_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.conv3(graph_feature(x2, self.k)).max(dim=-1)[0]
        features = torch.cat((x1, x2, x3), dim=1)
        pooled = F.relu(self.bnmlp1(self.mlp1(features))).max(dim=2)[0]
        return pooled, features


class ParSeNet(nn.Module):
    def __init__(self, emb_size: int = 128, n_types: int = 10, k: int = 80):
        super().__init__()
        self.encoder = Encoder(6, k)
        self.conv1 = nn.Conv1d(1024 + 256, 512, 1)
        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.GroupNorm(4, 256)
        self.mlp_seg_prob1 = nn.Conv1d(256, 256, 1)
        self.mlp_seg_prob2 = nn.Conv1d(256, emb_size, 1)
        self.bn_seg_prob1 = nn.GroupNorm(4, 256)
        self.mlp_prim_prob1 = nn.Conv1d(256, 256, 1)
        self.mlp_prim_prob2 = nn.Conv1d(256, n_types, 1)
        self.bn_prim_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """`(B, 6, N)` — точки и нормали -> эмбеддинг `(B, emb, N)` и log-вероятности типа `(B, T, N)`."""
        n = points.shape[2]
        pooled, features = self.encoder(points)
        x = torch.cat([pooled.unsqueeze(-1).repeat(1, 1, n), features], 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x_all = F.relu(self.bn2(self.conv2(x)))
        embedding = self.mlp_seg_prob2(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))))
        types = F.log_softmax(self.mlp_prim_prob2(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all)))), dim=1)
        return embedding, types
