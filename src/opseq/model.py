"""Graphormer над графом поверхностей и головы шага (README, «Тело последовательностью операций»).

Узел: явные признаки + PointNet по точкам в собственной системе, плюс состояние. Внимание с
поправкой — MLP от признаков пары и embedding spd, по числу на голову; виртуальный узел
(последний) видит всех, его выход — тип шага (операция или «готово»). Головы шага идут по
порядку, каждая видит выбранное раньше: тип -> указатель на грань (опора, у скругления и фаски —
полоса) -> класс граней операции (нет / стенка / торец) -> у выдавливания булева -> «до k» или «без
поверхности» -> глубина и направление. До цикла (прогон с пустым состоянием): отношения пар и
вспомогательные головы — роль грани и «две грани одной операции» (README, «Данные v2»).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import OCFG
from .data import BOOLEANS, FACE_CLASSES, OP_TYPES, PAIR_DIM, RELATIONS, ROLES, STATE_DIM, STATIC_DIM, STEP_CLASSES


def mlp(*dims: int) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(dims[-2], dims[-1]))


class PointNet(nn.Module):
    def __init__(self, out: int):
        super().__init__()
        self.net = mlp(6, 64, 128, out)

    def forward(self, points: Tensor) -> Tensor:
        return self.net(points).amax(-2)


class Layer(nn.Module):
    """Pre-LN слой трансформера с аддитивной поправкой к вниманию."""

    def __init__(self, dim: int, heads: int, ffn: int, dropout: float, pair_dim: int):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.pair = nn.Linear(pair_dim, heads)
        # поправка виртуального узла: к нему и от него, своя на голову (как в Graphormer)
        self.virtual = nn.Parameter(torch.zeros(2, heads))
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, pair: Tensor, key_mask: Tensor) -> Tensor:
        b, n1, d = x.shape
        n = n1 - 1
        bias = x.new_zeros(b, self.heads, n1, n1)
        bias[:, :, :n, :n] = self.pair(pair).permute(0, 3, 1, 2)
        bias[:, :, :n, n] = self.virtual[0][None, :, None]
        bias[:, :, n, :] = self.virtual[1][None, :, None]
        bias = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        q, k, v = self.qkv(self.norm1(x)).view(b, n1, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, n1, d)))
        return x + self.drop(self.ffn(self.norm2(x)))


class OpseqNet(nn.Module):
    def __init__(self, cfg=OCFG):
        super().__init__()
        m = cfg.model
        d = self.dim = m.dim
        self.points = PointNet(m.point_dim)
        self.node = mlp(STATIC_DIM + m.point_dim, d, d)
        self.state = nn.Linear(STATE_DIM, d)
        self.virtual = nn.Parameter(torch.zeros(d))
        self.pair = nn.Linear(PAIR_DIM, m.pair_dim)
        self.spd = nn.Embedding(cfg.data.max_spd + 2, m.pair_dim)
        self.layers = nn.ModuleList(Layer(d, m.heads, m.ffn, m.dropout, m.pair_dim) for _ in range(m.layers))
        self.norm = nn.LayerNorm(d)
        # головы шага
        self.step_type = mlp(d, d, len(STEP_CLASSES))
        self.type_embed = nn.Embedding(len(OP_TYPES), d)
        self.support = mlp(2 * d, d, 1)
        self.faces = mlp(3 * d, d, len(FACE_CLASSES))
        self.boolean = mlp(2 * d, d, len(BOOLEANS))
        self.boolean_embed = nn.Embedding(len(BOOLEANS), d)
        self.to_query = mlp(3 * d, d, d)
        self.to_key = nn.Linear(d, d, bias=False)
        self.to_pair = nn.Linear(m.pair_dim, 1)
        self.none = nn.Linear(d, 1)
        self.depth = mlp(d, d, 1)
        self.direction = mlp(d, d, 1)
        # до цикла
        self.relation = mlp(2 * d + m.pair_dim, d, len(RELATIONS))
        self.role = mlp(d, d, len(ROLES))
        self.same_op = mlp(2 * d + m.pair_dim, d, 1)

    def embed_nodes(self, static: Tensor, points: Tensor, mask: Tensor) -> Tensor:
        """Признаки узла без состояния `(B, N, d)`: считаются раз на батч, только у настоящих узлов."""
        out = static.new_zeros(*mask.shape, self.dim)
        out[mask] = self.node(torch.cat([static[mask], self.points(points[mask])], -1))
        return out

    def embed_pairs(self, pair: Tensor, spd: Tensor) -> Tensor:
        return F.gelu(self.pair(pair) + self.spd(spd))

    def encode(self, nodes: Tensor, state: Tensor, pairs: Tensor, mask: Tensor) -> Tensor:
        """`(B, N + 1, d)`, последний — виртуальный узел."""
        b = nodes.shape[0]
        x = torch.cat([nodes + self.state(state), self.virtual.expand(b, 1, -1)], 1)
        key_mask = torch.cat([mask, mask.new_ones(b, 1)], 1)
        for layer in self.layers:
            x = layer(x, pairs, key_mask)
        return self.norm(x)

    # ----- шаг -----

    def type_logits(self, h: Tensor) -> Tensor:
        """`(len(STEP_CLASSES),)` по виртуальному узлу тела `h` `(N + 1, d)`."""
        return self.step_type(h[-1])

    def support_logits(self, h: Tensor, kind: int) -> Tensor:
        """Указатель на грань `(N,)` для операции типа `kind`; `h` — `(N, d)`."""
        return self.support(torch.cat([h, self.type_embed.weight[kind].expand_as(h)], -1)).squeeze(-1)

    def face_logits(self, h: Tensor, h_support: Tensor, kind: int) -> Tensor:
        """Класс граней для операции `(N, len(FACE_CLASSES))`."""
        return self.faces(torch.cat([h, h_support.expand_as(h), self.type_embed.weight[kind].expand_as(h)], -1))

    def boolean_logits(self, h_support: Tensor, h_profile: Tensor) -> Tensor:
        return self.boolean(torch.cat([h_support, h_profile], -1))

    def to_heads(self, h: Tensor, pairs: Tensor, support: int, h_profile: Tensor, boolean: int):
        """Указатель «до k» по узлам одного тела `(N + 1,)`, последний — «без поверхности»; запрос —
        для глубины и направления. `h` — `(N, d)`, `pairs` — `(N, N, pair_dim)`."""
        q = self.to_query(torch.cat([h[support], h_profile, self.boolean_embed.weight[boolean]], -1))
        logits = self.to_key(h) @ q / self.dim ** 0.5 + self.to_pair(pairs[support]).squeeze(-1)
        return torch.cat([logits, self.none(q)]), q

    def depth_direction(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return self.depth(q).squeeze(-1), self.direction(q).squeeze(-1)

    # ----- до цикла -----

    def pair_input(self, h: Tensor, pairs: Tensor, index: Tensor) -> Tensor:
        """Вход головы пары по `(P, 3)` — батч, i, j; симметричен по i, j для отношений."""
        bi, i, j = index.unbind(1)
        hi, hj = h[bi, i], h[bi, j]
        return torch.cat([hi + hj, hi * hj, pairs[bi, i, j]], -1)

    def relation_logits(self, h: Tensor, pairs: Tensor, index: Tensor) -> Tensor:
        return self.relation(self.pair_input(h, pairs, index))

    def role_logits(self, h: Tensor) -> Tensor:
        return self.role(h)

    def same_op_logits(self, h: Tensor, pairs: Tensor, index: Tensor) -> Tensor:
        return self.same_op(self.pair_input(h, pairs, index)).squeeze(-1)
