"""Сеть с головой границ: энкодер и головы `PrimitiveSegNet` плюс логит на ребро.

Голова берёт тот же `h` на треугольник, что головы эмбеддинга и типа, проецирует
его своим слоем и сравнивает соседние треугольники симметрично: у ребра нет
направления, и перестановка концов не должна менять ответ.

Голова отталкивающих связей (`head.repulsive`) устроена так же, но сравнивает
дальние пары треугольников — для Mutex Watershed (README, «Mutex Watershed»).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model import PrimitiveSegNet

from .config import HeadCfg

# Каналы нормали во входе: признаки идут как cx, cy, cz, nx, ny, nz, кривизны.
NORMAL = slice(3, 6)
# Параметры голов границ и отталкивающих связей: их нет в чекпоинтах основной архитектуры.
HEAD_PREFIXES = ("edge_", "pair.", "rep_")


class BoundarySegNet(PrimitiveSegNet):
    """Эмбеддинг и тип на треугольник, логит «одна грань» на ребро меша."""

    def __init__(self, in_channels: int, emb_size: int, head: HeadCfg, dropout: float = 0.0):
        super().__init__(in_channels, emb_size, dropout=dropout)
        self.edge_conv = nn.Conv1d(256, head.hidden, kernel_size=1)
        self.edge_norm = nn.GroupNorm(4, head.hidden)
        self.edge_out = nn.Conv1d(head.hidden, head.pair_channels, kernel_size=1)
        self.product = head.product
        pair_in = head.pair_channels * (3 if head.product else 2) + 1
        self.pair = nn.Sequential(nn.Linear(pair_in, head.pair_hidden), nn.ReLU(),
                                  nn.Linear(head.pair_hidden, 1))
        self.repulsive = head.repulsive
        if head.repulsive:
            self.rep_conv = nn.Conv1d(256, head.hidden, kernel_size=1)
            self.rep_norm = nn.GroupNorm(4, head.hidden)
            self.rep_out = nn.Conv1d(head.hidden, head.pair_channels, kernel_size=1)
            self.rep_pair = nn.Sequential(nn.Linear(pair_in, head.pair_hidden), nn.ReLU(),
                                          nn.Linear(head.pair_hidden, 1))

    def forward(self, x: Tensor, idx: list[Tensor], edges: Tensor | None = None,
                pairs: Tensor | None = None):
        """Эмбеддинг `(B, emb, N)`, логиты типа `(B, n_types, N)` и, если даны
        рёбра `(E, 2)`, логит «одна грань» `(E,)`. Если даны дальние пары `(P, 2)`,
        четвёртым выходом — их логит «одна грань» `(P,)`."""
        pooled, local = self.encoder(x, idx)
        h = torch.cat([pooled.unsqueeze(-1).expand(-1, -1, local.shape[2]), local], dim=1)
        h = F.dropout(F.relu(self.norm1(self.conv1(h))), self.dropout, self.training)
        h = F.dropout(F.relu(self.norm2(self.conv2(h))), self.dropout, self.training)
        e = F.dropout(F.relu(self.emb_norm(self.emb_conv(h))), self.dropout, self.training)
        t = F.dropout(F.relu(self.type_norm(self.type_conv(h))), self.dropout, self.training)
        edge = None if edges is None else self.edge_logits(h, x, edges)
        if pairs is None:
            return self.emb_out(e), self.type_out(t), edge
        b = self.rep_out(F.relu(self.rep_norm(self.rep_conv(h))))[0]
        return self.emb_out(e), self.type_out(t), edge, self._pair_logits(self.rep_pair, b, x, pairs)

    def edge_logits(self, h: Tensor, x: Tensor, edges: Tensor) -> Tensor:
        b = self.edge_out(F.relu(self.edge_norm(self.edge_conv(h))))[0]    # (C, N)
        return self._pair_logits(self.pair, b, x, edges)

    def _pair_logits(self, mlp: nn.Module, b: Tensor, x: Tensor, edges: Tensor) -> Tensor:
        bi, bj = b[:, edges[:, 0]].t(), b[:, edges[:, 1]].t()
        # Косинус между нормалями соседей: от поворота детали не зависит.
        n = x[0, NORMAL]
        cos = (n[:, edges[:, 0]] * n[:, edges[:, 1]]).sum(0).unsqueeze(1)
        parts = [(bi - bj).abs(), bi + bj] + ([bi * bj] if self.product else [])
        return mlp(torch.cat(parts + [cos], dim=1)).squeeze(1)


def boundary_strength(edge: Tensor):
    """Логиты «одна грань» -> сила границы `1 − P` на ребро для нарезки."""
    return (1.0 - torch.sigmoid(edge)).reshape(-1).double().cpu().numpy()


def head_config(head: HeadCfg, state_dict: dict) -> HeadCfg:
    """Конфиг голов под чекпоинт: голова отталкивающих связей есть, если её веса есть."""
    return dataclasses.replace(head, repulsive=any(k.startswith("rep_") for k in state_dict))


def load_backbone(model: BoundarySegNet, state_dict: dict) -> list[str]:
    """Веса основной архитектуры в сеть с головой границ. Возвращает ключи,
    оставшиеся случайными; чужие или недостающие ключи вне головы — ошибка."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    stray = [k for k in missing if not k.startswith(HEAD_PREFIXES)]
    if unexpected or stray:
        raise RuntimeError(f"чекпоинт не от основной архитектуры: лишние {unexpected}, недостающие {stray}")
    return missing


def head_parameters(model: BoundarySegNet) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.startswith(HEAD_PREFIXES)]
