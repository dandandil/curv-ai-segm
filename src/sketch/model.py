"""2D сеть сечения (README, «Эскиз из сечения», «2D сеть: как устроена»).

Вход на отрезок — координаты середины и нормаль поверхности, как у 3D сети. Слой — свёртка вдоль
цепочки с ядром 3 и шагом d: у отрезка i склейка [x(i−d) − x(i), x(i), x(i+d) − x(i)] (разность с
соседом — как пара EdgeConv в 3D; левый и правый сосед раздельно, без максимума: на цепочке
известно, что до и что после), матрица, LeakyReLU, матрица, прибавка к x(i) — слой дописывает
описание отрезка, а не заменяет его. Шаг удваивается: 1…64 дважды, обзор ±254 отрезка. Одна
матрица в слое и 7 слоёв недоучивались (`runs/section_lean7`: train = val, граница 0.023 против
0.017 у `section_xy`) — вторая матрица и второй проход возвращают ёмкость.
Выход: тип отрезка (CURVE_TYPES), на стык — граница и касание.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import SCFG
from .data import SEG_DIM, neighbors
from .section import CURVE_TYPES


class ChainConv(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(3 * c, c), nn.LeakyReLU(0.2), nn.Linear(c, c))

    def forward(self, x: Tensor, prev: Tensor, nxt: Tensor) -> Tensor:
        own = torch.arange(len(x), device=x.device)
        # за концом открытой цепочки соседа нет — разность 0
        xp = x[torch.where(prev >= 0, prev, own)]
        xn = x[torch.where(nxt >= 0, nxt, own)]
        return x + self.mix(torch.cat([xp - x, x, xn - x], -1))


class SectionNet(nn.Module):
    def __init__(self, cfg=SCFG.model):
        super().__init__()
        c = cfg.channels
        self.dilations = list(cfg.dilations)
        self.inp = nn.Linear(SEG_DIM, c)
        self.layers = nn.ModuleList(ChainConv(c) for _ in self.dilations)
        self.type_head = nn.Linear(c, len(CURVE_TYPES))
        self.junction = nn.Sequential(nn.Linear(2 * c, c), nn.LeakyReLU(0.2), nn.Linear(c, 2))

    def forward(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Логиты типа `(T, 4)`, границы и касания `(J,)`."""
        start, length, closed = batch["start"], batch["length"], batch["closed"]
        near = {d: (neighbors(start, length, closed, -d), neighbors(start, length, closed, d))
                for d in set(self.dilations)}
        x = self.inp(batch["feat"])
        for d, layer in zip(self.dilations, self.layers):
            x = layer(x, *near[d])
        j = self.junction(torch.cat([x[batch["junction_prev"]], x[batch["junction_next"]]], -1))
        return self.type_head(x), j[:, 0], j[:, 1]
