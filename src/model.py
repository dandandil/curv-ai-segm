"""Сеть сегментации: DGCNN-энкодер по мешевому графу и две головы.

Архитектура выписана по коду ParSeNet `src/PointNet.py` (классы
`DGCNNEncoderGn` и `PrimitivesEmbeddingDGCNGn`), а не по статье — они местами
расходятся, см. README, «Порядок работ». Числа каналов, группы в GroupNorm и
порядок слоёв повторены дословно.

Отход от оригинала один: соседи берутся из мешевого графа и приходят готовыми
снаружи, свои на каждый слой (`src/graph.py`), тогда как в оригинале каждый
слой заново ищет их kNN по своему входу.

Глобальный вектор считается max-pool'ом по рабочему мешу, как в оригинале.
Отдельная глобальная ветвь из кэша нужна только при нарезке на патчи: max-pool
по патчу дал бы разные значения и эмбеддинги стали бы несопоставимы. Деталь
проходит целиком, патчей нет, поэтому второй энкодер не поднимается.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Типы поверхностей. Шестой — B-spline: у ParSeNet типов тоже шесть, но он
# делит сплайны на открытые и закрытые ради SplineNet, а нам это не нужно.
PRIMITIVE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "bspline")


def graph_feature(x: Tensor, idx: Tensor) -> Tensor:
    """`(B, C, N)` и соседи `(N, K)` -> `(B, 2C, N, K)`.

    Конкатенация `[сосед − центр, центр]` — как `get_graph_feature` в
    оригинале: разность несёт локальную геометрию, центр — абсолютное значение.
    """
    b, c, n = x.shape
    k = idx.shape[1]
    center = x.unsqueeze(-1)
    neighbor = x[:, :, idx.reshape(-1)].reshape(b, c, n, k)
    return torch.cat([neighbor - center, center.expand(-1, -1, -1, k)], dim=1)


class EdgeConv(nn.Module):
    """Свёртка 1x1 по парам «центр-сосед» и max по соседям."""

    # Соседи собираются порциями по столько треугольников: тензор (C, N, K) целиком — 2.5 ГБ на кусок
    # 124 тысячи, и кусок 184 тысячи не влезал в 12 ГБ [измерено 2026-10-17] (README, «Ядро 60 тысяч»).
    CHUNK = 16384

    def __init__(self, c_in: int, c_out: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in * 2, c_out, kernel_size=1, bias=False),
            nn.GroupNorm(groups, c_out),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: Tensor, idx: Tensor) -> Tensor:
        """То же, что `block(graph_feature(x, idx)).max(-1)`, без тензора пар.

        Тензор `(N, K, C)` и GroupNorm по нему занимали 58% времени шага и
        больше половины памяти. Свёртка 1x1 по `[x_j − x_i, x_i]` линейна и
        раскладывается в `u_j + v_i`; статистики GroupNorm по всем парам
        выражаются через суммы по соседям; GroupNorm при известных статистиках
        и LeakyReLU монотонны, поэтому max по соседям переносится внутрь (у
        канала с отрицательным γ — min). В float64 выход и градиенты совпадают
        с исходной формой до 1e-15 (tests/test_edgeconv.py).
        """
        conv, gn = self.block[0], self.block[1]
        assert x.shape[0] == 1, "батч — одна деталь"
        c = x.shape[1]
        w = conv.weight[:, :, 0, 0]
        u = w[:, :c] @ x[0]                          # вклад соседа, (C_out, N)
        v = (w[:, c:] - w[:, :c]) @ x[0]             # вклад центра, (C_out, N)
        n, k = idx.shape
        flat = idx.reshape(-1)
        parts = [idx[i:i + self.CHUNK].reshape(-1) for i in range(0, n, self.CHUNK)]
        # транзитные порции: autograd хранит только индексы
        neigh_sum = torch.cat([u[:, p].view(u.shape[0], -1, k).sum(2) for p in parts], dim=1)
        count = torch.bincount(flat, minlength=n).to(u.dtype)   # сколько раз треугольник чей-то сосед

        # Первый и второй момент z = u_j + v_i по всем N·K парам, по каналу.
        mean = ((u * count).sum(1) + k * v.sum(1)) / (n * k)
        square = ((u * u * count).sum(1) + k * (v * v).sum(1)
                  + 2 * (v * neigh_sum).sum(1)) / (n * k)
        groups = gn.num_groups
        per = u.shape[0] // groups
        group_mean = mean.view(groups, per).mean(1)
        # E[z²] − E[z]² при почти постоянном канале может уйти в минус на округлении
        group_var = (square.view(groups, per).mean(1) - group_mean ** 2).clamp_min(0)
        mu = group_mean.repeat_interleave(per)
        scale = gn.weight / torch.sqrt(group_var + gn.eps).repeat_interleave(per)

        positive = scale[:, None] > 0
        extreme = []
        for p in parts:
            g = u[:, p].view(u.shape[0], -1, k)
            extreme.append(torch.where(positive, g.max(2).values, g.min(2).values))
        extreme = torch.cat(extreme, dim=1)
        out = scale[:, None] * (extreme + v - mu[:, None]) + gn.bias[:, None]
        return F.leaky_relu(out, self.block[2].negative_slope).unsqueeze(0)


class MeshEncoder(nn.Module):
    """EdgeConv по слою на набор колец, конкатенация их выходов и глобальный вектор.

    Первые три слоя дают 64, 64 и 128 каналов, 256 получается конкатенацией — в статье
    написано «each extracting a 256-D representation», и это расхождение уже
    стоило нам одного неверного замера. Каждый следующий слой (широкий контекст,
    README, «Аудит цепочки признаков») — ещё 128.
    """

    WIDTHS = (64, 64, 128)
    EXTRA = 128

    def __init__(self, in_channels: int, layers: int = 3):
        super().__init__()
        widths = list(self.WIDTHS[:layers]) + [self.EXTRA] * max(0, layers - len(self.WIDTHS))
        # Имена conv1, conv2, ... — как были у трёх слоёв: старые чекпоинты грузятся как есть.
        self.layers = len(widths)
        c = in_channels
        for i, w in enumerate(widths):
            setattr(self, f"conv{i + 1}", EdgeConv(c, w, groups=2))
            c = w
        self.out_channels = sum(widths)
        self.mlp = nn.Conv1d(self.out_channels, 1024, kernel_size=1)
        self.norm = nn.GroupNorm(8, 1024)

    def forward(self, x: Tensor, idx: list[Tensor]) -> tuple[Tensor, Tensor]:
        outs = []
        for i in range(self.layers):
            x = getattr(self, f"conv{i + 1}")(x, idx[i])
            outs.append(x)
        local = torch.cat(outs, dim=1)
        pooled = F.relu(self.norm(self.mlp(local))).max(dim=2).values
        return pooled, local


class PrimitiveSegNet(nn.Module):
    """Эмбеддинг на треугольник и тип поверхности на треугольник."""

    def __init__(self, in_channels: int, emb_size: int = 128,
                 n_types: int = len(PRIMITIVE_TYPES), dropout: float = 0.0, layers: int | None = None):
        super().__init__()
        from src.config import CFG
        # Слоёв столько, сколько наборов колец в конфиге.
        self.encoder = MeshEncoder(in_channels, layers or len(CFG.model.rings))
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1024 + self.encoder.out_channels, 512, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, 512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.norm2 = nn.GroupNorm(4, 256)

        self.emb_conv = nn.Conv1d(256, 256, kernel_size=1)
        self.emb_norm = nn.GroupNorm(4, 256)
        self.emb_out = nn.Conv1d(256, emb_size, kernel_size=1)

        self.type_conv = nn.Conv1d(256, 256, kernel_size=1)
        self.type_norm = nn.GroupNorm(4, 256)
        self.type_out = nn.Conv1d(256, n_types, kernel_size=1)

    def forward(self, x: Tensor, idx: list[Tensor]) -> tuple[Tensor, Tensor]:
        """`x` — `(B, C, N)`, `idx` — по набору соседей на слой.

        Возвращает эмбеддинг `(B, emb, N)` без нормировки (её делает лосс) и
        логиты типа `(B, n_types, N)`.
        """
        pooled, local = self.encoder(x, idx)
        h = torch.cat([pooled.unsqueeze(-1).expand(-1, -1, local.shape[2]), local], dim=1)
        h = F.dropout(F.relu(self.norm1(self.conv1(h))), self.dropout, self.training)
        h = F.dropout(F.relu(self.norm2(self.conv2(h))), self.dropout, self.training)

        e = F.dropout(F.relu(self.emb_norm(self.emb_conv(h))), self.dropout, self.training)
        t = F.dropout(F.relu(self.type_norm(self.type_conv(h))), self.dropout, self.training)
        return self.emb_out(e), self.type_out(t)
