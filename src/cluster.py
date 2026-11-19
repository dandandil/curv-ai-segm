"""Mean-shift по эмбеддингам — только на инференсе.

В обучении кластеризация не участвует: в `train_parsenet.py` оригинала стоит
`EmbeddingLoss(margin=1.0, if_mean_shift=False)`, и эмбеддинг учится чистым
triplet-лоссом. Отсюда рассогласование train/test — учим «стягивай и
расталкивай», а режем поиском мод плотности; оригинал это принимает, и мы тоже
(README, «Кластеризация»). Компенсируется связыванием порогов: margin должен
быть заметно больше ширины ядра, иначе сгустки перетекают и грани сливаются
при довольном лоссе.

Ядро гауссово и итераций десять — по коду оригинала; статья описывает
von Mises-Fisher и сходимость примерно за полсотни итераций.

Всё считается чанками: у нас 30 тысяч точек против семи тысяч у оригинала, и
матрица N x N целиком уже не нужна ни по памяти, ни по смыслу.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Во сколько раз расширять фиксированную ширину ядра, когда кластеров больше
# потолка: шаг мельче удвоения квантиля, чтобы не уйти от подобранной ширины
# далеко. [не подбиралось]
WIDTH_GROWTH = 1.25


@torch.no_grad()
def bandwidth(z: Tensor, quantile: float, n_samples: int, floor: float) -> Tensor:
    """Ширина ядра как квантиль расстояний до ближайших соседей по выборке.

    Клип снизу — как в оригинале: при вырожденно узкой ширине ядро становится
    дельта-функцией и mean-shift не двигается.
    """
    pick = torch.randperm(z.shape[0], device=z.device)[:n_samples]
    sample = z[pick]
    dist = (2.0 - 2.0 * sample @ sample.t()).clamp(min=0)
    # Квантиль приходит от `segment`, который удваивает его при переборе
    # кластеров, поэтому он может выйти за единицу — а соседей не больше, чем
    # точек в выборке.
    k = min(max(int(quantile * sample.shape[0]), 1), sample.shape[0])
    return dist.topk(k, dim=1, largest=False).values[:, -1].clamp(min=1e-6).sqrt().mean().clamp(min=floor)


@torch.no_grad()
def mean_shift(z: Tensor, width: Tensor, iterations: int, chunk: int) -> Tensor:
    """Сдвиг к модам плотности на единичной сфере.

    `z` обязан быть нормированным: квадрат расстояния берётся как `2 - 2<a, b>`,
    и всё ядро выражается через одно матричное произведение.
    """
    shifted = z.clone()
    for _ in range(iterations):
        out = torch.empty_like(shifted)
        for a in range(0, shifted.shape[0], chunk):
            part = shifted[a:a + chunk]
            weight = torch.exp(-(2.0 - 2.0 * part @ z.t()) / width ** 2 / 2)
            out[a:a + chunk] = weight @ z / weight.sum(1, keepdim=True)
        shifted = F.normalize(out, p=2, dim=1)
    return shifted


@torch.no_grad()
def extract(shifted: Tensor, z: Tensor, width: Tensor, chunk: int) -> tuple[Tensor, Tensor]:
    """Немаксимальное подавление: центры кластеров и метка на точку.

    Кандидат в центры — каждая сдвинутая точка. Соседями считаются кандидаты,
    у которых КВАДРАТ расстояния меньше ширины ядра: сравнение разнородных
    величин взято из оригинала (`nms`, там же комментарий «this could be
    b ** 2`»), и менять его — значит менять поведение, а не чинить опечатку.
    Среди соседей остаётся тот, к кому приписано больше точек.

    `z` обязан быть нормированным.
    """
    owner = _nearest(shifted, z, chunk)
    population = torch.bincount(owner, minlength=shifted.shape[0]).float()
    alive = torch.nonzero(population, as_tuple=False).squeeze(1)

    winners = []
    for a in range(0, alive.numel(), chunk):
        part = shifted[alive[a:a + chunk]]
        close = (2.0 - 2.0 * part @ shifted.t()) < width
        winners.append((close * population).argmax(dim=1))
    centers = torch.unique(torch.cat(winners))
    return centers, _nearest(shifted[centers], z, chunk)


def _nearest(centers: Tensor, y: Tensor, chunk: int) -> Tensor:
    """Индекс ближайшего центра для каждой точки."""
    out = torch.empty(y.shape[0], dtype=torch.long, device=y.device)
    for a in range(0, y.shape[0], chunk):
        out[a:a + chunk] = (y[a:a + chunk] @ centers.t()).argmax(dim=1)
    return out


@torch.no_grad()
def segment(embedding: Tensor, quantile: float, iterations: int, n_samples: int,
            floor: float, chunk: int, max_clusters: int,
            width: float | None = None) -> tuple[Tensor, Tensor]:
    """Эмбеддинг `(D, N)` -> метки кластеров `(N,)` и использованная ширина.

    `width` — фиксированная ширина ядра; без неё ширина оценивается по данным
    квантилем, как в оригинале. Квантиль оригинала нам не подходит: он давал
    ширину шире внутреннего разброса грани, и соседние сгустки сливались
    (README, «Кластеризация»).

    При слишком узкой ширине кластеров выходит больше, чем граней бывает на
    детали. Тогда ширина растёт — квантиль удваивается, как в `guard_mean_shift`
    оригинала, а фиксированная — в `WIDTH_GROWTH` раз. Потолок 50
    кластеров у оригинала взят из его датасета, у нас — из фильтра корпуса по
    числу граней.
    """
    z = F.normalize(embedding.t(), p=2, dim=1)
    fixed = None if width is None else torch.tensor(width, device=z.device)
    while True:
        current = bandwidth(z, quantile, n_samples, floor) if fixed is None else fixed
        shifted = mean_shift(z, current, iterations, chunk)
        _, labels = extract(shifted, z, current, chunk)
        if torch.unique(labels).numel() <= max_clusters:
            return labels, current
        if fixed is None:
            quantile *= 2
        else:
            fixed = fixed * WIDTH_GROWTH
