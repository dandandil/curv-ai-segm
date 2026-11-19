"""Лоссы: triplet на эмбеддинге и кросс-энтропия на голове типа.

Triplet повторяет `EmbeddingLoss.triplet_loss` из ParSeNet по смыслу и по
нормировке, но считается сразу по всем парам граней, а не по двум с половиной
десяткам случайных: граней у нас не больше пятидесяти, и полный перебор пар
стоит меньше, чем цикл по выборке.

Считается на сырых нормированных эмбеддингах, до всякой кластеризации:
mean-shift в обучении не участвует (`if_mean_shift=False` в оригинале).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Потолок числа точек, берущихся с одной грани. Из оригинала: там
# `min(N // число_граней + 1, 30)`.
MAX_SAMPLES_PER_FACE = 30


def triplet_loss(embedding: Tensor, labels: Tensor, margin: float,
                 mask: Tensor | None = None,
                 generator: torch.Generator | None = None) -> Tensor:
    """`embedding` — `(D, N)`, `labels` — `(N,)` номера граней B-Rep.

    `mask` отмечает треугольники, которые можно брать в выборку; грани, у
    которых не осталось ни одного, из лосса выпадают.
    """
    valid = labels >= 0 if mask is None else (labels >= 0) & mask
    labels = labels[valid]
    z = F.normalize(embedding.t()[valid], p=2, dim=1)

    faces = torch.unique(labels)
    if faces.numel() < 2:
        return embedding.sum() * 0.0

    n_samples = min(z.shape[0] // faces.numel() + 1, MAX_SAMPLES_PER_FACE)
    idx = _sample_per_face(labels, faces, n_samples, generator)
    n_faces = faces.numel()

    # Эмбеддинги нормированы, поэтому квадрат расстояния — это 2 - 2·<a, b>.
    flat = z[idx].reshape(n_faces * n_samples, -1)
    dist = (2.0 - 2.0 * flat @ flat.t()).view(n_faces, n_samples, n_faces, n_samples)
    within = dist.diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    # constraint[a, b, i, j]: якорь i и позитив j с грани a против негатива j
    # с грани b — та же расстановка индексов, что в оригинале.
    constraint = F.relu(within.unsqueeze(1) - dist.permute(0, 2, 1, 3) + margin)
    same_point = torch.eye(n_samples, dtype=torch.bool, device=z.device)
    constraint = constraint.masked_fill(same_point, 0.0)

    # Нормировка на число нарушенных ограничений, а не на их количество:
    # иначе пары, где всё уже выполнено, размывают градиент.
    per_pair = constraint.sum((2, 3)) / (constraint > 0).sum((2, 3)).clamp(min=1)
    other_face = ~torch.eye(n_faces, dtype=torch.bool, device=z.device)
    return per_pair[other_face].mean()


def type_loss(logits: Tensor, types: Tensor, mask: Tensor | None = None) -> Tensor:
    """Кросс-энтропия на голове типа. Неизвестный тип помечен -1 и пропускается."""
    if mask is not None:
        types = types.masked_fill(~mask, -1)
    return F.cross_entropy(logits.t(), types, ignore_index=-1)


def _sample_per_face(labels: Tensor, faces: Tensor, n_samples: int,
                     generator: torch.Generator | None) -> Tensor:
    """По `n_samples` индексов с каждой грани, с повторами: мелкой грани может
    не хватить треугольников, а выборка должна быть одного размера."""
    out = []
    for face in faces:
        pos = torch.nonzero(labels == face, as_tuple=False).squeeze(1)
        pick = torch.randint(pos.numel(), (n_samples,), device=labels.device,
                             generator=generator)
        out.append(pos[pick])
    return torch.stack(out)
