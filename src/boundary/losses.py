"""Цель, лосс и метрики головы границ."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def edge_targets(labels: Tensor, ambiguous: Tensor, edges: Tensor, pos_weight_max: float,
                 tangent: Tensor | None = None, tangent_weight: float = 1.0,
                 face_balance: bool = False) -> tuple[Tensor, Tensor]:
    """Цель «одна грань» на ребро и вес.

    Рёбра с одинаковыми метками, у которых хотя бы один треугольник в полосе у
    стыка, не учитываются: положение стыка после переноса разметки и
    скругления шумом плавает на кольцо, и штраф за сдвиг на кольцо учил бы
    шуму. Стыки остаются — у стыкового ребра оба треугольника в полосе всегда.
    Стыки редки, поэтому их вес — отношение классов на детали. Стыки между
    касательными гранями (`tangent`) дополнительно умножаются на
    `tangent_weight`: излома там почти нет, и сеть их недобирает.

    `face_balance` — вес не на ребро, а на грань: см. `face_weights`.
    `pos_weight_max` тогда не нужен — классы уравнены внутри каждой грани.
    """
    a, b = edges[:, 0], edges[:, 1]
    same = labels[a] == labels[b]
    inside = same & ~(ambiguous[a] | ambiguous[b])
    joint = ~same
    if face_balance:
        weight = face_weights(labels, inside, joint, edges)
    else:
        ratio = (inside.sum() / joint.sum().clamp(min=1)).clamp(max=pos_weight_max)
        weight = inside.float() + joint.float() * ratio.float()
    if tangent is not None:
        weight = torch.where(tangent & joint, weight * tangent_weight, weight)
    return same.float(), weight


def face_weights(labels: Tensor, inside: Tensor, joint: Tensor, edges: Tensor) -> Tensor:
    """Каждая грань весит одинаково: 1/2 на её внутренние рёбра, 1/2 на её стык.

    При весе на ребро грань весит по числу своих рёбер, и узкая грань в двадцать
    рёбер тонет рядом с широкой в тысячу. Зонд на признаках головы различает стык
    и внутренность узкой грани с AUC 0.81 одним правилом на все рёбра, а голова —
    0.68 (README, «Зонд на рёбрах»). Стыковое ребро получает долю от обеих
    граней. Половина на половину — то же отношение классов, что давал вес на
    ребро по детали в целом, только теперь внутри каждой грани. У грани без
    внутренних рёбер в лоссе остаётся только её стык.
    """
    a, b = edges[:, 0], edges[:, 1]
    n = int(labels.max()) + 1
    la, lb = labels[a], labels[b]
    n_inside = torch.bincount(la[inside], minlength=n).float()
    n_joint = torch.bincount(torch.cat([la[joint], lb[joint]]), minlength=n).float()
    weight = torch.zeros(len(a), device=labels.device)
    weight[inside] = 0.5 / n_inside[la[inside]]
    weight[joint] = 0.5 / n_joint[la[joint]] + 0.5 / n_joint[lb[joint]]
    return weight


def _softmin(x: Tensor, group: Tensor, n: int, tau: float) -> Tensor:
    """Мягкий минимум по группам, `−τ·log Σ exp(−x/τ)`; у пустой группы +inf.

    Жёсткий минимум двигал бы за шаг одно ребро; мягкий раздаёт градиент
    нескольким самым слабым. Сдвиг на максимум — только для устойчивости и
    на результат не влияет, поэтому без градиента.
    """
    y = -x / tau
    top = torch.full((n,), float("-inf"), device=x.device, dtype=x.dtype)
    top = top.scatter_reduce(0, group, y.detach(), "amax", include_self=True)
    total = torch.zeros(n, device=x.device, dtype=x.dtype).index_add(0, group, torch.exp(y - top[group]))
    return -tau * (top + torch.log(total))


def contrast_loss(logits: Tensor, labels: Tensor, edges: Tensor, margin: float, tau: float) -> Tensor:
    """Штраф за то, что водораздел не отделит соседние грани друг от друга.

    Водораздел держит два бассейна раздельными, если перевал между ними выше
    более мелкого из двух доньев хотя бы на `depth` (`partition.watershed`).
    Здесь то же условие для истинных граней: дно грани — мягкий минимум силы по
    её внутренним рёбрам, перевал пары — по рёбрам их стыка, штраф
    `max(0, margin − (перевал − max(дно_f, дно_g)))`. Кросс-энтропия учит каждое
    ребро по отдельности и довольна узкой гранью с силой 0.76 внутри и 0.86 на
    стыке; водоразделу нужен именно перепад (README, «Где теряются грани у
    `res60_tight`»).

    Дно берётся по рёбрам, оба треугольника которых не касаются чужой грани:
    стык после переноса разметки плавает на кольцо, и ребро у самого стыка
    учило бы шуму. У грани шириной в один-два треугольника таких рёбер нет, и
    она из этого члена выпадает.
    """
    strength = torch.sigmoid(-logits)                  # 1 − P(одна грань)
    a, b = edges[:, 0], edges[:, 1]
    la, lb = labels[a], labels[b]
    joint = la != lb
    if not joint.any():
        return logits.sum() * 0.0
    touch = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    touch[a[joint]] = True
    touch[b[joint]] = True
    inner = ~joint & ~touch[a] & ~touch[b]
    n = int(labels.max()) + 1
    bottom = _softmin(strength[inner], la[inner], n, tau)
    pair = torch.minimum(la, lb)[joint] * n + torch.maximum(la, lb)[joint]
    keys, inverse = torch.unique(pair, return_inverse=True)
    passes = _softmin(strength[joint], inverse, len(keys), tau)
    f, g = keys // n, keys % n
    ok = torch.isfinite(bottom[f]) & torch.isfinite(bottom[g])
    if not ok.any():
        return logits.sum() * 0.0
    gap = passes[ok] - torch.maximum(bottom[f[ok]], bottom[g[ok]])
    return F.relu(margin - gap).mean()


def repulsive_loss(logits: Tensor, labels: Tensor, pairs: Tensor) -> Tensor:
    """Кросс-энтропия «одна грань» на дальних парах, классы весят поровну.

    Как у Wolf et al. (TPAMI 2020), где лосс сбалансирован по классам для каждого
    вида связей: иначе вероятности смещены к частому классу, а Mutex Watershed
    сравнивает притягивающие и отталкивающие веса на одной шкале. Без баланса в
    проверке без обучения нарезка слипалась (README, «Mutex Watershed»).
    """
    target = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).float()
    n_pos = target.sum().clamp(min=1)
    n_neg = (1 - target).sum().clamp(min=1)
    weight = target * (0.5 / n_pos) + (1 - target) * (0.5 / n_neg)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum()


def edge_loss(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=1e-12)


@torch.no_grad()
def edge_scores(logits: Tensor, labels: Tensor, ambiguous: Tensor, edges: Tensor,
                tangent: Tensor | None = None) -> dict:
    """Точность и полнота стыков с допуском в одно кольцо.

    Предсказанный стык верен, если он стык или лежит в полосе у стыка. Стык
    найден, если стыком помечено само ребро или любое ребро его треугольников.

    Допуск считает верным любой стык в полосе, поэтому метрика не видит, что
    голова делает ВНУТРИ узких граней: у грани шириной в пару колец там вся
    внутренность и есть полоса. Прогон `runs/boundary_band` дал ту же F1 при
    другом лоссе, и различие нашлось только прямым замером силы границы внутри
    граней (README, «Полоса у стыка в лоссе головы»).
    """
    a, b = edges[:, 0], edges[:, 1]
    joint = labels[a] != labels[b]
    pred = logits < 0
    near = joint | ambiguous[a] | ambiguous[b]
    touched = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    touched[a[pred]] = True
    touched[b[pred]] = True
    precision = float((pred & near).sum() / pred.sum().clamp(min=1))
    found = (pred | touched[a] | touched[b]) & joint
    recall = float(found.sum() / joint.sum().clamp(min=1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {"precision": precision, "recall": recall, "f1": f1}
    if tangent is not None:
        # Полнота отдельно на касательных стыках; NaN, если их на детали нет.
        tangent = tangent & joint
        out["tangent_recall"] = float((found & tangent).sum() / tangent.sum()) if tangent.any() else float("nan")
    return out
