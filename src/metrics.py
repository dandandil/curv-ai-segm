"""Метрики сегментации.

Доля верных рёбер неинформативна: граница между гранями — это доли процента
треугольников, и её можно полностью проиграть, оставшись при 99% (README,
«Метрики»). Поэтому меряется другое — что именно сломалось:

* протечка — один кластер накрыл несколько граней B-Rep, ошибка необратимая:
  дальше по конвейеру такой кластер не разложится обратно;
* лишний разрез — одна грань разошлась по нескольким кластерам, ошибка
  дешёвая: куски сливаются переподгонкой примитива;
* грань без кластера — ни один кластер не взял её большинством;
* ширина полосы границы — на сколько треугольников ошибка отходит от истинной
  границы. Ошибка вплотную к границе неизбежна: истинная линия B-Rep после
  ремеша не лежит на рёбрах меша.

Отдельно те же числа по типам поверхностей: у конусов и торов мода в
пространстве кривизн вырождается в линию, а не в сгусток, и средняя цифра это
скрыла бы.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

NAN = float("nan")
# Грани мельче стольких треугольников не восстановит никакая сегментация: после
# децимации это одна-две ленты у ребра. Из счёта они выбрасываются.
MIN_FACE = 5


def overlap(true_labels: Tensor, pred_labels: Tensor) -> Tensor:
    """Таблица `(граней, кластеров)`: сколько треугольников грани попало в кластер.

    Треугольники с отрицательной меткой в таблицу не попадают: разметка их
    сейчас не порождает, но метрика не должна падать, если появятся.
    """
    keep = (true_labels >= 0) & (pred_labels >= 0)
    true_labels, pred_labels = true_labels[keep], pred_labels[keep]
    n_true = int(true_labels.max()) + 1 if keep.any() else 1
    n_pred = int(pred_labels.max()) + 1 if keep.any() else 1
    flat = torch.bincount(true_labels * n_pred + pred_labels, minlength=n_true * n_pred)
    return flat.view(n_true, n_pred).float()


def segment_scores(true_labels: Tensor, pred_labels: Tensor) -> dict:
    """Протечки, лишние разрезы, потерянные грани и IoU.

    Каждой грани сопоставляется доминирующий кластер — тот, куда попало больше
    всего её треугольников, — а каждому кластеру доминирующая грань.

    Протечка — грань, чей доминирующий кластер уже достался другой грани:
    слияние N граней в один кластер даёт N - 1 протечку. Лишний разрез —
    кластер, чья доминирующая грань уже владеет другим кластером: грань,
    разбитая на три куска, даёт два разреза.

    Прежняя версия засчитывала грань в протечку, только если она занимала не
    меньше 20% кластера, и при полном слиянии молчала: в одном кластере из
    двадцати граней каждая занимает около 5%, и метрика показывала 2-3 протечки
    вместо 19.
    """
    m = overlap(true_labels, pred_labels)
    alive = m.sum(1) > 0
    used = m.sum(0) > 0
    face_to_cluster = m.argmax(1)[alive]
    cluster_to_face = m.argmax(0)[used]

    per_face = m.sum(1, keepdim=True)
    per_cluster = m.sum(0, keepdim=True)
    iou = (m / (per_face + per_cluster - m).clamp(min=1)).max(1).values[alive]
    return {
        "leaks": float(face_to_cluster.numel() - torch.unique(face_to_cluster).numel()),
        "splits": float(cluster_to_face.numel() - torch.unique(cluster_to_face).numel()),
        "faces": int(alive.sum()),
        "clusters": int(used.sum()),
        "missed_frac": float((iou < 0.5).float().mean()),
        "mean_iou": float(iou.mean()),
    }


def boundary_distance(neighbors: Tensor, true_labels: Tensor, max_rings: int = 8) -> Tensor:
    """Расстояние в треугольниках до ближайшей границы граней B-Rep.

    Обход в ширину от полосы, у которой есть сосед из другой грани. Одна карта
    расстояний на весь меш, а не окрестность у каждого треугольника, поэтому
    считается за один проход.
    """
    valid = neighbors >= 0
    other = valid & (true_labels[neighbors.clamp(min=0)] != true_labels[:, None])
    dist = torch.full_like(true_labels, max_rings + 1)
    dist[other.any(1)] = 0
    for r in range(1, max_rings + 1):
        step = neighbors[dist == r - 1]
        reached = torch.zeros_like(dist, dtype=torch.bool)
        reached[step[step >= 0]] = True
        dist[reached & (dist > r)] = r
    return dist


def boundary_profile(true_labels: Tensor, pred_labels: Tensor, dist: Tensor,
                     max_rings: int = 8) -> dict:
    """Доля ошибочных треугольников по расстоянию до границы.

    Ошибочным считается тот, чей кластер не совпадает с кластером, взявшим его
    грань большинством: сами номера кластеров произвольны и сравнивать их
    напрямую с номерами граней нельзя.
    """
    m = overlap(true_labels, pred_labels)
    expected = m.argmax(1)[true_labels]
    wrong = pred_labels != expected
    return {f"wrong@{r}": float(wrong[dist == r].float().mean()) if (dist == r).any() else NAN
            for r in range(max_rings + 1)}


def type_miou(logits: Tensor, types: Tensor, n_classes: int) -> float:
    """Средний IoU по типам примитивов — метрика ParSeNet (`evaluate_miou`).

    Воспроизведена дословно, включая одну особенность: класс, которого нет ни в
    разметке, ни в предсказании, даёт IoU = eps/eps = 1 и входит в среднее. Это
    завышает метрику тем сильнее, чем меньше типов на детали, но менять формулу
    значит потерять сопоставимость с числами оригинала.
    """
    pred = logits.argmax(0)
    eps = torch.finfo(torch.float32).eps
    total = 0.0
    for c in range(n_classes):
        gt_c, pred_c = types == c, pred == c
        inter = float((gt_c & pred_c).sum()) + eps
        union = float((gt_c | pred_c).sum()) + eps
        total += inter / union
    return total / n_classes


def type_scores(logits: Tensor, types: Tensor, true_labels: Tensor) -> dict:
    """Точность головы типа: на треугольник и на грань (голосованием)."""
    pred = logits.argmax(0)
    known = types >= 0
    voted = overlap(true_labels, pred)

    # Тип грани берётся присваиванием, а не голосованием: все треугольники
    # одной грани B-Rep несут один тип по построению. Грани, у которых он
    # неизвестен, остаются -1 и выпадают из счёта — приравнять их к нулевому
    # классу нельзя, иначе сеть, назвавшая такую грань плоскостью, получила бы
    # зачёт ни за что.
    truth = torch.full((voted.shape[0],), -1, dtype=types.dtype, device=types.device)
    truth[true_labels[known]] = types[known]
    alive = (voted.sum(1) > 0) & (truth >= 0)
    return {
        "type_acc_tri": float((pred[known] == types[known]).float().mean()) if known.any() else NAN,
        "type_acc_face": float((voted.argmax(1)[alive] == truth[alive]).float().mean())
        if alive.any() else NAN,
    }


def by_type(scores_per_part: list[dict], key: str) -> dict:
    """Среднее метрики по деталям, сгруппированное по преобладающему типу."""
    out: dict[str, list[float]] = {}
    for row in scores_per_part:
        out.setdefault(row["dominant_type"], []).append(row[key])
    return {t: float(np.mean(v)) for t, v in sorted(out.items())}


# Границы точной полуширины грани в мм для разбивки метрик (README, «Узкие грани: ширина в мм и
# правило разметки»): 0.5 — порог правила узких граней, ниже остаются только развороты.
WIDTH_EDGES_MM = (0.5, 1.0, 3.0)
WIDTH_NAMES = ("le0.5", "0.5-1", "1-3", "gt3")


def face_halfwidth(adjacency: np.ndarray, labels: np.ndarray, cap: int = 32) -> np.ndarray:
    """Полуширина каждой грани в кольцах треугольников, не больше `cap`.

    Потолок 32 кольца — около 6 мм при 30 тысячах: выше верхней границы
    разбивки (3 мм) с запасом, иначе широкие грани все попали бы в одну полосу.

    Обход в ширину от полосы у стыка по рёбрам внутри граней; полуширина грани —
    наибольшее расстояние до края среди её треугольников.
    """
    a, b = adjacency[:, 0], adjacency[:, 1]
    same = labels[a] == labels[b]
    n = len(labels)
    dist = np.full(n, cap)
    front = np.zeros(n, bool)
    front[a[~same]] = front[b[~same]] = True
    dist[front] = 0
    sa, sb = a[same], b[same]
    for r in range(1, cap):
        nxt = np.zeros(n, bool)
        nxt[sb[front[sa]]] = nxt[sa[front[sb]]] = True
        nxt &= dist == cap
        dist[nxt] = r
        front = nxt
    width = np.zeros(int(labels.max()) + 1, int)
    np.maximum.at(width, labels, dist)
    return width


def face_records(true_labels: Tensor, pred_labels: Tensor, types: Tensor, halfwidth_mm: np.ndarray,
                 type_names) -> list[tuple[float, str, str]]:
    """По грани: IoU, тип поверхности, полоса полуширины — для `grouped_scores`. Грань без
    точной полуширины (NaN, детали Fusion) получает полосу «?»."""
    m = overlap(true_labels, pred_labels)
    per_face = m.sum(1, keepdim=True)
    iou = (m / (per_face + m.sum(0, keepdim=True) - m).clamp(min=1)).max(1).values.cpu().numpy()
    face_type = np.full(len(iou), -1)
    known = types >= 0
    face_type[true_labels[known].cpu().numpy()] = types[known].cpu().numpy()
    band = np.digitize(halfwidth_mm, WIDTH_EDGES_MM, right=True)
    alive = per_face[:, 0].cpu().numpy() > 0
    return [(float(iou[f]), type_names[face_type[f]] if face_type[f] >= 0 else "?",
             WIDTH_NAMES[band[f]] if np.isfinite(halfwidth_mm[f]) else "?")
            for f in np.nonzero(alive)[0] if f < len(halfwidth_mm)]


def grouped_scores(records: list[tuple[float, str, str]]) -> dict:
    """IoU и доля потерянных (IoU < 0.5) по типам и по полуширине, по всем граням
    вместе: мелкой группы на одной детали мало, и среднее по деталям шумело бы."""
    out = {}
    for pos, prefix in ((1, "type"), (2, "width")):
        groups: dict[str, list[float]] = {}
        for rec in records:
            groups.setdefault(rec[pos], []).append(rec[0])
        for name, v in groups.items():
            v = np.asarray(v)
            out[f"iou_{prefix}_{name}"] = float(v.mean())
            out[f"lost_{prefix}_{name}"] = float((v < 0.5).mean())
            out[f"n_{prefix}_{name}"] = int(len(v))
    return out


def seam_cuts(adjacency: np.ndarray, cores: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Вывод кусками (README, «Края кусков»): доля рёбер внутри граней, по которым нарезка развела
    кластеры, — на шве ядер и вдали от него. Выше на шве — режет сам шов."""
    a, b = adjacency[:, 0], adjacency[:, 1]
    same = true_labels[a] == true_labels[b]
    seam = cores[a] != cores[b]
    cut = pred_labels[a] != pred_labels[b]
    rate = lambda m: float(cut[m].mean()) if m.any() else NAN  # noqa: E731
    return {"seam_edges": int((same & seam).sum()), "seam_cut": rate(same & seam),
            "inner_edges": int((same & ~seam).sum()), "inner_cut": rate(same & ~seam)}
