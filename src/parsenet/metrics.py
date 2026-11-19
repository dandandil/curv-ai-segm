"""IoU сегментации ParSeNet: `SIOU_matched_segments` / `matching_iou` авторов.

Метки и кластеры переводятся в сплошную нумерацию, обе матрицы принадлежности
дополняются до 50 столбцов, пары сопоставляются венгерским алгоритмом по IoU
(у них lapsolver, здесь scipy — оба решают ту же задачу о назначениях на
квадратной матрице). Среднее — по всем парам, кроме «пусто — пусто»: лишний
кластер в паре с пустой гранью даёт 0, так что пере-нарезка штрафуется.
Больше 50 меток их код не допускает; здесь ширина растёт до нужной.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def segment_iou(target: np.ndarray, pred: np.ndarray, width: int = 50) -> float:
    target = np.unique(target, return_inverse=True)[1]
    pred = np.unique(pred, return_inverse=True)[1]
    width = max(width, int(target.max()) + 1, int(pred.max()) + 1)
    p = np.eye(width, dtype=np.float64)[pred]
    g = np.eye(width, dtype=np.float64)[target]
    dots = p.T @ g
    cost = dots / (p.sum(0)[:, None] + g.sum(0)[None, :] - dots + 1e-7)
    rows, cols = linear_sum_assignment(1.0 - cost)
    ious = []
    for r, c in zip(rows, cols):
        in_pred, in_gt = pred == r, target == c
        if not in_pred.any() and not in_gt.any():
            continue
        ious.append(np.logical_and(in_pred, in_gt).sum() / (np.logical_or(in_pred, in_gt).sum() + 1e-8))
    return float(np.mean(ious))
