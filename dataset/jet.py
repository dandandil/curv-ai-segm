"""Подгонка Monge-струи 4-й степени и инвариантные скаляры.

Копия `jets.monge_jet4` и `jets.stage_scalars` из `~/scan2cad/colouring_test`
с одним отличием: соседи передаются снаружи, а не ищутся внутри по k ближайшим.
kNN по расстоянию пробивает стенку тоньше окрестности (README, «Окрестность
струи»), а окрестность по связности меша строится в `features.py`.
"""
from __future__ import annotations

import numpy as np

POWERS = [(i, j) for i in range(5) for j in range(5) if i + j <= 4]
POS = {p: n for n, p in enumerate(POWERS)}

# Элемент (p, q) нормальной матрицы — момент степени (i_p + i_q, j_p + j_q).
_II = np.array([i for i, j in POWERS])
_JJ = np.array([j for i, j in POWERS])
_MI = _II[:, None] + _II[None, :]
_MJ = _JJ[:, None] + _JJ[None, :]
DERIV_NAMES = ["fx", "fy", "fxx", "fxy", "fyy", "fxxx", "fxxy", "fxyy", "fyyy",
               "fxxxx", "fxxxy", "fxxyy", "fxyyy", "fyyyy"]


def tangent_frame(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Произвольный ортонормированный касательный базис на нормали."""
    aux = np.tile([0.0, 0.0, 1.0], (len(normals), 1))
    aux[np.abs(normals[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
    t1 = np.cross(aux, normals)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(normals, t1)


def monge_jet4(points: np.ndarray, normals: np.ndarray, centers: np.ndarray,
               idx: np.ndarray, ridge: float = 1e-8, chunk: int = 20000
               ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Производные f в центрах, RMS-невязка и радиус окрестности, мм.

    `centers` — номера точек, в которых строится струя; `idx` формы (m, k) —
    номера их соседей в `points`.
    """
    V = np.asarray(points, float)
    N = np.asarray(normals, float)
    nv = len(centers)
    out = {nm: np.zeros(nv) for nm in DERIV_NAMES}
    resid = np.zeros(nv)
    radius = np.zeros(nv)
    for s in range(0, nv, chunk):
        e = min(s + chunk, nv)
        c = centers[s:e]
        n = N[c]
        t1, t2 = tangent_frame(n)
        d = V[idx[s:e]] - V[c][:, None, :]
        x = np.einsum("mkc,mc->mk", d, t1)
        y = np.einsum("mkc,mc->mk", d, t2)
        z = np.einsum("mkc,mc->mk", d, n)
        h = np.maximum(np.hypot(x, y).max(axis=1), 1e-12)
        radius[s:e] = h
        X, Y, Z = x / h[:, None], y / h[:, None], z / h[:, None]

        # A^T A через моменты, без матрицы плана (m, k, 15): разных моментов до
        # 8-й степени 45, остальные элементы матрицы — повторы.
        mnt = np.zeros((e - s, 9, 9))
        Atb = np.empty((e - s, 15))
        xa = np.ones_like(X)
        for a in range(9):
            yb = np.ones_like(Y)
            for b in range(9 - a):
                mnt[:, a, b] = np.einsum("mk,mk->m", xa, yb)
                if a + b <= 4:
                    Atb[:, POS[(a, b)]] = np.einsum("mk,mk,mk->m", xa, yb, Z)
                yb *= Y
            xa *= X

        AtA = mnt[:, _MI, _MJ]
        AtA += (ridge * np.trace(AtA, axis1=1, axis2=2)[:, None, None] / 15.0) * np.eye(15)
        coef_all = np.linalg.solve(AtA, Atb[:, :, None])[:, :, 0]

        # Невязка прямым предсказанием: через нормальные уравнения теряется
        # половина значащих цифр на разности близких больших чисел.
        pred = np.zeros_like(Z)
        xa = np.ones_like(X)
        for a in range(5):
            yb = np.ones_like(Y)
            for b in range(5 - a):
                pred += coef_all[:, POS[(a, b)], None] * (xa * yb)
                yb *= Y
            xa *= X
        resid[s:e] = np.sqrt(((pred - Z) ** 2).mean(axis=1)) * h

        def coef(i, j):
            return coef_all[:, POS[(i, j)]] * h ** (1 - i - j)

        out["fx"][s:e], out["fy"][s:e] = coef(1, 0), coef(0, 1)
        out["fxx"][s:e], out["fxy"][s:e], out["fyy"][s:e] = 2 * coef(2, 0), coef(1, 1), 2 * coef(0, 2)
        out["fxxx"][s:e], out["fxxy"][s:e] = 6 * coef(3, 0), 2 * coef(2, 1)
        out["fxyy"][s:e], out["fyyy"][s:e] = 2 * coef(1, 2), 6 * coef(0, 3)
        out["fxxxx"][s:e], out["fxxxy"][s:e] = 24 * coef(4, 0), 6 * coef(3, 1)
        out["fxxyy"][s:e] = 4 * coef(2, 2)
        out["fxyyy"][s:e], out["fyyyy"][s:e] = 6 * coef(1, 3), 24 * coef(0, 4)
    return out, resid, radius


def stage_scalars(J: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """‖S‖, ‖∇II‖ и H, K. Все инвариантны к повороту касательного базиса."""
    a, b = J["fx"], J["fy"]
    W = np.sqrt(1.0 + a ** 2 + b ** 2)
    detI = W ** 2
    e, f, g = J["fxx"] / W, J["fxy"] / W, J["fyy"] / W
    s11 = ((1 + b ** 2) * e - a * b * f) / detI
    s12 = ((1 + b ** 2) * f - a * b * g) / detI
    s21 = ((1 + a ** 2) * f - a * b * e) / detI
    s22 = ((1 + a ** 2) * g - a * b * f) / detI
    H = 0.5 * (s11 + s22)
    K = s11 * s22 - s12 * s21
    d1 = np.sqrt(np.maximum(4 * H ** 2 - 2 * K, 0.0))
    d2 = np.sqrt(J["fxxx"] ** 2 + 3 * J["fxxy"] ** 2
                 + 3 * J["fxyy"] ** 2 + J["fyyy"] ** 2)
    return dict(d1=d1, d2=d2, H=H, K=K)
