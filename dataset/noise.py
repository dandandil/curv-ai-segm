"""Модель шума сканера поверх идеальной CAD-геометрии.

Три компоненты, и они очень разной степени обоснованности:

1. `high_frequency` — откалибрована по реальному скану `1.stl`
   (`~/scan2cad/colouring_test/notes.md` §9): гауссово смещение вдоль нормали,
   sigma = 2 мкм, радиус пространственной корреляции 0.3 мм. Белый шум там
   отклонён — у него кривая шероховатости по радиусу плоская, у реального
   скана растёт вчетверо на том же интервале.

2. `waviness` — та же конструкция на длинных волнах. НЕ откалибрована.
   Основание, что она существует: на `1.stl` невязка струи занижена, потому
   что волнистость шире радиуса подгонки и полином её поглощает
   (`segmentation_notes.md`, из-за этого FIT_C подняли с 8 до 40).

3. `round_edges` — сканер не разрешает острое ребро, его измерение это
   локальное среднее по функции рассеяния. НЕ откалибрована. Фильтр Тобина: сила
   задаётся числом проходов от шага подложки, поэтому зависит от разрешения.
   `gaussian_blur` — то же размытие с σ в миллиметрах, от сетки не зависит
   (секция noise, `blur`).

Compонента 1 применима как есть только пока детали того же размера, что и
сканы, по которым её мерили (15-48 мм). Абсолютные микроны на деталь другого
масштаба не переносятся.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, factorized
from scipy.spatial import cKDTree

from src.config import CFG

# Значения и их обоснование — в configs/default.yaml, секция noise.
_C = CFG.noise



def _diffused_field(points: np.ndarray, corr: float, spacing: float,
                    rng: np.random.Generator,
                    k: int = _C.diffusion_k) -> np.ndarray:
    """Скалярное поле с нулевым средним, единичной дисперсией и заданным
    радиусом корреляции.

    Белое поле многократно усредняется по k ближайшим соседям. Один шаг
    усреднения смещает на радиус окрестности h = 0.525*spacing*sqrt(k), после
    n шагов ширина ядра растёт как h*sqrt(n) — отсюда число шагов. Большие
    радиусы корреляции берём большим k, а не большим числом шагов, иначе на
    68 тыс. вершин это сотни проходов.
    """
    h = 0.525 * spacing * np.sqrt(k)
    steps = max(1, int(round((corr / h) ** 2)))
    _, idx = cKDTree(points).query(points, k=k, workers=1)
    w = rng.standard_normal(len(points))
    for _ in range(steps):
        w = w[idx].mean(axis=1)
    # Диффузия гасит дисперсию, возвращаем к единичной.
    return w / max(w.std(), 1e-12)


# Длина корреляции гаусса у поля `_diffused_field`: ℓ ≈ 1.13·corr при любом шаге сетки
# [измерено: ровная сетка, corr 0.3 мм — ℓ 0.342 / 0.336 мм на шаге 0.09 / 0.045; corr 3 мм —
# ℓ 3.36 мм]. У `_fourier_field` та же оценка выходит на 3-6% короче заданной, поэтому
# коэффициент 1.18: у откалиброванного мелкого шума (corr 0.3 мм) длина та же, ~0.34 мм.
_FIELD_ELL_PER_CORR = 1.18
# Число случайных волн и порция точек в `_fourier_field`: память — порция × волны.
_FOURIER_WAVES = 256
_FOURIER_CHUNK = 20000


def _fourier_field(points: np.ndarray, corr: float, rng: np.random.Generator,
                   waves: int = _FOURIER_WAVES) -> np.ndarray:
    """То же поле, что `_diffused_field`, суммой случайных плоских волн.

    Волновые векторы из гаусса со стандартным отклонением 1/ℓ дают ковариацию
    exp(-r²/2ℓ²) — как у поля усреднением по соседям, но за один проход и без
    зависимости от сетки. Усреднение требует (corr / шаг)² проходов: на подложке в
    500 тысяч треугольников и волнах до 6 мм это тысячи проходов и треть времени
    сборки детали.
    """
    ell = _FIELD_ELL_PER_CORR * corr
    omega = rng.standard_normal((waves, 3)) / ell
    phase = rng.uniform(0.0, 2 * np.pi, waves)
    w = np.empty(len(points))
    for s in range(0, len(points), _FOURIER_CHUNK):
        w[s:s + _FOURIER_CHUNK] = np.cos(points[s:s + _FOURIER_CHUNK] @ omega.T + phase).sum(1)
    w -= w.mean()
    return w / max(w.std(), 1e-12)


def _field(points: np.ndarray, corr: float, spacing: float, rng: np.random.Generator, fourier: bool) -> np.ndarray:
    return _fourier_field(points, corr, rng) if fourier else _diffused_field(points, corr, spacing, rng)


def high_frequency(points: np.ndarray, normals: np.ndarray, spacing: float,
                   rng: np.random.Generator, sigma: float = _C.hf_sigma_mm,
                   corr: float = _C.hf_corr_mm, fourier: bool = False) -> np.ndarray:
    """Компонента 1: коррелированный шум вдоль нормали. Значения по умолчанию
    измерены, не назначены. `fourier` — поле волнами (`_fourier_field`)."""
    return points + (sigma * _field(points, corr, spacing, rng, fourier))[:, None] * normals


def waviness(points: np.ndarray, normals: np.ndarray, spacing: float,
             rng: np.random.Generator, sigma: float = _C.wave_sigma_mm,
             corr: float = _C.wave_corr_mm, fourier: bool = False) -> np.ndarray:
    """Компонента 2: длинноволновая деформация — остаточная ошибка сшивки
    кадров. Та же конструкция, другой масштаб.

    sigma и corr — ПРЕДПОЛОЖЕНИЯ. Порядок выбран так, чтобы длина волны была
    заведомо больше радиуса подгонки струи (0.42 мм при k=64), то есть чтобы
    полином её локально поглощал — именно это и наблюдалось на реальном скане.
    Калибровать по плоской грани реального скана: подогнать плоскость и
    посмотреть спектр отклонения по масштабам.
    """
    return points + (sigma * _field(points, corr, spacing, rng, fourier))[:, None] * normals


def mesh_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Рёбра меша в обе стороны — вход для `round_edges`."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])


def round_edges(points: np.ndarray, src: np.ndarray, dst: np.ndarray,
                radius: float, spacing: float,
                lam: float = _C.taubin_lambda,
                kpb: float = _C.taubin_kpb) -> np.ndarray:
    """Компонента 3: скругление острых рёбер низкочастотной фильтрацией.

    Сканер меряет не точку, а свёртку поверхности со своей функцией рассеяния,
    поэтому острое ребро выходит скруглённым. Одна фильтрация даёт нужное
    поведение сама: гладкие участки почти не двигаются, ребро скругляется.

    Тобин (lambda | mu), а не лапласиан: у лапласиана систематическая усадка,
    смещающая подогнанные радиусы (`segmentation_notes.md` п.9, R=8.992 против
    истинного 9.000).

    `radius` — целевой радиус ребра в единицах координат. [измерено: ребро
    90 градусов, шаг сетки 0.1 мм, струя k=64] радиус растёт как
    R ~ 0.5 * spacing * sqrt(итераций), отсюда пересчёт:

        итераций    1     2     4     8    16    32
        R ребра  0.060 0.060 0.072 0.108 0.198 0.288 мм
        сдвиг     0.00  0.12  2.70 14.76 45.48 103.5 мкм

    ВАЖНО: ниже примерно 0.6 * spacing радиус не опускается — это предел
    разрешения сетки, и запрос меньшего молча даёт этот предел. Шум
    накладывается на подложку (шаг ~0.09 мм), где реалистичные для сканера
    0.1 мм ещё выразимы; на рабочем меше (шаг ~0.21 мм при 30 тысячах) они уже
    нет, поэтому порядок этапов и важен.

    Настоящий радиус ребра реального сканера НЕ ИЗМЕРЕН. Измеряется по скану:
    найти ребро, острое в CAD, и посмотреть пик кривизны на нём.
    """
    mu = -lam / (kpb * lam - 1.0)          # условие Тобина на подавление усадки
    steps = max(1, int(round((radius / (0.5 * spacing)) ** 2)))
    counts = np.bincount(dst, minlength=len(points)).clip(min=1)[:, None]
    v = points.copy()
    for _ in range(steps):
        for step in (lam, mu):
            avg = np.zeros_like(v)
            np.add.at(avg, dst, v[src])
            v = v + step * (avg / counts - v)
    return v


# Пределы котангенса в запасном пути размытия: неотрицательные веса делают систему
# устойчивой при любом качестве сетки.
_ROBUST_COT = (0.0, 1e3)


def _laplacian(v: np.ndarray, f: np.ndarray, robust: bool) -> tuple[sp.csr_matrix, np.ndarray]:
    """Котангенсный лапласиан и площади вершин; `robust` — как в `gaussian_blur`."""
    n = len(v)
    tri_area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    # Вырожденный треугольник (после ремеша их до десятых процента) дал бы
    # котангенс в 1e30, и система не сходилась бы: веса он не получает. На мешах
    # без таких треугольников результат не меняется.
    live = tri_area > 1e-10 * np.median(tri_area)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = f[live, k], f[live, (k + 1) % 3], f[live, (k + 2) % 3]
        a, b = v[i] - v[o], v[j] - v[o]
        cot = (a * b).sum(1) / np.linalg.norm(np.cross(a, b), axis=1)
        if robust:
            cot = np.clip(cot, *_ROBUST_COT)
        rows += [i, j]; cols += [j, i]; vals += [-0.5 * cot, -0.5 * cot]
    area = np.zeros(n)
    np.add.at(area, f.ravel(), np.repeat(tri_area / 3, 3))
    # Вершина только из вырожденных треугольников остаётся на месте, а не делает систему вырожденной.
    area = np.maximum(area, 1e-6 * np.median(area))
    off = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    lap = off - sp.diags(np.asarray(off.sum(1)).ravel())     # положительно полуопределённый
    return lap, area


def gaussian_blur(points: np.ndarray, faces: np.ndarray, sigma: float, robust: bool = False) -> np.ndarray:
    """Компонента 3 в миллиметрах: размытие поверхности гауссом со стандартным
    отклонением `sigma` по самой поверхности.

    Неявный шаг уравнения теплопроводности `(M + t·L) X = M X₀` с котангенсным
    лапласианом `L` и площадями вершин `M`: ядро теплопроводности за время `t` — гаусс
    с дисперсией `2t` по каждой оси, отсюда `t = σ²/2`. В отличие от Тобина сила не
    зависит от шага сетки. Поверхность при этом усаживается там, где искривлена, на
    `~t·H` — как и при настоящей свёртке со сглаживающим пятном.

    `robust` — котангенсы обрезаются до `_ROBUST_COT`. На подложке бывают иглы с
    углами больше 170° и меньше 1° (у 18% деталей корпуса), котангенсы там до 1e5, и
    точная система не сходится; обрезка меняет оператор только на таких
    треугольниках. Используется как запасной путь (`dataset/cache.py`, `apply_noise`).
    """
    if sigma <= 0:
        return points
    v = np.asarray(points, float)
    f = np.asarray(faces, np.int64)
    lap, area = _laplacian(v, f, robust)
    t = 0.5 * sigma ** 2
    system = (sp.diags(area) + t * lap).tocsr()
    jacobi = sp.diags(1.0 / system.diagonal())
    out = np.empty_like(v)
    for c in range(3):
        out[:, c], info = cg(system, area * v[:, c], x0=v[:, c], rtol=1e-10, maxiter=2000, M=jacobi)
        if info:
            raise RuntimeError(f"размытие не сошлось за {info} итераций")
    return out


def rolloff(points: np.ndarray, coarse_points: np.ndarray, coarse_faces: np.ndarray,
            alpha: float, sigma: float, steps: int = 3) -> np.ndarray:
    """Компонента 5: широкий скат у рёбер. К поверхности подмешивается доля `alpha`
    размытой гауссом `sigma` порядка миллиметра: у рёбер она проседает, на гладком —
    не меняется. Так выглядят плоские грани сканов — «подушка» с просадкой у края на
    десятки микрон (README, «Грязный шум сканера»).

    Скат плавный на масштабе `sigma`, поэтому считается на прорежённом меше и переносится
    на вершины средним по четырём ближайшим: на подложке с иглами неявный шаг такого
    размера не сходится. Решение прямое — несколько неявных шагов с одной факторизацией.
    """
    if alpha <= 0:
        return points
    lap, area = _laplacian(coarse_points, coarse_faces, robust=True)
    solve = factorized((sp.diags(area) + (0.5 * sigma ** 2 / steps) * lap).tocsc())
    far = np.asarray(coarse_points, float)
    for _ in range(steps):
        far = np.column_stack([solve(area * far[:, c]) for c in range(3)])
    d, i = cKDTree(coarse_points).query(points, k=4)
    w = 1.0 / np.maximum(d, 1e-9)
    shift = ((far - coarse_points)[i] * w[..., None]).sum(1) / w.sum(1)[:, None]
    return points + alpha * shift


def bumps(points: np.ndarray, normals: np.ndarray, area: float, rng: np.random.Generator, density: float,
          radius: tuple[float, float], height: tuple[float, float]) -> np.ndarray:
    """Компонента 6: редкие кочки и ямки — гауссовы бугорки вдоль нормали, число —
    `density` на мм² площади, радиус и высота — равномерно из пределов, знак случайный."""
    k = rng.poisson(density * area)
    if k == 0:
        return points
    centers = points[rng.choice(len(points), k, replace=False)]
    r = rng.uniform(*radius, k)
    h = rng.uniform(*height, k) * rng.choice([-1.0, 1.0], k)
    tree = cKDTree(points)
    out = points.copy()
    for c, ri, hi in zip(centers, r, h):
        idx = tree.query_ball_point(c, 3 * ri)
        out[idx] += (hi * np.exp(-((points[idx] - c) ** 2).sum(1) / (2 * ri * ri)))[:, None] * normals[idx]
    return out


def ripple(points: np.ndarray, normals: np.ndarray, rng: np.random.Generator, sigma: float, period: float,
           spread_deg: float, waves: int = 64) -> np.ndarray:
    """Компонента 7: направленная рябь масштаба 0.5-1.5 мм вдоль нормали. Сумма плоских
    волн с периодом около `period` и волновыми векторами в конусе `spread_deg` вокруг
    случайного направления: узкий конус — полосы, как от ошибки фазы сканера с проекцией
    полос (RGO, 130326), широкий — извилистый рельеф (4.stl, Тест4)."""
    if sigma <= 0:
        return points
    d = rng.standard_normal(3)
    d /= np.linalg.norm(d)
    jitter = rng.standard_normal((waves, 2)) @ np.linalg.svd(d[None])[2][1:]   # поперёк d
    dirs = d + np.tan(np.radians(spread_deg)) * jitter
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    omega = dirs * (2 * np.pi / (period * rng.uniform(0.8, 1.25, waves)))[:, None]
    phase = rng.uniform(0.0, 2 * np.pi, waves)
    w = np.empty(len(points))
    for s in range(0, len(points), _FOURIER_CHUNK):
        w[s:s + _FOURIER_CHUNK] = np.cos(points[s:s + _FOURIER_CHUNK] @ omega.T + phase).sum(1)
    w -= w.mean()
    return points + (sigma * w / max(w.std(), 1e-12))[:, None] * normals
