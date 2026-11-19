"""Куски при фиксированном ребре (README, «Фиксированное ребро и куски»).

Деталь сканируется целиком, рабочий меш режется на куски: центры — дальние точки по
поверхности, ядро куска — круг по поверхности в `faces` треугольников вокруг центра, кусок — ядро
и кайма `halo_rings` колец смежности вокруг него (на выводе так же: ядро Вороного плюс кайма).
Ядро одного размера у всех кусков, кусок — сколько выйдет (решено пользователем 2026-10-17).
С детали берётся не больше `tiles` кусков, жадно по новым граням B-Rep: куски одной детали иначе
однообразны (решено пользователем 2026-10-16).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

# Если кусок больше потолка, ядро уменьшается во столько раз, пока не влезет. Выбрано.
CORE_SHRINK = 0.85


def _graph(adjacency: np.ndarray, n: int, weights: np.ndarray | None = None):
    w = np.ones(len(adjacency)) if weights is None else weights
    g = coo_matrix((w, (adjacency[:, 0], adjacency[:, 1])), shape=(n, n)).tocsr()
    return g + g.T


def centers(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """`k` дальних точек (по прямой — по поверхности на миллионах треугольников долго; для
    расстановки центров хватает), первая — случайная."""
    out = [int(rng.integers(len(points)))]
    d = np.linalg.norm(points - points[out[0]], axis=1)
    for _ in range(k - 1):
        out.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(points - points[out[-1]], axis=1))
    return np.array(out)


def pick(labels: np.ndarray, points: np.ndarray, seeds: np.ndarray, k: int) -> list[int]:
    """До `k` центров (`seeds` — координаты) жадно: первый — чья ячейка Вороного содержит больше граней
    B-Rep, каждый следующий — кто добавляет больше ещё не взятых граней (при равенстве — больше
    граней всего). `labels`, `points` — метки и центры треугольников, можно подвыборкой. Возвращает
    номера в `seeds`."""
    cell = cKDTree(seeds).query(points)[1]
    sets = [set(np.unique(labels[cell == i]).tolist()) for i in range(len(seeds))]
    chosen, covered = [], set()
    while len(chosen) < min(k, len(seeds)):
        i = max((i for i in range(len(seeds)) if i not in chosen),
                key=lambda i: (len(sets[i] - covered), len(sets[i])))
        chosen.append(i)
        covered |= sets[i]
    return chosen


def cut(adjacency: np.ndarray, points: np.ndarray, start: int, size: int, radius: float,
        limit: int = 65535) -> np.ndarray:
    """Кусок: `size` треугольников, ближайших по поверхности к `start` (Дейкстра по центрам) — круг с
    ровным краем; обход в ширину считал шаги, а не мм, и край выходил рваным. Островки, которые
    круг окружил, но не набрал (за узким перешейком), добавляются, пока кусок не больше `limit`:
    иначе внутри куска дыра, которой у детали нет. `radius` — потолок Дейкстры, мм: на детали
    в миллионы треугольников весь меш обходить незачем."""
    n = len(points)
    w = np.linalg.norm(points[adjacency[:, 0]] - points[adjacency[:, 1]], axis=1) + 1e-9
    d = dijkstra(_graph(adjacency, n, w), indices=start, limit=radius)   # дальше радиуса не считать
    inside = np.zeros(n, bool)
    inside[np.argsort(d)[:size]] = True
    inside &= np.isfinite(d)
    rest = ~inside[adjacency[:, 0]] & ~inside[adjacency[:, 1]]
    k, lab = connected_components(_graph(adjacency[rest], n), directed=False)
    sizes = np.bincount(lab[~inside], minlength=k)
    comps = np.flatnonzero(sizes)
    for comp in comps[np.argsort(sizes[comps])][:-1]:   # все остатки, кроме крупнейшего, с мелких
        if inside.sum() + sizes[comp] > limit:
            break
        inside |= (lab == comp) & ~inside
    return np.flatnonzero(inside)


def tile(adjacency: np.ndarray, points: np.ndarray, hop, start: int, core: int, halo: int, cap: int,
         radius: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Кусок вокруг `start`: ядро — `cut` в `core` треугольников, кусок — всё в `halo` кольцах смежности
    (`hop` — граф смежности) от ядра. Кусок больше `cap` — ядро меньше в CORE_SHRINK раз, пока не
    влезет: кайма — всё поле зрения сети, её не урезать. Возвращает треугольники куска по
    возрастанию, маску ядра среди них и сколько раз ядро уменьшалось."""
    size, shrunk = core, 0
    while True:
        c = cut(adjacency, points, start, size, radius, limit=int(size * 1.1))
        d = dijkstra(hop, indices=c, min_only=True, limit=halo + 0.5)
        sel = np.flatnonzero(np.isfinite(d))
        if len(sel) <= cap:
            return sel, np.isin(sel, c), shrunk
        size, shrunk = int(size * CORE_SHRINK), shrunk + 1
