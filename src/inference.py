"""Инференс на произвольном меше: меш -> кластеры и типы поверхностей.

Точка входа для внешнего кода: то же, что делает `notebooks/inference.ipynb`,
но без ноутбука и воспроизводимо. Конвейер — подложка, рабочий меш, признаки,
кольца, сеть, нарезка по графу (README, «Голова границ»).

Семя обязательно: соседи внутри кольца выбираются случайно, и без заданного
генератора два прогона на одном меше расходятся [измерено, 6 деталей val:
совпадение нарезок IoU 0.873, разница 1.7 кластера; с семенем — 1.000].
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.sparse.csgraph import connected_components, dijkstra

from dataset import features as FE
from dataset import remesh as R
from dataset import scan as SC
from dataset import tiles as T
from dataset.cache import SCAN_STREAM, apply_noise, nominal_noise_params
from src import graph as G
from src.boundary import partition as P
from src.boundary.config import BCFG, PartitionCfg
from src.boundary.model import BoundarySegNet, boundary_strength, head_config
from src.config import CFG, ROOT
from src.data import compress_curvature, tile_features
from src.model import PRIMITIVE_TYPES
from src.runs import thresholds

DEFAULT_CHECKPOINT = ROOT / "runs" / "joint50" / "best.pt"
# Меньше этого на входе не из чего строить подложку нужной подробности.
MIN_INPUT_FACES = 12
# Ниже этого пола кривизны на гладких участках вход считается бесшумным.
# Порог откалиброван на рабочей точке 30 тысяч треугольников: при более мелкой
# цели подложка грубее радиуса струи, и пол кривизны сжимается.
# [измерено, нижний квартиль |S·L| на подложке: чистая тесселяция CAD — 0.000
# на всех трёх проверенных деталях; она же с шумом обучения — 1.7-2.7;
# реальные сканы — 2.1-4.9]
CLEAN_FLOOR = 0.5
# Доля треугольников, заведомо лежащих на гладком: нижний квартиль.
FLOOR_QUANTILE = 0.25


@dataclass(frozen=True)
class Segmentation:
    """Результат на рабочем меше: он мельче входного, и метки заданы на нём."""

    mesh: trimesh.Trimesh
    noised: bool                            # накладывался ли шум обучения на подложку (кусками — прошёл ли сканер)
    labels: np.ndarray                      # кластер на треугольник, 0..K-1
    types: np.ndarray                       # тип поверхности на треугольник, индекс PRIMITIVE_TYPES
    cluster_types: dict[int, str]           # тип кластера голосованием его треугольников
    embedding: np.ndarray                   # (N, D), нормированный — для сравнения кластеров
    strength: np.ndarray                    # сила границы на ребро рабочего меша
    seconds: dict[str, float] = field(default_factory=dict)
    cores: np.ndarray | None = None         # вывод кусками: ядро на треугольник


def load_model(checkpoint: str | Path = DEFAULT_CHECKPOINT,
               device: torch.device | str | None = None) -> BoundarySegNet:
    """Веса варианта D. Держать модель между запросами: загрузка дороже прохода."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(Path(checkpoint), map_location=device)
    if state.get("log_curvature", False) != CFG.model.log_curvature:
        raise ValueError("чекпоинт обучен с другим преобразованием входа, чем задаёт конфиг")
    model = BoundarySegNet(state["channels"], CFG.model.emb_size, head_config(BCFG.model, state["model"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def noise_floor(features: np.ndarray, names: list[str]) -> float:
    """Пол кривизны на гладких участках: нижний квартиль `|S·L|`.

    У меша прямо из CAD он равен нулю — плоскость идеально плоская. У скана и у
    обучающего варианта он заметно выше нуля, и сеть обучена именно на таком.
    `features` — сырые, до логарифма: `CLEAN_FLOOR` откалиброван по ним.
    """
    return float(np.quantile(np.abs(features[names.index(f"S@{CFG.features.scales_rel[0]}")]), FLOOR_QUANTILE))


def _work_and_features(substrate: trimesh.Trimesh, target_faces: int):
    """Рабочий меш, признаки для сети `(C, N)`, имена каналов и пол кривизны."""
    work = R.decimate(substrate, target_faces=target_faces)
    (raw,), names, _ = FE.compute_many(substrate, [work])
    features = raw.T.astype(np.float32)
    floor = noise_floor(features, names)
    if CFG.model.log_curvature:
        features = compress_curvature(features, names)
    return work, features, names, floor


def prepare(mesh: trimesh.Trimesh, target_faces: int = R.TARGET_FACES,
            clean_input: bool | None = None, seed: int = 0):
    """Входной меш -> рабочий меш, признаки `(C, N)` и был ли наложен шум.

    Подложка равномерная — на ней считается струя; рабочий меш упрощён по
    квадратичной ошибке, поэтому его плотность идёт за кривизной.

    `clean_input` — вход не со сканера, а прямо из CAD: тогда на подложку
    накладывается та же модель шума, что при обучении. Без этого сеть работает
    плохо: она не видела идеально острых рёбер и нулевой кривизны на
    плоскостях [измерено, 10 деталей val, разметка перенесена на тот же
    рабочий меш: чистый вход — IoU 0.166 и 19.9 протечки на деталь, тот же
    вход с шумом — 0.706 и 3.7]. Модель шума задана в миллиметрах
    (`configs/default.yaml`, секция noise), то есть рассчитана на детали
    корпуса, 20-35 мм.

    `None` — решать по самому мешу, сравнивая пол кривизны с `CLEAN_FLOOR`;
    тогда для бесшумного входа признаки считаются дважды, и это стоит лишних
    секунды-двух. Подложка при этом не перестраивается: она дороже всего.
    """
    substrate = R.substrate(mesh, target_faces=target_faces)
    if clean_input is not True:
        work, features, names, floor = _work_and_features(substrate, target_faces)
        if clean_input is False or floor >= CLEAN_FLOOR:
            return work, features, names, False
    substrate = apply_noise(substrate, seed, nominal_noise_params())
    work, features, names, _ = _work_and_features(substrate, target_faces)
    return work, features, names, True


def segment(mesh: trimesh.Trimesh | str | Path, model: BoundarySegNet | None = None,
            seed: int = 0, target_faces: int = R.TARGET_FACES, clean_input: bool | None = None,
            device: torch.device | str | None = None) -> Segmentation:
    """Меш (или путь к нему) -> кластеры, каждый под один примитив.

    `seed` задаёт выборку соседей в кольцах; при одном и том же семени ответ
    повторяется в точности. `clean_input` — для мешей прямо из CAD; по
    умолчанию решается по самому мешу, см. `prepare`.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load(str(mesh), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < MIN_INPUT_FACES:
        raise ValueError(f"нужен треугольный меш минимум из {MIN_INPUT_FACES} граней")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("в координатах вершин есть NaN или бесконечность")

    model = model or load_model(device=device)
    device = next(model.parameters()).device
    seconds = {}

    t = time.perf_counter()
    work, features, _, noised = prepare(mesh, target_faces, clean_input, seed)
    seconds["подготовка"] = time.perf_counter() - t
    expected = model.encoder.conv1.block[0].in_channels // 2
    if features.shape[0] != expected:
        raise ValueError(f"признаков {features.shape[0]}, а сеть обучена на {expected}: "
                         "конфиг признаков разошёлся с чекпоинтом")

    adjacency = np.array(work.face_adjacency, dtype=np.int64)   # копия: тензор из trimesh не перезаписываем
    x = torch.from_numpy(features).to(device).unsqueeze(0)
    edges = torch.from_numpy(adjacency).to(device)

    t = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    neighbors = G.face_neighbors(adjacency, len(work.faces), device)
    idx = G.ring_neighbors(neighbors, CFG.model.rings, CFG.model.neighbors,
                           CFG.model.frontier_cap, CFG.model.frontier_per_ring, generator)
    seconds["кольца"] = time.perf_counter() - t

    t = time.perf_counter()
    with torch.no_grad():
        embedding, logits, edge = model(x, idx, edges)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds["сеть"] = time.perf_counter() - t

    z = F.normalize(embedding[0].t(), dim=1).double().cpu().numpy()
    strength = boundary_strength(edge)
    types = logits[0].argmax(0).cpu().numpy()

    t = time.perf_counter()
    # Пороги под разрешение конфига, как в обучении и ноутбуке (src/runs.py): на 60 тысячах
    # регион поглощается до 60 треугольников, а не до 30.
    labels = P.partition(len(work.faces), adjacency, strength, thresholds()[1], z)
    seconds["нарезка"] = time.perf_counter() - t

    cluster_types = {int(l): PRIMITIVE_TYPES[int(np.bincount(types[labels == l]).argmax())]
                     for l in np.unique(labels)}
    return Segmentation(work, noised, labels, types, cluster_types, z, strength, seconds)


# Вывод кусками — для прогонов на кэше кусков (configs/tile.yaml; README, «Края кусков»).
# Каналы входа без струи: центр и нормаль треугольника, как в `_build_tile`.
TILE_INPUT = ["cx", "cy", "cz", "nx", "ny", "nz"]


def fixed_edge_work(mesh: trimesh.Trimesh, scan_seed: int | None = None) -> trimesh.Trimesh:
    """Рабочий меш с ребром `tile.edge_mm`, как в кэше кусков (`dataset/cache.py`, `_build_tile`).

    `scan_seed` задан — вход из CAD и сначала идёт через виртуальный сканер с этим семенем; у детали
    кэша это семя из её meta.json, и рабочий меш тот же, что при сборке [проверено: abc_00257246,
    818 428 треугольников, вершины t0 совпали до 3e-6 мм]. Иначе вход — уже скан.
    """
    t = CFG.tile
    if t.jet:
        raise NotImplementedError("вывод кусками со струёй не сделан: tile-прогоны пока без неё")
    if scan_seed is not None:
        rng = np.random.default_rng([scan_seed, SCAN_STREAM])
        mesh, _ = SC.scan(mesh, SC.sample_params(rng), rng)
    # Вход грубее ребра (тесселяция CAD без сканера) не отвергается: упрощение его не меняет,
    # и сеть видит меш как есть (решено пользователем 2026-11-04).
    return R.decimate(mesh, int(mesh.area / (np.sqrt(3) / 4 * t.edge_mm ** 2)))


def tile_cores(work: trimesh.Trimesh, seed: int = 0) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Ядро на треугольник и куски. Ядра — ячейки Вороного по поверхности вокруг дальних точек, одна
    на `tile.faces` треугольников связной части; кусок — ядро и кайма `halo_rings` колец, как при
    обучении. Кусок больше `max_faces` — у части на ядро больше. Части меньше ядра (островки скана)
    собираются в общие ядра подряд. Кусок — треугольники по возрастанию и маска ядра среди них."""
    t = CFG.tile
    n = len(work.faces)
    adjacency = np.asarray(work.face_adjacency, np.int64)
    pts = work.triangles_center
    hop = T._graph(adjacency, n)
    geo = T._graph(adjacency, n, np.linalg.norm(pts[adjacency[:, 0]] - pts[adjacency[:, 1]], axis=1) + 1e-9)
    comp = connected_components(hop, directed=False)[1]
    parts = np.split(np.argsort(comp, kind="stable"), np.cumsum(np.bincount(comp))[:-1])
    rng = np.random.default_rng(seed)

    def with_halo(core):
        sel = np.flatnonzero(np.isfinite(dijkstra(hop, indices=core, min_only=True, limit=t.halo_rings + 0.5)))
        return sel, np.isin(sel, core)

    groups, pack = [], []
    for part in sorted(parts, key=len, reverse=True):
        if len(part) <= t.faces:
            if pack and sum(map(len, pack)) + len(part) > t.faces:
                groups.append([np.concatenate(pack)])
                pack = []
            pack.append(part)
            continue
        k = int(np.ceil(len(part) / t.faces))
        while True:
            seeds = part[T.centers(pts[part], k, rng)]
            nearest = dijkstra(geo, indices=seeds, min_only=True, return_predecessors=True)[2][part]
            cells = [part[nearest == s] for s in seeds]
            if all(len(with_halo(c)[0]) <= t.max_faces for c in cells):
                break
            k += 1
        groups.append(cells)
    if pack:
        groups.append([np.concatenate(pack)])

    cores = np.full(n, -1)
    tiles = []
    for core in (c for g in groups for c in g):
        cores[core] = len(tiles)
        tiles.append(with_halo(core))
    return cores, tiles


def tile_outputs(work: trimesh.Trimesh, model: BoundarySegNet, seed: int = 0, seconds: dict | None = None):
    """Сеть по кускам: эмбеддинг `(N, D)` (нормированный, float32), логиты типа `(N, T)`, сила ребра и
    ядро на треугольник. Эмбеддинг и тип — из куска, в ядре которого треугольник; сила ребра — из
    кусков, в ядре которых его треугольники (у ребра между ядрами — среднее двух). Рёбра куска — рёбра
    целого меша с обоими концами в нём: у ребра, где сходятся больше двух треугольников, пары в
    подмеше могут сложиться иначе, и ребро терялось."""
    seconds = {} if seconds is None else seconds
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    cores, tiles = tile_cores(work, seed)
    seconds["куски"] = time.perf_counter() - t0
    n = len(work.faces)
    adjacency = np.asarray(work.face_adjacency, np.int64)
    points, normals, _ = FE.face_frames(work)
    frames = np.concatenate([points, normals], axis=1).T.astype(np.float32)
    z = np.zeros((n, CFG.model.emb_size), np.float32)
    logits = np.zeros((n, len(PRIMITIVE_TYPES)), np.float32)
    s_sum, s_cnt = np.zeros(len(adjacency)), np.zeros(len(adjacency))
    local = np.full(n, -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    expected = model.encoder.conv1.block[0].in_channels // 2
    seconds["кольца"] = seconds["сеть"] = 0.0
    for sel, core in tiles:
        local[sel] = np.arange(len(sel))
        inside = np.flatnonzero((local[adjacency[:, 0]] >= 0) & (local[adjacency[:, 1]] >= 0))
        edges = local[adjacency[inside]]
        local[sel] = -1
        x, _ = tile_features(frames[:, sel], TILE_INPUT)
        if x.shape[0] != expected:
            raise ValueError(f"признаков {x.shape[0]}, а сеть обучена на {expected}")
        t0 = time.perf_counter()
        idx = G.ring_neighbors(G.face_neighbors(edges, len(sel), device), CFG.model.rings, CFG.model.neighbors,
                               CFG.model.frontier_cap, CFG.model.frontier_per_ring, generator)
        seconds["кольца"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            emb, lg, edge = model(torch.from_numpy(x).to(device).unsqueeze(0), idx,
                                  torch.from_numpy(edges).to(device))
        mask = torch.from_numpy(core).to(device)
        z[sel[core]] = F.normalize(emb[0].t()[mask], dim=1).cpu().numpy()
        logits[sel[core]] = lg[0].t()[mask].cpu().numpy()
        seconds["сеть"] += time.perf_counter() - t0
        own = core[edges[:, 0]] | core[edges[:, 1]]
        s_sum[inside[own]] += boundary_strength(edge)[own]
        s_cnt[inside[own]] += 1
    return z, logits, s_sum / s_cnt, cores


def segment_tiles(mesh: trimesh.Trimesh | str | Path, model: BoundarySegNet, scan_seed: int | None = None,
                  seed: int = 0, cfg: PartitionCfg | None = None) -> Segmentation:
    """Целая деталь кусками: рабочий меш с фиксированным ребром, сеть по кускам, одна нарезка на всю
    деталь (README, «Края кусков»). `scan_seed` — для входа из CAD, см. `fixed_edge_work`."""
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load(str(mesh), force="mesh")
    seconds = {}
    t0 = time.perf_counter()
    work = fixed_edge_work(mesh, scan_seed)
    seconds["рабочий меш"] = time.perf_counter() - t0
    z, logits, strength, cores = tile_outputs(work, model, seed, seconds)
    t0 = time.perf_counter()
    labels = P.partition(len(work.faces), np.asarray(work.face_adjacency), strength, cfg or thresholds()[1], z)
    seconds["нарезка"] = time.perf_counter() - t0
    types = logits.argmax(1)
    t = len(PRIMITIVE_TYPES)
    votes = np.bincount(labels * t + types, minlength=(labels.max() + 1) * t).reshape(-1, t)
    cluster_types = {l: PRIMITIVE_TYPES[v] for l, v in enumerate(votes.argmax(1))}
    return Segmentation(work, scan_seed is not None, labels, types, cluster_types, z, strength, seconds, cores)
