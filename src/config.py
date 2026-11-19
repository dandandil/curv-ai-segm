"""Типизированная конфигурация проекта.

Все настраиваемые величины лежат в `configs/default.yaml` с пометкой, измерены
они или выбраны. Модули берут оттуда свои умолчания, поэтому поменять рабочую
точку — это правка одного файла, а не трёх.
"""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
# Конфиг опыта — переменной окружения, чтобы не править default.yaml:
#     CURV_CONFIG=configs/wide_rings.yaml python scripts/train.py ...
CONFIG_PATH = Path(os.environ.get("CURV_CONFIG") or DEFAULT_CONFIG).resolve()


@dataclass(frozen=True)
class RemeshCfg:
    target_faces: int
    substrate_edge_rel: float
    substrate_min_ratio: float
    feature_deg: float
    iterations: int
    passes: int
    face_tol: float
    substrate_passes: int
    substrate_source: str


@dataclass(frozen=True)
class FeaturesCfg:
    scales_rel: tuple[float, ...]
    min_k: int
    ball_grow: float
    ball_grow_steps: int
    ball_chunk: int
    chunk_budget: int
    target_k: int


@dataclass(frozen=True)
class NoiseCfg:
    hf_sigma_mm: float
    hf_corr_mm: float
    wave_sigma_mm: float
    wave_corr_mm: float
    edge_radius_mm: float
    taubin_lambda: float
    taubin_kpb: float
    diffusion_k: int
    blur: str
    blur_sigma_mm: tuple[float, float]
    hf_sigma_range_mm: tuple[float, float]
    wave_sigma_range_mm: tuple[float, float]
    wave_corr_range_mm: tuple[float, float]
    fine_sigma_range_mm: tuple[float, float]
    fine_corr_range_mm: tuple[float, float]
    rolloff_alpha_range: tuple[float, float]
    rolloff_sigma_range_mm: tuple[float, float]
    bump_density_range: tuple[float, float]
    bump_radius_range_mm: tuple[float, float]
    bump_height_range_mm: tuple[float, float]
    ripple_sigma_range_mm: tuple[float, float]
    ripple_period_range_mm: tuple[float, float]
    ripple_spread_range_deg: tuple[float, float]
    # scan — подложка из виртуального сканера (dataset/scan.py) вместо шума на подложке;
    # scan_version поднимать при каждой правке сканера: от неё зависит метка кэша.
    model: str = "synthetic"
    scan_version: int = 0
    # Нижний предел вокселя TSDF сканера, мм; 0 — шаг подложки, как раньше (README, «Фиксированное
    # ребро и куски»).
    scan_min_voxel_mm: float = 0.0
    # Шаг пикселя на детали не больше, мм: кадр крупной детали растёт сверх MAX_PIXELS; 0 — не
    # ограничивать, как раньше (README, «Сборка кусков: как устроено»).
    scan_max_pitch_mm: float = 0.0


@dataclass(frozen=True)
class TessellationCfg:
    linear_rel: float
    angular: float
    min_size_rel: float
    merge_angle_deg: float
    merge_length_rel: float
    merge_radius_rel: float
    tangency_samples: int
    tangency_deg: float
    # Ниже — с умолчаниями, выключающими новое: старые конфиги собирают разметку как прежде.
    # Слияние сплайновых швов: угол нормалей меньше merge_angle_deg и скачок главных
    # кривизн меньше стольких мм⁻¹ (0 — не сливать).
    seam_curv_jump: float = 0.0
    # Точная полуширина слитых граней по CAD: шаг точек на рёбрах, мм (0 — не считать).
    halfwidth_spacing_mm: float = 0.0
    # Правило узких граней: полуширина до стольких мм (0 — не размечать); ступенька —
    # нормали соседей напротив друг друга расходятся меньше step_deg, разворот — больше
    # 180 - turn_deg (README, «Узкие грани: ширина в мм и правило разметки»).
    narrow_halfwidth_mm: float = 0.0
    narrow_step_deg: float = 30.0
    narrow_turn_deg: float = 30.0


@dataclass(frozen=True)
class ModelCfg:
    neighbors: int | list[int]      # одно число или своё на каждый слой
    rings: tuple[tuple[int, int, int], ...]
    frontier_cap: int
    frontier_per_ring: int
    log_curvature: bool
    emb_size: int
    dropout: float


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    weight_decay: float
    accum: int
    epochs: int
    lr_patience: int
    margin: float
    type_weight: float
    exclude_ambiguous: bool
    rotate: bool
    lr_find_min: float
    lr_find_max: float


@dataclass(frozen=True)
class ClusterCfg:
    bandwidth: float | None
    quantile: float
    bandwidth_samples: int
    bandwidth_floor: float
    iterations: int
    chunk: int
    max_clusters: int


@dataclass(frozen=True)
class BrepCfg:
    min_triangles: int
    boundary_rings: int
    min_fit_points: int
    rms_tol_rel: float
    parallel_deg: float
    tangent_rel: float
    rms_growth: float
    solver_iterations: int
    edge_tol_edges: float
    min_chain_edges: int
    joint_tol_rel: float
    rms_tol_mm: float = 0.0


@dataclass(frozen=True)
class TileCfg:
    """Фиксированное ребро и куски (README, «Фиксированное ребро и куски»). Секция необязательная:
    без неё сборка и загрузка — прежние."""
    edge_mm: float              # ребро рабочего меша
    faces: int                  # треугольников в ядре куска (с лоссом)
    halo_rings: int             # кайма вокруг ядра без лосса, колец смежности
    max_faces: int              # потолок куска с каймой: больше — ядро меньше
    tiles: int                  # кусков с детали не больше
    scale_mm: float             # L0: координаты делятся, кривизна умножается
    jet: bool                   # подложка и струя; без них рабочий меш — упрощение меша скана
    substrate_ratio: float      # подложка во столько раз подробнее рабочего меша, один проход ремеша
    jet_mm: tuple[float, ...]   # радиусы струи
    cache_dir: str              # каталог кэша в dataset/data (ссылка на диск DATA)
    # scan — рабочий меш из виртуального скана; cad — сама тесселяция CAD из parts_cad, как есть
    # (README, «Сегментация CAD-тесселяций»)
    source: str = "scan"


@dataclass(frozen=True)
class Config:
    remesh: RemeshCfg
    features: FeaturesCfg
    noise: NoiseCfg
    tessellation: TessellationCfg
    model: ModelCfg
    train: TrainCfg
    cluster: ClusterCfg
    brep: BrepCfg
    tile: TileCfg | None = None


_SECTIONS = {"remesh": RemeshCfg, "features": FeaturesCfg,
             "noise": NoiseCfg, "tessellation": TessellationCfg,
             "model": ModelCfg, "train": TrainCfg, "cluster": ClusterCfg,
             "brep": BrepCfg}


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Прочитать конфиг. Лишние и недостающие ключи — ошибка, а не умолчание:
    молча разошедшийся конфиг хуже отсутствующего. Исключение — поля с умолчанием,
    которое выключает добавленное позже: старым конфигам их не дописывать."""
    raw = yaml.safe_load(Path(path).read_text())
    parts = {}
    for name, cls in _SECTIONS.items():
        want = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING}
        got = set(raw.get(name, {}))
        if got - want or required - got:
            raise ValueError(f"секция {name}: лишние {got - want}, недостающие {required - got}")
        vals = dict(raw[name])
        if name == "features":
            vals["scales_rel"] = tuple(vals["scales_rel"])
        if name == "noise":
            for key in ("blur_sigma_mm", "hf_sigma_range_mm", "wave_sigma_range_mm", "wave_corr_range_mm",
                        "fine_sigma_range_mm", "fine_corr_range_mm", "rolloff_alpha_range",
                        "rolloff_sigma_range_mm", "bump_density_range", "bump_radius_range_mm",
                        "bump_height_range_mm", "ripple_sigma_range_mm", "ripple_period_range_mm",
                        "ripple_spread_range_deg"):
                vals[key] = tuple(vals[key])
        if name == "model":
            vals["rings"] = tuple(tuple(r) for r in vals["rings"])
        parts[name] = cls(**vals)
    if "tile" in raw:
        want = {f.name for f in fields(TileCfg)}
        required = {f.name for f in fields(TileCfg) if f.default is MISSING}
        got = set(raw["tile"])
        if got - want or required - got:
            raise ValueError(f"секция tile: лишние {got - want}, недостающие {required - got}")
        parts["tile"] = TileCfg(**{**raw["tile"], "jet_mm": tuple(raw["tile"]["jet_mm"])})
    return Config(**parts)


CFG = load_config(CONFIG_PATH)
