"""Конфиг `configs/sketch.yaml` (или файл из `SKETCH_CONFIG`, как `CURV_CONFIG` основного), с той же
строгостью, что основной: лишние и недостающие ключи — ошибка."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.config import ROOT

SKETCH_CONFIG = Path(os.environ.get("SKETCH_CONFIG", ROOT / "configs" / "sketch.yaml"))


@dataclass(frozen=True)
class DataCfg:
    root: str
    planes_per_part: int
    oblique_frac: float
    depth_margin: float
    coplanar_rel: float
    min_seg_rel: float
    min_chain: int
    label_rel: float
    tangent_deg: float
    seam_curvature: float
    seam_step_rel: float
    seam_angle_deg: float
    spline_join_deg: float
    unresolved_gap: int
    unresolved_turn_deg: float
    vertex_rel: float
    max_len: int


@dataclass(frozen=True)
class ModelCfg:
    channels: int
    dilations: list


@dataclass(frozen=True)
class TrainCfg:
    batch_segments: int
    lr: float
    weight_decay: float
    warmup: int
    epochs: int
    grad_clip: float
    type_weight: float
    boundary_weight: float
    tangent_weight: float
    peak_threshold: float
    peak_window: int
    split_rel: float
    match_rel: float


@dataclass(frozen=True)
class SketchConfig:
    data: DataCfg
    model: ModelCfg
    train: TrainCfg


_SECTIONS = {"data": DataCfg, "model": ModelCfg, "train": TrainCfg}


def load_sketch_config(path: str | Path = SKETCH_CONFIG) -> SketchConfig:
    raw = yaml.safe_load(Path(path).read_text())
    parts = {}
    for name, cls in _SECTIONS.items():
        want = {f.name for f in fields(cls)}
        got = set(raw.get(name, {}))
        if want != got:
            raise ValueError(f"секция {name}: лишние {got - want}, недостающие {want - got}")
        parts[name] = cls(**raw[name])
    return SketchConfig(**parts)


SCFG = load_sketch_config()
