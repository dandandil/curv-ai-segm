"""Конфиг варианта D: `configs/boundary.yaml`, с той же строгостью, что основной:
лишние и недостающие ключи — ошибка."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.config import ROOT

BOUNDARY_CONFIG = ROOT / "configs" / "boundary.yaml"


@dataclass(frozen=True)
class HeadCfg:
    hidden: int
    pair_channels: int
    pair_hidden: int
    product: bool
    repulsive: bool
    repulsive_per_layer: int


@dataclass(frozen=True)
class BoundaryTrainCfg:
    edge_weight: float
    pos_weight_max: float
    tangent_weight: float
    exclude_band: bool
    face_balance: bool
    contrast_weight: float
    contrast_margin: float
    contrast_tau: float
    repulsive_weight: float
    backbone_lr: float
    head_lr: float


@dataclass(frozen=True)
class PartitionCfg:
    method: str
    beta: float
    depth: float
    merge: float
    veto: float | None
    min_region: int


@dataclass(frozen=True)
class BoundaryConfig:
    model: HeadCfg
    train: BoundaryTrainCfg
    partition: PartitionCfg


_SECTIONS = {"model": HeadCfg, "train": BoundaryTrainCfg, "partition": PartitionCfg}


def load_boundary_config(path: str | Path = BOUNDARY_CONFIG) -> BoundaryConfig:
    raw = yaml.safe_load(Path(path).read_text())
    parts = {}
    for name, cls in _SECTIONS.items():
        want = {f.name for f in fields(cls)}
        got = set(raw.get(name, {}))
        if want != got:
            raise ValueError(f"секция {name}: лишние {got - want}, недостающие {want - got}")
        parts[name] = cls(**raw[name])
    return BoundaryConfig(**parts)


BCFG = load_boundary_config()
