"""Конфиг эталона ParSeNet: `configs/parsenet.yaml`, лишние и недостающие ключи — ошибка."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.config import ROOT


@dataclass(frozen=True)
class DataCfg:
    points: int
    train_points: int
    normal_noise: float


@dataclass(frozen=True)
class ModelCfg:
    emb_size: int
    neighbors: int


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    batch_size: int
    batches_per_step: int
    micro_batch: int
    lr_factor: float
    lr_patience: int
    lr_min: float
    margin: float
    max_segment_pairs: int
    samples_per_segment: int


@dataclass(frozen=True)
class ClusterCfg:
    quantile: float
    iterations: int
    bandwidth_samples: int
    bandwidth_floor: float
    max_clusters: int


@dataclass(frozen=True)
class ParsenetConfig:
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    cluster: ClusterCfg


_SECTIONS = {"data": DataCfg, "model": ModelCfg, "train": TrainCfg, "cluster": ClusterCfg}


def load(path: str | Path = ROOT / "configs" / "parsenet.yaml") -> ParsenetConfig:
    raw = yaml.safe_load(Path(path).read_text())
    parts = {}
    for name, cls in _SECTIONS.items():
        want = {f.name for f in fields(cls)}
        got = set(raw.get(name, {}))
        if want != got:
            raise ValueError(f"секция {name}: лишние {got - want}, недостающие {want - got}")
        parts[name] = cls(**raw[name])
    return ParsenetConfig(**parts)


PCFG = load()
