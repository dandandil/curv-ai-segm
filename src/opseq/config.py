"""Конфиг `configs/opseq.yaml`, с той же строгостью, что основной: лишние и недостающие ключи — ошибка."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.config import ROOT

OPSEQ_CONFIG = ROOT / "configs" / "opseq.yaml"


@dataclass(frozen=True)
class DataCfg:
    corpora: list
    train_corpora: list
    labels: str
    points: int
    max_spd: int
    max_nodes: int
    val_frac: float
    test_frac: float
    relation_sin: float
    relation_rel: float


@dataclass(frozen=True)
class ModelCfg:
    layers: int
    dim: int
    heads: int
    ffn: int
    dropout: float
    pair_dim: int
    point_dim: int


@dataclass(frozen=True)
class TrainCfg:
    batch: int
    batch_pairs: int
    lr: float
    weight_decay: float
    warmup: int
    epochs: int
    grad_clip: float
    type_weight: float
    support_weight: float
    faces_weight: float
    boolean_weight: float
    to_weight: float
    depth_weight: float
    direction_weight: float
    relation_weight: float
    role_weight: float
    same_op_weight: float


@dataclass(frozen=True)
class OpseqConfig:
    data: DataCfg
    model: ModelCfg
    train: TrainCfg


_SECTIONS = {"data": DataCfg, "model": ModelCfg, "train": TrainCfg}


def load_opseq_config(path: str | Path = OPSEQ_CONFIG) -> OpseqConfig:
    raw = yaml.safe_load(Path(path).read_text())
    parts = {}
    for name, cls in _SECTIONS.items():
        want = {f.name for f in fields(cls)}
        got = set(raw.get(name, {}))
        if want != got:
            raise ValueError(f"секция {name}: лишние {got - want}, недостающие {want - got}")
        parts[name] = cls(**raw[name])
    return OpseqConfig(**parts)


OCFG = load_opseq_config()
