"""Конфиг и пороги прогона для ноутбуков: всё по имени прогона, без переменных окружения.

Конфиг (разрешение, признаки, кэш) прогона записан в его `runs/<прогон>/train.sh` как
`CURV_CONFIG=...`; у прогонов на 30 тысячах треугольников, запущенных до опытов с
конфигами, `train.sh` нет — у них основной конфиг. `src.config` читает конфиг один раз
при первом импорте, поэтому `use_run` вызывается до импорта кода проекта.
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "default.yaml"
# Число треугольников рабочего меша, под которое заданы пороги в треугольниках
# (metrics.MIN_FACE, boundary.yaml min_region).
BASE_FACES = 30000


def run_config(run: str) -> Path:
    """Конфиг прогона: из его train.sh, иначе основной."""
    sh = ROOT / "runs" / run / "train.sh"
    found = re.search(r"CURV_CONFIG=(\S+)", sh.read_text()) if sh.exists() else None
    return (ROOT / found.group(1)).resolve() if found else DEFAULT.resolve()


def absorb_mm(run: str) -> float:
    """Порог поглощения скруглений и фасок прогона (`--absorb-blends-mm` в его train.sh), 0 — нет."""
    sh = ROOT / "runs" / run / "train.sh"
    found = re.search(r"--absorb-blends-mm\s+(\S+)", sh.read_text()) if sh.exists() else None
    return float(found.group(1)) if found else 0.0


def run_flags(run: str) -> dict:
    """Разметка и сплиты прогона из его train.sh: правило узких граней и имена сплитов."""
    sh = ROOT / "runs" / run / "train.sh"
    text = sh.read_text() if sh.exists() else ""
    split = lambda flag, default: (re.search(flag + r"\s+(\S+)", text) or [None, default])[1]
    return {"narrow": "--narrow-rule" in text, "train_split": split("--train-split", "train"),
            "val_split": split("--val-split", "val")}


def use_run(run: str) -> Path:
    """Поставить конфиг прогона и вернуть его путь. Если код проекта уже загружен с
    другим конфигом, это ошибка: нужно перезапустить ядро, а не молча считать на чужом."""
    path = run_config(run)
    if "src.config" in sys.modules:
        loaded = sys.modules["src.config"].CONFIG_PATH
        if loaded != path:
            raise RuntimeError(f"в ядре уже загружен конфиг {loaded.name}, а у прогона {run} — {path.name}: "
                               "перезапустите ядро и выполните ячейки заново")
        return path
    os.environ["CURV_CONFIG"] = str(path)
    return path


def thresholds():
    """`(min_face, partition)` под разрешение конфига: пороги заданы в треугольниках, и при
    другом числе треугольников рабочего меша это та же доля детали, что в обучении
    (README, «Разрешение»): на 60 тысячах 10 и 60."""
    from src.boundary.config import BCFG
    from src.config import CFG
    from src.metrics import MIN_FACE
    scale = CFG.remesh.target_faces / BASE_FACES
    return round(MIN_FACE * scale), dataclasses.replace(BCFG.partition,
                                                        min_region=round(BCFG.partition.min_region * scale))
