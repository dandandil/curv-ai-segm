"""Сплиты корпуса ABC: dataset/data/splits/abc_{train,val,test}.txt.

    python dataset/abc_splits.py

Отбор — `~/data/abc/abc_selected.csv` (abc_count.py: одно тело, уникальные, неаналитических
не больше 80%), здесь ещё 15-300 граней и диагональ до 1000 мм: родной масштаб, а крупнее —
вероятно, ошибка единиц (README, «Корпус ABC: план сборки»). Сплит — по документу Onshape:
варианты одного документа почти одинаковы и не должны попасть и в train, и в val. Порядок
внутри сплита случайный и постоянный: кэш собирается по порядку, и любая собранная часть —
случайная выборка сплита.
"""
from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.cache import ABC_PREFIX, SPLITS  # noqa: E402

SELECTED = Path.home() / "data" / "abc" / "abc_selected.csv"
FACES = (15, 300)
MAX_DIAG_MM = 1000.0
# Доли документов в val и test.
HELD_OUT = {"val": 0.01, "test": 0.01}
SEED = 0


def doc_split(doc: str) -> str:
    """Сплит документа по хешу его id: не зависит от состава отбора."""
    x = int(hashlib.sha1(doc.encode()).hexdigest()[:8], 16) / 16 ** 8
    edge = 0.0
    for name, share in HELD_OUT.items():
        edge += share
        if x < edge:
            return name
    return "train"


def main() -> None:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for r in csv.DictReader(SELECTED.open()):
        diag = float(np.linalg.norm([float(v) for v in r["extent_mm_sorted"].split()]))
        if FACES[0] <= int(r["faces"]) <= FACES[1] and diag <= MAX_DIAG_MM:
            splits[doc_split(r["doc"])].append(ABC_PREFIX + r["uid"])
    rng = random.Random(SEED)
    for name, uids in splits.items():
        uids.sort()
        rng.shuffle(uids)
        (SPLITS / f"{ABC_PREFIX}{name}.txt").write_text("\n".join(uids) + "\n")
        print(f"{name}: {len(uids)}")


if __name__ == "__main__":
    main()
