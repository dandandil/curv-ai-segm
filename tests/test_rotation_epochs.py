"""Поворот обучающей детали новый при каждой выдаче, а не один на прогон."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CachedParts, split_uids  # noqa: E402


def test_rotation_changes_between_epochs():
    uids = split_uids("train", "noisy")[:1]
    if not uids:
        pytest.skip("нет кэша")
    ds = CachedParts(uids, "noisy", rotate=True)
    a, b = ds[0]["features"], ds[0]["features"]
    assert not np.allclose(a[3:6], b[3:6])                 # нормали повёрнуты по-разному
    assert np.allclose(np.linalg.norm(a[3:6], axis=0), 1.0, atol=1e-4)
