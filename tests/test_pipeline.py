"""Дымовой тест конвейера: одна деталь проходит цепочку целиком.

Проверяются инварианты, которые ломались молча: соответствие признаков граням
по индексам, достижение целевого числа
треугольников, отсутствие нечисловых значений.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset import features as FE  # noqa: E402
from dataset import remesh as R  # noqa: E402


@pytest.fixture(scope="module")
def built():
    uids = sorted(p.name for p in D.PARTS.iterdir() if (p / "mesh.ply").exists())
    if not uids:
        pytest.skip("нет данных в data/parts")
    uid = uids[0]
    src = trimesh.load(str(D.PARTS / uid / "mesh.ply"), process=False)
    sub = R.substrate(src)
    work = R.decimate(sub)
    mats, names, info = FE.compute_many(sub, [work])
    return dict(src=src, sub=sub, work=work, xw=mats[0], names=names, info=info)


def test_substrate_is_finer_than_target(built):
    assert len(built["sub"].faces) >= R.SUBSTRATE_MIN_RATIO * R.TARGET_FACES * 0.9


def test_meshes_hit_target_size(built):
    assert len(built["work"].faces) >= 0.95 * R.TARGET_FACES


def test_features_aligned_with_faces(built):
    """Строк ровно столько же, сколько граней: выброс вырожденных сдвинул бы
    индексы, и матрица разъехалась бы с мешем молча."""
    assert built["xw"].shape[0] == len(built["work"].faces)
    assert built["xw"].shape[1] == len(built["names"])


def test_features_finite(built):
    assert np.isfinite(built["xw"]).all()


def test_radius_matches_request(built):
    for meta in built["info"]:
        assert not meta["missed"], meta


def test_thin_wall_not_seen_through():
    """Стенка тоньше окрестности: посреди плоскости кривизна — ноль. Соседи по
    расстоянию брали точки с обратной стороны и давали здесь сотни."""
    plate = trimesh.creation.box((20.0, 20.0, 0.5)).subdivide_to_size(0.25)
    rel = 0.8 / np.sqrt(plate.area)                  # h = 0.8 мм > толщины 0.5
    x, names = FE.compute(plate, plate, scales=(rel,))
    c = plate.triangles.mean(axis=1)
    middle = (np.abs(c[:, :2]).max(axis=1) < 5.0) & (np.abs(np.abs(c[:, 2]) - 0.25) < 1e-6)
    assert middle.sum() > 100
    assert np.abs(x[middle, names.index(f"S@{rel}")]).max() < 1e-3


def test_labels_cover_all_faces(built):
    face_id = np.load(D.PARTS / sorted(p.name for p in D.PARTS.iterdir())[0] / "face_id.npy")
    lab = R.transfer_labels(built["src"], face_id, built["work"])
    assert len(lab) == len(built["work"].faces)
    assert lab.min() >= 0


def test_cache_tag_separates_cache_settings(tmp_path, monkeypatch):
    """Метка каталога кэша: пусто у основного конфига, число треугольников при
    другом разрешении и отпечаток, если отличаются другие настройки кэша."""
    import yaml
    from dataset import cache
    from src.config import DEFAULT_CONFIG
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())

    def tag_for(edit):
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        edit(cfg)
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        monkeypatch.setattr(cache, "CONFIG_PATH", path)
        return cache._cache_tag()

    assert tag_for(lambda c: None) == ""
    assert tag_for(lambda c: c["model"].update(rings=[[1, 2, 1]])) == ""      # модель кэш не меняет
    assert tag_for(lambda c: c["remesh"].update(target_faces=60000)) == "60k"
    noisy = tag_for(lambda c: c["noise"].update(edge_radius_mm=raw["noise"]["edge_radius_mm"] * 2))
    assert noisy.startswith("30k-") and len(noisy) == len("30k-") + 8        # другой шум — свой кэш


def test_absorb_blends_splits_strip_in_middle():
    """Полоса-скругление между двумя гранями делится между ними примерно пополам."""
    from src.data import absorb_blends
    # цепочка треугольников: 0-3 грань A, 4-7 скругление, 8-11 грань B
    labels = np.array([0] * 4 + [1] * 4 + [2] * 4)
    adjacency = np.array([[i, i + 1] for i in range(11)])
    out = absorb_blends(labels, adjacency, np.array([False, True, False]))
    assert (out[:6] == 0).all() and (out[6:] == 2).all()
    lone = absorb_blends(np.array([1, 1]), np.array([[0, 1]]), np.array([False, True]))
    assert (lone == 1).all()
