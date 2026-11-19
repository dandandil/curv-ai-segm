"""Инференс на произвольном меше: воспроизводимость и форма ответа."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import inference as I  # noqa: E402
from src.model import PRIMITIVE_TYPES  # noqa: E402

# Мелкая цель: тест проверяет устройство, а не качество нарезки. На полной
# рабочей точке (30 тысяч) один прогон стоит секунды.
TARGET = 2000


@pytest.fixture(scope="module")
def model():
    if not I.DEFAULT_CHECKPOINT.exists():
        pytest.skip("нет чекпоинта")
    return I.load_model()


def test_same_seed_gives_same_labels(model):
    """Соседи в кольцах выбираются случайно: без семени ответ плавает."""
    mesh = trimesh.creation.box((10.0, 6.0, 3.0))
    a = I.segment(mesh, model, target_faces=TARGET)
    b = I.segment(mesh, model, target_faces=TARGET)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.types, b.types)


def test_answer_matches_mesh(model):
    mesh = trimesh.creation.box((10.0, 6.0, 3.0))
    out = I.segment(mesh, model, target_faces=TARGET)
    assert len(out.labels) == len(out.mesh.faces) == len(out.types)
    assert out.labels.min() == 0 and len(np.unique(out.labels)) == out.labels.max() + 1
    assert set(out.cluster_types) == set(np.unique(out.labels).tolist())
    assert set(out.cluster_types.values()) <= set(PRIMITIVE_TYPES)
    assert len(out.strength) == len(out.mesh.face_adjacency)
    assert np.allclose(np.linalg.norm(out.embedding, axis=1), 1.0)


def test_clean_input_is_detected(model):
    """У меша прямо из CAD кривизна на плоскостях ровно нулевая: сеть такого не
    видела, и шум обучения накладывается сам.

    Порог `CLEAN_FLOOR` откалиброван на рабочих 30 тысячах треугольников, а
    здесь цель мелкая и подложка грубее радиуса струи, поэтому пол кривизны
    сжат и порога не достигает. Проверяется то, что от признака и требуется:
    на чистом входе он ноль, на зашумлённом заметно больше.
    """
    box = trimesh.creation.box((10.0, 6.0, 3.0))
    assert I.segment(box, model, target_faces=TARGET).noised

    _, clean, names, _ = I.prepare(box, target_faces=TARGET, clean_input=False)
    assert I.noise_floor(clean, names) == 0.0

    # Шум накладывается на уже подробную сетку: на грубой он сгладился бы
    # ремешем подложки, и вход остался бы кусочно-плоским, то есть «чистым».
    rough = box.subdivide().subdivide().subdivide()
    rough.vertices = rough.vertices + np.random.default_rng(0).normal(0, 0.1, rough.vertices.shape)
    _, noisy, names, _ = I.prepare(rough, target_faces=TARGET, clean_input=False)
    assert I.noise_floor(noisy, names) > 10 * I.noise_floor(clean, names) + 0.1


def test_bad_input_is_refused(model):
    with pytest.raises(ValueError):
        I.segment(trimesh.Trimesh(np.zeros((3, 3)), np.array([[0, 1, 2]])), model, target_faces=TARGET)
