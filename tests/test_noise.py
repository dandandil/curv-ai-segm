"""Модель шума: размытие в миллиметрах и выбор параметров шума детали."""
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import cache as D  # noqa: E402
from dataset import noise as N  # noqa: E402


def test_gaussian_blur_keeps_plane_and_smooths_bump():
    """Плоскость остаётся плоскостью; узкий бугор сглаживается тем сильнее, чем больше σ."""
    n = 60
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(-1.5, 1.5, n))
    flat = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    faces = np.array([tri for i in range(n - 1) for j in range(n - 1)
                      for tri in ([i * n + j, i * n + j + 1, (i + 1) * n + j + 1],
                                  [i * n + j, (i + 1) * n + j + 1, (i + 1) * n + j])])
    assert np.abs(N.gaussian_blur(flat, faces, 0.1)[:, 2]).max() < 1e-9
    bump = flat.copy()
    bump[:, 2] = 0.05 * np.exp(-(x.ravel() ** 2 + y.ravel() ** 2) / (2 * 0.05 ** 2))
    peaks = [N.gaussian_blur(bump, faces, s)[:, 2].max() for s in (0.0, 0.05, 0.1)]
    assert peaks[0] > peaks[1] > peaks[2]


def test_noise_params_taubin_is_previous_noise():
    p = D.noise_params(3)
    c = D.CFG.noise
    if c.blur == "taubin":
        assert p == {"blur": "taubin", "edge_radius_mm": c.edge_radius_mm, "hf_sigma_mm": c.hf_sigma_mm,
                     "wave_sigma_mm": c.wave_sigma_mm, "wave_corr_mm": c.wave_corr_mm}


def test_noise_params_gauss_per_part_and_fixed(monkeypatch):
    noise = dataclasses.replace(D.CFG.noise, blur="gauss", blur_sigma_mm=(0.06, 0.30),
                                hf_sigma_range_mm=(0.001, 0.003), wave_sigma_range_mm=(0.0, 0.02),
                                wave_corr_range_mm=(1.5, 6.0))
    monkeypatch.setattr(D, "CFG", dataclasses.replace(D.CFG, noise=noise))
    a, b = D.noise_params(1), D.noise_params(2)
    assert a == D.noise_params(1) and a != b                    # своё у каждой детали, воспроизводимо
    for p in (a, b):
        assert 0.06 <= p["blur_sigma_mm"] <= 0.30 and 0.001 <= p["hf_sigma_mm"] <= 0.003
        assert 0.0 <= p["wave_sigma_mm"] <= 0.02 and 1.5 <= p["wave_corr_mm"] <= 6.0
    fixed = D.noise_params(1, "fixed-0.12")
    assert fixed["blur_sigma_mm"] == 0.12 and fixed["hf_sigma_mm"] == 0.002 and fixed["wave_corr_mm"] == 3.75
    assert D.nominal_noise_params()["blur_sigma_mm"] == 0.18
    assert D.check_variant("fixed-0.12") == "fixed-0.12"


def test_noise_params_fine_component(monkeypatch):
    """Мелкая компонента: выключенная не появляется в параметрах, включённая берётся
    последней и прежние параметры детали не сдвигает."""
    noise = dataclasses.replace(D.CFG.noise, blur="gauss", blur_sigma_mm=(0.06, 0.30),
                                hf_sigma_range_mm=(0.001, 0.003), wave_sigma_range_mm=(0.0, 0.02),
                                wave_corr_range_mm=(1.5, 6.0), fine_sigma_range_mm=(0.0, 0.0),
                                fine_corr_range_mm=(0.0, 0.0))
    monkeypatch.setattr(D, "CFG", dataclasses.replace(D.CFG, noise=noise))
    off = D.noise_params(1)
    assert not any(k.startswith("fine_") for k in off)
    noise = dataclasses.replace(noise, fine_sigma_range_mm=(0.0002, 0.001), fine_corr_range_mm=(0.05, 0.1))
    monkeypatch.setattr(D, "CFG", dataclasses.replace(D.CFG, noise=noise))
    on = D.noise_params(1)
    assert {k: on[k] for k in off} == off
    assert 0.0002 <= on["fine_sigma_mm"] <= 0.001 and 0.05 <= on["fine_corr_mm"] <= 0.1
    assert abs(D.noise_params(1, "fixed-0.18")["fine_sigma_mm"] - 0.0006) < 1e-12

def test_rolloff_keeps_plane_and_sinks_edge():
    """Скат: на плоскости ничего не делает, у ребра двугранного угла поверхность проседает."""
    n = 41
    g = np.linspace(-2.0, 2.0, n)
    x, y = np.meshgrid(g, g)
    faces = np.array([tri for i in range(n - 1) for j in range(n - 1)
                      for tri in ([i * n + j, i * n + j + 1, (i + 1) * n + j + 1],
                                  [i * n + j, (i + 1) * n + j + 1, (i + 1) * n + j])])
    flat = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    assert np.abs(N.rolloff(flat, flat, faces, 0.1, 1.0)[:, 2]).max() < 1e-9
    roof = np.c_[x.ravel(), y.ravel(), -np.abs(x.ravel())]          # ребро вдоль y
    out = N.rolloff(roof, roof, faces, 0.1, 1.0)
    ridge = np.abs(x.ravel()) < 1e-9
    assert (out[ridge, 2] < -1e-3).all()


def test_bumps_count_and_extent():
    """Кочки: без плотности поверхность не меняется; с плотностью сдвиг не больше высоты."""
    pts = np.c_[np.random.default_rng(0).uniform(0, 10, (20000, 2)), np.zeros(20000)]
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    rng = np.random.default_rng(1)
    assert (N.bumps(pts, nrm, 100.0, rng, 0.0, (0.1, 0.2), (0.002, 0.008)) == pts).all()
    out = N.bumps(pts, nrm, 100.0, rng, 0.3, (0.1, 0.2), (0.002, 0.008))
    assert 0 < np.abs(out[:, 2]).max() <= 0.008 * 1.5 and (out[:, :2] == pts[:, :2]).all()

def test_ripple_amplitude_and_direction():
    """Рябь: сдвиг только вдоль нормали, разброс равен sigma, без sigma — поверхность прежняя."""
    g = np.linspace(0, 6.0, 200)
    x, y = np.meshgrid(g, g)
    pts = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    out = N.ripple(pts, nrm, np.random.default_rng(0), 0.002, 1.0, 10.0)
    assert (out[:, :2] == pts[:, :2]).all() and abs(out[:, 2].std() - 0.002) < 1e-4
    assert (N.ripple(pts, nrm, np.random.default_rng(0), 0.0, 1.0, 10.0) == pts).all()

def test_fourier_field_matches_diffused_correlation():
    """Поле волнами: нулевое среднее, единичный разброс, длина корреляции как у поля
    усреднением по соседям (ℓ ≈ 1.13·corr), и воспроизводимо по семени."""
    g = np.linspace(0, 6.0, 120)
    x, y = np.meshgrid(g, g)
    pts = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    f = N._fourier_field(pts, 0.3, np.random.default_rng(0))
    assert abs(f.mean()) < 1e-9 and abs(f.std() - 1) < 1e-9
    assert np.array_equal(f, N._fourier_field(pts, 0.3, np.random.default_rng(0)))
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, len(pts), 200000), rng.integers(0, len(pts), 200000)
    r = np.linalg.norm(pts[i] - pts[j], axis=1)
    near = np.abs(r - 0.34) < 0.03                      # на r = ℓ гаусс даёт exp(-1/2) ≈ 0.61
    assert 0.45 < np.mean(f[i[near]] * f[j[near]]) < 0.75
