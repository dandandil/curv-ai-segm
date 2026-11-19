"""Сборка кэша обучения: STEP-меш корпуса -> рабочий меш, признаки, разметка.

Один вызов `build_part` делает всю цепочку для одной детали и складывает
результат в `data/cache/<uid>/<variant>/` (у конфига опыта с другими настройками
кэша — в `<variant>.<метка>`, см. `part_dir`). Подложка внутри сборки не
сохраняется — она нужна только как носитель шума и источник окрестностей для
струи (см. README, «Конвейер данных»).

Вариантов два: `clean` и `noisy`. Шум закладывается в датасет заранее, а не
генерится на шаге обучения: пересборка стоит секунды на деталь, и на каждом
шаге это неподъёмно. Следствие, которое надо понимать: реализация шума на
деталь одна, то есть это сдвиг домена, а не аугментация.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.sparse.csgraph import connected_components

from . import features as FE
from . import noise as N
from . import remesh as R
from . import step_substrate as S
from src.config import CFG, CONFIG_PATH, DEFAULT_CONFIG

DATA = Path(__file__).resolve().parent / "data"
# Детали из своей тесселяции STEP (step_mesh.py). Старый корпус `parts` —
# общий с работой по SAM-2 и тесселирован грубо для кривизны, здесь не читается.
PARTS = DATA / "parts_fine"
# Свой каталог деталей (opseq собирает тела Fusion с историей отдельно, не трогая общий).
if os.environ.get("CURV_PARTS"):
    PARTS = Path(os.environ["CURV_PARTS"])
# Тесселяции CAD (build_parts_cad.py) — по конфигу, а не переменной: без неё загрузчик молча прочитал бы
# детали скана тех же uid.
CAD_SOURCE = bool(CFG.tile) and CFG.tile.source == "cad"
if CAD_SOURCE:
    PARTS = DATA / "parts_cad"
RAW_STEP = DATA / "raw_step"
# Корпус ABC: детали `abc_<номер модели>`, STEP — как скачан (~/data/abc, fetch_step.sh).
ABC_STEP = Path.home() / "data" / "abc" / "sel"
ABC_PREFIX = "abc_"
CACHE = DATA / "cache"
if CFG.tile:
    # Кэш кусков — на диске DATA: dataset/data/<cache_dir> — ссылка туда на каждой машине.
    CACHE = DATA / CFG.tile.cache_dir
SPLITS = DATA / "splits"


VARIANTS = ("clean", "noisy")
# Вариант с фиксированным размытием для оценки по уровням шума: `fixed-<σ, мм>`,
# прочие параметры — середины пределов (`noise_params`).
FIXED_PREFIX = "fixed-"
# Поток случайных чисел для выбора параметров шума детали — отдельный от потока самого
# шума, чтобы выбор параметров не сдвигал реализацию полей.
PARAM_STREAM = 7
SCAN_STREAM = 8                # поток параметров и шума виртуального сканера
TILE_STREAM = 9                # поток центров кусков
# Медианное ребро рабочего меша больше стольких ребёр конфига — скан грубее нужного (кадр сканера
# не больше 1200 пикселей, у детали от ~600 тысяч мм² шаг пикселя растёт): деталь пропускается.
# ~1% корпуса [измерено 2026-10-16: 99-й процентиль размера — 0.323 мм, самая крупная — 0.666].
MAX_EDGE_SHARE = 1.17
# Кандидатов в центры кусков не больше: жадный выбор по граням смотрит на их ячейки Вороного, и
# крупнее ячейка — лишь грубее оценка; дальние точки на миллионах треугольников иначе долги.
MAX_CANDIDATES = 64
# Разметка для выбора кусков — по стольким треугольникам: грани кандидатов видно и по подвыборке.
PICK_SAMPLE = 300_000
# Деталь больше по диагонали не собирается: при потолке шага пикселя её скан — сотни миллионов
# треугольников. ~1% корпуса. Выбрано.
MAX_DIAG_MM = 1000.0

# Разделы конфига, от которых зависит содержимое кэша. Модель и обучение его не
# меняют: опыты с другими кольцами на том же разрешении читают один кэш.
CACHE_SECTIONS = ("remesh", "noise", "features", "tessellation")
# Необязательные компоненты шума: префикс ключей и ключ, нули в котором её выключают.
OPTIONAL_NOISE = {"fine_": "fine_sigma_range_mm", "rolloff_": "rolloff_alpha_range", "bump_": "bump_density_range",
                  "ripple_": "ripple_sigma_range_mm"}


def _cache_tag() -> str:
    """Метка каталога кэша: пусто у основного конфига, иначе число треугольников и,
    если отличается ещё что-то влияющее на кэш, отпечаток этих настроек.

    Одного числа треугольников мало: конфиг с другим шумом на тех же 30 тысячах
    молча прочитал бы чужой кэш.
    """
    def settings(path: Path) -> dict:
        raw = yaml.safe_load(path.read_text())
        out = {name: raw[name] for name in CACHE_SECTIONS}
        if "tile" in raw:                 # у прежних конфигов секции нет — метка их кэша та же
            # L0 применяется при загрузке — кэш от него не зависит
            out["tile"] = {k: v for k, v in raw["tile"].items() if k != "scale_mm"}
        # Выключенная необязательная компонента кэш не меняет: метка как до её появления.
        for prefix, key in OPTIONAL_NOISE.items():
            if not max(out["noise"].get(key, [0.0])):
                out["noise"] = {k: v for k, v in out["noise"].items() if not k.startswith(prefix)}
        return out

    mine, base = settings(CONFIG_PATH), settings(DEFAULT_CONFIG)
    if mine == base:
        return ""
    tag = f"{mine['remesh']['target_faces'] // 1000}k"
    rest, base_rest = ({**s, "remesh": {k: v for k, v in s["remesh"].items() if k != "target_faces"}}
                       for s in (mine, base))
    if rest != base_rest:
        tag += "-" + hashlib.sha1(json.dumps(rest, sort_keys=True).encode()).hexdigest()[:8]
    return tag


# Опыт с другими настройками кэша пишет его рядом: `cache/<uid>/<вариант>.<метка>`.
# Иначе пересборка затёрла бы рабочую точку, с которой её же и сравнивают.
CONFIG_TAG = _cache_tag()


# Предрасчитанные кольца лежат рядом с кэшем детали. У конфига опыта (CURV_CONFIG)
# свой файл: кольца основного конфига не затираются.
RINGS_FILE = ("rings.npz" if CONFIG_PATH == DEFAULT_CONFIG.resolve()
              else f"rings.{CONFIG_PATH.stem}.npz")


def part_dir(uid: str, variant: str) -> Path:
    """Каталог кэша детали: у конфига опыта свой, у основного — прежний."""
    return CACHE / uid / (f"{variant}.{CONFIG_TAG}" if CONFIG_TAG else variant)


# Из какой тесселяции собран кэш. Готовая деталь пропускается только при
# совпадении: иначе оборванная пересборка приняла бы старые детали за новые.
TESSELLATION = (f"{CFG.tessellation.linear_rel}/{CFG.tessellation.angular}/"
                f"{CFG.tessellation.min_size_rel}")


def step_path(uid: str) -> Path:
    """STEP детали: у Fusion — data/raw_step/<uid>.stp, у ABC — единственный .step в папке модели."""
    if uid.startswith(ABC_PREFIX):
        found = sorted((ABC_STEP / uid[len(ABC_PREFIX):]).glob("*.step"))
        if not found:
            raise FileNotFoundError(f"нет STEP модели {uid}")
        return found[0]
    return RAW_STEP / f"{uid}.stp"


def check_variant(variant: str) -> str:
    if variant not in VARIANTS and not variant.startswith(FIXED_PREFIX):
        raise ValueError(f"вариант {variant!r}, ожидается один из {VARIANTS} или {FIXED_PREFIX}<σ, мм>")
    return variant


def noise_params(seed: int, variant: str = "noisy", blur_sigma_mm: tuple[float, float] | None = None) -> dict:
    """Параметры шума детали.

    При `blur: taubin` — фиксированные из конфига, как было всегда. При `gauss` у
    варианта `noisy` каждый параметр берётся равномерно из своих пределов по семени
    детали — один вариант шума на деталь, но разный уровень у разных деталей
    (README, «Аугментация шума»); у `fixed-<σ>` размытие задано, остальное —
    середины пределов. `blur_sigma_mm` — свои пределы размытия вместо конфига.
    """
    c = CFG.noise
    if c.blur == "taubin":
        return {"blur": "taubin", "edge_radius_mm": c.edge_radius_mm, "hf_sigma_mm": c.hf_sigma_mm,
                "wave_sigma_mm": c.wave_sigma_mm, "wave_corr_mm": c.wave_corr_mm}
    ranges = {"blur_sigma_mm": blur_sigma_mm or c.blur_sigma_mm, "hf_sigma_mm": c.hf_sigma_range_mm,
              "wave_sigma_mm": c.wave_sigma_range_mm, "wave_corr_mm": c.wave_corr_range_mm}
    # Новые компоненты — в конец: прежние параметры берутся из потока теми же числами.
    if max(c.fine_sigma_range_mm) > 0:
        ranges |= {"fine_sigma_mm": c.fine_sigma_range_mm, "fine_corr_mm": c.fine_corr_range_mm}
    if max(c.rolloff_alpha_range) > 0:
        ranges |= {"rolloff_alpha": c.rolloff_alpha_range, "rolloff_sigma_mm": c.rolloff_sigma_range_mm}
    if max(c.bump_density_range) > 0:
        ranges |= {"bump_density": c.bump_density_range}
    if max(c.ripple_sigma_range_mm) > 0:
        ranges |= {"ripple_sigma_mm": c.ripple_sigma_range_mm, "ripple_period_mm": c.ripple_period_range_mm,
                   "ripple_spread_deg": c.ripple_spread_range_deg}
    if variant.startswith(FIXED_PREFIX):
        out = {k: 0.5 * (lo + hi) for k, (lo, hi) in ranges.items()}
        out["blur_sigma_mm"] = float(variant[len(FIXED_PREFIX):])
    else:
        rng = np.random.default_rng([seed, PARAM_STREAM])
        out = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
    return {"blur": "gauss", **out}


def nominal_noise_params() -> dict:
    """Середины пределов — для инференса, которому нужен один типичный уровень шума."""
    c = CFG.noise
    if c.blur == "taubin":
        return noise_params(0)
    return noise_params(0, f"{FIXED_PREFIX}{0.5 * sum(c.blur_sigma_mm)}")


def apply_noise(sub: trimesh.Trimesh, seed: int, params: dict | None = None) -> trimesh.Trimesh:
    """Компоненты шума на подложке: размытие, длинные волны, высокочастотный и, если
    включены, мелкий, скат у рёбер, рябь и кочки.
    `params` — из `noise_params`; по умолчанию — по семени для варианта `noisy`."""
    params = params or noise_params(seed)
    v = np.asarray(sub.vertices)
    f = np.asarray(sub.faces)
    n = np.asarray(sub.vertex_normals)
    step = float(np.median(np.linalg.norm(v[sub.edges[:, 0]] - v[sub.edges[:, 1]], axis=1)))
    rng = np.random.default_rng(seed)
    if params["blur"] == "taubin":
        src, dst = N.mesh_edges(f)
        v = N.round_edges(v, src, dst, radius=params["edge_radius_mm"], spacing=step)
    else:
        try:
            v = N.gaussian_blur(v, f, params["blur_sigma_mm"])
        except RuntimeError:
            # Подложка с иглами: точная система не сходится, размытие — с обрезанными
            # котангенсами. Отмечается в параметрах, а с ними в meta.json.
            v = N.gaussian_blur(v, f, params["blur_sigma_mm"], robust=True)
            params["blur_robust"] = True
    if "rolloff_alpha" in params:
        coarse = R.decimate(trimesh.Trimesh(v, f, process=False))
        v = N.rolloff(v, np.asarray(coarse.vertices, float), np.asarray(coarse.faces),
                      params["rolloff_alpha"], params["rolloff_sigma_mm"])
    # Новая модель шума — и поля волнами: усреднение по соседям на мелкой подложке стоило
    # трети времени сборки. Прежняя (taubin) остаётся до бита прежней.
    fourier = params["blur"] == "gauss"
    v = N.waviness(v, n, step, rng, sigma=params["wave_sigma_mm"], corr=params["wave_corr_mm"], fourier=fourier)
    v = N.high_frequency(v, n, step, rng, sigma=params["hf_sigma_mm"], fourier=fourier)
    if "fine_sigma_mm" in params:
        # Мелкий шум сканера: та же конструкция на корреляции ~шага вершин скана.
        v = N.high_frequency(v, n, step, rng, sigma=params["fine_sigma_mm"], corr=params["fine_corr_mm"],
                             fourier=True)
    if "ripple_sigma_mm" in params:
        v = N.ripple(v, n, rng, params["ripple_sigma_mm"], params["ripple_period_mm"], params["ripple_spread_deg"])
    if "bump_density" in params:
        v = N.bumps(v, n, float(sub.area), rng, params["bump_density"], CFG.noise.bump_radius_range_mm,
                    CFG.noise.bump_height_range_mm)
    return trimesh.Trimesh(v, f, process=False)


def make_substrate(uid: str, src: trimesh.Trimesh) -> tuple[trimesh.Trimesh, str]:
    """Подложка детали и откуда она: `tessellation` — ремеш тесселяции, как было;
    `step` — сетка gmsh прямо по STEP (`substrate_source: step`); если gmsh не справился
    (на пробе 1 деталь из 100 — периодическая поверхность), `tessellation-presplit` —
    ремеш подразбитой тесселяции."""
    if CFG.remesh.substrate_source == "tessellation":
        return R.substrate(src), "tessellation"
    edge = R.substrate_edge(src)
    try:
        return S.substrate(step_path(uid), edge, src, faces=int(R.TARGET_FACES * R.SUBSTRATE_MIN_RATIO)), "step"
    except S.StepMeshError:
        return R.substrate(R.presplit(src, edge)), "tessellation-presplit"


def build_part(uid: str, variant: str = "clean", seed: int = 0,
               overwrite: bool = False) -> dict:
    """Собрать кэш одной детали. Возвращает метаданные."""
    check_variant(variant)
    out = part_dir(uid, variant)
    meta_path = out / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("tessellation") == TESSELLATION:
            return meta

    part = PARTS / uid
    src = trimesh.load(str(part / "mesh.ply"), process=False)
    face_id = np.load(part / "face_id.npy")
    if CFG.tile:
        return _build_tile(uid, variant, seed, src, face_id, out, meta_path)

    if CFG.noise.model == "scan" and variant != "clean":
        from dataset import scan as SC     # open3d нужен только этому конфигу
        rng = np.random.default_rng([seed, SCAN_STREAM])
        scanned, p = SC.scan(src, SC.sample_params(rng), rng)
        sub, source = R.substrate(scanned), "scan"
        params = {"model": "scan", **asdict(p)}
    else:
        sub, source = make_substrate(uid, src)
        params = None if variant == "clean" else noise_params(seed, variant)
        if params:
            sub = apply_noise(sub, seed, params)

    work = R.decimate(sub)

    # Метки берутся с ИСХОДНОЙ геометрии: шум сдвигает поверхность на микроны,
    # это на три порядка меньше треугольника, так что на приписку не влияет.
    labels = R.transfer_labels(src, face_id, work)
    amb = R.ambiguous(work, labels)

    (xw,), names, scale_info = FE.compute_many(sub, [work])
    _, _, bad_w = FE.face_frames(work)

    out.mkdir(parents=True, exist_ok=True)
    # Массивы пишутся во временный файл: оборванная запись не заменит прежний кэш.
    # float32, а не float16: dII после обезразмеривания доходит до сотен тысяч
    # и во float16 молча уходит в бесконечность.
    tmp = out / "work.tmp.npz"
    np.savez_compressed(
        tmp,
        vertices=np.asarray(work.vertices, np.float32),
        faces=np.asarray(work.faces, np.int32),
        adjacency=np.asarray(work.face_adjacency, np.int32),
        labels=labels.astype(np.int32),
        ambiguous=amb,
        features=xw.astype(np.float32),
    )
    # meta.json — признак готовой детали, поэтому снимается до замены массивов и пишется
    # последним: между ними оборванная сборка оставит деталь несобранной, а не прежнюю
    # метадату при новых массивах.
    meta_path.unlink(missing_ok=True)
    os.replace(tmp, out / "work.npz")
    # Кольца подписаны только конфигом соседства, не мешем: от прежней сборки
    # они прошли бы проверку и легли бы на чужие треугольники. Файлов колец по
    # одному на конфиг опыта (`RINGS_FILE`), и все они от прежнего меша.
    for rings in out.glob("rings*.npz"):
        rings.unlink()
    # Глобальный меш больше не собирается (README, «Сборка кэша»); от прежней
    # сборки он остался бы лежать мёртвым грузом.
    (out / "global.npz").unlink(missing_ok=True)
    counts = np.bincount(labels[labels >= 0])
    counts = counts[counts > 0]
    meta = {
        "uid": uid, "variant": variant, "seed": seed, "tessellation": TESSELLATION, "noise": params,
        "substrate": source,
        "feature_names": names,
        "scales": scale_info,
        "substrate_faces": int(len(sub.faces)),
        "work_faces": int(len(work.faces)),
        "work_faces_short": bool(len(work.faces) < (1 - R.FACE_TOL) * R.TARGET_FACES),
        "brep_faces": int(len(np.unique(labels))),
        "ambiguous_frac": float(amb.mean()),
        "min_tri_per_brep": int(counts.min()) if len(counts) else 0,
        "median_tri_per_brep": int(np.median(counts)) if len(counts) else 0,
        "degenerate_normals": int(bad_w.sum()),
        "area_mm2": float(work.area),
        "bbox_diag_mm": float(np.linalg.norm(work.extents)),
        "watertight": bool(work.is_watertight),
    }
    tmp = out / "meta.tmp.json"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    os.replace(tmp, meta_path)
    return meta


def load_part(uid: str, variant: str = "clean", tile: int | None = None) -> dict:
    """Прочитать кэш обратно: метаданные и массивы рабочего меша (у кэша кусков — куска `tile`)."""
    d = part_dir(uid, variant)
    z = np.load((d if tile is None else d / f"t{tile}") / "work.npz")
    return {"meta": json.loads((d / "meta.json").read_text()),
            "work": {k: (z[k].astype(np.float32) if k == "features" else z[k]) for k in z.files}}


def _build_tile(uid: str, variant: str, seed: int, src: trimesh.Trimesh, face_id: np.ndarray,
                out: Path, meta_path: Path) -> dict:
    """Куски при фиксированном ребре (README, «Фиксированное ребро и куски»): деталь сканируется
    целиком, рабочий меш — упрощение скана (`tile.jet: false`) или подложки с ребром ребро/√ratio одним
    проходом, по площади на ребро `edge_mm`. Деталь не больше `faces` — один кусок, она сама;
    иначе до `tiles` кусков-кругов по поверхности вокруг дальних точек, выбранных жадно по новым
    граням B-Rep. Каждый кусок — свой каталог `t<i>` с `work.npz` (и потом кольцами); `t0` — самый
    богатый гранями, на него смотрит валидация. Признаки в мм: координаты (центр куска и L0 —
    при загрузке), нормаль и, со струёй, кривизна в 1/мм и ‖∇II‖ в 1/мм²."""
    import shutil
    from dataset import scan as SC
    from dataset import tiles as T
    t = CFG.tile
    tri = np.sqrt(3) / 4 * t.edge_mm ** 2
    diag = float(np.linalg.norm(src.extents))
    if CAD_SOURCE:
        # Тесселяция CAD как есть: метка треугольника — его грань, переносить нечего.
        scanned = base = work = src
        p = None
    else:
        if diag > MAX_DIAG_MM:
            raise RuntimeError(f"деталь {diag:.0f} мм по диагонали больше {MAX_DIAG_MM:.0f} — пропускается")
        rng = np.random.default_rng([seed, SCAN_STREAM])
        scanned, p = SC.scan(src, SC.sample_params(rng), rng)
        base = R.substrate_once(scanned, t.edge_mm / np.sqrt(t.substrate_ratio)) if t.jet else scanned
        work = R.decimate(base, int(base.area / tri))
    edge = float(np.median(work.edges_unique_length))
    if not CAD_SOURCE and edge > MAX_EDGE_SHARE * t.edge_mm:
        raise RuntimeError(f"скан грубее ребра: медиана {edge:.3f} мм при {t.edge_mm} — деталь пропускается")
    n = len(work.faces)
    pts = work.triangles_center
    if n <= t.faces:
        tiles = [(np.arange(n), np.ones(n, bool), 0)]     # вся деталь — ядро
    else:
        pick_rng = np.random.default_rng([seed, TILE_STREAM])
        k = min(int(np.ceil(n / t.faces)), MAX_CANDIDATES)     # ядра покрывают деталь
        hop = T._graph(work.face_adjacency, n)
        # Центры — только на связных частях меша не меньше ядра: на оторванном островке скана выходил
        # кусок из 26 треугольников [пилот 2026-10-17].
        comp = connected_components(hop, directed=False)[1]
        size = np.bincount(comp)[comp]
        big = np.flatnonzero(size >= min(t.faces, size.max()))
        seeds = big[T.centers(pts[big], k, pick_rng)]
        # Грани кандидатов — по подвыборке: разметка всего меша крупной детали (миллионы треугольников)
        # нужна только для выбора, сами куски размечаются ниже.
        samp = np.sort(pick_rng.choice(n, min(n, PICK_SAMPLE), replace=False))
        lab_s = np.asarray(face_id)[samp if CAD_SOURCE else R.closest_faces(src, pts[samp])]
        # У тесселяции ребро не задано, и радиус ядра в мм не оценить: Дейкстра без потолка.
        radius = np.inf if CAD_SOURCE else 3 * np.sqrt(t.faces * tri / np.pi)
        tiles = [T.tile(work.face_adjacency, pts, hop, int(seeds[i]), t.faces, t.halo_rings, t.max_faces, radius)
                 for i in T.pick(lab_s, pts[samp], pts[seeds], t.tiles)]

    # Ссылка на DATA: если диск не смонтирован, mkdir создал бы каталоги в tmpfs /run/media.
    if not CACHE.exists():
        raise FileNotFoundError(f"нет каталога кэша {CACHE} -> {CACHE.resolve()}: диск DATA не смонтирован?")
    meta_path.unlink(missing_ok=True)
    if out.exists():
        for old in out.glob("t*"):
            shutil.rmtree(old)
    info, all_labels, amb_frac, bad, names, scale_info = [], [], [], 0, None, []
    for i, (sel, core, shrunk) in enumerate(tiles):
        sub = work if len(sel) == n else work.submesh([sel], append=True)
        # Разметка и признаки — по куску: у края куска «неоднозначные» недосчитаны, но там кайма без лосса.
        labels = np.asarray(face_id)[sel] if CAD_SOURCE else R.transfer_labels(src, face_id, sub)
        amb = R.ambiguous(sub, labels)
        pw, nw, bad_w = FE.face_frames(sub)
        cols, names = [pw, nw], ["cx", "cy", "cz", "nx", "ny", "nz"]
        if t.jet:
            L = float(np.sqrt(base.area))
            coarse, FE.COARSE_ABOVE = FE.COARSE_ABOVE, np.inf    # радиус в мм, доля L тут ни при чём
            try:
                (xw,), jn, scale_info = FE.compute_many(base, [sub], tuple(r / L for r in t.jet_mm))
            finally:
                FE.COARSE_ABOVE = coarse
            for j, c in enumerate(jn[6:], 6):
                kind, rel = c.split("@")
                cols.append(xw[:, j:j + 1] / L ** (2 if kind == "dII" else 1))
                names.append(f"{kind}@{float(rel) * L:.2g}mm")
        d = out / f"t{i}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "work.tmp.npz"
        np.savez_compressed(
            tmp,
            vertices=np.asarray(sub.vertices, np.float32),
            faces=np.asarray(sub.faces, np.int32),
            adjacency=np.asarray(sub.face_adjacency, np.int32),
            labels=labels.astype(np.int32),
            ambiguous=amb,
            core=core,
            features=np.concatenate(cols, axis=1).astype(np.float32),
        )
        os.replace(tmp, d / "work.npz")
        info.append({"faces": int(len(sel)), "area_mm2": float(sub.area),
                     "brep_faces": int(len(np.unique(labels))),
                     "core_faces": int(core.sum()), "core_shrunk": shrunk,
                     "core_brep_faces": int(len(np.unique(labels[core])))})
        all_labels.append(labels)
        amb_frac.append(float(amb.mean()))
        bad += int(bad_w.sum())
    labels = np.concatenate(all_labels)
    counts = np.bincount(labels[labels >= 0])
    counts = counts[counts > 0]
    meta = {
        "uid": uid, "variant": variant, "seed": seed, "tessellation": TESSELLATION,
        "noise": None if CAD_SOURCE else {"model": "scan", **asdict(p)},
        "substrate": "cad" if CAD_SOURCE else "scan-remesh" if t.jet else "scan",
        "tiles": info,
        "feature_names": names,
        "scales": scale_info,
        "scan_faces": int(len(scanned.faces)),
        "substrate_faces": int(len(base.faces)),
        "work_faces": int(n),
        "brep_faces": int(len(np.unique(labels))),
        "ambiguous_frac": float(np.mean(amb_frac)),
        "min_tri_per_brep": int(counts.min()) if len(counts) else 0,
        "median_tri_per_brep": int(np.median(counts)) if len(counts) else 0,
        "degenerate_normals": bad,
        "area_mm2": float(work.area),
        "cad_area_mm2": float(src.area),
        "edge_mm": edge,
        "bbox_diag_mm": float(np.linalg.norm(work.extents)),
        "watertight": bool(work.is_watertight),
    }
    tmp = out / "meta.tmp.json"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    os.replace(tmp, meta_path)
    return meta
