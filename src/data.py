"""Чтение кэша для обучения: одна деталь — один элемент.

Батча нет: деталь занимает четыре с лишним гигабайта, и градиент копится по
деталям (README, «Бюджет»). Поэтому здесь не тензоры фиксированной формы, а
словарь массивов на деталь.

Тип поверхности читается из `brep.json` корпуса, а не из кэша: он там не
сохранён, а json маленький и лежит рядом с деталью.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataset.cache import DATA, PARTS, RINGS_FILE, SPLITS, TESSELLATION, load_part, part_dir

from . import graph as G
from .config import CFG
from .metrics import face_halfwidth
from .model import PRIMITIVE_TYPES
from .surfaces import CONE_LIMIT_DEG

# Каналы кривизны. У них хвост до 10^9 на рёбрах — артефакт подгонки струи на
# изломе, — и без сжатия он глушит координаты и нормаль (README, «Разброс
# каналов»).
CURVATURE_PREFIXES = ("k1@", "k2@", "S@", "dII@")


# Имена типов в brep.json: у нераспознанных OpenCASCADE генератор оставлял
# сырое имя перечисления, и сплайн — как раз такой случай.
TYPE_INDEX = {name: i for i, name in enumerate(PRIMITIVE_TYPES)}
TYPE_INDEX["geomabs_bsplinesurface"] = TYPE_INDEX["bspline"]
# Прочие неаналитические (в ABC выдавливание — 5.5% граней) для сети тоже сплайн: своего
# примитива у них нет (README, «Корпус ABC: план сборки»).
for _kind in ("geomabs_beziersurface", "geomabs_surfaceofextrusion", "geomabs_surfaceofrevolution",
              "geomabs_offsetsurface", "geomabs_othersurface"):
    TYPE_INDEX[_kind] = TYPE_INDEX["bspline"]


def _fresh(meta_path: Path) -> bool:
    """Кэш детали собран из текущей тесселяции. Деталь, исключённая из сборки,
    иначе осталась бы в сплите со старым кэшем другой формы."""
    return meta_path.exists() and json.loads(meta_path.read_text()).get("tessellation") == TESSELLATION


def split_uids(split: str, variant: str) -> list[str]:
    """Детали сплита, для которых кэш уже собран."""
    want = (SPLITS / f"{split}.txt").read_text().split()
    return [u for u in want if _fresh(part_dir(u, variant) / "meta.json")]


def part_meta(uid: str, variant: str) -> dict:
    """Метаданные детали без чтения самих массивов."""
    return json.loads((part_dir(uid, variant) / "meta.json").read_text())


def compress_curvature(features: np.ndarray, names: list[str]) -> np.ndarray:
    """Знаковый логарифм `sign(x)·log1p(|x|)` на каналах кривизны, `(C, N)`.

    Координаты и нормаль не трогаются: они порядка единицы и хвостов не имеют.
    Около нуля преобразование почти тождественно, сжимает оно только то, что
    больше единицы, и сохраняет порядок значений.
    """
    out = features.copy()
    for i, name in enumerate(names):
        if name.startswith(CURVATURE_PREFIXES):
            out[i] = np.sign(out[i]) * np.log1p(np.abs(out[i]))
    return out


def tile_features(features: np.ndarray, names: list[str], no_jet: bool = False) -> tuple[np.ndarray, list[str]]:
    """Признаки куска в мм (README, «Масштаб входа L0») к входу сети, `(C, N)`: координаты — от центра
    куска на L0, кривизна — на L0 (‖∇II‖ — на L0²), без струи — только координаты и нормаль."""
    L0 = CFG.tile.scale_mm
    out = features.copy()
    out[:3] = (out[:3] - out[:3].mean(axis=1, keepdims=True)) / L0
    for i, name in enumerate(names):
        if name.startswith(CURVATURE_PREFIXES):
            out[i] *= L0 ** (2 if name.startswith("dII@") else 1)
    if no_jet:
        keep = [i for i, n in enumerate(names) if not n.startswith(CURVATURE_PREFIXES)]
        out, names = out[keep], [names[i] for i in keep]
    return out, names


def input_names(names: list[str], no_jet: bool) -> list[str]:
    """Каналы на входе сети: без струи — без кривизны."""
    return [n for n in names if not (no_jet and n.startswith(CURVATURE_PREFIXES))]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Равномерно случайный поворот: нормированный гауссов кватернион даёт
    равномерное распределение на SO(3)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotate_features(features: np.ndarray, names: list[str], rot: np.ndarray) -> np.ndarray:
    """Повернуть деталь в пространстве признаков, `(C, N)`.

    Пересчитывать геометрию не нужно: кривизны от поворота не зависят, поворот
    затрагивает только координаты и нормаль. [измерено: поворот меша и пересчёт
    признаков дают расхождение кривизн с медианой 1e-5 и p99 в 0.002 после
    логарифма при размахе канала 1.7-5.0; заметно расходятся 0.3-0.7%
    треугольников на рёбрах, где подгонка струи плохо обусловлена]
    """
    out = features.copy()
    turn = rot.astype(features.dtype)
    for group in (("cx", "cy", "cz"), ("nx", "ny", "nz")):
        idx = [names.index(n) for n in group]
        out[idx] = turn @ features[idx]
    return out


def rings_signature() -> str:
    """Параметры, от которых зависят кольца. Кэш, собранный при других, не читается."""
    m = CFG.model
    return (f"rings={list(map(list, m.rings))};k={m.neighbors};"
            f"cap={m.frontier_cap};per_ring={m.frontier_per_ring}")


def load_rings(part_dir: Path) -> list[np.ndarray] | None:
    """Кольца детали по слоям из кэша или None, если их нет или они собраны при
    другом конфиге. Слои хранятся раздельно: число соседей у них может
    различаться."""
    path = part_dir / RINGS_FILE
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["signature"]) != rings_signature():
            return None
        if "rings" in z:                      # прежний формат: один массив на все слои
            return [r.astype(np.int32) for r in z["rings"]]
        return [z[f"l{i}"].astype(np.int32) for i in range(len(CFG.model.rings))]


def ring_indices(item: dict, device, generator: torch.Generator | None = None) -> list[Tensor]:
    """Соседи по кольцам на каждый слой: из кэша, если он есть, иначе обходом на GPU."""
    if item.get("rings") is not None:
        return [torch.from_numpy(r).to(device).long() for r in item["rings"]]
    neigh = G.face_neighbors(item["adjacency"], item["n_faces"], device)
    return G.ring_neighbors(neigh, CFG.model.rings, CFG.model.neighbors,
                            CFG.model.frontier_cap, CFG.model.frontier_per_ring, generator)


def face_types(uid: str, n_labels: int) -> np.ndarray:
    """Тип каждой слитой грани как индекс в `PRIMITIVE_TYPES`, неизвестный -1.

    Метки в face_id — номера ПОСЛЕ слияния швов, а `faces` в brep.json — исходные
    грани; перевод через merge_map. Раньше тип брался по исходному номеру, и на
    318 деталях корпуса из 407 со слиянием часть меток получала тип чужой грани
    (32.9% треугольников этих деталей). Сливаются только грани одного типа,
    поэтому тип слитой грани однозначен."""
    brep = json.loads((PARTS / uid / "brep.json").read_text())
    merge_map = brep["merge_map"] or list(range(brep["n_faces"]))
    types = np.full(n_labels, -1, np.int64)
    for face in brep["faces"]:
        label = merge_map[face["id"]]
        if label < n_labels:
            types[label] = TYPE_INDEX.get(_shape_type(face), -1)
    return types


def _shape_type(face: dict) -> str:
    """Тип грани по форме: конус с полууглом ближе `CONE_LIMIT_DEG` к 0° —
    цилиндр, к 90° — плоскость. В STEP так записаны профили вращения с
    наклоном в десятые доли градуса [24474_2e83b11c_0: 13 «конусов» из 15], и
    тип «конус» на грани-цилиндре учил голову типа противоречию."""
    if face["type"] != "cone" or "semi_angle" not in face.get("params", {}):
        return face["type"]
    angle = abs(np.degrees(face["params"]["semi_angle"]))
    if angle < CONE_LIMIT_DEG:
        return "cylinder"
    if angle > 90 - CONE_LIMIT_DEG:
        return "plane"
    return "cone"


# Метки операций Fusion 360 Gallery на исходную грань детали. Порядок граней тот же, что
# в brep.json: число граней сошлось на всех 1182 деталях val, а грани Fillet в 97% случаев
# касаются двух соседей (README, «Скругления и фаски — соседним граням»).
# dataset/data/seg — breps/seg архива s2.0.1 (на PC-108-7 ссылка, на PC-108-9 копия).
SEG_DIR = DATA / "seg"
SEG_NAMES = ("ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd")


def blend_faces(uid: str, labels: np.ndarray, adjacency: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                max_mm: float) -> np.ndarray:
    """Слитые грани, которые отдаются соседям: скругление (метка Fillet) радиусом до `max_mm`
    — у сплайнового, без радиуса, ширина до π/2·`max_mm`, дуга 90° — или фаска (Chamfer)
    шириной до `max_mm`. Ширина — по рабочему мешу: полуширина в кольцах на медианное ребро,
    вдвое. Слитая грань отдаётся, только если все её исходные грани такие."""
    brep = json.loads((PARTS / uid / "brep.json").read_text())
    merge_map = brep["merge_map"] or list(range(brep["n_faces"]))
    seg = np.loadtxt(SEG_DIR / f"{uid}.seg", dtype=int).reshape(-1)
    n = int(labels.max()) + 1
    edge = float(np.median(np.linalg.norm(vertices[faces] - vertices[np.roll(faces, 1, axis=1)], axis=2)))
    width = 2 * face_halfwidth(adjacency, labels) * edge
    blend, other = np.zeros(n, bool), np.zeros(n, bool)
    for face in brep["faces"]:
        label = merge_map[face["id"]]
        if label >= n:
            continue
        kind = SEG_NAMES[seg[face["id"]]]
        params = face["params"] or {}
        radius = params.get("minor_radius", params.get("radius"))
        if kind == "Fillet":
            small = radius <= max_mm if radius is not None else width[label] <= np.pi / 2 * max_mm
        else:
            small = kind == "Chamfer" and width[label] <= max_mm
        (blend if small else other)[label] = True
    return blend & ~other


def absorb_blends(labels: np.ndarray, adjacency: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Треугольники отданных граней получают метку ближайшей оставшейся: обход в ширину от
    оставшихся граней по соседству, так что полоса делится примерно по середине. Грань без
    оставшихся соседей сохраняет свою метку."""
    out = labels.copy()
    todo = blend[out]
    a, b = adjacency[:, 0], adjacency[:, 1]
    while todo.any():
        into_a, into_b = todo[a] & ~todo[b], todo[b] & ~todo[a]
        if not (into_a.any() or into_b.any()):
            break
        out[a[into_a]] = out[b[into_a]]
        out[b[into_b]] = out[a[into_b]]
        todo[a[into_a]] = todo[b[into_b]] = False
    return out


def narrow_groups(uid: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Правило узких граней из brep.json (README, «Узкие грани: ширина в мм и правило
    разметки»): итоговая группа каждой слитой грани и какие грани отдаются соседям. Без
    `label_map` (детали Fusion) — всё как есть."""
    brep = json.loads((PARTS / uid / "brep.json").read_text())
    if brep.get("label_map") is None:
        return np.arange(n), np.zeros(n, bool)
    lm = np.asarray(brep["label_map"], np.int64)[:n]
    return np.where(lm >= 0, lm, np.arange(n)), lm < 0


def exact_halfwidth(uid: str, remap: np.ndarray, n_final: int) -> np.ndarray:
    """Точная полуширина итоговых граней в мм из brep.json; у группы граней — наибольшая из них.
    NaN, если её нет (детали Fusion, собранные до точной полуширины)."""
    out = np.full(n_final, np.nan)
    hw = json.loads((PARTS / uid / "brep.json").read_text()).get("halfwidth_mm")
    if hw is None:
        return out
    hw = np.asarray(hw, float)[:len(remap)]
    final = remap[:len(hw)]
    ok = (final >= 0) & (hw >= 0)
    np.fmax.at(out, final[ok], hw[ok])
    return out


def relabel(labels: np.ndarray, adjacency: np.ndarray, group: np.ndarray,
            drop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Метки на треугольник после правил разметки: грани одной группы — одна метка, отданные
    делятся между соседями. Вторым — итоговая метка каждой слитой грани, -1 у отданной."""
    n = len(group)
    inter = np.where(drop[labels], n + labels, group[labels])
    kept, out = np.unique(absorb_blends(inter, adjacency, np.r_[np.zeros(n, bool), np.ones(n, bool)]),
                          return_inverse=True)
    target = np.where(drop, n + np.arange(n), group)
    pos = np.searchsorted(kept, target).clip(max=len(kept) - 1)
    return out, np.where(kept[pos] == target, pos, -1)


def rule_labels(uid: str, labels: np.ndarray, adjacency: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                narrow: bool, absorb_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Метки и тип на треугольник после правил разметки (узкие грани, поглощение скруглений) и итоговая
    метка каждой слитой грани. Одна для загрузчика и для вывода целой детали."""
    n = int(labels.max()) + 1
    types = face_types(uid, n)[labels]
    remap = np.arange(n)
    if absorb_mm or narrow:
        adjacency = np.asarray(adjacency, np.int64)
        group, drop = narrow_groups(uid, n) if narrow else (np.arange(n), np.zeros(n, bool))
        if absorb_mm:
            drop |= blend_faces(uid, labels, adjacency, vertices, faces, absorb_mm)
        if drop.any() or (group != np.arange(n)).any():
            labels, remap = relabel(labels, adjacency, group, drop)
            # Тип группы — тип большинства её треугольников: ступенька бывает из граней разных типов.
            t = len(PRIMITIVE_TYPES) + 1
            votes = np.bincount(labels * t + types + 1, minlength=(labels.max() + 1) * t).reshape(-1, t)
            types = (votes.argmax(1) - 1)[labels]
    return labels, types, remap


class CachedParts(Dataset):
    """Кэш как датасет. `uids` задаётся явно, чтобы отладочная подвыборка не
    требовала отдельного класса."""

    def __init__(self, uids: list[str], variant: str = "noisy", rotate: bool = False, absorb_mm: float = 0.0,
                 narrow: bool = False, no_jet: bool = False, random_tile: bool = False):
        self.uids = list(uids)
        self.variant = variant
        self.rotate = rotate          # аугментация поворотами — только на обучении
        self.absorb_mm = absorb_mm    # скругления и фаски до стольких мм — соседним граням (0 — нет)
        self.narrow = narrow          # правило узких граней из brep.json
        self.no_jet = no_jet          # без каналов кривизны, даже если они в кэше
        self.random_tile = random_tile  # кэш кусков: случайный кусок детали, иначе t0 (самый богатый гранями)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
        tile = None
        if CFG.tile:
            # Один кусок детали за выдачу (README, «Фиксированное ребро и куски»): на обучении — случайный,
            # семя из генератора torch, как у поворота ниже; на валидации — t0.
            tile = 0
            if self.random_tile:
                tile = int(torch.randint(0, len(part_meta(uid, self.variant)["tiles"]), ()).item())
        part = load_part(uid, self.variant, tile)
        work = part["work"]
        labels, types, remap = rule_labels(uid, work["labels"].astype(np.int64), work["adjacency"],
                                           work["vertices"], work["faces"], self.narrow, self.absorb_mm)
        features = work["features"].T.copy()           # (C, N), как ждёт свёртка
        names = part["meta"]["feature_names"]
        if CFG.tile:
            features, names = tile_features(features, names, self.no_jet)
        elif self.no_jet:
            keep = [i for i, n in enumerate(names) if not n.startswith(CURVATURE_PREFIXES)]
            features, names = features[keep], [names[i] for i in keep]
        if CFG.model.log_curvature:
            features = compress_curvature(features, names)
        if self.rotate:
            # Семя из генератора torch этого процесса: он сдвигается с каждой
            # выдачей, поэтому поворот новый на каждой эпохе и воспроизводим по
            # seed прогона. initial_seed() для этого не годится — у постоянных
            # воркеров он один на весь прогон, и поворот детали повторялся.
            seed = int(torch.randint(0, 2 ** 62, ()).item())
            features = rotate_features(features, names, random_rotation(np.random.default_rng(seed)))
        return {
            "uid": uid,
            "features": features,
            "rings": load_rings(part_dir(uid, self.variant) / (f"t{tile}" if tile is not None else "")),
            "labels": labels,
            "remap": remap,
            "halfwidth_mm": exact_halfwidth(uid, remap, int(labels.max()) + 1),
            "types": types,
            "ambiguous": work["ambiguous"].astype(bool),
            # кайма куска без лосса (README, «Ядро 60 тысяч»); у прежних кэшей каймы нет
            "core": work["core"].astype(bool) if "core" in work else np.ones(len(work["faces"]), bool),
            "adjacency": work["adjacency"],
            "n_faces": len(work["faces"]),
        }


def collate_single(batch: list[dict]) -> dict:
    """Батч из одной детали. DataLoader нужен ради воркеров, а не ради батча:
    чтение и распаковка npz идут параллельно шагу на GPU."""
    (item,) = batch
    return item
