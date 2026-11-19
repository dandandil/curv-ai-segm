"""Виртуальный сканер: CAD -> меш, как его отдаёт софт сканера (README, «Виртуальный сканер»).

Вместо шума, наложенного на подложку, шум возникает там же, где у настоящего
сканера: деталь снимается картами глубины с ракурсов, у каждого кадра — шум
глубины вдоль луча (сильнее на скользящих углах), размытие карты (скругляет рёбра),
периодическая рябь от проекции полос и ошибка сведения кадров; кадры сливаются в
поверхность объёмным слиянием (TSDF) или Poisson-реконструкцией, софт её сглаживает.
Поверх — то, чего у слияния нет: погрешность изготовления, заливка вогнутых углов и
выбросы. Сканеры будут разные (решено пользователем 2026-10-10), поэтому каждый параметр
у каждой детали свой, из пределов `RANGES`.

Дыр нет: места, которые общие ракурсы не видят, досматриваются ракурсами вдоль их
нормалей, а что не увидено и так — заклеивается. Остаётся одно тело.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.spatial import cKDTree

from dataset import noise as N
from dataset import remesh as R
from src.config import CFG


@dataclass(frozen=True)
class ScanParams:
    views: int                  # общих ракурсов
    pitch_mm: float             # шаг пикселя на детали
    depth_sigma_mm: float       # шум глубины вдоль луча при нормальном падении
    grazing_power: float        # шум растёт как 1/cos^k угла падения
    blur_px: float              # размытие карты глубины, пикселей
    ripple_mm: float            # амплитуда ряби от проекции полос
    ripple_period_mm: float
    reg_shift_mm: float         # ошибка сведения кадра: сдвиг
    reg_rot_deg: float          # и поворот
    fusion: str                 # "tsdf" или "poisson"
    voxel_pitches: float        # TSDF: вокселей на шаг пикселя; Poisson: шаг сетки в шагах пикселя
    trunc_voxels: float         # TSDF: усечение, вокселей
    smooth_iters: int           # сглаживание Лапласа софтом после слияния, обязательное
    wave_sigma_mm: float        # плавная деформация изготовленной детали относительно CAD
    wave_corr_mm: float
    fill_mm: float              # заливка вогнутых углов: подъём поверхности у ребра
    fill_width_mm: float        # и на каком расстоянии от ребра он сходит на нет
    spikes_per_cm2: float       # выбросы: одиночные вершины, вытянутые вдоль нормали
    spike_mm: float             # их наибольшая длина


# Пределы, из которых у каждой детали берутся параметры. Выбрано с запасом в обе стороны от
# сканов пользователя (README, «Грязный шум сканера»: размах на плоской грани 5-79 мкм,
# рябь ±1-4 мкм с периодом ~1 мм, скругление рёбер 0.076 мм по биссектрисе).
RANGES = {
    "views": (16, 48),
    "pitch_mm": (0.03, 0.09),
    "depth_sigma_mm": (0.003, 0.015),
    "grazing_power": (0.5, 1.5),
    "blur_px": (0.0, 1.5),
    "ripple_mm": (0.0, 0.005),
    "ripple_period_mm": (0.5, 2.0),
    "reg_shift_mm": (0.0, 0.008),
    "reg_rot_deg": (0.0, 0.008),
    "voxel_pitches": (0.6, 1.0),
    "trunc_voxels": (3.0, 6.0),
    "smooth_iters": (4, 10),
    "wave_sigma_mm": (0.005, 0.03),
    "wave_corr_mm": (1.5, 6.0),
    # на сетке шага подложки слияние само заливает углы на ~35 мкм у ребра, как реальный скан;
    # сверху 21 мкм дало +57, 30 — +68 (cad_1)
    "fill_mm": (0.0, 0.02),
    "fill_width_mm": (0.1, 0.4),
    "spikes_per_cm2": (0.0, 2.0),
    "spike_mm": (0.02, 0.2),
}
# Poisson у производителей нигде не заявлен, основной путь — объёмное слияние (пользователь, 2026-10-11)
TSDF_SHARE = 0.9
# Иногда софт сглаживает сильно, теряя детали ~0.1 мм, как Smooth Fusion у Artec (пользователь, 2026-10-11)
STRONG_SMOOTH_SHARE = 0.2
STRONG_SMOOTH_ITERS = (15, 40)
# Пикселей кадра по стороне не больше: на детали в 200 мм шаг 0.1 мм дал бы 4 млн лучей на
# кадр, а подложка всё равно ~500 тысяч треугольников и мельче не различит.
MAX_PIXELS = 1200
COS_GRAZING_MIN = 0.15        # круче этого угла сканер точку не берёт
# Скачок глубины между соседними пикселями больше стольких шагов — край заслонения: такие
# пиксели софт сканера выбрасывает (плоскость под предельным углом даёт 6.6 шага).
JUMP_PITCHES = 8.0
HOLE_EDGES = 5000             # дыры до стольких рёбер по краю заклеиваются
SHARP_DEG = 20.0              # рёбра CAD круче этого — острые, их заливает
COVER_SAMPLES = 30000
COVER_ROUNDS = 3
COVER_VIEWS = 8               # досматривающих ракурсов за круг


def sample_params(rng: np.random.Generator) -> ScanParams:
    u = {k: rng.uniform(*v) for k, v in RANGES.items()}
    if rng.random() < STRONG_SMOOTH_SHARE:
        u["smooth_iters"] = rng.uniform(*STRONG_SMOOTH_ITERS)
    return ScanParams(views=int(round(u["views"])), pitch_mm=u["pitch_mm"], depth_sigma_mm=u["depth_sigma_mm"],
                      grazing_power=u["grazing_power"], blur_px=u["blur_px"], ripple_mm=u["ripple_mm"],
                      ripple_period_mm=u["ripple_period_mm"], reg_shift_mm=u["reg_shift_mm"],
                      reg_rot_deg=u["reg_rot_deg"], fusion="tsdf" if rng.random() < TSDF_SHARE else "poisson",
                      voxel_pitches=u["voxel_pitches"], trunc_voxels=u["trunc_voxels"],
                      smooth_iters=int(round(u["smooth_iters"])), wave_sigma_mm=u["wave_sigma_mm"],
                      wave_corr_mm=u["wave_corr_mm"], fill_mm=u["fill_mm"], fill_width_mm=u["fill_width_mm"],
                      spikes_per_cm2=u["spikes_per_cm2"], spike_mm=u["spike_mm"])


@dataclass
class View:
    eye: np.ndarray
    rot: np.ndarray             # мир -> камера, строки — оси камеры x, y, z (z — вперёд)
    fx: float
    width: int
    pitch: float                # мм на пиксель у цели


def _look_at(eye: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = target - eye
    z /= np.linalg.norm(z)
    up = rng.normal(size=3)
    x = np.cross(up - (up @ z) * z, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def _fibonacci(n: int, rng: np.random.Generator) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    d = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))         # случайный поворот сетки ракурсов
    return d @ q


def _view(eye: np.ndarray, target: np.ndarray, span: float, pitch: float, rng) -> View:
    width = int(min(MAX_PIXELS, max(64, np.ceil(span / pitch))))
    if CFG.noise.scan_max_pitch_mm:
        # при фиксированном ребре кадр крупной детали не огрубляется: шаг не больше потолка
        width = max(width, int(np.ceil(span / CFG.noise.scan_max_pitch_mm)))
    dist = np.linalg.norm(target - eye)
    return View(eye, _look_at(eye, target, rng), fx=width / 2 * dist / (span / 2), width=width, pitch=span / width)


def _rays(view: View) -> np.ndarray:
    w = view.width
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(w) + 0.5)
    cam = np.stack([(u - w / 2) / view.fx, (v - w / 2) / view.fx, np.ones_like(u)], -1).reshape(-1, 3)
    d = cam / np.linalg.norm(cam, axis=1, keepdims=True)
    return np.c_[np.broadcast_to(view.eye, d.shape), d @ view.rot].astype(np.float32)


def _cast(scene, view: View):
    hit = scene.cast_rays(o3d.core.Tensor(_rays(view)))
    t = hit["t_hit"].numpy().reshape(view.width, view.width)
    n = hit["primitive_normals"].numpy().reshape(view.width, view.width, 3)
    return t, n


def _cover_views(mesh: trimesh.Trimesh, scene, views: list[View], p: ScanParams, rng) -> list[View]:
    """Досмотр мест, которых общие ракурсы не видят: ракурс вдоль нормали непокрытой точки."""
    pts, idx = trimesh.sample.sample_surface(mesh, COVER_SAMPLES, seed=int(rng.integers(2 ** 31)))
    nrm = mesh.face_normals[idx]
    span = float(np.linalg.norm(mesh.extents))
    hits = []
    for v in views:
        t, n = _cast(scene, v)
        d = _rays(v)[:, 3:].reshape(v.width, v.width, 3)
        ok = np.isfinite(t) & (np.abs((n * d).sum(-1)) > COS_GRAZING_MIN)
        hits.append(v.eye + d[ok] * t[ok, None])
    added = []
    for _ in range(COVER_ROUNDS):
        tree = cKDTree(np.concatenate(hits))
        far = tree.query(pts, distance_upper_bound=2 * p.pitch_mm)[0] > 2 * p.pitch_mm
        if far.mean() < 0.002:
            break
        miss = np.flatnonzero(far)
        pick = miss[rng.choice(len(miss), min(COVER_VIEWS, len(miss)), replace=False)]
        for i in pick:
            v = _view(pts[i] + nrm[i] * span / 2, pts[i], span / 3, p.pitch_mm, rng)
            t, n = _cast(scene, v)
            d = _rays(v)[:, 3:].reshape(v.width, v.width, 3)
            ok = np.isfinite(t) & (np.abs((n * d).sum(-1)) > COS_GRAZING_MIN)
            hits.append(v.eye + d[ok] * t[ok, None])
            added.append(v)
    return added


def _frame(scene, view: View, p: ScanParams, rng):
    """Кадр сканера: глубина по оси камеры с шумом, NaN — нет точки."""
    t, n = _cast(scene, view)
    d = _rays(view)[:, 3:].reshape(view.width, view.width, 3)
    cos = np.abs((n * d).sum(-1))
    ok = np.isfinite(t) & (cos > COS_GRAZING_MIN)
    t = np.where(ok, t, np.nan)
    sigma = p.depth_sigma_mm / np.clip(cos, COS_GRAZING_MIN, 1) ** p.grazing_power
    t = t + rng.normal(size=t.shape) * sigma
    if p.ripple_mm > 0:
        # полосы проектора неподвижны относительно камеры: рябь по координате кадра
        ang, phase = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        pix = np.mgrid[:view.width, :view.width]
        mm = (np.cos(ang) * pix[1] + np.sin(ang) * pix[0]) * view.pitch
        t = t + p.ripple_mm * np.sin(2 * np.pi * mm / p.ripple_period_mm + phase)
    jump = np.zeros_like(ok)
    for ax in (0, 1):
        big = np.abs(np.diff(t, axis=ax)) > JUMP_PITCHES * view.pitch
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        jump[tuple(lo)] |= big
        jump[tuple(hi)] |= big
    if jump.any():
        # без этого размытие смешивает ближнюю и дальнюю поверхность в «юбку» у ступеньки
        jump = binary_dilation(jump, iterations=1 + int(np.ceil(2 * p.blur_px)))
        t = np.where(jump, np.nan, t)
    if p.blur_px > 0:
        # нормированная свёртка по точкам кадра: фон в среднее не попадает
        w = gaussian_filter(np.isfinite(t).astype(float), p.blur_px)
        s = gaussian_filter(np.nan_to_num(t), p.blur_px)
        t = np.where(np.isfinite(t), s / np.maximum(w, 1e-9), np.nan)
    return t * (d @ view.rot[2])          # глубина по оси камеры, как в карте глубины


def _misregistered(view: View, p: ScanParams, centre: np.ndarray, rng) -> np.ndarray:
    """Мир -> камера с ошибкой сведения: кадр кладётся чуть сдвинутым и повёрнутым."""
    axis = rng.normal(size=3)
    rot = trimesh.transformations.rotation_matrix(np.radians(rng.normal() * p.reg_rot_deg), axis, centre)
    rot[:3, 3] += rng.normal(size=3) * p.reg_shift_mm
    ext = np.eye(4)
    ext[:3, :3] = view.rot
    ext[:3, 3] = -view.rot @ view.eye
    return ext @ np.linalg.inv(rot)


def _concave_edges(mesh: trimesh.Trimesh, step: float) -> np.ndarray:
    """Точки острых вогнутых рёбер CAD с шагом step."""
    sharp = (mesh.face_adjacency_angles > np.radians(SHARP_DEG)) & ~mesh.face_adjacency_convex
    e = mesh.vertices[mesh.face_adjacency_edges[sharp]]
    if not len(e):
        return np.empty((0, 3))
    n = np.maximum(2, np.ceil(np.linalg.norm(e[:, 1] - e[:, 0], axis=1) / step).astype(int))
    return np.concatenate([a + np.linspace(0, 1, k)[:, None] * (b - a) for (a, b), k in zip(e, n)])


def _corner_fill(out: trimesh.Trimesh, mesh: trimesh.Trimesh, p: ScanParams) -> None:
    """Заливка вогнутых углов: у реального скана (README, «Виртуальный сканер») поверхность в
    0.4 мм от вогнутого ребра поднята на ~35 мкм почти полкой, у слияния кадров — на 5-15."""
    if p.fill_mm <= 0:
        return
    edges = _concave_edges(mesh, p.fill_width_mm / 5)
    if not len(edges):
        return
    d = cKDTree(edges).query(out.vertices, distance_upper_bound=3 * p.fill_width_mm)[0]
    near = np.isfinite(d)
    lift = p.fill_mm * np.exp(-(d[near] / p.fill_width_mm) ** 4)
    v = out.vertices.copy()
    v[near] += lift[:, None] * out.vertex_normals[near]
    out.vertices = v


def _close_holes(out: trimesh.Trimesh) -> trimesh.Trimesh:
    """Заклейка дыр, как в софте сканера. TSDF прорезает острые выпуклые рёбра: обе грани клина
    лежат в пределах усечения, и виды с одной стороны затирают другую — вдоль ребра цепочка
    дырочек (на cad_1 до 638 граничных рёбер, все в 0.05 мм от выпуклого ребра)."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(out.vertices, float), np.asarray(out.faces, np.int32)))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=HOLE_EDGES, selfintersection=False)
    m = ms.current_mesh()
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    return max(out.split(only_watertight=False), key=lambda m: len(m.faces))


def _spikes(out: trimesh.Trimesh, p: ScanParams, rng) -> None:
    """Выбросы, которые софт не отфильтровал: одиночная вершина, вытянутая вдоль нормали в любую сторону."""
    n = rng.poisson(p.spikes_per_cm2 * out.area / 100)
    if not n:
        return
    i = rng.choice(len(out.vertices), n, replace=False)
    v = out.vertices.copy()
    v[i] += (rng.uniform(0.2, 1.0, n) * rng.choice([-1, 1], n) * p.spike_mm)[:, None] * out.vertex_normals[i]
    out.vertices = v


def scan(mesh: trimesh.Trimesh, p: ScanParams, rng: np.random.Generator) -> tuple[trimesh.Trimesh, ScanParams]:
    """CAD-меш -> меш, как после сканера и его софта, и параметры, с которыми он снят на
    деле: шаг пикселя поднимается до шага подложки, и в meta.json пишется фактический."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    centre = mesh.bounds.mean(0)
    span = float(np.linalg.norm(mesh.extents))
    # Мельче шага подложки сливать незачем: ремеш в подложку это всё равно сотрёт, а деталь ABC в
    # 100 мм при шаге 0.05 мм дала 24 млн треугольников, 22 ГБ и 40 минут ЦП.
    # При фиксированном ребре предел задаётся в мм (`scan_min_voxel_mm`): иначе он шёл бы за
    # плотностью подложки, и прореживание подложки огрубляло бы и сам скан.
    floor = CFG.noise.scan_min_voxel_mm or R.substrate_edge(mesh)
    p = replace(p, pitch_mm=max(p.pitch_mm, floor / p.voxel_pitches))
    views = [_view(centre + d * 2 * span, centre, span, p.pitch_mm, rng) for d in _fibonacci(p.views, rng)]
    views += _cover_views(mesh, scene, views, p, rng)

    if p.fusion == "tsdf":
        voxel = p.voxel_pitches * views[0].pitch
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=p.trunc_voxels * voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    points, normals = [], []
    for v in views:
        z = _frame(scene, v, p, rng)
        ext = _misregistered(v, p, centre, rng)
        if p.fusion == "tsdf":
            depth = o3d.geometry.Image(np.nan_to_num(z, nan=0.0).astype(np.float32))
            color = o3d.geometry.Image(np.zeros((v.width, v.width, 3), np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1.0, depth_trunc=1e9, convert_rgb_to_intensity=False)
            # центр пикселя u у Open3D — в целой координате, у наших лучей — в u + 0.5
            intr = o3d.camera.PinholeCameraIntrinsic(v.width, v.width, v.fx, v.fx, v.width / 2 - 0.5, v.width / 2 - 0.5)
            volume.integrate(rgbd, intr, ext)
        else:
            ok = np.isfinite(z)
            vv, uu = np.nonzero(ok)
            cam = np.stack([(uu + 0.5 - v.width / 2) / v.fx * z[ok], (vv + 0.5 - v.width / 2) / v.fx * z[ok], z[ok]], 1)
            inv = np.linalg.inv(ext)
            world = cam @ inv[:3, :3].T + inv[:3, 3]
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
            pc.orient_normals_towards_camera_location(inv[:3, 3])
            points.append(world)
            normals.append(np.asarray(pc.normals))
    if p.fusion == "tsdf":
        out = volume.extract_triangle_mesh()
    else:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
        pc.normals = o3d.utility.Vector3dVector(np.concatenate(normals))
        res = p.voxel_pitches * views[0].pitch
        depth = int(np.clip(np.ceil(np.log2(span / res)), 6, 11))
        # Один поток: по умолчанию Poisson берёт все ядра мимо OMP_NUM_THREADS, и при сборщике на
        # ядро потоки крутятся в ожидании [abc_00404785: 16 с в один поток, больше 4 минут в 24 при
        # 12 сборщиках; на PC-108-9 три такие детали держали загрузку 80 на 24 ядрах]
        out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, n_threads=1)
    out = trimesh.Trimesh(np.asarray(out.vertices), np.asarray(out.triangles), process=True)
    # софт сканера удаляет оторванные клочки; у слияния их до 80 на деталь (Onshape видел 15 тел)
    out = max(out.split(only_watertight=False), key=lambda m: len(m.faces))
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out = _close_holes(out)
    # деталь изготовлена с погрешностью: плавная волна вдоль нормали, мелкой фактуры не добавляет
    out.vertices = N.waviness(out.vertices, out.vertex_normals, 0.0, rng, sigma=p.wave_sigma_mm,
                              corr=p.wave_corr_mm, fourier=True)
    _corner_fill(out, mesh, p)
    # без него извлечение поверхности даёт «джинсу» в шаг сетки (угол нормали 4-8° против 1.3° у
    # реального скана); 5 итераций — 1.8°, рёбра и волнистость не меняются
    # без сохранения объёма: оно растягивает всю деталь, возвращая усадку скруглений
    trimesh.smoothing.filter_laplacian(out, lamb=0.5, iterations=p.smooth_iters, volume_constraint=False)
    _spikes(out, p, rng)
    return out, p
