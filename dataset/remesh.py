"""Равномерный ремеш детали под фиксированное число треугольников и перенос
разметки на них.

Первый этап конвейера: дальше всё работает только с ремешем, исходная
тесселяция не используется. Зачем он нужен — см. README, «Конвейер данных».
"""
from __future__ import annotations

import numpy as np
import pymeshlab
import trimesh

from src.config import CFG

_C = CFG.remesh

# Значения и их обоснование — в configs/default.yaml, секция remesh.
TARGET_FACES = _C.target_faces
SUBSTRATE_EDGE_REL = _C.substrate_edge_rel
SUBSTRATE_MIN_RATIO = _C.substrate_min_ratio
FEATURE_DEG = _C.feature_deg
# Сколько центроидов за раз переносить разметку: память ограничена порцией, а не мешем.
LABEL_CHUNK = 8192
# Меньше порцию не дробить: если не хватает и её, деталь действительно не по силам.
MIN_CHUNK = 64
FACE_TOL = _C.face_tol
# Порог формы треугольника в QEM pymeshlab: схлопывания, дающие треугольник
# хуже порога, штрафуются. [по умолчанию 0.3; 0.5 взято без подбора]
QUALITY_THR = 0.5


def target_edge(area: float, faces: int) -> float:
    """Сторона равностороннего треугольника, при которой их выйдет `faces`."""
    return float(np.sqrt(4.0 * area / (np.sqrt(3.0) * faces)))


def _run(mesh: trimesh.Trimesh, edge: float, iterations: int) -> pymeshlab.MeshSet:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, float),
                               np.asarray(mesh.faces, np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(edge), iterations=iterations,
        featuredeg=FEATURE_DEG, checksurfdist=False)
    return ms


def _to_trimesh(ms: pymeshlab.MeshSet) -> trimesh.Trimesh:
    """Ремешер оставляет в векторах удалённые элементы; без уплотнения
    vertex_matrix() бросает MissingCompactnessException."""
    m = ms.current_mesh()
    if not m.is_compact():
        m.compact()
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)


def substrate_edge(mesh: trimesh.Trimesh, edge_rel: float = SUBSTRATE_EDGE_REL,
                   target_faces: int = TARGET_FACES, min_ratio: float = SUBSTRATE_MIN_RATIO) -> float:
    """Шаг подложки: доля габарита, но не крупнее, чем нужно для `min_ratio` целей
    треугольников. При большой доле габарита работает только второе — подложка с
    постоянным числом треугольников, и тогда число точек в шаре струи одинаково у всех
    деталей: радиус и шаг оба пропорциональны √площади."""
    return min(float(np.linalg.norm(mesh.extents)) * edge_rel,
               target_edge(float(mesh.area), int(target_faces * min_ratio)))


def presplit(mesh: trimesh.Trimesh, edge: float) -> trimesh.Trimesh:
    """Тесселяция, подразбитая до рёбер не длиннее двух шагов подложки.

    На длинных тонких треугольниках тесселяции CAD ремешер за свои итерации не успевает
    прийти к ровной сетке: у 18% деталей корпуса подложка выходила с больше чем
    процентом углов-игл, и шаг недобирался. После подразбиения при тех же 6 итерациях
    игл 0-0.03% на 12 деталях из 12 (README, «Подложка из CAD»). Геометрия не
    меняется — фасетки только делятся."""
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=2 * edge)
    return trimesh.Trimesh(v, f, process=False)


def substrate(mesh: trimesh.Trimesh, edge_rel: float = SUBSTRATE_EDGE_REL,
              iterations: int = _C.iterations, target_faces: int = TARGET_FACES,
              min_ratio: float = SUBSTRATE_MIN_RATIO,
              passes: int = _C.substrate_passes) -> trimesh.Trimesh:
    """Мелкая подложка под наложение шума и под окрестности струи.

    Шаг задаётся долей габарита: он должен быть достаточно мелким в АБСОЛЮТНЫХ
    единицах, чтобы на нём была представима корреляция шума. Но у детали с
    малой площадью такой шаг даёт мало треугольников, и рабочему мешу неоткуда
    взяться — `decimate` умеет только упрощать. Поэтому число граней
    дополнительно поднимается до `min_ratio` от цели.

    Проверять фактический результат обязательно: ремешер принимает длину ребра,
    а не число граней, и систематически недодаёт — при расчётных 90 тысячах
    один проход давал 56.
    """
    need = int(target_faces * min_ratio)
    edge = substrate_edge(mesh, edge_rel, target_faces, min_ratio)
    ms = _run(mesh, edge, iterations)
    for _ in range(passes - 1):
        got = ms.current_mesh().face_number()
        if got >= need:
            break
        edge *= np.sqrt(got / need)
        ms = _run(mesh, edge, iterations)
    return _to_trimesh(ms)


def substrate_once(mesh: trimesh.Trimesh, edge: float, iterations: int = _C.iterations) -> trimesh.Trimesh:
    """Подложка при фиксированном ребре: один проход ремеша с заданным ребром. Подгонка под число
    треугольников (`substrate`) повторяла проход ради 1-3% недобора — втрое дольше при том же
    ребре (README, «Подложка одним проходом»)."""
    return _to_trimesh(_run(mesh, edge, iterations))


def remesh(mesh: trimesh.Trimesh, target_faces: int = TARGET_FACES,
           iterations: int = _C.iterations, passes: int = _C.passes,
           tol: float = FACE_TOL) -> trimesh.Trimesh:
    """Изотропный ремеш до заданного числа треугольников, острые рёбра
    сохраняются.

    Ремешер принимает длину ребра, а не число граней, и попадает в цель
    неточно. Поэтому длина пересчитывается по фактическому результату и
    прогон повторяется, пока промах больше `tol`.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, float),
                               np.asarray(mesh.faces, np.int32)))
    edge = target_edge(float(mesh.area), target_faces)
    for _ in range(passes):
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(edge),
            iterations=iterations,
            featuredeg=FEATURE_DEG,
            checksurfdist=False,
        )
        got = ms.current_mesh().face_number()
        if abs(got - target_faces) <= tol * target_faces:
            break
        # Число граней обратно пропорционально квадрату длины ребра.
        edge *= np.sqrt(got / target_faces)
    return _to_trimesh(ms)


def decimate(mesh: trimesh.Trimesh,
             target_faces: int = TARGET_FACES) -> trimesh.Trimesh:
    """Рабочий меш упрощением подложки: плотность идёт за кривизной.

    Квадратичная ошибка схлопывает плоские участки дёшево, а кривые дорого,
    поэтому адаптация получается по построению, без явного расчёта кривизны.
    Считать по этому мешу струю нельзя — она берётся с подложки, которая
    остаётся равномерной; неравномерность здесь влияет только на то, где сеть
    получает узлы.

    [измерено: 12 деталей, бюджет 10 тыс. треугольников]
                          мин. треуг. в грани   граней <20тр   разброс площади
      равномерный ремеш            20                2%             2.1x
      упрощение подложки           38                0%             5.1x

    Цена — вытянутые треугольники (форма 4.57 против 4.05 при 3.46 у
    равностороннего) и плотность, коррелирующая с кривизной.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, float),
                               np.asarray(mesh.faces, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces, preserveboundary=True, preservenormal=True,
        planarquadric=True, qualitythr=QUALITY_THR)
    return _to_trimesh(ms)


def transfer_labels(source: trimesh.Trimesh, face_id: np.ndarray,
                    target: trimesh.Trimesh) -> np.ndarray:
    """Метка грани B-Rep для каждого треугольника ремеша.

    Каждый новый треугольник получает метку той исходной грани, к которой
    ближе всего его центроид — по настоящей ближайшей точке на исходной
    поверхности, а не по центроиду исходного треугольника. Разница
    существенна у стыков: центроидная версия промахивается на полтреугольника,
    а там как раз и проходят границы граней.
    """
    return np.asarray(face_id)[closest_faces(source, target.triangles.mean(axis=1))]


def closest_faces(mesh: trimesh.Trimesh, points: np.ndarray, chunk: int = LABEL_CHUNK) -> np.ndarray:
    """Треугольник `mesh` с ближайшей к каждой точке точкой поверхности.

    Порциями: `closest_point` строит на точку список треугольников-кандидатов, и на
    части деталей эти списки взрываются. Одним вызовом на 120 тысяч центроидов сборщик
    занимал 64 ГБ и вешал машину (README, «Опыт „вчетверо подробнее“»); у тесселяции с
    огромными треугольниками не хватало и порции в 8192 — тогда порция дробится. Ответ
    для точки от порции не зависит.
    """
    out, s = [], 0
    while s < len(points):
        try:
            out.append(trimesh.proximity.closest_point(mesh, points[s:s + chunk])[2])
            s += chunk
        except MemoryError:
            if chunk <= MIN_CHUNK:
                raise
            chunk = max(MIN_CHUNK, chunk // 8)
    return np.concatenate(out)


def ambiguous(target: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    """Треугольники, у которых хотя бы один сосед по ребру из другой грани.

    Полоса шириной в один треугольник вдоль всех границ. Именно там перенос
    меток наименее надёжен: истинная граница B-Rep после ремеша не лежит на
    рёбрах меша, и попасть в неё точнее половины ребра невозможно.
    """
    adj = target.face_adjacency
    diff = labels[adj[:, 0]] != labels[adj[:, 1]]
    out = np.zeros(len(labels), bool)
    out[adj[diff].ravel()] = True
    return out
