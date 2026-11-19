"""Подложка прямо из STEP: сетка на поверхности CAD через gmsh, без ремеша тесселяции.

Ремеш тесселяции оставлял у 18% деталей корпуса больше процента углов-игл (больше
169° и меньше 1°) и не доходил до заданного шага, а точки подложки лежали на
фасетках, а не на поверхности. Сетка gmsh строится по каждой грани CAD с заданным
шагом фронтальным Делоне: иглы — 0% на 99 деталях из 100, грани сшиты по общим
рёбрам, точки на настоящей поверхности (README, «Подложка из CAD»).

gmsh запускается в дочернем процессе: его состояние глобальное, на кривом STEP он
может зависнуть или упасть внутри C++, и сборщик кэша не должен от этого вставать.

    python -m dataset.step_substrate <файл.stp> <шаг, мм> <выход.npz>
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from .remesh import closest_faces

# Фронтальный Делоне в gmsh (Mesh.Algorithm = 6): ровные треугольники заданного размера.
FRONTAL_DELAUNAY = 6
# Сколько ждать одну деталь, с: медиана 12 с, p90 23 с при 12 сборщиках [проба, 100 деталей].
TIMEOUT_S = 600
# Во сколько раз сетка может выйти больше заказанной, прежде чем считаться сбоем: на одной
# детали шаг по площади тесселяции вышел крохотным, и gmsh начал строить 31 млн узлов.
MAX_OVERSHOOT = 2.0


class StepMeshError(RuntimeError):
    """gmsh не построил сетку: упал, завис или выдал непригодную."""


def mesh_step(step: Path, size: float, faces: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вершины, треугольники сетки на поверхности STEP с шагом `size`, мм, и номер грани STEP на
    треугольник (порядок поверхностей gmsh совпал с порядком граней у 99 деталей из 99).
    Вызывается в дочернем процессе (`main`).

    `faces` — сколько треугольников заказано: шаг не мельче, чем нужно для них по
    площади самого STEP. Тесселяция бывает неполной, и шаг, посчитанный по её
    площади, выходил в разы мельче."""
    import gmsh

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        if faces:
            area = sum(gmsh.model.occ.getMass(2, t) for _, t in gmsh.model.getEntities(2))
            size = max(size, float(np.sqrt(4.0 * area / (np.sqrt(3.0) * faces))))
        for name in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax"):
            gmsh.option.setNumber(name, size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Algorithm", FRONTAL_DELAUNAY)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.full(int(tags.max()) + 1, -1, np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        faces, owner = [], []
        for k, (_, surface) in enumerate(gmsh.model.getEntities(2)):
            types, _, nodes = gmsh.model.mesh.getElements(2, surface)
            for t, n in zip(types, nodes):
                if t == 2:
                    faces.append(index[n.astype(np.int64)].reshape(-1, 3))
                    owner.append(np.full(len(faces[-1]), k, np.int32))
        return coords.reshape(-1, 3), np.concatenate(faces), np.concatenate(owner)
    finally:
        gmsh.finalize()


def substrate(step: Path, size: float, reference: trimesh.Trimesh, faces: int | None = None,
              timeout: float = TIMEOUT_S, with_faces: bool = False):
    """Подложка из STEP с шагом `size`, мм, ориентированная наружу как `reference`.

    Знак кривизны у нас несёт вогнутость, поэтому ориентация проверяется, а не
    принимается на веру: треугольники согласуются между собой, каждая связная часть
    разворачивается наружу, и итог сверяется с исходной тесселяцией (`reference`, у неё
    нормали наружу). На пробе gmsh сам отдавал согласованную ориентацию наружу.
    `with_faces` — ещё номер грани STEP на треугольник.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mesh.npz"
        try:
            run = subprocess.run([sys.executable, "-m", "dataset.step_substrate", str(step), repr(size), str(out),
                                  str(faces or 0)],
                                 cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
                                 timeout=timeout)
        except subprocess.TimeoutExpired as ex:
            raise StepMeshError(f"gmsh не уложился в {timeout:.0f} с") from ex
        if run.returncode or not out.exists():
            raise StepMeshError(f"gmsh упал: {(run.stderr or run.stdout).strip()[-300:]}")
        z = np.load(out)
        if faces and len(z["faces"]) > MAX_OVERSHOOT * faces:
            raise StepMeshError(f"сетка {len(z['faces'])} треугольников при заказанных {faces}")
        mesh = trimesh.Trimesh(z["vertices"], z["faces"], process=False)
        owner = z["owner"]
    mesh.merge_vertices()
    keep = mesh.nondegenerate_faces()
    mesh.update_faces(keep)
    owner = owner[keep]
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    faces = np.asarray(mesh.faces).copy()
    parts = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)))
    for part in parts:
        piece = trimesh.Trimesh(mesh.vertices, faces[part], process=False)
        # Замкнутая часть — наружу по знаку объёма; незамкнутая — по ближайшей точке тесселяции.
        inward = piece.volume < 0 if piece.is_watertight else _disagree(piece, reference) > 0.5
        if inward:
            faces[part] = faces[part][:, ::-1]
    mesh = trimesh.Trimesh(mesh.vertices, faces, process=False)
    wrong = _disagree(mesh, reference)
    if wrong > 0.01:
        raise StepMeshError(f"ориентация не согласуется с тесселяцией у {wrong:.1%} треугольников")
    return (mesh, owner) if with_faces else mesh


def _disagree(mesh: trimesh.Trimesh, reference: trimesh.Trimesh, sample: int = 20000) -> float:
    """Доля треугольников, чья нормаль смотрит против нормали тесселяции в ближайшей точке
    её поверхности; по выборке. Ближайший центроид не годится: у ребра он часто на соседней
    грани крупной тесселяции, и на детали с верной ориентацией давал 12% «против»."""
    idx = np.random.default_rng(0).choice(len(mesh.faces), min(sample, len(mesh.faces)), replace=False)
    tri = closest_faces(reference, mesh.triangles_center[idx])
    return float(np.mean((mesh.face_normals[idx] * reference.face_normals[tri]).sum(1) < 0))


def main() -> None:
    step, size, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    vertices, faces, owner = mesh_step(step, size, int(sys.argv[4]) or None)
    np.savez(out, vertices=vertices, faces=faces, owner=owner)


if __name__ == "__main__":
    main()
