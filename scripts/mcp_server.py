"""MCP-сервер сегментатора: агент присылает путь к мешу, получает кластеры и их сводку.

    CURV_CONFIG=configs/tile.yaml python scripts/mcp_server.py --checkpoint runs/tile50/best.pt

Слушает только 127.0.0.1 (README, «MCP-сервер сегментатора»): с ноутбука — через туннель
`ssh -N -L 8765:127.0.0.1:8765 PC-108-7`, адрес для клиента http://127.0.0.1:8765/mcp.
Модель грузится один раз; прогоны идут по одному, сеть занимает GPU целиком.
Описания для агента (инструкции, параметры, поля ответов) — по-английски: их читает модель.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import anyio
import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Как у воркера: сервер отдаёт прогон на кусках, конфиг задаётся до импорта src.config.
os.environ.setdefault("CURV_CONFIG", str(ROOT / "configs" / "tile.yaml"))

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from src import describe as D  # noqa: E402
from src import inference as INF  # noqa: E402
from src.config import CFG  # noqa: E402

DEFAULT_PORT = 8765
# Сколько сегментаций держать в памяти: у детали в миллион треугольников это ~100 МБ.
MAX_KEPT = 8

PrimitiveType = Literal["plane", "cylinder", "cone", "sphere", "torus", "bspline"]
assert set(PrimitiveType.__args__) == set(INF.PRIMITIVE_TYPES)
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]

INSTRUCTIONS = f"""\
Segments a triangle mesh of a mechanical part into clusters; each cluster is one surface that a single
primitive can represent ({", ".join(INF.PRIMITIVE_TYPES)}). A neural net predicts boundaries and per-triangle
surface types; clusters are cut along the boundaries.

Workflow:
1. The server runs on host PC-108-7 and reads meshes from ITS file system. Copy a local file first,
   e.g. `scp part.stl PC-108-7:/tmp/part.stl`, then call segment with that remote path.
2. segment(path) -> segmentation_id plus an overview (cluster count by type, bbox). Takes ~10-60 s
   (one run at a time; a call may wait for another). Repeating it with the same file and seeds is instant.
3. list_clusters(segmentation_id) -> clusters by area, largest first; filter by type / min area.
4. get_cluster(segmentation_id, cluster) -> one cluster with its neighbours along the boundary.
5. get_boundaries(segmentation_id, cluster) -> the cluster's boundaries as 3D polylines with edge kind
   (convex / concave / tangent) and the angle between the faces: outline of a face, material for sketches.
6. get_relations(segmentation_id) -> part-wide groups: common directions (extrusion / revolution
   candidates), coplanar planes, coaxial surfaces, equal radii, neighbours that are one surface.
7. cluster_at(segmentation_id, points) -> which cluster lies under given 3D points.
Every cluster carries `surface`: a least-squares primitive fitted to its vertices (the band next to the
boundary excluded), with RMS and the share of vertices within tolerance.

Facts:
- Input is treated as a 3D scan. For a clean CAD tessellation (STL exported from CAD) pass scan_seed:
  it is first run through a virtual scanner, as in training; without it a coarse tessellation is refused.
- Coordinates and lengths are in the mesh's own units, assumed millimetres; the frame is the input's.
- Labels live on an internal work mesh (fixed edge {CFG.tile.edge_mm} mm), not on the input triangles;
  'faces' counts work-mesh triangles.
- Cluster ids are 0..K-1 within one segmentation_id. The server keeps the last {MAX_KEPT} segmentations;
  an unknown id means it was dropped: call segment again.
- Known weaknesses: fillets and faces narrower than ~0.5 mm are often absorbed by neighbours; torus and
  bspline types are least reliable; one geometric face may be split into several clusters (check
  neighbours with low boundary strength).
- Tolerance: max(0.04 mm, 3 x median fit RMS of this part's clusters) - the noise level measured on the
  part itself. It decides `within_tolerance`, `inlier_share` and all relations; get_relations accepts an
  explicit tolerance_mm to loosen or tighten the grouping.
- The fitted kind may differ from the net's `type` (bspline clusters get the best simple primitive)."""


class _Fit(BaseModel):
    rms_mm: float = Field(description="Area-weighted RMS distance of the cluster's vertices to the primitive.")
    inlier_share: float = Field(description="Area share of vertices within the part tolerance, 0..1. Low: the "
                                            "cluster is mixed (absorbed fillet, a piece of another face) or the "
                                            "primitive is wrong.")
    within_tolerance: bool = Field(description="rms_mm <= part tolerance.")


AXIAL_RANGE = ("[min, max] of the cluster along `axis`, measured from {}: the face's extent (e.g. hole depth).")


class PlaneFit(_Fit):
    kind: Literal["plane"]
    normal: Vec3 = Field(description="Unit normal, pointing out of the material.")
    offset: float = Field(description="d in normal . x + d = 0.")
    point: Vec3 = Field(description="The cluster centre projected onto the plane.")


class CylinderFit(_Fit):
    kind: Literal["cylinder"]
    axis_point: Vec3 = Field(description="Point on the axis level with the cluster centre.")
    axis: Vec3 = Field(description="Unit axis direction.")
    radius: float
    axial_range: list[float] = Field(description=AXIAL_RANGE.format("axis_point"))


class ConeFit(_Fit):
    kind: Literal["cone"]
    apex: Vec3
    axis: Vec3 = Field(description="Unit axis direction, from the apex towards the opening.")
    half_angle_deg: float
    axial_range: list[float] = Field(description=AXIAL_RANGE.format("apex"))


class SphereFit(_Fit):
    kind: Literal["sphere"]
    center: Vec3
    radius: float


class TorusFit(_Fit):
    kind: Literal["torus"]
    center: Vec3
    axis: Vec3 = Field(description="Unit axis direction.")
    major_radius: float = Field(description="Distance from the axis to the tube centre.")
    minor_radius: float = Field(description="Tube radius (fillet radius for a rounded edge).")
    axial_range: list[float] = Field(description=AXIAL_RANGE.format("center"))


SurfaceFit = Annotated[PlaneFit | CylinderFit | ConeFit | SphereFit | TorusFit, Field(discriminator="kind")]


class Neighbour(BaseModel):
    cluster: int = Field(description="Neighbouring cluster id.")
    boundary_mm: float = Field(description="Length of the shared boundary.")
    strength: float = Field(description="Mean predicted boundary probability along the shared boundary, 0..1 "
                                        "(length-weighted). Near 1: a confident cut. Low (below ~0.5): the net "
                                        "is unsure; the two clusters may be one surface split in two.")


class ClusterRow(BaseModel):
    cluster: int = Field(description="Cluster id, 0..K-1.")
    type: PrimitiveType = Field(description="Primitive type: majority vote of the cluster's triangles.")
    type_share: float = Field(description="Area fraction of triangles whose own predicted type equals `type`, "
                                          "0..1. Low values: mixed or uncertain cluster.")
    faces: int = Field(description="Work-mesh triangles in the cluster.")
    area_mm2: float = Field(description="Surface area.")
    centroid: Vec3 = Field(description="Area-weighted mean of triangle centres; for curved surfaces it may lie "
                                       "off the surface.")
    bbox_min: Vec3 = Field(description="Axis-aligned bounding box, min corner.")
    bbox_max: Vec3 = Field(description="Axis-aligned bounding box, max corner.")
    mean_normal: Vec3 = Field(description="Area-weighted mean of outward triangle normals, NOT normalised: "
                                          "length ~1 means flat (a plane's normal), shorter means curved "
                                          "(for a full cylinder ~0).")
    surface: SurfaceFit | None = Field(description="Least-squares primitive fitted to the cluster's vertices "
                                                   "(normally of kind `type`, else the best simple one); "
                                                   "null if no fit converged.")


class ClusterSummary(ClusterRow):
    neighbour_count: int = Field(description="Number of adjacent clusters; see get_cluster for the list.")


class Cluster(ClusterRow):
    neighbours: list[Neighbour] = Field(description="Adjacent clusters, longest shared boundary first.")


class SegmentInfo(BaseModel):
    segmentation_id: str = Field(description="Handle for the other tools.")
    path: str = Field(description="Resolved mesh path on the server.")
    work_faces: int = Field(description="Triangles in the work mesh the labels live on.")
    area_mm2: float = Field(description="Total surface area.")
    bbox_min: Vec3 = Field(description="Part bounding box, min corner.")
    bbox_max: Vec3 = Field(description="Part bounding box, max corner.")
    clusters: int = Field(description="Number of clusters K.")
    clusters_by_type: dict[str, int] = Field(description="Cluster count per primitive type.")
    tolerance_mm: float = Field(description="Part tolerance: max(0.04, 3 x median fit RMS).")
    seconds: dict[str, float] = Field(description="Time per pipeline stage (stage names are in Russian).")


class ClusterPage(BaseModel):
    total: int = Field(description="Clusters matching the filter.")
    offset: int = Field(description="Offset of this page.")
    clusters: list[ClusterSummary] = Field(description="Clusters, largest area first.")


class Chain(BaseModel):
    clusters: list[int] = Field(description="The two clusters it separates.")
    closed: bool = Field(description="A closed loop (e.g. a hole rim); then the polyline's last point equals the first.")
    nodes: list[int] | None = Field(description="Node ids at the start and end (null for a loop). Chains meeting at "
                                                "a node share it exactly: chain the polylines of one cluster by node "
                                                "to get the face's outline.")
    length_mm: float = Field(description="Polyline length before simplification.")
    normal_angle_deg: float = Field(description="Mean angle between the two faces' outward normals along the "
                                                "chain: 0 = smooth continuation, 90 = right-angle edge, 180 = "
                                                "knife edge. Taken from the fitted primitives.")
    edge: Literal["convex", "concave", "tangent"] = Field(description="convex = outer edge (material corner), "
                                                                     "concave = inner corner (groove, step "
                                                                     "base), tangent = angle below 5 deg "
                                                                     "(smooth: fillet, blend or a split face).")
    strength: float | None = Field(description="Mean predicted boundary probability along the chain, 0..1.")
    points: list[Vec3] = Field(description="Polyline, simplified (Douglas-Peucker) to half the work-mesh edge; "
                                           "ends sit at the node centres.")


class Boundaries(BaseModel):
    cluster: int = Field(description="The queried cluster.")
    chains: list[Chain] = Field(description="Boundary chains of the cluster, longest first.")
    nodes: dict[int, Vec3] = Field(description="Centres of the nodes the chains end at.")


class CapPlane(BaseModel):
    cluster: int = Field(description="Plane cluster.")
    offset: float = Field(description="Signed position along the direction (direction . point): differences are "
                                      "step heights, pitches, thicknesses.")


class Direction(BaseModel):
    direction: Vec3 = Field(description="Unit direction, area-weighted over its members.")
    caps: list[CapPlane] = Field(description="Planes whose normal is along the direction (end faces of an "
                                             "extrusion along it), sorted by offset.")
    axial: list[int] = Field(description="Cylinders / cones / tori whose axis is along the direction.")
    walls: list[int] = Field(description="Planes whose normal is perpendicular to the direction (side walls of an "
                                         "extrusion along it).")
    area_mm2: float = Field(description="Total area of caps, axial and walls: how much of the part this "
                                        "direction explains.")


class Coplanar(BaseModel):
    clusters: list[int] = Field(description="Plane clusters on one plane, largest first.")
    normal: Vec3 = Field(description="Common normal (sign: the direction group's).")
    offset: float = Field(description="Mean position along the normal.")


class Coaxial(BaseModel):
    clusters: list[int] = Field(description="Cylinders / cones / tori on one axis, largest first.")
    axis: Vec3 = Field(description="Common axis direction.")
    axis_point: Vec3 = Field(description="A point on the axis (from the largest member).")


class EqualRadius(BaseModel):
    radius: float = Field(description="Mean radius.")
    clusters: list[int] = Field(description="Cylinders with this radius: hole patterns, repeated bosses.")


class SameSurface(BaseModel):
    clusters: list[int] = Field(description="Two adjacent clusters.")
    kind: str = Field(description="Their common primitive type.")
    rms_mm: float = Field(description="Worse of the two parts' RMS to the primitive fitted to both together.")
    boundary_strength: float = Field(description="Predicted boundary probability between them; low strength "
                                                 "plus a common primitive = most likely one face cut in two.")


class Relations(BaseModel):
    tolerance_mm: float = Field(description="Distance tolerance used.")
    parallel_deg: float = Field(description="Angle below which directions count as parallel.")
    directions: list[Direction] = Field(description="Direction groups, most area first.")
    coplanar: list[Coplanar] = Field(description="Groups of planes lying on one plane (not necessarily adjacent).")
    coaxial: list[Coaxial] = Field(description="Groups of surfaces of revolution sharing an axis.")
    equal_radius: list[EqualRadius] = Field(description="Groups of cylinders with equal radius.")
    same_surface: list[SameSurface] = Field(description="Adjacent pairs that fit one primitive within tolerance.")


class Hit(BaseModel):
    point: Vec3 = Field(description="Query point.")
    cluster: int = Field(description="Cluster of the nearest work-mesh triangle.")
    type: PrimitiveType = Field(description="That cluster's type.")
    distance_mm: float = Field(description="Distance to the nearest triangle centre; much more than the "
                                           "work-mesh edge means the point is off the surface.")


class Hits(BaseModel):
    hits: list[Hit] = Field(description="One entry per query point, in order.")


@dataclass
class Kept:
    """Сегментация в памяти сервера: сводка посчитана сразу, метки — для поиска по точке."""

    labels: np.ndarray
    tree: cKDTree
    clusters: list[Cluster]
    info: SegmentInfo
    fits: list[D.Fit]
    tolerance: float
    chains: list[Chain]
    nodes: list[list[float]]
    relations: dict[float, Relations]       # по допуску: агент может переспросить с другим


def describe(seg: INF.Segmentation) -> Kept:
    """Всё, что отдают инструменты, считается сразу после нарезки (решено пользователем)."""
    edges = D.cut_edges(seg.mesh, seg.labels, seg.strength)
    rows = D.summarize(seg.mesh, seg.labels, seg.types, seg.cluster_types, edges)
    fits, floor = D.fit_surfaces(seg.mesh, seg.labels, seg.cluster_types, CFG.brep)
    tol = D.part_tolerance(fits, floor)
    clusters = [Cluster(**r, surface=D.surface_params(f, tol)) for r, f in zip(rows, fits)]
    chains, nodes = D.boundaries(seg.mesh, seg.labels, fits, edges, CFG.brep, CFG.tile.edge_mm)
    chains = sorted((Chain(**c) for c in chains), key=lambda c: -c.length_mm)
    types = [c.type for c in clusters]
    info = SegmentInfo(
        segmentation_id="",
        path="",
        work_faces=len(seg.mesh.faces),
        area_mm2=round(float(seg.mesh.area), D.DIGITS),
        bbox_min=D._r(seg.mesh.bounds[0]),
        bbox_max=D._r(seg.mesh.bounds[1]),
        clusters=len(clusters),
        clusters_by_type={t: types.count(t) for t in INF.PRIMITIVE_TYPES if t in types},
        tolerance_mm=round(tol, D.DIGITS),
        seconds={k: round(v, 1) for k, v in seg.seconds.items()},
    )
    kept = Kept(seg.labels, cKDTree(seg.mesh.triangles_center), clusters, info, fits, tol, chains, nodes, {})
    kept.relations[tol] = Relations(**D.relations(rows, fits, tol, CFG.brep))
    return kept


def build_server(checkpoint: str) -> MCPServer:
    model = INF.load_model(checkpoint)
    kept: OrderedDict[str, Kept] = OrderedDict()
    gpu = asyncio.Lock()
    server = MCPServer("curv-ai-segm", instructions=INSTRUCTIONS)

    def get(segmentation_id: str) -> Kept:
        if segmentation_id not in kept:
            raise ToolError(f"нет сегментации {segmentation_id!r}: вызовите segment заново")
        return kept[segmentation_id]

    @server.tool()
    async def segment(
        path: Annotated[str, Field(description="Mesh file (STL, PLY, OBJ) on the server host PC-108-7; "
                                               "absolute or ~-relative.")],
        scan_seed: Annotated[int | None, Field(description="Set (any int, e.g. 0) when the mesh is a clean CAD "
                                                           "tessellation: it is first run through the virtual "
                                                           "scanner with this seed. Leave null for real scans.")] = None,
        seed: Annotated[int, Field(description="Random seed of the net's neighbour sampling; the same seed "
                                               "gives the same result.")] = 0,
    ) -> SegmentInfo:
        """Segment a mesh into primitive clusters and keep the result under a segmentation_id.
        Call this first; the other tools take its segmentation_id."""
        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise ToolError(f"файла нет на сервере: {file}")
        key = hashlib.sha1(f"{file}|{file.stat().st_mtime_ns}|{scan_seed}|{seed}".encode()).hexdigest()[:12]
        if key not in kept:
            async with gpu:
                try:
                    seg = await anyio.to_thread.run_sync(lambda: INF.segment_tiles(file, model, scan_seed, seed))
                except ValueError as e:  # вход не годится (например, меш грубее ребра): причину — агенту
                    raise ToolError(str(e)) from e
                t0 = time.perf_counter()
                k = await anyio.to_thread.run_sync(lambda: describe(seg))
            k.info.seconds["описание"] = round(time.perf_counter() - t0, 1)
            k.info.segmentation_id, k.info.path = key, str(file)
            kept[key] = k
            while len(kept) > MAX_KEPT:
                kept.popitem(last=False)
        kept.move_to_end(key)
        return kept[key].info

    @server.tool()
    async def list_clusters(
        segmentation_id: Annotated[str, Field(description="From segment.")],
        type: Annotated[PrimitiveType | None, Field(description="Only clusters of this primitive type.")] = None,
        min_area_mm2: Annotated[float, Field(description="Only clusters at least this large.")] = 0.0,
        limit: Annotated[int, Field(ge=1, description="Page size.")] = 100,
        offset: Annotated[int, Field(ge=0, description="Skip this many matching clusters.")] = 0,
    ) -> ClusterPage:
        """List clusters, largest area first, without neighbour lists (use get_cluster for those).
        Good first look at a part: the big planes and cylinders are its main faces."""
        rows = [c for c in get(segmentation_id).clusters
                if (type is None or c.type == type) and c.area_mm2 >= min_area_mm2]
        rows.sort(key=lambda c: -c.area_mm2)
        page = [ClusterSummary(**c.model_dump(exclude={"neighbours"}), neighbour_count=len(c.neighbours))
                for c in rows[offset:offset + limit]]
        return ClusterPage(total=len(rows), offset=offset, clusters=page)

    @server.tool()
    async def get_cluster(
        segmentation_id: Annotated[str, Field(description="From segment.")],
        cluster: Annotated[int, Field(ge=0, description="Cluster id.")],
    ) -> Cluster:
        """One cluster with all its neighbours: shared boundary length and how confident the cut is."""
        clusters = get(segmentation_id).clusters
        if cluster >= len(clusters):
            raise ToolError(f"кластеры 0..{len(clusters) - 1}")
        return clusters[cluster]

    @server.tool()
    async def get_boundaries(
        segmentation_id: Annotated[str, Field(description="From segment.")],
        cluster: Annotated[int, Field(ge=0, description="Cluster id.")],
        neighbour: Annotated[int | None, Field(description="Only the chains shared with this cluster.")] = None,
    ) -> Boundaries:
        """Boundary chains of a cluster as 3D polylines, with edge kind and the angle between the faces.
        Chained by node they give the face's outline; a planar face's outline is a sketch profile."""
        s = get(segmentation_id)
        if cluster >= len(s.clusters):
            raise ToolError(f"кластеры 0..{len(s.clusters) - 1}")
        chains = [c for c in s.chains if cluster in c.clusters and (neighbour is None or neighbour in c.clusters)]
        nodes = sorted({n for c in chains for n in c.nodes or []})
        return Boundaries(cluster=cluster, chains=chains, nodes={n: s.nodes[n] for n in nodes})

    @server.tool()
    async def get_relations(
        segmentation_id: Annotated[str, Field(description="From segment.")],
        tolerance_mm: Annotated[float | None, Field(gt=0, description="Distance tolerance for coplanar / coaxial "
                                                                      "/ equal radius / same surface. Default: "
                                                                      "the part tolerance from segment.")] = None,
    ) -> Relations:
        """Part-wide relations between the fitted primitives: direction groups (extrusion / revolution
        candidates with their caps, walls and axial faces), coplanar and coaxial groups, equal radii, and
        adjacent clusters that are one surface cut in two."""
        s = get(segmentation_id)
        tol = s.tolerance if tolerance_mm is None else tolerance_mm
        if tol not in s.relations:
            rows = [c.model_dump(exclude={"surface"}) for c in s.clusters]
            s.relations[tol] = await anyio.to_thread.run_sync(
                lambda: Relations(**D.relations(rows, s.fits, tol, CFG.brep)))
        return s.relations[tol]

    @server.tool()
    async def cluster_at(
        segmentation_id: Annotated[str, Field(description="From segment.")],
        points: Annotated[list[Vec3], Field(min_length=1, description="Query points [x, y, z] in mesh "
                                                                      "coordinates.")],
    ) -> Hits:
        """Which cluster lies under each point (nearest work-mesh triangle). Use it to map a location
        you found elsewhere (a section, a picture, a measurement) to a cluster id."""
        s = get(segmentation_id)
        dist, face = s.tree.query(np.asarray(points, dtype=np.float64))
        return Hits(hits=[Hit(point=D._r(p), cluster=int(s.labels[f]), type=s.clusters[s.labels[f]].type,
                              distance_mm=round(float(d), D.DIGITS)) for p, f, d in zip(points, face, dist)])

    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "tile50" / "best.pt"))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    if CFG.tile is None:
        raise SystemExit("нужен конфиг кусков: CURV_CONFIG=configs/tile.yaml")
    build_server(args.checkpoint).run("streamable-http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
