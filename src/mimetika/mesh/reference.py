"""A catalogue of single reference cells of every topological dimension.

Each entry is one cell -- a point, a segment, a polygon or a polyhedron --
embedded in ``R^3``, together with its **analytically known** measure so the
geometry layer can be checked against exact values rather than against itself.

Following the standing assumptions of the mimetic convergence theory, all cells
have **planar facets** and are **star-shaped**; the collection deliberately
includes irregular, tilted and non-convex (but still star-shaped) shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.mesh.mesh import Mesh

SQRT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class ReferenceCell:
    """A single reference cell with its exact measure."""

    name: str
    dim: int
    mesh: Mesh
    measure: float
    convex: bool = True

    def __repr__(self) -> str:  # keeps pytest ids readable
        return f"<{self.name} dim={self.dim}>"


def _rotation() -> np.ndarray:
    """A fixed generic rotation, to place planar cells in a tilted plane."""
    a, b, c = 0.3, -0.7, 1.1
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _lift(xy: np.ndarray, tilt: bool = False) -> np.ndarray:
    """Embed planar ``(N,2)`` coordinates into ``R^3``, optionally tilted."""
    pts = np.column_stack([np.asarray(xy, dtype=float), np.zeros(len(xy))])
    return pts @ _rotation().T if tilt else pts


# -- 0D -----------------------------------------------------------------------


def _points() -> list[ReferenceCell]:
    return [
        ReferenceCell("point-origin", 0, Mesh.from_points([[0.0, 0.0, 0.0]]), 1.0),
        ReferenceCell("point-offset", 0, Mesh.from_points([[0.4, -1.2, 2.5]]), 1.0),
    ]


# -- 1D -----------------------------------------------------------------------


def _segments() -> list[ReferenceCell]:
    unit = Mesh.from_segments([[0, 0, 0], [1, 0, 0]], [(0, 1)])
    # 3-4-5 right triangle in the xy-plane -> length exactly 5
    oblique = Mesh.from_segments([[0.5, -0.5, 1.0], [3.5, 3.5, 1.0]], [(0, 1)])
    diag = Mesh.from_segments([[0, 0, 0], [2, 2, 1]], [(0, 1)])  # length 3
    return [
        ReferenceCell("segment-unit", 1, unit, 1.0),
        ReferenceCell("segment-oblique", 1, oblique, 5.0),
        ReferenceCell("segment-diagonal", 1, diag, 3.0),
    ]


# -- 2D -----------------------------------------------------------------------

_TRI_IRREGULAR = np.array([[0.1, 0.2], [1.3, 0.0], [0.4, 0.9]])
_QUAD_IRREGULAR = np.array([[0.0, 0.0], [1.2, 0.0], [1.5, 1.0], [0.2, 0.8]])
_HEX_REGULAR = np.array(
    [[np.cos(t), np.sin(t)] for t in np.arange(6) * np.pi / 3.0]
)
# Star-shaped L-shape: non-convex, star-shaped about (0.5, 0.5).
_L_SHAPE = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)


def _polygon(name, xy, area, tilt=False, convex=True) -> ReferenceCell:
    mesh = Mesh.from_polygons(_lift(xy, tilt), [list(range(len(xy)))])
    return ReferenceCell(name, 2, mesh, area, convex)


def _polygons() -> list[ReferenceCell]:
    return [
        _polygon("triangle-unit", [[0, 0], [1, 0], [0, 1]], 0.5),
        _polygon("triangle-irregular", _TRI_IRREGULAR, 0.45),
        _polygon("square-unit", [[0, 0], [1, 0], [1, 1], [0, 1]], 1.0),
        _polygon("quad-irregular", _QUAD_IRREGULAR, 1.1),
        _polygon("hexagon-regular", _HEX_REGULAR, 1.5 * SQRT3),
        _polygon("lshape-nonconvex", _L_SHAPE, 3.0, convex=False),
        # tilted copies exercise the general R^3 embedding
        _polygon("square-tilted", [[0, 0], [1, 0], [1, 1], [0, 1]], 1.0, tilt=True),
        _polygon("lshape-tilted", _L_SHAPE, 3.0, tilt=True, convex=False),
    ]


# -- 3D -----------------------------------------------------------------------

_TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
_HEX_FACES = (
    (0, 3, 2, 1),  # z-
    (4, 5, 6, 7),  # z+
    (0, 1, 5, 4),  # y-
    (1, 2, 6, 5),  # x+
    (2, 3, 7, 6),  # y+
    (0, 4, 7, 3),  # x-
)
_PRISM_FACES = ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
_PYRAMID_FACES = ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))

_TET_IRREGULAR = np.array(
    [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.3, 0.9, 0.1], [0.1, 0.2, 1.4]]
)
# Shear preserves volume and keeps every quadrilateral face planar.
_HEX_SHEARED = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0.4, 0.25, 1], [1.4, 0.25, 1], [1.4, 1.25, 1], [0.4, 1.25, 1],
    ],
    dtype=float,
)
_PRISM = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=float
)
_PYRAMID = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.2]], dtype=float
)

# Unit cube whose top face is dented *inward* to an apex: non-convex, all facets
# planar (the dent is four triangles), star-shaped about a point near the base.
_DENT_DEPTH = 0.4
_CUBE_DENTED = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [0.5, 0.5, 1.0 - _DENT_DEPTH],  # 8: the dent apex
    ],
    dtype=float,
)
_CUBE_DENTED_FACES = (
    (0, 3, 2, 1),  # bottom
    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3),  # sides
    (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 4, 8),  # dented top
)


def _polyhedron(name, pts, faces, volume, convex=True) -> ReferenceCell:
    mesh = Mesh.from_cells(pts, [[list(f) for f in faces]])
    return ReferenceCell(name, 3, mesh, volume, convex)


def _tet_volume(p: np.ndarray) -> float:
    return abs(np.linalg.det(p[1:] - p[0])) / 6.0


def _polyhedra() -> list[ReferenceCell]:
    ref_tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cube = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    return [
        _polyhedron("tet-reference", ref_tet, _TET_FACES, 1.0 / 6.0),
        _polyhedron(
            "tet-irregular", _TET_IRREGULAR, _TET_FACES, _tet_volume(_TET_IRREGULAR)
        ),
        _polyhedron("cube-unit", cube, _HEX_FACES, 1.0),
        _polyhedron("hex-sheared", _HEX_SHEARED, _HEX_FACES, 1.0),
        _polyhedron("prism-triangular", _PRISM, _PRISM_FACES, 1.0),
        _polyhedron("pyramid-square", _PYRAMID, _PYRAMID_FACES, 1.2 / 3.0),
        _polyhedron(
            "cube-dented",
            _CUBE_DENTED,
            _CUBE_DENTED_FACES,
            1.0 - _DENT_DEPTH / 3.0,
            convex=False,
        ),
    ]


# -- catalogue ----------------------------------------------------------------


def reference_cells(dim: int | None = None) -> list[ReferenceCell]:
    """All reference cells, optionally restricted to one dimension."""
    cells = _points() + _segments() + _polygons() + _polyhedra()
    return cells if dim is None else [c for c in cells if c.dim == dim]
