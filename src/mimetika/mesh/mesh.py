"""Mesh: the object that ties topology and geometry together.

A :class:`Mesh` is the single handle the higher layers (dof, operators,
assembly) receive.  It adds no numerics of its own; it co-locates the
combinatorial complex and its metric.  Meshes of dimension 0, 1, 2 and 3 are
all supported, always embedded in ``R^3``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.geometry.metric import Geometry
from mimetika.topology.complex import CellComplex


@dataclass
class Mesh:
    complex: CellComplex
    geometry: Geometry

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Mesh":
        """A 0D mesh of isolated points."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cx = CellComplex.from_vertices(len(points))
        return cls(complex=cx, geometry=Geometry(cx, points))

    @classmethod
    def from_segments(
        cls, points: np.ndarray, segments: list[tuple[int, int]]
    ) -> "Mesh":
        """A 1D mesh of segments."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cx = CellComplex.from_segments(len(points), segments)
        return cls(complex=cx, geometry=Geometry(cx, points))

    @classmethod
    def from_polygons(cls, points: np.ndarray, polygons: list[list[int]]) -> "Mesh":
        """A 2D mesh of polygons given as ordered vertex loops."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cx = CellComplex.from_polygons(len(points), polygons)
        return cls(complex=cx, geometry=Geometry(cx, points))

    @classmethod
    def from_cells(cls, points: np.ndarray, cells: list[list[list[int]]]) -> "Mesh":
        """A 3D mesh from polyhedral cells (each a list of oriented face loops)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cx = CellComplex.from_polyhedra(len(points), cells)
        return cls(complex=cx, geometry=Geometry(cx, points))

    # -- accessors -----------------------------------------------------------

    @property
    def dim(self) -> int:
        return self.complex.dim

    def num_cells(self, k: int) -> int:
        return self.complex.num_cells(k)
