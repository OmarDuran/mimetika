"""Mesh: the object that ties topology and geometry together.

A :class:`Mesh` is the single handle the higher layers (dof, operators,
assembly) receive.  It does not add new numerics itself; it just co-locates the
combinatorial complex and its metric.
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

    @classmethod
    def from_cells(
        cls, points: np.ndarray, cells: list[list[list[int]]]
    ) -> "Mesh":
        """Build a mesh from vertex coordinates and polyhedral cells."""
        cx = CellComplex.from_polyhedra(len(points), cells)
        return cls(complex=cx, geometry=Geometry(cx, points))

    @property
    def dim(self) -> int:
        return self.complex.dim

    def num_cells(self, k: int) -> int:
        return self.complex.num_cells(k)
