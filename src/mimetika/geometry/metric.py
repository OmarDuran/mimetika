"""Geometry layer: coordinates and metric quantities.

Given vertex coordinates and a :class:`~mimetika.topology.complex.CellComplex`,
this computes the *metric* data the mimetic operators need: k-cell measures
(edge lengths, facet areas, cell volumes), centroids and facet normals.

Measures are computed for general (planar) polygonal facets and general
polyhedral cells, so the same code serves hexes and arbitrary polytopes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.topology.complex import CellComplex


@dataclass
class Geometry:
    """Metric data attached to a :class:`CellComplex`."""

    complex: CellComplex
    points: np.ndarray  # (n_vertices, 3)

    # cached measures / centroids, filled lazily
    _measures: dict[int, np.ndarray] | None = None
    _centroids: dict[int, np.ndarray] | None = None
    _facet_normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        if self.points.shape[1] != 3:
            raise ValueError("points must be (n_vertices, 3)")
        self._measures = {}
        self._centroids = {}

    # -- measures ------------------------------------------------------------

    def measure(self, k: int) -> np.ndarray:
        """Return the array of k-cell measures (length/area/volume)."""
        if k in self._measures:
            return self._measures[k]
        if k == 0:
            m = np.ones(self.complex.num_cells(0))
        elif k == 1:
            m = self._edge_lengths()
        elif k == 2:
            m = self._facet_areas()
        elif k == 3:
            m = self._cell_volumes()
        else:
            raise ValueError(f"measure not defined for k={k}")
        self._measures[k] = m
        return m

    def _edge_lengths(self) -> np.ndarray:
        ev = self.complex.edge_vertices
        d = self.points[ev[:, 1]] - self.points[ev[:, 0]]
        return np.linalg.norm(d, axis=1)

    def _facet_areas(self) -> np.ndarray:
        areas = np.empty(self.complex.num_cells(2))
        for fid, loop in enumerate(self.complex.facet_vertices):
            areas[fid] = np.linalg.norm(self._polygon_area_vector(loop))
        return areas

    def _cell_volumes(self) -> np.ndarray:
        """Signed-then-absolute volume via the divergence theorem over facets.

        V = (1/3) * sum_facets  (x_f . n_f) * A_f, using oriented facet normals
        pointing out of the cell.
        """
        vols = np.zeros(self.complex.num_cells(3))
        fc = self.facet_centroids()
        for cid, entry in enumerate(self.complex.cell_facets):
            acc = 0.0
            for fid, sgn in entry:
                area_vec = self._polygon_area_vector(self.complex.facet_vertices[fid])
                acc += sgn * float(np.dot(fc[fid], area_vec))
            vols[cid] = abs(acc) / 3.0
        return vols

    def _polygon_area_vector(self, loop: tuple[int, ...]) -> np.ndarray:
        """Vector area of a planar polygon (Newell's method); |.| is the area,
        direction is the loop normal by the right-hand rule."""
        p = self.points[list(loop)]
        return 0.5 * np.cross(p - p.mean(0), np.roll(p, -1, axis=0) - p.mean(0)).sum(0)

    # -- centroids -----------------------------------------------------------

    def centroids(self, k: int) -> np.ndarray:
        """Centroids of k-cells (vertex-mean approximation for k>=1)."""
        if k in self._centroids:
            return self._centroids[k]
        if k == 0:
            c = self.points
        elif k == 1:
            c = self.points[self.complex.edge_vertices].mean(axis=1)
        elif k == 2:
            c = self.facet_centroids()
        elif k == 3:
            c = np.array(
                [
                    np.mean(
                        [
                            self.points[list(self.complex.facet_vertices[fid])].mean(0)
                            for fid, _ in entry
                        ],
                        axis=0,
                    )
                    for entry in self.complex.cell_facets
                ]
            )
        else:
            raise ValueError(f"centroid not defined for k={k}")
        self._centroids[k] = c
        return c

    def facet_centroids(self) -> np.ndarray:
        if 2 in self._centroids:
            return self._centroids[2]
        c = np.array(
            [self.points[list(loop)].mean(0) for loop in self.complex.facet_vertices]
        )
        self._centroids[2] = c
        return c

    def facet_normals(self) -> np.ndarray:
        """Unit normals of facets, per the canonical facet loop orientation."""
        if self._facet_normals is None:
            n = np.array(
                [
                    self._polygon_area_vector(loop)
                    for loop in self.complex.facet_vertices
                ]
            )
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            self._facet_normals = n / np.where(norms == 0, 1.0, norms)
        return self._facet_normals
