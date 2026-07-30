"""Geometry layer: coordinates and metric quantities.

Given vertex coordinates and a :class:`~mimetika.topology.complex.CellComplex`,
this computes the *metric* data the mimetic operators need: k-cell measures
(lengths, areas, volumes), **true** centroids, facet normals and quadrature.

Everything works for cells of any dimension (segment, polygon, polyhedron)
embedded in ``R^3``, with the standing assumption -- shared with the mimetic
convergence theory -- that facets are **planar** and cells are **star-shaped**.

Centroids are true (measure-weighted) centroids, not vertex averages: the
mimetic consistency identities are exact only when the element reference point
is the actual centroid, so this matters for the patch tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.topology.complex import CellComplex

# Degree-2 symmetric rule on the reference tetrahedron (4 points, barycentric).
_TET_A = 0.585410196624968515
_TET_B = 0.138196601125010515
_TET_BARY = np.array(
    [
        [_TET_A, _TET_B, _TET_B, _TET_B],
        [_TET_B, _TET_A, _TET_B, _TET_B],
        [_TET_B, _TET_B, _TET_A, _TET_B],
        [_TET_B, _TET_B, _TET_B, _TET_A],
    ]
)

# 2-point Gauss on [0, 1] (exact for cubics).
_GAUSS2 = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))


@dataclass
class Geometry:
    """Metric data attached to a :class:`CellComplex`."""

    complex: CellComplex
    points: np.ndarray  # (n_vertices, 3)

    _measures: dict[int, np.ndarray] | None = None
    _centroids: dict[int, np.ndarray] | None = None
    _facet_normals: np.ndarray | None = None
    _facet_tangents: np.ndarray | None = None
    _quad_cache: dict | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be (n_vertices, 3)")
        self._measures = {}
        self._centroids = {}
        self._quad_cache = {}

    # -- measures ------------------------------------------------------------

    def measure(self, k: int) -> np.ndarray:
        """Array of k-cell measures (count/length/area/volume)."""
        if k in self._measures:
            return self._measures[k]
        if k == 0:
            m = np.ones(self.complex.num_cells(0))
        elif k == 1:
            m = self._edge_lengths()
        elif k == 2:
            m = self._polygon_areas()
        elif k == 3:
            m = self._cell_volumes()
        else:
            raise ValueError(f"measure not defined for k={k}")
        self._measures[k] = m
        return m

    def _edge_lengths(self) -> np.ndarray:
        ev = self.complex.edge_vertices
        return np.linalg.norm(self.points[ev[:, 1]] - self.points[ev[:, 0]], axis=1)

    def _polygon_areas(self) -> np.ndarray:
        return np.array(
            [
                np.linalg.norm(self._area_vector(loop))
                for loop in self.complex.polygon_loops
            ]
        )

    def _cell_volumes(self) -> np.ndarray:
        """Volume by the divergence theorem: ``V = (1/3) sum_e s_e (x_e . A_e)``."""
        fc = self.centroids(2)
        vols = np.zeros(self.complex.num_cells(3))
        for cid in range(self.complex.num_cells(3)):
            acc = 0.0
            for fid, sgn in self.complex.facets_of(3, cid):
                area_vec = self._area_vector(self.complex.polygon_loops[fid])
                acc += sgn * float(np.dot(fc[fid], area_vec))
            vols[cid] = abs(acc) / 3.0
        return vols

    def _area_vector(self, loop: tuple[int, ...]) -> np.ndarray:
        """Vector area of a planar polygon (Newell); direction by right-hand rule."""
        p = self.points[list(loop)]
        c = p.mean(0)
        return 0.5 * np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(0)

    # -- centroids -----------------------------------------------------------

    def centroids(self, k: int) -> np.ndarray:
        """True (measure-weighted) centroids of the k-cells."""
        if k in self._centroids:
            return self._centroids[k]
        if k == 0:
            c = self.points
        elif k == 1:
            c = self.points[self.complex.edge_vertices].mean(axis=1)
        else:
            c = np.array(
                [
                    self._quadrature_centroid(k, cid)
                    for cid in range(self.complex.num_cells(k))
                ]
            ).reshape(-1, 3)
        self._centroids[k] = c
        return c

    def _quadrature_centroid(self, k: int, cell_id: int) -> np.ndarray:
        pts, wts = self.quadrature(k, cell_id)
        return (wts[:, None] * pts).sum(0) / wts.sum()

    def facet_centroids(self) -> np.ndarray:
        """Centroids of the 2-cells (the facets of a 3D complex)."""
        return self.centroids(2)

    def facet_normals(self) -> np.ndarray:
        """Unit normals of the 2-cells, per their canonical loop orientation."""
        if self._facet_normals is None:
            n = np.array(
                [self._area_vector(loop) for loop in self.complex.polygon_loops]
            ).reshape(-1, 3)
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            self._facet_normals = n / np.where(norms == 0, 1.0, norms)
        return self._facet_normals

    # -- quadrature ----------------------------------------------------------

    def quadrature(self, k: int, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature on a k-cell, exact at least for quadratics.

        Returns ``(points (Nq,3), weights (Nq,))`` with ``sum(weights)`` equal to
        the k-cell measure.

        Cached: every facet is visited once per adjacent cell (twice in the
        interior) and every cell at least twice (centroid, then assembly), so
        recomputing dominates assembly on large meshes.
        """
        if self._quad_cache is None:
            self._quad_cache = {}
        hit = self._quad_cache.get((k, cell_id))
        if hit is not None:
            return hit
        out = self._quadrature(k, cell_id)
        self._quad_cache[(k, cell_id)] = out
        return out

    def clear_quadrature_cache(self) -> None:
        """Drop cached quadrature (frees memory on very large meshes)."""
        self._quad_cache = {}

    def _quadrature(self, k: int, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        if k == 0:
            return self.points[cell_id][None, :], np.ones(1)
        if k == 1:
            return self._segment_quadrature(cell_id)
        if k == 2:
            return self._polygon_quadrature(cell_id)
        if k == 3:
            return self._polyhedron_quadrature(cell_id)
        raise ValueError(f"quadrature not defined for k={k}")

    def _segment_quadrature(self, eid: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.points[self.complex.edge_vertices[eid]]
        length = np.linalg.norm(b - a)
        pts = a + _GAUSS2[:, None] * (b - a)
        return pts, np.full(2, length / 2.0)

    def _polygon_quadrature(self, fid: int) -> tuple[np.ndarray, np.ndarray]:
        """Fan-triangulate from the vertex mean with **signed** areas.

        Signed weights make the decomposition an exact algebraic identity for
        any apex, so non-convex (star-shaped) polygons integrate correctly even
        when the apex falls outside the polygon.
        """
        loop = self.complex.polygon_loops[fid]
        fp = self.points[list(loop)]
        apex = fp.mean(0)
        normal = self._area_vector(loop)
        normal = normal / np.linalg.norm(normal)

        nxt = np.roll(fp, -1, axis=0)
        # one cross product for the whole fan; degenerate triangles simply get
        # zero weight, so they need no special case
        areas = 0.5 * (np.cross(fp - apex, nxt - apex) @ normal)

        # degree-2 rule: the midpoints of each fan triangle (apex, fp_i, nxt_i)
        mids = np.stack(
            [0.5 * (apex + fp), 0.5 * (fp + nxt), 0.5 * (nxt + apex)], axis=1
        )
        return mids.reshape(-1, 3), np.repeat(areas / 3.0, 3)

    def _polyhedron_quadrature(self, cid: int) -> tuple[np.ndarray, np.ndarray]:
        """Subdivide into tets (cell apex -> facet centroid -> facet edge).

        Uses **signed** tet volumes over the outward-oriented boundary, so the
        decomposition is exact for any apex and any simple polyhedron.
        """
        facets = self.complex.facets_of(3, cid)
        apex = np.mean(
            [
                self.points[list(self.complex.polygon_loops[f])].mean(0)
                for f, _ in facets
            ],
            axis=0,
        )
        pts, wts = [], []
        for fid, sgn in facets:
            loop = self.complex.polygon_loops[fid]
            if sgn < 0:  # orient the loop outward for this cell
                loop = tuple(reversed(loop))
            fp = self.points[list(loop)]
            fc = fp.mean(0)
            for i in range(len(loop)):
                a, b = fp[i], fp[(i + 1) % len(loop)]
                vol = float(
                    np.dot(fc - apex, np.cross(a - apex, b - apex))
                ) / 6.0
                if vol == 0.0:
                    continue
                pts.append(_TET_BARY @ np.array([apex, fc, a, b]))
                wts.append(np.full(4, vol / 4.0))
        return np.vstack(pts), np.concatenate(wts)

    # -- legacy aliases (3D naming) -----------------------------------------

    def cell_quadrature(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature over a top-dimensional cell."""
        return self.quadrature(self.complex.dim, cell_id)

    def facet_quadrature(self, fid: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature over a facet of a top-dimensional cell."""
        return self.quadrature(self.complex.dim - 1, fid)

    def facet_tangents(self, fid: int) -> tuple[np.ndarray, np.ndarray]:
        """A deterministic orthonormal in-plane tangent frame ``(t1, t2)``.

        Depends only on the canonical facet normal, hence identical for the two
        cells sharing the facet -- so face-based DOFs are globally consistent.
        Computed once for all facets and cached.
        """
        t = self.facet_tangent_frames()
        return t[fid, 0], t[fid, 1]

    def facet_tangent_frames(self) -> np.ndarray:
        """``(n_facets, 2, 3)`` in-plane frames for every 2-cell, vectorised."""
        if self._facet_tangents is None:
            n = self.facet_normals()
            e = np.zeros_like(n)
            e[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
            t1 = e - (np.einsum("ij,ij->i", e, n))[:, None] * n
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            self._facet_tangents = np.stack([t1, np.cross(n, t1)], axis=1)
        return self._facet_tangents
