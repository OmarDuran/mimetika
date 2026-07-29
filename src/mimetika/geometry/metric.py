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

    # -- quadrature ----------------------------------------------------------

    # Degree-2 symmetric rule on the reference tetrahedron (4 points), exact for
    # quadratic integrands. Barycentric coords; weights sum to 1.
    _TET_RULE_A = 0.585410196624968515
    _TET_RULE_B = 0.138196601125010515

    def cell_quadrature(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature points/weights integrating quadratics exactly over a cell.

        The polyhedron is subdivided into tetrahedra (cell centroid -> face
        centroid -> face edge), each carrying a degree-2 rule.  Returns
        ``(points (Nq,3), weights (Nq,))`` with ``sum(weights) == |E|``.
        """
        a, b = self._TET_RULE_A, self._TET_RULE_B
        bary = np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]])
        xc = self.centroids(3)[cell_id]

        pts, wts = [], []
        for fid, _ in self.complex.cell_facets[cell_id]:
            loop = self.complex.facet_vertices[fid]
            fp = self.points[list(loop)]
            fc = fp.mean(0)
            m = len(loop)
            for i in range(m):
                tet = np.array([xc, fc, fp[i], fp[(i + 1) % m]])
                vol = abs(np.dot(tet[1] - tet[0],
                                 np.cross(tet[2] - tet[0], tet[3] - tet[0]))) / 6.0
                if vol == 0.0:
                    continue
                pts.append(bary @ tet)  # (4,3)
                wts.append(np.full(4, vol / 4.0))
        return np.vstack(pts), np.concatenate(wts)

    def facet_quadrature(self, fid: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature over one (planar) facet, exact for quadratics.

        Fan-triangulates the polygon from its centroid and applies the degree-2
        3-midpoint triangle rule.  Returns ``(points (Nq,3), weights (Nq,))``
        with ``sum(weights) == area(facet)``.
        """
        loop = self.complex.facet_vertices[fid]
        fp = self.points[list(loop)]
        fc = fp.mean(0)
        m = len(loop)
        pts, wts = [], []
        for i in range(m):
            a, b = fp[i], fp[(i + 1) % m]
            area = 0.5 * np.linalg.norm(np.cross(a - fc, b - fc))
            if area == 0.0:
                continue
            tri = np.array([fc, a, b])
            mids = np.array([(tri[0] + tri[1]) / 2,
                             (tri[1] + tri[2]) / 2,
                             (tri[2] + tri[0]) / 2])
            pts.append(mids)
            wts.append(np.full(3, area / 3.0))
        return np.vstack(pts), np.concatenate(wts)

    def facet_tangents(self, fid: int) -> tuple[np.ndarray, np.ndarray]:
        """A deterministic orthonormal in-plane tangent frame ``(t1, t2)``.

        Depends only on the canonical facet normal, hence identical for the two
        cells sharing the facet -- so face-based DOFs are globally consistent.
        """
        n = self.facet_normals()[fid]
        axis = int(np.argmin(np.abs(n)))
        e = np.zeros(3)
        e[axis] = 1.0
        t1 = e - np.dot(e, n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return t1, t2

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
