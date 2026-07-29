"""Combinatorial cell complex and signed incidence (boundary) operators.

This layer is *purely combinatorial*: it knows nothing about coordinates,
lengths, areas or volumes.  For a graded complex of dimension ``d`` it stores
the signed boundary operators

    ``d_k : C_k -> C_{k-1}``   (``boundary[k]``, shape ``(n_{k-1}, n_k)``)

for ``k = 1 .. d``.  The discrete exterior derivative on k-forms is the
transpose (coboundary) ``D_k = boundary[k+1].T`` and is provided by the
:mod:`mimetika.operators` layer.

Complexes of every dimension are supported:

    ``dim 0``  a set of points          (no differential operators)
    ``dim 1``  segments                 facets = vertices
    ``dim 2``  polygons                 facets = edges
    ``dim 3``  polyhedra                facets = polygons

Top-dimensional cells are always described by their boundary with a globally
consistent orientation, and the lower-dimensional cells are deduced with
consistent signs, so the fundamental mimetic identity

    ``boundary[k] @ boundary[k+1] == 0``

holds *exactly*, by construction, for any well-formed mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# Named ranks.  In 3D: vertex=0, edge=1, facet=2, cell=3.
VERTEX = 0
EDGE = 1


@dataclass
class CellComplex:
    """A graded cell complex with signed boundary operators.

    Attributes
    ----------
    dim
        Topological dimension ``d``.
    n_cells
        ``n_cells[k]`` = number of k-cells, for ``k = 0 .. dim``.
    boundary
        ``boundary[k]`` = signed incidence ``C_k -> C_{k-1}`` as a CSR matrix of
        shape ``(n_cells[k-1], n_cells[k])``, for ``k = 1 .. dim``.
        ``boundary[0]`` is ``None``.
    edge_vertices
        ``(n_edges, 2)`` array; each row ``(u, w)`` with ``u < w`` is an edge
        oriented from ``u`` to ``w``.  Empty when ``dim == 0``.
    polygon_loops
        Canonical ordered vertex loops of the 2-cells (the facets of a 3D
        complex, or the top cells of a 2D complex).  Empty when ``dim < 2``.
    """

    dim: int
    n_cells: list[int]
    boundary: list[sp.csr_matrix | None]
    edge_vertices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    polygon_loops: list[tuple[int, ...]] = field(default_factory=list)
    _csc_cache: dict[int, sp.csc_matrix] | None = field(
        default=None, repr=False, compare=False
    )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_vertices(cls, n_vertices: int) -> "CellComplex":
        """A 0-dimensional complex: isolated points, no differential operators."""
        return cls(dim=0, n_cells=[n_vertices], boundary=[None])

    @classmethod
    def from_segments(
        cls, n_vertices: int, segments: list[tuple[int, int]]
    ) -> "CellComplex":
        """A 1D complex of segments; the facets of a segment are its endpoints."""
        rows, cols, vals = [], [], []
        ev = []
        for sid, (a, b) in enumerate(segments):
            ev.append((a, b))
            rows += [b, a]
            cols += [sid, sid]
            vals += [1.0, -1.0]
        b1 = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_vertices, len(segments))
        )
        return cls(
            dim=1,
            n_cells=[n_vertices, len(segments)],
            boundary=[None, b1],
            edge_vertices=np.array(ev, dtype=np.int64).reshape(-1, 2),
        )

    @classmethod
    def from_polygons(
        cls, n_vertices: int, polygons: list[list[int]]
    ) -> "CellComplex":
        """A 2D complex of polygons given as ordered vertex loops."""
        edges = _EdgeTable()
        for loop in polygons:
            for a, b in _loop_pairs(loop):
                edges.get(a, b)

        b1 = edges.vertex_incidence(n_vertices)

        rows, cols, vals = [], [], []
        for pid, loop in enumerate(polygons):
            for a, b in _loop_pairs(loop):
                eid, sgn = edges.get(a, b)
                rows.append(eid)
                cols.append(pid)
                vals.append(float(sgn))
        b2 = sp.csr_matrix(
            (vals, (rows, cols)), shape=(edges.count, len(polygons))
        )

        return cls(
            dim=2,
            n_cells=[n_vertices, edges.count, len(polygons)],
            boundary=[None, b1, b2],
            edge_vertices=edges.as_array(),
            polygon_loops=[tuple(loop) for loop in polygons],
        )

    @classmethod
    def from_polyhedra(
        cls, n_vertices: int, cells: list[list[list[int]]]
    ) -> "CellComplex":
        """Build a 3D complex from polyhedral cells.

        Parameters
        ----------
        n_vertices
            Total number of vertices.
        cells
            ``cells[c]`` is a list of faces of cell ``c``; each face is an
            ordered loop of vertex ids, oriented consistently for that cell
            (right-hand rule w.r.t. the outward normal).
        """
        # 1) Facets: identify by vertex *set*; keep first-seen loop as canonical.
        facet_id: dict[frozenset[int], int] = {}
        facet_loops: list[tuple[int, ...]] = []
        cell_facets: list[list[tuple[int, int]]] = []
        for faces in cells:
            entry: list[tuple[int, int]] = []
            for loop in faces:
                key = frozenset(loop)
                if key not in facet_id:
                    facet_id[key] = len(facet_loops)
                    facet_loops.append(tuple(loop))
                    sign = 1
                else:
                    sign = _loop_orientation_sign(facet_loops[facet_id[key]], loop)
                entry.append((facet_id[key], sign))
            cell_facets.append(entry)

        # 2) Edges from the canonical facet loops.
        edges = _EdgeTable()
        for loop in facet_loops:
            for a, b in _loop_pairs(loop):
                edges.get(a, b)

        b1 = edges.vertex_incidence(n_vertices)

        # 3) boundary[2] : facets -> edges
        rows, cols, vals = [], [], []
        for fid, loop in enumerate(facet_loops):
            for a, b in _loop_pairs(loop):
                eid, sgn = edges.get(a, b)
                rows.append(eid)
                cols.append(fid)
                vals.append(float(sgn))
        b2 = sp.csr_matrix(
            (vals, (rows, cols)), shape=(edges.count, len(facet_loops))
        )

        # 4) boundary[3] : cells -> facets
        rows, cols, vals = [], [], []
        for cid, entry in enumerate(cell_facets):
            for fid, sgn in entry:
                rows.append(fid)
                cols.append(cid)
                vals.append(float(sgn))
        b3 = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(facet_loops), len(cells))
        )

        return cls(
            dim=3,
            n_cells=[n_vertices, edges.count, len(facet_loops), len(cells)],
            boundary=[None, b1, b2, b3],
            edge_vertices=edges.as_array(),
            polygon_loops=facet_loops,
        )

    # -- accessors -----------------------------------------------------------

    def num_cells(self, k: int) -> int:
        return self.n_cells[k]

    def boundary_matrix(self, k: int) -> sp.csr_matrix:
        """Signed incidence ``C_k -> C_{k-1}`` (``1 <= k <= dim``)."""
        if not 1 <= k <= self.dim:
            raise ValueError(f"boundary defined for 1..{self.dim}, got k={k}")
        return self.boundary[k]

    def _boundary_csc(self, k: int) -> sp.csc_matrix:
        """Column-oriented boundary matrix, cached (facet lookup is per-cell)."""
        if self._csc_cache is None:
            self._csc_cache = {}
        if k not in self._csc_cache:
            self._csc_cache[k] = self.boundary_matrix(k).tocsc()
        return self._csc_cache[k]

    def facets_of(self, k: int, cell_id: int) -> list[tuple[int, int]]:
        """``(facet_id, sign)`` pairs of the ``(k-1)``-facets of a k-cell."""
        m = self._boundary_csc(k)
        lo, hi = m.indptr[cell_id], m.indptr[cell_id + 1]
        return [
            (int(r), 1 if v > 0 else -1)
            for r, v in zip(m.indices[lo:hi], m.data[lo:hi])
        ]

    @property
    def cell_facets(self) -> list[list[tuple[int, int]]]:
        """Facets of every top-dimensional cell."""
        return [self.facets_of(self.dim, c) for c in range(self.n_cells[self.dim])]

    @property
    def facet_vertices(self) -> list[tuple[int, ...]]:
        """Vertex loops of the facets of the top-dimensional cells (3D only)."""
        return self.polygon_loops

    def cell_vertices(self, cell_id: int, k: int | None = None) -> set[int]:
        """Set of vertex ids of a k-cell (default: top dimension)."""
        k = self.dim if k is None else k
        if k == 0:
            return {cell_id}
        if k == 1:
            return set(int(v) for v in self.edge_vertices[cell_id])
        if k == 2:
            return set(self.polygon_loops[cell_id])
        verts: set[int] = set()
        for fid, _ in self.facets_of(k, cell_id):
            verts |= self.cell_vertices(fid, k - 1)
        return verts

    def is_simplex(self, cell_id: int, k: int | None = None) -> bool:
        """True if the k-cell is a simplex (``k+1`` vertices and ``k+1`` facets)."""
        k = self.dim if k is None else k
        if k == 0:
            return True
        if len(self.cell_vertices(cell_id, k)) != k + 1:
            return False
        return len(self.facets_of(k, cell_id)) == k + 1

    def verify_complex(self, atol: float = 1e-12) -> bool:
        """Check ``boundary[k] @ boundary[k+1] == 0`` for all k (dd = 0)."""
        for k in range(1, self.dim):
            prod = self.boundary[k] @ self.boundary[k + 1]
            if prod.nnz and np.abs(prod.data).max() > atol:
                return False
        return True


class _EdgeTable:
    """Deduplicating table of undirected edges, canonically oriented low->high."""

    def __init__(self) -> None:
        self._ids: dict[tuple[int, int], int] = {}
        self._list: list[tuple[int, int]] = []

    def get(self, a: int, b: int) -> tuple[int, int]:
        """Return ``(edge_id, sign)`` for the directed edge ``a -> b``."""
        key = (a, b) if a < b else (b, a)
        eid = self._ids.get(key)
        if eid is None:
            eid = len(self._list)
            self._ids[key] = eid
            self._list.append(key)
        return eid, (1 if a < b else -1)

    @property
    def count(self) -> int:
        return len(self._list)

    def as_array(self) -> np.ndarray:
        return np.array(self._list, dtype=np.int64).reshape(-1, 2)

    def vertex_incidence(self, n_vertices: int) -> sp.csr_matrix:
        """``boundary[1]``: edges -> vertices (``+1`` head, ``-1`` tail)."""
        rows, cols, vals = [], [], []
        for eid, (u, w) in enumerate(self._list):
            rows += [w, u]
            cols += [eid, eid]
            vals += [1.0, -1.0]
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_vertices, self.count))


def _loop_pairs(loop):
    """Consecutive (a, b) vertex pairs around a closed loop."""
    m = len(loop)
    return [(loop[i], loop[(i + 1) % m]) for i in range(m)]


def _loop_orientation_sign(canon: tuple[int, ...], cand: list[int]) -> int:
    """+1 if ``cand`` traverses the same cyclic direction as ``canon``, else -1.

    Both loops must enumerate the same vertex set.
    """
    idx = canon.index(cand[0])
    n = len(canon)
    if canon[(idx + 1) % n] == cand[1]:
        return 1
    if canon[(idx - 1) % n] == cand[1]:
        return -1
    raise ValueError("candidate loop is not a reorientation of the canonical loop")
