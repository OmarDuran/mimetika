"""Combinatorial cell complex and signed incidence (boundary) operators.

This layer is *purely combinatorial*: it knows nothing about coordinates,
lengths, areas or volumes.  It stores, for a graded complex of dimension ``d``,
the signed boundary operators

    ``d_k : C_k -> C_{k-1}``   (``boundary[k]``, shape ``(n_{k-1}, n_k)``)

for ``k = 1 .. d``.  The discrete exterior derivative on k-forms is the
transpose (coboundary) ``D_k = boundary[k+1].T`` and is provided by the
:mod:`mimetika.operators` layer.

The whole complex is built from a *polytopal* description: a list of cells,
each given as a list of faces, each face an ordered loop of vertex ids with a
globally consistent orientation (e.g. outward-normal / counter-clockwise seen
from outside the cell).  Edges and facets are deduced automatically with
consistent signs, so the fundamental mimetic identity

    ``boundary[k] @ boundary[k+1] == 0``

holds *exactly*, by construction, for any (well-formed) polyhedral mesh.
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
        Topological dimension ``d`` (3 for a volumetric mesh).
    n_cells
        ``n_cells[k]`` = number of k-cells, for ``k = 0 .. dim``.
    boundary
        ``boundary[k]`` = signed incidence ``C_k -> C_{k-1}`` as a CSR matrix of
        shape ``(n_cells[k-1], n_cells[k])``, for ``k = 1 .. dim``.
        ``boundary[0]`` is ``None``.
    edge_vertices
        ``(n_edges, 2)`` array; each row ``(u, w)`` with ``u < w`` is an edge
        oriented from ``u`` to ``w``.
    facet_vertices
        List of canonical vertex loops, one per facet ((dim-1)-cell).
    cell_facets
        List, per cell, of ``(facet_id, sign)`` pairs.
    """

    dim: int
    n_cells: list[int]
    boundary: list[sp.csr_matrix | None]
    edge_vertices: np.ndarray
    facet_vertices: list[tuple[int, ...]] = field(default_factory=list)
    cell_facets: list[list[tuple[int, int]]] = field(default_factory=list)

    # -- construction --------------------------------------------------------

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
        n_facets = len(facet_loops)

        # 2) Edges: from consecutive vertices of every canonical facet loop.
        edge_id: dict[tuple[int, int], int] = {}
        edge_list: list[tuple[int, int]] = []

        def get_edge(a: int, b: int) -> tuple[int, int]:
            """Return (edge_id, sign) for directed edge a->b (canonical low->high)."""
            key = (a, b) if a < b else (b, a)
            eid = edge_id.get(key)
            if eid is None:
                eid = len(edge_list)
                edge_id[key] = eid
                edge_list.append(key)
            return eid, (1 if a < b else -1)

        for loop in facet_loops:
            m = len(loop)
            for i in range(m):
                get_edge(loop[i], loop[(i + 1) % m])
        n_edges = len(edge_list)

        # 3) boundary[1] : edges -> vertices   (w - u for edge u->w)
        rows, cols, vals = [], [], []
        for eid, (u, w) in enumerate(edge_list):
            rows += [w, u]
            cols += [eid, eid]
            vals += [1.0, -1.0]
        b1 = sp.csr_matrix((vals, (rows, cols)), shape=(n_vertices, n_edges))

        # 4) boundary[2] : facets -> edges  (oriented edges around each loop)
        rows, cols, vals = [], [], []
        for fid, loop in enumerate(facet_loops):
            m = len(loop)
            for i in range(m):
                eid, sgn = get_edge(loop[i], loop[(i + 1) % m])
                rows.append(eid)
                cols.append(fid)
                vals.append(float(sgn))
        b2 = sp.csr_matrix((vals, (rows, cols)), shape=(n_edges, n_facets))

        # 5) boundary[3] : cells -> facets
        rows, cols, vals = [], [], []
        for cid, entry in enumerate(cell_facets):
            for fid, sgn in entry:
                rows.append(fid)
                cols.append(cid)
                vals.append(float(sgn))
        b3 = sp.csr_matrix((vals, (rows, cols)), shape=(n_facets, len(cells)))

        return cls(
            dim=3,
            n_cells=[n_vertices, n_edges, n_facets, len(cells)],
            boundary=[None, b1, b2, b3],
            edge_vertices=np.array(edge_list, dtype=np.int64).reshape(-1, 2),
            facet_vertices=facet_loops,
            cell_facets=cell_facets,
        )

    # -- accessors -----------------------------------------------------------

    def num_cells(self, k: int) -> int:
        return self.n_cells[k]

    def boundary_matrix(self, k: int) -> sp.csr_matrix:
        """Signed incidence ``C_k -> C_{k-1}`` (``1 <= k <= dim``)."""
        if not 1 <= k <= self.dim:
            raise ValueError(f"boundary defined for 1..{self.dim}, got k={k}")
        return self.boundary[k]

    def cell_vertices(self, cell_id: int) -> set[int]:
        """Set of vertex ids of a 3-cell (union over its facets)."""
        verts: set[int] = set()
        for fid, _ in self.cell_facets[cell_id]:
            verts.update(self.facet_vertices[fid])
        return verts

    def is_simplex(self, cell_id: int) -> bool:
        """True if the 3-cell is a tetrahedron (4 triangular facets, 4 vertices)."""
        entry = self.cell_facets[cell_id]
        if len(entry) != 4 or len(self.cell_vertices(cell_id)) != 4:
            return False
        return all(len(self.facet_vertices[fid]) == 3 for fid, _ in entry)

    def verify_complex(self, atol: float = 1e-12) -> bool:
        """Check ``boundary[k] @ boundary[k+1] == 0`` for all k (dd = 0)."""
        for k in range(1, self.dim):
            prod = self.boundary[k] @ self.boundary[k + 1]
            if prod.nnz and np.abs(prod.data).max() > atol:
                return False
        return True


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
