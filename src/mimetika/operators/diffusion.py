r"""Mimetic inner product for the Laplace / diffusion operator.

Discretises the flux inner product ``(F, G)_X = \int_E K^{-1} F . G`` on the
facet (flux, 2-form) degrees of freedom -- one normal flux per facet.  This is
the matrix that turns the metric-free ``div`` into the weak Laplacian
``div M^{-1} grad`` in the mixed formulation.

Reconstruction space
--------------------
* ``basis="const"`` (default): the three constant flux fields.  ``m = 3``.
  This is the classic lowest-order mimetic diffusion inner product; it carries a
  nontrivial stabilization on *every* polytope with more than 3 facets,
  including tetrahedra.
* ``basis="rt0"``: the lowest-order Raviart--Thomas space ``{a + b x}``,
  ``m = 4``.  On a tetrahedron (4 facets) this is unisolvent, ``D = m = 4``, so
  the stabilization vanishes and the method coincides with RT0 mixed FE; on
  hexahedra (6 facets) a 2-dimensional stabilization remains.

Orientation is automatic: building ``N`` with the canonical facet normals yields
``M_E`` directly in the global (canonical-normal) DOF convention, because the
consistency term is covariant under per-facet sign flips.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.mesh.mesh import Mesh
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    consistency_matrix,
    stabilization_dim,
)


class DiffusionInnerProduct:
    """Global flux inner product for the diffusion operator on facet DOFs."""

    def __init__(
        self, mesh: Mesh, K: np.ndarray | None = None, basis: str = "const"
    ) -> None:
        self.mesh = mesh
        self.K = np.eye(3) if K is None else np.asarray(K, dtype=float)
        self.Kinv = np.linalg.inv(self.K)
        if basis not in ("const", "rt0"):
            raise ValueError("basis must be 'const' or 'rt0'")
        self.basis = basis

    # -- reconstruction modes -----------------------------------------------

    def _modes(self, x: np.ndarray, xE: np.ndarray) -> np.ndarray:
        """Evaluate the flux reconstruction modes at points ``x`` (shape (Nq,3)).

        Returns ``(Nq, m, 3)`` -- for each point, ``m`` vector-valued modes.
        """
        nq = len(x)
        const = np.broadcast_to(np.eye(3), (nq, 3, 3))  # (Nq, 3 modes, 3 comp)
        if self.basis == "const":
            return const
        radial = (x - xE)[:, None, :]  # (Nq, 1, 3)
        return np.concatenate([const, radial], axis=1)  # (Nq, 4, 3)

    # -- local matrix -------------------------------------------------------

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """Return ``(M_E, facet_ids)`` for one cell."""
        g = self.mesh.geometry
        cx = self.mesh.complex
        entry = cx.cell_facets[cell_id]
        facet_ids = [fid for fid, _ in entry]

        normals = g.facet_normals()
        fcent = g.facet_centroids()
        xE = g.centroids(3)[cell_id]
        vol = g.measure(3)[cell_id]

        # N[i, j] = mode_j(x_{e_i}) . n_{e_i}   (canonical normals)
        modes_at_faces = self._modes(fcent[facet_ids], xE)  # (D, m, 3)
        N = np.einsum("imc,ic->im", modes_at_faces, normals[facet_ids])

        # Kbar[j, l] = (1/|E|) \int_E K^{-1} w_j . w_l
        qp, qw = g.cell_quadrature(cell_id)
        modes_q = self._modes(qp, xE)  # (Nq, m, 3)
        Kw = np.einsum("cd,imd->imc", self.Kinv, modes_q)  # K^{-1} w_l at qp
        Kbar = np.einsum("q,qjc,qlc->jl", qw, modes_q, Kw) / vol

        M1 = consistency_matrix(N, Kbar, vol)
        scale = float(np.mean(np.diag(M1)))
        M = assemble_local_inner_product(N, Kbar, vol, scale)
        return M, facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """Dimension of the stabilization space on one cell (0 = no stab.)."""
        g = self.mesh.geometry
        entry = self.mesh.complex.cell_facets[cell_id]
        facet_ids = [fid for fid, _ in entry]
        xE = g.centroids(3)[cell_id]
        modes = self._modes(g.facet_centroids()[facet_ids], xE)
        N = np.einsum("imc,ic->im", modes, g.facet_normals()[facet_ids])
        return stabilization_dim(N)

    # -- global assembly ----------------------------------------------------

    def assemble(self) -> sp.csr_matrix:
        """Assemble the global flux inner product over all facet DOFs."""
        n = self.mesh.num_cells(2)
        rows, cols, vals = [], [], []
        for cid in range(self.mesh.num_cells(3)):
            M, fids = self.local(cid)
            for a, fa in enumerate(fids):
                for b, fb in enumerate(fids):
                    rows.append(fa)
                    cols.append(fb)
                    vals.append(M[a, b])
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
