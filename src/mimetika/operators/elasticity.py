r"""Mimetic inner product for linear elasticity (Mimetic-AFW).

Discretises the stress inner product ``(\sigma, \tau) = \int_E C^{-1}\sigma:\tau``
of the Hellinger--Reissner (mixed, weakly-symmetric) formulation of elasticity,
following Beir\~ao da Veiga, ESAIM M2AN 44 (2010) 231--250.  This is the
mimetic counterpart of the Arnold--Falk--Winter mixed finite element and shares
its algebraic (saddle-point) structure.

Degrees of freedom
------------------
Per facet ``e`` the DOFs are the moments of the normal traction ``\tau n_e``
against a basis of linear vector fields ``[P_1(e)]^3`` on the facet -- 9 DOFs
per facet (a constant vector plus two in-plane linear slopes; eq. (2.8)-(2.11)).

Reconstruction space and the simplex property
--------------------------------------------
The consistency term reproduces the exact inner product on a reconstruction
space of tensor fields.  Here that space is the **full linear tensor space**
``[P_1(E)]^{3x3}`` (``m = 9 x 4 = 36`` modes).  On a tetrahedron the DOF count
is ``D = 9 x 4 = 36 = m`` and the DOFs are unisolvent for linear tensors, so
``ker(N^T) = {0}`` and **the stabilization vanishes** -- the method reduces to
the AFW (BDM_1-based) mixed element.  On hexahedra (``D = 54``) an 18-dimensional
stabilization remains.

(Beir\~ao's original scheme uses only the 9 modes ``C\nabla p``, ``p`` linear,
which is cheaper but keeps a stabilization on every element including simplices.
Enriching to the full linear tensor space is exactly the design -- anticipated
in that paper's Remark 4.1 -- that makes the stabilization simplex-consistent.)

The compliance tensor (isotropic) is, from eq. (2.4),

    ``C^{-1} tau = (1/2mu) tau - (lambda / (2mu(2mu+3lambda))) tr(tau) I`` .
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

DOFS_PER_FACET = 9


def compliance_contraction(T: np.ndarray, S: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """``C^{-1}T : S`` for stacked tensors ``T, S`` of shape ``(..., 3, 3)``."""
    TS = np.einsum("...ij,...ij->...", T, S)
    trT = np.einsum("...ii->...", T)
    trS = np.einsum("...ii->...", S)
    return TS / (2 * mu) - lam / (2 * mu * (2 * mu + 3 * lam)) * trT * trS


class ElasticityInnerProduct:
    """Global stress inner product for elasticity on facet (9-per-facet) DOFs."""

    def __init__(self, mesh: Mesh, mu: float = 1.0, lam: float = 1.0) -> None:
        self.mesh = mesh
        self.mu = float(mu)
        self.lam = float(lam)

    # -- reconstruction modes: full linear tensor fields [P_1(E)]^{3x3} ------

    def _cell_diameter(self, cell_id: int) -> float:
        verts = np.array(sorted(self.mesh.complex.cell_vertices(cell_id)))
        p = self.mesh.geometry.points[verts]
        d = p[:, None, :] - p[None, :, :]
        return float(np.sqrt((d**2).sum(-1)).max())

    def _eval_modes(self, x: np.ndarray, xE: np.ndarray, hE: float) -> np.ndarray:
        """Return ``(Nq, 36, 3, 3)``: the linear tensor basis at points ``x``.

        Mode = ``scalar * E_{rc}`` with scalar in ``{1, X0, X1, X2}``,
        ``Xk = (x - xE)_k / hE``, and ``E_{rc}`` the (r,c) matrix unit.
        """
        nq = len(x)
        X = np.concatenate([np.ones((nq, 1)), (x - xE) / hE], axis=1)  # (Nq,4)
        modes = np.zeros((nq, DOFS_PER_FACET * 4, 3, 3))
        idx = 0
        for r in range(3):
            for c in range(3):
                for sidx in range(4):
                    modes[:, idx, r, c] = X[:, sidx]
                    idx += 1
        return modes

    # -- DOF functionals -----------------------------------------------------

    def _facet_dofs_of_modes(self, cell_id: int, fid: int) -> np.ndarray:
        r"""``(9, 36)`` block: DOFs on facet ``fid`` of every reconstruction mode.

        DOF ``(k, bidx)`` of a stress field ``T`` is
        ``\int_e b_{bidx}(x) (T n_e)_k dx`` with ``b_0=1, b_1=s1, b_2=s2`` the
        in-plane linear scalar basis.
        """
        g = self.mesh.geometry
        n_e = g.facet_normals()[fid]
        t1, t2 = g.facet_tangents(fid)
        xe = g.facet_centroids()[fid]
        he = self._facet_diameter(fid)
        xE = g.centroids(3)[cell_id]
        hE = self._cell_diameter(cell_id)

        qp, qw = g.facet_quadrature(fid)
        b = np.stack(
            [np.ones(len(qp)), (qp - xe) @ t1 / he, (qp - xe) @ t2 / he], axis=1
        )  # (Nq, 3)
        modes = self._eval_modes(qp, xE, hE)  # (Nq, 36, 3, 3)
        Tn = np.einsum("qmij,j->qmi", modes, n_e)  # (Nq, 36, 3) = T n_e

        # N9[k, bidx, mode] = sum_q qw * b_bidx * (T n_e)_k
        N9 = np.einsum("q,qb,qmk->kbm", qw, b, Tn)  # (3, 3, 36)
        return N9.reshape(DOFS_PER_FACET, -1)  # (9, 36), rows ordered k*3+bidx

    def _facet_diameter(self, fid: int) -> float:
        loop = self.mesh.complex.facet_vertices[fid]
        p = self.mesh.geometry.points[list(loop)]
        d = p[:, None, :] - p[None, :, :]
        return float(np.sqrt((d**2).sum(-1)).max())

    # -- local matrix -------------------------------------------------------

    def _N_and_Kbar(self, cell_id: int):
        g = self.mesh.geometry
        entry = self.mesh.complex.cell_facets[cell_id]
        facet_ids = [fid for fid, _ in entry]
        xE = g.centroids(3)[cell_id]
        hE = self._cell_diameter(cell_id)
        vol = g.measure(3)[cell_id]

        N = np.vstack([self._facet_dofs_of_modes(cell_id, fid) for fid in facet_ids])

        qp, qw = g.cell_quadrature(cell_id)
        modes = self._eval_modes(qp, xE, hE)  # (Nq, 36, 3, 3)
        # Kbar[j,l] = (1/|E|) sum_q qw * C^{-1} T_j : T_l
        Ti = modes[:, :, None]  # (Nq,36,1,3,3)
        Tj = modes[:, None, :]  # (Nq,1,36,3,3)
        integrand = compliance_contraction(Ti, Tj, self.mu, self.lam)  # (Nq,36,36)
        Kbar = np.einsum("q,qjl->jl", qw, integrand) / vol
        return N, Kbar, vol, facet_ids

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """Return ``(M_E, facet_ids)`` for one cell (M_E is ``9n x 9n``)."""
        N, Kbar, vol, facet_ids = self._N_and_Kbar(cell_id)
        M1 = consistency_matrix(N, Kbar, vol)
        scale = float(np.mean(np.diag(M1)))
        M = assemble_local_inner_product(N, Kbar, vol, scale)
        return M, facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """Dimension of the stabilization space on one cell (0 on simplices)."""
        N, _, _, _ = self._N_and_Kbar(cell_id)
        return stabilization_dim(N)

    # -- global assembly ----------------------------------------------------

    def assemble(self) -> sp.csr_matrix:
        """Assemble the global stress inner product (9 DOFs per facet)."""
        n = DOFS_PER_FACET * self.mesh.num_cells(2)
        rows, cols, vals = [], [], []
        for cid in range(self.mesh.num_cells(3)):
            M, fids = self.local(cid)
            gdofs = np.concatenate(
                [DOFS_PER_FACET * fid + np.arange(DOFS_PER_FACET) for fid in fids]
            )
            for a in range(len(gdofs)):
                for b in range(len(gdofs)):
                    rows.append(gdofs[a])
                    cols.append(gdofs[b])
                    vals.append(M[a, b])
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
