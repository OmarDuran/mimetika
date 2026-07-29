r"""Mimetic inner product for linear elasticity (Mimetic-AFW).

Discretises the stress inner product ``(sigma, tau) = \int_E C^{-1} sigma : tau``
of the Hellinger--Reissner (mixed, weakly-symmetric) formulation, following
Beir\~ao da Veiga, ESAIM M2AN 44 (2010) 231--250.  This is the mimetic
counterpart of the Arnold--Falk--Winter mixed finite element and shares its
algebraic (saddle-point) structure.

Degrees of freedom
------------------
Per facet ``e``, the moments of the normal traction ``tau n_e`` against the
``P_1`` basis on the facet: ``d`` components times ``d`` scalar basis functions
(the constant plus the ``d-1`` in-facet coordinates), i.e. ``d^2`` DOFs per
facet -- 9 per face in 3D, matching eqs (2.8)--(2.11).

Reconstruction space and the simplex property
---------------------------------------------
The reconstruction space is the **full linear tensor space** ``[P_1(E)]^{d x d}``
(``m = d^2 (d+1)`` modes).  On a simplex (``d+1`` facets) the DOF count is
``D = d^2 (d+1) = m``, so ``ker(N^T) = {0}`` and **the stabilization vanishes** --
the scheme reduces to the AFW (BDM_1-based) mixed element.  On genuine polytopes
a stabilization remains (e.g. dimension 18 on a hexahedron).

The moment matrix
-----------------
The ``d^2`` **constant** stress modes admit a potential: ``C^{-1} T = grad v``
with ``v`` linear, so integrating by parts gives the canonical columns

    ``R_{(e,k,b),j} = `` the coefficient of the facet basis function ``b`` in the
    expansion of ``(v_j)_k`` restricted to facet ``e``,

which is exactly what makes local mixed solves reproduce linear displacements.
The remaining (genuinely linear) modes have no potential and are completed by
the minimum-norm solution of ``N^T R = |E| Kbar``.

The compliance tensor (isotropic, ``d`` dimensions) is

    ``C^{-1} T = (1/2mu) [ T - lambda/(2mu + d lambda) tr(T) I ]`` ,

which reduces to eq. (2.4) when ``d = 3``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.geometry.local_cell import LocalCell, mesh_frame
from mimetika.mesh.mesh import Mesh
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    complete_moments,
    stabilization_dim,
)


def compliance(T: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Apply ``C^{-1}`` to stacked tensors ``T`` of shape ``(..., d, d)``."""
    T = np.asarray(T, dtype=float)
    d = T.shape[-1]
    tr = np.einsum("...ii->...", T)
    eye = np.eye(d)
    return (T - lam / (2 * mu + d * lam) * tr[..., None, None] * eye) / (2 * mu)


def compliance_contraction(
    T: np.ndarray, S: np.ndarray, mu: float, lam: float
) -> np.ndarray:
    """``C^{-1} T : S`` for stacked tensors of shape ``(..., d, d)``."""
    return np.einsum("...ij,...ij->...", compliance(T, mu, lam), S)


class ElasticityInnerProduct:
    """Stress inner product for elasticity on facet DOFs (``d^2`` per facet)."""

    def __init__(self, mesh: Mesh, mu: float = 1.0, lam: float = 1.0) -> None:
        self.mesh = mesh
        self.mu = float(mu)
        self.lam = float(lam)
        self.frame = mesh_frame(mesh.geometry)

    # -- sizes ----------------------------------------------------------------

    def dofs_per_facet(self, d: int) -> int:
        return d * d

    def n_modes(self, d: int) -> int:
        return d * d * (d + 1)

    # -- reconstruction modes -------------------------------------------------

    def _tensor_units(self, d: int) -> np.ndarray:
        """``(d^2, d, d)``: the matrix units ``E_rc`` in row-major order."""
        units = np.zeros((d * d, d, d))
        for r in range(d):
            for c in range(d):
                units[r * d + c, r, c] = 1.0
        return units

    def _eval_modes(self, xi: np.ndarray, d: int, scale: float) -> np.ndarray:
        """``(nq, m, d, d)`` linear tensor basis; the ``d^2`` constants come first."""
        units = self._tensor_units(d)
        scalars = np.column_stack([np.ones(len(xi)), xi / scale])  # (nq, d+1)
        # mode index = s * d^2 + u  ->  constants (s = 0) occupy the first block
        return np.einsum("qs,uij->qsuij", scalars, units).reshape(
            len(xi), -1, d, d
        )

    # -- local matrices -------------------------------------------------------

    def _scale(self, lc: LocalCell) -> float:
        return float(lc.volume ** (1.0 / lc.dim))

    def local_matrices(self, cell_id: int):
        """Return ``(N, R, Kbar, volume, lc)`` for one cell, in the local frame."""
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        d, vol = lc.dim, lc.volume
        nb = d  # scalar basis functions per facet
        ndf = self.dofs_per_facet(d)
        scale = self._scale(lc)

        # ---- N: moments of the normal traction of each mode ----------------
        blocks = []
        for i in range(lc.n_facets):
            B, qw = lc.facet_scalar_basis(i)  # (nq, nb), (nq,)
            modes = self._eval_modes(lc.facet_quadrature[i][0], d, scale)
            Tn = np.einsum("qmij,j->qmi", modes, lc.facet_normals[i])  # (nq, m, d)
            blocks.append(
                np.einsum("q,qb,qmk->kbm", qw, B, Tn).reshape(ndf, -1)
            )
        N = np.vstack(blocks)

        # ---- Kbar: Gram matrix of the modes --------------------------------
        modes_q = self._eval_modes(lc.quad_points, d, scale)  # (nq, m, d, d)
        CiT = compliance(modes_q, self.mu, self.lam)
        Kbar = np.einsum(
            "q,qjab,qlab->jl", lc.quad_weights, CiT, modes_q
        ) / vol

        # ---- R: canonical columns for the d^2 constant modes ---------------
        # C^{-1} T_j is constant, so v_j(xi) = (C^{-1} T_j) xi is linear with
        # zero element mean (the local origin is the centroid).
        A = compliance(self._tensor_units(d), self.mu, self.lam)  # (d^2, d, d)
        rows = []
        for i in range(lc.n_facets):
            qp = lc.facet_quadrature[i][0]
            v = np.einsum("jkc,qc->qkj", A, qp)  # (nq, d, d^2) = (v_j)_k
            coeff = lc.expand_on_facet(i, v)  # (nb, d, d^2)
            rows.append(np.einsum("bkj->kbj", coeff).reshape(ndf, -1))
        R = complete_moments(N, Kbar, vol, np.vstack(rows))

        return N, R, Kbar, vol, lc

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(M_E, facet_ids)`` in the *global* (canonical-orientation) DOF basis."""
        N, R, Kbar, vol, lc = self.local_matrices(cell_id)
        M = assemble_local_inner_product(N, R, Kbar, vol)
        s = np.repeat(lc.signs, self.dofs_per_facet(lc.dim))
        return M * s[:, None] * s[None, :], lc.facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """Dimension of the stabilization space on one cell (0 on simplices)."""
        N, _, _, _, _ = self.local_matrices(cell_id)
        return stabilization_dim(N)

    # -- global assembly ------------------------------------------------------

    def assemble(self) -> sp.csr_matrix:
        """Assemble the global stress inner product."""
        d = self.mesh.dim
        ndf = self.dofs_per_facet(d)
        n = ndf * self.mesh.num_cells(d - 1)
        rows, cols, vals = [], [], []
        for cid in range(self.mesh.num_cells(d)):
            M, fids = self.local(cid)
            g = (ndf * np.asarray(fids)[:, None] + np.arange(ndf)).ravel()
            rows.append(np.repeat(g, len(g)))
            cols.append(np.tile(g, len(g)))
            vals.append(M.ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
