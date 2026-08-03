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
    apply_pseudo_inverse,
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


def cell_groups(mesh: Mesh):
    """Group cell ids by facet count.

    Yields ``(facet_ids, signs, cell_ids)`` with ``facet_ids`` and ``signs`` of
    shape ``(n_cells_in_group, n_facets)``.  Grouping is what makes the local
    matrices batchable: cells with equal facet counts have identical array
    shapes, so a whole group goes through one set of stacked NumPy
    linear-algebra calls instead of a Python loop.

    A free function because it is pure topology -- every facet-DOF inner product
    needs the same traversal, whatever it then does with it.
    """
    csc = mesh.complex._boundary_csc(mesh.dim)
    counts = np.diff(csc.indptr)
    for nf in np.unique(counts):
        cells = np.where(counts == nf)[0]
        starts = csc.indptr[cells][:, None] + np.arange(nf)
        yield csc.indices[starts], np.sign(csc.data[starts]), cells


class ElasticityInnerProduct:
    """Stress inner product for elasticity on facet DOFs (``d^2`` per facet)."""

    def __init__(
        self,
        mesh: Mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        material: "Material | None" = None,
    ) -> None:
        from mimetika.materials import Material, poisson_from_lame

        self.mesh = mesh
        d = mesh.dim
        if material is None:
            material = Material(
                shear_modulus=mu, poisson=poisson_from_lame(mu, lam, d)
            )
        self.material = material.expand(mesh.num_cells(d))
        # per-cell shear modulus and compliance coefficient; both stay finite at
        # nu = 1/2, which is what makes the incompressible limit unremarkable
        self._mu = np.broadcast_to(
            self.material.shear_modulus, (mesh.num_cells(d),)
        )
        self._a = np.broadcast_to(
            self.material.compliance_coefficient(d), (mesh.num_cells(d),)
        )
        self.mu = float(np.mean(self._mu))
        self.lam = lam
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

    def facet_data(self, lc: LocalCell, scale: float):
        """Small per-facet tensors that every local matrix is built from.

        Returns ``(moments, expansions)`` where

        * ``moments[i]``   is ``(nb, d+1)`` with ``\\int_{e_i} b phi_s`` -- the
          moments of the facet basis against the mode scalars ``{1, xi/h}``;
        * ``expansions[i]`` is ``(nb, d)`` with the ``L^2(e_i)`` expansion
          coefficients of the coordinate ``xi_c`` in the facet basis.

        Both are tiny, and they are all the geometry the tensor-product mode
        structure needs.
        """
        d, nf, nb = lc.dim, lc.n_facets, lc.dim
        if d != 3:
            return self._facet_data_quadrature(lc, scale)

        # Closed forms.  With the facet basis b_0 = 1, b_a = ((xi - xi_e).t_a)/h
        # centred on the facet centroid, every integral collapses onto the facet
        # area A, its centroid offset xi_e, and its second moment S:
        #     int_e b_0        = A            int_e b_a       = 0
        #     int_e b_0 xi     = A xi_e       int_e b_a xi    = (t_a S)/h
        #     gram_00 = A      gram_0a = 0    gram_ab = (t_a S t_b)/h^2
        A = lc.facet_measures
        h = np.sqrt(A)
        xi_e = lc.facet_centroids
        t = np.stack(lc.facet_tangents)  # (nf, 2, 3)
        S = self.mesh.geometry.facet_second_moments()[lc.facet_ids]

        W = np.einsum("iad,idc->iac", t, S) / h[:, None, None]  # (nf, 2, 3)
        T = np.einsum("iac,ibc->iab", W, t) / h[:, None, None]  # (nf, 2, 2)

        mom_x = np.empty((nf, nb, d))
        mom_x[:, 0] = A[:, None] * xi_e
        mom_x[:, 1:] = W

        moments = np.zeros((nf, nb, d + 1))
        moments[:, 0, 0] = A
        moments[:, :, 1:] = mom_x / scale

        X = np.empty((nf, nb, d))
        X[:, 0] = xi_e
        X[:, 1:] = np.linalg.solve(T, W)  # 2x2 systems
        return moments, X

    def _facet_data_quadrature(self, lc: LocalCell, scale: float):
        """Reference implementation by quadrature (used for ``d != 3``)."""
        d, nf, nb = lc.dim, lc.n_facets, lc.dim
        moments = np.empty((nf, nb, d + 1))
        grams = np.empty((nf, nb, nb))
        mom_x = np.empty((nf, nb, d))
        for i in range(nf):
            B, qw = lc.facet_scalar_basis(i)
            wB = qw[:, None] * B
            grams[i] = B.T @ wB
            mom_x[i] = wB.T @ lc.facet_quadrature[i][0]
            moments[i, :, 0] = wB.sum(0)
        moments[:, :, 1:] = mom_x / scale
        return moments, np.linalg.solve(grams, mom_x)

    def local_matrices(self, cell_id: int, with_facet_data: bool = False):
        """Return ``(N, R, Kbar, volume, lc)`` for one cell, in the local frame.

        Built from the **tensor-product structure** of the reconstruction space
        rather than by sampling dense mode arrays.  With modes
        ``T_{(s,r,c)} = phi_s E_{rc}`` and ``C^{-1}T = (T - a tr(T) I)/2mu``:

            ``N[(i,k,b),(s,r,c)]  = delta_kr n_i[c] moments[i][b,s]``
            ``R[(i,k,b),(r,c)]    = (delta_kr X_i[b,c] - a delta_rc X_i[b,k])/2mu``
            ``Kbar                = kron(G, (I - a vec(I) vec(I)^T)/2mu)``

        with ``G_ss' = (1/|E|) int_E phi_s phi_s'``.  Every object on the right
        is tiny, so no ``(nq, m, d, d)`` array is ever formed.
        """
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        d, vol = lc.dim, lc.volume
        scale = self._scale(lc)
        a, mu = float(self._a[cell_id]), float(self._mu[cell_id])
        eye = np.eye(d)

        moments, X = self.facet_data(lc, scale)

        # ---- N ---------------------------------------------------------------
        N = np.einsum(
            "kr,ic,ibs->ikbsrc", eye, lc.facet_normals, moments
        ).reshape(lc.n_facets * d * d, -1)

        # ---- R: canonical columns (the d^2 constant modes come first) -------
        R_canon = (
            np.einsum("kr,ibc->ikbrc", eye, X) - a * np.einsum("rc,ibk->ikbrc", eye, X)
        ).reshape(lc.n_facets * d * d, d * d) / (2 * mu)

        # ---- Kbar ------------------------------------------------------------
        phi = np.column_stack([np.ones(len(lc.quad_points)), lc.quad_points / scale])
        G = (phi.T @ (lc.quad_weights[:, None] * phi)) / vol  # (d+1, d+1)
        dvec = eye.reshape(-1)  # vec(I) picks out the trace
        block = (np.eye(d * d) - a * np.outer(dvec, dvec)) / (2 * mu)
        Kbar = np.kron(G, block)

        R = complete_moments(N, Kbar, vol, R_canon)
        if with_facet_data:
            return N, R, Kbar, vol, lc, X
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

    # -- batched construction (many cells at once) ----------------------------

    def cell_groups(self):
        """Group cell ids by facet count; see :func:`cell_groups`."""
        return cell_groups(self.mesh)

    def local_matrices_batched(self, facet_ids, signs, cell_ids):
        """``(N, R, Kbar, vol, X)`` for a whole group of equal-shaped cells.

        Mathematically identical to :meth:`local_matrices`, expressed with
        stacked arrays.  Only 3D is supported; callers fall back to the per-cell
        path otherwise.
        """
        g = self.mesh.geometry
        d, nb, ndf = 3, 3, 9
        nB, nf = facet_ids.shape
        a = self._a[cell_ids]  # (nB,)
        mu = self._mu[cell_ids]
        eye = np.eye(d)

        vol = g.measure(3)[cell_ids]
        scale = vol ** (1.0 / 3.0)
        xi_e = g.centroids(2)[facet_ids] - g.centroids(3)[cell_ids][:, None]
        n_out = signs[..., None] * g.facet_normals()[facet_ids]
        A = g.measure(2)[facet_ids]
        h = np.sqrt(A)
        t = g.facet_tangent_frames()[facet_ids]  # (nB, nf, 2, 3)
        S = g.facet_second_moments()[facet_ids]  # (nB, nf, 3, 3)

        # facet moments and coordinate expansions, in closed form
        W = np.einsum("bfad,bfdc->bfac", t, S) / h[..., None, None]
        T = np.einsum("bfac,bfec->bfae", W, t) / h[..., None, None]
        mom_x = np.concatenate([(A[..., None] * xi_e)[:, :, None, :], W], axis=2)
        moments = np.zeros((nB, nf, nb, d + 1))
        moments[:, :, 0, 0] = A
        moments[:, :, :, 1:] = mom_x / scale[:, None, None, None]
        X = np.concatenate(
            [xi_e[:, :, None, :], np.linalg.solve(T, W)], axis=2
        )  # (nB, nf, nb, d)

        N = np.einsum("kr,bfc,bfes->bfkesrc", eye, n_out, moments).reshape(
            nB, nf * ndf, -1
        )
        R_canon = (
            np.einsum("kr,bfec->bfkerc", eye, X)
            - a[:, None, None, None, None, None]
            * np.einsum("rc,bfek->bfkerc", eye, X)
        ).reshape(nB, nf * ndf, d * d) / (2.0 * mu)[:, None, None]

        # Kbar = kron(G, block); the first moment vanishes at the centroid, so G
        # is the identity bordered by the cell inertia tensor
        G = np.zeros((nB, d + 1, d + 1))
        G[:, 0, 0] = 1.0
        G[:, 1:, 1:] = g.cell_second_moments()[cell_ids] / (
            vol[:, None, None] * scale[:, None, None] ** 2
        )
        dvec = eye.reshape(-1)
        block = (
            np.eye(d * d)[None] - a[:, None, None] * np.outer(dvec, dvec)[None]
        ) / (2 * mu)[:, None, None]  # (nB, d^2, d^2), per cell
        Kbar = np.einsum("bij,bkl->bikjl", G, block).reshape(nB, 36, 36)

        # min-norm completion of the non-canonical columns
        target = vol[:, None, None] * Kbar[:, :, d * d :]
        R_rest = N @ np.linalg.solve(np.einsum("bij,bik->bjk", N, N), target)
        return N, np.concatenate([R_canon, R_rest], axis=2), Kbar, vol, X

    def local_inner_products_batched(self, N, R, Kbar, vol):
        """Batched ``M_E = M1 + M2`` for a group of cells."""
        M1 = (
            R @ apply_pseudo_inverse(Kbar, np.swapaxes(R, 1, 2))
        ) / vol[:, None, None]
        Q, Rm = np.linalg.qr(N)
        diag = np.abs(np.diagonal(Rm, axis1=1, axis2=2))
        deficient = diag.min(axis=1) <= 1e-12 * diag.max(axis=1) * max(N.shape[1:])

        s = M1.diagonal(axis1=1, axis2=2).mean(axis=1)
        M = M1 - s[:, None, None] * (Q @ np.swapaxes(Q, 1, 2))
        idx = np.arange(M.shape[1])
        M[:, idx, idx] += s[:, None]
        return 0.5 * (M + np.swapaxes(M, 1, 2)), deficient
