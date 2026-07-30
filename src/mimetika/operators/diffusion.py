r"""Mimetic inner product for the Laplace / diffusion operator.

Discretises the flux inner product ``(F, G)_X = \int_E K^{-1} F . G`` on the
facet (flux) degrees of freedom -- one average normal flux per facet.  This is
the matrix that turns the metric-free ``div`` into the weak Laplacian in the
mixed formulation.

Works for cells of any dimension: the DOFs of a ``d``-cell live on its
``(d-1)``-facets (endpoints of a segment, edges of a polygon, faces of a
polyhedron), and everything is computed in the cell's own affine frame (see
:class:`~mimetika.geometry.local_cell.LocalCell`).

Reconstruction space
--------------------
* ``basis="const"`` (default): the ``d`` constant flux fields.  The classic
  lowest-order mimetic diffusion inner product.
* ``basis="rt0"``: lowest-order Raviart--Thomas ``{a + b xi}``, ``m = d + 1``.
  On a simplex (``d+1`` facets) this is unisolvent, so ``D = m`` and the
  stabilization vanishes -- the scheme coincides with RT0 mixed FE.

Both mode families admit a potential, so **every** column of the moment matrix
``R`` is canonical here (``R_e = |e| (psi_e - psi_E)`` with ``grad psi = K^{-1} w``):
the constants give the classical closed form ``R_e = |e| K^{-1}(x_e - x_E)`` and
the radial mode a quadratic potential.  Hence ``M N = R`` holds exactly and
local mixed solves reproduce linear potentials exactly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.geometry.local_cell import LocalCell, mesh_frame
from mimetika.mesh.mesh import Mesh
from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    stabilization_dim,
)


class DiffusionInnerProduct:
    """Flux inner product for the diffusion operator on facet DOFs."""

    def __init__(
        self, mesh: Mesh, K: np.ndarray | None = None, basis: str = "const"
    ) -> None:
        self.mesh = mesh
        # K is either one ambient 3x3 tensor for the whole mesh, or one per cell
        # (shape (n_cells, 3, 3)) -- high-contrast media need the latter
        K = np.eye(3) if K is None else np.asarray(K, dtype=float)
        if K.ndim == 3 and len(K) != mesh.num_cells(mesh.dim):
            raise ValueError("per-cell K must have one tensor per cell")
        self.K = K
        if basis not in ("const", "rt0"):
            raise ValueError("basis must be 'const' or 'rt0'")
        self.basis = basis
        # one common frame for the whole mesh, so neighbouring cells agree on
        # the meaning of a shared facet's DOFs
        self.frame = mesh_frame(mesh.geometry)

    # -- reconstruction modes -------------------------------------------------

    def n_modes(self, d: int) -> int:
        return d + 1 if self.basis == "rt0" else d

    def _modes(self, xi: np.ndarray, d: int) -> np.ndarray:
        """``(nq, m, d)``: the flux modes evaluated at local points ``xi``."""
        nq = len(xi)
        const = np.broadcast_to(np.eye(d), (nq, d, d))
        if self.basis == "const":
            return const
        return np.concatenate([const, xi[:, None, :]], axis=1)

    def _potentials(self, xi: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
        """``(nq, m)``: potentials ``psi_j`` with ``grad psi_j = K^{-1} w_j``.

        Constant mode ``e_j`` -> ``psi = (K^{-1} e_j) . xi`` (linear);
        radial mode ``xi``    -> ``psi = 0.5 xi^T K^{-1} xi`` (quadratic).
        """
        lin = xi @ Kinv  # (nq, d) -- column j is (K^{-1} e_j) . xi
        if self.basis == "const":
            return lin
        quad = 0.5 * np.einsum("qi,ij,qj->q", xi, Kinv, xi)
        return np.column_stack([lin, quad])

    # -- local matrices -------------------------------------------------------

    def _cell_tensor(self, cell_id: int) -> np.ndarray:
        """The ambient conductivity of one cell."""
        return self.K[cell_id] if self.K.ndim == 3 else self.K

    def local_matrices(self, cell_id: int):
        """Return ``(N, R, Kbar, volume, lc)`` for one cell, in the local frame."""
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        d, vol = lc.dim, lc.volume
        Kloc = lc.project_tensor(self._cell_tensor(cell_id))
        Kinv = np.linalg.inv(Kloc)

        # N: average normal flux of each mode on each facet (planar facets, so
        # the normal trace of every mode is constant and the centroid suffices).
        modes_f = self._modes(lc.facet_centroids, d)  # (nf, m, d)
        N = np.einsum("imc,ic->im", modes_f, lc.facet_normals)

        # Kbar: Gram matrix of the modes in the continuous inner product.
        modes_q = self._modes(lc.quad_points, d)  # (nq, m, d)
        Kw = np.einsum("cd,qmd->qmc", Kinv, modes_q)
        Kbar = np.einsum("q,qjc,qlc->jl", lc.quad_weights, modes_q, Kw) / vol

        # R: canonical moments  R_e = |e| (mean_e psi - mean_E psi).
        psi_E = (lc.quad_weights @ self._potentials(lc.quad_points, Kinv)) / vol
        R = np.empty_like(N)
        for i in range(lc.n_facets):
            qp, qw = lc.facet_quadrature[i]
            psi_e = (qw @ self._potentials(qp, Kinv)) / lc.facet_measures[i]
            R[i] = lc.facet_measures[i] * (psi_e - psi_E)

        return N, R, Kbar, vol, lc

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(M_E, facet_ids)`` in the *global* (canonical-orientation) DOF basis."""
        N, R, Kbar, vol, lc = self.local_matrices(cell_id)
        M = assemble_local_inner_product(N, R, Kbar, vol)
        # local frame uses outward normals; convert to canonical facet normals
        S = lc.signs
        return M * S[:, None] * S[None, :], lc.facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """Dimension of the stabilization space on one cell (0 on simplices)."""
        N, _, _, _, _ = self.local_matrices(cell_id)
        return stabilization_dim(N)

    # -- global assembly ------------------------------------------------------

    def assemble(self, dofmap=None) -> sp.csr_matrix:
        """Assemble the global flux inner product over all facet DOFs.

        ``dofmap`` (a :class:`~mimetika.dof.facet_dofs.FacetDofMap`) decides
        which cells share a facet's DOF.  With none, or with a conforming map,
        facet ``f`` *is* DOF ``f`` and the result is unchanged; a map that
        duplicates fracture facets sends each side to its own DOF.
        """
        d = self.mesh.dim
        n = self.mesh.num_cells(d - 1) if dofmap is None else dofmap.n_dofs
        rows, cols, vals = [], [], []
        for cid in range(self.mesh.num_cells(d)):
            M, fids = self.local(cid)
            f = (
                np.asarray(fids)
                if dofmap is None
                else dofmap.cell_dofs(cid, fids)
            )
            rows.append(np.repeat(f, len(f)))
            cols.append(np.tile(f, len(f)))
            vals.append(M.ravel())
        matrix = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        # The DOF is the *integrated* normal flux, so that the discrete divergence
        # can be the bare signed incidence.  The local matrices are built in the
        # facet-average convention, so the measures are moved here -- into the
        # inner product, where every other metric quantity already lives.
        area = self.mesh.geometry.measure(self.mesh.dim - 1)
        if dofmap is None:
            scale = 1.0 / area
        else:  # a duplicated facet's DOFs inherit their parent facet's measure
            scale = np.ones(n)
            for facet in range(len(area)):
                for cell, _ in dofmap.sides(facet):
                    scale[int(dofmap.dofs(cell, facet)[0])] = 1.0 / area[facet]
        weights = sp.diags(scale)
        return (weights @ matrix @ weights).tocsr()
