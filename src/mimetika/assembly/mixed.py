r"""Global mixed (saddle-point) problems.

Assembles the element-level mimetic operators into the global systems and
solves them.  Two elliptic problems are provided, sharing one algebraic shape:

**Mixed Poisson** -- unknowns: normal flux per facet, pressure per cell::

    [  M   -B^T ] [ F ]   [ -g_D ]
    [  B    0   ] [ p ] = [  b   ]

with ``div F = f``, ``F = -K grad p`` and Dirichlet data ``p = p_D`` on the
boundary.  ``B`` is the *purely topological* discrete divergence: the transpose
of the signed incidence scaled by facet measures, so ``(B F)_E = \int_E div F``.

**Mixed elasticity** (Hellinger--Reissner, weakly imposed symmetry) -- unknowns:
traction moments per facet, displacement per cell, rotation multiplier per
cell::

    [  M    D^T   A^T ] [ sigma ]   [ g_D ]
    [  D     0     0  ] [ u     ] = [  f  ]
    [  A     0     0  ] [ s     ]   [  0  ]

with ``div sigma = f``, ``sigma = C eps(u)`` imposed weakly, and ``u = u_D`` on
the boundary.

In both cases the facet terms of interior facets cancel between the two
adjacent cells (the trace is single-valued and the outward normals are
opposite), so the explicit boundary data appears only on ``\partial\Omega``.

Because the local inner products satisfy strong consistency ``M N = R``, these
global solves are **exact** whenever the exact solution is in the reconstruction
space -- e.g. a linear potential or a linear displacement, on any mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.assembly.local import skew_generators
from mimetika.mesh.mesh import Mesh
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.elasticity import ElasticityInnerProduct


# -- shared helpers -----------------------------------------------------------


def boundary_facets(mesh: Mesh) -> np.ndarray:
    """Indices of the facets that belong to exactly one cell."""
    inc = abs(mesh.complex.boundary_matrix(mesh.dim))
    counts = np.asarray(inc.sum(axis=1)).ravel()
    return np.where(counts == 1)[0]


def facet_cell_signs(mesh: Mesh) -> dict[int, int]:
    """For each boundary facet, its incidence sign in the one cell holding it."""
    b = mesh.complex.boundary_matrix(mesh.dim).tocoo()
    signs: dict[int, int] = {}
    counts: dict[int, int] = {}
    for f, v in zip(b.row, b.data):
        counts[f] = counts.get(f, 0) + 1
        signs[f] = int(np.sign(v))
    return {f: s for f, s in signs.items() if counts[f] == 1}


def discrete_divergence(mesh: Mesh) -> sp.csr_matrix:
    """``B`` with ``(B F)_E = \\int_E div F``: incidence scaled by facet measures.

    Purely topological up to the facet measures -- this is the discrete
    exterior derivative, not a fitted operator.
    """
    d = mesh.dim
    inc = mesh.complex.boundary_matrix(d)  # (n_facets, n_cells), signed
    return (inc.T @ sp.diags(mesh.geometry.measure(d - 1))).tocsr()


def _solve_saddle(A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    return spla.spsolve(A.tocsc(), rhs)


@dataclass
class MixedSolution:
    """Solution of a global mixed problem, split into its blocks."""

    blocks: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.blocks[key]


# -- mixed Poisson ------------------------------------------------------------


class MixedPoisson:
    """Global mixed Poisson problem with Dirichlet data."""

    def __init__(
        self,
        mesh: Mesh,
        K: np.ndarray | None = None,
        basis: str = "const",
    ) -> None:
        self.mesh = mesh
        self.inner = DiffusionInnerProduct(mesh, K=K, basis=basis)

    @property
    def n_flux(self) -> int:
        return self.mesh.num_cells(self.mesh.dim - 1)

    @property
    def n_pressure(self) -> int:
        return self.mesh.num_cells(self.mesh.dim)

    def source_vector(self, source) -> np.ndarray:
        """``b_E = \\int_E f``, by quadrature."""
        d = self.mesh.dim
        b = np.zeros(self.n_pressure)
        if source is None:
            return b
        for c in range(self.n_pressure):
            qp, qw = self.mesh.geometry.quadrature(d, c)
            b[c] = qw @ np.asarray(source(qp)).ravel()
        return b

    def dirichlet_vector(self, potential) -> np.ndarray:
        """``g_e = s_e \\int_e p_D`` on boundary facets, zero elsewhere."""
        d = self.mesh.dim
        g = np.zeros(self.n_flux)
        if potential is None:
            return g
        for f, s in facet_cell_signs(self.mesh).items():
            qp, qw = self.mesh.geometry.quadrature(d - 1, f)
            g[f] = s * (qw @ np.asarray(potential(qp)).ravel())
        return g

    def assemble(self, source=None, dirichlet=None):
        """Return ``(A, rhs)`` of the global saddle-point system."""
        M = self.inner.assemble()
        B = discrete_divergence(self.mesh)
        A = sp.bmat([[M, -B.T], [B, None]], format="csr")
        rhs = np.concatenate(
            [-self.dirichlet_vector(dirichlet), self.source_vector(source)]
        )
        return A, rhs

    def solve(self, source=None, dirichlet=None) -> MixedSolution:
        A, rhs = self.assemble(source, dirichlet)
        x = _solve_saddle(A, rhs)
        return MixedSolution(
            {"flux": x[: self.n_flux], "pressure": x[self.n_flux :]}
        )

    # -- interpolants (for error measurement) --------------------------------

    def interpolate_pressure(self, potential) -> np.ndarray:
        """Element means of the exact potential."""
        d = self.mesh.dim
        out = np.zeros(self.n_pressure)
        for c in range(self.n_pressure):
            qp, qw = self.mesh.geometry.quadrature(d, c)
            out[c] = (qw @ np.asarray(potential(qp)).ravel()) / qw.sum()
        return out

    def interpolate_flux(self, flux) -> np.ndarray:
        """Average normal flux on each facet, w.r.t. the canonical normal."""
        d = self.mesh.dim
        normals = (
            self.mesh.geometry.facet_normals()
            if d == 3
            else _facet_normals_low_dim(self.mesh)
        )
        out = np.zeros(self.n_flux)
        for f in range(self.n_flux):
            qp, qw = self.mesh.geometry.quadrature(d - 1, f)
            F = np.asarray(flux(qp), dtype=float)
            out[f] = (qw @ (F @ normals[f])) / qw.sum()
        return out


def _facet_normals_low_dim(mesh: Mesh) -> np.ndarray:
    """Ambient canonical normals of the facets of a 2D mesh (edges)."""
    from mimetika.geometry.local_cell import mesh_frame

    frame = mesh_frame(mesh.geometry)
    ev = mesh.complex.edge_vertices
    p = mesh.geometry.points
    t = ((p[ev[:, 1]] - p[ev[:, 0]]) @ frame)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n_local = np.column_stack([t[:, 1], -t[:, 0]])
    return n_local @ frame.T


# -- mixed elasticity ---------------------------------------------------------


class MixedElasticity:
    """Global weakly-symmetric Hellinger--Reissner problem with Dirichlet data."""

    def __init__(self, mesh: Mesh, mu: float = 1.0, lam: float = 1.0) -> None:
        self.mesh = mesh
        self.inner = ElasticityInnerProduct(mesh, mu=mu, lam=lam)
        self.d = mesh.dim
        self.ndf = self.inner.dofs_per_facet(self.d)
        self.n_skew = len(skew_generators(self.d))

    @property
    def n_stress(self) -> int:
        return self.ndf * self.mesh.num_cells(self.d - 1)

    @property
    def n_cells(self) -> int:
        return self.mesh.num_cells(self.d)

    def _facet_dofs(self, fid: int) -> np.ndarray:
        return self.ndf * fid + np.arange(self.ndf)

    def assemble_operators(self):
        """Assemble ``(M, D, A)``: inner product, discrete div and asymmetry."""
        from mimetika.assembly.local import elasticity_local_operators

        d, ndf = self.d, self.ndf
        M = self.inner.assemble()

        rows_d, cols_d, vals_d = [], [], []
        rows_a, cols_a, vals_a = [], [], []
        for c in range(self.n_cells):
            _, Dv, As, lc = elasticity_local_operators(self.inner, c)
            gdofs = np.concatenate([self._facet_dofs(f) for f in lc.facet_ids])
            signs = np.repeat(lc.signs, ndf)
            # local blocks are scaled by |E| and use outward normals
            Dloc = lc.volume * Dv * signs
            Aloc = lc.volume * As * signs
            for k in range(d):
                nz = np.nonzero(Dloc[k])[0]
                rows_d += [c * d + k] * len(nz)
                cols_d += list(gdofs[nz])
                vals_d += list(Dloc[k, nz])
            for p in range(self.n_skew):
                nz = np.nonzero(Aloc[p])[0]
                rows_a += [c * self.n_skew + p] * len(nz)
                cols_a += list(gdofs[nz])
                vals_a += list(Aloc[p, nz])

        D = sp.csr_matrix(
            (vals_d, (rows_d, cols_d)), shape=(d * self.n_cells, self.n_stress)
        )
        A = sp.csr_matrix(
            (vals_a, (rows_a, cols_a)),
            shape=(self.n_skew * self.n_cells, self.n_stress),
        )
        return M, D, A

    def dirichlet_vector(self, displacement) -> np.ndarray:
        """Boundary displacement data, expanded in the facet ``P_1`` basis."""
        from mimetika.assembly.local import elasticity_local_operators
        from mimetika.geometry.local_cell import LocalCell

        g = np.zeros(self.n_stress)
        if displacement is None:
            return g
        on_boundary = facet_cell_signs(self.mesh)
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid not in on_boundary:
                    continue
                qp = lc.facet_quadrature[i][0]
                u = np.asarray(displacement(lc.to_ambient(qp)), dtype=float) @ lc.frame
                coeff = lc.expand_on_facet(i, u).T.ravel()
                g[self._facet_dofs(fid)] += lc.signs[i] * coeff
        return g

    def source_vector(self, body_force) -> np.ndarray:
        """``\\int_E f`` per cell (matching the ``|E|``-scaled divergence block)."""
        out = np.zeros(self.d * self.n_cells)
        if body_force is None:
            return out
        from mimetika.geometry.local_cell import LocalCell

        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            qp, qw = self.mesh.geometry.quadrature(self.d, c)
            f = np.asarray(body_force(qp), dtype=float) @ lc.frame
            out[c * self.d : (c + 1) * self.d] = qw @ f
        return out

    def assemble(self, body_force=None, dirichlet=None):
        """Return ``(A, rhs)`` of the global saddle-point system."""
        M, D, A = self.assemble_operators()
        blocks = [[M, D.T, A.T], [D, None, None], [A, None, None]]
        if self.n_skew == 0:
            blocks = [[M, D.T], [D, None]]
        S = sp.bmat(blocks, format="csr")
        rhs = np.concatenate(
            [
                self.dirichlet_vector(dirichlet),
                self.source_vector(body_force),
                np.zeros(self.n_skew * self.n_cells),
            ]
        )
        return S, rhs

    def solve(self, body_force=None, dirichlet=None) -> MixedSolution:
        S, rhs = self.assemble(body_force, dirichlet)
        x = _solve_saddle(S, rhs)
        n1 = self.n_stress
        n2 = n1 + self.d * self.n_cells
        return MixedSolution(
            {"stress": x[:n1], "displacement": x[n1:n2], "rotation": x[n2:]}
        )

    # -- interpolants (for error measurement) --------------------------------

    def interpolate_displacement(self, displacement) -> np.ndarray:
        """Element means of the exact displacement, in the mesh frame."""
        from mimetika.geometry.local_cell import LocalCell

        out = np.zeros(self.d * self.n_cells)
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            qp, qw = self.mesh.geometry.quadrature(self.d, c)
            u = np.asarray(displacement(qp), dtype=float) @ lc.frame
            out[c * self.d : (c + 1) * self.d] = (qw @ u) / qw.sum()
        return out

    def interpolate_stress(self, stress) -> np.ndarray:
        """Traction moments of the exact stress on every facet."""
        from mimetika.geometry.local_cell import LocalCell

        out = np.zeros(self.n_stress)
        done: set[int] = set()
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid in done:
                    continue
                done.add(fid)
                qp, qw = lc.facet_quadrature[i]
                B, _ = lc.facet_scalar_basis(i)
                amb = lc.to_ambient(qp)
                S = np.asarray(stress(amb), dtype=float)
                S = np.einsum("ai,qab,bj->qij", lc.frame, S, lc.frame)
                # canonical orientation = outward times the incidence sign
                Tn = np.einsum("qij,j->qi", S, lc.signs[i] * lc.facet_normals[i])
                out[self._facet_dofs(fid)] = np.einsum(
                    "q,qb,qk->kb", qw, B, Tn
                ).ravel()
        return out
