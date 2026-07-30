r"""Global mixed (saddle-point) problems.

Assembles the element-level mimetic operators into the global systems and
solves them.  Two elliptic problems are provided, sharing one algebraic shape:

**Mixed Poisson** -- unknowns: normal flux per facet, pressure per cell::

    [  M   -B^T ] [ F ]   [ -g_D ]
    [ -B    0   ] [ p ] = [  -b  ]

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

from mimetika.assembly.local import skew_generators
from mimetika.mesh.mesh import Mesh
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.solver.saddle import solve_saddle


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


def discrete_divergence(mesh: Mesh, dofmap=None) -> sp.csr_matrix:
    """``B`` with ``(B F)_E = \\int_E div F``: incidence scaled by facet measures.

    Purely topological up to the facet measures -- this is the discrete
    exterior derivative, not a fitted operator.

    With a duplicating ``dofmap`` a fracture facet contributes to **one** cell
    per column, so ``un+`` and ``un-`` no longer cancel and the mass exchanged
    with the fracture is free rather than identically zero.
    """
    d = mesh.dim
    if dofmap is None:
        inc = mesh.complex.boundary_matrix(d)  # (n_facets, n_cells), signed
        return (inc.T @ sp.diags(mesh.geometry.measure(d - 1))).tocsr()

    area = mesh.geometry.measure(d - 1)
    rows, cols, vals = [], [], []
    for c in range(mesh.num_cells(d)):
        for f, sgn in mesh.complex.facets_of(d, c):
            rows.append(c)
            cols.append(int(dofmap.dofs(c, f)[0]))
            vals.append(sgn * area[f])
    return sp.csr_matrix(
        (vals, (rows, cols)), shape=(mesh.num_cells(d), dofmap.n_dofs)
    )


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
        self._M: sp.csr_matrix | None = None

    def inner_product(self) -> sp.csr_matrix:
        """The assembled flux inner product (cached -- it is the costly part)."""
        if self._M is None:
            self._M = self.inner.assemble()
        return self._M

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
        """Return ``(A, rhs)`` of the global saddle-point system.

        The second block row is negated so that ``A`` is genuinely **symmetric
        indefinite** -- which is what MINRES and PETSc's ``fieldsplit`` expect.
        The solution is unchanged.
        """
        M = self.inner_product()
        B = discrete_divergence(self.mesh)
        A = sp.bmat([[M, -B.T], [-B, None]], format="csr")
        rhs = np.concatenate(
            [-self.dirichlet_vector(dirichlet), -self.source_vector(source)]
        )
        return A, rhs

    def solve(self, source=None, dirichlet=None, **kwargs) -> MixedSolution:
        """Assemble and solve; ``kwargs`` go to :func:`solve_saddle`."""
        A, rhs = self.assemble(source, dirichlet)
        x = solve_saddle(A, rhs, (self.n_flux, self.n_pressure), **kwargs)
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


#: names :meth:`MixedElasticity.assemble_constrained` accepts.  Callers that
#: forward boundary data opaquely -- the contact driver -- key off this instead
#: of listing conditions themselves, so adding one here is the only change a new
#: boundary condition needs.
BOUNDARY_ARGUMENTS = (
    "body_force",
    "dirichlet",
    "extra_rhs",
    "traction",
    "traction_facets",
    "roller_facets",
)


def _constrain(A: sp.csr_matrix, rhs: np.ndarray, dofs, values):
    """Symmetric row/column elimination imposing ``x[dofs] = values``.

    The pinned equation is scaled by the diagonal entry it replaces rather than
    written as a bare ``1``.  With a stiff material the stress block carries
    entries of order ``1/G`` -- around ``1e-10`` for rock -- so a unit diagonal
    would be ten orders of magnitude larger than everything around it, and a
    direct factorisation reports a zero pivot on the *rest* of the matrix.
    (SuperLU's internal scaling hides this; MUMPS does not, and fails with
    ``KSP_DIVERGED_PC_FAILED``.)  Scaling leaves the solution untouched.
    """
    A = A.tolil(copy=True)
    rhs = rhs - np.asarray(A[:, dofs] @ values).ravel()
    diagonal = np.abs(A.diagonal())
    reference = diagonal[diagonal > 0].mean() if np.any(diagonal > 0) else 1.0
    for d, v in zip(dofs, values):
        scale = diagonal[d] if diagonal[d] > 0 else reference
        A[d, :] = 0.0
        A[:, d] = 0.0
        A[d, d] = scale
        rhs[d] = v * scale
    return A.tocsr(), rhs


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

    def __init__(
        self, mesh: Mesh, mu: float = 1.0, lam: float = 1.0, contact=None
    ) -> None:
        self.mesh = mesh
        self.contact = contact  # optional FractureContact: adds compliance to M
        self.inner = ElasticityInnerProduct(mesh, mu=mu, lam=lam)
        self.d = mesh.dim
        self.ndf = self.inner.dofs_per_facet(self.d)
        self.n_skew = len(skew_generators(self.d))
        self._ops = None

    @property
    def n_stress(self) -> int:
        return self.ndf * self.mesh.num_cells(self.d - 1)

    @property
    def n_cells(self) -> int:
        return self.mesh.num_cells(self.d)

    def _facet_dofs(self, fid: int) -> np.ndarray:
        return self.ndf * fid + np.arange(self.ndf)

    def assemble_operators(self):
        """Assemble ``(M, D, A)``: inner product, discrete div and asymmetry.

        Cached -- this is by far the costly part of a solve.
        """
        if self._ops is not None:
            return self._ops
        from mimetika.assembly.local import skew_generators
        from mimetika.operators.inner_product import assemble_local_inner_product

        d, ndf, nsk = self.d, self.ndf, self.n_skew
        gens = skew_generators(d)

        # One pass builds all three operators.  Assembling M separately would
        # recompute every local matrix a second time.
        m_rows, m_cols, m_vals = [], [], []
        d_rows, d_cols, d_vals = [], [], []
        a_rows, a_cols, a_vals = [], [], []

        for facet_ids, signs_g, cells in self.inner.cell_groups():
            nB, nf = facet_ids.shape
            if d == 3:
                N, R, Kbar, vol, X = self.inner.local_matrices_batched(
                    facet_ids, signs_g, cells
                )
                Mloc, deficient = self.inner.local_inner_products_batched(
                    N, R, Kbar, vol
                )
                if deficient.any():  # rare; redo those cells with the scalar path
                    for b in np.where(deficient)[0]:
                        n1, r1, k1, v1, _ = self.inner.local_matrices(int(cells[b]))
                        Mloc[b] = assemble_local_inner_product(n1, r1, k1, v1)
            else:
                # the batched kernel is written for d = 3; other dimensions take
                # the scalar path, which is dimension-generic
                Mloc = np.empty((nB, nf * ndf, nf * ndf))
                X = np.empty((nB, nf, d, d))
                for b, c in enumerate(cells):
                    N, R, Kbar, vol, _, Xc = self.inner.local_matrices(
                        int(c), with_facet_data=True
                    )
                    Mloc[b] = assemble_local_inner_product(N, R, Kbar, vol)
                    X[b] = Xc

            gdofs = (
                ndf * facet_ids[:, :, None] + np.arange(ndf)
            ).reshape(nB, nf * ndf)
            sgn = np.repeat(signs_g, ndf, axis=1)

            Mloc = Mloc * sgn[:, :, None] * sgn[:, None, :]  # -> canonical DOFs
            m_rows.append(np.repeat(gdofs, nf * ndf, axis=1).ravel())
            m_cols.append(np.tile(gdofs, (1, nf * ndf)).ravel())
            m_vals.append(Mloc.ravel())

            # div_h, scaled by |E|: the constant moment of each component
            cols = (np.arange(nf)[:, None] * ndf + np.arange(d)[None, :] * d).ravel()
            d_rows.append(
                (cells[:, None] * d + np.tile(np.arange(d), nf)[None, :]).ravel()
            )
            d_cols.append(gdofs[:, cols].ravel())
            d_vals.append(np.repeat(signs_g, d, axis=1).ravel())

            # as_h, scaled by |E|: pair the tractions with the rigid rotations
            Aloc = np.einsum("pkc,bfec->bpfke", gens, X).reshape(nB, nsk, nf * ndf)
            Aloc = Aloc * sgn[:, None, :]
            a_rows.append(
                np.repeat(cells[:, None] * nsk + np.arange(nsk), nf * ndf, axis=1).ravel()
            )
            a_cols.append(np.tile(gdofs[:, None, :], (1, nsk, 1)).ravel())
            a_vals.append(Aloc.ravel())

        def build(rows, cols, vals, shape):
            return sp.csr_matrix(
                (
                    np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols)),
                ),
                shape=shape,
            )

        n_sig = self.n_stress
        M = build(m_rows, m_cols, m_vals, (n_sig, n_sig))
        if self.contact is not None:
            # a compliant fracture adds compliance in series on its facets
            M = (M + self.contact.assemble(n_sig)).tocsr()
        self._ops = (
            M,
            build(d_rows, d_cols, d_vals, (d * self.n_cells, n_sig)),
            build(a_rows, a_cols, a_vals, (nsk * self.n_cells, n_sig)),
        )
        return self._ops

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

    def assemble(self, body_force=None, dirichlet=None, extra_rhs=None):
        """Return ``(A, rhs)`` of the global saddle-point system.

        ``extra_rhs`` is added to the stress block; the contact driver uses it
        to carry the augmented-Lagrangian prestress.
        """
        M, D, A = self.assemble_operators()
        blocks = [[M, D.T, A.T], [D, None, None], [A, None, None]]
        if self.n_skew == 0:
            blocks = [[M, D.T], [D, None]]
        S = sp.bmat(blocks, format="csr")
        traction_rhs = self.dirichlet_vector(dirichlet)
        if extra_rhs is not None:
            traction_rhs = traction_rhs + np.asarray(extra_rhs, dtype=float)
        rhs = np.concatenate(
            [
                traction_rhs,
                self.source_vector(body_force),
                np.zeros(self.n_skew * self.n_cells),
            ]
        )
        return S, rhs

    # -- traction (essential) boundary conditions --------------------------------

    def traction_moments(self, facets, traction) -> np.ndarray:
        """Traction DOF values on the given facets.

        Prescribing a traction is **essential** in the Hellinger--Reissner form,
        because the traction *is* a degree of freedom.  It is also what makes the
        problem well posed at ``nu = 1/2``: with displacement prescribed all
        round, the hydrostatic stress is only a Lagrange multiplier for
        ``div u = 0`` and its level is undetermined.

        ``traction(x)`` may return either

        * ``(nq, 3, 3)`` -- the **stress tensor**, from which the traction is
          formed against the canonical facet normal, or
        * ``(nq, 3)`` -- the traction vector itself, taken with respect to that
          same canonical normal (the one :meth:`facet_frame` returns, pointing
          out of the ``+1`` incidence cell).

        Passing the tensor is the safer of the two: the caller then never has to
        know which way a given facet's normal points, and the values produced
        here agree with :meth:`interpolate_stress` facet by facet.  A vector
        traction assembled against the *wrong* normal is silently sign-flipped,
        which is exactly the failure this signature exists to avoid.
        """
        from mimetika.geometry.local_cell import LocalCell

        ndf = self.ndf
        wanted = {int(f) for f in facets}
        out, seen = {}, set()
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid not in wanted or fid in seen:
                    continue
                seen.add(fid)
                qp, qw = lc.facet_quadrature[i]
                B, _ = lc.facet_scalar_basis(i)
                # the canonical normal, in local components
                normal = lc.signs[i] * lc.facet_normals[i]
                val = np.asarray(traction(lc.to_ambient(qp)), dtype=float)
                if val.ndim == 3:  # stress tensor: rotate, then contract
                    S = np.einsum("ai,qab,bj->qij", lc.frame, val, lc.frame)
                    t = np.einsum("qij,j->qi", S, normal)
                else:  # traction vector: project onto the mesh frame
                    t = val @ lc.frame
                out[fid] = np.einsum("q,qb,qk->kb", qw, B, t).ravel()
        dofs = np.concatenate([ndf * f + np.arange(ndf) for f in sorted(out)])
        return dofs, np.concatenate([out[f] for f in sorted(out)])

    def roller_dofs(self, facets) -> np.ndarray:
        """Traction DOFs to pin for a **roller** (normal displacement, free slip).

        A roller prescribes the *normal* displacement and a *vanishing shear*
        traction.  In Hellinger--Reissner those land on opposite sides: the
        displacement is natural and rides along in ``dirichlet``, while the shear
        traction is essential and must be constrained -- which is what this
        returns (the tangential components of each facet's traction moments).

        Restricted to **axis-aligned** facets.  For a general normal the shear
        components are a rotation of the stored ones rather than a subset of
        them, so pinning them means a change of basis on the facet block; the
        rectangular domains this is used for do not need it, and silently
        applying the wrong constraint would be far worse than refusing.
        """
        d, ndf = self.d, self.ndf
        frame = self.inner.frame  # (3, d): DOF components live in the mesh frame
        out = []
        for f in sorted({int(x) for x in facets}):
            normal = self.mesh.geometry.facet_frame(f)[0] @ frame
            axis = int(np.argmax(np.abs(normal)))
            if not np.isclose(abs(normal[axis]), 1.0, atol=1e-10):
                raise ValueError(
                    f"facet {f} has normal {normal}, which is not axis aligned; "
                    "roller constraints need a rotated facet basis"
                )
            base = ndf * f
            # DOF layout is component-major: index = component * d + basis
            out.append(
                np.concatenate(
                    [base + k * d + np.arange(d) for k in range(d) if k != axis]
                )
            )
        return np.concatenate(out) if out else np.zeros(0, dtype=int)

    def assemble_constrained(
        self,
        body_force=None,
        dirichlet=None,
        extra_rhs=None,
        traction=None,
        traction_facets=(),
        roller_facets=(),
    ):
        """``(A, rhs)`` with **every** boundary condition applied.

        The single place that knows the full set of conditions.  Callers that
        need to add their own constraints on top -- the contact driver pins the
        fracture traction to the current multiplier -- start from here rather
        than reimplementing the boundary handling, which is what lets them stay
        ignorant of what conditions exist.
        """
        S, rhs = self.assemble(body_force, dirichlet, extra_rhs)
        if len(traction_facets):
            dofs, values = self.traction_moments(traction_facets, traction)
            S, rhs = _constrain(S, rhs, dofs, values)
        if len(roller_facets):
            dofs = self.roller_dofs(roller_facets)
            S, rhs = _constrain(S, rhs, dofs, np.zeros(len(dofs)))
        return S, rhs

    @property
    def block_sizes(self) -> tuple[int, int]:
        return (self.n_stress, (self.d + self.n_skew) * self.n_cells)

    def split(self, x: np.ndarray) -> MixedSolution:
        """Unpack a raw solution vector into its fields."""
        n1 = self.n_stress
        n2 = n1 + self.d * self.n_cells
        return MixedSolution(
            {"stress": x[:n1], "displacement": x[n1:n2], "rotation": x[n2:]}
        )

    def solve(self, **kwargs) -> MixedSolution:
        """Assemble and solve.

        Boundary arguments go to :meth:`assemble_constrained`; anything else is
        forwarded to :func:`~mimetika.solver.saddle.solve_saddle`.
        """
        boundary = {k: kwargs.pop(k) for k in BOUNDARY_ARGUMENTS if k in kwargs}
        S, rhs = self.assemble_constrained(**boundary)
        return self.split(solve_saddle(S, rhs, self.block_sizes, **kwargs))

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

    def interpolate_rotation(self, grad_displacement) -> np.ndarray:
        """Element means of the rotation multiplier ``s = skw(grad u)``.

        ``grad_displacement`` maps ambient points ``(N,3)`` to ``(N,3,3)`` with
        ``[q,i,j] = du_i/dx_j``.  The multiplier is the skew part of the
        displacement gradient, expressed in the basis of :func:`skew_generators`.
        """
        from mimetika.geometry.local_cell import LocalCell

        gens = skew_generators(self.d)
        out = np.zeros(self.n_skew * self.n_cells)
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            qp, qw = self.mesh.geometry.quadrature(self.d, c)
            G = np.asarray(grad_displacement(qp), dtype=float)
            G = np.einsum("ai,qab,bj->qij", lc.frame, G, lc.frame)
            skew = 0.5 * (G - np.swapaxes(G, 1, 2))
            vals = 0.5 * np.einsum("pij,qij->qp", gens, skew)
            out[c * self.n_skew : (c + 1) * self.n_skew] = (qw @ vals) / qw.sum()
        return out

    def cell_stress(self, dofs) -> np.ndarray:
        """Cell-mean stress tensors ``(n_cells, d, d)`` in the mesh frame.

        The same contraction the trace operator performs, kept general instead of
        contracted against the identity::

            ``sigma_E[i,j] = (1/|E|) sum_e s_e int_e (sigma n_e)_i (x - x_E)_j``

        Exact for stress fields that are linear on each facet, and available in
        any dimension -- unlike the 3D-only flux/stress reconstructions in
        :mod:`mimetika.postprocess`.
        """
        from mimetika.geometry.local_cell import LocalCell

        d, ndf = self.d, self.ndf
        inner = self.inner
        dofs = np.asarray(dofs, dtype=float)
        out = np.zeros((self.n_cells, d, d))
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, inner.frame)
            _, X = inner.facet_data(lc, inner._scale(lc))  # (nf, nb, d)
            block = dofs[
                (ndf * np.asarray(lc.facet_ids)[:, None] + np.arange(ndf)).ravel()
            ].reshape(lc.n_facets, d, d)  # (facet, component, basis)
            out[c] = np.einsum("f,fib,fbj->ij", lc.signs, block, X) / lc.volume
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
