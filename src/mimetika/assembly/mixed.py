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
global solves are exact whenever the exact solution is in the reconstruction
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
    """``B`` with ``(B F)_E = \\int_E div F``: the signed incidence matrix.

    Purely topological -- integer entries, no geometry at all.  This is the
    discrete exterior derivative, and Stokes makes it exact.

    That requires the flux DOF to be the *integrated* normal flux
    ``F_e = int_e F.n``, which is the natural evaluation of a ``(d-1)``-form on a
    facet, rather than the facet *average*.  With the average convention a
    ``diag(|e|)`` enters this operator, putting metric into the one object that
    should carry none; the measures belong in the inner product.

    With a duplicating ``dofmap`` a fracture facet contributes to **one** cell
    per column, so ``un+`` and ``un-`` no longer cancel and the mass exchanged
    with the fracture is free rather than identically zero.
    """
    d = mesh.dim
    if dofmap is None:
        # Cached on the mesh.  This operator is *metric-free apart from the facet
        # measures* -- signed incidence, nothing else -- so it is the same matrix
        # for every physics on a given mesh: Darcy, elasticity, and the flow block
        # of poromechanics all want this one object, and the multiphysics systems
        # assemble it several times.
        cached = getattr(mesh, "_discrete_divergence", None)
        if cached is not None:
            return cached
        divergence = mesh.complex.boundary_matrix(d).T.tocsr()
        try:
            mesh._discrete_divergence = divergence
        except AttributeError:  # frozen mesh: correctness does not depend on it
            pass
        return divergence

    rows, cols, vals = [], [], []
    for c in range(mesh.num_cells(d)):
        for f, sgn in mesh.complex.facets_of(d, c):
            rows.append(c)
            cols.append(int(dofmap.dofs(c, f)[0]))
            vals.append(float(sgn))
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
    """Global mixed Poisson problem with Dirichlet data.

    The default flux space is the de Rham (consistency-only) product:
    ``d`` facet ``P_1`` moments per facet, no stabilization.  Passing
    ``basis='const'`` or ``'rt0'`` selects the stabilized
    :class:`~mimetika.operators.diffusion.DiffusionInnerProduct` (one
    average flux per facet), retained as the ``M_1 + M_2`` example.
    """

    def __init__(
        self,
        mesh: Mesh,
        K: np.ndarray | None = None,
        basis: str | None = None,
        inner=None,
    ) -> None:
        self.mesh = mesh
        if inner is None:
            if basis is not None:
                inner = DiffusionInnerProduct(mesh, K=K, basis=basis)
            else:
                from mimetika.operators.derham import (
                    DeRhamDiffusionInnerProduct,
                )

                inner = DeRhamDiffusionInnerProduct(mesh, K=K)
        self.inner = inner
        d = mesh.dim
        self.ndf = (
            inner.dofs_per_facet(d) if hasattr(inner, "dofs_per_facet") else 1
        )
        self._M: sp.csr_matrix | None = None

    def inner_product(self) -> sp.csr_matrix:
        """The assembled flux inner product (cached -- it is the costly part)."""
        if self._M is None:
            self._M = self.inner.assemble()
        return self._M

    @property
    def n_flux(self) -> int:
        return self.ndf * self.mesh.num_cells(self.mesh.dim - 1)

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
        """Boundary-pressure pairing on boundary facets, zero elsewhere.

        One average per facet for the stabilized space; the facet ``P_1``
        moments of ``p_D`` for the de Rham space.
        """
        from mimetika.geometry.local_cell import LocalCell

        d, ndf = self.mesh.dim, self.ndf
        g = np.zeros(self.n_flux)
        if potential is None:
            return g
        on_boundary = facet_cell_signs(self.mesh)
        if ndf == 1:
            for f, sgn in on_boundary.items():
                qp, qw = self.mesh.geometry.quadrature(d - 1, f)
                # pairs with the *integrated* flux DOF: the facet mean of p_D
                g[f] = sgn * (qw @ np.asarray(potential(qp)).ravel()) / qw.sum()
            return g
        done: set[int] = set()
        for c in range(self.n_pressure):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid not in on_boundary or fid in done:
                    continue
                done.add(fid)
                qp, _ = lc.facet_quadrature[i]
                p = np.asarray(potential(lc.to_ambient(qp)), dtype=float).ravel()
                g[ndf * fid : ndf * (fid + 1)] = (
                    lc.signs[i] * lc.expand_on_facet(i, p)
                )
        return g

    def divergence(self) -> sp.csr_matrix:
        """The discrete divergence on this space's flux DOFs.

        Signed incidence on the facet DOF for the stabilized space; on the
        constant (``b = 0``) moment of each facet block for the de Rham
        space -- topological either way.
        """
        if self.ndf == 1:
            return discrete_divergence(self.mesh)
        d, ndf = self.mesh.dim, self.ndf
        rows, cols, vals = [], [], []
        for c in range(self.n_pressure):
            for f, sgn in self.mesh.complex.facets_of(d, c):
                rows.append(c)
                cols.append(ndf * f)
                vals.append(float(np.sign(sgn)))
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_pressure, self.n_flux)
        )

    def assemble(self, source=None, dirichlet=None):
        """Return ``(A, rhs)`` of the global saddle-point system.

        The second block row is negated so that ``A`` is genuinely **symmetric
        indefinite** -- which is what MINRES and PETSc's ``fieldsplit`` expect.
        The solution is unchanged.
        """
        M = self.inner_product()
        B = self.divergence()
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
        """Flux DOFs of the exact field, w.r.t. the canonical normals.

        Integrated normal flux per facet for the stabilized space; the full
        facet ``P_1`` moments for the de Rham space.
        """
        from mimetika.geometry.local_cell import LocalCell

        d, ndf = self.mesh.dim, self.ndf
        out = np.zeros(self.n_flux)
        if ndf == 1:
            normals = (
                self.mesh.geometry.facet_normals()
                if d == 3
                else _facet_normals_low_dim(self.mesh)
            )
            for f in range(self.mesh.num_cells(d - 1)):
                qp, qw = self.mesh.geometry.quadrature(d - 1, f)
                F = np.asarray(flux(qp), dtype=float)
                out[f] = qw @ (F @ normals[f])  # integrated, not averaged
            return out
        done: set[int] = set()
        for c in range(self.n_pressure):
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid in done:
                    continue
                done.add(fid)
                qp, qw = lc.facet_quadrature[i]
                B, _ = lc.facet_scalar_basis(i)
                Fn = (
                    np.asarray(flux(lc.to_ambient(qp)), dtype=float) @ lc.frame
                ) @ lc.facet_normals[i]
                out[ndf * fid : ndf * (fid + 1)] = lc.signs[i] * np.einsum(
                    "q,qb,q->b", qw, B, Fn
                )
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


def constraint_scales(A: sp.spmatrix, dofs) -> np.ndarray:
    """The diagonal each pinned row is scaled by in :func:`_constrain`.

    Exposed so the condensed form can reproduce the elimination exactly.
    """
    diagonal = np.abs(A.diagonal())
    reference = diagonal[diagonal > 0].mean() if np.any(diagonal > 0) else 1.0
    dofs = np.asarray(dofs)
    return np.where(diagonal[dofs] > 0, diagonal[dofs], reference)


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
    dofs = np.asarray(dofs, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    scales = constraint_scales(A, dofs)
    rhs = rhs - np.asarray(A[:, dofs] @ values).ravel()

    # Drop every entry in a pinned row or column in one pass, then put the scaled
    # diagonal back.  Doing it per DOF through LIL assignment is quadratic.
    coo = A.tocoo()
    pinned = np.zeros(A.shape[0], dtype=bool)
    pinned[dofs] = True
    keep = ~(pinned[coo.row] | pinned[coo.col])
    matrix = sp.csr_matrix(
        (
            np.concatenate([coo.data[keep], scales]),
            (
                np.concatenate([coo.row[keep], dofs]),
                np.concatenate([coo.col[keep], dofs]),
            ),
        ),
        shape=A.shape,
    )
    rhs[dofs] = values * scales
    return matrix, rhs


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
    """Global weakly-symmetric Hellinger--Reissner problem with Dirichlet data.

    This is the classic three-field formulation ``(sigma, u, s)``, kept as
    the reference implementation and the base class.  The standard formulation
    is the four-field split
    (:class:`~mimetika.assembly.four_field.FourFieldElasticity`), which carries
    the solid pressure explicitly and is solution-identical by exact
    congruence; the contact driver and the benchmarks build that one.
    """

    def __init__(
        self,
        mesh: Mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        contact=None,
        inner=None,
    ) -> None:
        self.mesh = mesh
        self.contact = contact  # optional FractureContact: adds compliance to M
        # any facet-DOF stress inner product works here -- the assembly asks the
        # space for its sizes and offsets rather than assuming a layout.  The
        # default is the de Rham (consistency-only) product; passing e.g.
        # ElasticityInnerProduct (stabilized) or LumpedDeviatoricStress
        # (diagonal) swaps the discretisation.
        if inner is None:
            # both formulations default to the same de Rham member (the
            # deviatoric space; here its volumetric completion is folded back
            # in), so three- and four-field solves stay solution-identical
            # under defaults.  DeRhamElasticityInnerProduct (linear-trace
            # member, = AFW on simplices) and the stabilized
            # ElasticityInnerProduct remain available explicitly.
            from mimetika.operators.derham import DeRhamDeviatoricStress

            inner = DeRhamDeviatoricStress(mesh, mu=mu, lam=lam)
        self.inner = inner
        self.d = mesh.dim
        self.ndf = self.inner.dofs_per_facet(self.d)
        self.n_skew = len(skew_generators(self.d))
        self._ops = None
        self._space_ops = None

    @property
    def n_stress(self) -> int:
        return self.ndf * self.mesh.num_cells(self.d - 1)

    @property
    def n_cells(self) -> int:
        return self.mesh.num_cells(self.d)

    def _facet_dofs(self, fid: int) -> np.ndarray:
        return self.ndf * fid + np.arange(self.ndf)

    def assemble_operators(self):
        """Assemble ``(M, D, A)`` with ``M`` the **full** compliance.

        When the space keeps the volumetric part out of its own operator
        (``volumetric_included = False``: the lumped space), it is folded back
        in here as ``W^T diag(c) W`` -- one rank-one update per cell, Woodbury
        in matrix form.  That completes the compliance but fills in the facet
        couplings of each cell, so the diagonality of the lumped ``M`` is lost;
        the four-field assembly (:class:`.four_field.FourFieldElasticity`)
        exists to avoid exactly this step.
        """
        if self._ops is not None:
            return self._ops
        M, D, A = self._space_operators()
        if not getattr(self.inner, "volumetric_included", True):
            W, cvec = self.inner.volumetric_operator()
            M = (M + W.T @ sp.diags(cvec) @ W).tocsr()
        self._ops = (M, D, A)
        return self._ops

    def _space_operators(self):
        """``(M, D, A)`` with ``M`` the *space's own* inner product.

        For AFW that is the full compliance; for the lumped space it is the
        diagonal deviatoric part only.  The fracture-contact compliance, which
        is facet compliance in series whatever the space, is included.  Cached
        -- this is by far the costly part of a solve, and both the three- and
        four-field assemblies start from it.
        """
        if self._space_ops is not None:
            return self._space_ops
        from mimetika.assembly.local import skew_generators

        d, ndf, nsk = self.d, self.ndf, self.n_skew
        gens = skew_generators(d)

        # This pass builds D and A only; M is the space's own :meth:`assemble`.
        d_rows, d_cols, d_vals = [], [], []
        a_rows, a_cols, a_vals = [], [], []

        for facet_ids, signs_g, cells in self.inner.cell_groups():
            nB, nf = facet_ids.shape
            if (d == 2 and hasattr(self.inner, "facet_data_batched")
                    and self.inner.facet_basis_size(d) == d):
                X = self.inner.facet_data_batched(facet_ids, cells)
            elif d == 3:
                *_, X = self.inner.local_matrices_batched(
                    facet_ids, signs_g, cells
                )
            else:
                # X is stacked, not pre-allocated: its facet-basis extent is d for
                # AFW and 1 for LumpedDeviatoricStress.  Pre-allocating (nB, nf, d, d)
                # broadcast a (nf, 1, d) block up to (nf, d, d) silently.
                blocks = []
                for c in cells:
                    *_, Xc = self.inner.local_matrices(
                        int(c), with_facet_data=True
                    )
                    blocks.append(Xc)
                X = np.stack(blocks)

            gdofs = (
                ndf * facet_ids[:, :, None] + np.arange(ndf)
            ).reshape(nB, nf * ndf)
            sgn = np.repeat(signs_g, ndf, axis=1)

            # div_h, scaled by |E|: where the constant moment of each component
            # sits is a property of the stress space, not of the assembly -- see
            # `constant_moment_offsets`.  The values below stay pure signs, so D
            # remains topological.
            offsets = self.inner.constant_moment_offsets(d)
            cols = (np.arange(nf)[:, None] * ndf + offsets[None, :]).ravel()
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
        # M is the space's own operator.  Accumulating ``M1 + M2`` here instead would
        # add a stabilisation term that is nonzero for a space not built that way --
        # it destroys the diagonality of LumpedDeviatoricStress.  Identical to the
        # accumulated form for AFW.
        M = self.inner.assemble()
        if self.contact is not None:
            # a compliant fracture adds compliance in series on its facets
            M = (M + self.contact.assemble(n_sig)).tocsr()
        self._space_ops = (
            M,
            build(d_rows, d_cols, d_vals, (d * self.n_cells, n_sig)),
            build(a_rows, a_cols, a_vals, (nsk * self.n_cells, n_sig)),
        )
        return self._space_ops

    def dirichlet_vector(self, displacement) -> np.ndarray:
        """Boundary displacement data, expanded in the facet ``P_1`` basis."""
        from mimetika.assembly.local import elasticity_local_operators
        from mimetika.geometry.local_cell import LocalCell

        g = np.zeros(self.n_stress)
        if displacement is None:
            return g
        on_boundary = facet_cell_signs(self.mesh)
        if not on_boundary:
            return g
        # only cells touching a boundary facet contribute -- iterating all of
        # them builds tens of thousands of LocalCells to visit a few hundred
        bm = self.mesh.complex.boundary_matrix(self.d).tocsr()
        boundary_cells = np.unique(np.concatenate(
            [bm[int(f)].indices for f in on_boundary]
        ))
        for c in boundary_cells:
            lc = LocalCell.build(self.mesh.geometry, int(c), self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid not in on_boundary:
                    continue
                qp = lc.facet_quadrature[i][0]
                u = np.asarray(displacement(lc.to_ambient(qp)), dtype=float) @ lc.frame
                # keep the moments this space carries: (d, nb_full) -> (d, nb)
                nb = self.inner.facet_basis_size(self.d)
                coeff = lc.expand_on_facet(i, u).T[:, :nb].ravel()
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
        traction assembled against the *wrong* normal is silently sign-flipped.
        """
        from mimetika.geometry.local_cell import LocalCell

        ndf = self.ndf
        wanted = {int(f) for f in facets}
        out, seen = {}, set()
        # only cells adjacent to a wanted facet can contribute
        bm = self.mesh.complex.boundary_matrix(self.d).tocsr()
        adjacent = (np.unique(np.concatenate(
            [bm[f].indices for f in sorted(wanted)]
        )) if wanted else np.empty(0, dtype=int))
        for c in adjacent:
            c = int(c)
            lc = LocalCell.build(self.mesh.geometry, c, self.inner.frame)
            for i, fid in enumerate(lc.facet_ids):
                if fid not in wanted or fid in seen:
                    continue
                seen.add(fid)
                qp, qw = lc.facet_quadrature[i]
                # keep the moments this space carries: (nq, nb_full) -> (nq, nb)
                B, _ = lc.facet_scalar_basis(i)
                B = B[:, : self.inner.facet_basis_size(self.d)]
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
            # DOF layout is component-major: index = component * nb + basis, with
            # nb = facet_basis_size (d for AFW, 1 for the lumped space).  Using d in
            # place of nb pins every DOF on the facet when nb < d, over-constraining
            # the system.
            nb = self.inner.facet_basis_size(d)
            out.append(
                np.concatenate(
                    [base + k * nb + np.arange(nb) for k in range(d) if k != axis]
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
        """``(A, rhs)`` with every boundary condition applied.

        The single place that knows the full set of conditions.  Callers that
        need to add their own constraints on top -- the contact driver pins the
        fracture traction to the current multiplier -- start from here rather
        than reimplementing the boundary handling.
        """
        S, rhs = self.assemble(body_force, dirichlet, extra_rhs)
        if len(traction_facets):
            dofs, values = self.traction_moments(traction_facets, traction)
            S, rhs = _constrain(S, rhs, dofs, values)
        if len(roller_facets):
            dofs = self.roller_dofs(roller_facets)
            S, rhs = _constrain(S, rhs, dofs, np.zeros(len(dofs)))
        return S, rhs

    def constitutive_rows(self, contact: bool = True) -> sp.csr_matrix:
        """The stress-row block ``[M | D^T | A^T]``, matching :meth:`split`.

        Applied to a solution vector it evaluates the constitutive functional on
        every traction DOF; its residual against the assembled right-hand side
        is the boundary-displacement pairing the contact driver reads the
        fracture jump from.  A formulation with more fields overrides this so
        the driver never has to know the block layout.

        ``contact=False`` strips the embedded fracture compliance and returns
        the rows of the **unfractured** system.  The distinction matters for a
        compliant fracture: at the solution the fractured row is satisfied
        *exactly*, so its residual is zero, while the unfractured residual
        equals ``A_f sigma`` -- the jump itself.
        """
        M, D, A = self.assemble_operators()
        if not contact and self.contact is not None:
            M = (M - self.contact.assemble(self.n_stress)).tocsr()
        return sp.hstack([M, D.T, A.T], format="csr")

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
        nb = inner.facet_basis_size(d)  # d for AFW, 1 for the lumped space
        dofs = np.asarray(dofs, dtype=float)
        out = np.zeros((self.n_cells, d, d))
        for c in range(self.n_cells):
            lc = LocalCell.build(self.mesh.geometry, c, inner.frame)
            _, X = inner.facet_data(lc, inner._scale(lc))  # (nf, nb, d)
            block = dofs[
                (ndf * np.asarray(lc.facet_ids)[:, None] + np.arange(ndf)).ravel()
            ].reshape(lc.n_facets, d, nb)  # (facet, component, basis)
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
                B = B[:, : self.inner.facet_basis_size(self.d)]
                amb = lc.to_ambient(qp)
                S = np.asarray(stress(amb), dtype=float)
                S = np.einsum("ai,qab,bj->qij", lc.frame, S, lc.frame)
                # canonical orientation = outward times the incidence sign
                Tn = np.einsum("qij,j->qi", S, lc.signs[i] * lc.facet_normals[i])
                out[self._facet_dofs(fid)] = np.einsum(
                    "q,qb,qk->kb", qw, B, Tn
                ).ravel()
        return out
