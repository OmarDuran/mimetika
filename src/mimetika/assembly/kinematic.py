r"""Kinematic-rotation elasticity: the rotation defined, not enforced.

In the weakly-symmetric Hellinger--Reissner formulations
(:class:`~mimetika.assembly.mixed.MixedElasticity`,
:class:`~mimetika.assembly.four_field.FourFieldElasticity`) the rotation ``s``
is a Lagrange multiplier: the row ``A sigma = 0`` enforces symmetry, and ``s``
is determined only through the constraint pairing -- the zero block, and with
it the multiplier inf-sup question.

This module closes the same system the other way.  Taking the skew part of the
constitutive relation ``C^{-1} sigma = grad u - S^* s`` (isotropy leaves the
skew part untouched up to ``1/2mu``) gives the identity

    ``s = skw(grad u) - skw(sigma)/2mu`` ,

which is **Cauchy's definition of the macroscopic rotation**, corrected by the
stress asymmetry so that it is exact even before symmetry is achieved.  Using
it as the rotation row,

    ``(1/2mu) A sigma + 2|E| s - G u = 0`` ,

the rotation block becomes the diagonal mass ``2|E|`` -- ``s`` is a locally
eliminable *kinematic* variable, not a multiplier.  There is no zero block, no
inf-sup condition for the rotation, and no Cosserat physics: no curvature term
and no internal length enter anywhere.  Symmetry of the stress becomes an
*output* (it holds to the consistency order) instead of an imposed constraint.
This is why the kinematic-rotation closure needs no rotation stabilization of
any kind: a kinematically *defined* quantity never carries an inf-sup burden.

The gauge modes, and the blended row
------------------------------------
With an *independent* facet stress the pure definition row is rank-deficient:
the relaxed model has the exact gauge symmetry ``sigma -> sigma + tau``,
``s -> s - skw(tau)/2mu`` for any cell-wise constant skew ``tau`` with
continuous tractions (globally constant in 2D; in 3D jumps parallel to the
facet normal are admissible, so the gauge space grows with the mesh).  A
formulation whose stress is a *dependent* variable never sees these modes;
keeping ``sigma`` primary, the definition row alone drops the angular-momentum
content of the system.  The well-posed closure *blends* the two:

    ``(1-theta) (1/2mu) A sigma - 2 theta |E| s + theta G u = 0`` ,
    ``theta in (0, 1)`` ,

which is consistent for every ``theta`` (the exact solution satisfies both
ingredients separately) and kills every gauge mode (they would need
``(1-theta)/theta = -1``).  Eliminating ``s`` now *adds*
``(1-theta)/theta * A^T A / (4 mu |E|)`` to the stress block -- definiteness
improves rather than degrades.  Both limits are degenerate: ``theta = 1`` is
the pure definition (gauge kernel), ``theta = 0`` the zero-block multiplier.
The default ``theta = 1/2`` makes ``s`` the average of its two continuum
expressions, ``skw(grad u)`` and ``-skw(sigma)/2mu`` shifted by ``skw(grad u)``.

The price is symmetry of the *linear system*: the rotation row is not the
transpose of the ``A^T s`` coupling in the constitutive row, so the matrix is
only quasi-symmetric.  Solvers must treat it as nonsymmetric -- ``direct``
works; MINRES does not and is refused.

The operator ``G``
------------------
``(G u)_{E,p} = |E| * gens_p : grad_h(u)|_E`` needs a discrete displacement
gradient from cell values -- genuinely new information (contracting the
constitutive row against cell-wise skew tests would be a *dependent* equation
and the system would be singular).  Here ``grad_h`` is a weighted least-squares
fit over the face neighbours, supplemented on Dirichlet boundary facets by the
boundary datum at the facet centroid.  The fit is exact for affine
displacements on any mesh, which is what keeps the AFW patch test exact.  On
boundary facets carrying a traction condition the datum is *not* known;
``traction_facets`` passed to :meth:`assemble_constrained` are simply excluded
from the fit, so cells next to such facets must retain at least ``d``
independent neighbour directions (checked, with a clear error).

The two-point variant (mu-weighted facet averages in place of the
least-squares fit) belongs to the lumped stage and trades exactness on general
meshes for the minimal two-point stencil on face-orthogonal ones.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.local import skew_generators
from mimetika.assembly.mixed import (
    MixedElasticity,
    MixedSolution,
    facet_cell_signs,
)
from mimetika.solver.saddle import solve_saddle


def skew_gradient(mesh, frame, dirichlet=None, exclude_facets=()):
    """``(G, b)`` with ``(G u + b)_{E,p} = |E| gens_p : grad_h(u)|_E``.

    ``grad_h`` is the per-cell weighted least-squares gradient built from the
    differences to face-neighbour cell values and, on Dirichlet boundary
    facets, to the boundary datum at the facet centroid (whose contribution is
    returned in the affine part ``b``).  Weights ``1/|offset|^2`` make the fit
    scale-invariant; exactness for affine fields holds for any weights.

    ``exclude_facets`` lists facets that must not enter the fit: boundary
    facets whose datum is unknown (traction conditions) and **fracture**
    facets, across which the displacement is discontinuous -- differencing
    through them would smear the jump into a spurious rotation.
    """
    d = mesh.dim
    gens = skew_generators(d)
    nsk = len(gens)
    n_cells = mesh.num_cells(d)
    centroids_c = mesh.geometry.centroids(d) @ frame  # local coordinates
    centroids_f = mesh.geometry.centroids(d - 1) @ frame
    vol = mesh.geometry.measure(d)
    inc = mesh.complex.boundary_matrix(d).tocsr()  # (n_facets, n_cells)
    on_boundary = facet_cell_signs(mesh)
    excluded = {int(f) for f in exclude_facets}

    rows, cols, vals = [], [], []
    b = np.zeros(nsk * n_cells)
    csc = inc.tocsc()
    for c in range(n_cells):
        offsets, neighbours, data = [], [], []
        for f in csc.indices[csc.indptr[c] : csc.indptr[c + 1]]:
            if int(f) in excluded:
                continue  # traction facet (no datum) or fracture (jump inside)
            adjacent = inc.indices[inc.indptr[f] : inc.indptr[f + 1]]
            if len(adjacent) == 2:
                other = int(adjacent[0] if adjacent[1] == c else adjacent[1])
                offsets.append(centroids_c[other] - centroids_c[c])
                neighbours.append(other)
                data.append(None)
            elif dirichlet is not None:
                # boundary datum at the facet centroid closes the fit
                value = np.asarray(
                    dirichlet(mesh.geometry.centroids(d - 1)[int(f)][None]),
                    dtype=float,
                ).reshape(-1)[:3] @ frame
                offsets.append(centroids_f[int(f)] - centroids_c[c])
                neighbours.append(None)
                data.append(value)
        if len(offsets) < d:
            raise ValueError(
                f"cell {c} has only {len(offsets)} usable directions for the "
                f"least-squares gradient (need {d}); too many traction facets "
                f"or missing Dirichlet data"
            )
        X = np.asarray(offsets)  # (m, d)
        w = 1.0 / np.einsum("mi,mi->m", X, X)
        L = np.linalg.solve(X.T @ (w[:, None] * X), X.T * w)  # (d, m)

        # grad_h u_{k,c'} = sum_m L[c',m] (u_m,k - u_E,k); contract with gens
        for p in range(nsk):
            row = c * nsk + p
            coeff = np.einsum("kc,cm->km", gens[p], L) * vol[c]  # (d, m)
            for m, other in enumerate(neighbours):
                if other is None:
                    b[row] += coeff[:, m] @ data[m]
                else:
                    rows.extend([row] * d)
                    cols.extend(other * d + np.arange(d))
                    vals.extend(coeff[:, m])
            # the -u_E terms, summed over all fit rows
            rows.extend([row] * d)
            cols.extend(c * d + np.arange(d))
            vals.extend(-coeff.sum(axis=1))
    G = sp.csr_matrix(
        (vals, (rows, cols)), shape=(nsk * n_cells, d * n_cells)
    )
    return G, b


def two_point_skew_gradient(mesh, frame, mu, dirichlet=None, exclude_facets=()):
    """``(G, b)`` with ``(G u + b)_{E,p} = sum_e |e| gens_p : (u_hat_e (x) n_e)``.

    The **two-point** skew gradient: the surface form
    ``int_E gens_p : grad u = oint gens_p : (u (x) n)`` with the facet value
    ``u_hat_e`` reconstructed from at most the two neighbouring cells, by a
    ``mu``-weighted average:

    * interior facet: the ``mu/delta``-weighted average of the two cell values
      (``delta`` = distance from the cell centre to the facet plane).  The
      mu-weighting is not cosmetic: continuity of the shear traction gives
      ``mu_L g_L = mu_R g_R`` for the normal derivatives, which is exactly the
      cancellation that keeps the average consistent across a shear-modulus
      jump (the layered-shear state is reproduced exactly);
    * Dirichlet boundary facet: the datum at the facet centroid (into ``b``);
    * excluded facet (traction, or a fracture with its jump): the one-sided
      cell value -- an admissible closure because ``sum_e |e| n_e = 0``.

    Exact for affine displacements on tensor-product (Cartesian, graded)
    grids, where the cell-centre line crosses each facet at its centroid; on
    general face-orthogonal complexes the crossing point and the centroid
    differ by ``O(h)`` -- the consistency trade intrinsic to two-point
    reconstructions.
    """
    from mimetika.geometry.local_cell import LocalCell

    d = mesh.dim
    gens = skew_generators(d)
    nsk = len(gens)
    n_cells = mesh.num_cells(d)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n_cells,))
    inc = mesh.complex.boundary_matrix(d).tocsr()
    excluded = {int(f) for f in exclude_facets}
    centroids = mesh.geometry.centroids(d)
    centroids_f = mesh.geometry.centroids(d - 1)

    rows, cols, vals = [], [], []
    b = np.zeros(nsk * n_cells)

    def add(cell, p, coeff, other):
        """coeff (d,) multiplies the cell value of ``other``."""
        rows.extend([cell * nsk + p] * d)
        cols.extend(other * d + np.arange(d))
        vals.extend(coeff)

    for c in range(n_cells):
        lc = LocalCell.build(mesh.geometry, c, frame)
        for i, f in enumerate(lc.facet_ids):
            n_out = lc.facet_normals[i]  # outward, local frame
            weight_n = lc.facet_measures[i] * n_out
            adjacent = inc.indices[inc.indptr[f] : inc.indptr[f + 1]]
            interior = len(adjacent) == 2 and int(f) not in excluded
            for p in range(nsk):
                coeff = gens[p] @ weight_n  # (d,): multiplies u_hat components
                if interior:
                    other = int(adjacent[0] if adjacent[1] == c else adjacent[1])
                    x_f = lc.facet_centroids[i]
                    d_c = abs(x_f @ n_out)
                    d_o = abs((x_f - lc.to_local(centroids[other])[0]) @ n_out)
                    w_c = mu[c] / max(d_c, 1e-300)
                    w_o = mu[other] / max(d_o, 1e-300)
                    add(c, p, coeff * (w_c / (w_c + w_o)), c)
                    add(c, p, coeff * (w_o / (w_c + w_o)), other)
                elif len(adjacent) == 1 and int(f) not in excluded and dirichlet is not None:
                    value = np.asarray(
                        dirichlet(centroids_f[int(f)][None]), dtype=float
                    ).reshape(-1)[:3] @ frame
                    b[c * nsk + p] += coeff @ value
                else:  # excluded, or boundary without a datum: one-sided
                    add(c, p, coeff, c)
    G = sp.csr_matrix(
        (vals, (rows, cols)), shape=(nsk * n_cells, d * n_cells)
    )
    return G, b


class KinematicRotationElasticity(MixedElasticity):
    """Three-field AFW with the rotation defined kinematically.

    Same fields and boundary interface as
    :class:`~mimetika.assembly.mixed.MixedElasticity`; only the rotation row
    differs (module docstring).  The system is quasi-symmetric -- use a direct
    solver.
    """

    #: blend between the kinematic definition (``theta = 1``, gauge-singular)
    #: and the angular-momentum constraint (``theta = 0``, zero-block
    #: multiplier); any value strictly between is well posed.
    blend = 0.5

    #: how the skew gradient is reconstructed: ``"least_squares"`` (exact for
    #: affine fields on any mesh) or ``"two_point"`` (mu-weighted facet
    #: averages -- minimal stencil, exact on tensor grids).
    gradient = "least_squares"

    def _rotation_row(self, dirichlet, traction_facets=()):
        """``(Arow, Grow, Srow, b)`` for ``Arow sigma + Grow u + Srow s = b``.

        The blended row (module docstring):
        ``(1-theta)(1/2mu) A sigma - 2 theta |E| s + theta (G u + b_G) = 0``.
        """
        M, D, A = self.assemble_operators()
        theta = float(self.blend)
        if not 0.0 < theta < 1.0:
            raise ValueError(
                f"blend must lie strictly in (0, 1): theta = 1 has the "
                f"constant-skew gauge kernel and theta = 0 is the zero-block "
                f"multiplier form; got {theta}"
            )
        mu = np.repeat(self.inner._mu, self.n_skew)
        vol = np.repeat(self.mesh.geometry.measure(self.d), self.n_skew)
        excluded = set(int(f) for f in traction_facets)
        if self.contact is not None:
            # a fracture is an internal boundary: differencing the displacement
            # across it would smear the jump into a spurious rotation of the
            # fault-adjacent cells, at amplitude ~ jump/h.  Fault-adjacent
            # cells fit their gradient one-sidedly instead.
            excluded |= set(int(f) for f in self.contact.facets)
        if self.gradient == "two_point":
            G, b = two_point_skew_gradient(
                self.mesh,
                self.inner.frame,
                self.inner._mu,
                dirichlet,
                exclude_facets=excluded,
            )
        else:
            G, b = skew_gradient(
                self.mesh, self.inner.frame, dirichlet, exclude_facets=excluded
            )
        return (
            sp.diags((1.0 - theta) / (2.0 * mu)) @ A,
            (theta * G).tocsr(),
            sp.diags(-2.0 * theta * vol),
            -theta * b,
        )

    def assemble(
        self, body_force=None, dirichlet=None, extra_rhs=None, traction_facets=()
    ):
        M, D, A = self.assemble_operators()
        Ar, G, S, b = self._rotation_row(dirichlet, traction_facets)
        blocks = [[M, D.T, A.T], [D, None, None], [Ar, G, S]]
        Smat = sp.bmat(blocks, format="csr")
        traction_rhs = self.dirichlet_vector(dirichlet)
        if extra_rhs is not None:
            traction_rhs = traction_rhs + np.asarray(extra_rhs, dtype=float)
        rhs = np.concatenate(
            [traction_rhs, self.source_vector(body_force), b]
        )
        return Smat, rhs

    def assemble_constrained(
        self,
        body_force=None,
        dirichlet=None,
        extra_rhs=None,
        traction=None,
        traction_facets=(),
        roller_facets=(),
    ):
        from mimetika.assembly.mixed import _constrain

        S, rhs = self.assemble(
            body_force, dirichlet, extra_rhs, traction_facets=traction_facets
        )
        if len(traction_facets):
            dofs, values = self.traction_moments(traction_facets, traction)
            S, rhs = _constrain(S, rhs, dofs, values)
        if len(roller_facets):
            dofs = self.roller_dofs(roller_facets)
            S, rhs = _constrain(S, rhs, dofs, np.zeros(len(dofs)))
        return S, rhs

    def solve(self, **kwargs) -> MixedSolution:
        from mimetika.assembly.mixed import BOUNDARY_ARGUMENTS

        if kwargs.get("method", "direct") != "direct":
            raise ValueError(
                "the kinematic-rotation system is quasi-symmetric; MINRES does "
                "not apply -- use method='direct'"
            )
        kwargs.setdefault("method", "direct")
        boundary = {k: kwargs.pop(k) for k in BOUNDARY_ARGUMENTS if k in kwargs}
        S, rhs = self.assemble_constrained(**boundary)
        return self.split(solve_saddle(S, rhs, self.block_sizes, **kwargs))


class KinematicRotationFourField(FourFieldElasticity):
    """Four-field split with the rotation defined kinematically.

    The solid-pressure construction is untouched -- the hydrostatic stress is
    symmetric, so ``p_s`` never enters the rotation identity -- and the exact
    congruence with the three-field kinematic system holds exactly as for the
    multiplier forms.
    """

    blend = KinematicRotationElasticity.blend
    gradient = KinematicRotationElasticity.gradient

    def _rotation_row(self, dirichlet, traction_facets=()):
        return KinematicRotationElasticity._rotation_row(
            self, dirichlet, traction_facets
        )

    def assemble(
        self, body_force=None, dirichlet=None, extra_rhs=None, traction_facets=()
    ):
        M, D, A = self.assemble_operators()
        Gamma, B = self.solid_pressure_blocks()
        Ar, G, S, b = self._rotation_row(dirichlet, traction_facets)
        n_ps = self.n_cells
        blocks = [
            [M, Gamma.T, D.T, A.T],
            [Gamma, B, None, None],
            [D, None, None, None],
            [Ar, sp.csr_matrix((Ar.shape[0], n_ps)), G, S],
        ]
        Smat = sp.bmat(blocks, format="csr")
        traction_rhs = self.dirichlet_vector(dirichlet)
        if extra_rhs is not None:
            traction_rhs = traction_rhs + np.asarray(extra_rhs, dtype=float)
        rhs = np.concatenate(
            [
                traction_rhs,
                np.zeros(self.n_cells),
                self.source_vector(body_force),
                b,
            ]
        )
        return Smat, rhs

    assemble_constrained = KinematicRotationElasticity.assemble_constrained
    solve = KinematicRotationElasticity.solve


class TwoPointElasticity(KinematicRotationElasticity):
    """Three-field two-point elasticity: lumped stress + two-point rotation.

    The lumped inner product is built here (orthogonality guarded at
    construction); the kinematic rotation row uses the two-point skew
    gradient.  Note the three-field assembly must fold the volumetric
    rank-one term back into ``M``, so **this arrangement is not diagonal** --
    it exists for congruence testing and as the stepping stone; the scheme
    with every structural payoff is :class:`TwoPointFourField`.
    """

    gradient = "two_point"

    def __init__(
        self,
        mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        material=None,
        collocation=None,
        contact=None,
        orthogonality_tol: float = 1e-9,
    ) -> None:
        from mimetika.operators.lumped import LumpedDeviatoricStress

        inner = LumpedDeviatoricStress(
            mesh,
            mu=mu,
            lam=lam,
            material=material,
            collocation=collocation,
            orthogonality_tol=orthogonality_tol,
        )
        super().__init__(mesh, contact=contact, inner=inner)


class TwoPointFourField(KinematicRotationFourField):
    """The mimetic two-point stress scheme -- the endpoint of the lumping.

    Fields ``(sigma, p_s, u, s)`` with every structural property assembled at
    once, on a face-orthogonal complex with positive collocation distances:

    * ``M`` **diagonal** (the lumped deviatoric inner product; the volumetric
      part rides on ``p_s``, never folded back);
    * ``B`` and the rotation ``(s, s)`` block diagonal; ``Gamma``, ``A``,
      ``G`` all two-point: **each facet couples only its two neighbouring
      cells** -- the minimal fill-in of a cell-centred method, reached from
      the traction side;
    * no zero block anywhere: ``sigma`` eliminates facet-wise, ``p_s`` and
      ``s`` cell-wise, and no multiplier inf-sup condition exists;
    * no Cosserat terms: the rotation is the Cauchy macro rotation, closed by
      the blended kinematic row.

    The system is quasi-symmetric (direct solvers).  Consistency is that of
    the two-point ingredients: exact for constant stresses on tensor grids
    (patch test, layered shear across mu-jumps), O(h) on general
    face-orthogonal complexes.
    """

    gradient = "two_point"

    def __init__(
        self,
        mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        material=None,
        collocation=None,
        contact=None,
        orthogonality_tol: float = 1e-9,
    ) -> None:
        from mimetika.operators.lumped import LumpedDeviatoricStress

        inner = LumpedDeviatoricStress(
            mesh,
            mu=mu,
            lam=lam,
            material=material,
            collocation=collocation,
            orthogonality_tol=orthogonality_tol,
        )
        super().__init__(mesh, contact=contact, inner=inner)
