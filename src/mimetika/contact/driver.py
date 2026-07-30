r"""Contact driver: turns a :mod:`~mimetika.contact.laws` law into a solve.

The driver owns everything a constitutive law should not know about -- the
rotation into the facet frame, the conversion between traction *moments* and
pointwise values, assembly into the mixed elasticity system, and the outer
iteration.  A law only supplies its projection.

Augmented Lagrangian (Uzawa)
----------------------------
The multiplier ``lambda`` **is** the physical contact traction.  Each outer
iteration

1. solves the mechanics with the fracture traction *constrained* to ``lambda``
   (an essential condition, since the traction is a DOF here),
2. recovers the gap ``g`` from the jump operator, and
3. updates ``lambda <- law.project(lambda + r g)``.

It matters that the traction is constrained rather than tied to ``lambda + r g``
through a compliance: with the augmented relation inside the operator the solved
traction is the *trial* value, so an open fracture comes out carrying tension.
Constraining it keeps ``t = lambda`` exactly, and an open point is then
genuinely traction free.

The jump operator is the assembled traction row of the **unfractured** system,

    ``Jump_f(x) = ( M sigma + D^T u + A^T s )_f = -g_f`` ,

evaluated on the solution -- a linear functional, so it can be applied even
though that row was replaced by the constraint.

An exactly linear law needs no outer iteration at all: the driver detects it via
:meth:`~mimetika.contact.laws.ContactLaw.linear_compliance` and does one solve
with the compliance block.

Enforcement
-----------
``"averaged"`` applies the law to the facet-mean traction -- one state per
facet, which is what most discrete-fracture codes do.  ``"pointwise"`` applies
it at the facet quadrature points and re-integrates, which resolves partial
contact within a facet at the cost of state per point.  The choice is the
caller's; the law is written the same way either way.

Stepping
--------
:meth:`ContactDriver.solve_step` advances **one** step.  The caller owns the
loop, which keeps the driver free to be embedded in a staggered poromechanics
scheme later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mimetika.assembly.contact import FractureContact
from mimetika.assembly.mixed import MixedElasticity
from mimetika.contact.laws import ContactLaw


@dataclass
class ContactState:
    """Everything carried from one step to the next."""

    multiplier: np.ndarray  # traction moments per fracture facet, (nf, 9)
    internal: np.ndarray  # law state, (nf * points_per_facet, n_state)
    jump: np.ndarray  # facet-frame jump at the enforcement points
    solution: object = None
    iterations: int = 0
    converged: bool = True


@dataclass
class ContactDriver:
    """Solves mixed elasticity with a fracture contact law, one step at a time."""

    mesh: object
    facets: np.ndarray
    law: ContactLaw
    mu: float = 1.0
    lam: float = 1.0
    augmentation: float | None = None  # AL parameter r; None => derived
    enforcement: str = "averaged"  # or "pointwise"
    relaxation: float = 0.5  # Uzawa under-relaxation; 1.0 = none
    max_iterations: int = 200
    tolerance: float = 1e-10

    _geom: FractureContact = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.facets = np.asarray(sorted(int(f) for f in self.facets), dtype=np.int64)
        if self.enforcement not in ("averaged", "pointwise"):
            raise ValueError("enforcement must be 'averaged' or 'pointwise'")
        # geometry helper: facet frames, Gram matrices, compliance blocks
        self.dim = self.mesh.dim
        self.ndf = self.dim * self.dim
        self._geom = FractureContact(self.mesh, self.facets)
        self._r = (
            self.default_augmentation()
            if self.augmentation is None
            else np.full(self.n_points, float(self.augmentation))
        )

    def default_augmentation(self) -> np.ndarray:
        """A per-point augmentation parameter ``r``, from geometry and moduli.

        Uzawa converges only when ``r`` is comparable to the *stiffness the
        fracture sees*: the update ``lambda <- P(lambda + r g)`` contracts when
        ``r < 2 / compliance``, and oscillates in a two-cycle otherwise.  The
        surrounding rock behaves as a spring of compliance ``L / (2 mu + lambda)``
        where ``L`` is the distance from the two adjacent cell centroids to the
        facet, so the natural choice is its inverse.

        ``L`` is measured directly as ``|(x_f - x_E) . n_f|`` summed over the two
        cells.  A ``volume / area`` shortcut would be exact only for boxes -- for
        a tetrahedron it gives ``h/6`` instead of ``h/4``, mis-scaling ``r`` badly
        enough to stall the iteration.
        """
        d = self.dim
        modulus = 2.0 * self.mu + self.lam
        cell_c = self.mesh.geometry.centroids(d)
        facet_c = self.mesh.geometry.centroids(d - 1)
        out = []
        for f in self.facets:
            normal = self.mesh.geometry.facet_frame(int(f))[0]
            cells = self.mesh.complex.boundary_matrix(d).tocsr()[int(f)].indices
            length = sum(
                abs((facet_c[int(f)] - cell_c[int(c)]) @ normal) for c in cells
            )
            out += [modulus / length] * self.points_per_facet(int(f))
        return np.array(out)

    # -- enforcement points ---------------------------------------------------

    def points_per_facet(self, facet: int) -> int:
        if self.enforcement == "averaged":
            return 1
        return len(self.mesh.geometry.quadrature(self.dim - 1, int(facet))[0])

    @property
    def n_points(self) -> int:
        return sum(self.points_per_facet(int(f)) for f in self.facets)

    def _basis(self, facet: int):
        """``(values (npts, 3), weights (npts,))`` of the facet ``P_1`` basis."""
        d, k = self.dim, self.dim - 1
        g = self.mesh.geometry
        if self.enforcement == "averaged":
            row = np.zeros((1, d))
            row[0, 0] = 1.0
            return row, np.array([g.measure(k)[int(facet)]])
        qp, qw = g.quadrature(k, int(facet))
        rel = qp - g.centroids(k)[int(facet)]
        h = g.measure(k)[int(facet)] ** (1.0 / k)
        tangents = g.facet_frame(int(facet))[1:]
        cols = [np.ones(len(qp))] + [rel @ t / h for t in tangents]
        return np.column_stack(cols), qw

    def _frame(self, facet: int) -> np.ndarray:
        """``(d, d)`` rotation taking facet-frame components to mesh components."""
        return self._geom.rotation(int(facet))

    # -- moments <-> facet-frame values ---------------------------------------

    def to_values(self, moments: np.ndarray, facet: int) -> np.ndarray:
        """Traction moments on a facet -> facet-frame values at the enforcement points."""
        d = self.dim
        gram = self._geom.facet_gram(int(facet))
        coeffs = np.linalg.solve(gram, moments.reshape(d, d).T).T  # (comp, basis)
        B, _ = self._basis(int(facet))
        vals = coeffs @ B.T  # (3 comp, npts), global components
        return (self._frame(int(facet)).T @ vals).T  # -> facet frame, (npts, 3)

    def to_moments(self, values: np.ndarray, facet: int) -> np.ndarray:
        """Facet-frame values at the enforcement points -> traction moments."""
        glob = (self._frame(int(facet)) @ np.atleast_2d(values).T).T  # (npts, 3)
        B, w = self._basis(int(facet))
        return np.einsum("p,pb,pk->kb", w, B, glob).ravel()

    def _slice(self, index: int) -> slice:
        start = sum(self.points_per_facet(int(f)) for f in self.facets[:index])
        return slice(start, start + self.points_per_facet(int(self.facets[index])))

    # -- gather -----------------------------------------------------------------

    def tractions(self, stress: np.ndarray) -> np.ndarray:
        """Facet-frame traction at every enforcement point."""
        return np.vstack(
            [
                self.to_values(
                    stress[self.ndf * int(f) : self.ndf * (int(f) + 1)], int(f)
                )
                for f in self.facets
            ]
        )

    def gap(self, problem, solution) -> np.ndarray:
        """Facet-frame gap at the enforcement points, from the jump operator.

        ``g = -( M sigma + D^T u + A^T s )_f``, with ``M`` the *unfractured*
        inner product.  Positive normal component means the fracture is open.
        """
        M, D, A = problem.assemble_operators()
        r = (
            M @ solution["stress"]
            + D.T @ solution["displacement"]
            + A.T @ solution["rotation"]
        )
        out = []
        for f in self.facets:
            coeffs = -r[self.ndf * int(f) : self.ndf * (int(f) + 1)].reshape(
                self.dim, self.dim
            )  # (comp, basis)
            B, _ = self._basis(int(f))
            out.append((self._frame(int(f)).T @ (coeffs @ B.T)).T)
        return np.vstack(out)

    # -- the solve ----------------------------------------------------------------

    def _problem(self, compliance) -> MixedElasticity:
        """Mixed elasticity carrying a given facet-frame compliance on the fracture."""
        geom = FractureContact(self.mesh, self.facets, facet_compliance=compliance)
        return MixedElasticity(self.mesh, mu=self.mu, lam=self.lam, contact=geom)

    def initial_state(self) -> ContactState:
        return ContactState(
            multiplier=np.zeros((len(self.facets), self.ndf)),
            internal=self.law.initial_state(self.n_points),
            jump=np.zeros((self.n_points, self.dim)),
        )

    def solve_step(
        self, dirichlet=None, body_force=None, state: ContactState | None = None,
        dt: float | None = None, **kwargs,
    ) -> ContactState:
        """Advance one load/time step. The caller owns the loop."""
        state = self.initial_state() if state is None else state
        kwargs.setdefault("method", "direct")

        exact = self.law.linear_compliance(self.dim)
        if exact is not None:  # linear law: one solve, no outer iteration
            problem = self._problem(exact)
            sol = problem.solve(body_force=body_force, dirichlet=dirichlet, **kwargs)
            traction = self.tractions(sol["stress"])
            return ContactState(
                multiplier=np.zeros_like(state.multiplier),
                internal=state.internal,
                jump=np.einsum("ij,pj->pi", exact, traction),
                solution=sol,
                iterations=1,
            )

        # Uzawa: constrain the traction to lambda, recover the gap, project
        problem = MixedElasticity(self.mesh, mu=self.mu, lam=self.lam)
        lam, g_prev, internal = state.multiplier.copy(), state.jump, state.internal
        dofs = np.concatenate(
            [self.ndf * int(f) + np.arange(self.ndf) for f in self.facets]
        )

        change, scale, g, projected = np.inf, 1.0, state.jump, None
        for it in range(1, self.max_iterations + 1):
            sol = self._solve_with_traction(
                problem, dofs, lam.ravel(), dirichlet, body_force, **kwargs
            )
            g = self.gap(problem, sol)
            trial = (
                np.vstack(
                    [self.to_values(lam[i], int(f)) for i, f in enumerate(self.facets)]
                )
                + self._r[:, None] * g
            )
            projected, internal = self.law.project(trial, internal, g, g_prev, dt)

            target = np.vstack(
                [
                    self.to_moments(projected[self._slice(i)], int(f))
                    for i, f in enumerate(self.facets)
                ]
            )
            # Under-relax: in the *sliding* regime the tangential update is not a
            # contraction, and the plain fixed point settles into a limit cycle.
            lam_new = lam + self.relaxation * (target - lam)
            change = np.abs(lam_new - lam).max()
            scale = max(np.abs(lam_new).max(), 1.0)
            lam = lam_new
            if change <= self.tolerance * scale:
                break

        internal = self.law.advance(projected, g, internal, dt, g_prev)
        return ContactState(
            multiplier=lam, internal=internal, jump=g, solution=sol,
            iterations=it, converged=change <= self.tolerance * scale,
        )

    def _solve_with_traction(
        self, problem, dofs, values, dirichlet, body_force, **kwargs
    ):
        """Solve with the fracture traction constrained to ``values``."""
        from mimetika.solver.saddle import solve_saddle

        S, rhs = problem.assemble(body_force=body_force, dirichlet=dirichlet)
        S = S.tolil(copy=True)
        rhs = rhs - S[:, dofs] @ values
        for d, v in zip(dofs, values):
            S[d, :] = 0.0
            S[:, d] = 0.0
            S[d, d] = 1.0
            rhs[d] = v
        blocks = (problem.n_stress, (problem.d + problem.n_skew) * problem.n_cells)
        x = solve_saddle(S.tocsr(), rhs, blocks, **kwargs)
        n1 = problem.n_stress
        n2 = n1 + problem.d * problem.n_cells
        from mimetika.assembly.mixed import MixedSolution

        return MixedSolution(
            {"stress": x[:n1], "displacement": x[n1:n2], "rotation": x[n2:]}
        )
