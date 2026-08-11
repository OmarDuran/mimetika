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
from dataclasses import replace as dc_replace

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.contact import FractureContact
from mimetika.contact.laws import ContactLaw
from mimetika.contact.map import ContactMap, fixed_point, newton
from mimetika.solver.saddle import solve_saddle


@dataclass
class ContactState:
    """Everything carried from one step to the next."""

    multiplier: np.ndarray  # traction moments per fracture facet, (nf, 9)
    internal: np.ndarray  # law state, (nf * points_per_facet, n_state)
    jump: np.ndarray  # facet-frame jump at the enforcement points
    solution: object = None
    #: the mechanics problem the solution belongs to -- needed to interpret it
    #: (cell stresses, DOF layout), and the caller never built it
    problem: object = None
    iterations: int = 0
    converged: bool = True
    #: the condensed map of the solve, when one was built -- pass it back as
    #: ``solve_step(..., reuse=...)`` to skip the factorisation and ``Ghat``
    #: when only the rhs or a law parameter changed
    condensed: object = None


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
    #: in-situ traction at the enforcement points, ``(n_points, dim)``.  Fracture
    #: state, not a boundary condition: a law constrains the *total* traction, so
    #: an incremental solve has to tell it what it is sitting on top of.
    prestress: np.ndarray | None = None
    #: traction DOFs per facet of the stress space the mechanics uses:
    #: ``d^2`` for AFW (the default), ``d`` for the lumped space, whose facet
    #: basis is the constant alone.  Must match the problems the ``mechanics``
    #: factory builds -- the driver reads and pins stress DOFs by this stride.
    dofs_per_facet: int | None = None

    _geom: FractureContact = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.facets = np.asarray(sorted(int(f) for f in self.facets), dtype=np.int64)
        if self.enforcement not in ("averaged", "pointwise"):
            raise ValueError("enforcement must be 'averaged' or 'pointwise'")
        # geometry helper: facet frames, Gram matrices, compliance blocks
        self.dim = self.mesh.dim
        self.ndf = (
            self.dim * self.dim
            if self.dofs_per_facet is None
            else int(self.dofs_per_facet)
        )
        self.nb = self.ndf // self.dim  # facet basis size: d for AFW, 1 lumped
        if self.nb * self.dim != self.ndf or not 1 <= self.nb <= self.dim:
            raise ValueError(
                f"dofs_per_facet must be d * nb with 1 <= nb <= d; got "
                f"{self.ndf} in dimension {self.dim}"
            )
        self._geom = FractureContact(
            self.mesh, self.facets, dofs_per_facet=self.ndf
        )
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
        """``(values (npts, 3), weights (npts,))`` of the facet ``P_1`` basis.

        The linear functions are scaled by ``sqrt(|f|)``, which is the scaling
        :meth:`FractureContact.facet_gram` and
        :meth:`LocalCell.facet_scalar_basis` -- the basis the stress DOFs are
        actually defined against -- both use.

        It used to be ``|f| ** (1/k)``.  That agrees for ``k = 2`` (a polygonal
        facet of a 3D cell) and disagrees for ``k = 1`` (an edge of a 2D cell),
        where it gives ``|f|`` instead of ``sqrt(|f|)``.  The Gram matrix then
        came out a factor ``|f|`` too small, so ``to_values`` inverted a different
        basis from the one ``to_moments`` integrated against and the round trip
        was not the identity -- silently, and only in 2D.
        """
        d, k = self.dim, self.dim - 1
        g = self.mesh.geometry
        if self.enforcement == "averaged":
            row = np.zeros((1, self.nb))
            row[0, 0] = 1.0
            return row, np.array([g.measure(k)[int(facet)]])
        qp, qw = g.quadrature(k, int(facet))
        rel = qp - g.centroids(k)[int(facet)]
        h = np.sqrt(g.measure(k)[int(facet)])
        tangents = g.facet_frame(int(facet))[1:]
        cols = [np.ones(len(qp))] + [rel @ t / h for t in tangents]
        return np.column_stack(cols)[:, : self.nb], qw

    def _frame(self, facet: int) -> np.ndarray:
        """``(d, d)`` rotation taking facet-frame components to mesh components."""
        return self._geom.rotation(int(facet))

    # -- moments <-> facet-frame values ---------------------------------------

    def to_values(self, moments: np.ndarray, facet: int) -> np.ndarray:
        """Traction moments on a facet -> facet-frame values at the enforcement points."""
        d = self.dim
        gram = self._geom.facet_gram(int(facet))[: self.nb, : self.nb]
        coeffs = np.linalg.solve(gram, moments.reshape(d, self.nb).T).T  # (comp, basis)
        B, _ = self._basis(int(facet))
        vals = coeffs @ B.T  # (3 comp, npts), global components
        return (self._frame(int(facet)).T @ vals).T  # -> facet frame, (npts, 3)

    def to_moments(self, values: np.ndarray, facet: int) -> np.ndarray:
        """Facet-frame values at the enforcement points -> traction moments."""
        glob = (self._frame(int(facet)) @ np.atleast_2d(values).T).T  # (npts, 3)
        B, w = self._basis(int(facet))
        return np.einsum("p,pb,pk->kb", w, B, glob).ravel()

    def expand_to_points(self, per_facet) -> np.ndarray:
        """Repeat one value per facet across that facet's enforcement points.

        Only the driver knows how many points a facet carries -- one under
        ``averaged``, one per quadrature point under ``pointwise`` -- so data
        supplied per facet, such as an in-situ prestress, has to be expanded
        here rather than by the caller.
        """
        per_facet = np.atleast_2d(np.asarray(per_facet, dtype=float))
        return np.repeat(
            per_facet,
            [self.points_per_facet(int(f)) for f in self.facets],
            axis=0,
        )

    def per_facet(self, values) -> np.ndarray:
        """Collapse enforcement-point values to one per facet -- the inverse of
        :meth:`expand_to_points`.

        Anything indexed by facet -- a plot against position, a VTU cell array --
        needs this first.  Under ``pointwise`` there are several points per facet,
        so indexing a point-valued array with facet indices silently reads the
        wrong entries and pairs them with the wrong facets.
        """
        values = np.atleast_2d(np.asarray(values, dtype=float))
        if len(values) == len(self.facets):
            return values
        return np.vstack([values[self._slice(i)].mean(axis=0)
                          for i in range(len(self.facets))])

    def _slice(self, index: int) -> slice:
        # cumulative offsets, cached: the naive per-call sum is quadratic in
        # the facet count and showed up as millions of generator evaluations
        offsets = self.__dict__.get("_point_offsets")
        if offsets is None:
            counts = [self.points_per_facet(int(f)) for f in self.facets]
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)
            self.__dict__["_point_offsets"] = offsets
        return slice(int(offsets[index]), int(offsets[index + 1]))

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

    def gap(self, problem, solution, rhs=None) -> np.ndarray:
        """Facet-frame gap at the enforcement points, from the jump operator.

        ``g = -( M sigma + D^T u + A^T s - b_f )_f``, with ``M`` the
        *unfractured* inner product and ``b_f`` the mechanics right-hand side
        on those rows (pass ``rhs`` to include it -- see :meth:`jump_offset`).
        Positive normal component means the fracture is open.
        """
        x = np.concatenate([solution[k] for k in solution.blocks])
        r = problem.constitutive_rows(contact=False) @ x
        if rhs is not None:
            r = r - np.asarray(rhs, dtype=float)
        out = []
        for f in self.facets:
            coeffs = -r[self.ndf * int(f) : self.ndf * (int(f) + 1)].reshape(
                self.dim, self.nb
            )  # (comp, basis)
            B, _ = self._basis(int(f))
            out.append((self._frame(int(f)).T @ (coeffs @ B.T)).T)
        return np.vstack(out)

    # -- the linear operators the map is made of -------------------------------

    def moment_operator(self) -> sp.csr_matrix:
        """``W``: facet-frame values at the enforcement points -> traction moments.

        The matrix form of :meth:`to_moments`, assembled once.  Columns are
        indexed ``(point, component)`` and rows ``(facet, component, basis)``.
        """
        rows, cols, vals = [], [], []
        for i, f in enumerate(self.facets):
            frame = self._frame(int(f))  # (d, d): facet frame -> mesh components
            B, w = self._basis(int(f))
            # W[(k, b), (p, j)] = w[p] B[p, b] frame[k, j]
            block = np.einsum("p,pb,kj->kbpj", w, B, frame)
            base_row = self.ndf * i
            base_col = self.dim * self._slice(i).start
            idx = np.indices(block.shape)
            rows.append(base_row + idx[0].ravel() * self.nb + idx[1].ravel())
            cols.append(base_col + idx[2].ravel() * self.dim + idx[3].ravel())
            vals.append(block.ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.ndf * len(self.facets), self.dim * self.n_points),
        )

    def jump_operator(self, problem) -> sp.csr_matrix:
        """``J``: solution vector -> facet-frame gap at the enforcement points.

        The gap is a **linear** functional of the solution -- the assembled
        traction row of the *unfractured* system
        (:meth:`~mimetika.assembly.mixed.MixedElasticity.constitutive_rows`,
        in whatever field layout the problem uses; in three-field terms

            ``g_f = -( M sigma + D^T u + A^T s )_f`` ),

        rotated into the facet frame and evaluated at the enforcement points.
        Because it is linear it can be applied even though that row was replaced
        by the contact constraint, and because it is a matrix the contact map
        never needs the mesh.

        No ``Gram^{-1}`` here, and it is worth saying why, because the residual
        looks like a moment vector and :meth:`to_values` *does* invert the Gram.
        The two convert different objects.  A traction DOF **is** a moment
        ``m = int_e (sigma n) b``, so recovering a traction's pointwise values
        needs ``Gram^{-1} m`` -- that is :meth:`to_values`.  The jump term
        ``int_e [[u]] . (tau n)`` is instead paired *against* that moment DOF:
        writing ``(tau n) = sum_b phi_b b_b`` gives ``m = Gram phi``, so the
        pairing already carries a ``Gram^{-1}`` and the residual emerges as the
        expansion **coefficients** of the jump, ready to evaluate against the
        basis.  Inserting a second ``Gram^{-1}`` divides the jump by ``|e|``, and
        the resulting slip then grows like ``1/h`` under refinement.
        """
        # the assembled stress-row block of the *unfractured* system, in
        # whatever field layout the problem uses -- three- and four-field
        # evaluate the same functional.  Stripping the fracture compliance is
        # essential: at the solution the fractured row is satisfied exactly, so
        # its residual is zero, whereas the unfractured residual is A_f sigma,
        # i.e. the jump this operator exists to extract.
        traction_rows = problem.constitutive_rows(contact=False)

        blocks, rows, cols, vals = [], [], [], []
        for i, f in enumerate(self.facets):
            frame = self._frame(int(f))
            B, _ = self._basis(int(f))
            # values[(p, j)] = -sum_{k,b} frame[k, j] B[p, b] residual[(k, b)]
            block = -np.einsum("kj,pb->pjkb", frame, B)
            idx = np.indices(block.shape)
            base_row = self.dim * self._slice(i).start
            base_col = self.ndf * int(f)
            rows.append(base_row + idx[0].ravel() * self.dim + idx[1].ravel())
            cols.append(base_col + idx[2].ravel() * self.nb + idx[3].ravel())
            vals.append(block.ravel())
        gather = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim * self.n_points, traction_rows.shape[0]),
        )
        return (gather @ traction_rows).tocsr()

    def jump_offset(self, rhs: np.ndarray) -> np.ndarray:
        """Gap contribution of the mechanics right-hand side on the fault rows.

        The gap is the **residual** of the replaced constitutive rows,
        ``g = -(row . z - b_f)``, at the enforcement points: whatever the
        assembly put into ``b_f`` on the fault facets -- notably the Biot
        pore-pressure coupling -- belongs in the gap.  Reading ``J z`` alone
        instead imposes a spurious jump with ``b_f``'s coefficients, which on
        an unstructured mesh alternate facet to facet and rattle the whole
        contact solution.
        """
        out = []
        for f in self.facets:
            coeffs = np.asarray(
                rhs[self.ndf * int(f): self.ndf * (int(f) + 1)], dtype=float
            ).reshape(self.dim, self.nb)
            B, _ = self._basis(int(f))
            out.append((self._frame(int(f)).T @ (coeffs @ B.T)).T)
        return np.vstack(out)

    def contact_dofs(self) -> np.ndarray:
        """Indices of the fracture traction unknowns, in ``moment_operator`` order."""
        return np.concatenate(
            [self.ndf * int(f) + np.arange(self.ndf) for f in self.facets]
        )

    def contact_geometry(self, compliance=None) -> FractureContact:
        """The fracture geometry, optionally carrying a linear compliance block.

        Constitutive and geometric data -- the driver's own business.  Handed to
        whoever builds the mechanics so the fracture is embedded in ``A``.
        """
        return FractureContact(
            self.mesh,
            self.facets,
            facet_compliance=compliance,
            dofs_per_facet=self.ndf,
        )

    def contact_map(self, problem, matrix, rhs, **solver) -> ContactMap:
        """Assemble the nonlinear algebraic map ``y = CD(x)`` for a given system.

        ``(matrix, rhs)`` is the caller's mechanics, boundary conditions already
        applied.  Everything mesh-dependent is baked into matrices here, so the
        returned map is pure algebra.
        """
        solver = solver or {"method": "direct"}
        return ContactMap(
            matrix=matrix,
            rhs=rhs,
            dofs=self.contact_dofs(),
            to_moments=self.moment_operator(),
            jump=self.jump_operator(problem),
            gap_shift=self.jump_offset(rhs),
            augmentation=self._r,
            law=self.law,
            block_sizes=problem.block_sizes,
            solver=solver,
            prestress=self.prestress,
        )

    def initial_state(self) -> ContactState:
        return ContactState(
            multiplier=np.zeros((len(self.facets), self.ndf)),
            internal=self.law.initial_state(self.n_points),
            jump=np.zeros((self.n_points, self.dim)),
        )

    # -- convenience: build the map and drive it to its fixed point ---------------

    def solve_step(
        self, mechanics, state: ContactState | None = None,
        dt: float | None = None, condense: bool = False,
        solver: str = "picard", reuse=None, recover: bool = True, **kwargs,
    ) -> ContactState:
        """Advance one load/time step by solving ``x = CD(x)``.

        ``mechanics(contact) -> (problem, A, rhs)`` is supplied by the caller and
        is the *only* route by which boundary conditions, materials or a
        pore-pressure right-hand side reach the contact problem.  The driver
        never names one.

        A law with an exact linear compliance needs no iteration at all: the
        compliance goes straight into ``A`` and one solve finishes it.
        """
        state = self.initial_state() if state is None else state
        kwargs.setdefault("method", "direct")

        exact = self.law.linear_compliance(self.dim)
        if exact is not None:
            problem, A, rhs = mechanics(self.contact_geometry(exact))
            sol = problem.split(solve_saddle(A, rhs, problem.block_sizes, **kwargs))
            traction = self.tractions(sol["stress"])
            return ContactState(
                multiplier=np.zeros_like(state.multiplier),
                internal=state.internal,
                jump=np.einsum("ij,pj->pi", exact, traction),
                solution=sol,
                problem=problem,
                iterations=1,
            )

        problem, A, rhs = mechanics(None)
        cd = self.contact_map(problem, A, rhs, **kwargs)
        # condensing eliminates the mechanics once, after which each iteration is
        # a small dense matvec instead of a global solve -- worth it whenever the
        # iteration count exceeds the number of contact unknowns
        driven = cd
        if condense or solver == "newton":
            driven = cd.condense(reuse=reuse)
            if self.augmentation is None and solver != "newton":
                # the condensed operator is the exact fracture compliance, so it
                # beats the geometric estimate -- decisively so for a fault that
                # cuts the domain, where the local guess is far too stiff
                driven = driven.rescaled()
        common = dict(
            x0=self.values_of(state.multiplier),
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            internal=state.internal,
            g_prev=state.jump,
            dt=dt,
        )
        if solver == "newton":
            result = newton(driven, **common)
        elif solver == "picard":
            result = fixed_point(driven, relaxation=self.relaxation, **common)
        else:
            raise ValueError(f"unknown solver {solver!r}")
        evaluation = result.evaluation
        if evaluation.solution is None and recover:
            # condensed: recover the field once.  Callers that only read the
            # jump (an outer iteration on a law parameter) skip this with
            # ``recover=False`` -- one back-substitution saved per solve.
            z = driven.recover(result.x) if hasattr(driven, "recover") else None
            if z is not None:
                evaluation = dc_replace(evaluation, solution=z)
            else:
                evaluation = cd(result.x, internal=state.internal,
                                g_prev=state.jump, dt=dt)
        internal = self.law.advance(
            evaluation.value, evaluation.gap, evaluation.internal, dt, state.jump
        )
        return ContactState(
            multiplier=(cd.to_moments @ result.x.ravel()).reshape(
                len(self.facets), self.ndf
            ),
            internal=internal,
            jump=evaluation.gap,
            solution=(problem.split(evaluation.solution)
                      if evaluation.solution is not None else None),
            problem=problem,
            iterations=result.iterations,
            converged=result.converged,
            condensed=driven if driven is not cd else None,
        )

    def values_of(self, multiplier: np.ndarray) -> np.ndarray:
        """Traction moments per facet -> facet-frame values, the space ``CD`` uses."""
        multiplier = np.atleast_2d(multiplier)
        return np.vstack(
            [self.to_values(multiplier[i], int(f)) for i, f in enumerate(self.facets)]
        )[:, : self.dim]


def elastic_mechanics(mesh, mu: float = 1.0, lam: float = 1.0, **boundary):
    """A ``mechanics`` factory for homogeneous mixed elasticity.

    Belongs to the **caller** side of the seam: it is what closes over the
    boundary data so the driver never sees any.  For a different problem --
    per-cell materials, poromechanics, a pressure-driven right-hand side -- write
    another factory with the same three-value signature; nothing downstream can
    tell the difference.

    Builds the **four-field** formulation -- the standard one.  The driver only
    touches the problem through ``block_sizes``, ``split`` and
    ``constitutive_rows``, so a factory returning the classic three-field
    :class:`~mimetika.assembly.mixed.MixedElasticity` works identically.
    """
    from mimetika.assembly.four_field import FourFieldElasticity

    def build(contact=None):
        problem = FourFieldElasticity(mesh, mu=mu, lam=lam, contact=contact)
        matrix, rhs = problem.assemble_constrained(**boundary)
        return problem, matrix, rhs

    return build
