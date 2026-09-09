r"""The contact problem as a nonlinear algebraic function ``y = CD(x)``.

Contact is a fixed-point problem in the contact traction, and nothing more::

    y = CD(x) ,    CD(x) = P( x + r g(x) ) ,    g(x) = J z ,    A(x) z = b(x)

where ``A(x)`` is the mechanics system with the fracture traction DOFs pinned to
``x``.  One evaluation is: pin, solve, read the gap, project.  The solution of
``x = CD(x)`` is the converged contact state.

The inputs are a matrix, a right-hand side, an index set, two linear maps and a
projection -- no mesh, material, boundary condition or problem object.  So what
assembles ``(A, b)`` and what iterates on ``CD`` are both interchangeable, and
``CD`` can be exercised on stub operators.

What ``x`` contains
-------------------
``x`` and ``y`` are the same object: the contact traction at the enforcement
points, in the facet frame, shape ``(n_points, dim)``, normal component first::

    dim = 2:   x[p] = (t_n, t_t)            one shear direction
    dim = 3:   x[p] = (t_n, t_t1, t_t2)     two shear directions

with ``t_n < 0`` in compression and ``g_n > 0`` open.  The space is the same for
every law; what changes is the subset ``CD`` can return and the state carried:

======================  ========================================  ================  ===
law                     admissible set of ``y = CD(x)``           internal state    dt
======================  ========================================  ================  ===
FrictionlessBilateral   ``t_t = 0``, ``t_n`` free                 none              no
LinearContact           all of ``R^dim`` (bonded)                 none              no
SignoriniCoulomb        ``t_n <= 0``, ``|t_t| <= -mu t_n + c``    slip (1)          no
AssociativeMohrCoulomb  same set, closest-point return            slip (1)          no
RateAndStateFriction    ``t_n <= 0, |t_t| <= -mu(V, theta) t_n``  slip, theta (2)   yes
======================  ========================================  ================  ===

:class:`LinearContact` constrains nothing, so its projection is the identity and
its fixed point is reached in one evaluation; the driver takes it through the
compliance block instead, so it does not reach ``CD``.  The sets of
:class:`FrictionlessBilateral` and of :class:`SignoriniCoulomb` without cohesion
are cones through the origin, so their projections commute with positive
scaling; cohesion shifts the set off the origin and breaks that.

The conversion to the traction moments the linear system constrains is the
linear map ``to_moments``; the gap comes back through the linear map ``jump``.
Both are supplied as matrices by the discretisation.

Prestress
---------
A contact law constrains the total traction: Signorini says the total normal
traction is compressive, not the increment.  When only an increment is solved
for -- a depletion response on top of an in-situ state -- a unilateral condition
shown the increment alone reads a tensile increment on a closed fault as
opening.  ``prestress`` carries the in-situ traction at the enforcement points,
added before the projection and removed after, so ``x`` stays the incremental
unknown while the law sees the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.solver.saddle import solve_saddle


def driving_gap(gap: np.ndarray, g_prev=None) -> np.ndarray:
    """What the augmentation multiplies: total normal gap, tangential increment.

    The normal condition ``g_n >= 0`` constrains the absolute gap, so the normal
    term is driven by the total jump.  Coulomb friction opposes the slip rate:
    eq. (2e) of Frigo et al. (2025) reads ``g_T . t_T = tau_max |g_T|`` with
    ``g_T`` a rate, which a quasi-static scheme discretises as the backward
    increment ``Delta_n g_T = g_T,n - g_T,n-1``.

    Driving the tangential part with the total jump is equivalent only under
    monotone proportional loading (the first step from rest, or a path along a
    fixed direction).  Once the slip direction rotates or reverses, the total
    jump points along the accumulated path and the traction lags the direction
    it should oppose.
    """
    gap = np.asarray(gap, dtype=float)
    if g_prev is None:
        return gap
    out = gap.copy()
    out[:, 1:] = gap[:, 1:] - np.atleast_2d(np.asarray(g_prev, dtype=float))[:, 1:]
    return out


@dataclass
class MapEvaluation:
    """One evaluation of :class:`ContactMap`."""

    value: np.ndarray  # y = CD(x), (n_points, dim)
    gap: np.ndarray  # g(x) at the enforcement points, (n_points, dim)
    internal: np.ndarray  # law state after the projection
    solution: np.ndarray  # the raw solution vector z of the pinned system

    @property
    def residual(self) -> np.ndarray:
        """``CD(x) - x`` is not available here -- see :meth:`ContactMap.residual`."""
        raise AttributeError("use ContactMap.residual(x, ...)")


@dataclass
class ContactMap:
    """The nonlinear algebraic map ``y = CD(x)`` of augmented-Lagrangian contact.

    Parameters
    ----------
    matrix, rhs
        The assembled mechanics system, all boundary conditions applied.
    dofs
        Indices of the fracture traction unknowns, in the order ``to_moments``
        produces them.
    to_moments
        ``(len(dofs), n_points * dim)``: facet-frame values at the enforcement
        points to traction moments.
    jump
        ``(n_points * dim, len(rhs))``: solution vector to facet-frame gap.
    augmentation
        ``r``, one per enforcement point.  The iteration contracts only for
        ``r`` comparable to the stiffness the fracture sees.
    law
        Supplies the projection onto the admissible set.
    block_sizes
        Field block sizes, forwarded to :func:`solve_saddle`.
    """

    matrix: sp.spmatrix
    rhs: np.ndarray
    dofs: np.ndarray
    to_moments: sp.spmatrix
    jump: sp.spmatrix
    augmentation: np.ndarray
    law: object
    block_sizes: tuple
    solver: dict = field(default_factory=lambda: {"method": "direct"})
    #: in-situ traction at the enforcement points; the law sees ``x + prestress``
    prestress: np.ndarray | None = None
    #: gap contribution of the mechanics rhs on the replaced fault rows,
    #: ``(n_points, dim)``.  The gap is the residual of those rows,
    #: ``-(row . z - b_f)``; reading ``J z`` alone imposes a spurious jump
    #: equal to ``b_f``'s coefficients (e.g. the Biot pore-coupling term).
    gap_shift: np.ndarray | None = None

    @property
    def n_points(self) -> int:
        return len(self.augmentation)

    @property
    def dim(self) -> int:
        return self.jump.shape[0] // self.n_points

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_points, self.dim)

    def initial_guess(self) -> np.ndarray:
        return np.zeros(self.shape)

    # -- the map ---------------------------------------------------------------

    def __call__(
        self, x, internal=None, g_prev=None, dt=None
    ) -> MapEvaluation:
        """Evaluate ``y = CD(x)``: pin, solve, read the gap, project."""
        from mimetika.assembly.mixed import _constrain

        x = np.asarray(x, dtype=float).reshape(self.shape)
        moments = self.to_moments @ x.ravel()
        A, b = _constrain(self.matrix, self.rhs, self.dofs, moments)
        z = solve_saddle(A, b, self.block_sizes, **self.solver)

        gap = (self.jump @ z).reshape(self.shape)
        if self.gap_shift is not None:
            gap = gap + self.gap_shift
        offset = 0.0 if self.prestress is None else self.prestress
        trial = x + offset + self.augmentation[:, None] * driving_gap(gap, g_prev)
        if internal is None:
            internal = self.law.initial_state(self.n_points)
        if getattr(self.law, "wants_augmentation", False):
            self.law._augmentation = self.augmentation
        y, internal = self.law.project(trial, internal, gap, g_prev, dt)
        return MapEvaluation(
            value=np.asarray(y).reshape(self.shape) - offset,
            gap=gap,
            internal=internal,
            solution=z,
        )

    # -- condensation ------------------------------------------------------------

    def condense(self, reuse: "CondensedContactMap | None" = None
                 ) -> "CondensedContactMap":
        """Reduce to the contact unknowns alone -- no linear solve per evaluation.

        The nonlinear system has ``n_points * dim`` unknowns, a handful per
        fracture facet.  The constrained matrix does not depend on ``x``:
        pinning zeroes the same rows and columns whatever the pinned values are,
        and the right-hand side depends on them affinely::

            ``b(x) = b_0 + B W x`` ,   ``z(x) = A^{-1} b(x)`` ,
            ``g(x) = g_0 + Ghat x`` ,  ``Ghat = J A^{-1} B W`` .

        One factorisation and ``n_points * dim + 1`` back-substitutions give the
        small dense ``Ghat``, after which ``CD`` is a matvec and a projection,
        with no global solve.

        Worth it whenever the iteration count exceeds the contact DOF count,
        the usual case for friction; for a very large fracture and a near-linear
        law the uncondensed form can still win.

        ``reuse`` skips the factorisation and ``Ghat``: pass the condensed map
        of a previous system with the same matrix (a new load level, a new law
        parameter) and only the affine offset is redone -- one back-substitution
        instead of ``n + 1`` plus a factorisation.
        """
        from mimetika.assembly.mixed import constraint_scales

        n = self.n_points * self.dim
        shift = 0.0 if self.gap_shift is None else self.gap_shift
        # with zero pinned values _constrain reduces to zeroing the pinned rhs
        # entries (rhs - A[:, dofs] @ 0, then rhs[dofs] = 0 * scales)
        b0 = np.asarray(self.rhs, dtype=float).copy()
        b0[self.dofs] = 0.0

        if reuse is not None and reuse.factor is not None:
            if reuse.b0 is not None and np.array_equal(b0, reuse.b0):
                # identical rhs (same load level): the affine offset is
                # already correct -- no back-substitution at all
                return replace(
                    reuse,
                    augmentation=self.augmentation,
                    law=self.law,
                    prestress=self.prestress,
                )
            base = reuse.factor.solve(b0)
            return replace(
                reuse,
                gap_offset=(self.jump @ base).reshape(self.shape) + shift,
                augmentation=self.augmentation,
                law=self.law,
                prestress=self.prestress,
                base=base,
                b0=b0,
            )

        from mimetika.assembly.mixed import _constrain

        scales = constraint_scales(self.matrix, self.dofs)
        A0, _ = _constrain(self.matrix, self.rhs, self.dofs,
                           np.zeros(len(self.dofs)))

        # b(v) - b_0 = -A[:, dofs] v, with the pinned rows overwritten by scale * v
        columns = -self.matrix[:, self.dofs].tolil()
        columns[self.dofs, :] = sp.diags(scales)
        load = (columns.tocsr() @ self.to_moments).tocsc()  # (N, n), sparse

        factor = spla.splu(sp.csc_matrix(A0))
        base = factor.solve(b0)
        # contract J A^{-1} B W in column blocks of 64: only the projection of
        # the (N, n) dense response matrix onto the fault rows is needed, so it
        # is never materialised
        gap_matrix = np.empty((self.jump.shape[0], n))
        step = 64
        for j0 in range(0, n, step):
            block = np.asarray(load[:, j0:j0 + step].todense())
            gap_matrix[:, j0:j0 + step] = self.jump @ factor.solve(block)
        return CondensedContactMap(
            gap_offset=(self.jump @ base).reshape(self.shape) + shift,
            gap_matrix=gap_matrix,
            augmentation=self.augmentation,
            law=self.law,
            shape=self.shape,
            prestress=self.prestress,
            factor=factor,
            load=load,
            base=base,
            b0=b0,
        )

    def residual(self, x, **kwargs) -> np.ndarray:
        """``CD(x) - x`` -- zero exactly at the contact solution."""
        x = np.asarray(x, dtype=float).reshape(self.shape)
        return self(x, **kwargs).value - x


@dataclass
class FixedPointResult:
    x: np.ndarray
    evaluation: MapEvaluation
    iterations: int
    converged: bool
    change: float


def fixed_point(
    contact_map: ContactMap,
    x0=None,
    relaxation: float = 0.5,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    internal=None,
    g_prev=None,
    dt=None,
) -> FixedPointResult:
    """Solve ``x = CD(x)`` by relaxed Picard iteration.

    While the fracture sticks the tangential update is a contraction and
    ``relaxation = 1`` converges; while it slides it is not, and the undamped
    iteration settles into a limit cycle of constant amplitude.
    """
    def settled(x, change):
        """``change <= tolerance * max(|x|, 1)``, with a finiteness guard.

        Without the guard a diverging iteration reports success: once ``x``
        overflows, ``tolerance * max(|x|, 1)`` is ``inf`` and ``change <= inf``
        passes.
        """
        if not (np.all(np.isfinite(x)) and np.isfinite(change)):
            return False
        return bool(change <= tolerance * max(np.abs(x).max(), 1.0))

    x = contact_map.initial_guess() if x0 is None else np.array(x0, dtype=float)
    change, evaluation = np.inf, None
    with np.errstate(over="ignore", invalid="ignore"):
        for iteration in range(1, max_iterations + 1):
            evaluation = contact_map(x, internal=internal, g_prev=g_prev, dt=dt)
            internal = evaluation.internal
            step = evaluation.value - x
            x = x + relaxation * step
            change = np.abs(relaxation * step).max()
            if settled(x, change):
                break
            if not np.all(np.isfinite(x)):  # diverged: no point continuing
                break
    converged = settled(x, change)
    return FixedPointResult(
        x=x,
        evaluation=evaluation,
        iterations=iteration,
        converged=converged,
        change=change,
    )


@dataclass
class CondensedContactMap:
    """``CD`` with the mechanics eliminated: ``g(x) = g_0 + Ghat x``.

    Same interface as :class:`ContactMap`, so :func:`fixed_point` accepts
    either, but every evaluation is a small dense matvec instead of a global
    solve.  An evaluation returns no solution vector; :meth:`recover` rebuilds
    it at the converged ``x`` with one back-substitution.
    """

    gap_offset: np.ndarray  # g_0, (n_points, dim)
    gap_matrix: np.ndarray  # Ghat, (n_points * dim, n_points * dim)
    augmentation: np.ndarray
    law: object
    shape: tuple
    prestress: np.ndarray | None = None
    #: retained pieces of the condensation, for :meth:`recover` and for
    #: rebuilding the affine offset under a new rhs (``ContactMap.condense``
    #: with ``reuse``) without refactorising
    factor: object = None  # the splu factor of the pinned matrix
    load: object = None  # sparse (N, n): moment values -> rhs contribution
    base: np.ndarray | None = None  # A0^{-1} b_0
    b0: np.ndarray | None = None  # the pinned rhs the base belongs to

    @property
    def n_points(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    def initial_guess(self) -> np.ndarray:
        return np.zeros(self.shape)

    def jacobi_augmentation(self, safety: float = 1.0) -> np.ndarray:
        """Per-point ``r`` read off the condensed compliance ``Ghat``.

        The augmentation must match the stiffness the fracture sees, which is
        ``Ghat`` itself, the whole domain's response rather than a local
        estimate.  ``r_p = safety / max_j |Ghat_(pj,pj)|`` is the Jacobi choice,
        making the diagonal of ``I + r Ghat`` vanish.

        A geometric estimate from the two cells adjacent to the facet assumes
        the fracture is loaded through its immediate neighbours; for a fault
        cutting the entire domain the compliance is that of the whole block and
        the estimate can be an order of magnitude too stiff, enough to make the
        iteration diverge.
        """
        n_points, dim = self.shape
        diagonal = np.abs(np.diag(self.gap_matrix)).reshape(n_points, dim)
        largest = diagonal.max(axis=1)
        return safety / np.where(largest > 0, largest, 1.0)

    def rescaled(self, safety: float = 1.0) -> "CondensedContactMap":
        """A copy whose augmentation comes from :meth:`jacobi_augmentation`."""
        return replace(self, augmentation=self.jacobi_augmentation(safety))

    def gap(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.shape)
        return self.gap_offset + (self.gap_matrix @ x.ravel()).reshape(self.shape)

    def recover(self, x) -> np.ndarray | None:
        """Full solution vector at multiplier ``x`` -- one back-substitution.

        ``z(x) = A_0^{-1}(b_0 + B W x) = base + A_0^{-1}(B W x)``.  Replaces
        the full pinned solve (a second factorisation) that recovering the
        fields otherwise costs.  ``None`` when the factor was not retained.
        """
        if self.factor is None:
            return None
        x = np.asarray(x, dtype=float).ravel()
        return self.base + self.factor.solve(np.asarray(self.load @ x).ravel())

    def __call__(self, x, internal=None, g_prev=None, dt=None) -> MapEvaluation:
        x = np.asarray(x, dtype=float).reshape(self.shape)
        gap = self.gap(x)
        offset = 0.0 if self.prestress is None else self.prestress
        trial = x + offset + self.augmentation[:, None] * driving_gap(gap, g_prev)
        if internal is None:
            internal = self.law.initial_state(self.n_points)
        if getattr(self.law, "wants_augmentation", False):
            self.law._augmentation = self.augmentation
        y, internal = self.law.project(trial, internal, gap, g_prev, dt)
        return MapEvaluation(
            value=np.asarray(y).reshape(self.shape) - offset,
            gap=gap,
            internal=internal,
            solution=None,
        )

    def residual(self, x, **kwargs) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.shape)
        return self(x, **kwargs).value - x


def projection_tangent(law, trial, internal=None, g=None, g_prev=None, dt=None,
                       step: float = 1e-7) -> np.ndarray:
    """``dP/dt`` at ``trial``: ``(n_points, dim, dim)`` blocks.

    Uses the law's analytic ``tangent`` when it has one, and central differences
    otherwise.  The projection acts pointwise, so its Jacobian is block diagonal
    and the finite-difference cost is ``2 * dim`` evaluations of the whole array
    -- negligible next to a single global solve.
    """
    trial = np.atleast_2d(np.asarray(trial, dtype=float))
    if hasattr(law, "tangent"):
        return law.tangent(trial)

    n, dim = trial.shape
    if internal is None:
        internal = law.initial_state(n)
    out = np.zeros((n, dim, dim))
    scale = max(np.abs(trial).max(), 1.0)
    for j in range(dim):
        shift = np.zeros(dim)
        shift[j] = step * scale
        plus, _ = law.project(trial + shift, internal, g, g_prev, dt)
        minus, _ = law.project(trial - shift, internal, g, g_prev, dt)
        out[:, :, j] = (np.asarray(plus) - np.asarray(minus)) / (2 * step * scale)
    return out


def projection_gap_tangent(law, trial, internal=None, g=None, g_prev=None,
                           dt=None, step: float = 1e-5) -> np.ndarray:
    """``dP/dg`` at fixed trial: ``(n_points, dim, dim)`` blocks.

    Zero for plain Coulomb, whose projection reads the jump only through the
    trial; nonzero for a law whose coefficients depend on the jump (slip
    weakening, rate and state).  There ``dP/dg . Ghat`` is the weakening
    feedback and belongs in the Newton Jacobian; dropping it degrades Newton to
    a Picard-like alternation that fails to converge near the nucleation fold
    while the equilibrium branch still exists.  Central differences per gap
    component, pointwise blocks.
    """
    g = np.atleast_2d(np.asarray(g, dtype=float))
    trial = np.atleast_2d(np.asarray(trial, dtype=float))
    n, dim = g.shape
    if internal is None:
        internal = law.initial_state(n)
    out = np.zeros((n, dim, dim))
    scale = max(np.abs(g).max(), 1e-6)
    for j in range(dim):
        shift = np.zeros(dim)
        shift[j] = step * scale
        plus, _ = law.project(trial, internal, g + shift, g_prev, dt)
        minus, _ = law.project(trial, internal, g - shift, g_prev, dt)
        out[:, :, j] = (np.asarray(plus) - np.asarray(minus)) / (2 * step * scale)
    return out


def newton(
    condensed,
    x0=None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
    internal=None,
    g_prev=None,
    dt=None,
    damping: float = 1.0,
) -> FixedPointResult:
    r"""Semismooth Newton on ``F(x) = CD(x) - x = 0``, for a condensed map.

    Picard converges only when ``CD`` is a contraction, which needs the
    augmentation to match the fracture compliance and that compliance to be
    close to diagonal.  Neither holds for a fault cutting the domain: ``Ghat``
    is dense, every facet feels every other, and no scalar ``r`` makes
    ``I + r Ghat`` a contraction -- rescaling ``r`` cannot fix a spectral-radius
    problem caused by off-diagonal coupling.

    With

        ``F(x) = P(x + r (g_0 + Ghat x)) - x`` ,
        ``J    = T (I + r Ghat) - I`` ,   ``T = dP/dt`` ,

    the step is a dense solve of size ``n_points * dim``, small because the map
    is condensed.  For an affine law (a frictionless fault) the residual is
    linear and this converges in a single iteration.

    Requires the condensed form: ``Ghat`` has to be available explicitly.
    """
    if not hasattr(condensed, "gap_matrix"):
        raise TypeError("newton needs a condensed map; call ContactMap.condense()")

    n_points, dim = condensed.shape
    size = n_points * dim
    augmented = np.repeat(condensed.augmentation, dim)[:, None] * condensed.gap_matrix
    trial_jacobian = np.eye(size) + augmented  # d(trial)/dx

    x = np.zeros(condensed.shape) if x0 is None else np.array(x0, dtype=float)
    change, evaluation = np.inf, None
    for iteration in range(1, max_iterations + 1):
        evaluation = condensed(x, internal=internal, g_prev=g_prev, dt=dt)
        internal = evaluation.internal
        residual = (evaluation.value - x).ravel()
        change = np.abs(residual).max()
        if change <= tolerance * max(np.abs(x).max(), 1.0):
            break

        offset = 0.0 if condensed.prestress is None else condensed.prestress
        trial = x + offset + condensed.augmentation[:, None] * evaluation.gap
        blocks = projection_tangent(
            condensed.law, trial, internal, evaluation.gap, g_prev, dt
        )
        jacobian = sp.block_diag(blocks, format="csr") @ trial_jacobian - np.eye(size)
        if getattr(condensed.law, "gap_dependent", False):
            # laws whose coefficients read the jump need the dP/dg . Ghat
            # chain-rule term, the weakening feedback
            gap_blocks = projection_gap_tangent(
                condensed.law, trial, internal, evaluation.gap, g_prev, dt
            )
            jacobian = jacobian + (
                sp.block_diag(gap_blocks, format="csr") @ condensed.gap_matrix
            )
        step = np.linalg.solve(np.asarray(jacobian), -residual)
        x = x + damping * step.reshape(condensed.shape)

    converged = bool(
        np.all(np.isfinite(x)) and change <= tolerance * max(np.abs(x).max(), 1.0)
    )
    return FixedPointResult(
        x=x, evaluation=evaluation, iterations=iteration,
        converged=converged, change=change,
    )
