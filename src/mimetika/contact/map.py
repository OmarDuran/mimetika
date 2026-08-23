r"""The contact problem as a nonlinear algebraic function ``y = CD(x)``.

Contact is a fixed-point problem in the contact traction, and nothing more::

    y = CD(x) ,    CD(x) = P( x + r g(x) ) ,    g(x) = J z ,    A(x) z = b(x)

where ``A(x)`` is the mechanics system with the fracture traction DOFs *pinned*
to ``x``.  One evaluation is: pin, solve, read the gap, project.  The solution of
``x = CD(x)`` is the converged contact state.

Why this is the right seam
--------------------------
Everything here is algebra: a matrix, a right-hand side, an index set, two linear
maps and a projection.  There is no mesh, no material, no boundary condition and
no problem object.  That matters three ways:

* **The mechanics is interchangeable.**  Whatever assembles ``(A, b)`` -- mixed
  elasticity, poromechanics, a pressure-driven right-hand side, per-cell
  materials -- is invisible here, so adding a boundary condition upstream needs
  no change at all in the contact code.
* **The iteration is interchangeable.**  ``CD`` is just a function, so the
  relaxed Picard iteration in :func:`fixed_point` can be swapped for Newton or
  Anderson acceleration without touching the map.
* **It is testable without a mesh.**  Feed ``CD`` any ``x`` and check ``y``:
  contraction, the fixed point, and the projection can each be checked on stub
  operators, separately from the discretisation.

What ``x`` contains
-------------------
``x`` and ``y`` are always the same object: the **contact traction at the
enforcement points, in the facet frame**, shape ``(n_points, dim)``.  The
components are ordered normal first::

    dim = 2:   x[p] = (t_n, t_t)            one shear direction
    dim = 3:   x[p] = (t_n, t_t1, t_t2)     two shear directions

with the sign convention ``t_n < 0`` in compression and ``g_n > 0`` open.  The
*space* is therefore identical for every law -- what changes is the subset of it
that ``CD`` can return, and what the law carries alongside:

=========================  ==================  ==========================================  =====================  =======
law                        ``x`` components    admissible set of ``y = CD(x)``             internal state         needs
                                                                                                                  ``dt``
=========================  ==================  ==========================================  =====================  =======
:class:`FrictionlessBilateral`  ``(t_n, t_t..)``  ``t_t = 0``; ``t_n`` free, either sign     none                   no
:class:`LinearContact`     ``(t_n, t_t..)``    all of ``R^dim`` (bonded, no constraint)    none                   no
:class:`SignoriniCoulomb`  ``(t_n, t_t..)``    ``t_n <= 0``, ``|t_t| <= -mu t_n + c``      slip magnitude (1)     no
:class:`RateAndStateFriction`  ``(t_n, t_t..)``  ``t_n <= 0``, ``|t_t| <= -mu(V, theta) t_n``  slip, ``theta`` (2)  yes
=========================  ==================  ==========================================  =====================  =======

Two consequences.  :class:`LinearContact` imposes no constraint at
all, so its projection is the identity and its fixed point is reached in one
evaluation -- which is why the driver short-circuits it through the compliance
block instead and it never reaches ``CD`` in practice.  And the sets of
:class:`FrictionlessBilateral` and :class:`SignoriniCoulomb` without cohesion
are **cones through the origin**, so their projections commute with positive
scaling; cohesion shifts the set off the origin and breaks that.

The conversion to the traction *moments* the linear system actually constrains
is the linear map ``to_moments``; the gap comes back through the linear map
``jump``.  Both are supplied as matrices by whoever knows the discretisation.

Prestress
---------
A contact law constrains the **total** traction: Signorini says the total normal
traction is compressive, not that some increment is.  When only an increment is
solved for -- a depletion response on top of an in-situ state -- the law must
still be shown the total, or a unilateral condition will read a tensile
*increment* on a firmly closed fault as opening.  ``prestress`` carries the
in-situ traction at the enforcement points: it is added before the projection
and removed after, so ``x`` stays the incremental unknown the mechanics
constrains while the law sees physical reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.solver.saddle import solve_saddle


def driving_gap(gap: np.ndarray, g_prev=None) -> np.ndarray:
    """What the augmentation multiplies: total normal gap, tangential *increment*.

    The two components are not treated alike.
    The normal condition ``g_n >= 0`` is a statement about the *absolute* gap, so
    the normal term is driven by the total jump.  Coulomb friction instead
    opposes the slip **rate**: eq. (2e) of Frigo et al. (2025) reads
    ``g_T . t_T = tau_max |g_T|`` with ``g_T`` a rate, which a quasi-static scheme
    discretises as the backward increment ``Delta_n g_T = g_T,n - g_T,n-1``.

    Driving the tangential part with the total jump instead is equivalent only
    while the loading is monotone and proportional -- the first step from rest,
    or any path along a fixed direction.  As soon as the slip direction rotates
    or reverses, the total jump still points along the accumulated path and the
    traction lags the direction it should oppose.
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
        The assembled mechanics system, **all** boundary conditions applied.
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
    #: ``(n_points, dim)``.  The gap is the *residual* of those rows,
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

        The nonlinear system is small: ``n_points * dim`` unknowns, a handful
        per fracture facet.  Two facts remove the need for a global solve per
        evaluation.

        The constrained matrix does **not** depend on ``x``.  Pinning zeroes the
        same rows and columns whatever the pinned values are; only the
        right-hand side moves.  And that dependence is *affine*::

            ``b(x) = b_0 + B W x`` ,   ``z(x) = A^{-1} b(x)`` ,
            ``g(x) = g_0 + Ghat x`` ,  ``Ghat = J A^{-1} B W`` .

        So one factorisation and ``n_points * dim + 1`` back-substitutions give a
        small dense ``Ghat``, after which ``CD`` is a matvec and a projection.
        The Uzawa iteration then touches the global system not at all.

        Worth it whenever the iteration count exceeds the contact DOF count,
        which is the usual case for friction; for a very large fracture and a
        near-linear law the uncondensed form can still win.

        ``reuse`` skips the factorisation and ``Ghat`` entirely: pass the
        condensed map of a *previous* system with the **same matrix** (a new
        load level, a new law parameter) and only the affine offset is redone
        -- one back-substitution instead of ``n + 1`` plus a factorisation.
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
        # contract J A^{-1} B W in column blocks: the full response matrix is
        # (N, n) dense -- gigabytes at scale -- but only its projection onto
        # the fault rows survives, so never materialise it
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

    While the fracture *sticks* the tangential update is a contraction and
    ``relaxation = 1`` converges; while it *slides* it is not, and the plain
    iteration settles into a limit cycle of constant amplitude rather than
    converging.  Damping restores convergence.

    Separate from :class:`ContactMap`: the map is the problem, and a Newton or
    Anderson variant would replace only this function.
    """
    def settled(x, change):
        """Converged means *small*, which a non-finite iterate never is.

        Without the finiteness guard a diverging iteration reports success: once
        ``x`` overflows, ``tolerance * max(|x|, 1)`` is ``inf`` and the test
        ``change <= inf`` passes.
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

    Same interface as :class:`ContactMap` -- :func:`fixed_point` cannot tell them
    apart -- but every evaluation is a small dense matvec instead of a global
    solve.  The solution vector is no longer available, which is the one thing
    given up: recover it with a single final :class:`ContactMap` evaluation at
    the converged ``x``.
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

        The augmentation must match the stiffness the fracture actually sees, and
        ``Ghat`` *is* that stiffness -- exactly, including the whole domain's
        response, not a local estimate.  Taking ``r_p = 1 / max_j |Ghat_(pj,pj)|``
        is the Jacobi choice, which makes the diagonal of ``I + r Ghat`` vanish.

        A geometric estimate based on the two cells adjacent to the facet
        assumes the fracture is loaded through its immediate neighbours; for a
        fault cutting the entire domain the compliance is that of the whole
        block, and the estimate can be an order of magnitude too stiff --
        enough to make the iteration diverge.
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

    Zero for plain Coulomb -- the projection reads the jump only through the
    trial -- but not for a law whose *coefficients* depend on the jump (slip
    weakening, rate and state).  There the term ``dP/dg . Ghat`` belongs in
    the Newton Jacobian: it is exactly the destabilising feedback of the
    weakening, and dropping it degrades Newton to a Picard-like alternation
    that spirals near the nucleation fold while the equilibrium branch still
    exists.  Central differences per gap component, pointwise blocks.
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

    Picard is only a good solver when ``CD`` is a contraction, which needs the
    augmentation to match the fracture compliance *and* that compliance to be
    close to diagonal.  Neither holds for a fault that cuts the domain: the
    condensed operator ``Ghat`` is dense, every facet feels every other, and no
    scalar ``r`` makes ``I + r Ghat`` a contraction.  Rescaling ``r`` cannot fix
    a spectral radius problem caused by off-diagonal coupling.

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
            # chain-rule term -- the weakening feedback itself
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
