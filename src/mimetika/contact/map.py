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

Two consequences worth stating.  :class:`LinearContact` imposes no constraint at
all, so its projection is the identity and its fixed point is reached in one
evaluation -- which is why the driver short-circuits it through the compliance
block instead and it never reaches ``CD`` in practice.  And the sets of
:class:`FrictionlessBilateral` and :class:`SignoriniCoulomb` without cohesion
are **cones through the origin**, so their projections commute with positive
scaling; cohesion shifts the set off the origin and breaks that.

The conversion to the traction *moments* the linear system actually constrains
is the linear map ``to_moments``; the gap comes back through the linear map
``jump``.  Both are supplied as matrices by whoever knows the discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from mimetika.solver.saddle import solve_saddle


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
        trial = x + self.augmentation[:, None] * gap
        if internal is None:
            internal = self.law.initial_state(self.n_points)
        y, internal = self.law.project(trial, internal, gap, g_prev, dt)
        return MapEvaluation(
            value=np.asarray(y).reshape(self.shape),
            gap=gap,
            internal=internal,
            solution=z,
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

    Under-relaxation is not cosmetic.  While the fracture *sticks* the tangential
    update is a contraction and ``relaxation = 1`` converges; while it *slides* it
    is not, and the plain iteration settles into a limit cycle of constant
    amplitude rather than converging.  Damping restores convergence.

    Deliberately separate from :class:`ContactMap`: the map is the problem, this
    is one way of solving it, and a Newton or Anderson variant would replace only
    this function.
    """
    def settled(x, change):
        """Converged means *small*, which a non-finite iterate never is.

        Without the finiteness guard a diverging iteration reports success: once
        ``x`` overflows, ``tolerance * max(|x|, 1)`` is ``inf`` and the test
        ``change <= inf`` passes.  Divergence would be indistinguishable from
        convergence in the returned flag.
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
