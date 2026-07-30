r"""Contact laws on a fracture: the constitutive part, free of any DOF.

Every law relates the **traction** on the fracture to the **displacement jump**.
Both are presented in the facet frame ``(n, t1, t2)`` with one fixed convention:

    ``g_n > 0``  the fracture is **open** (gap)
    ``t_n < 0``  the fracture is in **compression**

so Signorini reads ``g_n >= 0``, ``t_n <= 0``, ``g_n t_n = 0``.  A law never
sees a degree of freedom, a mesh, or a basis: the driver owns the rotation into
this frame, the moment/point conversion, assembly and the solve.

Taxonomy
--------
The general contract is an **implicit** relation ``C(t, g, state) = 0``, because
that is the only form that covers unilateral contact -- a compliance ``g = A t``
cannot express ``t_n <= 0``.  Laws differ along axes that each force something
on the driver:

======================  ==========================  ============================
axis                    values                      consequence for the driver
======================  ==========================  ============================
relation form           compliance / implicit       compliance => one linear solve
smoothness              smooth / nonsmooth          nonsmooth => outer iteration
tangent symmetry        symmetric / not             friction is **not** symmetric
path dependence         none / incremental / rate   load steps / time steps
internal state          none / slip / slip + theta  state array, committed per step
enforcement             averaged / pointwise        where the projection is applied
======================  ==========================  ============================

============================  ===========  =========  ==========  =========  ============
model                         form         smooth     symmetric   path       state
============================  ===========  =========  ==========  =========  ============
:class:`LinearContact`        compliance   yes        yes         none       none
:class:`SignoriniCoulomb`     implicit     no         no          increment  slip
:class:`RateAndStateFriction` implicit     stiff      no          rate       slip, theta
============================  ===========  =========  ==========  =========  ============

Solution strategy
-----------------
The driver uses an **augmented Lagrangian** (Uzawa) outer iteration in which the
multiplier ``lambda`` *is* the physical contact traction: the mechanics is solved
with the fracture traction constrained to ``lambda``, the gap ``g`` is recovered,
and then

    ``lambda <- project(lambda + r g)`` .

So the only thing a nonsmooth law has to supply is its projection onto the
admissible set -- that is what :meth:`ContactLaw.project` is.  The augmentation
``r`` is not free: the iteration contracts only for ``r < 2 / compliance``, so
the driver derives it from the stiffness the fracture actually sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContactLaw(ABC):
    """Base class: relates facet traction to displacement jump."""

    #: internal variables carried per enforcement point
    n_state: int = 0
    #: needs the jump of the previous step (slip history)
    path_dependent: bool = False
    #: needs a time increment (slip *rate*)
    rate_dependent: bool = False
    #: whether the exact tangent is symmetric (friction is not)
    symmetric_tangent: bool = True

    def initial_state(self, n_points: int) -> np.ndarray:
        return np.zeros((n_points, self.n_state))

    def linear_compliance(self, dim: int = 3) -> np.ndarray | None:
        """``A_f`` in the facet frame when the law is exactly linear, else ``None``.

        Shape ``(dim, dim)``: the components are ``(n, t_1, ..., t_{dim-1})``, so
        a 2D fracture has one shear direction and a 3D one has two.  A law that
        returns a matrix here is solved in **one** linear solve, with no outer
        iteration and no projection.
        """
        return None

    @abstractmethod
    def project(
        self,
        trial: np.ndarray,
        state: np.ndarray,
        g: np.ndarray | None = None,
        g_prev: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a trial traction onto the admissible set.

        Parameters
        ----------
        trial
            ``(n, 3)`` trial traction ``lambda + r g`` in the facet frame.
        state
            ``(n, n_state)`` internal variables.
        g, g_prev, dt
            Jump, previous jump and time step -- supplied for path- and
            rate-dependent laws.

        Returns
        -------
        ``(traction, state)``, both ``(n, 3)`` / ``(n, n_state)``.
        """

    def advance(self, traction, g, state, dt=None, g_prev=None) -> np.ndarray:
        """Commit internal variables at the end of a converged step."""
        return state


class LinearContact(ContactLaw):
    """``A_f sigma n = [[u]]`` -- linear springs, always bonded.

    Allows tension and interpenetration: there is no unilateral condition.  The
    law is exactly representable, so the driver solves it in a single linear
    solve rather than iterating.
    """

    symmetric_tangent = True

    def __init__(self, normal_stiffness: float = 1.0, shear_stiffness: float = 1.0):
        self.normal_stiffness = float(normal_stiffness)
        self.shear_stiffness = float(shear_stiffness)

    def linear_compliance(self, dim: int = 3) -> np.ndarray:
        return np.diag(
            [1.0 / self.normal_stiffness]
            + [1.0 / self.shear_stiffness] * (dim - 1)
        )

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        return trial, state


class FrictionlessBilateral(ContactLaw):
    """A closed, frictionless fault: ``t_t = 0`` and ``g_n = 0``.

    Bilateral in the normal direction -- the fault is held shut and may carry
    tension -- and free to slide tangentially.  The projection keeps the normal
    traction and zeroes the shear, so the converged state has no opening and no
    shear stress, which is exactly the classical frictionless crack.

    Why not ``SignoriniCoulomb(friction=0)``
    ---------------------------------------
    That law also clips the normal traction to compression, which is right for a
    total-stress problem and **wrong for an incremental one**.  A fault sitting
    under tens of MPa of in-situ compression stays firmly closed, so an
    incremental solve -- where only the depletion response is computed -- must
    not read an incremental normal tension as opening.  Using Signorini there
    would open the fault spuriously wherever the increment happens to be
    tensile.  The choice between the two is a modelling decision about *what the
    unknown is*, not about the physics of the fault.
    """

    n_state = 0
    symmetric_tangent = True

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        trial = np.atleast_2d(np.asarray(trial, dtype=float))
        t = np.zeros_like(trial)
        t[:, 0] = trial[:, 0]  # normal traction is whatever holds the fault shut
        return t, state


class SignoriniCoulomb(ContactLaw):
    """Unilateral contact with Coulomb friction (Alart--Curnier projection).

    Normal:      ``g_n >= 0``, ``t_n <= 0``, ``g_n t_n = 0``  (no interpenetration,
                 no tension -- the fracture may open and lose contact).
    Tangential:  ``|t_t| <= -mu t_n``; sticking inside the cone, sliding on it.

    The projection is the Alart--Curnier one: clip the normal traction to the
    compressive half-line, then project the tangential traction onto the
    friction disk whose radius follows from the *projected* normal traction --
    so an open point carries no shear, automatically.

    State is the accumulated tangential slip, which the law itself does not use
    but which makes the slip path available to callers and to rate-dependent
    laws derived from this one.
    """

    n_state = 1  # accumulated slip magnitude
    path_dependent = True
    symmetric_tangent = False  # Coulomb friction is non-associated

    def __init__(self, friction: float = 0.6, cohesion: float = 0.0):
        self.friction = float(friction)
        self.cohesion = float(cohesion)

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        trial = np.atleast_2d(np.asarray(trial, dtype=float))
        t = np.empty_like(trial)

        # normal: onto the compressive half-line
        t[:, 0] = np.minimum(trial[:, 0], 0.0)

        # tangential: onto the friction disk of radius -mu t_n + c
        radius = np.maximum(-self.friction * t[:, 0] + self.cohesion, 0.0)
        shear = trial[:, 1:]
        mag = np.linalg.norm(shear, axis=1)
        scale = np.where(mag > radius, radius / np.maximum(mag, 1e-300), 1.0)
        t[:, 1:] = shear * scale[:, None]
        return t, state

    def advance(self, traction, g, state, dt=None, g_prev=None):
        state = np.array(np.atleast_2d(state), dtype=float, copy=True)
        if g is not None:
            ref = np.zeros_like(np.atleast_2d(g)) if g_prev is None else np.atleast_2d(g_prev)
            state[:, 0] += np.linalg.norm(np.atleast_2d(g)[:, 1:] - ref[:, 1:], axis=1)
        return state

    # -- diagnostics ---------------------------------------------------------

    def status(self, traction, tol: float = 1e-10) -> np.ndarray:
        """``0`` open, ``1`` stick, ``2`` slip -- one label per enforcement point."""
        t = np.atleast_2d(np.asarray(traction, dtype=float))
        radius = np.maximum(-self.friction * t[:, 0] + self.cohesion, 0.0)
        mag = np.linalg.norm(t[:, 1:], axis=1)
        out = np.ones(len(t), dtype=int)
        out[t[:, 0] > -tol] = 0
        out[(t[:, 0] <= -tol) & (mag >= radius - tol)] = 2
        return out


class RateAndStateFriction(SignoriniCoulomb):
    """Rate- and state-dependent friction (regularised, aging law).

    The friction coefficient is no longer constant:

        ``mu(V, theta) = mu0 + a ln(V/V0) + b ln(V0 theta / Dc)``

    with slip rate ``V = |g_t - g_t_prev| / dt`` and state ``theta`` evolving by
    the aging law ``dtheta/dt = 1 - V theta / Dc``, integrated implicitly so the
    update is unconditionally stable:

        ``theta_new = (theta + dt) / (1 + dt V / Dc)`` .

    Everything else -- the unilateral normal condition and the projection onto
    the friction disk -- is inherited, which is the point of the taxonomy: only
    the *radius* of the disk changes.
    """

    n_state = 2  # accumulated slip, state variable theta
    rate_dependent = True

    def __init__(
        self,
        mu0: float = 0.6,
        a: float = 0.010,
        b: float = 0.015,
        Dc: float = 1e-4,
        V0: float = 1e-6,
        theta0: float = 1.0,
        Vmin: float = 1e-16,
    ):
        super().__init__(friction=mu0)
        self.mu0, self.a, self.b = float(mu0), float(a), float(b)
        self.Dc, self.V0 = float(Dc), float(V0)
        self.theta0, self.Vmin = float(theta0), float(Vmin)

    def initial_state(self, n_points: int) -> np.ndarray:
        state = np.zeros((n_points, self.n_state))
        state[:, 1] = self.theta0
        return state

    def slip_rate(self, g, g_prev, dt) -> np.ndarray:
        if g is None or g_prev is None or not dt:
            return np.full(len(np.atleast_2d(g if g is not None else [[0, 0, 0]])), self.Vmin)
        dg = np.atleast_2d(g)[:, 1:] - np.atleast_2d(g_prev)[:, 1:]
        return np.maximum(np.linalg.norm(dg, axis=1) / dt, self.Vmin)

    def friction_coefficient(self, V, theta) -> np.ndarray:
        theta = np.maximum(theta, 1e-300)
        return np.maximum(
            self.mu0
            + self.a * np.log(V / self.V0)
            + self.b * np.log(self.V0 * theta / self.Dc),
            0.0,
        )

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        trial = np.atleast_2d(np.asarray(trial, dtype=float))
        state = np.atleast_2d(np.asarray(state, dtype=float))
        V = self.slip_rate(g, g_prev, dt)
        mu = self.friction_coefficient(V, state[:, 1])

        t = np.empty_like(trial)
        t[:, 0] = np.minimum(trial[:, 0], 0.0)
        radius = np.maximum(-mu * t[:, 0] + self.cohesion, 0.0)
        shear = trial[:, 1:]
        mag = np.linalg.norm(shear, axis=1)
        scale = np.where(mag > radius, radius / np.maximum(mag, 1e-300), 1.0)
        t[:, 1:] = shear * scale[:, None]
        return t, state

    def advance(self, traction, g, state, dt=None, g_prev=None):
        state = np.array(np.atleast_2d(state), dtype=float, copy=True)
        V = self.slip_rate(g, g_prev, dt)
        if g is not None and g_prev is not None:
            state[:, 0] += np.linalg.norm(
                np.atleast_2d(g)[:, 1:] - np.atleast_2d(g_prev)[:, 1:], axis=1
            )
        if dt:  # implicit aging law -- unconditionally stable
            state[:, 1] = (state[:, 1] + dt) / (1.0 + dt * V / self.Dc)
        return state
