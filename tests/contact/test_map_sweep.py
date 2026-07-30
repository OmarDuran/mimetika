r"""Systematic parametric sweep of ``y = CD(x)``, for every contact law.

The point-checks in :mod:`test_map` pin specific values.  This file asks the
different question: **does ``CD`` behave for every ``x``?**  Each law is swept
over a grid spanning tension and compression, pure normal and pure shear, and
twelve orders of magnitude, and every sample is required to satisfy the
invariants that hold by construction *whatever* ``x`` is:

============================  ====================================================
invariant                     why it must hold for all ``x``
============================  ====================================================
finite                        a projection of a finite trial is finite; a ``nan``
                              here is silent corruption downstream
shape preserved               ``CD: R^(P x D) -> R^(P x D)``
deterministic                 the map is a function, not a procedure with memory
admissible                    ``y`` lies in the law's set -- that is what ``P`` is
idempotent (``P(P(t))=P(t)``) a projection applied twice is the projection; this
                              is the general form of "``y`` is in the set"
unilateral sign               ``y_n <= 0`` for laws with a Signorini condition
scale equivariance            for cone-type laws ``P(s t) = s P(t)``, ``s > 0``
============================  ====================================================

Because the map is pure algebra, the sweep needs no mesh: the stub system below
gives ``g(x) = (b - c x) / s`` elementwise, so the response is known in closed
form and any failure is the law's or the map's, never the discretisation's.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.contact.laws import (
    AssociativeMohrCoulomb,
    FrictionlessBilateral,
    LinearContact,
    RateAndStateFriction,
    SignoriniCoulomb,
)
from mimetika.contact.map import ContactMap, fixed_point

DIM = 3  # 3D: the friction set is a disk, the hardest case
POINTS = 4
COUPLING, STIFFNESS = 1.0, 2.0

# every law that can be driven through the map, by name
LAWS = {
    "frictionless-bilateral": FrictionlessBilateral(),
    "linear": LinearContact(normal_stiffness=2.0, shear_stiffness=3.0),
    "signorini-mu0.2": SignoriniCoulomb(friction=0.2),
    "signorini-mu0.6": SignoriniCoulomb(friction=0.6),
    "signorini-cohesion": SignoriniCoulomb(friction=0.4, cohesion=0.5),
    "rate-and-state": RateAndStateFriction(mu0=0.6, a=0.01, b=0.015, Dc=1e-4),
    # the associative return mapping: same admissible set, closest-point projection
    "associative-mc": AssociativeMohrCoulomb(friction=0.6, eps_n=1.0, eps_t=1.0),
    "associative-mc-anisotropic": AssociativeMohrCoulomb(
        friction=0.6, cohesion=0.3, eps_n=0.5, eps_t=2.0
    ),
}
UNILATERAL = ("signorini-mu0.2", "signorini-mu0.6", "signorini-cohesion",
              "rate-and-state", "associative-mc", "associative-mc-anisotropic")
#: laws whose admissible set is a cone through the origin, so ``P`` commutes
#: with positive scaling.  Cohesion shifts the set off the origin and breaks it.
CONE = ("frictionless-bilateral", "signorini-mu0.2", "signorini-mu0.6",
        "associative-mc")


def stub_map(law, r=0.4, load=3.0, points=POINTS, dim=DIM):
    """``CD`` on a diagonal stub: ``g(x) = (load - COUPLING x) / STIFFNESS``."""
    size = points * dim
    matrix = sp.bmat(
        [
            [sp.eye(size) * 4.0, sp.eye(size) * COUPLING],
            [sp.eye(size) * COUPLING, sp.eye(size) * STIFFNESS],
        ],
        format="csr",
    )
    rhs = np.concatenate([np.zeros(size), np.full(size, load)])
    return ContactMap(
        matrix=matrix,
        rhs=rhs,
        dofs=np.arange(size),
        to_moments=sp.eye(size, format="csr"),
        jump=sp.hstack(
            [sp.csr_matrix((size, size)), sp.eye(size, format="csr")], format="csr"
        ),
        augmentation=np.full(points, r),
        law=law,
        block_sizes=(size, size),
        # these tests exercise CD, not the linear solver, so the backend is
        # pinned: PETSc pays KSP setup and a MUMPS factorisation per solve, which
        # on a stub this small is pure overhead and makes the suite's runtime
        # depend on which environment it runs in
        solver={"method": "direct", "backend": "scipy"},
    )


def sweep_points(dim=DIM):
    """A grid of trial tractions spanning the interesting regimes and scales."""
    magnitudes = [0.0, 1e-8, 1e-3, 1.0, 1e3, 1e8]
    out = [np.zeros(dim)]
    for m in magnitudes:
        for sign in (-1.0, 1.0):
            out.append(np.array([sign * m] + [0.0] * (dim - 1)))  # pure normal
            shear = np.zeros(dim)
            shear[1] = sign * m
            out.append(shear)  # pure shear
            mixed = np.full(dim, sign * m)
            out.append(mixed)  # combined
            skewed = np.array([-m] + [sign * m * 10.0] * (dim - 1))
            out.append(skewed)  # shear far outside any cone
    return np.array(out)


SWEEP = sweep_points()
SWEEP_IDS = [f"x{i}" for i in range(len(SWEEP))]


def evaluate(law, x_row, **kwargs):
    """``CD`` applied with every enforcement point set to the same trial value."""
    cd = stub_map(law, **kwargs)
    x = np.tile(x_row, (cd.n_points, 1))
    return cd, cd(x, dt=1e-3)


# -- the sweep -----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_the_map_is_finite_and_shape_preserving(name, x_row):
    cd, evaluation = evaluate(LAWS[name], x_row)
    assert evaluation.value.shape == cd.shape
    assert evaluation.gap.shape == cd.shape
    assert np.all(np.isfinite(evaluation.value)), f"{name}: non-finite y"
    assert np.all(np.isfinite(evaluation.gap)), f"{name}: non-finite gap"


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_the_map_is_deterministic(name, x_row):
    """A function, not a procedure: the same input gives the same output."""
    _, first = evaluate(LAWS[name], x_row)
    _, second = evaluate(LAWS[name], x_row)
    assert np.array_equal(first.value, second.value)


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_the_output_is_admissible(name, x_row):
    """``P(P(t)) = P(t)``: applying the projection again must change nothing.

    The general statement that ``y`` lies in the admissible set, without this
    file needing to know the shape of each law's set.
    """
    law = LAWS[name]
    _, evaluation = evaluate(law, x_row)
    y = evaluation.value
    again, _ = law.project(y, law.initial_state(len(y)), evaluation.gap, None, 1e-3)
    assert np.allclose(again, y, rtol=1e-12, atol=1e-12), f"{name}: not idempotent"


@pytest.mark.parametrize("name", UNILATERAL)
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_unilateral_laws_never_return_tension(name, x_row):
    """``y_n <= 0`` at every enforcement point, for every trial state."""
    _, evaluation = evaluate(LAWS[name], x_row)
    assert np.all(evaluation.value[:, 0] <= 0.0), f"{name}: tension returned"


@pytest.mark.parametrize("name", UNILATERAL)
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_an_open_point_carries_no_shear(name, x_row):
    """Where the normal traction is zero the friction radius is too."""
    law = LAWS[name]
    _, evaluation = evaluate(law, x_row)
    y = evaluation.value
    open_points = np.abs(y[:, 0]) < 1e-14
    cohesion = getattr(law, "cohesion", 0.0)
    if cohesion == 0.0 and open_points.any():
        assert np.allclose(y[open_points, 1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("name", CONE)
@pytest.mark.parametrize("scale", [1e-6, 0.5, 3.0, 1e6])
@pytest.mark.parametrize("x_row", SWEEP[::5], ids=SWEEP_IDS[::5])
def test_cone_laws_are_positively_homogeneous(name, scale, x_row):
    """``P(s t) = s P(t)`` for ``s > 0`` when the admissible set is a cone.

    Checked on the projection directly: scaling the *trial* is what the identity
    is about, and running it through the solve as well would confound it with
    the response of the stub system.
    """
    law = LAWS[name]
    trial = np.tile(x_row, (POINTS, 1))
    state = law.initial_state(POINTS)
    base, _ = law.project(trial, state, None, None, 1e-3)
    scaled, _ = law.project(scale * trial, state, None, None, 1e-3)
    assert np.allclose(scaled, scale * np.asarray(base), rtol=1e-10, atol=1e-12)


# -- the fixed point, swept over the augmentation ---------------------------------------


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("r", [0.1, 0.4, 1.0, 1.9])
def test_the_fixed_point_is_reached_and_admissible(name, r):
    """Inside the contraction range ``0 < r < 2 s / c`` every law must converge."""
    law = LAWS[name]
    result = fixed_point(
        stub_map(law, r=r), relaxation=0.5, tolerance=1e-9, max_iterations=5000,
        dt=1e-3,
    )
    assert result.converged, f"{name} r={r}: did not converge"
    assert np.all(np.isfinite(result.x))
    again, _ = law.project(
        result.x, law.initial_state(POINTS), result.evaluation.gap, None, 1e-3
    )
    assert np.allclose(again, result.x, atol=1e-10)


@pytest.mark.parametrize("name", list(LAWS))
def test_the_fixed_point_does_not_depend_on_where_it_started(name):
    """A well-posed map: the same solution from tension, compression and zero."""
    law = LAWS[name]
    answers = []
    for start in (-1e4, 0.0, 1e4):
        result = fixed_point(
            stub_map(law),
            x0=np.full((POINTS, DIM), start),
            relaxation=0.5,
            tolerance=1e-9,
            max_iterations=5000,
            dt=1e-3,
        )
        assert result.converged, f"{name}: from {start}"
        answers.append(result.x)
    assert np.allclose(answers[0], answers[1], atol=1e-8)
    assert np.allclose(answers[1], answers[2], atol=1e-8)


@pytest.mark.parametrize("name", list(LAWS))
def test_divergence_is_reported_as_divergence(name):
    """An augmentation far outside the contraction range must not report success.

    Guards the overflow trap: once the iterate overflows, ``tolerance * |x|`` is
    ``inf`` and a naive ``change <= tol * |x|`` test passes, so a diverging run
    would be indistinguishable from a converged one.
    """
    # a *compressive* load, so the unilateral laws are actually engaged: under
    # tension they clamp to zero traction on the first evaluation, which is a
    # genuine fixed point (an open fault) and cannot diverge for any r
    result = fixed_point(
        stub_map(LAWS[name], r=50.0, load=-3.0),
        relaxation=1.0,
        max_iterations=400,
        dt=1e-3,
    )
    assert not result.converged


@pytest.mark.parametrize("name", UNILATERAL)
def test_a_tensile_load_opens_the_fault_regardless_of_augmentation(name):
    """The flip side: under tension the answer is ``t = 0`` for *any* ``r``.

    A unilateral law clamps the first trial to zero normal traction and stays
    there, so this is one of the few places where an arbitrarily large
    augmentation is harmless.  A **cohesive** fault still carries shear up to
    ``c`` once open -- that is what cohesion means -- so only the normal
    component is required to vanish.
    """
    law = LAWS[name]
    cohesion = getattr(law, "cohesion", 0.0)
    for r in (0.4, 50.0):
        result = fixed_point(stub_map(law, r=r, load=3.0), relaxation=1.0, dt=1e-3)
        assert result.converged
        assert np.allclose(result.x[:, 0], 0.0, atol=1e-12)
        assert np.all(
            np.linalg.norm(result.x[:, 1:], axis=1) <= cohesion + 1e-12
        )
        assert np.all(result.evaluation.gap > 0.0)  # genuinely open


# -- the sweep grid itself ---------------------------------------------------------------


def test_the_sweep_actually_covers_the_regimes():
    """Guard the guard: an all-zero or one-sided grid would pass everything."""
    assert len(SWEEP) > 30
    assert (SWEEP[:, 0] < 0).any() and (SWEEP[:, 0] > 0).any()  # both signs
    assert (np.abs(SWEEP) > 1e6).any() and (np.abs(SWEEP[SWEEP != 0]) < 1e-6).any()
    shear_only = (SWEEP[:, 0] == 0) & (np.abs(SWEEP[:, 1:]) > 0).any(axis=1)
    assert shear_only.any()


@pytest.mark.parametrize("name", UNILATERAL)
def test_the_sweep_would_catch_a_broken_projection(name):
    """The invariants have teeth: a law that skips clipping must fail them."""
    law = LAWS[name]
    tensile = np.tile(np.array([5.0] + [0.0] * (DIM - 1)), (POINTS, 1))
    projected, _ = law.project(tensile, law.initial_state(POINTS), None, None, 1e-3)
    assert np.all(np.asarray(projected)[:, 0] <= 0.0)
    assert not np.allclose(projected, tensile)  # it really did something


# -- condensation: the same map, with the mechanics eliminated --------------------------
#
# The nonlinear system is small -- ``n_points * dim`` unknowns -- so paying a
# global linear solve per residual evaluation is backwards.  The constrained
# matrix does not depend on ``x`` and the right-hand side depends on it affinely,
# so the gap condenses to ``g(x) = g_0 + Ghat x`` and ``CD`` becomes a dense
# matvec.  These tests require that this changes the cost and *nothing else*.


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("x_row", SWEEP, ids=SWEEP_IDS)
def test_condensation_reproduces_the_full_map(name, x_row):
    """Not merely close: the condensed form is the same algebra, so it is exact."""
    law = LAWS[name]
    cd = stub_map(law)
    condensed = cd.condense()
    x = np.tile(x_row, (POINTS, 1))
    full, small = cd(x, dt=1e-3), condensed(x, dt=1e-3)
    assert np.allclose(small.value, full.value, rtol=1e-12, atol=1e-12)
    assert np.allclose(small.gap, full.gap, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", list(LAWS))
def test_condensed_and_full_reach_the_same_fixed_point(name):
    law = LAWS[name]
    cd = stub_map(law, load=-3.0)
    settings = dict(relaxation=0.5, tolerance=1e-12, max_iterations=5000, dt=1e-3)
    full = fixed_point(cd, **settings)
    small = fixed_point(cd.condense(), **settings)
    assert full.converged and small.converged
    assert np.allclose(full.x, small.x, atol=1e-10)
    assert full.iterations == small.iterations


def test_the_condensed_gap_is_affine():
    """``g(x) = g_0 + Ghat x`` -- superposition must hold exactly."""
    condensed = stub_map(LAWS["signorini-mu0.6"]).condense()
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((POINTS, DIM)), rng.standard_normal((POINTS, DIM))
    lhs = condensed.gap(a + b) - condensed.gap(np.zeros((POINTS, DIM)))
    rhs = (condensed.gap(a) - condensed.gap_offset) + (
        condensed.gap(b) - condensed.gap_offset
    )
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_the_condensed_system_is_the_size_of_the_contact_problem():
    """``Ghat`` is ``n_points * dim`` square -- not the size of the mechanics."""
    cd = stub_map(LAWS["signorini-mu0.6"])
    condensed = cd.condense()
    n = POINTS * DIM
    assert condensed.gap_matrix.shape == (n, n)
    assert condensed.shape == cd.shape
    assert cd.matrix.shape[0] > n  # the mechanics really was larger


def test_condensation_drops_the_solution_vector():
    """The one thing given up, stated rather than discovered later."""
    condensed = stub_map(LAWS["signorini-mu0.6"]).condense()
    assert condensed(np.zeros((POINTS, DIM))).solution is None


# -- the driving gap: total normal, incremental tangential ------------------------------


def test_the_normal_term_uses_the_total_gap_and_the_shear_the_increment():
    """Frigo et al. (2025) eqs (10a)/(10b): the two components differ on purpose.

    ``g_N >= 0`` constrains the *absolute* gap, while Coulomb friction opposes
    the slip **rate**, discretised as the backward increment.  Driving the shear
    with the total jump is equivalent only under monotone proportional loading.
    """
    from mimetika.contact.map import driving_gap

    gap = np.array([[-2.0, 5.0, 1.0]])
    previous = np.array([[99.0, 4.0, 0.5]])
    driven = driving_gap(gap, previous)
    assert driven[0, 0] == -2.0  # normal: total, previous ignored
    assert np.allclose(driven[0, 1:], [1.0, 0.5])  # shear: the increment
    assert np.array_equal(driving_gap(gap, None), gap)  # first step: they coincide


def test_the_shear_traction_follows_the_slip_increment_not_the_total():
    """Under a rotating load the traction must oppose the *current* increment."""
    from mimetika.contact.map import driving_gap

    total = np.array([[-1.0, -0.05, 0.05]])  # accumulated path points up-left
    previous = np.array([[0.0, 0.05, 0.05]])  # previous step was up-right
    driven = driving_gap(total, previous)
    increment = driven[0, 1:] / np.linalg.norm(driven[0, 1:])
    assert np.allclose(increment, [-1.0, 0.0], atol=1e-12)
    # the total jump points somewhere quite different
    naive = total[0, 1:] / np.linalg.norm(total[0, 1:])
    assert not np.allclose(naive, increment, atol=1e-2)
