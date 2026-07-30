r"""``y = CD(x)`` as a nonlinear algebraic function, tested without a mesh.

That the contact problem is *only* algebra is the claim the design makes, so it
is the claim these tests check: every :class:`ContactMap` below is built from
hand-written 2x2 matrices.  No mesh, no material, no boundary condition, no
mixed-elasticity problem appears anywhere in this file.

The stub system is small enough to have a closed form.  With one enforcement
point, ``to_moments = [1]`` and the system

    ``[[a, c], [c, d]] z = [b0, b1]`` ,   ``dofs = [0]`` ,   ``J = [0, 1]``

pinning ``z_0 = x`` leaves ``z_1 = (b1 - c x) / d``, so

    ``g(x) = (b1 - c x) / d``   and   ``CD(x) = P( x + r g(x) )`` .

With the identity projection the fixed point is ``x* = b1 / c``, where
``g(x*) = 0`` -- the bilateral contact condition -- and every statement below
can be checked against those two formulas rather than against a previous run.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.contact.laws import ContactLaw
from mimetika.contact.map import ContactMap, fixed_point

# The coupling sign is not free.  ``g(x) = (b1 - c x)/d`` must *decrease* as the
# traction grows -- pushing harder closes the gap -- so ``c/d > 0``.  With the
# opposite sign the map has multiplier ``1 + r|c|/d > 1`` and no ``r`` converges,
# which is a statement about contact being unstable, not about the solver.
A_11, C_12, D_22 = 4.0, 1.0, 2.0
B_0, B_1 = 0.0, 3.0


class Identity(ContactLaw):
    """No admissible set: the projection is the identity."""

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        return np.atleast_2d(trial), state


class ClampNormal(ContactLaw):
    """Signorini's normal condition alone: ``t_n <= 0``."""

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        trial = np.atleast_2d(np.asarray(trial, dtype=float))
        return np.minimum(trial, 0.0), state


def stub_map(law=None, r=0.5):
    matrix = sp.csr_matrix(np.array([[A_11, C_12], [C_12, D_22]]))
    return ContactMap(
        matrix=matrix,
        rhs=np.array([B_0, B_1]),
        dofs=np.array([0]),
        to_moments=sp.csr_matrix(np.array([[1.0]])),
        jump=sp.csr_matrix(np.array([[0.0, 1.0]])),
        augmentation=np.array([r]),
        law=Identity() if law is None else law,
        block_sizes=(1, 1),
    )


def exact_gap(x):
    return (B_1 - C_12 * x) / D_22


EXACT_FIXED_POINT = B_1 / C_12


# -- the map is a function -------------------------------------------------------


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.0, 7.5])
def test_the_gap_is_the_closed_form(x):
    """``g(x)`` must be the exact response of the pinned system."""
    evaluation = stub_map()(np.array([[x]]))
    assert evaluation.gap.item() == pytest.approx(exact_gap(x), rel=1e-10)


@pytest.mark.parametrize("x", [-2.0, 0.0, 1.0, 7.5])
@pytest.mark.parametrize("r", [0.1, 0.5, 2.0])
def test_the_map_is_the_closed_form(x, r):
    """``CD(x) = x + r g(x)`` under the identity projection."""
    value = stub_map(r=r)(np.array([[x]])).value.item()
    assert value == pytest.approx(x + r * exact_gap(x), rel=1e-10)


def test_the_pinned_unknown_really_is_pinned():
    """``z_0 = x`` exactly -- the constraint, not a penalty."""
    for x in (-3.0, 0.0, 2.5):
        evaluation = stub_map()(np.array([[x]]))
        assert evaluation.solution[0] == pytest.approx(x, rel=1e-12)


def test_the_map_is_affine_under_an_affine_projection():
    """``CD`` inherits the linearity of the system: superposition must hold."""
    cd = stub_map()
    at = lambda x: cd(np.array([[x]])).value.item()  # noqa: E731
    assert at(2.0) - at(1.0) == pytest.approx(at(3.0) - at(2.0), rel=1e-10)


# -- the fixed point -------------------------------------------------------------


def test_the_residual_vanishes_only_at_the_fixed_point():
    cd = stub_map()
    assert cd.residual(np.array([[EXACT_FIXED_POINT]])).item() == pytest.approx(
        0.0, abs=1e-12
    )
    assert abs(cd.residual(np.array([[EXACT_FIXED_POINT + 1.0]])).item()) > 1e-3


def test_the_gap_closes_at_the_fixed_point():
    """The physical content: ``g(x*) = 0`` is bilateral contact."""
    assert exact_gap(EXACT_FIXED_POINT) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("r", [0.2, 0.5, 1.0])
def test_fixed_point_finds_it(r):
    # multiplier 1 - r c / d is 0.9 at r = 0.2, so reaching 1e-14 takes ~700
    # iterations: a slow contraction is still a contraction
    result = fixed_point(
        stub_map(r=r), relaxation=1.0, tolerance=1e-14, max_iterations=5000
    )
    assert result.converged
    assert result.x.item() == pytest.approx(EXACT_FIXED_POINT, rel=1e-8)


def test_fixed_point_starts_from_the_supplied_guess():
    """Seeding at the answer must converge immediately, not re-derive it."""
    result = fixed_point(
        stub_map(), x0=np.array([[EXACT_FIXED_POINT]]), tolerance=1e-14
    )
    assert result.iterations == 1
    assert result.x.item() == pytest.approx(EXACT_FIXED_POINT, rel=1e-10)


def test_the_contraction_condition_is_the_expected_one():
    """``CD`` has multiplier ``1 - r c / d``: contracts exactly for ``0 < r < 2d/c``."""
    for r, contracts in ((0.5, True), (3.9, True), (4.1, False), (8.0, False)):
        multiplier = abs(1.0 - r * C_12 / D_22)
        assert (multiplier < 1.0) is contracts
        result = fixed_point(stub_map(r=r), relaxation=1.0, max_iterations=5000)
        assert result.converged is contracts


def test_too_large_an_augmentation_diverges():
    """The contraction condition is real: ``r c / d > 2`` oscillates outwards.

    Documents *why* the augmentation is derived from stiffness rather than
    guessed -- the same failure the mesh-based driver shows.
    """
    result = fixed_point(
        stub_map(r=8.0), relaxation=1.0, max_iterations=60, tolerance=1e-14
    )
    assert not result.converged


def test_relaxation_rescues_a_divergent_augmentation():
    """Damping widens the range of ``r`` that converges -- the reason it exists."""
    plain = fixed_point(stub_map(r=5.0), relaxation=1.0, max_iterations=200)
    damped = fixed_point(stub_map(r=5.0), relaxation=0.2, max_iterations=200)
    assert not plain.converged
    assert damped.converged
    assert damped.x.item() == pytest.approx(EXACT_FIXED_POINT, rel=1e-6)


def test_iteration_count_is_reported_honestly():
    result = fixed_point(stub_map(), relaxation=0.05, max_iterations=5)
    assert not result.converged
    assert result.iterations == 5


# -- the law is what shapes the answer ---------------------------------------------


def test_the_projection_is_applied():
    """A clamping law must change the map wherever the identity would go positive."""
    x = np.array([[1.0]])
    free = stub_map(law=Identity())(x).value.item()
    clamped = stub_map(law=ClampNormal())(x).value.item()
    assert free > 0.0
    assert clamped == 0.0


def test_a_clamping_law_moves_the_fixed_point_off_the_bilateral_one():
    """With ``t <= 0`` enforced, the gap no longer closes -- it opens instead."""
    result = fixed_point(stub_map(law=ClampNormal()), tolerance=1e-14)
    assert result.converged
    assert result.x.item() <= 0.0
    assert result.x.item() != pytest.approx(EXACT_FIXED_POINT, rel=1e-3)
    assert exact_gap(result.x.item()) > 0.0  # open


# -- shape and plumbing --------------------------------------------------------------


def test_shape_is_inferred_from_the_operators():
    cd = stub_map()
    assert cd.n_points == 1
    assert cd.dim == 1
    assert cd.shape == (1, 1)
    assert cd.initial_guess().shape == (1, 1)


def test_a_flat_input_is_accepted():
    """``CD`` is a function on vectors; the ``(n_points, dim)`` shape is a view."""
    cd = stub_map()
    assert cd(np.array([1.0])).value.item() == pytest.approx(
        cd(np.array([[1.0]])).value.item()
    )


def test_multiple_points_and_components():
    """Two points in 2D: the operators, not the map, carry the dimensions."""
    n, dim = 2, 2
    size = n * dim
    matrix = sp.csr_matrix(np.eye(2 * size) + 0.1)
    cd = ContactMap(
        matrix=matrix,
        rhs=np.ones(2 * size),
        dofs=np.arange(size),
        to_moments=sp.eye(size, format="csr"),
        jump=sp.hstack(
            [sp.csr_matrix((size, size)), sp.eye(size, format="csr")], format="csr"
        ),
        augmentation=np.full(n, 0.3),
        law=Identity(),
        block_sizes=(size, size),
    )
    assert cd.shape == (n, dim)
    evaluation = cd(np.zeros((n, dim)))
    assert evaluation.value.shape == (n, dim)
    assert evaluation.gap.shape == (n, dim)
