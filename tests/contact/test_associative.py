r"""The associative Mohr--Coulomb return mapping (closest-point projection).

Same admissible set as :class:`SignoriniCoulomb`, different return mapping, and
therefore different physics.  Three things are checked, each against something
independent of the implementation:

* the projection really is the **closest point** -- compared against a
  constrained numerical minimisation, not against a stored answer;
* the **consistent tangent** really is the derivative -- compared against
  central finite differences;
* it really is **different** from the partial return -- so that the two are not
  quietly the same code path under another name.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from mimetika.contact.laws import AssociativeMohrCoulomb, SignoriniCoulomb

FRICTION = 0.6
METRICS = [(1.0, 1.0, 0.0), (1.0, 2.5, 0.4), (3.0, 0.5, 1.2)]


def law(eps_n=1.0, eps_t=1.0, cohesion=0.0):
    return AssociativeMohrCoulomb(
        friction=FRICTION, cohesion=cohesion, eps_n=eps_n, eps_t=eps_t
    )


def closest_point(trial, eps_n, eps_t, cohesion):
    """The projection by constrained minimisation -- the definition, not the code."""

    def distance(x):
        return 0.5 * (
            (x[0] - trial[0]) ** 2 / eps_n
            + np.sum((x[1:] - trial[1:]) ** 2) / eps_t
        )

    constraints = [
        {"type": "ineq", "fun": lambda x: -x[0]},
        {
            "type": "ineq",
            "fun": lambda x: cohesion - FRICTION * x[0] - np.linalg.norm(x[1:]),
        },
    ]
    best, best_value = None, np.inf
    for start in (
        np.array([min(trial[0], 0.0), 0.0, 0.0]),
        np.zeros(3),
        np.array([min(trial[0], 0.0) - 1.0, 0.0, 0.0]),
    ):
        result = minimize(
            distance, start, constraints=constraints, method="SLSQP",
            options={"maxiter": 800, "ftol": 1e-16},
        )
        feasible = -result.x[0] > -1e-7 and (
            cohesion - FRICTION * result.x[0] - np.linalg.norm(result.x[1:]) > -1e-7
        )
        if feasible and result.fun < best_value:
            best, best_value = result.x, result.fun
    return best


# -- it is the closest point ---------------------------------------------------------


@pytest.mark.parametrize("eps_n,eps_t,cohesion", METRICS)
def test_the_projection_is_the_metric_closest_point(eps_n, eps_t, cohesion):
    """Against a numerical optimiser: the tolerance is SLSQP's, not ours."""
    model = law(eps_n, eps_t, cohesion)
    rng = np.random.default_rng(1)
    worst, checked = 0.0, 0
    for _ in range(120):
        trial = rng.normal(scale=4.0, size=3)
        reference = closest_point(trial, eps_n, eps_t, cohesion)
        if reference is None:
            continue
        checked += 1
        got, _ = model.project(trial[None, :], None)
        worst = max(worst, np.abs(got[0] - reference).max())
    assert checked > 100
    assert worst < 1e-5, f"max deviation from the optimum {worst:.2e}"


@pytest.mark.parametrize("eps_n,eps_t,cohesion", METRICS)
def test_the_result_is_always_admissible(eps_n, eps_t, cohesion):
    model = law(eps_n, eps_t, cohesion)
    rng = np.random.default_rng(2)
    trials = rng.normal(scale=5.0, size=(2000, 3))
    got, _ = model.project(trials, None)
    assert np.all(got[:, 0] <= 1e-12)
    radius = cohesion - FRICTION * got[:, 0]
    assert np.all(np.linalg.norm(got[:, 1:], axis=1) <= radius + 1e-10)


def test_an_admissible_trial_is_left_alone():
    """Inside the cone the projection must be the identity, not a near-identity."""
    model = law()
    inside = np.array([[-4.0, 1.0, 0.5], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got, _ = model.project(inside, None)
    assert np.allclose(got, inside, atol=1e-14)


def test_the_projection_is_idempotent():
    rng = np.random.default_rng(3)
    model = law(1.0, 2.5, 0.4)
    once, _ = model.project(rng.normal(scale=4.0, size=(500, 3)), None)
    twice, _ = model.project(once, None)
    assert np.allclose(twice, once, atol=1e-12)


def test_the_shear_stays_collinear_with_the_trial():
    """Rotational symmetry about the normal axis: the projection cannot rotate shear."""
    rng = np.random.default_rng(4)
    model = law(1.0, 2.5, 0.4)
    for _ in range(200):
        trial = rng.normal(scale=4.0, size=3)
        got, _ = model.project(trial[None, :], None)
        a, b = trial[1:], got[0, 1:]
        if np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9:
            cross = a[0] * b[1] - a[1] * b[0]
            assert abs(cross) < 1e-10 * np.linalg.norm(a) * np.linalg.norm(b)


# -- the consistent tangent ------------------------------------------------------------


@pytest.mark.parametrize("eps_n,eps_t,cohesion", METRICS)
def test_the_tangent_is_the_derivative(eps_n, eps_t, cohesion):
    """Central differences -- the whole point of the law is quadratic convergence."""
    model = law(eps_n, eps_t, cohesion)
    rng = np.random.default_rng(5)
    worst, step = 0.0, 1e-6
    for _ in range(300):
        trial = rng.normal(scale=3.0, size=3)
        exact = model.tangent(trial[None, :])[0]
        approx = np.zeros((3, 3))
        for j in range(3):
            shift = np.zeros(3)
            shift[j] = step
            plus, _ = model.project((trial + shift)[None, :], None)
            minus, _ = model.project((trial - shift)[None, :], None)
            approx[:, j] = (plus[0] - minus[0]) / (2 * step)
        worst = max(worst, np.abs(exact - approx).max())
    assert worst < 1e-6, f"tangent disagrees with finite differences by {worst:.2e}"


def test_the_tangent_is_the_identity_where_the_map_is():
    model = law()
    inside = np.array([[-4.0, 1.0, 0.5]])
    assert np.allclose(model.tangent(inside)[0], np.eye(3), atol=1e-14)


def test_sliding_softens_the_tangent_along_the_slip_direction():
    """``I - m (x) m``: no stiffness along the slip -- slip costs no extra traction."""
    model = law()
    sliding = np.array([[-1.0, 10.0, 0.0]])  # far outside the cone, shear along +y
    tangent = model.tangent(sliding)[0]
    direction = np.array([1.0, 0.0])
    along = direction @ tangent[1:, 1:] @ direction
    across = np.array([0.0, 1.0]) @ tangent[1:, 1:] @ np.array([0.0, 1.0])
    assert along < across  # softened along the slip, stiffer across it


def test_the_associative_coupling_block_is_present():
    """``dt_T/dt_N != 0`` is what distinguishes associative from the partial return."""
    model = law()
    sliding = np.array([[-1.0, 10.0, 0.0]])
    tangent = model.tangent(sliding)[0]
    assert np.abs(tangent[1:, 0]).max() > 1e-6  # shear responds to normal
    assert np.abs(tangent[0, 1:]).max() > 1e-6  # and normal to shear


def test_the_tangent_is_non_symmetric():
    """Coulomb friction is not associated in the classical sense; the block is not symmetric."""
    model = law(eps_n=3.0, eps_t=0.5)
    tangent = model.tangent(np.array([[-1.0, 10.0, 0.0]]))[0]
    assert not np.allclose(tangent, tangent.T, atol=1e-8)


# -- it differs from the partial return -------------------------------------------------


def test_it_is_not_the_same_as_the_partial_return():
    """If these agreed everywhere, the new law would be dead code."""
    associative = law(1.0, 2.5, 0.4)
    partial = SignoriniCoulomb(friction=FRICTION, cohesion=0.4)
    rng = np.random.default_rng(6)
    trials = rng.normal(scale=4.0, size=(400, 3))
    a, _ = associative.project(trials, None)
    b, _ = partial.project(trials, None)
    differing = np.abs(a - b).max(axis=1) > 1e-8
    assert differing.sum() > 50, "the two return mappings barely differ"


def test_sliding_changes_the_normal_traction_only_in_the_associative_law():
    """The physical signature: the closest-point correction is not purely radial.

    The partial return updates ``t_N`` first and then projects the shear at that
    fixed ``t_N``, so sliding never alters the normal traction.  The associative
    one moves along the cone normal, which has a component along the axis --
    the traction-space image of dilatancy.
    """
    trial = np.array([[-1.0, 5.0, 0.0]])  # compressive, shear well outside the cone
    associative, _ = law().project(trial, None)
    partial, _ = SignoriniCoulomb(friction=FRICTION).project(trial, None)
    assert partial[0, 0] == pytest.approx(-1.0)  # unchanged by the shear correction
    assert associative[0, 0] < -1.0 - 1e-6  # driven further into compression


def test_the_two_agree_when_the_trial_is_admissible():
    """Both are the identity inside the cone, so they can only differ on the boundary."""
    inside = np.array([[-4.0, 1.0, 0.5], [-2.0, 0.0, 0.0]])
    a, _ = law().project(inside, None)
    b, _ = SignoriniCoulomb(friction=FRICTION).project(inside, None)
    assert np.allclose(a, b, atol=1e-14)


def test_the_metric_selects_the_projection():
    """``eps_N``/``eps_T`` are not free knobs: they choose which point is closest."""
    trial = np.array([[-1.0, 5.0, 0.0]])
    isotropic, _ = law(1.0, 1.0).project(trial, None)
    normal_stiff, _ = law(0.01, 1.0).project(trial, None)
    assert not np.allclose(isotropic, normal_stiff, atol=1e-6)
    # a very stiff normal metric resists moving t_N, approaching the partial return
    partial, _ = SignoriniCoulomb(friction=FRICTION).project(trial, None)
    assert abs(normal_stiff[0, 0] - partial[0, 0]) < abs(
        isotropic[0, 0] - partial[0, 0]
    )
