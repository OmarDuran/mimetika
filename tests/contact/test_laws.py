"""Contact laws: the constitutive layer, tested without any mesh or solver.

A law is a pure function of ``(traction, jump, state)`` in the facet frame, so
it can be checked directly against the conditions it is supposed to encode.
"""

import numpy as np
import pytest

from mimetika.contact import (
    ContactLaw,
    LinearContact,
    RateAndStateFriction,
    SignoriniCoulomb,
)


# -- the taxonomy itself --------------------------------------------------------


def test_taxonomy_flags_are_declared():
    """Each law advertises what it needs; the driver keys off these."""
    lin, mc, rs = LinearContact(), SignoriniCoulomb(), RateAndStateFriction()

    assert (lin.n_state, lin.path_dependent, lin.rate_dependent) == (0, False, False)
    assert lin.symmetric_tangent and lin.linear_compliance() is not None

    assert (mc.n_state, mc.path_dependent, mc.rate_dependent) == (1, True, False)
    assert not mc.symmetric_tangent and mc.linear_compliance() is None

    assert (rs.n_state, rs.path_dependent, rs.rate_dependent) == (2, True, True)
    assert not rs.symmetric_tangent and rs.linear_compliance() is None


def test_every_law_implements_the_contract():
    for law in (LinearContact(), SignoriniCoulomb(), RateAndStateFriction()):
        assert isinstance(law, ContactLaw)
        state = law.initial_state(4)
        assert state.shape == (4, law.n_state)
        t, s = law.project(np.zeros((4, 3)), state, dt=1.0)
        assert t.shape == (4, 3) and s.shape == state.shape
        assert law.advance(t, np.zeros((4, 3)), s, dt=1.0).shape == state.shape


def test_contact_law_is_abstract():
    with pytest.raises(TypeError):
        ContactLaw()


# -- linear ----------------------------------------------------------------------


def test_linear_compliance_is_diagonal_in_the_facet_frame():
    law = LinearContact(normal_stiffness=4.0, shear_stiffness=8.0)
    assert np.allclose(law.linear_compliance(), np.diag([0.25, 0.125, 0.125]))


def test_linear_projection_is_the_identity():
    """Bonded: tension and interpenetration are both admissible."""
    law = LinearContact()
    trial = np.random.default_rng(0).standard_normal((5, 3))
    out, _ = law.project(trial, law.initial_state(5))
    assert np.allclose(out, trial)


# -- Signorini + Coulomb ----------------------------------------------------------


def test_normal_projection_removes_tension():
    law = SignoriniCoulomb(friction=0.5)
    t, _ = law.project(
        np.array([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]), law.initial_state(2)
    )
    assert t[0, 0] == 0.0  # tension clipped: the fracture opens
    assert t[1, 0] == -3.0  # compression preserved


def test_open_point_carries_no_shear():
    """The friction radius follows the *projected* normal traction."""
    law = SignoriniCoulomb(friction=0.6)
    t, _ = law.project(np.array([[5.0, 1.0, 2.0]]), law.initial_state(1))
    assert np.allclose(t, 0.0)


@pytest.mark.parametrize("mu", [0.2, 0.6, 1.0])
def test_tangential_projection_onto_the_friction_disk(mu):
    law = SignoriniCoulomb(friction=mu)
    tn = -2.0
    radius = mu * abs(tn)

    inside = np.array([[tn, 0.3 * radius, 0.4 * radius]])
    assert np.allclose(law.project(inside, law.initial_state(1))[0], inside)  # stick

    outside = np.array([[tn, 10 * radius, 0.0]])
    t_out, _ = law.project(outside, law.initial_state(1))
    assert np.isclose(np.linalg.norm(t_out[0, 1:]), radius)  # slip: on the cone
    assert np.allclose(t_out[0, 1:] / np.linalg.norm(t_out[0, 1:]), [1.0, 0.0])


def test_cohesion_shifts_the_friction_bound():
    law = SignoriniCoulomb(friction=0.0, cohesion=1.5)
    t, _ = law.project(np.array([[-1.0, 10.0, 0.0]]), law.initial_state(1))
    assert np.isclose(np.linalg.norm(t[0, 1:]), 1.5)


def test_status_labels():
    law = SignoriniCoulomb(friction=0.5)
    t = np.array([[0.0, 0.0, 0.0], [-2.0, 0.1, 0.0], [-2.0, 1.0, 0.0]])
    assert list(law.status(t)) == [0, 1, 2]  # open, stick, slip


def test_slip_accumulates_over_steps():
    law = SignoriniCoulomb()
    state = law.initial_state(1)
    g0 = np.array([[0.0, 0.0, 0.0]])
    g1 = np.array([[0.0, 0.3, 0.4]])
    state = law.advance(None, g1, state, g_prev=g0)
    assert np.isclose(state[0, 0], 0.5)
    state = law.advance(None, np.array([[0.0, 0.3, 1.6]]), state, g_prev=g1)
    assert np.isclose(state[0, 0], 1.7)


# -- rate and state ------------------------------------------------------------------


def test_rate_and_state_initialises_theta():
    law = RateAndStateFriction(theta0=2.5)
    state = law.initial_state(3)
    assert np.allclose(state[:, 1], 2.5) and np.allclose(state[:, 0], 0.0)


def test_friction_increases_with_slip_rate():
    """The direct effect ``a ln(V/V0)``."""
    law = RateAndStateFriction(mu0=0.6, a=0.01, b=0.0, V0=1e-6)
    theta = np.array([1.0])
    assert law.friction_coefficient(np.array([1e-5]), theta) > law.friction_coefficient(
        np.array([1e-7]), theta
    )


def test_friction_increases_with_state():
    """The evolution effect ``b ln(V0 theta / Dc)``."""
    law = RateAndStateFriction(mu0=0.6, a=0.0, b=0.015, Dc=1e-4, V0=1e-6)
    V = np.array([1e-6])
    assert law.friction_coefficient(V, np.array([10.0])) > law.friction_coefficient(
        V, np.array([1.0])
    )


def test_aging_law_is_stable_and_reaches_steady_state():
    """theta -> Dc/V under sustained slip, for a deliberately large dt."""
    law = RateAndStateFriction(Dc=1e-4, V0=1e-6)
    V, dt = 1e-3, 1e-2
    state = law.initial_state(1)
    g_prev = np.array([[0.0, 0.0, 0.0]])
    for k in range(2000):
        g = np.array([[0.0, V * dt * (k + 1), 0.0]])
        state = law.advance(None, g, state, dt=dt, g_prev=g_prev)
        g_prev = g
        assert state[0, 1] > 0.0  # never negative
    assert np.isclose(state[0, 1], law.Dc / V, rtol=1e-3)


def test_rate_and_state_inherits_the_unilateral_normal_condition():
    law = RateAndStateFriction()
    t, _ = law.project(np.array([[3.0, 1.0, 0.0]]), law.initial_state(1), dt=1.0)
    assert np.allclose(t, 0.0)


def test_slip_rate_floor_avoids_log_of_zero():
    law = RateAndStateFriction()
    g = np.zeros((2, 3))
    assert np.all(law.slip_rate(g, g, 1.0) >= law.Vmin)
    assert np.all(np.isfinite(law.friction_coefficient(law.slip_rate(g, g, 1.0),
                                                       np.array([1.0, 1.0]))))
