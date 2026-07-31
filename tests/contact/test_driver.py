"""Contact driver: augmented-Lagrangian solve of fracture contact.

The driver owns the rotation into the facet frame, the moment/point conversion,
assembly and the Uzawa iteration; the law only supplies a projection.  These
tests pin the physics the combination has to reproduce:

* tension opens the fracture and leaves it **traction free**;
* compression closes it with **no interpenetration**;
* shear sticks inside the friction cone and slides exactly on it.
"""

import numpy as np
import pytest

from mimetika.contact import (
    elastic_mechanics,
    ContactDriver,
    LinearContact,
    RateAndStateFriction,
    SignoriniCoulomb,
)
from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.fracture import facets_on_plane

MU = LAM = 1.0
OEDOMETER = 2 * MU + LAM  # confined modulus: sigma_xx = (2mu+lam) e


def setup(shape=(2, 2, 2), law=None, tets=False, **kw):
    mesh = (structured_tets if tets else structured_box)(*shape)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    law = SignoriniCoulomb(friction=0.6) if law is None else law
    return mesh, tags, ContactDriver(mesh, tags, law, mu=MU, lam=LAM, **kw)


def load(normal=0.0, shear=0.0):
    """Confined normal strain plus a shear, prescribed on the whole boundary."""

    def u(x):
        x = np.atleast_2d(x)
        out = np.zeros((len(x), 3))
        out[:, 0] = normal * x[:, 0]
        out[:, 1] = shear * x[:, 0]
        return out

    return u


# -- the augmentation parameter ---------------------------------------------------


def test_augmentation_is_derived_from_stiffness_and_geometry():
    """r must match the inverse compliance the fracture sees, or Uzawa cycles."""
    mesh, tags, d = setup()
    # two half-cells of 0.25 each: compliance = 0.5 / (2mu+lam)
    assert np.allclose(d._r, OEDOMETER / 0.5)


def test_augmentation_uses_the_true_centroid_distance_on_tets():
    """A volume/area shortcut is exact for boxes but wrong for tetrahedra."""
    mesh, tags, d = setup(tets=True)
    g = mesh.geometry
    f = int(tags[0])
    cells = mesh.complex.boundary_matrix(3).tocsr()[f].indices
    length = sum(
        abs((g.centroids(2)[f] - g.centroids(3)[int(c)]) @ g.facet_normals()[f])
        for c in cells
    )
    assert np.isclose(d._r[0], OEDOMETER / length)


def test_explicit_augmentation_is_respected():
    _, _, d = setup(augmentation=3.5)
    assert np.allclose(d._r, 3.5)


def test_too_large_an_augmentation_fails_to_converge():
    """Documents why r is derived rather than guessed."""
    mesh, _, d = setup(augmentation=200.0, max_iterations=60)
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01)))
    assert not state.converged


# -- moments <-> facet-frame values ------------------------------------------------


@pytest.mark.parametrize("mode", ["averaged", "pointwise"])
def test_moment_value_round_trip(mode):
    mesh, tags, d = setup(enforcement=mode)
    rng = np.random.default_rng(0)
    for f in tags:
        vals = rng.standard_normal((d.points_per_facet(int(f)), 3))
        back = d.to_values(d.to_moments(vals, int(f)), int(f))
        if mode == "averaged":
            assert np.allclose(back, vals)  # constant part is exactly recovered
        else:
            # values -> moments -> values is a PROJECTION, not the identity: a 3D
            # facet has more quadrature points than P_1 basis functions, so the
            # first step is a genuine loss.  Idempotence is the sharp statement,
            # and unlike a shape check it would catch a wrong basis.
            assert back.shape == vals.shape
            again = d.to_values(d.to_moments(back, int(f)), int(f))
            assert np.allclose(again, back)


@pytest.mark.parametrize("mode", ["averaged", "pointwise"])
def test_the_moment_round_trip_is_exact(mode):
    """``moments -> values -> moments`` IS the identity, in the other direction.

    This is the direction the contact constraint actually uses -- the multiplier
    is converted to moments and pinned -- so it has to be lossless, and for
    ``pointwise`` it is: the enforcement points carry the full ``P_1`` basis.

    It was NOT lossless in 2D.  ``to_moments`` integrated against a basis scaled
    by ``|f| ** (1/k)`` while ``to_values`` inverted a Gram built with
    ``sqrt(|f|)``; those agree only for ``k = 2``, so in 2D the round trip
    destroyed 88% of the linear part.  The old test asserted only ``back.shape``
    in this mode and could never have seen it.
    """
    mesh, tags, d = setup(enforcement=mode)
    rng = np.random.default_rng(3)
    for f in tags:
        moments = rng.standard_normal(d.ndf)
        back = d.to_moments(d.to_values(moments, int(f)), int(f))
        if mode == "pointwise":
            assert np.allclose(back, moments, atol=1e-12)
        else:  # averaged keeps only the constant part, by construction
            block = back.reshape(d.dim, d.dim)
            original = moments.reshape(d.dim, d.dim)
            assert np.allclose(block[:, 0], original[:, 0], atol=1e-12)
            assert np.allclose(block[:, 1:], 0.0, atol=1e-12)


def test_enforcement_point_counts():
    mesh, tags, avg = setup(enforcement="averaged")
    _, _, pw = setup(enforcement="pointwise")
    assert avg.n_points == len(tags)
    assert pw.n_points > avg.n_points


def test_unknown_enforcement_is_rejected():
    with pytest.raises(ValueError, match="enforcement"):
        setup(enforcement="nodal")


# -- linear law: no iteration, exact ------------------------------------------------


def test_linear_law_solves_in_one_pass_and_matches_the_contact_law():
    kn = 0.5
    mesh, tags, d = setup(law=LinearContact(kn, kn))
    delta = OEDOMETER * 0.01 / kn

    def u(x):
        x = np.atleast_2d(x)
        side = 1.0 if x[:, 0].mean() > 0.5 else 0.0
        out = np.zeros((len(x), 3))
        out[:, 0] = 0.01 * x[:, 0] + delta * side
        return out

    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=u))
    assert state.iterations == 1  # exactly linear: no outer loop
    assert np.allclose(state.jump[:, 0], delta, atol=1e-9)


# -- Signorini ------------------------------------------------------------------------


def test_tension_opens_the_fracture_with_zero_traction():
    mesh, tags, d = setup()
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=+0.01)))
    t = d.tractions(state.solution["stress"])
    assert state.converged
    assert np.allclose(t[:, 0], 0.0, atol=1e-12)  # traction free
    assert (state.jump[:, 0] > 0).all()  # genuinely open
    assert list(np.unique(d.law.status(t))) == [0]


def test_compression_closes_without_interpenetration():
    mesh, tags, d = setup()
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01)))
    t = d.tractions(state.solution["stress"])
    assert state.converged
    assert np.allclose(state.jump[:, 0], 0.0, atol=1e-8)  # no interpenetration
    assert np.allclose(t[:, 0], -OEDOMETER * 0.01, rtol=1e-6)  # exact closed stress
    assert (t[:, 0] < 0).all()


def test_complementarity_holds():
    """g_n >= 0, t_n <= 0 and g_n t_n = 0 at every enforcement point."""
    mesh, tags, d = setup()
    for normal in (+0.01, -0.01, 0.0):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=normal)))
        t = d.tractions(state.solution["stress"])
        g = state.jump
        assert (g[:, 0] > -1e-8).all()
        assert (t[:, 0] < 1e-10).all()
        assert np.allclose(g[:, 0] * t[:, 0], 0.0, atol=1e-9)


# -- Coulomb friction --------------------------------------------------------------------


def test_stick_below_the_cone():
    mesh, tags, d = setup()
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.002)))
    t = d.tractions(state.solution["stress"])
    bound = -d.law.friction * t[:, 0]
    assert (np.linalg.norm(t[:, 1:], axis=1) < bound).all()
    assert np.allclose(state.jump[:, 1:], 0.0, atol=1e-8)  # no slip
    assert list(np.unique(d.law.status(t))) == [1]


def test_slip_saturates_exactly_on_the_cone():
    mesh, tags, d = setup()
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)))
    t = d.tractions(state.solution["stress"])
    bound = -d.law.friction * t[:, 0]
    assert np.allclose(np.linalg.norm(t[:, 1:], axis=1), bound, rtol=1e-6)
    assert (np.abs(state.jump[:, 1]) > 1e-4).all()  # genuine slip
    assert list(np.unique(d.law.status(t))) == [2]


def test_shear_traction_is_capped_however_hard_we_push():
    mesh, tags, d = setup()
    magnitudes = []
    for shear in (0.05, 0.2, 1.0):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=shear)))
        t = d.tractions(state.solution["stress"])
        magnitudes.append(np.linalg.norm(t[0, 1:]))
    assert np.allclose(magnitudes, magnitudes[0], rtol=1e-6)


@pytest.mark.parametrize("friction", [0.2, 0.6, 1.0])
def test_the_cap_scales_with_the_friction_coefficient(friction):
    mesh, tags, d = setup(law=SignoriniCoulomb(friction=friction))
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.5)))
    t = d.tractions(state.solution["stress"])
    assert np.allclose(
        np.linalg.norm(t[:, 1:], axis=1), friction * OEDOMETER * 0.01, rtol=1e-5
    )


# -- stepping and state ---------------------------------------------------------------


def test_caller_drives_the_loop_and_slip_accumulates():
    mesh, tags, d = setup()
    state, history = None, []
    for shear in (0.05, 0.10, 0.15):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=shear)), state=state)
        assert state.converged
        history.append(state.internal[0, 0])
    assert history[0] < history[1] < history[2]  # monotone accumulation


def test_initial_state_shape():
    mesh, tags, d = setup()
    s = d.initial_state()
    assert s.multiplier.shape == (len(tags), 9)
    assert s.internal.shape == (d.n_points, d.law.n_state)
    assert s.jump.shape == (d.n_points, 3)


# -- rate and state ----------------------------------------------------------------------


def test_rate_dependence_changes_the_friction_reached():
    law = RateAndStateFriction(mu0=0.6, a=0.01, b=0.015, Dc=1e-4, V0=1e-6)
    mesh, tags, d = setup(law=law)
    ratios = []
    for dt in (1e-2, 1e-4):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)), dt=dt)
        t = d.tractions(state.solution["stress"])
        ratios.append(np.linalg.norm(t[0, 1:]) / -t[0, 0])
    assert not np.isclose(ratios[0], ratios[1], rtol=1e-3)  # rate matters
    assert ratios[1] > ratios[0]  # faster slip -> stronger (direct effect)


def test_state_variable_evolves_across_steps():
    law = RateAndStateFriction(Dc=1e-4)
    mesh, tags, d = setup(law=law)
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)), dt=1e-3)
    assert not np.isclose(state.internal[0, 1], law.theta0)


# -- enforcement modes and other meshes -----------------------------------------------------


def test_both_enforcement_modes_agree_on_a_uniform_problem():
    """With a spatially uniform state, averaging loses nothing."""
    results = {}
    for mode in ("averaged", "pointwise"):
        mesh, tags, d = setup(enforcement=mode)
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)))
        assert state.converged
        results[mode] = d.tractions(state.solution["stress"])[0]
    # the two modes are different discretisations of the projection, so they
    # agree on the physics (normal traction, shear magnitude) rather than bitwise
    assert np.isclose(results["averaged"][0], results["pointwise"][0], rtol=1e-6)
    assert np.isclose(
        np.linalg.norm(results["averaged"][1:]),
        np.linalg.norm(results["pointwise"][1:]),
        rtol=1e-4,
    )


def test_contact_on_a_tetrahedral_mesh():
    mesh, tags, d = setup(tets=True)
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)))
    t = d.tractions(state.solution["stress"])
    assert state.converged
    assert (t[:, 0] < 0).all()
    assert np.allclose(
        np.linalg.norm(t[:, 1:], axis=1), -d.law.friction * t[:, 0], rtol=1e-5
    )


def test_relaxation_is_required_in_the_sliding_regime():
    """Without it the Uzawa fixed point settles into a limit cycle.

    The tangential update is a contraction while sticking, but not while
    sliding, so the plain iteration (relaxation = 1) can oscillate with constant
    amplitude on meshes where fracture facets couple strongly.
    """
    mesh, tags, plain = setup(tets=True, relaxation=1.0, max_iterations=40)
    stuck = plain.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)))
    assert not stuck.converged

    mesh, tags, damped = setup(tets=True)  # default relaxation
    ok = damped.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-0.01, shear=0.05)))
    assert ok.converged
    t = damped.tractions(ok.solution["stress"])
    assert np.allclose(
        np.linalg.norm(t[:, 1:], axis=1) / -t[:, 0], damped.law.friction, rtol=1e-6
    )


def test_the_jump_operator_recovers_a_prescribed_offset():
    """The jump operator against a **known** jump -- it had no such test.

    ``test_linear_law_solves_in_one_pass_and_matches_the_contact_law`` looks like
    this check but is not: a linear law takes the compliance shortcut, where the
    jump is ``A_f t`` and the jump operator is never evaluated.  The operator is
    used only on the Uzawa/Newton path, so nothing pinned its scaling.

    Here a very soft fracture is pulled open by a prescribed discontinuous
    displacement, so the jump is known in advance.  Being h-independent is the
    sharp part: a spurious ``Gram`` factor in the operator scales the jump by the
    facet measure, which is invisible on one mesh and obvious across two.
    """
    from mimetika.assembly.mixed import MixedElasticity
    from mimetika.solver.saddle import solve_saddle

    offset = 0.02
    recovered = []
    for n in (2, 4):
        mesh = structured_box(n, n, n)
        tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
        driver = ContactDriver(mesh, tags, SignoriniCoulomb(friction=0.6),
                               mu=MU, lam=LAM)

        def prescribed(x):
            x = np.atleast_2d(x)
            out = np.zeros((len(x), 3))
            out[:, 1] = np.where(x[:, 0] > 0.5, offset, 0.0)  # tangential step
            return out

        problem = MixedElasticity(mesh, mu=MU, lam=LAM,
                                  contact=driver.contact_geometry(
                                      np.diag([1e6, 1e6, 1e6])))
        matrix, rhs = problem.assemble_constrained(dirichlet=prescribed)
        solution = problem.split(
            solve_saddle(matrix, rhs, problem.block_sizes, method="direct")
        )
        vector = np.concatenate([solution["stress"], solution["displacement"],
                                 solution["rotation"]])
        jump = (driver.jump_operator(problem) @ vector).reshape(
            driver.n_points, driver.dim
        )
        recovered.append(np.abs(jump[:, 1]).max())

    # a soft fracture takes up essentially the whole prescribed offset ...
    assert recovered[0] == pytest.approx(offset, rel=0.05)
    # ... and the answer must not depend on the facet size
    assert recovered[1] == pytest.approx(recovered[0], rel=0.05)


def test_the_jump_operator_matches_the_moment_conversion_up_to_the_gram():
    """Pins *why* the operator carries no ``Gram^{-1}`` while ``to_values`` does.

    They convert different objects.  A traction DOF is a moment, so its values
    need ``Gram^{-1}``.  The jump residual is paired *against* that moment DOF,
    so the pairing already supplies one and the residual is already a coefficient
    vector.  Adding a second inverse divides the jump by ``|e|``.
    """
    mesh, tags, driver = setup()
    facet = int(tags[0])
    gram = driver._geom.facet_gram(facet)
    moments = np.arange(1.0, driver.ndf + 1.0)
    #  to_values solves the Gram; the jump operator's shape function does not
    as_traction = driver.to_values(moments, facet)
    as_jump = driver.to_values(
        (gram @ moments.reshape(driver.dim, driver.dim).T).T.ravel(), facet
    )
    assert not np.allclose(as_traction, as_jump)
    scale = mesh.geometry.measure(mesh.dim - 1)[facet]
    assert np.allclose(as_jump[:, 0], scale * as_traction[:, 0], rtol=1e-10)
