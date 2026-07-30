r"""Benchmark 0 of Novikov et al.: in-situ state and depletion response, no fault.

Two independent claims, tested separately:

* the in-situ profiles printed in the paper follow from the Table 2 parameters
  (a check on the *setup*, needing no solver);
* the mixed poromechanics solver reproduces the uniaxial depletion response
  exactly (a check on the *solver*, against a closed form).

Tolerances are set by what the paper prints -- three or four significant
figures -- for the published comparisons, and by round-off for the closed
forms, which are exact for this discretisation.
"""

import numpy as np
import pytest

from benchmarks.contact_mechanics.benchmark_0 import (
    depletion_response,
    in_situ_report,
)
from benchmarks.contact_mechanics.common import Parameters, linear_fit

# eq. (17)-(19): (intercept [Pa], gradient [Pa/m]), y measured upwards
PUBLISHED = {
    "sigma_yy": (-82.60e6, 23.60e3),
    "sigma_xx": (-57.05e6, 16.30e3),
    "pressure": (35.00e6, -10.06e3),
    "sigma_normal_70": (-60.04e6, 17.15e3),
    "sigma_shear_70": (8.21e6, -2.35e3),
}


@pytest.fixture
def parameters():
    return Parameters()


# -- the in-situ state derives from Table 2 ---------------------------------------


@pytest.mark.parametrize("field", list(PUBLISHED))
def test_in_situ_profiles_match_the_paper(parameters, field):
    """Derived from ``(rho_s, rho_fl, phi, g, D0, K0, alpha, p0)`` -- not tabulated.

    ``sigma_shear_70`` is compared in magnitude: its sign depends on which way
    the fault tangent is taken, and the paper's choice is opposite to ours.

    The tolerance is set by the paper's own precision -- it prints three
    significant figures, so ``2.35`` against a computed ``2.340`` is agreement,
    not disagreement.
    """
    intercept, gradient = in_situ_report(parameters)[field]
    want_intercept, want_gradient = PUBLISHED[field]
    if field == "sigma_shear_70":
        intercept, gradient = abs(intercept), abs(gradient)
        want_intercept, want_gradient = abs(want_intercept), abs(want_gradient)
    assert intercept == pytest.approx(want_intercept, rel=5e-3)
    assert gradient == pytest.approx(want_gradient, rel=5e-3)


def test_bulk_density_and_uniaxial_modulus(parameters):
    assert parameters.bulk_density == pytest.approx(2406.2, rel=1e-4)
    assert parameters.uniaxial_modulus == pytest.approx(15.79e9, rel=1e-3)


def test_the_in_situ_state_is_genuinely_linear_in_depth(parameters):
    """The paper states these as linear profiles; a fit must be exact, not close."""
    y = np.linspace(-2000.0, 2000.0, 17)
    for values in (
        parameters.vertical_stress(y),
        parameters.horizontal_stress(y),
        parameters.pressure(y),
    ):
        intercept, gradient = linear_fit(y, values)
        assert np.allclose(values, intercept + gradient * y, rtol=1e-12)


def test_effective_stress_ratio_is_the_earth_pressure_coefficient(parameters):
    """``K0`` acts on *effective* stress -- using total stress would be a classic slip."""
    y = np.linspace(-1000.0, 1000.0, 9)
    effective_vertical = parameters.vertical_stress(y) + parameters.biot * (
        parameters.pressure(y)
    )
    effective_horizontal = parameters.horizontal_stress(y) + parameters.biot * (
        parameters.pressure(y)
    )
    assert np.allclose(
        effective_horizontal, parameters.earth_pressure * effective_vertical
    )


def test_the_state_is_compressive_everywhere(parameters):
    y = np.linspace(-parameters.height / 2, parameters.height / 2, 21)
    assert np.all(parameters.vertical_stress(y) < 0)
    assert np.all(parameters.horizontal_stress(y) < 0)
    assert np.all(parameters.pressure(y) > 0)


@pytest.mark.parametrize("dip", [90.0, 70.0, 45.0])
def test_resolved_traction_is_a_genuine_rotation(parameters, dip):
    """Normal and shear must equal the tensor contraction, and preserve invariants."""
    y = np.linspace(-500.0, 500.0, 5)
    normal_vector, tangent = parameters.fault_basis(dip)
    stress = parameters.stress_tensor(y)
    normal, shear = parameters.resolved(y, dip=dip)

    assert np.allclose(normal_vector @ tangent, 0.0)
    assert np.allclose(np.linalg.norm(normal_vector), 1.0)
    # the trace is rotation invariant
    other = np.einsum("i,qij,j->q", tangent, stress, tangent)
    assert np.allclose(normal + other, np.trace(stress, axis1=1, axis2=2))
    assert np.allclose(shear, np.einsum("i,qij,j->q", normal_vector, stress, tangent))


def test_a_vertical_fault_sees_the_horizontal_stress_and_no_shear(parameters):
    """At ``dip = 90`` the fault normal is ``e_x``: a sanity anchor for the rotation."""
    y = np.linspace(-500.0, 500.0, 5)
    normal, shear = parameters.resolved(y, dip=90.0)
    assert np.allclose(normal, parameters.horizontal_stress(y))
    assert np.allclose(shear, 0.0, atol=1e-6)


# -- the depletion response is reproduced by the solver ------------------------------


@pytest.fixture(scope="module")
def response():
    return depletion_response(Parameters(), n=6)


def test_uniaxial_strain_is_exact(response):
    parameters = Parameters()
    assert response["vertical_strain"] == pytest.approx(
        parameters.vertical_strain, rel=1e-10
    )


def test_compaction_matches_the_published_value(response):
    assert response["compaction"] == pytest.approx(-0.32, abs=5e-3)


def test_incremental_horizontal_stresses_are_exact(response):
    parameters = Parameters()
    assert response["sigma_xx_effective"] == pytest.approx(
        parameters.horizontal_effective_increment, rel=1e-10
    )
    assert response["sigma_xx_total"] == pytest.approx(
        parameters.horizontal_total_increment, rel=1e-10
    )
    assert response["sigma_xx_effective"] == pytest.approx(-3.97e6, rel=1e-3)
    assert response["sigma_xx_total"] == pytest.approx(18.53e6, rel=1e-3)


def test_the_free_surface_carries_no_vertical_stress(response):
    """The top is traction free, so ``sigma_yy`` must vanish, not merely be small."""
    assert response["sigma_yy_total"] < 1e-6 * abs(Parameters().depletion)


@pytest.mark.parametrize("n", [3, 6, 12])
def test_the_response_is_mesh_independent(n):
    """A uniform state: refinement must change nothing at all."""
    parameters = Parameters()
    got = depletion_response(parameters, n=n)
    assert got["vertical_strain"] == pytest.approx(
        parameters.vertical_strain, rel=1e-10
    )


def test_rollers_are_what_make_it_uniaxial():
    """Guard the premise: free lateral slip is what produces the closed form.

    Compared against the same problem with the sides **fully clamped** rather
    than free to slide.  A genuinely unconfined block would be the more obvious
    contrast, but prescribing traction all round leaves the rigid-body modes
    undetermined and the system singular -- clamping is the well-posed
    alternative, and it gives a visibly different answer.
    """
    from mimetika.assembly.mixed import boundary_facets
    from mimetika.assembly.poromechanics import PoroMechanics
    from mimetika.materials import Material
    from mimetika.mesh import structured_quads

    parameters = Parameters()
    mesh = structured_quads(4, 4)
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=parameters.shear_modulus,
            poisson=parameters.poisson,
            biot=parameters.biot,
        ),
    )
    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][1] - 1.0) < 1e-12]
    sides = [f for f in boundary if f not in set(top)]

    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    no_traction = lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3))  # noqa: E731
    common = dict(
        dt=None,
        dirichlet=zero,
        pressure=parameters.depletion,
        traction=no_traction,
        traction_facets=top,
    )
    rolling = problem.solve(roller_facets=sides, **common)
    clamped = problem.solve(**common)

    rolling_xx = problem.mechanics.cell_stress(rolling["stress"])[:, 0, 0].mean()
    clamped_xx = problem.mechanics.cell_stress(clamped["stress"])[:, 0, 0].mean()
    assert abs(rolling_xx - clamped_xx) > 1e-2 * abs(rolling_xx)
    # and only the rolling one is the uniaxial closed form
    assert rolling_xx == pytest.approx(
        parameters.horizontal_total_increment, rel=1e-10
    )


def test_roller_dofs_reject_a_non_axis_aligned_facet():
    """The restriction is documented, so it must be enforced rather than assumed."""
    from mimetika.assembly.mixed import MixedElasticity, boundary_facets
    from mimetika.mesh import structured_triangles

    mesh = structured_triangles(2, 2)
    problem = MixedElasticity(mesh, mu=1.0, lam=1.0)
    diagonal = [
        f
        for f in range(mesh.num_cells(1))
        if abs(abs(mesh.geometry.facet_frame(f)[0][0]) - 1.0) > 1e-10
        and abs(abs(mesh.geometry.facet_frame(f)[0][1]) - 1.0) > 1e-10
    ]
    assert diagonal, "the triangulation should have diagonal edges"
    with pytest.raises(ValueError, match="axis aligned"):
        problem.roller_dofs(diagonal[:1])
    # ... while the axis-aligned boundary edges are accepted
    axis_aligned = [f for f in boundary_facets(mesh)]
    assert len(problem.roller_dofs(axis_aligned)) > 0
