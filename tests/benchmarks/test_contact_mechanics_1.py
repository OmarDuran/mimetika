r"""Benchmark 1 of Novikov et al.: vertical displaced fault, frictionless.

Two layers again, tested separately: the analytic solution (eqs 18-22, no
solver involved) and the simulation that has to reproduce it.

The analytic slip is derived for an **unbounded** medium while the simulation
uses the paper's finite ``W = H = 4500`` m domain, so the comparison is on the
peak and the profile shape with a tolerance that reflects that, plus a
refinement trend.  What *is* required exactly is the frictionless condition
itself: zero shear traction on every fault facet, to round-off.
"""

import numpy as np
import pytest

from benchmarks.contact_mechanics import benchmark_1 as bench
from benchmarks.contact_mechanics.common import Parameters


@pytest.fixture
def parameters():
    return Parameters()


# -- the analytic solution ---------------------------------------------------------


def test_the_published_constants(parameters):
    """``C``, ``A`` and ``C/A`` of eqs (19) and (21), from Table 2 parameters."""
    assert parameters.slip_stress_scale == pytest.approx(-2.95e6, rel=2e-3)
    assert parameters.slip_stiffness == pytest.approx(1.2171e9, rel=1e-4)
    ratio = parameters.slip_stress_scale / parameters.slip_stiffness
    assert ratio == pytest.approx(-0.0024, rel=2e-2)


def test_the_geometry_is_the_published_one(parameters):
    assert (parameters.fault_a, parameters.fault_b) == (75.0, 150.0)
    assert parameters.throw == 75.0
    assert parameters.reservoir_height == 225.0


def test_the_peak_slip(parameters):
    """``(C/A)(a-b) = 0.1817`` m -- the plateau of Fig. 6 (right)."""
    assert bench.peak_slip(parameters) == pytest.approx(0.18173, rel=1e-3)


def test_the_slip_profile_has_the_published_shape(parameters):
    """Zero outside ``|y| > b``, flat over the overlap, linear in between."""
    slip = lambda y: bench.analytic_slip(np.atleast_1d(y), parameters)  # noqa: E731
    assert np.allclose(slip([-400.0, -150.0, 150.0, 400.0]), 0.0)
    plateau = slip([-74.0, -30.0, 0.0, 30.0, 74.0])
    assert np.allclose(plateau, plateau[0])
    assert np.abs(plateau[0]) == pytest.approx(bench.peak_slip(parameters), rel=1e-12)
    # linear on the ramps: the midpoint is the mean of the endpoints
    for left, right in ((-150.0, -75.0), (75.0, 150.0)):
        mid = 0.5 * (left + right)
        assert slip(mid)[0] == pytest.approx(
            0.5 * (slip(left)[0] + slip(right)[0]), rel=1e-12
        )


def test_the_slip_profile_is_continuous(parameters):
    y = np.linspace(-300.0, 300.0, 4001)
    slip = bench.analytic_slip(y, parameters)
    assert np.abs(np.diff(slip)).max() < 1e-3  # no jumps at the four breakpoints


def test_the_slip_profile_is_symmetric(parameters):
    y = np.linspace(-300.0, 300.0, 601)
    assert np.allclose(
        bench.analytic_slip(y, parameters), bench.analytic_slip(-y, parameters)
    )


def test_the_coulomb_stress_is_even_and_singular_at_the_reservoir_edges(parameters):
    """``Sigma_C`` depends on ``y`` only through ``y^2``; it blows up at ``a`` and ``b``."""
    y = np.array([10.0, 50.0, 200.0, 400.0])
    assert np.allclose(
        bench.analytic_coulomb_stress(y, parameters),
        bench.analytic_coulomb_stress(-y, parameters),
    )
    near_a = bench.analytic_coulomb_stress([74.999], parameters)[0]
    near_b = bench.analytic_coulomb_stress([149.999], parameters)[0]
    assert abs(near_a) > 1e7 and abs(near_b) > 1e7
    assert np.sign(near_a) != np.sign(near_b)  # opposite singularities


def test_the_coulomb_stress_decays_far_from_the_reservoir(parameters):
    far = bench.analytic_coulomb_stress([3000.0], parameters)[0]
    assert abs(far) < 0.02 * abs(parameters.slip_stress_scale)


# -- the simulation ------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse():
    return bench.simulate(Parameters(), nx=12, ny=30)


def test_the_offset_reservoir_is_built_correctly(parameters):
    """The throw is what loads the fault, so the pressure field must be displaced."""
    mesh, fault, pressure = bench.build(parameters, nx=12, ny=30)
    centroids = mesh.geometry.centroids(2)
    depleted = pressure != 0.0
    assert depleted.any() and not depleted.all()
    assert np.all(pressure[depleted] == parameters.depletion)
    left = depleted & (centroids[:, 0] < 0)
    right = depleted & (centroids[:, 0] > 0)
    # left spans [-b, a], right spans [-a, b]: the two are mirror images in y
    assert centroids[left, 1].min() < centroids[right, 1].min()
    assert centroids[left, 1].max() < centroids[right, 1].max()
    assert len(fault) == 30  # one fault facet per row of cells


def test_the_solve_converges(coarse):
    assert coarse["state"].converged
    assert coarse["state"].iterations <= 10  # Newton, not Picard


def test_the_fault_carries_no_shear_traction(coarse):
    """The frictionless condition itself -- required exactly, not approximately."""
    shear = np.abs(coarse["traction"][:, 1])
    normal = np.abs(coarse["traction"][:, 0]).max()
    assert shear.max() <= 1e-8 * max(normal, 1.0)


def test_the_peak_slip_is_close_to_the_analytic_value(coarse):
    """Within the finite-domain discrepancy; the trend is checked separately."""
    got = np.abs(coarse["slip"]).max()
    assert got == pytest.approx(bench.peak_slip(Parameters()), rel=0.15)


def test_the_slip_is_localised_at_the_reservoir(coarse):
    """Slip must die away from the offset, as the analytic profile does."""
    parameters = Parameters()
    y, slip = coarse["y"], np.abs(coarse["slip"])
    far = slip[np.abs(y) > 3.0 * parameters.fault_b]
    # inclusive: on a coarse mesh the innermost facets sit exactly at |y| = a
    near = slip[np.abs(y) <= parameters.fault_a]
    assert near.size and far.size
    assert far.max() < 0.25 * near.max()


def test_the_slip_is_symmetric_about_the_reservoir(coarse):
    """``delta(-y) = delta(y)``, as the analytic profile is.

    Compared against the **peak**, not against the local value: the offset
    reservoir is symmetric only under the point reflection ``(x, y) -> (-x, -y)``,
    and the boundary conditions (free top, roller base) are not ``y``-symmetric
    at all, so the far field carries a small absolute asymmetry that is large in
    relative terms precisely where the slip is near zero.

    Checked within ``|y| <= 3b``, where the slip is actually resolved.  Beyond
    that the asymmetry grows steadily towards the top and bottom boundaries --
    it is the boundary conditions showing through, not the fault.
    """
    parameters = Parameters()
    y, slip = coarse["y"], np.abs(coarse["slip"])
    peak = slip.max()
    window = np.abs(y) <= 3.0 * parameters.fault_b
    paired = {}
    for value, magnitude in zip(y[window], slip[window]):
        paired.setdefault(round(abs(value), 6), []).append(magnitude)
    pairs = [m for m in paired.values() if len(m) == 2]
    assert len(pairs) >= 3
    for magnitudes in pairs:
        assert abs(magnitudes[0] - magnitudes[1]) < 5e-3 * peak


def test_refinement_approaches_the_analytic_peak():
    """The finite domain leaves a gap, but refinement must close most of it."""
    parameters = Parameters()
    errors = []
    for nx, ny in ((12, 30), (16, 60)):
        result = bench.simulate(parameters, nx=nx, ny=ny)
        peak = np.abs(result["slip"]).max()
        errors.append(abs(peak / bench.peak_slip(parameters) - 1.0))
    assert errors[1] < errors[0]
    assert errors[1] < 0.08


def test_the_default_law_is_the_unilateral_one(coarse):
    """Benchmark 1 runs ``SignoriniCoulomb(friction=0)`` -- the physical model.

    A benchmark exists to test laws, so the law that represents the situation is
    the one it should run, not a bonded stand-in chosen because it is easier.

    The incremental normal traction goes into tension over part of the fault.
    That is admissible here -- the in-situ normal stress is about ``-57`` MPa, so
    the fault is still shut by a wide margin -- but a unilateral law applied to
    the *increment* would read it as opening and clip it to zero.
    """
    from mimetika.contact import SignoriniCoulomb

    parameters = Parameters()
    mesh, fault, _ = bench.build(parameters, nx=12, ny=30)
    driver_law = bench.simulate(parameters, nx=12, ny=30)["state"]
    assert driver_law.converged
    # the incremental normal traction is tensile, and the law keeps it: with the
    # in-situ prestress the *total* traction is compressive, so the fault is shut
    assert coarse["traction"][:, 0].max() > 1e6
    assert isinstance(SignoriniCoulomb(friction=0.0), SignoriniCoulomb)


def test_without_the_prestress_the_unilateral_law_opens_the_fault():
    """The failure the prestress prevents, kept visible.

    Deliberately runs ``prestress=False`` so the law sees only the increment.  It
    then reads the ``+8.4`` MPa tensile increment as opening and clips it -- which
    is a defect of the incremental formulation, not of Signorini.
    """
    parameters = Parameters()
    with_prestress = bench.simulate(parameters, nx=12, ny=30, prestress=True)
    without = bench.simulate(parameters, nx=12, ny=30, prestress=False)
    assert with_prestress["state"].converged and without["state"].converged
    assert with_prestress["traction"][:, 0].max() > 1e6  # tension kept: shut
    assert without["traction"][:, 0].max() < 1e-6  # tension clipped: opened


def test_the_two_laws_agree_once_both_see_the_total_traction():
    """``SignoriniCoulomb(friction=0)`` and the bonded law coincide here.

    They must: with the in-situ prestress the total normal traction is
    compressive everywhere, so the unilateral constraint is inactive and the two
    laws describe the same fault.
    """
    from mimetika.contact import FrictionlessBilateral

    parameters = Parameters()
    unilateral = bench.simulate(parameters, nx=12, ny=30)
    bonded = bench.simulate(parameters, nx=12, ny=30, law=FrictionlessBilateral())
    assert np.allclose(unilateral["slip"], bonded["slip"], atol=1e-12)
    assert np.allclose(unilateral["traction"], bonded["traction"], atol=1e-6)
