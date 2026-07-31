r"""Terzaghi's consolidation column, 2D and 3D, against the closed form.

The column exists to exercise the boundary conditions -- applied traction,
rollers, a drained face and sealed faces -- on a problem whose answer is known
exactly.  So the tests check the boundary conditions as directly as they check
the numbers:

* the section stays **uniform** (the rollers really do impose uniaxial strain),
* the answer does not depend on the column's **width** or its lateral cell count,
* **2D and 3D agree**, which they must, having the same 1D solution,
* the error **converges** at first order, so what remains is discretization.

The closed form is itself checked against an independent route -- the settlement
series is verified by integrating the pressure series -- so a mistake in the
analytic side cannot quietly define the answer.
"""

import numpy as np
import pytest

from benchmarks.poroelasticity.terzaghi import (
    TIME_FACTORS,
    Column,
    analytic_consolidation,
    analytic_pressure,
    compare,
    simulate,
)

# coarse but adequate; the convergence test is what justifies the tolerances
COARSE = dict(axial=30, lateral=2, per_decade=8)


@pytest.fixture(scope="module")
def column():
    return Column()


@pytest.fixture(scope="module")
def solved(column):
    """One 2D and one 3D run, shared by every test -- these are the slow part."""
    return {
        2: simulate(column, dim=2, **COARSE),
        3: simulate(column, dim=3, axial=20, lateral=2, per_decade=8),
    }


# -- the closed form, checked against itself -------------------------------------------


def test_the_drained_face_is_at_zero_and_the_base_is_the_maximum():
    """Zero at the drained top, and monotone down the column to the sealed base.

    Monotonicity rather than ``argmax``: at early ``T`` the profile is flat at 1
    to machine precision over almost the whole column, so the maximum is a tie and
    its index says nothing.
    """
    for factor in TIME_FACTORS:
        profile = analytic_pressure(np.linspace(0, 1, 51), factor)
        assert profile[-1] == pytest.approx(0.0, abs=1e-12)  # z/L = 1, drained
        assert profile[0] == pytest.approx(profile.max(), abs=1e-12)  # base
        assert np.all(np.diff(profile) <= 1e-12), f"not monotone at T = {factor:g}"


def test_the_early_column_is_undrained_away_from_the_top():
    """As ``T -> 0`` the pressure is ``p_0`` everywhere but a thin top layer."""
    profile = analytic_pressure(np.linspace(0.0, 0.9, 40), 1e-6)
    assert np.allclose(profile, 1.0, atol=1e-6)


def test_the_late_column_has_fully_drained():
    assert np.max(np.abs(analytic_pressure(np.linspace(0, 1, 40), 5.0))) < 1e-4
    assert analytic_consolidation(5.0)[0] == pytest.approx(1.0, abs=1e-4)


def test_the_settlement_series_is_the_integral_of_the_pressure_series():
    """Independent route to ``U``: ``w = (sigma_0 L - alpha int p dz)/K_v``.

    If both series were derived the same way this would be circular; they are
    not -- one is integrated numerically here and compared to the closed form.
    """
    grid = np.linspace(0.0, 1.0, 20001)
    for factor in TIME_FACTORS:
        mean = np.trapezoid(analytic_pressure(grid, factor), grid)
        assert analytic_consolidation(factor)[0] == pytest.approx(1.0 - mean, abs=1e-6)


def test_the_load_partition_shifts_the_start_but_not_the_end(column):
    """``U(0+) = 1 - beta`` and ``U(inf) = 1``, for any beta."""
    beta = column.load_partition
    assert 0.0 < beta < 1.0
    assert analytic_consolidation(1e-12, beta)[0] == pytest.approx(1 - beta, abs=1e-6)
    assert analytic_consolidation(20.0, beta)[0] == pytest.approx(1.0, abs=1e-6)
    # Terzaghi's own column: incompressible fluid and grains, so nothing is instant
    assert analytic_consolidation(1e-12, 1.0)[0] == pytest.approx(0.0, abs=1e-6)


def test_the_derived_constants_are_self_consistent(column):
    """``p_0``, ``beta`` and ``c_v`` are three views of the same two numbers."""
    assert column.load_partition == pytest.approx(
        column.biot * column.initial_pressure / column.load
    )
    assert column.consolidation_coefficient == pytest.approx(
        column.permeability / column.viscosity / column.storage
    )


# -- the discrete column ----------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_the_pressure_profile_matches_at_every_time_factor(column, solved, dim):
    for row in compare(column, solved[dim]):
        assert row["rms"] < 5e-2, f"T = {row['factor']:g}: rms {row['rms']:.2e}"
        assert row["max"] < 8e-2, f"T = {row['factor']:g}: max {row['max']:.2e}"


@pytest.mark.parametrize("dim", [2, 3])
def test_the_consolidation_curve_matches(column, solved, dim):
    for row in compare(column, solved[dim]):
        assert abs(row["consolidation"] - row["consolidation_exact"]) < 3e-2


@pytest.mark.parametrize("dim", [2, 3])
def test_the_cross_section_stays_uniform(column, solved, dim):
    """The rollers must give uniaxial strain: no lateral variation at all.

    This is the boundary-condition test.  A column that bulges is not confined,
    and it would still produce a plausible-looking profile down the axis.
    """
    for row in compare(column, solved[dim]):
        assert row["spread"] < 1e-6, f"T = {row['factor']:g}: {row['spread']:.2e}"


def test_two_and_three_dimensions_agree(column, solved):
    """A 3D box under uniaxial strain has no freedom a 2D one lacks."""
    for two, three in zip(compare(column, solved[2]), compare(column, solved[3])):
        assert two["factor"] == three["factor"]
        assert abs(two["consolidation"] - three["consolidation"]) < 5e-3


def test_the_column_fully_consolidates(column, solved):
    """By ``T = 1`` almost all of the settlement has happened, in both dimensions."""
    for dim in (2, 3):
        last = compare(column, solved[dim])[-1]
        assert last["factor"] == 1.0
        assert last["consolidation"] > 0.9


# -- the answer must not depend on things it cannot depend on ---------------------------


def test_the_width_does_not_matter(column):
    """A 1D problem cannot know how wide the column is."""
    narrow = simulate(Column(width=0.1), dim=2, **COARSE)
    wide = simulate(Column(width=25.0), dim=2, **COARSE)
    for a, b in zip(compare(Column(width=0.1), narrow), compare(Column(width=25.0), wide)):
        assert a["consolidation"] == pytest.approx(b["consolidation"], abs=1e-9)


def test_the_lateral_cell_count_does_not_matter(column):
    one = simulate(column, dim=2, axial=30, lateral=1, per_decade=8)
    five = simulate(column, dim=2, axial=30, lateral=5, per_decade=8)
    for a, b in zip(compare(column, one), compare(column, five)):
        assert a["consolidation"] == pytest.approx(b["consolidation"], abs=1e-9)
        assert a["rms"] == pytest.approx(b["rms"], abs=1e-9)


# -- convergence, which is what licenses the tolerances above ---------------------------


def test_refinement_converges_at_first_order(column):
    """Backward Euler in time, lowest-order mixed in space: halve h and dt, halve the error.

    Without this the tolerances above are arbitrary; with it they are a statement
    about a converging scheme.
    """
    errors = []
    for axial, per_decade in ((30, 8), (60, 16), (120, 32)):
        result = simulate(column, dim=2, axial=axial, lateral=2, per_decade=per_decade)
        rows = {row["factor"]: row for row in compare(column, result)}
        errors.append(rows[1e-2]["rms"])
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.6 < coarse / fine < 2.4, f"rate off: {errors}"


# -- the limits the solver is meant to be robust in --------------------------------------


def test_an_incompressible_fluid_recovers_the_textbook_column():
    """``1/M = 0`` and ``alpha = 1`` give ``beta = 1``: no instant settlement."""
    column = Column(inverse_biot_modulus=0.0, biot=1.0)
    assert column.load_partition == pytest.approx(1.0)
    assert column.initial_pressure == pytest.approx(column.load)  # fluid takes it all
    result = simulate(column, dim=2, **COARSE)
    rows = compare(column, result)
    assert rows[0]["consolidation"] < 0.05  # starts from zero, unlike the default
    for row in rows:
        assert row["rms"] < 5e-2
        assert abs(row["consolidation"] - row["consolidation_exact"]) < 3e-2


def test_a_nearly_incompressible_skeleton_still_solves():
    """``nu -> 1/2``: the mixed form should not notice."""
    column = Column(poisson=0.49999)
    result = simulate(column, dim=2, **COARSE)
    for row in compare(column, result):
        assert np.isfinite(row["rms"])
        assert row["rms"] < 5e-2
        assert row["spread"] < 1e-6
