r"""Terzaghi's consolidation column, 2D and 3D, against the closed form.

The column exists to exercise the boundary conditions -- applied traction,
rollers, a drained face and sealed faces -- on a problem whose answer is known
exactly.  So the tests check the boundary conditions as directly as they check
the numbers:

* the section stays **uniform** (the rollers really do impose uniaxial strain),
* the answer does not depend on the column's **width** or its lateral cell count,
* **2D and 3D agree**, which they must, having the same 1D solution,
* the error **converges** at first order, so what remains is discretization.

Both cell families are run: ``cart`` (quadrilaterals, hexahedra) and ``simplex``
(triangles, tetrahedra).  They are not two spellings of one test.  On simplices
the mimetic stabilization vanishes, so ``M = M1`` alone; and because every cell
is split the same way, the mesh is not laterally symmetric, so the column is only
one-dimensional in the limit.  ``cart`` is uniform across its section to machine
precision and ``simplex`` is not -- the two therefore carry different tolerances,
with the simplex asymmetry pinned by a *convergence* test rather than a bound.

The closed form is itself checked against an independent route -- the settlement
series is verified by integrating the pressure series -- so a mistake in the
analytic side cannot quietly define the answer.
"""

import numpy as np
import pytest

from benchmarks.poroelasticity.consolidation_soil import (
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
    """One run per (family, dim), shared by every test -- these are the slow part."""
    return {
        (family, dim): simulate(
            column, dim=dim, family=family,
            **(COARSE if dim == 2 else dict(axial=20, lateral=2, per_decade=8)),
        )
        for family in ("cart", "simplex")
        for dim in (2, 3)
    }


#: ``cart`` is exactly one-dimensional; ``simplex`` is not.  Splitting every cell
#: the same way breaks the column's left-right symmetry, so a triangulated or
#: tetrahedralised column has a genuine lateral variation that converges away
#: under refinement rather than being zero.  Two tolerances, one reason.
UNIFORM = {"cart": 1e-6, "simplex": 2e-1}

#: Below this the drained boundary layer (thickness ~2 sqrt(T)) is thinner than a
#: cell, and the unstabilised Biot discretisation rings.  See
#: ``test_the_early_time_oscillation_is_confined_to_the_drained_face``.
RESOLVED = 1e-3


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


FAMILIES = [(f, d) for f in ("cart", "simplex") for d in (2, 3)]


@pytest.mark.parametrize("family,dim", FAMILIES)
def test_the_pressure_profile_matches_once_the_layer_is_resolved(
    column, solved, family, dim
):
    """Both families track the closed form wherever the mesh can see the front."""
    for row in compare(column, solved[(family, dim)]):
        if row["factor"] < RESOLVED:
            continue
        assert row["rms"] < 5e-2, f"{family} {dim}D T={row['factor']:g}: {row['rms']:.2e}"
        assert row["max"] < 8e-2, f"{family} {dim}D T={row['factor']:g}: {row['max']:.2e}"


@pytest.mark.parametrize("family,dim", FAMILIES)
def test_the_consolidation_curve_matches(column, solved, family, dim):
    """Settlement is an integral of the pressure, so the early-time ringing --
    which is local to a few cells and changes sign -- largely cancels out of it.
    It is checked at *every* time factor, including the unresolved ones."""
    for row in compare(column, solved[(family, dim)]):
        assert abs(row["consolidation"] - row["consolidation_exact"]) < 3e-2


@pytest.mark.parametrize("family,dim", FAMILIES)
def test_the_cross_section_is_uniform_to_the_family_s_limit(
    column, solved, family, dim
):
    """The rollers must impose uniaxial strain.

    On ``cart`` that is exact -- any lateral variation at all is a broken
    boundary condition.  On ``simplex`` the mesh itself is not laterally
    symmetric, so the bound is loose here and
    :func:`test_the_simplex_section_converges_to_uniform` is what pins it down.
    """
    for row in compare(column, solved[(family, dim)]):
        assert row["spread"] < UNIFORM[family], (
            f"{family} {dim}D T={row['factor']:g}: spread {row['spread']:.2e}"
        )


@pytest.mark.parametrize("family", ["cart", "simplex"])
def test_two_and_three_dimensions_agree(column, solved, family):
    """A 3D box under uniaxial strain has no freedom a 2D one lacks."""
    two = compare(column, solved[(family, 2)])
    three = compare(column, solved[(family, 3)])
    for a, b in zip(two, three):
        assert a["factor"] == b["factor"]
        assert abs(a["consolidation"] - b["consolidation"]) < 5e-3


@pytest.mark.parametrize("family,dim", FAMILIES)
def test_the_column_fully_consolidates(column, solved, family, dim):
    last = compare(column, solved[(family, dim)])[-1]
    assert last["factor"] == 1.0
    assert last["consolidation"] > 0.9


# -- the answer must not depend on things it cannot depend on ---------------------------


def test_the_width_does_not_matter(column):
    """A 1D problem cannot know how wide the column is."""
    narrow, wide = Column(width=0.1), Column(width=25.0)
    a = compare(narrow, simulate(narrow, dim=2, **COARSE))
    b = compare(wide, simulate(wide, dim=2, **COARSE))
    for one, other in zip(a, b):
        assert one["consolidation"] == pytest.approx(other["consolidation"], abs=1e-9)


def test_the_lateral_cell_count_does_not_matter_on_cart(column):
    """On quadrilaterals the column is exactly 1D, so extra columns change nothing."""
    one = compare(column, simulate(column, dim=2, axial=30, lateral=1, per_decade=8))
    five = compare(column, simulate(column, dim=2, axial=30, lateral=5, per_decade=8))
    for a, b in zip(one, five):
        assert a["consolidation"] == pytest.approx(b["consolidation"], abs=1e-9)
        assert a["rms"] == pytest.approx(b["rms"], abs=1e-9)


def test_the_simplex_section_converges_to_uniform(column):
    """The triangulated column is 1D only in the limit -- but it does get there.

    Every quad is split the same way, which breaks left-right symmetry, so the
    section varies with the lateral count.  That spread must shrink under lateral
    refinement -- otherwise the asymmetry is a bug rather than a discretisation
    error.  The settlement, being axial, must instead stay put.
    """
    spreads, settlements = [], []
    for lateral in (2, 4, 8):
        result = simulate(column, dim=2, axial=60, lateral=lateral,
                          per_decade=12, family="simplex")
        row = {x["factor"]: x for x in compare(column, result)}[1e-4]
        spreads.append(row["spread"])
        settlements.append(abs(row["consolidation"] - row["consolidation_exact"]))
    # The spread is the mesh asymmetry and must shrink -- over the range, not term
    # by term: its sign depends on how the diagonals meet the boundary, so it is
    # not monotone in the column count (measured 0.0176, 0.0197, 0.0155).
    assert spreads[-1] < spreads[0], spreads

    # The settlement is an *axial* quantity, fixed by axial=60 and the time steps.
    # Lateral refinement must leave it alone, and does: it sits on the axial error
    # floor at ~5e-4, varying by a few percent (measured 5.02, 5.13, 5.31 e-4).
    # Requiring it to converge laterally would be requiring the wrong thing.
    assert max(settlements) < 1.3 * min(settlements), settlements
    assert max(settlements) < 5e-3, settlements


def test_the_early_time_oscillation_is_confined_to_the_drained_face(column):
    """A known limitation, pinned so it cannot spread unnoticed.

    Below ``T ~ (dz/L)^2`` the drained boundary layer is thinner than a cell and
    the unstabilised Biot system rings -- the Vermeer-Verruijt condition
    ``dt >= h^2/(6 c_v)``, a *lower* bound on the step.  What matters is that the
    ringing stays next to the drained face; the rest of the column must be clean.
    """
    result = simulate(column, dim=2, **COARSE)
    layers = result["profiles"][1e-5]
    elevation, pressure = layers[:, 0], layers[:, 1] / column.initial_pressure
    interior = elevation < 0.9
    assert np.allclose(pressure[interior], 1.0, atol=1e-3), "ringing reached the interior"
    assert np.abs(pressure[~interior] - 1.0).max() > 1e-4  # it is there, near the face


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
