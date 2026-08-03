r"""The condensation benchmark: the structural claims behind the table.

Timing exactness is a property of the machine, so the assertions here are the
ones that cannot legitimately vary: the sizes, the reduction, and a *loose*
ordering of the timings (the measured gap is ~50x; asserting 3x leaves room
for any amount of CI noise without ever passing a broken condensation).
"""

import numpy as np
import pytest

from benchmarks.elasticity.condensation import measure

N = 5


@pytest.fixture(scope="module")
def rows():
    return {r["name"]: r for r in measure(n=N, repeats=1)}


def test_the_sizes_are_what_the_formulations_say(rows):
    cells = N**3
    facets = 3 * N * N * (N + 1)
    assert rows["AFW 3-field (full)"]["dofs"] == 9 * facets + 6 * cells
    assert rows["AFW 4-field (full)"]["dofs"] == 9 * facets + 7 * cells
    assert rows["lumped 3-field (full)"]["dofs"] == 3 * facets + 6 * cells
    for name in ("lumped 4-field (condensed)", "two-point 4-field (condensed)"):
        assert rows[name]["dofs"] == 3 * facets + 7 * cells
        # the factorised system is cell-centred: (1 + d + nsk) per cell
        assert rows[name]["reduced"] == 7 * cells


def test_the_condensed_forms_factorise_dramatically_less(rows):
    afw = rows["AFW 4-field (full)"]
    for name in ("lumped 4-field (condensed)", "two-point 4-field (condensed)"):
        assert rows[name]["fill"] * 5 < afw["fill"]
        assert rows[name]["time"] * 3 < afw["time"]


def test_the_three_field_lumped_cannot_be_condensed(rows):
    """It runs full -- the fold-in re-couples the facets (see test_condense)."""
    assert rows["lumped 3-field (full)"]["reduced"] is None


def test_every_formulation_solved_its_system(rows):
    for r in rows.values():
        assert np.all(np.isfinite(r["solution"]))
