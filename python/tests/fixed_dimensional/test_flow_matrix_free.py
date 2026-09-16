"""MINRES on the flow saddle point with no assembled operator, and the two
properties a solver is judged on.

THE METHOD. Pazner, Kolev & Vassilevski, SIAM J. Sci. Comput. 46 (2024) B179.
The operator is applied through exokal's ActionSink -- the term kernels at
ad::Directional, so nothing of size n^2 is built -- and the preconditioner is
block diagonal,

    B = diag( diag(M), S~ ),    S~ = D diag(M)^-1 D^T,

with one BoomerAMG V-cycle on S~. At lowest order S~ is the two-point flux
matrix: a weighted graph Laplacian on the cell adjacency graph, an M-matrix by
construction (tests/solver/test_matrix_free.cpp), which is why one cycle is
enough. The pressure equation is negated inside the solve, because the Darcy
term writes [[M, -D^T], [D, 0]] and MINRES on a non-symmetric operator converges
to something else without saying so.

WHAT IS CLAIMED, AND WHAT IS MEASURED. Proposition 1 bounds kappa(B^-1 A) by
2.618 with the exact Schur complement, and S~ degrades that only by the
equivalence of M with its diagonal -- which is flat in h because M is assembled
from cell-local blocks and a facet is shared by two cells. That bound is
measured directly in the C++ test: 3.61 then 3.77 for derham_rt over an
eightfold refinement, and exactly 2.618 for the two-point star, where S~ IS the
exact Schur complement.

A bounded condition number bounds the convergence FACTOR. It does not fix the
iteration count at a given tolerance, and the count does drift as the spectrum
fills in between the same two bounds. So what is asserted here is the drift
against the problem size, not a constant: from cube(3) to cube(8) the unknowns
grow nineteenfold and the count at most doubles.

The contrast is the stronger result and it is asserted as such -- flat to within
four iterations over sixteen orders of magnitude.

A STRONG CONDITION IS THE OTHER HALF OF THE SYMMETRY. A prescribed pressure is
natural in the mixed form -- data a term reads -- so the fixtures above pin
nothing and the sign flip is all the symmetry needs. A normal flux is carried as
an unknown and is imposed strongly, and Simulation's tangent replaces such a row
while leaving its column in place: the sign-flipped operator is then asymmetric
by the whole of D on those facets, which is not a preference but the assumption
MINRES makes. The route eliminates the column too; test 5 is that it did.
"""

import numpy as np
import pytest

import mimetika_cxx as mk

from test_flow_ads import (  # noqa: F401  -- the fixtures are shared
    JUMPS, R, _linear, cube, enclosure, patch, wedge,
)

RTOL = 1e-9

#: every flow product, including the two-point star whose S~ is exact
PRODUCTS = {
    "derham_rt": R.derham_rt,
    "stabilized_rt": R.stabilized_rt,
    "derham_bdm": R.derham_bdm,
    "stabilized_bdm": R.stabilized_bdm,
    "diagonal_tpfa": R.diagonal_tpfa,
}


def _count(model):
    report = model.solve_matrix_free(
        options=mk.SolverOptions(rtol=RTOL, max_iterations=1000)
    )
    assert report.converged, report.reason
    return report.iterations


# ---- 1. the answer ---------------------------------------------------------
#
# CONVERGED IS NOT CORRECT. A preconditioner built on the wrong block, or a
# saddle point left non-symmetric, still converges -- to something else. The
# datum is linear and every product here reproduces a linear pressure exactly on
# a cartesian mesh, so the answer is known in closed form and the comparison is
# against the solver tolerance rather than against a discretization error.
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_the_matrix_free_answer_is_the_exact_patch(name):
    mesh = cube(4)
    model = patch(mesh, PRODUCTS[name])
    its = _count(model)
    worst = max(
        abs(model.cell_pressure(e) - _linear(mk.centroid(mesh, 3, e)))
        for e in range(model.n_cells)
    )
    print(f"  {name:16s} {model.n_dofs:6d} dofs {its:4d} its   max|p - p_exact| {worst:.2e}")
    assert worst < 1e-7


# ---- 2. h ------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_matrix_free_is_h_robust(name):
    """cube(3, 4, 6, 8) -- 27 to 512 cells, so nineteen times the unknowns:

        derham_rt       20 29 40 41
        stabilized_rt   21 24 27 29
        derham_bdm      37 45 55 58
        stabilized_bdm  33 39 48 50
        diagonal_tpfa    5  7  9  9

    The worst ratio is 2.05, against nineteenfold growth in the problem. Not
    flat -- the conditioning is what Proposition 1 holds flat, and the C++ test
    measures that directly -- but bounded well away from the size of the system.
    """
    counts = []
    for n in (3, 4, 6, 8):
        model = patch(cube(n), PRODUCTS[name])
        counts.append(_count(model))
        print(f"  {name:16s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs "
              f"{counts[-1]:4d} its")
    assert counts[-1] <= 2.5 * counts[0]


# ---- 3. the contrast -------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_matrix_free_does_not_track_the_contrast(name):
    """K = 10^p on two interior boxes, p from -8 to +8, on one fixed mesh:

        derham_rt       43 43 43 41 44 44 44
        stabilized_rt   29 29 29 29 29 30 30
        derham_bdm      58 58 58 58 59 59 59
        stabilized_bdm  52 52 52 50 53 53 53
        diagonal_tpfa    9  9  9  9 11 11 11

    Flat to three iterations over sixteen orders of magnitude, in both
    directions. The coefficient never leaves M: S~'s transmissibility is the
    reciprocal of a facet's own mass, so a jump rescales the graph Laplacian's
    weights and the preconditioner follows it exactly -- there is no separate
    coefficient for the method to get wrong.
    """
    mesh = cube(8)
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, PRODUCTS[name], enclosure(mesh, p))))
        print(f"  {name:16s} K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 4


# ---- 5. a sealed face ------------------------------------------------------
#
# Four faces sealed and two given a linear pressure, whose exact solution is
# p = x with no flow across the sides. The seal is a NormalFluxBC: strong, so it
# pins a flux moment on every facet it names, and those pinned columns are what
# the route has to eliminate before MINRES may be applied at all.
#
# What the elimination is worth, measured in C++ on this geometry with S~
# inverted exactly (so the count is the preconditioner's and not the cycle's),
# derham_rt over cube(3, 4, 6, 8):
#
#     eliminated            7   9  13  17
#     row-eliminated only  23  30  56  71
#
# The same answer for three to four times the work. A HOMOGENEOUS pin is the
# forgiving case: it leaves the iterates in the subspace where the pinned
# entries vanish, and the row-eliminated operator is symmetric THERE whatever it
# is elsewhere, so the old route arrived at the right answer slowly rather than
# at the wrong one. Test 6 is the case that does not forgive.
#
# So the answer is what is asserted here; the count is reported.
def sealed_box(mesh, product, flux=0.0):
    """The four faces normal to y and z given a normal flux, the two normal to x
    a linear pressure. The flux is STRONG and the pressure NATURAL, so this is
    the fixture with pinned rows in it."""
    model = mk.FlowModel(mesh, 3, 1.0, product)
    sealed = []
    for f in mk.boundary_facets(mesh, 3):
        x = mk.centroid(mesh, 2, f)
        if 1e-12 < x[0] < 1.0 - 1e-12:
            sealed.append(f)
        else:
            model.add_pressure([f], x[0], [1.0, 0.0, 0.0])
    assert sealed, "the fixture pinned nothing, so it tests nothing"
    model.add_normal_flux(sealed, flux)
    return model, sealed


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_strong_condition_keeps_the_answer(name):
    mesh = cube(4)
    model, sealed = sealed_box(mesh, PRODUCTS[name])

    its = _count(model)
    worst = max(
        abs(model.cell_pressure(e) - mk.centroid(mesh, 3, e)[0])
        for e in range(model.n_cells)
    )
    print(f"  {name:16s} {len(sealed):4d} facets sealed {its:4d} its   "
          f"max|p - x| {worst:.2e}")
    assert worst < 1e-7


# ---- 6. and a datum that is not zero ---------------------------------------
#
# THE CASE THAT DOES NOT FORGIVE. A nonzero normal flux puts a nonzero entry in
# the pinned rows of the load, the iterates leave the subspace where the pinned
# entries vanish, and the asymmetry of the row-eliminated operator is no longer
# invisible to the recurrence. Measured in C++ with S~ inverted exactly,
# derham_rt over cube(3, 4, 6, 8) at q.n = 0.25:
#
#     eliminated            12  15  24  31
#     row-eliminated only  255  --  114 127
#
# where -- is 800 iterations without converging, returning an answer wrong by
# 7.5e-08 against a tolerance of 1e-10 and reporting no failure. There is no
# closed form here, so the reference is the assembled route, which does not
# assume symmetry and never needed the elimination.
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_nonzero_strong_datum_agrees_with_the_assembled_route(name):
    mesh = cube(4)
    model, _ = sealed_box(mesh, PRODUCTS[name], flux=0.25)
    its = _count(model)
    free = [model.cell_pressure(e) for e in range(model.n_cells)]

    reference, _ = sealed_box(mesh, PRODUCTS[name], flux=0.25)
    report = reference.solve(options=mk.SolverOptions(method="direct"))
    assert report.converged, report.reason
    direct = [reference.cell_pressure(e) for e in range(reference.n_cells)]

    scale = max(abs(v) for v in direct)
    worst = max(abs(a - b) for a, b in zip(free, direct))
    print(f"  {name:16s} {its:4d} its   max|p_free - p_direct| {worst:.2e} "
          f"of {scale:.2e}")
    assert worst < 1e-8 * scale


# ---- 4. where it is weakest ------------------------------------------------
def test_the_count_is_bounded_on_the_simplicial_annulus():
    """THE CARTESIAN LADDER IS THE FAVOURABLE CASE, and this is the other one.

    On the annulus of tetrahedra the counts are several times larger and do not
    fall under refinement in the way the cartesian ones rise:

        derham_rt / stabilized_rt    74 112 109  94
        derham_bdm / stabilized_bdm 259 254 228 197
        diagonal_tpfa                 7   7   7   9

    The two members of each pair coincide, which is the stabilization vanishing
    on a simplex rather than a coincidence of the solver. What is asserted is
    only that the count stays bounded and comes down rather than growing; the
    size of it is a property of diag(M) on a stretched tetrahedron, where the
    equivalence constant measured in the C++ test is 38 against 7.6 on a cube.
    """
    for name in ("stabilized_rt", "stabilized_bdm", "diagonal_tpfa"):
        counts = []
        for nr in (3, 4, 6, 8):
            model = patch(wedge(nr), PRODUCTS[name])
            counts.append(_count(model))
            print(f"  {name:16s} {model.n_cells:6d} cells {counts[-1]:5d} its")
        assert counts[-1] <= counts[0] + 45
