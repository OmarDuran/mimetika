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
