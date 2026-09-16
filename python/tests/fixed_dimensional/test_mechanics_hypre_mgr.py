"""MGR on the strongly-symmetric stress, importing the ADS file's fixtures.

SEPARATE FILE, SEPARATE PROCESS. MGR's AMG F-relaxation allocates a hierarchy
per solve, and after the ~50 solves the ADS file runs the two together abort the
interpreter mid-suite. Split, each is a ctest executable of its own and neither
accumulates the other's state.

MGR does not precondition the stress block: it eliminates it, F = sigma and
C = u, leaving a displacement operator for BoomerAMG. It carries none of ADS's
auxiliary hierarchies and is several times faster a cycle, so the question is
which of the three properties survive the reduction.

ALL THREE DO, once the F block is solved rather than smoothed. The solver picks
AMG there wherever a facet carries more than one moment: a scalar flux block is
a mass matrix and Jacobi damps it, a stress block is not.

Both families are tested. The weak one takes MGR's two-level reduction, which
converges only with the absolute-row-sum interpolation the solver now selects
for it -- under hypre's default signed point diagonal the level-1 F diagonal
changes sign in 3D and BoomerAMG is handed an indefinite coarse operator. See
mgr_interp_type in linear_solver/hypre.hpp.
"""

import pytest

from test_mechanics_hypre_ads import (  # noqa: F401  -- the fixtures are shared
    CELLS, JUMPS, MU, POISSON, S, STRONG_LADDER, _count, _hypre, fixed,
    lame_checkerboard, mesh_of, patch, strong, F,
)


def weak(mesh, lame=1.0, lame_field=None):
    """sigma NOT symmetric, symmetry imposed by the rotation multiplier: d
    copies of one scalar H(div) space, a copy a tensor row."""
    return patch(mesh, S.stabilized_bdm, lame, lame_field, F.weak_symmetry)


def weak_deviatoric(mesh, lame=1.0, lame_field=None):
    """The same with the total pressure carried as a fourth field, which sets
    the compliance's trace coefficient to zero: A is then EXACTLY block diagonal
    by tensor row -- measured max |A| across rows 0.0 against a/(2mu) -- and
    cond(A) is flat in lambda, 1.036e1 at lambda = 1, 1e4 and 1e8 against
    1.5e1 to 6.0e8 for the three-field form."""
    return patch(mesh, S.stabilized_bdm, lame, lame_field, F.weak_symmetry_deviatoric)


WEAK = {"three-field": weak, "four-field": weak_deviatoric}


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_mgr_is_h_robust(cells):
    """The same relative bound the ADS ladder uses, and for the same reason:

        hexahedra    18 23 26
        prisms       27 33 37
        tetrahedra   36 34 35

    The worst ratio is 1.44. These are with AMG on the F block, which the
    solver selects wherever a facet carries more than one moment; under Jacobi
    the tetrahedral column is 74 63 58 -- twice as long for the same work.
    """
    _hypre()
    counts = []
    for n in STRONG_LADDER:
        mesh = mesh_of(cells, n)
        model = strong(mesh)
        counts.append(_count(model, mesh, "mgr"))
        print(f"  {cells:11s} mgr {model.n_cells:6d} cells "
              f"{model.n_dofs:8d} dofs   {counts[-1]:4d} its")
    assert max(counts) <= 1.5 * min(counts)


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_mgr_does_not_track_the_contrast(cells):
    """lambda jumps eight orders of magnitude and the count moves by at most 10
    percent off the jumped baseline:

        hexahedra    18 | 19 19 19 19
        prisms       27 | 28 28 28 28
        tetrahedra   36 | 37 37 37 37

    The first column is the uniform material and the rest the checkerboard; the
    step between them is the mesh seeing two materials, not the size of the
    jump. Both are bounded separately, spread 4 and factor 2.
    """
    _hypre()
    mesh = fixed(cells)
    counts = []
    for p in JUMPS:
        field = lame_checkerboard(mesh, 1.0, p)
        counts.append(_count(strong(mesh, lame_field=field), mesh, "mgr"))
        print(f"  {cells:11s} mgr  lambda_in = 1e{p:+03d}   {counts[-1]:4d} its")
    uniform, jumped = counts[0], counts[1:]
    assert max(jumped) <= min(jumped) + 4
    assert max(jumped) <= 2 * uniform


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_mgr_does_not_track_the_incompressibility(cells):
    """The property the F-relaxation buys, and every family has it:

        hexahedra    18 19 19 19 19
        prisms       27 29 29 32 34
        tetrahedra   36 38 46 52 60

    lambda moves through four orders of magnitude and the count at most
    doubles. Under Jacobi the same sweep runs 74, 92, 204, 542, 1648 on
    tetrahedra -- twenty-two fold, and growing like sqrt(2 mu + d lam).

    WHY IT IS THE F BLOCK. The compliance is diagonal on the
    Frobenius-orthonormal modes: five deviatoric eigenvalues at 1/2mu and one
    hydrostatic at 1/(2mu + d lam), which vanishes as lam -> infinity. Jacobi
    cannot damp a mode that small. The coarse block never sees it -- div of a
    constant stress is zero, so the hydrostatic direction lies in ker B, hence
    orthogonal to range(B^T), and never enters S = -B M^-1 B^T. A direct F-solve
    is flat outright: 22, 23, 23, 23, 23.
    """
    _hypre()
    mesh = fixed(cells)
    counts = []
    for nu in POISSON:
        lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)
        counts.append(_count(strong(mesh, lame=lam), mesh, "mgr"))
        print(f"  {cells:11s} nu = {nu:<7}  lambda = {lam:10.1f}   {counts[-1]:4d} its")
    assert max(counts) <= 3 * min(counts)


# ---- the weak family -------------------------------------------------------
#
# sigma is d copies of ONE scalar H(div) space, a copy a tensor row: div is
# exactly row-block-diagonal, and the compliance couples the rows only through
# the trace, by lam/(2mu + d lam) <= 1/d in the three-field form and by nothing
# at all in the four-field one.
#
# That duplication is also why the path needs its own interpolation. A facet
# carries d moments a row and d^2 in all, so it holds 6 higher moments in 3D
# against 2 in 2D; A_00's off-diagonal row sums over |diagonal| then average
# 0.86 in 2D and 2.55 in 3D, every 3D row above one. Under hypre's default
# SIGNED point diagonal the level-1 F diagonal goes negative on 31 percent of
# the facet constants and BoomerAMG is handed an indefinite coarse operator.
# mgr_interp_type selects the absolute row sum here and the point diagonal on
# the strong family, which wants the opposite -- see linear_solver/hypre.hpp.


@pytest.mark.parametrize("form", sorted(WEAK))
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_weak_symmetry_mgr_is_h_robust(form, cells):
    """The same relative bound the strong ladder uses:

        three-field   21 25 29   31 35 36   89 76 63
        four-field    19 22 24   27 30 31   72 65 54
                      hexahedra   prisms   tetrahedra

    The worst ratio is 1.41, and the tetrahedral column FALLS with refinement:
    its coarse end is the expensive one, as it is for ADS.
    """
    _hypre()
    counts = []
    for n in STRONG_LADDER:
        mesh = mesh_of(cells, n)
        model = WEAK[form](mesh)
        counts.append(_count(model, mesh, "mgr"))
        print(f"  {cells:11s} {form:11s} {model.n_cells:6d} cells "
              f"{model.n_dofs:8d} dofs   {counts[-1]:4d} its")
    assert max(counts) <= 1.5 * min(counts)


@pytest.mark.parametrize("form", sorted(WEAK))
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_weak_symmetry_mgr_does_not_track_the_contrast(form, cells):
    """lambda jumps eight orders of magnitude and the count stops moving after
    the first step, exactly as on the strong family:

        three-field   21 | 26 26 26 26   31 | 40 41 41 41   89 | 98 99 99 99
        four-field    19 | 23 23 23 23   27 | 30 31 31 31   72 | 75 75 75 75
                          hexahedra          prisms            tetrahedra

    The first column is the uniform material; the step to the checkerboard is
    the mesh seeing two materials at all, not the size of the jump.
    """
    _hypre()
    mesh = fixed(cells)
    counts = []
    for p in JUMPS:
        field = lame_checkerboard(mesh, 1.0, p)
        counts.append(_count(WEAK[form](mesh, lame_field=field), mesh, "mgr"))
        print(f"  {cells:11s} {form:11s} lambda_in = 1e{p:+03d}   {counts[-1]:4d} its")
    uniform, jumped = counts[0], counts[1:]
    assert max(jumped) <= min(jumped) + 4
    assert max(jumped) <= 2 * uniform


# THE ONE PROPERTY THE WEAK FAMILY DOES NOT HAVE, AND ONLY ON SIMPLICES.
#
#     nu = 0.25, 0.4, 0.49, 0.499, 0.4999
#     three-field   21  24  33  38   42     31 37  50  62   67    89 121 262 433 1079
#     four-field    19  21  27  30   33     27 29  35  39   43    72  76 112 237  360
#                        hexahedra                prisms              tetrahedra
#
# Hexahedra and prisms hold the strong family's 3x bound in both forms. The
# tetrahedral column does not: 12 fold in the three-field form and 5 in the
# four-field one, so the total pressure recovers more than half the growth but
# not the property. The bounds below are the measured behaviour, asserted per
# family so a regression is caught and an improvement is visible rather than
# hidden under one loose cap.
NU_BOUND = {
    ("hexahedra", "three-field"): 3.0, ("hexahedra", "four-field"): 3.0,
    ("prisms", "three-field"): 3.0, ("prisms", "four-field"): 3.0,
    ("tetrahedra", "three-field"): 14.0, ("tetrahedra", "four-field"): 6.0,
}


@pytest.mark.parametrize("form", sorted(WEAK))
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_weak_symmetry_mgr_incompressibility(form, cells):
    """h- and contrast-robust everywhere; nu-robust on hexahedra and prisms and
    NOT on tetrahedra. See the table above for the measured counts."""
    _hypre()
    mesh = fixed(cells)
    counts = []
    for nu in POISSON:
        lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)
        counts.append(_count(WEAK[form](mesh, lame=lam), mesh, "mgr"))
        print(f"  {cells:11s} {form:11s} nu = {nu:<7} lambda = {lam:10.1f}   "
              f"{counts[-1]:4d} its")
    assert max(counts) <= NU_BOUND[(cells, form)] * min(counts)
