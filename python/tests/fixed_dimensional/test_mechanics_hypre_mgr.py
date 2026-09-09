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

Only stabilized_vem is tested. The weak family takes MGR's two-level reduction,
and that path does not converge -- see the note on mgr_vem_split in
linear_solver/hypre.hpp.
"""

import pytest

from test_mechanics_hypre_ads import (  # noqa: F401  -- the fixtures are shared
    CELLS, JUMPS, MU, POISSON, STRONG_LADDER, _count, _hypre, fixed,
    lame_checkerboard, mesh_of, strong,
)


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
