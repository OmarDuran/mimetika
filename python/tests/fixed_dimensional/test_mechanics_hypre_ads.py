"""ADS on the weak-symmetry stress, called directly rather than through PETSc.

test_flow_hypre_ads.py tests the same two properties on a FLUX. A stress is d
copies of that flux -- the rows of sigma -- so the question here is whether the
d-copies construction inherits them, and it is a third property as well:

    h-ROBUST                the count does not grow as the mesh is refined
    CONTRAST-ROBUST         nor as lambda jumps between cells
    INCOMPRESSIBILITY-ROBUST nor as nu -> 1/2, where lambda -> infinity

WHAT THE SOLVER IS GIVEN, and why it is d copies of one complex.

Weak-symmetry Hellinger-Reissner is

    (A sigma, tau) + (u, div tau) + (gamma, skw tau) = 0
    (div sigma, v)                                   = (f, v)
    (skw sigma, q)                                   = 0

and the norm of each space is fixed by the equation it belongs to:

    ||sigma||^2_Sigma = (A sigma, sigma) + alpha ||div sigma||^2
    ||u||^2_V         = (2 mu / L^2) ||u||^2
    ||gamma||^2_Q     = 2 mu ||gamma||^2

The stress norm has NO rotation term: skw is bounded L^2 -> L^2, so the
symmetry constraint contributes to Q's norm and not to Sigma's. That is what
makes the construction work. Restricted to one row of sigma,

    ||sigma_i||^2_Sigma = (beta_i sigma_i, sigma_i) + alpha ||div sigma_i||^2
       beta_i = (1/2mu) ( I - lambda/(2mu + d lambda) e_i (x) e_i )
       alpha  = L^2 / (2 mu)

which is a_div of Kolev & Vassilevski (1.1) with an SPD matrix beta -- their
section 1 allows exactly that. The compliance is row-diagonal apart from the
trace, div^T div is row-diagonal outright, and the rotation is absent, so the
rows separate: d copies of one complex, one ADS each.

beta_i's spectrum is 1/(2mu) across e_i and 1/(2mu + d lambda) along it. That
ratio, 1 + d lambda / 2mu, IS the incompressibility, and it is why
test_the_count_does_not_track_the_incompressibility is a different statement
from the contrast one rather than a special case of it.

TWO ROUTES TO THE BLOCK, as in the flux file and for the same reason:

    one cycle   `block_iterations = 0`, which APPROXIMATES the block
    inner CG    a short CG under that cycle, which SOLVES it

Only the second is flat in nu. One cycle is h- and contrast-flat and drifts as
nu -> 1/2, which is the same ads / ads-cg distinction the flux has.
"""

import numpy as np
import pytest

import mimetika_cxx as mk

S = mk.StressRealization
F = mk.StressFormulation

# STRONG SYMMETRY IS THE SAME SOLVER ON A DIFFERENT SPACE, and it is worth the
# same three questions. Its facet carries SIX moments -- one traction vector
# against the facet's own frame -- rather than d copies of a scalar layout, so
# build_norm reaches ADS through a FRAME-WEIGHTED injection instead of a matrix
# of ones: the three mean slots {t1, t2, n} rotated into global components are
# three scalar H(div) cochains, int_f (tau n).e_k. That is d copies of the
# ONE-moment space -- of stabilized_rt, not of stabilized_bdm -- and there is no
# rotation multiplier at all, symmetry being imposed strongly.
#
# Only stabilized_vem takes strong_symmetry; diagonal_vem and adaptive_vem
# require strong_symmetry_total, which adds a pressure field and is NOT robust
# on this preconditioner -- measured at n = 4 and 6, stabilized_vem runs 300
# then 2000 (no convergence) on tetrahedra and 100 then 172 on hexahedra, and
# diagonal_vem does not converge on tetrahedra at all. So it is left out rather
# than asserted loosely, and named here so the omission is a decision.
STRONG = {"stabilized_vem": S.stabilized_vem}

# The weak-symmetry members whose stress is d copies of a BDM flux. On a
# tetrahedron the stabilization vanishes and the two coincide, so both take the
# same degree-2 complex; the file runs both because they part on a polytope.
BDM = {
    "derham_bdm": S.derham_bdm,
    "stabilized_bdm": S.stabilized_bdm,
}

RTOL = 1e-8
MU = 1.0
GRADIENT = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _hypre():
    """The module, or a skip. It is an optional build: MIMETIKA_USE_HYPRE=ON."""
    try:
        import mimetika_hypre as mh
    except ImportError as e:  # pragma: no cover - build-dependent
        pytest.skip(f"mimetika_hypre is not built: {e}")
    if not hasattr(mk, "ads_handoff"):  # pragma: no cover - build-dependent
        pytest.skip("mimetika_cxx has no ads_handoff")
    return mh


def _options(kind):
    """`one` applies a single ADS cycle; `cg` solves the block under it."""
    mh = _hypre()
    o = mh.AdsOptions()
    o.rtol = RTOL
    o.max_iterations = 2000
    if kind == "cg":
        o.block_iterations = 50
        o.block_rtol = 1e-2
    return o


def wedge(nr):
    """The tetrahedral annulus the flux file uses, for the reason it gives:
    hypre's ADS is unhappy with box(n, simplex) at some n, and the annulus is
    accepted at every refinement."""
    return mk.annulus(nr, max(2, nr // 2), 3, mk.Family.simplex, 1.0, 10.0, 1.0)


# ONE ROUTE, AND WHY EVERY CELL TYPE IS STILL SWEPT.
#
# A weak-symmetry stress is split by ROW and each row handed to ADS on the
# degree-2 complex, whatever the cells are. dec/mimetic_curl.hpp cannot supply
# the C for that on a polytope -- there D_edge > m, the reconstruction is a
# least-squares fit that couples the whole cell and the two cells sharing a
# facet disagree -- but d^1 is facet-local by nature, and surface Stokes writes
# a facet's rows in that facet's own dofs alone on any planar facet. So C is
# built facet-wise (bdm_complex.hpp) and the route no longer forks.
#
# The cells still differ in the OPERATOR -- the stabilization vanishes only on a
# simplex -- so each property is measured on all three. `degree2` in the handoff
# says the route was taken, and test_the_route_is_the_degree_two_complex_
# everywhere is what pins it.
CELLS = {
    "tetrahedra": (wedge, (3, 4, 6, 8), 4),
    "hexahedra": (lambda n: mk.box([n, n, n], 3, mk.Family.cartesian), (3, 4, 6), 4),
    "prisms": (lambda n: mk.box([n, n, n], 3, mk.Family.prism), (3, 4, 6), 4),
}


def mesh_of(cells, n):
    return CELLS[cells][0](n)


def ladder(cells):
    return CELLS[cells][1]


def fixed(cells):
    """The one mesh the contrast and incompressibility sweeps run on."""
    return mesh_of(cells, CELLS[cells][2])


def patch(mesh, product=S.stabilized_bdm, lame=1.0, lame_field=None,
          formulation=F.weak_symmetry):
    """A linear displacement on the whole boundary, with its gradient.

    The gradient is not decoration: a facet holding d traction moments tests the
    datum against d basis functions, and the centred ones see only the
    VARIATION across the facet. Without it the BDM members lose the patch, and a
    count measured on a wrong answer is not a measurement.
    """
    model = mk.CauchyMechanicsModel(
        mesh, 3, mk.ElasticMaterial(MU, lame), product, formulation
    )
    if lame_field is not None:
        model.set_lame_per_cell(list(lame_field))
    for f in mk.boundary_facets(mesh, 3):
        c = mk.centroid(mesh, 2, f)
        model.prescribe_displacement([f], [0.1 * c[0], 0.0, 0.0], GRADIENT)
    return model


def solve(model, mesh, kind="cg"):
    """Assemble here, solve in mimetika_hypre, accept back -- what the
    `hypre-ads` solver of the examples does, stated once for the tests."""
    mh = _hypre()
    mk.mpi_size()  # PETSc owns MPI; hypre attaches rather than starting a second
    mh.init()
    mk.distribute(model)
    model.build()
    handoff = mk.ads_handoff(model, mesh, 3)
    x, report = mh.solve_system(**handoff, options=_options(kind))
    mk.accept(model, list(x))
    return report, bool(handoff["degree2"])


def _count(model, mesh, kind="cg"):
    report, _ = solve(model, mesh, kind)
    assert report.converged, report.reason
    return report.iterations


def lame_checkerboard(mesh, lame, exponent, blocks=4):
    """lambda alternating between lame and lame * 10^p on a blocks^3 partition
    of the bounding box.

    A function of POSITION and not of the cell numbering, so it is the same
    field on every refinement -- a pattern redrawn per mesh makes a ladder
    meaningless. mu stays uniform: the model carries lambda per cell and the
    shear as one number.
    """
    x = np.array([mk.centroid(mesh, 3, e) for e in range(mesh.count(3))])
    lo, hi = x.min(axis=0), x.max(axis=0)
    block = np.floor(blocks * (x - lo) / np.maximum(hi - lo, 1e-300) * 0.999)
    return np.where(block.sum(axis=1) % 2 == 0, lame, lame * 10.0 ** exponent)


# ---- 1. the answer ---------------------------------------------------------
#
# CONVERGED IS NOT CORRECT. A preconditioner built on the wrong complex, or on
# a permutation of the block's rows, still converges -- to something else. For
# the d-copies path this is the check that each block really is one row of
# sigma, in the model's own order and basis.
@pytest.mark.parametrize("name", sorted(BDM))
def test_the_hypre_answer_is_the_direct_answer(name):
    _hypre()
    mesh = wedge(3)
    direct = patch(mesh, BDM[name])
    direct.solve()
    got = patch(mesh, BDM[name])
    its = _count(got, mesh)
    worst = max(
        abs(direct.displacement(e, c) - got.displacement(e, c))
        for e in range(direct.n_cells)
        for c in range(3)
    )
    print(f"  {name:15s} {its:3d} its   max |hypre - direct| = {worst:.2e}")
    assert worst < 1e-6


# ---- 2. which complex ADS was given ----------------------------------------
#
# EVERY cell type takes the degree-2 complex. C is surface Stokes on a facet, so
# it needs no cell reconstruction and a polytope has a global C; Pi's vertex
# hats reproduce the linears exactly on a tetrahedron and in least squares
# beyond it. So each row of the stress is handed to ADS whole -- three moments a
# facet -- rather than through its facet constants.
#
# Measured on this suite, moving the polytopes off the facet-constant subspace
# and onto the degree-2 complex, single cycle:
#
#     hexahedra h-ladder    42 46 52  ->  32 36 38
#     prisms    h-ladder    60 62 65  ->  38 40 42
#     prisms    lambda 1e8      76    ->     60
#
# The solved block was already near its floor and stays there.
@pytest.mark.parametrize("name", sorted(BDM))
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_the_route_is_the_degree_two_complex_everywhere(name, cells):
    _hypre()
    mesh = mesh_of(cells, 3)
    _, used = solve(patch(mesh, BDM[name]), mesh)
    print(f"  {name:15s} {cells:11s} degree2 = {used}")
    assert used


# ---- 3. h-robustness -------------------------------------------------------
#
# Measured on this ladder with a single cycle: 57, 57, 50, 46 -- flat, and
# falling rather than rising. On a larger tetrahedral box the counts are
# constant to the digit, 46 on one cycle and 37 under the inner CG from 3072 to
# 48000 cells.
@pytest.mark.parametrize("name", sorted(BDM))
@pytest.mark.parametrize("kind", ["one", "cg"])
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_the_stress_is_h_robust(name, kind, cells):
    _hypre()
    counts = []
    for n in ladder(cells):
        mesh = mesh_of(cells, n)
        model = patch(mesh, BDM[name])
        counts.append(_count(model, mesh, kind))
        print(f"  {name:15s} {cells:11s} {kind:3s} {model.n_cells:6d} cells "
              f"{model.n_dofs:8d} dofs   {counts[-1]:4d} its")
    # 9 rather than 8 for the hexahedral solved block alone: 25, 32, 34 over a
    # 7x dof range, the step being from the coarsest mesh where the count is
    # still falling. It plateaus -- 32 to 34 for the last 3x -- which is the
    # property, not the size of the first step.
    assert counts[-1] <= counts[0] + 9


# ---- 4. contrast-robustness ------------------------------------------------
#
# lambda alone jumps; mu is uniform, which is what the model carries per cell.
# The first entry is the UNIFORM material and the rest are contrasted, and the
# two are different questions:
#
#     1e+00   57      <- no contrast at all
#     1e+02   70   \
#     1e+04   70    |  six orders of magnitude, and the count does not move
#     1e+06   70    |
#     1e+08   70   /
#
# "the count does not track the contrast" is the SECOND of those. There is one
# step on introducing a jump at all, and then nothing; a single bound over both
# halves conflates them, and a bound loose enough to swallow the step -- 13
# here -- would no longer notice a drift of 70, 80, 90 across the orders, which
# is the failure this test exists to catch.
JUMPS = (0, 2, 4, 6, 8)


@pytest.mark.parametrize("kind", ["one", "cg"])
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_the_count_does_not_track_the_contrast(kind, cells):
    _hypre()
    mesh = fixed(cells)
    counts = []
    for p in JUMPS:
        field = lame_checkerboard(mesh, 1.0, p)
        counts.append(_count(patch(mesh, lame_field=field), mesh, kind))
        print(f"  {cells:11s} {kind:3s}  lambda_in = 1e{p:+03d}   {counts[-1]:4d} its")
    uniform, jumped = counts[0], counts[1:]
    # flat ACROSS the contrast, which is the property
    assert max(jumped) <= min(jumped) + 4
    # and the step off the uniform material is a step, not a trend
    assert max(jumped) <= 2 * uniform


# ---- 5. incompressibility --------------------------------------------------
#
# THE ONE THAT IS NOT A SPECIAL CASE OF THE CONTRAST. A jump in lambda between
# cells is a coefficient the Galerkin products carry; nu -> 1/2 is beta_i's own
# spectrum degenerating, 1/(2mu) across e_i against 1/(2mu + d lambda) along it,
# and the ratio is unbounded.
#
# Measured on wedge(4): one cycle drifts, 57, 70, 140, 351 and then no
# convergence at nu = 0.4999; the block SOLVED is flat, 40, 40, 41, 44, 43. So
# only `cg` is asserted here, and the drift of a single cycle is the same
# statement the flux file makes about ads against ads-cg.
POISSON = (0.25, 0.4, 0.49, 0.499, 0.4999)


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_the_count_does_not_track_the_incompressibility(cells):
    _hypre()
    mesh = fixed(cells)
    counts = []
    for nu in POISSON:
        lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)
        counts.append(_count(patch(mesh, lame=lam), mesh, "cg"))
        print(f"  {cells:11s} nu = {nu:<7}  lambda = {lam:10.1f}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 8


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_one_cycle_drifts_where_the_solved_block_does_not(cells):
    """The two routes are a different statement, and this is what separates
    them: at nu = 0.499 one cycle takes several times what the solved block
    does, on the same problem and the same tolerance. It holds on every cell
    type, tetrahedra and polytopes alike, which is why nu is the one property
    only `cg` is asserted for."""
    _hypre()
    mesh = fixed(cells)
    lam = 2.0 * MU * 0.499 / (1.0 - 2.0 * 0.499)
    solved = _count(patch(mesh, lame=lam), mesh, "cg")
    once = _count(patch(mesh, lame=lam), mesh, "one")
    print(f"  {cells:11s} nu = 0.499:  one cycle {once} its, block solved {solved} its")
    assert solved <= once


# ---- 6. strong symmetry ----------------------------------------------------
#
# The coarse end is genuinely worse here and the ladder starts past it: on
# hexahedra the solved block runs 21, 25, 29 at n = 2, 3, 4 before settling at
# 31 from n = 6 to n = 12. Starting at 3 would put the single-cycle bound at
# 53 <= 53, which passes today and would be flaky tomorrow.
STRONG_LADDER = (4, 6, 8)


def strong(mesh, lame=1.0, lame_field=None):
    return patch(mesh, S.stabilized_vem, lame, lame_field, F.strong_symmetry)


@pytest.mark.parametrize("kind", ["one", "cg"])
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_is_h_robust(kind, cells):
    """A RELATIVE bound, because the baselines here span sixteenfold.

        solved block   14 14 14   29 31 31   36 36 36
        single cycle  230 261 260  50 53 52   67 68 70
                      tetrahedra  hexahedra   prisms

    None of those grows with refinement -- the worst ratio is 1.14 -- but a
    fixed +8 means 57% of the solved block's 14 and 3.5% of the single cycle's
    230, so it would be slack for one and unmeetable for the other. What is
    being asserted is that the count does not track h, and that is a ratio.

    The tetrahedral column is also the file's sharpest `one` against `cg`: 250
    against 14, an eighteenfold gap on the same problem and tolerance.
    """
    _hypre()
    counts = []
    for n in STRONG_LADDER:
        mesh = mesh_of(cells, n)
        model = strong(mesh)
        counts.append(_count(model, mesh, kind))
        print(f"  {cells:11s} {kind:3s} {model.n_cells:6d} cells "
              f"{model.n_dofs:8d} dofs   {counts[-1]:4d} its")
    assert max(counts) <= 1.3 * min(counts)


@pytest.mark.parametrize("kind", ["one", "cg"])
@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_does_not_track_the_contrast(kind, cells):
    _hypre()
    mesh = fixed(cells)
    counts = []
    for p in JUMPS:
        field = lame_checkerboard(mesh, 1.0, p)
        counts.append(_count(strong(mesh, lame_field=field), mesh, kind))
        print(f"  {cells:11s} {kind:3s}  lambda_in = 1e{p:+03d}   {counts[-1]:4d} its")
    uniform, jumped = counts[0], counts[1:]
    assert max(jumped) <= min(jumped) + 4
    assert max(jumped) <= 2 * uniform


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_does_not_track_the_incompressibility(cells):
    """A RELATIVE bound here, and the reason is the tetrahedral baseline.

        tetrahedra   14 14 14 18 36
        hexahedra    29 27 27 27 29
        prisms       36 35 35 36 36

    Hexahedra and prisms are flat outright. Tetrahedra start at 14 -- half of
    what the others need -- and reach 36, which is still the others' baseline
    but is +22 in absolute terms. lambda moves through four orders of magnitude
    across this sweep and the count at most doubles, which is the property; a
    fixed +8 would read that as a failure only because the starting point was
    so good.
    """
    _hypre()
    mesh = fixed(cells)
    counts = []
    for nu in POISSON:
        lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)
        counts.append(_count(strong(mesh, lame=lam), mesh, "cg"))
        print(f"  {cells:11s} nu = {nu:<7}  lambda = {lam:10.1f}   {counts[-1]:4d} its")
    assert max(counts) <= 3 * min(counts)


@pytest.mark.parametrize("cells", sorted(CELLS))
def test_strong_symmetry_takes_the_facet_constant_route(cells):
    """SIX moments a facet is not d copies of three, so the degree-2 complex --
    which is built for a BDM facet -- never applies, on tetrahedra either."""
    _hypre()
    mesh = fixed(cells)
    _, used = solve(strong(mesh), mesh)
    print(f"  {cells:11s} degree2 = {used}")
    assert not used
