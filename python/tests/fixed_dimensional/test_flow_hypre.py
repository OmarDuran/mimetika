"""The flow problem through mimetika_hypre, called DIRECTLY rather than through
PETSc's PCHYPRE, on BOTH preconditioners the module offers.

    ADS   the Riesz map, with the flux block preconditioned by the auxiliary
          space divergence solver -- sections 1 to 7 here
    MGR   a multigrid reduction that ELIMINATES the flux instead, leaving the
          cell-centred Laplacian for BoomerAMG -- section 8

test_flow_ads.py states the two properties and tests them on the PETSc route.
This file tests the same two on the other route, because they are different
code: `mimetika_hypre` links its own hypre, is handed the assembled system and
the norm as plain arrays through `mk.ads_handoff`, and calls hypre itself.
Nothing about the METHOD differs, so a disagreement between the two files is a
defect in one of the bridges.

    h-ROBUST         the count does not grow as the mesh is refined
    CONTRAST-ROBUST  the count does not grow as the coefficient jumps

Both preconditioners are held to both, on every flow realization.

WHAT DIFFERS IS WHAT ADS IS TOLD FOR A BDM FACET.

    one moment a facet   derham_rt, stabilized_rt, adaptive_rt. The complex is
                         the mesh's own signed incidence, G = d0 and C = d1,
                         and ADS builds Pi from those and the coordinates.
    d moments a facet    the block is reached DIRECTLY, whatever the cells are,
                         by handing ADS the degree-2 complex
                         P3 -> N2E2 -> BDM1 and the interpolations Pi_rt, Pi_nd
                         through HYPRE_ADSSetInterpolations. `degree2` in the
                         handoff says so, and
                         test_the_degree_two_complex_is_used_on_every_cell_type
                         is what pins it -- the facet-constant subspace it
                         replaced converges at similar counts under the inner
                         CG, so nothing else here would notice the wrong route
                         being taken.

WHY IT IS NOT SIMPLEX-ONLY. dec/mimetic_curl.hpp computes C per CELL from a
reconstruction, which on a polytope is a least-squares fit (D_edge > m) that
couples the whole cell, so the two cells sharing a facet disagree. But d^1 is
facet-local by nature: on f the normal curl is the 2D surface curl of the
tangential trace, so surface Stokes writes a facet's rows in that facet's own
dofs alone, on a hexahedron as much as on a tetrahedron. C is built that way
(bdm_complex.hpp), and Pi's vertex hats reproduce the linears exactly on a
tetrahedron and in least squares beyond it.

TWO ROUTES TO THE BLOCK, as in the PETSc file and for the same reason:

    one cycle   `block_iterations = 0`. Cheap, and it APPROXIMATES the block.
    inner CG    a short CG under that cycle, so the block is SOLVED.

Only the second is flat. The first is bounded and drifts, and the drift is the
block solve rather than the norm.
"""

import numpy as np
import pytest

import mimetika_cxx as mk

R = mk.FluxRealization

# the products with ONE unknown per facet: what ADS is written for
ONE_PER_FACET = {
    "derham_rt": R.derham_rt,
    "stabilized_rt": R.stabilized_rt,
    "adaptive_rt": R.adaptive_rt,
}

# d moments per facet. On a tetrahedron the stabilization vanishes and the two
# ARE the same operator (test_flow_ads.py proves that on the cell spectra), so
# the degree-2 complex -- whose Pi_rt is the stabilized product's own N -- is
# the right one for both. On a polytope they differ, and each gets its own.
BDM = {
    "derham_bdm": R.derham_bdm,
    "stabilized_bdm": R.stabilized_bdm,
}

RTOL = 1e-8
GRADIENT = (1.0, 2.0, -1.0)


def _hypre():
    """The module, or a skip. It is an optional build: MIMETIKA_USE_HYPRE=ON."""
    try:
        import mimetika_hypre as mh
    except ImportError as e:
        pytest.skip(f"mimetika_hypre is not built: {e}")
    if not hasattr(mk, "ads_handoff"):
        pytest.skip("mimetika_cxx has no ads_handoff")
    return mh


def _options(kind):
    """`one` applies a single ADS cycle; `cg` solves the block under it;
    `mgr` replaces the Riesz map entirely with a multigrid reduction."""
    mh = _hypre()
    o = mh.AdsOptions()
    o.rtol = RTOL
    o.max_iterations = 2000
    if kind in ("cg", "mgr"):
        o.block_iterations = 50
        o.block_rtol = 1e-2
    if kind == "mgr":
        o.mgr = True
    return o


def _linear(x):
    return GRADIENT[0] * x[0] + GRADIENT[1] * x[1] + GRADIENT[2] * x[2]


def patch(mesh, product=R.stabilized_rt, permeability=None):
    """A linear pressure on the whole boundary, with its gradient.

    The gradient is not decoration: a facet holding d flux moments tests the
    datum against d basis functions, and the centred ones see only the
    VARIATION across the facet. Without it the BDM products lose the patch and
    a count measured on a wrong answer is not a measurement.
    """
    model = mk.FlowModel(mesh, 3, 1.0, product)
    if permeability is not None:
        model.set_permeability(permeability)
    for f in mk.boundary_facets(mesh, 3):
        model.add_pressure([f], _linear(mk.centroid(mesh, 2, f)), list(GRADIENT))
    return model


def solve(model, mesh, kind="cg"):
    """Assemble here, solve in mimetika_hypre, accept back -- what the
    `hypre-ads` solver of the examples does, stated once for the tests."""
    mh = _hypre()
    mk.mpi_size()  # PETSc owns MPI; hypre attaches to it rather than starting it
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


def cube(n):
    return mk.box([n, n, n], 3, mk.Family.cartesian)


# THE SIMPLICIAL MESH IS THE ANNULUS, NOT THE KUHN BOX, for the reason
# test_flow_ads.py gives: hypre's ADS is unhappy with box(n, simplex) at some n.
# Measured here through this route, stabilized_rt on box(4, simplex) converges
# at contrast 1e-8, 1 and 1e8 and does NOT at 1e-2, 1e2 and 1e4 -- non-monotone
# in the coefficient, so it is the mesh and not the method. The annulus is
# accepted at every refinement.
def wedge(nr, family=mk.Family.simplex):
    return mk.annulus(nr, max(2, nr // 2), 3, family, 1.0, 10.0, 1.0)


# ---- 1. the answer ---------------------------------------------------------
#
# CONVERGED IS NOT CORRECT. A preconditioner built on the wrong complex, or on
# a permutation of the block's rows, still converges -- to something else. For
# the degree-2 path this is the check that C's rows really are the model's flux
# dofs, in its order and its basis.
@pytest.mark.parametrize("name", sorted({**ONE_PER_FACET, **BDM}))
def test_the_hypre_answer_is_the_direct_answer(name):
    _hypre()
    product = {**ONE_PER_FACET, **BDM}[name]
    mesh = wedge(6)
    direct = patch(mesh, product)
    direct.solve()
    got = patch(mesh, product)
    its = _count(got, mesh)
    worst = max(
        abs(direct.cell_pressure(e) - got.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  {name:15s} {its:3d} its   max |hypre - direct| = {worst:.2e}")
    assert worst < 1e-6


# ---- 2. which complex ADS was given ----------------------------------------
@pytest.mark.parametrize("name", sorted(BDM))
def test_the_degree_two_complex_is_used_on_every_cell_type(name):
    """C is surface Stokes on a facet, so it needs no cell reconstruction and a
    polytope has a global C after all; Pi's vertex hats reproduce the linears
    exactly on a tetrahedron and in least squares beyond it. So a BDM block gets
    the degree-2 complex whatever the cells are, and ADS sees the WHOLE row
    rather than its facet constants -- three moments a facet instead of one."""
    _hypre()
    _, on_tets = solve(patch(wedge(4), BDM[name]), wedge(4))
    _, on_hexes = solve(patch(cube(3), BDM[name]), cube(3))
    print(f"  {name:15s} tetrahedra {on_tets}   hexahedra {on_hexes}")
    assert on_tets, "a tetrahedral BDM block should be given the degree-2 complex"
    assert on_hexes, "a polytopal BDM block should be given it too"


def test_one_moment_a_facet_never_takes_the_degree_two_complex():
    _hypre()
    for name, product in sorted(ONE_PER_FACET.items()):
        _, used = solve(patch(wedge(4), product), wedge(4))
        print(f"  {name:15s} degree2 = {used}")
        assert not used


# ---- 3. h-robustness -------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ONE_PER_FACET))
def test_hypre_ads_cg_is_h_robust(name):
    _hypre()
    counts = []
    for n in (3, 4, 6, 8):
        mesh = cube(n)
        model = patch(mesh, ONE_PER_FACET[name])
        counts.append(_count(model, mesh))
        print(f"  {name:15s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


@pytest.mark.parametrize("name", sorted(BDM))
def test_the_bdm_block_is_h_robust_on_polytopes(name):
    """The degree-2 route on hexahedra: C facet-wise, Pi's hats in least
    squares. Measured against the facet-constant subspace it replaced, single
    cycle on the mechanics ladder: 42, 46, 52 -> 32, 36, 38."""
    _hypre()
    counts = []
    for n in (3, 4, 6, 8):
        mesh = cube(n)
        model = patch(mesh, BDM[name])
        counts.append(_count(model, mesh))
        print(f"  {name:15s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


@pytest.mark.parametrize("name", sorted(BDM))
def test_the_bdm_block_is_h_robust_on_tetrahedra(name):
    """The degree-2 route: ADS acts on the BDM block itself."""
    _hypre()
    counts = []
    for nr in (4, 6, 8, 12):
        mesh = wedge(nr)
        model = patch(mesh, BDM[name])
        report, used = solve(model, mesh)
        assert report.converged, report.reason
        assert used
        counts.append(report.iterations)
        print(f"  {name:15s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


@pytest.mark.parametrize(
    "family", [mk.Family.simplex, mk.Family.prism], ids=["simplex", "prism"]
)
def test_hypre_ads_cg_is_h_robust_across_cell_types(family):
    _hypre()
    counts = []
    for nr in (4, 6, 8, 12):
        mesh = wedge(nr, family)
        counts.append(_count(patch(mesh, R.stabilized_rt), mesh))
        print(f"  {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


def test_one_cycle_is_bounded_but_not_flat():
    _hypre()
    counts = []
    for nr in (4, 6, 8, 12):
        mesh = wedge(nr)
        counts.append(_count(patch(mesh, R.stabilized_rt), mesh, "one"))
        print(f"  one cycle  {counts[-1]:4d} its")
    assert counts[-1] <= 2 * counts[0]
    flat = [_count(patch(wedge(nr), R.stabilized_rt), wedge(nr)) for nr in (4, 12)]
    print(f"  the same two meshes with the block solved: {flat[0]} -> {flat[1]} its")
    assert flat[1] <= flat[0] + 6


# ---- 4. contrast-robustness ------------------------------------------------
#
# The enclosure of Kolev & Vassilevski's Figure 6.1, so the two test files and
# the paper read the same problem: two interior cubes of one material inside
# another, K_in over fourteen orders of magnitude. Their Table 6.1 holds 13-18
# ADS-CG iterations across the range.
JUMPS = (-7, -4, -2, 0, 2, 4, 7)


def enclosure(mesh, exponent):
    """K = 10^p on [1/4,1/2]^3 u [1/2,3/4]^3, 1 elsewhere."""
    k = np.ones(mesh.count(3))
    for e in range(mesh.count(3)):
        x = mk.centroid(mesh, 3, e)
        inner = all(0.25 < x[i] < 0.5 for i in range(3))
        outer = all(0.5 < x[i] < 0.75 for i in range(3))
        if inner or outer:
            k[e] = 10.0 ** exponent
    return k


@pytest.mark.parametrize("kind", ["one", "cg"])
def test_the_count_does_not_track_the_contrast(kind):
    _hypre()
    mesh = cube(6)
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, R.stabilized_rt, enclosure(mesh, p)), mesh, kind))
        print(f"  {kind:3s}  K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 8


@pytest.mark.parametrize("name", sorted(BDM))
@pytest.mark.parametrize("kind", ["one", "cg"])
def test_the_bdm_count_does_not_track_the_contrast(name, kind):
    """The norm's K-scalar is per CELL, so it reaches a d-moment block exactly
    as it reaches a one-moment one, and the coarse operator is Galerkin either
    way -- the coefficient is in the auxiliary spaces because it is in A."""
    _hypre()
    mesh = cube(6)
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, BDM[name], enclosure(mesh, p)), mesh, kind))
        print(f"  {name:15s} {kind:3s}  K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    # The solved block is flat; a single cycle only APPROXIMATES it, so it is
    # bounded and drifts -- 16 at K = 1 against 25 at 1e-7, over fourteen orders
    # of contrast. Held to separate bounds for that reason, not to make room.
    assert max(counts) <= min(counts) + (8 if kind == "cg" else 10)


# ---- 5. contrast on a CHECKERBOARD ------------------------------------------
#
# The enclosure above has ONE bounded interface. A checkerboard jumps across
# EVERY facet, so the auxiliary spaces never see a patch on which the
# coefficient is smooth, and it is the harder statement of the same property.
#
# The pattern is a function of position in the bounding box and not of the cell
# numbering, so it is the SAME field on every mesh -- a pattern redrawn per mesh
# makes an h-ladder meaningless.
def checkerboard(mesh, exponent, cells_per_side=4):
    """K = 10^p on alternate blocks of a cells_per_side^3 partition of the box."""
    x = np.array([mk.centroid(mesh, 3, e) for e in range(mesh.count(3))])
    lo, hi = x.min(axis=0), x.max(axis=0)
    block = np.floor(cells_per_side * (x - lo) / np.maximum(hi - lo, 1e-300) * 0.999)
    return np.where(block.sum(axis=1) % 2 == 0, 1.0, 10.0 ** exponent)


@pytest.mark.parametrize("name", sorted({**ONE_PER_FACET, **BDM}))
@pytest.mark.parametrize("kind", ["one", "cg"])
def test_the_count_does_not_track_a_checkerboard(name, kind):
    """MESH-ALIGNED: on a cartesian box each block is a union of whole cells, so
    the jump lies on facets and the count is flat over all fourteen orders."""
    _hypre()
    mesh = cube(6)
    product = {**ONE_PER_FACET, **BDM}[name]
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, product, checkerboard(mesh, p)), mesh, kind))
        print(f"  {name:15s} {kind:3s}  K = 1e{p:+03d}   {counts[-1]:4d} its")
    # the two routes are held to different bounds for the reason
    # test_one_cycle_is_bounded_but_not_flat states: a single cycle
    # APPROXIMATES the block, so it is bounded and drifts -- derham_rt runs 13
    # at K = 1 and 22 at either extreme -- while the solved block is flat.
    assert max(counts) <= min(counts) + (8 if kind == "cg" else 14)


# NOT MESH-ALIGNED, and this is where the property stops -- on ONE side.
#
# On the annulus the same pattern is stated in cartesian coordinates over a
# radial tetrahedral mesh, so a block boundary cuts cells at an arbitrary angle
# and no auxiliary patch sees a smooth coefficient. A SOFT inclusion is still
# flat -- 12 iterations at 1e-7, as at 1 -- and a HARD one is not:
#
#     K_in            1e+4   1e+7   1e+8
#     stabilized_rt     12     24     41
#     stabilized_bdm    14     42   1163
#
# and the 1e+8 figure moves with refinement (608 on wedge(12)), so the sweep
# stops at a 1e4 inclusion, which is the range the claim holds over. The rest
# is recorded rather than asserted: a bound around a number that swings by an
# order of magnitude guards nothing.
#
# It is not this route's defect. The same problem through PETSc's PCHYPRE takes
# 98 iterations at 1e+8 on wedge(6) and DIVERGES on wedge(12), where this one
# converges in 608 -- both routes meet the same wall, and this one degrades
# instead of failing.
MISALIGNED_JUMPS = (-7, -4, -2, 0, 2, 4)


@pytest.mark.parametrize("name", sorted({"stabilized_rt": None, **BDM}))
def test_a_misaligned_checkerboard_is_flat_to_a_1e4_inclusion(name):
    _hypre()
    mesh = wedge(6)
    product = {**ONE_PER_FACET, **BDM}[name]
    counts = []
    for p in MISALIGNED_JUMPS:
        counts.append(_count(patch(mesh, product, checkerboard(mesh, p)), mesh))
        print(f"  {name:15s} K = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 8


# ---- 8. MGR: reduction instead of the Riesz map ----------------------------
#
# The Riesz map preconditions the flux block and pairs it with the pressure
# block-diagonally. MGR instead ELIMINATES the flux: F = q, C = p, and the
# coarse operator is
#
#     S = -D M^-1 D^T ,
#
# the cell-centred Laplacian -- one unknown a cell, which is BoomerAMG's ground
# and is what the two-point family already is. The flux block here carries no
# div term, so its F-relaxation is not facing the a_div near-nullspace that ADS
# exists for, which is why this is available for flow and not for the stress.
#
# TWO LEVELS WHEN A FACET CARRIES MORE THAN ONE MOMENT. div is topological and
# reads only the constant moment, D = [0 | D_0], so the higher moments lie
# entirely in ker D. Reducing the whole flux at once asks relaxation to invert M
# on a divergence-free subspace it cannot damp -- measured as 15, 22, 28 over
# the ladder against 12, 13, 14 when the moments are split off first, and no
# convergence at all with an AMG V-cycle on the F block. Splitting them makes
# level 0 EXACT on the constraint: the eliminated block contributes nothing to
# it, so level 1 is the RT0 system. The solver identifies the constants from the
# operator -- a flux dof is one exactly when it appears in a constraint row --
# rather than from an assumed dof ordering, so it holds for either route.
MGR_PRODUCTS = {**ONE_PER_FACET, **BDM}


@pytest.mark.parametrize("name", sorted(MGR_PRODUCTS))
def test_mgr_is_h_robust(name):
    _hypre()
    counts = []
    for n in (4, 6, 8):
        mesh = cube(n)
        model = patch(mesh, MGR_PRODUCTS[name])
        counts.append(_count(model, mesh, "mgr"))
        print(f"  {name:15s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs   "
              f"{counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 4


@pytest.mark.parametrize("name", sorted(MGR_PRODUCTS))
def test_mgr_does_not_track_the_contrast(name):
    """One bounded interface, fourteen orders across it."""
    _hypre()
    mesh = cube(6)
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, MGR_PRODUCTS[name], enclosure(mesh, p)),
                             mesh, "mgr"))
        print(f"  {name:15s} K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 4


@pytest.mark.parametrize("name", sorted(MGR_PRODUCTS))
def test_mgr_does_not_track_a_checkerboard(name):
    """Jumps on every interior interface, which is the sharper question -- and
    the one where MGR beats the Riesz map on a d-moment facet: measured spread
    3 against ADS's 6."""
    _hypre()
    mesh = cube(6)
    counts = []
    for p in JUMPS:
        counts.append(_count(patch(mesh, MGR_PRODUCTS[name], checkerboard(mesh, p)),
                             mesh, "mgr"))
        print(f"  {name:15s} K = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 4


@pytest.mark.parametrize("name", sorted(MGR_PRODUCTS))
def test_the_mgr_answer_is_the_direct_answer(name):
    """A count measured on a wrong answer is not a measurement."""
    _hypre()
    mesh = cube(4)
    direct = patch(mesh, MGR_PRODUCTS[name])
    direct.solve()
    got = patch(mesh, MGR_PRODUCTS[name])
    solve(got, mesh, "mgr")
    worst = max(
        abs(direct.cell_pressure(e) - got.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  {name:15s} max |mgr - direct| = {worst:.2e}")
    assert worst < 1e-6



# ---- 9. adaptive_rt with the selection ACTUALLY ON --------------------------
#
# adaptive_rt derives a binary eta per cell from the degeneracy scan: 1 where
# the stabilized member is used, 0 on the cells the scan flags. At exokal's
# DEFAULT threshold none of these meshes has a flagged cell, so every other
# test in this file runs adaptive_rt as an alias of stabilized_rt and says
# nothing about the adaptive path. The threshold here is chosen to flag a real
# fraction of the annulus, and the first test refuses to let that go silently
# vacuous.
#
# What ADS is told does NOT change with eta: the space is one moment a facet
# either way, so G, C and the coordinates are the same and only the entries of
# A_h move, cell by cell. That is the same kind of variation as a jumping K,
# which is why the count is expected to hold -- and does.
ADAPTIVE_PERCENT = 80.0


def _flagged(mesh):
    """Cells the scan flags at ADAPTIVE_PERCENT, from the mesh alone."""
    percent = np.asarray(mk.cell_collapse_percent(mesh, 3))
    judged = percent[~np.isnan(percent)]
    return int((judged < ADAPTIVE_PERCENT).sum()), int(judged.size)


def _adaptive(mesh, permeability=None):
    model = patch(mesh, R.adaptive_rt, permeability)
    model.set_degeneracy_percent(ADAPTIVE_PERCENT)
    return model


def test_the_adaptive_selection_is_active_on_this_ladder():
    """Without this the two tests below would pass on stabilized_rt."""
    for n in (4, 6, 8):
        mesh = wedge(n)
        flagged, judged = _flagged(mesh)
        print(f"  wedge({n})  {flagged:4d} of {judged:4d} cells flagged")
        assert flagged > 0, "the selection is inactive; the tests below are vacuous"


def test_adaptive_rt_is_h_robust_with_the_selection_on():
    _hypre()
    counts = []
    for n in (4, 6, 8):
        mesh = wedge(n)
        flagged, judged = _flagged(mesh)
        model = _adaptive(mesh)
        counts.append(_count(model, mesh))
        print(f"  wedge({n})  {flagged:4d}/{judged:4d} flagged  "
              f"{model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 6


def test_adaptive_rt_does_not_track_the_contrast_with_the_selection_on():
    """The selection and the coefficient jump at once."""
    _hypre()
    mesh = wedge(6)
    counts = []
    for p in JUMPS:
        counts.append(_count(_adaptive(mesh, enclosure(mesh, p)), mesh))
        print(f"  K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 6
