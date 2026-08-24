"""The auxiliary-space divergence solver on the flow problem, and the two
properties that make it worth having.

The Riesz map is only as scalable as the way its H(div) block is inverted. A
Cholesky of that block is exact and creates fill, so the cost per iteration
grows even where the count does not. ADS -- Kolev & Vassilevski, SIAM J. Sci.
Comput. 34 (2012), A3079, on the Hiptmair-Xu auxiliary-space framework --
inverts it by splitting along the de Rham complex instead: no fill, and linear
in the unknowns.

WHAT ADS IS TOLD, and it is the whole of section 5.3 of that paper: the block
itself, the discrete gradient G, the discrete curl C, and the vertex
coordinates. From C, G and the coordinates hypre builds the Raviart-Thomas
interpolation Pi (Proposition 5.1) and the Nedelec interpolation Pi^V, and the
auxiliary operators are the VARIATIONAL ones of variant (B), Pi^T A Pi and
C^T A C. Nothing else is supplied and nothing else is needed -- which is the
paper's central practical claim, and the reason the MATERIAL COEFFICIENT needs
no separate treatment in the solver: it is in A, so it is in every Galerkin
product built from A.

The two properties tested here are the two the theory promises, and they are
different claims:

    h-ROBUST         the count does not grow as the mesh is refined. The
                     preconditioned operator's condition number is bounded by
                     the inf-sup and continuity constants of the space, and
                     those do not know h.
    CONTRAST-ROBUST  the count does not grow as the coefficient jumps. The
                     bound is uniform in alpha and beta (section 4), and the
                     paper's Table 6.1 holds 13-18 ADS-CG iterations over
                     sixteen orders of magnitude in the coefficient.

The one-facet-unknown products are the ones ADS is defined for -- it is a
statement about the lowest-order Raviart-Thomas space -- and among them
stabilized_rt is the one that carries the claim onto polytopes, so it is run on
a honeycomb of hexagonal prisms as well as on simplices and hexahedra.

TWO ROUTES TO THE BLOCK, and they are not the same method:

    ads     one ADS V-cycle as the preconditioner of the block. Cheap, and it
            APPROXIMATES the block rather than solving it.
    ads-cg  a short CG under that cycle, so the block is solved to a tolerance.

Only the second is h-flat. The first is bounded but drifts, and the drift is
the block solve rather than the norm -- which is exactly what these two
together measure.
"""

import numpy as np
import pytest

import mimetika_cxx as mk

from _meshes import honeycomb

R = mk.FluxRealization

# the products with ONE unknown per facet: ADS is written for that space
ONE_PER_FACET = {
    "derham_rt": R.derham_rt,
    "stabilized_rt": R.stabilized_rt,
    "adaptive_rt": R.adaptive_rt,
}

RTOL = 1e-8


def _opts(kind):
    common = dict(method="gmres", preconditioner="riesz", rtol=RTOL, max_iterations=2000)
    if kind == "riesz":
        return mk.SolverOptions(**common)
    if kind == "ads":
        return mk.SolverOptions(riesz_block_pc="ads", **common)
    # a short CG under the cycle: the block solved to a tolerance, not sampled
    return mk.SolverOptions(
        riesz_block_pc="ads", riesz_block_its=50, riesz_block_rtol=1e-2, **common
    )


GRADIENT = (1.0, 2.0, -1.0)


def _linear(x):
    return GRADIENT[0] * x[0] + GRADIENT[1] * x[1] + GRADIENT[2] * x[2]


def patch(mesh, product=R.stabilized_rt, permeability=None):
    """A linear pressure prescribed on the whole boundary: no constraint rows,
    and for the mimetic products an answer known in closed form."""
    model = mk.FlowModel(mesh, 3, 1.0, product)
    if permeability is not None:
        model.set_permeability(permeability)
    for f in mk.boundary_facets(mesh, 3):
        model.add_pressure([f], _linear(mk.centroid(mesh, 2, f)))
    return model


def cube(n):
    return mk.box([n, n, n], 3, mk.Family.cartesian)


# THE SIMPLICIAL MESH IS THE ANNULUS, NOT THE KUHN BOX. hypre's ADS setup
# fails outright on box(n, simplex) for some n -- n = 2 and 4 refuse with HYPRE
# error 12 while 3, 5 and 6 solve -- so the simplicial sweeps here run on the
# annulus, which ADS accepts at every refinement. The box failure is hypre's
# own and is reported as such; it is not exercised from this file, because a
# failed hypre setup leaves the library unusable for the rest of the process.
def wedge(nr, family=mk.Family.simplex):
    return mk.annulus(nr, nr // 2, 3, family, 1.0, 10.0, 1.0)


def _needs_hypre():
    try:
        patch(cube(3)).solve(options=_opts("ads"))
    except RuntimeError as e:  # a PETSc built without hypre has no ADS to run
        pytest.skip(f"hypre/ADS unavailable: {e}")


def _count(model, kind):
    report = model.solve(options=_opts(kind))
    assert report.converged, report.reason
    return report.iterations


# ---- 1. the answer ----------------------------------------------------------
#
# CONVERGED IS NOT CORRECT. An auxiliary space is a preconditioner, so it may
# not move the answer at all: a cycle built on the wrong complex, or on a
# permutation of the block's own rows, still converges -- to something else.
@pytest.mark.parametrize("name", sorted(ONE_PER_FACET))
@pytest.mark.parametrize("kind", ["ads", "ads-cg"])
def test_the_auxiliary_space_answer_is_the_direct_answer(name, kind):
    _needs_hypre()
    mesh = wedge(8)
    direct = patch(mesh, ONE_PER_FACET[name])
    direct.solve()
    ads = patch(mesh, ONE_PER_FACET[name])
    its = _count(ads, kind)
    worst = max(
        abs(direct.cell_pressure(e) - ads.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  {name:14s} {kind:6s} {its:3d} its   max |ads - direct| = {worst:.2e}")
    assert worst < 1e-6


# ---- 2. h-robustness -------------------------------------------------------
#
# Four meshes over a fiftyfold in unknowns. ads-cg must be FLAT, which is the
# theorem; a single V-cycle need only stay BOUNDED, and the gap between those
# two statements is the block solve.
@pytest.mark.parametrize("name", sorted(ONE_PER_FACET))
def test_ads_cg_is_h_robust(name):
    _needs_hypre()
    counts = []
    for n in (3, 4, 6, 8):
        model = patch(cube(n), ONE_PER_FACET[name])
        counts.append(_count(model, "ads-cg"))
        print(f"  {name:14s} {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


# and on simplices and prisms, where the facet count per cell is 4 and 5 rather
# than 6: the auxiliary spaces are built from the complex, so the cell type is
# a property of the input rather than of the method
@pytest.mark.parametrize(
    "family", [mk.Family.simplex, mk.Family.prism], ids=["simplex", "prism"]
)
def test_ads_cg_is_h_robust_across_cell_types(family):
    _needs_hypre()
    counts = []
    for nr in (4, 6, 8, 12):
        model = patch(wedge(nr, family), R.stabilized_rt)
        counts.append(_count(model, "ads-cg"))
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {counts[-1]:4d} its")
    assert counts[-1] <= counts[0] + 6


def test_one_ads_cycle_is_bounded_but_not_flat():
    _needs_hypre()
    counts = []
    for n in (3, 4, 6, 8):
        model = patch(cube(n))
        counts.append(_count(model, "ads"))
        print(f"  one cycle  {model.n_cells:6d} cells {counts[-1]:4d} its")
    # bounded: a V-cycle is a fixed operator, so the count cannot run away
    assert counts[-1] <= 2 * counts[0]
    # and the drift is the block solve, not the norm: solving the block to a
    # tolerance on the same meshes removes it
    flat = [_count(patch(cube(n)), "ads-cg") for n in (3, 8)]
    print(f"  the same two meshes under ads-cg: {flat[0]} -> {flat[1]} its")
    assert flat[1] <= flat[0] + 6


# ---- 3. contrast-robustness ------------------------------------------------
#
# The enclosure of the paper's Figure 6.1: two interior cubes of one material
# inside another. K_in sweeps sixteen orders of magnitude, and the count is not
# allowed to track it -- their Table 6.1 holds 13-18 ADS-CG iterations across
# the same range.
#
# BOTH DIRECTIONS MATTER, and they fail differently: a soft inclusion is where
# the norm has to carry K and a hard one is where it must not carry the cell's
# own K alone. FlowModel::norm_permeability is the scalar that does both, and
# test_the_norm_has_to_carry_the_permeability below is what says so.
JUMPS = (-8, -4, -2, 0, 2, 4, 8)


def enclosure(mesh, exponent):
    """Fig. 6.1's geometry: K = 10^p on [1/4,1/2]^3 u [1/2,3/4]^3, 1 elsewhere."""
    k = np.ones(mesh.count(3))
    for e in range(mesh.count(3)):
        x = mk.centroid(mesh, 3, e)
        inner = all(0.25 < x[i] < 0.5 for i in range(3))
        outer = all(0.5 < x[i] < 0.75 for i in range(3))
        if inner or outer:
            k[e] = 10.0 ** exponent
    return k


@pytest.mark.parametrize("kind", ["ads", "ads-cg"])
def test_the_count_does_not_track_the_contrast(kind):
    _needs_hypre()
    mesh = cube(8)
    counts = []
    for p in JUMPS:
        model = patch(mesh, R.stabilized_rt, enclosure(mesh, p))
        counts.append(_count(model, kind))
        print(f"  {kind:6s} K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 8


# THE NORM IS PART OF THE PRECONDITIONER, and this is the measurement that says
# the permeability has to be in it. A is the block ADS is handed, so the
# coefficient reaches the auxiliary spaces through Pi^T A Pi whatever the norm
# does -- but W, the pressure factor, stands for the Schur complement
# B star_K^-1 B^T, and a W that ignores K is the Gram matrix of a norm the
# operator does not have. The soft inclusion is where that shows.
def test_the_norm_has_to_carry_the_permeability():
    _needs_hypre()
    mesh = cube(8)
    k = enclosure(mesh, -8)

    carried = patch(mesh, R.stabilized_rt, k)
    with_k = _count(carried, "ads-cg")

    ignored = patch(mesh, R.stabilized_rt, k)
    ignored.set_norm_permeability(np.ones(mesh.count(3)))  # W without K
    without_k = _count(ignored, "ads-cg")

    print(f"  soft inclusion 1e-8: norm with K {with_k} its, norm without K {without_k} its")
    assert without_k > 2 * with_k


# ---- 4. the polytope -------------------------------------------------------
#
# stabilized_rt is the generalization of RT to polytopes, so the honeycomb is
# where that claim is tested: eight facets a cell against RT_0's four modes.
# Both properties are asked for again, because neither follows from the
# simplicial case -- the auxiliary spaces are built from a complex whose faces
# are hexagons here.
def test_the_polytope_is_h_robust():
    _needs_hypre()
    counts, errors = [], []
    for n in (2, 3, 4, 5):
        # the domain is fixed and the cells shrink: a refinement, not a bigger box
        mesh = honeycomb(2 * n, 2 * n, n, s=1.0 / n, h=1.0 / n)
        model = patch(mesh, R.stabilized_rt)
        counts.append(_count(model, "ads-cg"))
        errors.append(
            max(
                abs(model.cell_pressure(e) - _linear(mk.centroid(mesh, 3, e)))
                for e in range(model.n_cells)
            )
        )
        print(f"  honeycomb {model.n_cells:5d} cells {model.n_dofs:6d} dofs   "
              f"{counts[-1]:3d} its   max |p - p_exact| {errors[-1]:.1e}")
    assert counts[-1] <= counts[0] + 6
    # and the product reproduces a linear pressure on hexagonal prisms, which
    # is the consistency the polytopal claim is about -- to the solver's own
    # tolerance, not to discretization
    assert max(errors) < 1e-6


def test_the_polytope_is_contrast_robust():
    _needs_hypre()
    mesh = honeycomb(6, 6, 3, s=1.0 / 3.0, h=1.0 / 3.0)
    cen = np.array([mk.centroid(mesh, 3, e) for e in range(mesh.count(3))])
    mid = 0.5 * (cen.min(0) + cen.max(0))
    span = 0.25 * (cen.max(0) - cen.min(0))
    # the same INTERIOR inclusion as Fig. 6.1, on hexagonal prisms: a blob that
    # does not touch the boundary the pressure is prescribed on
    inside = (np.abs(cen - mid) < span).all(1)
    counts = []
    for p in (-8, -4, 0, 4, 8):
        k = np.where(inside, 10.0 ** p, 1.0)
        model = patch(mesh, R.stabilized_rt, k)
        counts.append(_count(model, "ads-cg"))
        print(f"  honeycomb K_in = 1e{p:+03d}   {counts[-1]:4d} its")
    assert max(counts) <= min(counts) + 8


# WHERE IT STOPS BEING FLAT, and it is not the cell type. A hard region filling
# HALF the domain and touching the boundary the pressure is prescribed on costs
# roughly three times the uniform count at 1e+8 -- 52 iterations on the
# cartesian box and 48 on the honeycomb, against 14 and 14 for the interior
# inclusion of the same contrast. The paper's own benchmark is the interior
# inclusion (Fig. 6.1), and its Table 6.1 reports the deterioration it does see
# for one extreme of alpha as a property of the coefficient-independent
# stopping norm rather than of the cycle.
#
# The bound here is loose on purpose: it is the measured behaviour, pinned so a
# regression would show, not a claim of robustness.
@pytest.mark.parametrize(
    "mesh_of,tag", [(lambda: cube(8), "cartesian"),
                    (lambda: honeycomb(6, 6, 3, s=1.0 / 3.0, h=1.0 / 3.0), "honeycomb")],
    ids=["cartesian", "honeycomb"],
)
def test_a_hard_region_against_the_boundary_costs_a_factor_not_an_order(mesh_of, tag):
    _needs_hypre()
    mesh = mesh_of()
    cen = np.array([mk.centroid(mesh, 3, e) for e in range(mesh.count(3))])
    half = cen[:, 0] > 0.5 * (cen[:, 0].min() + cen[:, 0].max())
    uniform = _count(patch(mesh, R.stabilized_rt, np.ones(mesh.count(3))), "ads-cg")
    hard = _count(patch(mesh, R.stabilized_rt, np.where(half, 1e8, 1.0)), "ads-cg")
    print(f"  {tag:10s} half the domain at 1e+8: {hard} its against {uniform} uniform")
    assert hard <= 4 * uniform


# ---- 5. the tensor ---------------------------------------------------------
#
# The paper's section 6.2 takes beta to be the SPE10 permeability matrix, so a
# tensor is not an extension of the theory but a case of it. What the model has
# to do is put the tensor where it belongs: in the star, hence in A, hence in
# every auxiliary operator; and reduced to the one scalar the divergence term
# can carry, in the norm.
def test_the_tensor_reduces_the_way_the_star_reads_it():
    mesh = cube(3)
    n = mesh.count(3)

    iso = patch(mesh, R.stabilized_rt, np.full(n, 7.0))
    iso.build()
    assert np.allclose(np.asarray(iso.norm_permeability), 7.0)

    # on a cube the facet normals are the axes, so the area-weighted normal
    # average is the trace over three -- and the anisotropy is NOT read as its
    # smallest eigenvalue, which is what a harmonic mean would do
    aniso = patch(mesh, R.stabilized_rt, np.tile([100.0, 100.0, 0.01], (n, 1)))
    aniso.build()
    assert np.allclose(np.asarray(aniso.norm_permeability), (100.0 + 100.0 + 0.01) / 3.0)

    # a full symmetric tensor is accepted, and one that is not definite is not
    full = patch(mesh, R.stabilized_rt, np.tile([2.0, 3.0, 4.0, 0.5, 0.2, 0.1], (n, 1)))
    full.build()
    assert np.all(np.asarray(full.norm_permeability) > 0.0)
    with pytest.raises(Exception):
        patch(mesh, R.stabilized_rt, np.tile([1.0, 1.0, 1.0, 5.0, 0.0, 0.0], (n, 1)))


@pytest.mark.parametrize("kind", ["ads", "ads-cg"])
def test_an_anisotropic_tensor_is_solved_and_is_the_direct_answer(kind):
    _needs_hypre()
    mesh = cube(6)
    n = mesh.count(3)
    K = np.tile([1.0, 1.0, 1e-4], (n, 1))
    direct = patch(mesh, R.stabilized_rt, K)
    direct.solve()
    it = patch(mesh, R.stabilized_rt, K)
    its = _count(it, kind)
    worst = max(
        abs(direct.cell_pressure(e) - it.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  kz/kx = 1e-4  {kind:6s} {its:4d} its   max |ads - direct| = {worst:.2e}")
    assert worst < 1e-6


# ---- 6. the products ADS is not for ----------------------------------------
#
# ADS is a statement about the lowest-order space: one unknown per facet, in
# 3D. derham_bdm carries d moments on each facet, so there is no ADS to run on
# its block -- and the solver says so by falling back to the exact factorization
# rather than preconditioning a space it was not given the complex for. What
# must not happen is a wrong answer.
def test_the_bdm_block_falls_back_rather_than_misapplying_ads():
    _needs_hypre()
    mesh = wedge(8)
    direct = patch(mesh, R.derham_bdm)
    direct.solve()
    asked = patch(mesh, R.derham_bdm)
    its = _count(asked, "ads")
    worst = max(
        abs(direct.cell_pressure(e) - asked.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  derham_bdm asked for ads: {its} its, max |ads - direct| = {worst:.2e}")
    assert worst < 1e-6


# The two-point star leaves a DIAGONAL first block, so the flux is eliminated
# cell by cell and what reaches a solver is the finite-volume pressure system.
# There is no H(div) block left to hand ADS, and the request is answered by the
# multigrid of the operator that does remain -- so the ADS options are accepted
# and the answer is the direct one, which is what this pins.
def test_the_two_point_star_is_condensed_rather_than_preconditioned():
    _needs_hypre()
    mesh = cube(6)
    direct = patch(mesh, R.diagonal_tpfa)
    direct.solve()
    counts = []
    for kind in ("ads", "ads-cg"):
        model = patch(mesh, R.diagonal_tpfa)
        counts.append(_count(model, kind))
        worst = max(
            abs(direct.cell_pressure(e) - model.cell_pressure(e)) for e in range(direct.n_cells)
        )
        assert worst < 1e-6
    print(f"  diagonal_tpfa condensed: ads {counts[0]} its, ads-cg {counts[1]} its "
          f"(the same system either way)")
    assert counts[0] == counts[1]
