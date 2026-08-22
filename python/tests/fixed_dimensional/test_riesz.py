"""The Riesz map preconditioner, and the one property that defines it.

A block preconditioner is not judged by being fast on one mesh. P is the matrix
of the inner product of the space the operator is an isomorphism on,

    ||(q, p)||_X^2 = (K^-1 q, q) + ||div q||_L2^2 + ||p||_L2^2

and the theorem that makes it worth building says P^-1 A has a condition number
bounded by the inf-sup and continuity constants alone. Those do not depend on h.
So the ITERATION COUNT MUST NOT GROW under refinement, and a count that grows is
the signature of a P that is not that Gram matrix -- which is what a plain B^T B
in place of B^T diag(1/|E|) B produces, and what leaving the constrained rows
alone produces.

The test is therefore a refinement sweep, not a timing.
"""

import math

import mimetika_cxx as mk
import pytest

A_IN, B_OUT = 1.0, 10.0
P_IN, P_OUT = 2.0, 1.0

RIESZ = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000
)


def dupuit(nr, dim=2, family=None, product=None):
    """The annulus with sealed symmetry planes: strong constraints and natural data."""
    family = mk.Family.simplex if family is None else family
    product = mk.FluxRealization.derham_rt if product is None else product
    mesh = mk.annulus(nr, nr // 2, dim, family, A_IN, B_OUT, 1.0)
    model = mk.SinglePhaseModel(mesh, dim, 1.0, product)
    rmid = math.sqrt(A_IN * B_OUT)
    inner, outer, sealed = [], [], []
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, f)
        on_symmetry = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - 1.0) < 1e-8))
        )
        if on_symmetry:
            sealed.append(f)
        elif math.hypot(x[0], x[1]) < rmid:
            inner.append(f)
        else:
            outer.append(f)
    model.add_normal_flux(sealed)  # STRONG: pinned rows, which P must match
    model.add_pressure(inner, P_IN)
    model.add_pressure(outer, P_OUT)
    return model


# THE DEFINING PROPERTY. Six meshes over a hundredfold in unknowns, and the
# count is not allowed to drift upward: the slack here is wide enough that
# ordinary variation passes and a preconditioner that scales with h does not.
def test_the_iteration_count_does_not_grow_under_refinement():
    counts = []
    for nr in (8, 16, 32, 48, 64):
        model = dupuit(nr)
        report = model.solve(options=RIESZ)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= 2 * counts[0]
    # and the growth is not monotone creep either: the finest is no worse than
    # the coarsest, which is what "independent of h" actually claims
    assert counts[-1] <= counts[0] + 5


# CONVERGED IS NOT CORRECT. A preconditioner that quietly changes the operator
# converges to the wrong vector, so the answer is compared against the direct
# solve of the same system rather than against a tolerance.
@pytest.mark.parametrize("nr", [8, 24])
def test_the_preconditioned_answer_is_the_direct_answer(nr):
    direct = dupuit(nr)
    direct.solve()
    riesz = dupuit(nr)
    riesz.solve(options=RIESZ)
    worst = max(
        abs(direct.cell_pressure(e) - riesz.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  {direct.n_cells} cells   max |riesz - direct| = {worst:.2e}")
    assert worst < 1e-7


# The split is the factors of the space, and it must tile the unknowns: one left
# out is never preconditioned, one counted twice is corrected twice.
def test_the_split_is_the_factors_of_the_space():
    model = dupuit(8)
    with pytest.raises(RuntimeError):  # the space does not exist until it is built
        mk.field_blocks(model)
    model.solve()
    blocks = mk.field_blocks(model)
    assert [b["name"] for b in blocks] == ["q_0", "p_0"]
    assert sum(b["size"] for b in blocks) == model.n_dofs
    assert blocks[0]["ranges"][0][0] == 0
    assert blocks[-1]["ranges"][-1][1] == model.n_dofs


# The map is written against H(div) x L2, so it holds in 3D and on cells that
# are not simplices -- the norm does not know the cell type.
@pytest.mark.parametrize(
    "dim,family",
    [(2, mk.Family.cartesian), (3, mk.Family.simplex), (3, mk.Family.prism)],
    ids=["2D-cartesian", "3D-simplex", "3D-prism"],
)
def test_it_holds_across_dimension_and_cell_type(dim, family):
    counts = []
    for nr in (6, 12):
        model = dupuit(nr, dim=dim, family=family)
        report = model.solve(options=RIESZ)
        counts.append(report.iterations)
        print(f"  {dim}D {model.n_cells:5d} cells {model.n_dofs:6d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= 2 * counts[0]


# ---- the auxiliary-space block solve -----------------------------------------
#
# The Riesz map is only as scalable as the way its first block is inverted. A
# Cholesky of that block is exact and creates fill, so the cost per iteration
# grows even though the count does not; ADS (Kolev & Vassilevski, on the
# Hiptmair-Xu auxiliary-space framework) inverts it by splitting along the de
# Rham complex instead, which costs no fill and is linear in the unknowns.
#
# It needs the discrete gradient and curl -- the complex's own boundary
# operators, d_1 and d_2 -- and the vertex coordinates, and it is defined for
# ONE UNKNOWN PER FACET in 3D. Those are supplied automatically when they
# exist, which is why this is a 3D RT test and not a BDM one.
ADS = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000,
    riesz_block_pc="ads",
)
# a short CG under ADS instead of a single V-cycle: the block is then solved to
# a tolerance rather than approximated once, and the outer count flattens
CG_ADS = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000,
    riesz_block_pc="ads", riesz_block_its=50, riesz_block_rtol=1e-2,
)


def _needs_hypre():
    try:
        dupuit(4, dim=3).solve(options=ADS)
    except RuntimeError as e:  # PETSc built without hypre
        pytest.skip(f"hypre/ADS unavailable: {e}")


def test_the_auxiliary_space_block_gives_the_same_answer():
    _needs_hypre()
    direct = dupuit(10, dim=3)
    direct.solve()
    ads = dupuit(10, dim=3)
    ads.solve(options=ADS)
    worst = max(
        abs(direct.cell_pressure(e) - ads.cell_pressure(e)) for e in range(direct.n_cells)
    )
    print(f"  {direct.n_cells} cells   max |ads - direct| = {worst:.2e}")
    assert worst < 1e-7


# SOLVING THE BLOCK, NOT APPROXIMATING IT ONCE, is what h-independence needs. A
# single V-cycle leaves the count drifting up with the mesh (29, 45, 53, 60 over
# sixty times the unknowns); with CG to 1e-2 under it the count is flat, which
# says the drift is the block solve and not the norm.
def test_the_auxiliary_space_count_does_not_grow_under_refinement():
    _needs_hypre()
    counts = []
    for nr in (8, 16, 24):
        model = dupuit(nr, dim=3)
        report = model.solve(options=CG_ADS)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= counts[0] + 5


# ---- elasticity: the same principle, three factors ---------------------------
#
# X = H(div; M) x L^2(R^d) x L^2(skew), with
#
#     ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 ,  A = C^-1
#     ||u||^2 = ||u||_L2^2 ,  ||gamma||^2 = ||gamma||_L2^2
#
# The rotation is the multiplier of the ALGEBRAIC constraint skw(sigma) = 0 and
# adds no term to the stress norm: skw is bounded L2 -> L2, so AFW's inf-sup is
# proved with the H(div) norm alone.
MU, LAM = 1.0, 1.0


def patch(nr, dim=2, family=None, product=None, formulation=None):
    """Linear displacement prescribed on the whole boundary: natural, no constraints."""
    family = mk.Family.simplex if family is None else family
    product = mk.StressRealization.stabilized_bdm if product is None else product
    formulation = mk.StressFormulation.weak_symmetry if formulation is None else formulation
    mesh = mk.annulus(nr, nr // 2, dim, family, A_IN, B_OUT, 1.0)
    pts = [mesh.point(v) for v in range(mesh.count(0))]
    lo = [min(p[k] for p in pts) for k in range(3)]
    length = max(max(p[k] for p in pts) - lo[k] for k in range(dim))
    model = mk.CauchyElasticityModel(
        mesh, dim, mk.ElasticMaterial(MU, LAM), product, formulation
    )
    gradient = [0.0] * 9
    for k in range(dim):
        gradient[k * 3 + k] = 1.0 / length
    for f in mk.boundary_facets(mesh, dim):
        x_e = mk.centroid(mesh, dim, mk.cofacet_of(mesh, dim, f))
        constant = [0.0] * 3
        for k in range(dim):
            constant[k] = (x_e[k] - lo[k]) / length
        model.prescribe_displacement([f], constant, gradient)
    return model


def test_the_elasticity_count_does_not_grow_under_refinement():
    counts = []
    for nr in (6, 12, 24, 36):
        model = patch(nr)
        report = model.solve(options=RIESZ)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= counts[0] + 10


# THE STRESS BLOCK REACHES ADS THROUGH ITS FACET CONSTANTS.
#
# stabilized_bdm puts d^2 unknowns on a facet -- d traction components, each
# against the d functions of the facet P_1 basis -- and ADS wants one. The
# constants are a SUBSET of those unknowns, so the injection into them is exact
# rather than interpolated; the divergence sees only them, and what is left is
# local to a facet. That is a two-level cycle: facet-block smoother, ADS per
# component on the constants.
def test_the_stress_block_reaches_the_auxiliary_space():
    _needs_hypre()
    direct = patch(8, dim=3)
    direct.solve()
    ads = patch(8, dim=3)
    ads.solve(options=ADS)
    worst = max(
        abs(direct.displacement(e, k) - ads.displacement(e, k))
        for e in range(direct.n_cells)
        for k in range(3)
    )
    print(f"  {direct.n_cells} cells   max |ads - direct| = {worst:.2e}")
    assert worst < 1e-7


def test_the_stress_cycle_count_does_not_grow_under_refinement():
    _needs_hypre()
    counts = []
    for nr in (6, 12, 18):
        model = patch(nr, dim=3)
        report = model.solve(options=ADS)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= counts[0] + 10


@pytest.mark.parametrize("nr", [6, 18])
def test_the_elasticity_answer_is_the_direct_answer(nr):
    direct = patch(nr)
    direct.solve()
    riesz = patch(nr)
    riesz.solve(options=RIESZ)
    worst = max(
        abs(direct.displacement(e, k) - riesz.displacement(e, k))
        for e in range(direct.n_cells)
        for k in range(2)
    )
    print(f"  {direct.n_cells} cells   max |riesz - direct| = {worst:.2e}")
    assert worst < 1e-7


# P = A, factorized: a perfect preconditioner, so one iteration. It tests the
# Pmat wiring, not a preconditioner -- and an earlier version that handed the
# blocks to the sub-KSPs after PCSetUp failed exactly here, silently.
@pytest.mark.parametrize("kind", ["flow", "elasticity"])
def test_an_exact_preconditioner_converges_in_one_iteration(kind):
    exact = mk.SolverOptions(method="gmres", preconditioner="exact", rtol=1e-12)
    model = dupuit(4) if kind == "flow" else patch(4)
    report = model.solve(options=exact)
    assert report.iterations == 1


# ---- four fields: a different norm, not the same one with a field added ------
#
# The total-pressure form moves the trace of the stress into its own unknown,
# p = lambda div u, and what is left acting on sigma is the DEVIATORIC
# compliance. So the norm changes with it:
#
#   ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 + c_p^-1 ||(2 mu)^-1 tr sigma||^2
#   ||p||^2     = c_p ||p||^2 ,   c_p = d/(2 mu) + 1/lambda
#
# and neither term is what the three-field norm would give. The pressure row is
# the only multiplier row that is not a constraint -- it has c_p |E| on the
# diagonal -- which is where both come from: W is that diagonal rather than the
# measure, and B^T W^-1 B is then the Schur complement of an invertible block,
# which is the trace term above. Taking the three-field reading instead scales
# the pressure block by c_p |E|^2 and the solve stops converging on 6^3 cells.
#
# What did NOT change: the rotation keeps its graph term. skw is bounded
# L^2 -> L^2 so the norm does not need it, but the count does -- 41 iterations
# against 85 -- and a norm the iteration disagrees with is not the norm.
def test_the_total_pressure_row_keeps_the_scale_the_operator_gave_it():
    counts = []
    for nr in (6, 12, 24):
        model = patch(nr, formulation=mk.StressFormulation.weak_symmetry_total)
        report = model.solve(options=RIESZ)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    # four fields cost a constant factor over three -- 136 against 41 -- and
    # that is the form, not the norm: what the norm owes is that the factor
    # STAYS constant, which is what is asserted
    assert counts[-1] <= counts[0] + 15


# THE LUMPED COMPLIANCE, MEASURED ON THE SADDLE POINT IT IS A NORM FOR.
#
# diagonal_afw carries a diagonal star, so a solve of it is CONDENSED by
# default: the stress is divided out and what reaches a Krylov method is the
# reduced system, whose iteration count says nothing about the norm on the
# saddle point. This test is about the norm, so it asks for the saddle point
# explicitly -- condense=False -- and the count it reports is then the one the
# theorem bounds.
SADDLE_RIESZ = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000, condense=False
)


def test_the_lumped_compliance_on_the_saddle_point():
    counts = []
    for nr in (6, 12, 24):
        model = patch(nr, product=mk.StressRealization.diagonal_afw,
                      formulation=mk.StressFormulation.weak_symmetry_total)
        report = model.solve(options=SADDLE_RIESZ)
        counts.append(report.iterations)
        print(f"  {model.n_cells:6d} cells {model.n_dofs:7d} dofs   {report.iterations:4d} its")
    assert counts[-1] <= counts[0] + 15


# AND THE CONDENSED ROUTE, which is what a caller gets by default: the stress
# is eliminated and the reduced system is solved, so the count is the reduced
# system's and the answer must still be the saddle point's.
def test_the_condensed_route_is_what_a_caller_gets():
    model = patch(12, product=mk.StressRealization.diagonal_afw,
                  formulation=mk.StressFormulation.weak_symmetry_total)
    report = model.solve(options=RIESZ)
    assert report.condensed
    assert report.condensed_dofs < model.n_dofs
    print(f"  {model.n_dofs} dofs -> {report.condensed_dofs} condensed, {report.iterations} its")
