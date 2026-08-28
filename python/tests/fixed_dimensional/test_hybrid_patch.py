"""The hybridized route, against the linear global patch.

Hybridization breaks the H(div)-conformity cell by cell and restores it with a
Lagrange multiplier on every facet -- the trace of the potential. The
multiplier space is therefore the NORMAL TRACE space of the local field, and
its width is the moments a facet carries:

    RT      q.n|_f constant           1 per facet
    BDM     q.n|_f linear             d per facet
    AFW     (sigma n)|_f linear       d.nb per facet   -- a vector P_1 field
    wrench  the rigid-motion moments  d(d+1)/2         -- a displacement screw

Giving BDM one multiplier per facet would enforce continuity of the MEAN flux
alone, the linear part of q.n could jump, and the hybridized system would stop
being the mixed one. So the width is a statement to test, not a parameter.

The claim of the method is EQUIVALENCE -- the same discrete problem, a
different elimination -- so the test is not that the interface solve converges
but that it lands on the monolithic answer, which on this patch is the exact
field. Anything that reaches the multiplier wrongly (a datum truncated to its
constant moment, a recovery that reads the kinematic fields in the local
saddle's order rather than the space's) still converges, and to something else.
"""

import numpy as np
import pytest

import mimetika_cxx as mk

MU = LAM = 1.0
GRADIENT = (1.0, 2.0, -1.0)          # p = g.x, and u = (I + W)(x)/L for the stress
CG = mk.SolverOptions(method="cg", preconditioner="hypre", rtol=1e-12, max_iterations=3000)
DIRECT = mk.SolverOptions()

FLUX = {
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
    "adaptive_rt": mk.FluxRealization.adaptive_rt,
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "stabilized_bdm": mk.FluxRealization.stabilized_bdm,
}
# product -> formulation. The diagonal members are left out: they are
# mathematically incomplete and are being developed, so their hybrid behaviour
# is not a statement this file should pin.
STRESS = {
    "derham_bdm": (mk.StressRealization.derham_bdm, mk.StressFormulation.weak_symmetry),
    "stabilized_bdm": (mk.StressRealization.stabilized_bdm, mk.StressFormulation.weak_symmetry),
    "stabilized_vem": (mk.StressRealization.stabilized_vem, mk.StressFormulation.strong_symmetry),
}
FAMILIES = {
    "cartesian": mk.Family.cartesian,
    "simplex": mk.Family.simplex,
    "prism": mk.Family.prism,
}
# Where a product reproduces a linear field, which is not everywhere: the
# two-point star is consistent only on a K-orthogonal mesh, and derham_rt's
# in-plane consistency on prisms is a pinned deficit of its own.
NOT_EXACT = {("diagonal_tpfa", "simplex"), ("diagonal_tpfa", "prism"),
             ("derham_rt", "prism")}


def box(family, n=3):
    return mk.box([n, n, n], 3, FAMILIES[family], [1.0, 2.0, 3.0])


def _linear(x):
    return sum(GRADIENT[k] * x[k] for k in range(3))


# ---- flow --------------------------------------------------------------------


def flow_patch(mesh, product):
    """p = g.x on the whole boundary, as an AFFINE datum.

    The gradient is not decoration: a facet carrying d flux moments tests the
    datum against d basis functions, and the centred ones see only the
    variation across the facet. Hybridized, that datum is the pinned
    multiplier, so a truncated one is a truncated multiplier.
    """
    m = mk.FlowModel(mesh, 3, 1.0, FLUX[product])
    for f in mk.boundary_facets(mesh, 3):
        m.add_pressure([f], _linear(mk.centroid(mesh, 2, f)), list(GRADIENT))
    m.build()
    return m


def flow_pressures(model):
    return np.array([model.cell_pressure(e) for e in range(model.n_cells)])


@pytest.mark.parametrize("product", sorted(FLUX))
def test_the_flow_multiplier_is_the_normal_trace_space(product):
    """One per facet at lowest order, d at BDM order -- the trace space."""
    mesh = box("simplex", 2)
    boundary = mk.boundary_facets(mesh, 3)
    free = mesh.count(2) - len(boundary)
    model = flow_patch(mesh, product)
    report = model.solve_hybrid(options=CG)
    per_facet = report.condensed_dofs / free
    print(f"  {product:15s} {report.condensed_dofs} multipliers over {free} free facets"
          f" = {per_facet:.0f}, moments/facet {model.moments_per_facet}")
    assert report.condensed_dofs == free * model.moments_per_facet


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("product", sorted(FLUX))
def test_the_flow_hybrid_answer_is_the_direct_answer(product, family):
    """Equivalence, which is the whole claim: same problem, two eliminations."""
    mesh = box(family)
    direct = flow_patch(mesh, product)
    direct.solve(options=DIRECT)
    hybrid = flow_patch(mesh, product)
    report = hybrid.solve_hybrid(options=CG)
    assert report.converged, report.reason
    a, b = flow_pressures(direct), flow_pressures(hybrid)
    worst = float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-300))
    print(f"  {product:15s} {family:10s} {report.iterations:3d} cg its"
          f"   |hybrid - direct| = {worst:.2e}")
    assert worst < 1e-8


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("product", sorted(FLUX))
def test_the_flow_hybrid_reproduces_the_linear_patch(product, family):
    mesh = box(family)
    model = flow_patch(mesh, product)
    model.solve_hybrid(options=CG)
    exact = np.array([_linear(mk.centroid(mesh, 3, e)) for e in range(model.n_cells)])
    worst = float(np.abs(flow_pressures(model) - exact).max() / np.abs(exact).max())
    print(f"  {product:15s} {family:10s} |p_h - p| = {worst:.2e}")
    if (product, family) in NOT_EXACT:
        return  # consistent only where it claims to be; pinned elsewhere
    assert worst < 1e-8


@pytest.mark.parametrize("product", ["derham_bdm", "stabilized_bdm"])
def test_the_bdm_multiplier_needs_the_affine_datum(product):
    """The other half of the width: d multipliers want d coefficients.

    Pinned on its constant slot alone the multiplier carries a datum that is
    constant on each facet, and the patch is lost by a factor no refinement
    removes -- while a one-moment facet cannot tell the difference.
    """
    mesh = box("simplex")
    exact = np.array([_linear(mk.centroid(mesh, 3, e)) for e in range(mesh.count(3))])

    def solved(gradient):
        m = mk.FlowModel(mesh, 3, 1.0, FLUX[product])
        for f in mk.boundary_facets(mesh, 3):
            m.add_pressure([f], _linear(mk.centroid(mesh, 2, f)),
                           list(GRADIENT) if gradient else None)
        m.build()
        m.solve_hybrid(options=CG)
        return float(np.abs(flow_pressures(m) - exact).max() / np.abs(exact).max())

    affine, number = solved(True), solved(False)
    print(f"  {product:15s} affine datum {affine:.2e}   value alone {number:.2e}")
    assert affine < 1e-8
    assert number > 1e-3


# ---- mechanics ---------------------------------------------------------------


def stress_patch(mesh, realization):
    """u = G x on the whole boundary, written about each cell's centroid."""
    product, form = STRESS[realization]
    m = mk.CauchyMechanicsModel(mesh, 3, mk.ElasticMaterial(MU, LAM), product, form)
    gradient = [0.0] * 9
    for k in range(3):
        gradient[k * 3 + k] = GRADIENT[k]
    facets = mk.boundary_facets(mesh, 3)
    cells = mk.cofacets_of(mesh, 3, facets)
    for f, cell in zip(facets, cells):
        x = mk.centroid(mesh, 3, int(cell))
        m.prescribe_displacement([f], [GRADIENT[k] * x[k] for k in range(3)], gradient)
    m.build()
    return m


def stress_fields(model):
    u = np.array([[model.displacement(e, k) for k in range(3)] for e in range(model.n_cells)])
    s = np.array([model.cell_stress(e) for e in range(model.n_cells)])
    return u, s


@pytest.mark.parametrize("realization", sorted(STRESS))
def test_the_mechanics_multiplier_is_the_displacement_trace(realization):
    """d.nb componentwise -- a vector P_1 field -- and 6 on the wrench axis."""
    mesh = box("simplex", 2)
    boundary = mk.boundary_facets(mesh, 3)
    free = mesh.count(2) - len(boundary)
    model = stress_patch(mesh, realization)
    report = model.solve_hybrid(options=CG)
    per_facet = report.condensed_dofs // free
    print(f"  {realization:15s} {report.condensed_dofs} multipliers over {free} free facets"
          f" = {per_facet} per facet")
    assert report.condensed_dofs == free * per_facet
    assert per_facet == (6 if "vem" in realization else 9)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("realization", sorted(STRESS))
def test_the_mechanics_hybrid_answer_is_the_direct_answer(realization, family):
    """Displacement AND stress: the recovery has to put every field back.

    The kinematic unknowns live in the local saddle as one vector per cell --
    the divergence rows then the asymmetry rows -- while the space keeps them
    as separate fields. Reading that back in the saddle's order writes the
    rotation into the displacement's slots, which is invisible on the wrench
    axis (no rotation field) and wrong on the componentwise one.
    """
    if "vem" in realization and family != "simplex":
        pytest.skip("the strongly-symmetric family is exercised on tetrahedra here")
    mesh = box(family)
    direct = stress_patch(mesh, realization)
    direct.solve(options=DIRECT)
    hybrid = stress_patch(mesh, realization)
    report = hybrid.solve_hybrid(options=CG)
    assert report.converged, report.reason
    ua, sa = stress_fields(direct)
    ub, sb = stress_fields(hybrid)
    du = float(np.abs(ua - ub).max() / max(float(np.abs(ua).max()), 1e-300))
    ds = float(np.abs(sa - sb).max() / max(float(np.abs(sa).max()), 1e-300))
    print(f"  {realization:15s} {family:10s} {report.iterations:3d} cg its"
          f"   |u| {du:.2e}   |sigma| {ds:.2e}")
    assert du < 1e-8
    assert ds < 1e-8


@pytest.mark.parametrize("realization", sorted(STRESS))
def test_the_mechanics_hybrid_reproduces_the_linear_patch(realization):
    mesh = box("simplex")
    model = stress_patch(mesh, realization)
    model.solve_hybrid(options=CG)
    u, _ = stress_fields(model)
    exact = np.array([[GRADIENT[k] * mk.centroid(mesh, 3, e)[k] for k in range(3)]
                      for e in range(model.n_cells)])
    worst = float(np.abs(u - exact).max() / np.abs(exact).max())
    print(f"  {realization:15s} |u_h - u| = {worst:.2e}")
    assert worst < 1e-8


def test_a_prescribed_traction_is_refused_rather_than_dropped():
    """The interface load carries the cell rows and the multiplier datum only.

    A traction is a sigma-row datum and exokal's hybrid_interface_load has no
    sigma-row term, so it would be dropped in silence -- measured, a
    traction-driven column came back zero with the interface reporting zero
    iterations. Refused until the load can carry it.
    """
    mesh = box("simplex", 2)
    product, form = STRESS["stabilized_vem"]
    m = mk.CauchyMechanicsModel(mesh, 3, mk.ElasticMaterial(MU, LAM), product, form)
    base, top = [], []
    for f in mk.boundary_facets(mesh, 3):
        (base if mk.centroid(mesh, 2, f)[2] < 1e-9 else top).append(f)
    m.prescribe_displacement(base, [0.0, 0.0, 0.0], [0.0] * 9)
    traction = [0.0] * 9
    traction[8] = -0.5
    m.add_traction(top, traction)
    m.build()
    with pytest.raises(Exception, match="traction"):
        m.solve_hybrid(options=CG)
