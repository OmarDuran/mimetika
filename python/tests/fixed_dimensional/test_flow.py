"""Single-phase Darcy flow through the Python interface.

The Python-side mirror of tests/model/test_flow_model.cpp: the same column
with a linear pressure it must reproduce exactly, the same Dupuit annulus it
can only approximate, and the same three flux products -- including what each
one does NOT claim, which is asserted as a refusal rather than skipped.
"""

import math

import pytest

import mimetika_cxx as mk

BDM = mk.FluxRealization.derham_bdm
RT = mk.FluxRealization.derham_rt
STABILIZED = mk.FluxRealization.stabilized_rt
TPFA = mk.FluxRealization.diagonal_tpfa
ADAPTIVE = mk.FluxRealization.adaptive_rt

FAMILIES = [mk.Family.cartesian, mk.Family.simplex, mk.Family.prism]
PRODUCTS = [BDM, RT, STABILIZED]


class Outcome:
    def __init__(self):
        self.max_err = 0.0
        self.rms_err = 0.0
        self.cells = 0
        self.dofs = 0


def run(m, dim, high, low, sealed, p_high, p_low, exact, how=BDM, degeneracy=None) -> Outcome:
    prob = mk.FlowModel(m, dim, 1.0, how)
    prob.add_normal_flux(sealed)
    prob.add_pressure(high, p_high)
    prob.add_pressure(low, p_low)
    if degeneracy is not None:
        prob.set_degeneracy_percent(degeneracy)
    prob.solve()

    out = Outcome()
    out.cells = prob.n_cells
    out.dofs = prob.n_dofs
    acc = 0.0
    for e in range(out.cells):
        d = prob.cell_pressure(e) - exact(mk.centroid(m, dim, e))
        out.max_err = max(out.max_err, abs(d))
        acc += d * d
    out.rms_err = math.sqrt(acc / out.cells)
    return out


# the facet sets of a column: the two ends, and everything else sealed
def column_case(n, dim, family, how=BDM, degeneracy=None) -> Outcome:
    h, p_hi, p_lo = 1.0, 2.0, 1.0
    m = mk.column(n, dim, family, h)
    axis = dim - 1
    top, base, side = [], [], []
    for f in mk.boundary_facets(m, dim):
        z = mk.centroid(m, dim - 1, f)[axis]
        if abs(z - h) < 1e-9:
            top.append(f)
        elif abs(z) < 1e-9:
            base.append(f)
        else:
            side.append(f)
    # p = p_lo at the base, p_hi at the top: linear in between
    return run(
        m, dim, top, base, side, p_hi, p_lo,
        lambda x: p_lo + (p_hi - p_lo) * x[axis] / h,
        how,
        degeneracy,
    )


# the facet sets of a quarter annulus: the two radii, symmetry planes sealed
def annulus_case(nr, nt, dim, family, how=BDM, degeneracy=None) -> Outcome:
    a, b, hz, p_a, p_b = 1.0, 10.0, 1.0, 2.0, 1.0
    m = mk.annulus(nr, nt, dim, family, a, b, hz)
    rmid = math.sqrt(a * b)
    inner, outer, sealed = [], [], []
    for f in mk.boundary_facets(m, dim):
        x = mk.centroid(m, dim - 1, f)
        r = math.hypot(x[0], x[1])
        sym = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - hz) < 1e-8))
        )
        if sym:
            sealed.append(f)
        elif r < rmid:
            inner.append(f)
        else:
            outer.append(f)
    return run(
        m, dim, inner, outer, sealed, p_a, p_b,
        lambda x: p_a + (p_b - p_a) * math.log(math.hypot(x[0], x[1]) / a) / math.log(b / a),
        how,
        degeneracy,
    )


# WHAT EACH PRODUCT CLAIMS, stated once. A configuration outside a product's
# claim is not a failure and not a silent skip -- it is reported as refused, and
# a separate test checks that the refusal is an exception rather than a wrong
# answer.
def supported(r, dim, family) -> bool:
    # All three claim every structured family in both dimensions. RT_0's
    # original argument was d+1 modes against d+1 facets -- a simplex, with a
    # hexahedron a different case rather than a coarser one -- but the
    # consistency-only families now enrich with curl-type divergence-free
    # fields until the facet moments are unisolvent, which reaches the tensor
    # cells too. The claim is bounded by FACET COUNT, not by cell type.
    del r, dim, family
    return True


CLAIMED = [
    (r, dim, f)
    for r in PRODUCTS
    for dim in (2, 3)
    for f in FAMILIES
    if supported(r, dim, f)
]
_ID = lambda c: f"{str(c[0]).split('.')[-1]}-{c[1]}D-{str(c[2]).split('.')[-1]}"  # noqa: E731


# A LINEAR PRESSURE IS REPRODUCED EXACTLY. The flux space contains the constant
# fields, so the gradient of a linear pressure is represented with no error at
# all. Anything above round-off here is a broken space, not a coarse mesh.
@pytest.mark.parametrize("how,dim,family", CLAIMED, ids=[_ID(c) for c in CLAIMED])
def test_the_column_reproduces_the_linear_solution_exactly(how, dim, family):
    o = column_case(6, dim, family, how)
    print(
        f"  column  {mk.flux_realization_name(how):<10} {dim}D "
        f"{mk.family_name(family):<10} {o.cells:5d} cells {o.dofs:7d} dofs   "
        f"max {o.max_err:.2e}   rms {o.rms_err:.2e}"
    )
    assert o.max_err < 1e-10


# AND THE DUPUIT PROFILE, which is not in the space: the radial harmonic is
# approximated, so the error is a resolution and not a defect. It must fall with
# refinement, which the second size checks.
@pytest.mark.parametrize("how,dim,family", CLAIMED, ids=[_ID(c) for c in CLAIMED])
def test_the_annulus_reproduces_dupuit(how, dim, family):
    coarse = annulus_case(8, 4, dim, family, how)
    fine = annulus_case(16, 8, dim, family, how)
    print(
        f"  annulus {mk.flux_realization_name(how):<10} {dim}D "
        f"{mk.family_name(family):<10} {coarse.cells:5d} -> {fine.cells:5d} cells   "
        f"max {coarse.max_err:.2e} -> {fine.max_err:.2e}   "
        f"rms {coarse.rms_err:.2e} -> {fine.rms_err:.2e}"
    )
    assert coarse.max_err < 5e-2
    assert fine.rms_err < coarse.rms_err  # refinement helps, so it is resolution


# THE SPACES ARE NOT THE SAME SIZE, which is the concrete content of "different
# discretizations": d moments per facet against one, and all three exact on a
# linear pressure.
# ---- the two-point product --------------------------------------------------
#
# diagonal_tpfa is exokal's, and so is the question of where it is consistent:
# it reconstructs nothing, its M is the diagonal primal-dual star, and it is
# strongly consistent only where the mesh is K-ORTHOGONAL. exokal tests that.
# What is tested here is that the PYTHON INTERFACE reaches it -- that the enum
# maps to the realization it names and the model built from it lays out the
# space it should.
#
# Its space is RT's -- one flux per facet -- so a binding that mixed the two up
# would still assemble, still solve, and only the count would notice.
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: str(f).split(".")[-1])
def test_the_two_point_product_lays_out_one_flux_per_facet(dim, family):
    tpfa = column_case(6, dim, family, TPFA)
    rt = column_case(6, dim, family, RT)
    print(
        f"  tpfa    {dim}D {mk.family_name(family):<10} {tpfa.dofs:6d} dofs "
        f"(rt {rt.dofs})   max {tpfa.max_err:.2e}"
    )
    assert tpfa.dofs == rt.dofs
    assert tpfa.cells == rt.cells


# AND IT REPRODUCES A LINEAR PRESSURE WHERE IT CLAIMS TO: the hexahedral and
# prismatic columns are K-orthogonal, the tetrahedral one is not. That boundary
# is exokal's to test against the geometry; this asserts only the half mimetika
# depends on.
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "family", [mk.Family.cartesian, mk.Family.prism], ids=["cartesian", "prism"]
)
def test_the_two_point_product_is_exact_where_the_column_is_orthogonal(dim, family):
    o = column_case(6, dim, family, TPFA)
    print(f"  tpfa    {dim}D {mk.family_name(family):<10}   max {o.max_err:.2e}")
    assert o.max_err < 1e-10


# ---- the adaptive product -------------------------------------------------
#
# adaptive_rt is exokal's per-cell SELECTION between the stabilized product
# and the diagonal star, carried as eta in {0, 1}, and exokal tests its
# algebra. What is tested here is the INTERFACE: that eta is DERIVED rather
# than given -- ones by default, 0 on the cells the metric-degeneracy scan
# flags at the threshold set_degeneracy_percent names -- and that the model
# reports the selection as built.
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: str(f).split(".")[-1])
def test_the_default_selection_is_the_stabilized_product(dim, family):
    # a structured annulus has no collapsed cell, so nothing is flagged and
    # eta is ones: the stabilized product everywhere
    ada = annulus_case(8, 4, dim, family, ADAPTIVE)
    stab = annulus_case(8, 4, dim, family, STABILIZED)
    print(
        f"  adaptive {dim}D {mk.family_name(family):<10} "
        f"max {ada.max_err:.6e} (stabilized {stab.max_err:.6e})"
    )
    assert abs(ada.max_err - stab.max_err) < 1e-12
    assert ada.dofs == stab.dofs


def test_a_total_selection_is_the_two_point_star():
    # on a uniform mesh every cell sits at ~100% of its node-star mean, so a
    # threshold above that flags them all: eta is zeros, and the operator is
    # diagonal_tpfa everywhere
    ada = annulus_case(8, 4, 3, mk.Family.simplex, ADAPTIVE, degeneracy=1e6)
    tpfa = annulus_case(8, 4, 3, mk.Family.simplex, TPFA)
    print(f"  adaptive 3D simplex all-flagged   max {ada.max_err:.6e} "
          f"(tpfa {tpfa.max_err:.6e})")
    assert abs(ada.max_err - tpfa.max_err) < 1e-12


# THE SCAN FLAGS THE COLLAPSED CELL AND NO OTHER: a unit cube with a slab
# 10^-6 of its height glued on top. The slab is 10^-4 % of its node-star mean
# -- under exokal's default 0.01% -- so with NO threshold given it takes the
# diagonal star and the cube keeps the stabilized product. The selection the
# model reports is the one the operator was built from, which is the eta cell
# data to write next to the solution.
def two_cells_one_collapsed(t=1e-6):
    pts = [[x, y, z] for z in (0.0, 1.0, 1.0 + t) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    hexa = lambda b: [  # noqa: E731 -- the six quads of layer pair (b, b+4)
        [b + 0, b + 1, b + 3, b + 2], [b + 4, b + 6, b + 7, b + 5],
        [b + 0, b + 4, b + 5, b + 1], [b + 2, b + 3, b + 7, b + 6],
        [b + 0, b + 2, b + 6, b + 4], [b + 1, b + 5, b + 7, b + 3],
    ]
    return mk.Mesh.from_polyhedra(pts, [hexa(0), hexa(4)])


def test_the_scan_flags_the_collapsed_cell():
    m = two_cells_one_collapsed()
    prob = mk.FlowModel(m, 3, 1.0, ADAPTIVE)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    with pytest.raises(Exception):
        prob.eta  # as built -- so not before the build
    prob.solve()
    print(f"  adaptive collapsed slab   eta = {list(prob.eta)}")
    assert list(prob.eta) == [1.0, 0.0]
    # and the constant pressure is reproduced across the seam
    assert abs(prob.cell_pressure(0) - 1.0) < 1e-10
    assert abs(prob.cell_pressure(1) - 1.0) < 1e-10

    # a wider threshold flags the cube too: the scan is the caller's dial
    prob = mk.FlowModel(m, 3, 1.0, ADAPTIVE)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.set_degeneracy_percent(1e9)
    prob.solve()
    assert list(prob.eta) == [0.0, 0.0]


def test_the_selection_defaults_to_ones_on_a_sound_mesh():
    m = mk.column(4, 3, mk.Family.cartesian, 1.0)
    prob = mk.FlowModel(m, 3, 1.0, ADAPTIVE)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.solve()
    assert list(prob.eta) == [1.0] * m.count(3)


# THE SECOND SELECTOR: conditioning. A sound cartesian column has blocks of
# modest conditioning, so a threshold above them selects nothing and one
# below them selects everything -- the two members again, reached by the other
# dial -- and the count of switched cells is reported as built.
def test_the_conditioning_selector_reaches_both_members():
    m = mk.column(4, 3, mk.Family.cartesian, 1.0)
    sp = mk.flux_cell_spectra(m, 3, ADAPTIVE, 1.0, None, None)
    worst = max(sp["cond"])
    assert worst > 1.0 and math.isfinite(worst)

    prob = mk.FlowModel(m, 3, 1.0, ADAPTIVE)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.set_cond_threshold(2.0 * worst)
    prob.solve()
    assert list(prob.eta) == [1.0] * m.count(3)
    assert prob.n_ill_conditioned == 0

    prob = mk.FlowModel(m, 3, 1.0, ADAPTIVE)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.set_cond_threshold(0.5)
    prob.solve()
    assert list(prob.eta) == [0.0] * m.count(3)
    assert prob.n_ill_conditioned == m.count(3)
    assert abs(prob.cell_pressure(0) - 1.0) < 1e-10

    # the spectra binding makes the same selection, and says how many
    sp = mk.flux_cell_spectra(m, 3, ADAPTIVE, 1.0, None, 0.5)
    assert list(sp["eta"]) == [0.0] * m.count(3)
    assert sp["n_ill_conditioned"] == m.count(3)


def test_the_conditioning_threshold_is_refused_off_the_adaptive_product():
    m = mk.column(4, 3, mk.Family.cartesian, 1.0)
    prob = mk.FlowModel(m, 3, 1.0, STABILIZED)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.set_cond_threshold(1e3)
    with pytest.raises(Exception):
        prob.solve()


# THE THRESHOLD BELONGS TO adaptive_rt: named beside any other realization it
# is refused at solve() rather than silently ignored.
def test_the_threshold_is_refused_off_the_adaptive_product():
    m = mk.column(4, 3, mk.Family.cartesian, 1.0)
    prob = mk.FlowModel(m, 3, 1.0, STABILIZED)
    prob.add_pressure(mk.boundary_facets(m, 3), 1.0)
    prob.set_degeneracy_percent(1.0)
    with pytest.raises(Exception):
        prob.solve()


def test_the_products_lay_out_different_spaces():
    bdm = column_case(6, 3, mk.Family.simplex, BDM)
    rt = column_case(6, 3, mk.Family.simplex, RT)
    mfd = column_case(6, 3, mk.Family.simplex, STABILIZED)
    print(
        f"  3D simplex column   derham {bdm.dofs}   derham_rt {rt.dofs}   "
        f"stabilized {mfd.dofs} dofs"
    )
    assert rt.dofs < bdm.dofs
    assert mfd.dofs == rt.dofs  # both are one flux per facet


# RT AND THE STABILIZED PRODUCT RETURN THE SAME SOLVED FIELD ON A SIMPLEX -- and
# they are NOT the same operator, which is the more interesting half. Their
# matrices differ by about 3%, yet the pressures agree to round-off: both spaces
# contain the constants, so both are consistent, and on this problem the
# stabilization does not reach the cell pressures. It is also the reason a
# model-level comparison must never be used to conclude two operators are the
# same -- this test would have said so, and it would have been wrong.
@pytest.mark.parametrize("nr", [8, 16])
def test_rt_and_the_stabilized_product_coincide_on_a_simplex(nr):
    rt = annulus_case(nr, nr // 2, 3, mk.Family.simplex, RT)
    mfd = annulus_case(nr, nr // 2, 3, mk.Family.simplex, STABILIZED)
    print(
        f"  annulus {rt.cells:4d} cells   RT max {rt.max_err:.6e}   "
        f"MFD max {mfd.max_err:.6e}   |diff| {abs(rt.max_err - mfd.max_err):.2e}"
    )
    assert abs(rt.max_err - mfd.max_err) < 1e-12
    assert abs(rt.rms_err - mfd.rms_err) < 1e-12


def drum(n):
    """An n-gonal prism as a single cell: n side quads and two caps, n+2 facets.

    The one family whose facet count is a free parameter, so the limit can be
    located rather than assumed.
    """
    pts = [[math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), z] for z in (0.0, 1.0)
           for i in range(n)]
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    return mk.Mesh.from_polyhedra(pts, [faces])


def builds(n, how):
    try:
        m = drum(n)
        model = mk.FlowModel(m, 3, 1.0, how)
        model.add_pressure(mk.boundary_facets(m, 3), 1.0)
        model.solve()
        return True
    except Exception:
        return False


# WHERE THE CONSISTENCY-ONLY FAMILY STOPS, and that it stops by refusing.
#
# The enrichment is not unbounded: past a cap on the facet count the cell is
# refused at once, before any search. The stabilized product stabilizes instead
# of enriching and has no such limit, which is what makes the pair the test --
# the refusal belongs to the consistency-only argument, not to the cell being
# difficult. A silent wrong answer is the one thing that would not be correct.
@pytest.mark.parametrize("n,accepted", [(11, True), (12, False), (16, False)])
def test_the_consistency_only_product_refuses_past_the_facet_limit(n, accepted):
    assert builds(n, RT) is accepted
    assert builds(n, STABILIZED) is True  # no limit: it never had to enrich
    print(f"  {n + 2:2d} facets   derham_rt {'accepts' if accepted else 'refuses'}, "
          f"stabilized_rt accepts")
