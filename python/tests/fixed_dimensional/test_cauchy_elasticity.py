"""Cauchy elasticity through the Python interface.

The Python-side mirror of tests/model/test_cauchy_elasticity.cpp: the same two
problems (a confined column with a closed form, and Lame's thick-walled tube
without one), the same three cell families, the same two products, the same
tolerances.
"""

import math

import pytest

import mimetika_cxx as mk

MU, LAM = 1.0, 1.0

DERHAM = mk.StressRealization.derham_bdm
DERHAM_RT = mk.StressRealization.derham_rt
STABILIZED = mk.StressRealization.stabilized_bdm
VEM = mk.StressRealization.stabilized_vem
DIAGONAL_VEM = mk.StressRealization.diagonal_vem
ADAPTIVE_VEM = mk.StressRealization.adaptive_vem
WEAK = mk.StressFormulation.weak_symmetry
STRONG = mk.StressFormulation.strong_symmetry
STRONG_TOTAL = mk.StressFormulation.strong_symmetry_total

FAMILIES = [mk.Family.cartesian, mk.Family.simplex, mk.Family.prism]
PRODUCTS = [DERHAM, STABILIZED]


class Outcome:
    def __init__(self):
        self.max_err = 0.0
        self.rms_err = 0.0
        self.stress_err = 0.0
        self.cells = 0
        self.dofs = 0
        self.stabilized = 0


def _material(mu=MU, lam=LAM):
    return mk.ElasticMaterial(mu, lam)


# ---- the column: confined uniaxial compression ------------------------------
#
# Rollers on the base and every side, a uniform compressive traction on top.
# Confinement removes every lateral strain by geometry alone, so elasticity
# gives the whole answer in closed form.
def column_case(n, dim, family, how, form=WEAK, degeneracy=None) -> Outcome:
    h, load = 1.0, 0.5
    m = mk.column(n, dim, family, h)
    axis = dim - 1

    loaded, confined = [], []
    for f in mk.boundary_facets(m, dim):
        z = mk.centroid(m, dim - 1, f)[axis]
        (loaded if abs(z - h) < 1e-9 else confined).append(f)

    applied = [0.0] * 9
    applied[axis * 3 + axis] = -load

    model = mk.CauchyElasticityModel(m, dim, _material(), how, form)
    model.add_traction(loaded, applied)
    model.add_free_slip(confined)
    if degeneracy is not None:
        model.set_degeneracy_percent(degeneracy)
    model.solve()

    k_oed = model.material().oedometer()
    s_exact = LAM / k_oed * (-load)
    e_exact = -load / k_oed

    out = Outcome()
    out.cells = model.n_cells
    out.dofs = model.n_dofs
    out.stabilized = model.n_stabilized
    acc = 0.0
    for e in range(out.cells):
        x = mk.centroid(m, dim, e)[axis]
        d = model.displacement(e, axis) - e_exact * x
        out.max_err = max(out.max_err, abs(d))
        acc += d * d
    out.rms_err = math.sqrt(acc / out.cells)
    # the LATERAL normal traction, on the confined facets whose normal is not the
    # column axis -- the base carries the axial reaction instead
    for f in confined:
        nrm = mk.boundary_outward_normal(m, dim, f)
        if abs(nrm[axis]) > 1e-9:
            continue
        out.stress_err = max(out.stress_err, abs(model.normal_traction(f) - s_exact))
    return out


# ---- the annulus: Lame's thick-walled tube ----------------------------------
class Lame:
    def __init__(self, A, B, mat):
        self.A, self.B, self.mat = A, B, mat

    def u_r(self, r):
        return self.A * r / (2.0 * (self.mat.lame + self.mat.shear)) + self.B / (
            2.0 * self.mat.shear * r
        )


def annulus_case(nr, nt, dim, family, how, mat=None, form=WEAK) -> Outcome:
    mat = mat if mat is not None else _material()
    a, b, hz, p_a, p_b = 1.0, 4.0, 1.0, 1.0, 0.25
    m = mk.annulus(nr, nt, dim, family, a, b, hz)

    ex = Lame(
        (a * a * p_a - b * b * p_b) / (b * b - a * a),
        (p_a - p_b) * a * a * b * b / (b * b - a * a),
        mat,
    )

    rmid = math.sqrt(a * b)
    inner, outer, sym = [], [], []
    for f in mk.boundary_facets(m, dim):
        x = mk.centroid(m, dim - 1, f)
        r = math.hypot(x[0], x[1])
        on_sym = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - hz) < 1e-8))
        )
        if on_sym:
            sym.append(f)
        elif r < rmid:
            inner.append(f)
        else:
            outer.append(f)

    si, so = [0.0] * 9, [0.0] * 9
    for k in range(3):
        si[k * 3 + k] = -p_a
        so[k * 3 + k] = -p_b

    model = mk.CauchyElasticityModel(m, dim, mat, how, form)
    model.add_traction(inner, si)
    model.add_traction(outer, so)
    model.add_free_slip(sym)
    model.solve()

    out = Outcome()
    out.cells = model.n_cells
    out.dofs = model.n_dofs
    out.stabilized = model.n_stabilized
    acc = 0.0
    for e in range(out.cells):
        x = mk.centroid(m, dim, e)
        r = math.hypot(x[0], x[1])
        # the RADIAL displacement, projected out of the cell's vector unknown
        ur = (model.displacement(e, 0) * x[0] + model.displacement(e, 1) * x[1]) / r
        d = ur - ex.u_r(r)
        out.max_err = max(out.max_err, abs(d))
        acc += d * d
    out.rms_err = math.sqrt(acc / out.cells)
    return out


# A LINEAR DISPLACEMENT IS REPRODUCED EXACTLY, on every cell type, in either
# dimension, by both products.
@pytest.mark.parametrize("how", PRODUCTS, ids=lambda r: str(r).split(".")[-1])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: str(f).split(".")[-1])
def test_the_column_reproduces_the_linear_displacement_exactly(how, dim, family):
    o = column_case(4, dim, family, how)
    print(
        f"  column  {mk.stress_realization_name(how):<14} {dim}D "
        f"{mk.family_name(family):<10} {o.cells:4d} cells {o.dofs:6d} dofs   "
        f"u {o.max_err:.2e}   sigma_lat {o.stress_err:.2e}   stab {o.stabilized}"
    )
    assert o.max_err < 1e-10
    assert o.stress_err < 1e-10
    # a simplex mesh never stabilizes, whichever product is asked for
    if family == mk.Family.simplex:
        assert o.stabilized == 0
    # and the de Rham product never stabilizes at all
    if how == DERHAM:
        assert o.stabilized == 0


# AND LAME IS APPROXIMATED AND CONVERGES. The radial solution carries 1/r^2 in
# the stress and 1/r in the displacement, so no polynomial reconstruction
# contains it: the error is a resolution, and it must fall with refinement.
@pytest.mark.parametrize("how", PRODUCTS, ids=lambda r: str(r).split(".")[-1])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: str(f).split(".")[-1])
def test_the_annulus_reproduces_lame(how, dim, family):
    coarse = annulus_case(6, 3, dim, family, how)
    fine = annulus_case(12, 6, dim, family, how)
    print(
        f"  annulus {mk.stress_realization_name(how):<14} {dim}D "
        f"{mk.family_name(family):<10} {coarse.cells:4d} -> {fine.cells:5d} cells   "
        f"u max {coarse.max_err:.2e} -> {fine.max_err:.2e}   "
        f"rms {coarse.rms_err:.2e} -> {fine.rms_err:.2e}"
    )
    assert coarse.max_err < 5e-2
    assert fine.rms_err < coarse.rms_err  # refinement helps, so it is resolution


# ON A SIMPLEX MESH THE TWO PRODUCTS ARE ONE ELEMENT, and this carries it from a
# single cell up to a solved problem. On a polytope they are genuinely different
# discretizations, which the second half measures rather than glosses.
@pytest.mark.parametrize("dim", [2, 3])
def test_the_two_products_are_one_element_on_a_simplex_mesh(dim):
    # the material must not be trivial: at lam = 0 the volumetric term is
    # switched off and agreement would say nothing about the trace pairing
    for mat in (_material(MU, 0.0), _material(MU, LAM), _material(MU, 100.0 * MU)):
        b = annulus_case(6, 3, dim, mk.Family.simplex, DERHAM, mat)
        a = annulus_case(6, 3, dim, mk.Family.simplex, STABILIZED, mat)
        print(
            f"  {dim}D simplex   nu {mat.poisson():.4f}   derham_bdm {b.max_err:.12e}"
            f"   stabilized_bdm {a.max_err:.12e}  |diff| {abs(b.max_err - a.max_err):.2e}"
        )
        assert abs(b.max_err - a.max_err) < 1e-10
        assert abs(b.rms_err - a.rms_err) < 1e-10
        assert b.stabilized == 0
        assert a.stabilized == 0

    # and on a polytopal mesh they are different discretizations
    cb = annulus_case(6, 3, dim, mk.Family.cartesian, DERHAM)
    ca = annulus_case(6, 3, dim, mk.Family.cartesian, STABILIZED)
    print(
        f"  {dim}D cartesian derham_bdm {cb.max_err:.12e}   "
        f"stabilized_bdm {ca.max_err:.12e}  |diff| {abs(cb.max_err - ca.max_err):.2e}"
    )
    assert cb.stabilized == 0  # enriched to unisolvence instead
    assert ca.stabilized == ca.cells  # stabilized on ker(N^T)
    assert abs(cb.max_err - ca.max_err) > 1e-10


# derham_rt IS REFUSED AT CONSTRUCTION. It is a sound inner product and not an
# element -- one constant traction vector per facet cannot control the rigid
# rotations across a mesh -- so the model declines it rather than assembling a
# singular saddle point.
def test_the_model_refuses_the_realization_that_is_not_an_element():
    m = mk.column(2, 3, mk.Family.simplex)
    with pytest.raises(ValueError):
        mk.CauchyElasticityModel(m, 3, _material(), DERHAM_RT)


# NEITHER PRODUCT LOCKS. As nu -> 1/2 the volumetric term dominates the
# compliance, which is where one would expect an approximation to tell. It does
# not: this is a MIXED method, and locking is a pathology of displacement-based
# formulations. So the test is a comparison rather than a threshold.
@pytest.mark.parametrize("how", PRODUCTS, ids=lambda r: str(r).split(".")[-1])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "family", [mk.Family.simplex, mk.Family.cartesian], ids=lambda f: str(f).split(".")[-1]
)
def test_neither_product_locks_as_the_material_becomes_incompressible(how, dim, family):
    nu = 0.4999
    stiff = _material(MU, 2.0 * MU * nu / (1.0 - 2.0 * nu))
    soft_c = annulus_case(6, 3, dim, family, how)
    hard_c = annulus_case(6, 3, dim, family, how, stiff)
    hard_f = annulus_case(12, 6, dim, family, how, stiff)
    rate = math.log2(hard_c.rms_err / hard_f.rms_err)
    print(
        f"  {dim}D {mk.family_name(family):<9} {mk.stress_realization_name(how):<14}  "
        f"nu 0.25 rms {soft_c.rms_err:.3e}   nu 0.4999 rms {hard_c.rms_err:.3e} -> "
        f"{hard_f.rms_err:.3e}  rate {rate:.2f}"
    )
    # no degradation at the limit, and the rate holds
    assert hard_c.rms_err < 1.1 * soft_c.rms_err
    assert rate > 1.5


# ---- the strongly-symmetric vem family ---------------------------------------
#
# exokal's rigid-motion ansatz (Dassi-Lovadina-Visinoni): six traction moments
# per facet carried whole, the displacement as the six rigid-motion
# coefficients per cell, no rotation multiplier -- symmetry lives in the
# reconstruction space. exokal tests the operators; what is tested here is the
# INTERFACE: the layout, the boundary conditions written in the six-slot facet
# basis, the readbacks, and adaptive_vem's derived selection.

STRONG_FAMILIES = [mk.Family.cartesian, mk.Family.simplex, mk.Family.prism]


@pytest.mark.parametrize("form", [STRONG, STRONG_TOTAL],
                         ids=["strong_symmetry", "strong_symmetry_total"])
@pytest.mark.parametrize("family", STRONG_FAMILIES, ids=lambda f: str(f).split(".")[-1])
def test_the_strong_column_reproduces_the_confined_solution_exactly(family, form):
    o = column_case(4, 3, family, VEM, form)
    print(
        f"  column  stabilized_vem {str(form).split('.')[-1]:<22} "
        f"{mk.family_name(family):<10} {o.cells:4d} cells {o.dofs:6d} dofs   "
        f"max {o.max_err:.2e}   stress {o.stress_err:.2e}"
    )
    assert o.max_err < 1e-12
    assert o.stress_err < 1e-12


def test_the_diagonal_vem_star_is_exact_where_the_column_is_orthogonal():
    # the cartesian column's cells are face-orthogonal with isotropic second
    # moment, which is where the two-point member's resultants are consistent
    o = column_case(4, 3, mk.Family.cartesian, DIAGONAL_VEM, STRONG_TOTAL)
    print(f"  column  diagonal_vem cartesian   max {o.max_err:.2e}")
    assert o.max_err < 1e-12


def test_the_adaptive_vem_selection_matches_its_members_at_the_ends():
    stab = column_case(4, 3, mk.Family.simplex, VEM, STRONG_TOTAL)
    ones = column_case(4, 3, mk.Family.simplex, ADAPTIVE_VEM, STRONG_TOTAL)
    star = column_case(4, 3, mk.Family.cartesian, DIAGONAL_VEM, STRONG_TOTAL)
    allf = column_case(4, 3, mk.Family.cartesian, ADAPTIVE_VEM, STRONG_TOTAL, degeneracy=1e9)
    print(
        f"  adaptive_vem ends   default max {ones.max_err:.2e} (stab {stab.max_err:.2e})   "
        f"all-flagged max {allf.max_err:.2e} (star {star.max_err:.2e})"
    )
    assert abs(ones.max_err - stab.max_err) < 1e-14
    assert abs(allf.max_err - star.max_err) < 1e-14
    assert ones.dofs == stab.dofs  # same mesh, same layout
    assert allf.dofs == star.dofs


def test_the_adaptive_vem_selection_is_reported_as_built():
    m = mk.column(3, 3, mk.Family.cartesian, 1.0)
    model = mk.CauchyElasticityModel(m, 3, _material(), ADAPTIVE_VEM, STRONG_TOTAL)
    model.add_traction(mk.boundary_facets(m, 3), [0.0] * 9)
    with pytest.raises(Exception):
        model.eta  # as built -- so not before the build
    model.solve()
    assert list(model.eta) == [1.0] * model.n_cells


# THE SECOND SELECTOR, BY CONDITIONING, reaches the same two ends: a threshold
# above every block's conditioning selects nothing, one below selects all.
def test_the_adaptive_vem_conditioning_selector_reaches_both_members():
    m = mk.column(4, 3, mk.Family.cartesian, 1.0)
    sp = mk.stress_cell_spectra(m, 3, ADAPTIVE_VEM, STRONG_TOTAL, MU, LAM, None, None)
    worst = max(sp["cond"])
    assert worst > 1.0 and math.isfinite(worst)

    # the confined column, so the all-diagonal solve is well posed (a zero
    # traction everywhere would leave the rigid modes free)
    loaded, confined = [], []
    for f in mk.boundary_facets(m, 3):
        (loaded if abs(mk.centroid(m, 2, f)[2] - 1.0) < 1e-9 else confined).append(f)
    applied = [0.0] * 9
    applied[8] = -0.5

    def confined_model():
        model = mk.CauchyElasticityModel(m, 3, _material(), ADAPTIVE_VEM, STRONG_TOTAL)
        model.add_traction(loaded, applied)
        model.add_free_slip(confined)
        return model

    model = confined_model()
    model.set_cond_threshold(2.0 * worst)
    model.solve()
    assert list(model.eta) == [1.0] * model.n_cells
    assert model.n_ill_conditioned == 0

    model = confined_model()
    model.set_cond_threshold(0.5)
    model.solve()
    assert list(model.eta) == [0.0] * model.n_cells
    assert model.n_ill_conditioned == model.n_cells

    sp = mk.stress_cell_spectra(m, 3, ADAPTIVE_VEM, STRONG_TOTAL, MU, LAM, None, 0.5)
    assert list(sp["eta"]) == [0.0] * model.n_cells
    assert sp["n_ill_conditioned"] == model.n_cells


# THE REFUSALS: the symmetry axis is one decision, the diagonal members demand
# the total pressure, the ansatz is three-dimensional, and the threshold
# belongs to adaptive_vem. Each is an exception where the choice was made.
@pytest.mark.parametrize(
    "how,form,dim",
    [
        (VEM, WEAK, 3),                                       # axis disagreement
        (DERHAM, STRONG, 3),                                  # and the reverse
        (DIAGONAL_VEM, STRONG, 3),                            # diagonal needs total
        (ADAPTIVE_VEM, STRONG, 3),                            # the blend inherits it
        (VEM, STRONG, 2),                                     # 3D construction
    ],
    ids=["strong-under-weak", "weak-under-strong", "diagonal-plain", "adaptive-plain", "2d"],
)
def test_the_strong_wiring_refuses_what_it_must(how, form, dim):
    m = mk.column(2, dim, mk.Family.cartesian, 1.0)
    with pytest.raises(Exception):
        mk.CauchyElasticityModel(m, dim, _material(), how, form)


def test_the_threshold_is_refused_off_the_adaptive_vem_product():
    o = None
    with pytest.raises(Exception):
        o = column_case(3, 3, mk.Family.cartesian, VEM, STRONG, degeneracy=1.0)
    assert o is None
