r"""The four-field split: exact congruence, honest solid pressure, sparsity.

The four-field formulation is an *algebraic rearrangement*, not a new
discretisation: eliminating the solid pressure must reproduce the three-field
compliance **exactly**.  So the headline tests solve the same problem both ways
-- with data chosen *outside* the reconstruction space, so the discrete
solutions carry genuine discretisation error -- and demand agreement to solver
precision.  Two exact solutions agreeing would prove nothing.

The remaining claims are the ones the split exists for: ``p_s`` equals the
discrete trace ``tr_h(sigma)/d`` (the first invariant is a primary unknown),
the lumped inner product stays **diagonal** in the assembled system (the
three-field assembly provably fills it in), and in poromechanics the pore
pressure reaches the mechanics only through a **diagonal** cell--cell block --
the stress row never sees it.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.assembly.four_field import FourFieldElasticity, FourFieldPoroMechanics
from mimetika.assembly.mixed import MixedElasticity
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.materials import Material
from mimetika.mesh import (
    structured_box,
    structured_quads,
    structured_tets,
    structured_triangles,
)
from mimetika.operators.lumped import LumpedDeviatoricStress

MU, LAM = 1.3, 2.7
EXACT = {"method": "direct"}

# linear displacement (for patch tests: inside the reconstruction space)
U_A = np.array([0.31, -0.42, 0.17])
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])


def linear_displacement(x):
    return U_A + np.atleast_2d(x) @ U_B.T


def exact_stress(d, mu=MU, lam=LAM):
    eps = 0.5 * (U_B + U_B.T)[:d, :d]
    return 2 * mu * eps + lam * np.trace(eps) * np.eye(d)


def quadratic_displacement(x):
    """Outside every reconstruction space -- forces discretisation error."""
    x = np.atleast_2d(x)
    return np.column_stack(
        [
            0.3 * x[:, 0] ** 2 + 0.1 * x[:, 1] - 0.2 * x[:, 0] * x[:, 2],
            -0.25 * x[:, 1] ** 2 + 0.4 * x[:, 0] * x[:, 1],
            0.15 * x[:, 2] ** 2 + 0.05 * x[:, 0] - 0.1 * x[:, 1] * x[:, 2],
        ]
    )


def body_force(x):
    return np.tile([0.2, -0.4, 0.1], (len(np.atleast_2d(x)), 1))


# (mesh factory, space): the lumped space only on meshes whose centroids form
# an orthogonal complex; AFW on simplices (no stabilization) and on polytopes
# (stabilization active), so the congruence is tested with and without it.
CASES = {
    "afw-tet": (lambda: structured_tets(1, 1, 1), "afw"),
    "afw-hex": (lambda: structured_box(2, 2, 2), "afw"),
    "afw-tri": (lambda: structured_triangles(2, 2), "afw"),
    "lumped-quad": (lambda: structured_quads(3, 3), "lumped"),
    "lumped-hex": (lambda: structured_box(2, 2, 2), "lumped"),
}


def make_inner(mesh, space, material=None):
    if space == "afw":
        return None  # MixedElasticity's default
    if material is not None:
        return LumpedDeviatoricStress(mesh, material=material)
    return LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)


def solid_pressure_from_trace(inner, stress):
    """``p_s = tr_h(sigma)/d`` straight from the volumetric coupling."""
    W, _ = inner.volumetric_operator()
    d = inner.mesh.dim
    vol = inner.mesh.geometry.measure(d)
    return 2.0 * inner._mu * (W @ stress) / (d * vol)


# -- elasticity: congruence and the honest unknown ------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_four_field_matches_three_field_elasticity(name):
    """Same (sigma, u, s) as the three-field solve, on a *non-exact* solution."""
    make, space = CASES[name]
    mesh = make()
    three = MixedElasticity(mesh, MU, LAM, inner=make_inner(mesh, space))
    four = FourFieldElasticity(mesh, MU, LAM, inner=make_inner(mesh, space))

    kwargs = dict(dirichlet=quadratic_displacement, body_force=body_force)
    sol3 = three.solve(**kwargs, **EXACT)
    sol4 = four.solve(**kwargs, **EXACT)

    for field in ("stress", "displacement", "rotation"):
        assert np.allclose(sol4[field], sol3[field], rtol=1e-8, atol=1e-9), field


@pytest.mark.parametrize("name", list(CASES))
def test_solid_pressure_is_the_discrete_trace(name):
    """Row two evaluates ``p_s = tr_h(sigma)/d`` -- the unknown is honest."""
    make, space = CASES[name]
    mesh = make()
    four = FourFieldElasticity(mesh, MU, LAM, inner=make_inner(mesh, space))
    sol = four.solve(
        dirichlet=quadratic_displacement, body_force=body_force, **EXACT
    )
    expected = solid_pressure_from_trace(four.inner, sol["stress"])
    assert np.allclose(sol["solid_pressure"], expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", list(CASES))
def test_four_field_patch_test(name):
    """A linear displacement is reproduced exactly, ``p_s`` included."""
    make, space = CASES[name]
    mesh = make()
    d = mesh.dim
    four = FourFieldElasticity(mesh, MU, LAM, inner=make_inner(mesh, space))
    sol = four.solve(dirichlet=linear_displacement, **EXACT)

    assert np.allclose(
        sol["displacement"],
        four.interpolate_displacement(linear_displacement),
        atol=1e-9,
    )
    p_exact = np.trace(exact_stress(d)) / d
    assert np.allclose(sol["solid_pressure"], p_exact, atol=1e-9)

    I1, J2 = four.stress_invariants(sol)
    dev = exact_stress(d) - p_exact * np.eye(d)
    assert np.allclose(I1, d * p_exact, atol=1e-8)
    assert np.allclose(J2, 0.5 * (dev * dev).sum(), atol=1e-8)


def test_lumped_inner_product_stays_diagonal():
    """The whole point: four-field keeps M diagonal; three-field cannot."""
    mesh = structured_quads(3, 3)
    four = FourFieldElasticity(
        mesh, inner=LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    )
    M4, _, _ = four.assemble_operators()
    assert (M4 - sp.diags(M4.diagonal())).nnz == 0

    three = MixedElasticity(
        mesh, inner=LumpedDeviatoricStress(mesh, mu=MU, lam=LAM)
    )
    M3, _, _ = three.assemble_operators()
    assert (M3 - sp.diags(M3.diagonal())).nnz > 0  # Woodbury fold-in fills in


def test_nu_zero_is_rejected():
    """At ``nu = 0`` the solid-pressure row would be identically zero."""
    mesh = structured_quads(2, 2)
    four = FourFieldElasticity(mesh, mu=1.0, lam=0.0)  # lam = 0  <=>  nu = 0
    with pytest.raises(ValueError, match="four-field"):
        four.assemble()


# -- poromechanics: congruence and the diagonal Biot block ----------------------

POISSON = [0.15, 0.3, 0.49]


def material_for(nu):
    return Material(
        shear_modulus=1.0, poisson=nu, biot=0.9, inverse_biot_modulus=0.1
    )


def poromech_pair(name, nu):
    make, space = CASES[name]
    material = material_for(nu)
    # each problem gets its own mesh instance; the inner must see *that* mesh
    mesh5 = make()
    five = PoroMechanics(
        mesh5, material, stress_inner=make_inner(mesh5, space, material)
    )
    mesh4 = make()
    four = FourFieldPoroMechanics(
        mesh4, material, stress_inner=make_inner(mesh4, space, material)
    )
    return five, four


def pressure_bc(x):
    x = np.atleast_2d(x)
    return 1.0 + 0.5 * x[:, 0] - 0.3 * x[:, 1]


def fluid_source(x):
    x = np.atleast_2d(x)
    return 0.2 + 0.1 * x[:, 0]


@pytest.mark.parametrize("name", ["afw-tet", "lumped-quad", "lumped-hex"])
@pytest.mark.parametrize("nu", POISSON)
def test_four_field_poromechanics_matches_five_field(name, nu):
    """Identical (sigma, u, s, q, p) over two backward-Euler steps."""
    five, four = poromech_pair(name, nu)
    kwargs = dict(
        dt=0.05,
        dirichlet=quadratic_displacement,
        body_force=body_force,
        pressure_bc=pressure_bc,
        source=fluid_source,
    )
    sol5 = five.solve(**kwargs, **EXACT)
    sol4 = four.solve(**kwargs, **EXACT)
    for field in ("stress", "displacement", "rotation", "flux", "pressure"):
        assert np.allclose(sol4[field], sol5[field], rtol=1e-7, atol=1e-9), field

    # a second step exercises the previous-state terms
    sol5b = five.solve(previous=sol5, **kwargs, **EXACT)
    sol4b = four.solve(previous=sol4, **kwargs, **EXACT)
    for field in ("stress", "displacement", "rotation", "flux", "pressure"):
        assert np.allclose(sol4b[field], sol5b[field], rtol=1e-7, atol=1e-9), field

    # and the reported solid pressure is the honest trace throughout
    expected = solid_pressure_from_trace(four.mechanics.inner, sol4b["stress"])
    assert np.allclose(sol4b["solid_pressure"], expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("name", ["afw-tet", "lumped-quad"])
def test_quasi_steady_pressure_data_matches(name):
    """``dt = None``: given pressure enters only the diagonal ``p_s`` row."""
    five, four = poromech_pair(name, 0.3)

    def pressure(x):
        x = np.atleast_2d(x)
        return -2.0 + x[:, 0] + 0.5 * x[:, 1]

    kwargs = dict(dt=None, dirichlet=quadratic_displacement, pressure=pressure)
    sol5 = five.solve(**kwargs, **EXACT)
    sol4 = four.solve(**kwargs, **EXACT)
    for field in ("stress", "displacement", "rotation"):
        assert np.allclose(sol4[field], sol5[field], rtol=1e-7, atol=1e-9), field
    expected = solid_pressure_from_trace(four.mechanics.inner, sol4["stress"])
    assert np.allclose(sol4["solid_pressure"], expected, rtol=1e-7, atol=1e-9)


def test_biot_coupling_is_diagonal():
    """The stress row never sees ``p_f``; the ``(p_s, p_f)`` block is diagonal."""
    mesh = structured_quads(3, 3)
    material = material_for(0.3)
    four = FourFieldPoroMechanics(
        mesh, material, stress_inner=LumpedDeviatoricStress(mesh, material=material)
    )
    S, _, _ = four.assemble(dt=0.1)
    n1 = four.n_stress
    nc = four.n_cells
    n4 = four._flux_offset + four.n_flux

    assert S[:n1, n4:].nnz == 0  # no trace operator in the stress row
    block = S[n1 : n1 + nc, n4:]
    assert (block - sp.diags(np.asarray(block.diagonal()).ravel())).nnz == 0
    # and the mechanics inner product is still diagonal in the global matrix
    M = S[:n1, :n1]
    assert (M - sp.diags(np.asarray(M.diagonal()).ravel())).nnz == 0
