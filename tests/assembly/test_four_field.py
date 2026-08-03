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
from mimetika.assembly.mixed import MixedElasticity, boundary_facets
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


# -- robustness: material contrast and the incompressible limit (mechanics) -----
#
# The split moves the volumetric compliance onto ``p_s``, so these regimes are
# where it could fail *differently* from the three-field form: a modulus
# contrast scales ``Gamma`` and ``B`` per cell, and ``nu -> 1/2`` sends the
# ``p_s`` row to its ``gamma = -1`` limit.  Exact uniform/piecewise states make
# any weakness visible with no discretisation error to hide behind.


def layered(mesh, values, axis=1, split=0.5):
    """Per-cell array taking ``values[0]`` below ``split`` and ``values[1]`` above."""
    centroids = mesh.geometry.centroids(mesh.dim)
    return np.where(centroids[:, axis] < split, values[0], values[1])


def checkerboard(mesh, values, split=0.5):
    """Per-cell array alternating between ``values`` on a checkerboard pattern."""
    centroids = mesh.geometry.centroids(mesh.dim)[:, : mesh.dim]
    parity = np.floor(centroids / split).astype(int).sum(axis=1) % 2
    return np.where(parity == 0, values[0], values[1])


def poisson_for(shear, inverse_modulus, dim):
    """The ``nu`` pairing with ``shear`` to give a prescribed bulk compliance.

    Holding ``K`` fixed while raising ``mu`` drives ``nu`` towards ``-1`` --
    auxetic, so these cells have ``a < 0``: exactly the sign the four-field
    split must tolerate (only ``a = 0`` is degenerate).
    """
    t = 2.0 * np.asarray(shear, dtype=float) * inverse_modulus
    c = (1.0 - t) / (t * dim)
    return c / (1.0 + 2.0 * c)


def layered_shear(mesh, shear, amplitude=0.5, split=0.5):
    """Exact pure-shear state of a two-layer medium stacked along ``y``.

    The stress is uniform and **traceless**, so the exact solid pressure is
    zero -- the contrast must not leak into ``p_s``.
    """
    tensor = np.zeros((3, 3))
    tensor[0, 1] = tensor[1, 0] = amplitude
    lower, upper = float(shear[0]), float(shear[1])

    def displacement(x):
        x = np.atleast_2d(x)
        y = x[:, 1]
        below = amplitude * y / lower
        above = amplitude * split / lower + amplitude * (y - split) / upper
        out = np.zeros((len(x), 3))
        out[:, 0] = np.where(y < split, below, above)
        return out

    def stress(x):
        return np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))

    return stress, displacement


def contrast_problem(mesh, material, space):
    from mimetika.operators.elasticity import ElasticityInnerProduct

    inner = (
        LumpedDeviatoricStress(mesh, material=material)
        if space == "lumped"
        else ElasticityInnerProduct(mesh, material=material)
    )
    return FourFieldElasticity(mesh, inner=inner)


CONTRAST_CASES = {
    "afw-hex": (lambda: structured_box(2, 2, 2), "afw"),
    "lumped-quad": (lambda: structured_quads(4, 4), "lumped"),
}


@pytest.mark.parametrize("name", list(CONTRAST_CASES))
@pytest.mark.parametrize("contrast", [1e3, 1e6, 1e9])
def test_layered_shear_survives_the_modulus_contrast(name, contrast):
    """Exact to the accuracy the conditioning allows; ``p_s`` stays at zero.

    The attainable accuracy degrades like ``contrast * eps_machine`` -- the
    optimal double-precision rate at a condition number proportional to the
    jump -- and the traceless state must not leak into the solid pressure.
    """
    make, space = CONTRAST_CASES[name]
    mesh = make()
    shear = (1.0, contrast)
    material = Material(shear_modulus=layered(mesh, shear), poisson=0.3)
    problem = contrast_problem(mesh, material, space)
    stress, displacement = layered_shear(mesh, shear)

    sol = problem.solve(dirichlet=displacement, **EXACT)
    exact = problem.interpolate_stress(stress)
    scale = np.abs(exact).max()
    error = np.abs(sol["stress"] - exact).max() / scale
    assert error < 1e-13 * contrast + 1e-12, f"contrast {contrast:g}: {error:.3e}"
    assert np.abs(sol["solid_pressure"]).max() < (1e-13 * contrast + 1e-11) * scale


@pytest.mark.parametrize("name", list(CONTRAST_CASES))
@pytest.mark.parametrize("contrast", [1e2, 1e4, 1e6])
def test_checkerboard_shear_contrast_puts_the_hydrostatic_load_on_p_s(
    name, contrast
):
    """``mu`` alternating cell by cell, ``K`` matched: the load lives on ``p_s``.

    Matching the bulk compliance forces the stiff cells auxetic (``a < 0``), so
    this doubles as the sign test for the ``p_s`` diagonal.  A hydrostatic
    stress engages only the volumetric compliance, identical in every cell, so
    the state is uniform, and the four-field split must report it entirely
    through ``p_s = tr(sigma)/d`` with the facet field carrying no deviation.
    """
    make, space = CONTRAST_CASES[name]
    mesh = make()
    d = mesh.dim
    shear = checkerboard(mesh, (1.0, contrast))
    # the matched bulk compliance must keep every nu inside (-1, 1/2) *and*
    # away from the degenerate nu = 0; in 2D (where inv_modulus = (1-2nu)/2mu)
    # that means scaling it with the contrast -- the stiff cell then sits at
    # nu = -1/2 and the soft one just under 1/2
    target = 1.0 / 3.0 if d == 3 else 1.0 / contrast
    material = Material(
        shear_modulus=shear, poisson=poisson_for(shear, target, d)
    )
    inverse = material.inverse_modulus(d)
    assert np.allclose(inverse, inverse[0], rtol=1e-12)  # the premise
    assert np.any(material.compliance_coefficient(d) < 0)  # really auxetic

    problem = contrast_problem(mesh, material, space)
    amplitude = 2.0
    tensor = np.zeros((3, 3))
    tensor[:d, :d] = amplitude * np.eye(d)
    strain = amplitude * inverse[0] * np.eye(3)  # C^{-1}(p I) = p inv_modulus I

    sol = problem.solve(
        dirichlet=lambda x: np.atleast_2d(x) @ strain.T, **EXACT
    )
    exact = problem.interpolate_stress(
        lambda x: np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))
    )
    scale = np.abs(exact).max()
    assert np.abs(sol["stress"] - exact).max() / scale < 1e-13 * contrast + 1e-11
    assert np.allclose(sol["solid_pressure"], amplitude, rtol=1e-13 * contrast + 1e-10)


def boundary_face(mesh, value=1.0, axis=1):
    """Boundary facets on the plane ``x[axis] == value``."""
    centroids = mesh.geometry.centroids(mesh.dim - 1)
    return [
        f
        for f in boundary_facets(mesh)
        if abs(centroids[f][axis] - value) < 1e-12
    ]


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("nu", [0.45, 0.49999, 0.5])
def test_the_incompressible_limit_is_exact(name, nu):
    """``nu -> 1/2`` including the limit itself: constant stress, exact ``p_s``.

    At ``nu = 1/2`` the ``p_s`` row sits at ``gamma = -1``, ``B = d|E|/2mu`` --
    nothing degenerates -- but the hydrostatic level is only determined once a
    traction is prescribed somewhere, so one face carries the stress data.
    """
    make, space = CASES[name]
    mesh = make()
    d = mesh.dim
    material = Material(shear_modulus=MU, poisson=nu)
    problem = contrast_problem(mesh, material, space)

    tensor = np.zeros((3, 3))
    tensor[:d, :d] = np.diag([3.0, -1.0, -2.0])[:d, :d] + 1.5 * np.eye(d)
    a = float(material.compliance_coefficient(d))
    strain = np.zeros((3, 3))
    strain[:d, :d] = (
        tensor[:d, :d] - a * np.trace(tensor[:d, :d]) * np.eye(d)
    ) / (2.0 * MU)

    def stress(x):
        return np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))

    sol = problem.solve(
        dirichlet=lambda x: np.atleast_2d(x) @ strain.T,
        traction=stress,
        traction_facets=boundary_face(mesh),
        **EXACT,
    )
    exact = problem.interpolate_stress(stress)
    scale = np.abs(exact).max()
    assert np.abs(sol["stress"] - exact).max() < 1e-10 * scale
    assert np.allclose(
        sol["solid_pressure"], np.trace(tensor[:d, :d]) / d, atol=1e-10 * scale
    )


def test_contrast_next_to_incompressible_cells():
    """A stiff layer at ``nu = 1/2`` against a soft compressible one.

    Pure shear is traceless, so the answer is independent of ``nu`` -- any
    sensitivity of the split to the incompressible cells is numerical.
    """
    mesh = structured_box(2, 2, 2)
    contrast = 1e8
    shear = (1.0, contrast)
    material = Material(
        shear_modulus=layered(mesh, shear), poisson=layered(mesh, (0.15, 0.5))
    )
    problem = contrast_problem(mesh, material, "afw")
    stress, displacement = layered_shear(mesh, shear)

    sol = problem.solve(dirichlet=displacement, **EXACT)
    exact = problem.interpolate_stress(stress)
    scale = np.abs(exact).max()
    assert np.abs(sol["stress"] - exact).max() / scale < 1e-13 * contrast + 1e-12
    assert np.abs(sol["solid_pressure"]).max() < (1e-13 * contrast + 1e-11) * scale


# -- robustness: incompressible fluid and the undrained regime (six-field) ------


UNDRAINED_CASES = ["afw-tet", "lumped-quad", "lumped-hex"]


@pytest.mark.parametrize("name", UNDRAINED_CASES)
@pytest.mark.parametrize("nu", [0.3, 0.49999])
def test_the_undrained_incompressible_fluid_response_is_exact(name, nu):
    """Sealed boundary, ``1/M = 0``: one step is the exact undrained state.

    With every facet sealed and zero previous fluid content, the first backward
    Euler step enforces ``alpha div u + p/M = 0`` -- for an incompressible
    fluid, ``div u = 0`` exactly, and the closed form is Skempton ``B = 1``:

        ``p = -tr(sigma)/(d alpha)`` ,   ``eps = C^{-1}(sigma + alpha p I)`` .

    The state is uniform, so the discrete solution must reproduce it exactly at
    any time step -- including ``nu = 0.49999``, where the storage
    ``S = d alpha^2 inv_modulus`` is tiny but the response is unchanged.
    """
    make, space = CASES[name]
    mesh = make()
    d = mesh.dim
    alpha = 0.9
    material = Material(
        shear_modulus=1.0, poisson=nu, biot=alpha, inverse_biot_modulus=0.0
    )
    inner = make_inner(mesh, space, material)
    problem = FourFieldPoroMechanics(mesh, material, stress_inner=inner)

    total = np.zeros((3, 3))
    total[:d, :d] = np.diag([3.0, -1.0, -2.0])[:d, :d] + 2.0 * np.eye(d)
    trace = np.trace(total[:d, :d])
    p_undrained = -trace / (d * alpha)

    effective = total[:d, :d] + alpha * p_undrained * np.eye(d)
    a = float(material.compliance_coefficient(d))
    strain = (effective - a * np.trace(effective) * np.eye(d)) / 2.0
    assert abs(np.trace(strain)) < 1e-12  # the undrained state is isochoric
    grad = np.zeros((3, 3))
    grad[:d, :d] = strain

    def stress(x):
        return np.broadcast_to(total, (len(np.atleast_2d(x)), 3, 3))

    sol = problem.solve(
        dt=0.37,  # any step: the sealed uniform state is stationary
        dirichlet=lambda x: np.atleast_2d(x) @ grad.T,
        traction=stress,
        traction_facets=boundary_face(mesh),
        no_flow=boundary_facets(mesh),
        **EXACT,
    )
    exact = problem.mechanics.interpolate_stress(stress)
    scale = np.abs(exact).max()
    assert np.allclose(sol["pressure"], p_undrained, rtol=1e-9)
    assert np.abs(sol["flux"]).max() < 1e-9 * scale
    assert np.abs(sol["stress"] - exact).max() < 1e-9 * scale
    assert np.allclose(sol["solid_pressure"], trace / d, atol=1e-9 * scale)
    assert np.abs(problem.volumetric_strain(sol)).max() < 1e-11 * scale


def test_the_doubly_incompressible_transient_stays_nonsingular():
    """``nu = 1/2`` and ``1/M = 0`` together: every diagonal the pressure has
    vanishes (``delta = Delta = S = 0``), leaving pure constraints -- the system
    must remain a well-posed saddle point."""
    mesh = structured_quads(2, 2)
    material = Material(
        shear_modulus=1.0, poisson=0.5, biot=0.9, inverse_biot_modulus=0.0
    )
    problem = FourFieldPoroMechanics(mesh, material)

    tensor = np.diag([3.0, -1.0, 0.0])

    def stress(x):
        return np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))

    matrix, _, _ = problem.assemble(
        dt=0.1,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        traction=stress,
        traction_facets=boundary_face(mesh),
    )
    singular_values = np.linalg.svd(matrix.toarray(), compute_uv=False)
    assert singular_values[-1] > 1e-8 * singular_values[0]


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
