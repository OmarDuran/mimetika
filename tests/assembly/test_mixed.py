"""Global mixed (saddle-point) problems.

The headline check is the **global patch test**: because the local inner
products satisfy strong consistency ``M N = R``, a solution that lies in the
reconstruction space -- a linear potential, or a linear displacement -- must be
reproduced *exactly* by the global solve, on any mesh.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from mimetika.assembly.local import skew_generators
from mimetika.assembly.mixed import (
    MixedElasticity,
    MixedPoisson,
    boundary_facets,
    discrete_divergence,
)
from mimetika.mesh import structured_box, structured_tets

MU, LAM = 1.3, 2.7
K_ANISO = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])

# linear scalar field  p = P_A + P_B . x
P_A, P_B = 0.37, np.array([0.8, -1.3, 0.55])
# linear vector field  u = U_A + U_B x
U_A = np.array([0.31, -0.42, 0.17])
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])

MESHES = [
    ("hex-1", structured_box(1, 1, 1)),
    ("hex-2", structured_box(2, 2, 2)),
    ("tet-1", structured_tets(1, 1, 1)),
    ("tet-2", structured_tets(2, 2, 2)),
]
IDS = [name for name, _ in MESHES]
ONLY = [m for _, m in MESHES]


def potential(x):
    return P_A + np.atleast_2d(x) @ P_B


def displacement(x):
    return U_A + np.atleast_2d(x) @ U_B.T


def grad_displacement(x):
    return np.broadcast_to(U_B, (len(np.atleast_2d(x)), 3, 3))


def exact_stress():
    eps = 0.5 * (U_B + U_B.T)
    return 2 * MU * eps + LAM * np.trace(eps) * np.eye(3)


def stress_field(x):
    return np.broadcast_to(exact_stress(), (len(np.atleast_2d(x)), 3, 3))


def flux_field(x):
    return np.broadcast_to(-K_ANISO @ P_B, (len(np.atleast_2d(x)), 3))


# -- the discrete divergence is topological -----------------------------------


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_discrete_divergence_matches_incidence(mesh):
    B = discrete_divergence(mesh)
    inc = mesh.complex.boundary_matrix(mesh.dim)
    expected = inc.T @ sp.diags(mesh.geometry.measure(mesh.dim - 1))
    assert (abs(B - expected) > 1e-14).nnz == 0


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_discrete_divergence_of_constant_flux_vanishes(mesh):
    """Gauss: a constant field has zero net flux through a closed cell."""
    B = discrete_divergence(mesh)
    problem = MixedPoisson(mesh, K=K_ANISO)
    assert np.allclose(B @ problem.interpolate_flux(flux_field), 0.0, atol=1e-12)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_boundary_facets_form_a_closed_surface(mesh):
    bf = boundary_facets(mesh)
    n = mesh.geometry.facet_normals()[bf]
    area = mesh.geometry.measure(2)[bf]
    # the unit cube: 6 unit-area sides, and the outward normals cancel
    assert np.isclose(area.sum(), 6.0)
    assert np.allclose(np.abs((area[:, None] * n).sum(0)), 0.0, atol=1e-12)


# -- global patch test: mixed Poisson -----------------------------------------


@pytest.mark.parametrize("basis", ["const", "rt0"])
@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_poisson_reproduces_linear_potential(mesh, basis):
    problem = MixedPoisson(mesh, K=K_ANISO, basis=basis)
    sol = problem.solve(source=None, dirichlet=potential)
    assert np.allclose(
        sol["pressure"], problem.interpolate_pressure(potential), atol=1e-10
    )
    assert np.allclose(sol["flux"], problem.interpolate_flux(flux_field), atol=1e-10)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_poisson_constant_solution(mesh):
    problem = MixedPoisson(mesh, K=K_ANISO)
    sol = problem.solve(dirichlet=lambda x: np.full(len(np.atleast_2d(x)), 3.25))
    assert np.allclose(sol["flux"], 0.0, atol=1e-10)
    assert np.allclose(sol["pressure"], 3.25, atol=1e-10)


def test_poisson_system_is_a_symmetric_saddle_point():
    """[[M, -B^T], [-B, 0]] is symmetric indefinite, as MINRES/fieldsplit need."""
    problem = MixedPoisson(structured_box(2, 2, 2))
    A, rhs = problem.assemble(dirichlet=potential)
    n = problem.n_flux + problem.n_pressure
    assert A.shape == (n, n) and rhs.shape == (n,)
    assert (abs(A - A.T) > 1e-12).nnz == 0
    ev = np.linalg.eigvalsh(A.toarray())
    assert ev.min() < 0 < ev.max()  # genuinely indefinite


def test_elasticity_system_is_symmetric():
    problem = MixedElasticity(structured_box(1, 1, 1))
    S, _ = problem.assemble(dirichlet=displacement)
    assert (abs(S - S.T) > 1e-9).nnz == 0


@pytest.mark.parametrize("method", ["direct", "minres"])
def test_poisson_solver_methods_agree(method):
    """MINRES with the block preconditioner reproduces the direct solve."""
    problem = MixedPoisson(structured_box(2, 2, 2), K=K_ANISO)
    sol = problem.solve(dirichlet=potential, backend="scipy", method=method)
    assert np.allclose(
        sol["pressure"], problem.interpolate_pressure(potential), atol=1e-8
    )


# -- global patch test: mixed elasticity --------------------------------------


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_elasticity_reproduces_linear_displacement(mesh):
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    sol = problem.solve(body_force=None, dirichlet=displacement)
    assert np.allclose(
        sol["displacement"], problem.interpolate_displacement(displacement), atol=1e-9
    )
    assert np.allclose(
        sol["stress"], problem.interpolate_stress(stress_field), atol=1e-8
    )
    assert np.allclose(
        sol["rotation"], problem.interpolate_rotation(grad_displacement), atol=1e-9
    )


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_elasticity_rigid_body_motion_has_no_stress(mesh):
    """Translation + infinitesimal rotation must produce zero stress."""
    W = np.array([[0.0, 0.7, -0.4], [-0.7, 0.0, 0.25], [0.4, -0.25, 0.0]])
    a = np.array([1.1, -0.6, 0.3])
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    sol = problem.solve(dirichlet=lambda x: a + np.atleast_2d(x) @ W.T)

    assert np.allclose(sol["stress"], 0.0, atol=1e-8)
    assert np.allclose(
        sol["displacement"],
        problem.interpolate_displacement(lambda x: a + np.atleast_2d(x) @ W.T),
        atol=1e-9,
    )
    # the multiplier picks up exactly the rotation
    expected = np.tile(
        [0.5 * np.sum(S * W) for S in skew_generators(3)], problem.n_cells
    )
    assert np.allclose(sol["rotation"], expected, atol=1e-9)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_elasticity_weak_symmetry_is_enforced(mesh):
    """``as_h(sigma_h) = 0`` is a constraint of the system, so it holds exactly."""
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    _, _, A = problem.assemble_operators()
    sol = problem.solve(body_force=None, dirichlet=displacement)
    assert np.allclose(A @ sol["stress"], 0.0, atol=1e-9)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_global_elasticity_equilibrium_is_enforced(mesh):
    """``div_h sigma_h`` equals the prescribed load; here zero."""
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    _, D, _ = problem.assemble_operators()
    sol = problem.solve(body_force=None, dirichlet=displacement)
    assert np.allclose(D @ sol["stress"], 0.0, atol=1e-9)


def test_elasticity_block_sizes():
    mesh = structured_box(2, 2, 2)
    problem = MixedElasticity(mesh)
    M, D, A = problem.assemble_operators()
    n_sig = 9 * mesh.num_cells(2)
    assert M.shape == (n_sig, n_sig)
    assert D.shape == (3 * mesh.num_cells(3), n_sig)
    assert A.shape == (3 * mesh.num_cells(3), n_sig)


def test_elasticity_exactness_is_uniform_in_lambda():
    mesh = structured_tets(1, 1, 1)
    for lam in (1.0, 1e4, 1e7):
        problem = MixedElasticity(mesh, mu=1.0, lam=lam)
        sol = problem.solve(body_force=None, dirichlet=displacement)
        assert np.allclose(
            sol["displacement"],
            problem.interpolate_displacement(displacement),
            atol=1e-8,
        )


# -- convergence smoke test ---------------------------------------------------


def test_poisson_error_decreases_under_refinement():
    """A smooth non-polynomial solution: the error must shrink with h."""

    def p(x):
        return np.prod(np.sin(np.pi * np.atleast_2d(x)), axis=1)

    def f(x):
        return 3 * np.pi**2 * p(x)

    errs = []
    for n in (2, 4):
        mesh = structured_box(n, n, n)
        problem = MixedPoisson(mesh)
        sol = problem.solve(source=f, dirichlet=p)
        e = sol["pressure"] - problem.interpolate_pressure(p)
        errs.append(np.sqrt(mesh.geometry.measure(3) @ e**2))
    assert errs[1] < 0.5 * errs[0]  # at least first order
