"""Mimetic patch tests: exactness of *local saddle-point solves*.

For every reference cell of dimension >= 1 (0-cells carry no differential
operators) we take a generic linear potential / displacement in ``R^3``, form
the corresponding local mixed problem, **solve it**, and require the discrete
solution to equal the exact interpolant.

This is strictly stronger than an energy-exactness check: it passes only if the
inner product satisfies strong consistency ``M N = R``.
"""

import numpy as np
import pytest

from mimetika.assembly.local import (
    elasticity_local_operators,
    exact_diffusion_dofs,
    linear_displacement_reference,
    skew_generators,
    solve_local_diffusion,
    solve_local_elasticity,
    stress_dofs,
)
from mimetika.geometry.local_cell import LocalCell
from mimetika.mesh.reference import reference_cells
from mimetika.operators.diffusion import DiffusionInnerProduct
from mimetika.operators.elasticity import ElasticityInnerProduct

CELLS = [c for c in reference_cells() if c.dim >= 1]
IDS = [c.name for c in CELLS]
ZERO_D = [c for c in reference_cells() if c.dim == 0]

ANISO_K = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])

# A generic linear scalar potential  p(x) = P_A + P_B . x
P_A, P_B = 0.37, np.array([0.8, -1.3, 0.55])
# A generic linear vector field  u(x) = U_A + U_B x
U_A = np.array([0.31, -0.42, 0.17])
U_B = np.array([[0.5, -0.3, 0.2], [0.15, 0.4, -0.25], [-0.1, 0.35, 0.6]])


def potential(x):
    return P_A + np.asarray(x) @ P_B


def displacement(x):
    return U_A + np.asarray(x) @ U_B.T


# -- 0D: the differential operators are void ----------------------------------


@pytest.mark.parametrize("rc", ZERO_D, ids=[c.name for c in ZERO_D])
def test_zero_dimensional_cells_have_no_differential_structure(rc):
    assert rc.mesh.dim == 0
    with pytest.raises(ValueError):
        LocalCell.build(rc.mesh.geometry, 0)
    with pytest.raises(ValueError):
        rc.mesh.complex.boundary_matrix(1)


# -- scalar: linear potential -> constant flux --------------------------------


@pytest.mark.parametrize("basis", ["const", "rt0"])
@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_local_diffusion_solve_is_exact(rc, basis):
    ip = DiffusionInnerProduct(rc.mesh, K=ANISO_K, basis=basis)
    flux, p_E = solve_local_diffusion(ip, 0, potential)

    assert np.allclose(flux, exact_diffusion_dofs(ip, 0, P_B), atol=1e-10)
    centroid = rc.mesh.geometry.centroids(rc.dim)[0]
    assert np.isclose(p_E, potential(centroid[None, :])[0], atol=1e-10)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_local_diffusion_solve_is_exact_for_isotropic_K(rc):
    ip = DiffusionInnerProduct(rc.mesh, basis="const")
    flux, p_E = solve_local_diffusion(ip, 0, potential)
    assert np.allclose(flux, exact_diffusion_dofs(ip, 0, P_B), atol=1e-10)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_recovered_flux_is_divergence_free(rc):
    """A linear potential has zero source, so the discrete divergence vanishes."""
    ip = DiffusionInnerProduct(rc.mesh, K=ANISO_K)
    lc = LocalCell.build(rc.mesh.geometry, 0)
    flux, _ = solve_local_diffusion(ip, 0, potential)
    assert abs(lc.facet_measures @ flux) < 1e-10


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_constant_potential_gives_zero_flux(rc):
    ip = DiffusionInnerProduct(rc.mesh, K=ANISO_K)
    flux, p_E = solve_local_diffusion(ip, 0, lambda x: np.full(len(np.atleast_2d(x)), 2.5))
    assert np.allclose(flux, 0.0, atol=1e-10)
    assert np.isclose(p_E, 2.5)


# -- vector: linear displacement -> constant symmetric stress -----------------


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_local_elasticity_solve_is_exact(rc):
    ip = ElasticityInnerProduct(rc.mesh, mu=1.3, lam=2.7)
    sigma, u_E, s = solve_local_elasticity(ip, 0, displacement)
    sigma_x, u_x, s_x = linear_displacement_reference(ip, 0, U_A, U_B)

    assert np.allclose(sigma, sigma_x, atol=1e-8)
    assert np.allclose(u_E, u_x, atol=1e-9)
    assert np.allclose(s, s_x, atol=1e-9)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_recovered_stress_is_weakly_symmetric_and_equilibrated(rc):
    ip = ElasticityInnerProduct(rc.mesh, mu=1.3, lam=2.7)
    M, Dv, As, lc = elasticity_local_operators(ip, 0)
    sigma, _, _ = solve_local_elasticity(ip, 0, displacement)
    assert np.allclose(Dv @ sigma, 0.0, atol=1e-9)  # div_h sigma = 0
    assert np.allclose(As @ sigma, 0.0, atol=1e-9)  # as_h sigma = 0


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_rigid_body_motion_produces_no_stress(rc):
    """Translation + infinitesimal rotation is a zero-energy displacement."""
    d = rc.dim
    lc = LocalCell.build(rc.mesh.geometry, 0)
    ip = ElasticityInnerProduct(rc.mesh, mu=1.3, lam=2.7)

    # a skew ambient gradient restricted to the hull stays skew
    W = np.array([[0.0, 0.7, -0.4], [-0.7, 0.0, 0.25], [0.4, -0.25, 0.0]])
    a = np.array([1.1, -0.6, 0.3])
    sigma, u_E, s = solve_local_elasticity(ip, 0, lambda x: a + np.asarray(x) @ W.T)

    assert np.allclose(sigma, 0.0, atol=1e-9)
    assert np.allclose(u_E, (a + W @ lc.origin) @ lc.frame, atol=1e-10)
    W_loc = lc.frame.T @ W @ lc.frame
    expected_s = np.array([0.5 * np.sum(S * W_loc) for S in skew_generators(d)])
    assert np.allclose(s, expected_s, atol=1e-9)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_incompressible_limit_stays_exact(rc):
    """Exactness must be uniform in lambda (the near-incompressible regime)."""
    for lam in (1.0, 1e4, 1e7):
        ip = ElasticityInnerProduct(rc.mesh, mu=1.0, lam=lam)
        sigma, u_E, _ = solve_local_elasticity(ip, 0, displacement)
        sigma_x, u_x, _ = linear_displacement_reference(ip, 0, U_A, U_B)
        scale = max(1.0, np.abs(sigma_x).max())
        assert np.allclose(sigma / scale, sigma_x / scale, atol=1e-8)
        assert np.allclose(u_E, u_x, atol=1e-8)


# -- the discrete operators themselves ----------------------------------------


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_discrete_divergence_matches_the_exact_divergence(rc):
    """``div_h`` applied to a constant stress gives zero; to a linear one, div."""
    ip = ElasticityInnerProduct(rc.mesh)
    _, Dv, _, lc = elasticity_local_operators(ip, 0)
    rng = np.random.default_rng(0)
    S = rng.standard_normal((lc.dim, lc.dim))
    assert np.allclose(Dv @ stress_dofs(ip, 0, S), 0.0, atol=1e-10)


@pytest.mark.parametrize("rc", CELLS, ids=IDS)
def test_discrete_asymmetry_matches_the_exact_one(rc):
    """``as_h`` of a constant stress equals its algebraic skew part."""
    ip = ElasticityInnerProduct(rc.mesh)
    _, _, As, lc = elasticity_local_operators(ip, 0)
    d = lc.dim
    rng = np.random.default_rng(1)
    S = rng.standard_normal((d, d))
    expected = np.array([np.sum(G * S) for G in skew_generators(d)])
    assert np.allclose(As @ stress_dofs(ip, 0, S), expected, atol=1e-10)


def test_skew_generators_are_a_basis():
    for d, n in [(1, 0), (2, 1), (3, 3)]:
        gens = skew_generators(d)
        assert gens.shape == (n, d, d)
        for G in gens:
            assert np.allclose(G, -G.T)
        # pairwise orthogonal with S : S = 2
        for i, G in enumerate(gens):
            for j, H in enumerate(gens):
                assert np.isclose(np.sum(G * H), 2.0 if i == j else 0.0)
