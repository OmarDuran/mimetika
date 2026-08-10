"""Cell-centred reconstruction of facet-based flux and stress unknowns.

The reconstructions must be **exact for constant fields**: that is what makes
them the natural partners of a scheme that reproduces constant fluxes and
stresses exactly, rather than an ad-hoc smoothing.
"""

import numpy as np
import pytest

from mimetika.assembly.mixed import MixedElasticity, MixedPoisson
from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.reference import reference_cells
from mimetika.postprocess import (
    mean_stress,
    principal_stresses,
    reconstruct_flux,
    reconstruct_stress,
    von_mises,
)

# exactness of the reconstruction is a discretisation property: use a direct
# solve so the iterative tolerance does not enter
EXACT = {"method": "direct"}
K_ANISO = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
MU, LAM = 1.3, 2.7
GRAD = np.array([0.7, -1.1, 0.45])
U_A = np.array([0.013, -0.007, 0.004])
U_B = np.array([[0.02, -0.01, 0.006], [0.005, 0.015, -0.008], [-0.003, 0.011, 0.02]])

MESHES = [
    ("hex", structured_box(2, 2, 2)),
    ("tet", structured_tets(1, 1, 1)),
    ("prism", reference_cells(dim=3)[4].mesh),
    ("dented", reference_cells(dim=3)[6].mesh),
]
IDS = [n for n, _ in MESHES]
ONLY = [m for _, m in MESHES]


def linear_potential(x):
    return 0.3 + np.atleast_2d(x) @ GRAD


def linear_displacement(x):
    return U_A + np.atleast_2d(x) @ U_B.T


def exact_stress():
    eps = 0.5 * (U_B + U_B.T)
    return 2 * MU * eps + LAM * np.trace(eps) * np.eye(3)


# -- flux ---------------------------------------------------------------------


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_flux_reconstruction_is_exact_for_constant_velocity(mesh):
    problem = MixedPoisson(mesh, K=K_ANISO, basis="const")
    sol = problem.solve(dirichlet=linear_potential, **EXACT)
    velocity = reconstruct_flux(mesh, sol["flux"])
    expected = -K_ANISO @ GRAD
    assert velocity.shape == (mesh.num_cells(3), 3)
    assert np.allclose(velocity, expected, atol=1e-10)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_flux_reconstruction_of_interpolant(mesh):
    """Reconstruction inverts interpolation on constant fields."""
    problem = MixedPoisson(mesh, K=K_ANISO, basis="const")
    field = np.array([0.4, -0.9, 1.3])
    dofs = problem.interpolate_flux(
        lambda x: np.broadcast_to(field, (len(np.atleast_2d(x)), 3))
    )
    assert np.allclose(reconstruct_flux(mesh, dofs), field, atol=1e-12)


def test_zero_flux_reconstructs_to_zero():
    mesh = structured_box(2, 2, 2)
    assert np.allclose(reconstruct_flux(mesh, np.zeros(mesh.num_cells(2))), 0.0)


# -- stress -------------------------------------------------------------------


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_stress_reconstruction_is_exact_for_constant_stress(mesh):
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    sol = problem.solve(dirichlet=linear_displacement, **EXACT)
    sigma = reconstruct_stress(mesh, sol["stress"])
    assert sigma.shape == (mesh.num_cells(3), 3, 3)
    assert np.allclose(sigma, exact_stress(), atol=1e-9)


@pytest.mark.parametrize("mesh", ONLY, ids=IDS)
def test_stress_reconstruction_of_interpolant(mesh):
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    field = exact_stress()
    dofs = problem.interpolate_stress(
        lambda x: np.broadcast_to(field, (len(np.atleast_2d(x)), 3, 3))
    )
    assert np.allclose(reconstruct_stress(mesh, dofs), field, atol=1e-10)


# -- derived measures ---------------------------------------------------------


def test_von_mises_of_hydrostatic_stress_vanishes():
    s = np.broadcast_to(2.5 * np.eye(3), (4, 3, 3))
    assert np.allclose(von_mises(s), 0.0)
    assert np.allclose(mean_stress(s), 2.5)


def test_von_mises_of_uniaxial_stress_equals_the_axial_value():
    s = np.zeros((1, 3, 3))
    s[0, 0, 0] = 3.0
    assert np.isclose(von_mises(s)[0], 3.0)


def test_principal_stresses_are_sorted_eigenvalues():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3, 3))
    sym = 0.5 * (A + np.swapaxes(A, 1, 2))
    p = principal_stresses(sym)
    assert p.shape == (5, 3)
    assert (np.diff(p, axis=1) >= -1e-12).all()
    assert np.allclose(p.sum(1), np.einsum("nii->n", sym))


def test_derived_measures_on_a_real_solve():
    mesh = structured_box(2, 2, 2)
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    sol = problem.solve(dirichlet=linear_displacement, **EXACT)
    sigma = reconstruct_stress(mesh, sol["stress"])
    exact = exact_stress()
    assert np.allclose(von_mises(sigma), von_mises(exact[None])[0], atol=1e-9)
    assert np.allclose(mean_stress(sigma), np.trace(exact) / 3, atol=1e-9)
