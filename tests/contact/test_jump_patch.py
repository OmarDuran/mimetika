r"""A patch test for the jump operator ``g = -(M sigma + D^T u + A^T s)|_f``.

The fracture row of the *unfractured* system is the traction residual on a facet.
Feed it the degrees of freedom of a field that the discretization represents
exactly -- piecewise linear displacement, piecewise constant stress, continuous
traction -- and it must return that field's displacement jump at the facet, to
round-off.  Anything less means the fracture terms carry a consistency error that
no mesh refinement in a benchmark can distinguish from a modelling error.

The degrees of freedom are **constructed**, never sampled from a discontinuous
function.  Interpolating a field that jumps across the fault would evaluate it at
cell quadrature points, and on a structured mesh some of those sit exactly *on*
the fault plane, where a ``x > x_f`` side test is ambiguous.  That excludes one
quadrature point from the cell mean and scales the recovered jump by the weight
of the rest -- a clean-looking rational factor that is an artefact of the test
and not a property of the operator.  Cell means of a linear field are known in
closed form, so they are written down directly instead.
"""

import numpy as np
import pytest

from mimetika.assembly.mixed import MixedElasticity
from mimetika.contact import ContactDriver, SignoriniCoulomb
from mimetika.materials import compliance_coefficient
from mimetika.mesh import structured_quads
from mimetika.mesh.fracture import facets_on_plane

MU, NU = 1.0, 0.25
LAM = 2 * MU * NU / (1 - 2 * NU)
DIM = 2
STRESS_LEFT = np.array([[2.0, 0.7], [0.7, -1.0]])

MESHES = [(2, 1, 1.0), (2, 2, 1.0), (4, 3, 0.7), (6, 4, 1.3)]


def _strain(stress):
    """Hooke inverted with the same compliance the discretization uses."""
    a = compliance_coefficient(DIM, NU)
    return (stress - a * np.trace(stress) * np.eye(DIM)) / (2 * MU)


def _constant_stress_dofs(problem, stress):
    tensor = np.zeros((3, 3))
    tensor[:DIM, :DIM] = stress
    return problem.interpolate_stress(
        lambda x: np.broadcast_to(tensor, (len(np.atleast_2d(x)), 3, 3))
    )


def recovered_and_exact(nx, ny, height, jump_yy, offset):
    """Run the operator on exact DOFs; return ``(recovered, exact)`` jumps.

    ``jump_yy`` perturbs ``sigma_yy`` on the right of the fault.  The fault normal
    is ``e_x``, so that perturbation leaves the traction ``sigma . n`` continuous:
    the pair is in equilibrium, but the strains differ, so the displacement jump
    grows linearly along the fault.  ``offset`` adds a rigid translation on top.
    """
    mesh = structured_quads(nx, ny, lengths=(1.0, height))
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    facet = int(tags[0])
    problem = MixedElasticity(mesh, mu=MU, lam=LAM)
    frame = problem.inner.frame  # (3, d): ambient -> mesh components

    stress_right = STRESS_LEFT + np.array([[0.0, 0.0], [0.0, jump_yy]])
    strain_left, strain_right = _strain(STRESS_LEFT), _strain(stress_right)

    centroids = mesh.geometry.centroids(DIM)
    right = centroids[:, 0] > 0.5

    # -- stress: constant per side, so no discontinuity is ever sampled
    sigma = _constant_stress_dofs(problem, STRESS_LEFT)
    if jump_yy:
        other = _constant_stress_dofs(problem, stress_right)
        for cell in np.where(right)[0]:
            for fid, _ in mesh.complex.facets_of(DIM, int(cell)):
                if int(fid) != facet:  # the fault facet keeps the left traction
                    block = slice(problem.ndf * int(fid), problem.ndf * (int(fid) + 1))
                    sigma[block] = other[block]

    # -- displacement: the cell mean of a linear field is its value at the centroid
    displacement = np.zeros(DIM * mesh.num_cells(DIM))
    for cell in range(mesh.num_cells(DIM)):
        strain = strain_right if right[cell] else strain_left
        ambient = np.zeros(3)
        ambient[:DIM] = strain @ centroids[cell][:DIM]
        if right[cell]:
            ambient = ambient + offset
        displacement[cell * DIM : (cell + 1) * DIM] = ambient @ frame

    rotation = np.zeros(problem.n_skew * mesh.num_cells(DIM))
    driver = ContactDriver(mesh, tags, SignoriniCoulomb(), mu=MU, lam=LAM)
    dofs = np.concatenate([sigma, displacement, rotation])
    recovered = (driver.jump_operator(problem) @ dofs).reshape(-1, DIM)[0]

    ambient = np.zeros(3)
    ambient[:DIM] = (strain_right - strain_left) @ mesh.geometry.centroids(1)[facet][:DIM]
    exact = driver._frame(facet).T @ ((ambient + offset) @ frame)
    return recovered, exact


@pytest.mark.parametrize("nx,ny,height", MESHES)
def test_a_rigid_offset_is_recovered_exactly(nx, ny, height):
    """No strain jump: the operator must return the translation itself."""
    got, exact = recovered_and_exact(nx, ny, height, 0.0, np.array([0.03, -0.02, 0.0]))
    assert np.abs(got - exact).max() < 1e-13
    assert np.abs(exact).max() > 1e-3  # the test would pass trivially on zero


@pytest.mark.parametrize("nx,ny,height", MESHES)
def test_a_strain_jump_is_recovered_exactly(nx, ny, height):
    """Equilibrated stress pair, differing strains: the jump varies along the fault."""
    got, exact = recovered_and_exact(nx, ny, height, 1.3, np.zeros(3))
    assert np.abs(got - exact).max() < 1e-13
    assert np.abs(exact).max() > 1e-3


@pytest.mark.parametrize("nx,ny,height", MESHES)
def test_the_two_superpose(nx, ny, height):
    got, exact = recovered_and_exact(nx, ny, height, 1.3, np.array([0.03, -0.02, 0.0]))
    assert np.abs(got - exact).max() < 1e-13


def test_the_recovered_jump_follows_the_facet_and_is_not_a_constant():
    """Guards the guard: a stuck operator returning one value would pass the above.

    With a strain jump the exact answer depends on where the facet centroid sits,
    so refining the mesh in ``y`` must move it.  If it does not, the comparison
    above is matching a constant rather than tracking the field.
    """
    values = [recovered_and_exact(2, n, 1.0, 1.3, np.zeros(3))[0][1] for n in (1, 2, 4)]
    assert len({round(v, 9) for v in values}) == 3


def test_a_continuous_field_produces_no_jump():
    """The null case: same stress on both sides, one linear displacement, zero jump."""
    got, exact = recovered_and_exact(4, 3, 0.7, 0.0, np.zeros(3))
    assert np.abs(exact).max() < 1e-14
    assert np.abs(got).max() < 1e-13
