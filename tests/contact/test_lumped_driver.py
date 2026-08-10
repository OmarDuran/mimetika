r"""The contact driver on the lumped stress space (``d`` DOFs per facet).

The driver reads and pins traction DOFs by the stride of the stress space; for
the lumped space that is ``d`` per facet, with the constant as the only facet
basis function.  These tests mirror the AFW driver tests on the lumped
four-field mechanics and add the cross-formulation checks: the *physics* on
the fault -- normal traction, the friction cap, closure under compression --
must agree between AFW and lumped, because the fault states here are constant
per facet and both spaces resolve constants exactly.
"""

import numpy as np
import pytest

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.contact import ContactDriver, SignoriniCoulomb
from mimetika.contact.driver import elastic_mechanics
from mimetika.mesh import structured_box
from mimetika.mesh.fracture import facets_on_plane
from mimetika.operators.lumped import LumpedDeviatoricStress

MU, LAM = 1.0, 1.0


def lumped_mechanics(mesh, mu=MU, lam=LAM, **boundary):
    """The ``mechanics`` factory for the lumped four-field formulation."""

    def build(contact=None):
        problem = FourFieldElasticity(
            mesh,
            contact=contact,
            inner=LumpedDeviatoricStress(mesh, mu=mu, lam=lam),
        )
        matrix, rhs = problem.assemble_constrained(**boundary)
        return problem, matrix, rhs

    return build


def setup(shape=(2, 2, 2), law=None):
    mesh = structured_box(*shape)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    law = SignoriniCoulomb(friction=0.6) if law is None else law
    driver = ContactDriver(
        mesh, tags, law, mu=MU, lam=LAM, dofs_per_facet=mesh.dim
    )
    return mesh, tags, driver


def load(normal=0.0, shear=0.0):
    def u(x):
        x = np.atleast_2d(x)
        out = np.zeros((len(x), 3))
        out[:, 0] = normal * x[:, 0]
        out[:, 1] = shear * x[:, 0]
        return out

    return u


def test_moment_value_round_trip_is_the_identity():
    """``to_values . to_moments = id`` on the constant facet basis."""
    _, tags, driver = setup()
    assert driver.ndf == 3 and driver.nb == 1
    values = np.array([[0.7, -0.3, 0.2]])
    for f in tags:
        moments = driver.to_moments(values, int(f))
        assert moments.shape == (3,)
        back = driver.to_values(moments, int(f))
        assert np.allclose(back, values, atol=1e-12)


def test_compression_closes_without_interpenetration():
    mesh, tags, driver = setup()
    state = driver.solve_step(
        lumped_mechanics(mesh, dirichlet=load(normal=-0.01))
    )
    assert state.converged
    t = driver.tractions(state.solution["stress"])
    assert np.all(t[:, 0] < 0)  # compressive normal traction
    assert np.abs(state.jump[:, 0]).max() < 1e-10  # no interpenetration


def test_shear_traction_is_capped_at_the_friction_coefficient():
    mesh, tags, driver = setup()
    state = driver.solve_step(
        lumped_mechanics(mesh, dirichlet=load(normal=-0.01, shear=0.05))
    )
    assert state.converged
    t = driver.tractions(state.solution["stress"])
    ratio = np.linalg.norm(t[:, 1:], axis=1) / -t[:, 0]
    assert np.allclose(ratio, 0.6, rtol=1e-6)


def test_afw_and_lumped_agree_on_the_fault_physics():
    """The determined fault quantities coincide; the artifact stays small.

    The normal traction and the in-plane shear are set by the confined load
    and the friction cap -- constant states both spaces resolve exactly, so
    they must agree tightly.  The out-of-plane tangential component is *not*
    determined (the exact value is zero); the slipping return mapping leaves a
    discretization-level residue there that legitimately differs between the
    spaces, so it is only bounded, not compared.
    """
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    boundary = dict(dirichlet=load(normal=-0.01, shear=0.05))

    afw_driver = ContactDriver(
        mesh, tags, SignoriniCoulomb(friction=0.6), mu=MU, lam=LAM
    )
    afw = afw_driver.solve_step(elastic_mechanics(mesh, MU, LAM, **boundary))

    lump_driver = ContactDriver(
        mesh, tags, SignoriniCoulomb(friction=0.6), mu=MU, lam=LAM,
        dofs_per_facet=mesh.dim,
    )
    lump = lump_driver.solve_step(lumped_mechanics(mesh, **boundary))

    t_afw = afw_driver.tractions(afw.solution["stress"])
    t_lump = lump_driver.tractions(lump.solution["stress"])
    # t_N is determined exactly; the primary shear inherits the (undetermined)
    # out-of-plane residue at second order through the saturated cap, so it is
    # compared at that level rather than at round-off
    assert np.allclose(t_lump[:, 0], t_afw[:, 0], rtol=1e-6)
    assert np.allclose(t_lump[:, 1], t_afw[:, 1], rtol=1e-4)
    for t in (t_afw, t_lump):
        assert np.abs(t[:, 2]).max() < 1e-2 * np.abs(t[:, 0]).min()
        ratio = np.linalg.norm(t[:, 1:], axis=1) / -t[:, 0]
        assert np.allclose(ratio, 0.6, rtol=1e-6)


def test_a_mismatched_stride_is_rejected():
    mesh = structured_box(2, 2, 2)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError, match="dofs_per_facet"):
        ContactDriver(
            mesh, tags, SignoriniCoulomb(friction=0.6), dofs_per_facet=5
        )
