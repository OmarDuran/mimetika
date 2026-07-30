"""Linear contact law on fracture facets, in the mixed elasticity problem.

The reference cases are piecewise-linear displacement fields with a prescribed
jump, for which the contact law gives the opening in closed form.  Because both
the field and the stress lie in the reconstruction space, the discrete solution
must reproduce them to round-off.

Note on the boundary data: a lateral face touching the fracture plane has
quadrature points *exactly* on it, so an ``x > 0.5`` indicator is ambiguous
there.  The side is therefore decided per facet, from the mean of its points.
"""

import numpy as np
import pytest

from mimetika.assembly.contact import FractureContact
from mimetika.assembly.mixed import MixedElasticity
from mimetika.mesh import structured_box, structured_tets
from mimetika.mesh.fracture import facets_on_plane, read_fracture_tags
from mimetika.mesh.readers import read_vtu

MU, LAM = 1.0, 1.0
STRAIN = 0.01


def setup(shape=(2, 2, 2), kn=0.5, ks=0.5, tets=False):
    mesh = (structured_tets if tets else structured_box)(*shape)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    contact = FractureContact(mesh, tags, normal_stiffness=kn, shear_stiffness=ks)
    return mesh, tags, contact


def jump_field(slope, jump, component=0):
    """``u_c = slope*x + jump*H(x-0.5)`` -- side chosen per facet, not per point."""

    def u(x):
        x = np.atleast_2d(x)
        side = 1.0 if x[:, 0].mean() > 0.5 else 0.0
        out = np.zeros((len(x), 3))
        out[:, component] = slope * x[:, 0] + jump * side
        return out

    return u


# -- the compliance block -------------------------------------------------------


def test_local_compliance_is_diagonal_in_the_facet_frame():
    mesh, tags, ct = setup(kn=2.0, ks=5.0)
    g = mesh.geometry
    for f in tags:
        n = g.facet_normals()[f]
        t1, t2 = g.facet_tangents(f)
        A = ct.local_compliance(int(f))
        assert np.isclose(n @ A @ n, 1 / 2.0)  # 1/k_n along the normal
        assert np.isclose(t1 @ A @ t1, 1 / 5.0)  # 1/k_s in shear
        assert np.isclose(t2 @ A @ t2, 1 / 5.0)
        assert np.allclose(n @ A @ t1, 0.0, atol=1e-12)
        assert np.allclose(A, A.T)


def test_facet_gram_matches_quadrature():
    """The closed-form Gram must equal the integral of the P1 basis products."""
    from mimetika.geometry.local_cell import LocalCell

    mesh, tags, ct = setup()
    ip = MixedElasticity(mesh).inner
    for f in tags:
        cell = ct.mesh.complex.boundary_matrix(3).tocsr()[f].indices[0]
        lc = LocalCell.build(mesh.geometry, int(cell), ip.frame)
        i = lc.facet_ids.index(int(f))
        B, qw = lc.facet_scalar_basis(i)
        assert np.allclose(ct.facet_gram(int(f)), B.T @ (qw[:, None] * B), atol=1e-12)


def test_compliance_block_is_symmetric_positive_definite():
    mesh, tags, ct = setup()
    for f in tags:
        blk = ct.block(int(f))
        assert blk.shape == (9, 9)
        assert np.allclose(blk, blk.T)
        assert np.linalg.eigvalsh(blk).min() > 0


def test_assembled_contact_touches_only_fracture_facets():
    mesh, tags, ct = setup()
    A = ct.assemble()
    assert A.shape == (9 * mesh.num_cells(2), 9 * mesh.num_cells(2))
    touched = {int(d) // 9 for d in np.unique(A.tocoo().row)}
    assert touched == {int(f) for f in tags}


def test_contact_does_not_duplicate_any_dof():
    """t+ + t- = 0 by equilibrium, so the traction block stays shared."""
    mesh, tags, ct = setup()
    plain = MixedElasticity(mesh, mu=MU, lam=LAM)
    frac = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    assert frac.n_stress == plain.n_stress == 9 * mesh.num_cells(2)


def test_system_stays_symmetric_with_contact():
    mesh, tags, ct = setup()
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    S, _ = pb.assemble(dirichlet=jump_field(STRAIN, 0.0))
    assert (abs(S - S.T) > 1e-10).nnz == 0


# -- normal opening --------------------------------------------------------------


@pytest.mark.parametrize("kn", [0.1, 0.5, 10.0])
def test_normal_opening_matches_the_contact_law(kn):
    """Confined extension: sigma_xx = (2mu+lam) e and the opening is sigma/k_n."""
    mesh, tags, ct = setup(kn=kn, ks=kn)
    sxx = (2 * MU + LAM) * STRAIN
    delta = sxx / kn

    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=jump_field(STRAIN, delta), method="direct")

    exact = pb.interpolate_stress(
        lambda x: np.broadcast_to(
            np.diag([sxx, LAM * STRAIN, LAM * STRAIN]), (len(np.atleast_2d(x)), 3, 3)
        )
    )
    assert np.allclose(sol["stress"], exact, atol=1e-12)
    for f in tags:
        assert np.allclose(ct.opening(sol["stress"], int(f)), [delta, 0, 0], atol=1e-10)


def test_displacement_shows_the_prescribed_jump():
    kn = 0.5
    mesh, tags, ct = setup(kn=kn)
    sxx = (2 * MU + LAM) * STRAIN
    delta = sxx / kn
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=jump_field(STRAIN, delta), method="direct")

    u = sol["displacement"].reshape(-1, 3)
    x = mesh.geometry.centroids(3)[:, 0]
    assert np.allclose(u[x < 0.5, 0], STRAIN * 0.25, atol=1e-10)
    assert np.allclose(u[x > 0.5, 0], STRAIN * 0.75 + delta, atol=1e-10)


# -- shear slip --------------------------------------------------------------------


@pytest.mark.parametrize("ks", [0.25, 2.0])
def test_shear_slip_matches_the_contact_law(ks):
    """Simple shear across the fracture: sigma_xy = mu*g, slip = sigma_xy/k_s."""
    mesh, tags, ct = setup(kn=1e6, ks=ks)  # rigid in tension, compliant in shear
    g = STRAIN
    sxy = MU * g
    slip = sxy / ks

    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=jump_field(g, slip, component=1), method="direct")

    for f in tags:
        op = ct.opening(sol["stress"], int(f))
        assert np.isclose(op[1], slip, atol=1e-8)  # tangential
        assert abs(op[0]) < 1e-8  # no opening


# -- rigid body motion: the test the naive jump operator fails ----------------------


def test_rigid_translation_produces_no_traction_and_no_opening():
    mesh, tags, ct = setup()
    a = np.array([1.1, -0.6, 0.3])
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=lambda x: np.broadcast_to(a, (len(np.atleast_2d(x)), 3)),
                   method="direct")
    assert np.allclose(sol["stress"], 0.0, atol=1e-10)
    for f in tags:
        assert np.allclose(ct.opening(sol["stress"], int(f)), 0.0, atol=1e-10)


def test_rigid_rotation_produces_no_traction_and_no_opening():
    """The jump operator must include the rotation lever arm.

    A jump built from cell displacements alone would see the two sides of the
    fracture move differently under a rigid rotation and invent a traction.
    """
    mesh, tags, ct = setup()
    W = np.array([[0.0, 0.7, -0.4], [-0.7, 0.0, 0.25], [0.4, -0.25, 0.0]])
    a = np.array([1.1, -0.6, 0.3])
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(
        dirichlet=lambda x: a + np.atleast_2d(x) @ W.T, method="direct"
    )

    assert np.allclose(sol["stress"], 0.0, atol=1e-9)
    for f in tags:
        assert np.allclose(ct.opening(sol["stress"], int(f)), 0.0, atol=1e-9)
    # the multiplier still recovers the rotation itself
    from mimetika.assembly.local import skew_generators

    expected = np.tile(
        [0.5 * np.sum(S * W) for S in skew_generators(3)], pb.n_cells
    )
    assert np.allclose(sol["rotation"], expected, atol=1e-9)


# -- limits ---------------------------------------------------------------------------


def test_rigid_fracture_recovers_the_unfractured_solution():
    """k -> infinity: zero compliance, so the fracture disappears."""
    mesh, tags, ct = setup(kn=1e12, ks=1e12)
    u = jump_field(STRAIN, 0.0)
    stiff = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct).solve(
        dirichlet=u, method="direct"
    )
    plain = MixedElasticity(mesh, mu=MU, lam=LAM).solve(dirichlet=u, method="direct")
    assert np.allclose(stiff["stress"], plain["stress"], atol=1e-8)
    assert np.allclose(stiff["displacement"], plain["displacement"], atol=1e-10)


def test_soft_fracture_carries_almost_no_traction():
    """k -> 0: the fracture stops transmitting traction.

    Only the traction *on the fracture* vanishes.  The rest of the body still
    carries stress, because the lateral walls hold prescribed displacements and
    keep forcing it to deform -- the two halves decouple from each other, not
    from their own boundary data.
    """
    transmitted = []
    for kn in (1e-4, 1e-8):
        mesh, tags, ct = setup(kn=kn, ks=kn)
        pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
        sol = pb.solve(dirichlet=jump_field(STRAIN, 0.0), method="direct")
        frac = np.concatenate([9 * int(f) + np.arange(9) for f in tags])
        other = np.setdiff1d(np.arange(pb.n_stress), frac)
        transmitted.append(np.abs(sol["stress"][frac]).max())
        assert np.abs(sol["stress"][other]).max() > 1e-3  # still loaded

    # the transmitted traction falls off in proportion to the stiffness
    assert transmitted[1] < 1e-3 * transmitted[0]


def test_compliance_adds_in_series():
    """Halving the stiffness doubles the opening at fixed traction."""
    sxx = (2 * MU + LAM) * STRAIN
    openings = []
    for kn in (1.0, 0.5):
        mesh, tags, ct = setup(kn=kn, ks=kn)
        pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
        sol = pb.solve(dirichlet=jump_field(STRAIN, sxx / kn), method="direct")
        openings.append(ct.opening(sol["stress"], int(tags[0]))[0])
    assert np.isclose(openings[1], 2 * openings[0], rtol=1e-8)


# -- other meshes ----------------------------------------------------------------------


def test_contact_on_a_tetrahedral_mesh():
    mesh, tags, ct = setup(kn=0.5, ks=0.5, tets=True)
    assert len(tags) > 0
    sxx = (2 * MU + LAM) * STRAIN
    delta = sxx / 0.5
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=jump_field(STRAIN, delta), method="direct")
    for f in tags:
        assert np.allclose(ct.opening(sol["stress"], int(f)), [delta, 0, 0], atol=1e-9)


def test_contact_driven_from_a_tagged_vtu(tagged_vtu):
    path, _, tags = tagged_vtu(
        shape=(2, 2, 2), point=(0.5, 0, 0), normal=(1, 0, 0), name="contact.vtu"
    )
    mesh = read_vtu(path)
    resolved = read_fracture_tags(path, mesh)
    kn = 0.5
    ct = FractureContact(mesh, resolved, normal_stiffness=kn, shear_stiffness=kn)
    sxx = (2 * MU + LAM) * STRAIN
    pb = MixedElasticity(mesh, mu=MU, lam=LAM, contact=ct)
    sol = pb.solve(dirichlet=jump_field(STRAIN, sxx / kn), method="direct")
    assert np.allclose(
        ct.opening(sol["stress"], int(resolved[0])), [sxx / kn, 0, 0], atol=1e-9
    )
