"""Fracture contact in 2D.

Same laws, same driver, same physics -- only the dimension changes:

======================  =======  =======
                         d = 2    d = 3
======================  =======  =======
facet                    edge     polygon
stress DOFs per facet    4        9
rotation per cell        1        3
reconstruction modes     12       36
stabilisation-free       triangle tetrahedron
friction set             interval disk
======================  =======  =======

The 2D case is the cheap testbed for the contact laws, and it also guards the
orientation convention: the canonical facet normal must point out of the ``+1``
incidence cell.  In 3D that holds by construction from the loop orientation; in
2D it has to be enforced, and getting it wrong silently inverts Signorini --
tension closes the fracture and compression opens it.
"""

import numpy as np
import pytest

from mimetika.assembly.mixed import MixedElasticity
from mimetika.contact import ContactDriver, elastic_mechanics, LinearContact, SignoriniCoulomb
from mimetika.mesh import structured_quads, structured_triangles
from mimetika.mesh.fracture import facets_on_plane

MU = LAM = 1.0
OEDOMETER = 2 * MU + LAM
STRAIN = 0.01


def setup(n=2, law=None, tris=False, **kw):
    mesh = (structured_triangles if tris else structured_quads)(n, n)
    tags = facets_on_plane(mesh, [0.5, 0, 0], [1, 0, 0])
    law = SignoriniCoulomb(friction=0.6) if law is None else law
    return mesh, tags, ContactDriver(mesh, tags, law, mu=MU, lam=LAM, **kw)


def load(normal=0.0, shear=0.0):
    def u(x):
        x = np.atleast_2d(x)
        out = np.zeros((len(x), 3))
        out[:, 0] = normal * x[:, 0]
        out[:, 1] = shear * x[:, 0]
        return out

    return u


# -- the 2D discretization itself --------------------------------------------------


@pytest.mark.parametrize("tris", [False, True], ids=["quads", "triangles"])
def test_2d_elasticity_sizes(tris):
    mesh, _, d = setup(tris=tris)
    pb = MixedElasticity(mesh, mu=MU, lam=LAM)
    assert (pb.d, pb.ndf, pb.n_skew) == (2, 4, 1)
    assert pb.n_stress == 4 * mesh.num_cells(1)
    M, D, A = pb.assemble_operators()
    assert D.shape == (2 * mesh.num_cells(2), pb.n_stress)
    assert A.shape == (1 * mesh.num_cells(2), pb.n_stress)


def test_stabilization_vanishes_on_triangles_but_not_quads():
    """The simplex property in 2D: 3 edges x 4 DOFs = 12 = d^2(d+1)."""
    from mimetika.operators.elasticity import ElasticityInnerProduct

    quads, _, _ = setup(tris=False)
    tris, _, _ = setup(tris=True)
    assert ElasticityInnerProduct(tris).stabilization_dim(0) == 0
    assert ElasticityInnerProduct(quads).stabilization_dim(0) == 16 - 12


@pytest.mark.parametrize("tris", [False, True], ids=["quads", "triangles"])
def test_2d_patch_test_is_exact(tris):
    """A linear displacement must be reproduced to round-off in 2D too."""
    mesh, _, _ = setup(tris=tris)
    pb = MixedElasticity(mesh, mu=MU, lam=LAM)
    a = np.array([0.013, -0.007, 0.0])
    B = np.array([[0.02, -0.01, 0.0], [0.005, 0.015, 0.0], [0.0, 0.0, 0.0]])
    u = lambda x: a + np.atleast_2d(x) @ B.T  # noqa: E731
    sol = pb.solve(dirichlet=u, method="direct")
    assert np.allclose(
        sol["displacement"], pb.interpolate_displacement(u), atol=1e-12
    )


# -- the orientation convention ------------------------------------------------------


@pytest.mark.parametrize("tris", [False, True], ids=["quads", "triangles"])
def test_facet_normal_points_out_of_the_plus_incidence_cell(tris):
    """The convention Signorini depends on -- checked in 2D *and* 3D."""
    from mimetika.mesh import structured_box, structured_tets

    meshes = [
        (structured_triangles if tris else structured_quads)(2, 2),
        structured_tets(1, 1, 1) if tris else structured_box(2, 2, 2),
    ]
    for mesh in meshes:
        d = mesh.dim
        g = mesh.geometry
        for f in range(mesh.num_cells(d - 1)):
            col = mesh.complex.boundary_matrix(d).tocsr()[f]
            plus = [int(c) for c, v in zip(col.indices, col.data) if v > 0]
            if not plus:
                continue
            n = g.facet_frame(f)[0]
            away = g.centroids(d - 1)[f] - g.centroids(d)[plus[0]]
            assert float(away @ n) > 0


def test_facet_frame_is_orthonormal_in_2d():
    mesh, _, _ = setup()
    for f in range(mesh.num_cells(1)):
        frame = mesh.geometry.facet_frame(f)
        assert frame.shape == (2, 3)
        assert np.allclose(frame @ frame.T, np.eye(2), atol=1e-12)


# -- contact physics in 2D ------------------------------------------------------------


@pytest.mark.parametrize("tris", [False, True], ids=["quads", "triangles"])
def test_tension_opens_the_fracture(tris):
    mesh, tags, d = setup(tris=tris)
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=+STRAIN)))
    t = d.tractions(state.solution["stress"])
    assert state.converged
    assert np.allclose(t[:, 0], 0.0, atol=1e-12)
    assert (state.jump[:, 0] > 0).all()


@pytest.mark.parametrize("tris", [False, True], ids=["quads", "triangles"])
def test_compression_closes_without_interpenetration(tris):
    mesh, tags, d = setup(tris=tris)
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN)))
    t = d.tractions(state.solution["stress"])
    assert state.converged
    assert np.allclose(state.jump[:, 0], 0.0, atol=1e-8)
    assert np.allclose(t[:, 0], -OEDOMETER * STRAIN, rtol=1e-6)


def test_complementarity_holds_in_2d():
    mesh, tags, d = setup()
    for normal in (+STRAIN, -STRAIN, 0.0):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=normal)))
        t = d.tractions(state.solution["stress"])
        assert (state.jump[:, 0] > -1e-8).all()
        assert (t[:, 0] < 1e-10).all()
        assert np.allclose(state.jump[:, 0] * t[:, 0], 0.0, atol=1e-9)


def test_stick_and_slip_in_2d():
    """In 2D the friction set is an interval, not a disk."""
    mesh, tags, d = setup()

    stuck = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN, shear=0.002)))
    t = d.tractions(stuck.solution["stress"])
    assert (np.abs(t[:, 1]) < -d.law.friction * t[:, 0]).all()
    assert list(np.unique(d.law.status(t))) == [1]

    sliding = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN, shear=0.05)))
    t = d.tractions(sliding.solution["stress"])
    assert np.allclose(np.abs(t[:, 1]), -d.law.friction * t[:, 0], rtol=1e-6)
    assert list(np.unique(d.law.status(t))) == [2]


@pytest.mark.parametrize("friction", [0.2, 0.6, 1.0])
def test_the_2d_cap_scales_with_friction(friction):
    mesh, tags, d = setup(law=SignoriniCoulomb(friction=friction))
    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN, shear=0.5)))
    t = d.tractions(state.solution["stress"])
    assert np.allclose(np.abs(t[:, 1]), friction * OEDOMETER * STRAIN, rtol=1e-5)


def test_shear_traction_is_capped_in_2d():
    mesh, tags, d = setup()
    caps = []
    for shear in (0.05, 0.2, 1.0):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN, shear=shear)))
        caps.append(abs(d.tractions(state.solution["stress"])[0, 1]))
    assert np.allclose(caps, caps[0], rtol=1e-6)


# -- driver plumbing in 2D --------------------------------------------------------------


def test_driver_sizes_in_2d():
    mesh, tags, d = setup()
    assert (d.dim, d.ndf) == (2, 4)
    assert d.n_points == len(tags)
    s = d.initial_state()
    assert s.multiplier.shape == (len(tags), 4)
    assert s.jump.shape == (len(tags), 2)


@pytest.mark.parametrize("mode", ["averaged", "pointwise"])
def test_moment_value_round_trip_in_2d(mode):
    mesh, tags, d = setup(enforcement=mode)
    rng = np.random.default_rng(0)
    for f in tags:
        vals = rng.standard_normal((d.points_per_facet(int(f)), 2))
        back = d.to_values(d.to_moments(vals, int(f)), int(f))
        if mode == "averaged":
            assert np.allclose(back, vals)
        else:
            assert back.shape == vals.shape


def test_compliance_block_is_4x4_in_2d():
    mesh, tags, d = setup()
    for f in tags:
        blk = d._geom.block(int(f))
        assert blk.shape == (4, 4)
        assert np.allclose(blk, blk.T)
        assert np.linalg.eigvalsh(blk).min() > 0


def test_linear_law_in_2d_solves_in_one_pass():
    kn = 0.5
    mesh, tags, d = setup(law=LinearContact(kn, kn))
    delta = OEDOMETER * STRAIN / kn

    def u(x):
        x = np.atleast_2d(x)
        side = 1.0 if x[:, 0].mean() > 0.5 else 0.0
        out = np.zeros((len(x), 3))
        out[:, 0] = STRAIN * x[:, 0] + delta * side
        return out

    state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=u))
    assert state.iterations == 1
    assert np.allclose(state.jump[:, 0], delta, atol=1e-9)


def test_stepping_accumulates_slip_in_2d():
    mesh, tags, d = setup()
    state, history = None, []
    for shear in (0.05, 0.10, 0.15):
        state = d.solve_step(elastic_mechanics(mesh, MU, LAM, dirichlet=load(normal=-STRAIN, shear=shear)), state=state)
        assert state.converged
        history.append(state.internal[0, 0])
    assert history[0] < history[1] < history[2]
