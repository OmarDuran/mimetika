import numpy as np
import pytest

from mimetika.mesh import single_tetrahedron, structured_box
from mimetika.operators.diffusion import DiffusionInnerProduct

ANISO_K = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.0]])


@pytest.mark.parametrize("basis", ["const", "rt0"])
def test_local_matrix_is_spd(basis):
    for mesh in (single_tetrahedron(), structured_box(1, 1, 1)):
        M, _ = DiffusionInnerProduct(mesh, K=ANISO_K, basis=basis).local(0)
        assert np.allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0


def test_stabilization_dims():
    tet, hexm = single_tetrahedron(), structured_box(1, 1, 1)
    # const (m=3): tet has 4 facets -> 1, hex has 6 -> 3.
    assert DiffusionInnerProduct(tet, basis="const").stabilization_dim(0) == 1
    assert DiffusionInnerProduct(hexm, basis="const").stabilization_dim(0) == 3
    # rt0 (m=4): vanishes on the tetrahedron, 2 on the hexahedron.
    assert DiffusionInnerProduct(tet, basis="rt0").stabilization_dim(0) == 0
    assert DiffusionInnerProduct(hexm, basis="rt0").stabilization_dim(0) == 2


def test_consistency_constant_flux_hex():
    # M reproduces int_E K^{-1} F.F for any constant flux F (const basis).
    mesh = structured_box(1, 1, 1)
    ip = DiffusionInnerProduct(mesh, K=ANISO_K, basis="const")
    M, fids = ip.local(0)
    normals = mesh.geometry.facet_normals()[fids]
    vol = mesh.geometry.measure(3)[0]
    Kinv = np.linalg.inv(ANISO_K)
    for F in (np.array([1.0, 0, 0]), np.array([0.7, -0.4, 1.2])):
        g = normals @ F  # DOF vector of the constant flux
        assert np.isclose(g @ M @ g, vol * (F @ Kinv @ F))


def test_rt0_exact_on_linear_flux_on_tet():
    # With no stabilization, M equals the continuous inner product on all of RT0.
    tet = single_tetrahedron()
    ip = DiffusionInnerProduct(tet, K=ANISO_K, basis="rt0")
    M, fids = ip.local(0)
    g = tet.geometry
    normals = g.facet_normals()[fids]
    fc = g.facet_centroids()[fids]
    xE = g.centroids(3)[0]
    a, b = np.array([0.2, 0.5, -0.3]), 0.8  # F = a + b (x - xE)
    gd = np.einsum("ic,ic->i", a + b * (fc - xE), normals)
    qp, qw = g.cell_quadrature(0)
    F = a + b * (qp - xE)
    Kinv = np.linalg.inv(ANISO_K)
    exact = np.einsum("q,qc,cd,qd->", qw, F, Kinv, F)
    assert np.isclose(gd @ M @ gd, exact)


def test_global_assembly_symmetric_and_shared_facets():
    mesh = structured_box(2, 1, 1)
    A = DiffusionInnerProduct(mesh).assemble()
    assert A.shape == (mesh.num_cells(2), mesh.num_cells(2))
    assert (abs(A - A.T) > 1e-12).nnz == 0
    assert np.linalg.eigvalsh(A.toarray()).min() > 0
