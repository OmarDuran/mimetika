import numpy as np

from mimetika.mesh import single_tetrahedron, structured_box
from mimetika.operators.elasticity import (
    ElasticityInnerProduct,
    compliance_contraction,
)


def test_local_matrix_spd():
    for mesh in (single_tetrahedron(), structured_box(1, 1, 1)):
        M, _ = ElasticityInnerProduct(mesh, mu=1.3, lam=2.7).local(0)
        assert np.allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0


def test_stabilization_vanishes_on_simplex():
    tet = single_tetrahedron()
    ip = ElasticityInnerProduct(tet)
    M, _ = ip.local(0)
    assert M.shape == (36, 36)  # 9 DOFs x 4 facets
    assert ip.stabilization_dim(0) == 0  # <-- the design requirement


def test_stabilization_active_on_hexahedron():
    hexm = structured_box(1, 1, 1)
    ip = ElasticityInnerProduct(hexm)
    M, _ = ip.local(0)
    assert M.shape == (54, 54)  # 9 DOFs x 6 facets
    assert ip.stabilization_dim(0) == 18  # 54 - 36


def test_consistency_exact_on_linear_tensor_fields():
    # On the tet (pure consistency) M reproduces int C^{-1} T:T for every
    # linear tensor field T in the reconstruction space.
    mu, lam = 1.3, 2.7
    tet = single_tetrahedron()
    ip = ElasticityInnerProduct(tet, mu=mu, lam=lam)
    N, _, _, _ = ip._N_and_Kbar(0)
    M, _ = ip.local(0)
    qp, qw = tet.geometry.cell_quadrature(0)
    xE = tet.geometry.centroids(3)[0]
    hE = ip._cell_diameter(0)
    modes = ip._eval_modes(qp, xE, hE)
    rng = np.random.default_rng(0)
    for _ in range(4):
        c = rng.standard_normal(36)
        g = N @ c
        T = np.einsum("m,qmij->qij", c, modes)
        exact = np.einsum("q,q->", qw, compliance_contraction(T, T, mu, lam))
        assert np.isclose(g @ M @ g, exact)


def test_incompressible_limit_stays_spd():
    # Uniform stability in lambda -> the matrix stays well-defined as lam -> inf.
    hexm = structured_box(1, 1, 1)
    for lam in (1.0, 1e3, 1e6):
        M, _ = ElasticityInnerProduct(hexm, mu=1.0, lam=lam).local(0)
        assert np.linalg.eigvalsh(M).min() > 0


def test_global_assembly_shape_and_symmetry():
    mesh = structured_box(1, 1, 1)
    A = ElasticityInnerProduct(mesh).assemble()
    assert A.shape == (9 * mesh.num_cells(2), 9 * mesh.num_cells(2))
    assert (abs(A - A.T) > 1e-10).nnz == 0
