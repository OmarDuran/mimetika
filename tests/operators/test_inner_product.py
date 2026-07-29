import numpy as np

from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    consistency_matrix,
    nullspace_basis,
    stabilization_dim,
)


def test_nullspace_basis_orthonormal_and_complementary():
    # N: 5 DOFs, 2 modes -> left null space has dim 3.
    rng = np.random.default_rng(0)
    N = rng.standard_normal((5, 2))
    C = nullspace_basis(N)
    assert C.shape == (5, 3)
    assert np.allclose(C.T @ C, np.eye(3))  # orthonormal
    assert np.allclose(N.T @ C, 0.0)  # C spans ker(N^T)


def test_nullspace_empty_when_full_rank():
    N = np.eye(4)
    assert nullspace_basis(N).shape == (4, 0)
    assert stabilization_dim(N) == 0


def test_consistency_matrix_is_exact_on_range():
    # M1 reproduces the exact inner product |E| c^T Kbar c for g = N c.
    rng = np.random.default_rng(1)
    N = rng.standard_normal((6, 3))
    A = rng.standard_normal((3, 3))
    Kbar = A @ A.T + np.eye(3)  # SPD
    vol = 2.0
    M1 = consistency_matrix(N, Kbar, vol)
    for _ in range(5):
        c = rng.standard_normal(3)
        g = N @ c
        assert np.isclose(g @ M1 @ g, vol * (c @ Kbar @ c))


def test_stabilization_zero_iff_full_rank():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 4))
    Kbar = A @ A.T + np.eye(4)
    # D = m = 4 -> no stabilization; M_E is SPD from consistency alone.
    N = rng.standard_normal((4, 4))
    M = assemble_local_inner_product(N, Kbar, 1.0, stability_scale=1.0)
    assert np.allclose(M, consistency_matrix(N, Kbar, 1.0))
    assert np.linalg.eigvalsh(M).min() > 0


def test_local_matrix_spd_with_stabilization():
    rng = np.random.default_rng(3)
    N = rng.standard_normal((6, 3))
    A = rng.standard_normal((3, 3))
    Kbar = A @ A.T + np.eye(3)
    M = assemble_local_inner_product(N, Kbar, 1.5, stability_scale=0.5)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0
