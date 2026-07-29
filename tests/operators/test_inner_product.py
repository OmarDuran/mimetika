import numpy as np

from mimetika.operators.inner_product import (
    assemble_local_inner_product,
    complete_moments,
    consistency_matrix,
    consistency_residual,
    min_norm_moments,
    nullspace_basis,
    stabilization_dim,
)


def _spd(rng, m):
    A = rng.standard_normal((m, m))
    return A @ A.T + m * np.eye(m)


def _valid_pair(rng, D, m, volume):
    """A consistent ``(N, R, Kbar)`` triple with ``N^T R = |E| Kbar``."""
    N = rng.standard_normal((D, m))
    Kbar = _spd(rng, m)
    R = complete_moments(N, Kbar, volume, np.zeros((D, 0)))
    return N, R, Kbar


def test_nullspace_basis_orthonormal_and_complementary():
    rng = np.random.default_rng(0)
    N = rng.standard_normal((5, 2))
    C = nullspace_basis(N)
    assert C.shape == (5, 3)
    assert np.allclose(C.T @ C, np.eye(3))
    assert np.allclose(N.T @ C, 0.0)


def test_nullspace_empty_when_full_rank():
    N = np.eye(4)
    assert nullspace_basis(N).shape == (4, 0)
    assert stabilization_dim(N) == 0


def test_min_norm_moments_solves_the_gram_identity():
    rng = np.random.default_rng(1)
    N = rng.standard_normal((7, 3))
    target = rng.standard_normal((3, 3))
    R = min_norm_moments(N, target)
    assert np.allclose(N.T @ R, target)
    # minimum norm => R lies in range(N)
    assert np.allclose(nullspace_basis(N).T @ R, 0.0)


def test_complete_moments_preserves_canonical_columns():
    rng = np.random.default_rng(2)
    D, m, vol = 8, 4, 1.7
    N = rng.standard_normal((D, m))
    Kbar = _spd(rng, m)
    # two canonical columns, consistent with the Gram identity by construction
    R_canon = min_norm_moments(N, vol * Kbar[:, :2])
    R = complete_moments(N, Kbar, vol, R_canon)
    assert R.shape == (D, m)
    assert np.allclose(R[:, :2], R_canon)  # canonical columns untouched
    assert np.allclose(N.T @ R, vol * Kbar)  # identity holds for all columns


def test_strong_consistency_M_N_equals_R():
    """The defining property: M N = R, not merely N^T M N = |E| Kbar."""
    rng = np.random.default_rng(3)
    for D, m in [(4, 4), (6, 3), (9, 4)]:
        vol = 0.7
        N, R, Kbar = _valid_pair(rng, D, m, vol)
        M = assemble_local_inner_product(N, R, Kbar, vol)
        assert consistency_residual(M, N, R) < 1e-10


def test_energy_identity_follows_from_consistency():
    rng = np.random.default_rng(4)
    vol = 1.3
    N, R, Kbar = _valid_pair(rng, 6, 3, vol)
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert np.allclose(N.T @ M @ N, vol * Kbar)


def test_stabilization_zero_iff_full_rank():
    rng = np.random.default_rng(5)
    vol = 1.0
    N, R, Kbar = _valid_pair(rng, 4, 4, vol)  # D == m
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert stabilization_dim(N) == 0
    assert np.allclose(M, consistency_matrix(R, Kbar, vol))
    assert np.linalg.eigvalsh(M).min() > 0


def test_local_matrix_spd_with_stabilization():
    rng = np.random.default_rng(6)
    vol = 1.5
    N, R, Kbar = _valid_pair(rng, 6, 3, vol)
    assert stabilization_dim(N) == 3
    M = assemble_local_inner_product(N, R, Kbar, vol)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0


def test_consistency_matrix_alone_is_singular_when_D_exceeds_m():
    rng = np.random.default_rng(7)
    vol = 1.0
    N, R, Kbar = _valid_pair(rng, 6, 3, vol)
    M1 = consistency_matrix(R, Kbar, vol)
    assert np.linalg.matrix_rank(M1) == 3  # stabilization is what makes it SPD
