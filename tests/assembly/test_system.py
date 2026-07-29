import numpy as np
import scipy.sparse as sp

from mimetika.assembly import apply_dirichlet, hodge_laplacian
from mimetika.operators import DiagonalHodge


def test_laplacian_is_symmetric(box_222):
    h = DiagonalHodge(box_222.geometry)
    L = hodge_laplacian(box_222, h, 0)
    assert (abs(L - L.T) > 1e-12).nnz == 0


def test_laplacian_annihilates_constants(box_222):
    h = DiagonalHodge(box_222.geometry)
    L = hodge_laplacian(box_222, h, 0)
    ones = np.ones(box_222.num_cells(0))
    assert np.allclose(L @ ones, 0.0)


def test_apply_dirichlet_enforces_values():
    # Small SPD system; constrain dof 0 = 5, dof 2 = -1.
    A = sp.csr_matrix(
        np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, 2]])
    )
    b = np.zeros(3)
    dofs = np.array([0, 2])
    vals = np.array([5.0, -1.0])
    A2, b2 = apply_dirichlet(A, b, dofs, vals)
    x = np.linalg.solve(A2.toarray(), b2)
    assert np.isclose(x[0], 5.0)
    assert np.isclose(x[2], -1.0)
    # interior row 1: 2*x1 = x0 + x2 -> x1 = (5 + -1)/2 = 2
    assert np.isclose(x[1], 2.0)


def test_apply_dirichlet_keeps_symmetry():
    A = sp.csr_matrix(np.array([[4.0, 1, 1], [1, 3, 0], [1, 0, 5]]))
    b = np.array([1.0, 2.0, 3.0])
    A2, _ = apply_dirichlet(A, b, np.array([0]), np.array([1.0]))
    assert (abs(A2 - A2.T) > 1e-12).nnz == 0
