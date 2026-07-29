"""Assembly layer: build global operators and linear systems.

Combines the metric-free exterior derivatives with a Hodge inner product to form
the weak mimetic operators.  The canonical building block is the Hodge Laplacian
on k-forms; for k=0 this is the standard weak Poisson operator

    ``L_0 = D_0^T  M_1  D_0``

with ``D_0 = grad`` and ``M_1`` the Hodge on 1-forms (edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mimetika.mesh.mesh import Mesh
from mimetika.operators.derivative import exterior_derivative
from mimetika.operators.hodge import HodgeOperator


@dataclass
class LinearSystem:
    """A global linear system ``A x = b`` in scipy form (backend-neutral)."""

    A: sp.csr_matrix
    b: np.ndarray


def hodge_laplacian(mesh: Mesh, hodge: HodgeOperator, k: int) -> sp.csr_matrix:
    """Primal weak Laplacian on k-forms: ``D_k^T M_{k+1} D_k``.

    This is the ``d*d`` half of the full Hodge Laplacian, which is the relevant
    operator for a Dirichlet Poisson problem when ``k = 0``.
    """
    d = exterior_derivative(mesh, k)
    m = hodge.matrix(k + 1)
    return (d.T @ m @ d).tocsr()


def apply_dirichlet(
    A: sp.csr_matrix,
    b: np.ndarray,
    dofs: np.ndarray,
    values: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Impose ``x[dofs] = values`` by symmetric row/column elimination.

    Returns new ``(A, b)``; the constrained rows/cols are replaced by identity.
    """
    dofs = np.asarray(dofs, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    b = b.astype(float, copy=True)

    # 1) Move known values to the RHS using the *original* columns.
    b = b - np.asarray(A[:, dofs] @ values).ravel()

    # 2) Zero the constrained rows (CSR) and columns (CSC).
    A = A.tocsr(copy=True).astype(float)
    for d in dofs:
        A.data[A.indptr[d] : A.indptr[d + 1]] = 0.0
    A = A.tocsc()
    for d in dofs:
        A.data[A.indptr[d] : A.indptr[d + 1]] = 0.0
    A.eliminate_zeros()

    # 3) Restore identity on the diagonal and set the RHS to the known values.
    A = A.tolil()
    for d, v in zip(dofs, values):
        A[d, d] = 1.0
        b[d] = v
    return A.tocsr(), b
