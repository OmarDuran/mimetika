"""Matrix/vector backend abstraction.

The assembly and solver layers work with scipy sparse matrices by default and
transparently convert to PETSc (petsc4py) objects when PETSc is installed and
requested.  Keeping the conversion here means no other layer imports petsc4py.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def petsc_available() -> bool:
    try:
        import petsc4py  # noqa: F401

        return True
    except ImportError:
        return False


def with_explicit_diagonal(A: sp.spmatrix) -> sp.csr_matrix:
    """Ensure every diagonal entry is *structurally* present (possibly zero).

    Saddle-point systems have an identically-zero trailing block, so those
    diagonal entries are absent from the sparsity pattern -- and PETSc's
    factorisations reject a matrix "missing diagonal entries".  Adding explicit
    zeros changes nothing numerically.
    """
    A = A.tocoo()
    n = min(A.shape)
    idx = np.arange(n)
    out = sp.csr_matrix(
        (
            np.concatenate([A.data, np.zeros(n)]),
            (np.concatenate([A.row, idx]), np.concatenate([A.col, idx])),
        ),
        shape=A.shape,
    )
    return out


def insert_petsc_options(options: str) -> None:
    """Push a PETSc options string (e.g. ``"-ksp_view -ksp_monitor"``) into the
    global options database, where ``KSP.setFromOptions()`` will pick it up.

    This is the authoritative way to inspect or override the solver: the values
    come from PETSc itself, not from anything this library reports.
    """
    from petsc4py import PETSc

    opts = PETSc.Options()
    tokens = options.split()
    i = 0
    while i < len(tokens):
        key = tokens[i].lstrip("-")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            opts[key] = tokens[i + 1]
            i += 2
        else:
            opts[key] = None
            i += 1


def to_petsc_mat(A: sp.spmatrix, ensure_diagonal: bool = True):
    """Convert a scipy sparse matrix to a PETSc AIJ ``Mat``."""
    from petsc4py import PETSc

    A = with_explicit_diagonal(A) if ensure_diagonal else A.tocsr()
    mat = PETSc.Mat().createAIJ(
        size=A.shape, csr=(A.indptr, A.indices, A.data)
    )
    mat.assemble()
    return mat


def to_petsc_vec(v: np.ndarray):
    """Convert a numpy array to a PETSc ``Vec``."""
    from petsc4py import PETSc

    vec = PETSc.Vec().createWithArray(np.ascontiguousarray(v, dtype=float))
    return vec
