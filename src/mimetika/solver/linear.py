"""Linear solver layer.

Solves ``A x = b`` via PETSc's KSP when petsc4py is available, otherwise via a
scipy direct/iterative fallback.  The public API is backend-neutral: callers
pass a :class:`~mimetika.assembly.system.LinearSystem` and get a numpy solution.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from mimetika.assembly.backend import petsc_available
from mimetika.assembly.system import LinearSystem


def solve(
    system: LinearSystem,
    backend: str = "auto",
    ksp_type: str = "cg",
    pc_type: str = "gamg",
    rtol: float = 1e-10,
    max_it: int = 10000,
) -> np.ndarray:
    """Solve ``A x = b``.

    Parameters
    ----------
    backend
        ``"petsc"``, ``"scipy"`` or ``"auto"`` (petsc if available).
    ksp_type, pc_type, rtol, max_it
        PETSc KSP/PC options (ignored by the scipy backend).
    """
    if backend == "auto":
        backend = "petsc" if petsc_available() else "scipy"

    if backend == "petsc":
        return _solve_petsc(system, ksp_type, pc_type, rtol, max_it)
    if backend == "scipy":
        return _solve_scipy(system)
    raise ValueError(f"unknown backend {backend!r}")


def _solve_scipy(system: LinearSystem) -> np.ndarray:
    return spla.spsolve(system.A.tocsc(), system.b)


def _solve_petsc(
    system: LinearSystem, ksp_type: str, pc_type: str, rtol: float, max_it: int
) -> np.ndarray:
    from petsc4py import PETSc

    from mimetika.assembly.backend import to_petsc_mat, to_petsc_vec

    A = to_petsc_mat(system.A)
    b = to_petsc_vec(system.b)
    x = b.duplicate()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.setFromOptions()
    ksp.solve(b, x)

    return x.getArray().copy()
