r"""Solvers for the global mixed (saddle-point) systems.

The mixed systems are **symmetric indefinite**::

    [ M   B^T ] [ x ]   [ f ]
    [ B    0  ] [ y ] = [ g ]

so the usual SPD machinery (CG, plain AMG) does not apply.  Two strategies are
provided:

* ``method="direct"`` -- a sparse LU factorisation.  Reliable, and the right
  choice up to a few hundred thousand unknowns.
* ``method="minres"`` -- MINRES with a block preconditioner.  With PETSc this
  is a ``fieldsplit``/Schur preconditioner (the Schur complement approximated by
  ``B diag(M)^{-1} B^T``, PETSc's ``selfp``); with scipy it is the equivalent
  block-diagonal preconditioner built explicitly.  This is what scales to the
  large stress systems, where a direct factorisation is out of reach.

Preconditioner variants (``preconditioner=``):

* ``"cpr"`` (default) -- the mixed-formulation analogue of the **Constrained
  Pressure Residual** preconditioner of reservoir simulation.  CPR's idea is to
  solve the elliptic pressure subsystem with AMG and follow it with a cheap
  global smoother.  Here the elliptic operator is not extracted by an IMPES-style
  decoupling: in mixed form it *is* the Schur complement ``B diag(M)^{-1} B^T``,
  the cell-centred pressure Laplacian, which ``selfp`` already assembles.  So the
  configuration is AMG (hypre BoomerAMG) on the Schur block and a cheap
  incomplete factorisation on the leading block, both applied **once**
  (``preonly``).
* ``"schur"`` -- PETSc's defaults for the sub-blocks: an inner GMRES solve to
  ``rtol 1e-5`` with ILU on *each* block, every outer iteration.

``cpr`` is both faster and more correct.  Faster because the default re-solves a
large inner system per outer iteration (roughly 10x on the fault mesh).  More
correct because MINRES requires a **fixed, symmetric positive-definite**
preconditioner: a nested GMRES solve makes the preconditioner *variable*, which
formally invalidates the MINRES recurrence, whereas ``preonly`` with ICC/AMG
keeps it fixed and symmetric.

PETSc is used when ``petsc4py`` is importable; otherwise everything falls back
to scipy transparently.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.assembly.backend import petsc_available


def solve_saddle(
    A: sp.spmatrix,
    rhs: np.ndarray,
    block_sizes: tuple[int, ...],
    backend: str = "auto",
    method: str = "minres",
    rtol: float = 1e-12,
    max_it: int = 5000,
    verbose: bool = False,
    options: str | None = None,
    preconditioner: str = "cpr",
    scale_blocks: bool = True,
) -> np.ndarray:
    """Solve a symmetric indefinite saddle-point system.

    Parameters
    ----------
    A, rhs
        The assembled system.
    block_sizes
        Sizes of the field blocks, e.g. ``(n_flux, n_pressure)``.  The first
        block is treated as the "velocity-like" field, the rest as multipliers.
    backend
        ``"petsc"``, ``"scipy"`` or ``"auto"``.
    method
        ``"direct"`` or ``"minres"``.
    scale_blocks
        Nondimensionalise the two blocks against each other before solving (see
        :func:`block_scaling`).  On by default; it is a similarity transform, so
        the solution is unchanged.
    """
    if backend == "auto":
        backend = "petsc" if petsc_available() else "scipy"

    scaling = block_scaling(A, block_sizes) if scale_blocks else None
    if scaling is not None:
        D = sp.diags(scaling)
        A, rhs = (D @ A @ D).tocsr(), scaling * rhs

    if backend == "petsc":
        x = _solve_petsc(
            A, rhs, block_sizes, method, rtol, max_it, verbose, options,
            preconditioner,
        )
    elif options:
        raise ValueError("PETSc options were given but the scipy backend is in use")
    else:
        x = _solve_scipy(A, rhs, block_sizes, method, rtol, max_it, verbose)
    return scaling * x if scaling is not None else x


def block_scaling(A: sp.spmatrix, block_sizes) -> np.ndarray | None:
    """Diagonal ``D`` making the two blocks of ``D A D`` commensurate.

    In physical units the leading block is a compliance, of order ``1/G``, while
    the constraint block is a discrete divergence of order one.  For rock
    (``G ~ 1e10``) that is an eleven-order-of-magnitude spread, and the resulting
    condition number is *intrinsic* to the choice of units, not an artefact --
    which is why row-equilibration cannot touch it: row one already mixes both
    scales, so its maximum is dominated by the constraint entries.

    Scaling the two fields against each other does fix it.  With
    ``sigma = s tilde-sigma`` and ``u = tilde-u / s``, the transformed blocks are
    ``s^2 M`` and ``B``, so ``s = sqrt(|B|/|M|)`` brings both to order one --
    exactly nondimensionalising stress by the modulus.  In practice this takes
    the fault-benchmark systems from ``cond ~ 2e11`` to ``~1e3``, which is the
    difference between MUMPS reporting a zero pivot and factorising cleanly.

    Returns ``None`` when the scaling would be a no-op or cannot be formed.
    """
    n0 = int(block_sizes[0])
    total = A.shape[0]
    if not 0 < n0 < total:
        return None
    leading = abs(A[:n0, :n0]).max()
    constraint = abs(A[n0:, :n0]).max()
    if not (leading > 0 and constraint > 0):
        return None
    s = float(np.sqrt(constraint / leading))
    if not np.isfinite(s) or 0.1 < s < 10.0:  # already commensurate
        return None
    scaling = np.empty(total)
    scaling[:n0] = s
    scaling[n0:] = 1.0 / s
    return scaling


# -- scipy --------------------------------------------------------------------


def _block_preconditioner(A: sp.csr_matrix, block_sizes) -> spla.LinearOperator:
    """Block-diagonal ``diag(M)`` / Schur preconditioner for MINRES."""
    n0 = block_sizes[0]
    A = A.tocsr()
    M = A[:n0, :n0]
    B = A[n0:, :n0]

    dM = M.diagonal()
    dM = np.where(np.abs(dM) < 1e-300, 1.0, dM)
    # approximate Schur complement  S ~ B diag(M)^{-1} B^T   (SPD)
    S = (B @ sp.diags(1.0 / dM) @ B.T).tocsc()
    S = S + sp.eye(S.shape[0], format="csc") * (1e-12 * abs(S.diagonal()).max())
    S_solve = spla.factorized(S)

    def apply(v):
        out = np.empty_like(v)
        out[:n0] = v[:n0] / dM
        out[n0:] = S_solve(v[n0:])
        return out

    return spla.LinearOperator(A.shape, matvec=apply, dtype=float)


def _solve_scipy(A, rhs, block_sizes, method, rtol, max_it, verbose):
    if method == "direct":
        return spla.spsolve(A.tocsc(), rhs)

    P = _block_preconditioner(A, block_sizes)
    it = {"n": 0}

    def cb(_):
        it["n"] += 1

    x, info = spla.minres(
        A.tocsr(), rhs, M=P, rtol=rtol, maxiter=max_it, callback=cb
    )
    if verbose:
        print(f"    scipy MINRES: {it['n']} iterations, info={info}")
    if info != 0:
        raise RuntimeError(f"MINRES did not converge (info={info})")
    return x


# -- PETSc --------------------------------------------------------------------


def _cpr_options() -> str:
    """Sub-block options for the CPR-style preconditioner.

    ``hypre`` is required for the AMG stage: ``gamg`` was measured to produce an
    *indefinite* preconditioner on the elasticity Schur block (PETSc reason -8,
    with a visibly wrong answer), which MINRES cannot use.  Without hypre we
    fall back to ICC, which is symmetric and safe if less scalable.
    """
    from petsc4py import PETSc

    schur = "hypre" if PETSc.Sys.hasExternalPackage("hypre") else "icc"
    return (
        "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type icc "
        f"-fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type {schur}"
    )


def _solve_petsc(
    A, rhs, block_sizes, method, rtol, max_it, verbose, options=None,
    preconditioner="cpr",
):
    from petsc4py import PETSc

    from mimetika.assembly.backend import (
        insert_petsc_options,
        to_petsc_mat,
        to_petsc_vec,
    )

    mat = to_petsc_mat(A)
    b = to_petsc_vec(rhs)
    x = b.duplicate()

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)

    if method == "direct":
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        # PETSc's built-in serial LU struggles on large indefinite systems;
        # prefer MUMPS when the installation provides it.
        for pkg in ("mumps", "superlu"):
            if PETSc.Sys.hasExternalPackage(pkg):
                pc.setFactorSolverType(pkg)
                if verbose:
                    print(f"    PETSc direct solver: {pkg}")
                break
    else:
        n0 = block_sizes[0]
        total = sum(block_sizes)
        is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=PETSc.IntType))
        is1 = PETSc.IS().createGeneral(np.arange(n0, total, dtype=PETSc.IntType))

        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(("0", is0), ("1", is1))
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.DIAG)
        # approximate the Schur complement by B diag(M)^{-1} B^T
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
        ksp.setTolerances(rtol=rtol, max_it=max_it)
        if preconditioner == "cpr":
            # inserted before setFromOptions so an explicit --petsc-opts wins
            insert_petsc_options(_cpr_options())
        elif preconditioner != "schur":
            raise ValueError(f"unknown preconditioner {preconditioner!r}")

    if options:  # user options last, so they override the built-in defaults
        insert_petsc_options(options)
    ksp.setFromOptions()  # let -ksp_* / -pc_* command-line options win
    ksp.solve(b, x)

    reason = ksp.getConvergedReason()
    if verbose:
        # every field below is queried from the live PETSc objects
        pc = ksp.getPC()
        detail = f"    PETSc KSP={ksp.getType()} PC={pc.getType()}"
        if pc.getType() == "lu":
            detail += f" ({pc.getFactorSolverType()})"
        elif pc.getType() == "fieldsplit":
            # name the sub-block solvers: 'fieldsplit' alone does not say
            # whether the CPR configuration is actually in force
            try:
                subs = pc.getFieldSplitSubKSP()
                inner = " + ".join(
                    f"{k.getType()}/{k.getPC().getType()}" for k in subs
                )
                detail += f"[{inner}]"
            except Exception:  # pragma: no cover - PETSc build dependent
                pass
        detail += (
            f" | {ksp.getIterationNumber()} iterations,"
            f" reason={reason}, rnorm={ksp.getResidualNorm():.3e}"
        )
        print(detail)
    if reason < 0:
        raise RuntimeError(f"PETSc KSP diverged (reason={reason})")
    return x.getArray().copy()
