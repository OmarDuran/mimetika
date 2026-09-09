r"""Solvers for the global mixed (saddle-point) systems.

The mixed systems are symmetric indefinite::

    [ M   B^T ] [ x ]   [ f ]
    [ B    0  ] [ y ] = [ g ]

so CG and plain AMG do not apply.  Two strategies:

* ``method="direct"`` -- sparse LU, up to a few hundred thousand unknowns.
* ``method="minres"`` -- MINRES with a block preconditioner.  With PETSc a
  ``fieldsplit``/Schur preconditioner, the Schur complement approximated by
  ``B diag(M)^{-1} B^T`` (PETSc's ``selfp``); with scipy the same block-diagonal
  preconditioner built explicitly.

Preconditioner variants (``preconditioner=``):

* ``"cpr"`` (default) -- the mixed-formulation analogue of the Constrained
  Pressure Residual preconditioner of reservoir simulation.  The elliptic
  operator needs no IMPES-style decoupling: in mixed form it is the Schur
  complement ``B diag(M)^{-1} B^T``, the cell-centred pressure Laplacian that
  ``selfp`` assembles.  Configuration: AMG (hypre BoomerAMG) on the Schur block,
  ICC on the leading block, both applied once (``preonly``).
* ``"schur"`` -- PETSc's sub-block defaults: an inner GMRES solve to ``rtol
  1e-5`` with ILU on each block, every outer iteration.

``cpr`` is roughly 10x faster on the fault mesh, where ``schur`` re-solves a
large inner system per outer iteration.  It is also the only admissible choice:
MINRES requires a fixed SPD preconditioner, and a nested GMRES solve makes the
preconditioner vary between iterations, invalidating the MINRES recurrence.

PETSc is used when ``petsc4py`` is importable, otherwise scipy.
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

    The leading block is a compliance, of order ``1/G``; the constraint block is
    a discrete divergence, of order one.  For rock (``G ~ 1e10``) that is an
    eleven-decade spread set by the choice of units, so row equilibration cannot
    remove it: row one mixes both scales and its maximum is dominated by the
    constraint entries.

    With ``sigma = s tilde-sigma`` and ``u = tilde-u / s`` the transformed blocks
    are ``s^2 M`` and ``B``, so ``s = sqrt(|B|/|M|)`` brings both to order one --
    stress nondimensionalised by the modulus.  On the fault-benchmark systems
    this takes ``cond ~ 2e11`` to ``~1e3``; without it MUMPS reports a zero
    pivot.

    Returns ``None`` when ``0.1 < s < 10`` (already commensurate) or the blocks
    are empty or identically zero.
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

    ``hypre`` is required for the AMG stage: ``gamg`` produced an indefinite
    preconditioner on the elasticity Schur block (PETSc reason -8), which MINRES
    cannot use.  Without hypre, ICC: symmetric, less scalable.
    """
    from petsc4py import PETSc

    schur = "hypre" if PETSc.Sys.hasExternalPackage("hypre") else "icc"
    return (
        "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type icc "
        f"-fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type {schur}"
    )


#: MUMPS ``INFOG(1)`` codes that mean "the workspace estimate was too small",
#: as opposed to a genuinely singular matrix (which is ``-10``).
_MUMPS_OUT_OF_SPACE = (-8, -9, -14, -15, -17, -20)


def _retry_if_out_of_workspace(ksp, b, x, verbose):
    """Grow MUMPS's working space and refactorise, if that is why it failed.

    MUMPS sizes its workspace from a symbolic estimate of the fill-in; when the
    numeric factorisation exceeds it, PETSc reports ``KSP_DIVERGED_PC_FAILED``,
    the same code as for a zero pivot.  ``INFOG(1)`` separates them: the codes in
    :data:`_MUMPS_OUT_OF_SPACE` are a sizing failure over a well-posed system,
    and the remedy is to raise ``ICNTL(14)``, the percentage by which the
    estimate is inflated, and factorise again.  High cell aspect ratios generate
    more fill-in than the estimate anticipates and trip this.

    ``ICNTL(14)`` is escalated on demand through 100, 400, 1600, so the common
    case pays no extra memory.  Returns the final converged reason, or ``None``
    if no retry was warranted.
    """
    from petsc4py import PETSc

    if ksp.getConvergedReason() >= 0:
        return None
    pc = ksp.getPC()
    if pc.getType() != "lu" or pc.getFactorSolverType() != "mumps":
        return None
    try:
        infog = pc.getFactorMatrix().getMumpsInfog(1)
    except Exception:  # pragma: no cover - older petsc4py without the accessor
        return None
    if infog not in _MUMPS_OUT_OF_SPACE:
        return None  # a real numerical failure; let the caller raise

    # A spent PC cannot be re-run: PETSc caches the failed factorisation, and
    # ICNTL(14) must be set before the numeric phase, so each attempt gets a
    # fresh KSP over the same operator.
    mat = ksp.getOperators()[0]
    options = PETSc.Options()
    try:
        for extra in (100, 400, 1600):
            if verbose:
                print(f"    MUMPS INFOG(1)={infog}: retrying with ICNTL(14)={extra}")
            options["mat_mumps_icntl_14"] = extra
            retry = PETSc.KSP().create()
            retry.setOperators(mat)
            retry.setType("preonly")
            retry_pc = retry.getPC()
            retry_pc.setType("lu")
            retry_pc.setFactorSolverType("mumps")
            retry.setFromOptions()
            retry.solve(b, x)
            if retry.getConvergedReason() >= 0:
                return retry.getConvergedReason()
        return retry.getConvergedReason()
    finally:
        # the options database is global and outlives this call
        options.delValue("mat_mumps_icntl_14")


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
    reason = _retry_if_out_of_workspace(ksp, b, x, verbose)

    reason = ksp.getConvergedReason() if reason is None else reason
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
