r"""Mimetic inner products (the metric / material heart of the method).

Implements the *consistency + stability* construction of local inner-product
(mass) matrices common to the mimetic finite difference family
(Brezzi--Lipnikov--Simoncini) and its elasticity extension
(Beir\~ao da Veiga, ESAIM M2AN 44 (2010) 231--250, section 4).

For one element ``E`` with ``D`` degrees of freedom and a *reconstruction space*
of ``m <= D`` modes one builds

    ``N``    (D x m) : the DOFs of each reconstruction mode,
    ``R``    (D x m) : the matching *moment* matrix,
    ``Kbar`` (m x m) : Gram matrix of the modes in the continuous inner product,

related by the fundamental identity  ``N^T R = |E| Kbar``.  Then

    ``M_E = M1 + M2``,
    ``M1  = (1/|E|) R Kbar^{-1} R^T``            (consistency, rank m),
    ``M2  = s C C^T``,  C = orthonormal basis of ker(N^T)   (stability).

Two properties matter, and both are checked in the tests:

* **Strong consistency**: ``M_E N = R`` exactly.  (``M1 N = R`` because
  ``R^T N = |E| Kbar``; ``M2 N = 0`` because ``C^T N = 0``.)  This is what makes
  a *local mixed solve* reproduce exact polynomial fields -- strictly stronger
  than the energy identity ``N^T M N = |E| Kbar``, which alone does **not**
  give exact local solves on general polytopes.

* **Stabilization vanishes on simplices**: ``M2 = 0`` iff ``ker(N^T) = {0}`` iff
  ``D = m``.  Choosing the reconstruction space equal to the target mixed-FE
  space -- unisolvent on a simplex -- gives ``D = m`` there, so stabilization is
  active only on genuine polytopes.

Where does ``R`` come from?
---------------------------
A column of ``R`` is *canonical* whenever the corresponding mode ``w`` admits a
potential (``metric^{-1} w = grad psi``): integrating by parts moves the pairing
onto the facets, where the DOFs live, giving an exactly computable column.  The
remaining columns -- modes with no potential -- are not determined by any
identity; they are completed by the **minimum-norm** solution of
``N^T R = |E| Kbar``.  (Completing *every* column that way reproduces the
projection form ``M1 = |E| N (N^T N)^{-1} Kbar (N^T N)^{-1} N^T``, which is why
that form fails strong consistency: it discards the canonical columns.)
"""

from __future__ import annotations

import numpy as np


def nullspace_basis(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis of ``{x : A^T x = 0}`` = ``range(A)^perp`` in ``R^D``.

    Shape ``(D, D - rank A)``; an empty ``(D, 0)`` array when the columns of
    ``A`` already span ``R^D`` (the simplex case).
    """
    A = np.asarray(A, dtype=float)
    D = A.shape[0]
    if A.size == 0:
        return np.eye(D)
    U, s, _ = np.linalg.svd(A, full_matrices=True)
    tol = rtol * (s[0] if s.size else 1.0) * max(A.shape)
    return U[:, int(np.sum(s > tol)) :]


def stabilization_dim(N: np.ndarray) -> int:
    """Dimension of the local stabilization space ``dim ker(N^T)``.

    Exactly zero when the reconstruction is unisolvent for the DOFs (a simplex
    with a matched reconstruction space); positive on genuine polytopes.
    """
    return int(nullspace_basis(N).shape[1])


def min_norm_moments(N: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimum-norm solution of ``N^T R = target``: ``R = N (N^T N)^{-1} target``."""
    N = np.asarray(N, dtype=float)
    return N @ np.linalg.solve(N.T @ N, np.asarray(target, dtype=float))


def complete_moments(
    N: np.ndarray, Kbar: np.ndarray, volume: float, R_canonical: np.ndarray
) -> np.ndarray:
    """Full ``R`` from its canonical leading columns plus a min-norm completion.

    ``R_canonical`` holds the first ``mc`` columns (the modes that admit a
    potential); the remaining ``m - mc`` columns are the minimum-norm solution
    of ``N^T R = |E| Kbar`` restricted to those columns.
    """
    R_canonical = np.asarray(R_canonical, dtype=float)
    mc = R_canonical.shape[1]
    m = np.asarray(Kbar).shape[0]
    if mc == m:
        return R_canonical
    rest = min_norm_moments(N, volume * np.asarray(Kbar)[:, mc:])
    return np.hstack([R_canonical, rest])


def apply_pseudo_inverse(Kbar: np.ndarray, B: np.ndarray, rtol: float = 1e-12):
    """``Kbar^+ B`` for symmetric positive *semi*-definite ``Kbar`` (batched).

    ``Kbar`` is only positive **semi**-definite in the incompressible limit: the
    elasticity Gram matrix is ``kron(G, (I - a vec(I) vec(I)^T)/2mu)``, whose
    eigenvalue ``1 - a d`` vanishes exactly at ``a = 1/d``, i.e. ``nu = 1/2``.
    That direction is the hydrostatic mode, which stores no energy -- so it is
    not a defect to be regularised away but a genuine null space, and the
    pseudo-inverse is the right inverse to use.

    It costs nothing in accuracy: ``ker(Kbar) subset ker(R)`` (a mode with zero
    compliance energy also has zero moments), so the rows of ``R`` lie in
    ``range(Kbar)`` and ``R Kbar^+ Kbar = R`` holds exactly.  Strong consistency
    ``M1 N = R`` therefore survives the limit intact.

    Leading dimensions broadcast, so a whole group of cells is done at once.
    """
    w, V = np.linalg.eigh(np.asarray(Kbar, dtype=float))
    tol = rtol * np.max(w, axis=-1, keepdims=True)
    inv = np.where(w > tol, 1.0 / np.where(w > tol, w, 1.0), 0.0)
    return V @ (inv[..., None] * (np.swapaxes(V, -1, -2) @ np.asarray(B, dtype=float)))


def consistency_matrix(R: np.ndarray, Kbar: np.ndarray, volume: float) -> np.ndarray:
    """Consistency term ``M1 = (1/|E|) R Kbar^+ R^T``; satisfies ``M1 N = R``."""
    R = np.asarray(R, dtype=float)
    return (R @ apply_pseudo_inverse(Kbar, R.T)) / volume


def range_projector(A: np.ndarray, rtol: float = 1e-12) -> tuple[np.ndarray, int]:
    """Orthonormal basis of ``range(A)`` and its rank, via a *thin* SVD.

    The stabilization only ever needs ``I - Q Q^T`` (the projector onto
    ``ker(A^T)``), never an explicit basis of the null space -- and a thin SVD
    of a ``D x m`` matrix is far cheaper than the full one, which would build a
    ``D x D`` factor just to discard most of it.
    """
    A = np.asarray(A, dtype=float)
    # QR is markedly cheaper than SVD and suffices when A has full column rank,
    # which is the usual case; |diag(R)| reveals when it does not.
    if A.shape[0] >= A.shape[1] and A.size:
        Q, Rm = np.linalg.qr(A)
        diag = np.abs(np.diag(Rm))
        if diag.size and diag.min() > rtol * diag.max() * max(A.shape):
            return Q, A.shape[1]
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = rtol * (s[0] if s.size else 1.0) * max(A.shape)
    rank = int(np.sum(s > tol))
    return U[:, :rank], rank


def assemble_local_inner_product(
    N: np.ndarray,
    R: np.ndarray,
    Kbar: np.ndarray,
    volume: float,
    stability_scale: float | None = None,
) -> np.ndarray:
    """Local inner-product matrix ``M_E = M1 + M2`` (see module docstring).

    Parameters
    ----------
    N, R
        ``(D, m)`` consistency and moment matrices with ``N^T R = |E| Kbar``.
    Kbar
        ``(m, m)`` SPD Gram matrix of the reconstruction modes.
    volume
        Element measure ``|E|``.
    stability_scale
        Positive scalar sizing the stabilization.  Defaults to the mean diagonal
        of ``M1``, which matches its spectral scaling.

    Returns
    -------
    ``(D, D)`` symmetric positive-definite matrix satisfying ``M_E N = R``.
    When ``D == m`` the stabilization is identically zero.
    """
    M1 = consistency_matrix(R, Kbar, volume)
    Q, rank = range_projector(N)
    if rank == N.shape[0]:  # ker(N^T) = {0}: the simplex case, no stabilization
        M = M1
    else:
        s = float(np.mean(np.diag(M1))) if stability_scale is None else stability_scale
        # M2 = s (I - Q Q^T), the projector onto ker(N^T)
        M = M1 - s * (Q @ Q.T)
        M[np.diag_indices_from(M)] += s
    return 0.5 * (M + M.T)  # symmetrize against round-off


def consistency_residual(M: np.ndarray, N: np.ndarray, R: np.ndarray) -> float:
    """``max |M N - R|`` -- the strong-consistency diagnostic."""
    return float(np.abs(np.asarray(M) @ np.asarray(N) - np.asarray(R)).max())
