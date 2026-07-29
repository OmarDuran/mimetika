r"""Mimetic inner products (the metric / material heart of the method).

This module implements the *consistency + stability* construction of local
inner-product (mass) matrices common to the mimetic finite difference family
(Brezzi--Lipnikov--Simoncini) and its elasticity extension
(Beir\~ao da Veiga, ESAIM M2AN 44 (2010) 231--250, section 4).

For one element ``E`` with ``D`` local degrees of freedom and a *reconstruction
space* of ``m`` polynomial modes (``m <= D``), one builds

    ``N`` (D x m)  : the DOFs of each reconstruction mode,
    ``Kbar`` (m x m, SPD) : the Gram matrix of the modes in the continuous
                            inner product (``Kbar_{jl} = (1/|E|) <w_j, w_l>``).

The local inner product is ``M_E = M1 + M2`` with

    consistency   ``M1 = N (N^T N)^{-1} (|E| Kbar) (N^T N)^{-1} N^T`` ,
    stability     ``M2 = s * C C^T`` ,  C = orthonormal basis of ker(N^T).

The two properties that matter (both checked in the tests):

* **Consistency / exactness.**  For any DOF vector produced by the
  reconstruction, ``g = N c``, one has ``g^T M1 g = |E| c^T Kbar c`` -- i.e. M1
  reproduces the *exact* continuous inner product on the reconstruction space,
  and ``M2 g = 0`` there (``C^T N = 0``).

* **Stabilization vanishes on simplices.**  ``M2 = 0`` iff ``ker(N^T) = {0}``
  iff ``D = m`` with ``N`` full rank.  Choosing the reconstruction space equal
  to the target mixed-FE space -- which is unisolvent on a simplex -- yields
  ``D = m`` on tetrahedra, so the stabilization is active only on genuine
  polytopes (hexahedra, general polyhedra).

The projection form of ``M1`` above is used in place of the algebraically
equivalent ``(1/|E|) R Kbar^{-1} R^T`` (with ``R`` the moment matrix and
``N^T R = |E| Kbar``) because it needs only ``N`` and ``Kbar`` -- both obtained
by quadrature -- and so extends cleanly to enriched reconstruction spaces.
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
    rank = int(np.sum(s > tol))
    return U[:, rank:]


def stabilization_dim(N: np.ndarray) -> int:
    """Dimension of the local stabilization space ``dim ker(N^T)``.

    Exactly zero when the reconstruction is unisolvent for the DOFs (simplex
    with a matched reconstruction space); positive on genuine polytopes.  A
    cheap, exact diagnostic independent of the stability scaling.
    """
    return int(nullspace_basis(N).shape[1])


def consistency_matrix(N: np.ndarray, Kbar: np.ndarray, volume: float) -> np.ndarray:
    """Consistency term ``M1`` (projection form). Exact on ``range(N)``."""
    N = np.asarray(N, dtype=float)
    Kbar = np.asarray(Kbar, dtype=float)
    G = N.T @ N
    P = np.linalg.solve(G, N.T)  # (N^T N)^{-1} N^T ,  shape (m, D)
    return volume * (P.T @ Kbar @ P)


def assemble_local_inner_product(
    N: np.ndarray,
    Kbar: np.ndarray,
    volume: float,
    stability_scale: float,
) -> np.ndarray:
    """Local inner-product matrix ``M_E = M1 + M2`` (see module docstring).

    Parameters
    ----------
    N
        ``(D, m)`` matrix of the DOFs of the reconstruction modes.
    Kbar
        ``(m, m)`` SPD Gram matrix of the modes in the continuous inner product.
    volume
        Element measure ``|E|``.
    stability_scale
        Positive scalar sizing the stabilization; chosen to match the spectral
        scaling of ``M1`` (e.g. ``mean(diag(M1))``).

    Returns
    -------
    ``(D, D)`` symmetric positive-definite matrix.  When ``D == m`` the
    stabilization is identically zero (purely consistent inner product).
    """
    M1 = consistency_matrix(N, Kbar, volume)

    C = nullspace_basis(N)
    M2 = np.zeros_like(M1) if C.shape[1] == 0 else stability_scale * (C @ C.T)

    M = M1 + M2
    return 0.5 * (M + M.T)  # symmetrize against round-off
