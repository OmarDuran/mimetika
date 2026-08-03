r"""Exact static condensation of a diagonal leading block.

The lumped stress inner product exists so that the stress can be eliminated
from the saddle-point system *exactly*: with ``M`` diagonal, the partition

    [ M    C1 ] [ sigma ]   [ g ]
    [ C2   E  ] [   y   ] = [ b ]

gives ``sigma = M^{-1}(g - C1 y)`` facet by facet, and the remaining unknowns
satisfy the **reduced** system

    (E - C2 M^{-1} C1) y = b - C2 M^{-1} g .

Because ``M^{-1}`` is a diagonal scaling, forming the reduced matrix is a
sparse triple product with no fill beyond neighbour coupling: every block of
``C1``/``C2`` connects a facet to its two adjacent cells, so the reduced
system is **cell-centred** with a two-point stencil -- ``(1 + d + d(d-1)/2)``
unknowns per cell for the four-field formulations.  The condensation is exact
algebra, not an approximation: the recovered ``sigma`` satisfies the original
system to round-off.

This is the payoff of the lumped inner product.  The AFW operator couples the
facets of each cell, so its stress block cannot be condensed without a global
factorisation -- which is the whole solve.  The three-field lumped assembly is
equally stuck: folding the volumetric term back into ``M`` re-couples the
facets.  Only the four-field arrangements keep ``M`` diagonal in the assembled
system, which is why the efficient path is four-field + lumped.

Works for symmetric (multiplier) and quasi-symmetric (kinematic rotation)
systems alike -- ``C1`` and ``C2`` are handled independently.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def eliminate_leading_diagonal(S: sp.spmatrix, rhs: np.ndarray, n0: int):
    """Reduce out the leading ``n0`` unknowns; their block must be diagonal.

    Returns ``(reduced, reduced_rhs, recover)`` where ``recover(y)`` maps a
    solution of the reduced system back to the full vector.  Raises if the
    leading block is not diagonal -- condensation is exact or it is not done.
    """
    S = S.tocsr()
    M = S[:n0, :n0]
    diagonal = np.asarray(M.diagonal()).ravel()
    if (M - sp.diags(diagonal)).nnz != 0:
        raise ValueError(
            "the leading block is not diagonal; exact condensation applies to "
            "the lumped four-field arrangements only (AFW couples the facets "
            "of each cell, and the three-field lumped assembly folds the "
            "volumetric term back into M)"
        )
    if np.any(diagonal == 0.0):
        raise ValueError("the leading diagonal has zero entries")
    inv = 1.0 / diagonal
    C1 = S[:n0, n0:]
    C2 = S[n0:, :n0]
    E = S[n0:, n0:]
    reduced = (E - C2 @ sp.diags(inv) @ C1).tocsr()
    g, b = rhs[:n0], rhs[n0:]
    reduced_rhs = b - C2 @ (inv * g)

    def recover(y: np.ndarray) -> np.ndarray:
        sigma = inv * (g - C1 @ y)
        return np.concatenate([sigma, y])

    return reduced, reduced_rhs, recover


def solve_condensed(S: sp.spmatrix, rhs: np.ndarray, n0: int) -> np.ndarray:
    """Solve by exact stress condensation: reduce, factorise, back-substitute."""
    reduced, reduced_rhs, recover = eliminate_leading_diagonal(S, rhs, n0)
    y = spla.splu(reduced.tocsc()).solve(reduced_rhs)
    return recover(y)
