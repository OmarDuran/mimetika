"""Hodge / mass operators (where the metric enters).

The exterior derivative is metric-free; *all* geometric information in a mimetic
method enters through the Hodge star (equivalently, the mass / inner-product
matrix) on k-forms.  This module defines the interface and a lowest-fidelity,
positive-definite diagonal implementation sufficient to drive the assembly and
solver layers end to end.

Extension points (deliberately left as follow-ups, each a self-contained upgrade
that does not change any caller):

* ``CircumcentricHodge`` -- the geometrically-consistent diagonal DEC star
  ``*_k = diag(|dual_k| / |primal_k|)`` using the circumcentric dual mesh.
* ``PolytopalHodge`` -- a dense-per-cell consistency+stability inner product
  (``M = M_consistency + M_stability``) assembled cell by cell, the genuinely
  "mimetic on polytopes" inner product.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from mimetika.geometry.metric import Geometry


class HodgeOperator(ABC):
    """Abstract inner-product / Hodge-star provider on k-forms."""

    @abstractmethod
    def matrix(self, k: int) -> sp.spmatrix:
        """Return the (symmetric positive-definite) mass matrix on k-forms."""


class DiagonalHodge(HodgeOperator):
    """A diagonal mass matrix ``M_k = diag(measure_k)``.

    This is the mass-lumped inner product: SPD, cheap, and exact enough to
    exercise the full pipeline.  It is *not* the geometrically-consistent DEC
    star (see module docstring) -- swap in ``CircumcentricHodge`` for that.
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    def matrix(self, k: int) -> sp.dia_matrix:
        m = self.geometry.measure(k)
        return sp.diags(np.where(m == 0, 1.0, m), format="dia")

    def inverse(self, k: int) -> sp.dia_matrix:
        m = self.geometry.measure(k)
        return sp.diags(1.0 / np.where(m == 0, 1.0, m), format="dia")
