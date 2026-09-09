"""Hodge / mass operators (where the metric enters).

The exterior derivative is metric-free; *all* geometric information in a mimetic
method enters through the Hodge star (equivalently, the mass / inner-product
matrix) on k-forms.  This module defines the interface and one diagonal, SPD
implementation of it.

Extension points, not yet implemented (each a self-contained upgrade that does
not change any caller):

* ``CircumcentricHodge`` -- the geometrically-consistent diagonal DEC star
  ``*_k = diag(|dual_k| / |primal_k|)`` using the circumcentric dual mesh.
* ``PolytopalHodge`` -- a dense-per-cell consistency+stability inner product
  (``M = M1 + M2``) assembled cell by cell.
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
    """A diagonal mass matrix ``M_k = diag(measure_k)``, zero measures set to 1.

    The mass-lumped inner product: SPD and cheap.  Not the geometrically
    consistent DEC star, which is an extension point (see module docstring).
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    def matrix(self, k: int) -> sp.dia_matrix:
        m = self.geometry.measure(k)
        return sp.diags(np.where(m == 0, 1.0, m), format="dia")

    def inverse(self, k: int) -> sp.dia_matrix:
        m = self.geometry.measure(k)
        return sp.diags(1.0 / np.where(m == 0, 1.0, m), format="dia")
