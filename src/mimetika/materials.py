r"""Material properties, possibly varying from cell to cell.

Parametrised by the Poisson ratio rather than by Lame's ``lambda``: ``lambda ->
infinity`` as ``nu -> 1/2``, so a ``lambda``-based interface cannot represent an
incompressible solid.

The compliance used by the Hellinger--Reissner form is

    ``C^{-1} T = ( T - a tr(T) I ) / (2 mu)`` ,   ``a = nu / (1 - 2 nu + d nu)``

with ``a = 1/d`` at ``nu = 1/2``, the deviatoric projector.  The mixed
formulation inverts ``C^{-1}``, bounded in the incompressible limit, hence
locking free; a displacement formulation would invert ``C``, which is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compliance_coefficient(dim: int, poisson) -> np.ndarray:
    """``a = nu / (1 - 2 nu + d nu)``; finite at ``nu = 1/2`` (giving ``1/d``)."""
    nu = np.asarray(poisson, dtype=float)
    if np.any(nu > 0.5 + 1e-12) or np.any(nu <= -1.0):
        raise ValueError("Poisson ratio must lie in (-1, 1/2]")
    return nu / (1.0 - 2.0 * nu + dim * nu)


def poisson_from_lame(mu, lam, dim: int = 3) -> np.ndarray:
    """Recover ``nu`` from Lame parameters (``lam = inf`` -> ``nu = 1/2``)."""
    mu, lam = np.asarray(mu, dtype=float), np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(np.isinf(lam), 0.5, lam / (2.0 * (lam + mu)))
    return nu


@dataclass
class Material:
    """Poroelastic properties; scalars are broadcast to every cell.

    ``bulk_modulus`` is the drained bulk modulus ``K``, ``inf`` at ``nu = 1/2``.
    The couplings ``alpha/(dK)`` and ``alpha^2/K`` then vanish, so the
    incompressible skeleton needs no special case downstream.
    """

    shear_modulus: np.ndarray = 1.0
    poisson: np.ndarray = 0.25
    biot: np.ndarray = 1.0
    #: inverse Biot modulus ``1/M``; zero for an incompressible fluid
    inverse_biot_modulus: np.ndarray = 0.0
    permeability: np.ndarray = 1.0
    viscosity: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus",
            "poisson",
            "biot",
            "inverse_biot_modulus",
            "permeability",
        ):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))

    def expand(self, n_cells: int) -> "Material":
        """Broadcast every scalar field to one value per cell."""
        out = {}
        for name in (
            "shear_modulus",
            "poisson",
            "biot",
            "inverse_biot_modulus",
            "permeability",
        ):
            v = np.asarray(getattr(self, name), dtype=float)
            out[name] = np.broadcast_to(v, (n_cells,)).copy() if v.ndim == 0 else v
        return Material(viscosity=self.viscosity, **out)

    # -- derived ---------------------------------------------------------------

    def compliance_coefficient(self, dim: int) -> np.ndarray:
        return compliance_coefficient(dim, self.poisson)

    @property
    def bulk_modulus(self) -> np.ndarray:
        """Drained bulk modulus ``K = 2 mu (1 + nu) / (3 (1 - 2 nu))``; ``inf`` at 1/2."""
        nu = self.poisson
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                np.isclose(nu, 0.5),
                np.inf,
                2.0 * self.shear_modulus * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu)),
            )

    def inverse_modulus(self, dim: int) -> np.ndarray:
        """``tr(C^{-1} T) / tr(T) = (1-2nu) / (2 mu (1-2nu+d nu))``.

        Volumetric compliance of the skeleton; both poroelastic couplings are
        built from it.  Zero at ``nu = 1/2``.  Equals ``1/(3K)`` in 3D; the
        ``d``-dependence matters because the benchmarks are plane strain.
        """
        nu = self.poisson
        return (1.0 - 2.0 * nu) / (
            2.0 * self.shear_modulus * (1.0 - 2.0 * nu + dim * nu)
        )

    def pressure_coupling(self, dim: int) -> np.ndarray:
        """``alpha / (d K)``, as it multiplies the trace operator in the stress row."""
        return self.biot * self.inverse_modulus(dim)

    def storage(self, dim: int) -> np.ndarray:
        """``S = d alpha^2 / (d K) + 1/M``; zero for incompressible fluid and solid."""
        return dim * self.biot**2 * self.inverse_modulus(dim) + self.inverse_biot_modulus

    @property
    def mobility(self) -> np.ndarray:
        """``K / mu_f``, the coefficient entering the Darcy inner product."""
        return self.permeability / self.viscosity
