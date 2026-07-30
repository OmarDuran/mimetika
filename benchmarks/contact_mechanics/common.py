r"""Shared setup for the fault-reactivation contact-mechanics benchmarks.

Novikov, Voskov et al. (2024), *Benchmark study of fault reactivation induced
by pressure depletion*.  A depleting reservoir at depth loads a fault; the
question is when, where and how much it slips.

Geometry and sign conventions
-----------------------------
``y`` is measured **upwards** from the reservoir reference level, so depth is
``D0 - y``.  Stresses are **tension positive** throughout (the paper's figures
are too), which makes every in-situ stress negative.  Effective stress uses the
Biot convention ``sigma' = sigma + alpha p``.

The in-situ state is *derived*, not tabulated
---------------------------------------------
Everything in the paper's eq. (17)-(19) follows from the Table 2 parameters:

    ``rho_b = phi rho_fl + (1 - phi) rho_s``          bulk density
    ``sigma_yy(y) = -rho_b g (D0 - y)``               lithostatic
    ``p(y)        = p0 - rho_fl g y``                 hydrostatic
    ``sigma'_xx   = K0 sigma'_yy``                    lateral earth pressure
    ``sigma_xx    = sigma'_xx - alpha p``

Reproducing the paper's coefficients from the parameters -- rather than pasting
them in -- is what makes the setup checkable; :mod:`tests` does exactly that.

One deviation: the tabulated fluid density gives a pressure gradient of
``1020 * 9.81 = 10.01`` kPa/m, while the paper quotes ``10.06`` kPa/m.  The
latter corresponds to ``rho_fl = 1025``, which is the value used here so that
the published in-situ profiles are reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """Table 2 of the benchmark paper, in SI units."""

    width: float = 4500.0  # W
    height: float = 4500.0  # H
    depth: float = 3500.0  # D0, to the reservoir reference level
    reservoir_height: float = 224.5  # h; see `compaction` below

    shear_modulus: float = 6500e6  # G
    poisson: float = 0.15  # nu
    biot: float = 0.9  # alpha
    earth_pressure: float = 0.5  # K0, effective horizontal / vertical

    depletion: float = -25e6  # Delta p
    reference_pressure: float = 35e6  # p0 at y = 0
    fluid_density: float = 1025.0  # rho_fl (see module docstring)
    solid_density: float = 2650.0  # rho_s
    porosity: float = 0.15  # phi
    gravity: float = 9.81  # g

    dip: float = 90.0  # theta, degrees from horizontal
    friction: float = 0.52  # mu
    fault_permeability: float = 0.0  # kappa

    # -- derived material constants ------------------------------------------

    @property
    def bulk_density(self) -> float:
        return self.porosity * self.fluid_density + (
            1.0 - self.porosity
        ) * self.solid_density

    @property
    def uniaxial_modulus(self) -> float:
        """``Kv = 2G(1-nu)/(1-2nu)`` -- the oedometer (confined) modulus."""
        return (
            2.0
            * self.shear_modulus
            * (1.0 - self.poisson)
            / (1.0 - 2.0 * self.poisson)
        )

    # -- the in-situ state ----------------------------------------------------

    @property
    def vertical_gradient(self) -> float:
        """``d sigma_yy / dy`` -- positive: the rock gets lighter upwards."""
        return self.bulk_density * self.gravity

    @property
    def pressure_gradient(self) -> float:
        """``dp/dy`` -- negative: hydrostatic pressure falls upwards."""
        return -self.fluid_density * self.gravity

    def pressure(self, y) -> np.ndarray:
        """Initial pore pressure ``p0(y)``."""
        return self.reference_pressure + self.pressure_gradient * np.asarray(
            y, dtype=float
        )

    def vertical_stress(self, y) -> np.ndarray:
        """Total vertical stress, tension positive."""
        return -self.bulk_density * self.gravity * (
            self.depth - np.asarray(y, dtype=float)
        )

    def horizontal_stress(self, y) -> np.ndarray:
        """Total horizontal stress from the lateral earth-pressure ratio."""
        effective = self.earth_pressure * (
            self.vertical_stress(y) + self.biot * self.pressure(y)
        )
        return effective - self.biot * self.pressure(y)

    def stress_tensor(self, y) -> np.ndarray:
        """``(n, 2, 2)`` in-situ total stress; the state is diagonal in ``(x, y)``."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.zeros((len(y), 2, 2))
        out[:, 0, 0] = self.horizontal_stress(y)
        out[:, 1, 1] = self.vertical_stress(y)
        return out

    # -- resolved on the fault -------------------------------------------------

    def fault_basis(self, dip: float | None = None):
        """``(normal, tangent)`` of a fault dipping ``dip`` degrees from horizontal."""
        theta = np.radians(self.dip if dip is None else dip)
        tangent = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])
        return normal, tangent

    def resolved(self, y, dip: float | None = None):
        """``(sigma_normal, sigma_shear)`` of the in-situ state on the fault.

        ``sigma_normal < 0`` in compression; the sign of the shear follows the
        chosen tangent, so only its magnitude is convention free.
        """
        normal, tangent = self.fault_basis(dip)
        stress = self.stress_tensor(y)
        return (
            np.einsum("i,qij,j->q", normal, stress, normal),
            np.einsum("i,qij,j->q", tangent, stress, normal),
        )

    # -- the depletion response (paper section 2.4) ----------------------------

    @property
    def vertical_strain(self) -> float:
        """``eps_yy = alpha Delta p / Kv`` under uniaxial (confined) strain."""
        return self.biot * self.depletion / self.uniaxial_modulus

    @property
    def compaction(self) -> float:
        """``Delta h = h eps_yy``.

        The paper reports ``-0.32`` m, which fixes the reservoir thickness at
        ``h = 224.5`` m -- the value :attr:`reservoir_height` defaults to, since
        Table 2 lists the domain size but not the reservoir's own thickness.
        """
        return self.reservoir_height * self.vertical_strain

    @property
    def horizontal_effective_increment(self) -> float:
        """``Delta sigma'_xx = nu/(1-nu) alpha Delta p``."""
        return (
            self.poisson
            / (1.0 - self.poisson)
            * self.biot
            * self.depletion
        )

    @property
    def horizontal_total_increment(self) -> float:
        """``Delta sigma_xx = Delta sigma'_xx - alpha Delta p``."""
        return self.horizontal_effective_increment - self.biot * self.depletion


def linear_fit(y, values):
    """``(intercept, gradient)`` of a field that the benchmark states as linear."""
    gradient, intercept = np.polyfit(np.asarray(y, dtype=float), values, 1)
    return intercept, gradient
