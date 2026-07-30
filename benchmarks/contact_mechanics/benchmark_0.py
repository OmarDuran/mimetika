r"""Benchmark 0 -- in-situ state and depletion response, no fault (paper 2.4).

Two things are checked, and they are checked differently on purpose:

1. **The in-situ state** is analytic.  It is *derived* from the Table 2
   parameters and compared with the coefficients the paper prints, so an error
   in the setup shows up before any solver runs.

2. **The depletion response** is simulated.  A uniformly depleted, laterally
   confined domain compacts uniaxially, and the mixed poromechanics solver must
   reproduce the closed forms

       ``eps_yy = alpha Delta p / Kv`` ,
       ``Delta sigma'_xx = nu/(1-nu) alpha Delta p`` ,   ``sigma_yy = 0`` .

   Lateral confinement is imposed with **rollers** -- prescribed normal
   displacement and free slip -- which in Hellinger--Reissner means pinning the
   *shear traction* DOFs, since the displacement side is natural there.

Run with ``python -m benchmarks.contact_mechanics.benchmark_0``.
"""

from __future__ import annotations

import argparse

import numpy as np

from mimetika.assembly.mixed import boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.materials import Material
from mimetika.mesh import structured_quads

from benchmarks.contact_mechanics.common import Parameters, linear_fit


def in_situ_report(parameters: Parameters, samples: int = 11):
    """Fitted ``(intercept, gradient)`` of every in-situ field the paper lists."""
    y = np.linspace(-parameters.height / 2, parameters.height / 2, samples)
    normal, shear = parameters.resolved(y, dip=70.0)
    return {
        "sigma_yy": linear_fit(y, parameters.vertical_stress(y)),
        "sigma_xx": linear_fit(y, parameters.horizontal_stress(y)),
        "pressure": linear_fit(y, parameters.pressure(y)),
        "sigma_normal_70": linear_fit(y, normal),
        "sigma_shear_70": linear_fit(y, shear),
    }


def depletion_response(parameters: Parameters, n: int = 8):
    """Simulate uniform depletion of a confined block; return the key responses.

    The mesh is the unit square: the response is a *strain*, so the domain size
    only enters through ``Delta h = h eps_yy``, applied afterwards.
    """
    mesh = structured_quads(n, n)
    problem = PoroMechanics(
        mesh,
        Material(
            shear_modulus=parameters.shear_modulus,
            poisson=parameters.poisson,
            biot=parameters.biot,
        ),
    )
    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][1] - 1.0) < 1e-12]
    rollers = [f for f in boundary if f not in set(top)]

    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    solution = problem.solve(
        dt=None,
        dirichlet=zero,
        pressure=parameters.depletion,
        traction=lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3)),
        traction_facets=top,  # free surface
        roller_facets=rollers,  # laterally confined, base supported
    )
    stress = problem.mechanics.cell_stress(solution["stress"])
    strain = float(problem.volumetric_strain(solution).mean())
    return {
        "vertical_strain": strain,
        "compaction": parameters.reservoir_height * strain,
        "sigma_xx_total": float(stress[:, 0, 0].mean()),
        "sigma_xx_effective": float(
            stress[:, 0, 0].mean() + parameters.biot * parameters.depletion
        ),
        "sigma_yy_total": float(np.abs(stress[:, 1, 1]).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=8, help="cells per side")
    parser.add_argument("--dip", type=float, default=70.0)
    arguments = parser.parse_args()

    parameters = Parameters(dip=arguments.dip)
    print("Benchmark 0 -- in-situ state and depletion response\n")
    print(f"  bulk density      {parameters.bulk_density:10.1f} kg/m^3")
    print(f"  uniaxial modulus  {parameters.uniaxial_modulus:10.4e} Pa"
          "     (paper 15.79e9)\n")

    published = {
        "sigma_yy": (-82.60e6, 23.60e3),
        "sigma_xx": (-57.05e6, 16.30e3),
        "pressure": (35.00e6, -10.06e3),
        "sigma_normal_70": (-60.04e6, 17.15e3),
        "sigma_shear_70": (8.21e6, -2.35e3),
    }
    print("  in-situ profiles           intercept [MPa]   gradient [kPa/m]")
    for name, (intercept, gradient) in in_situ_report(parameters).items():
        want = published[name]
        print(
            f"    {name:<18} {intercept / 1e6:+9.2f} ({want[0] / 1e6:+7.2f})"
            f"   {gradient / 1e3:+8.2f} ({want[1] / 1e3:+7.2f})"
        )
    print("    (the shear sign is the tangent convention; the magnitude agrees)\n")

    response = depletion_response(parameters, n=arguments.n)
    print("  depletion response                 simulated        closed form")
    rows = [
        ("vertical strain", response["vertical_strain"], parameters.vertical_strain),
        ("compaction [m]", response["compaction"], parameters.compaction),
        (
            "d sigma'_xx [Pa]",
            response["sigma_xx_effective"],
            parameters.horizontal_effective_increment,
        ),
        (
            "d sigma_xx [Pa]",
            response["sigma_xx_total"],
            parameters.horizontal_total_increment,
        ),
        ("free-surface sigma_yy", response["sigma_yy_total"], 0.0),
    ]
    for name, got, want in rows:
        print(f"    {name:<22} {got:+16.6e}  {want:+16.6e}")


if __name__ == "__main__":
    main()
