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
from dataclasses import replace

import numpy as np

from mimetika.assembly.mixed import boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.materials import Material
from mimetika.mesh import structured_quads
from mimetika.postprocess import MixedDimensionalSeries

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


# -- Fig. 4: combined stresses across a finite reservoir --------------------------------


def finite_reservoir(parameters: Parameters, nx: int = 20, ny: int = 120):
    """Deplete a reservoir of finite thickness inside the full domain.

    Different from :func:`depletion_response`, and harder.  That one depletes the
    **whole** domain, which is why it reproduces the uniaxial closed form to
    round-off -- there is nothing for the rock to arch over.  Fig. 4 needs a
    ``h = 225`` m reservoir inside a ``4500`` m domain, so the surrounding rock
    carries part of the load and the stress steps sharply at the reservoir top
    and bottom.  The uniaxial formulae survive only in the interior.
    """
    width, height = parameters.width, parameters.height
    mesh = structured_quads(
        nx, ny, lengths=(width, height), origin=(-width / 2, -height / 2)
    )
    material = Material(
        shear_modulus=parameters.shear_modulus,
        poisson=parameters.poisson,
        biot=parameters.biot,
    )
    problem = PoroMechanics(mesh, material)

    half = 0.5 * parameters.reservoir_height
    spacing = height / ny
    if abs(half / spacing - round(half / spacing)) > 1e-9:
        raise ValueError(
            f"ny={ny} puts the reservoir boundary at y={half} inside a cell "
            f"(dy={spacing}); the depletion is assigned per cell, so a boundary "
            "that bisects a cell shifts the step by half a cell.  Choose ny a "
            f"multiple of {round(height / (2 * half) * 2)} -- e.g. 40, 80, 120, 200."
        )
    centroids = mesh.geometry.centroids(2)
    pressure = np.where(np.abs(centroids[:, 1]) < half, parameters.depletion, 0.0)

    facets = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(facets[f][1] - height / 2) < 1e-9]
    solution = problem.solve(
        dt=None,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        pressure=pressure,
        traction=lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3)),
        traction_facets=top,
        roller_facets=[f for f in boundary if f not in set(top)],
    )
    return mesh, problem.mechanics.cell_stress(solution["stress"]), pressure


def combined_stress_profile(
    parameters: Parameters, nx: int = 20, ny: int = 120,
    dip: float = 70.0, extent: float = 250.0,
):
    """Fig. 4: combined stresses resolved on a ``dip``-degree plane.

    Sampled along a line at ``dip`` to the horizontal through the reservoir
    centre -- the line the fault would occupy -- and resolved onto that plane.
    "Combined" means in-situ **plus** the depletion increment, which is what the
    figure plots.
    """
    mesh, stress, _ = finite_reservoir(parameters, nx=nx, ny=ny)
    normal, tangent = parameters.fault_basis(dip)
    centroids = mesh.geometry.centroids(2)

    theta = np.radians(dip)
    y = np.linspace(-extent, extent, 2 * int(extent) + 1)
    x = y * np.cos(theta) / np.sin(theta)  # the line at `dip` through the origin

    perpendicular, parallel = [], []
    for xi, yi in zip(x, y):
        # nearest cell centre; ties at a reservoir boundary are why `ny` must put
        # that boundary on a cell face (checked in `finite_reservoir`)
        cell = int(np.argmin((centroids[:, 0] - xi) ** 2 + (centroids[:, 1] - yi) ** 2))
        total = stress[cell] + parameters.stress_tensor(yi)[0]  # increment + in-situ
        perpendicular.append(normal @ total @ normal)
        parallel.append(tangent @ total @ normal)
    return {
        "y": y,
        "normal": np.array(perpendicular),
        "shear": np.array(parallel),
    }


def combined_analytic(y, parameters: Parameters, dip: float = 70.0):
    """The analytic combined profile of Fig. 4, as a function of ``y``.

    The reference curve the paper plots alongside the simulators.  It is exactly
    piecewise because the reservoir is infinitely wide: no arching, so outside
    the depleted band the increment is identically zero and the combined stress
    is the in-situ state, while inside it is the in-situ state plus the uniaxial
    increment ``Delta sigma_xx`` (with ``Delta sigma_yy = 0``).  The two branches
    keep the in-situ depth gradient, so neither is flat.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    normal, tangent = parameters.fault_basis(dip)
    inside = np.abs(y) < 0.5 * parameters.reservoir_height

    stress = parameters.stress_tensor(y)  # (n, 2, 2) in-situ
    stress[inside, 0, 0] += parameters.horizontal_total_increment
    return (
        np.einsum("i,qij,j->q", normal, stress, normal),
        np.einsum("i,qij,j->q", tangent, stress, normal),
    )


def combined_closed_form(parameters: Parameters, dip: float = 70.0):
    """``(outside, inside)`` resolved stresses expected far from the reservoir edges.

    Outside is simply the in-situ state.  Inside, the uniaxial increment adds
    ``Delta sigma_xx`` with ``Delta sigma_yy = 0``, resolved on the same plane.
    """
    normal, tangent = parameters.fault_basis(dip)
    insitu = parameters.stress_tensor(0.0)[0]
    increment = np.diag([parameters.horizontal_total_increment, 0.0])
    combined = insitu + increment
    return {
        "outside": (normal @ insitu @ normal, tangent @ insitu @ normal),
        "inside": (normal @ combined @ normal, tangent @ combined @ normal),
    }


def depletion_series(
    path,
    parameters: Parameters,
    steps: int = 5,
    nx: int = 8,
    ny: int = 120,
    dip: float = 70.0,
):
    """The finite-reservoir stress state over a depletion ramp, as a ``.pvd``.

    **Bulk only** -- benchmark 0 has no fault, so there is no lower-dimensional
    part to write.  The series exists so the stress state can be watched building
    up: the combined stresses resolved on the ``dip``-degree plane are exactly the
    quantities the paper plots in Fig. 4, and the step at the reservoir edges is
    the feature to look for.
    """
    normal, tangent = parameters.fault_basis(dip)
    series = None
    for level in np.linspace(0.0, parameters.depletion, steps + 1)[1:]:
        stage = replace(parameters, depletion=float(level))
        mesh, increment, pressure = finite_reservoir(stage, nx=nx, ny=ny)
        if series is None:
            series = MixedDimensionalSeries(path, mesh)

        y = mesh.geometry.centroids(2)[:, 1]
        combined = parameters.stress_tensor(y) + increment  # in-situ + increment
        series.write(
            abs(level) / 1e6,   # the time coordinate is |Delta p| in MPa
            bulk={
                "pressure": pressure,
                "sigma_xx": combined[:, 0, 0],
                "sigma_yy": combined[:, 1, 1],
                "sigma_xy": combined[:, 0, 1],
                "sigma_normal": np.einsum("i,qij,j->q", normal, combined, normal),
                "sigma_shear": np.einsum("i,qij,j->q", tangent, combined, normal),
            },
        )
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=8, help="cells per side")
    parser.add_argument("--dip", type=float, default=70.0)
    parser.add_argument("--vtu", nargs="?", const="out/benchmark_0",
                        help="write a PVD depletion series to this path stem")
    parser.add_argument("--steps", type=int, default=5)
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

    if arguments.vtu:
        series = depletion_series(arguments.vtu, parameters,
                                  steps=arguments.steps, dip=arguments.dip)
        print(f"\n  wrote {series.collection} "
              f"({arguments.steps} depletion steps, bulk part only -- no fault)")


if __name__ == "__main__":
    main()
