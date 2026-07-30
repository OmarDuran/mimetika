r"""Benchmark 1 -- vertical displaced fault, frictionless (paper section 3).

A reservoir offset across a vertical fault is depleted by ``-25`` MPa.  The
throw puts reservoir against seal on both sides of the fault, which loads it in
shear; with no friction the fault slips until it carries no shear stress at all.

Geometry (Fig. 5).  The reservoir spans ``[-b, a]`` on one side of the fault and
``[-a, b]`` on the other, with ``a = 75`` m, ``b = 150`` m -- a thickness of
``a + b = 225`` m and a throw of ``b - a = 75`` m.

Analytic solution (Jansen & Meulenbroek 2022, quoted as eqs 18-22).  The
pre-slip Coulomb stress and the resulting slip are

    ``Sigma_C(y) = (C/2) ln[ (y-a)^2 (y+a)^2 / ( (y-b)^2 (y+b)^2 ) ]``
    ``delta(y)   = (C/A) x { 0, -(y+b), a-b, (y-b), 0 }``  on the five intervals

with ``C = (1-2nu) alpha p / (2 pi (1-nu))`` and ``A = G / (2 pi (1-nu))``.  Both
are derived for an unbounded medium; the simulation uses the paper's finite
``W = H = 4500`` m domain, which is why the comparison below is on the profile
shape and peak rather than pointwise.

The contact law is :class:`SignoriniCoulomb` with ``friction = 0``: the physical
model, unilateral and frictionless.  A benchmark exists to *test* laws, so the
one that represents the situation is the one that should be run.

Making it work requires giving the law the **total** traction.  Signorini
constrains ``t_N <= 0`` on the total stress, and this is an *incremental* problem
-- only the depletion response is solved for.  The incremental normal traction
reaches ``+8.4`` MPa in tension, but the fault sits on ``-57`` MPa of in-situ
compression and is shut by a wide margin, so the law must be told what it is
sitting on: that is what ``prestress`` carries.  With it, Signorini correctly
finds the fault closed and agrees with :class:`FrictionlessBilateral` to
round-off; without it, it reads the tensile increment as opening.  The deficiency
was in the incremental formulation, not in the law.

Run with ``python -m benchmarks.contact_mechanics.benchmark_1``.
"""

from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.mixed import MixedElasticity, boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.contact import ContactDriver, FrictionlessBilateral, SignoriniCoulomb
from mimetika.materials import Material
from mimetika.mesh import structured_quads
from mimetika.mesh.fracture import facets_on_plane
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.solver.saddle import solve_saddle

from benchmarks.contact_mechanics.common import Parameters


# -- the analytic solution ----------------------------------------------------------


def analytic_coulomb_stress(y, parameters: Parameters) -> np.ndarray:
    """``Sigma_C(y)`` -- eq. (18).  Singular at ``y = +-a`` and ``y = +-b``."""
    y = np.asarray(y, dtype=float)
    a, b = parameters.fault_a, parameters.fault_b
    numerator = (y - a) ** 2 * (y + a) ** 2
    denominator = (y - b) ** 2 * (y + b) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * parameters.slip_stress_scale * np.log(numerator / denominator)


def analytic_slip(y, parameters: Parameters) -> np.ndarray:
    """``delta(y)`` -- eq. (20), the five-interval piecewise profile.

    Continuous, piecewise linear, and flat at ``(C/A)(a-b)`` over the overlap
    ``|y| < a`` where reservoir faces reservoir.
    """
    y = np.asarray(y, dtype=float)
    a, b = parameters.fault_a, parameters.fault_b
    scale = parameters.slip_stress_scale / parameters.slip_stiffness
    shape = np.select(
        [y <= -b, y <= -a, y < a, y < b],
        [np.zeros_like(y), -(y + b), np.full_like(y, a - b), y - b],
        default=np.zeros_like(y),
    )
    return scale * shape


def peak_slip(parameters: Parameters) -> float:
    """``|delta|`` over the overlap -- ``0.1817`` m for the published parameters."""
    return abs(
        parameters.slip_stress_scale
        / parameters.slip_stiffness
        * (parameters.fault_a - parameters.fault_b)
    )


# -- the simulation -------------------------------------------------------------------


def build(parameters: Parameters, nx: int = 20, ny: int = 60):
    """Mesh, fault tags and the depletion pressure field of the offset reservoir."""
    width, height = parameters.width, parameters.height
    mesh = structured_quads(
        nx, ny, lengths=(width, height), origin=(-width / 2, -height / 2)
    )
    fault = facets_on_plane(mesh, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    # the reservoir is displaced across the fault: [-b, a] on the left, [-a, b]
    # on the right, so each side faces seal over part of the throw
    centroids = mesh.geometry.centroids(2)
    a, b = parameters.fault_a, parameters.fault_b
    left = centroids[:, 0] < 0.0
    lower = np.where(left, -b, -a)
    upper = np.where(left, a, b)
    inside = (centroids[:, 1] > lower) & (centroids[:, 1] < upper)
    pressure = np.where(inside, parameters.depletion, 0.0)
    return mesh, fault, pressure


def mechanics_factory(mesh, parameters: Parameters, pressure):
    """A ``mechanics`` factory carrying the depletion load and the roller frame.

    The pore pressure reaches the fault entirely through ``extra_rhs``: in the
    quasi-steady Biot problem the pressure is data, and its whole contribution to
    the stress row is ``-(alpha/(dK)) T^T p``.  The contact driver therefore
    never sees a pressure, a material or a boundary condition.
    """
    material = Material(
        shear_modulus=parameters.shear_modulus,
        poisson=parameters.poisson,
        biot=parameters.biot,
    )
    poro = PoroMechanics(mesh, material)
    coupling = sp.diags(poro.material.pressure_coupling(2)) @ poro.trace_operator()
    extra = -(coupling.T @ pressure)

    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][1] - parameters.height / 2) < 1e-9]
    rollers = [f for f in boundary if f not in set(top)]

    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    free = lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3))  # noqa: E731

    def factory(contact=None):
        problem = MixedElasticity(mesh, contact=contact)
        problem.inner = ElasticityInnerProduct(mesh, material=material)
        problem._ops = None
        matrix, rhs = problem.assemble_constrained(
            dirichlet=zero,
            extra_rhs=extra,
            traction=free,
            traction_facets=top,  # the overburden increment is zero
            roller_facets=rollers,
        )
        return problem, matrix, rhs

    return factory


def pre_slip_stress(parameters: Parameters, nx: int = 20, ny: int = 60):
    """Coulomb stress on the **locked** fault -- Fig. 6 (left), eq. (18).

    The other half of the benchmark, and a different computation: the fault is
    not allowed to slip at all, so this is the plain continuous medium under the
    depletion load.  The mechanics factory is called with no contact block and
    solved directly -- no contact map, no iteration.

    On a vertical fault the Coulomb stress is just the shear stress, and eq. (18)
    is singular at the four reservoir edges ``y = +-a, +-b``, so a cell-centred
    value can only ever track it away from those points.
    """
    mesh, fault, pressure = build(parameters, nx=nx, ny=ny)
    problem, matrix, rhs = mechanics_factory(mesh, parameters, pressure)(None)
    solution = problem.split(
        solve_saddle(matrix, rhs, problem.block_sizes, method="direct")
    )

    # read the shear traction **on the fault facets**, not from the cells beside
    # them.  In Hellinger--Reissner the facet traction moments are primary
    # unknowns, so this is the value on the plane itself; a cell-centred sigma_xy
    # is sampled half a cell away, where the stress has already decayed, and no
    # amount of refinement in y fixes an error in x.
    driver = ContactDriver(
        mesh, fault, FrictionlessBilateral(), mu=parameters.shear_modulus, lam=1.0
    )
    traction = driver.tractions(solution["stress"])
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    return {
        "y": y[order],
        "coulomb_stress": traction[order, 1],
        "solution": solution,
    }


def insitu_prestress(mesh, fault, parameters: Parameters) -> np.ndarray:
    """In-situ fault traction at the enforcement points, ``(n_facets, 2)``.

    A contact law constrains the **total** traction, so an incremental solve has
    to tell it what it is sitting on: on a vertical fault the in-situ normal
    traction is ``sigma_xx(y) ~ -57`` MPa and the shear vanishes.  Without this
    a unilateral law reads the tensile *increment* as opening; with it, the same
    law correctly finds the fault shut.
    """
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    prestress = np.zeros((len(y), 2))
    prestress[:, 0] = parameters.horizontal_stress(y)  # sigma_xx on a vertical fault
    return prestress


def simulate(
    parameters: Parameters,
    nx: int = 20,
    ny: int = 60,
    law=None,
    prestress: bool = True,
):
    """Solve the displaced-fault problem; return slip against ``y``.

    ``law`` defaults to ``SignoriniCoulomb(friction=0)`` -- the physical model.
    Pass :class:`FrictionlessBilateral` to run the bonded variant: with
    ``prestress=True`` the two agree to round-off because the fault really is
    shut, and with ``prestress=False`` the unilateral one opens it, which is the
    failure this benchmark is able to detect.
    """
    mesh, fault, pressure = build(parameters, nx=nx, ny=ny)
    driver = ContactDriver(
        mesh,
        fault,
        SignoriniCoulomb(friction=0.0) if law is None else law,
        prestress=(
            insitu_prestress(mesh, fault, parameters) if prestress else None
        ),
        mu=parameters.shear_modulus,
        lam=2.0
        * parameters.shear_modulus
        * parameters.poisson
        / (1.0 - 2.0 * parameters.poisson),
        relaxation=1.0,  # the projection is affine here, so no damping is needed
        tolerance=1e-10,
        max_iterations=400,
    )
    state = driver.solve_step(
        mechanics_factory(mesh, parameters, pressure), solver="newton"
    )
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    return {
        "y": y[order],
        "slip": state.jump[order, 1],
        "traction": driver.tractions(state.solution["stress"])[order],
        "state": state,
        "cells": mesh.num_cells(2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=60)
    arguments = parser.parse_args()

    parameters = Parameters()
    print("Benchmark 1 -- vertical displaced fault, frictionless\n")
    print(f"  a = {parameters.fault_a:g} m, b = {parameters.fault_b:g} m, "
          f"throw = {parameters.throw:g} m, h = {parameters.reservoir_height:g} m")
    print(f"  C = {parameters.slip_stress_scale:.4e} Pa      (paper -2.95e6)")
    print(f"  A = {parameters.slip_stiffness:.4e} Pa      (paper  1.2171e9)")
    print(f"  C/A = {parameters.slip_stress_scale / parameters.slip_stiffness:.6f}"
          "        (paper -0.0024)")
    print(f"  peak |delta| = {peak_slip(parameters):.4f} m\n")

    result = simulate(parameters, nx=arguments.nx, ny=arguments.ny)
    state = result["state"]
    print(f"  mesh {arguments.nx} x {arguments.ny} = {result['cells']} cells, "
          f"{len(result['y'])} fault facets")
    print(f"  converged={state.converged} in {state.iterations} iterations\n")

    exact = analytic_slip(result["y"], parameters)
    print("      y [m]      slip [m]     analytic [m]")
    for y, got, want in zip(result["y"], np.abs(result["slip"]), np.abs(exact)):
        if abs(y) <= 1.2 * parameters.fault_b:
            print(f"   {y:9.1f}   {got:11.5f}   {want:11.5f}")
    print(f"\n  peak slip: simulated {np.abs(result['slip']).max():.4f} m, "
          f"analytic {peak_slip(parameters):.4f} m")
    print(f"  max |shear traction| on the fault: "
          f"{np.abs(result['traction'][:, 1]).max():.3e} Pa (must be ~0)")


if __name__ == "__main__":
    main()
