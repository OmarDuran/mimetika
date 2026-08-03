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
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.mixed import MixedElasticity, boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.contact import ContactDriver, FrictionlessBilateral, SignoriniCoulomb
from mimetika.materials import Material
from mimetika.mesh import (
    graded_coordinates,
    graded_quads,
    graded_triangles,
    structured_quads,
)
from mimetika.mesh.fracture import facets_on_plane
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.postprocess import (
    MixedDimensionalSeries,
    contact_fields,
    mechanics_fields,
)
from mimetika.solver.saddle import solve_saddle

from benchmarks.contact_mechanics.common import Parameters

#: Benchmark 1 runs on a much larger domain than Table 2's 4500 m box, because
#: eqs. (18)-(20) are posed for an **unbounded** medium and a box that small does
#: not approximate one.  Two separate truncations bite, and both were measured:
#:
#: * **Width.**  The paper says this itself in Sect. 4.1 (p. 11) -- its own
#:   results deviate from the semi-analytical ones and "this discrepancy
#:   disappears if the width W of the simulation domain is increased", after
#:   which it reruns at W = 18,000 m (Figs. 10, 11).  Peak slip here goes
#:   -2.61% -> +1.25% on the same change, and is converged in W by 36 km.
#: * **Height.**  The fault runs the *full* height (p. 9), so H is also the fault
#:   length, and a fault stopping at +-H/2 leaves an end effect the infinite-fault
#:   solution has no counterpart for -- a spurious far-field slip tail.  It decays
#:   as H grows: H = 4500 -> +1.25% peak / 5.6 mm tail; H = 9000 -> +0.10% / 2.2 mm;
#:   H = 18000 -> -0.96% / 1.0 mm.  By 18 km the peak has converged onto the
#:   scheme's own fault-compliance error, measured independently at -0.92%.
#:
#: H = 9000 m is used: the tail is small and the peak sits on the analytic value.
#: The cost is nil -- the far field is meshed at ``boundary_spacing`` = 500 m and
#: only has to be present, not resolved.
WIDE_DOMAIN = dict(width=18000.0, height=9000.0)


def wide_parameters(**overrides) -> Parameters:
    """Table 2 on :data:`WIDE_DOMAIN`; pass ``width``/``height`` to override."""
    return Parameters(**{**WIDE_DOMAIN, **overrides})


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


#: Half-width of the uniformly refined near field, in metres, and the factor by
#: which it is refined relative to ``spacing``.  The analytic slip is supported on
#: ``|y| <= b = 150`` m but the *numerical* solution carries a far-field tail well
#: beyond it, so resolving only the reservoir edges leaves the region that sets
#: that tail on stretched cells.  ``[-400, 400]`` in both directions at half the
#: reservoir spacing covers it; outside, the mesh coarsens to ``boundary_spacing``.
NEAR_FIELD = None
NEAR_FIELD_REFINEMENT = 1.0


def build(parameters: Parameters, nx: int = 20, ny: int = 60, spacing=None,
          boundary_spacing: float = 500.0, triangles: bool = False,
          near_field: float | None = NEAR_FIELD):
    """Mesh, fault tags and the depletion pressure field of the offset reservoir.

    With ``spacing`` given the mesh is **graded**: nodes are placed exactly on the
    reservoir edges ``y = +-a, +-b`` and on the fault ``x = 0``, uniform at
    ``spacing`` across the reservoir, coarsening geometrically outwards.  That is
    the right mesh for this problem -- the features span 300 m inside a 4500 m
    domain, so a uniform grid spends almost all its cells where nothing happens.
    Passing ``nx``/``ny`` instead gives the uniform mesh, kept for comparison.
    """
    width, height = parameters.width, parameters.height
    if spacing is not None:
        a, b = parameters.fault_a, parameters.fault_b
        # Table 3 gives 2 m at the refined region and 100 m at the domain
        # boundary, for their 4500 m box.  The domain here is 18 km wide and 9 km
        # tall (see ``Parameters``) because the analytic solution is posed on an
        # unbounded medium, and 100 m cells over that extension would be 24k cells
        # of pure filler.  They buy nothing: coarsening the *extension* from 100 m
        # to 2 km moves the peak slip by 0.03% and leaves the far-field tail
        # unchanged, while the mesh shrinks 10x and the solve runs 22x faster.
        # The far field only has to be there, not resolved.  The reservoir itself
        # is still meshed at ``spacing``.
        fine = spacing / NEAR_FIELD_REFINEMENT
        near = None if near_field is None else (-near_field, near_field)
        ys = graded_coordinates([-b, -a, a, b], (-height / 2, height / 2),
                                spacing, max_spacing=boundary_spacing,
                                window=near, window_spacing=fine)
        xs = graded_coordinates([0.0], (-width / 2, width / 2),
                                spacing, max_spacing=boundary_spacing,
                                window=near, window_spacing=fine)
        mesh = graded_triangles(xs, ys) if triangles else graded_quads(xs, ys)
    else:
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
        problem = MixedElasticity(
            mesh,
            contact=contact,
            inner=ElasticityInnerProduct(mesh, material=material),
        )
        matrix, rhs = problem.assemble_constrained(
            dirichlet=zero,
            extra_rhs=extra,
            traction=free,
            traction_facets=top,  # the overburden increment is zero
            roller_facets=rollers,
        )
        return problem, matrix, rhs

    return factory


def pre_slip_stress(parameters: Parameters, nx: int = 20, ny: int = 60,
                    spacing=None, boundary_spacing: float = 500.0,
                    triangles: bool = False):
    """Coulomb stress on the **locked** fault -- Fig. 6 (left), eq. (18).

    The other half of the benchmark, and a different computation: the fault is
    not allowed to slip at all, so this is the plain continuous medium under the
    depletion load.  The mechanics factory is called with no contact block and
    solved directly -- no contact map, no iteration.

    On a vertical fault the Coulomb stress is just the shear stress, and eq. (18)
    is singular at the four reservoir edges ``y = +-a, +-b``, so a cell-centred
    value can only ever track it away from those points.
    """
    mesh, fault, pressure = build(parameters, nx=nx, ny=ny, spacing=spacing,
                                  boundary_spacing=boundary_spacing,
                                  triangles=triangles)
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
    spacing=None,
    boundary_spacing: float = 500.0,
    triangles: bool = False,
    enforcement: str = "averaged",
):
    """Solve the displaced-fault problem; return slip against ``y``.

    ``law`` defaults to ``SignoriniCoulomb(friction=0)`` -- the physical model.
    Pass :class:`FrictionlessBilateral` to run the bonded variant: with
    ``prestress=True`` the two agree to round-off because the fault really is
    shut, and with ``prestress=False`` the unilateral one opens it, which is the
    failure this benchmark is able to detect.
    """
    mesh, fault, pressure = build(parameters, nx=nx, ny=ny, spacing=spacing,
                                  boundary_spacing=boundary_spacing,
                                  triangles=triangles)
    driver = ContactDriver(
        mesh,
        fault,
        SignoriniCoulomb(friction=0.0) if law is None else law,
        prestress=None,
        mu=parameters.shear_modulus,
        lam=2.0
        * parameters.shear_modulus
        * parameters.poisson
        / (1.0 - 2.0 * parameters.poisson),
        enforcement=enforcement,
        relaxation=1.0,  # the projection is affine here, so no damping is needed
        tolerance=1e-10,
        max_iterations=400,
    )
    if prestress:  # per facet; the driver knows the enforcement layout
        driver.prestress = driver.expand_to_points(
            insitu_prestress(mesh, fault, parameters)
        )
    state = driver.solve_step(
        mechanics_factory(mesh, parameters, pressure), solver="newton"
    )
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    # collapse to one value per facet BEFORE ordering: under pointwise
    # enforcement these arrays have several rows per facet
    slip = driver.per_facet(state.jump)
    traction = driver.per_facet(driver.tractions(state.solution["stress"]))
    return {
        "y": y[order],
        "slip": slip[order, 1],
        "traction": traction[order],
        "state": state,
        "cells": mesh.num_cells(2),
    }


def depletion_series(
    path,
    parameters: Parameters,
    steps: int = 6,
    spacing: float = 6.25,
    boundary_spacing: float = 500.0,
):
    """Write the fault's response to a depletion ramp as a ``.pvd`` time series.

    Depletion **is** the time axis here: the problem is quasi-static, so nothing
    depends on real time, but the paper tracks the slip-patch boundaries against
    incremental pressure (Fig. 12) and that is the sweep worth watching.  Each
    step re-solves from scratch -- the frictionless law is path independent, so
    there is no history to carry, and pretending otherwise would be misleading.

    Both parts are written.  The fracture carries the traction and the
    displacement jump -- a fault is a contact interface, not a thin material, so
    that is its whole state.  The surrounding rock carries the displacement and
    stress that drive it, plus the depletion pressure that is the load, and a
    fault plotted without its surroundings cannot be read.
    """
    mesh, fault, _ = build(parameters, spacing=spacing,
                           boundary_spacing=boundary_spacing)
    series = MixedDimensionalSeries(path, mesh, fault)
    lame = (2.0 * parameters.shear_modulus * parameters.poisson
            / (1.0 - 2.0 * parameters.poisson))

    for level in np.linspace(0.0, parameters.depletion, steps + 1)[1:]:
        stage = replace(parameters, depletion=float(level))
        _, _, pressure = build(stage, spacing=spacing,
                               boundary_spacing=boundary_spacing)
        driver = ContactDriver(
            mesh, fault, SignoriniCoulomb(friction=0.0),
            mu=stage.shear_modulus, lam=lame,
            prestress=insitu_prestress(mesh, fault, stage),
        )
        state = driver.solve_step(
            mechanics_factory(mesh, stage, pressure), solver="newton"
        )
        if not state.converged:
            raise RuntimeError(f"depletion {level:.3e} Pa did not converge")
        # the time coordinate is |Delta p| in MPa, which is what Fig. 12 uses
        series.write(
            abs(level) / 1e6,
            bulk=mechanics_fields(state.problem, state.solution, pressure),
            fracture=contact_fields(driver, state),
        )
    return series


def figure_6(parameters: Parameters | None = None, spacing: float = 6.25,
             boundary_spacing: float = 500.0,
             path: str = "benchmarks/contact_mechanics/benchmark_1_fig6.png") -> str:
    """Reproduce Fig. 6: pre-slip Coulomb stress (left) and the resulting slip (right).

    Both panels are the frictionless fault.  The left one holds it **locked** --
    no slip permitted -- so the shear it carries is the driving stress
    ``Sigma_C`` of eq. (18); on a vertical fault with no in-situ shear the Coulomb
    stress is just ``sigma_xy``.  The right one lets it slip, the total shear then
    vanishes, and what remains is the tent of eq. (20).

    Laid out the paper's way round: the quantity on the horizontal axis, depth on
    the vertical, over ``|y| <= 250`` m.  The analytic markers deliberately skip
    ``y = +-a, +-b``, where eq. (18) is logarithmically singular and no
    cell-centred value can follow it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = parameters or wide_parameters()
    locked = pre_slip_stress(parameters, spacing=spacing,
                             boundary_spacing=boundary_spacing)
    slipped = simulate(parameters, spacing=spacing,
                       boundary_spacing=boundary_spacing)

    window = 250.0
    fine = np.linspace(-window, window, 4001)
    marks = np.linspace(-245.0, 245.0, 21)  # 24.5 m apart, clear of +-75 and +-150
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 6.4), sharey=True)

    exact = analytic_coulomb_stress(fine, parameters)
    exact[~np.isfinite(exact)] = np.nan
    left.plot(exact / 1e6, fine, "r-", lw=1.4, zorder=3)
    left.plot(analytic_coulomb_stress(marks, parameters) / 1e6, marks, "ro",
              ms=5, mfc="none", lw=0, label="analytical", zorder=4)
    y, sigma = np.asarray(locked["y"]), np.asarray(locked["coulomb_stress"])
    inside = np.abs(y) <= window
    left.plot(sigma[inside] / 1e6, y[inside], "b-", lw=1.4, label="mimetika")
    left.set_xlabel(r"$\Sigma_C\ (=\sigma_{xy})$   (MPa)")
    left.set_ylabel(r"$y$   (m)")
    left.set_xlim(-20, 20)
    left.set_title("Pre-slip Coulomb stress")

    right.plot(analytic_slip(fine, parameters), fine, "r-", lw=1.4, zorder=3)
    right.plot(analytic_slip(marks, parameters), marks, "ro", ms=5, mfc="none",
               lw=0, label="analytical", zorder=4)
    y, slip = np.asarray(slipped["y"]), np.abs(np.asarray(slipped["slip"]))
    inside = np.abs(y) <= window
    right.plot(slip[inside], y[inside], "b-", lw=1.4, label="mimetika")
    right.set_xlabel(r"$\delta$   (m)")
    right.set_xlim(-0.005, 0.2)
    right.set_title("Resulting slip")

    peak, target = slip.max(), peak_slip(parameters)
    right.text(0.04, 0.03,
               f"peak {peak:.4f} m\nanalytic {target:.4f} m\n({100*(peak/target-1):+.2f}%)",
               transform=right.transAxes, fontsize=9, va="bottom",
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    fig.suptitle("Benchmark 1 -- vertical displaced fault, frictionless    "
                 f"(W = {parameters.width:.0f} m, H = {parameters.height:.0f} m)",
                 fontsize=11)
    for axis in (left, right):
        axis.set_ylim(-window, window)
        axis.grid(alpha=0.25)
        for edge in (parameters.fault_a, parameters.fault_b):
            axis.axhline(edge, color="k", lw=0.6, ls=":")
            axis.axhline(-edge, color="k", lw=0.6, ls=":")
        axis.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--spacing", type=float, default=None,
                        help="graded mesh: cell size at the reservoir edges")
    parser.add_argument("--vtu", nargs="?", const="out/benchmark_1",
                        help="write a PVD depletion series to this path stem")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--width", type=float, default=None,
                        help="domain width W in m (default: see Parameters)")
    parser.add_argument("--height", type=float, default=None,
                        help="domain height H in m; also the fault length")
    parser.add_argument("--figure", nargs="?",
                        const="benchmarks/contact_mechanics/benchmark_1_fig6.png",
                        help="write the Fig. 6 comparison to this path")
    arguments = parser.parse_args()

    parameters = wide_parameters(**{
        name: value
        for name, value in (("width", arguments.width), ("height", arguments.height))
        if value is not None
    })
    print("Benchmark 1 -- vertical displaced fault, frictionless\n")
    print(f"  a = {parameters.fault_a:g} m, b = {parameters.fault_b:g} m, "
          f"throw = {parameters.throw:g} m, h = {parameters.reservoir_height:g} m")
    print(f"  C = {parameters.slip_stress_scale:.4e} Pa      (paper -2.95e6)")
    print(f"  A = {parameters.slip_stiffness:.4e} Pa      (paper  1.2171e9)")
    print(f"  C/A = {parameters.slip_stress_scale / parameters.slip_stiffness:.6f}"
          "        (paper -0.0024)")
    print(f"  peak |delta| = {peak_slip(parameters):.4f} m\n")

    if arguments.vtu:
        series = depletion_series(
            arguments.vtu, parameters, steps=arguments.steps,
            spacing=arguments.spacing or 6.25,
        )
        print(f"  wrote {series.collection} "
              f"({arguments.steps} depletion steps; part 0 = rock cells, "
              f"part 1 = fault)\n")

    if arguments.figure:
        print("  wrote", figure_6(parameters, spacing=arguments.spacing or 6.25,
                                  path=arguments.figure), "\n")

    result = simulate(parameters, nx=arguments.nx, ny=arguments.ny,
                      spacing=arguments.spacing)
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
