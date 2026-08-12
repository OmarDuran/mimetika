r"""Consolidation of a soil layer (Coussy Sect. 5.2.2) -- the Terzaghi problem.

A saturated poroelastic layer of thickness ``h`` rests on a rigid impervious
base; its upper surface is drained and loaded at ``t = 0`` by a step
compressive traction ``sigma_0``.  Because the fluid cannot escape
instantaneously it carries the whole load at first; it then drains through
the surface and the load transfers to the skeleton.  The layer settles as it
does so -- the celebrated consolidation problem of Terzaghi (1923), solved
as Sect. 5.2.2 of Coussy, *Poromechanics* (Wiley, 2004).

This benchmark reproduces **Coussy's Figure 5.3**: the normalized
overpressure ``pbar`` against the normalized depth ``zbar = z/h`` measured
from the drained surface, for ``tbar = c_f t / h^2`` in {0.001, 0.01, 0.1,
0.25, 0.5, 1.0} (his Eq. 5.87), with the **early-time** similarity solution
``pbar = erf(zbar / (2 sqrt(tbar)))`` (Eq. 5.83) dashed at ``tbar = 0.1``
and ``0.25`` -- at 0.001 and 0.01 it cannot be distinguished from the exact
series, exactly as the book notes.  The same figure is produced for both
cell families (cartesian, simplex), each with the 2D and 3D columns overlaid
on the closed form.

Why this problem and not a harder one
-------------------------------------
Every boundary condition the fault benchmarks rely on appears here exactly once,
and the answer is known in closed form:

* an **applied traction** on the loaded face,
* **rollers** on the sides and base -- the uniaxial-strain constraint,
* a **drained** face (pressure natural, ``p = 0``),
* **sealed** faces (flux essential, pinned to zero).

Get one of them wrong and the column stops being one-dimensional: it bulges, or
it drains from the wrong face, or it never reaches the right final settlement.
None of those failures is subtle once the profile is plotted against the analytic
curve, which is the point -- the fault benchmarks fail *quietly*.

The 2D and 3D columns must produce the **same** curve.  A 3D box under uniaxial
strain has no extra freedom to use, so any spread between them is a bug in the
boundary conditions or the coupling rather than in the physics.

The closed form
---------------
With ``zeta`` measured from the drained surface and the **time factor**
``T = c_v t / L^2``,

    ``p/p_0 = (4/pi) sum_m 1/(2m+1) sin[(2m+1) pi zeta/2] exp[-(2m+1)^2 pi^2 T/4]``

    ``U(T)  = 1 - beta (8/pi^2) sum_m 1/(2m+1)^2 exp[-(2m+1)^2 pi^2 T/4]``

resting on five constants derived from the material:

    ``K_v  = 2 G (1-nu)/(1-2nu)``           the oedometer (confined) modulus
    ``S    = 1/M + alpha^2/K_v``            storage under uniaxial strain
    ``c_v  = (k/mu_f) / S``                 the consolidation coefficient
    ``p_0  = alpha sigma_0 / (K_v S)``      the undrained pressure
    ``beta = alpha^2 / (K_v S)``            the fluid's share of the load

The ``beta`` in ``U`` is not decoration.  Terzaghi's original column has
``alpha = 1`` and an incompressible fluid, so ``beta = 1``, nothing settles until
fluid leaves, and ``U`` starts at zero -- which is the form every textbook
prints.  A general Biot column settles ``1 - beta`` of the way *instantly*, and
comparing it against the ``beta = 1`` curve makes a correct solver look broken by
a constant offset at early time.

``S`` uses ``K_v``, not the bulk ``K``: the column cannot strain laterally, so
the stiffness the fluid feels is the confined one.  Using ``K`` here is the
classic slip, and it shows up as a consolidation rate that is wrong by a fixed
factor while the profile *shape* still looks perfect.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from mimetika.assembly.four_field import FourFieldPoroMechanics
from mimetika.assembly.mixed import boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.materials import Material
from mimetika.mesh import (
    structured_box,
    structured_quads,
    structured_tets,
    structured_triangles,
)
from mimetika.simulation import (
    FlowBC,
    MechanicsBC,
    PoromechanicsSolver,
    RKTimeStepping,
)

#: the six time factors of Coussy's Figure 5.3
TIME_FACTORS = (0.001, 0.01, 0.1, 0.25, 0.5, 1.0)

#: the times at which the book dashes the early-time similarity solution --
#: at 0.001 and 0.01 it coincides with the series to plotting accuracy
EARLY_TIME_SHOWN = (0.1, 0.25)

#: ``family -> (2D generator, 3D generator)``.  Both families must give the same
#: answer, but they do not exercise the same code: on simplices the mimetic
#: stabilization term vanishes identically, so ``simplex`` runs the scheme with
#: ``M = M1`` alone while ``cart`` also carries ``M2``.
FAMILIES = {
    "cart": (structured_quads, structured_box),
    "simplex": (structured_triangles, structured_tets),
}


@dataclass(frozen=True)
class Column:
    """Geometry, material and load.  Defaults are a stiff, low-permeability rock."""

    height: float = 10.0  # L
    width: float = 1.0  # lateral extent; the answer must not depend on it
    shear_modulus: float = 6.0e8  # G
    poisson: float = 0.2  # nu
    biot: float = 0.9  # alpha
    inverse_biot_modulus: float = 1.0e-10  # 1/M; 0 for an incompressible fluid
    permeability: float = 1.0e-13  # k
    viscosity: float = 1.0e-3  # mu_f
    load: float = 1.0e7  # sigma_0, compressive and positive

    # -- derived ---------------------------------------------------------------

    @property
    def oedometer_modulus(self) -> float:
        """``K_v = 2G(1-nu)/(1-2nu)`` -- the stiffness felt under zero lateral strain."""
        return (
            2.0
            * self.shear_modulus
            * (1.0 - self.poisson)
            / (1.0 - 2.0 * self.poisson)
        )

    @property
    def storage(self) -> float:
        """``S = 1/M + alpha^2/K_v``, the *uniaxial* storage coefficient."""
        return self.inverse_biot_modulus + self.biot**2 / self.oedometer_modulus

    @property
    def consolidation_coefficient(self) -> float:
        """``c_v = (k/mu_f)/S``."""
        return self.permeability / self.viscosity / self.storage

    @property
    def initial_pressure(self) -> float:
        """``p_0 = alpha sigma_0/(K_v S)`` -- the undrained response at ``t = 0+``."""
        return self.biot * self.load / (self.oedometer_modulus * self.storage)

    @property
    def final_settlement(self) -> float:
        """``sigma_0 L / K_v`` -- the drained compaction, once ``p`` has gone."""
        return self.load * self.height / self.oedometer_modulus

    @property
    def load_partition(self) -> float:
        r"""``beta = alpha^2/(K_v S) = alpha p_0/sigma_0`` -- the fluid's share at ``t=0+``.

        The fraction of the applied load the fluid takes instantaneously, and so
        the fraction of the settlement that is *delayed*.  The remaining
        ``1 - beta`` happens at once, undrained, because a compressible fluid in
        compressible grains can be squeezed without draining.

        Terzaghi's original problem has ``beta = 1``: with ``alpha = 1`` and
        ``1/M = 0`` the storage is exactly ``1/K_v``, nothing moves until fluid
        leaves, and the textbook ``U(T)`` starting from zero is recovered.  Any
        other material settles before the clock starts.
        """
        return self.biot**2 / (self.oedometer_modulus * self.storage)

    def time(self, factor) -> np.ndarray:
        """Physical time from the dimensionless time factor ``T``."""
        return (
            np.asarray(factor, dtype=float)
            * self.height**2
            / self.consolidation_coefficient
        )

    def material(self) -> Material:
        return Material(
            shear_modulus=self.shear_modulus,
            poisson=self.poisson,
            biot=self.biot,
            inverse_biot_modulus=self.inverse_biot_modulus,
            permeability=self.permeability,
            viscosity=self.viscosity,
        )


# -- the closed form ------------------------------------------------------------------


def _terms_for(factor: float, cap: int = 200_000) -> int:
    """Enough terms that the last one has decayed to nothing.

    The mode ``m`` decays as ``exp[-(2m+1)^2 pi^2 T/4]``, so a fixed truncation is
    a trap: at ``T = 1`` a dozen terms suffice, while at ``T = 1e-5`` the series is
    still essentially the square wave of the initial condition and needs some
    hundreds.  Truncating too early does not merely lose accuracy, it produces
    Gibbs ripples that can move the location of the maximum.
    """
    factor = max(float(factor), 1e-30)
    need = 0.5 * (2.0 / np.pi) * np.sqrt(40.0 / factor)
    return int(min(max(50.0, need + 10.0), cap))


def analytic_pressure(elevation, factor, terms: int | None = None) -> np.ndarray:
    """``p/p_0`` at height ``z/L`` above the sealed base, at time factor ``T``.

    ``zeta = 1 - z/L`` is the distance from the *drained* top, which is where the
    series is anchored: the sine vanishes there and its derivative vanishes at the
    sealed base.  ``terms`` defaults to a count chosen from ``T``.
    """
    zeta = 1.0 - np.asarray(elevation, dtype=float)
    odd = 2 * np.arange(terms if terms else _terms_for(factor)) + 1
    phase = np.sin(np.outer(zeta, odd) * np.pi / 2.0)
    decay = np.exp(-(odd**2) * np.pi**2 * float(factor) / 4.0)
    return (4.0 / np.pi) * (phase * (decay / odd)).sum(axis=1)


def early_time_pressure(depth, factor) -> np.ndarray:
    """Coussy Eq. 5.83: ``pbar = erf(zbar / (2 sqrt(tbar)))``.

    The similarity solution of the half-space: at early times the diffusion
    has not yet felt the sealed base, so the layer behaves as if
    semi-infinite.  ``depth`` is ``zbar``, measured from the drained
    surface.
    """
    from scipy.special import erf

    return erf(np.asarray(depth, dtype=float)
               / (2.0 * np.sqrt(float(factor))))


def analytic_consolidation(factor, partition: float = 1.0, terms: int = 400):
    r"""Degree of consolidation ``U = w(t)/w_inf``, ending at 1.

    ``U = 1 - beta (8/pi^2) sum_m exp[-(2m+1)^2 pi^2 T/4] / (2m+1)^2``, which
    follows from ``w = (sigma_0 L - alpha \int p\,dz)/K_v`` and the same series.

    ``partition`` is ``beta = alpha^2/(K_v S)``.  The textbook curve is
    ``beta = 1`` and starts at ``U = 0``; a general Biot column starts at
    ``1 - beta``, having already settled undrained.  Leaving ``beta`` out is an
    easy way to "find" an error in a solver that is in fact correct.
    """
    odd = 2 * np.arange(terms) + 1
    f = np.atleast_1d(np.asarray(factor, dtype=float))
    decay = np.exp(-np.outer(f, odd**2) * np.pi**2 / 4.0)
    total = (decay / odd**2).sum(axis=1)
    # The neglected tail is not negligible at small T, where every mode is still
    # alive: sum_{m>=N} 1/(2m+1)^2 ~ 1/(4N), which at N = 400 is 6e-4 -- enough to
    # shift U(0+) in the fourth digit and look like a solver error.  The whole
    # series is pi^2/8, so the tail is known exactly; weighting it by the slowest
    # surviving decay is exact as T -> 0 and vanishes when T is large.
    head = (1.0 / odd**2).sum()
    total = total + decay[:, -1] * (np.pi**2 / 8.0 - head)
    return 1.0 - partition * (8.0 / np.pi**2) * total


# -- the discrete column --------------------------------------------------------------


def build(column: Column, dim: int, axial: int, lateral: int, family: str = "cart"):
    """Mesh plus the facet sets naming each boundary condition.

    The column axis is the **last** coordinate in either dimension -- ``y`` in 2D,
    ``z`` in 3D -- so one set of code drives both.  ``family`` picks quadrilaterals
    and hexahedra (``cart``) or triangles and tetrahedra (``simplex``).
    """
    try:
        planar, solid = FAMILIES[family]
    except KeyError:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    if dim == 2:
        mesh = planar(lateral, axial, lengths=(column.width, column.height))
    elif dim == 3:
        mesh = solid(
            lateral, lateral, axial,
            lengths=(column.width, column.width, column.height),
        )
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    axis = dim - 1
    centroids = mesh.geometry.centroids(dim - 1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary if abs(centroids[f][axis] - column.height) < 1e-9]
    confined = [f for f in boundary if f not in set(top)]
    return mesh, top, confined


def simulate(
    column: Column,
    dim: int = 2,
    factors=TIME_FACTORS,
    axial: int = 60,
    lateral: int = 2,
    per_decade: int = 12,
    family: str = "cart",
    poromechanics=FourFieldPoroMechanics,
    **solver,
):
    """March the column through the time factors; sample ``p`` and the settlement.

    Backward Euler from the **unloaded** state.  That is the right initial
    condition and it needs no special case: with zero previous fluid content the
    first step enforces ``alpha div u + p/M = 0``, which *is* the undrained
    response, so the instantaneous pressure rise falls out of the scheme rather
    than being imposed on it.

    Steps are geometric.  The solution decays like ``exp(-pi^2 T/4)``, so uniform
    steps would either crawl through the tail or miss the whole early transient.
    """
    mesh, top, confined = build(column, dim, axial, lateral, family)
    axis = dim - 1

    load = np.zeros((3, 3))
    load[axis, axis] = -column.load  # tension positive: a push is negative
    applied = lambda x: np.broadcast_to(  # noqa: E731
        load, (len(np.atleast_2d(x)), 3, 3)
    )
    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731

    targets = sorted(float(f) for f in factors)
    schedule = _schedule(targets, per_decade)

    # the full poromechanics solver: solved flow, adaptive (geometric) time
    # schedule, drained top (pressure natural), sealed everywhere else (flux
    # essential).  ``poromechanics=PoroMechanics`` selects the classic
    # five-field system; the four-field default agrees to solver round-off.
    stepping = RKTimeStepping(schedule=[column.time(f) for f in schedule])
    engine = PoromechanicsSolver(
        mesh, column.material(),
        flow="solved",
        bc=MechanicsBC(dirichlet=zero, traction=applied,
                       traction_facets=top, roller_facets=confined),
        flow_bc=FlowBC(flux_facets=confined),
        time=stepping,
        poromechanics=poromechanics,
        linear_solver="direct",
    )
    poro = engine.poro

    centroids = mesh.geometry.centroids(dim)
    elevation = centroids[:, axis] / column.height
    volume = mesh.geometry.measure(dim)

    previous, profiles, settlement = None, {}, {}
    for (t, step), factor in zip(stepping.steps(), schedule):
        solution = engine.flow_step(previous=previous, dt=step, **solver)
        previous = solution
        if factor in targets:
            profiles[factor] = _layers(elevation, solution["pressure"])
            settlement[factor] = _settlement(poro, solution, column, axis, volume)
    return {
        "dim": dim,
        "family": family,
        "profiles": profiles,
        "settlement": settlement,
        "cells": mesh.num_cells(dim),
    }


def _schedule(targets, per_decade: int) -> list[float]:
    """Geometric time factors that land exactly on every target."""
    out, previous = [], targets[0] / 10.0
    for target in targets:
        span = max(np.log10(target / previous), 0.0)
        n = max(1, int(round(per_decade * span)))
        out.extend(np.geomspace(previous, target, n + 1)[1:].tolist())
        previous = target
    return [targets[0] / 10.0] + out


def _layers(elevation, pressure):
    """Collapse the cross-section: ``(z/L, mean p, lateral spread)`` per layer.

    The spread is the diagnostic.  A correctly confined column is uniform across
    its section, so anything above round-off means the lateral boundary is not
    doing its job -- which is exactly the failure this benchmark exists to catch.
    """
    keys = np.round(elevation, 9)
    out = []
    for key in np.unique(keys):
        band = pressure[keys == key]
        out.append((float(key), float(band.mean()), float(band.max() - band.min())))
    return np.array(out)


def _settlement(poro, solution, column: Column, axis: int, volume) -> float:
    """Top-surface settlement from the mean axial strain, ``-<eps_zz> L``.

    ``eps = C^{-1}(sigma + alpha p I)``, so the axial strain comes off the stress
    and pressure directly; under uniaxial strain the volumetric strain *is* the
    axial one, which is what ``volumetric_strain`` returns.
    """
    strain = poro.volumetric_strain(solution)
    return -float(np.sum(strain * volume) / np.sum(volume)) * column.height


# -- reporting ------------------------------------------------------------------------


def compare(column: Column, result) -> list[dict]:
    """Numeric against analytic, per time factor."""
    rows = []
    for factor, layers in result["profiles"].items():
        elevation, pressure, spread = layers[:, 0], layers[:, 1], layers[:, 2]
        exact = analytic_pressure(elevation, factor) * column.initial_pressure
        scale = column.initial_pressure
        rows.append({
            "factor": factor,
            "rms": float(np.sqrt(np.mean((pressure - exact) ** 2)) / scale),
            "max": float(np.abs(pressure - exact).max() / scale),
            "spread": float(spread.max() / scale),
            "consolidation": result["settlement"][factor] / column.final_settlement,
            "consolidation_exact": float(
                analytic_consolidation(factor, column.load_partition)[0]
            ),
        })
    return rows


CELL_NAMES = {"cart": ("quadrilaterals", "hexahedra"),
              "simplex": ("triangles", "tetrahedra")}


#: where along each curve the ``tbar`` label sits (a depth per factor,
#: spread like the book's)
_LABEL_DEPTH = {0.001: 0.10, 0.01: 0.22, 0.1: 0.38, 0.25: 0.55,
                0.5: 0.72, 1.0: 0.90}


def figure(column: Column, results, path: str = "consolidation_soil.png",
           family: str = "cart") -> str:
    """Coussy's Figure 5.3, with the 2D and 3D numerics over the closed form.

    Normalized overpressure ``pbar`` against normalized depth ``zbar``
    (downward, 0 at the drained surface); solid lines the exact series
    (Eq. 5.87), dashed the early-time similarity solution (Eq. 5.83) at
    ``tbar = 0.1, 0.25``; markers the mimetic columns.
    """
    planar, solid = CELL_NAMES[family]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    markers = {2: "o", 3: "x"}

    fine = np.linspace(0.0, 1.0, 400)  # zbar, from the drained surface
    for factor in TIME_FACTORS:
        pbar = analytic_pressure(1.0 - fine, factor)
        ax.plot(pbar, fine, "k-", lw=1.2)
        z_lab = _LABEL_DEPTH[factor]
        p_lab = float(analytic_pressure(1.0 - z_lab, factor)[0])
        ax.annotate(rf"$\bar t = {factor:g}$", (p_lab, z_lab),
                    xytext=(p_lab + 0.03, z_lab - 0.015), fontsize=9)
    for factor in EARLY_TIME_SHOWN:
        ax.plot(early_time_pressure(fine, factor), fine, "k--", lw=1.0)

    for result in results:
        marker = markers[result["dim"]]
        thin = max(1, len(next(iter(result["profiles"].values()))) // 24)
        for factor in TIME_FACTORS:
            layers = result["profiles"][factor][::thin]
            ax.plot(layers[:, 1] / column.initial_pressure,
                    1.0 - layers[:, 0], marker, color="k", ms=4,
                    mfc="none", mew=0.8, lw=0, alpha=0.85)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # depth increases downward, as in the book
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(r"$\bar p$")
    ax.set_ylabel(r"$\bar z$")
    ax.grid(alpha=0.2)
    ax.set_title(
        f"Consolidation of a soil layer -- Coussy Fig. 5.3 ({family})\n"
        f"solid: exact series   dashed: early-time erf "
        rf"($\bar t$ = 0.1, 0.25)   o: 2D {planar}   x: 3D {solid}",
        fontsize=10, pad=28,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axial", type=int, default=160)
    parser.add_argument("--lateral", type=int, default=2)
    parser.add_argument("--per-decade", type=int, default=32)
    parser.add_argument("--prefix",
                        default="benchmarks/poroelasticity/consolidation_soil")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--families", nargs="+", default=sorted(FAMILIES))
    args = parser.parse_args(argv)

    column = Column()
    print("Consolidation of a soil layer (Coussy Sect. 5.2.2 -- Terzaghi)")
    print(f"  K_v = {column.oedometer_modulus:.4e} Pa    S = {column.storage:.4e} 1/Pa")
    print(f"  c_v = {column.consolidation_coefficient:.4e} m^2/s")
    print(f"  p_0 = {column.initial_pressure:.4e} Pa   "
          f"({column.initial_pressure / column.load:.3f} sigma_0)")
    print(f"  beta = {column.load_partition:.4f}   w_inf = {column.final_settlement:.4e} m")

    for family in args.families:
        planar, solid = CELL_NAMES[family]
        results = []
        for dim in args.dims:
            # simplices split each cell into 2 (2D) or 6 (3D), so the same axial
            # count is a much larger system; keep 3D affordable
            axial = args.axial if dim == 2 else max(20, args.axial // 2)
            result = simulate(column, dim=dim, axial=axial, lateral=args.lateral,
                              per_decade=args.per_decade, family=family)
            results.append(result)
            name = planar if dim == 2 else solid
            print(f"\n  {family} / {dim}D {name}, {result['cells']} cells")
            print("     T        rms(p)/p0   max(p)/p0   lateral    U num    U exact")
            for row in compare(column, result):
                print("   %7.0e   %9.2e   %9.2e   %9.2e   %6.4f   %6.4f"
                      % (row["factor"], row["rms"], row["max"], row["spread"],
                         row["consolidation"], row["consolidation_exact"]))
        path = f"{args.prefix}_{family}.png"
        print("\nwrote", figure(column, results, path, family=family))


if __name__ == "__main__":
    main()
