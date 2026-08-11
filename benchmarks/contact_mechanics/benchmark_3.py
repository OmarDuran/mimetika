r"""Benchmark 3 -- inclined displaced fault, slip-weakening friction (paper 4.2).

The configuration of benchmark 2, with a linear slip-weakening friction law
(paper Eq. 23): the coefficient falls from ``mu_s = 0.52`` to
``mu_d = 0.20`` over a critical slip distance ``delta_c = 0.02`` m.  Slip
reduces the fault's carrying capacity, and below the *nucleation pressure*
``p*`` no quasi-static equilibrium exists -- a seismic event.  The paper's
semi-analytical estimate (Uenishi & Rice 2003 as modified by Jansen &
Meulenbroek 2022) is ``p* = -17.41`` MPa; its DARTS simulation arrives at
``p* = -17.27`` MPa.  The benchmark sweeps the depletion until the
friction fixed point stops settling and brackets ``p*``.

Resolution.  The Uenishi--Rice critical nucleation length for these
parameters is ``h* ~ 1.16 G delta_c / ((1 - nu)(mu_s - mu_d) |sigma_n'|)
~ 16 m``.  The default mesh is the paper's DARTS grid (its Table 3: 2 m
cells on the fault at the reservoir, 100 m at the boundary), which
resolves ``h*`` and lets the quasi-static fixed point follow the aseismic
patch growth until it runs away at ``p*``.  On coarse fault spacings
(tens of metres) a slipping patch is supercritical as soon as it is
resolvable and the computed ``p*`` collapses onto the mesh-dependent
constant-friction slip onset instead -- expect that when raising
``--spacing``.  The figure shows the runaway cascade of fixed-point
iterates at nucleation.

Reference (APA): Novikov, A., Shokrollahzadeh Behbahani, S., Voskov, D.,
Hajibeygi, H., & Jansen, J.-D. (2024). Benchmarking numerical simulation of
induced fault slip with semi-analytical solutions. *Geomechanics and
Geophysics for Geo-Energy and Geo-Resources, 10*, 182.
https://doi.org/10.1007/s40948-024-00896-1

Run with ``python -m benchmarks.contact_mechanics.benchmark_3``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from mimetika.contact import ContactDriver, SignoriniCoulomb

from benchmarks.contact_mechanics.benchmark_2 import (
    build,
    insitu_prestress,
    mechanics_factory,
    pressure_field,
    wide_parameters,
)
from benchmarks.contact_mechanics.common import Parameters
from benchmarks.contact_mechanics.reference_data import DATA as REFERENCE_DATA
from benchmarks.contact_mechanics.reference_data import load as load_reference


def reference_pre_nucleation():
    """Semi-analytical slip just before nucleation (paper Fig. 14, at the
    paper's ``p* = -17.41`` MPa): two patches with peaks 1.86 / 3.74 mm.

    This is the only genuine slip curve in the 4TU dataset -- the ``delta``
    rows of Figs. 6 and 9/10 are byte-identical copies of this one.
    """
    if not REFERENCE_DATA.exists():
        return None
    data = load_reference("14 right")
    return data["y"], data["delta"]


class SlipWeakening(SignoriniCoulomb):
    """Linear slip-weakening Coulomb friction (paper Eq. 23).

    ``mu(|g_t|) = max(mu_d, mu_s - (mu_s - mu_d) |g_t| / delta_c)``.  Only
    the radius of the friction disk changes; the unilateral normal condition
    and the Alart--Curnier projection are inherited, with the coefficient
    evaluated per enforcement point from the current jump.
    """

    path_dependent = True  # the projection needs the jump

    def __init__(self, static: float = 0.52, dynamic: float = 0.20,
                 critical: float = 0.02):
        super().__init__(friction=static)
        self.static, self.dynamic = float(static), float(dynamic)
        self.critical = float(critical)

    def _mu(self, g) -> np.ndarray | float:
        if g is None:
            return self.static
        slip = np.linalg.norm(np.atleast_2d(g)[:, 1:], axis=1)
        return np.maximum(
            self.dynamic,
            self.static - (self.static - self.dynamic) * slip / self.critical,
        )

    def project(self, trial, state, g=None, g_prev=None, dt=None):
        # the parent projection broadcasts ``self.friction`` against the
        # points, so a per-point coefficient slots straight in
        self.friction = self._mu(g)
        try:
            return super().project(trial, state, g=g, g_prev=g_prev, dt=dt)
        finally:
            self.friction = self.static


def simulate(parameters: Parameters, law: SlipWeakening,
             spacing: float = 2.0, built=None):
    """Quasi-static solve at ``parameters.depletion``; None if no equilibrium."""
    if built is None:
        built = build(parameters, spacing)
    mesh, fault, _ = built
    pressure = pressure_field(mesh, parameters)
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    factory = mechanics_factory(mesh, parameters, pressure)
    pre = insitu_prestress(mesh, fault, parameters, pressure)

    # outer fixed point on the friction coefficient: solve with mu frozen at
    # the previous iterate's slip, then update.  This follows the *stable*
    # quasi-static branch; at nucleation the update runs away instead of
    # settling, which is the physical loss of equilibrium.
    inner = SignoriniCoulomb(friction=law.static)
    mu_pts = None
    state, slip_pts, converged = None, None, False
    y_all = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order_f = np.argsort(y_all)
    y_sorted = y_all[order_f]
    history = []
    for outer in range(40):
        driver = ContactDriver(
            mesh, fault, inner, prestress=None,
            mu=parameters.shear_modulus, lam=lam,
            tolerance=1e-10, max_iterations=200,
        )
        driver.prestress = driver.expand_to_points(pre)
        if mu_pts is not None:
            inner.friction = mu_pts  # per-point array broadcasts in the law
        state = driver.solve_step(factory, solver="newton")
        jump = driver.per_facet(state.jump)
        slip_new = np.abs(jump[:, 1])
        mu_new = np.maximum(
            law.dynamic,
            law.static - (law.static - law.dynamic) * slip_new / law.critical,
        )
        mu_new = driver.expand_to_points(mu_new[:, None]).ravel()
        history.append((y_sorted, np.abs(jump[order_f, 1])))
        if mu_pts is not None and np.abs(mu_new - mu_pts).max() < 1e-4:
            converged = bool(state.converged)
            break
        if slip_new.max() > 50 * law.critical:  # runaway: no equilibrium
            converged = False
            break
        mu_pts = mu_new
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    slip = driver.per_facet(state.jump)[:, 1]
    return {
        "y": y[order], "slip": slip[order], "state": state, "built": built,
        "converged": converged, "history": history,
        "peak": float(np.abs(slip).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="mesh size on the fault at the reservoir "
                             "(2 m = the paper's DARTS grid, Table 3)")
    parser.add_argument("--start", type=float, default=-16.5)
    parser.add_argument("--stop", type=float, default=-19.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--mu-s", type=float, default=0.52)
    parser.add_argument("--mu-d", type=float, default=0.20)
    parser.add_argument("--delta-c", type=float, default=0.02)
    parser.add_argument("--figure",
                        default="benchmarks/contact_mechanics/benchmark_3.png")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()

    parameters = wide_parameters()
    law = SlipWeakening(arguments.mu_s, arguments.mu_d, arguments.delta_c)
    print("Benchmark 3 -- inclined displaced fault, slip-weakening friction\n")
    print(f"  mu_s = {law.static:g}, mu_d = {law.dynamic:g}, "
          f"delta_c = {law.critical:g} m")
    print("  paper: p* = -17.41 MPa (semi-analytical), "
          "-17.27 MPa (DARTS)\n")

    built, last, nucleated, nucleation_res = None, None, None, None
    for level in np.arange(arguments.start, arguments.stop - 1e-9,
                           -abs(arguments.step)):
        stage = replace(parameters, depletion=level * 1e6)
        res = simulate(stage, law, spacing=arguments.spacing, built=built)
        built = res["built"]
        status = ("converged" if res["converged"] else "NO EQUILIBRIUM")
        print(f"  p = {level:7.2f} MPa   peak |slip| = "
              f"{res['peak'] * 1e3:7.3f} mm   {status}")
        if not res["converged"] or res["peak"] > 50 * law.critical:
            nucleated, nucleation_res = level, res
            break
        last = (level, res)

    if nucleated is not None:
        upper = f"{last[0]:.2f}" if last is not None else f"> {nucleated:.2f}"
        print(f"\n  nucleation bracketed: p* in ({nucleated:.2f}, {upper}) MPa")
        print("  (paper: -17.41 / -17.27 MPa; here p* tracks the "
              "mesh-dependent slip onset -- see the docstring's h* caveat)")
    elif last is not None:
        print("\n  no nucleation down to the last level -- extend --stop")

    ref = reference_pre_nucleation()
    if ref is not None and last is not None:
        yr, dr = ref
        ours = np.interp(yr, last[1]["y"], np.abs(last[1]["slip"]))
        sel = np.abs(yr) <= 100.0
        rms = np.sqrt(np.mean((ours[sel] - np.abs(dr[sel])) ** 2))
        print(f"\n  last equilibrium (p = {last[0]:.2f} MPa) vs 4TU Fig. 14 "
              f"(paper's pre-nucleation state at p = -17.41 MPa):")
        print(f"    peaks {np.abs(last[1]['slip']).max() * 1e3:.2f} / "
              f"{np.abs(dr).max() * 1e3:.2f} mm, rms = {rms * 1e3:.2f} mm")
        print("    (different pressures -- the comparison is of the "
              "pre-nucleation *state*, not pointwise equality)")

    if nucleation_res is not None and arguments.figure and not arguments.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 6.0))
        for k, (yy, ss) in enumerate(nucleation_res["history"]):
            sel = np.abs(yy) < 150.0
            ax.plot(np.abs(ss[sel]) * 1e3, yy[sel], lw=1.3,
                    label=f"fixed-point iterate {k}")
        if ref is not None:
            yr, dr = ref
            keep = (np.abs(yr) < 150.0) & (np.abs(dr) > 1e-6)
            ax.plot(np.abs(dr[keep]) * 1e3, yr[keep], "ko", ms=3.0,
                    mfc="white", mew=0.8, lw=0,
                    label="semi-analytical pre-nucleation (4TU)")
        ax.axhline(parameters.fault_a, color="k", lw=0.6, ls=":")
        ax.axhline(-parameters.fault_a, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"$|\delta|$   (mm)")
        ax.set_ylabel("y   (m)")
        ax.set_title("Benchmark 3 -- nucleation cascade at "
                     f"p = {nucleated:g} MPa\n(slip-weakening runaway: no "
                     "quasi-static equilibrium)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=8)
        fig.tight_layout()
        fig.savefig(arguments.figure, dpi=150)
        print(f"  wrote {arguments.figure}")


if __name__ == "__main__":
    main()
