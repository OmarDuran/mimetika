r"""Benchmark 3 -- inclined displaced fault, slip-weakening friction (paper 4.2).

The configuration of benchmark 2, with a linear slip-weakening friction law
(paper Eq. 23): the coefficient falls from ``mu_s = 0.52`` to
``mu_d = 0.20`` over a critical slip distance ``delta_c = 0.02`` m.  Slip
reduces the fault's carrying capacity, and below the *nucleation pressure*
``p*`` no quasi-static equilibrium exists -- a seismic event.  The paper's
semi-analytical estimate (Uenishi & Rice 2003 as modified by Jansen &
Meulenbroek 2022) is ``p* = -17.41`` MPa; its DARTS simulation arrives at
``p* = -17.27`` MPa.  The benchmark continues the *coupled* slip-weakening
solve down the depletion levels, each warm-started from the previous
equilibrium, until the stable branch is lost -- that level brackets ``p*``.

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
    poromechanics_solver,
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
    gap_dependent = True  # ...and the Newton Jacobian needs dP/dg

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
             spacing: float = 2.0, built=None, mu0=None):
    """Quasi-static solve at ``parameters.depletion``; ``None`` past the fold.

    Outer fixed point on a *frozen* friction coefficient: solve plain Coulomb
    with ``mu`` from the previous iterate's slip, update, repeat.  This is
    the physical branch tracker: the linearisation of the ``mu``-update map
    is exactly the slip-weakening stability operator, so the iteration
    contracts precisely while the quasi-static branch is stable and diverges
    at the Uenishi--Rice fold.  For that equivalence to hold the iteration
    must *enter* each level near the branch -- warm-start ``mu0`` from the
    previous depletion level.  (Solving the coupled law directly with Newton
    is not an alternative near the fold: the stable and fully-weakened
    equilibria draw close and Newton hops basins.)
    """
    if built is None:
        mesh, fault, pressure = build(parameters, spacing)
        built = (mesh, fault, pressure, {})
    mesh, fault, _, cache = built
    pressure = pressure_field(mesh, parameters)
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    pre = insitu_prestress(mesh, fault, parameters, pressure)

    inner = SignoriniCoulomb(friction=law.static)
    driver = ContactDriver(
        mesh, fault, inner, prestress=None,
        mu=parameters.shear_modulus, lam=lam,
        tolerance=1e-10, max_iterations=200,
    )
    solver = poromechanics_solver(mesh, parameters, cache=cache,
                                  driver=driver)
    # inner solves are deliberately cold (state=None): with a warm ``g_prev``
    # the tangential driving becomes increment-based and near-threshold
    # facets flip slip direction on noise-scale increments, which loses the
    # branch *earlier*.  Only ``mu`` is continued across levels.
    # near the fold the contraction factor approaches 1 and the iteration
    # creeps: an iteration cap cannot tell slow convergence from divergence,
    # but the mu-update magnitude can -- it shrinks on the stable side and
    # grows past the fold
    mu_pts, converged = mu0, False
    last_change, growing = None, 0
    for outer in range(600):
        if mu_pts is not None:
            inner.friction = mu_pts  # per-point array broadcasts in the law
        state = solver.step(pressure, prestress=pre,
                            recover=False)  # the loop reads only slip
        slip_new = np.abs(driver.per_facet(state.jump)[:, 1])
        mu_new = driver.expand_to_points(np.maximum(
            law.dynamic,
            law.static - (law.static - law.dynamic) * slip_new / law.critical,
        )[:, None]).ravel()
        change = (np.inf if mu_pts is None
                  else float(np.abs(mu_new - mu_pts).max()))
        if change < 1e-4:
            converged = bool(state.converged)
            break
        if slip_new.max() > 50 * law.critical:  # runaway: past the fold
            break
        if last_change is not None and np.isfinite(last_change):
            growing = growing + 1 if change > last_change else 0
            if growing >= 10:  # persistently growing update: divergence
                break
        last_change = change
        mu_pts = mu_new
    # one recovering solve for the fields the postprocessing reads
    state = solver.step(pressure, prestress=pre)
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    slip = driver.per_facet(state.jump)[:, 1]
    converged = converged and float(np.abs(slip).max()) < 50 * law.critical

    # resolved fault profile: the stress space carries BDM_1 facet moments,
    # so traction and jump vary *linearly* along every facet -- twice the
    # information of a facet average (and of a collocated FV on the same
    # mesh).  Read both at the two facet Gauss points.
    pw = ContactDriver(mesh, fault, SignoriniCoulomb(friction=law.static),
                       prestress=None, mu=parameters.shear_modulus, lam=lam,
                       enforcement="pointwise")
    _, _, rhs_full = solver.mechanics(pressure)  # cached residual datum
    gap_pts = pw.gap(state.problem, state.solution, rhs=rhs_full)
    tot_pts = (pw.expand_to_points(pre)
               + pw.tractions(state.solution["stress"]))
    y_pts = np.concatenate([mesh.geometry.quadrature(1, int(f))[0][:, 1]
                            for f in fault])
    slip_pts = np.abs(gap_pts[:, 1])
    mu_p = np.maximum(law.dynamic, law.static
                      - (law.static - law.dynamic) * slip_pts / law.critical)
    sc_pts = np.abs(tot_pts[:, 1]) + mu_p * tot_pts[:, 0]
    op = np.argsort(y_pts)
    # post-slip Coulomb function with the slip-weakening coefficient (Fig. 14
    # left): total effective traction = in-situ prestress + solved increment
    total = driver.per_facet(
        driver.prestress + driver.tractions(state.solution["stress"])
    )
    mu_fac = np.maximum(
        law.dynamic,
        law.static - (law.static - law.dynamic) * np.abs(slip) / law.critical,
    )
    coulomb = np.abs(total[:, 1]) + mu_fac * total[:, 0]
    return {
        "y": y[order], "slip": slip[order], "coulomb": coulomb[order],
        "y_pts": y_pts[op], "slip_pts": slip_pts[op],
        "coulomb_pts": sc_pts[op],
        "state": state, "built": built, "mu": mu_pts,
        "converged": converged,
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
    parser.add_argument("--refine", type=float, default=0.02,
                        help="bisect the nucleation bracket down to this "
                             "width in MPa (0 disables); Fig. 14 is drawn "
                             "at the deepest converged level")
    parser.add_argument("--mu-s", type=float, default=0.52)
    parser.add_argument("--mu-d", type=float, default=0.20)
    parser.add_argument("--delta-c", type=float, default=0.02)
    parser.add_argument("--figure",
                        default="benchmarks/contact_mechanics/benchmark_3.png")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cascade", action="store_true",
                        help="also write the continuation-cascade diagnostic "
                             "figure (not a paper figure)")
    arguments = parser.parse_args()

    parameters = wide_parameters()
    law = SlipWeakening(arguments.mu_s, arguments.mu_d, arguments.delta_c)
    print("Benchmark 3 -- inclined displaced fault, slip-weakening friction\n")
    print(f"  mu_s = {law.static:g}, mu_d = {law.dynamic:g}, "
          f"delta_c = {law.critical:g} m")
    print("  paper: p* = -17.41 MPa (semi-analytical), "
          "-17.27 MPa (DARTS)\n")

    built, last, nucleated, nucleation_res = None, None, None, None
    mu0, profiles = None, []
    for level in np.arange(arguments.start, arguments.stop - 1e-9,
                           -abs(arguments.step)):
        stage = replace(parameters, depletion=level * 1e6)
        res = simulate(stage, law, spacing=arguments.spacing, built=built,
                       mu0=mu0)
        built = res["built"]
        status = ("converged" if res["converged"] else "NO EQUILIBRIUM")
        print(f"  p = {level:7.2f} MPa   peak |slip| = "
              f"{res['peak'] * 1e3:7.3f} mm   {status}")
        profiles.append((level, res["y"], np.abs(res["slip"])))
        if not res["converged"] or res["peak"] > 50 * law.critical:
            nucleated, nucleation_res = level, res
            break
        last = (level, res)
        mu0 = res["mu"]  # continue the friction field along the branch

    if nucleated is not None and last is not None and arguments.refine > 0:
        # bisect toward the fold: each probe rides the cached factorization,
        # and the deepest converged state is the honest pre-nucleation one
        lo, hi = nucleated, last[0]
        mu0 = last[1]["mu"]
        while hi - lo > arguments.refine + 1e-12:
            mid = 0.5 * (lo + hi)
            stage = replace(parameters, depletion=mid * 1e6)
            res = simulate(stage, law, spacing=arguments.spacing, built=built,
                           mu0=mu0)
            status = ("converged" if res["converged"] else "NO EQUILIBRIUM")
            print(f"  p = {mid:8.3f} MPa   peak |slip| = "
                  f"{res['peak'] * 1e3:7.3f} mm   {status}   (bisection)")
            if res["converged"]:
                hi, last, mu0 = mid, (mid, res), res["mu"]
                profiles.append((mid, res["y"], np.abs(res["slip"])))
            else:
                lo = mid
        nucleated = lo

    if nucleated is not None:
        upper = f"{last[0]:.3f}" if last is not None else f"> {nucleated:.3f}"
        print(f"\n  nucleation bracketed: p* in ({nucleated:.3f}, {upper}) MPa")
        print("  (paper: -17.41 MPa semi-analytical, -17.27 MPa DARTS; on "
              "coarse fault spacings p* shifts to the mesh-dependent slip "
              "onset -- see the docstring)")
    elif last is not None:
        print("\n  no nucleation down to the last level -- extend --stop")

    ref = reference_pre_nucleation()
    if ref is not None and last is not None:
        yr, dr = ref
        ours = np.interp(yr, last[1]["y_pts"], last[1]["slip_pts"])
        sel = np.abs(yr) <= 100.0
        rms = np.sqrt(np.mean((ours[sel] - np.abs(dr[sel])) ** 2))
        print(f"\n  last equilibrium (p = {last[0]:.2f} MPa) vs 4TU Fig. 14 "
              f"(paper's pre-nucleation state at p = -17.41 MPa):")
        print(f"    peaks {np.abs(last[1]['slip']).max() * 1e3:.2f} / "
              f"{np.abs(dr).max() * 1e3:.2f} mm, rms = {rms * 1e3:.2f} mm")
        print("    (different pressures -- the comparison is of the "
              "pre-nucleation *state*, not pointwise equality)")

    stem, dot, ext = arguments.figure.rpartition(".")
    base = stem if dot else arguments.figure
    suffix = f"{dot}{ext}" if dot else ".png"

    if last is not None and arguments.figure and not arguments.no_plots:
        # paper Fig. 14: post-slip Coulomb stress (left) and slip (right) in
        # the upper patch at the last quasi-static equilibrium
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        level, res = last
        fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(9.0, 6.0),
                                                sharey=True)
        # the law constrains the facet-*mean* Coulomb stress, so that is the
        # enforced quantity to plot; the slip is a kinematic field whose
        # per-facet linear (BDM) variation is genuine resolution
        selc = (res["y"] >= 60.0) & (res["y"] <= 80.0)
        left_ax.plot(res["coulomb"][selc] / 1e6, res["y"][selc], lw=1.5,
                     label="mimetic-AFW-BDM")
        sel = (res["y_pts"] >= 60.0) & (res["y_pts"] <= 80.0)
        right_ax.plot(res["slip_pts"][sel] * 1e3, res["y_pts"][sel], lw=1.5)
        if REFERENCE_DATA.exists():
            rl = load_reference("14 left")
            rr = load_reference("14 right")
            for ax, (yy, vv), scale in ((left_ax, (rl["y"], rl["Sigma_C_post"]),
                                         1e-6),
                                        (right_ax, (rr["y"], rr["delta"]),
                                         1e3)):
                keep = (yy >= 60.0) & (yy <= 80.0)
                ax.plot(vv[keep][::12] * scale, yy[keep][::12], "ro", ms=3.5,
                        mfc="white", mew=0.9, lw=0)
            left_ax.plot([], [], "ro", ms=3.5, mfc="white", mew=0.9, lw=0,
                         label="semi-analytical (4TU)")
        for ax in (left_ax, right_ax):
            ax.axhline(parameters.fault_a, color="k", lw=0.6, ls=":")
            ax.grid(alpha=0.25)
        left_ax.axvline(0.0, color="k", lw=0.6)
        left_ax.set_xlabel(r"$\Sigma_C$   (MPa)")
        left_ax.set_ylabel("y   (m)")
        left_ax.set_ylim(60, 80)
        right_ax.set_xlabel(r"$|\delta|$   (mm)")
        left_ax.legend(fancybox=True, framealpha=0.9, edgecolor="0.8",
                       fontsize=8.5)
        fig.suptitle("Benchmark 3 -- slip-weakening, last equilibrium at "
                     f"p = {level:g} MPa   (Novikov et al. 2024, Fig. 14, "
                     "paper p* = -17.41 MPa)", fontsize=10)
        fig.tight_layout()
        path = f"{base}_fig14{suffix}"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")

    if (len(profiles) > 1 and arguments.cascade
            and arguments.figure and not arguments.no_plots):
        # the continuation: slip profiles level by level up to (and including)
        # the nucleated state, against the semi-analytical pre-nucleation one
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 6.0))
        for level, yy, ss in profiles:
            sel = np.abs(yy) < 150.0
            runaway = nucleated is not None and level == nucleated
            ax.plot(np.abs(ss[sel]) * 1e3, yy[sel],
                    lw=2.0 if runaway else 1.2,
                    color="crimson" if runaway else None,
                    label=f"p = {level:g} MPa"
                          + (" (no equilibrium)" if runaway else ""))
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
        ax.set_title("Benchmark 3 -- continuation to nucleation\n"
                     "(coupled slip-weakening, warm-started levels)",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=7)
        fig.tight_layout()
        path = f"{base}_cascade{suffix}"
        fig.savefig(path, dpi=150)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
