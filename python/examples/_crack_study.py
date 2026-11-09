#!/usr/bin/env python3
r"""The four-level refinement of the single-crack benchmark, both symmetry families.

Produces two figures:

  crack_levels.png        one row per refinement -- the mesh at the crack, the
                          normal traction, and the tangential sliding -- with
                          both constructions on each row against eq. (34).
  crack_convergence.png   the two error norms against the crack element size,
                          with least-squares rates.

ONLY THE CRACK IS REFINED. The far field is held at a fixed coarse size, so the
cell count grows with the crack's resolution and not with the domain's: the
ladder below runs 25 -> 200 crack faces for a few thousand cells, where a
uniform mesh would need millions. The domain is held large enough that the
O((2b/L)^2) far-field truncation sits below the discretization error at every
level -- otherwise the finest level measures the truncation floor and the fitted
rate is a property of the box.
"""

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import mimetika_cxx as mk

import numpy as np

import crack_compression as C

PRODUCTS = ("stabilized_bdm", "stabilized_vem")
STYLE = {
    "stabilized_bdm": ("tab:blue", "o", "weak symmetry (stabilized_bdm)"),
    "stabilized_vem": ("tab:red", "s", "strong symmetry (stabilized_vem)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="0.08,0.04,0.02,0.01",
                    help="crack element sizes, coarse to fine")
    ap.add_argument("--domain", type=float, default=60.0)
    ap.add_argument("--size-far", type=float, default=3.0)
    ap.add_argument("--refine-radius", type=float, default=0.4,
                    help="hold the crack size within this distance of the crack, "
                         "so a NEIGHBOURHOOD refines and not just the crack line")
    ap.add_argument("--grading", type=float, default=2.0)
    ap.add_argument("--psi", type=float, default=20.0)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--trim", type=float, default=0.10,
                    help="fraction of the half-length dropped at EACH tip; 0.10 is\n                         the paper's central 90%")
    ap.add_argument("--trim-sweep", default="0.0,0.05,0.10,0.20",
                    help="also report the fitted rates at these trims, to show how\n                         much of a rate is the tip singularity")
    ap.add_argument("--out", default="crack")
    opt = ap.parse_args()

    args = argparse.Namespace(
        mesh="gmsh", mesh_file=None, half_length=1.0, domain=opt.domain, psi=opt.psi,
        theta=opt.theta, size_far=opt.size_far, grading=opt.grading,
        refine_radius=opt.refine_radius, young=25000.0,
        poisson=0.25, sigma=100.0, c_n=1e9, c_t=1e-3, trim=opt.trim, vtu="")
    levels = [float(v) for v in opt.levels.split(",")]
    b = args.half_length

    t_ex, g_ex_peak = C.analytic(np.array([b]), b, args.young, args.poisson,
                                 args.sigma, args.psi, args.theta)
    print("single crack under compression -- CMAME 368 (2020) 113161, Section 5.2")
    print(f"  domain {args.domain:g} m = {args.domain/(2*b):.0f} crack lengths, "
          f"far-field truncation ~ {(2*b/args.domain)**2:.2%}")
    print(f"  the crack size is held within {args.refine_radius:g} m of the crack, then "
          f"graded to {args.size_far:g} m over {args.grading:g} m")
    print(f"  exact: t_N = {float(t_ex):.4f} MPa, peak g_T = {float(g_ex_peak[0]):.4e} m\n")
    print(f"  {'h':>7}{'cells':>8}{'faces':>7}  {'product':<17}"
          f"{'t_N':>10}{'g_T peak':>12}{'err t_N':>10}{'err g_T':>10}")

    rows, curves, meshes, profiles = [], {}, {}, {}
    for h in levels:
        for product in PRODUCTS:
            mesh, model, n, t, *_ = C.build(product, h, args)
            model.solve(progress=False, options=mk.SolverOptions())
            xi, t_n, t_t, g_t, g_n, _ = C.read_crack(model, n, t)
            e_t, e_g, kept = C.compare(xi, t_n, g_t, args)
            curves[(h, product)] = (xi, t_n, g_t)
            profiles.setdefault(product, {})[h] = (xi.tolist(), t_n.tolist(), g_t.tolist())
            meshes[h] = C.__dict__  # placeholder, replaced below
            rows.append(dict(h=h, product=product, cells=int(mesh.count(3)),
                             faces=int(model.n_fracture_faces), kept=kept,
                             n_dofs=int(model.n_dofs),
                             t_N=float(t_n.mean()), g_T=float(np.abs(g_t).max()),
                             g_N=float(np.abs(g_n).max()), t_T=float(np.abs(t_t).max()),
                             err_t_N=float(e_t), err_g_T=float(e_g)))
            r = rows[-1]
            print(f"  {h:>7g}{r['cells']:>8}{r['faces']:>7}  {product:<17}"
                  f"{r['t_N']:>10.4f}{r['g_T']:>12.4e}{r['err_t_N']:>10.2e}"
                  f"{r['err_g_T']:>10.2e}")

    print(f"  {'':>22}  {'exact (eq. 34)':<17}{float(t_ex):>10.4f}"
          f"{float(g_ex_peak[0]):>12.4e}")
    print("\n  least-squares rates against h:")
    rates = {}
    for product in PRODUCTS:
        mine = [r for r in rows if r["product"] == product]
        hs = [r["h"] for r in mine]
        rates[product] = (C.rate(hs, [r["err_t_N"] for r in mine]),
                          C.rate(hs, [r["err_g_T"] for r in mine]))
        print(f"    {product:<17} m(t_N) = {rates[product][0]:5.2f}   "
              f"m(g_T) = {rates[product][1]:5.2f}    (paper: 1.11 and 1.39)")

    print("\n  THE TIPS ARE SINGULAR, so the fitted rate depends on how much of them is kept:")
    print(f"    {'kept':>8}  {'stabilized_bdm':>26}  {'stabilized_vem':>26}")
    print(f"    {'':>8}  {'m(t_N)':>12}{'m(g_T)':>14}  {'m(t_N)':>12}{'m(g_T)':>14}")
    for tr in [float(v) for v in opt.trim_sweep.split(",")]:
        cells = []
        for product in PRODUCTS:
            hs, et, eg = [], [], []
            for h in levels:
                xi, t_n, g_t = (np.asarray(v) for v in profiles[product][h])
                a, b_, _ = C.compare(xi, t_n, g_t, args, trim=tr)
                hs.append(h); et.append(a); eg.append(b_)
            cells.append((C.rate(hs, et), C.rate(hs, eg)))
        print(f"    {1 - tr:>7.0%}  {cells[0][0]:>12.2f}{cells[0][1]:>14.2f}"
              f"  {cells[1][0]:>12.2f}{cells[1][1]:>14.2f}"
              + ("   <- the paper's window" if abs(tr - 0.10) < 1e-12 else ""))

    Path(f"{opt.out}_levels.json").write_text(json.dumps(
        {"setup": vars(args), "levels": levels, "rows": rows,
         "rates": {k: list(v) for k, v in rates.items()},
         "profiles": profiles}, indent=2))

    draw_levels(f"{opt.out}_levels.png", levels, curves, args)
    draw_convergence(f"{opt.out}_convergence.png", levels, rows, rates, args)
    print(f"\n  wrote {opt.out}_levels.png, {opt.out}_convergence.png, {opt.out}_levels.json")


def draw_levels(path, levels, curves, args):
    """One row per refinement: the mesh at the crack, t_N, and g_T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import meshio

    b, psi = args.half_length, math.radians(args.psi)
    fine = np.linspace(0.0, 2 * b, 400)
    t_ex, g_ex = C.analytic(fine, b, args.young, args.poisson,
                            args.sigma, args.psi, args.theta)

    fig, ax = plt.subplots(len(levels), 3, figsize=(14.0, 3.5 * len(levels)),
                           gridspec_kw={"width_ratios": [1.0, 1.3, 1.3]})
    for i, h in enumerate(levels):
        path_i = (f"/tmp/crack_psi{args.psi:g}_b{b:g}_L{args.domain:g}"
                  f"_h{h:g}_f{args.size_far:g}_r{args.refine_radius:g}.vtu")
        m = meshio.read(path_i)
        p, tri = m.points, m.cells[0].data[:, :3]
        a = ax[i, 0]
        a.triplot(p[:, 0], p[:, 1], tri, lw=0.2, color="0.62")
        a.plot([-b * math.cos(psi), b * math.cos(psi)],
               [-b * math.sin(psi), b * math.sin(psi)],
               color="crimson", lw=2.2, solid_capstyle="butt", zorder=5)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.0, 1.0); a.set_aspect("equal")
        n_faces = len(curves[(h, "stabilized_bdm")][0])
        a.set_title(f"h = {h:g} m — {len(m.cells[0].data)} prisms, {n_faces} crack faces",
                    fontsize=9.5)
        a.set_ylabel("y [m]"); a.tick_params(labelsize=8)
        if i == len(levels) - 1:
            a.set_xlabel("x [m]")

        for j, (exact, pick, ylab, ttl) in enumerate((
                (t_ex, lambda c: c[1], r"$t_N$ [MPa]", "normal traction"),
                (g_ex, lambda c: np.abs(c[2]), r"$\|g_T\|$ [m]", "tangential sliding"))):
            a = ax[i, j + 1]
            if j == 0:
                a.axhline(float(np.atleast_1d(exact)[0]), color="k", lw=1.4,
                          label="eq. (34)", zorder=4)
            else:
                a.plot(fine - b, exact, color="k", lw=1.4, label="eq. (34)", zorder=4)
            for product, (col, mkr, lab) in STYLE.items():
                c = curves[(h, product)]
                a.plot(c[0], pick(c), mkr, color=col, ms=2.2, lw=0.6, ls="-",
                       label=lab, alpha=0.85)
            a.axvspan(-b, -b * (1 - args.trim), color="0.92", zorder=0)
            a.axvspan(b * (1 - args.trim), b, color="0.92", zorder=0)
            a.set_ylabel(ylab); a.grid(alpha=0.3); a.tick_params(labelsize=8)
            if j == 0:
                a.set_ylim(-22, -2)
            else:
                a.set_ylim(0, 1.25 * max(g_ex.max(),
                                         max(np.abs(curves[(h, p)][2]).max()
                                             for p in STYLE)))
            if i == 0:
                a.set_title(ttl, fontsize=10)
                a.legend(fontsize=7.5, loc="lower center" if j == 0 else "upper right")
            if i == len(levels) - 1:
                a.set_xlabel(r"$\xi - b$ along the crack [m]")

    fig.suptitle("single crack under compression, CMAME 368 (2020) §5.2 — four refinements of the "
                 "crack, both symmetry constructions on each mesh\n"
                 fr"$\psi={args.psi:g}^\circ$, $\theta={args.theta:g}^\circ$, "
                 fr"$\bar\sigma={args.sigma:g}$ MPa, domain ${args.domain/(2*b):.0f}\times 2b$   "
                 "(shaded: the 5% tips the error norms drop)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=140)


def draw_convergence(path, levels, rows, rates, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for j, (key, ttl, paper) in enumerate((
            ("err_t_N", r"normal traction  $\|t_N - t_N^h\|_2 / \|t_N\|_2$", 1.11),
            ("err_g_T", r"tangential sliding  $\|g_T - g_T^h\|_2 / \|g_T\|_2$", 1.39))):
        a = ax[j]
        for product, (col, mkr, lab) in STYLE.items():
            mine = [r for r in rows if r["product"] == product]
            hs = np.array([r["h"] for r in mine])
            es = np.array([r[key] for r in mine])
            a.loglog(hs, es, mkr + "-", color=col, ms=6, lw=1.2,
                     label=f"{lab}   m = {rates[product][j]:.2f}")
        h0 = np.array(levels, float)
        ref = (h0 / h0[0]) ** paper
        a.loglog(h0, ref * max(r[key] for r in rows if r["h"] == h0[0]), "k--", lw=1.0,
                 label=f"paper's rate  m = {paper}")
        a.set_xlabel("crack element size h [m]"); a.set_ylabel("relative error")
        a.set_title(ttl, fontsize=10); a.grid(which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("single crack under compression — convergence on the central 90% of the trace",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
