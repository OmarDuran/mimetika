#!/usr/bin/env python3
r"""A single crack under far-field compression, against the closed-form solution.

THE BENCHMARK. Section 5.2 of Franceschini, Castelletto, White & Tchelepi,
*Algebraically stabilized Lagrange multiplier method for frictional contact
mechanics with hydraulically active fractures*, CMAME 368 (2020) 113161: a
straight crack of length 2b in an infinite plane under uniaxial compression
sigma_bar, the crack inclined at psi to the loading axis. With zero cohesion the
closed form is their eq. (34),

    t_N     = -sigma_bar sin^2(psi)
    g_T(xi) = 4(1 - nu^2)/E . sigma_bar sin(psi) (cos(psi) - sin(psi) tan(theta))
              . sqrt(b^2 - (b - xi)^2)                              0 <= xi <= 2b

-- a CONSTANT normal traction and an ELLIPTIC sliding profile, zero at both tips.

WHY theta = 0 MAKES THIS A LINEAR PROBLEM, and why that is the point. Coulomb
caps the shear at tau_max = c - t_N tan(theta); at c = 0 and theta = 0 the cap is
zero, so the crack carries no shear at all and slides freely while staying
closed. The contact state is then known in advance -- g_N = 0, t_T = 0 -- and
there is no active set to iterate for. One linear solve reproduces the whole
benchmark, which makes it a test of the DISCRETIZATION rather than of a
nonlinear solver. mimetika reaches that state with the linear Martins-Oden law
alone: a stiff normal direction and a soft tangential one.

    model.linear_contact(c_n, c_T)      D_c = (1/c_n) n(x)n + (1/c_T) (I - n(x)n)

THE ARGUMENTS ARE STIFFNESSES, whatever the symbol says: the measured ratio of
traction to jump is c_n normally and c_T tangentially. So the frictionless
closed limit is c_n LARGE and c_T SMALL, and the defaults below are the gentlest
pair that reaches it -- 1e9 and 1e-3 already give t_N to six figures and
g_N ~ 1e-8 m, with none of the conditioning cost of pushing further.

THE MESH IS GENERATED, not structured (see _crack_mesh.py). gmsh meshes the
square with the crack EMBEDDED as an internal curve, graded about it, and the
triangulation is extruded one layer into prisms. Two reasons, and the second
matters more than the first:

  * RESOLUTION. A structured box ties the crack's resolution to the domain's, so
    the paper's 108-432 interface elements would cost a million cells. Here
    --size-crack sets the crack's element size directly.
  * THE STRUCTURED CASE HIDES BUGS. An axis-aligned crack in a box is mirror
    symmetric about its own plane: the facet normals do not alternate and the
    per-facet right-hand-side coefficients cancel in pairs. Both are properties
    this code has previously been wrong about and passed anyway. An inclined
    crack through an unstructured grading has neither.

--mesh box keeps the structured route, with the crack on y = 0 and the LOAD
rotated by psi instead, which is the same boundary value problem in a rotated
frame. It is the cheap cross-check, not the benchmark.

WHY 3D. The strong-symmetry family carries the d(d+1)/2 rigid-motion moments of
the traction on a facet, which is a 3D construction -- six in space, and the
plane does not have them. The paper extrudes its own 2D domain for the same
reason. One layer of prisms, with sigma_zz = nu(sigma_xx + sigma_yy), is exactly
plane strain: eps_zz comes out identically zero, so the condition is satisfied
rather than imposed.

THE BOUNDARY, which is where this is easy to get wrong:

  * the four LATERAL faces carry the far field as the affine datum
    u = eps_inf . x. It removes every rigid mode (a pure-traction box is
    singular, and in Hellinger-Reissner the displacement is a cell unknown, so
    there is no nodal pin to fall back on) and both families reproduce an affine
    field exactly, so the datum contributes no error of its own.
  * the two Z-FACES ARE NOT BOUNDARY, they are the slab. They carry free slip --
    the plane-strain symmetry condition -- and nothing else. Handing them the
    datum as well clamps the interior of the slab to the unperturbed field and
    the crack can no longer slide: measured peak g_T 6.4e-04 m against the exact
    4.7e-03, a factor of 7, while t_N stays right to five figures because the
    normal traction is statically determinate and does not notice. A plausible,
    converged, wrong answer.

What the datum DOES contribute is TRUNCATION: the crack perturbs the stress by
O((b/r)^2), so a box of half-width L/2 carries an O((2b/L)^2) error that belongs
to the DOMAIN and not to the mesh. At the default L = 20b that is about 1%, and
it does not fall under refinement -- which is why --domain is a convergence knob
as much as --size-crack is, and why the two are reported apart.

Run it:

    PYTHONPATH=.. python crack_compression.py

which meshes the crack, solves one weak-symmetry and one strong-symmetry
product, and prints the readings against eq. (34). Then:

    PYTHONPATH=.. python crack_compression.py --product stabilized_vem
    PYTHONPATH=.. python crack_compression.py --size-crack 0.02    # ~100 faces
    PYTHONPATH=.. python crack_compression.py --levels 0.1,0.05,0.025
    PYTHONPATH=.. python crack_compression.py --mesh box --cells 40
    PYTHONPATH=.. python crack_compression.py --theta 30           # the paper's friction
    PYTHONPATH=.. python crack_compression.py --domain 40          # a quieter far field
    PYTHONPATH=.. python crack_compression.py --vtu out            # the crack, for ParaView

THE ERROR IS MEASURED ON THE CENTRAL 90% of the crack trace, as the paper does:
the traction oscillates near the tips, where the discrete crack cannot open
against a fixed tip node, and a norm over the whole trace measures that artefact
instead of the method. The excluded fraction is reported, and --trim 0 turns it
off.
"""

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401  -- BEFORE mimetika_cxx; see python/_shadowing.py
import mimetika_cxx as mk

import numpy as np

import _crack_mesh
import _realizations
from _stages import stage

DIM = 3

#: the paper's material and loading (Section 5.2), in MPa and metres
E_DEFAULT = 25000.0
NU_DEFAULT = 0.25
SIGMA_DEFAULT = 100.0
PSI_DEFAULT = 20.0
B_DEFAULT = 1.0

#: one of each family, which is the comparison this example is for
DEFAULT_PRODUCTS = ("stabilized_bdm", "stabilized_vem")


# ---- the closed form ---------------------------------------------------------
def analytic(xi, b, E, nu, sigma, psi_deg, theta_deg):
    """eq. (34): (t_N, g_T) at arclength xi along the crack, 0 <= xi <= 2b."""
    psi, theta = math.radians(psi_deg), math.radians(theta_deg)
    t_n = -sigma * math.sin(psi) ** 2
    amp = (4.0 * (1.0 - nu ** 2) / E) * (
        sigma * math.sin(psi) * (math.cos(psi) - math.sin(psi) * math.tan(theta)))
    under = np.maximum(b * b - (b - np.asarray(xi, float)) ** 2, 0.0)
    return t_n, amp * np.sqrt(under)


def far_field(E, nu, sigma, psi_deg, tilt_crack):
    """(sigma_inf, eps_inf, crack normal, crack tangent) as 3x3 and 3-vectors.

    TWO WAYS TO THE SAME STATE. With the crack tilted (the generated mesh) the
    load is the plain uniaxial sigma = -sigma_bar e_x (x) e_x and the crack
    carries the angle. With the crack on y = 0 (the structured box) the crack is
    axis-aligned and the LOAD carries it, sigma = -sigma_bar d (x) d with
    d = (cos psi, -sin psi, 0). Either way n . sigma n = -sigma_bar sin^2 psi.

    Both are closed in z by sigma_zz = nu (sigma_xx + sigma_yy); the isotropic
    inverse eps = (1+nu)/E sigma - (nu/E) tr(sigma) I then has eps_zz = 0
    identically, which IS the plane-strain condition.
    """
    psi = math.radians(psi_deg)
    c, s = math.cos(psi), math.sin(psi)
    if tilt_crack:
        sig = np.diag([-sigma, 0.0, 0.0])
        normal, tangent = np.array([-s, c, 0.0]), np.array([c, s, 0.0])
    else:
        d = np.array([c, -s, 0.0])
        sig = -sigma * np.outer(d, d)
        normal, tangent = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    sig[2, 2] = nu * (sig[0, 0] + sig[1, 1])
    eps = (1.0 + nu) / E * sig - (nu / E) * np.trace(sig) * np.eye(3)
    return sig, eps, normal, tangent


# ---- the model ---------------------------------------------------------------
def build(product, level, args):
    """The mesh, the crack, the far-field datum and the contact law."""
    b = args.half_length
    if args.mesh == "gmsh":
        # the cache key must carry EVERY geometric argument, or a run that
        # changes only --domain silently reuses the previous box
        path = args.mesh_file or (
            f"/tmp/crack_psi{args.psi:g}_b{args.half_length:g}_L{args.domain:g}"
            f"_h{level:g}_f{args.size_far:g}_r{getattr(args,'refine_radius',0.0):g}.vtu")
        if args.mesh_file is None or not Path(path).exists():
            _crack_mesh.inclined_crack(
                path, half_length=b, domain=args.domain, psi=args.psi,
                size_crack=level, size_far=args.size_far, grading=args.grading,
                refine_radius=getattr(args, "refine_radius", 0.0))
        mesh = mk.read_vtu(path)
        sets = mk.fracture_node_sets(path)
        if "f1" not in sets:
            raise SystemExit(f"{path} carries no f1 node set: {', '.join(sets) or 'none'}")
        node_sets, thickness = [list(sets["f1"])], level
    else:
        cells = int(level)
        h = args.domain / cells
        if abs(b / h - round(b / h)) > 1e-9:
            raise SystemExit(f"--mesh box needs the crack tip on a node: b = {b} is not a "
                             f"multiple of h = {h:.6g}; pick cells so cells*{b}/{args.domain} "
                             f"is a whole number")
        if cells % 2:
            raise SystemExit("--cells must be even, or y = 0 is not a node plane")
        mesh = mk.box([cells, cells, 1], DIM, mk.Family.cartesian,
                      [args.domain, args.domain, h], [-args.domain / 2, -args.domain / 2, 0.0])
        pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
        node_sets = [[int(v) for v in np.flatnonzero(
            (np.abs(pts[:, 1]) < 1e-9) & (np.abs(pts[:, 0]) <= b + 1e-9))]]
        thickness = h

    how, form = _realizations.resolve(product)
    mu = args.young / (2.0 * (1.0 + args.poisson))
    lam = args.young * args.poisson / ((1.0 + args.poisson) * (1.0 - 2.0 * args.poisson))
    model = mk.CauchyMechanicsModel(mesh, DIM, mk.ElasticMaterial(mu, lam), how, form)
    model.fracture(node_sets, [1])
    model.linear_contact(args.c_n, args.c_t)

    _, eps, normal, tangent = far_field(args.young, args.poisson, args.sigma,
                                        args.psi, args.mesh == "gmsh")
    grad = [float(v) for v in eps.reshape(9)]
    slab, lateral = [], []
    for f in model.outer_boundary_facets():
        x = mk.centroid(mesh, DIM - 1, f)
        (slab if min(abs(x[2]), abs(x[2] - thickness)) < 1e-9 * max(1.0, thickness)
         else lateral).append(f)
    model.add_free_slip(slab)
    # ONE CALL PER FACET: the datum is u = a + B (x - x_E) about the facet's
    # cofacet centroid, so the constant is u(x_E) and not u(0). A single call
    # over the whole list is accepted in silence and is wrong by O(h).
    for f in lateral:
        x_e = mk.centroid(mesh, DIM, mk.cofacet_of(mesh, DIM, f))
        model.prescribe_displacement([f], [float(v) for v in eps @ np.array(x_e)], grad)
    return mesh, model, normal, tangent, len(lateral), len(slab)


# ---- reading the crack back --------------------------------------------------
def read_crack(model, normal, tangent):
    """(xi, t_N, t_T, g_N, g_T) per crack face, ordered along the crack.

    A facet's canonical normal belongs to that facet alone, so across one
    fracture they alternate -- and a flip swaps which side is "plus", so the
    jump's AMBIENT direction flips with it. t_N and g_N survive (quadratic in n,
    or invariant because the sides swap too), but a tangential profile read
    straight out of the per-face frame alternates in sign. On a generated mesh
    that is not hypothetical: the normals genuinely alternate there, where on a
    structured axis-aligned crack they do not. So the readings are carried back
    into ambient axes and oriented ONCE, against the crack's own normal.
    """
    f = model.fracture_faces()
    n, t1, t2 = f["normal"], f["tangent1"], f["tangent2"]
    jump, trac = f["jump"], f["traction"]

    side = np.sign(n @ normal)
    side[side == 0.0] = 1.0
    amb_j = (jump[:, 0:1] * n + jump[:, 1:2] * t1 + jump[:, 2:3] * t2) * side[:, None]
    amb_t = (trac[:, 0:1] * n + trac[:, 1:2] * t1 + trac[:, 2:3] * t2) * side[:, None]

    xi = f["centroid"] @ tangent
    order = np.argsort(xi)
    return (xi[order], trac[order, 0], amb_t[order] @ tangent,
            amb_j[order] @ tangent, amb_j[order] @ normal, f["area"][order])


def compare(xi, t_n, g_t, args, trim=None):
    """Relative L2 errors against eq. (34), away from the tips.

    THE TIPS ARE EXCLUDED BECAUSE THE EXACT SOLUTION IS SINGULAR THERE. g_T goes
    like sqrt(b^2 - (b - xi)^2), whose derivative blows up at both ends, and the
    traction carries the r^-1/2 crack-tip field. An L2 norm taken over the whole
    trace is then dominated by how well a given mesh resolves a singularity --
    which is a statement about the mesh at the tip, not about the method, and it
    is why a rate measured that way comes out flat however good the scheme is.
    `trim` is the fraction of the half-length dropped at EACH tip, so 0.10 keeps
    the central 90% of the trace, as the paper does.
    """
    b = args.half_length
    t = args.trim if trim is None else trim
    keep = np.abs(xi) <= b * (1.0 - t)
    if keep.sum() < 2:
        return float("nan"), float("nan"), int(keep.sum())
    t_ex, g_ex = analytic(xi[keep] + b, b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    e_t = np.linalg.norm(t_n[keep] - t_ex) / np.linalg.norm(np.full(keep.sum(), t_ex))
    e_g = np.linalg.norm(g_t[keep] - g_ex) / (np.linalg.norm(g_ex) or 1.0)
    return e_t, e_g, int(keep.sum())


def rate(hs, errs):
    """The least-squares slope of log(error) against log(h)."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    ok = (hs > 0) & (errs > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0])


# ---- one run -----------------------------------------------------------------
def run(product, level, args):
    mesh, model, normal, tangent, n_lat, n_slab = build(product, level, args)
    label = f"h={level:g}" if args.mesh == "gmsh" else f"{int(level)}^2"
    with stage(f"{product:<16s} {label:>9s}  {mesh.count(DIM)} cells, "
               f"{n_lat} datum + {n_slab} slip facets"):
        model.solve(progress=False, options=mk.SolverOptions())
    xi, t_n, t_t, g_t, g_n, area = read_crack(model, normal, tangent)

    b = args.half_length
    t_ex, g_ex = analytic(xi + b, b, args.young, args.poisson, args.sigma, args.psi, args.theta)
    e_t, e_g, kept = compare(xi, t_n, g_t, args)
    out = {
        "product": product, "level": level, "cells": int(mesh.count(DIM)),
        "n_dofs": model.n_dofs, "crack_faces": int(model.n_fracture_faces), "kept": kept,
        "t_N_mean": float(t_n.mean()), "t_N_exact": float(t_ex),
        "g_T_peak": float(np.abs(g_t).max()), "g_T_peak_exact": float(g_ex.max()),
        "g_N_max": float(np.abs(g_n).max()), "t_T_max": float(np.abs(t_t).max()),
        "err_t_N": float(e_t), "err_g_T": float(e_g),
    }
    if args.vtu:
        out["vtu"] = f"{args.vtu}_{product}_{level:g}.vtu"
        model.write_fracture_vtu(out["vtu"])
    return out, (xi, t_n, g_t)


def report(rows):
    print()
    print(f"  {'product':<17}{'cells':>8}{'faces':>7}{'t_N':>11}{'g_T peak':>12}"
          f"{'g_N':>10}{'t_T':>9}{'err t_N':>10}{'err g_T':>10}")
    for r in rows:
        print(f"  {r['product']:<17}{r['cells']:>8}{r['crack_faces']:>7}"
              f"{r['t_N_mean']:>11.4f}{r['g_T_peak']:>12.4e}"
              f"{r['g_N_max']:>10.1e}{r['t_T_max']:>9.1e}"
              f"{r['err_t_N']:>10.2e}{r['err_g_T']:>10.2e}")
    print(f"  {'exact (eq. 34)':<17}{'':>8}{'':>7}{rows[0]['t_N_exact']:>11.4f}"
          f"{rows[0]['g_T_peak_exact']:>12.4e}{0.0:>10.1e}{0.0:>9.1e}")


def draw(path, curves, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = args.half_length
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fine = np.linspace(0.0, 2 * b, 400)
    t_ex, g_ex = analytic(fine, b, args.young, args.poisson, args.sigma, args.psi, args.theta)
    ax[0].axhline(t_ex, color="k", lw=1.3, label="eq. (34)")
    ax[1].plot(fine - b, g_ex, color="k", lw=1.3, label="eq. (34)")
    for name, (xi, t_n, g_t) in curves.items():
        ax[0].plot(xi, t_n, "o-", ms=3, lw=0.8, label=name)
        ax[1].plot(xi, np.abs(g_t), "o-", ms=3, lw=0.8, label=name)
    ax[0].set_xlabel(r"$\xi - b$ along the crack [m]"); ax[0].set_ylabel(r"$t_N$ [MPa]")
    ax[1].set_xlabel(r"$\xi - b$ along the crack [m]"); ax[1].set_ylabel(r"$\|g_T\|$ [m]")
    for a in ax:
        a.axvspan(-b, -b * (1 - args.trim), color="0.9", zorder=0)
        a.axvspan(b * (1 - args.trim), b, color="0.9", zorder=0)
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(fr"single crack under compression: $\psi={args.psi}^\circ$, "
                 fr"$\theta={args.theta}^\circ$, $\bar\sigma={args.sigma}$ MPa, "
                 fr"domain ${args.domain/(2*b):.0f}\times 2b$   (shaded: trimmed tips)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default=",".join(DEFAULT_PRODUCTS),
                    help="comma-separated stress products; both symmetry families are "
                         f"available: {', '.join(_realizations.names())}. "
                         f"Default {','.join(DEFAULT_PRODUCTS)} -- one of each.")
    ap.add_argument("--mesh", default="gmsh", choices=("gmsh", "box"),
                    help="gmsh: the crack is inclined and embedded in a graded, unstructured "
                         "triangulation extruded one layer (the benchmark). box: the crack is "
                         "axis-aligned in a structured box and the LOAD is rotated instead "
                         "(the cheap cross-check, and a mirror-symmetric configuration that "
                         "has hidden contact bugs before).")
    ap.add_argument("--mesh-file", default=None,
                    help="read this .vtu instead of generating one; it must carry an "
                         "f1_node_set PointData flag on the crack nodes")
    ap.add_argument("--size-crack", type=float, default=0.05,
                    help="gmsh: element size ON the crack, so it carries about 2b/size faces")
    ap.add_argument("--size-far", type=float, default=1.5,
                    help="gmsh: element size at the far boundary")
    ap.add_argument("--refine-radius", type=float, default=0.4,
                    help="gmsh: the crack element size is held within this distance of the "
                         "crack before grading out. Refining a TUBE rather than the crack "
                         "line is what makes the cell count grow like h^-2 and lets a bulk "
                         "convergence rate be observed at all.")
    ap.add_argument("--grading", type=float, default=None,
                    help="gmsh: distance over which the size grows (default: one crack length)")
    ap.add_argument("--cells", type=int, default=40, help="box: cells across the domain")
    ap.add_argument("--levels", default="",
                    help="a comma-separated ladder for a convergence study -- --size-crack "
                         "values under --mesh gmsh, --cells values under --mesh box")
    ap.add_argument("--domain", type=float, default=20.0,
                    help="box width in metres; the far field is imposed on its boundary, so "
                         "this sets the O((2b/L)^2) truncation floor the mesh cannot cross")
    ap.add_argument("--half-length", type=float, default=B_DEFAULT, help="b; the crack is 2b long")
    ap.add_argument("--psi", type=float, default=PSI_DEFAULT, help="crack inclination [deg]")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="friction angle [deg]. 0 is the frictionless limit this example "
                         "reproduces with a linear law; the paper's own case is 30, which "
                         "changes only the expected g_T amplitude here -- the LAW stays "
                         "linear, so a nonzero value compares against a different closed "
                         "form, it does not run a Coulomb solve.")
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT, help="sigma_bar [MPa]")
    ap.add_argument("--young", type=float, default=E_DEFAULT, help="E [MPa]")
    ap.add_argument("--poisson", type=float, default=NU_DEFAULT, help="nu")
    ap.add_argument("--c-n", type=float, default=1e9,
                    help="normal STIFFNESS: large closes the crack (g_N -> 0)")
    ap.add_argument("--c-t", type=float, default=1e-3,
                    help="tangential STIFFNESS: small frees the sliding (t_T -> 0)")
    ap.add_argument("--trim", type=float, default=0.10,
                    help="fraction of the HALF-length dropped at each tip before the error "
                         "norms, so 0.10 keeps the central 90%% of the trace, which is what "
                         "the paper does. THE EXACT SOLUTION IS SINGULAR AT THE TIPS -- g_T "
                         "has a square-root profile, so its slope is infinite there, and the "
                         "traction carries the r^-1/2 crack-tip field -- so a norm over the "
                         "whole trace measures how a fixed mesh resolves a singularity and "
                         "not how the method converges. --trim 0 turns it off, and the rate "
                         "it then reports is not a rate.")
    ap.add_argument("--plot", default="crack_compression.png", help="'' to skip")
    ap.add_argument("--json", default="crack_compression.json", help="'' to skip")
    ap.add_argument("--vtu", default="", help="prefix for per-run fracture .vtu files")
    args = ap.parse_args()

    products = [p.strip() for p in args.product.split(",") if p.strip()]
    for p in products:
        if p not in _realizations.names():
            raise SystemExit(f"unknown product {p!r}; choose from "
                             f"{', '.join(_realizations.names())}")
    default_level = args.size_crack if args.mesh == "gmsh" else args.cells
    ladder = ([float(v) for v in args.levels.split(",")] if args.levels else [default_level])

    b = args.half_length
    t_ex, g_ex = analytic(np.array([b]), b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    print("single crack under compression -- CMAME 368 (2020) 113161, Section 5.2")
    print(f"  E = {args.young} MPa, nu = {args.poisson}, sigma_bar = {args.sigma} MPa, "
          f"psi = {args.psi} deg, theta = {args.theta} deg, 2b = {2*b} m")
    print(f"  {'inclined crack, graded unstructured mesh' if args.mesh == 'gmsh' else
             'axis-aligned crack, structured box, load rotated'}")
    print(f"  domain {args.domain} m = {args.domain/(2*b):.0f} crack lengths "
          f"-> far-field truncation ~ {(2*b/args.domain)**2:.1%}")
    print(f"  exact: t_N = {t_ex:.4f} MPa, peak g_T = {g_ex[0]:.4e} m")
    print()

    rows, curves = [], {}
    for product in products:
        for level in ladder:
            out, curve = run(product, level, args)
            rows.append(out)
            curves[f"{product} ({out['crack_faces']} faces)"] = curve
    report(rows)

    if len(ladder) > 1:
        print()
        print("  convergence on the central trace:")
        for product in products:
            mine = [r for r in rows if r["product"] == product]
            hs = [r["level"] if args.mesh == "gmsh" else 1.0 / r["level"] for r in mine]
            print(f"    {product:<17} m(t_N) = {rate(hs, [r['err_t_N'] for r in mine]):5.2f}"
                  f"   m(g_T) = {rate(hs, [r['err_g_T'] for r in mine]):5.2f}"
                  f"   (paper: 1.11 and 1.39)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"setup": vars(args), "exact": {"t_N": float(t_ex), "g_T_peak": float(g_ex[0])},
             "runs": rows}, indent=2))
        print(f"\n  wrote {args.json}")
    if args.plot:
        draw(args.plot, curves, args)


if __name__ == "__main__":
    main()
