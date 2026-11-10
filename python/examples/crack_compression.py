#!/usr/bin/env python3
r"""A single crack under far-field compression, against the closed-form solution.

THE BENCHMARK. Section 5.2 of Franceschini, Castelletto, White & Tchelepi,
*Algebraically stabilized Lagrange multiplier method for frictional contact
mechanics with hydraulically active fractures*, CMAME 368 (2020) 113161: a
straight crack of length 2b in an infinite plane under uniaxial compression
sigma_bar, inclined at psi to the loading axis. With zero cohesion the closed
form is their eq. (34),

    t_N     = -sigma_bar sin^2(psi)
    g_T(xi) = 4(1 - nu^2)/E . sigma_bar sin(psi) (cos(psi) - sin(psi) tan(theta))
              . sqrt(b^2 - (b - xi)^2)                              0 <= xi <= 2b

-- a CONSTANT normal traction and an ELLIPTIC sliding profile, zero at both tips.

WHY theta = 0 MAKES THIS A LINEAR PROBLEM. Coulomb caps the shear at
tau_max = c - t_N tan(theta); at c = 0 and theta = 0 the cap is zero, so the
crack carries no shear and slides freely while staying closed. The contact state
is known in advance -- g_N = 0, t_T = 0 -- and there is no active set to iterate
for: one linear solve is the whole benchmark, which makes it a test of the
discretization and not of a nonlinear solver. The linear law reaches that state
with a stiff normal direction and a soft tangential one,

    model.linear_contact(c_n, c_T)        t_N = c_n g_N ,   t_T = c_T g_T

THE ARGUMENTS ARE STIFFNESSES: the measured ratio of traction to jump is c_n
normally and c_T tangentially, so the frictionless closed limit is c_n LARGE and
c_T SMALL. The defaults, 1e9 and 1e-3, already sit in it -- the sliding is
identical to five figures from c_T = 1e-2 down to 1e-6.

THE FORMULATION is the cut one: model.fracture() duplicates the fracture facets,
so the stress lives in H(div; Omega \ Gamma), and the fracture becomes a stratum
of its own carrying the two mortar traces of the displacement -- with their
rotations in the strong family, whose facet unknowns are the six rigid-motion
moments -- and the contact traction as a multiplier. The four readings drawn
here are that formulation's own unknowns. The uncut formulation, which replaces
the constitutive rows of interior facets (algebraic_constraints/contact), is not
exercised by this example.

WHY 3D. The strong-symmetry family is a 3D construction, so the plane problem is
solved on one layer of prisms, as the paper extrudes its own. With
sigma_zz = nu (sigma_xx + sigma_yy) the far field has eps_zz = 0 identically:
plane strain is satisfied, not imposed.

THE BOUNDARY, which is where this is easy to get wrong:

  * the four LATERAL faces carry the far field as the affine datum
    u = eps_inf . x. It removes every rigid mode -- in Hellinger-Reissner the
    displacement is a cell unknown, so there is no nodal pin to fall back on --
    and both families reproduce an affine field exactly.
  * the two Z-FACES ARE NOT BOUNDARY, they are the slab: free slip, the
    plane-strain symmetry condition, and nothing else. Given the datum as well,
    the slab is clamped to the unperturbed field and the crack cannot slide:
    measured g_T seven times too small, with t_N still right to five figures.

THE MESH (see _crack_mesh.py) embeds the crack in a graded triangulation, so it
is conformal across it, and tags the crack's nodes as f1_node_set. The crack
size is held within --refine-radius of the crack and released to --size-far
over --grading on a linear ramp.

THE LADDER IS A UNIFORM REFINEMENT: the first level is meshed and every later
one splits each triangle in four, so h halves exactly, the crack's face count
doubles exactly, and the grading pattern is the same at every level. That is
not a refinement of taste. A re-meshed ladder draws a new triangulation at the
tips each level, and that mesh-to-mesh variation is as large as the convergence
being measured: on one grading the same two families fitted 0.75 / 0.08 and
0.58 / 0.74 re-meshed against 1.51 / 0.45 and 0.99 / 1.21 uniformly refined.

THE ERROR IS TAKEN AWAY FROM THE TIPS, on the central 90% of the trace as the
paper does, because the exact solution is singular there: g_T has a square-root
profile, so its slope is unbounded at both ends, and the traction carries the
r^-1/2 crack-tip field. Over the whole trace the weak family's traction rate
reads 0.06; on the central 80% it reads 0.79. A norm that includes the tips
measures how a mesh resolves a singularity.

MEASURED, at the defaults (psi = 20 deg, three uniform refinements, 25 / 50 / 100
crack faces on 1592 / 6368 / 25472 prisms), means and errors on the central 90%:

                         t_N [MPa]    g_T peak [m]   err t_N    err g_T
    stabilized_bdm  25   -11.6830     4.8124e-03     1.29e-02   1.40e-03
     (weak)         50   -11.6942     4.8179e-03     4.59e-03   7.64e-04
                   100   -11.6997     4.8194e-03     1.22e-03   4.73e-04
    stabilized_vem  25   -11.7909     5.8532e-03     1.49e-01   2.74e-01
     (strong)       50   -11.6975     5.2195e-03     6.45e-02   1.09e-01
                   100   -11.6976     4.9787e-03     2.42e-02   4.51e-02
    exact                -11.6978     4.8209e-03

    rates           weak 1.71 (t_N), 0.79 (g_T)    strong 1.31, 1.30
    paper                1.11        1.39

Both constructions converge monotonically in both quantities. The strong family
matches the paper's rates from an error twenty to a hundred times the weak
family's; the weak sliding is within 0.2 percent from the coarsest mesh, which
is why its fitted rate is the least settled number here.

COST. The slab is quasi-two-dimensional, so the solve is MUMPS: the three levels
of both families take about 35 s and peak at 7 GB. One more uniform refinement
is 100k prisms and about 30 GB -- it is not a default.

Run it:

    PYTHONPATH=.. python crack_compression.py

which writes crack_uniform_refinement.png (one row per level: the mesh, then
t_N, g_N, t_T and g_T, every construction on each panel),
crack_uniform_refinement_convergence.png and crack_uniform_refinement.json. Then:

    PYTHONPATH=.. python crack_compression.py --product stabilized_vem
    PYTHONPATH=.. python crack_compression.py --product derham_bdm,stabilized_bdm
    PYTHONPATH=.. python crack_compression.py --levels 0.16,0.08,0.04    # cheaper
    PYTHONPATH=.. python crack_compression.py --recombine                # quadrilaterals
    PYTHONPATH=.. python crack_compression.py --no-uniform-refinement    # re-mesh each level
    PYTHONPATH=.. python crack_compression.py --mesh box --levels 40,80  # no gmsh
    PYTHONPATH=.. python crack_compression.py --redraw                   # figures only

--recombine has gmsh pair the triangles into quadrilaterals, so the slab is
hexahedra with the few prisms the pairing leaves, and the outputs take the
suffix _quads.

--mesh box lays the crack on y = 0 of a structured box and rotates the LOAD by
psi instead -- the same boundary value problem in a rotated frame, with no mesh
generator. It is the cheap cross-check: it ties the crack's resolution to the
domain's, and it is mirror symmetric about the crack plane, the configuration
in which alternating facet normals and per-facet right-hand sides cancel.
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
DEFAULT_PRODUCTS = ("stabilized_bdm", "stabilized_vem")  # one of each family
DEFAULT_LEVELS = {"gmsh": "0.08,0.04,0.02", "box": "40,80"}
COLORS = ("tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown")
MARKERS = ("o", "s", "^", "D", "v", "P")


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
    """(sigma_inf, eps_inf, crack normal, crack tangent).

    With the crack tilted (the generated mesh) the load is the plain uniaxial
    sigma = -sigma_bar e_x (x) e_x. With the crack on y = 0 (the box) the LOAD
    carries the angle, sigma = -sigma_bar d (x) d with d = (cos psi, -sin psi, 0).
    Either way n . sigma n = -sigma_bar sin^2 psi. Both are closed in z by
    sigma_zz = nu (sigma_xx + sigma_yy), for which the isotropic inverse
    eps = (1+nu)/E sigma - (nu/E) tr(sigma) I has eps_zz = 0.
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
def refinement_base(args):
    """The size the uniformly refined ladder is meshed at, or None when each level
    is re-meshed."""
    return args.levels[0] if args.mesh == "gmsh" and args.uniform_refinement else None


def mesh_path(args, level):
    """Where the generated mesh for this geometry and size field is cached.

    The key carries EVERY argument the mesh depends on -- a run that changes
    only --domain must not reuse the previous box -- and a tag for the size-field
    recipe, bumped when _crack_mesh changes what the same arguments produce.
    """
    return (f"/tmp/crack_v3_psi{args.psi:g}_b{args.half_length:g}_L{args.domain:g}"
            f"_h{level:g}_f{args.size_far:g}_r{args.refine_radius:g}_g{args.grading:g}"
            f"_u{refinement_base(args) or 0:g}{'_q' if args.recombine else ''}.vtu")


def build(product, level, args):
    """The mesh, the cut crack, the far-field datum and the contact law."""
    b = args.half_length
    if args.mesh == "gmsh":
        path = mesh_path(args, level)
        if not Path(path).exists():
            # a uniform refinement meshes ONCE at the base size and splits; the
            # level is the crack's face size either way, so everything
            # downstream reads the two kinds of ladder alike
            base, splits = refinement_base(args), 0
            if base:
                splits = round(math.log2(base / level))
                if splits < 0 or abs(base / 2 ** splits - level) > 1e-12 * base:
                    raise SystemExit(f"uniform refinement: {level:g} is not {base:g} halved "
                                     f"a whole number of times; fix --levels or use "
                                     f"--no-uniform-refinement")
            _crack_mesh.inclined_crack(
                path, half_length=b, domain=args.domain, psi=args.psi,
                size_crack=base or level, refinements=splits, size_far=args.size_far,
                grading=args.grading, refine_radius=args.refine_radius,
                recombine=args.recombine)
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
                             f"multiple of h = {h:.6g}; pick cells so that "
                             f"cells * {b} / {args.domain} is a whole number")
        if cells % 2:
            raise SystemExit("--mesh box needs an even cell count, or y = 0 is not a node plane")
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
        z = mk.centroid(mesh, DIM - 1, f)[2]
        (slab if min(abs(z), abs(z - thickness)) < 1e-9 * max(1.0, thickness)
         else lateral).append(f)
    model.add_free_slip(slab)
    # ONE CALL PER FACET: the datum is u = a + B (x - x_E) about the facet's
    # cofacet centroid, so the constant is u(x_E) and not u(0). A single call
    # over the whole list is accepted in silence and is wrong by O(h).
    for f in lateral:
        x_e = mk.centroid(mesh, DIM, mk.cofacet_of(mesh, DIM, f))
        model.prescribe_displacement([f], [float(v) for v in eps @ np.array(x_e)], grad)
    return mesh, model, normal, tangent


def solver_options(args):
    """A direct solve, by MUMPS unless told otherwise.

    The slab is one cell thick, so the system is two-dimensional in its
    connectivity and nested dissection keeps the fill near n log n. The default
    factorization does not order that way: measured on a graded mesh, 12.5k
    cells in 51 s and 15k cells unfinished after 15 minutes, against 9 s and
    3.5 GB for MUMPS at 12.4k cells.
    """
    return mk.SolverOptions(factorization=args.factorization)


# ---- reading the crack back --------------------------------------------------
def read_crack(model, normal, tangent):
    """(xi, t_N, t_T, g_N, g_T) per crack face, ordered along the crack.

    The per-face frame {n, t1, t2} need not agree between faces: a facet's
    canonical normal is its own, and a flip swaps which side is "plus", so the
    jump's ambient direction flips with it. t_N and g_N are invariant, but a
    tangential profile read straight out of the frame can alternate in sign. So
    the readings are carried back into ambient axes and oriented ONCE, against
    the crack's own normal, then resolved along the crack's own tangent. xi is
    measured from the crack's centre, -b at one tip and +b at the other.
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
            amb_j[order] @ normal, amb_j[order] @ tangent)


def central(xi, args, trim=None):
    """The faces away from the tips: |xi| <= (1 - trim) b."""
    return np.abs(xi) <= args.half_length * (1.0 - (args.trim if trim is None else trim))


def compare(xi, t_n, g_t, args, trim=None):
    """Relative L2 errors against eq. (34) on the central faces.

    The tips are excluded because the exact solution is singular there -- see
    the module docstring. `trim` is the fraction of the half-length dropped at
    EACH tip, so 0.10 keeps the central 90% of the trace.
    """
    b = args.half_length
    keep = central(xi, args, trim)
    if keep.sum() < 2:
        return float("nan"), float("nan")
    t_ex, g_ex = analytic(xi[keep] + b, b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    e_t = np.linalg.norm(t_n[keep] - t_ex) / np.linalg.norm(np.full(keep.sum(), t_ex))
    e_g = np.linalg.norm(g_t[keep] - g_ex) / (np.linalg.norm(g_ex) or 1.0)
    return float(e_t), float(e_g)


def rate(hs, errs):
    """The least-squares slope of log(error) against log(h)."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    ok = (hs > 0) & (errs > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0])


# ---- the study ---------------------------------------------------------------
def solve_all(args):
    rows, profiles = [], {p: {} for p in args.products}
    for level in args.levels:
        for product in args.products:
            mesh, model, normal, tangent = build(product, level, args)
            with stage(f"{product:<26s} level {level:<6g} {mesh.count(DIM):>6} cells"):
                model.solve(progress=False, options=solver_options(args))
            xi, t_n, t_t, g_n, g_t = read_crack(model, normal, tangent)
            # sliding reported positive, as the paper draws it; the tangential
            # traction takes the same orientation, so t_T = c_T g_T keeps its sign
            sgn = 1.0 if g_t.sum() >= 0.0 else -1.0
            g_t, t_t = sgn * g_t, sgn * t_t
            e_t, e_g = compare(xi, t_n, g_t, args)
            keep = central(xi, args)
            profiles[product][f"{level:g}"] = {
                "xi": xi.tolist(), "t_N": t_n.tolist(), "t_T": t_t.tolist(),
                "g_N": g_n.tolist(), "g_T": g_t.tolist()}
            rows.append(dict(
                level=level, product=product, cells=int(mesh.count(DIM)),
                faces=int(model.n_fracture_faces), n_dofs=int(model.n_dofs),
                t_N=float(t_n[keep].mean()), g_T=float(np.abs(g_t).max()),
                g_N=float(np.abs(g_n[keep]).max()), t_T=float(np.abs(t_t[keep]).max()),
                err_t_N=e_t, err_g_T=e_g))
            if args.vtu:
                model.write_fracture_vtu(f"{args.vtu}_{product}_{level:g}.vtu")
    return rows, profiles


def fit_rates(rows, args):
    """Per product: the slopes of the two errors against the crack's ACTUAL face
    size 2b / faces -- the mesh that was solved on, not the size asked for."""
    rates = {}
    for product in args.products:
        mine = [r for r in rows if r["product"] == product]
        hs = [2.0 * args.half_length / r["faces"] for r in mine]
        rates[product] = (rate(hs, [r["err_t_N"] for r in mine]),
                          rate(hs, [r["err_g_T"] for r in mine]))
    return rates


def report(rows, rates, args):
    b = args.half_length
    t_ex, g_ex = analytic(np.array([b]), b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    print(f"\n  on the central {1 - args.trim:.0%} of the trace:")
    print(f"  {'product':<26}{'cells':>7}{'faces':>6}{'t_N':>10}{'g_T peak':>12}"
          f"{'|g_N|':>9}{'|t_T|':>9}{'err t_N':>10}{'err g_T':>10}")
    for r in rows:
        print(f"  {r['product']:<26}{r['cells']:>7}{r['faces']:>6}{r['t_N']:>10.4f}"
              f"{r['g_T']:>12.4e}{r['g_N']:>9.1e}{r['t_T']:>9.1e}"
              f"{r['err_t_N']:>10.2e}{r['err_g_T']:>10.2e}")
    print(f"  {'exact (eq. 34)':<26}{'':>7}{'':>6}{float(t_ex):>10.4f}{float(g_ex[0]):>12.4e}"
          f"{0.0:>9.1e}{0.0:>9.1e}")
    if len(args.levels) > 1:
        print("\n  least-squares rates (paper: 1.11 and 1.39):")
        for product in args.products:
            print(f"    {product:<26} m(t_N) = {rates[product][0]:5.2f}   "
                  f"m(g_T) = {rates[product][1]:5.2f}")


# ---- the figures -------------------------------------------------------------
def label(product):
    family = "strong" if product in _realizations.STRONG else "weak"
    return f"{family} symmetry ({product})"


def draw_mesh(a, args, level):
    """The cells around the crack, in the plane of the slab."""
    b, psi = args.half_length, math.radians(args.psi)
    if args.mesh == "gmsh":
        import meshio
        from matplotlib.collections import PolyCollection
        m = meshio.read(mesh_path(args, level))
        names = {"wedge": "prisms", "hexahedron": "hexahedra"}
        parts = []
        for block in m.cells:
            # the z = 0 face of each cell: the first half of its nodes
            flat = block.data[:, :block.data.shape[1] // 2]
            xy = m.points[flat][:, :, :2]
            near = (np.abs(xy[:, :, 0]).min(axis=1) < 1.8) & (np.abs(xy[:, :, 1]).min(axis=1) < 1.4)
            a.add_collection(PolyCollection(xy[near], facecolors="none", edgecolors="0.6",
                                            linewidths=0.25))
            parts.append(f"{len(flat)} {names.get(block.type, block.type)}")
        tip = (b * math.cos(psi), b * math.sin(psi))
        count = " + ".join(parts)
    else:
        h = args.domain / int(level)
        for g in np.arange(-2.0, 2.0 + 0.5 * h, h):
            a.plot([-2, 2], [g, g], lw=0.3, color="0.6")
            a.plot([g, g], [-2, 2], lw=0.3, color="0.6")
        tip = (b, 0.0)
        count = f"{int(level)}x{int(level)} hexahedra"
    a.plot([-tip[0], tip[0]], [-tip[1], tip[1]], color="crimson", lw=2.2,
           solid_capstyle="butt", zorder=5)
    a.set_xlim(-1.7, 1.7); a.set_ylim(-1.25, 1.25); a.set_aspect("equal")
    return count


def draw_four(path, profiles, args):
    """Rows: levels. Columns: the mesh, then t_N, g_N, t_T, g_T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = args.half_length
    fine = np.linspace(0.0, 2 * b, 400)
    t_ex, g_ex = analytic(fine, b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    # (key, axis label, exact, what the linear law itself leaves, title)
    panels = (
        ("t_N", r"$t_N$ [MPa]", np.full_like(fine, t_ex), None,
         r"normal traction — exact $-\bar\sigma\sin^2\psi$"),
        ("g_N", r"$g_N$ [m]", np.zeros_like(fine), np.full_like(fine, t_ex / args.c_n),
         r"normal jump — exact 0 (closed); dashed $t_N/c_n$"),
        ("t_T", r"$t_T$ [MPa]", np.zeros_like(fine), args.c_t * g_ex,
         r"tangential traction — exact 0 (frictionless); dashed $c_T\,g_T$"),
        ("g_T", r"$g_T$ [m]", g_ex, None,
         r"tangential jump (sliding) — exact: eq. (34)"),
    )
    n = len(args.levels)
    fig, ax = plt.subplots(n, 5, figsize=(22.0, 4.0 * n), squeeze=False,
                           gridspec_kw={"width_ratios": [0.95, 1.2, 1.2, 1.2, 1.2]})
    for i, level in enumerate(args.levels):
        key_l = f"{level:g}"
        count = draw_mesh(ax[i, 0], args, level)
        faces = len(profiles[args.products[0]][key_l]["xi"])
        ax[i, 0].set_title(f"{count}, {faces} crack faces", fontsize=10)
        ax[i, 0].set_ylabel("y [m]"); ax[i, 0].tick_params(labelsize=8)
        for j, (key, ylab, exact, law, title) in enumerate(panels, start=1):
            a = ax[i, j]
            a.plot(fine - b, exact, color="k", lw=1.5, label="exact", zorder=4)
            if law is not None:
                a.plot(fine - b, law, color="k", lw=1.0, ls="--",
                       label="what the linear law leaves", zorder=4)
            for k, product in enumerate(args.products):
                c = profiles[product][key_l]
                a.plot(c["xi"], c[key], MARKERS[k % len(MARKERS)], ls="-", lw=0.7, ms=2.6,
                       color=COLORS[k % len(COLORS)], alpha=0.85, label=label(product))
            a.axvspan(-b, -b * (1 - args.trim), color="0.92", zorder=0)
            a.axvspan(b * (1 - args.trim), b, color="0.92", zorder=0)
            a.set_ylabel(ylab); a.grid(alpha=0.3); a.tick_params(labelsize=8)
            a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            if key == "t_N":
                a.set_ylim(2.05 * t_ex, 0.0)
            if i == 0:
                a.set_title(title, fontsize=9.5)
                a.legend(fontsize=7.5, loc="best")
        for a in ax[i]:
            if i == n - 1:
                a.set_xlabel("x [m]" if a is ax[i, 0] else r"$\xi$ from the crack centre [m]")

    ladder = ("uniform refinement: every triangle split in four per level"
              if refinement_base(args)
              else "structured box" if args.mesh == "box" else "re-meshed per level")
    fig.suptitle(
        "single crack under compression, CMAME 368 (2020) §5.2 — cut fracture, mortar "
        "traces, traction multiplier\n"
        fr"$\psi={args.psi:g}^\circ$, $\theta={args.theta:g}^\circ$, "
        fr"$\bar\sigma={args.sigma:g}$ MPa, $c_n={args.c_n:g}$, $c_T={args.c_t:g}$, "
        fr"domain ${args.domain/(2*b):.0f}\times 2b$;  {ladder}  "
        f"(shaded: the tips the error norms drop)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)


def draw_convergence(path, rows, rates, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = args.half_length
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for j, (key, title, paper) in enumerate((
            ("err_t_N", r"normal traction  $\|t_N - t_N^h\|_2 / \|t_N\|_2$", 1.11),
            ("err_g_T", r"tangential sliding  $\|g_T - g_T^h\|_2 / \|g_T\|_2$", 1.39))):
        a = ax[j]
        for k, product in enumerate(args.products):
            mine = [r for r in rows if r["product"] == product]
            a.loglog([2.0 * b / r["faces"] for r in mine], [r[key] for r in mine],
                     MARKERS[k % len(MARKERS)] + "-", color=COLORS[k % len(COLORS)], ms=6,
                     lw=1.2, label=f"{label(product)}   m = {rates[product][j]:.2f}")
        hs = np.array(sorted({2.0 * b / r["faces"] for r in rows}, reverse=True))
        top = max(r[key] for r in rows if 2.0 * b / r["faces"] == hs[0])
        a.loglog(hs, (hs / hs[0]) ** paper * top, "k--", lw=1.0,
                 label=f"paper's rate  m = {paper}")
        a.set_xlabel("crack face size 2b / faces [m]"); a.set_ylabel("relative error")
        a.set_title(title, fontsize=10); a.grid(which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"single crack under compression — convergence on the central "
                 f"{1 - args.trim:.0%} of the trace", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default=",".join(DEFAULT_PRODUCTS),
                    help="comma-separated stress products, drawn together on every panel. "
                         f"Both symmetry families: {', '.join(_realizations.names())}. "
                         f"Default {','.join(DEFAULT_PRODUCTS)} -- one of each.")
    ap.add_argument("--mesh", default="gmsh", choices=("gmsh", "box"),
                    help="gmsh: inclined crack embedded in a graded triangulation, extruded "
                         "one layer. box: axis-aligned crack in a structured box, load "
                         "rotated instead; needs no mesh generator")
    ap.add_argument("--levels", default=None,
                    help="the ladder, coarse to fine: crack face sizes under --mesh gmsh "
                         f"(default {DEFAULT_LEVELS['gmsh']}), cells across the domain under "
                         f"--mesh box (default {DEFAULT_LEVELS['box']})")
    ap.add_argument("--uniform-refinement", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="gmsh: mesh the first level and reach the rest by splitting every "
                         "triangle in four, so each level must halve the one before. "
                         "--no-uniform-refinement re-meshes each level, which costs fewer "
                         "cells and adds mesh-to-mesh noise as large as the convergence")
    ap.add_argument("--recombine", action="store_true",
                    help="gmsh: pair the triangles into quadrilaterals -- hexahedra after the "
                         "extrusion, with the few prisms the pairing leaves. The default "
                         "output names gain the suffix _quads")
    ap.add_argument("--refine-radius", type=float, default=0.4,
                    help="gmsh: the crack size is held within this distance of the crack")
    ap.add_argument("--grading", type=float, default=8.0,
                    help="gmsh: distance over which the size then grows to --size-far")
    ap.add_argument("--size-far", type=float, default=5.0,
                    help="gmsh: element size at the far boundary")
    ap.add_argument("--domain", type=float, default=60.0,
                    help="box width in metres; the far field is imposed on its boundary, "
                         "so the truncation is O((2b/L)^2) and does not fall with the mesh")
    ap.add_argument("--half-length", type=float, default=1.0, help="b; the crack is 2b long")
    ap.add_argument("--psi", type=float, default=20.0, help="crack inclination [deg]")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="friction angle [deg] of the CLOSED FORM compared against. The law "
                         "stays linear and frictionless, so only 0 is the problem solved; "
                         "the paper's own case is 30")
    ap.add_argument("--sigma", type=float, default=100.0, help="sigma_bar [MPa]")
    ap.add_argument("--young", type=float, default=25000.0, help="E [MPa]")
    ap.add_argument("--poisson", type=float, default=0.25, help="nu")
    ap.add_argument("--c-n", type=float, default=1e9,
                    help="normal STIFFNESS: large closes the crack (g_N -> 0)")
    ap.add_argument("--c-t", type=float, default=1e-3,
                    help="tangential STIFFNESS: small frees the sliding (t_T -> 0)")
    ap.add_argument("--trim", type=float, default=0.10,
                    help="fraction of the half-length dropped at each tip before the means "
                         "and error norms; 0.10 is the paper's central 90%%")
    ap.add_argument("--factorization", default="mumps",
                    help="the direct factorization; mumps orders the quasi-2D slab by "
                         "nested dissection")
    ap.add_argument("--out", default=None,
                    help="prefix of <out>.png, <out>_convergence.png and <out>.json; "
                         "crack_uniform_refinement by default, crack_compression for a "
                         "re-meshed or structured ladder, with _quads appended under "
                         "--recombine")
    ap.add_argument("--redraw", action="store_true",
                    help="redraw the figures from <out>.json without solving")
    ap.add_argument("--vtu", default="", help="prefix for per-run fracture .vtu files")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("crack_uniform_refinement"
                    if args.mesh == "gmsh" and args.uniform_refinement else "crack_compression")
        if args.mesh == "gmsh" and args.recombine:
            args.out += "_quads"
    store = Path(f"{args.out}.json")

    if args.redraw:
        data = json.loads(store.read_text())
        vars(args).update(data["setup"])
        rows, profiles = data["rows"], data["profiles"]
    else:
        args.products = [p.strip() for p in args.product.split(",") if p.strip()]
        for p in args.products:
            if p not in _realizations.names():
                raise SystemExit(f"unknown product {p!r}; choose from "
                                 f"{', '.join(_realizations.names())}")
        args.levels = [float(v) for v in
                       (args.levels or DEFAULT_LEVELS[args.mesh]).split(",")]

    b = args.half_length
    t_ex, g_ex = analytic(np.array([b]), b, args.young, args.poisson,
                          args.sigma, args.psi, args.theta)
    print("single crack under compression -- CMAME 368 (2020) 113161, Section 5.2")
    print(f"  E = {args.young:g} MPa, nu = {args.poisson:g}, sigma_bar = {args.sigma:g} MPa, "
          f"psi = {args.psi:g} deg, theta = {args.theta:g} deg, 2b = {2 * b:g} m")
    print(f"  domain {args.domain:g} m = {args.domain / (2 * b):.0f} crack lengths, "
          f"far-field truncation ~ {(2 * b / args.domain) ** 2:.2%}")
    print(f"  exact: t_N = {float(t_ex):.4f} MPa, peak g_T = {float(g_ex[0]):.4e} m\n")

    if not args.redraw:
        rows, profiles = solve_all(args)
    rates = fit_rates(rows, args)
    report(rows, rates, args)

    if not args.redraw:
        store.write_text(json.dumps({"setup": vars(args), "rows": rows, "rates": rates,
                                     "profiles": profiles}, indent=2))
    draw_four(f"{args.out}.png", profiles, args)
    written = [f"{args.out}.png"]
    if len(args.levels) > 1:
        draw_convergence(f"{args.out}_convergence.png", rows, rates, args)
        written.append(f"{args.out}_convergence.png")
    print(f"\n  wrote {', '.join(written + [str(store)])}")


if __name__ == "__main__":
    main()
