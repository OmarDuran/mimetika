"""Drilling of a borehole (Coussy 2004, Sect. 5.2.3) -- six-field DeRham.

An infinite porous layer under hydrostatic initial stress ``sigma = -w I``
and pore pressure ``p0``; at ``t = 0`` a borehole of radius ``a`` is drilled
instantaneously: on the wall the fluid pressure becomes ``p1`` and the
radial stress ``-p1``.  Plane strain, cylindrical symmetry.  The exact
fields (Coussy Eqs. 5.99, 5.101, 5.109, 5.111):

    p(r,t)  = p0 + (p1 - p0) pbar(r,t)
    xi(r,t) = (p1 - w)/(2 mu) a^2/r + a b (p1 - p0)/(K + 4mu/3) xibar(r,t)

with ``pbar``, ``xibar`` Bessel-integral inversions of the radial diffusion
with diffusivity ``c_f = k M (K + 4mu/3)/(K_u + 4mu/3)``, ``K_u = K + b^2 M``
(Eq. 5.22).  The mimetika 2D model is exactly plane strain (its compliance
coefficient is ``lam/(2(lam + mu))``), so the 3D Lame parameters carry over
unchanged.

Discretization: quarter annulus meshed by **gmsh** (graded triangles), the
six-field formulation with the de Rham defaults -- no stabilization in any
block.  Rollers and sealed facets on the symmetry planes; tractions and
pressures on the wall and the far boundary.

Reference (APA)
---------------
Coussy, O. (2004). *Poromechanics*. John Wiley & Sons.
https://doi.org/10.1002/0470092718  (ISBN 0-470-84920-7)

The problem and its exact solution are Section 5.2.3, "Drilling of a
Borehole" (Chapter 5, "Problems of Poroelasticity"), pp. 129-133; the
diffusivity coefficient is Eq. (5.22), p. 117.  In-text: (Coussy, 2004,
Section 5.2.3).

Run:  python benchmarks/poroelasticity/borehole.py [--h-in 0.2 --steps 80]
"""

from __future__ import annotations

import argparse

import numpy as np
from mimetika.assembly.four_field import FourFieldPoroMechanics
from mimetika.assembly.mixed import MixedSolution, boundary_facets
from mimetika.materials import Material
from mimetika.mesh.mesh import Mesh
from mimetika.simulation import (
    FlowBC,
    MechanicsBC,
    PoromechanicsSolver,
    RKTimeStepping,
)

# -- problem data (mu, lam are the 3D Lame parameters; plane strain) ----------
A_R, R_OUT = 1.0, 40.0
MU, LAM = 1.0, 1.0
B_BIOT, M_BIOT = 0.8, 10.0
K3 = LAM + 2.0 * MU / 3.0
KU3 = K3 + B_BIOT**2 * M_BIOT
CF = 1.0 * M_BIOT * (K3 + 4 * MU / 3) / (KU3 + 4 * MU / 3)  # k = 1
P0, P1, W = 1.0, 2.0, 3.0
NU = LAM / (2.0 * (LAM + MU))


# -- exact solution: Talbot inversion of Coussy (5.100) and (5.110) -----------
#
# In dimensionless variables (a = c_f = 1, T = c_f t / a^2):
#   pbar*(s)  = K0(rt sqrt(s)) / (s K0(sqrt(s)))
#   xibar*(s) = ((1/rt) K1(sqrt(s)) - K1(rt sqrt(s))) / (s^{3/2} K0(sqrt(s)))
# The Bessel-integral forms (5.101)/(5.111) are equivalent but numerically
# hostile (oscillatory); the Laplace inversion of the closed forms is smooth.

from pathlib import Path

import mpmath as mp

_CACHE_FILE = Path(__file__).parent / "_cache" / "borehole_exact.npz"
_cache: dict = {}
if _CACHE_FILE.exists():
    _d = np.load(_CACHE_FILE)
    _cache = {
        (str(k), float(r), float(T)): float(v)
        for k, r, T, v in zip(_d["kind"], _d["r"], _d["T"], _d["val"])
    }


def _save_cache() -> None:
    _CACHE_FILE.parent.mkdir(exist_ok=True)
    keys = list(_cache)
    np.savez_compressed(
        _CACHE_FILE,
        kind=np.array([k[0] for k in keys]),
        r=np.array([k[1] for k in keys]),
        T=np.array([k[2] for k in keys]),
        val=np.array([_cache[k] for k in keys]),
    )


def pbar_exact(r, T):
    key = ("p", round(float(r), 10), round(float(T), 12))
    if key not in _cache:
        rt = float(r) / A_R

        def F(s):
            q = mp.sqrt(s)
            return mp.besselk(0, rt * q) / (s * mp.besselk(0, q))

        _cache[key] = float(mp.invertlaplace(F, float(T), method="talbot", degree=14))
    return _cache[key]


def xibar_exact(r, T):
    key = ("x", round(float(r), 10), round(float(T), 12))
    if key not in _cache:
        rt = float(r) / A_R

        def F(s):
            q = mp.sqrt(s)
            return (
                mp.besselk(1, q) / rt - mp.besselk(1, rt * q)
            ) / (s * q * mp.besselk(0, q))

        _cache[key] = float(mp.invertlaplace(F, float(T), method="talbot", degree=14))
    return _cache[key]


# -- gmsh quarter annulus ------------------------------------------------------

def _triangulation(h_in: float, h_out: float):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    c = occ.addPoint(0, 0, 0)
    p1 = occ.addPoint(A_R, 0, 0)
    p2 = occ.addPoint(R_OUT, 0, 0)
    p3 = occ.addPoint(0, R_OUT, 0)
    p4 = occ.addPoint(0, A_R, 0)
    l_bottom = occ.addLine(p1, p2)
    arc_out = occ.addCircleArc(p2, c, p3)
    l_left = occ.addLine(p3, p4)
    arc_in = occ.addCircleArc(p4, c, p1)
    loop = occ.addCurveLoop([l_bottom, arc_out, l_left, arc_in])
    occ.addPlaneSurface([loop])
    occ.synchronize()

    fld = gmsh.model.mesh.field
    dist = fld.add("Distance")
    fld.setNumbers(dist, "CurvesList", [arc_in])
    fld.setNumber(dist, "Sampling", 200)
    thr = fld.add("Threshold")
    fld.setNumber(thr, "InField", dist)
    fld.setNumber(thr, "SizeMin", h_in)
    fld.setNumber(thr, "SizeMax", h_out)
    fld.setNumber(thr, "DistMin", 0.0)
    fld.setNumber(thr, "DistMax", R_OUT - A_R)
    fld.setAsBackgroundMesh(thr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    pts = coords.reshape(-1, 3)
    remap = {int(t): i for i, t in enumerate(tags)}
    tris = []
    for etype, _, conn in zip(*gmsh.model.mesh.getElements(2)):
        if etype == 2:  # 3-node triangles
            tris = [
                [remap[int(n)] for n in tri]
                for tri in np.asarray(conn).reshape(-1, 3)
            ]
    gmsh.finalize()
    return pts, tris


def quarter_annulus_2d(h_in: float, h_out: float) -> Mesh:
    pts, tris = _triangulation(h_in, h_out)
    return Mesh.from_polygons(pts, tris)


def extruded_prisms(h_in: float, h_out: float, layers: int, height: float) -> Mesh:
    """Extrude the gmsh triangulation into vertical prisms (polyhedral cells).

    Rollers and sealed faces on top and bottom make the extrusion exactly
    plane strain, so the 2D exact solution remains the reference; the cells,
    however, are genuine polytopes and exercise the 3D de Rham enrichment.
    """
    pts2, tris = _triangulation(h_in, h_out)
    n2 = len(pts2)
    # counter-clockwise triangles, so the face orientations below are outward
    def ccw(t):
        a, b, c = (pts2[i][:2] for i in t)
        u, v = b - a, c - a
        return t if u[0] * v[1] - u[1] * v[0] > 0 else t[::-1]

    tris = [ccw(list(t)) for t in tris]
    dz = height / layers
    points = np.vstack([
        np.column_stack([pts2[:, 0], pts2[:, 1], np.full(n2, l * dz)])
        for l in range(layers + 1)
    ])
    cells = []
    for l in range(layers):
        lo, hi = l * n2, (l + 1) * n2
        for a, b, c in tris:
            bot = [c + lo, b + lo, a + lo]          # outward -z
            top = [a + hi, b + hi, c + hi]          # outward +z
            sides = [
                [a + lo, b + lo, b + hi, a + hi],   # t x z = outward
                [b + lo, c + lo, c + hi, b + hi],
                [c + lo, a + lo, a + hi, c + hi],
            ]
            cells.append([bot, top, *sides])
    return Mesh.from_cells(points, cells)


# -- benchmark -----------------------------------------------------------------

def _split_six(problem, x):
    """Unpack the six-field vector [sigma, ps, u, s, q, p]."""
    n1 = problem.n_stress
    n1p = n1 + problem.n_cells
    n2 = n1p + problem.d * problem.n_cells
    n3 = n2 + problem.n_skew * problem.n_cells
    n4 = n3 + problem.n_flux
    return MixedSolution(
        {
            "stress": x[:n1],
            "solid_pressure": x[n1:n1p],
            "displacement": x[n1p:n2],
            "rotation": x[n2:n3],
            "flux": x[n3:n4],
            "pressure": x[n4:],
        }
    )


def _initial_fields(prev, n_tri):
    """t = 0^- state: uniform pressure, zero net displacement and rotation."""
    zero_v = np.zeros((n_tri, 3))
    return {
        "pressure": prev["pressure"],
        "pbar": np.zeros(n_tri),
        "solid_pressure": np.full(n_tri, -W),
        "displacement": zero_v,
        "xi": zero_v,
        "xi_radial": np.zeros(n_tri),
        "rotation": np.zeros(n_tri),
    }


def run(h_in, h_out, n_steps, T_end, report_at, vtk=None, dim=2,
        layers=2, height=1.0):
    if dim == 2:
        mesh = quarter_annulus_2d(h_in, h_out)
    else:
        mesh = extruded_prisms(h_in, h_out, layers, height)
    d = mesh.dim
    n_tri = mesh.num_cells(d)
    kind = "triangles" if d == 2 else f"prisms ({layers} layers)"
    print(f"gmsh quarter annulus: {n_tri} {kind}, R/a = {R_OUT / A_R:g}")

    material = Material(
        shear_modulus=MU,
        poisson=NU,
        biot=B_BIOT,
        inverse_biot_modulus=1.0 / M_BIOT,
        permeability=1.0,
        viscosity=1.0,
    )

    # boundary classification by facet centroid
    fc = mesh.geometry.centroids(d - 1)
    bnd = boundary_facets(mesh)
    tol = 1e-8
    on_x = [f for f in bnd if abs(fc[f][1]) < tol]  # y = 0 plane
    on_y = [f for f in bnd if abs(fc[f][0]) < tol]  # x = 0 plane
    sym = on_x + on_y
    if d == 3:  # plane-strain closure: rollers + sealed on top and bottom
        sym += [
            f for f in bnd
            if abs(fc[f][2]) < tol or abs(fc[f][2] - height) < tol
        ]
    radial = [f for f in bnd if f not in set(sym)]
    rmid = 0.5 * (A_R + R_OUT)
    inner = [f for f in radial if np.linalg.norm(fc[f][:2]) < rmid]
    outer = [f for f in radial if np.linalg.norm(fc[f][:2]) >= rmid]

    def traction(x):
        r = np.linalg.norm(np.atleast_2d(x)[:, :2], axis=1)
        s = np.where(r < rmid, -P1, -W)
        out = np.zeros((len(s), 3, 3))
        out[:, 0, 0] = out[:, 1, 1] = out[:, 2, 2] = s
        return out

    def pressure_bc(x):
        r = np.linalg.norm(np.atleast_2d(x)[:, :2], axis=1)
        return np.where(r < rmid, P1, P0)

    dt = (T_end * A_R**2 / CF) / n_steps

    # the full poromechanics solver on its constant-dt fast path: the matrix
    # and every boundary datum are constant in time, so the first flow_step
    # assembles and factorizes ONCE and every later step is a
    # back-substitution plus the previous-state update of the pressure rows
    import time

    engine = PoromechanicsSolver(
        mesh, material,
        flow="solved",
        bc=MechanicsBC(traction=traction, traction_facets=inner + outer,
                       roller_facets=sym),
        flow_bc=FlowBC(flux_facets=sym, pressure=pressure_bc),
        time=RKTimeStepping(dt=dt),
        poromechanics=FourFieldPoroMechanics,  # DeRham defaults
        linear_solver="direct",
    )
    problem = engine.poro

    # initial state compatible with the plane-strain closure (u_z = 0):
    # in-plane -w, and the confined vertical stress szz0 = 2 lam eps2 - b p0
    # (the change fields still match Coussy exactly, by linearity)
    eps2 = (B_BIOT * P0 - W) / (2.0 * (LAM + MU))
    szz0 = 2.0 * LAM * eps2 - B_BIOT * P0
    sigma0 = np.diag([-W, -W, szz0])
    init_stress = problem.mechanics.interpolate_stress(
        lambda x: np.broadcast_to(sigma0, (len(np.atleast_2d(x)), 3, 3))
    )
    prev = MixedSolution(
        {"stress": init_stress, "pressure": np.full(n_tri, P0)}
    )

    t0 = time.perf_counter()

    series = None
    if vtk is not None:
        from mimetika.postprocess.series import MixedDimensionalSeries

        series = MixedDimensionalSeries(vtk, mesh)
        cc = mesh.geometry.centroids(d)
        rr = np.linalg.norm(cc[:, :2], axis=1)
        err = cc[:, :2] / rr[:, None]
        frame = problem.mechanics.inner.frame
        eps0 = (B_BIOT * P0 - W) / (2.0 * (LAM + MU))

        def fields(sol):
            u = sol["displacement"].reshape(-1, d) @ frame.T  # (n, 3) ambient
            xi = u.copy()
            xi[:, :2] -= eps0 * cc[:, :2]  # Coussy's xi: net of the pre-strain
            return {
                "pressure": sol["pressure"],
                "pbar": (sol["pressure"] - P0) / (P1 - P0),
                "solid_pressure": sol["solid_pressure"],
                "displacement": u,
                "xi": xi,
                "xi_radial": np.einsum("ci,ci->c", xi[:, :2], err),
                "rotation": sol["rotation"],
            }

        series.write(0.0, bulk=_initial_fields(prev, n_tri))

    t0 = time.perf_counter()
    T, results = 0.0, {}
    for _ in range(n_steps):
        sol = engine.flow_step(previous=prev)  # fast path: one factorization
        prev = sol
        T += dt * CF / A_R**2
        for Tr in report_at:
            if abs(T - Tr) < 0.5 * dt * CF / A_R**2 and Tr not in results:
                results[Tr] = sol
        if series is not None:
            series.write(T, bulk=fields(sol))
    print(f"{n_steps} steps: {time.perf_counter() - t0:.1f}s")
    if series is not None:
        print(f"wrote {series.collection} ({n_steps + 1} timesteps)")
    return mesh, problem, results


def compare(runs, path):
    """Overlay the runs (one linestyle per dimension) against the exact curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    rs = np.linspace(A_R * 1.001, 6.0 * A_R, 40)
    times = sorted({Tr for _, _, _, results in runs for Tr in results})
    colors = {Tr: f"C{i}" for i, Tr in enumerate(times)}
    styles = {2: "-", 3: "--"}

    for dim, mesh, problem, results in runs:
        d = mesh.dim
        cc = mesh.geometry.centroids(d)
        r = np.linalg.norm(cc[:, :2], axis=1)
        er = cc[:, :2] / r[:, None]
        frame = problem.mechanics.inner.frame
        sel = r < 6.0 * A_R
        print(f"-- {dim}D --")
        print(f"{'T':>6} {'rms(pbar)':>12} {'rms(xibar)':>12}")
        for Tr, sol in sorted(results.items()):
            pb_num = (sol["pressure"] - P0) / (P1 - P0)
            u = sol["displacement"].reshape(-1, d) @ frame.T
            ur = np.einsum("ci,ci->c", u[:, :2], er)
            ur = ur - (B_BIOT * P0 - W) / (2.0 * (LAM + MU)) * r
            xib_num = (ur - (P1 - W) * A_R**2 / (2 * MU * r)) * (
                (K3 + 4 * MU / 3) / (A_R * B_BIOT * (P1 - P0))
            )
            rq = np.round(r[sel], 1)
            pmap = {x: pbar_exact(x, Tr) for x in np.unique(rq)}
            xmap = {x: xibar_exact(x, Tr) for x in np.unique(rq)}
            e_p = np.sqrt(np.mean((pb_num[sel] - [pmap[x] for x in rq]) ** 2))
            e_x = np.sqrt(np.mean((xib_num[sel] - [xmap[x] for x in rq]) ** 2))
            print(f"{Tr:6.2f} {e_p:12.3e} {e_x:12.3e}")

            order = np.argsort(r[sel])
            kw = dict(ls=styles[dim], lw=1.1, color=colors[Tr])
            if dim == 2:  # sparse markers so the 3D overlap is visible on top
                kw.update(marker="s", ms=2.2, mfc=colors[Tr], markevery=7)
            ax1.plot(r[sel][order], pb_num[sel][order], **kw)
            ax2.plot(r[sel][order], xib_num[sel][order], **kw)

    for Tr in times:  # exact: one marker set per time, on top
        ax1.plot(rs, [pbar_exact(x, Tr) for x in rs], "o", ms=2.8,
                 mfc="white", mew=0.9, color=colors[Tr])
        ax2.plot(rs, [xibar_exact(x, Tr) for x in rs], "o", ms=2.8,
                 mfc="white", mew=0.9, color=colors[Tr])

    ax1.set_xlabel("r / a"), ax1.set_ylabel(r"$\bar{p}$")
    ax2.set_xlabel("r / a"), ax2.set_ylabel(r"$\bar{\xi}$")
    handles = [Line2D([], [], color=colors[Tr], lw=1.4, label=f"T = {Tr:g}")
               for Tr in times]
    ax1.legend(handles=handles, fancybox=True, framealpha=0.9,
               edgecolor="0.8", fontsize=8, title="time", title_fontsize=8)
    style = [
        Line2D([], [], color="k", lw=1.1, marker="s", ms=2.2,
               label="mimetic-AFW-BDM six fields, 2D (triangles)"),
        Line2D([], [], color="k", lw=1.1, ls="--",
               label="mimetic-AFW-BDM six fields, 3D (prisms)"),
        Line2D([], [], color="k", ls="", marker="o", ms=3, mfc="white",
               mew=0.9, label="exact (Coussy 5.2.3)"),
    ]
    ax2.legend(handles=style, fancybox=True, framealpha=0.9,
               edgecolor="0.8", fontsize=7.5)
    for ax in (ax1, ax2):
        ax.set_xlim(1, 6)
        ax.grid(True, color="gray", alpha=0.35, lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        'Section 5.2.3, "Drilling of a Borehole" '
        '(Chapter 5, "Problems of Poroelasticity", Coussy, 2004)',
        fontsize=10,
    )
    fig.text(
        0.02, 0.02,
        "Postprocessing of the DOFs:  "
        r"$\bar{p} = (p_E - p_0)/(p_1 - p_0)$ from the cell pore-pressure"
        " unknowns;\n"
        r"$\bar{\xi}$ from the cell-mean displacement projected on $e_r$,"
        r" minus the pre-drilling strain $\epsilon_0 x$ and the undrained"
        r" term $(p_1{-}\varpi)a^2/2\mu r$,"
        "\n"
        r"scaled by $(K + 4\mu/3)/(a\,b\,(p_1 - p_0))$;"
        " cell values are sorted by $r = |x|$ (all layers in 3D).",
        fontsize=6.5, color="0.35", va="bottom", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    _save_cache()
    print(f"wrote {path}  (exact-solution cache: {len(_cache)} entries)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-in", type=float, default=0.35)
    ap.add_argument("--h-out", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--t-end", type=float, default=0.8)
    ap.add_argument("--report", type=float, nargs="+", default=[0.1, 0.4, 0.8])
    ap.add_argument("--out", default="benchmarks/poroelasticity/borehole.png")
    ap.add_argument(
        "--vtk",
        nargs="?",
        const="benchmarks/poroelasticity/borehole_vtk/borehole",
        default=None,
        help="write a .pvd/.vtu time series per dimension to this path stem",
    )
    ap.add_argument("--dims", type=int, nargs="+", choices=(2, 3),
                    default=[2, 3])
    ap.add_argument("--h-in3", type=float, default=0.35,
                    help="wall mesh size for the 3D extrusion")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--height", type=float, default=1.0)
    args = ap.parse_args(argv)
    runs = []
    for dim in args.dims:
        h = args.h_in if dim == 2 else args.h_in3
        vtk = f"{args.vtk}_{dim}d" if args.vtk else None
        mesh, problem, results = run(
            h, args.h_out, args.steps, args.t_end, args.report,
            vtk=vtk, dim=dim, layers=args.layers, height=args.height,
        )
        runs.append((dim, mesh, problem, results))
    compare(runs, args.out)


if __name__ == "__main__":
    main()
