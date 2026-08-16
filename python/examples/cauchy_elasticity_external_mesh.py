#!/usr/bin/env python3
"""Cauchy elasticity on a mesh read from a .vtu: the linear patch test.

The other elasticity example builds its own annulus and compares against Lame's
closed form. This one takes the mesh as DATA -- whatever is in the file, of
whatever cell type -- and so cannot know a closed form for it. What it can do
instead is prescribe one:

    u(x) = (x - x_min) / L ,     L = the largest bounding-box extent

which runs from 0 at the low corner of the box to 1 at the far end of its
longest side, and is imposed on EVERY boundary facet. Pure Dirichlet: no
traction is given anywhere, so the displacement data alone fixes the solution
and no rigid-body mode survives.

A linear field is the one case where the answer is known on any mesh at all.
The datum is affine, both integrals it needs are already carried by the stress
operators' facet moments, so no boundary quadrature enters and the mixed method
REPRODUCES the field rather than approximating it. The error is therefore
round-off and not discretization -- on a good mesh and a bad one alike, which
is what makes this a test of the method rather than of the mesh.

The stress that goes with it is uniform,

    eps = sym(grad u) = I_d / L ,    sigma = 2 mu eps + lam tr(eps) I ,

so sigma_kk = (2 mu + d lam) / L on the d meshed axes, and that is checked too.

Run it:

    PYTHONPATH=.. python cauchy_elasticity_external_mesh.py --mesh domain.vtu
    PYTHONPATH=.. python cauchy_elasticity_external_mesh.py --mesh domain.vtu --vtu out.vtu

with --make-mesh to produce a sample file first, if there is none to hand:

    PYTHONPATH=.. python cauchy_elasticity_external_mesh.py --make-mesh domain.vtu
"""

import argparse
import math

import mimetika_cxx as mk
import numpy as np
from _stages import stage

MU, LAM = 1.0, 1.0

PRODUCTS = {
    "derham_afw": mk.StressRealization.derham_afw,
    "derham_afw_rt": mk.StressRealization.derham_afw_rt,
    "stabilized_afw": mk.StressRealization.stabilized_afw,
}


def bounding_box(mesh):
    """The box the mesh occupies, and the characteristic length of the domain."""
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return lo, hi, float((hi - lo)[: mesh.dim].max())


def prescribe_linear_displacement(model, mesh, dim, lo, length):
    """u = (x - lo) / L on every boundary facet, as an affine datum per facet.

    The datum is written about the CELL centroid -- u = a + B (x - x_E) -- so
    the gradient is the same everywhere and only the constant moves.
    """
    gradient = [0.0] * 9
    for k in range(dim):
        gradient[k * 3 + k] = 1.0 / length

    facets = mk.boundary_facets(mesh, dim)
    for f in facets:
        x_e = mk.centroid(mesh, dim, mk.cofacet_of(mesh, dim, f))
        constant = [0.0] * 3
        for k in range(dim):
            constant[k] = (x_e[k] - lo[k]) / length
        model.prescribe_displacement([f], constant, gradient)
    return len(facets)


def exact(x, lo, length, dim):
    return [(x[k] - lo[k]) / length for k in range(dim)]


def make_mesh(path):
    """A sample .vtu, so the example runs without a mesh from somewhere else."""
    mesh = mk.annulus(8, 4, 2, mk.Family.simplex, 1.0, 10.0, 1.0)
    mk.write_vtu(mesh, path)
    print(f"wrote {path}: {mesh.count(2)} cells, {mesh.count(0)} vertices")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", help="the .vtu to solve on")
    ap.add_argument("--make-mesh", help="write a sample .vtu to this path and stop")
    ap.add_argument("--product", default="derham_afw", choices=sorted(PRODUCTS))
    ap.add_argument(
        "--nu", type=float, default=None, help="Poisson ratio (default lam = mu = 1)"
    )
    ap.add_argument("--vtu", help="write the solution to this .vtu")
    args = ap.parse_args()

    if args.make_mesh:
        make_mesh(args.make_mesh)
        return
    if not args.mesh:
        ap.error("--mesh is required (or --make-mesh to produce one)")

    with stage(f"reading {args.mesh}"):
        mesh = mk.read_vtu(args.mesh)
    dim = mesh.dim
    if dim not in (2, 3):
        raise SystemExit(
            f"{args.mesh}: top cells are {dim}-dimensional, expected 2 or 3"
        )

    with stage("bounding box"):
        lo, hi, length = bounding_box(mesh)
    lam = LAM if args.nu is None else 2.0 * MU * args.nu / (1.0 - 2.0 * args.nu)
    mat = mk.ElasticMaterial(MU, lam)
    how = PRODUCTS[args.product]

    print(f"{args.mesh}: {dim}D, {mesh.count(dim)} cells, {mesh.count(0)} vertices")
    print(
        f"  box  {np.array2string(lo[:dim], precision=4)} .. "
        f"{np.array2string(hi[:dim], precision=4)}"
    )
    print(f"  characteristic length L = {length:.6g}")
    print(f"  mu = {mat.shear}, lam = {mat.lame}  (nu = {mat.poisson():.4f})")
    print(f"  {mk.stress_realization_name(how)}\n")

    with stage("building the stress operators"):
        model = mk.CauchyElasticityModel(mesh, dim, mat, how)
    with stage("prescribing u on the boundary"):
        n_facets = prescribe_linear_displacement(model, mesh, dim, lo, length)
    model.solve(progress=True)
    print(f"\n  u = (x - x_min)/L on all {n_facets} boundary facets, pure Dirichlet")
    print(
        f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized\n"
    )

    # ---- the field the method should reproduce exactly ----------------------
    with stage("measuring the displacement error"):
        worst = rms = 0.0
        for e in range(model.n_cells):
            want = exact(mk.centroid(mesh, dim, e), lo, length, dim)
            for k in range(dim):
                d = model.displacement(e, k) - want[k]
                worst = max(worst, abs(d))
                rms += d * d
        rms = math.sqrt(rms / (model.n_cells * dim))
    print()
    print(f"  max |u - u_exact|  {worst:11.3e}")
    print(f"  rms |u - u_exact|  {rms:11.3e}   <- round-off, not discretization\n")

    # ---- and the uniform stress that goes with it ---------------------------
    want_kk = (2.0 * mat.shear + dim * mat.lame) / length
    off = 0.0
    diag = 0.0
    for e in range(model.n_cells):
        s = model.cell_stress(e)
        for k in range(dim):
            diag = max(diag, abs(s[k * 3 + k] - want_kk))
            for c in range(dim):
                if k != c:
                    off = max(off, abs(s[k * 3 + c]))
    print(f"  sigma_kk expected  {want_kk:11.6f}   (2 mu + d lam) / L")
    print(f"  max deviation      {diag:11.3e}")
    print(f"  max shear          {off:11.3e}")

    if args.vtu:
        with stage(f"writing {args.vtu}"):
            n = model.n_cells
            u = np.zeros((n, 3))
            u_exact = np.zeros((n, 3))
            for e in range(n):
                want = exact(mk.centroid(mesh, dim, e), lo, length, dim)
                for k in range(dim):
                    u[e, k] = model.displacement(e, k)
                    u_exact[e, k] = want[k]
            mk.write_vtu(
                mesh,
                args.vtu,
                {
                    "displacement": u,
                    "displacement_exact": u_exact,
                    "error": u - u_exact,
                    "stress": np.array([model.cell_stress(e) for e in range(n)]),
                },
            )
    

if __name__ == "__main__":
    main()
