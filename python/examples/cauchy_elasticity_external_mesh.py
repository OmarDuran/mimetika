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

import mimetika_cxx as mk
import numpy as np

from _stages import stage

MU, LAM = 1.0, 1.0

# THE LINEAR SOLVER. "riesz" is the Riesz map of the space the operator is an
# isomorphism on -- P is the Gram matrix of its norm -- so its iteration count
# does not grow with the mesh. "direct" is a full factorization: exact, and the
# wrong instrument past a few hundred thousand unknowns.
SOLVERS = {
    "riesz": mk.SolverOptions(
        method="gmres", preconditioner="riesz", rtol=1e-12, max_iterations=2000
    ),
    "direct": mk.SolverOptions(),
}

PRODUCTS = {
    "derham_afw": mk.StressRealization.derham_afw,
    "derham_afw_rt": mk.StressRealization.derham_afw_rt,
    "stabilized_afw": mk.StressRealization.stabilized_afw,
}


def fmt(v):
    """A short fixed-width row of numbers, so exact and numerical line up."""
    return "[" + " ".join(f"{x:+.4f}" for x in v) + "]"


def bounding_box(mesh):
    """The box the mesh occupies, and the characteristic length of the domain."""
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return lo, hi, float((hi - lo)[: mesh.dim].max())


def skew_generators(dim):
    """The (i < j) pairs, in the order the rotation unknown numbers them."""
    return [(i, j) for i in range(dim) for j in range(i + 1, dim)]


def gradient_of(dim, length, spin):
    """grad u = (I + W) / L, with W the skew part of magnitude `spin`.

    The symmetric part is the dilation, so the STRESS IS UNCHANGED by the spin
    -- sigma sees sym(grad u) only. What the spin changes is the rotation, which
    is otherwise identically zero and tests nothing: the AFW multiplier exists
    to enforce symmetry weakly, and a field with no rotation never asks it to.
    """
    g = [0.0] * 9
    for k in range(dim):
        g[k * 3 + k] = 1.0 / length
    for i, j in skew_generators(dim):
        g[i * 3 + j] += spin / length
        g[j * 3 + i] -= spin / length
    return g


def prescribe_linear_displacement(model, mesh, dim, lo, gradient):
    """u = grad u . (x - lo) on every boundary facet, as an affine datum.

    The datum is written about the CELL centroid -- u = a + B (x - x_E) -- so
    the gradient is the same everywhere and only the constant moves.
    """
    facets = mk.boundary_facets(mesh, dim)
    for f in facets:
        x_e = mk.centroid(mesh, dim, mk.cofacet_of(mesh, dim, f))
        constant = exact_displacement(x_e, lo, gradient, dim) + [0.0] * (3 - dim)
        model.prescribe_displacement([f], constant, gradient)
    return len(facets)


def exact_displacement(x, lo, gradient, dim):
    return [sum(gradient[k * 3 + c] * (x[c] - lo[c]) for c in range(dim)) for k in range(dim)]


def exact_rotation(gradient, dim):
    """skew(grad u), component by component in the generator order."""
    return [gradient[i * 3 + j] for i, j in skew_generators(dim)]


def exact_stress(gradient, mat, dim):
    """sigma = 2 mu sym(grad u) + lam tr(grad u) I, as a full 3x3."""
    trace = sum(gradient[k * 3 + k] for k in range(dim))
    s = [0.0] * 9
    for i in range(dim):
        for j in range(dim):
            sym = 0.5 * (gradient[i * 3 + j] + gradient[j * 3 + i])
            s[i * 3 + j] = 2.0 * mat.shear * sym + (mat.lame * trace if i == j else 0.0)
    return s


def make_mesh(path):
    """A sample .vtu, so the example runs without a mesh from somewhere else."""
    mesh = mk.annulus(8, 4, 2, mk.Family.simplex, 1.0, 10.0, 1.0)
    mk.write_vtu(mesh, path)
    print(f"wrote {path}: {mesh.count(2)} cells, {mesh.count(0)} vertices")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", help="the .vtu to solve on")
    ap.add_argument("--make-mesh", help="write a sample .vtu to this path and stop")
    ap.add_argument("--product", default="stabilized_afw", choices=sorted(PRODUCTS))
    ap.add_argument(
        "--nu", type=float, default=None, help="Poisson ratio (default lam = mu = 1)"
    )
    ap.add_argument("--vtu", help="write the solution to this .vtu")
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVERS))
    ap.add_argument("--spin", type=float, default=0.5,
                    help="magnitude of the skew part of grad u (0 leaves the rotation zero)")
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
    print(f"  {mk.stress_realization_name(how)}, {args.solver} solver\n")

    gradient = gradient_of(dim, length, args.spin)
    with stage("building the stress operators"):
        model = mk.CauchyElasticityModel(mesh, dim, mat, how)
    with stage("prescribing u on the boundary"):
        n_facets = prescribe_linear_displacement(model, mesh, dim, lo, gradient)
    report = model.solve(progress=True, options=SOLVERS[args.solver])
    if args.solver != "direct":
        print(f"  {args.solver}: {report.iterations} iterations, {report.reason}")
    print(f"\n  u = (I + W)(x - x_min)/L on all {n_facets} boundary facets, pure Dirichlet")
    print(f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized")

    # ---- the three fields, each against the value the datum determines -------
    #
    # The method reproduces a linear displacement exactly, so ALL THREE are
    # known in closed form and none of them is a discretization error: what is
    # left is round-off, or the residual tolerance of an iterative solve.
    u_hat = exact_rotation(gradient, dim)
    s_hat = exact_stress(gradient, mat, dim)
    n_rot = model.n_rotations

    with stage("measuring displacement, rotation and stress"):
        du = dr = ds = dso = 0.0
        got_u = [0.0] * dim
        got_r = [0.0] * n_rot
        got_s = [0.0] * 9
        for e in range(model.n_cells):
            want_u = exact_displacement(mk.centroid(mesh, dim, e), lo, gradient, dim)
            for k in range(dim):
                got_u[k] = model.displacement(e, k)
                du = max(du, abs(got_u[k] - want_u[k]))
            for k in range(n_rot):
                got_r[k] = model.rotation(e, k)
                dr = max(dr, abs(got_r[k] - u_hat[k]))
            s = model.cell_stress(e)
            for k in range(9):
                got_s[k] = s[k]
            for k in range(dim):
                ds = max(ds, abs(s[k * 3 + k] - s_hat[k * 3 + k]))
                for c in range(dim):
                    if k != c:
                        dso = max(dso, abs(s[k * 3 + c] - s_hat[k * 3 + c]))
    print()

    floor = "round-off" if args.solver == "direct" else f"the {args.solver} tolerance"
    diag = [k * 3 + k for k in range(dim)]
    off = [k * 3 + c for k in range(dim) for c in range(dim) if k != c]
    # the off-diagonal is zero for this field, so its magnitude says it all and
    # a row of zeros would only stretch the table
    worst_off = max(abs(got_s[k]) for k in off)
    x_last = mk.centroid(mesh, dim, model.n_cells - 1)
    rows = [
        ("displacement", exact_displacement(x_last, lo, gradient, dim), got_u, du),
        ("rotation", u_hat, got_r, dr),
        ("sigma diagonal", [s_hat[k] for k in diag], [got_s[k] for k in diag], ds),
        ("max |sigma_ij|", [0.0], [worst_off], dso),
    ]
    w = max(len(fmt(r[1])) for r in rows)
    print(f"  {'field':15s} {'exact':>{w}s} {'numerical (last cell)':>{w}s} {'max error':>11s}")
    for name, want, got, err in rows:
        print(f"  {name:15s} {fmt(want):>{w}s} {fmt(got):>{w}s} {err:11.3e}")
    print(f"\n  every error above is {floor}, not discretization")

    if args.vtu:
        with stage(f"writing {args.vtu}"):
            n = model.n_cells
            u = np.zeros((n, 3))
            u_exact = np.zeros((n, 3))
            rot = np.zeros((n, 3))
            rot_exact = np.zeros((n, 3))
            sig = np.zeros((n, 9))
            for e in range(n):
                want = exact_displacement(mk.centroid(mesh, dim, e), lo, gradient, dim)
                for k in range(dim):
                    u[e, k] = model.displacement(e, k)
                    u_exact[e, k] = want[k]
                for k in range(n_rot):
                    rot[e, k] = model.rotation(e, k)
                    rot_exact[e, k] = u_hat[k]
                sig[e] = model.cell_stress(e)
            mk.write_vtu(
                mesh,
                args.vtu,
                {
                    "displacement": u,
                    "displacement_exact": u_exact,
                    "displacement_error": u - u_exact,
                    "rotation": rot,
                    "rotation_exact": rot_exact,
                    "rotation_error": rot - rot_exact,
                    "stress": sig,
                    "stress_exact": np.tile(np.array(s_hat), (n, 1)),
                    "stress_error": sig - np.array(s_hat),
                },
            )


if __name__ == "__main__":
    main()
