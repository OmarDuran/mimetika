#!/usr/bin/env python3
"""Single-phase Darcy flow on a mesh read from a .vtu: the linear patch test.

The Dupuit example builds its own annulus and compares against a logarithmic
closed form. This one takes the mesh as DATA and so cannot know a closed form
for it. It prescribes one instead:

    p(x) = ((x - x_min) . n) / L ,   n = diag / L ,   L = |diag|

the diagonal of the bounding box, running from 0 at the low corner of the box
to 1 at the high one, imposed on EVERY boundary facet. Pure Dirichlet: no flux
is given anywhere, so the pressure data alone fixes the solution.

A linear pressure is the one field whose answer is known on any mesh. Darcy
sends it to a CONSTANT flux, q = -K grad p, which the lowest-order flux space
contains exactly; the cell unknown is a cell average, and the average of a
linear function is its value at the centroid. So the computed pressure should
equal p(x_E) to round-off, on a good mesh and a bad one alike.

    THAT HOLDS FOR derham_rt AND stabilized_rt, AND NOT FOR derham_bdm.

The reason is the datum, not the method. A prescribed pressure is natural here,
carried per facet as a single number, and the term places it entirely on the
CONSTANT moment of the facet's flux. RT has exactly one moment per facet, so
that is the whole datum and nothing is lost. BDM has `dim` of them, and the
linear part of p across each facet belongs to the higher moments -- which
receive zero. The mechanics side does not have this problem: its displacement
datum is affine, u = a + B (x - x_E), so both moments are supplied.

The resulting error converges at first order, so it reads as a discretization
error and is not one. The example reports it rather than hiding it: run with
--product derham_bdm to see it.

Run it:

    PYTHONPATH=.. python single_phase_flow_external_mesh.py --mesh domain.vtu
    PYTHONPATH=.. python single_phase_flow_external_mesh.py --mesh domain.vtu --vtu out.vtu

with --make-mesh to produce a sample file first, if there is none to hand:

    PYTHONPATH=.. python single_phase_flow_external_mesh.py --make-mesh domain.vtu
"""

import argparse
import math

import mimetika_cxx as mk
import numpy as np

from _stages import stage

MOBILITY = 1.0

# THE LINEAR SOLVER. "riesz" is the Riesz map of H(div) x L^2 -- P is the Gram
# matrix of ||q||^2 = (K^-1 q,q) + ||div q||^2 and ||p||^2 = ||p||_L2^2 -- so
# its iteration count does not grow with the mesh. "direct" is a full
# factorization: exact, and the wrong instrument past a few hundred thousand
# unknowns.
def solvers(rtol):
    """The linear solvers, at the residual tolerance asked for.

    "riesz" is the Riesz map of the space the operator is an isomorphism on --
    P is the Gram matrix of its norm -- so its iteration count does not grow
    with the mesh. "direct" is a full factorization: exact, and the wrong
    instrument past a few hundred thousand unknowns.

    THE TOLERANCE IS ON THE RESIDUAL, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        "direct": mk.SolverOptions(),
    }


SOLVER_NAMES = ("direct", "riesz")
DEFAULT_RTOL = 1e-12


PRODUCTS = {
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
}


def bounding_box(mesh):
    """The box the mesh occupies, its diagonal, and the characteristic length."""
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = (hi - lo)[: mesh.dim]
    length = float(np.linalg.norm(diag))
    return lo, hi, diag / length, length


def exact(x, lo, direction, length, dim):
    return float(np.dot(np.asarray(x)[:dim] - lo[:dim], direction) / length)


def prescribe_linear_pressure(model, mesh, dim, lo, direction, length):
    """p on every boundary facet, at the value the field takes at its centroid.

    The datum is one number per facet, so the value that belongs there is the
    facet average -- and for a linear field that is exactly the centroid value.
    """
    facets = mk.boundary_facets(mesh, dim)
    for f in facets:
        x_f = mk.centroid(mesh, dim - 1, f)
        model.add_pressure([f], exact(x_f, lo, direction, length, dim))
    return len(facets)


def make_mesh(path):
    """A sample .vtu, so the example runs without a mesh from somewhere else."""
    mesh = mk.annulus(8, 4, 2, mk.Family.simplex, 1.0, 10.0, 1.0)
    mk.write_vtu(mesh, path)
    print(f"wrote {path}: {mesh.count(2)} cells, {mesh.count(0)} vertices")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", help="the .vtu to solve on")
    ap.add_argument("--make-mesh", help="write a sample .vtu to this path and stop")
    ap.add_argument("--product", default="stabilized_rt", choices=sorted(PRODUCTS))
    ap.add_argument("--vtu", help="write the solution to this .vtu")
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVER_NAMES))
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                    help="residual tolerance of the iterative solver")
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
        lo, hi, direction, length = bounding_box(mesh)
    how = PRODUCTS[args.product]

    print(f"{args.mesh}: {dim}D, {mesh.count(dim)} cells, {mesh.count(0)} vertices")
    print(
        f"  box  {np.array2string(lo[:dim], precision=4)} .. "
        f"{np.array2string(hi[:dim], precision=4)}"
    )
    print(f"  characteristic length L = {length:.6g}  (the box diagonal)")
    print(f"  mobility = {MOBILITY}")
    print(f"  {mk.flux_realization_name(how)}, {args.solver} solver\n")

    with stage("building the flux operators"):
        model = mk.SinglePhaseModel(mesh, dim, MOBILITY, how)
    with stage("prescribing p on the boundary"):
        n_facets = prescribe_linear_pressure(model, mesh, dim, lo, direction, length)
    report = model.solve(progress=True, options=solvers(args.rtol)[args.solver])
    if args.solver != "direct":
        print(f"  {args.solver}: {report.iterations} iterations, {report.reason}")
    print(
        f"\n  p = ((x - x_min).n)/L on all {n_facets} boundary facets, pure Dirichlet"
    )
    print(
        f"  {model.n_cells} cells, {model.n_dofs} dofs, "
        f"{model.moments_per_facet} moment(s) per facet\n"
    )

    with stage("measuring the pressure error"):
        worst = rms = 0.0
        for e in range(model.n_cells):
            x = mk.centroid(mesh, dim, e)
            d = model.cell_pressure(e) - exact(x, lo, direction, length, dim)
            worst = max(worst, abs(d))
            rms += d * d
        rms = math.sqrt(rms / model.n_cells)
    print()
    print(f"  max |p - p_exact|  {worst:11.3e}")
    print(f"  rms |p - p_exact|  {rms:11.3e}")

    # THE VERDICT IS THE MEASUREMENT, not the moment count. One moment per facet
    # means the datum loses nothing; it does not by itself mean the answer is
    # exact, and on a general polyhedral mesh it is not.
    # an iterative solve cannot show the round-off floor, so the threshold is
    # the one that solver can actually reach
    exact_below = 1e-10 if args.solver == "direct" else 1e-7
    if worst < exact_below:
        print(
            "\n  exact: the datum is a facet constant and the facet carries one moment"
        )
    elif model.moments_per_facet == 1:
        print(
            f"\n  NOT exact ({worst:.3e}), and the datum is not the reason: the facet"
        )
        print("  carries one moment, so nothing of it was dropped. The linear field is")
        print("  not being reproduced on this mesh.")
    else:
        print(
            f"\n  NOT exact: the facet carries {model.moments_per_facet} moments and the datum"
        )
        print("  supplies only the constant one, so the linear part of p across each")
        print(
            "  facet is dropped. First order, and it looks like a discretization error."
        )

    if args.vtu:
        with stage(f"writing {args.vtu}"):
            n = model.n_cells
            p = np.array([model.cell_pressure(e) for e in range(n)])
            q = np.array(
                [
                    exact(mk.centroid(mesh, dim, e), lo, direction, length, dim)
                    for e in range(n)
                ]
            )
            mk.write_vtu(
                mesh, args.vtu, {"pressure": p, "pressure_exact": q, "error": p - q}
            )


if __name__ == "__main__":
    main()
