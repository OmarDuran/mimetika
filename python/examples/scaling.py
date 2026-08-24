#!/usr/bin/env python3
"""How the solve scales, on a mesh that changes only in size.

A scaling study needs a mesh whose shape is fixed under refinement. The annulus
curves and grades, so refining it changes the conditioning as well as the
count, and a timing at two resolutions is then two different problems. A box
changes only the size.

    python scaling.py --n 16                      # 16^3 cells
    mpirun -n 4 python scaling.py --n 16          # the same problem, shared out
    python scaling.py --n 16 --family simplex     # six tetrahedra per cell
    python scaling.py --n 16 --physics elasticity

The problem is the linear patch test of the corresponding external-mesh
example: a linear field prescribed on the whole boundary, whose answer the
reconstructing products reproduce exactly. For those the error column is a
check, not a convergence study -- it must stay at the solver tolerance as the
mesh grows, and anything else means the scaling was measured on a wrong answer.

Two cases do not reproduce it, and there the error column is not that check.
diagonal_afw's linear moment slots are inconsistent on every mesh, so it is
never exact. And in FLOW, derham_bdm's datum is one scalar per facet, which
loads the constant moment alone and leaves the higher ones zero: exact on a
Cartesian box, first order on simplices. Mechanics has neither problem with
derham_bdm -- its displacement datum is affine, so both moments are supplied.
Both cases time the right operator on an inexact answer, which is what a
scaling study wants.

Defaults are deliberately small: a scaling curve is read from its shape, which
is visible long before a mesh becomes inconvenient. Raise --n when the times
are large enough to compare.
"""

import argparse
import os
import sys

import mimetika_cxx as mk
import numpy as np

import _realizations as rz

FAMILIES = {"cartesian": mk.Family.cartesian, "simplex": mk.Family.simplex,
            "prism": mk.Family.prism}

# The product is part of the measurement. Two realizations of
# the same space cost very different amounts -- diagonal_tpfa's M is diagonal
# where stabilized_rt's is dense per cell -- so a scaling curve belongs to a
# product as much as to a mesh.
FLUX_PRODUCTS = {
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
}
# The realizations, product and formulation together: a `_total` name is the
# four-field form. Timing the two separately is the point here -- four fields
# adds a scalar per cell and a row to the pairing, so it is a different cost
# curve and not a variant of the same one.
STRESS_PRODUCTS = rz.STRESS
SOLVERS = ("direct", "riesz", "ads")
MU, LAM = 1.0, 1.0


def solver_options(name, rtol, block_its, block_rtol):
    """The solver, and how hard the block is solved.

    The Riesz map needs the first block inverted, not solved: what the theory
    asks of it is spectral equivalence, and an inner Krylov run to a tight
    tolerance buys an accuracy the outer iteration cannot use. `block_its = 0`
    applies one ADS cycle as a fixed operator -- the cheapest thing that is
    still a Riesz map -- and anything larger is an inner CG, which makes the
    preconditioner vary and promotes the outer method to FGMRES.
    """
    if name == "direct":
        return mk.SolverOptions()
    common = dict(method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000)
    if name == "ads":
        inner = dict(riesz_block_its=block_its, riesz_block_rtol=block_rtol) if block_its else {}
        return mk.SolverOptions(riesz_block_pc="ads", riesz_exact_limit=10**9, **inner, **common)
    return mk.SolverOptions(**common)


def flow(mesh, dim, lo, direction, length, product):
    """p = ((x - x_min).n)/L on every boundary facet."""
    model = mk.FlowModel(mesh, dim, 1.0, FLUX_PRODUCTS[product])
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, f)
        model.add_pressure([f], float(np.dot(np.asarray(x)[:dim] - lo[:dim], direction) / length))
    return model


def elasticity(mesh, dim, lo, direction, length, product):
    """u = (x - x_min)/L on every boundary facet, as an affine datum."""
    how, form = rz.resolve(product)
    model = mk.CauchyMechanicsModel(mesh, dim, mk.ElasticMaterial(MU, LAM), how, form)
    gradient = [0.0] * 9
    for k in range(dim):
        gradient[k * 3 + k] = 1.0 / length
    facets = mk.boundary_facets(mesh, dim)
    for f, cell in zip(facets, mk.cofacets_of(mesh, dim, facets)):
        x = mk.centroid(mesh, dim, int(cell))
        constant = [float((x[k] - lo[k]) / length) for k in range(dim)] + [0.0] * (3 - dim)
        model.prescribe_displacement([f], constant, gradient)
    return model


def error_of(physics, model, mesh, dim, lo, direction, length):
    """The worst departure from the field that was prescribed."""
    if physics == "flow":
        return max(
            abs(model.cell_pressure(e)
                - float(np.dot(np.asarray(mk.centroid(mesh, dim, e))[:dim] - lo[:dim], direction)
                        / length))
            for e in range(model.n_cells))
    return max(
        abs(model.displacement(e, k) - float((mk.centroid(mesh, dim, e)[k] - lo[k]) / length))
        for e in range(model.n_cells) for k in range(dim))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=12, help="cells along each axis")
    ap.add_argument("--dim", type=int, default=3, choices=(2, 3))
    ap.add_argument("--family", default="cartesian", choices=sorted(FAMILIES))
    ap.add_argument("--physics", default="flow", choices=("flow", "elasticity"))
    ap.add_argument("--product", default=None,
                    help="flux or stress realization; the default is the lowest-order "
                         "stabilized one of the physics chosen. A stress name ending "
                         "_total is the four-field form")
    ap.add_argument("--formulation", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVERS))
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--block-its", type=int, default=0,
                    help="inner CG iterations on the Riesz block; 0 applies one cycle")
    ap.add_argument("--block-rtol", type=float, default=1e-2,
                    help="tolerance of that inner CG")
    ap.add_argument("--vtu", help="write the partition and the solution here")
    args = ap.parse_args()
    rz.reject_formulation_flag(args.formulation)

    root = mk.mpi_rank() == 0
    if not root:
        sys.stdout = open(os.devnull, "w")

    n = (args.n, args.n, args.n if args.dim == 3 else 1)
    mesh = mk.box(n, args.dim, FAMILIES[args.family])
    lo = np.zeros(3)
    direction = np.ones(args.dim) / np.sqrt(args.dim)
    length = 1.0

    known = FLUX_PRODUCTS if args.physics == "flow" else STRESS_PRODUCTS
    product = args.product or ("stabilized_rt" if args.physics == "flow" else "stabilized_bdm")
    if product not in known:
        raise SystemExit(f"--product {product} is not a {args.physics} realization; "
                         f"choose from {', '.join(sorted(known))}")
    if args.physics == "flow":
        model = flow(mesh, args.dim, lo, direction, length, product)
    else:
        model = elasticity(mesh, args.dim, lo, direction, length, product)
    report = model.solve(
        options=solver_options(args.solver, args.rtol, args.block_its, args.block_rtol))
    error = error_of(args.physics, model, mesh, args.dim, lo, direction, length)

    block = f", block {report.block_solver}" if report.block_solver else ""
    print(f"  {args.physics}, {product}, {args.family}, {args.solver}{block}, "
          f"{mk.mpi_size()} process(es)")
    print(f"  {mesh.count(args.dim)} cells, {model.n_dofs} dofs")
    print(f"  {'assembly':<16}{report.assembly_seconds:8.2f} s")
    print(f"  {'matrix':<16}{report.matrix_seconds:8.2f} s")
    print(f"  {'preconditioner':<16}{report.preconditioner_seconds:8.2f} s")
    print(f"  {'iteration':<16}{report.solve_seconds:8.2f} s   {report.iterations} iterations")
    if mk.mpi_size() > 1:
        print(f"  {'off-rank':<16}{100 * report.off_rank_fraction:8.1f} %")
    print(f"  {'max error':<16}{error:8.1e}")

    if args.vtu and root:
        fields = {"rank": mk.cell_ranks(mesh, args.dim, max(mk.mpi_size(), 2)).astype(float)}
        if args.physics == "flow":
            fields["pressure"] = np.array([model.cell_pressure(e) for e in range(model.n_cells)])
        mk.write_vtu(mesh, args.vtu, fields)
        print(f"  wrote {args.vtu}")


if __name__ == "__main__":
    main()
