#!/usr/bin/env python3
"""Flow, of Darcy type, on a mesh read from a .vtu: the linear patch test.

The Dupuit example builds its own annulus and compares against a logarithmic
closed form. This one takes the mesh as data and so cannot know a closed form
for it. It prescribes one instead:

    p(x) = ((x - x_min) . n) / L ,   n = diag / L ,   L = |diag|

the diagonal of the bounding box, running from 0 at the low corner of the box
to 1 at the high one, imposed on every boundary facet. Pure Dirichlet: no flux
is given anywhere, so the pressure data alone fixes the solution.

A linear pressure is the one field whose answer is known on any mesh. Darcy
sends it to a constant flux, q = -K grad p, which the lowest-order flux space
contains exactly; the cell unknown is a cell average, and the average of a
linear function is its value at the centroid. So the computed pressure should
equal p(x_E) to round-off, on a good mesh and a bad one alike.

    It holds for every product here, RT and BDM alike -- provided the
    datum is given as a FIELD and not as a number.

A prescribed pressure is natural here: it enters the flux row as
int_f p_D (tau.n), and a facet carrying d flux moments tests it against d basis
functions. The chart is equilibrated, so the facet Gram is |f| I and moment b
wants (1/|f|) int_f p_D phi_b. For RT there is one moment, the basis is the
constant, and a single number -- the facet average, which for a linear field is
the centroid value -- is the whole datum. For BDM the higher basis functions
are centred: they are blind to a constant and see exactly the VARIATION of p
across the facet. A number cannot express that, so `add_pressure` takes the
gradient too, and prescribe_linear_pressure passes it.

Given only the number, the BDM products lose the patch -- 2.9e-02 on a Kuhn box,
1.3e+00 on a simplicial annulus, against round-off with the gradient. That is
the datum, not the discretization: the same operator is exact once its rows are
told what the field does inside each facet.

Run it:

    PYTHONPATH=.. python flow_external_mesh.py --mesh domain.vtu
    PYTHONPATH=.. python flow_external_mesh.py --mesh domain.vtu --vtu out.vtu

with --make-mesh to produce a sample file first, if there is none to hand:

    PYTHONPATH=.. python flow_external_mesh.py --make-mesh domain.vtu
"""

import argparse
import os
import sys

import mimetika_cxx as mk

import _hypre
import numpy as np

from _diagnostics import write_report
from _errors import error_table, l2_norms
from _stages import stage

MOBILITY = 1.0


def solvers(rtol):
    """The linear solvers, at the residual tolerance asked for.

    "riesz" is the Riesz map of H(div) x L^2 -- P is the Gram matrix of
    ||q||^2 = (K^-1 q,q) + ||div q||^2 and ||p||^2 = ||p||_L2^2 -- so its
    iteration count does not grow with the mesh. "direct" is a full
    factorization: exact, and the wrong instrument past a few hundred thousand
    unknowns.

    "ads" and "ads-cg" are the same map with its first block inverted by an
    auxiliary-space solver instead of a factorization; "riesz" picks that route
    by itself once the block is large enough, so they are here to be measured
    against the default rather than to be needed.

    The tolerance is on the residual, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        # The same map, with the big block inverted by an auxiliary space.
        #
        # P's first block is H(div)-like and it is most of the unknowns. A
        # Cholesky of it is exact and creates fill, so the cost per iteration
        # grows with the mesh even though the count does not. ADS splits it
        # along the de Rham complex instead -- the discrete gradient and curl
        # are the complex's own boundary operators, handed over rather than
        # rebuilt -- and costs no fill: on a refinement to 96k unknowns it holds
        # 0.14 us per dof per iteration while icc goes 0.17 -> 1.4 and Cholesky
        # 0.11 -> 0.68.
        #
        # ADS is a 3D construction and is written for ONE unknown per facet
        # (derham_rt, stabilized_rt, and the eta = 1 cells of adaptive_rt). A
        # facet carrying d moments -- derham_bdm, stabilized_bdm -- is not
        # refused: it reaches ADS through the facet-constant subspace, as a
        # two-level cycle whose coarse operator is where ADS runs.
        "ads": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads",
        ),
        # a short CG under ADS rather than a single V-cycle: the block is then
        # solved to a tolerance instead of approximated once, and the outer
        # count stops drifting up with the mesh (60 -> 21 at 96k unknowns).
        # Each iteration costs more, so which of the two wins is a measurement.
        # the same cycle with the inner CG stated explicitly. The budget is
        # what MEASURES the map rather than the budget: at 50 steps to 1e-2 the
        # outer count reads the cap instead of the preconditioner -- 29 against
        # 23 on the h-ladder, 205 against 132 at nu = 0.4999 -- and on a mesh
        # written in metres rather than in unit lengths it does not converge at
        # all. Solved to 1e-6 the count is the Riesz map's.
        "ads-cg": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads", riesz_block_its=500, riesz_block_rtol=1e-6,
        ),
        "direct": mk.SolverOptions(),
    }


SOLVER_NAMES = ("direct", "riesz", "ads", "ads-cg", _hypre.NAME)
DEFAULT_RTOL = 1e-9


PRODUCTS = {
    "derham_bdm": mk.FluxRealization.derham_bdm,
    # The BDM order's other half: the same [P_1]^d space and the same d
    # moments per facet as derham_bdm, reached by PENALIZING the surplus
    # rather than by enriching to unisolvence. On a simplex the kernel is
    # empty, the stabilization vanishes and the two coincide with BDM_1; on a
    # polytope they are different operators. It is the flux the stabilized_bdm
    # stress is d copies of.
    "stabilized_bdm": mk.FluxRealization.stabilized_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    # One flux per facet and no reconstruction: M is the diagonal primal-dual
    # star, M_ff = (|sigma*|/|sigma|) / (n.K n), which is the two-point flux
    # approximation. The same space as derham_rt and stabilized_rt -- one
    # unknown per facet -- and a different operator: it is strongly consistent
    # only where the mesh is K-orthogonal, so on a box it reproduces a linear
    # pressure to round-off and on a curved or polytopal one it does not.
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
    # The per-cell selection between the two above, carried as eta in {0, 1}:
    # the stabilized product everywhere (eta = 1) and the diagonal star on the
    # cells the metric-degeneracy scan flags (eta = 0), which avoids a
    # reconstruction over a collapsed cell. The scan runs at exokal's default
    # threshold unless --degeneracy-percent names one; the selection as built
    # is the eta cell data written to the .vtu.
    "adaptive_rt": mk.FluxRealization.adaptive_rt,
}


def bounding_box(mesh):
    """The box the mesh occupies, its diagonal, and the characteristic length."""
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = (hi - lo)[: mesh.dim]
    length = float(np.linalg.norm(diag))
    return lo, hi, diag / length, length


# THE TWO FIELDS, in the non-dimensional s = (x - x_min)/L.
#
#   linear      p = s.n          grad p = n/L        q = -lambda n/L      f = 0
#   quadratic   p = |s|^2 L/2    grad p = s          q = -lambda s        f = -lambda d/L
#
# The linear one is a PATCH TEST: q is constant, the lowest-order space holds
# it exactly, and the method reproduces p to round-off. The quadratic one is
# not: q is linear, which RT_0 spans, but the cell pressure is constant and
# cannot follow a quadratic, so what is measured is convergence rather than
# reproduction -- and the source is what makes it a different problem at all,
# div q = f with f constant and nonzero.
def exact(x, lo, direction, length, dim, quadratic=False):
    s = (np.asarray(x)[:dim] - lo[:dim]) / length
    if quadratic:
        return float(0.5 * length * np.dot(s, s))
    return float(np.dot(s, direction))


def exact_flux_at(x, lo, direction, length, dim, quadratic=False):
    """q = -lambda K grad p at a point, K = I. Constant if p is affine."""
    if quadratic:
        return -MOBILITY * (np.asarray(x)[:dim] - lo[:dim]) / length
    return -MOBILITY * np.asarray(direction) / length


def source_density(length, dim, quadratic):
    """f = div q. Zero for the affine field, -lambda d / L for the quadratic."""
    return -MOBILITY * dim / length if quadratic else 0.0


def cell_fields(model, mesh, dim, lo, direction, length, quadratic=False):
    """|E|, p_h, Pi_0 p, Pi_0 q_h and the exact q, one row per cell.

    Everything downstream is a norm, so the loop only gathers: three calls into
    the model per cell and no arithmetic, which numpy then does at once.

    Pi_0 of an affine field is its centroid value, so the linear case needs no
    quadrature. The quadratic one is read at the centroid too -- that is Pi_0
    up to O(h^2), the order the method itself converges at, so it measures the
    same thing on every mesh in the ladder rather than a mesh-dependent mix.
    """
    n = model.n_cells
    x = np.array([mk.centroid(mesh, dim, e) for e in range(n)])
    volume = np.fromiter((mk.measure(mesh, dim, e) for e in range(n)), float, n)
    p_h = np.fromiter((model.cell_pressure(e) for e in range(n)), float, n)
    q_h = np.array([model.cell_flux(e) for e in range(n)])
    p = np.array([exact(xe, lo, direction, length, dim, quadratic) for xe in x])
    q = np.array([exact_flux_at(xe, lo, direction, length, dim, quadratic) for xe in x])
    return volume, p_h, p, q_h, q


def report_error(volume, p_h, p, q_h, q, dim, moments):
    """The table for u = p and u = q; returns the per-cell ||e||_{L2(E)} of each."""
    print()
    # With one moment per facet cell_flux IS Pi_0 q_h, so the flux row bounds
    # ||q - q_h||_D below; with d moments it pairs only the constant one and
    # carries a reconstruction error of its own.
    print("  e = Pi_0(u - u_h),   Pi_0 v|_E = |E|^-1 int_E v")
    print("  S:  p -> ||Pi_0 p||_D    q -> ||q||_D"
          + ("    (e_q <= ||q - q_h||_D)" if moments == 1
             else f"    ({moments} moments/facet: e_q also reconstructs)"))
    print()
    # Both scales are read off the exact field the same way, which is what lets
    # the two fields be compared on one table.
    scale_p, _ = l2_norms(volume, p)
    scale_q, _ = l2_norms(volume, q[:, :dim])
    return error_table(volume, [
        ("p", p_h - p, scale_p),
        ("q", q_h[:, :dim] - q[:, :dim], scale_q),
    ])


def prescribe_pressure(model, mesh, dim, lo, direction, length, quadratic=False):
    """p on every boundary facet: the centroid value AND the gradient.

    THE GRADIENT IS NOT DECORATION. The flux row of a boundary facet carries
    int_f p_D (tau.n), and a facet holding d flux moments tests that datum
    against d basis functions, not one. The centroid value is the whole datum
    only where the facet carries a single moment -- there the basis is the
    constant and the facet average is exactly the centroid value of a linear
    field. With d moments the higher basis functions are centred, so they see
    only the VARIATION of p across the facet, which the gradient supplies and a
    number cannot. Omit it and the BDM products lose the patch they reproduce
    exactly: 2.9e-02 on the Kuhn box against 8.4e-15 with it.
    """
    facets = mk.boundary_facets(mesh, dim)
    for f in facets:
        x_f = mk.centroid(mesh, dim - 1, f)
        if quadratic:
            # grad p = s, which MOVES: the datum is the field's linearization
            # about each facet, exact to O(h^2) -- the order the method
            # converges at, so it does not set the rate.
            g = [(float(x_f[k]) - lo[k]) / length if k < dim else 0.0 for k in range(3)]
        else:
            g = [float(direction[k]) / length if k < dim else 0.0 for k in range(3)]
        model.add_pressure([f], exact(x_f, lo, direction, length, dim, quadratic), g)
    return len(facets)


# The partition, written out with the answer. A partition is a picture: a rank
# holding a disconnected piece, or most of the mesh, is obvious in ParaView and
# invisible in a timing. `--partition N` previews an N-way split without
# running on N processes; with no argument it shows the one this run used.
def partition_field(mesh, dim, requested):
    parts = requested if requested else mk.mpi_size()
    if parts < 2:
        return {}
    return {"rank": mk.cell_ranks(mesh, dim, parts).astype(float)}


def make_mesh(path):
    """A sample .vtu, so the example runs without a mesh from somewhere else."""
    mesh = mk.annulus(8, 4, 2, mk.Family.simplex, 1.0, 10.0, 1.0)
    mk.write_vtu(mesh, path)
    print(f"wrote {path}: {mesh.count(2)} cells, {mesh.count(0)} vertices")


# One process speaks and writes. Under mpirun every rank runs this file and
# solves the same problem -- the algebra is shared out, the script is not -- so
# without this the report appears N times and N processes race to write the
# same .vtu. The solve itself is unaffected: every rank takes part in it, and
# every rank ends up with the whole answer.
# What the run is shared out over, said once rather than inferred from N copies
# of the output. The balance is the partition's own report: a bisection that
# has gone wrong shows up here as a rank holding most of the mesh, long before
# it shows up as a timing.
def report_processes(mesh, dim):
    size = mk.mpi_size()
    if size < 2:
        return
    ranks = mk.cell_ranks(mesh, dim, size)
    counts = np.bincount(ranks, minlength=size)
    print(f"  {size} processes, {counts.min()}..{counts.max()} cells each")


def only_root():
    if mk.mpi_rank() == 0:
        return True
    sys.stdout = open(os.devnull, "w")
    return False


def report_complex(mesh):
    """The complex's entity counts, stratum by stratum.

    These are what every degree-of-freedom formula reads from: the lowest-order
    flux space is f + c (one flux per facet, one pressure per cell), the BDM
    layer d*f + c. Reported once so a count in the output can be checked
    against the formula rather than trusted.
    """
    names = ["vertices", "edges", "faces", "cells"]
    parts = [
        f"{mesh.count(k)} {names[k] if k < mesh.dim else 'cells'}"
        for k in range(mesh.dim + 1)
    ]
    print("  complex: " + ", ".join(parts))


def main():
    root = only_root()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", help="the .vtu to solve on")
    ap.add_argument("--make-mesh", help="write a sample .vtu to this path and stop")
    ap.add_argument("--product", default="stabilized_rt", choices=sorted(PRODUCTS))
    ap.add_argument(
        "--partition",
        type=int,
        default=0,
        help="write an N-way mesh partition into the .vtu (default: the processes in use)",
    )
    ap.add_argument("--vtu", help="write the solution to this .vtu")
    ap.add_argument("--solver", default="riesz", choices=sorted(SOLVER_NAMES))
    # hypre-ads only. The default applies ONE ADS cycle per application, which
    # is cheapest where K is smooth; a jumping coefficient needs the block
    # solved instead -- one cycle stops converging past a contrast of ~1e4 --
    # and this is how many CG steps to allow under the cycle.
    ap.add_argument("--ads-block-its", type=int, default=None, metavar="N",
                    help="hypre-ads: CG steps on the flux block (default: one ADS "
                         "cycle). Use ~20 for a jumping coefficient")
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="hybridize: eliminate the flux cell by cell and solve the facet-pressure "
        "multiplier system, which is SPD -- conjugate gradients and multigrid, on any "
        "product. The boundary roles swap: a prescribed pressure PINS a multiplier, "
        "a normal flux loads a free row, an unconditioned facet is sealed. Serial only.",
    )
    ap.add_argument(
        "--output",
        default="",
        help="folder for the mesh diagnostics; they are off unless this is given",
    )
    ap.add_argument(
        "--degeneracy-percent",
        type=float,
        default=None,
        help="a cell is degenerate below this percent of its node-star mean; defaults to exokal's default_degeneracy_percent",
    )
    ap.add_argument(
        "--cond-threshold",
        type=float,
        default=None,
        help="adaptive_rt: a cell whose stabilized flux block has lambda_max/lambda_min above "
             "this takes the diagonal star as well (composes with --degeneracy-percent)",
    )
    ap.add_argument(
        "--field",
        default="linear",
        choices=("linear", "quadratic"),
        help="linear: p affine, q constant, no source -- the patch test the method "
             "reproduces exactly. quadratic: p = |x - x_min|^2 / 2L, so q is linear and "
             "the source div q = -lambda d / L is constant and NONZERO, which the "
             "constant cell pressure can only converge to",
    )
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="build the Jacobian and the preconditioner, and stop before the iteration",
    )
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
    if args.solver == _hypre.NAME and not _hypre.available():
        raise SystemExit(_hypre.why_unavailable())
    if (args.solver.startswith("ads") or args.solver == _hypre.NAME) and dim != 3:
        raise SystemExit(
            f"--solver {args.solver} is a 3D construction (it needs the discrete "
            f"gradient and curl of a 3-complex); {args.mesh} is {dim}D. "
            "Use --solver riesz."
        )

    if args.output and root:
        out = os.path.join(args.output, os.path.splitext(os.path.basename(args.mesh))[0])
        with stage(f"diagnosing the mesh into {out}"):
            bad, warn, degenerate = write_report(args.mesh, out, args.degeneracy_percent)
        print(
            f"  diagnostics: {bad} violation(s), {warn} warning(s), "
            f"{degenerate} degenerate cell(s)"
        )

    with stage("bounding box"):
        lo, hi, direction, length = bounding_box(mesh)
    how = PRODUCTS[args.product]

    print(f"{args.mesh}: {dim}D, {mesh.count(dim)} cells, {mesh.count(0)} vertices")
    report_complex(mesh)
    print(
        f"  box  {np.array2string(lo[:dim], precision=4)} .. "
        f"{np.array2string(hi[:dim], precision=4)}"
    )
    print(f"  characteristic length L = {length:.6g}  (the box diagonal)")
    print(f"  mobility = {MOBILITY}")
    # What actually runs, not what was asked for: --hybrid removes the H(div)
    # block a Riesz map would split along, so the interface system gets
    # conjugate gradients and an algebraic multigrid whatever --solver said.
    solver_name = "cg + boomeramg (hybridized)" if args.hybrid else f"{args.solver} solver"
    print(f"  {mk.flux_realization_name(how)}, {solver_name}")
    if args.hybrid and args.solver != "riesz":
        print(f"  note: --solver {args.solver} does not apply to the interface system "
              f"and is ignored")
    report_processes(mesh, dim)
    print()

    with stage("creating the model"):
        model = mk.FlowModel(mesh, dim, MOBILITY, how)
        # the threshold reaches the model only where it is the model's: for
        # any other product it stays what it always was, the diagnostics dial
        if args.product == "adaptive_rt" and args.degeneracy_percent is not None:
            model.set_degeneracy_percent(args.degeneracy_percent)
        if args.product == "adaptive_rt" and args.cond_threshold is not None:
            model.set_cond_threshold(args.cond_threshold)
    quadratic = args.field == "quadratic"
    with stage("prescribing p on the boundary"):
        n_facets = prescribe_pressure(model, mesh, dim, lo, direction, length, quadratic)
        # div q = f, the same constant in every cell. The count comes from the
        # MESH and not from the model: n_cells is filled by the build, which
        # has not run yet, so asking the model here silently sets no source at
        # all -- and the run then solves a different problem whose error does
        # not converge.
        f = source_density(length, dim, quadratic)
        if f != 0.0:
            model.set_source([f] * mesh.count(dim))
    if args.assemble_only:
        # the direct hypre path is not one of the PETSc option sets, so it
        # assembles through its own bridge rather than through model.assemble
        if args.solver == _hypre.NAME:
            report = _hypre.assemble(model, mesh, dim)
        else:
            report = model.assemble(progress=True, options=solvers(args.rtol)[args.solver])
        complex_note = " (degree 2, with Pi)" if getattr(report, "degree2", False) else ""
        print(
            f"\n  assembled: A, b {report.matrix_seconds:.2f} s;  "
            f"complex {report.preconditioner_seconds:.2f} s{complex_note}"
        )
        print(
            f"  {model.n_cells} cells, {model.n_dofs} dofs, "
            f"{model.moments_per_facet} moment"
            f"{'' if model.moments_per_facet == 1 else 's'} per facet"
        )
        if report.condensed:
            print(f"  flux eliminated exactly: {model.n_dofs} -> {report.condensed_dofs} "
                  f"unknowns; the preconditioner built is the reduced system's "
                  f"({report.block_solver})")
        return
    if args.hybrid:
        report = model.solve_hybrid(
            progress=True,
            options=mk.SolverOptions(
                method="cg", preconditioner="hypre", rtol=args.rtol, max_iterations=2000
            ),
        )
    elif args.solver == _hypre.NAME:
        report = _hypre.solve(model, mesh, dim, _hypre.options(args.rtol, block_iterations=args.ads_block_its))
    else:
        report = model.solve(progress=True, options=solvers(args.rtol)[args.solver])
    # The two assemblies, always. They are what scales with the mesh, and they
    # are separate costs: the Jacobian is the physics, the preconditioner is the
    # price of being able to solve it iteratively.
    print(
        f"\n  assembly: jacobian {report.assembly_seconds:.2f} s + matrix "
        f"{report.matrix_seconds:.2f} s, preconditioner "
        f"{report.preconditioner_seconds:.2f} s"
    )
    if args.hybrid and report.condensed_dofs:
        print(f"  the linear system solved: {report.condensed_dofs} facet multipliers "
              f"({model.n_dofs} in the mixed space, eliminated cell by cell)")
    label = "cg" if args.hybrid else args.solver
    if args.hybrid or args.solver != "direct":
        print(
            f"  {label}: {report.iterations} iterations to rtol {args.rtol:.1e}"
            f", {report.reason} in {report.solve_seconds:.2f} s"
        )
    print(
        f"\n  p = ((x - x_min).n)/L on all {n_facets} boundary facets, pure Dirichlet"
    )
    print(
        f"  {model.n_cells} cells, {model.n_dofs} dofs, "
        f"{model.moments_per_facet} moment{'' if model.moments_per_facet == 1 else 's'} per facet\n"
    )
    if args.product == "adaptive_rt":
        # the selection as built: how many cells the scan handed to the
        # diagonal star, which is the number that explains the error below
        n_star = int((model.eta == 0.0).sum())
        pct = args.degeneracy_percent
        print(
            f"  adaptive_rt: {n_star} cell(s) on the diagonal star, "
            f"{model.n_cells - n_star} on the stabilized product "
            f"(threshold {'default' if pct is None else f'{pct}%'}"
            + (f"; {model.n_ill_conditioned} switched by cond > {args.cond_threshold:g}"
               if args.cond_threshold is not None else "")
            + ")\n"
        )

    # The flux is reconstructed per cell, and the gather is collective.
    #
    # A rank reconstructs only the cells it owns, so the field is summed over
    # the processes -- and that sum is an MPI_Allreduce, which every rank has to
    # reach. It happens here, before the norms, because a rank measuring its own
    # partial flux would report an error that is an artefact of the partition;
    # and unconditionally, because putting it behind `--vtu` or `and root`
    # deadlocks the run -- rank 0 waits in the reduction for ranks that already
    # went on to exit.
    with stage("reconstructing p and q"):
        volume, p_h, p_exact, q_h, q_exact = cell_fields(model, mesh, dim, lo, direction,
                                                         length, quadratic)
        q_h = mk.gather_cells(model, q_h)
    cell_error = report_error(volume, p_h, p_exact, q_h, q_exact, dim,
                              model.moments_per_facet)

    if args.vtu and root:
        with stage(f"writing {args.vtu}"):
            # ONE TRIPLE PER UNKNOWN: the discrete field, the exact field it
            # is measured against, and the error -- so the .vtu carries the
            # error table rather than the material for recomputing it.
            #
            # THE ERROR FIELDS ARE THE TABLE'S ROWS, cell by cell:
            # ||e||_{L2(E)} = |E|^{1/2} |e_E|, whose extremes over the cells
            # are its min_E and max_E columns and whose l2 norm over the cells
            # is its ||e||_D. So thresholding on pressure_error in ParaView
            # lands exactly on the cells the table's maximum came from.
            #
            # They carry |E|^{1/2} because a norm over a cell does, which is
            # why a small cell can sit low in that column whatever its error;
            # `volume` is written so the pointwise error can be recovered by
            # dividing it out.
            #
            # The exact flux is one constant vector, broadcast: written per
            # cell so that flux and flux_exact are the same kind of field in
            # ParaView and can be differenced or glyphed against each other.
            exact_rows = np.zeros((len(p_h), 3))
            exact_rows[:, :dim] = q_exact
            fields = {
                "pressure": p_h,
                "pressure_exact": p_exact,
                "pressure_error": cell_error["p"],
                "flux": q_h,
                "flux_exact": exact_rows,
                "flux_error": cell_error["q"],
                "volume": volume,
            }
            if args.product == "adaptive_rt":
                fields["eta"] = model.eta
            fields.update(partition_field(mesh, dim, args.partition))
            mk.write_vtu(mesh, args.vtu, fields)


if __name__ == "__main__":
    main()
