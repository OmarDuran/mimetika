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
import os
import sys

import mimetika_cxx as mk
import numpy as np

from _diagnostics import write_report
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

    "ads" and "ads-cg" are the same map with its first block inverted by an
    auxiliary-space solver instead of a factorization; "riesz" picks that route
    by itself once the block is large enough, so they are here to be MEASURED
    against the default rather than to be needed.

    THE TOLERANCE IS ON THE RESIDUAL, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        # THE SAME MAP, WITH THE BIG BLOCK INVERTED BY AN AUXILIARY SPACE.
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
        # Only for ONE UNKNOWN PER FACET in 3D (derham_rt, stabilized_rt),
        # which is the space ADS is written for.
        "ads": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads",
        ),
        # a short CG under ADS rather than a single V-cycle: the block is then
        # SOLVED to a tolerance instead of approximated once, and the outer
        # count stops drifting up with the mesh (60 -> 21 at 96k unknowns).
        # Each iteration costs more, so which of the two wins is a measurement.
        "ads-cg": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads", riesz_block_its=50, riesz_block_rtol=1e-2,
        ),
        "direct": mk.SolverOptions(),
    }


SOLVER_NAMES = ("direct", "riesz", "ads", "ads-cg")
DEFAULT_RTOL = 1e-9


PRODUCTS = {
    "derham_bdm": mk.FluxRealization.derham_bdm,
    "derham_rt": mk.FluxRealization.derham_rt,
    "stabilized_rt": mk.FluxRealization.stabilized_rt,
    # ONE FLUX PER FACET AND NO RECONSTRUCTION: M is the diagonal primal-dual
    # star, M_ff = (|sigma*|/|sigma|) / (n.K n), which is the two-point flux
    # approximation. The same space as derham_rt and stabilized_rt -- one
    # unknown per facet -- and a different operator: it is strongly consistent
    # only where the mesh is K-ORTHOGONAL, so on a box it reproduces a linear
    # pressure to round-off and on a curved or polytopal one it does not.
    "diagonal_tpfa": mk.FluxRealization.diagonal_tpfa,
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


# THE PARTITION, WRITTEN OUT WITH THE ANSWER. A partition is a picture: a rank
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


# ONE PROCESS SPEAKS AND WRITES. Under mpirun every rank runs this file and
# solves the same problem -- the algebra is shared out, the script is not -- so
# without this the report appears N times and N processes race to write the
# same .vtu. The solve itself is unaffected: every rank takes part in it, and
# every rank ends up with the whole answer.
# WHAT THE RUN IS SHARED OUT OVER, said once rather than inferred from eight
# copies of the output. The balance is the partition's own report: a bisection
# that has gone wrong shows up here as a rank holding most of the mesh, long
# before it shows up as a timing.
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
    ap.add_argument(
        "--output",
        default="diagnostics",
        help="folder for the mesh diagnostics; empty string to skip them",
    )
    ap.add_argument(
        "--degeneracy-percent",
        type=float,
        default=0.1,
        help="a cell is degenerate below this percent of its neighborhood mean measure",
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
    if args.solver.startswith("ads") and dim != 3:
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
    print(
        f"  box  {np.array2string(lo[:dim], precision=4)} .. "
        f"{np.array2string(hi[:dim], precision=4)}"
    )
    print(f"  characteristic length L = {length:.6g}  (the box diagonal)")
    print(f"  mobility = {MOBILITY}")
    print(f"  {mk.flux_realization_name(how)}, {args.solver} solver")
    report_processes(mesh, dim)
    print()

    with stage("creating the model"):
        model = mk.SinglePhaseModel(mesh, dim, MOBILITY, how)
    with stage("prescribing p on the boundary"):
        n_facets = prescribe_linear_pressure(model, mesh, dim, lo, direction, length)
    if args.assemble_only:
        report = model.assemble(progress=True, options=solvers(args.rtol)[args.solver])
        print(
            f"\n  assembled: matrix {report.matrix_seconds:.2f} s, "
            f"preconditioner {report.preconditioner_seconds:.2f} s"
        )
        print(
            f"  {model.n_cells} cells, {model.n_dofs} dofs, "
            f"{model.moments_per_facet} moment(s) per facet"
        )
        return
    report = model.solve(progress=True, options=solvers(args.rtol)[args.solver])
    # THE TWO ASSEMBLIES, ALWAYS. They are what scales with the mesh, and they
    # are separate costs: the Jacobian is the physics, the preconditioner is the
    # price of being able to solve it iteratively.
    print(
        f"\n  assembly: jacobian {report.assembly_seconds:.2f} s + matrix "
        f"{report.matrix_seconds:.2f} s, preconditioner "
        f"{report.preconditioner_seconds:.2f} s"
    )
    if args.solver != "direct":
        print(
            f"  {args.solver}: {report.iterations} iterations to rtol {args.rtol:.1e}"
            f", {report.reason} in {report.solve_seconds:.2f} s"
        )
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
    elif args.product == "diagonal_tpfa":
        # NOT A DEFECT, AND NOT THE DATUM: the two-point flux reconstructs
        # nothing, and is strongly consistent only where the segment joining two
        # cell centroids meets their shared facet squarely. Off that condition
        # it loses the linear field, which is the price of a diagonal M.
        print(f"\n  NOT exact ({worst:.3e}), and this is what diagonal_tpfa claims: the")
        print("  two-point flux is consistent only where the mesh is K-ORTHOGONAL, and")
        print("  this one is not. Run --product stabilized_rt on the same mesh to see")
        print("  the same space reproduce the field exactly.")
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

    # THE FLUX IS RECONSTRUCTED PER CELL, AND THE GATHER IS COLLECTIVE.
    #
    # A rank reconstructs only the cells it owns, so the field is summed over
    # the processes -- and that sum is an MPI_Allreduce, which EVERY rank has to
    # reach. Computing it inside the `and root` branch below deadlocks the run:
    # rank 0 waits in the reduction for ranks that already went on to exit.
    flux = None
    if args.vtu:
        flux = mk.gather_cells(
            model, np.array([model.cell_flux(e) for e in range(model.n_cells)])
        )

    if args.vtu and root:
        with stage(f"writing {args.vtu}"):
            n = model.n_cells
            p = np.array([model.cell_pressure(e) for e in range(n)])
            q = np.array(
                [
                    exact(mk.centroid(mesh, dim, e), lo, direction, length, dim)
                    for e in range(n)
                ]
            )
            fields = {"pressure": p, "pressure_exact": q, "error": p - q, "flux": flux}
            fields.update(partition_field(mesh, dim, args.partition))
            mk.write_vtu(mesh, args.vtu, fields)


if __name__ == "__main__":
    main()
