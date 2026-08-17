#!/usr/bin/env python3
"""THE ANSWER MUST NOT DEPEND ON HOW MANY PROCESSES SOLVED IT.

This is the first step of distributing the solver, and the only property worth
testing at it: with the system distributed across MPI_COMM_WORLD, every number
the model reports has to be what one process reports, to round-off.

WHAT IS DISTRIBUTED HERE and what is not. PETSc chooses a contiguous range of
rows per rank; the matrix, the preconditioner, the vectors, the Krylov method
and the field split all live on that layout, and the solve is genuinely
parallel. ASSEMBLY IS STILL REPLICATED -- every rank builds the whole triplet
list and inserts only its own rows -- and the solution is gathered back to
every rank at the end. So the mesh partition, the ghost exchange and the
owned-cell assembly are NOT exercised, and are not meant to be: this isolates
the layout, so a disagreement here can only be the layout.

The iteration count is checked as strictly as the answer. It is the sharper
instrument of the two: a field split whose index sets are wrong on some rank
still converges, to the right answer, in a different number of steps.

    python mpi_layout.py --write ref.json      # one process, the reference
    mpirun -n 2 python mpi_layout.py --check ref.json
"""

import argparse
import json
import math
import sys

import mimetika_cxx as mk

A_IN, B_OUT = 1.0, 10.0
P_IN, P_OUT = 2.0, 1.0
MU, LAM = 1.0, 1.0

RIESZ = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000
)
DIRECT = mk.SolverOptions()


def dupuit(nr, dim=2):
    """Sealed symmetry planes and prescribed pressures: constraints and data both."""
    mesh = mk.annulus(nr, nr // 2, dim, mk.Family.simplex, A_IN, B_OUT, 1.0)
    model = mk.SinglePhaseModel(mesh, dim, 1.0, mk.FluxRealization.derham_rt)
    rmid = math.sqrt(A_IN * B_OUT)
    inner, outer, sealed = [], [], []
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, f)
        on_symmetry = (
            abs(x[0]) < 1e-8
            or abs(x[1]) < 1e-8
            or (dim == 3 and (abs(x[2]) < 1e-8 or abs(x[2] - 1.0) < 1e-8))
        )
        if on_symmetry:
            sealed.append(f)
        elif math.hypot(x[0], x[1]) < rmid:
            inner.append(f)
        else:
            outer.append(f)
    model.add_normal_flux(sealed)
    model.add_pressure(inner, P_IN)
    model.add_pressure(outer, P_OUT)
    return model, mesh


def patch(nr, dim=2):
    """A linear displacement on the whole boundary: the elasticity saddle point."""
    mesh = mk.annulus(nr, nr // 2, dim, mk.Family.simplex, A_IN, B_OUT, 1.0)
    pts = [mesh.point(v) for v in range(mesh.count(0))]
    lo = [min(p[k] for p in pts) for k in range(3)]
    length = max(max(p[k] for p in pts) - lo[k] for k in range(dim))
    model = mk.CauchyElasticityModel(
        mesh, dim, mk.ElasticMaterial(MU, LAM), mk.StressRealization.stabilized_afw
    )
    gradient = [0.0] * 9
    for k in range(dim):
        gradient[k * 3 + k] = 1.0 / length
    for f in mk.boundary_facets(mesh, dim):
        x_e = mk.centroid(mesh, dim, mk.cofacet_of(mesh, dim, f))
        constant = [0.0] * 3
        for k in range(dim):
            constant[k] = (x_e[k] - lo[k]) / length
        model.prescribe_displacement([f], constant, gradient)
    return model, mesh


# Each case reports numbers that depend on EVERY unknown -- the two norms below
# are sums over all cells -- so an unknown left unpreconditioned or counted
# twice on some rank cannot hide behind a spot check.
def flow_case(nr, solver):
    model, mesh = dupuit(nr)
    report = model.solve(options=solver)
    p = [model.cell_pressure(e) for e in range(model.n_cells)]
    return {
        "dofs": model.n_dofs,
        "iterations": report.iterations,
        "sum": sum(p),
        "sum_sq": sum(v * v for v in p),
        "first": p[0],
        "last": p[-1],
    }


def elasticity_case(nr, solver):
    model, mesh = patch(nr)
    report = model.solve(options=solver)
    u = [model.displacement(e, k) for e in range(model.n_cells) for k in range(2)]
    return {
        "dofs": model.n_dofs,
        "iterations": report.iterations,
        "sum": sum(u),
        "sum_sq": sum(v * v for v in u),
        "first": u[0],
        "last": u[-1],
    }


# ADS is a 3D construction and needs one unknown per facet, so its case is the
# 3D annulus rather than the plane one the others use.
ADS = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000,
    riesz_block_pc="ads", riesz_exact_limit=10**9,
)


def flow_3d_case(nr, solver):
    model, mesh = dupuit(nr, dim=3)
    report = model.solve(options=solver)
    p = [model.cell_pressure(e) for e in range(model.n_cells)]
    return {
        "dofs": model.n_dofs,
        "iterations": report.iterations,
        "sum": sum(p),
        "sum_sq": sum(v * v for v in p),
        "first": p[0],
        "last": p[-1],
    }


CASES = {
    "flow-direct": lambda: flow_case(12, DIRECT),
    "flow-riesz": lambda: flow_case(12, RIESZ),
    "elasticity-direct": lambda: elasticity_case(8, DIRECT),
    "elasticity-riesz": lambda: elasticity_case(8, RIESZ),
    "flow-3d-ads": lambda: flow_3d_case(8, ADS),
}

# THE ONE CASE WHOSE COUNT MAY MOVE. ADS is hypre's, and the algebraic
# multigrid inside it coarsens the auxiliary spaces from the matrix as it is
# distributed -- so its hierarchy, and its iteration count, depend a little on
# the number of ranks. That is the solver's own business; the ANSWER is not
# allowed to move, and is compared as strictly as every other case.
LOOSE_COUNTS = {"flow-3d-ads": 15}


def run():
    out = {}
    for name, case in CASES.items():
        try:
            out[name] = case()
        except RuntimeError as e:  # a PETSc without hypre has no ADS to run
            if "ads" not in name:
                raise
            print(f"  {name}: unavailable ({str(e)[:60]})")
    return out


def compare(got, want, tol=1e-9):
    """Every number, on every case. The counts must match exactly."""
    bad = []
    for name, ref in want.items():
        if name not in got:
            bad.append(f"{name}: missing")
            continue
        mine = got[name]
        slack = LOOSE_COUNTS.get(name, 0)
        if abs(mine["iterations"] - ref["iterations"]) > slack:
            bad.append(
                f"{name}: {mine['iterations']} iterations against {ref['iterations']} "
                "on one process -- the split differs, not just the arithmetic"
            )
        if mine["dofs"] != ref["dofs"]:
            bad.append(f"{name}: {mine['dofs']} dofs against {ref['dofs']}")
        for key in ("sum", "sum_sq", "first", "last"):
            scale = max(1.0, abs(ref[key]))
            if abs(mine[key] - ref[key]) > tol * scale:
                bad.append(f"{name}: {key} {mine[key]:.17g} against {ref[key]:.17g}")
    return bad


# THE PARTITION IS A PROPERTY OF THE MATRIX, not of the clock. What it does is
# put a rank's unknowns next to each other, and the measure of that is how much
# of the matrix has columns another rank owns -- which a mat-vec must
# communicate. Timing it on a laptop measures the laptop; this does not.
def locality(nr=16):
    out = {}
    for name, on in (("index", False), ("partition", True)):
        model, _ = dupuit(nr)
        options = mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000,
            partition=on,
        )
        report = model.solve(options=options)
        out[name] = report.off_rank_fraction
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", help="solve and write the reference to this file")
    ap.add_argument("--check", help="solve and compare against this file")
    ap.add_argument(
        "--expect-ranks",
        type=int,
        help="fail unless PETSC_COMM_WORLD has exactly this many processes",
    )
    args = ap.parse_args()

    # A LAUNCHER THAT DOES NOT MATCH THE RUNTIME GIVES SINGLETONS. Each process
    # then believes it is alone, takes the sequential path, solves the whole
    # problem and agrees with the reference perfectly -- a passing test that
    # exercised nothing. So the rank count is asserted before anything else.
    size = mk.mpi_size()
    if args.expect_ranks is not None and size != args.expect_ranks:
        print(
            f"MPI_COMM_WORLD has {size} process(es), expected {args.expect_ranks}: "
            "the launcher and the MPI the extension loaded do not match, so this "
            "would have tested the sequential path on every rank",
            file=sys.stderr,
        )
        sys.exit(2)

    got = run()
    if args.write:
        with open(args.write, "w") as f:
            json.dump(got, f, indent=1)
        print(f"wrote {args.write}")
        return
    if not args.check:
        ap.error("--write or --check is required")

    with open(args.check) as f:
        want = json.load(f)
    bad = compare(got, want)
    for name in CASES:
        print(f"  {name:20s} {got[name]['dofs']:7d} dofs {got[name]['iterations']:5d} its")
    print(f"\n  solved on {size} process(es)")
    if size > 1:
        share = locality()
        print(
            f"  off-rank entries: {100 * share['index']:.1f}% split by index, "
            f"{100 * share['partition']:.1f}% partitioned"
        )
        if share["partition"] > 0.1 or share["partition"] >= share["index"]:
            bad.append(
                f"the partition did not localize the matrix: {share['partition']:.3f} "
                f"off-rank against {share['index']:.3f} split by index"
            )
    if bad:
        print("\nDISAGREES WITH THE SERIAL SOLVE:")
        for line in bad:
            print("  " + line)
        sys.exit(1)
    print("\nevery case agrees with the serial solve, iteration counts included")


if __name__ == "__main__":
    main()
