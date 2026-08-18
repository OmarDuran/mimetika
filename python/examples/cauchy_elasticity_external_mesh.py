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
import os
import sys

import mimetika_cxx as mk
import numpy as np

from _diagnostics import write_report
from _stages import stage

MU, LAM = 1.0, 1.0

# THE LINEAR SOLVER. "riesz" is the Riesz map of the space the operator is an
# isomorphism on -- P is the Gram matrix of its norm -- so its iteration count
# does not grow with the mesh. "direct" is a full factorization: exact, and the
# wrong instrument past a few hundred thousand unknowns.
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
        # THE SAME MAP, WITH THE STRESS BLOCK INVERTED BY AN AUXILIARY SPACE.
        #
        # P's stress block is H(div; M) and it is most of the unknowns; a
        # Cholesky of it is exact and creates fill, so its cost per iteration
        # grows with the mesh (0.62 -> 8.0 us per dof per iteration over a
        # refinement to 100k unknowns) even though the iteration count does
        # not.
        #
        # ADS wants ONE unknown per facet and the stabilized_bdm facet carries
        # d^2 -- d traction components, each against the d functions of the
        # facet P_1 basis. The route to it is the facet-CONSTANT subspace:
        # those moments are a subset of the unknowns, so the injection into
        # them is exact, they are where the divergence lives, and what is left
        # over is local to a facet and belongs to a smoother. So the block gets
        # a two-level cycle -- facet-block smoother, ADS per component on the
        # constants -- which holds ~3 us per dof per iteration flat.
        "ads": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads",
        ),
        # the same cycle with the inner CG stated explicitly rather than left
        # to the default the two-level path chooses (50 iterations to 1e-2)
        "ads-cg": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000,
            riesz_block_pc="ads", riesz_block_its=50, riesz_block_rtol=1e-2,
        ),
        "direct": mk.SolverOptions(),
    }


SOLVER_NAMES = ("direct", "riesz", "ads", "ads-cg")
DEFAULT_RTOL = 1e-9


PRODUCTS = {
    "derham_bdm": mk.StressRealization.derham_bdm,
    "derham_rt": mk.StressRealization.derham_rt,
    "stabilized_bdm": mk.StressRealization.stabilized_bdm,
    # NO RECONSTRUCTION: d per facet, one constant traction vector, and M the
    # diagonal primal-dual star -- the two-point stress. Half the unknowns of
    # the BDM products and an eighth of the matrix entries, on every mesh; it
    # is CONSISTENT only where the mesh is face-orthogonal, which a box of
    # hexahedra is and a tetrahedral or polyhedral one is not.
    #
    # It exists in four fields only, so --formulation is set for it rather than
    # asked of the caller.
    "diagonal_tpsa": mk.StressRealization.diagonal_tpsa,
}

FORMULATIONS = {
    "weak_symmetry": mk.StressFormulation.weak_symmetry,
    "weak_symmetry_total": mk.StressFormulation.weak_symmetry_total,
}


def formulation_for(product, asked):
    """Three fields or four, with the one product that has no choice honoured.

    diagonal_tpsa is diagonal only when the compliance is (2 mu)^-1, which is
    what the total-pressure form gives; in three fields the trace couples the
    traction components and the product does not exist. Asking for the pair
    that cannot be built is refused here rather than deeper down.
    """
    if product == "diagonal_tpsa":
        if asked == "weak_symmetry":
            raise SystemExit(
                "--product diagonal_tpsa exists only in the four-field form: "
                "pass --formulation weak_symmetry_total, or omit it"
            )
        return mk.StressFormulation.weak_symmetry_total
    return FORMULATIONS[asked]



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

    The cofacet cells are looked up in one batch.  ``cofacet_of`` rebuilds the
    coboundary operator on every call, which is O(mesh) each time: per facet it
    cost 1.31 s here against 3 ms for every other part of this loop, and the
    batched form is 2400x faster because it builds that operator once.
    """
    facets = mk.boundary_facets(mesh, dim)
    cells = mk.cofacets_of(mesh, dim, facets)
    for f, cell in zip(facets, cells):
        x_e = mk.centroid(mesh, dim, int(cell))
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
    ap.add_argument("--product", default="stabilized_bdm", choices=sorted(PRODUCTS))
    ap.add_argument("--formulation", default="weak_symmetry", choices=sorted(FORMULATIONS),
                    help="three fields, or four with the total pressure p = lambda div u")
    ap.add_argument(
        "--nu", type=float, default=None, help="Poisson ratio (default lam = mu = 1)"
    )
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
    form = formulation_for(args.product, args.formulation)
    print(f"  {mk.stress_realization_name(how)}, {mk.stress_formulation_name(form)}, "
          f"{args.solver} solver")
    report_processes(mesh, dim)
    print()

    gradient = gradient_of(dim, length, args.spin)
    with stage("creating the model"):
        model = mk.CauchyElasticityModel(mesh, dim, mat, how, form)
    with stage("prescribing u on the boundary"):
        n_facets = prescribe_linear_displacement(model, mesh, dim, lo, gradient)
    if args.assemble_only:
        # THE TWO BUILDS ALONE. They are what scales with the mesh, and a caller
        # measuring them should not have to wait for a Krylov method to
        # converge -- nor be told a time without being told they finished.
        report = model.assemble(progress=True, options=solvers(args.rtol)[args.solver])
        print(
            f"\n  assembled: matrix {report.matrix_seconds:.2f} s, "
            f"preconditioner {report.preconditioner_seconds:.2f} s"
        )
        print(f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized")
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
        # THE STRESS IS RECONSTRUCTED FROM THE CELL'S OPERATORS, and on several
        # processes each holds only its own; every other cell would read zero.
        # The displacement and the rotation are read from the solution, which
        # every process has in full, so only this one is gathered.
        stress = mk.gather_cells(
            model, np.array([model.cell_stress(e) for e in range(model.n_cells)])
        )
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
            s = stress[e]
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

    if args.vtu and root:
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
                # ALREADY GATHERED, and it must not be gathered again here: a
                # gather is collective, this block runs on one process only,
                # and the others would wait for it forever.
                sig[e] = stress[e]
            fields = {
                    "displacement": u,
                    "displacement_exact": u_exact,
                    "displacement_error": u - u_exact,
                    "rotation": rot,
                    "rotation_exact": rot_exact,
                    "rotation_error": rot - rot_exact,
                    "stress": sig,
                    "stress_exact": np.tile(np.array(s_hat), (n, 1)),
                    "stress_error": sig - np.array(s_hat),
            }
            fields.update(partition_field(mesh, dim, args.partition))
            mk.write_vtu(mesh, args.vtu, fields)


if __name__ == "__main__":
    main()
