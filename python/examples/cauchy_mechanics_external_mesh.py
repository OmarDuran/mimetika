#!/usr/bin/env python3
"""Cauchy mechanics on a mesh read from a .vtu: the linear patch test.

The other elasticity example builds its own annulus and compares against Lame's
closed form. This one takes the mesh as data -- whatever is in the file, of
whatever cell type -- and so cannot know a closed form for it. It prescribes
one instead:

    u(x) = (x - x_min) / L ,     L = the largest bounding-box extent

which runs from 0 at the low corner of the box to 1 at the far end of its
longest side, and is imposed on every boundary facet. Pure Dirichlet: no
traction is given anywhere, so the displacement data alone fixes the solution
and no rigid-body mode survives.

A linear field is the one case where the answer is known on any mesh at all.
The datum is affine, both integrals it needs are already carried by the stress
operators' facet moments, so no boundary quadrature enters and the mixed method
reproduces the field rather than approximating it. The error is therefore
round-off and not discretization, on a good mesh and a bad one alike.

The stress that goes with it is uniform,

    eps = sym(grad u) = I_d / L ,    sigma = 2 mu eps + lam tr(eps) I ,

so sigma_kk = (2 mu + d lam) / L on the d meshed axes, and that is checked too.

Run it:

    PYTHONPATH=.. python cauchy_mechanics_external_mesh.py --mesh domain.vtu
    PYTHONPATH=.. python cauchy_mechanics_external_mesh.py --mesh domain.vtu --vtu out.vtu

with --make-mesh to produce a sample file first, if there is none to hand:

    PYTHONPATH=.. python cauchy_mechanics_external_mesh.py --make-mesh domain.vtu
"""

import argparse
import os
import sys

import mimetika_cxx as mk
import numpy as np

from _diagnostics import write_report
from _errors import error_table, l2_norms
from _stages import stage

MU, LAM = 1.0, 1.0


def solvers(rtol):
    """The linear solvers, at the residual tolerance asked for.

    "riesz" is the Riesz map of the space the operator is an isomorphism on --
    P is the Gram matrix of its norm -- so its iteration count does not grow
    with the mesh. "direct" is a full factorization: exact, and the wrong
    instrument past a few hundred thousand unknowns.

    The tolerance is on the residual, not on the answer. An iterative solve
    cannot show the round-off floor a direct one leaves, so a patch test read
    through it is bounded by this number rather than by the method.
    """
    return {
        "riesz": mk.SolverOptions(
            method="gmres", preconditioner="riesz", rtol=rtol, max_iterations=2000
        ),
        # The same map, with the stress block inverted by an auxiliary space.
        #
        # P's stress block is H(div; M) and it is most of the unknowns; a
        # Cholesky of it is exact and creates fill, so its cost per iteration
        # grows with the mesh (0.62 -> 8.0 us per dof per iteration over a
        # refinement to 100k unknowns) even though the iteration count does
        # not.
        #
        # ADS wants one unknown per facet and the stabilized_bdm facet carries
        # d^2 -- d traction components, each against the d functions of the
        # facet P_1 basis. The route to it is the facet-constant subspace:
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
    # No reconstruction: d per facet, one constant traction vector, and M the
    # diagonal primal-dual star -- the two-point stress. Half the unknowns of
    # the BDM products and an eighth of the matrix entries, on every mesh; it
    # is consistent only where the mesh is face-orthogonal, which a box of
    # hexahedra is and a tetrahedral or polyhedral one is not.
    #
    # It exists in four fields only, so --formulation is set for it rather than
    # asked of the caller.
    "diagonal_afw": mk.StressRealization.diagonal_afw,
    # The strongly-symmetric family (Dassi-Lovadina-Visinoni): six traction
    # moments per facet carried whole, the displacement as the cell's six
    # rigid-motion coefficients, no rotation multiplier. stabilized_vem builds
    # either strong formulation; diagonal_vem is its two-point member and
    # adaptive_vem the per-cell selection between them, both under
    # strong_symmetry_total only -- where M can be diagonal at all.
    "stabilized_vem": mk.StressRealization.stabilized_vem,
    "diagonal_vem": mk.StressRealization.diagonal_vem,
    "adaptive_vem": mk.StressRealization.adaptive_vem,
}

STRONG_PRODUCTS = {"stabilized_vem", "diagonal_vem", "adaptive_vem"}

FORMULATIONS = {
    "weak_symmetry": mk.StressFormulation.weak_symmetry,
    "weak_symmetry_total": mk.StressFormulation.weak_symmetry_total,
    "strong_symmetry": mk.StressFormulation.strong_symmetry,
    "strong_symmetry_total": mk.StressFormulation.strong_symmetry_total,
}


def formulation_for(product, asked):
    """Three fields or four, with the one product that has no choice honoured.

    diagonal_afw is diagonal only when the compliance is (2 mu)^-1, which is
    what the total-pressure form gives; in three fields the trace couples the
    traction components and the product does not exist. Asking for the pair
    that cannot be built is refused here rather than deeper down.
    """
    # Omitted is not the same as asked for. `asked` is None when the caller
    # said nothing, and then the product decides: three fields for the BDM
    # ones, four for diagonal_afw, which has no other form. Only an explicit
    # --formulation weak_symmetry with diagonal_afw is a contradiction, and it
    # is the only case refused.
    if product == "diagonal_afw":
        if asked not in (None, "weak_symmetry_total"):
            raise SystemExit(
                "--product diagonal_afw exists only in the four-field form: drop "
                "--formulation, or pass weak_symmetry_total"
            )
        return mk.StressFormulation.weak_symmetry_total
    # the strong family: the symmetry axis is the product's, and the diagonal
    # members demand the total pressure
    if product in ("diagonal_vem", "adaptive_vem"):
        if asked not in (None, "strong_symmetry_total"):
            raise SystemExit(
                f"--product {product} exists only under strong_symmetry_total: drop "
                "--formulation, or pass strong_symmetry_total"
            )
        return mk.StressFormulation.strong_symmetry_total
    if product == "stabilized_vem":
        if asked in ("weak_symmetry", "weak_symmetry_total"):
            raise SystemExit(
                "--product stabilized_vem carries its symmetry in the space: use "
                "strong_symmetry or strong_symmetry_total"
            )
        return FORMULATIONS[asked or "strong_symmetry"]
    if asked in ("strong_symmetry", "strong_symmetry_total"):
        raise SystemExit(
            f"--product {product} imposes symmetry weakly: use weak_symmetry or "
            "weak_symmetry_total, or a vem product"
        )
    return FORMULATIONS[asked or "weak_symmetry"]



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

    The symmetric part is the dilation, so the stress is unchanged by the spin
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

    The datum is written about the cell centroid -- u = a + B (x - x_E) -- so
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

    These are what every degree-of-freedom formula reads from: the weak stress
    space is d^2 f + (d + d(d-1)/2) c, the strong one 6f + 6c (plus c with the
    total pressure). Reported once so a count in the output can be checked
    against the formula rather than trusted.
    """
    names = ["vertices", "edges", "faces", "cells"]
    parts = [
        f"{mesh.count(k)} {names[k] if k < mesh.dim else 'cells'}"
        for k in range(mesh.dim + 1)
    ]
    print("  complex: " + ", ".join(parts))


ap_default_solver = "riesz"


def main():
    root = only_root()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", help="the .vtu to solve on")
    ap.add_argument("--make-mesh", help="write a sample .vtu to this path and stop")
    ap.add_argument("--product", default="stabilized_bdm", choices=sorted(PRODUCTS))
    ap.add_argument("--formulation", default=None, choices=sorted(FORMULATIONS),
                    help="three fields, or four with the total pressure p = lambda div u; "
                         "the default follows the product")
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
    ap.add_argument("--solver", default=ap_default_solver, choices=sorted(SOLVER_NAMES))
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="hybridize: eliminate the stress cell by cell and solve the facet "
        "multiplier system, which is SPD -- conjugate gradients and multigrid. "
        "The boundary roles swap: a prescribed displacement PINS a multiplier "
        "and a traction loads the free rows. Serial only.",
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
        help="adaptive_vem: a cell whose stabilized vem block has lambda_max/lambda_min above "
             "this takes the diagonal star as well (composes with --degeneracy-percent)",
    )
    ap.add_argument(
        "--rotation-jump",
        type=float,
        default=None,
        metavar="C",
        help="diagonal_afw: the facet-jump stabilization of the rotation multiplier, "
             "-J gamma on the rotation row with J the two-point Laplacian of the half "
             "weights C mu |f| delta. J annihilates constant rotations, so the patch "
             "test below is unchanged by it. Off by default; not available with --hybrid",
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
    report_complex(mesh)
    print(
        f"  box  {np.array2string(lo[:dim], precision=4)} .. "
        f"{np.array2string(hi[:dim], precision=4)}"
    )
    print(f"  characteristic length L = {length:.6g}")
    print(f"  mu = {mat.shear}, lam = {mat.lame}  (nu = {mat.poisson():.4f})")
    form = formulation_for(args.product, args.formulation)
    # What actually runs, not what was asked for. --hybrid does not take the
    # solver named on the command line: the elimination removes the H(div)
    # block a Riesz map or ADS would split along, so the interface system gets
    # conjugate gradients and an algebraic multigrid whatever --solver said.
    solver_name = "cg + boomeramg (hybridized)" if args.hybrid else f"{args.solver} solver"
    print(f"  {mk.stress_realization_name(how)}, {mk.stress_formulation_name(form)}, "
          f"{solver_name}")
    if args.rotation_jump:
        print(f"  rotation jump c = {args.rotation_jump:g}: -J gamma on the rotation row")
    if args.hybrid and args.solver != ap_default_solver:
        print(f"  note: --solver {args.solver} does not apply to the interface system "
              f"and is ignored")
    report_processes(mesh, dim)
    print()

    gradient = gradient_of(dim, length, args.spin)
    with stage("creating the model"):
        model = mk.CauchyMechanicsModel(mesh, dim, mat, how, form)
        # the threshold reaches the model only where it is the model's: for
        # any other product it stays what it always was, the diagnostics dial
        if args.product == "adaptive_vem" and args.degeneracy_percent is not None:
            model.set_degeneracy_percent(args.degeneracy_percent)
        if args.product == "adaptive_vem" and args.cond_threshold is not None:
            model.set_cond_threshold(args.cond_threshold)
        if args.rotation_jump is not None:
            model.set_rotation_jump(args.rotation_jump)
    with stage("prescribing u on the boundary"):
        n_facets = prescribe_linear_displacement(model, mesh, dim, lo, gradient)
    if args.assemble_only:
        # The two builds alone. They are what scales with the mesh, and a caller
        # measuring them should not have to wait for a Krylov method to
        # converge -- nor be told a time without being told they finished.
        report = model.assemble(progress=True, options=solvers(args.rtol)[args.solver])
        print(
            f"\n  assembled: matrix {report.matrix_seconds:.2f} s, "
            f"preconditioner {report.preconditioner_seconds:.2f} s"
        )
        print(f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized")
        if report.condensed:
            print(f"  sigma eliminated exactly: {model.n_dofs} -> {report.condensed_dofs} "
                  f"unknowns; the preconditioner built is the reduced system's "
                  f"({report.block_solver})")
        return
    if args.hybrid:
        # The interface system is SPD, and the H(div) block a Riesz map or ADS
        # splits along is gone: CG with an algebraic multigrid is what it takes.
        report = model.solve_hybrid(
            progress=True,
            options=mk.SolverOptions(
                method="cg", preconditioner="hypre", rtol=args.rtol, max_iterations=2000
            ),
        )
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
    # The size the linear solver was handed, which is not the model's dof
    # count whenever a field was eliminated first. Hybridized it is the facet
    # multipliers; condensed it is the cells; monolithic the two agree.
    # Printing it settles what an iteration count is per.
    if report.condensed and report.condensed_dofs:
        solved = report.condensed_dofs
        of_what = "facet multipliers" if args.hybrid else "cell unknowns"
        print(f"  the linear system solved: {solved} {of_what}"
              f"  ({model.n_dofs} in the mixed space, eliminated first)")
    else:
        print(f"  the linear system solved: {model.n_dofs} unknowns, the mixed space")

    label = "cg" if args.hybrid else args.solver
    if args.hybrid or args.solver != "direct":
        print(
            f"  {label}: {report.iterations} iterations to rtol {args.rtol:.1e}"
            f", {report.reason} in {report.solve_seconds:.2f} s"
        )
    # What was actually solved, when the stress left the system. A facet-
    # diagonal star is eliminated exactly -- division, not factorization --
    # and strong_symmetry_total (sigma, u, p) collapses from 6f + 7c unknowns
    # to the 7c cell unknowns of (u, p): six rigid-motion coefficients and one
    # total pressure per cell, a two-point system that is symmetric
    # quasi-definite (u positive, p negative -- the pressure mass c_p|E| rules
    # out SPD), which is why it takes GMRES + BoomerAMG rather than CG.
    if report.condensed:
        per_cell = report.condensed_dofs // model.n_cells
        print(
            f"  sigma eliminated exactly: {model.n_dofs} -> {report.condensed_dofs} unknowns "
            f"({per_cell} per cell), symmetric quasi-definite, "
            f"gmres + {report.block_solver}"
        )
    # THE VALIDITY GATE OF THE DIAGONAL STAR, which exokal leaves to the
    # consumer: a facet the cell centroid does not see squarely carries a
    # non-positive weight, M is not positive there, and the elimination
    # divides by it -- the answer then collapses toward zero while the
    # residual still CONVERGES. That is worse than a wrong answer, so it is
    # shouted rather than footnoted.
    if args.product in ("diagonal_vem", "adaptive_vem"):
        bad = model.n_invalid_star
        if bad:
            print(f"\n  *** {bad} cell(s) carry a NON-POSITIVE star weight: the diagonal")
            print("  *** product is INVALID on this mesh -- the field below is meaningless.")
            print("  *** Use stabilized_vem or adaptive_vem (whose default keeps the")
            print("  *** stabilized product everywhere the scan does not flag).")
    print(f"\n  u = (I + W)(x - x_min)/L on all {n_facets} boundary facets, pure Dirichlet")
    print(f"  {model.n_cells} cells, {model.n_dofs} dofs, {model.n_stabilized} stabilized")
    if args.product == "adaptive_vem":
        n_star = int((model.eta == 0.0).sum())
        pct = args.degeneracy_percent
        print(f"  adaptive_vem: {n_star} cell(s) on the diagonal star, "
              f"{model.n_cells - n_star} on the stabilized vem product "
              f"(threshold {'default' if pct is None else f'{pct}%'}"
              + (f"; {model.n_ill_conditioned} switched by cond > {args.cond_threshold:g}"
                 if args.cond_threshold is not None else "")
              + ")")

    # ---- the three fields, each against the value the datum determines -------
    #
    # The method reproduces a linear displacement exactly, so all three are
    # known in closed form and none of them is a discretization error: what is
    # left is round-off, or the residual tolerance of an iterative solve.
    u_hat = exact_rotation(gradient, dim)
    s_hat = exact_stress(gradient, mat, dim)
    # The split the material sees: the volumetric stress is the mean of the
    # diagonal, tr(sigma)/d, and the deviator is what is left. The split is a
    # linear read of the same reconstruction, so each part inherits exactly
    # the accuracy of the full tensor -- reported separately because a defect
    # that lives in one part alone (a missing trace, a spurious deviator) is
    # invisible in a single max-error number.
    sm_hat = sum(s_hat[k * 3 + k] for k in range(dim)) / dim
    d_hat = [s_hat[k] - (sm_hat if k in (0, 4, 8) and k // 3 < dim else 0.0) for k in range(9)]
    n_rot = model.n_rotations

    with stage("reconstructing u, gamma and sigma"):
        # The stress is reconstructed from the cell's operators, and on several
        # processes each holds only its own; every other cell would read zero.
        # The displacement and the rotation are read from the solution, which
        # every process has in full, so only this one is gathered.
        n = model.n_cells
        stress = mk.gather_cells(
            model, np.array([model.cell_stress(e) for e in range(n)])
        )
        volume = np.fromiter((mk.measure(mesh, dim, e) for e in range(n)), float, n)
        x = np.array([mk.centroid(mesh, dim, e) for e in range(n)])
        u = np.array([[model.displacement(e, k) for k in range(dim)] for e in range(n)])
        rot = np.array([[model.rotation(e, k) for k in range(n_rot)] for e in range(n)])
        # The exact side needs no projection. u is affine, so Pi_0 u is u at the
        # centroid; gamma and sigma are constant, so Pi_0 leaves them alone.
        grad = np.array(gradient).reshape(3, 3)[:dim, :dim]
        u_exact = (x[:, :dim] - lo[:dim]) @ grad.T
        # the meshed block of the 3x3 layout, and the split the material sees:
        # volumetric = tr(sigma)/d, deviatoric = what is left of the block
        block = [i * 3 + j for i in range(dim) for j in range(dim)]
        on_diagonal = [k * dim + k for k in range(dim)]
        sigma, sigma_hat = stress[:, block], np.array(s_hat)[block]
        vol = sigma[:, on_diagonal].mean(axis=1)
        dev = sigma.copy()
        dev[:, on_diagonal] -= vol[:, None]
        dev_hat = sigma_hat.copy()
        dev_hat[on_diagonal] -= sm_hat

    print()
    print("  the exact fields, which the method reproduces:")
    print()
    print(f"    u at the last cell   {fmt(u_exact[-1])}")
    print(f"    gamma = skw(grad u)  {fmt(u_hat)}")
    print(f"    sigma diagonal       {fmt([s_hat[k * 3 + k] for k in range(dim)])}")
    print(f"    sigma volumetric     {fmt([sm_hat])}")
    print(f"    sigma deviatoric     {fmt(dev_hat[on_diagonal])}")

    print()
    print("  error.  e = Pi_0(v - v_h), with Pi_0 w|_E = |E|^-1 int_E w the L2")
    print("  projection onto cell-wise constants; D is the domain and E a cell.")
    print("  u and gamma are cell moments over |E| and sigma is the cell average")
    print("  of the facet tractions, so each row is a whole discretization error.")
    print("  S is the norm the row is measured against: ||Pi_0 u||_D for u,")
    print("  |grad u|_F |D|^(1/2) for gamma, and ||sigma||_D for all three stress")
    print("  rows, which keeps those three comparable. It is named per row rather")
    print("  than read off the row because sigma_dev vanishes for this field, as")
    print("  does gamma at --spin 0.")
    print()
    root_area = np.sqrt(float(volume.sum()))
    scale_u, _ = l2_norms(volume, u_exact)
    error_table(volume, [
        ("u", u - u_exact, scale_u),
        ("gamma", rot - np.array(u_hat), root_area * np.linalg.norm(np.array(gradient))),
        ("sigma", sigma - sigma_hat, root_area * np.linalg.norm(sigma_hat)),
        ("sigma vol", vol - sm_hat, root_area * np.linalg.norm(sigma_hat)),
        ("sigma dev", dev - dev_hat, root_area * np.linalg.norm(sigma_hat)),
    ])

    if args.vtu and root:
        with stage(f"writing {args.vtu}"):
            # Everything here was built by the measurement above, `stress`
            # included -- and it must not be gathered again: a gather is
            # collective, this block runs on one process only, and the others
            # would wait for it forever. What is left is padding the meshed
            # components out to the three ParaView expects, and re-expressing
            # the split over the full 3x3 layout.
            def padded(values, columns):
                out = np.zeros((n, 3))
                out[:, :columns] = values
                return out

            u3, rot3 = padded(u, dim), padded(rot, n_rot)
            # The split, per cell: volumetric = tr(sigma)/d as a scalar, the
            # deviator as the full tensor with that mean removed from the
            # meshed diagonal.
            dev9 = stress.copy()
            for k in range(dim):
                dev9[:, k * 3 + k] -= vol
            fields = {
                    "displacement": u3,
                    "displacement_exact": padded(u_exact, dim),
                    "displacement_error": u3 - padded(u_exact, dim),
                    "rotation": rot3,
                    "rotation_exact": padded(np.tile(u_hat, (n, 1)), n_rot),
                    "rotation_error": rot3 - padded(np.tile(u_hat, (n, 1)), n_rot),
                    "stress": stress,
                    "stress_exact": np.tile(np.array(s_hat), (n, 1)),
                    "stress_error": stress - np.array(s_hat),
                    "stress_volumetric": vol,
                    "stress_volumetric_exact": np.full(n, sm_hat),
                    "stress_volumetric_error": vol - sm_hat,
                    "stress_deviatoric": dev9,
                    "stress_deviatoric_exact": np.tile(np.array(d_hat), (n, 1)),
                    "stress_deviatoric_error": dev9 - np.array(d_hat),
            }
            if args.product == "adaptive_vem":
                fields["eta"] = model.eta
            fields.update(partition_field(mesh, dim, args.partition))
            mk.write_vtu(mesh, args.vtu, fields)


if __name__ == "__main__":
    main()
