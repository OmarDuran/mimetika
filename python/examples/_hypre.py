r"""The `hypre-ads` and `hypre-mgr` solvers of the flow and mechanics examples.

The same Riesz map every other solver in those examples uses --
P = diag(M + B^T W^-1 B, W) -- with its first block inverted by hypre's ADS
called DIRECTLY rather than through PETSc's PCHYPRE.

WHY A SECOND MODULE AND A HANDOFF.  mimetika_cxx links PETSc, PETSc links its
own libHYPRE, and two copies of hypre in one process export the same names --
which one a call reaches is decided by load order.  So the direct path lives in
`mimetika_hypre`, whose hypre symbols are hidden at link time.  The two modules
can then be imported together, but they share no types: a mesh built in one is
not a mesh in the other.  What crosses between them is plain data --
`mk.ads_handoff` gives the assembled system and the norm, `mh.solve_system`
returns the answer, `mk.accept` writes it back.

The norm is not restated on the way: `ads_handoff` returns what build_norm
gives PETSc, so both paths are preconditioned by the same map and a difference
between them is the library rather than the method.

ADS is written for ONE unknown per facet in 3D -- derham_rt, stabilized_rt and
the eta = 1 cells of adaptive_rt -- and it takes those on the mesh's own
complex.  A facet carrying three moments, derham_bdm and stabilized_bdm, is
lifted to the degree-2 rung P3 -> N2E2 -> BDM1 with its interpolations Pi_rt,
Pi_nd; where that upgrade does not apply -- not 3D, or not three moments a
facet -- the block falls back to the facet-constant subspace, a two-level cycle
whose coarse operator P^T A0 P is where ADS runs.  All five run on one process
and on several.

A weak-symmetry stress is d COPIES of the H(div) space, split by ROW: the
multipliers are handed over unmerged, so no C^T W_gamma^-1 C term couples one
row of sigma to another and each row is one a_div problem on the same complex.
"""

import sys
import time

import mimetika_cxx as mk

NAME = "hypre-ads"
# THE OTHER PRECONDITIONER THIS MODULE OFFERS.
#
# ADS preconditions the first block of the Riesz map; MGR eliminates it instead
# -- F = the flux or the stress, C = the multipliers -- leaving
#
#     S = -D M^-1 D^T
#
# for BoomerAMG. On flow that is the cell-centred Laplacian, one unknown a cell,
# which is what the two-point family already is: h- and contrast-robust, and the
# faster of the two.
#
# ON THE STRESS IT IS BOUNDED RATHER THAN FLAT. Eliminating the stress
# reconstructs the displacement operator, so the count tracks nu. Measured by
# test_mechanics_hypre_mgr and test_mechanics_hypre_ads, stabilized_vem under
# strong symmetry on tetrahedra at rtol 1e-8:
#
#     nu    0.25  0.40  0.49  .499  .4999
#     mgr     36    38    46    52     60
#     ads     14    14    14    18     36     (the block solved, not one cycle)
#
# The stress gets AMG on its F block, which the solver picks wherever a facet
# carries more than one moment. Jacobi there cannot damp the hydrostatic mode --
# the compliance's one small eigenvalue, 1/(2mu + d lam) against 1/2mu on the
# five deviatoric ones -- and the count then grows like sqrt(2mu + d lam): the
# same sweep reads 74, 92, 204, 542, 1648 under Jacobi. It is the F block and
# NOT the coarse one, since div of a constant stress is zero and the hydrostatic
# direction therefore never enters S = -B M^-1 B^T.
#
# AMG there costs about 43 percent more a solve at nu = 1/4 on 1.2e5 cells.
# mgr_relax_sweeps must be ODD -- an even count does not converge, on flow or on
# the stress; linear_solver/hypre.hpp carries the sweep-count timings.
#
# THE SCOPE OF THE ROBUSTNESS CLAIM: stabilized_vem, AMG on F. On its own
# ladders test_mechanics_hypre_mgr asserts a ratio of at most 1.5 in h (worst
# measured 1.44), at most 2 across an eight-decade lambda jump, and at most 3
# over nu = 0.25 .. 0.4999. Nothing is asserted for the weak family.
#
# THE WEAK PATH DOES NOT CONVERGE, AND NOTHING REFUSES IT: a facet carrying d
# moments takes MGR's two-level reduction and stalls at the iteration cap, flow
# derham_bdm and stabilized_bdm alike. The note in linear_solver/hypre.hpp has
# it surviving every F-relaxation, interpolation type and reduction depth.
MGR_NAME = "hypre-mgr"
HYPRE_NAMES = (NAME, MGR_NAME)


def available():
    """Whether the module was built. It is optional: MIMETIKA_USE_HYPRE=ON."""
    try:
        import mimetika_hypre  # noqa: F401
    except ImportError:
        return False
    return hasattr(mk, "ads_handoff")


def why_unavailable():
    return (
        f"--solver {NAME} needs the mimetika_hypre module, which is built by a "
        "separate configuration:\n"
        "  cmake -S . -B build-hypre -DMIMETIKA_USE_HYPRE=ON "
        "-DMIMETIKA_HYPRE_ROOT=<hypre install> -DMIMETIKA_USE_PETSC=OFF "
        "-DMIMETIKA_BUILD_PYTHON=ON\n"
        "It cannot be part of mimetika_cxx: that links PETSc, and PETSc links "
        "its own libHYPRE."
    )


def require_serial():
    """Refuse a distributed run before any work, and say so once.

    The solver refuses too, but by then every rank has assembled and every rank
    raises, so the message arrives N times behind N tracebacks. Checked here it
    is one line.
    """
    n = mk.mpi_size()
    if n > 1:
        if mk.mpi_rank() == 0:
            sys.stderr.write(
                f"\n  --solver {NAME} is serial and this run has {n} ranks.\n"
                "  The direct hypre path creates each matrix over the whole index range,\n"
                "  so on more than one rank every process claims every row. Use\n"
                "  --solver ads or ads-cg for a distributed run, or drop mpirun.\n")
            sys.stderr.flush()
        raise SystemExit(1)


def options(rtol, cycle_type=13, amg_theta=0.25, ams_theta=0.25, max_iterations=2000,
            block_iterations=None, block_rtol=None, mgr=False):
    """The ADS knobs, amg_theta and ams_theta included: PETSc registers those
    two and never queries them, so they are reachable only here.

    `block_iterations` is the one that changes the character of the solve. The
    default applies ONE ADS cycle per application, which is what the Riesz map
    asks for and what is cheapest where the coefficient is smooth. A jumping
    coefficient needs the block solved rather than approximated -- one cycle
    stops converging past a contrast of about 1e4, for PETSc's ADS as much as
    for this one -- and a short CG under the same cycle restores it. Measured
    on a 93k-cell industrial mesh at rtol 1e-5: one cycle is 81 iterations in
    38 s, a CG to 1e-2 is 9 in 48 s, and to 1e-6 is 8 in 116 s.
    """
    import mimetika_hypre as mh

    o = mh.AdsOptions()
    o.rtol = rtol
    o.max_iterations = max_iterations
    o.cycle_type = cycle_type
    o.amg_theta = amg_theta
    o.ams_theta = ams_theta
    if block_iterations is not None:
        o.block_iterations = block_iterations
    if block_rtol is not None:
        o.block_rtol = block_rtol
    o.mgr = mgr
    return o


def assemble(model, mesh, dim):
    """Assemble and stop, for --assemble-only.

    Two costs, reported apart because they are different objects: A and b, the
    saddle-point system; and the complex the Riesz map's first block is
    preconditioned through -- the discrete gradient and curl, and for three
    moments a facet the degree-2 rung P3 -> N2E2 -> BDM1 with its
    interpolations Pi_rt, Pi_nd.
    """
    import mimetika_hypre as mh

    mk.mpi_size()
    mh.init()

    t0 = time.perf_counter()
    _stage("assembling A, b")
    mk.distribute(model)
    model.build()
    matrix = time.perf_counter() - t0
    _stage_done(matrix)

    _stage("assembling the discrete complex")
    t1 = time.perf_counter()
    handoff = mk.ads_handoff(model, mesh, dim)
    precond = time.perf_counter() - t1
    _stage_done(precond)
    return _Assembly(matrix, precond, handoff)


class _Assembly:
    """The fields the examples read from a SolveReport after --assemble-only."""

    def __init__(self, matrix_seconds, preconditioner_seconds, handoff):
        self.matrix_seconds = matrix_seconds
        self.preconditioner_seconds = preconditioner_seconds
        self.assembly_seconds = matrix_seconds + preconditioner_seconds
        self.condensed = False
        self.condensed_dofs = 0
        self.degree2 = bool(handoff.get("degree2", False))
        self.block_solver = ("ads, degree-2 complex" if self.degree2 else "ads")


def _reduced_onto(model):
    """What MGR's C block is: the multipliers the first factor pairs with."""
    return "the displacement" if hasattr(model, "n_rotations") else "the pressure"


def solve(model, mesh, dim, opts):
    """Assemble in mimetika_cxx, solve in mimetika_hypre, accept back.

    Returns an object with the fields the examples read from a SolveReport:
    iterations, reason, converged, and the two assembly timings the direct path
    reports as one setup.
    """
    import mimetika_hypre as mh

    # the other solvers assemble inside solve(); here it is explicit, so it is
    # timed here too rather than reported as zero
    # ONE MPI INITIALIZATION IN THE PROCESS, AND PETSc OWNS IT.
    #
    # build() partitions, which asks MPI for the communicator size, so MPI has
    # to be up before the model is built. Both modules can start it, and they
    # must not both: mpi_size() brings up PetscSession, and hypre's init then
    # sees MPI_Initialized and attaches instead of initializing a second time.
    mk.mpi_size()
    mh.init()

    # The stages are announced before they run: the other solvers report from
    # inside solve(), so this path would print nothing for the whole assembly.
    # stderr, unbuffered, in the shape the C++ Stage class uses.
    t0 = time.perf_counter()
    _stage("assembling")
    # the partition is numbered inside build(), so it has to be asked for first
    mk.distribute(model)
    model.build()
    handoff = mk.ads_handoff(model, mesh, dim)
    assembly = time.perf_counter() - t0
    _stage_done(assembly)

    _stage("solving")
    t1 = time.perf_counter()
    x, report = mh.solve_system(**handoff, options=opts)
    _stage_done(time.perf_counter() - t1)
    _line("preconditioner", report.setup_seconds)
    _line("iteration", report.solve_seconds)
    # WHICH PRECONDITIONER RAN. A facet carrying three moments is split by ROW
    # and each row handed to ADS on the degree-2 complex -- on ANY cell shape,
    # because C is facet-wise and needs no cell reconstruction. Anything else
    # reaches ADS through its lowest-order space: the mesh's own complex for one
    # moment a facet, the frame-weighted coarse space for a strong-symmetry
    # stress. The two converge at similar counts, so the count does not say
    # which ran.
    if opts.mgr:
        # ONE LEVEL OR TWO, read off the operator: two where the multiplier rows
        # touch some but not all of the flux dofs, the untouched higher moments
        # lying in ker D and eliminated first; one where they touch every one --
        # flow, and the strong stress, whose six facet moments pair against RM(E).
        _note(f"mgr reduction onto {_reduced_onto(model)}")
    else:
        _note("ads on the degree-2 complex" if handoff["degree2"]
              else "ads on the lowest-order complex")

    # A capped solve still returns a vector, and the examples read it into an
    # error table: unreported it looks like a discretization that stopped
    # converging -- 3.8e-02 then 3.9e-01 on a mesh ladder.
    if not report.converged:
        if _root():
            sys.stdout.flush()
            sys.stderr.write(
                f"\n  {MGR_NAME if opts.mgr else NAME}: DID NOT CONVERGE -- "
                f"{report.iterations} iterations, "
                f"{report.reason}. The answer below is whatever the last iterate was.\n")
            sys.stderr.flush()
    mk.accept(model, list(x))
    return _Report(report, assembly, handoff)


# Rank 0 alone reports: every rank runs the same script, so N ranks writing to
# an unbuffered stderr interleave N copies of each stage line. Same convention
# as the C++ Stage class.
def _root():
    return mk.mpi_rank() == 0


def _stage(what):
    if not _root():
        return
    sys.stdout.flush()
    sys.stderr.write(f"  {what} ...")
    sys.stderr.flush()


def _stage_done(seconds):
    if not _root():
        return
    sys.stderr.write(f" {seconds:.2f} s\n")
    sys.stderr.flush()


def _note(text):
    if not _root():
        return
    sys.stdout.flush()
    sys.stderr.write(f"    {text}\n")
    sys.stderr.flush()


def _line(what, seconds):
    if not _root():
        return
    sys.stdout.flush()
    sys.stderr.write(f"    {what} ... {seconds:.2f} s\n")
    sys.stderr.flush()


class _Report:
    """A SolveReport-shaped view, so the examples print one table."""

    def __init__(self, r, assembly_seconds=0.0, handoff=None):
        # which complex ADS ran on, for the block_solver line below
        self.degree2 = bool(handoff.get("degree2", False)) if handoff else False
        self.iterations = r.iterations
        self.reason = r.reason
        self.converged = r.converged
        self.residual = r.residual
        self.solve_seconds = r.setup_seconds + r.solve_seconds
        self.matrix_seconds = 0.0
        self.preconditioner_seconds = r.setup_seconds
        self.assembly_seconds = assembly_seconds
        self.condensed = False
        self.condensed_dofs = 0
        self.block_solver = ("ads on the degree-2 complex" if self.degree2
                             else "ads on the facet-constant subspace")
        self.off_rank_fraction = 0.0
