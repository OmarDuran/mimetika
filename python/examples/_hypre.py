r"""The `hypre-ads` solver of flow.py and flow_external_mesh.py.

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
the eta = 1 cells of adaptive_rt -- and it takes those directly.  A facet
carrying d moments, derham_bdm and stabilized_bdm, reaches it through the
facet-constant subspace: the block is then a two-level cycle whose coarse
operator P^T A0 P is where ADS runs, with a symmetric SOR sweep as the smoother
because what the coarse space omits is the divergence-free part and that is not
facet-local.  All five run on one process and on several.

What is NOT here is a weak-symmetry stress: d COPIES of an H(div) space need
the per-component split the PETSc path builds, and the solve refuses it.
"""

import sys
import time

import mimetika_cxx as mk

NAME = "hypre-ads"


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
            block_iterations=None, block_rtol=None):
    """The ADS knobs, including the two PETSc registers and never queries.

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
    return o


def assemble(model, mesh, dim):
    """Assemble and stop, for --assemble-only.

    Two costs, reported apart because they are different objects: A and b, the
    saddle-point system; and the complex the Riesz map's first block is
    preconditioned through -- the discrete gradient and curl, and for a facet
    carrying d moments the degree-2 rung P3 -> N2E2 -> BDM1 with its
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

    # THE STAGES ARE ANNOUNCED BEFORE THEY RUN, NOT AFTER.
    #
    # The other solvers report from inside solve(), so this path printing
    # nothing left a minute of real work looking like a hung process. Written
    # to stderr unbuffered and in the same shape, so one run reads as one run.
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
    # WHICH OF THE TWO PRECONDITIONERS RAN. A facet carrying d moments is split
    # by row onto the degree-2 complex where every cell is a tetrahedron, and
    # reaches ADS through the facet-constant subspace where one is not. They
    # converge at similar counts, so no other line here distinguishes them --
    # and a mesh that is hybrid by a single cell takes the second.
    _note("ads on the degree-2 complex" if handoff["degree2"]
          else "ads on the facet-constant subspace")

    # A PRECONDITIONER THAT DID NOT CONVERGE STILL RETURNS A VECTOR.
    #
    # The examples read the answer and print an error table; without this a
    # capped solve reads as a discretization that stopped converging -- 3.8e-02
    # then 3.9e-01 on a mesh ladder, which is not a rate, it is a failure.
    if not report.converged:
        if _root():
            sys.stdout.flush()
            sys.stderr.write(
                f"\n  {NAME}: DID NOT CONVERGE -- {report.iterations} iterations, "
                f"{report.reason}. The answer below is whatever the last iterate was.\n")
            sys.stderr.flush()
    mk.accept(model, list(x))
    return _Report(report, assembly, handoff)


# RANK 0 ALONE REPORTS.
#
# Every rank runs the same script and reaches the same stage at a slightly
# different moment; eight of them writing to an unbuffered stderr produces
# "assembling ... assembling ... assembling ..." on one line and eight
# durations on the next. This mirrors what the C++ Stage class does.
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
        # WHICH PRECONDITIONER RAN, because there are two and they are not
        # interchangeable. A facet carrying d moments is split by ROW onto the
        # degree-2 complex where every cell is a tetrahedron, and reaches ADS
        # through the facet-constant subspace where one is not. Both converge,
        # at similar counts, so nothing else printed here would distinguish
        # them -- and on a mesh that is hybrid by one cell it is the second.
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
