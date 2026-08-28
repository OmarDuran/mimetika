#pragma once

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/linear_solver/condense.hpp"
#include "mimetika/linear_solver/linear.hpp"

// PETSc, with a direct factorization first.
//
// A direct solve is the right instrument while a discretization is being
// validated: it answers "is this operator right" with no preconditioner in
// between. If a direct solve gives the wrong displacement field, the
// discretization is wrong — there is nowhere else for the error to have come
// from.
//
// MUMPS rather than PETSc's built-in LU because the systems here are
// saddle points: indefinite, so the factorization needs symmetric pivoting to
// stay stable, and PETSc's own LU does not do it well. MUMPS also handles the
// zero diagonal blocks — the (u,u) and (gamma,gamma) blocks that make this a
// saddle point in the first place — without a shift.
//
// An iterative path is the same object with a different prefix, which is why
// the KSP is configured from options rather than hard-coded: `-ksp_type
// fgmres -pc_type fieldsplit` selects one without recompiling, and the
// matrix-free operator already built can be attached to it later.

namespace mimetika::solver {

// One PETSc initialization for the process, torn down at exit. PETSc is
// global state and initializing it twice is an error, so it is owned here
// rather than by whoever happens to solve first.
class PetscSession {
 public:
  static PetscSession& instance() {
    static PetscSession s;
    return s;
  }

  PetscSession(const PetscSession&) = delete;
  PetscSession& operator=(const PetscSession&) = delete;

 private:
  PetscSession() {
    PetscBool ready = PETSC_FALSE;
    PetscInitialized(&ready);
    if (!ready) {
      int argc = 0;
      char** argv = nullptr;
      PetscInitialize(&argc, &argv, nullptr, nullptr);
      owned_ = true;
    }
  }
  ~PetscSession() {
    if (owned_) PetscFinalize();
  }
  bool owned_{false};
};

inline void check(PetscErrorCode e, const char* what) {
  if (e != 0) throw std::runtime_error(std::string("petsc: ") + what);
}

// Not every factorization exists on every communicator. PETSc's own LU
// and Cholesky, SuperLU and ICC are sequential codes; asked for on several
// processes they fail at setup rather than distributing themselves. MUMPS is
// the one in this build that does both, and it is already the choice for these
// saddle points, so a sequential name becomes it. An incomplete factorization
// has no distributed form at all and becomes block Jacobi over the ranks, each
// block keeping the incomplete factorization it asked for.
inline std::string parallel_package(const std::string& package) {
  return (package.empty() || package == "petsc" || package == "superlu") ? "mumps" : package;
}

// How the system is solved, as an argument rather than an environment. The
// MIMETIKA_FACTOR environment variable and the PETSc options database are
// invisible to the caller, absent from the Python surface, silently ignored
// when misspelled, and cannot be set differently for two solves in one
// process.
//
// A misspelled value here is refused by PETSc and surfaces as an exception.
// The norm of the product space. P is its Gram matrix, and nothing else.
//
// A maps X to its dual, so a Krylov method -- which needs an operator X -> X --
// requires a map X' -> X. The canonical one is the Riesz map of the inner
// product of X, and with it P^{-1}A has a condition number bounded by the
// inf-sup and continuity constants alone: independent of h. P is therefore not
// an approximation of A. Write the norm and P is determined.
//
//   FLOW      X = H(div) x L^2
//             ||q||^2 = (K^{-1} q, q) + ||div q||^2 ,   ||p||^2 = ||p||_{L2}^2
//
//   ELASTICITY  X = H(div; M) x L^2(R^d) x L^2(skew)
//             ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 ,  A = C^{-1}
//             ||u||^2 = ||u||_{L2}^2 ,   ||r||^2 = ||r||_{L2}^2
//
// Both are the same statement: the first factor carries the material inner
// product plus the graph term of its differential, and every factor after it
// carries plain L^2. The multiplier for weak symmetry is an L^2 factor like the
// displacement -- it is not special, and giving it anything else is a different
// preconditioner.
//
// How each term reads in this dof basis:
//
//   (A sigma, sigma)  is the assembled (0,0) block. The discrete Hodge is that
//                     form, so it is taken rather than rebuilt.
//
//   ||div sigma||^2   is not B^T B. The facet dof is the measure-weighted
//                     moment, so (B sigma)_E = int_E div sigma, the integral;
//                     div sigma is constant on the cell, so its square
//                     integrates to (B sigma)_E^2 / |E| and the term is
//                     B^T diag(1/|E|) B. On a uniform mesh that is a constant
//                     factor and passes for a tuning knob; on a graded one it
//                     varies cell by cell and no constant repairs it.
//
//   ||u||^2           is diag(|E|): a cell dof is the value on the cell.
//
// So one quantity -- the cell measure -- fixes every block.
struct SpaceNorm {
  // the factors of X, as index sets, first factor first
  std::vector<std::vector<int>> factors;
  // the L2 weight of every unknown of factors 1.., one vector per such factor:
  // the measure of the cell that unknown belongs to.
  std::vector<std::vector<double>> l2_weight;

  // The graph term is not stated separately: it is B^T W^{-1} B with W the
  // multiplier block above, which is the whole content of
  //
  //     P = diag( M + B^T W^{-1} B ,  W ) .
  //
  // W is the L2 mass in the dof basis, and that basis differs between the two.
  // Flow's cell unknown is the value of p on the cell, so ||p||^2 = sum p^2 |E|
  // and W = |E|. Elasticity's cell unknowns are moments -- u_dof = int_E u, as
  // CauchyMechanicsModel::displacement shows by dividing by the measure to
  // report a mean -- so ||u||^2 = sum (u_dof/|E|)^2 |E| = sum u_dof^2 / |E| and
  // W = 1/|E|. The rotation is a moment likewise.
  //
  // Measured on the Lame annulus, cond(P^{-1}A) with W = |E| is 8e2 and rising
  // with refinement; with W = 1/|E| it is 3.2, 3.4, 3.5 over the same three
  // meshes -- flat.

  // A constrained unknown is not in the space. Its row of A is the constraint,
  // scale * e_i^T, not a form; leaving the norm's entries there preconditions an
  // equation that is not the one being solved, and the iteration count starts
  // growing with h again. P carries the same row, so those unknowns contribute
  // the identity to P^{-1}A and drop out of the Krylov space.
  // Which multipliers contribute a graph term: the differential constraint does
  // (factor 1), an algebraic one does not. AFW's inf-sup is proved with
  // ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 -- skw is bounded
  // L^2 -> L^2, so the rotation adds nothing to the stress norm.
  std::size_t differential_factors{1};
  bool carries_graph_term(std::size_t f) const { return f >= 1 && f <= differential_factors; }

  std::vector<int> pinned;
  std::vector<double> pinned_diagonal;

  // Rank-one additions to the first block: each contributes w (t . x)^2, with t
  // supported on `rank_one_dofs`. What the three-field stress norm needs to stay
  // lambda-free; see CauchyMechanicsModel::norm_trace_terms.
  std::vector<std::vector<int>> rank_one_dofs;
  std::vector<std::vector<double>> rank_one_row;
  std::vector<double> rank_one_weight;

  // The de Rham maps an auxiliary-space solver needs.
  //
  // ADS preconditions an H(div) operator by splitting it along the complex --
  // a field becomes a vector potential in H(curl) plus a part carried by the
  // vertex spaces -- and the maps that take it there are the discrete gradient
  // (edges x vertices) and curl (faces x edges). Those are not a new
  // construction: they are the boundary operators of the complex as stored, so
  // a library built on a chain complex hands them over instead of
  // reconstructing them from an element table.
  //
  // Empty means no auxiliary solver is possible and a factorization is used.
  struct Incidence {
    int rows{0}, cols{0};
    std::vector<int> row, col;
    std::vector<double> value;

    bool empty() const { return value.empty(); }
  };
  Incidence discrete_gradient;  // d_1 : vertices -> edges
  Incidence discrete_curl;      // d_2 : edges -> faces

  // ADS also needs the vertex coordinates, one row of `space_dim` per column of
  // the gradient. They are not redundant with the maps above: the complex is
  // metric-free, and the auxiliary spaces the solver builds are spaces of
  // piecewise linear fields -- it recovers their interpolation by applying the
  // maps to the coordinate functions x, y, z. This is the whole metric content
  // ADS asks for.
  std::vector<double> vertex_coordinates;  // row-major, n_vertices x space_dim
  int space_dim{3};

  // Which process owns each entity the two maps address -- vertices, edges,
  // faces -- by the same rule and the same partition that owns the unknowns.
  //
  // The maps above are stated in the complex's own numbering, which is a
  // serial numbering. Distributed, hypre needs each of the three spaces laid
  // out across the ranks, and the faces must be laid out exactly as the block
  // is: its row i has to be that block's row i. That lets the solver renumber
  // all three consistently, without asking the complex to be distributed.
  //
  // Empty on one process, where there is nothing to lay out.
  std::vector<std::vector<int>> entity_owner;  // [k][entity]

  // The lowest-order subspace of the first factor, when the first factor is
  // not itself lowest order.
  //
  // ADS is written for one unknown per facet. Flow's RT space is that already;
  // the AFW stress space is not -- a facet carries d traction components, each
  // measured against the d functions of the facet P_1 basis, so d^2 unknowns
  // sit on it. The auxiliary-space argument still applies, and this is the map
  // it applies through: the injection of the facet-constant moments, which are
  // a subset of the degrees of freedom rather than a computed interpolation,
  // because the space is defined by its moments and the constant one is one of
  // them.
  //
  // Columns are ordered component-major -- all facets of component 0, then
  // component 1 -- so that each component is a contiguous run of the coarse
  // space and ADS can be given it as the scalar H(div) problem it expects.
  //
  // Rows are global unknowns; build_riesz maps them into the block.
  Incidence lowest_order;
  int lowest_order_components{1};

  bool empty() const { return factors.empty(); }
};

struct SolverOptions {
  // "direct" is KSPPREONLY with a full factorization. Anything else names a
  // Krylov method PETSc knows: "gmres", "minres", "cg", "fgmres".
  std::string method{"direct"};
  // the factorization package, used when the preconditioner is one:
  // "superlu", "mumps", or "petsc" for the built-in.
  std::string factorization{"superlu"};
  // the PC type: "lu", "ilu", "jacobi", "none", "fieldsplit", ...
  std::string preconditioner{"lu"};
  // How the Riesz blocks are inverted. The first factor is SPD but large -- it
  // is most of the unknowns -- so a complete factorization of it costs about
  // what a direct solve of the whole system costs, in time and in fill: exact,
  // and it does not scale.
  //
  // An approximate inverse is still a Riesz map as long as it is spectrally
  // equivalent to the block: the iteration count rises by a constant and stops
  // depending on the mesh. "gamg" is algebraic multigrid, which is the
  // scalable choice; "lu" is the exact one, for small problems and for
  // checking that an approximation is what changed an answer.
  // Eliminate the first field first, when the caller says it can be. A
  // diagonal star -- diagonal_tpfa, diagonal_afw -- makes that block diagonal,
  // and then the flux or the stress is divided out cell by cell and what is
  // solved is the finite volume system itself. Off is the saddle point, which
  // every other product must have.
  bool condense{true};
  std::string riesz_block_pc{};
  // The Riesz block is SPD and is solved as such, by MUMPS.
  //
  // `factorization` defaults to SuperLU because the whole system is an
  // indefinite saddle point. The first Riesz factor has no such structure: a
  // material inner product plus B^T W^-1 B, symmetric positive definite. So it
  // takes a Cholesky, half the fill and half the work of an LU.
  //
  // Which package is not a detail. Measured on the H(div) block of the 22k-cell
  // polyhedral mesh (77k unknowns), solving to 1e-9:
  //
  //     Cholesky / MUMPS    64 iterations    1.1 s
  //     Cholesky / PETSc    38              64.1 s
  //     Cholesky / SuperLU  -- SuperLU has no Cholesky
  //
  // PETSc's own factorization takes fewer iterations, because it is the more
  // exact of the two, and is sixty times slower: it orders the matrix
  // naturally, and the fill of a natural ordering on an unstructured
  // three-dimensional block is ruinous. MUMPS reorders before it factors.
  //
  // An empty value falls back to `factorization`, which is SuperLU -- and
  // SuperLU cannot do a Cholesky at all, so that fallback is an error rather
  // than a slow path. Naming MUMPS here keeps the default working.
  std::string riesz_block_factorization{"mumps"};
  // How the first factor is inverted, and it is a memory decision.
  //
  //   0   exact: a complete factorization. 21 iterations flat, and fill that
  //       grows with the block -- 450 MB at 33k unknowns, extrapolating to
  //       tens of gigabytes on a mesh of tens of thousands of polyhedra.
  //   >0  that many preconditioned CG steps instead. 25-28 iterations, so the
  //       outer count barely moves, and no fill: 46 MB at the same size. The
  //       outer method must then be flexible, which is applied automatically.
  //   -1  choose: exact while the block is small enough to factor, inexact
  //       above it. The threshold is where the fill stops being affordable
  //       rather than where the method changes character.
  int riesz_block_its{-1};
  double riesz_block_rtol{1e-4};
  // First-factor unknowns above which the exact solve is refused. Set from
  // where the fill stops being affordable rather than from where the method
  // changes: with MUMPS the H(div) block of a 22k-cell polyhedral mesh -- 77k
  // unknowns -- factors in 1.5 s, so the limit sits well above it.
  int riesz_exact_limit{400000};
  // First-factor unknowns above which the auxiliary-space solver is preferred
  // to the exact one, when the complex makes it possible at all. Against a
  // factorization that reorders, the exact block is the cheaper way to apply P
  // at every size that fits in memory -- see the crossover note in factorize()
  // -- so the default matches riesz_exact_limit and ADS is reached only where
  // the factorization is refused for its fill.
  int riesz_ads_limit{400000};
  // The coarse solve of the two-level cycle. 0 applies the per-component ADS
  // cycles once; >0 wraps them in that many CG steps to riesz_coarse_rtol.
  // Under the scale-free norm the graph term is what the block is made of,
  // and the graph term lives on the facet constants -- the coarse space --
  // so one ADS sweep there leaves the cycle's contraction to hypre's: the
  // inner CG under the cycle grew 56 -> 100 -> 168 across three refinements
  // of the Lame annulus. Solved to a tolerance, the coarse correction is what
  // the two-level argument assumes, and the inner count is what it should be.
  // A coarse CG makes the cycle a varying operator, so the block's Krylov
  // method is promoted to flexible CG when this is on.
  int riesz_coarse_its{0};
  double riesz_coarse_rtol{1e-2};
  // Renumber by the mesh partition when there is more than one process. Off,
  // the rows are split by index and a rank's unknowns come from all over the
  // mesh; the answer is the same either way, and the difference is how much of
  // the matrix crosses a process boundary.
  bool partition{true};
  double rtol{1e-10};
  double atol{1e-50};
  int max_iterations{1000};

  bool direct() const { return method == "direct"; }
};

class PetscSolver final : public LinearSolver {
 public:
  // `type` selects the factorization package; MUMPS is the default because
  // these systems are indefinite. An empty prefix means the KSP also reads
  // command-line options, so an iterative method can be selected at run time.
  // SUPERLU BY DEFAULT, NOT MUMPS.
  //
  // The mixed form is an INDEFINITE SADDLE POINT: the multiplier blocks put
  // structural zeros on the diagonal, so a factorization lives or dies on its
  // pivoting. MUMPS sizes its working array from a symbolic estimate, and
  // delayed pivots on a saddle point overrun that array -- it SEGVs inside the
  // factorization, on a well-posed system, returning no error at all. Raising
  // ICNTL(14) to 200% does not rescue it. That is what cost benchmark 3 its
  // first working run.
  //
  // SuperLU is an UNSYMMETRIC supernodal factorization with genuine partial
  // pivoting: it allocates as it goes, so there is no estimate to overrun, and
  // it makes no assumption about the sign structure of the diagonal. It is the
  // right default for this class of system; at these sizes the cost difference
  // is not what decides anything.
  //
  // The choice stays a constructor argument, and MIMETIKA_FACTOR overrides it
  // at run time, so a solver can be swapped without a rebuild when one of them
  // misbehaves -- which is exactly how this was diagnosed.
  explicit PetscSolver(SolverOptions options = {}, std::string prefix = "")
      : opts_(std::move(options)), prefix_(std::move(prefix)) {
    PetscSession::instance();
  }

  const SolverOptions& options() const { return opts_; }

  // What factorize() spent, so a caller can report the two halves of it
  // separately: the matrix is linear in the assembly, the preconditioner is
  // what decides whether a mesh is reachable at all.
  double matrix_seconds() const { return matrix_seconds_; }
  double preconditioner_seconds() const { return preconditioner_seconds_; }

  // How much of the matrix crosses a process boundary. PETSc stores an MPIAIJ
  // row in two pieces -- the columns this rank owns and the rest -- and the
  // second is exactly what a mat-vec has to communicate. It is the measure of
  // a partition that does not depend on the machine, the load or the timer.
  double off_rank_fraction() const {
    const double total = local_entries_ + off_rank_entries_;
    return total > 0.0 ? off_rank_entries_ / total : 0.0;
  }

  // The factors of the product space. Required by the "riesz" preconditioner
  // and ignored by every other one.
  void set_norm(SpaceNorm s) { norm_ = std::move(s); }

  // Which unknowns may be divided out, named by the caller because only the
  // model knows which field is the flux or the stress. Naming them is a
  // permission and not an instruction: the matrix is asked whether that block
  // really is diagonal, and a product whose star couples a cell's facets is
  // solved as the saddle point it is.
  void set_condensable(std::vector<int> dofs) { condensable_ = std::move(dofs); }

  // Who owns each unknown, one rank per global unknown, in the caller's own
  // numbering.
  //
  // Without this the rows are split by index, and an index is not a place: a
  // rank's rows are then scattered over the whole mesh, every mat-vec is
  // nearly all off-rank communication, and the layout that a parallel
  // preconditioner assumes -- that a rank holds a subdomain -- is absent.
  // With it the solver renumbers so that each rank's unknowns are the
  // contiguous block PETSc requires, and hands the answer back in the
  // caller's numbering as if nothing had happened.
  //
  // Ignored on one process, where there is nothing to renumber for.
  void set_owners(std::vector<int> owner_of_dof) { owners_ = std::move(owner_of_dof); }

  // Distributed assembly needs nothing from the solver, by convention. A
  // process assembles every cell that contributes to a row it owns -- its own
  // and its halo -- so those rows arrive complete and the rest are dropped,
  // exactly as they are when the assembly is replicated. No stash, no
  // exchange, and one code path for both.
  //
  // The alternative, assembling only owned cells and letting the matrix carry
  // the rest, was measured: PETSc's off-process stash turned a 2 second matrix
  // into a 310 second one on a 22k-cell mesh.
  void set_local_assembly(bool) {}

  std::string name() const override {
    return "petsc/" +
           (opts_.direct() ? opts_.factorization : opts_.method + "+" + opts_.preconditioner);
  }

  ~PetscSolver() override { release(); }
  PetscSolver(const PetscSolver&) = delete;
  PetscSolver& operator=(const PetscSolver&) = delete;

  // Bind the operator once. A transient linear problem at constant dt has a
  // tangent that never moves, so the assembly, the symbolic analysis and the
  // numeric factorization are all done once and every step after that is a
  // back-substitution. Terzaghi takes 400 steps and the borehole 400: paying
  // MUMPS for each of them is the difference between minutes and seconds, and
  // nothing in the answer changes.
  void factorize(const SparseSystem& A) {
    release();
    n_ = static_cast<PetscInt>(A.n);
    claim_rows();
    bound_ = &A;  // build_riesz reads the triplets, so bind before the KSP
    const auto t0 = std::chrono::steady_clock::now();
    build_matrix(A);
    const auto t1 = std::chrono::steady_clock::now();
    build_ksp();
    if (distributed_) {
      check(VecCreateMPI(comm_, n_local_, n_, &rhs_), "VecCreate");
    } else {
      check(VecCreateSeq(PETSC_COMM_SELF, n_, &rhs_), "VecCreate");
    }
    check(VecDuplicate(rhs_, &sol_), "VecDuplicate");
    // force the factorization now rather than on the first solve, so that the
    // cost shows up where it is paid
    check(KSPSetUp(ksp_), "KSPSetUp");
    measure_locality();
    const auto t2 = std::chrono::steady_clock::now();
    matrix_seconds_ = slowest(std::chrono::duration<double>(t1 - t0).count());
    preconditioner_seconds_ = slowest(std::chrono::duration<double>(t2 - t1).count());
    bound_ = &A;
  }

  // Solve against the bound operator. Refuses an unbound solver rather than
  // silently factorizing, because a caller reaching here without binding has
  // a different bug than a slow one.
  SolveReport solve(const std::vector<double>& b, std::vector<double>& x) {
    if (ksp_ == nullptr) {
      throw std::logic_error("PetscSolver::solve: no operator bound; call factorize() first");
    }
    if (static_cast<PetscInt>(b.size()) != n_) {
      throw std::invalid_argument("PetscSolver: right-hand side size");
    }
    for (PetscInt i = own_begin_; i < own_end_; ++i) {
      const auto source =
          static_cast<std::size_t>(old_of_.empty() ? i : old_of_[static_cast<std::size_t>(i)]);
      check(VecSetValue(rhs_, i, b[source], INSERT_VALUES), "VecSetValue");
    }
    check(VecAssemblyBegin(rhs_), "VecAssembly");
    check(VecAssemblyEnd(rhs_), "VecAssembly");
    return run(*bound_, b, x);
  }

  // the one-shot form: bind, solve, and keep the factorization in case the
  // same operator comes back
  SolveReport solve(const SparseSystem& A, const std::vector<double>& b,
                    std::vector<double>& x) override {
    if (opts_.condense && !condensable_.empty() && block_is_diagonal(A, condensable_)) {
      return solve_condensed(A, b, x);
    }
    return ordinary(A, b, x);
  }

  // Build what a solve would build, and nothing more -- for the caller that
  // measures the two assemblies without iterating. Measuring the
  // factorization of a saddle the solve immediately eliminates reports a
  // cost, in time and in memory, that no run ever pays: on a facet-diagonal
  // star the block is divided out and the preconditioner that matters is the
  // reduced system's. So the same gate decides here as in solve(), and the
  // report carries the same condensation facts.
  SolveReport prepare(const SparseSystem& A, const std::vector<double>& b) {
    SolveReport r;
    r.converged = true;
    r.reason = "assembled";
    if (opts_.condense && !condensable_.empty() && block_is_diagonal(A, condensable_)) {
      PetscMPIInt size = 1, rank = 0;
      MPI_Comm_size(PETSC_COMM_WORLD, &size);
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      const bool spread = size > 1;
      if (!spread || owners_.size() == static_cast<std::size_t>(A.n)) {
        const auto t0 = std::chrono::steady_clock::now();
        const Condensation c =
            condense(A, b, condensable_, spread ? &owners_ : nullptr, static_cast<int>(rank));
        const auto t1 = std::chrono::steady_clock::now();
        SolverOptions inner = opts_;
        inner.condense = false;
        if (inner.preconditioner == "riesz" || inner.preconditioner == "exact") {
          inner.preconditioner = "hypre";
          if (inner.direct()) inner.method = "gmres";
        }
        PetscSolver sub(inner);
        if (spread) {
          std::vector<int> reduced_owners(c.rest.size(), 0);
          for (std::size_t i = 0; i < c.rest.size(); ++i) {
            reduced_owners[i] = owners_[static_cast<std::size_t>(c.rest[i])];
          }
          sub.set_owners(std::move(reduced_owners));
        }
        sub.factorize(c.reduced);
        r.condensed = true;
        r.condensed_dofs = c.size();
        r.matrix_seconds = slowest(std::chrono::duration<double>(t1 - t0).count()) +
                           sub.matrix_seconds();
        r.preconditioner_seconds = sub.preconditioner_seconds();
        r.block_solver = inner.preconditioner;
        return r;
      }
    }
    factorize(A);
    r.matrix_seconds = matrix_seconds_;
    r.preconditioner_seconds = preconditioner_seconds_;
    return r;
  }

  // A RESIDUAL THIS LARGE IS NOT A CONVERGED SOLVE, WHATEVER THE
  // FACTORIZATION SAID.
  //
  // A direct solver handed a singular matrix returns a vector, reports
  // CONVERGED, and leaves 1e18 in the answer -- measured, on diagonal_afw
  // under an essential stress condition (relative residual 9.4e3) and on the
  // Kuhn tetrahedra (5.8e3). The residual is one pass over the triplets, which
  // is nothing beside a factorization, and it is the only thing that
  // distinguishes an answer from a vector.
  //
  // The bound is one: a relative residual of 1 is what the answer x = 0
  // leaves, so anything at or above it is not a solution by any reading. The
  // singular cases measured are 9.4e3, 5.8e3 and 3e20; a
  // converged condensed solve leaves 8e-3, because recovering the eliminated
  // field divides by a small diagonal and amplifies whatever the reduced solve
  // left. A tighter bound would call that a failure, and it is not one.
  //
  // Not on several processes. A rank holds its own rows and its halo, and the
  // eliminated field is recovered only where its row is here, so the sum below
  // is over rows whose columns this rank cannot all evaluate. The number would
  // be large and mean nothing.
  SolveReport checked(const SparseSystem& A, const std::vector<double>& b,
                      const std::vector<double>& x, SolveReport r) const {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (!r.converged || size > 1) return r;
    r.residual = true_residual(A, b, x);
    if (!(r.residual < 1.0)) {
      r.converged = false;
      char said[64];
      std::snprintf(said, sizeof(said), "%.2e", r.residual);
      r.reason = "the operator is singular, or the factorization failed: the answer leaves a "
                 "relative residual of " +
                 std::string(said);
    }
    return r;
  }

 private:
  // The ordinary path, with the one retry it owns.
  //
  // hypre builds the auxiliary-space block from the maps and coordinates it is
  // given, and on some partitions of some meshes the cycle it returns is
  // near-singular -- measured on the 22k-cell polyhedral mesh: fine serially
  // and at 2..7, 9 and 10 ranks, stagnation and then breakdown at 8, 11 and
  // 12, with no hypre knob (relaxation, cycle type, AMG thresholds) changing
  // the outcome. That is a property of the partition rather than of the
  // problem -- the same block under the exact factorization takes its usual
  // count -- so the failure is not handed back: the block is replaced by the
  // exact one, the solve is repeated once, and the report says so.
  SolveReport ordinary(const SparseSystem& A, const std::vector<double>& b,
                       std::vector<double>& x) {
    if (bound_ != &A) factorize(A);
    SolveReport r = solve(b, x);
    if (!r.converged && ads_failed_) {
      const std::string why = r.reason.substr(0, r.reason.find(" -- "));
      opts_.riesz_block_pc = "cholesky";
      factorize(A);
      r = solve(b, x);
      r.reason += " -- the auxiliary-space (ADS) block was unusable on this partition (" + why +
                  ") and was replaced by the exact one";
    }
    r.matrix_seconds = matrix_seconds_;
    r.preconditioner_seconds = preconditioner_seconds_;
    r.off_rank_fraction = off_rank_fraction();
    return checked(A, b, x, r);
  }

  // The condensed solve, which is the same solver on a smaller problem.
  //
  // S is handed to a second PetscSolver with this one's options and no
  // condensable set, so it takes the ordinary path -- the same methods, the
  // same factorizations, the same reporting -- and there is one implementation
  // of a solve rather than two. What this adds is the elimination either side
  // of it, and the timing of the elimination itself, which belongs to the
  // matrix rather than to the iteration.
  //
  // One process. The elimination reads whole rows and whole columns of the
  // eliminated unknowns, and distributed assembly gives a rank its own rows and
  // its halo -- so on several processes the outer products would be emitted
  // twice on the halo and missing nowhere. Rather than half-condense, a
  // distributed run takes the saddle point, which is correct.
  SolveReport solve_condensed(const SparseSystem& A, const std::vector<double>& b,
                              std::vector<double>& x) {
    // PETSC_COMM_WORLD, not comm_: comm_ is chosen inside factorize(), which
    // has not run yet, so it still says SELF here.
    PetscMPIInt size = 1, rank = 0;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const bool spread = size > 1;

    // Without an ownership there is nothing to condense on several processes.
    // Each rank holds its own rows and its halo, so a rank that cannot tell
    // which reduced rows are its own would emit some twice and some never. A
    // caller that distributes without set_owners gets the saddle point, which
    // is correct.
    if (spread && owners_.size() != static_cast<std::size_t>(A.n)) {
      return ordinary(A, b, x);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const Condensation c =
        condense(A, b, condensable_, spread ? &owners_ : nullptr, static_cast<int>(rank));

    const auto t1 = std::chrono::steady_clock::now();

    SolverOptions inner = opts_;
    inner.condense = false;

    // The reduced system is itself a saddle point and keeps its Riesz map.
    //
    // Eliminating the first field does not flatten what is left. For
    // diagonal_vem the reduced unknowns are the displacement and the
    // volumetric stress, and that pair is symmetric quasi-definite --
    // measured: u positive (5.5e1 .. 5.8e3), p negative (-7.9e-2 .. -3.7e-2).
    // An algebraic multigrid over the whole of it preconditions a saddle point
    // as though it were elliptic, which is the wrong tool.
    //
    // What it takes is the same map one level down: P = diag(A + B^T W^-1 B, W)
    // over the reduced unknowns, which is what build_riesz already assembles.
    // It needs the split, and the caller's norm carries it -- factor 0 is the
    // field being eliminated, so the reduced norm is the factors after it, with
    // their indices carried through `slot`.
    //
    // Without that split -- a norm whose multipliers were merged into one
    // factor, which is what the full system wants -- there is nothing to split
    // on, and the fallback is the multigrid.
    SpaceNorm reduced_norm;
    const bool reduced_riesz =
        (inner.preconditioner == "riesz" || inner.preconditioner == "exact") &&
        norm_.factors.size() >= 3 && norm_.l2_weight.size() + 1 == norm_.factors.size();
    if (reduced_riesz) {
      for (std::size_t f = 1; f < norm_.factors.size(); ++f) {
        std::vector<int> mapped;
        std::vector<double> weight;
        mapped.reserve(norm_.factors[f].size());
        for (std::size_t k = 0; k < norm_.factors[f].size(); ++k) {
          const int slot = c.slot[static_cast<std::size_t>(norm_.factors[f][k])];
          if (slot < 0) continue;  // eliminated, so not a factor of what is left
          mapped.push_back(slot);
          if (f >= 2) weight.push_back(norm_.l2_weight[f - 1][k]);
        }
        reduced_norm.factors.push_back(std::move(mapped));
        if (f >= 2) reduced_norm.l2_weight.push_back(std::move(weight));
      }
      // the pinned unknowns, carried through the same map
      for (std::size_t k = 0; k < norm_.pinned.size(); ++k) {
        const int slot = c.slot[static_cast<std::size_t>(norm_.pinned[k])];
        if (slot < 0) continue;
        reduced_norm.pinned.push_back(slot);
        reduced_norm.pinned_diagonal.push_back(
            k < norm_.pinned_diagonal.size() ? norm_.pinned_diagonal[k] : 1.0);
      }
      inner.preconditioner = "riesz";
      if (inner.direct()) inner.method = "gmres";
    } else if (inner.preconditioner == "riesz" || inner.preconditioner == "exact") {
      // A Riesz map is a statement about a space that is no longer there, when
      // the split went with it: no H(div) block survives the elimination, so
      // what is left is preconditioned as the finite volume operator it is.
      inner.preconditioner = "hypre";
      if (inner.direct()) inner.method = "gmres";
    }

    PetscSolver sub(inner);
    if (reduced_riesz) sub.set_norm(std::move(reduced_norm));
    if (spread) {
      // the reduced unknowns inherit the partition they came from: a cell's
      // pressure, displacement and rotation belong to whoever owned the cell
      std::vector<int> reduced_owners(c.rest.size(), 0);
      for (std::size_t i = 0; i < c.rest.size(); ++i) {
        reduced_owners[i] = owners_[static_cast<std::size_t>(c.rest[i])];
      }
      sub.set_owners(std::move(reduced_owners));
    }
    std::vector<double> y;
    SolveReport r = sub.solve(c.reduced, c.rhs, y);
    x = c.expand(y, b);

    r.condensed = true;
    r.condensed_dofs = c.size();
    if (r.block_solver.empty()) r.block_solver = inner.preconditioner;
    // the residual the caller is owed is the original system's, not the
    // reduced one's, and the same bound applies to it
    r = checked(A, b, x, r);

    return r;
  }

  // P, assembled from the norm above. Nothing here decides anything: every
  // block is the term the norm names, read in this dof basis.
  //
  // It is a second matrix, not an edit of the sub-solvers. PETSc takes the
  // preconditioner from Pmat in KSPSetOperators(ksp, Amat, Pmat) and a
  // fieldsplit reads its diagonal blocks from there. Handing the blocks to the
  // sub-KSPs instead -- after PCSetUp, by KSPSetOperators on each -- is undone
  // the next time the outer KSP sets up and rebuilds them from Pmat, which
  // leaves the preconditioner silently equal to the operator: it converges on
  // nothing and reports DIVERGED_ITS.
  void build_riesz(KSP ksp, PC pc) {
    const std::size_t nf = norm_.factors.size();
    if (nf < 2 || norm_.l2_weight.size() != nf - 1) {
      throw std::invalid_argument(
          "PetscSolver: the 'riesz' preconditioner needs the space norm; call set_norm() with "
          "one index set per factor and L2 weights for every factor after the first");
    }

    // A factor is a contiguous run: a general index set makes
    // MatCreateSubMatrix search for every row it is asked for, and on a block
    // of several hundred thousand that search is the whole cost of building
    // the preconditioner -- minutes, against the second the extraction itself
    // takes from a stride.
    // An index set is per process. Every rank names the unknowns of the factor
    // that it owns -- the union over ranks is the factor, which is what
    // PCFIELDSPLIT wants; naming all of them everywhere would give each field
    // n_ranks copies of itself.
    std::vector<IS> sets(nf, nullptr);
    for (std::size_t b = 0; b < nf; ++b) {
      const auto& all = norm_.factors[b];
      std::vector<PetscInt> idx;
      idx.reserve(all.size());
      for (const int g : all) {
        if (owns(at(g))) idx.push_back(at(g));
      }
      std::sort(idx.begin(), idx.end());
      const auto m = static_cast<PetscInt>(idx.size());
      bool contiguous = !idx.empty();
      for (std::size_t k = 1; k < idx.size() && contiguous; ++k) {
        contiguous = idx[k] == idx[k - 1] + 1;
      }
      if (contiguous) {
        check(ISCreateStride(comm_, m, idx.front(), 1, &sets[b]), "ISCreateStride");
      } else {
        check(ISCreateGeneral(comm_, m, idx.data(), PETSC_COPY_VALUES, &sets[b]),
              "ISCreateGeneral");
        check(ISSort(sets[b]), "ISSort");
      }
    }

    // P, built from the triplets in one pass.
    //
    // Every block of P is already present in the assembly, so none of it needs
    // to be extracted or multiplied out:
    //
    //   material   the (0,0) entries of A, taken as they stand
    //   graph      B^T W^-1 B, and B is one row of A per multiplier. A row has
    //              only as many entries as the cell has facets, so the outer
    //              product of a row with itself is a handful of entries and the
    //              whole term is a single pass -- no sparse matrix product, and
    //              no matrix the size of the first factor to hold its result.
    //   L2         the multiplier diagonal, W
    //
    // Doing it through MatCreateSubMatrix and MatTransposeMatMult instead costs
    // an extraction of a block that is most of the operator and a product whose
    // intermediate is larger again: on a mesh of tens of thousands of polyhedra
    // that is the whole time to reach a solve, and gigabytes that this pass
    // never allocates.
    const SparseSystem& A = *bound_;
    std::vector<int> factor_of(static_cast<std::size_t>(n_), -1);
    std::vector<double> weight(static_cast<std::size_t>(n_), 0.0);
    for (std::size_t f = 0; f < nf; ++f) {
      const auto& idx = norm_.factors[f];
      for (std::size_t k = 0; k < idx.size(); ++k) {
        factor_of[static_cast<std::size_t>(idx[k])] = static_cast<int>(f);
        if (f >= 1) weight[static_cast<std::size_t>(idx[k])] = norm_.l2_weight[f - 1][k];
      }
    }
    // W is the Schur scale, and for one row the operator states it.
    //
    // Everything the norm does to a multiplier is done because its row says
    // nothing about itself: a constraint has no diagonal, so the scale of its
    // multiplier has to be named from outside, and above it is named by the
    // measure. The four-field form has one row that is not of that kind. The
    // total pressure is a definition, p = lambda div u, and its row carries
    // c_p |E| p with c_p = d/(2 mu) + 1/lambda -- a scale the operator states,
    // and not the measure. Taking the stand-in there instead rescales the
    // pressure block by c_p |E|^2; on 6^3 cells the solve stopped converging.
    //
    // Its B^T W^-1 B is then not a graph term but the Schur complement of a
    // block that is invertible, which is to say the trace part of the stress
    // norm this form asks for:
    //
    //   ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2
    //                                  + c_p^-1 ||(2 mu)^-1 tr sigma||^2
    //
    // The deviatoric compliance is what A is once p carries the trace, so
    // without that term nothing in the norm sees a trace at all. It is kept
    // for exactly the discretization that needs it: with a lumped compliance
    // the mass controls the deviator only, and dropping the term costs
    // diagonal_afw a quarter of its count (1192 against 960), while
    // stabilized_bdm, whose mass is a real one, hardly moves (135 against 148).
    //
    // Both follow from the diagonal alone -- no field named here, and no
    // formulation known here.
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto i = static_cast<std::size_t>(A.row[k]);
      if (A.col[k] == A.row[k] && factor_of[i] >= 1 && A.value[k] != 0.0) {
        weight[i] = std::abs(A.value[k]);
      }
    }

    std::vector<char> is_pinned(static_cast<std::size_t>(n_), 0);
    for (const int i : norm_.pinned) is_pinned[static_cast<std::size_t>(i)] = 1;

    // the rows of the constraint blocks, gathered. Only the multiplier rows are
    // kept, and each holds a few entries, so this is a small fraction of A.
    std::vector<Index> b_begin(static_cast<std::size_t>(n_) + 1, 0);
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto i = static_cast<std::size_t>(A.row[k]);
      if (factor_of[i] >= 1 && norm_.carries_graph_term(static_cast<std::size_t>(factor_of[i])) &&
          factor_of[static_cast<std::size_t>(A.col[k])] == 0) {
        ++b_begin[i + 1];
      }
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_); ++i) b_begin[i + 1] += b_begin[i];
    std::vector<Index> b_col(static_cast<std::size_t>(b_begin.back()));
    std::vector<double> b_val(b_col.size());
    {
      std::vector<Index> at(b_begin.begin(), b_begin.end() - 1);
      for (std::size_t k = 0; k < A.nnz(); ++k) {
        const auto i = static_cast<std::size_t>(A.row[k]);
        if (factor_of[i] >= 1 && norm_.carries_graph_term(static_cast<std::size_t>(factor_of[i])) &&
            factor_of[static_cast<std::size_t>(A.col[k])] == 0) {
          const auto slot = static_cast<std::size_t>(at[i]++);
          b_col[slot] = A.col[k];
          b_val[slot] = A.value[k];
        }
      }
    }

    // The graph term is never materialized as triplets.
    //
    // B^T W^-1 B is an outer product per constraint row, so writing it as
    // triplets costs the square of each row's entry count: on this mesh that
    // is 122 million of them against 90 million for the material block, and
    // they then have to be sorted alongside it. A row's outer product is a
    // dense block over the columns that row touches, which is exactly what one
    // MatSetValues call takes -- so the term goes straight into the matrix,
    // and only the material block passes through a triplet list.
    std::vector<Index> p_row, p_col;
    std::vector<double> p_val;
    const auto emit = [&](Index i, Index j, double v) {
      if (is_pinned[static_cast<std::size_t>(i)] || is_pinned[static_cast<std::size_t>(j)]) return;
      p_row.push_back(i);
      p_col.push_back(j);
      p_val.push_back(v);
    };
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      if (factor_of[static_cast<std::size_t>(A.row[k])] == 0 &&
          factor_of[static_cast<std::size_t>(A.col[k])] == 0) {
        emit(A.row[k], A.col[k], A.value[k]);
      }
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_); ++i) {
      if (factor_of[i] >= 1 && !is_pinned[i]) {
        emit(static_cast<Index>(i), static_cast<Index>(i), weight[i]);
      }
    }
    // the rank-one corrections, as outer products over the dofs each touches
    for (std::size_t t = 0; t < norm_.rank_one_weight.size(); ++t) {
      const std::vector<int>& idx = norm_.rank_one_dofs[t];
      const std::vector<double>& row = norm_.rank_one_row[t];
      const double w = norm_.rank_one_weight[t];
      for (std::size_t i = 0; i < idx.size(); ++i) {
        for (std::size_t j = 0; j < idx.size(); ++j) {
          emit(idx[i], idx[j], w * row[i] * row[j]);
        }
      }
    }
    for (std::size_t k = 0; k < norm_.pinned.size(); ++k) {
      const Index i = norm_.pinned[k];
      p_row.push_back(i);
      p_col.push_back(i);
      p_val.push_back(k < norm_.pinned_diagonal.size() ? norm_.pinned_diagonal[k] : 1.0);
    }

    // preallocate for both parts: the material triplets, and every column of a
    // constraint row against every other column of that row. Counted over all
    // rows and reduced afterwards, because a constraint row's outer product
    // reaches facets this rank may not own.
    std::vector<PetscInt> diag, off;
    count_into(diag, off, p_row, p_col);
    for (std::size_t r = 0; r < static_cast<std::size_t>(n_); ++r) {
      const auto b = static_cast<std::size_t>(b_begin[r]);
      const auto e = static_cast<std::size_t>(b_begin[r + 1]);
      for (std::size_t a = b; a < e; ++a) {
        const PetscInt i = at(b_col[a]);
        for (std::size_t c = b; c < e; ++c) {
          const PetscInt j = at(b_col[c]);
          const bool in_diagonal = !distributed_ || (owns(i) && owns(j));
          ++(in_diagonal ? diag : off)[static_cast<std::size_t>(i)];
        }
      }
    }
    finish_counts(diag, off);
    Mat P = new_matrix(diag, off, "MatCreate(P)");
    diag = std::vector<PetscInt>();
    off = std::vector<PetscInt>();

    scatter_by_row(P, p_row, p_col, p_val, n_);
    p_row = std::vector<Index>();
    p_col = std::vector<Index>();
    p_val = std::vector<double>();

    // one dense block per constraint row: (1/W_r) b_r b_r^T, over the columns
    // that row touches. Pinned unknowns are not in the space, so a row is
    // skipped where it would reach one.
    {
      std::vector<PetscInt> cols;
      std::vector<PetscScalar> blk;
      for (std::size_t r = 0; r < static_cast<std::size_t>(n_); ++r) {
        const auto b = static_cast<std::size_t>(b_begin[r]);
        const auto e = static_cast<std::size_t>(b_begin[r + 1]);
        if (e == b) continue;
        const double inv_w = 1.0 / weight[r];
        cols.clear();
        for (std::size_t a = b; a < e; ++a) {
          if (!is_pinned[static_cast<std::size_t>(b_col[a])]) cols.push_back(at(b_col[a]));
        }
        if (cols.empty()) continue;
        const std::size_t m = cols.size();
        blk.assign(m * m, 0.0);
        std::size_t ia = 0;
        for (std::size_t a = b; a < e; ++a) {
          if (is_pinned[static_cast<std::size_t>(b_col[a])]) continue;
          std::size_t ic = 0;
          for (std::size_t c = b; c < e; ++c) {
            if (is_pinned[static_cast<std::size_t>(b_col[c])]) continue;
            blk[ia * m + ic] = b_val[a] * b_val[c] * inv_w;
            ++ic;
          }
          ++ia;
        }
        // the block's rows are this constraint's columns -- the facets of one
        // cell -- and each rank inserts the ones it owns. Every rank that
        // assembles the cell computes the same block, so a row inserted twice
        // would be doubled.
        if (!distributed_) {
          check(MatSetValues(P, static_cast<PetscInt>(m), cols.data(), static_cast<PetscInt>(m),
                             cols.data(), blk.data(), ADD_VALUES),
                "P(graph block)");
        } else {
          for (std::size_t ir = 0; ir < m; ++ir) {
            if (!owns(cols[ir])) continue;
            check(MatSetValues(P, 1, cols.data() + ir, static_cast<PetscInt>(m), cols.data(),
                               blk.data() + ir * m, ADD_VALUES),
                  "P(graph block)");
          }
        }
      }
    }
    b_col = std::vector<Index>();
    b_val = std::vector<double>();
    b_begin = std::vector<Index>();

    check(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY), "assembly(P)");
    check(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY), "assembly(P)");

    // P is symmetric positive definite by construction -- a material inner
    // product plus B^T W^-1 B plus positive diagonals -- and saying so lets a
    // Cholesky be taken of its blocks: half the fill of an LU.
    check(MatSetOption(P, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption(symmetric)");
    check(MatSetOption(P, MAT_SPD, PETSC_TRUE), "MatSetOption(spd)");

    check(KSPSetOperators(ksp, M_, P), "KSPSetOperators(A, P)");
    check(PCSetType(pc, PCFIELDSPLIT), "PCSetType(fieldsplit)");
    for (std::size_t b = 0; b < nf; ++b) {
      check(PCFieldSplitSetIS(pc, std::to_string(b).c_str(), sets[b]), "PCFieldSplitSetIS");
    }
    check(PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE), "PCFieldSplitSetType");

    // exact or not, decided on the size of the block being inverted: the fill
    // of a complete factorization is what stops this scaling, not its speed
    const auto n0 = static_cast<int>(norm_.factors[0].size());
    const bool inexact_block =
        opts_.riesz_block_its > 0 || (opts_.riesz_block_its < 0 && n0 > opts_.riesz_exact_limit);
    // When the complex is there, ADS is the default for a big block.
    //
    // The two incidence maps are supplied only when the first factor is one
    // unknown per facet in 3D, which is the space ADS is written for; without
    // them the auxiliary decomposition does not exist and an incomplete
    // factorization is the fallback. On a 96k-dof block the difference is not
    // marginal -- ADS holds ~0.14 us per dof per iteration across a refinement
    // where icc goes 0.17 -> 1.4 and Cholesky 0.11 -> 0.68, because both of
    // those pay for fill and ADS pays for none.
    const bool ads_possible =
        !norm_.discrete_gradient.empty() && !norm_.discrete_curl.empty() &&
        norm_.vertex_coordinates.size() ==
            static_cast<std::size_t>(norm_.discrete_gradient.cols) *
                static_cast<std::size_t>(norm_.space_dim);
    // ADS is written for one unknown per facet: the discrete curl's rows must
    // be the block's rows. The strongly-symmetric stress carries the
    // six-component traction moment vector whole, so the two disagree row for
    // row and the auxiliary decomposition does not exist -- neither directly
    // nor through the facet-constant subspace, whose injection is written for
    // d copies of a scalar layout.
    const bool one_per_facet =
        !norm_.discrete_curl.empty() &&
        static_cast<std::size_t>(n0) == static_cast<std::size_t>(norm_.discrete_curl.rows);
    // Where the crossover is. Against MUMPS, which reorders before it factors,
    // the exact block wins at every size that fits in memory here -- 0.58 s
    // against 1.00 at 119k unknowns, 1.71 against 2.86 at 322k -- because the
    // iteration count is lower (39 against 60) and each application is a pair
    // of triangular solves rather than a multigrid cycle over three auxiliary
    // spaces.
    //
    // So the auxiliary-space route is taken where the factorization is refused
    // for its fill, which is what riesz_exact_limit is for, and not before.
    // Ask for it with riesz_block_pc = "ads" to have it sooner.
    const bool through_subspace = !norm_.lowest_order.empty();
    // Both routes are distributed: ADS on a block whose unknowns are the
    // facets, and the two-level cycle that reaches it through the
    // facet-constant subspace. Each needs the entities laid out on the
    // partition, which is the one thing that must be supplied for it.
    const bool ads_possible_here =
        ads_possible && (!distributed_ || norm_.entity_owner.size() >= 3);
    const bool use_ads = ads_possible_here && (one_per_facet || through_subspace) &&
                         (inexact_block || (!through_subspace && n0 > opts_.riesz_ads_limit));
    std::string b0_pc = !opts_.riesz_block_pc.empty() ? opts_.riesz_block_pc
                        : use_ads                     ? "ads"
                        : inexact_block               ? "icc"
                                                      : "cholesky";
    // A named 'ads' that cannot be built is re-decided rather than obeyed.
    // Without the one-unknown-per-facet layout there is nothing to attach --
    // hypre would be handed a curl whose rows are not the block's -- and
    // throwing mid-setup fails a run the exact block solves. The same command
    // then works on every product, and block_solver reports what ran.
    if (b0_pc == "ads" && !(ads_possible_here && (one_per_facet || through_subspace))) {
      b0_pc = inexact_block ? "icc" : "cholesky";
    }

    // A two-level cycle is not a complete solver, and one application of it is
    // not enough. Where ADS acts on the block itself, a single V-cycle is
    // already a good inverse -- flow converges in 33 iterations with one. Where
    // it acts through the facet-constant subspace, one cycle leaves the
    // non-constant moments to a smoother and the outer count is 195 against
    // Cholesky's 25; solving the block with a short CG under the same cycle
    // brings it to 37 and costs less than the difference. So the cycle asks for
    // the inner Krylov itself rather than waiting for the block to be big
    // enough to trigger it.
    const bool two_level = b0_pc == "ads" && !norm_.lowest_order.empty();
    // One cycle is what the Riesz map asks for.
    //
    // What the theory wants of the first block is spectral equivalence, not an
    // accurate solve: an inner Krylov run to a tight tolerance buys a precision
    // the outer iteration cannot use, and pays for it in every application.
    // Measured on 790k unknowns over four processes, block solved by
    //
    //     one ADS cycle          8.3 s    60 outer iterations
    //     CG(5)  to 1e-1        11.6      40
    //     CG(10) to 1e-2        19.3      35
    //     CG(200) to 1e-4       37.4      36
    //
    // Fewer outer iterations, more time: the count falls and the cost per count
    // rises faster. So ADS is applied once unless the caller asks otherwise,
    // and stays a fixed operator that plain GMRES may use.
    //
    // The subspace route is the exception, measured above: a cycle whose coarse
    // space is a subspace corrects less of the block than one acting on the
    // block itself.
    const bool single_cycle = b0_pc == "ads" && !two_level && opts_.riesz_block_its < 0;
    const bool inner_krylov = (inexact_block && !single_cycle) || two_level;
    // The two-level budget is "to tolerance". Capped at 50, the inner CG
    // under the cycle stopped short on the stress block -- median 56, 100, 168
    // steps are what rtol 1e-2 takes over three refinements of the Lame
    // annulus -- and the outer count then measured the cap: 21, 22, 36 under
    // refinement, and 131 on one rank against 103 on two, the hypre
    // hierarchy leaking into the count. Solved to tolerance the outer count
    // is the Riesz map's, 20-22 flat and rank-independent (21 against 22).
    // Where the block converges inside 50 steps anyway -- hybrid_mesh_l_3,
    // 2.4M unknowns, 53 outer iterations either way -- the budget is moot.
    const int block_its = opts_.riesz_block_its > 0 ? opts_.riesz_block_its
                          : two_level              ? 500
                                                   : 200;
    const double block_rtol = opts_.riesz_block_its > 0 ? opts_.riesz_block_rtol
                              : two_level              ? 1e-2
                                                       : opts_.riesz_block_rtol;
    // an inner Krylov makes the preconditioner a varying operator, which only a
    // flexible outer method may use; applying it under plain gmres is a silent
    // wrong answer, so the promotion happens here rather than in the caller
    if (inner_krylov) check(KSPSetType(ksp, KSPFGMRES), "KSPSetType(fgmres)");
    // Orthogonality, kept. P^-1 A has a cluster at 1 from the constrained
    // rows and its other eigenvalues a few decades away, and classical
    // Gram-Schmidt without refinement loses the Krylov basis to that
    // contrast: the recursive residual stalls while the true one is already
    // down, and the iteration only notices at a restart -- measured on the
    // Dupuit annulus as 39, 66, 67 iterations at restart 30 and 109, 110, 207
    // at restart 100, always a restart plus nine. Re-orthogonalized when the
    // loss is detected, the same ladder is 13, 13, 14: the map's count. The
    // check is cheap next to an application of P; a no-op for a non-GMRES
    // method.
    PetscCallAbort(comm_, KSPGMRESSetCGSRefinementType(ksp, KSP_GMRES_CGS_REFINE_IFNEEDED));

    // The sub-solvers are set through the options database, before setup.
    //
    // PCFieldSplitGetSubKSP requires the PC to be set up, and setting it up is
    // what performs the factorizations -- with whatever sub-solver the split
    // defaults to. Reaching in afterwards to change the type therefore changes
    // nothing that has already been paid for: the default LU of the first
    // factor is taken first, which is the entire cost being avoided. Naming
    // them by prefix leaves the choice in place when the setup happens.
    const std::string p0 = prefix_ + "fieldsplit_0_";
    const std::string p1 = prefix_ + "fieldsplit_1_";
    const auto set = [](const std::string& key, const std::string& value) {
      PetscOptionsSetValue(nullptr, ("-" + key).c_str(), value.c_str());
    };
    // an incomplete factorization is sequential; over several ranks it becomes
    // one per rank, which is what PCBJACOBI is
    if (distributed_ && (b0_pc == "icc" || b0_pc == "ilu")) {
      set(p0 + "sub_pc_type", b0_pc);
      b0_pc = "bjacobi";
    }
    if (inner_krylov) {
      set(p0 + "ksp_type", two_level && opts_.riesz_coarse_its > 0 ? "fcg" : "cg");
      set(p0 + "ksp_max_it", std::to_string(block_its));
      set(p0 + "ksp_rtol", std::to_string(block_rtol));
      set(p0 + "pc_type", b0_pc);
    } else {
      set(p0 + "ksp_type", "preonly");
      set(p0 + "pc_type", b0_pc);
      std::string pkg = !opts_.riesz_block_factorization.empty()
                            ? opts_.riesz_block_factorization
                            : opts_.factorization;
      if (distributed_) pkg = parallel_package(pkg);
      if ((b0_pc == "lu" || b0_pc == "cholesky") && !pkg.empty() && pkg != "petsc") {
        set(p0 + "pc_factor_mat_solver_type", pkg);
      }
    }
    // the L2 factors are diagonal, so Jacobi inverts them exactly and anything
    // heavier is wasted
    set(p1 + "ksp_type", "preonly");
    set(p1 + "pc_type", "jacobi");

    // An auxiliary-space solver on the first factor.
    //
    // ADS preconditions an H(div) operator by splitting it along the de Rham
    // complex -- a field becomes a vector potential in H(curl) plus a part
    // carried by the vertex spaces -- and preconditioning each piece where
    // multigrid actually works. Classical AMG cannot: the near-null space here
    // is the divergence-free fields, not the constants, which is why BoomerAMG
    // on this block needs more iterations the finer the mesh. ADS costs no
    // fill and is linear in the unknowns.
    //
    // Its two maps are the discrete gradient and curl -- the complex's own
    // boundary operators -- and they must be attached before the sub-PC is set
    // up, because setting up is what builds the hierarchy. So the split is
    // brought up with a PC that costs nothing, the maps are attached, and the
    // real type is set; the outer KSPSetUp then does the work once.
    ads_block_ = b0_pc == "ads";
    block_solver_ = inner_krylov ? "cg+" + b0_pc : b0_pc;
    if (b0_pc == "cholesky" || b0_pc == "lu") {
      const std::string pkg = !opts_.riesz_block_factorization.empty()
                                  ? opts_.riesz_block_factorization
                                  : opts_.factorization;
      block_solver_ += "/" + (distributed_ ? parallel_package(pkg) : pkg);
    }
    if (b0_pc == "ads") {
      set(p0 + "pc_type", "none");
      check(PCSetUp(pc), "PCSetUp(fieldsplit)");
      PetscInt n_split = 0;
      KSP* sub = nullptr;
      check(PCFieldSplitGetSubKSP(pc, &n_split, &sub), "PCFieldSplitGetSubKSP");
      PC sub_pc = nullptr;
      // One application, or a short CG under it. A single ADS V-cycle is a
      // fixed linear operator, which plain gmres may use; riesz_block_its > 0
      // asks for an inner Krylov instead, which sharpens the block at the price
      // of making the preconditioner vary -- the outer method was already
      // promoted to fgmres above for exactly that.
      if (!inner_krylov) {
        check(KSPSetType(sub[0], KSPPREONLY), "sub KSPSetType");
      } else {
        check(KSPSetType(sub[0], KSPCG), "sub KSPSetType(cg)");
        check(KSPSetTolerances(sub[0], block_rtol, PETSC_DEFAULT, PETSC_DEFAULT, block_its),
              "sub KSPSetTolerances");
      }
      check(KSPGetPC(sub[0], &sub_pc), "sub KSPGetPC");
      // One unknown per facet, or a lowest-order subspace of one that is not.
      // The first is ADS as hypre offers it; the second is the auxiliary-space
      // argument applied one level up, and is what the AFW stress space needs.
      if (norm_.lowest_order.empty()) {
        attach_ads(sub_pc);
      } else {
        build_lowest_order_cycle(sub_pc);
      }
      // set up here, so the hierarchy is built where factorize() is timed
      // rather than inside the first KSPSolve
      check(PCSetUp(sub_pc), "PCSetUp(ads)");
      PetscFree(sub);
    }

    riesz_.push_back(P);
    for (IS& s : sets) ISDestroy(&s);
  }

  // The entities, laid out like the unknowns. Same sort -- by owner, stable
  // within a rank -- so a rank's faces are contiguous and in the order its
  // block rows are in, which is the one thing hypre cannot be told and must
  // simply be true.
  struct EntityLayout {
    std::vector<PetscInt> new_of, old_of;
    // where each rank's run starts, n_ranks + 1 of them: needed to build a
    // space on top of this one, which is what the coarse space of the
    // two-level cycle is
    std::vector<PetscInt> first;
    PetscInt begin{0}, end{0}, local{0}, total{0};

    bool owns(PetscInt i) const { return i >= begin && i < end; }
    PetscInt count(int rank) const {
      return first[static_cast<std::size_t>(rank) + 1] - first[static_cast<std::size_t>(rank)];
    }
  };

  EntityLayout layout_of(const std::vector<int>& owner, int n_ranks) const {
    EntityLayout out;
    out.total = static_cast<PetscInt>(owner.size());
    std::vector<PetscInt> count(static_cast<std::size_t>(n_ranks) + 1, 0);
    for (const int r : owner) {
      if (r < 0 || r >= n_ranks) {
        throw std::invalid_argument("PetscSolver: an entity is owned by no rank in this run");
      }
      ++count[static_cast<std::size_t>(r) + 1];
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(n_ranks); ++r) count[r + 1] += count[r];
    out.begin = count[static_cast<std::size_t>(rank_)];
    out.end = count[static_cast<std::size_t>(rank_) + 1];
    out.local = out.end - out.begin;
    out.first = count;
    out.new_of.resize(owner.size());
    out.old_of.resize(owner.size());
    std::vector<PetscInt> at_slot(count.begin(), count.end() - 1);
    for (std::size_t e = 0; e < owner.size(); ++e) {
      const auto slot = at_slot[static_cast<std::size_t>(owner[e])]++;
      out.new_of[e] = slot;
      out.old_of[static_cast<std::size_t>(slot)] = static_cast<PetscInt>(e);
    }
    return out;
  }

  // An incidence matrix over two such layouts: the complex's numbers on the
  // way in, the partition's on the way out, and only the rows this rank owns.
  Mat to_mat(const SpaceNorm::Incidence& a, const EntityLayout& rows,
             const EntityLayout& cols) const {
    std::vector<PetscInt> diag(static_cast<std::size_t>(rows.local), 0);
    std::vector<PetscInt> off(static_cast<std::size_t>(rows.local), 0);
    for (std::size_t k = 0; k < a.value.size(); ++k) {
      const PetscInt i = rows.new_of[static_cast<std::size_t>(a.row[k])];
      if (!rows.owns(i)) continue;
      const PetscInt j = cols.new_of[static_cast<std::size_t>(a.col[k])];
      ++(cols.owns(j) ? diag : off)[static_cast<std::size_t>(i - rows.begin)];
    }
    Mat M = nullptr;
    check(MatCreate(comm_, &M), "MatCreate(incidence)");
    check(MatSetType(M, MATMPIAIJ), "MatSetType(incidence)");
    check(MatSetSizes(M, rows.local, cols.local, rows.total, cols.total), "MatSetSizes(incidence)");
    check(MatMPIAIJSetPreallocation(M, 0, diag.data(), 0, off.data()), "preallocate(incidence)");
    for (std::size_t k = 0; k < a.value.size(); ++k) {
      const PetscInt i = rows.new_of[static_cast<std::size_t>(a.row[k])];
      if (!rows.owns(i)) continue;
      const PetscInt j = cols.new_of[static_cast<std::size_t>(a.col[k])];
      check(MatSetValues(M, 1, &i, 1, &j, &a.value[k], INSERT_VALUES), "incidence entry");
    }
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly(incidence)");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly(incidence)");
    return M;
  }

  // how many rows of the first factor's block this rank holds
  PetscInt block_rows_local() const {
    PetscInt n = 0;
    for (const int g : norm_.factors[0]) {
      if (owns(at(g))) ++n;
    }
    return n;
  }

  // A rectangular map whose two spaces are laid out independently -- the
  // interpolation of the cycle, whose rows are the block's and whose columns
  // are the coarse space's. Rows already carry the solver's numbering, so each
  // rank inserts the ones it holds.
  Mat to_mat_rectangular(const SpaceNorm::Incidence& a, PetscInt local_rows,
                         PetscInt local_cols) const {
    PetscInt row_begin = 0;
    MPI_Scan(&local_rows, &row_begin, 1, MPIU_INT, MPI_SUM, comm_);
    row_begin -= local_rows;
    PetscInt col_begin = 0;
    MPI_Scan(&local_cols, &col_begin, 1, MPIU_INT, MPI_SUM, comm_);
    col_begin -= local_cols;
    std::vector<PetscInt> diag(static_cast<std::size_t>(local_rows), 0);
    std::vector<PetscInt> off(static_cast<std::size_t>(local_rows), 0);
    const auto ours = [&](PetscInt i) { return i >= row_begin && i < row_begin + local_rows; };
    for (std::size_t k = 0; k < a.value.size(); ++k) {
      const PetscInt i = a.row[k];
      if (!ours(i)) continue;
      const PetscInt j = a.col[k];
      const bool in_diagonal = j >= col_begin && j < col_begin + local_cols;
      ++(in_diagonal ? diag : off)[static_cast<std::size_t>(i - row_begin)];
    }
    Mat M = nullptr;
    check(MatCreate(comm_, &M), "MatCreate(interpolation)");
    check(MatSetType(M, MATMPIAIJ), "MatSetType(interpolation)");
    check(MatSetSizes(M, local_rows, local_cols, a.rows, a.cols), "MatSetSizes(interpolation)");
    check(MatMPIAIJSetPreallocation(M, 0, diag.data(), 0, off.data()),
          "preallocate(interpolation)");
    for (std::size_t k = 0; k < a.value.size(); ++k) {
      const PetscInt i = a.row[k];
      if (!ours(i)) continue;
      const PetscInt j = a.col[k];
      check(MatSetValues(M, 1, &i, 1, &j, &a.value[k], INSERT_VALUES), "interpolation entry");
    }
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly(interpolation)");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly(interpolation)");
    return M;
  }

  // ADS on a block whose unknowns are the facets, with the two maps of the
  // complex and the vertex coordinates the auxiliary spaces are built from.
  // `local_rows` is how many rows of the operator this rank holds -- the first
  // factor's owned unknowns when ADS acts on the block itself, one copy's
  // worth of faces when it acts on a component of a coarse space. It is
  // checked rather than assumed, because hypre cannot be told the numbering
  // and a mismatch would silently precondition a permuted operator.
  void attach_ads(PC pc, PetscInt local_rows = -1) {
    check(PCSetType(pc, PCHYPRE), "PCSetType(hypre)");
    check(PCHYPRESetType(pc, "ads"), "PCHYPRESetType(ads)");
    if (norm_.discrete_gradient.empty() || norm_.discrete_curl.empty()) {
      throw std::invalid_argument(
          "PetscSolver: 'ads' needs the discrete gradient and curl of the complex");
    }
    const std::size_t n_vertex = static_cast<std::size_t>(norm_.discrete_gradient.cols);
    if (norm_.vertex_coordinates.size() != n_vertex * static_cast<std::size_t>(norm_.space_dim)) {
      throw std::invalid_argument("PetscSolver: 'ads' needs one coordinate per vertex");
    }
    Mat G = nullptr;
    Mat C = nullptr;
    PetscInt local_vertices = static_cast<PetscInt>(n_vertex);
    std::vector<PetscReal> xyz;
    if (!distributed_) {
      G = to_mat(norm_.discrete_gradient);
      C = to_mat(norm_.discrete_curl);
      xyz.assign(norm_.vertex_coordinates.begin(), norm_.vertex_coordinates.end());
    } else {
      // The three spaces, laid out on the partition.
      //
      // hypre is handed the maps between them, so all three have to agree with
      // each other and the faces have to agree with the block: C's row i is
      // the block's row i, or the solver is preconditioning a permutation of
      // its own operator. Nothing here communicates -- every rank sorts the
      // same ownership array the same way.
      PetscMPIInt size = 1;
      MPI_Comm_size(comm_, &size);
      if (norm_.entity_owner.size() < 3) {
        throw std::invalid_argument(
            "PetscSolver: 'ads' on several processes needs the owner of every vertex, edge and "
            "face");
      }
      const EntityLayout vertices = layout_of(norm_.entity_owner[0], size);
      const EntityLayout edges = layout_of(norm_.entity_owner[1], size);
      const EntityLayout faces = layout_of(norm_.entity_owner[2], size);
      PetscInt owned_dofs = local_rows;
      if (owned_dofs < 0) {
        owned_dofs = 0;
        for (const int g : norm_.factors[0]) {
          if (owns(at(g))) ++owned_dofs;
        }
      }
      if (owned_dofs != faces.local) {
        throw std::invalid_argument(
            "PetscSolver: 'ads' needs the faces laid out as the block is; this rank owns " +
            std::to_string(owned_dofs) + " unknowns of the first factor and " +
            std::to_string(faces.local) + " faces");
      }
      G = to_mat(norm_.discrete_gradient, edges, vertices);
      C = to_mat(norm_.discrete_curl, faces, edges);
      local_vertices = vertices.local;
      xyz.resize(static_cast<std::size_t>(vertices.local) *
                 static_cast<std::size_t>(norm_.space_dim));
      for (PetscInt i = 0; i < vertices.local; ++i) {
        const auto old = static_cast<std::size_t>(vertices.old_of[static_cast<std::size_t>(
            vertices.begin + i)]);
        for (int c = 0; c < norm_.space_dim; ++c) {
          xyz[static_cast<std::size_t>(i) * static_cast<std::size_t>(norm_.space_dim) +
              static_cast<std::size_t>(c)] =
              norm_.vertex_coordinates[old * static_cast<std::size_t>(norm_.space_dim) +
                                       static_cast<std::size_t>(c)];
        }
      }
    }
    check(PCHYPRESetDiscreteGradient(pc, G), "PCHYPRESetDiscreteGradient");
    check(PCHYPRESetDiscreteCurl(pc, C), "PCHYPRESetDiscreteCurl");
    // the metric, and the only metric ADS is told: x, y, z at the vertices,
    // from which it interpolates the auxiliary vector spaces
    check(PCSetCoordinates(pc, norm_.space_dim, local_vertices, xyz.data()),
          "PCSetCoordinates(ads)");

    // ADS's own knobs are hypre's, and hypre's are reachable only through the
    // options database: the cycle type, the relaxation, the AMG parameters of
    // each auxiliary space. A PC built in code and set up in code never reads
    // that database, so a -pc_hypre_ads_* given on the command line is accepted
    // and silently ignored unless the database is read here.
    //
    // Reading it is not free, which is why it is conditional. PCSetFromOptions
    // writes PETSc's own defaults for every hypre parameter it knows, and
    // those are not hypre's: on the 22k-cell polyhedral mesh they cost 1.6x
    // the time per application (3.7 s became 6.0 s at the same tolerance). So
    // the database is read only when the caller has actually set one of these,
    // and left alone otherwise.
    //
    // What the knobs are worth, measured on that mesh: the default cycle takes
    // 44 iterations and 3.7 s, and `-pc_hypre_ads_cycle_type 11` -- which
    // solves the vector Poisson problems more thoroughly -- takes 18 and 2.6.
    // It is not the default because it breaks GMRES down on eight processes
    // here.
    //
    // The cycle also carries the coefficient jump. On the 3D cartesian patch
    // with K a checkerboard in {1, 1/c} and n = 16, the count over
    // c = 1, 1e2, 1e4, 1e6 is 16, 14, 26, 29 by default and 12, 12, 12, 16 at
    // cycle 7: flat in h either way, and the growth in c halved. Cycles 3 and
    // 13 measure the same. Serial only -- the parallel breakdown noted above
    // is why none of them is the default.
    //
    // What is NOT reachable is the auxiliary hierarchy's own strength
    // threshold. PETSc registers -pc_hypre_ads_amg_theta,
    // -pc_hypre_ads_ams_theta and -pc_hypre_ads_ams_cycle_type but never
    // queries them: each is accepted, reported by -options_left as unused, and
    // changes no count. They are left out of the list below so that setting
    // one does not pay for PCSetFromOptions and buy nothing. K reaches the
    // auxiliary problems regardless -- ADS forms them as Galerkin products
    // Pi^T A Pi and C^T A C, and A carries K -- but theta stays at hypre's
    // default, which is the one parameter a jumping K would ask to change.
    {
      static const char* const knobs[] = {
          "-pc_hypre_ads_cycle_type",   "-pc_hypre_ads_relax_type",
          "-pc_hypre_ads_relax_times",  "-pc_hypre_ads_relax_weight",
          "-pc_hypre_ads_omega",        "-pc_hypre_ads_print_level"};
      const char* own = nullptr;
      check(PCGetOptionsPrefix(pc, &own), "PCGetOptionsPrefix(ads)");
      const std::string mine = own != nullptr ? own : "";
      bool asked = false;
      for (const char* knob : knobs) {
        PetscBool has = PETSC_FALSE;
        PetscOptionsHasName(nullptr, mine.c_str(), knob, &has);
        asked = asked || has == PETSC_TRUE;
      }
      if (asked) {
        // the entry that brought the split up cheaply says "none"; a PC
        // reading that would undo itself, so the type is written down first
        PetscOptionsSetValue(nullptr, ("-" + mine + "pc_type").c_str(), "hypre");
        PetscOptionsSetValue(nullptr, ("-" + mine + "pc_hypre_type").c_str(), "ads");
        check(PCSetFromOptions(pc), "PCSetFromOptions(ads)");
      }
    }
    riesz_.push_back(G);
    riesz_.push_back(C);
  }

  // A two-level cycle whose coarse space is the facet constants.
  //
  // The AFW stress block is not an ADS problem: a facet carries d traction
  // components measured against the d functions of its P_1 basis, so d^2
  // unknowns sit on it and hypre would not know what a facet is. But the
  // auxiliary-space argument is about a subspace where the operator is
  // spectrally equivalent to something a solver exists for, and here that
  // subspace is written down rather than interpolated: the facet-constant
  // moments are a subset of the degrees of freedom, so the injection is a
  // matrix of ones.
  //
  //   smoother   Chebyshev/Jacobi on the whole block -- the higher moments are
  //              local to a facet, and what is local is what a smoother is for
  //   coarse     the constants, one H(div) problem per component: the coupling
  //              between components is the material's, bounded and dropped by
  //              an additive split, and each diagonal block is what ADS takes
  //
  // The coarse operator is Galerkin, P^T A P, so nothing about the physics is
  // restated at the coarse level -- it is the same operator seen on the
  // subspace.
  // d copies of a one-unknown-per-facet space, split and handed to ADS, one
  // component at a time and in sequence.
  //
  // The material couples the copies -- through the trace in three fields, not
  // at all in four where M is diagonal -- and an additive split drops that
  // coupling, which is a preconditioner's privilege. Each diagonal block is
  // then the scalar H(div) problem ADS is written for.
  //
  // What the split drops -- the coupling between components: the rotation, and
  // the trace the total pressure takes -- is real. Applied additively the count
  // grows with the mesh (388 against 199 at 68k unknowns); multiplicative
  // recovers most of it for the same work, since each component then sees the
  // residual the ones before it left.
  //
  // A point-block smoother over a facet's d tractions was the other candidate
  // and is not usable: with a lumped compliance those blocks are near-singular,
  // and inverting them diverges outright by 8^3 cells.
  void split_by_component(PC pc, const SpaceNorm::Incidence& local, int nc) {
    const PetscInt per = local.cols / nc;
    // the block rows this rank holds, as one range: MPI_Scan is collective, so
    // it is taken once here rather than per row
    PetscInt block_begin = 0, block_end = local.rows;
    if (distributed_) {
      const PetscInt mine_rows = block_rows_local();
      MPI_Scan(&mine_rows, &block_end, 1, MPIU_INT, MPI_SUM, comm_);
      block_begin = block_end - mine_rows;
    }
    std::vector<std::vector<PetscInt>> rows(static_cast<std::size_t>(nc));
    for (std::size_t k = 0; k < local.value.size(); ++k) {
      const int c = local.col[k] / per;
      rows[static_cast<std::size_t>(c)].push_back(local.row[k]);
    }
    check(PCSetType(pc, PCFIELDSPLIT), "PCSetType(component split)");
    const char* own = nullptr;
    check(PCGetOptionsPrefix(pc, &own), "PCGetOptionsPrefix(component split)");
    const std::string mine = own != nullptr ? own : "";
    std::vector<IS> parts(static_cast<std::size_t>(nc), nullptr);
    for (int c = 0; c < nc; ++c) {
      auto& r = rows[static_cast<std::size_t>(c)];
      std::sort(r.begin(), r.end());
      if (distributed_) {
        // a rank names the rows of this component that it holds; the union
        // over the ranks is the component
        std::vector<PetscInt> ours;
        for (const PetscInt i : r) {
          if (i >= block_begin && i < block_end) ours.push_back(i);
        }
        r = std::move(ours);
      }
      check(ISCreateGeneral(comm_, static_cast<PetscInt>(r.size()), r.data(), PETSC_COPY_VALUES,
                            &parts[static_cast<std::size_t>(c)]),
            "ISCreateGeneral(component)");
      check(PCFieldSplitSetIS(pc, std::to_string(c).c_str(), parts[static_cast<std::size_t>(c)]),
            "PCFieldSplitSetIS(component)");
      const std::string key = mine + "fieldsplit_" + std::to_string(c) + "_";
      PetscOptionsSetValue(nullptr, ("-" + key + "ksp_type").c_str(), "preonly");
      PetscOptionsSetValue(nullptr, ("-" + key + "pc_type").c_str(), "none");
    }
    check(PCFieldSplitSetType(pc, PC_COMPOSITE_MULTIPLICATIVE), "PCFieldSplitSetType(component)");
    check(PCSetUp(pc), "PCSetUp(component split)");
    PetscInt n_part = 0;
    KSP* sub = nullptr;
    check(PCFieldSplitGetSubKSP(pc, &n_part, &sub), "PCFieldSplitGetSubKSP(component)");
    for (PetscInt c = 0; c < n_part; ++c) {
      PC cpc = nullptr;
      check(KSPSetType(sub[c], KSPPREONLY), "component KSPSetType");
      check(KSPGetPC(sub[c], &cpc), "component KSPGetPC");
      PetscInt local_rows = -1;
      if (distributed_) {
        Mat a = nullptr, b = nullptr;
        check(KSPGetOperators(sub[c], &a, &b), "component KSPGetOperators");
        check(MatGetLocalSize(b, &local_rows, nullptr), "MatGetLocalSize(component)");
      }
      attach_ads(cpc, local_rows);
      check(PCSetUp(cpc), "PCSetUp(component ads)");
    }
    PetscFree(sub);
    for (IS& s : parts) ISDestroy(&s);
  }

  void build_lowest_order_cycle(PC pc) {
    const auto& inj = norm_.lowest_order;
    const int nc = norm_.lowest_order_components;
    if (nc < 1 || inj.cols % nc != 0) {
      throw std::invalid_argument("PetscSolver: lowest-order injection is not component-blocked");
    }
    // The block's row of each unknown. A fieldsplit gives its sub-matrix the
    // rows of its index set in that set's order, which is ascending in the
    // solver's numbering -- so on several processes the block's rows are
    // grouped by rank exactly as the unknowns are, and the injection's rows
    // must be numbered the same way.
    const std::vector<int>& first = norm_.factors[0];
    std::vector<PetscInt> order(first.size());
    std::iota(order.begin(), order.end(), PetscInt{0});
    std::sort(order.begin(), order.end(), [&](PetscInt a, PetscInt b) {
      return at(first[static_cast<std::size_t>(a)]) < at(first[static_cast<std::size_t>(b)]);
    });
    std::vector<int> position(static_cast<std::size_t>(n_), -1);
    for (std::size_t i = 0; i < order.size(); ++i) {
      position[static_cast<std::size_t>(first[static_cast<std::size_t>(order[i])])] =
          static_cast<int>(i);
    }
    SpaceNorm::Incidence local;
    local.rows = static_cast<int>(first.size());
    local.cols = inj.cols;
    local.row.reserve(inj.value.size());
    local.col.reserve(inj.value.size());
    local.value.reserve(inj.value.size());
    for (std::size_t k = 0; k < inj.value.size(); ++k) {
      const int r = position[static_cast<std::size_t>(inj.row[k])];
      if (r < 0) throw std::invalid_argument("PetscSolver: injection row is not in the block");
      local.row.push_back(r);
      local.col.push_back(inj.col[k]);
      local.value.push_back(inj.value[k]);
    }
    position = std::vector<int>();
    order = std::vector<PetscInt>();

    // The coarse space is the whole block when a facet carries one moment per
    // component -- diagonal_afw, and derham_rt's layout generally. The
    // injection is then a permutation, and a two-level cycle over it is a cycle
    // whose coarse problem is its fine one: every application pays for a
    // smoother, a Galerkin product and a coarse solve of the same size, and the
    // inner CG pays for it fifty times over. On a 22k-cell mesh that does not
    // converge in any useful time.
    //
    // What the block actually is, in that case, is d copies of a
    // one-unknown-per-facet space -- exactly what ADS takes -- so it is split
    // by component and handed over directly, with no cycle at all.
    if (local.cols == local.rows) {
      split_by_component(pc, local, nc);
      return;
    }

    // The coarse space, laid out on the partition.
    //
    // Serially it is copy-major: every face of copy 0, then copy 1. That
    // ordering cannot survive distribution -- it would give the first ranks
    // whole copies and the last ranks none -- so distributed it becomes
    // (rank, copy, face): each process holds its own faces of every copy, and
    // a copy is still a contiguous run within a process, which is what the
    // coarse split needs and what makes each component's rows the faces this
    // process owns, in the order ADS is given them.
    EntityLayout faces;
    std::vector<PetscInt> coarse_first;
    PetscInt coarse_local = local.cols;
    if (distributed_) {
      PetscMPIInt size = 1;
      MPI_Comm_size(comm_, &size);
      if (norm_.entity_owner.size() < 3) {
        throw std::invalid_argument(
            "PetscSolver: the two-level cycle needs the owner of every vertex, edge and face");
      }
      faces = layout_of(norm_.entity_owner[2], size);
      coarse_first.assign(static_cast<std::size_t>(size) + 1, 0);
      for (int r = 0; r < size; ++r) {
        coarse_first[static_cast<std::size_t>(r) + 1] =
            coarse_first[static_cast<std::size_t>(r)] + nc * faces.count(r);
      }
      coarse_local = nc * faces.local;
      // the injection's columns, renumbered into that layout
      const PetscInt per_copy = inj.cols / nc;
      for (std::size_t k = 0; k < local.col.size(); ++k) {
        const PetscInt column = local.col[static_cast<std::size_t>(k)];
        const PetscInt copy = column / per_copy;
        const PetscInt face = column % per_copy;
        const int owner = norm_.entity_owner[2][static_cast<std::size_t>(face)];
        const PetscInt within = faces.new_of[static_cast<std::size_t>(face)] -
                                faces.first[static_cast<std::size_t>(owner)];
        local.col[static_cast<std::size_t>(k)] =
            coarse_first[static_cast<std::size_t>(owner)] + copy * faces.count(owner) + within;
      }
    }
    Mat interpolation = distributed_
                            ? to_mat_rectangular(local, block_rows_local(), coarse_local)
                            : to_mat(local);

    // the facet's unknowns are contiguous, so telling the block how big a
    // facet is costs nothing and is what lets the smoother invert one exactly
    const PetscInt per_facet =
        static_cast<PetscInt>(local.rows) / (static_cast<PetscInt>(inj.cols) / nc);
    {
      Mat block_a = nullptr, block_p = nullptr;
      check(PCGetOperators(pc, &block_a, &block_p), "PCGetOperators(block)");
      if (block_p != nullptr && per_facet > 1) {
        check(MatSetBlockSize(block_p, per_facet), "MatSetBlockSize(block)");
      }
    }

    check(PCSetType(pc, PCMG), "PCSetType(mg)");
    check(PCMGSetLevels(pc, 2, nullptr), "PCMGSetLevels");
    check(PCMGSetType(pc, PC_MG_MULTIPLICATIVE), "PCMGSetType");
    check(PCMGSetCycleType(pc, PC_MG_CYCLE_V), "PCMGSetCycleType");
    check(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH), "PCMGSetGalerkin");
    check(PCMGSetInterpolation(pc, 1, interpolation), "PCMGSetInterpolation");
    riesz_.push_back(interpolation);

    // THE SMOOTHER CARRIES THE DIV-FREE PART, so it is a sweep and not a
    // facet-local inverse.
    //
    // What the coarse space does not carry is the non-constant moments -- and
    // those are exactly the DIVERGENCE-FREE directions, only the constant
    // moment reaching div. That is the near-nullspace Kolev and Vassilevski
    // (SISC 34-6, A3079) say must be addressed explicitly and "cannot be
    // handled by simple relaxation on the fine grid", and for which they use a
    // convergent Gauss-Seidel smoother rather than a point method.
    //
    // Inverting each facet's block exactly looks right -- the non-constant
    // moments do live on one facet -- and is not: it splits facet from facet,
    // and a div-free field is global. Measured on the AFW block, mean inner CG
    // steps per cycle application at 3^3, 6^3, 8^3 cells, and the solve time
    // at 8^3:
    //
    //     Chebyshev + point-block Jacobi   16.8  18.1  18.5   3.67 s
    //     Richardson + symmetric SOR       12.9  14.1  15.4   3.11 s
    //     Chebyshev + symmetric SOR        11.0  12.5  13.7   2.94 s
    //
    // The sweep couples neighbouring facets, which is what a div-free field
    // needs; Chebyshev on top of it is free. The one-copy flow block does not
    // move (3.1) -- there the coarse space already does the work.
    KSP smoother = nullptr;
    check(PCMGGetSmoother(pc, 1, &smoother), "PCMGGetSmoother");
    check(KSPSetType(smoother, KSPCHEBYSHEV), "smoother KSPSetType");
    check(KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2),
          "smoother KSPSetTolerances");
    PC smooth_pc = nullptr;
    check(KSPGetPC(smoother, &smooth_pc), "smoother KSPGetPC");
    check(PCSetType(smooth_pc, PCSOR), "smoother PCSetType");
    check(PCSORSetSymmetric(smooth_pc, SOR_LOCAL_SYMMETRIC_SWEEP), "smoother PCSORSetSymmetric");

    // The coarse solve is a split by component, and each component is only
    // reachable after the coarse operator exists -- which is what setting the
    // cycle up computes. So it is brought up with a PC that costs nothing, and
    // the real ones are attached to the sub-solves afterwards.
    KSP coarse = nullptr;
    check(PCMGGetCoarseSolve(pc, &coarse), "PCMGGetCoarseSolve");
    if (opts_.riesz_coarse_its > 0) {
      check(KSPSetType(coarse, KSPCG), "coarse KSPSetType(cg)");
      check(KSPSetTolerances(coarse, opts_.riesz_coarse_rtol, PETSC_DEFAULT, PETSC_DEFAULT,
                             opts_.riesz_coarse_its),
            "coarse KSPSetTolerances");
      check(KSPSetNormType(coarse, KSP_NORM_PRECONDITIONED), "coarse KSPSetNormType");
    } else {
      check(KSPSetType(coarse, KSPPREONLY), "coarse KSPSetType");
    }
    PC coarse_pc = nullptr;
    check(KSPGetPC(coarse, &coarse_pc), "coarse KSPGetPC");
    // One copy needs no split. A flux has a single H(div) field, so its coarse
    // space is already the scalar problem ADS takes; wrapping it in a
    // one-field fieldsplit hands hypre a submatrix it builds its AMG hierarchy
    // from differently, and that segfaults inside BoomerAMG rather than
    // failing. The split exists for the d copies of a stress.
    if (nc == 1) {
      check(PCSetUp(pc), "PCSetUp(mg)");
      attach_ads(coarse_pc);
      check(PCSetUp(coarse_pc), "PCSetUp(coarse ads)");
      return;
    }
    check(PCSetType(coarse_pc, PCFIELDSPLIT), "coarse PCSetType(fieldsplit)");
    // one component of the coarse space: every face, serially; this rank's
    // faces of that copy, distributed -- contiguous either way
    const PetscInt per = distributed_ ? faces.local : inj.cols / nc;
    const PetscInt coarse_begin =
        distributed_ ? coarse_first[static_cast<std::size_t>(rank_)] : 0;
    std::vector<IS> parts(static_cast<std::size_t>(nc), nullptr);
    const char* cprefix = nullptr;
    check(PCGetOptionsPrefix(coarse_pc, &cprefix), "PCGetOptionsPrefix(coarse)");
    const std::string cp = cprefix != nullptr ? cprefix : "";
    for (int c = 0; c < nc; ++c) {
      check(ISCreateStride(comm_, per, coarse_begin + c * per, 1,
                           &parts[static_cast<std::size_t>(c)]),
            "ISCreateStride(component)");
      check(PCFieldSplitSetIS(coarse_pc, std::to_string(c).c_str(),
                              parts[static_cast<std::size_t>(c)]),
            "PCFieldSplitSetIS(component)");
      const std::string key = cp + "fieldsplit_" + std::to_string(c) + "_";
      PetscOptionsSetValue(nullptr, ("-" + key + "ksp_type").c_str(), "preonly");
      PetscOptionsSetValue(nullptr, ("-" + key + "pc_type").c_str(), "none");
    }
    // SYMMETRIC-MULTIPLICATIVE, and the symmetry is not decoration: this cycle
    // is applied inside a CG, so the composition has to stay SPD. Plain
    // multiplicative is faster per sweep and wrecks it -- 70, 100, 148 inner
    // steps over three refinements against 17, 18, 18, growing with h, which
    // is CG being handed a non-symmetric preconditioner.
    //
    // What the split composes is the d copies of the coarse H(div) problem,
    // and what an ADDITIVE composition drops is the coupling between them --
    // which is the TRACE, since the compliance couples the copies through it.
    // That coupling is negligible on a unit box, which is why additive looked
    // adequate, and it is not on a mesh written in metres: with lambda > 0 the
    // hydrostatic direction is where the compliance goes singular, and the
    // graph term's D^2 makes it dominate as the domain grows. Measured on 5^3
    // hexahedra of a box of side L, ads-cg outer iterations:
    //
    //     L            1     10    100    300
    //     additive    34     28     51     80
    //     symmetric   34     26     36     41
    //
    // At lambda -> 0 there is no trace coupling and both are flat, which is
    // what says the trace is the mechanism. A 6^3 box of 1000 x 1000 x 500 m
    // went from not converging at all to 29 iterations, against 22 for the
    // exact-block map.
    check(PCFieldSplitSetType(coarse_pc, PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE),
          "coarse PCFieldSplitSetType");
    check(PCSetUp(pc), "PCSetUp(mg)");
    check(PCSetUp(coarse_pc), "PCSetUp(coarse fieldsplit)");
    PetscInt n_part = 0;
    KSP* csub = nullptr;
    check(PCFieldSplitGetSubKSP(coarse_pc, &n_part, &csub), "PCFieldSplitGetSubKSP(coarse)");
    for (PetscInt c = 0; c < n_part; ++c) {
      PC cpc = nullptr;
      check(KSPSetType(csub[c], KSPPREONLY), "coarse component KSPSetType");
      check(KSPGetPC(csub[c], &cpc), "coarse component KSPGetPC");
      // the component's rows are this rank's faces, which is what ADS is told
      attach_ads(cpc, distributed_ ? faces.local : -1);
      check(PCSetUp(cpc), "PCSetUp(coarse ads)");
    }
    PetscFree(csub);
    for (IS& s : parts) ISDestroy(&s);
  }

  // An incidence matrix of the complex as a PETSc operator.
  static Mat to_mat(const SpaceNorm::Incidence& a) {
    std::vector<PetscInt> per_row(static_cast<std::size_t>(a.rows), 0);
    for (const int r : a.row) ++per_row[static_cast<std::size_t>(r)];
    Mat M = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &M), "MatCreate(incidence)");
    check(MatSetType(M, MATSEQAIJ), "MatSetType(incidence)");
    check(MatSetSizes(M, a.rows, a.cols, a.rows, a.cols), "MatSetSizes(incidence)");
    check(MatSeqAIJSetPreallocation(M, 0, per_row.data()), "preallocate(incidence)");
    for (std::size_t k = 0; k < a.value.size(); ++k) {
      const PetscInt i = a.row[k], j = a.col[k];
      check(MatSetValues(M, 1, &i, 1, &j, &a.value[k], INSERT_VALUES), "incidence entry");
    }
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly(incidence)");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly(incidence)");
    return M;
  }

  // Which rows this process owns.
  //
  // One process and the solver is what it was: PETSC_COMM_SELF, sequential
  // matrices, every row local. Several and the algebra is distributed across
  // MPI_COMM_WORLD, in the layout PETSc itself chooses -- a contiguous run per
  // rank, which is what its Mat and Vec require and what every parallel
  // preconditioner assumes.
  //
  // Assembly is still replicated at this stage: every rank builds the whole
  // triplet list and inserts only the rows it owns, so a wrong answer here can
  // only come from the layout.
  void claim_rows() {
    PetscMPIInt size = 1, rank = 0;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    distributed_ = size > 1;
    comm_ = distributed_ ? PETSC_COMM_WORLD : PETSC_COMM_SELF;
    rank_ = rank;
    if (!distributed_) {
      n_local_ = n_;
      own_begin_ = 0;
      own_end_ = n_;
      return;
    }
    if (owners_.size() == static_cast<std::size_t>(n_)) {
      renumber(size);
      return;
    }
    PetscInt local = PETSC_DECIDE;
    check(PetscSplitOwnership(comm_, &local, &n_), "PetscSplitOwnership");
    PetscInt scan = 0;
    MPI_Scan(&local, &scan, 1, MPIU_INT, MPI_SUM, comm_);
    n_local_ = local;
    own_begin_ = scan - local;
    own_end_ = scan;
  }

  // The renumbering is a sort, and nothing more: the unknowns of rank 0 first,
  // in their original order, then rank 1's. Every rank computes the same one
  // from the same ownership array -- no communication, and no rank's answer
  // can disagree about where an unknown lives.
  //
  // Stable within a rank, so the locality the caller's numbering already had
  // survives inside each block; what changes is only that a rank's unknowns
  // become adjacent, which is what makes its rows a subdomain rather than a
  // stride through the mesh.
  void renumber(int n_ranks) {
    const auto n = static_cast<std::size_t>(n_);
    std::vector<PetscInt> count(static_cast<std::size_t>(n_ranks) + 1, 0);
    for (const int owner : owners_) {
      if (owner < 0 || owner >= n_ranks) {
        throw std::invalid_argument("PetscSolver: an unknown is owned by no rank in this run");
      }
      ++count[static_cast<std::size_t>(owner) + 1];
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(n_ranks); ++r) count[r + 1] += count[r];
    own_begin_ = count[static_cast<std::size_t>(rank_)];
    own_end_ = count[static_cast<std::size_t>(rank_) + 1];
    n_local_ = own_end_ - own_begin_;

    new_of_.resize(n);
    old_of_.resize(n);
    std::vector<PetscInt> at(count.begin(), count.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      const auto slot = at[static_cast<std::size_t>(owners_[i])]++;
      new_of_[i] = slot;
      old_of_[static_cast<std::size_t>(slot)] = static_cast<PetscInt>(i);
    }
  }

  // the caller's numbering to the solver's, and the identity when there is no
  // renumbering to do
  PetscInt at(Index i) const {
    return new_of_.empty() ? static_cast<PetscInt>(i) : new_of_[static_cast<std::size_t>(i)];
  }

  bool owns(PetscInt i) const { return i >= own_begin_ && i < own_end_; }

  // A phase takes as long as its slowest process, and a report from one of them
  // is a sample rather than a duration -- the ranks of a partitioned solve
  // differ by whatever their subdomains differ by. So what is reported is the
  // maximum, which is also what the wall clock outside measures.
  double slowest(double seconds) const {
    if (!distributed_) return seconds;
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return seconds;
  }

  // the two halves of an MPIAIJ row, summed over the ranks
  void measure_locality() {
    local_entries_ = off_rank_entries_ = 0.0;
    if (!distributed_ || M_ == nullptr) return;
    Mat diag = nullptr, off = nullptr;
    const PetscInt* colmap = nullptr;
    if (MatMPIAIJGetSeqAIJ(M_, &diag, &off, &colmap) != 0) return;
    MatInfo info;
    if (diag != nullptr && MatGetInfo(diag, MAT_LOCAL, &info) == 0) local_entries_ = info.nz_used;
    if (off != nullptr && MatGetInfo(off, MAT_LOCAL, &info) == 0) off_rank_entries_ = info.nz_used;
    double both[2] = {local_entries_, off_rank_entries_};
    MPI_Allreduce(MPI_IN_PLACE, both, 2, MPI_DOUBLE, MPI_SUM, comm_);
    local_entries_ = both[0];
    off_rank_entries_ = both[1];
  }

  // The nonzero counts PETSc preallocates from, in two steps because they come
  // from two places: the triplets, and the terms P adds that were never
  // triplets. Both are counted over all rows in the solver's numbering, and
  // only then reduced to the rows this rank owns.
  void count_into(std::vector<PetscInt>& diag, std::vector<PetscInt>& off,
                  const std::vector<Index>& row, const std::vector<Index>& col) const {
    if (diag.empty()) diag.assign(static_cast<std::size_t>(n_), 0);
    if (off.empty()) off.assign(static_cast<std::size_t>(n_), 0);
    for (std::size_t k = 0; k < row.size(); ++k) {
      const PetscInt i = at(row[k]);
      // a rank that does not own the row cannot tell that row's diagonal block
      // from its off-diagonal one -- that is the owner's ownership range -- so
      // what it contributes is counted as off-diagonal. Over-allocating the
      // off-diagonal block costs memory, never correctness.
      const bool in_diagonal = !distributed_ || (owns(i) && owns(at(col[k])));
      ++(in_diagonal ? diag : off)[static_cast<std::size_t>(i)];
    }
  }

  // and reduced to the rows this rank owns, which are the only ones it writes
  void finish_counts(std::vector<PetscInt>& diag, std::vector<PetscInt>& off) const {
    if (!distributed_) {
      for (PetscInt& c : diag) c = std::min(c, n_);
      off.clear();
      return;
    }
    std::vector<PetscInt> d(static_cast<std::size_t>(n_local_)), o(static_cast<std::size_t>(n_local_));
    for (PetscInt i = own_begin_; i < own_end_; ++i) {
      const auto k = static_cast<std::size_t>(i - own_begin_);
      d[k] = std::min(diag[static_cast<std::size_t>(i)], n_local_);
      o[k] = std::min(off[static_cast<std::size_t>(i)], n_ - n_local_);
    }
    diag = std::move(d);
    off = std::move(o);
  }

  void count_rows(const std::vector<Index>& row, const std::vector<Index>& col,
                  std::vector<PetscInt>& diag, std::vector<PetscInt>& off) const {
    diag.clear();
    off.clear();
    count_into(diag, off, row, col);
    finish_counts(diag, off);
  }

  // Create and preallocate, sequential or distributed, from those counts.
  Mat new_matrix(const std::vector<PetscInt>& diag, const std::vector<PetscInt>& off,
                 const char* what) const {
    Mat M = nullptr;
    check(MatCreate(comm_, &M), what);
    check(MatSetType(M, distributed_ ? MATMPIAIJ : MATSEQAIJ), what);
    check(MatSetSizes(M, distributed_ ? n_local_ : n_, distributed_ ? n_local_ : n_, n_, n_), what);
    if (distributed_) {
      check(MatMPIAIJSetPreallocation(M, 0, diag.data(), 0, off.data()), what);
    } else {
      check(MatSeqAIJSetPreallocation(M, 0, diag.data()), what);
    }
    check(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), what);
    // an explicitly stored zero is structure, not noise, and must survive
    check(MatSetOption(M, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE), what);
    return M;
  }

  // A matrix from triplets, one call per row rather than one per entry.
  //
  // Handing PETSc one triplet at a time costs a search of the row for every
  // entry, and an assembly of tens of thousands of polyhedra emits of order
  // 10^8 of them. Grouping by row first -- a counting sort, one pass to count
  // and one to place -- turns that into one call per row, and PETSc sums the
  // duplicates inside each call as before. The scratch is scoped so it is
  // returned before anything else is allocated.
  Mat assemble(const std::vector<Index>& row, const std::vector<Index>& col,
               const std::vector<double>& val, PetscInt n) const {
    // Triplets, not nonzeros: a row's triplet count is an upper bound on its
    // nonzero count, and on a small mesh it can exceed the dimension, which
    // PETSc rejects outright -- count_rows caps them.
    std::vector<PetscInt> diag, off;
    count_rows(row, col, diag, off);
    Mat M = new_matrix(diag, off, "MatCreate");
    diag = std::vector<PetscInt>();
    off = std::vector<PetscInt>();
    scatter_by_row(M, row, col, val, n);
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly");
    return M;
  }

  // The scatter alone, for a matrix the caller preallocated itself -- the
  // preconditioner adds terms that never become triplets, so it has to own the
  // preallocation.
  void scatter_by_row(Mat M, const std::vector<Index>& row, const std::vector<Index>& col,
                      const std::vector<double>& val, PetscInt n) const {
    const auto rows = static_cast<std::size_t>(n);
    const std::size_t nnz = val.size();
    // The counting sort is done in the solver's numbering, so a row of the
    // matrix is a row of the partition: the entries a rank owns end up
    // adjacent here as well as in PETSc.
    std::vector<PetscInt> begin(rows + 1, 0);
    for (std::size_t k = 0; k < nnz; ++k) ++begin[static_cast<std::size_t>(at(row[k])) + 1];
    for (std::size_t i = 0; i < rows; ++i) begin[i + 1] += begin[i];

    std::vector<PetscInt> slot_of(begin.begin(), begin.end() - 1);
    std::vector<PetscInt> cols(nnz);
    std::vector<PetscScalar> vals(nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
      const auto slot = static_cast<std::size_t>(slot_of[static_cast<std::size_t>(at(row[k]))]++);
      cols[slot] = at(col[k]);
      vals[slot] = val[k];
    }
    slot_of = std::vector<PetscInt>();
    for (PetscInt i = 0; i < n; ++i) {
      // a row is written by the rank that owns it and by no other, whether
      // the triplets came from the whole mesh or from a subdomain and its halo
      if (!owns(i)) continue;
      const auto b = static_cast<std::size_t>(begin[static_cast<std::size_t>(i)]);
      const auto e = static_cast<std::size_t>(begin[static_cast<std::size_t>(i) + 1]);
      if (e == b) continue;
      check(MatSetValues(M, 1, &i, static_cast<PetscInt>(e - b), cols.data() + b, vals.data() + b,
                         ADD_VALUES),
            "MatSetValues");
    }
  }

  void build_matrix(const SparseSystem& A) { M_ = assemble(A.row, A.col, A.value, n_); }

  void build_ksp() {
    KSP ksp = nullptr;
    check(KSPCreate(comm_, &ksp), "KSPCreate");
    check(KSPSetOperators(ksp, M_, M_), "KSPSetOperators");
    check(KSPSetType(ksp, opts_.direct() ? KSPPREONLY : opts_.method.c_str()), "KSPSetType");
    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    // "riesz" is this layer's name, not a PETSc type: it resolves to a
    // fieldsplit whose blocks are the Riesz map, set up below.
    // "exact" is a diagnostic: P = A, factorized. It preconditions perfectly,
    // so a Krylov method must converge in one iteration -- a test of the Pmat
    // wiring rather than of any preconditioner. If this takes more than one
    // step, KSPSetOperators(ksp, Amat, Pmat) is not reaching the solver and
    // nothing built on top of it can be trusted.
    const bool exact = !opts_.direct() && opts_.preconditioner == "exact";
    const bool riesz = !opts_.direct() && opts_.preconditioner == "riesz";
    const std::string pc_type = (opts_.direct() || exact)
                                    ? std::string(PCLU)
                                    : (riesz ? std::string(PCFIELDSPLIT) : opts_.preconditioner);
    check(PCSetType(pc, pc_type.c_str()), "PCSetType");
    // the package is a property of a factorization, so it is set only when the
    // preconditioner is one; naming it otherwise is how a silent no-op happens
    // Complete factorizations only. Naming a package on an incomplete one
    // changes what it computes -- PCILU under a package that offers no ILU
    // quietly becomes an exact solve, and the iteration count then says the
    // preconditioner is excellent when there is no iteration happening.
    const bool factorizing = pc_type == "lu" || pc_type == "cholesky";
    const std::string package =
        distributed_ ? parallel_package(opts_.factorization) : opts_.factorization;
    if (factorizing && !package.empty() && package != "petsc") {
      check(PCFactorSetMatSolverType(pc, package.c_str()), "PCFactorSetMatSolverType");
    }
    if (!opts_.direct()) {
      check(KSPSetTolerances(ksp, opts_.rtol, opts_.atol, PETSC_DEFAULT,
                             static_cast<PetscInt>(opts_.max_iterations)),
            "KSPSetTolerances");
    }
    if (exact) {
      Mat P = nullptr;
      check(MatDuplicate(M_, MAT_COPY_VALUES, &P), "MatDuplicate(P=A)");
      check(KSPSetOperators(ksp, M_, P), "KSPSetOperators(A, P=A)");
      riesz_.push_back(P);
    }
    if (riesz) build_riesz(ksp, pc);
    // MUMPS workspace headroom.
    //
    // MUMPS sizes its working array from a symbolic estimate. On an indefinite
    // saddle point -- which every mixed form is, with structural zeros on the
    // diagonal of the multiplier blocks -- delayed pivots make the real fill
    // far exceed that estimate, and MUMPS then writes past the array rather
    // than reporting a shortage: a SEGV inside the factorization, on a system
    // that is perfectly well posed. The default headroom (ICNTL(14), ~20%) is
    // enough for the small cases and not for a 90k-dof fault mesh.
    //
    // Set on the global options database as strings so this compiles whether or
    // not PETSc was built with the MUMPS headers exposed; PETSc ignores an
    // option no solver claims.
    // Only when MUMPS is the package. Set unconditionally these are options no
    // solver claims, and PETSc reports every run as having unused options.
    if (factorizing && package == "mumps") {
      PetscOptionsSetValue(nullptr, "-mat_mumps_icntl_14", "200");
      PetscOptionsSetValue(nullptr, "-mat_mumps_icntl_24", "1");  // detect null pivots
    }

    if (!prefix_.empty()) {
      check(KSPSetOptionsPrefix(ksp, prefix_.c_str()), "KSPSetOptionsPrefix");
    }
    // options last, so the command line can override every choice above —
    // which is how an iterative method is selected without a recompile
    check(KSPSetFromOptions(ksp), "KSPSetFromOptions");
    ksp_ = ksp;
  }

  SolveReport run(const SparseSystem& A, const std::vector<double>& b, std::vector<double>& x) {
    KSP ksp = ksp_;
    Vec sol = sol_;
    const PetscInt n = n_;
    SolveReport out;
    ads_failed_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    const PetscErrorCode e = KSPSolve(ksp, rhs_, sol);
    out.solve_seconds =
        slowest(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (e == 0) {
      KSPConvergedReason why = KSP_CONVERGED_ITERATING;
      KSPGetConvergedReason(ksp, &why);
      PetscInt its = 0;
      KSPGetIterationNumber(ksp, &its);
      out.converged = why > 0;
      out.iterations = static_cast<int>(its);
      const char* text = nullptr;
      KSPGetConvergedReasonString(ksp, &text);
      out.reason = text != nullptr ? text : "";
      // A breakdown under an auxiliary-space block is almost always the block.
      //
      // ADS is built by hypre from the maps and the coordinates it is given,
      // and on some partitions of some meshes what it returns is not positive
      // definite -- CG says so outright (PC_FAILED), GMRES merely breaks down
      // with its preconditioned residual falling and its true residual stuck.
      // Neither message points at the preconditioner, so the flag does: the
      // one-shot solve reads it and retries with the exact block; a caller on
      // the bind-once path holds the operator and makes that call itself.
      ads_failed_ = !out.converged && ads_block_ &&
                    (why == KSP_DIVERGED_BREAKDOWN || why == KSP_DIVERGED_PC_FAILED ||
                     why == KSP_DIVERGED_INDEFINITE_PC);
      if (ads_failed_) {
        out.reason += " -- the auxiliary-space (ADS) block is not usable on this partition";
      }

      // The answer goes back to every rank. The model that reads it -- cell
      // pressures, stresses, the error norms the examples print -- is still
      // replicated, so each rank needs the whole vector, not its slice. One
      // gather at the end of the solve is what keeps the rest of the code
      // unchanged while the algebra is distributed.
      Vec whole = sol;
      VecScatter gather = nullptr;
      if (distributed_) {
        check(VecScatterCreateToAll(sol, &gather, &whole), "VecScatterCreateToAll");
        check(VecScatterBegin(gather, sol, whole, INSERT_VALUES, SCATTER_FORWARD), "VecScatter");
        check(VecScatterEnd(gather, sol, whole, INSERT_VALUES, SCATTER_FORWARD), "VecScatter");
      }
      const PetscScalar* v = nullptr;
      check(VecGetArrayRead(whole, &v), "VecGetArrayRead");
      if (old_of_.empty()) {
        x.assign(v, v + n);
      } else {
        // back to the caller's numbering: it knows nothing of the partition
        x.assign(static_cast<std::size_t>(n), 0.0);
        for (PetscInt i = 0; i < n; ++i) {
          x[static_cast<std::size_t>(old_of_[static_cast<std::size_t>(i)])] = v[i];
        }
      }
      check(VecRestoreArrayRead(whole, &v), "VecRestoreArrayRead");
      if (distributed_) {
        VecDestroy(&whole);
        VecScatterDestroy(&gather);
      }
      // reported from the system, not from the backend's own iteration: a
      // direct solve reports zero iterations and would otherwise say nothing
      out.residual = true_residual(A, b, x);
      out.block_solver = block_solver_;
    } else {
      out.reason = "KSPSolve failed";
    }

    if (e != 0) throw std::runtime_error("PetscSolver: " + out.reason);
    return out;
  }

  void release() {
    if (ksp_ != nullptr) KSPDestroy(&ksp_);
    if (sol_ != nullptr) VecDestroy(&sol_);
    if (rhs_ != nullptr) VecDestroy(&rhs_);
    for (Mat& r : riesz_) {
      if (r != nullptr) MatDestroy(&r);
    }
    riesz_.clear();
    if (M_ != nullptr) MatDestroy(&M_);
    ksp_ = nullptr;
    sol_ = rhs_ = nullptr;
    M_ = nullptr;
    bound_ = nullptr;
    new_of_ = std::vector<PetscInt>();
    old_of_ = std::vector<PetscInt>();
  }

  SolverOptions opts_;
  bool ads_failed_{false};  // the last run died in the auxiliary-space block
  double matrix_seconds_{0.0}, preconditioner_seconds_{0.0};
  SpaceNorm norm_;
  std::vector<int> condensable_;
  std::vector<Mat> riesz_;  // the diagonal blocks of the Riesz map
  std::string prefix_;
  // the parallel layout: one process leaves these at [0, n) on PETSC_COMM_SELF
  MPI_Comm comm_{PETSC_COMM_SELF};
  bool distributed_{false};
  int rank_{0};
  PetscInt n_local_{0}, own_begin_{0}, own_end_{0};
  // the partition, and the permutation it induces: new_of_ takes the caller's
  // numbering to the solver's and old_of_ brings the answer back
  std::vector<int> owners_;
  std::vector<PetscInt> new_of_, old_of_;
  double local_entries_{0.0}, off_rank_entries_{0.0};
  bool ads_block_{false};
  std::string block_solver_;
  Mat M_{nullptr};
  KSP ksp_{nullptr};
  Vec rhs_{nullptr}, sol_{nullptr};
  PetscInt n_{0};
  const SparseSystem* bound_{nullptr};
};

}  // namespace mimetika::solver
