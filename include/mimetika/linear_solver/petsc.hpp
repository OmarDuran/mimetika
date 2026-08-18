#pragma once

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/linear_solver/linear.hpp"

// PETSC, WITH A DIRECT FACTORIZATION FIRST.
//
// A direct solve is the right instrument while a discretization is being
// validated: it answers "is this operator right" without a preconditioner
// standing between the question and the answer. If a direct solve gives the
// wrong displacement field, the discretization is wrong — there is nowhere
// else for the error to have come from. That is worth a great deal when the
// alternative is debugging a Krylov method and a mixed-form operator at the
// same time.
//
// MUMPS rather than PETSc's built-in LU because the systems here are
// saddle points: indefinite, so the factorization needs symmetric pivoting to
// stay stable, and PETSc's own LU does not do it well. MUMPS also handles the
// zero diagonal blocks — the (u,u) and (gamma,gamma) blocks that make this a
// saddle point in the first place — without a shift.
//
// AN ITERATIVE PATH IS THE SAME OBJECT with a different prefix, which is why
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

// A FACTORIZATION IS NOT A FACTORIZATION ON EVERY COMMUNICATOR. PETSc's own LU
// and Cholesky, SuperLU and ICC are sequential codes; asked for on several
// processes they fail at setup rather than distributing themselves. MUMPS is
// the one in this build that does both, and it is already the choice for these
// saddle points, so a sequential name becomes it. An incomplete factorization
// has no distributed form at all and becomes block Jacobi over the ranks, each
// block keeping the incomplete factorization it asked for.
inline std::string parallel_package(const std::string& package) {
  return (package.empty() || package == "petsc" || package == "superlu") ? "mumps" : package;
}

// HOW THE SYSTEM IS SOLVED, as an argument rather than an environment.
//
// Every one of these was reachable only through the MIMETIKA_FACTOR environment
// variable or the PETSc options database, which is not an interface: invisible
// to the caller, absent from the Python surface, silently ignored when
// misspelled, and impossible to set differently for two solves in one process.
//
// A misspelled value here is refused by PETSc and surfaces as an exception.
// THE NORM OF THE PRODUCT SPACE. P is its Gram matrix, and nothing else.
//
// A maps X to its DUAL, so a Krylov method -- which needs an operator X -> X --
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
// Both are the same statement: the FIRST factor carries the material inner
// product plus the graph term of its differential, and every factor after it
// carries plain L^2. The multiplier for weak symmetry is an L^2 factor like the
// displacement -- it is not special, and giving it anything else is a different
// preconditioner.
//
// HOW EACH TERM READS IN THIS DOF BASIS, which is the only part that is not
// textbook:
//
//   (A sigma, sigma)  is the assembled (0,0) block. The discrete Hodge IS that
//                     form, so it is taken rather than rebuilt.
//
//   ||div sigma||^2   is NOT B^T B. The facet dof is the measure-weighted
//                     moment, so (B sigma)_E = int_E div sigma, the INTEGRAL;
//                     div sigma is constant on the cell, so its square
//                     integrates to (B sigma)_E^2 / |E| and the term is
//                     B^T diag(1/|E|) B. On a uniform mesh that is a constant
//                     factor and passes for a tuning knob; on a graded one it
//                     varies cell by cell and no constant repairs it.
//
//   ||u||^2           is diag(|E|): a cell dof is the VALUE on the cell.
//
// So one quantity -- the cell measure -- fixes every block, which is what makes
// this one norm rather than a set of separately tuned matrices.
struct SpaceNorm {
  // the factors of X, as index sets, first factor first
  std::vector<std::vector<int>> factors;
  // the L2 weight of every unknown of factors 1.., one vector per such factor:
  // the measure of the cell that unknown belongs to.
  std::vector<std::vector<double>> l2_weight;

  // The graph term is NOT stated separately: it is B^T W^{-1} B with W the
  // multiplier block above, which is the whole content of
  //
  //     P = diag( M + B^T W^{-1} B ,  W ) .
  //
  // W IS THE L2 MASS IN THE DOF BASIS, and that basis differs between the two.
  // Flow's cell unknown is the VALUE of p on the cell, so ||p||^2 = sum p^2 |E|
  // and W = |E|. Elasticity's cell unknowns are MOMENTS -- u_dof = int_E u, as
  // CauchyElasticityModel::displacement shows by dividing by the measure to
  // report a mean -- so ||u||^2 = sum (u_dof/|E|)^2 |E| = sum u_dof^2 / |E| and
  // W = 1/|E|. The rotation is a moment likewise.
  //
  // Measured on the Lame annulus, cond(P^{-1}A) with W = |E| is 8e2 and rising
  // with refinement; with W = 1/|E| it is 3.2, 3.4, 3.5 over the same three
  // meshes -- flat, which is the property the map exists to have.

  // A CONSTRAINED UNKNOWN IS NOT IN THE SPACE. Its row of A is the constraint,
  // scale * e_i^T, not a form; leaving the norm's entries there preconditions an
  // equation that is not the one being solved, and the iteration count starts
  // growing with h again. P carries the same row, so those unknowns contribute
  // the identity to P^{-1}A and drop out of the Krylov space.
  // Which multipliers contribute a graph term: the DIFFERENTIAL constraint does
  // (factor 1), an ALGEBRAIC one does not. AFW's inf-sup is proved with
  // ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 -- skw is bounded
  // L^2 -> L^2, so the rotation adds nothing to the stress norm.
  std::size_t differential_factors{1};
  bool carries_graph_term(std::size_t f) const { return f >= 1 && f <= differential_factors; }

  std::vector<int> pinned;
  std::vector<double> pinned_diagonal;

  // THE DE RHAM MAPS AN AUXILIARY-SPACE SOLVER NEEDS.
  //
  // ADS preconditions an H(div) operator by splitting it along the complex --
  // a field becomes a vector potential in H(curl) plus a part carried by the
  // vertex spaces -- and the maps that take it there are the discrete GRADIENT
  // (edges x vertices) and CURL (faces x edges). Those are not a new
  // construction: they are the boundary operators of the complex as stored,
  // which is why a library built on a chain complex can hand them over instead
  // of reconstructing them from an element table.
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

  // ADS also needs the VERTEX COORDINATES, one row of `space_dim` per column of
  // the gradient. They are not redundant with the maps above: the complex is
  // metric-free, and the auxiliary spaces the solver builds are spaces of
  // piecewise linear FIELDS -- it recovers their interpolation by applying the
  // maps to the coordinate functions x, y, z. This is the whole metric content
  // ADS asks for.
  std::vector<double> vertex_coordinates;  // row-major, n_vertices x space_dim
  int space_dim{3};

  // WHICH PROCESS OWNS EACH ENTITY the two maps address -- vertices, edges,
  // faces -- by the same rule and the same partition that owns the unknowns.
  //
  // The maps above are stated in the complex's own numbering, which is a
  // serial numbering. Distributed, hypre needs each of the three spaces laid
  // out across the ranks, and the FACES must be laid out exactly as the block
  // is: its row i has to be that block's row i. This is what lets the solver
  // renumber all three consistently, without asking the complex to be
  // distributed.
  //
  // Empty on one process, where there is nothing to lay out.
  std::vector<std::vector<int>> entity_owner;  // [k][entity]

  // THE LOWEST-ORDER SUBSPACE OF THE FIRST FACTOR, when the first factor is
  // not itself lowest order.
  //
  // ADS is written for ONE UNKNOWN PER FACET. Flow's RT space is that already;
  // the AFW stress space is not -- a facet carries d traction components, each
  // measured against the d functions of the facet P_1 basis, so d^2 unknowns
  // sit on it. The auxiliary-space argument still applies, and this is the map
  // it applies through: the injection of the facet-CONSTANT moments, which are
  // a subset of the degrees of freedom rather than a computed interpolation,
  // because the space is defined by its moments and the constant one is one of
  // them.
  //
  // Columns are ordered component-major -- all facets of component 0, then
  // component 1 -- so that each component is a contiguous run of the coarse
  // space and ADS can be given it as the scalar H(div) problem it expects.
  //
  // Rows are GLOBAL unknowns; build_riesz maps them into the block.
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
  // HOW THE RIESZ BLOCKS ARE INVERTED. The first factor is SPD but LARGE -- it
  // is most of the unknowns -- so a complete factorization of it costs about
  // what a direct solve of the whole system costs, in time and in fill. That is
  // exact and it does not scale.
  //
  // An approximate inverse is still a Riesz map as long as it is spectrally
  // equivalent to the block: the iteration count rises by a constant and stops
  // depending on the mesh. "gamg" is algebraic multigrid, which is the
  // scalable choice; "lu" is the exact one, for small problems and for
  // checking that an approximation is what changed an answer.
  std::string riesz_block_pc{};
  // THE RIESZ BLOCK IS SPD, AND IS SOLVED AS SUCH -- BY MUMPS.
  //
  // `factorization` defaults to SuperLU because the whole system is an
  // INDEFINITE saddle point. The first Riesz factor has no such structure: a
  // material inner product plus B^T W^-1 B, symmetric positive definite. So it
  // takes a CHOLESKY, half the fill and half the work of an LU.
  //
  // WHICH PACKAGE IS NOT A DETAIL. Measured on the H(div) block of the 22k-cell
  // polyhedral mesh (77k unknowns), solving to 1e-9:
  //
  //     Cholesky / MUMPS    64 iterations    1.1 s
  //     Cholesky / PETSc    38              64.1 s
  //     Cholesky / SuperLU  -- SuperLU has no Cholesky
  //
  // PETSc's own factorization takes FEWER iterations, because it is the more
  // exact of the two, and is sixty times slower: it orders the matrix
  // naturally, and the fill of a natural ordering on an unstructured
  // three-dimensional block is ruinous. MUMPS reorders before it factors.
  //
  // An empty value falls back to `factorization`, which is SuperLU -- and
  // SuperLU cannot do a Cholesky at all, so that fallback is an error rather
  // than a slow path. Naming MUMPS here is what keeps the default working.
  std::string riesz_block_factorization{"mumps"};
  // How the first factor is inverted, and it is a MEMORY decision.
  //
  //   0   exact: a complete factorization. 21 iterations flat, and fill that
  //       grows with the block -- 450 MB at 33k unknowns, extrapolating to
  //       tens of gigabytes on a mesh of tens of thousands of polyhedra.
  //   >0  that many preconditioned CG steps instead. 25-28 iterations, so the
  //       outer count barely moves, and NO FILL: 46 MB at the same size. The
  //       outer method must then be flexible, which is applied automatically.
  //   -1  choose: exact while the block is small enough to factor, inexact
  //       above it. The threshold is where the fill stops being affordable
  //       rather than where the method changes character.
  int riesz_block_its{-1};
  double riesz_block_rtol{1e-4};
  // First-factor unknowns above which the exact solve is refused. Set from
  // where the FILL stops being affordable rather than from where the method
  // changes: with MUMPS the H(div) block of a 22k-cell polyhedral mesh -- 77k
  // unknowns -- factors in 1.5 s, so the limit sits well above it.
  int riesz_exact_limit{400000};
  // First-factor unknowns above which the AUXILIARY-SPACE solver is preferred
  // to the exact one, when the complex makes it possible at all. Lower than
  // the limit above, and for a different reason: a Cholesky of this block
  // still succeeds at 77k unknowns, it just stops being the cheapest way to
  // apply P. Measured on a uniform refinement, solve time
  //
  //     dofs     12k    41k    96k
  //     cholesky 0.05   0.47   2.55 s
  //     ads      0.08   0.30   0.79 s
  //
  // -- the crossover is around 25k, and past it the gap only widens, because
  // Cholesky's cost per iteration grows with the fill and ADS's does not.
  int riesz_ads_limit{400000};
  // RENUMBER BY THE MESH PARTITION when there is more than one process. Off,
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

  // HOW MUCH OF THE MATRIX CROSSES A PROCESS BOUNDARY. PETSc stores an MPIAIJ
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

  // WHO OWNS EACH UNKNOWN, one rank per global unknown, in the caller's own
  // numbering.
  //
  // Without this the rows are split by index, and an index is not a place: a
  // rank's rows are then scattered over the whole mesh, every mat-vec is
  // nearly all off-rank communication, and the layout that a parallel
  // preconditioner assumes -- that a rank holds a SUBDOMAIN -- is absent.
  // With it the solver renumbers so that each rank's unknowns are the
  // contiguous block PETSc requires, and hands the answer back in the
  // caller's numbering as if nothing had happened.
  //
  // Ignored on one process, where there is nothing to renumber for.
  void set_owners(std::vector<int> owner_of_dof) { owners_ = std::move(owner_of_dof); }

  // DISTRIBUTED ASSEMBLY NEEDS NOTHING FROM THE SOLVER, and that is the point
  // of the convention it uses. A process assembles every cell that contributes
  // to a row it owns -- its own and its halo -- so those rows arrive complete
  // and the rest are dropped, exactly as they are when the assembly is
  // replicated. No stash, no exchange, and one code path for both.
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

  // BIND THE OPERATOR ONCE. A transient linear problem at constant dt has a
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
    if (bound_ != &A) factorize(A);
    SolveReport r = solve(b, x);
    r.matrix_seconds = matrix_seconds_;
    r.preconditioner_seconds = preconditioner_seconds_;
    r.off_rank_fraction = off_rank_fraction();
    return r;
  }

 private:
  // P, ASSEMBLED FROM THE NORM ABOVE. Nothing here decides anything: every
  // block is the term the norm names, read in this dof basis.
  //
  // IT IS A SECOND MATRIX, not an edit of the sub-solvers. PETSc takes the
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

    // A FACTOR IS A CONTIGUOUS RUN, and saying so is not a micro-optimization:
    // a general index set makes MatCreateSubMatrix search for every row it is
    // asked for, and on a block of several hundred thousand that search is the
    // whole cost of building the preconditioner -- minutes, against the second
    // the extraction itself takes from a stride.
    // AN INDEX SET IS PER PROCESS. Every rank names the unknowns of the factor
    // that it OWNS -- the union over ranks is the factor, which is what
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

    // P, BUILT FROM THE TRIPLETS IN ONE PASS.
    //
    // Every block of P is already present in the assembly, so none of it needs
    // to be extracted or multiplied out:
    //
    //   material   the (0,0) entries of A, taken as they stand
    //   graph      B^T W^-1 B, and B is one ROW of A per multiplier. A row has
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

    // THE GRAPH TERM IS NEVER MATERIALIZED AS TRIPLETS.
    //
    // B^T W^-1 B is an outer product per constraint row, so writing it as
    // triplets costs the SQUARE of each row's entry count: on this mesh that
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

    // P IS SYMMETRIC POSITIVE DEFINITE BY CONSTRUCTION -- a material inner
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
    // WHEN THE COMPLEX IS THERE, ADS IS THE DEFAULT FOR A BIG BLOCK.
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
    // THE CROSSOVER IS NOT WHERE IT WAS MEASURED TO BE.
    //
    // ADS was made the default above 25k unknowns on a comparison against a
    // factorization that did not reorder. Against MUMPS, which does, the exact
    // block wins at every size that fits in memory here -- 0.58 s against 1.00
    // at 119k unknowns, 1.71 against 2.86 at 322k -- because the iteration
    // count is lower (39 against 60) and each application is a pair of
    // triangular solves rather than a multigrid cycle over three auxiliary
    // spaces.
    //
    // So the auxiliary-space route is taken where the factorization is refused
    // for its FILL, which is what riesz_exact_limit is for, and not before.
    // Ask for it with riesz_block_pc = "ads" to have it sooner.
    const bool through_subspace = !norm_.lowest_order.empty();
    // BOTH ROUTES ARE DISTRIBUTED: ADS on a block whose unknowns are the
    // facets, and the two-level cycle that reaches it through the
    // facet-constant subspace. Each needs the entities laid out on the
    // partition, which is the one thing that must be supplied for it.
    const bool ads_possible_here =
        ads_possible && (!distributed_ || norm_.entity_owner.size() >= 3);
    const bool use_ads = ads_possible_here &&
                         (inexact_block || (!through_subspace && n0 > opts_.riesz_ads_limit));
    std::string b0_pc = !opts_.riesz_block_pc.empty() ? opts_.riesz_block_pc
                        : use_ads                     ? "ads"
                        : inexact_block               ? "icc"
                                                      : "cholesky";

    // A TWO-LEVEL CYCLE IS NOT A COMPLETE SOLVER, and one application of it is
    // not enough. Where ADS acts on the block itself, a single V-cycle is
    // already a good inverse -- flow converges in 33 iterations with one. Where
    // it acts through the facet-constant subspace, one cycle leaves the
    // non-constant moments to a smoother and the outer count is 195 against
    // Cholesky's 25; solving the block with a short CG under the same cycle
    // brings it to 37 and costs less than the difference. So the cycle asks for
    // the inner Krylov itself rather than waiting for the block to be big
    // enough to trigger it.
    const bool two_level = b0_pc == "ads" && !norm_.lowest_order.empty();
    // ONE CYCLE IS WHAT THE RIESZ MAP ASKS FOR.
    //
    // What the theory wants of the first block is SPECTRAL EQUIVALENCE, not an
    // accurate solve: an inner Krylov run to a tight tolerance buys a precision
    // the outer iteration cannot use, and pays for it in every application.
    // Measured on 790k unknowns over four processes, block solved by
    //
    //     one ADS cycle          8.3 s    60 outer iterations
    //     CG(5)  to 1e-1        11.6      40
    //     CG(10) to 1e-2        19.3      35
    //     CG(200) to 1e-4       37.4      36     <- what this used to default to
    //
    // Fewer outer iterations, more time: the count falls and the cost per count
    // rises faster. So ADS is applied ONCE unless the caller asks otherwise,
    // and stays a fixed operator that plain GMRES may use.
    //
    // The subspace route is the exception, and was measured too: one cycle of
    // its two-level cycle leaves 195 outer iterations against 37 with a short
    // CG under it, because a cycle whose coarse space is a SUBSPACE corrects
    // less of the block than one acting on the block itself.
    const bool single_cycle = b0_pc == "ads" && !two_level && opts_.riesz_block_its < 0;
    const bool inner_krylov = (inexact_block && !single_cycle) || two_level;
    const int block_its = opts_.riesz_block_its > 0 ? opts_.riesz_block_its
                          : two_level              ? 50
                                                   : 200;
    const double block_rtol = opts_.riesz_block_its > 0 ? opts_.riesz_block_rtol
                              : two_level              ? 1e-2
                                                       : opts_.riesz_block_rtol;
    // an inner Krylov makes the preconditioner a VARYING operator, which only a
    // flexible outer method may use; applying it under plain gmres is a silent
    // wrong answer, so the promotion happens here rather than in the caller
    if (inner_krylov) check(KSPSetType(ksp, KSPFGMRES), "KSPSetType(fgmres)");

    // THE SUB-SOLVERS ARE SET THROUGH THE OPTIONS DATABASE, before setup.
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
      set(p0 + "ksp_type", "cg");
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
    // the L2 factors are DIAGONAL, so Jacobi inverts them exactly and anything
    // heavier is wasted
    set(p1 + "ksp_type", "preonly");
    set(p1 + "pc_type", "jacobi");

    // AN AUXILIARY-SPACE SOLVER ON THE FIRST FACTOR.
    //
    // ADS preconditions an H(div) operator by splitting it along the de Rham
    // complex -- a field becomes a vector potential in H(curl) plus a part
    // carried by the vertex spaces -- and preconditioning each piece where
    // multigrid actually works. Classical AMG cannot: the near-null space here
    // is the DIVERGENCE-FREE fields, not the constants, which is why BoomerAMG
    // on this block needs more iterations the finer the mesh. ADS costs no
    // fill and is linear in the unknowns, which is the only thing that scales.
    //
    // Its two maps are the discrete gradient and curl -- the complex's own
    // boundary operators -- and they must be attached BEFORE the sub-PC is set
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
      // ONE APPLICATION, or a short CG under it. A single ADS V-cycle is a
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
      // ONE UNKNOWN PER FACET, or a lowest-order subspace of one that is not.
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

  // THE ENTITIES, LAID OUT LIKE THE UNKNOWNS. Same sort -- by owner, stable
  // within a rank -- so a rank's faces are contiguous and in the order its
  // block rows are in, which is the one thing hypre cannot be told and must
  // simply be true.
  struct EntityLayout {
    std::vector<PetscInt> new_of, old_of;
    // where each rank's run starts, n_ranks + 1 of them: needed to build a
    // space ON TOP of this one, which is what the coarse space of the
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

  // ADS on a block whose unknowns ARE the facets, with the two maps of the
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
      // THE THREE SPACES, LAID OUT ON THE PARTITION.
      //
      // hypre is handed the maps between them, so all three have to agree with
      // each other AND the faces have to agree with the block: C's row i is
      // the block's row i, or the solver is preconditioning a permutation of
      // its own operator. Nothing here communicates -- every rank sorts the
      // same ownership array the same way -- which is the point of an
      // ownership rule that needs no negotiation.
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

    // ADS'S OWN KNOBS ARE HYPRE'S, and hypre's are reachable only through the
    // options database: the cycle type, the relaxation, the AMG parameters of
    // each auxiliary space. A PC built in code and set up in code never reads
    // that database, so a -pc_hypre_ads_* asked for on the command line was
    // being accepted and silently ignored.
    //
    // READING IT IS NOT FREE, which is why it is conditional. PCSetFromOptions
    // writes PETSc's own defaults for every hypre parameter it knows, and
    // those are not hypre's: on the 22k-cell polyhedral mesh they cost 1.6x
    // the time per application (3.7 s became 6.0 s at the same tolerance). So
    // the database is read only when the caller has actually set one of these,
    // and left alone otherwise.
    //
    // What the knobs are worth, measured on that mesh: the default cycle takes
    // 44 iterations and 3.7 s, and `-pc_hypre_ads_cycle_type 11` -- which
    // solves the vector Poisson problems more thoroughly -- takes 18 and 2.6.
    // It is NOT the default because it breaks GMRES down on eight processes
    // here, and a preconditioner that is twice as fast until it fails is not a
    // better default than one that works.
    {
      static const char* const knobs[] = {
          "-pc_hypre_ads_cycle_type",   "-pc_hypre_ads_relax_type",
          "-pc_hypre_ads_relax_times",  "-pc_hypre_ads_relax_weight",
          "-pc_hypre_ads_omega",        "-pc_hypre_ams_cycle_type",
          "-pc_hypre_ads_amg_alpha_theta", "-pc_hypre_ads_amg_beta_theta",
          "-pc_hypre_ads_print_level"};
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

  // A TWO-LEVEL CYCLE WHOSE COARSE SPACE IS THE FACET CONSTANTS.
  //
  // The AFW stress block is not an ADS problem: a facet carries d traction
  // components measured against the d functions of its P_1 basis, so d^2
  // unknowns sit on it and hypre would not know what a facet is. But the
  // auxiliary-space argument is about a SUBSPACE where the operator is
  // spectrally equivalent to something a solver exists for, and here that
  // subspace is written down rather than interpolated: the facet-constant
  // moments are a subset of the degrees of freedom, so the injection is a
  // matrix of ones.
  //
  //   smoother   Chebyshev/Jacobi on the whole block -- the higher moments are
  //              LOCAL to a facet, and what is local is what a smoother is for
  //   coarse     the constants, one H(div) problem per component: the coupling
  //              between components is the material's, bounded and dropped by
  //              an additive split, and each diagonal block is what ADS takes
  //
  // The coarse operator is Galerkin, P^T A P, so nothing about the physics is
  // restated at the coarse level -- it is the same operator seen on the
  // subspace.
  void build_lowest_order_cycle(PC pc) {
    const auto& inj = norm_.lowest_order;
    const int nc = norm_.lowest_order_components;
    if (nc < 1 || inj.cols % nc != 0) {
      throw std::invalid_argument("PetscSolver: lowest-order injection is not component-blocked");
    }
    // THE BLOCK'S ROW OF EACH UNKNOWN. A fieldsplit gives its sub-matrix the
    // rows of its index set in that set's order, which is ascending in the
    // SOLVER's numbering -- so on several processes the block's rows are
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

    // THE COARSE SPACE, LAID OUT ON THE PARTITION.
    //
    // Serially it is copy-major: every face of copy 0, then copy 1. That
    // ordering cannot survive distribution -- it would give the first ranks
    // whole copies and the last ranks none -- so distributed it becomes
    // (rank, copy, face): each process holds its own faces of every copy, and
    // a copy is still a contiguous run WITHIN a process, which is what the
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

    // THE SMOOTHER IS FACET-LOCAL, and exactly so.
    //
    // What the coarse space does not carry is the non-constant moments, and
    // those live ON one facet: the divergence sees only the constants, so the
    // rest of a facet's block is coupled to the mesh through the material mass
    // alone. Inverting each facet's block exactly is therefore the right
    // smoother rather than an expensive one -- it is d^2 x d^2, one dense
    // solve per facet -- and point Jacobi, which splits those moments from
    // each other, is what makes the cycle look weak.
    KSP smoother = nullptr;
    check(PCMGGetSmoother(pc, 1, &smoother), "PCMGGetSmoother");
    check(KSPSetType(smoother, KSPCHEBYSHEV), "smoother KSPSetType");
    check(KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2),
          "smoother KSPSetTolerances");
    PC smooth_pc = nullptr;
    check(KSPGetPC(smoother, &smooth_pc), "smoother KSPGetPC");
    check(PCSetType(smooth_pc, PCPBJACOBI), "smoother PCSetType");

    // The coarse solve is a split by component, and each component is only
    // reachable after the coarse operator exists -- which is what setting the
    // cycle up computes. So it is brought up with a PC that costs nothing, and
    // the real ones are attached to the sub-solves afterwards.
    KSP coarse = nullptr;
    check(PCMGGetCoarseSolve(pc, &coarse), "PCMGGetCoarseSolve");
    check(KSPSetType(coarse, KSPPREONLY), "coarse KSPSetType");
    PC coarse_pc = nullptr;
    check(KSPGetPC(coarse, &coarse_pc), "coarse KSPGetPC");
    // ONE COPY NEEDS NO SPLIT. A flux has a single H(div) field, so its coarse
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
    check(PCFieldSplitSetType(coarse_pc, PC_COMPOSITE_ADDITIVE), "coarse PCFieldSplitSetType");
    check(PCSetUp(pc), "PCSetUp(mg)");
    check(PCSetUp(coarse_pc), "PCSetUp(coarse fieldsplit)");
    PetscInt n_part = 0;
    KSP* csub = nullptr;
    check(PCFieldSplitGetSubKSP(coarse_pc, &n_part, &csub), "PCFieldSplitGetSubKSP(coarse)");
    for (PetscInt c = 0; c < n_part; ++c) {
      PC cpc = nullptr;
      check(KSPSetType(csub[c], KSPPREONLY), "coarse component KSPSetType");
      check(KSPGetPC(csub[c], &cpc), "coarse component KSPGetPC");
      // the component's rows ARE this rank's faces, which is what ADS is told
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

  // WHICH ROWS THIS PROCESS OWNS.
  //
  // One process and the solver is what it was: PETSC_COMM_SELF, sequential
  // matrices, every row local. Several and the algebra is distributed across
  // MPI_COMM_WORLD, in the layout PETSc itself chooses -- a contiguous run per
  // rank, which is what its Mat and Vec require and what every parallel
  // preconditioner assumes.
  //
  // ASSEMBLY IS STILL REPLICATED at this stage: every rank builds the whole
  // triplet list and inserts only the rows it owns. That is deliberate and it
  // is not the end state -- it makes the solve parallel while leaving the
  // partition, the ghosts and the owned-cell assembly for the step that
  // introduces them, so a wrong answer here can only come from the layout.
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

  // THE RENUMBERING IS A SORT, and nothing more: the unknowns of rank 0 first,
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

  // A PHASE TAKES AS LONG AS ITS SLOWEST PROCESS, and a report from one of them
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

  // THE NONZERO COUNTS PETSc PREALLOCATES FROM, in two steps because they come
  // from two places: the triplets, and the terms P adds that were never
  // triplets. Both are counted over ALL rows in the solver's numbering, and
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

  // A MATRIX FROM TRIPLETS, one call per row rather than one per entry.
  //
  // Handing PETSc one triplet at a time costs a search of the row for every
  // entry, and an assembly of tens of thousands of polyhedra emits of order
  // 10^8 of them. Grouping by row first -- a counting sort, one pass to count
  // and one to place -- turns that into one call per row, and PETSc sums the
  // duplicates inside each call as before. The scratch is scoped so it is
  // returned before anything else is allocated.
  Mat assemble(const std::vector<Index>& row, const std::vector<Index>& col,
               const std::vector<double>& val, PetscInt n) const {
    // TRIPLETS, NOT NONZEROS: a row's triplet count is an upper bound on its
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
    // THE COUNTING SORT IS DONE IN THE SOLVER'S NUMBERING, so a row of the
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
    // "exact" is a DIAGNOSTIC: P = A, factorized. It preconditions perfectly,
    // so a Krylov method must converge in one iteration -- which is what makes
    // it a test of the Pmat wiring rather than of any preconditioner. If this
    // takes more than one step, KSPSetOperators(ksp, Amat, Pmat) is not
    // reaching the solver and nothing built on top of it can be trusted.
    const bool exact = !opts_.direct() && opts_.preconditioner == "exact";
    const bool riesz = !opts_.direct() && opts_.preconditioner == "riesz";
    const std::string pc_type = (opts_.direct() || exact)
                                    ? std::string(PCLU)
                                    : (riesz ? std::string(PCFIELDSPLIT) : opts_.preconditioner);
    check(PCSetType(pc, pc_type.c_str()), "PCSetType");
    // the package is a property of a FACTORIZATION, so it is set only when the
    // preconditioner is one; naming it otherwise is how a silent no-op happens
    // COMPLETE factorizations only. Naming a package on an incomplete one
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
    // MUMPS WORKSPACE HEADROOM, and it is not a tuning knob here.
    //
    // MUMPS sizes its working array from a symbolic estimate. On an INDEFINITE
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
    // ONLY WHEN MUMPS IS THE PACKAGE. Set unconditionally these are options no
    // solver claims, and PETSc reports every run as having unused options --
    // noise that trains a reader to ignore the one that matters.
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
      // A BREAKDOWN UNDER AN AUXILIARY-SPACE BLOCK IS ALMOST ALWAYS THE BLOCK.
      //
      // ADS is built by hypre from the maps and the coordinates it is given,
      // and on some partitions of some meshes what it returns is not positive
      // definite -- CG says so outright (PC_FAILED), GMRES merely breaks down
      // with its preconditioned residual falling and its true residual stuck.
      // Neither message points at the preconditioner, so this one does.
      if (!out.converged && ads_block_ &&
          (why == KSP_DIVERGED_BREAKDOWN || why == KSP_DIVERGED_PC_FAILED ||
           why == KSP_DIVERGED_INDEFINITE_PC)) {
        out.reason += " -- the auxiliary-space (ADS) block is not usable on this partition; "
                      "try riesz_block_pc='cholesky', or a different number of processes";
      }

      // THE ANSWER GOES BACK TO EVERY RANK. The model that reads it -- cell
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
      // reported from the SYSTEM, not from the backend's own iteration: a
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
  double matrix_seconds_{0.0}, preconditioner_seconds_{0.0};
  SpaceNorm norm_;
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
