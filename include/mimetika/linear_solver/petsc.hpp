#pragma once

#include <petscksp.h>

#include <algorithm>
#include <cmath>
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

// HOW THE SYSTEM IS SOLVED, as an argument rather than an environment.
//
// Every one of these was reachable only through the MIMETIKA_FACTOR environment
// variable or the PETSc options database, which is not an interface: invisible
// to the caller, absent from the Python surface, silently ignored when
// misspelled, and impossible to set differently for two solves in one process.
//
// A misspelled value here is refused by PETSc and surfaces as an exception.
// THE NORM OF THE PRODUCT SPACE. P is its matrix, and nothing else.
//
// A maps X to its DUAL, so a Krylov method -- which needs an operator X -> X --
// requires a map X' -> X. The canonical one is the Riesz map of the inner
// product of X, and with it P^{-1}A has a condition number bounded by the
// inf-sup and continuity constants alone: independent of h. So P is not a
// clever approximation of A. It is the Gram matrix of
//
//     ||(q,p)||_X^2 = (K^{-1} q, q) + ||div q||_{L2}^2 + ||p||_{L2}^2
//
// and the only question is how each term reads in THIS dof basis.
//
//   (K^{-1} q, q)  is A00, already assembled: the discrete Hodge is that form.
//
//   ||div q||^2    is NOT B^T B. The flux dof is the measure-weighted moment
//                  int_f q.n, so (Bq)_E = sum of moments = int_E div q, the
//                  INTEGRAL. div q is constant on the cell, so its value is
//                  (Bq)_E/|E| and its square integrates to (Bq)_E^2/|E|:
//                  the term is B^T diag(1/|E|) B. On a uniform mesh that is a
//                  constant factor and looks like a tuning parameter; on a
//                  graded one it varies cell by cell and no constant repairs it.
//
//   ||p||^2        is diag(|E|): the cell dof is the VALUE on the cell, not its
//                  moment, so the mass is the measure.
//
// One quantity therefore fixes both blocks -- the cell measure -- which is what
// makes this one norm rather than two tuned matrices.
struct BlockSplit {
  std::vector<std::vector<int>> blocks;
  std::vector<double> cell_measure;  // |E|, one per cell unknown

  // A CONSTRAINED UNKNOWN IS NOT IN THE SPACE. Its row of A is the constraint,
  // scale * e_i^T, not a form; leaving the norm's entries there preconditions an
  // equation that is not the one being solved, and the iteration count starts
  // growing with h again. P carries the same row, so those unknowns contribute
  // the identity to P^{-1}A and drop out of the Krylov space.
  std::vector<int> pinned;
  std::vector<double> pinned_diagonal;

  bool empty() const { return blocks.empty(); }
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

  // The factors of the product space. Required by the "riesz" preconditioner
  // and ignored by every other one.
  void set_split(BlockSplit s) { split_ = std::move(s); }

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
    build_matrix(A);
    build_ksp();
    check(VecCreateSeq(PETSC_COMM_SELF, n_, &rhs_), "VecCreate");
    check(VecDuplicate(rhs_, &sol_), "VecDuplicate");
    // force the factorization now rather than on the first solve, so that the
    // cost shows up where it is paid
    check(KSPSetUp(ksp_), "KSPSetUp");
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
    for (PetscInt i = 0; i < n_; ++i) {
      check(VecSetValue(rhs_, i, b[static_cast<std::size_t>(i)], INSERT_VALUES), "VecSetValue");
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
    return solve(b, x);
  }

 private:
  // THE RIESZ MAP OF THE PRODUCT SPACE, AS A SEPARATE PRECONDITIONER MATRIX.
  //
  // The operator of a mixed form is an isomorphism from the product space to
  // its dual, so the preconditioner that makes its condition number
  // mesh-INDEPENDENT is the Riesz map of that space: block diagonal, one block
  // per factor (Arnold-Falk-Winther; Mardal-Winther).
  //
  //     H(div) x L^2   ->   diag( A00 + B^T B ,  M_p )
  //
  // A00 is the discrete Hodge already assembled and B the divergence, so the
  // H(div) inner product <q,q> + <div q, div q> is ALGEBRA ON THE ASSEMBLED
  // OPERATOR and needs no second assembly. Only the L^2 mass of the cell
  // unknown comes from outside, as the cell measures.
  //
  // IT IS A SECOND MATRIX, not an edit of the sub-solvers. PETSc takes the
  // preconditioner from Pmat in KSPSetOperators(ksp, Amat, Pmat) and a
  // fieldsplit reads its diagonal blocks from there. Handing the blocks to the
  // sub-KSPs instead -- after PCSetUp, by KSPSetOperators on each -- is undone
  // the next time the outer KSP sets up and rebuilds them from Pmat, which
  // leaves the preconditioner silently equal to the operator: it converges on
  // nothing and reports DIVERGED_ITS.
  //
  // The (1,1) block of the operator is EMPTY, which is why it cannot serve as
  // its own preconditioner; the mass matrix taking its place is the whole
  // content of the method.
  void build_riesz(KSP ksp, PC pc) {
    if (split_.blocks.size() < 2) {
      throw std::invalid_argument(
          "PetscSolver: the 'riesz' preconditioner needs a block split; call set_split()");
    }
    const auto& flux = split_.blocks[0];
    const auto& cell = split_.blocks[1];
    const auto n0 = static_cast<PetscInt>(flux.size());
    const auto n1 = static_cast<PetscInt>(cell.size());

    std::vector<IS> sets(split_.blocks.size(), nullptr);
    for (std::size_t b = 0; b < split_.blocks.size(); ++b) {
      check(ISCreateGeneral(PETSC_COMM_SELF, static_cast<PetscInt>(split_.blocks[b].size()),
                            split_.blocks[b].data(), PETSC_COPY_VALUES, &sets[b]),
            "ISCreateGeneral");
    }

    // the flux block: A00 + B^T B, the H(div) inner product
    Mat A00 = nullptr, B = nullptr, BtB = nullptr;
    check(MatCreateSubMatrix(M_, sets[0], sets[0], MAT_INITIAL_MATRIX, &A00), "sub(0,0)");
    check(MatCreateSubMatrix(M_, sets[1], sets[0], MAT_INITIAL_MATRIX, &B), "sub(1,0)");
    // ||div q||^2 = (Bq)^T diag(1/|E|) (Bq): scale the rows of B by |E|^{-1/2}
    // so that B^T B is that form, rather than the unweighted one
    Vec inv_root = nullptr;
    check(VecCreateSeq(PETSC_COMM_SELF, n1, &inv_root), "VecCreateSeq");
    for (PetscInt i = 0; i < n1; ++i) {
      const auto k = static_cast<std::size_t>(i);
      const double w = k < split_.cell_measure.size() ? split_.cell_measure[k] : 1.0;
      check(VecSetValue(inv_root, i, 1.0 / std::sqrt(w), INSERT_VALUES), "VecSetValue");
    }
    check(VecAssemblyBegin(inv_root), "VecAssembly");
    check(VecAssemblyEnd(inv_root), "VecAssembly");
    check(MatDiagonalScale(B, inv_root, nullptr), "row scale B");
    VecDestroy(&inv_root);
    check(MatTransposeMatMult(B, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BtB), "B^T diag(1/|E|) B");
    check(MatAXPY(A00, 1.0, BtB, DIFFERENT_NONZERO_PATTERN), "A00 += B^T B");
    MatDestroy(&B);
    MatDestroy(&BtB);

    // scatter it, and the cell mass, into a preconditioner matrix of full size
    std::vector<PetscInt> per_row(static_cast<std::size_t>(n_), 1);
    for (PetscInt i = 0; i < n0; ++i) {
      PetscInt ncols = 0;
      const PetscInt* cols = nullptr;
      check(MatGetRow(A00, i, &ncols, &cols, nullptr), "MatGetRow");
      per_row[static_cast<std::size_t>(flux[static_cast<std::size_t>(i)])] = ncols;
      check(MatRestoreRow(A00, i, &ncols, &cols, nullptr), "MatRestoreRow");
    }
    Mat P = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &P), "MatCreate(P)");
    check(MatSetType(P, MATSEQAIJ), "MatSetType(P)");
    check(MatSetSizes(P, n_, n_, n_, n_), "MatSetSizes(P)");
    check(MatSeqAIJSetPreallocation(P, 0, per_row.data()), "preallocate(P)");
    for (PetscInt i = 0; i < n0; ++i) {
      PetscInt ncols = 0;
      const PetscInt* cols = nullptr;
      const PetscScalar* vals = nullptr;
      check(MatGetRow(A00, i, &ncols, &cols, &vals), "MatGetRow");
      const PetscInt gi = flux[static_cast<std::size_t>(i)];
      for (PetscInt k = 0; k < ncols; ++k) {
        const PetscInt gj = flux[static_cast<std::size_t>(cols[k])];
        check(MatSetValues(P, 1, &gi, 1, &gj, &vals[k], INSERT_VALUES), "P(flux)");
      }
      check(MatRestoreRow(A00, i, &ncols, &cols, &vals), "MatRestoreRow");
    }
    // the cell unknown is piecewise constant, so its L^2 mass is diagonal and
    // its entry is the cell measure
    for (PetscInt i = 0; i < n1; ++i) {
      const auto k = static_cast<std::size_t>(i);
      const double w = k < split_.cell_measure.size() ? split_.cell_measure[k] : 1.0;
      const PetscInt gi = cell[k];
      check(MatSetValues(P, 1, &gi, 1, &gi, &w, INSERT_VALUES), "P(mass)");
    }
    check(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY), "assembly(P)");
    check(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY), "assembly(P)");
    MatDestroy(&A00);

    // the constrained rows, given the row A gives them
    if (!split_.pinned.empty()) {
      check(MatZeroRowsColumns(P, static_cast<PetscInt>(split_.pinned.size()), split_.pinned.data(),
                               1.0, nullptr, nullptr),
            "MatZeroRowsColumns(P)");
      for (std::size_t k = 0; k < split_.pinned.size(); ++k) {
        const PetscInt i = split_.pinned[k];
        const double d = k < split_.pinned_diagonal.size() ? split_.pinned_diagonal[k] : 1.0;
        check(MatSetValues(P, 1, &i, 1, &i, &d, INSERT_VALUES), "P(pinned)");
      }
      check(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY), "assembly(P)");
      check(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY), "assembly(P)");
    }

    check(KSPSetOperators(ksp, M_, P), "KSPSetOperators(A, P)");
    check(PCSetType(pc, PCFIELDSPLIT), "PCSetType(fieldsplit)");
    for (std::size_t b = 0; b < sets.size(); ++b) {
      check(PCFieldSplitSetIS(pc, std::to_string(b).c_str(), sets[b]), "PCFieldSplitSetIS");
    }
    check(PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE), "PCFieldSplitSetType");

    // each factor is inverted exactly: the flux block is SPD, the cell block
    // diagonal. Anything cheaper is a different preconditioner and no longer
    // the Riesz map.
    check(PCSetUp(pc), "PCSetUp(fieldsplit)");
    PetscInt n_split = 0;
    KSP* sub = nullptr;
    check(PCFieldSplitGetSubKSP(pc, &n_split, &sub), "PCFieldSplitGetSubKSP");
    for (PetscInt b = 0; b < n_split; ++b) {
      PC sub_pc = nullptr;
      check(KSPSetType(sub[b], KSPPREONLY), "sub KSPSetType");
      check(KSPGetPC(sub[b], &sub_pc), "sub KSPGetPC");
      check(PCSetType(sub_pc, b == 0 ? PCLU : PCJACOBI), "sub PCSetType");
    }
    PetscFree(sub);

    riesz_.push_back(P);
    for (IS& s : sets) ISDestroy(&s);
  }

  void build_matrix(const SparseSystem& A) {
    const PetscInt n = n_;

    // Count per row first: preallocating is the difference between assembling
    // in seconds and in minutes, and PETSc will not tell you which one you
    // chose.
    std::vector<PetscInt> per_row(A.n, 0);
    for (std::size_t k = 0; k < A.nnz(); ++k) ++per_row[static_cast<std::size_t>(A.row[k])];
    // TRIPLETS, NOT NONZEROS. An assembly emits one triplet per contribution
    // and PETSc sums the duplicates, so a row's triplet count is an upper
    // bound on its nonzero count -- and on a small mesh it can exceed the
    // matrix dimension, which PETSc rejects outright. Clamping asks for the
    // most a row could possibly hold, which is exactly what a preallocation
    // hint is for.
    for (PetscInt& c : per_row) c = std::min(c, n);
    for (PetscInt i = 0; i < n; ++i) ++per_row[static_cast<std::size_t>(i)];  // the diagonal
    for (PetscInt& c : per_row) c = std::min(c, n);

    Mat M = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &M), "MatCreate");
    check(MatSetType(M, MATSEQAIJ), "MatSetType");
    check(MatSetSizes(M, n, n, n, n), "MatSetSizes");
    check(MatSeqAIJSetPreallocation(M, 0, per_row.data()), "preallocate");
    // duplicate triplets are summed, which is what an assembly produces
    check(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption");
    // an explicitly stored zero is structure, not noise, and must survive
    check(MatSetOption(M, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE), "MatSetOption");
    // THE DIAGONAL IS MADE STRUCTURALLY PRESENT, zero or not.
    //
    // The multiplier block of a mixed form has no diagonal entry at all: the
    // (p, p) block is empty, not small. A factorization that indexes the
    // diagonal then refuses the matrix outright -- PETSc's ILU reports
    // "Matrix is missing diagonal entries" -- and a fieldsplit cannot extract
    // a block it cannot address. An explicit zero costs one entry per row and
    // changes no product.
    const double zero = 0.0;
    for (PetscInt i = 0; i < n; ++i) {
      check(MatSetValues(M, 1, &i, 1, &i, &zero, ADD_VALUES), "MatSetValues");
    }
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto i = static_cast<PetscInt>(A.row[k]);
      const auto j = static_cast<PetscInt>(A.col[k]);
      check(MatSetValues(M, 1, &i, 1, &j, &A.value[k], ADD_VALUES), "MatSetValues");
    }
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly");

    M_ = M;
  }

  void build_ksp() {
    KSP ksp = nullptr;
    check(KSPCreate(PETSC_COMM_SELF, &ksp), "KSPCreate");
    check(KSPSetOperators(ksp, M_, M_), "KSPSetOperators");
    check(KSPSetType(ksp, opts_.direct() ? KSPPREONLY : opts_.method.c_str()), "KSPSetType");
    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    // "riesz" is this layer's name, not a PETSc type: it resolves to a
    // fieldsplit whose blocks are the Riesz map, set up below.
    const bool riesz = !opts_.direct() && opts_.preconditioner == "riesz";
    const std::string pc_type = opts_.direct()
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
    if (factorizing && !opts_.factorization.empty() && opts_.factorization != "petsc") {
      check(PCFactorSetMatSolverType(pc, opts_.factorization.c_str()), "PCFactorSetMatSolverType");
    }
    if (!opts_.direct()) {
      check(KSPSetTolerances(ksp, opts_.rtol, opts_.atol, PETSC_DEFAULT,
                             static_cast<PetscInt>(opts_.max_iterations)),
            "KSPSetTolerances");
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
    if (factorizing && opts_.factorization == "mumps") {
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
    const PetscErrorCode e = KSPSolve(ksp, rhs_, sol);
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

      const PetscScalar* v = nullptr;
      check(VecGetArrayRead(sol, &v), "VecGetArrayRead");
      x.assign(v, v + n);
      check(VecRestoreArrayRead(sol, &v), "VecRestoreArrayRead");
      // reported from the SYSTEM, not from the backend's own iteration: a
      // direct solve reports zero iterations and would otherwise say nothing
      out.residual = true_residual(A, b, x);
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
  }

  SolverOptions opts_;
  BlockSplit split_;
  std::vector<Mat> riesz_;  // the diagonal blocks of the Riesz map
  std::string prefix_;
  Mat M_{nullptr};
  KSP ksp_{nullptr};
  Vec rhs_{nullptr}, sol_{nullptr};
  PetscInt n_{0};
  const SparseSystem* bound_{nullptr};
};

}  // namespace mimetika::solver
