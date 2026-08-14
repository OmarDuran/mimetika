#pragma once

#include <algorithm>
#include <petscksp.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/solver/linear.hpp"

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

class PetscSolver final : public LinearSolver {
 public:
  // `type` selects the factorization package; MUMPS is the default because
  // these systems are indefinite. An empty prefix means the KSP also reads
  // command-line options, so an iterative method can be selected at run time.
  explicit PetscSolver(std::string factorization = "mumps", std::string prefix = "")
      : factor_(std::move(factorization)), prefix_(std::move(prefix)) {
    PetscSession::instance();
  }

  std::string name() const override { return "petsc/" + factor_; }

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

    Mat M = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &M), "MatCreate");
    check(MatSetType(M, MATSEQAIJ), "MatSetType");
    check(MatSetSizes(M, n, n, n, n), "MatSetSizes");
    check(MatSeqAIJSetPreallocation(M, 0, per_row.data()), "preallocate");
    // duplicate triplets are summed, which is what an assembly produces
    check(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption");
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
    check(KSPSetType(ksp, KSPPREONLY), "KSPSetType");  // the direct path
    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    check(PCSetType(pc, PCLU), "PCSetType");
    if (!factor_.empty() && factor_ != "petsc") {
      check(PCFactorSetMatSolverType(pc, factor_.c_str()), "PCFactorSetMatSolverType");
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
    if (M_ != nullptr) MatDestroy(&M_);
    ksp_ = nullptr;
    sol_ = rhs_ = nullptr;
    M_ = nullptr;
    bound_ = nullptr;
  }

  std::string factor_, prefix_;
  Mat M_{nullptr};
  KSP ksp_{nullptr};
  Vec rhs_{nullptr}, sol_{nullptr};
  PetscInt n_{0};
  const SparseSystem* bound_{nullptr};
};

}  // namespace mimetika::solver
