#pragma once

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

  SolveReport solve(const SparseSystem& A, const std::vector<double>& b,
                    std::vector<double>& x) override {
    if (b.size() != A.n) throw std::invalid_argument("PetscSolver: right-hand side size");
    const auto n = static_cast<PetscInt>(A.n);

    // Count per row first: preallocating is the difference between assembling
    // in seconds and in minutes, and PETSc will not tell you which one you
    // chose.
    std::vector<PetscInt> per_row(A.n, 0);
    for (std::size_t k = 0; k < A.nnz(); ++k) ++per_row[static_cast<std::size_t>(A.row[k])];

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

    Vec rhs = nullptr, sol = nullptr;
    check(VecCreateSeq(PETSC_COMM_SELF, n, &rhs), "VecCreate");
    check(VecDuplicate(rhs, &sol), "VecDuplicate");
    for (PetscInt i = 0; i < n; ++i) {
      check(VecSetValue(rhs, i, b[static_cast<std::size_t>(i)], INSERT_VALUES), "VecSetValue");
    }
    check(VecAssemblyBegin(rhs), "VecAssembly");
    check(VecAssemblyEnd(rhs), "VecAssembly");

    KSP ksp = nullptr;
    check(KSPCreate(PETSC_COMM_SELF, &ksp), "KSPCreate");
    check(KSPSetOperators(ksp, M, M), "KSPSetOperators");
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

    SolveReport out;
    const PetscErrorCode e = KSPSolve(ksp, rhs, sol);
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

    KSPDestroy(&ksp);
    VecDestroy(&sol);
    VecDestroy(&rhs);
    MatDestroy(&M);
    if (e != 0) throw std::runtime_error("PetscSolver: " + out.reason);
    return out;
  }

 private:
  std::string factor_, prefix_;
};

}  // namespace mimetika::solver
