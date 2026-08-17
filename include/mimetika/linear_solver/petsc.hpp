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
  // first-factor unknowns above which the exact solve is refused
  int riesz_exact_limit{50000};
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

  // The factors of the product space. Required by the "riesz" preconditioner
  // and ignored by every other one.
  void set_norm(SpaceNorm s) { norm_ = std::move(s); }

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
    bound_ = &A;  // build_riesz reads the triplets, so bind before the KSP
    const auto t0 = std::chrono::steady_clock::now();
    build_matrix(A);
    const auto t1 = std::chrono::steady_clock::now();
    build_ksp();
    check(VecCreateSeq(PETSC_COMM_SELF, n_, &rhs_), "VecCreate");
    check(VecDuplicate(rhs_, &sol_), "VecDuplicate");
    // force the factorization now rather than on the first solve, so that the
    // cost shows up where it is paid
    check(KSPSetUp(ksp_), "KSPSetUp");
    const auto t2 = std::chrono::steady_clock::now();
    matrix_seconds_ = std::chrono::duration<double>(t1 - t0).count();
    preconditioner_seconds_ = std::chrono::duration<double>(t2 - t1).count();
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
    SolveReport r = solve(b, x);
    r.matrix_seconds = matrix_seconds_;
    r.preconditioner_seconds = preconditioner_seconds_;
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
    std::vector<IS> sets(nf, nullptr);
    for (std::size_t b = 0; b < nf; ++b) {
      const auto& idx = norm_.factors[b];
      const auto m = static_cast<PetscInt>(idx.size());
      bool contiguous = !idx.empty();
      for (std::size_t k = 1; k < idx.size() && contiguous; ++k) {
        contiguous = idx[k] == idx[k - 1] + 1;
      }
      if (contiguous) {
        check(ISCreateStride(PETSC_COMM_SELF, m, idx.front(), 1, &sets[b]), "ISCreateStride");
      } else {
        check(ISCreateGeneral(PETSC_COMM_SELF, m, idx.data(), PETSC_COPY_VALUES, &sets[b]),
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
    // constraint row against every other column of that row
    std::vector<PetscInt> per_row(static_cast<std::size_t>(n_), 0);
    for (const Index r : p_row) ++per_row[static_cast<std::size_t>(r)];
    for (std::size_t r = 0; r < static_cast<std::size_t>(n_); ++r) {
      const auto b = static_cast<std::size_t>(b_begin[r]);
      const auto e = static_cast<std::size_t>(b_begin[r + 1]);
      for (std::size_t a = b; a < e; ++a) {
        per_row[static_cast<std::size_t>(b_col[a])] += static_cast<PetscInt>(e - b);
      }
    }
    for (PetscInt& c : per_row) c = std::min(c, n_);

    Mat P = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &P), "MatCreate(P)");
    check(MatSetType(P, MATSEQAIJ), "MatSetType(P)");
    check(MatSetSizes(P, n_, n_, n_, n_), "MatSetSizes(P)");
    check(MatSeqAIJSetPreallocation(P, 0, per_row.data()), "preallocate(P)");
    check(MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption(P)");
    check(MatSetOption(P, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE), "MatSetOption(P)");
    per_row = std::vector<PetscInt>();

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
          if (!is_pinned[static_cast<std::size_t>(b_col[a])]) cols.push_back(b_col[a]);
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
        check(MatSetValues(P, static_cast<PetscInt>(m), cols.data(), static_cast<PetscInt>(m),
                           cols.data(), blk.data(), ADD_VALUES),
              "P(graph block)");
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
    const int block_its = opts_.riesz_block_its > 0 ? opts_.riesz_block_its : 200;
    const std::string b0_pc =
        !opts_.riesz_block_pc.empty() ? opts_.riesz_block_pc : (inexact_block ? "icc" : "lu");
    // an inner Krylov makes the preconditioner a VARYING operator, which only a
    // flexible outer method may use; applying it under plain gmres is a silent
    // wrong answer, so the promotion happens here rather than in the caller
    if (inexact_block) check(KSPSetType(ksp, KSPFGMRES), "KSPSetType(fgmres)");

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
    if (inexact_block) {
      set(p0 + "ksp_type", "cg");
      set(p0 + "ksp_max_it", std::to_string(block_its));
      set(p0 + "ksp_rtol", std::to_string(opts_.riesz_block_rtol));
      set(p0 + "pc_type", b0_pc);
    } else {
      set(p0 + "ksp_type", "preonly");
      set(p0 + "pc_type", b0_pc);
      if ((b0_pc == "lu" || b0_pc == "cholesky") && !opts_.factorization.empty() &&
          opts_.factorization != "petsc") {
        set(p0 + "pc_factor_mat_solver_type", opts_.factorization);
      }
    }
    // the L2 factors are DIAGONAL, so Jacobi inverts them exactly and anything
    // heavier is wasted
    set(p1 + "ksp_type", "preonly");
    set(p1 + "pc_type", "jacobi");

    riesz_.push_back(P);
    for (IS& s : sets) ISDestroy(&s);
  }

  // A MATRIX FROM TRIPLETS, one call per row rather than one per entry.
  //
  // Handing PETSc one triplet at a time costs a search of the row for every
  // entry, and an assembly of tens of thousands of polyhedra emits of order
  // 10^8 of them. Grouping by row first -- a counting sort, one pass to count
  // and one to place -- turns that into one call per row, and PETSc sums the
  // duplicates inside each call as before. The scratch is scoped so it is
  // returned before anything else is allocated.
  static Mat assemble(const std::vector<Index>& row, const std::vector<Index>& col,
                      const std::vector<double>& val, PetscInt n) {
    const auto rows = static_cast<std::size_t>(n);
    const std::size_t nnz = val.size();

    std::vector<PetscInt> begin(rows + 1, 0);
    for (std::size_t k = 0; k < nnz; ++k) ++begin[static_cast<std::size_t>(row[k]) + 1];
    std::vector<PetscInt> per_row(rows, 0);
    for (std::size_t i = 0; i < rows; ++i) {
      // TRIPLETS, NOT NONZEROS: a row's triplet count is an upper bound on its
      // nonzero count, and on a small mesh it can exceed the dimension, which
      // PETSc rejects outright
      per_row[i] = std::min(begin[i + 1], n);
      begin[i + 1] += begin[i];
    }

    Mat M = nullptr;
    check(MatCreate(PETSC_COMM_SELF, &M), "MatCreate");
    check(MatSetType(M, MATSEQAIJ), "MatSetType");
    check(MatSetSizes(M, n, n, n, n), "MatSetSizes");
    check(MatSeqAIJSetPreallocation(M, 0, per_row.data()), "preallocate");
    check(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption");
    // an explicitly stored zero is structure, not noise, and must survive
    check(MatSetOption(M, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE), "MatSetOption");
    per_row = std::vector<PetscInt>();
    scatter_by_row(M, row, col, val, n);
    check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "assembly");
    check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "assembly");
    return M;
  }

  // The scatter alone, for a matrix the caller preallocated itself -- the
  // preconditioner adds terms that never become triplets, so it has to own the
  // preallocation.
  static void scatter_by_row(Mat M, const std::vector<Index>& row, const std::vector<Index>& col,
                             const std::vector<double>& val, PetscInt n) {
    const auto rows = static_cast<std::size_t>(n);
    const std::size_t nnz = val.size();
    std::vector<PetscInt> begin(rows + 1, 0);
    for (std::size_t k = 0; k < nnz; ++k) ++begin[static_cast<std::size_t>(row[k]) + 1];
    for (std::size_t i = 0; i < rows; ++i) begin[i + 1] += begin[i];

    std::vector<PetscInt> at(begin.begin(), begin.end() - 1);
    std::vector<PetscInt> cols(nnz);
    std::vector<PetscScalar> vals(nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
      const auto slot = static_cast<std::size_t>(at[static_cast<std::size_t>(row[k])]++);
      cols[slot] = static_cast<PetscInt>(col[k]);
      vals[slot] = val[k];
    }
    at = std::vector<PetscInt>();
    for (PetscInt i = 0; i < n; ++i) {
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
    check(KSPCreate(PETSC_COMM_SELF, &ksp), "KSPCreate");
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
    if (factorizing && !opts_.factorization.empty() && opts_.factorization != "petsc") {
      check(PCFactorSetMatSolverType(pc, opts_.factorization.c_str()), "PCFactorSetMatSolverType");
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
    const auto t0 = std::chrono::steady_clock::now();
    const PetscErrorCode e = KSPSolve(ksp, rhs_, sol);
    out.solve_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
  double matrix_seconds_{0.0}, preconditioner_seconds_{0.0};
  SpaceNorm norm_;
  std::vector<Mat> riesz_;  // the diagonal blocks of the Riesz map
  std::string prefix_;
  Mat M_{nullptr};
  KSP ksp_{nullptr};
  Vec rhs_{nullptr}, sol_{nullptr};
  PetscInt n_{0};
  const SparseSystem* bound_{nullptr};
};

}  // namespace mimetika::solver
