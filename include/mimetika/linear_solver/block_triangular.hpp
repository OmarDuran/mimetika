#pragma once

#include <petscksp.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "mimetika/linear_solver/cell_schwarz.hpp"
#include "mimetika/linear_solver/linear.hpp"

// Block upper-triangular preconditioning of the mixed saddle point, under
// right-preconditioned GMRES.
//
// The operator, once every non-stress row is negated, is
//
//     K = [ M    -B^T ]        B the divergence stacked over the asymmetry,
//         [ -B    0   ],
//
// symmetric, and its Schur complement is -S with S = B M^-1 B^T. The
// preconditioner is
//
//     P = [ tau Mt   -B^T ]
//         [ 0        -S~  ],
//
// Mt the cell additive Schwarz of cell_schwarz.hpp and S~ = B Mt^-1 B^T under
// one BoomerAMG V-cycle. With the exact blocks P^-1 K has one eigenvalue and
// GMRES terminates in two steps (Pazner, Kolev & Vassilevski, Remark 2); with
// these it is what was measured, against block-diagonal MINRES on the same
// blocks at its best scaling:
//
//                           MINRES, tau = 8   this, tau = 2, restart 50
//     simplex box 8^3            120                   81
//     stretched annulus 6        357                  264
//
// and 242 and a failure to converge for the diagonal-M route this replaces.
//
// TAU = 2. Additive Schwarz sums the inverses of overlapping cells, and an
// interior facet belongs to two, so Mt^-1 is too large relative to M^-1 by
// about the overlap; tau rescales the (1, 1) block against the Schur block.
// Measured over tau in {0.5, 1, 2, 4, 8}: 2 is optimal on regular tetrahedra
// (81; 91 at tau = 1) and within five percent of optimal on stretched ones,
// where 1 is marginally better.
//
// RESTART 50. The Krylov basis is (restart + 1) vectors of the system's size,
// 1.3 GB at 3.2 million unknowns. Against no restart it costs one percent more
// iterations on regular cells and twelve on stretched ones; 30 costs five and
// twenty-two, 100 nothing and five.
//
// B^T IS SPARSE, and that is the condition for any of this to pay: through the
// kernels it would be a second operator apply an iteration, doubling the cost
// of each and returning the iterations saved.
//
// THE RESIDUAL REPORTED IS THE TRUE ONE, recomputed from the answer with one
// operator apply. Right preconditioning makes GMRES's own estimate the
// unpreconditioned residual, but that estimate drifts over a long run; and a
// solver comparison read off preconditioned residuals is the trap minres.hpp
// describes.

namespace mimetika::solver {

struct BlockTriangularOptions {
  double tau{2.0};
  int restart{50};
  double rtol{1e-10};
  double atol{1e-50};
  int max_iterations{1000};
};

struct BlockTriangularReport {
  bool converged{false};
  int iterations{0};
  double residual{0.0};  // ||b - K x||_2 / ||b||_2, recomputed from x
  std::string reason;
  double setup_seconds{0.0};  // the AMG hierarchy on S~
  double solve_seconds{0.0};
};

namespace block_triangular_detail {

inline void check(PetscErrorCode ierr, const char* what) {
  if (ierr != 0) {
    throw std::runtime_error(std::string("block_triangular_gmres: PETSc failed in ") + what);
  }
}

using ApplyK = std::function<void(const std::vector<double>&, std::vector<double>&)>;

struct Context {
  ApplyK apply_k;
  const std::vector<Index>* stress{nullptr};
  const std::vector<Index>* rest{nullptr};
  const CellSchwarz* m11{nullptr};
  const SparseColumns* b{nullptr};
  double tau{2.0};
  KSP schur{nullptr};
  Vec schur_in{nullptr}, schur_out{nullptr};
  std::vector<double> v, y, rs, ys;  // scratch, allocated once
  std::string error;                 // a C++ exception, carried across the C callbacks
};

// PETSc objects, destroyed however the solve leaves
struct Owned {
  Mat S{nullptr}, K{nullptr};
  KSP outer{nullptr}, inner{nullptr};
  Vec b{nullptr}, x{nullptr}, in{nullptr}, out{nullptr};
  ~Owned() {
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&in);
    VecDestroy(&out);
    KSPDestroy(&outer);
    KSPDestroy(&inner);
    MatDestroy(&K);
    MatDestroy(&S);
  }
};

inline PetscErrorCode mult(Mat A, Vec x, Vec out) {
  Context* c = nullptr;
  PetscCall(MatShellGetContext(A, &c));
  const PetscScalar* xa;
  PetscCall(VecGetArrayRead(x, &xa));
  std::copy(xa, xa + c->v.size(), c->v.begin());
  PetscCall(VecRestoreArrayRead(x, &xa));
  try {
    c->apply_k(c->v, c->y);
  } catch (const std::exception& ex) {
    c->error = ex.what();
    return PETSC_ERR_LIB;
  }
  PetscScalar* oa;
  PetscCall(VecGetArray(out, &oa));
  std::copy(c->y.begin(), c->y.end(), oa);
  PetscCall(VecRestoreArray(out, &oa));
  return PETSC_SUCCESS;
}

// z = P^-1 r:  z_r = -S~^-1 r_r,  z_s = (tau Mt)^-1 (r_s - B^T S~^-1 r_r)
inline PetscErrorCode precondition(PC pc, Vec r, Vec z) {
  Context* c = nullptr;
  PetscCall(PCShellGetContext(pc, &c));
  const auto& stress = *c->stress;
  const auto& rest = *c->rest;
  const PetscScalar* ra;
  PetscCall(VecGetArrayRead(r, &ra));
  PetscScalar* si;
  PetscCall(VecGetArray(c->schur_in, &si));
  for (std::size_t k = 0; k < rest.size(); ++k) si[k] = ra[rest[k]];
  PetscCall(VecRestoreArray(c->schur_in, &si));
  for (std::size_t k = 0; k < stress.size(); ++k) c->rs[k] = ra[stress[k]];
  PetscCall(VecRestoreArrayRead(r, &ra));

  PetscCall(KSPSolve(c->schur, c->schur_in, c->schur_out));

  const PetscScalar* so;
  PetscCall(VecGetArrayRead(c->schur_out, &so));
  c->b->subtract_transpose(so, c->rs.data());
  c->m11->apply(c->rs.data(), c->ys.data(), c->tau);
  PetscScalar* za;
  PetscCall(VecGetArray(z, &za));
  for (std::size_t k = 0; k < rest.size(); ++k) za[rest[k]] = -so[k];
  for (std::size_t k = 0; k < stress.size(); ++k) za[stress[k]] = c->ys[k];
  PetscCall(VecRestoreArray(z, &za));
  PetscCall(VecRestoreArrayRead(c->schur_out, &so));
  return PETSC_SUCCESS;
}

}  // namespace block_triangular_detail

// K x = rhs by right-preconditioned GMRES. `apply_k` is the symmetrized operator
// (negated non-stress rows, pinned columns eliminated); `stress` and `rest` are
// the global indices of the two blocks, in their local numberings' order; the
// Schur CSR is taken by PETSc WITHOUT A COPY and must outlive the call, which it
// does by being borrowed here. x starts from zero.
inline BlockTriangularReport block_triangular_gmres(
    block_triangular_detail::ApplyK apply_k, const std::vector<Index>& stress,
    const std::vector<Index>& rest, const CellSchwarz& m11, const SparseColumns& b, Csr& schur,
    const std::vector<double>& rhs, std::vector<double>& x, const BlockTriangularOptions& opt) {
  using namespace block_triangular_detail;
  using clock = std::chrono::steady_clock;
  const std::size_t n = rhs.size();
  if (stress.size() + rest.size() != n || m11.n_stress != stress.size() ||
      static_cast<std::size_t>(schur.n) != rest.size()) {
    throw std::invalid_argument("block_triangular_gmres: the blocks do not partition the system");
  }

  BlockTriangularReport rep;
  Owned own;
  Context ctx;
  ctx.apply_k = std::move(apply_k);
  ctx.stress = &stress;
  ctx.rest = &rest;
  ctx.m11 = &m11;
  ctx.b = &b;
  ctx.tau = opt.tau;
  ctx.v.assign(n, 0.0);
  ctx.y.assign(n, 0.0);
  ctx.rs.assign(stress.size(), 0.0);
  ctx.ys.assign(stress.size(), 0.0);

  // S~ as AIJ over the CSR arrays themselves when the index widths agree
  std::vector<PetscInt> ptr_copy, col_copy;
  PetscInt* ptr = nullptr;
  PetscInt* col = nullptr;
  if (std::is_same_v<PetscInt, int>) {
    ptr = schur.ptr.data();
    col = schur.col.data();
  } else {
    ptr_copy.assign(schur.ptr.begin(), schur.ptr.end());
    col_copy.assign(schur.col.begin(), schur.col.end());
    ptr = ptr_copy.data();
    col = col_copy.data();
  }
  const auto t_setup = clock::now();
  check(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, schur.n, schur.n, ptr, col, schur.val.data(), &own.S),
        "MatCreateSeqAIJWithArrays");

  // one V-cycle on S~, set up once: the same operator returns every iteration
  check(KSPCreate(PETSC_COMM_SELF, &own.inner), "KSPCreate(schur)");
  check(KSPSetOptionsPrefix(own.inner, "mechanics_schur_"), "KSPSetOptionsPrefix(schur)");
  check(KSPSetOperators(own.inner, own.S, own.S), "KSPSetOperators(schur)");
  check(KSPSetType(own.inner, KSPPREONLY), "KSPSetType(schur)");
  PC inner_pc = nullptr;
  check(KSPGetPC(own.inner, &inner_pc), "KSPGetPC(schur)");
  check(PCSetType(inner_pc, PCHYPRE), "PCSetType(hypre) -- is PETSc built with hypre?");
  check(PCHYPRESetType(inner_pc, "boomeramg"), "PCHYPRESetType(boomeramg)");
  check(KSPSetFromOptions(own.inner), "KSPSetFromOptions(schur)");
  check(MatCreateVecs(own.S, &own.out, &own.in), "MatCreateVecs(schur)");
  check(KSPSetUp(own.inner), "KSPSetUp(schur)");
  ctx.schur = own.inner;
  ctx.schur_in = own.in;
  ctx.schur_out = own.out;
  rep.setup_seconds = std::chrono::duration<double>(clock::now() - t_setup).count();

  const auto nn = static_cast<PetscInt>(n);
  check(MatCreateShell(PETSC_COMM_SELF, nn, nn, nn, nn, &ctx, &own.K), "MatCreateShell");
  check(MatShellSetOperation(own.K, MATOP_MULT, (void (*)(void))mult), "MatShellSetOperation");
  check(KSPCreate(PETSC_COMM_SELF, &own.outer), "KSPCreate");
  check(KSPSetOptionsPrefix(own.outer, "mechanics_mf_"), "KSPSetOptionsPrefix");
  check(KSPSetOperators(own.outer, own.K, own.K), "KSPSetOperators");
  check(KSPSetType(own.outer, KSPGMRES), "KSPSetType(gmres)");
  check(KSPGMRESSetRestart(own.outer, opt.restart), "KSPGMRESSetRestart");
  check(KSPSetPCSide(own.outer, PC_RIGHT), "KSPSetPCSide");
  check(KSPSetTolerances(own.outer, opt.rtol, opt.atol, PETSC_DEFAULT, opt.max_iterations),
        "KSPSetTolerances");
  PC pc = nullptr;
  check(KSPGetPC(own.outer, &pc), "KSPGetPC");
  check(PCSetType(pc, PCSHELL), "PCSetType(shell)");
  check(PCShellSetContext(pc, &ctx), "PCShellSetContext");
  check(PCShellSetApply(pc, precondition), "PCShellSetApply");
  check(KSPSetFromOptions(own.outer), "KSPSetFromOptions");

  check(MatCreateVecs(own.K, &own.x, &own.b), "MatCreateVecs");
  PetscScalar* ba;
  check(VecGetArray(own.b, &ba), "VecGetArray(b)");
  std::copy(rhs.begin(), rhs.end(), ba);
  check(VecRestoreArray(own.b, &ba), "VecRestoreArray(b)");
  check(VecSet(own.x, 0.0), "VecSet(x)");

  const auto t_solve = clock::now();
  const PetscErrorCode ierr = KSPSolve(own.outer, own.b, own.x);
  rep.solve_seconds = std::chrono::duration<double>(clock::now() - t_solve).count();
  if (!ctx.error.empty()) throw std::runtime_error("block_triangular_gmres: " + ctx.error);
  check(ierr, "KSPSolve");

  PetscInt its = 0;
  check(KSPGetIterationNumber(own.outer, &its), "KSPGetIterationNumber");
  KSPConvergedReason why;
  check(KSPGetConvergedReason(own.outer, &why), "KSPGetConvergedReason");
  const char* why_text = nullptr;
  check(KSPGetConvergedReasonString(own.outer, &why_text), "KSPGetConvergedReasonString");
  rep.iterations = static_cast<int>(its);
  rep.converged = why > 0;
  rep.reason = why_text != nullptr ? why_text : "";

  const PetscScalar* xa;
  check(VecGetArrayRead(own.x, &xa), "VecGetArrayRead(x)");
  x.assign(xa, xa + n);
  check(VecRestoreArrayRead(own.x, &xa), "VecRestoreArrayRead(x)");

  // the true residual, from the answer
  ctx.apply_k(x, ctx.y);
  double rr = 0.0, bb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = rhs[i] - ctx.y[i];
    rr += d * d;
    bb += rhs[i] * rhs[i];
  }
  rep.residual = bb > 0.0 ? std::sqrt(rr / bb) : std::sqrt(rr);
  return rep;
}

}  // namespace mimetika::solver
