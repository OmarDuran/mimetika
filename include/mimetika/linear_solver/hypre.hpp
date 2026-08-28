#pragma once

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "mimetika/linear_solver/linear.hpp"
#include "mimetika/linear_solver/space_norm.hpp"

// The Riesz map with its first block handed to hypre DIRECTLY.
//
// THIS HEADER MUST NOT MEET PETSc IN ONE BINARY. PETSc links its own libHYPRE
// and both copies export HYPRE_ADSCreate and the rest; which one a call
// reaches is then decided by load order. A target that includes this one links
// mimetika_hypre and not mimetika_petsc.
//
// What the direct path buys is the part of hypre PETSc does not forward. PETSc
// registers -pc_hypre_ads_amg_theta, -pc_hypre_ads_ams_theta and
// -pc_hypre_ads_ams_cycle_type and never queries them -- each is accepted,
// reported by -options_left as unused, and changes no count -- so the strength
// threshold of the auxiliary hierarchies, the one parameter a jumping
// coefficient asks to change, is unreachable through PCHYPRE. Here it is
// HYPRE_ADSSetAMGOptions and HYPRE_ADSSetAMSOptions.
//
// The preconditioner is the same one the PETSc path builds, and deliberately
// so -- the comparison between the two is then the library and not the method:
//
//     P = diag( M + B^T W^-1 B , W )
//
// with M the assembled (0,0) block, B the differential and W the L2 weight of
// each factor after the first. ADS inverts the first block; the rest is a
// diagonal scaling. The outer method is FlexGMRES because an inner ADS cycle
// makes P a varying operator, which plain GMRES may not use.
//
// ADS is written for ONE unknown per facet in 3D -- derham_rt, stabilized_rt,
// and the eta = 1 cells of adaptive_rt. A facet carrying d moments needs the
// facet-constant subspace of the PETSc path, which is not built here: solve()
// refuses it rather than handing hypre a curl whose rows are not the block's.

namespace mimetika::solver {

// One MPI and one hypre initialization for the process, torn down at exit.
class HypreSession {
 public:
  // Idempotent, and callable before anything else. The model's own build()
  // partitions, which asks MPI for the communicator size, so MPI has to be up
  // before the model is built -- the PETSc path gets that from PetscInitialize
  // inside its solve(), and this path has to ask for it.
  static void ensure() {
    static HypreSession session;
    (void)session;
  }

 private:
  HypreSession() {
    int ready = 0;
    MPI_Initialized(&ready);
    if (!ready) {
      int argc = 0;
      char** argv = nullptr;
      MPI_Init(&argc, &argv);
      owns_mpi_ = true;
    }
    HYPRE_Initialize();
  }
  ~HypreSession() {
    HYPRE_Finalize();
    if (owns_mpi_) MPI_Finalize();
  }
  bool owns_mpi_{false};
};

struct HypreOptions {
  double rtol{1e-8};
  int max_iterations{500};
  // The ADS cycle. 11-14 apply three scalar AMG solves in place of one
  // monolithic vector solve; 13 is the 5-level multiplicative (034515430)
  // that measured fastest per application on the hybrid mesh.
  int ads_cycle_type{13};
  // The auxiliary hierarchies' strength thresholds -- what PETSc cannot set.
  // hypre's own defaults are 0.25 for both.
  double amg_theta{0.25};
  double ams_theta{0.25};
  int amg_coarsen_type{10};
  int amg_agg_levels{1};
  int amg_relax_type{3};
  // hypre's own ADS defaults are 10, 1, 3, 0.25, 0, 0. interp_type and Pmax
  // are the two that matter under a jumping coefficient: 6 (extended+i) with
  // Pmax = 4 truncates the interpolation to four entries a row, and on a 1e6
  // checkerboard at 16^3 that is the difference between converging and not.
  int amg_interp_type{0};
  int amg_pmax{0};
  int print_level{0};
  // ONE CYCLE BY DEFAULT. An inner CG is the option, not the rule.
  //
  // One ADS cycle is what the Riesz theory asks for, and on a well-conditioned
  // problem it is also much the cheaper: on a 93k-cell industrial mesh at
  // rtol 1e-5, 315k unknowns,
  //
  //     block 0    (one cycle)      81 iterations   38 s
  //     block 20   to 1e-2           9 iterations   48 s
  //     block 50   to 1e-2           9 iterations   74 s
  //     block 200  to 1e-6           8 iterations  116 s and worse
  //
  // so solving the block buys a tenth of the iterations at three times the
  // cost. What it is FOR is a jumping coefficient: at 16^3 on a checkerboard
  // K one cycle takes 16, 78, and then fails to converge at 1e4 and 1e6, and
  // an inner CG returns it to 8, 8, 8, 9. PETSc's single cycle fails there
  // too, so that is the cycle's own limit rather than a defect of either
  // wiring -- and it is why PETSc offers `ads` and `ads-cg` as two choices
  // instead of one compromise. This is `ads`; set block_iterations for the
  // other.
  //
  // The outer method is flexible either way, so an inner Krylov is always
  // admissible here.
  int block_iterations{0};
  double block_rtol{1e-2};
  // extra ADS cycles per application, inside whatever the above does
  int ads_iterations{1};
};

struct HypreReport {
  bool converged{false};
  int iterations{0};
  double residual{0.0};
  double setup_seconds{0.0};
  double solve_seconds{0.0};
  std::string reason;
};

class HypreSolver {
 public:
  using Options = HypreOptions;
  using Report = HypreReport;

  // Solve A x = b with the Riesz map above. `norm` supplies the factors, the
  // L2 weights and the complex's two boundary operators.
  Report solve(const SparseSystem& A, const std::vector<double>& b, std::vector<double>& x,
               const SpaceNorm& norm, const Options& opts = Options{}) {
    HypreSession::ensure();
    // SERIAL ONLY, AND IT SAYS SO RATHER THAN HANGING.
    //
    // Every IJMatrix here is created over the whole index range, [0, n) on the
    // communicator, so on more than one rank each process claims every row and
    // the assembly deadlocks -- which is what an 8-way run looked like. The
    // block preconditioner is serial for the same reason: apply() indexes the
    // local vector data directly, which is the whole vector only on one rank.
    //
    // Making this distributed means partitioning the three spaces the way
    // PetscSolver's layout_of does, renumbering the injection into it, and
    // gathering the flux block across ranks in apply(). Until then the PETSc
    // path is the parallel one.
    {
      int size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (size > 1) {
        throw std::invalid_argument(
            "HypreSolver: the direct hypre path is serial; this run has " +
            std::to_string(size) +
            " ranks. Use --solver ads or ads-cg for a distributed run, or run this one "
            "on a single process.");
      }
    }
    if (norm.empty()) throw std::invalid_argument("HypreSolver: the norm has no factors");
    if (norm.discrete_gradient.empty() || norm.discrete_curl.empty()) {
      throw std::invalid_argument("HypreSolver: ADS needs the discrete gradient and curl");
    }
    if (!norm.lowest_order.empty()) {
      throw std::invalid_argument(
          "HypreSolver: this facet carries more than one moment; ADS reaches it only "
          "through the facet-constant subspace, which the direct path does not build");
    }
    const auto n = static_cast<HYPRE_BigInt>(A.n);
    const std::vector<int>& flux = norm.factors[0];
    const auto nf = static_cast<HYPRE_BigInt>(flux.size());
    if (static_cast<HYPRE_BigInt>(norm.discrete_curl.rows) != nf) {
      throw std::invalid_argument("HypreSolver: the curl's rows are not the first factor");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // where each global unknown sits in the flux block, or -1
    std::vector<int> in_flux(A.n, -1);
    for (std::size_t i = 0; i < flux.size(); ++i) {
      in_flux[static_cast<std::size_t>(flux[i])] = static_cast<int>(i);
    }
    // which factor each unknown belongs to, and whether it carries the graph
    // term. A linear scan per entry would be quadratic, so this is a table.
    std::vector<int> factor_of(A.n, -1);
    for (std::size_t f = 0; f < norm.factors.size(); ++f) {
      for (const int i : norm.factors[f]) factor_of[static_cast<std::size_t>(i)] = static_cast<int>(f);
    }
    // A CONSTRAINED UNKNOWN IS NOT IN THE SPACE.
    //
    // Its row of A is the constraint, scale * e_i^T, and not a form. Leaving
    // the norm's entries on it preconditions an equation that is not the one
    // being solved; P carries the same row instead, so the unknown contributes
    // the identity to P^-1 A and drops out of the Krylov space. Omitting this
    // is what made the direct path lose contrast robustness -- 16, 75, 1570,
    // diverged over K = 1 .. 1e6 against the PETSc path's 16, 14, 26, 29.
    std::vector<char> is_pinned(A.n, 0);
    for (const int i : norm.pinned) is_pinned[static_cast<std::size_t>(i)] = 1;
    // 1/W for every unpinned unknown outside the first factor
    std::vector<double> inv_w(A.n, 0.0);
    for (std::size_t f = 1; f < norm.factors.size(); ++f) {
      const auto& idx = norm.factors[f];
      const auto& w = norm.l2_weight[f - 1];
      for (std::size_t k = 0; k < idx.size(); ++k) {
        const auto i = static_cast<std::size_t>(idx[k]);
        if (is_pinned[i]) continue;
        inv_w[i] = w[k] != 0.0 ? 1.0 / w[k] : 0.0;
      }
    }
    // the constrained rows carry their own diagonal, in both blocks
    for (std::size_t k = 0; k < norm.pinned.size(); ++k) {
      const auto i = static_cast<std::size_t>(norm.pinned[k]);
      const double d =
          k < norm.pinned_diagonal.size() ? norm.pinned_diagonal[k] : 1.0;
      if (in_flux[i] < 0) inv_w[i] = d != 0.0 ? 1.0 / d : 0.0;
    }

    // A0 = M + sum_f B_f^T diag(1/W_f) B_f, in the flux numbering.
    //
    // B is read off A itself: a row outside the first factor with a column
    // inside it IS the differential, so the graph term needs no second
    // assembly. Rows are gathered first because the product pairs every two
    // entries of the same row.
    std::vector<std::vector<std::pair<int, double>>> b_rows(
        static_cast<std::size_t>(A.n));
    Triplets m;
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto r = static_cast<std::size_t>(A.row[k]);
      const auto c = static_cast<std::size_t>(A.col[k]);
      if (is_pinned[r] || is_pinned[c]) continue;  // not in the space
      const int fr = in_flux[r], fc = in_flux[c];
      if (fr >= 0 && fc >= 0) {
        m.add(fr, fc, A.value[k]);
      } else if (fr < 0 && fc >= 0 && factor_of[r] >= 1 &&
                 norm.carries_graph_term(static_cast<std::size_t>(factor_of[r]))) {
        b_rows[r].emplace_back(fc, A.value[k]);
      }
    }
    for (std::size_t r = 0; r < b_rows.size(); ++r) {
      const double s = inv_w[r];
      if (s == 0.0) continue;
      for (const auto& [i, vi] : b_rows[r]) {
        for (const auto& [j, vj] : b_rows[r]) m.add(i, j, s * vi * vj);
      }
    }
    // and the constrained flux unknowns, with the diagonal A gave them
    for (std::size_t k = 0; k < norm.pinned.size(); ++k) {
      const auto i = static_cast<std::size_t>(norm.pinned[k]);
      if (in_flux[i] < 0) continue;
      m.add(in_flux[i], in_flux[i],
            k < norm.pinned_diagonal.size() ? norm.pinned_diagonal[k] : 1.0);
    }

    Mat a_full = to_mat(A, n);
    Mat a0 = to_mat(m, nf);
    Mat grad = to_mat(norm.discrete_gradient);
    Mat curl = to_mat(norm.discrete_curl);

    // the metric, and the only metric ADS is told
    const auto nv = static_cast<HYPRE_BigInt>(norm.discrete_gradient.cols);
    if (norm.vertex_coordinates.size() !=
        static_cast<std::size_t>(nv) * static_cast<std::size_t>(norm.space_dim)) {
      throw std::invalid_argument("HypreSolver: one coordinate per vertex is required");
    }
    // THE COORDINATES ARE SHIFTED TO THE ORIGIN.
    //
    // ADS builds its vector interpolation from the coordinate functions, so
    // what it needs from them is the LINEAR part. A mesh written in a projected
    // coordinate system carries an offset that dwarfs the domain -- an
    // industrial mesh here spans 1e4 metres about an origin 5.3e5 away, a ratio
    // of 594 -- and the linear part is then a rounding error on the constant.
    // Subtracting the minimum changes the span of {1, x, y, z} not at all and
    // restores the precision.
    Vecs xyz;
    std::vector<double> origin(3, 0.0);
    for (int d = 0; d < norm.space_dim && d < 3; ++d) {
      double lo = std::numeric_limits<double>::infinity();
      for (HYPRE_BigInt v = 0; v < nv; ++v) {
        lo = std::min(lo, norm.vertex_coordinates[static_cast<std::size_t>(v) *
                                                      static_cast<std::size_t>(norm.space_dim) +
                                                  static_cast<std::size_t>(d)]);
      }
      origin[static_cast<std::size_t>(d)] = std::isfinite(lo) ? lo : 0.0;
    }
    for (int d = 0; d < 3; ++d) {
      std::vector<double> c(static_cast<std::size_t>(nv), 0.0);
      if (d < norm.space_dim) {
        for (HYPRE_BigInt v = 0; v < nv; ++v) {
          c[static_cast<std::size_t>(v)] =
              norm.vertex_coordinates[static_cast<std::size_t>(v) *
                                          static_cast<std::size_t>(norm.space_dim) +
                                      static_cast<std::size_t>(d)] -
              origin[static_cast<std::size_t>(d)];
        }
      }
      xyz.push_back(to_vec(c));
    }

    Block block;
    block.n_flux = static_cast<int>(nf);
    block.flux = &flux;
    block.inv_w = &inv_w;
    HYPRE_ADSCreate(&block.ads);
    HYPRE_ADSSetDiscreteGradient(block.ads, mat_of(grad));
    HYPRE_ADSSetDiscreteCurl(block.ads, mat_of(curl));
    HYPRE_ADSSetCoordinateVectors(block.ads, vec_of(xyz[0]), vec_of(xyz[1]), vec_of(xyz[2]));
    HYPRE_ADSSetCycleType(block.ads, opts.ads_cycle_type);
    HYPRE_ADSSetPrintLevel(block.ads, opts.print_level);
    HYPRE_ADSSetMaxIter(block.ads, opts.ads_iterations);  // a preconditioner, not a solver
    HYPRE_ADSSetTol(block.ads, 0.0);
    // THE KNOBS PETSc DOES NOT FORWARD.
    HYPRE_ADSSetAMGOptions(block.ads, opts.amg_coarsen_type, opts.amg_agg_levels,
                           opts.amg_relax_type, opts.amg_theta, opts.amg_interp_type,
                           opts.amg_pmax);
    HYPRE_ADSSetAMSOptions(block.ads, 11, opts.amg_coarsen_type, opts.amg_agg_levels,
                           opts.amg_relax_type, opts.ams_theta, opts.amg_interp_type,
                           opts.amg_pmax);

    block.a0 = mat_of(a0);
    block.rhs = to_vec(std::vector<double>(static_cast<std::size_t>(nf), 0.0));
    block.sol = to_vec(std::vector<double>(static_cast<std::size_t>(nf), 0.0));
    HYPRE_ADSSetup(block.ads, block.a0, vec_of(block.rhs), vec_of(block.sol));

    // the inner CG on the block, with that cycle as its preconditioner
    if (opts.block_iterations > 0) {
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &block.inner);
      HYPRE_PCGSetMaxIter(block.inner, opts.block_iterations);
      HYPRE_PCGSetTol(block.inner, opts.block_rtol);
      HYPRE_PCGSetTwoNorm(block.inner, 1);
      HYPRE_PCGSetPrintLevel(block.inner, 0);
      // the cast hypre's own examples use: the generic pointer type is stated
      // over HYPRE_Matrix/HYPRE_Vector, the ADS entry points over the ParCSR
      // ones, and the library dispatches on the object it is handed
      HYPRE_PCGSetPrecond(block.inner,
                          reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSolve),
                          reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSetup), block.ads);
      HYPRE_ParCSRPCGSetup(block.inner, block.a0, vec_of(block.rhs), vec_of(block.sol));
    }

    HYPRE_Solver ksp = nullptr;
    HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &ksp);
    HYPRE_FlexGMRESSetKDim(ksp, 50);
    HYPRE_FlexGMRESSetMaxIter(ksp, opts.max_iterations);
    HYPRE_FlexGMRESSetTol(ksp, opts.rtol);
    HYPRE_FlexGMRESSetPrintLevel(ksp, opts.print_level);
    HYPRE_FlexGMRESSetPrecond(ksp, apply, setup, reinterpret_cast<HYPRE_Solver>(&block));

    Vec rhs = to_vec(b);
    Vec sol = to_vec(std::vector<double>(A.n, 0.0));
    HYPRE_ParCSRFlexGMRESSetup(ksp, mat_of(a_full), vec_of(rhs), vec_of(sol));
    const auto t1 = std::chrono::steady_clock::now();
    HYPRE_ParCSRFlexGMRESSolve(ksp, mat_of(a_full), vec_of(rhs), vec_of(sol));
    const auto t2 = std::chrono::steady_clock::now();

    Report r;
    HYPRE_Int its = 0;
    HYPRE_FlexGMRESGetNumIterations(ksp, &its);
    HYPRE_FlexGMRESGetFinalRelativeResidualNorm(ksp, &r.residual);
    r.iterations = static_cast<int>(its);
    r.converged = r.residual < opts.rtol * 10.0 && r.iterations < opts.max_iterations;
    r.reason = r.converged ? "CONVERGED_RTOL" : "DIVERGED_ITS";
    r.setup_seconds = std::chrono::duration<double>(t1 - t0).count();
    r.solve_seconds = std::chrono::duration<double>(t2 - t1).count();

    x.assign(A.n, 0.0);
    read_back(sol, x);

    HYPRE_ParCSRFlexGMRESDestroy(ksp);
    if (block.inner != nullptr) HYPRE_ParCSRPCGDestroy(block.inner);
    HYPRE_ADSDestroy(block.ads);
    return r;
  }

 private:
  // ---- small owning wrappers, so a throw does not leak hypre objects -------
  struct Mat {
    HYPRE_IJMatrix ij{nullptr};
    Mat() = default;
    Mat(const Mat&) = delete;
    Mat& operator=(const Mat&) = delete;
    Mat(Mat&& o) noexcept : ij(o.ij) { o.ij = nullptr; }
    Mat& operator=(Mat&& o) noexcept {
      if (this != &o) { if (ij != nullptr) HYPRE_IJMatrixDestroy(ij); ij = o.ij; o.ij = nullptr; }
      return *this;
    }
    ~Mat() { if (ij != nullptr) HYPRE_IJMatrixDestroy(ij); }
  };
  struct Vec {
    HYPRE_IJVector ij{nullptr};
    Vec() = default;
    Vec(const Vec&) = delete;
    Vec& operator=(const Vec&) = delete;
    Vec(Vec&& o) noexcept : ij(o.ij) { o.ij = nullptr; }
    Vec& operator=(Vec&& o) noexcept {
      if (this != &o) { if (ij != nullptr) HYPRE_IJVectorDestroy(ij); ij = o.ij; o.ij = nullptr; }
      return *this;
    }
    ~Vec() { if (ij != nullptr) HYPRE_IJVectorDestroy(ij); }
  };
  using Vecs = std::vector<Vec>;

  static HYPRE_ParCSRMatrix mat_of(const Mat& m) {
    void* obj = nullptr;
    HYPRE_IJMatrixGetObject(m.ij, &obj);
    return static_cast<HYPRE_ParCSRMatrix>(obj);
  }
  static HYPRE_ParVector vec_of(const Vec& v) {
    void* obj = nullptr;
    HYPRE_IJVectorGetObject(v.ij, &obj);
    return static_cast<HYPRE_ParVector>(obj);
  }

  // triplets, summed by (row, col)
  struct Triplets {
    std::vector<int> row, col;
    std::vector<double> value;
    void add(int r, int c, double v) {
      row.push_back(r);
      col.push_back(c);
      value.push_back(v);
    }
  };

  // ONE CALL, NOT ONE PER ENTRY.
  //
  // HYPRE_IJMatrixAddToValues per triplet is O(nnz) calls into the library and
  // is what made a 93k-cell industrial mesh -- 3.3 million entries -- appear to
  // hang. The triplets are summed by (row, col) here and handed over as whole
  // rows, which is the shape SetValues takes.
  static Mat build_mat(std::vector<int> row, std::vector<int> col, std::vector<double> val,
                       HYPRE_BigInt rows, HYPRE_BigInt cols) {
    std::vector<std::size_t> order(val.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return row[a] != row[b] ? row[a] < row[b] : col[a] < col[b];
    });
    std::vector<HYPRE_BigInt> r_of, c_of;
    std::vector<HYPRE_Int> n_of;
    std::vector<double> v_of;
    c_of.reserve(order.size());
    v_of.reserve(order.size());
    for (std::size_t k = 0; k < order.size();) {
      const int r = row[order[k]];
      r_of.push_back(r);
      HYPRE_Int count = 0;
      while (k < order.size() && row[order[k]] == r) {
        const int c = col[order[k]];
        double sum = 0.0;
        while (k < order.size() && row[order[k]] == r && col[order[k]] == c) {
          sum += val[order[k]];
          ++k;
        }
        c_of.push_back(c);
        v_of.push_back(sum);
        ++count;
      }
      n_of.push_back(count);
    }
    Mat out;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &out.ij);
    HYPRE_IJMatrixSetObjectType(out.ij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(out.ij);
    if (!r_of.empty()) {
      HYPRE_IJMatrixSetValues(out.ij, static_cast<HYPRE_Int>(r_of.size()), n_of.data(),
                              r_of.data(), c_of.data(), v_of.data());
    }
    HYPRE_IJMatrixAssemble(out.ij);
    return out;
  }

  static Mat to_mat(const Triplets& t, HYPRE_BigInt n) {
    return build_mat(t.row, t.col, t.value, n, n);
  }

  static Mat to_mat(const SparseSystem& a, HYPRE_BigInt n) {
    std::vector<int> r(a.nnz()), c(a.nnz());
    std::vector<double> v(a.nnz());
    for (std::size_t k = 0; k < a.nnz(); ++k) {
      r[k] = static_cast<int>(a.row[k]);
      c[k] = static_cast<int>(a.col[k]);
      v[k] = a.value[k];
    }
    return build_mat(std::move(r), std::move(c), std::move(v), n, n);
  }

  static Mat to_mat(const SpaceNorm::Incidence& inc) {
    return build_mat(inc.row, inc.col, inc.value, inc.rows, inc.cols);
  }

  static Vec to_vec(const std::vector<double>& v) {
    Vec out;
    const auto n = static_cast<HYPRE_BigInt>(v.size());
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &out.ij);
    HYPRE_IJVectorSetObjectType(out.ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(out.ij);
    std::vector<HYPRE_BigInt> idx(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<HYPRE_BigInt>(i);
    HYPRE_IJVectorSetValues(out.ij, static_cast<HYPRE_Int>(v.size()), idx.data(), v.data());
    HYPRE_IJVectorAssemble(out.ij);
    return out;
  }

  static void read_back(const Vec& v, std::vector<double>& out) {
    std::vector<HYPRE_BigInt> idx(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) idx[i] = static_cast<HYPRE_BigInt>(i);
    HYPRE_IJVectorGetValues(v.ij, static_cast<HYPRE_Int>(out.size()), idx.data(), out.data());
  }

  // ---- the block preconditioner -------------------------------------------
  //
  // y = P^-1 r, with ADS on the first factor and a diagonal scaling on the
  // rest. The vectors are the full system's, so the flux entries are gathered
  // into the block's own vector and scattered back; serially that is a copy
  // over an index set and no communication.
  struct Block {
    HYPRE_Solver ads{nullptr};
    HYPRE_Solver inner{nullptr};
    HYPRE_ParCSRMatrix a0{nullptr};
    Vec rhs, sol;
    int n_flux{0};
    const std::vector<int>* flux{nullptr};
    const std::vector<double>* inv_w{nullptr};
  };

  static double* data_of(HYPRE_ParVector v) {
    return hypre_VectorData(hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(v)));
  }

  static HYPRE_Int setup(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector, HYPRE_Vector) { return 0; }

  static HYPRE_Int apply(HYPRE_Solver s, HYPRE_Matrix, HYPRE_Vector rv, HYPRE_Vector xv) {
    Block& b = *reinterpret_cast<Block*>(s);
    const double* r = data_of(reinterpret_cast<HYPRE_ParVector>(rv));
    double* x = data_of(reinterpret_cast<HYPRE_ParVector>(xv));
    double* fr = data_of(vec_of(b.rhs));
    double* fx = data_of(vec_of(b.sol));

    const auto& flux = *b.flux;
    for (std::size_t i = 0; i < flux.size(); ++i) {
      fr[i] = r[static_cast<std::size_t>(flux[i])];
      fx[i] = 0.0;
    }
    if (b.inner != nullptr) {
      HYPRE_ParCSRPCGSolve(b.inner, b.a0, vec_of(b.rhs), vec_of(b.sol));
    } else {
      HYPRE_ADSSolve(b.ads, b.a0, vec_of(b.rhs), vec_of(b.sol));
    }

    const auto& iw = *b.inv_w;
    for (std::size_t i = 0; i < iw.size(); ++i) x[i] = r[i] * iw[i];
    for (std::size_t i = 0; i < flux.size(); ++i) {
      x[static_cast<std::size_t>(flux[i])] = fx[i];
    }
    return 0;
  }
};

}  // namespace mimetika::solver
