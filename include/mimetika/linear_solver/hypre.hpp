#pragma once

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
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
// and the eta = 1 cells of adaptive_rt -- and takes those directly. A facet
// carrying d moments reaches it through the facet-constant subspace, and the
// block is then a two-level cycle: the Galerkin operator P^T A0 P on the facet
// constants, where ADS runs, under a symmetric SOR sweep.
//
// What it does not take is d COPIES of an H(div) space -- a weak-symmetry
// stress -- which needs the per-component split the PETSc path builds.

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

  // Who owns each unknown. Empty, or one rank, means serial.
  void set_owners(std::vector<int> owner_of_dof) { owners_ = std::move(owner_of_dof); }

  // A space laid out on the partition.
  //
  // hypre's IJ interface gives each rank a CONTIGUOUS run of global indices,
  // and the complex's own numbering is not contiguous per rank -- so every
  // space is renumbered by (owner, original index), which is a stable sort and
  // therefore the same permutation on every process. `new_of` takes the
  // complex's index to hypre's; `old_of` inverts it.
  struct Layout {
    std::vector<int> new_of, old_of, first;
    int begin{0}, end{0}, local{0}, total{0};
    bool owns(int i) const { return i >= begin && i < end; }
  };

  static Layout layout_of(const std::vector<int>& owner, int ranks, int rank) {
    Layout out;
    out.total = static_cast<int>(owner.size());
    std::vector<int> count(static_cast<std::size_t>(ranks) + 1, 0);
    for (const int r : owner) {
      if (r < 0 || r >= ranks) throw std::invalid_argument("HypreSolver: an entity has no owner");
      ++count[static_cast<std::size_t>(r) + 1];
    }
    for (int r = 0; r < ranks; ++r) count[static_cast<std::size_t>(r) + 1] += count[static_cast<std::size_t>(r)];
    out.first = count;
    out.new_of.assign(owner.size(), -1);
    out.old_of.assign(owner.size(), -1);
    std::vector<int> at = count;
    for (std::size_t i = 0; i < owner.size(); ++i) {
      const int slot = at[static_cast<std::size_t>(owner[i])]++;
      out.new_of[i] = slot;
      out.old_of[static_cast<std::size_t>(slot)] = static_cast<int>(i);
    }
    out.begin = count[static_cast<std::size_t>(rank)];
    out.end = count[static_cast<std::size_t>(rank) + 1];
    out.local = out.end - out.begin;
    return out;
  }

  // the identity layout, for one rank
  static Layout serial_layout(int n) {
    Layout out;
    out.total = n;
    out.local = n;
    out.end = n;
    out.first = {0, n};
    out.new_of.resize(static_cast<std::size_t>(n));
    out.old_of.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) out.new_of[static_cast<std::size_t>(i)] = i;
    out.old_of = out.new_of;
    return out;
  }

  // Solve A x = b with the Riesz map above. `norm` supplies the factors, the
  // L2 weights and the complex's two boundary operators.
  Report solve(const SparseSystem& A, const std::vector<double>& b, std::vector<double>& x,
               const SpaceNorm& norm, const Options& opts = Options{}) {
    HypreSession::ensure();
    int ranks = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // EVERY SPACE IS RENUMBERED ONTO THE PARTITION.
    //
    // hypre's IJ interface gives each rank a contiguous run of global indices,
    // and neither the unknowns nor the complex's entities are contiguous per
    // rank in their own numbering. So the four spaces this needs -- the whole
    // system, the faces the flux sits on, the edges and the vertices -- are
    // each sorted by (owner, index) and every matrix is stated in the new
    // numbering. A rank then inserts only its own rows, which is what the
    // interface asks and what a run over more than one process was deadlocking
    // on before.
    if (ranks > 1 && owners_.size() != A.n) {
      throw std::invalid_argument(
          "HypreSolver: a distributed run needs the owner of every unknown; "
          "call set_owners()");
    }
    if (ranks > 1 && norm.entity_owner.size() < 3) {
      throw std::invalid_argument(
          "HypreSolver: a distributed run needs the owner of every vertex, edge and face");
    }
    if (norm.empty()) throw std::invalid_argument("HypreSolver: the norm has no factors");
    if (norm.discrete_gradient.empty() || norm.discrete_curl.empty()) {
      throw std::invalid_argument("HypreSolver: ADS needs the discrete gradient and curl");
    }
    const auto n = static_cast<HYPRE_BigInt>(A.n);
    const std::vector<int>& flux = norm.factors[0];
    const auto nf = static_cast<HYPRE_BigInt>(flux.size());
    // ONE MOMENT PER FACET, OR THE SUBSPACE THAT IS.
    //
    // ADS takes a scalar H(div) problem: one unknown per face, which is what
    // the curl's rows address. A facet carrying d moments -- derham_bdm,
    // stabilized_bdm -- reaches it through the facet-constant subspace, and
    // the block is then a two-level cycle whose coarse operator is where ADS
    // runs. `lowest_order` is that injection, and it is a matrix of ones
    // because the constant moment IS one of the unknowns.
    const bool two_level = !norm.lowest_order.empty();
    if (!two_level && static_cast<HYPRE_BigInt>(norm.discrete_curl.rows) != nf) {
      throw std::invalid_argument("HypreSolver: the curl's rows are not the first factor");
    }
    if (two_level && norm.lowest_order.cols != norm.discrete_curl.rows * norm.lowest_order_components) {
      throw std::invalid_argument(
          "HypreSolver: the coarse space is not the faces of the complex");
    }
    if (two_level && norm.lowest_order_components != 1) {
      throw std::invalid_argument(
          "HypreSolver: the direct path takes one copy of the H(div) space; d copies -- a "
          "weak-symmetry stress -- need the component split the PETSc path builds");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // the four spaces, on the partition
    const Layout dofs = ranks > 1 ? layout_of(owners_, ranks, rank)
                                  : serial_layout(static_cast<int>(A.n));
    const Layout verts = ranks > 1 ? layout_of(norm.entity_owner[0], ranks, rank)
                                   : serial_layout(norm.discrete_gradient.cols);
    const Layout edges = ranks > 1 ? layout_of(norm.entity_owner[1], ranks, rank)
                                   : serial_layout(norm.discrete_gradient.rows);
    const Layout faces = ranks > 1 ? layout_of(norm.entity_owner[2], ranks, rank)
                                   : serial_layout(norm.discrete_curl.rows);
    // The block's own numbering. With one moment a facet it IS the faces --
    // the same owners in the same order, so the same permutation -- and with d
    // moments it is d times larger and the faces are its coarse space.
    std::vector<int> flux_owner(flux.size(), 0);
    if (ranks > 1) {
      for (std::size_t i = 0; i < flux.size(); ++i) {
        flux_owner[i] = owners_[static_cast<std::size_t>(flux[i])];
      }
    }
    const Layout block_l = ranks > 1 ? layout_of(flux_owner, ranks, rank)
                                     : serial_layout(static_cast<int>(flux.size()));

    // Where each global unknown sits in the flux block, in HYPRE's numbering.
    //
    // The flux block IS the face space -- one moment per facet, checked above --
    // so a flux unknown's row in A0 is its face's row after renumbering, and
    // that is the same row the discrete curl has. Position in factors[0] is
    // the face index; the layout takes it to hypre's.
    std::vector<int> in_flux(A.n, -1);
    for (std::size_t i = 0; i < flux.size(); ++i) {
      in_flux[static_cast<std::size_t>(flux[i])] = block_l.new_of[i];
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

    Mat a_full = to_mat(A, dofs);
    Mat a0 = to_mat(m, block_l);
    Mat grad = to_mat(norm.discrete_gradient, edges, verts);
    Mat curl = to_mat(norm.discrete_curl, faces, edges);

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
      xyz.push_back(to_vec(c, verts));
    }

    // The preconditioner works on LOCAL data: hypre hands apply() this rank's
    // rows of the outer vectors, and the block's vectors hold this rank's
    // faces. A dof and the face it sits on are owned by the same rank -- both
    // partitions come from the same exokal call -- so the gather is a local
    // permutation and needs no communication.
    // THE COARSE SPACE, WHERE ONE IS NEEDED.
    //
    // The injection's rows are global unknowns and its columns the faces; both
    // are renumbered onto the partition, and the coarse operator is the
    // Galerkin product P^T A0 P -- so nothing about the physics is restated at
    // the coarse level, it is the same operator seen on the subspace. ADS then
    // runs there, on one unknown per face, which is what it is written for.
    Mat inject;
    if (two_level) {
      std::vector<int> ir, ic;
      std::vector<double> iv;
      ir.reserve(norm.lowest_order.value.size());
      ic.reserve(norm.lowest_order.value.size());
      iv.reserve(norm.lowest_order.value.size());
      for (std::size_t k = 0; k < norm.lowest_order.value.size(); ++k) {
        const int g = norm.lowest_order.row[k];           // a global unknown
        const int r = in_flux[static_cast<std::size_t>(g)];
        if (r < 0) continue;                              // not in the flux block
        ir.push_back(r);
        ic.push_back(faces.new_of[static_cast<std::size_t>(norm.lowest_order.col[k])]);
        iv.push_back(norm.lowest_order.value[k]);
      }
      inject = build_mat(std::move(ir), std::move(ic), std::move(iv), block_l.total, faces.total,
                         block_l.begin, block_l.end, faces.begin, faces.end);
    }

    Block block;
    block.two_level = two_level;
    block.n_flux = static_cast<int>(nf);
    block.inv_w_local.assign(static_cast<std::size_t>(dofs.local), 0.0);
    block.flux_at.assign(static_cast<std::size_t>(dofs.local), -1);
    for (std::size_t i = 0; i < A.n; ++i) {
      const int at = dofs.new_of[i];
      if (!dofs.owns(at)) continue;
      const std::size_t local = static_cast<std::size_t>(at - dofs.begin);
      block.inv_w_local[local] = inv_w[i];
      if (in_flux[i] >= 0) {
        if (!block_l.owns(in_flux[i])) {
          throw std::invalid_argument(
              "HypreSolver: a flux unknown and its face are on different ranks");
        }
        block.flux_at[local] = in_flux[i] - block_l.begin;
      }
    }
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
    block.rhs = zero_vec(block_l);
    block.sol = zero_vec(block_l);
    if (two_level) {
      block.p = mat_of(inject);
      // hypre builds a matrix's communication package lazily, and its internal
      // parallel routines -- the Galerkin product among them -- assume it is
      // already there. Absent, RAP walks a null offd map and segfaults, which
      // is what a 2-rank run did.
      if (hypre_ParCSRMatrixCommPkg(block.p) == nullptr) hypre_MatvecCommPkgCreate(block.p);
      if (hypre_ParCSRMatrixCommPkg(block.a0) == nullptr) hypre_MatvecCommPkgCreate(block.a0);
      block.a_coarse = hypre_ParCSRMatrixRAP(block.p, block.a0, block.p);
      if (hypre_ParCSRMatrixCommPkg(block.a_coarse) == nullptr) {
        hypre_MatvecCommPkgCreate(block.a_coarse);
      }
      block.c_rhs = zero_vec(faces);
      block.c_sol = zero_vec(faces);
      block.resid = zero_vec(block_l);
      block.vtemp = zero_vec(block_l);
      block.ztemp = zero_vec(block_l);
      block.diag = block_diagonal(m, block_l);
      HYPRE_ADSSetup(block.ads, block.a_coarse, vec_of(block.c_rhs), vec_of(block.c_sol));
    } else {
      HYPRE_ADSSetup(block.ads, block.a0, vec_of(block.rhs), vec_of(block.sol));
    }

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
      if (two_level) {
        HYPRE_PCGSetPrecond(block.inner, cycle, cycle_setup,
                            reinterpret_cast<HYPRE_Solver>(&block));
      } else {
        HYPRE_PCGSetPrecond(block.inner,
                            reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSolve),
                            reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSetup), block.ads);
      }
      HYPRE_ParCSRPCGSetup(block.inner, block.a0, vec_of(block.rhs), vec_of(block.sol));
    }

    HYPRE_Solver ksp = nullptr;
    HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &ksp);
    HYPRE_FlexGMRESSetKDim(ksp, 50);
    HYPRE_FlexGMRESSetMaxIter(ksp, opts.max_iterations);
    HYPRE_FlexGMRESSetTol(ksp, opts.rtol);
    HYPRE_FlexGMRESSetPrintLevel(ksp, opts.print_level);
    HYPRE_FlexGMRESSetPrecond(ksp, apply, setup, reinterpret_cast<HYPRE_Solver>(&block));

    Vec rhs = to_vec(b, dofs);
    Vec sol = to_vec(std::vector<double>(A.n, 0.0), dofs);
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
    read_back(sol, dofs, x);

    HYPRE_ParCSRFlexGMRESDestroy(ksp);
    if (block.inner != nullptr) HYPRE_ParCSRPCGDestroy(block.inner);
    if (block.a_coarse != nullptr) hypre_ParCSRMatrixDestroy(block.a_coarse);
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
  // rows/cols are ALREADY in hypre's numbering; `mine` is this rank's row run.
  static Mat build_mat(std::vector<int> row, std::vector<int> col, std::vector<double> val,
                       HYPRE_BigInt rows, HYPRE_BigInt cols, int row_begin, int row_end,
                       int col_begin, int col_end) {
    std::vector<std::size_t> order;
    order.reserve(val.size());
    for (std::size_t i = 0; i < val.size(); ++i) {
      if (row[i] >= row_begin && row[i] < row_end) order.push_back(i);  // this rank's rows
    }
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
    (void)rows;
    (void)cols;
    Mat out;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, row_begin, row_end - 1, col_begin, col_end - 1,
                         &out.ij);
    HYPRE_IJMatrixSetObjectType(out.ij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(out.ij);
    if (!r_of.empty()) {
      HYPRE_IJMatrixSetValues(out.ij, static_cast<HYPRE_Int>(r_of.size()), n_of.data(),
                              r_of.data(), c_of.data(), v_of.data());
    }
    HYPRE_IJMatrixAssemble(out.ij);
    return out;
  }

  static Mat to_mat(const Triplets& t, const Layout& l) {
    return build_mat(t.row, t.col, t.value, l.total, l.total, l.begin, l.end, l.begin, l.end);
  }

  static Mat to_mat(const SparseSystem& a, const Layout& l) {
    std::vector<int> r(a.nnz()), c(a.nnz());
    std::vector<double> v(a.nnz());
    for (std::size_t k = 0; k < a.nnz(); ++k) {
      r[k] = l.new_of[static_cast<std::size_t>(a.row[k])];
      c[k] = l.new_of[static_cast<std::size_t>(a.col[k])];
      v[k] = a.value[k];
    }
    return build_mat(std::move(r), std::move(c), std::move(v), l.total, l.total, l.begin, l.end,
                     l.begin, l.end);
  }

  static Mat to_mat(const SpaceNorm::Incidence& inc, const Layout& r, const Layout& c) {
    std::vector<int> row(inc.row.size()), col(inc.col.size());
    for (std::size_t k = 0; k < inc.row.size(); ++k) {
      row[k] = r.new_of[static_cast<std::size_t>(inc.row[k])];
      col[k] = c.new_of[static_cast<std::size_t>(inc.col[k])];
    }
    // The column range is the COLUMN space's own local run: hypre pairs a
    // rectangular matrix's columns with the vector it multiplies, so the
    // gradient's columns are partitioned like the vertices and the curl's like
    // the edges. Giving every rank all the columns is what a 2-rank run was
    // dying on.
    return build_mat(std::move(row), std::move(col), inc.value, r.total, c.total, r.begin, r.end,
                     c.begin, c.end);
  }

  // `v` is indexed the caller's way; `l` says where each entry goes and which
  // of them this rank owns.
  static Vec to_vec(const std::vector<double>& v, const Layout& l) {
    Vec out;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, l.begin, l.end - 1, &out.ij);
    HYPRE_IJVectorSetObjectType(out.ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(out.ij);
    std::vector<HYPRE_BigInt> idx;
    std::vector<double> val;
    idx.reserve(static_cast<std::size_t>(l.local));
    val.reserve(static_cast<std::size_t>(l.local));
    for (std::size_t i = 0; i < v.size(); ++i) {
      const int at = l.new_of[i];
      if (!l.owns(at)) continue;
      idx.push_back(at);
      val.push_back(v[i]);
    }
    if (!idx.empty()) {
      HYPRE_IJVectorSetValues(out.ij, static_cast<HYPRE_Int>(idx.size()), idx.data(), val.data());
    }
    HYPRE_IJVectorAssemble(out.ij);
    return out;
  }

  // a zero vector over a layout, for the block's working space
  static Vec zero_vec(const Layout& l) {
    return to_vec(std::vector<double>(static_cast<std::size_t>(l.total), 0.0), l);
  }

  // This rank's entries, put back where the caller expects them. Every rank
  // ends up with its own rows only; the caller gathers if it wants the whole
  // answer, exactly as the PETSc path leaves it.
  static void read_back(const Vec& v, const Layout& l, std::vector<double>& out) {
    std::vector<HYPRE_BigInt> idx;
    std::vector<int> where;
    idx.reserve(static_cast<std::size_t>(l.local));
    where.reserve(static_cast<std::size_t>(l.local));
    for (std::size_t i = 0; i < out.size(); ++i) {
      const int at = l.new_of[i];
      if (!l.owns(at)) continue;
      idx.push_back(at);
      where.push_back(static_cast<int>(i));
    }
    std::vector<double> got(idx.size(), 0.0);
    if (!idx.empty()) {
      HYPRE_IJVectorGetValues(v.ij, static_cast<HYPRE_Int>(idx.size()), idx.data(), got.data());
    }
    for (std::size_t k = 0; k < where.size(); ++k) {
      out[static_cast<std::size_t>(where[k])] = got[k];
    }
    if (l.total != l.local) {
      MPI_Allreduce(MPI_IN_PLACE, out.data(), static_cast<int>(out.size()), MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  }

  // ---- the block preconditioner -------------------------------------------
  //
  // y = P^-1 r, with ADS on the first factor and a diagonal scaling on the
  // rest. The vectors are the full system's, so the flux entries are gathered
  // into the block's own vector and scattered back; serially that is a copy
  // over an index set and no communication.
  std::vector<int> owners_;

  struct Block {
    HYPRE_Solver ads{nullptr};
    HYPRE_Solver inner{nullptr};
    bool two_level{false};
    HYPRE_ParCSRMatrix p{nullptr};        // the injection
    HYPRE_ParCSRMatrix a_coarse{nullptr};  // P^T A0 P
    Vec c_rhs, c_sol, resid;
    // the relaxation's own scratch. hypre uses Vtemp and Ztemp to stage the
    // off-rank part of a sweep, so they must not be the vector the cycle is
    // holding its residual in -- sharing them is invisible on one process and
    // corrupts on several.
    Vec vtemp, ztemp;
    std::vector<double> diag;              // of A0, for the smoother
    HYPRE_ParCSRMatrix a0{nullptr};
    Vec rhs, sol;
    int n_flux{0};
    // per LOCAL row of the outer vector: 1/W, and where it sits in the block
    std::vector<double> inv_w_local;
    std::vector<int> flux_at;
  };

  static double* data_of(HYPRE_ParVector v) {
    return hypre_VectorData(hypre_ParVectorLocalVector(reinterpret_cast<hypre_ParVector*>(v)));
  }

  // the diagonal of the block, this rank's rows, for the smoother
  static std::vector<double> block_diagonal(const Triplets& t, const Layout& l) {
    std::vector<double> d(static_cast<std::size_t>(l.local), 0.0);
    for (std::size_t k = 0; k < t.value.size(); ++k) {
      if (t.row[k] != t.col[k]) continue;
      if (!l.owns(t.row[k])) continue;
      d[static_cast<std::size_t>(t.row[k] - l.begin)] += t.value[k];
    }
    for (double& v : d) v = v != 0.0 ? 1.0 / v : 1.0;
    return d;
  }

  static HYPRE_Int setup(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector, HYPRE_Vector) { return 0; }

  static HYPRE_Int cycle_setup(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector, HYPRE_Vector) {
    return 0;
  }

  // relax_type 6 is hypre's symmetric hybrid Gauss-Seidel: a forward sweep and
  // a backward one, which is what keeps the composition symmetric so a CG may
  // use it. It relaxes u toward solving A0 u = f rather than replacing u.
  static void smooth(Block& b, HYPRE_ParVector f, HYPRE_ParVector u) {
    hypre_BoomerAMGRelax(b.a0, f, nullptr, 6, 0, 1.0, 1.0, nullptr, u, vec_of(b.vtemp),
                         vec_of(b.ztemp));
  }

  // A TWO-LEVEL CYCLE, SYMMETRIC SO A CG MAY USE IT.
  //
  //   smoother  A SYMMETRIC SOR SWEEP, not a point method. What the coarse
  //             space does not carry is the non-constant moments, and those
  //             are the divergence-free directions -- only the constant moment
  //             reaches div. Kolev and Vassilevski say that near-nullspace
  //             "cannot be handled by simple relaxation on the fine grid", and
  //             the measurement agrees: with damped Jacobi here the outer count
  //             ran 18, 24, 30 over three refinements and failed at a contrast
  //             of 1e4, where a sweep is flat. A point smoother splits facet
  //             from facet and a div-free field is global.
  //   coarse    the facet constants, one H(div) problem, and that is ADS.
  //
  // Pre-smooth, restrict the residual, correct, prolong, post-smooth: the
  // composition is symmetric, which the inner CG requires of its
  // preconditioner.
  static HYPRE_Int cycle(HYPRE_Solver s, HYPRE_Matrix, HYPRE_Vector rv, HYPRE_Vector xv) {
    Block& b = *reinterpret_cast<Block*>(s);
    auto* r = reinterpret_cast<HYPRE_ParVector>(rv);
    auto* x = reinterpret_cast<HYPRE_ParVector>(xv);
    double* xd = data_of(x);
    const double* rd = data_of(r);
    const std::size_t n = b.diag.size();

    for (std::size_t i = 0; i < n; ++i) xd[i] = 0.0;
    smooth(b, r, x);  // pre-smooth

    // residual, restricted to the coarse space
    double* res = data_of(vec_of(b.resid));
    for (std::size_t i = 0; i < n; ++i) res[i] = rd[i];
    hypre_ParCSRMatrixMatvec(-1.0, b.a0, x, 1.0, vec_of(b.resid));
    hypre_ParCSRMatrixMatvecT(1.0, b.p, vec_of(b.resid), 0.0, vec_of(b.c_rhs));

    double* cs = data_of(vec_of(b.c_sol));
    const std::size_t nc = static_cast<std::size_t>(
        hypre_VectorSize(hypre_ParVectorLocalVector(vec_of(b.c_sol))));
    for (std::size_t i = 0; i < nc; ++i) cs[i] = 0.0;
    HYPRE_ADSSolve(b.ads, b.a_coarse, vec_of(b.c_rhs), vec_of(b.c_sol));

    hypre_ParCSRMatrixMatvec(1.0, b.p, vec_of(b.c_sol), 1.0, x);  // prolong and add

    smooth(b, r, x);  // post-smooth; with symm SOR the composition stays SPD
    (void)rd;
    (void)n;
    return 0;
  }

  static HYPRE_Int apply(HYPRE_Solver s, HYPRE_Matrix, HYPRE_Vector rv, HYPRE_Vector xv) {
    Block& b = *reinterpret_cast<Block*>(s);
    const double* r = data_of(reinterpret_cast<HYPRE_ParVector>(rv));
    double* x = data_of(reinterpret_cast<HYPRE_ParVector>(xv));
    double* fr = data_of(vec_of(b.rhs));
    double* fx = data_of(vec_of(b.sol));

    for (std::size_t i = 0; i < b.flux_at.size(); ++i) {
      if (b.flux_at[i] >= 0) {
        fr[static_cast<std::size_t>(b.flux_at[i])] = r[i];
        fx[static_cast<std::size_t>(b.flux_at[i])] = 0.0;
      }
    }
    if (b.inner != nullptr) {
      HYPRE_ParCSRPCGSolve(b.inner, b.a0, vec_of(b.rhs), vec_of(b.sol));
    } else if (b.two_level) {
      cycle(reinterpret_cast<HYPRE_Solver>(&b), nullptr,
            reinterpret_cast<HYPRE_Vector>(vec_of(b.rhs)),
            reinterpret_cast<HYPRE_Vector>(vec_of(b.sol)));
    } else {
      HYPRE_ADSSolve(b.ads, b.a0, vec_of(b.rhs), vec_of(b.sol));
    }

    for (std::size_t i = 0; i < b.inv_w_local.size(); ++i) {
      x[i] = b.flux_at[i] >= 0 ? fx[static_cast<std::size_t>(b.flux_at[i])]
                               : r[i] * b.inv_w_local[i];
    }
    return 0;
  }
};

}  // namespace mimetika::solver
