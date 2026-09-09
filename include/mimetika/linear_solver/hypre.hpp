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
#include "exokal/numerics/dense.hpp"
#include "mimetika/linear_solver/space_norm.hpp"

// The Riesz map with its first block handed to hypre directly.
//
// This header must not meet PETSc in one binary: PETSc links its own libHYPRE,
// both copies export HYPRE_ADSCreate and the rest, and load order decides which
// a call reaches. A target that includes this one links mimetika_hypre and not
// mimetika_petsc.
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
// ADS is written for one unknown per facet in 3D -- derham_rt, stabilized_rt,
// and the eta = 1 cells of adaptive_rt -- and takes those directly. A facet
// carrying d moments reaches it through the facet-constant subspace, and the
// block is then a two-level cycle: the Galerkin operator P^T A0 P on the facet
// constants, where ADS runs, under an l1-scaled Gauss-Seidel sweep.
//
// What it does not take is d copies of an H(div) space -- a weak-symmetry
// stress -- which needs the per-component split the PETSc path builds.

#include <cstdlib>

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

// The sweep over the copies runs forward only.
//
// The backward pass exists only to make the cycle symmetric so that a CG may
// precondition with it. Dropped, the cycle costs half as much -- d applications
// of ADS instead of 2d -- and the inner Krylov has to tolerate a non-symmetric
// preconditioner, which FlexGMRES does; the outer method already is one for the
// same reason. Measured on the hybrid meshes: level 1 12.45 s -> 8.25 s, level 2
// 144 s -> 88 s, with the outer count moving 44 -> 46 and 47 -> 49 and all 87
// robustness assertions unchanged.
//
// Set MIMETIKA_ADS_SYMMETRIC_SWEEP to restore the symmetric cycle and the inner
// CG, which is what the earlier measurements in this file were taken with.
inline bool forward_only() {
  static const bool off = std::getenv("MIMETIKA_ADS_SYMMETRIC_SWEEP") != nullptr;
  return !off;
}

struct HypreOptions {
  double rtol{1e-8};
  int max_iterations{500};
  // The ADS cycle. 11-14 apply three scalar AMG solves in place of one
  // monolithic vector solve; 13 is the 5-level multiplicative (034515430)
  // that measured fastest per application on the hybrid mesh.
  int ads_cycle_type{13};
  // MGR instead of the Riesz map. Reduce the flux away rather than precondition
  // it: F = the flux, C = the pressure, and the coarse operator is
  //
  //     S = -D M^-1 D^T ,
  //
  // the cell-centred Laplacian -- one unknown a cell, which is BoomerAMG's own
  // ground and is what the two-point family already is. The flux block here is
  // a mass matrix (no div term), so F-relaxation is not facing the a_div
  // near-nullspace ADS exists for.
  bool mgr{false};
  // F-relaxation on the flux block, and the default is read off the operator.
  // hypre's methods: 0 Jacobi, 1 AMG V-cycle, 2 AMG, 9/99 direct.
  //
  // -1 picks it: Jacobi where a facet carries one moment, AMG where it carries
  // more. A scalar flux block is a mass matrix and Jacobi damps it; a stress
  // block is not. The compliance is diagonal on the Frobenius-orthonormal modes
  // with five deviatoric eigenvalues at 1/2mu and one hydrostatic at
  // 1/(2mu + d lam), and Jacobi cannot damp a mode that small -- the count then
  // grows like sqrt(2mu + d lam), measured on tetrahedra as 74, 92, 204, 542,
  // 1648 over nu = 1/4 .. 0.4999. It is the F block and not the coarse one:
  // div(constant) = 0 puts the hydrostatic direction in ker B, hence orthogonal
  // to range(B^T), so it never enters S = -B M^-1 B^T.
  //
  // AMG on that block gives 36, 38, 46, 52, 60 -- 1.7 times over four orders of
  // lambda against 22 -- and h and the contrast improve with it. It costs about
  // 43 percent more a solve at nu = 1/4 on a 1.2e5-cell mesh and repays that
  // twenty-sevenfold at 0.4999. Direct is flat outright, 22 23 23 23 23, but
  // scales badly.
  int mgr_frelax{-1};
  // What stands in for A_FF^-1 in Wp = -A_FF^-1 A_FC, hence in the Galerkin
  // coarse operator A_CC + A_CF Wp. hypre reads all of these off A itself:
  // 0 injection, 1 l1-Jacobi (the diagonal plus the off-diagonal row mass),
  // 2 the plain diagonal (hypre's default), 3 classical modified, 4 approximate
  // inverse.
  int mgr_interp_type{2};
  // The strong-VEM split, by the roles the dofs play.
  //
  // stabilized_vem carries q = d(d+1)/2 traction moments a facet and the whole
  // of RM(E) a cell. Dv pairs them, and the pairing is block lower triangular:
  // a translation is constant on a facet so it reads only the constant
  // traction, while a rotation omega ^ (x - x_E) = omega ^ (x_f - x_E) +
  // omega ^ (x - x_f) reads the constants and the linear moments. Read off the
  // matrix on a tet: the three translation rows touch facet slots {0, 1, 3},
  // the three rotation rows touch all six.
  //
  //   0  off, one reduction of the whole stress -- what MGR has always done
  //   1  F = sigma, then u_r; C = u_t. The coarse block is then the cell
  //      translations alone, a vector Laplacian, rather than translations and
  //      rotations together as one scalar system.
  //   5  the same cut, with the mean traction eliminated rather than kept.
  //      F0 = ker Pi (the stabilization's own subspace, lambda-free), F1 = the
  //      mean traction (the consistency term), C = all of u. The C block is
  //      then displacement only and definite, which is what BoomerAMG needs;
  //      every partition holding stress in C leaves a saddle and diverges.
  //      Against split 0 this separates the stabilization from the consistency
  //      instead of reducing the whole stress at once.
  //   4  the mean / zero-mean cut, which is the one the pairing respects.
  //      T_h(f) carries four groups: the mean tangential traction (2 dofs), the
  //      in-plane torque n_f ^ (x - x_f) (1), the mean normal traction (1) and
  //      the two linear normal moments (2). The last two have zero mean over
  //      the face, and a translation is constant on a face, so Proposition
  //      3.1's alpha_E reads only the mean pair. Dassi-Lovadina-Visinoni
  //      unisolve the space in that order: the means first, then the torque,
  //      then the bending. So
  //
  //        F0 = the zero-mean traction   invisible to the translations
  //        F1 = u_r                      paired with the moment F0 carries
  //        C  = the mean traction, u_t   one vector a face against one a cell
  //
  //      and the coarse block is the lowest-order mixed problem. Detected off
  //      the matrix: the mean dofs are the ones a translation row touches.
  //   2  F = sigma minus its constant normal slot, then u_r; C = sigma_n u u_t.
  //      A uniform hydrostatic stress has traction p n, constant and purely
  //      normal, so on an L^2-orthonormal facet basis its moments are p |f| on
  //      the n chi_0 slot and exactly zero on the other five. The locking
  //      direction therefore lives in one slot a facet, and holding that slot
  //      in C keeps it out of every diagonal Wp.
  //
  // Level 1 divides by a diagonal level 0 created -- the (u, u) block is zero
  // in the saddle point -- and not by one it gutted, which is what breaks the
  // moment split on a BDM facet: there level 0's Galerkin update subtracts most
  // of the constants' diagonal and level 1's prolongation reaches 6.4e+02.
  int mgr_vem_split{0};
  // Cycles of the coarse solver. One is a preconditioner; many approximate an
  // exact coarse solve, which is how to tell a bad coarse operator from a bad
  // coarse solver.
  int mgr_coarse_iterations{1};
  // Lift the cell block's smallest eigenvalue, for MGR only.
  //
  // The two-field compliance is diagonal on the Frobenius-orthonormal modes
  // with five deviatoric eigenvalues at 1/2mu and one hydrostatic eigenvalue at
  // 1/(2mu + d lam). As lam -> infinity that one vanishes and the cell block
  // loses a rank, the lost direction being the hydrostatic constant stress --
  // it lies in ker(P_dev P) and in ker(S) = range(N), so both surviving terms
  // annihilate it. MGR interpolates with diag(A_FF)^-1, and a diagonal cannot
  // represent an inverse blowing up along one vector, so that direction is left
  // with condition number O(lam) and the count grows like its square root:
  // measured on tetrahedra 74, 92, 204, 542, 1648 over nu = 1/4 .. 0.4999,
  // whose ratios 2.66 and 3.04 track sqrt(lam)'s 3.19 and 3.17.
  //
  // So lift it diagonally, because a diagonal is what the interpolation reads.
  // Read off the assembled matrix alone -- no geometry, no mu, no lam: take
  // each cell's flux block, and where its smallest eigenvalue has fallen below
  // the second smallest, add (lambda_2 - lambda_1) diag(v v^T) with v the
  // corresponding eigenvector. Only the preconditioner's copy is touched; the
  // Krylov keeps the true operator, so the answer does not move.
  double mgr_hydrostatic_lift{0.0};
  // The sweep count must be odd: an even number of sweeps returns the F block
  // to where it started. Measured on the hybrid ladder, MGR converges at 1, 3,
  // 5, 7, 9 sweeps and stalls at the iteration cap at 2, 4, 6, 8 -- on flow
  // stabilized_rt (17, cap, 14, cap) and on the strong stress alike, the even
  // runs burning time in proportion to the count. solve() refuses an even
  // count when opts.mgr is set.
  //
  // 3 is the default, the wall-clock minimum. More relaxation costs about 70 ms
  // an iteration on a 390 ms cycle -- F-relaxation is the small part of it, the
  // coarse AMG solve the large -- and buys iterations until it stops paying.
  // Measured on the strong stress at l_3, nu = 1/4, as (iterations, seconds):
  //
  //     1 sweep  (218, 100.3)    5 sweeps  (88, 66.1)
  //     3 sweeps (107,  64.9)    7 sweeps  (78, 69.1)    9 sweeps (86, 84.6)
  //
  // so it is non-monotone and 3 to 7 is the basin. Against one sweep 3 is 35
  // percent faster, which is why the default is not 1.
  //
  // 5 is the setting for an incompressible run, at about 2 percent more wall
  // clock here. At l_1 over nu = 0.25, 0.4, 0.49, 0.499, 0.4999 one sweep gives
  // 79, 110, 318, 304 and then does not converge; three give 46, 60, 96, 185,
  // 209; five give 38, 47, 67, 105, 193. The gap widens with nu well before the
  // limit, so the 2 percent inverts there.
  int mgr_relax_sweeps{3};
  // The auxiliary hierarchies' strength thresholds -- what PETSc cannot set.
  // hypre's own defaults are 0.25 for both.
  double amg_theta{0.25};
  double ams_theta{0.25};
  int amg_coarsen_type{10};
  int amg_agg_levels{1};
  // 3, hybrid Gauss-Seidel. The l1-scaled type 8 is the parallel-convergent
  // choice and is what the cycle's own smoother uses, but INSIDE ADS it is a
  // loss: measured on hybrid_mesh_l_2 it left the cycle count unchanged (576
  // against 582 calls at two ranks) and made every call dearer, 94 ms against
  // 81, for 47 s against 44 at one rank.
  int amg_relax_type{3};
  // hypre's own ADS defaults are 10, 1, 3, 0.25, 0, 0. interp_type and Pmax
  // are the two that matter under a jumping coefficient: 6 (extended+i) with
  // Pmax = 4 truncates the interpolation to four entries a row, and on a 1e6
  // checkerboard at 16^3 that is the difference between converging and not.
  int amg_interp_type{0};
  int amg_pmax{0};
  int print_level{0};
  // One cycle by default; an inner Krylov is the option.
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
  // cost. What it is for is a jumping coefficient: at 16^3 on a checkerboard
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
    // Every space is renumbered onto the partition. hypre's IJ interface gives
    // each rank a contiguous run of global indices, and neither the unknowns
    // nor the complex's entities are contiguous per rank in their own
    // numbering. So the four spaces this needs -- the whole system, the faces
    // the flux sits on, the edges and the vertices -- are each sorted by
    // (owner, index), every matrix is stated in the new numbering, and a rank
    // inserts only its own rows.
    if (ranks > 1 && owners_.size() != A.n) {
      throw std::invalid_argument(
          "HypreSolver: a distributed run needs the owner of every unknown; "
          "call set_owners()");
    }
    if (ranks > 1 && !opts.mgr && norm.entity_owner.size() < 3) {
      throw std::invalid_argument(
          "HypreSolver: a distributed run needs the owner of every vertex, edge and face");
    }
    if (norm.empty()) throw std::invalid_argument("HypreSolver: the norm has no factors");
    if (opts.mgr && opts.mgr_relax_sweeps % 2 == 0) {
      throw std::invalid_argument(
          "HypreSolver: mgr_relax_sweeps must be odd; an even count leaves the "
          "F relaxation where it started and the reduction does not converge");
    }
    // MGR reduces A, not the Riesz map: it partitions the assembled system by
    // F/C markers and never forms A0, so A0's graph term, the complex and the
    // ADS hierarchies are unused under it.
    const bool need_ads = !opts.mgr;
    if (need_ads && (norm.discrete_gradient.empty() || norm.discrete_curl.empty())) {
      throw std::invalid_argument("HypreSolver: ADS needs the discrete gradient and curl");
    }
    const auto n = static_cast<HYPRE_BigInt>(A.n);
    const std::vector<int>& flux = norm.factors[0];
    const auto nf = static_cast<HYPRE_BigInt>(flux.size());
    // One moment per facet, or the subspace that is.
    //
    // ADS takes a scalar H(div) problem: one unknown per face, which is what
    // the curl's rows address. A facet carrying d moments -- derham_bdm,
    // stabilized_bdm -- reaches it through the facet-constant subspace, and
    // the block is then a two-level cycle whose coarse operator is where ADS
    // runs. `lowest_order` is that injection, a matrix of ones because the
    // constant moment is one of the unknowns.
    const bool two_level = need_ads && !norm.lowest_order.empty();
    if (need_ads && !two_level && static_cast<HYPRE_BigInt>(norm.discrete_curl.rows) != nf) {
      throw std::invalid_argument("HypreSolver: the curl's rows are not the first factor");
    }
    if (two_level && norm.lowest_order.cols != norm.discrete_curl.rows * norm.lowest_order_components) {
      throw std::invalid_argument(
          "HypreSolver: the coarse space is not the faces of the complex");
    }
    // d copies of the complex, for a stress.
    //
    // A flux is one H(div) field. A stress is d of them side by side -- the
    // rows of the tensor -- and they are coupled only through the material.
    // The coarse space is then d contiguous runs of the facet constants, and
    // each run is a scalar H(div) problem on the same de Rham complex: one
    // curl, one gradient, one set of coordinates, d solvers built on them.
    //
    // What the split drops is the coupling between components -- the rotation,
    // and the trace term of the compliance. Both are bounded independently of
    // h and of the material, so dropping them costs a constant in the count
    // and not a rate.
    const int copies = two_level ? norm.lowest_order_components : 1;
    if (copies < 1) throw std::invalid_argument("HypreSolver: a space with no copies");

    const auto t0 = std::chrono::steady_clock::now();

    // the four spaces, on the partition
    const Layout dofs = ranks > 1 ? layout_of(owners_, ranks, rank)
                                  : serial_layout(static_cast<int>(A.n));
    const Layout verts = !need_ads ? serial_layout(0)
                         : ranks > 1 ? layout_of(norm.entity_owner[0], ranks, rank)
                                     : serial_layout(norm.discrete_gradient.cols);
    const Layout edges = !need_ads ? serial_layout(0)
                         : ranks > 1 ? layout_of(norm.entity_owner[1], ranks, rank)
                                     : serial_layout(norm.discrete_gradient.rows);
    const Layout faces = !need_ads ? serial_layout(0)
                         : ranks > 1 ? layout_of(norm.entity_owner[2], ranks, rank)
                                     : serial_layout(norm.discrete_curl.rows);
    // The block's own numbering. With one moment a facet it is the faces --
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
    // The flux block is the face space -- one moment per facet, checked above --
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
    // A constrained unknown is not in the space. Its row of A is the
    // constraint, scale * e_i^T, and not a form. Leaving the norm's entries on
    // it preconditions an equation that is not the one being solved; P carries
    // the same row instead, so the unknown contributes the identity to P^-1 A
    // and drops out of the Krylov space. Omitted, this path loses contrast
    // robustness -- 16, 75, 1570, diverged over K = 1 .. 1e6 against the PETSc
    // path's 16, 14, 26, 29.
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
    // Which flux dofs the constraint touches. div is topological and reads only
    // a facet's constant moment, so these are the constants -- identified from
    // the operator rather than from an assumed dof ordering.
    std::vector<char> in_constraint(A.n, 0);
    // By magnitude, not by presence: a slot a row does not pair with is
    // structurally there and numerically zero -- measured at 3e-15 against 0.9
    // on the strong-symmetry stress -- so counting entries rather than weighing
    // them marks every dof.
    for (std::size_t r = 0; r < b_rows.size(); ++r) {
      double rmax = 0.0;
      for (const auto& [c3, v3] : b_rows[r]) rmax = std::max(rmax, std::abs(v3));
      for (const auto& [c2, v2] : b_rows[r]) {
        if (std::abs(v2) <= 1e-10 * rmax) continue;
        (void)v2;
        (void)v2;
        // c2 is the block numbering; old_of takes it to a position in `flux`,
        // and flux[] to the global unknown
        const auto pos = block_l.old_of.empty()
                             ? static_cast<std::size_t>(c2)
                             : static_cast<std::size_t>(block_l.old_of[(std::size_t)c2]);
        if (pos < flux.size()) in_constraint[(std::size_t)flux[pos]] = 1;
      }
    }

    if (need_ads) {
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
    }

    Mat a_full = to_mat(A, dofs);
    // Declared here, assigned under the gate: block.a0 and the ADS hierarchies
    // hold raw handles into these and are read below the gate, so their scope
    // has to outlive it.
    Block block;
    Mat a0, grad, curl, pi_rt, pi_nd;
    Vecs xyz;
    std::vector<Mat> inject;
    if (need_ads) {
    a0 = to_mat(m, block_l);
    grad = to_mat(norm.discrete_gradient, edges, verts);
    curl = to_mat(norm.discrete_curl, faces, edges);

    // The interpolations, when the space is not lowest order. With a BDM facet
    // ADS cannot build Pi from the coordinates -- that construction assumes one
    // unknown a facet -- so the caller supplies both. Their columns are the
    // vector nodal space, 3 to a vertex, which is a fourth partition and needs
    // its own owners when distributed.
    const bool supplied_pi = !norm.rt_interpolation.empty() && !norm.nd_interpolation.empty();
    if (supplied_pi) {
      if (norm.rt_interpolation.rows != norm.discrete_curl.rows ||
          norm.nd_interpolation.rows != norm.discrete_curl.cols ||
          norm.rt_interpolation.cols != norm.nd_interpolation.cols) {
        throw std::invalid_argument(
            "HypreSolver: the interpolations do not match the complex they interpolate into");
      }
      if (ranks > 1 &&
          static_cast<int>(norm.interpolation_owner.size()) != norm.rt_interpolation.cols) {
        throw std::invalid_argument(
            "HypreSolver: the interpolations are distributed but their columns have no owners");
      }
    }
    // The vector nodal space is a fourth partition, renumbered by the same
    // (owner, index) sort as the other three, so every rank agrees on it
    // without communicating; its local run is the column range of the
    // rectangular Pi matrices.
    const Layout vnodes =
        (supplied_pi && ranks > 1) ? layout_of(norm.interpolation_owner, ranks, rank)
                                   : serial_layout(supplied_pi ? norm.rt_interpolation.cols : 1);

    if (supplied_pi) {
      pi_rt = to_mat(norm.rt_interpolation, faces, vnodes);
      pi_nd = to_mat(norm.nd_interpolation, edges, vnodes);
    }

    // the metric, and the only metric ADS is told
    const auto nv = static_cast<HYPRE_BigInt>(norm.discrete_gradient.cols);
    if (norm.vertex_coordinates.size() !=
        static_cast<std::size_t>(nv) * static_cast<std::size_t>(norm.space_dim)) {
      throw std::invalid_argument("HypreSolver: one coordinate per vertex is required");
    }
    // The coordinates are shifted to the origin. ADS builds its vector
    // interpolation from the coordinate functions, so what it reads is their
    // linear part. A mesh in a projected coordinate system carries an offset
    // that dwarfs the domain -- an industrial mesh here spans 1e4 metres about
    // an origin 5.3e5 away, a ratio of 594 -- and the linear part is then a
    // rounding error on the constant. Subtracting the minimum leaves the span
    // of {1, x, y, z} unchanged.
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

    // The coarse space, where one is needed.
    //
    // The injection's rows are global unknowns and its columns the faces; both
    // are renumbered onto the partition, and the coarse operator is the
    // Galerkin product P^T A0 P -- the same operator seen on the subspace, with
    // nothing about the physics restated there. ADS then runs on one unknown
    // per face, which is what it is written for.
    //
    // One injection per copy. The columns are copy-major -- c * n_facet + f --
    // so copy c is the run [c*n_facet, (c+1)*n_facet) and splitting on the
    // column index is the whole of the component split. d narrow injections
    // rather than one wide one make P_c^T A0 P_c the diagonal sub-block
    // directly, which is the operator ADS takes.
    inject.resize(static_cast<std::size_t>(copies));
    if (two_level) {
      std::vector<std::vector<int>> ir(static_cast<std::size_t>(copies)), ic(
          static_cast<std::size_t>(copies));
      std::vector<std::vector<double>> iv(static_cast<std::size_t>(copies));
      for (std::size_t k = 0; k < norm.lowest_order.value.size(); ++k) {
        const int g = norm.lowest_order.row[k];           // a global unknown
        const int r = in_flux[static_cast<std::size_t>(g)];
        if (r < 0) continue;                              // not in the flux block
        const int col = norm.lowest_order.col[k];
        const int c = col / faces.total, f = col % faces.total;
        if (c < 0 || c >= copies) {
          throw std::invalid_argument("HypreSolver: the injection's column is not a copy of a face");
        }
        ir[static_cast<std::size_t>(c)].push_back(r);
        ic[static_cast<std::size_t>(c)].push_back(faces.new_of[static_cast<std::size_t>(f)]);
        iv[static_cast<std::size_t>(c)].push_back(norm.lowest_order.value[k]);
      }
      for (int c = 0; c < copies; ++c) {
        inject[static_cast<std::size_t>(c)] = build_mat(
            std::move(ir[static_cast<std::size_t>(c)]), std::move(ic[static_cast<std::size_t>(c)]),
            std::move(iv[static_cast<std::size_t>(c)]), block_l.total, faces.total, block_l.begin,
            block_l.end, faces.begin, faces.end);
      }
    }

    // The preconditioner works on local data: hypre hands apply() this rank's
    // rows of the outer vectors, and the block's vectors hold this rank's
    // faces. A dof and the face it sits on are owned by the same rank -- both
    // partitions come from the same exokal call -- so the gather flux_at
    // records is a local permutation and needs no communication.
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
    // A monolithic Pi forces a cycle below 10. hypre splits the cycle types at
    // 10: above, it wants the scalar triple Pix/Piy/Piz and errors out on a
    // monolithic one. 13 is the default because it is the best of the scalar
    // cycles for RT; with interpolations supplied the choice is not available.
    const int ads_cycle = supplied_pi && opts.ads_cycle_type > 10 ? 1 : opts.ads_cycle_type;
    // d solvers, one complex. Each copy of the H(div) space is a different
    // operator -- its own Galerkin product -- but the space is the same, so
    // the gradient, the curl, the coordinates and the interpolations are
    // handed to every one of them unchanged.
    block.ads.assign(static_cast<std::size_t>(copies), nullptr);
    for (int c = 0; c < copies; ++c) {
      HYPRE_Solver& a = block.ads[static_cast<std::size_t>(c)];
      HYPRE_ADSCreate(&a);
      HYPRE_ADSSetDiscreteGradient(a, mat_of(grad));
      HYPRE_ADSSetDiscreteCurl(a, mat_of(curl));
      HYPRE_ADSSetCoordinateVectors(a, vec_of(xyz[0]), vec_of(xyz[1]), vec_of(xyz[2]));
      // Set AFTER the coordinates: ADS builds Pi from them only when none is
      // given, and both being present is allowed -- the supplied one wins.
      if (supplied_pi) {
        HYPRE_ADSSetInterpolations(a, mat_of(pi_rt), nullptr, nullptr, nullptr, mat_of(pi_nd),
                                   nullptr, nullptr, nullptr);
      }
      HYPRE_ADSSetCycleType(a, ads_cycle);
      HYPRE_ADSSetPrintLevel(a, opts.print_level);
      HYPRE_ADSSetMaxIter(a, opts.ads_iterations);  // a preconditioner, not a solver
      HYPRE_ADSSetTol(a, 0.0);
      // the auxiliary hierarchies' parameters, which PETSc does not forward
      HYPRE_ADSSetAMGOptions(a, opts.amg_coarsen_type, opts.amg_agg_levels, opts.amg_relax_type,
                             opts.amg_theta, opts.amg_interp_type, opts.amg_pmax);
      HYPRE_ADSSetAMSOptions(a, supplied_pi ? 1 : 11, opts.amg_coarsen_type, opts.amg_agg_levels,
                             opts.amg_relax_type, opts.ams_theta, opts.amg_interp_type,
                             opts.amg_pmax);
    }

    block.a0 = mat_of(a0);
    block.rhs = zero_vec(block_l);
    block.sol = zero_vec(block_l);
    if (two_level) {
      // hypre builds a matrix's communication package lazily, and its internal
      // parallel routines -- the Galerkin product among them -- assume it is
      // already there. Absent, RAP walks a null offd map and segfaults, which
      // is what a 2-rank run did.
      if (hypre_ParCSRMatrixCommPkg(block.a0) == nullptr) hypre_MatvecCommPkgCreate(block.a0);
      block.resid = zero_vec(block_l);
      block.vtemp = zero_vec(block_l);
      block.ztemp = zero_vec(block_l);
      block.diag = block_diagonal(m, block_l);
      // the l1 norms the scaled smoother needs; without them relax type 8 is
      // silently the unscaled hybrid sweep again
      hypre_BoomerAMGRelaxComputeL1Norms(block.a0, 8, 0, 0, nullptr, &block.l1_norms);
      for (int c = 0; c < copies; ++c) {
        const auto k = static_cast<std::size_t>(c);
        block.p.push_back(mat_of(inject[k]));
        if (hypre_ParCSRMatrixCommPkg(block.p[k]) == nullptr) {
          hypre_MatvecCommPkgCreate(block.p[k]);
        }
        block.a_coarse.push_back(hypre_ParCSRMatrixRAP(block.p[k], block.a0, block.p[k]));
        if (hypre_ParCSRMatrixCommPkg(block.a_coarse[k]) == nullptr) {
          hypre_MatvecCommPkgCreate(block.a_coarse[k]);
        }
        block.c_rhs.push_back(zero_vec(faces));
        block.c_sol.push_back(zero_vec(faces));
        HYPRE_ADSSetup(block.ads[k], block.a_coarse[k], vec_of(block.c_rhs[k]),
                       vec_of(block.c_sol[k]));
      }
    } else {
      HYPRE_ADSSetup(block.ads[0], block.a0, vec_of(block.rhs), vec_of(block.sol));
    }

    // The inner Krylov on A0: FlexGMRES even where CG is admissible.
    //
    // A0 is SPD, and without two_level so is the preconditioner -- one ADS
    // cycle, cycle 13 = 01234543210, palindromic -- so CG applies. It is
    // nonetheless slower, by the stopping criterion rather than the work a
    // step: block_rtol = 1e-2 is read on ||r||_2, which FlexGMRES minimizes
    // and CG does not, CG being optimal in the A-norm of the error. At that
    // tolerance the two differ by the whole cost -- 93 s against 33 s on 5.2e5
    // cells at the same 6 outer iterations -- and FlexGMRES pays k(k+1)/2
    // inner products for the k steps it takes, not for KDim. Under two_level
    // the forward-only cycle omits the backward sweep, and CG is then
    // inadmissible rather than merely slower.
    if (opts.block_iterations > 0 && forward_only()) {
      block.inner_is_flex = true;
      HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &block.inner);
      HYPRE_FlexGMRESSetKDim(block.inner, opts.block_iterations);
      HYPRE_FlexGMRESSetMaxIter(block.inner, opts.block_iterations);
      HYPRE_FlexGMRESSetTol(block.inner, opts.block_rtol);
      HYPRE_FlexGMRESSetPrintLevel(block.inner, 0);
      if (two_level) {
        HYPRE_FlexGMRESSetPrecond(block.inner, cycle, cycle_setup,
                                  reinterpret_cast<HYPRE_Solver>(&block));
      } else {
        HYPRE_FlexGMRESSetPrecond(block.inner,
                                  reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSolve),
                                  reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSetup),
                                  block.ads[0]);
      }
      HYPRE_ParCSRFlexGMRESSetup(block.inner, block.a0, vec_of(block.rhs),
                                 vec_of(block.sol));
    } else if (opts.block_iterations > 0) {
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
                            reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_ADSSetup),
                            block.ads[0]);
      }
      HYPRE_ParCSRPCGSetup(block.inner, block.a0, vec_of(block.rhs), vec_of(block.sol));
    }
    }  // need_ads

    HYPRE_Solver ksp = nullptr;
    HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &ksp);
    HYPRE_FlexGMRESSetKDim(ksp, 50);
    HYPRE_FlexGMRESSetMaxIter(ksp, opts.max_iterations);
    HYPRE_FlexGMRESSetTol(ksp, opts.rtol);
    HYPRE_FlexGMRESSetPrintLevel(ksp, opts.print_level);
    Vec rhs = to_vec(b, dofs);
    Vec sol = to_vec(std::vector<double>(A.n, 0.0), dofs);
    HYPRE_Solver mgr = nullptr, mgr_amg = nullptr;
    std::vector<HYPRE_Int> mgr_marker, mgr_cidx;
    HYPRE_Int mgr_ncpts[1] = {0};
    HYPRE_Int* mgr_lvl[1] = {nullptr};
    if (opts.mgr) {
      // The markers, and why there are two levels when a facet carries more
      // than one moment.
      //
      // div is topological, so it reads only the constant moment of a facet:
      // D = [0 | D_0]. The higher moments lie in ker D, so a single reduction
      // of the whole flux asks F-relaxation to invert M on a divergence-free
      // subspace, which it cannot damp -- measured on stabilized_bdm as 71,
      // 1736 and then no convergence over three refinements where RT held 9,
      // 10, 11. Splitting them off first is meant to make level 0 exact on the
      // constraint, leaving the RT0 system.
      //
      // The two-level path has never converged. Measured 2026-09: flow
      // stabilized_bdm and derham_bdm both stall at the iteration cap, and so
      // does every weak-symmetry stress, which carries d moments a facet and so
      // always takes this branch. The RT products are one moment a facet and
      // take the one-level branch, and are the only MGR cases under test.
      //
      // Not a tuning failure. F-relaxation 0, 1, 2, 9 and 99 -- the last two
      // direct solves of the F block -- all stall, as do interpolation types 0
      // through 4 and reduction depths of one, two and three levels (the third
      // peeling the algebraic multiplier onto its own level). HYPRE_MGRSetup
      // reports no error. The outer residual does not decrease at all,
      // reduction factor 1.000000 a step against the strong stress's 0.987,
      // 0.950, 0.912: inert rather than weak, which reads as a
      // calling-convention mismatch and not a numerical one.
      //
      // To resume: hypre can print the dofmap it builds
      // (HYPRE_MGR_PRINT_FINE_MATRIX); comparing it against mgr_marker below
      // separates a wrong call from a wrong multi-level reduction. Until then a
      // facet carrying more than one moment should use ADS.
      //
      //   marker 0   the higher facet moments      F at level 0
      //   marker 1   the facet constants           F at level 1
      //   marker 2   the multipliers               C throughout
      //
      // The constants are read off norm.lowest_order, whose rows are them; with
      // one moment a facet there is no such injection and one level suffices.

      std::size_t n_touch = 0, n_flux_dofs = 0;
      for (std::size_t i = 0; i < A.n; ++i)
        if (in_flux[i] >= 0) { ++n_flux_dofs; n_touch += in_constraint[i] ? 1 : 0; }
      const bool split_moments = n_touch > 0 && n_touch < n_flux_dofs;
      if (std::getenv("MIMETIKA_MGR_MARKERS") != nullptr && rank == 0) {
        std::fprintf(stderr, "[mgr] flux %zu, touched by the constraint %zu -> %s\n",
                     n_flux_dofs, n_touch, split_moments ? "two levels" : "one level");
        std::fflush(stderr);
      }
      // By slot, not by row width. q = d(d+1)/2 dofs a facet in the order
      // {t1, t2, n^(x-x_f), n chi_0, n chi_1, n chi_2} and the same count a
      // cell in the order {alpha, omega}. Verified against known stress states:
      // a dilation puts 7.97 on slot 3 and 6e-13 on slots 0, 1, and slots 2, 4
      // and 5 vanish on any constant stress -- they are ker Pi_E.
      //
      //   mean traction {0, 1, 3}   what a translation reads
      //   ker Pi        {2, 4, 5}   zero mean over the facet, the stabilizer's
      //   u             {0,1,2}=alpha, {3,4,5}=omega
      //
      // A width heuristic got this wrong on warped hexahedral faces, where the
      // normal varies over the face and a constant test vector picks up every
      // slot.
      std::vector<char> is_rotation(A.n, 0), is_normal(A.n, 0);
      bool vem_split = false;
      if (opts.mgr_vem_split >= 1 && norm.space_dim > 0) {
        const auto d = static_cast<std::size_t>(norm.space_dim);
        const std::size_t q = d * (d + 1) / 2;
        if (flux.size() % q == 0 && norm.factors.size() > 1 &&
            norm.factors[1].size() % q == 0) {
          for (std::size_t k = 0; k < flux.size(); ++k) {
            const auto slot = k % q;
            const bool mean = slot < d - 1 || slot == d;  // {0,1} and {3} in 3D
            if (mean) is_normal[static_cast<std::size_t>(flux[k])] = 1;
          }
          for (std::size_t k = 0; k < norm.factors[1].size(); ++k) {
            if (k % q >= d) is_rotation[static_cast<std::size_t>(norm.factors[1][k])] = 1;
          }
          vem_split = true;
        }
        if (std::getenv("MIMETIKA_MGR_MARKERS") != nullptr && rank == 0) {
          std::size_t n_c = 0, n_r = 0;
          for (std::size_t k = 0; k < flux.size(); ++k) {
            n_c += is_normal[static_cast<std::size_t>(flux[k])] ? 1 : 0;
          }
          for (std::size_t i2 = 0; i2 < A.n; ++i2) n_r += is_rotation[i2] ? 1 : 0;
          std::fprintf(stderr,
                       "[mgr] vem split: stress %zu = %zu mean + %zu ker Pi; %zu rotations -> %s\n",
                       flux.size(), n_c, flux.size() - n_c, n_r, vem_split ? "two levels" : "off");
          std::fflush(stderr);
        }
      }
      const std::vector<char>& is_const = in_constraint;
      mgr_marker.assign(static_cast<std::size_t>(dofs.local), 0);
      for (std::size_t i = 0; i < A.n; ++i) {
        const int at = dofs.new_of[i];
        if (!dofs.owns(at)) continue;
        const int f = factor_of[i];
        HYPRE_Int mk2 = 0;
        if (vem_split && opts.mgr_vem_split == 5) {
          // ker Pi first, then the mean traction, and all of u stays coarse:
          // the C block is displacement only, hence definite, which is what
          // BoomerAMG needs and what every stress-in-C partition failed
          mk2 = f < 1 ? (is_normal[i] ? 1 : 0) : 2;
        } else if (vem_split) {
          if (f < 1) mk2 = is_normal[i] ? 2 : 0;       // sigma_n rides in C
          else mk2 = is_rotation[i] ? 1 : 2;           // u_r | u_t
        } else if (f >= 1) {
          mk2 = split_moments ? 2 : 1;                 // a multiplier
        } else if (split_moments) {
          mk2 = is_const[i] ? 1 : 0;                   // constant or higher
        }
        mgr_marker[static_cast<std::size_t>(at - dofs.begin)] = mk2;
      }
      const HYPRE_Int nfac = (split_moments || vem_split) ? 3 : 2;
      const HYPRE_Int n_lvl = (split_moments || vem_split) ? 2 : 1;
      // not static: the one-level branch writes into these, and a static would
      // carry that into the next solve in the same process
      mgr_cidx.clear();
      if (split_moments || vem_split) { mgr_cidx.push_back(1); mgr_cidx.push_back(2); }
      else { mgr_cidx.push_back(1); }
      std::vector<HYPRE_Int> lvl1_cidx{2};
      HYPRE_Int ncpts2[2] = {static_cast<HYPRE_Int>(mgr_cidx.size()), 1};
      HYPRE_Int* lvl2[2] = {mgr_cidx.data(), lvl1_cidx.data()};

      HYPRE_MGRCreate(&mgr);
      HYPRE_MGRSetCpointsByPointMarkerArray(mgr, nfac, n_lvl, ncpts2, lvl2,
                                            mgr_marker.data());
      HYPRE_MGRSetNonCpointsToFpoints(mgr, 1);
      HYPRE_MGRSetInterpType(mgr, opts.mgr_interp_type);
      HYPRE_MGRSetMaxIter(mgr, 1);   // a preconditioner, not a solver
      HYPRE_MGRSetTol(mgr, 0.0);
      HYPRE_MGRSetPrintLevel(mgr, opts.print_level);
      // one moment a facet is a scalar flux; more is a stress
      const bool one_moment =
          norm.discrete_curl.rows > 0 &&
          flux.size() == static_cast<std::size_t>(norm.discrete_curl.rows);
      const int frelax = opts.mgr_frelax >= 0 ? opts.mgr_frelax : (one_moment ? 0 : 2);
      HYPRE_MGRSetFRelaxMethod(mgr, frelax);
      HYPRE_MGRSetNumRelaxSweeps(mgr, opts.mgr_relax_sweeps);
      HYPRE_BoomerAMGCreate(&mgr_amg);
      HYPRE_BoomerAMGSetMaxIter(mgr_amg, opts.mgr_coarse_iterations);
      HYPRE_BoomerAMGSetTol(mgr_amg, opts.mgr_coarse_iterations > 1 ? 1e-10 : 0.0);
      HYPRE_BoomerAMGSetPrintLevel(mgr_amg, 0);
      HYPRE_BoomerAMGSetCoarsenType(mgr_amg, opts.amg_coarsen_type);
      HYPRE_BoomerAMGSetRelaxType(mgr_amg, opts.amg_relax_type);
      HYPRE_BoomerAMGSetStrongThreshold(mgr_amg, opts.amg_theta);
      HYPRE_BoomerAMGSetInterpType(mgr_amg, opts.amg_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(mgr_amg, opts.amg_pmax);
      HYPRE_MGRSetCoarseSolver(mgr, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, mgr_amg);
      // The lift, read off A. Each constraint row's flux support is a cell, so
      // the cell blocks come from the assembled matrix without asking the mesh.
      // hypre_MGRSolve runs on the hierarchy its own setup built, not on the
      // matrix the Krylov hands it, so setting MGR up on the lifted copy leaves
      // the outer operator untouched.
      Mat a_lift;
      if (opts.mgr_hydrostatic_lift > 0.0) {
        std::vector<double> bump(A.n, 0.0);
        std::vector<int> foff(static_cast<std::size_t>(nf) + 1, 0);
        for (std::size_t k = 0; k < A.nnz(); ++k) {
          const int r = in_flux[static_cast<std::size_t>(A.row[k])];
          if (r >= 0 && in_flux[static_cast<std::size_t>(A.col[k])] >= 0) ++foff[(std::size_t)r + 1];
        }
        for (HYPRE_BigInt i = 0; i < nf; ++i) foff[(std::size_t)i + 1] += foff[(std::size_t)i];
        std::vector<int> mcol(static_cast<std::size_t>(foff[(std::size_t)nf]));
        std::vector<double> mval(mcol.size());
        {
          std::vector<int> at = foff;
          for (std::size_t k = 0; k < A.nnz(); ++k) {
            const int r = in_flux[static_cast<std::size_t>(A.row[k])];
            const int c = in_flux[static_cast<std::size_t>(A.col[k])];
            if (r < 0 || c < 0) continue;
            const auto sl = static_cast<std::size_t>(at[(std::size_t)r]++);
            mcol[sl] = c;
            mval[sl] = A.value[k];
          }
        }
        std::vector<int> local(static_cast<std::size_t>(nf), -1);
        std::size_t lifted = 0;
        for (std::size_t r = 0; r < b_rows.size(); ++r) {
          if (b_rows[r].empty()) continue;
          std::vector<int> F;
          for (const auto& [f2, v2] : b_rows[r]) {
            (void)v2;
            if (f2 >= 0 && f2 < nf) F.push_back(f2);
          }
          const std::size_t n2 = F.size();
          if (n2 < 2) continue;
          for (std::size_t a2 = 0; a2 < n2; ++a2) local[(std::size_t)F[a2]] = static_cast<int>(a2);
          exokal::numerics::Dense Me(n2, n2);
          for (std::size_t a2 = 0; a2 < n2; ++a2) {
            for (int k = foff[(std::size_t)F[a2]]; k < foff[(std::size_t)F[a2] + 1]; ++k) {
              const int lj = local[(std::size_t)mcol[(std::size_t)k]];
              if (lj >= 0) Me(a2, static_cast<std::size_t>(lj)) += mval[(std::size_t)k];
            }
          }
          for (std::size_t a2 = 0; a2 < n2; ++a2) local[(std::size_t)F[a2]] = -1;
          const exokal::numerics::SymmetricEigen e2 = exokal::numerics::symmetric_eigen(Me);
          const double l1 = e2.values.front(), l2 = e2.values[1];
          if (!(l2 > 0.0) || l1 >= opts.mgr_hydrostatic_lift * l2) continue;
          const double gap = l2 - l1;
          for (std::size_t a2 = 0; a2 < n2; ++a2) {
            const double vi = e2.vectors(a2, 0);
            bump[static_cast<std::size_t>(flux[(std::size_t)F[a2]])] += gap * vi * vi;
          }
          ++lifted;
        }
        if (std::getenv("MIMETIKA_MGR_MARKERS") != nullptr && rank == 0) {
          std::fprintf(stderr, "[mgr] hydrostatic lift on %zu of %zu cells\n", lifted,
                       b_rows.size());
          std::fflush(stderr);
        }
        Triplets t2;
        for (std::size_t k = 0; k < A.nnz(); ++k) t2.add(A.row[k], A.col[k], A.value[k]);
        for (std::size_t i = 0; i < A.n; ++i) {
          if (bump[i] != 0.0) t2.add(static_cast<int>(i), static_cast<int>(i), bump[i]);
        }
        a_lift = to_mat(t2, dofs);
      }
      HYPRE_MGRSetup(mgr, opts.mgr_hydrostatic_lift > 0.0 ? mat_of(a_lift) : mat_of(a_full),
                     vec_of(rhs), vec_of(sol));
      HYPRE_FlexGMRESSetPrecond(ksp,
                                reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_MGRSolve),
                                reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_MGRSetup), mgr);
    } else {
      HYPRE_FlexGMRESSetPrecond(ksp, apply, setup, reinterpret_cast<HYPRE_Solver>(&block));
    }
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
    if (mgr != nullptr) HYPRE_MGRDestroy(mgr);
    if (mgr_amg != nullptr) HYPRE_BoomerAMGDestroy(mgr_amg);
    if (block.l1_norms != nullptr) hypre_TFree(block.l1_norms, HYPRE_MEMORY_HOST);
    if (block.inner != nullptr) {
      if (block.inner_is_flex) HYPRE_ParCSRFlexGMRESDestroy(block.inner);
      else HYPRE_ParCSRPCGDestroy(block.inner);
    }
    for (auto* c : block.a_coarse) {
      if (c != nullptr) hypre_ParCSRMatrixDestroy(c);
    }
    for (auto a : block.ads) HYPRE_ADSDestroy(a);
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

  // One SetValues call for every local row at once, not one per entry:
  // HYPRE_IJMatrixAddToValues per triplet is O(nnz) calls into the library and
  // is what made a 93k-cell industrial mesh -- 3.3 million entries -- appear to
  // hang. The triplets are summed by (row, col) here and handed over as whole
  // rows, the shape SetValues takes.
  //
  // rows/cols are already in hypre's numbering; [row_begin, row_end) is this
  // rank's row run and [col_begin, col_end) its column run.
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
    // The column range is the column space's own local run: hypre pairs a
    // rectangular matrix's columns with the vector it multiplies, so the
    // gradient's columns are partitioned like the vertices and the curl's like
    // the edges. Giving every rank all the columns fails on more than one rank.
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
    // one per copy of the H(div) space: a flux has one, a stress has d
    std::vector<HYPRE_Solver> ads;
    HYPRE_Solver inner{nullptr};
    bool inner_is_flex{false};
    HYPRE_Real* l1_norms{nullptr};  // for the l1-scaled smoother
    bool two_level{false};
    std::vector<HYPRE_ParCSRMatrix> p;         // the injection of each copy
    std::vector<HYPRE_ParCSRMatrix> a_coarse;  // P_c^T A0 P_c
    std::vector<Vec> c_rhs, c_sol;
    Vec resid;
    // the relaxation's own scratch. hypre uses Vtemp and Ztemp to stage the
    // off-rank part of a sweep, so they must not be the vector the cycle is
    // holding its residual in -- sharing them is invisible on one process and
    // corrupts on several.
    Vec vtemp, ztemp;
    std::vector<double> diag;              // of A0, for the smoother
    HYPRE_ParCSRMatrix a0{nullptr};
    Vec rhs, sol;
    int n_flux{0};
    // per local row of the outer vector: 1/W, and where it sits in the block
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

  // relax_type 8 is hypre's l1-scaled hybrid symmetric Gauss-Seidel: a forward
  // sweep and a backward one, so the composition stays symmetric and a CG may
  // use it. It relaxes u toward solving A0 u = f rather than replacing u.
  //
  // l1-scaled, not plain hybrid (type 6). "Hybrid" is Gauss-Seidel inside a
  // rank and Jacobi across ranks, so the smoother weakens as ranks are added
  // and the cycle weakens with it. Measured on hybrid_mesh_l_2, the inner CG
  // needed 3.87 applications of the cycle at one rank, 5.5 at two and 6.1 at
  // four -- which ate the whole parallel gain: the time per ADS call fell 78
  // to 45 ms while the number of calls rose 348 to 660. The l1 scaling is
  // convergent independently of the partition, which is why Kolev and
  // Vassilevski use it (section 5.2) and why the count is now flat in ranks.
  static void smooth(Block& b, HYPRE_ParVector f, HYPRE_ParVector u) {
    hypre_BoomerAMGRelax(b.a0, f, nullptr, 8, 0, 1.0, 1.0, b.l1_norms, u, vec_of(b.vtemp),
                         vec_of(b.ztemp));
  }

  // A two-level cycle: pre-smooth, restrict the residual, correct on the
  // coarse space, prolong, post-smooth.
  //
  //   smoother  a symmetric l1-scaled Gauss-Seidel sweep, not a point method
  //             -- see smooth(). What the coarse space does not carry is the
  //             non-constant moments, and those are the divergence-free
  //             directions, only the constant moment reaching div. Kolev and
  //             Vassilevski say that near-nullspace "cannot be handled by
  //             simple relaxation on the fine grid", and the measurement
  //             agrees: with damped Jacobi here the outer count ran 18, 24, 30
  //             over three refinements and failed at a contrast of 1e4, where
  //             a sweep is flat. A point smoother splits facet from facet and
  //             a div-free field is global.
  //   coarse    the facet constants, one H(div) problem, and that is ADS.
  //
  // The composition is symmetric only when the sweep over the copies runs both
  // ways. forward_only() is the default, so both the inner and the outer
  // Krylov method are FlexGMRES.
  static HYPRE_Int cycle(HYPRE_Solver s, HYPRE_Matrix, HYPRE_Vector rv, HYPRE_Vector xv) {
    Block& b = *reinterpret_cast<Block*>(s);
    auto* r = reinterpret_cast<HYPRE_ParVector>(rv);
    auto* x = reinterpret_cast<HYPRE_ParVector>(xv);
    double* xd = data_of(x);
    const double* rd = data_of(r);
    const std::size_t n = b.diag.size();

    for (std::size_t i = 0; i < n; ++i) xd[i] = 0.0;
    // The smoother is not redundant even where the copy split covers every
    // unknown of the block: that split is a block Gauss-Seidel over the copies,
    // and what it converges slowly on is the coupling between them -- the
    // rotation and the trace -- whose high-frequency part the sweep damps.
    // Dropped, the mechanics ladder stops converging: 9.4e-02 then 2.7e-01, a
    // rate of -1.8.
    //
    // One sweep, not two. Measured on the hybrid meshes the outer count is
    // unchanged at one sweep -- stabilized_vem 16 and 17 at levels 1 and 2,
    // stabilized_bdm 18 and 16 -- while the solve drops 2.18 s to 1.78 s and
    // 20.76 s to 16.45 s for vem, 29.63 s to 26.19 s for bdm. On strong
    // symmetry the facet slots the injection does not carry are a mass matrix,
    // cond 2.3 after diagonal scaling with no near-nullspace, so one sweep
    // already resolves them.
    const int sweeps = 1;
    for (int q = 0; q < sweeps; ++q) smooth(b, r, x);  // pre-smooth

    // Multiplicative over the copies. Additive -- every copy correcting the
    // same residual -- is symmetric and far too weak: on a weak-symmetry
    // stress it ran 44, 155, 1385 over three refinements where PETSc's
    // multiplicative split held 27, 26, 29. The copies are coupled through the
    // material, so a copy has to see what the ones before it did.
    //
    // A single forward sweep is not symmetric, and is the default all the
    // same -- see forward_only(). The backward sweep runs only when
    // MIMETIKA_ADS_SYMMETRIC_SWEEP restores it and there is more than one
    // copy; with one copy this is the plain two-level correction.
    const auto correct = [&](std::size_t c) {
      double* rs = data_of(vec_of(b.resid));
      for (std::size_t i = 0; i < n; ++i) rs[i] = rd[i];
      hypre_ParCSRMatrixMatvec(-1.0, b.a0, x, 1.0, vec_of(b.resid));
      hypre_ParCSRMatrixMatvecT(1.0, b.p[c], vec_of(b.resid), 0.0, vec_of(b.c_rhs[c]));
      double* cs = data_of(vec_of(b.c_sol[c]));
      const std::size_t ncoarse = static_cast<std::size_t>(
          hypre_VectorSize(hypre_ParVectorLocalVector(vec_of(b.c_sol[c]))));
      for (std::size_t i = 0; i < ncoarse; ++i) cs[i] = 0.0;
      HYPRE_ADSSolve(b.ads[c], b.a_coarse[c], vec_of(b.c_rhs[c]), vec_of(b.c_sol[c]));
      hypre_ParCSRMatrixMatvec(1.0, b.p[c], vec_of(b.c_sol[c]), 1.0, x);  // prolong and add
    };
    const std::size_t copies = b.ads.size();
    for (std::size_t c = 0; c < copies; ++c) correct(c);
    if (copies > 1 && !forward_only()) {
      for (std::size_t c = copies; c-- > 0;) correct(c);
    }

    for (int q = 0; q < sweeps; ++q) smooth(b, r, x);  // post-smooth
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
      if (b.inner_is_flex)
        HYPRE_ParCSRFlexGMRESSolve(b.inner, b.a0, vec_of(b.rhs), vec_of(b.sol));
      else
        HYPRE_ParCSRPCGSolve(b.inner, b.a0, vec_of(b.rhs), vec_of(b.sol));
    } else if (b.two_level) {
      cycle(reinterpret_cast<HYPRE_Solver>(&b), nullptr,
            reinterpret_cast<HYPRE_Vector>(vec_of(b.rhs)),
            reinterpret_cast<HYPRE_Vector>(vec_of(b.sol)));
    } else {
      HYPRE_ADSSolve(b.ads[0], b.a0, vec_of(b.rhs), vec_of(b.sol));
    }

    for (std::size_t i = 0; i < b.inv_w_local.size(); ++i) {
      x[i] = b.flux_at[i] >= 0 ? fx[static_cast<std::size_t>(b.flux_at[i])]
                               : r[i] * b.inv_w_local[i];
    }
    return 0;
  }
};

}  // namespace mimetika::solver
