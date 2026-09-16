#include <cmath>
#include <cstdio>
#include <array>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/numerics/dense.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/matrix_free.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

// S~ = D diag(M)^-1 D^T, and the one property it rests on.
//
// The block preconditioner for the mixed flow saddle point is diag(M~, S~) with
// M~ = diag(M): its optimality (Pazner, Kolev & Vassilevski, SIAM J. Sci.
// Comput. 46 (2024) B179, Prop. 1) is uniform ONLY where M is spectrally
// equivalent to its diagonal. They obtain that from the
// interpolation-histopolation basis at high order. At lowest order it comes for
// free from the assembly instead: M is built from cell-local blocks of F x F,
// F the facets of a cell, and a facet is shared by two cells, so
//
//     cond(diag(M)^-1/2 M diag(M)^-1/2) <= 2 max_E cond(diag(M_E)^-1/2 M_E diag(M_E)^-1/2),
//
// a bound in the CELL SHAPE and not in h. What is measured below is that the
// quantity is indeed flat under refinement, per realization and per cell family
// -- the lowest-order counterpart of their Figure 2.

using exokal::numerics::Dense;
using mimetika::FlowModel;
using mimetika::solver::approximate_schur;
using mimetika::solver::field_blocks;
using mimetika::solver::SparseSystem;
using mimetika::mesh::Family;
using Realization = FlowModel::Realization;

namespace {

struct Blocks {
  SparseSystem A;
  std::vector<graphos::Index> flux, pressure;
  std::size_t cells{0};
};

// The linear pressure datum the flow examples use: the facet value and the
// gradient, so a facet carrying d moments tests the datum against all d.
Blocks build(int n, Family family, Realization how) {
  const exokal::Mesh mesh = mimetika::mesh::box({n, n, n}, 3, family);
  FlowModel model(mesh, 3, 1.0, how);
  for (const graphos::Index f : mimetika::boundary_facets(mesh.topology(), 3)) {
    const auto x = exokal::centroid(mesh, 2, f);
    model.flow().emplace<mimetika::PressureBC>(std::vector<graphos::Index>{f}, x[0],
                                               std::array<double, 3>{1.0, 0.0, 0.0});
  }
  model.build();
  Blocks b;
  b.A = model.system();
  b.cells = static_cast<std::size_t>(mesh.topology().count(3));
  // the space names a field "q_0" / "p_0": the stratum's index is part of it
  for (const auto& blk : field_blocks(model.simulation().epoch())) {
    if (blk.name.rfind("q", 0) == 0) b.flux = blk.indices();
    if (blk.name.rfind("p", 0) == 0) b.pressure = blk.indices();
  }
  if (b.flux.empty() || b.pressure.empty()) {
    throw std::runtime_error("the flow space did not name a flux and a pressure block");
  }
  return b;
}

Dense dense_block(const SparseSystem& A, const std::vector<graphos::Index>& rows,
                  const std::vector<graphos::Index>& cols) {
  const std::vector<int> ri = mimetika::solver::slot_of(A.n, rows);
  const std::vector<int> ci = mimetika::solver::slot_of(A.n, cols);
  Dense out(rows.size(), cols.size());
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const int r = ri[static_cast<std::size_t>(A.row[k])];
    const int c = ci[static_cast<std::size_t>(A.col[k])];
    if (r >= 0 && c >= 0) out(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) += A.value[k];
  }
  return out;
}

// cond of the Jacobi-preconditioned block: exactly what diag(M)^-1 stands in for
double diagonal_condition(const Dense& M) {
  const std::size_t n = M.rows();
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double d = std::fabs(M(i, i));
    s[i] = d > 0.0 ? 1.0 / std::sqrt(d) : 1.0;
  }
  Dense P(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) P(i, j) = s[i] * M(i, j) * s[j];
  }
  const std::vector<double> w = exokal::numerics::symmetric_eigen(P).values;
  return w.back() / w.front();
}

const std::vector<std::pair<std::string, Realization>> kProducts{
    {"derham_rt", Realization::derham_rt},
    {"stabilized_rt", Realization::stabilized_rt},
    {"derham_bdm", Realization::derham_bdm},
    {"stabilized_bdm", Realization::stabilized_bdm},
    {"diagonal_tpfa", Realization::diagonal_tpfa},
};

const std::vector<std::pair<std::string, Family>> kFamilies{
    {"cartesian", Family::cartesian}, {"simplex", Family::simplex}, {"prism", Family::prism}};

}  // namespace

// ---- 1. the assembled operator ---------------------------------------------
MIMETIKA_TEST(the_approximate_schur_is_a_symmetric_m_matrix) {
  for (const auto& [pname, how] : kProducts) {
    for (const auto& [fname, family] : kFamilies) {
      const Blocks b = build(3, family, how);
      const SparseSystem S = approximate_schur(b.A, b.flux, b.pressure);
      CHECK(S.n == b.cells);

      Dense d(S.n, S.n);
      for (std::size_t k = 0; k < S.nnz(); ++k) {
        d(static_cast<std::size_t>(S.row[k]), static_cast<std::size_t>(S.col[k])) += S.value[k];
      }
      double asym = 0.0, worst_off = 0.0, min_diag = 1e300, worst_sum = 0.0;
      std::size_t widest = 0;
      for (std::size_t i = 0; i < S.n; ++i) {
        double sum = 0.0;
        std::size_t width = 0;
        for (std::size_t j = 0; j < S.n; ++j) {
          asym = std::max(asym, std::fabs(d(i, j) - d(j, i)));
          sum += d(i, j);
          if (d(i, j) != 0.0) ++width;
          if (i != j) worst_off = std::max(worst_off, d(i, j));  // must be <= 0
        }
        min_diag = std::min(min_diag, d(i, i));
        worst_sum = std::min(worst_sum, sum);  // >= 0: weakly diagonally dominant
        widest = std::max(widest, width);
      }
      // WHAT THE MATRIX-FREE ROUTE DOES NOT ALLOCATE. A triplet costs two
      // indices and a double; the saddle point is the largest object a solve
      // builds and S~ is a few entries a cell, so the ratio is the memory the
      // route saves by assembling the preconditioner alone.
      const double bytes = static_cast<double>(2 * sizeof(graphos::Index) + sizeof(double));
      std::printf("  %-15s %-10s %4zu cells  widest row %2zu  min diag %.3e  "
                  "worst off %+.1e  worst row sum %+.1e   nnz(A) %7zu -> nnz(S~) %6zu "
                  "(%.1f%% of %.2f MiB)\n",
                  pname.c_str(), fname.c_str(), S.n, widest, min_diag, worst_off, worst_sum,
                  b.A.nnz(), S.nnz(),
                  100.0 * static_cast<double>(S.nnz()) / static_cast<double>(b.A.nnz()),
                  bytes * static_cast<double>(b.A.nnz()) / (1024.0 * 1024.0));
      // The preconditioner is smaller than the operator it stands for, and how
      // much smaller is the product's own property: measured 3.2 to 5.2 percent
      // of nnz(A) for the BDM members, 21 to 31 for the RT ones, and 48 to 54
      // for the two-point star -- whose A is already nearly diagonal, so there
      // is little to save and S~ is the exact Schur complement anyway.
      CHECK(S.nnz() < b.A.nnz());
      CHECK(asym < 1e-12);            // D diag^-1 D^T is symmetric by construction
      CHECK(worst_off <= 1e-12);      // an M-matrix has no positive off-diagonal
      CHECK(min_diag > 0.0);
      CHECK(worst_sum > -1e-10);      // the interior rows sum to zero, the boundary to more
    }
  }
}

// ---- 2. the assumption ------------------------------------------------------
MIMETIKA_TEST(the_flux_mass_is_spectrally_equivalent_to_its_diagonal) {
  for (const auto& [pname, how] : kProducts) {
    for (const auto& [fname, family] : kFamilies) {
      // The eigensolve is dense and Jacobi is cubic a sweep, so the ladder
      // stops where the flux block does: two points are enough to see whether
      // the quantity moves with h, and the claim is that it does not.
      constexpr std::size_t kCap = 400;
      std::vector<double> kappa;
      std::vector<std::size_t> size;
      for (const int n : {2, 3}) {
        const Blocks b = build(n, family, how);
        if (b.flux.size() > kCap) break;
        size.push_back(b.flux.size());
        kappa.push_back(diagonal_condition(dense_block(b.A, b.flux, b.flux)));
      }
      std::printf("  %-15s %-10s  cond(diag^-1/2 M diag^-1/2) =", pname.c_str(), fname.c_str());
      for (std::size_t i = 0; i < kappa.size(); ++i) {
        std::printf("  %8.3f (%zu dofs)", kappa[i], size[i]);
      }
      std::printf("\n");
      CHECK(!kappa.empty());
      // FLAT, not merely bounded: the bound is in the cell shape, and the two
      // meshes are the same shape refined.
      if (kappa.size() > 1) CHECK(kappa.back() <= 1.10 * kappa.front() + 1e-9);
    }
  }
}

// ---- 3. the operator -------------------------------------------------------
//
// THE MATRIX-FREE APPLY IS NOT NEW CODE. exokal's ActionSink runs the same term
// kernels at ad::Directional, so y = J(x)v costs about twice a residual and
// nothing of size n^2 is built; Simulation::apply wraps it and carries the
// constrained rows through apply_to_action. The problem is linear, so J(x) = A
// at every state and the two must agree to round-off -- which is what makes the
// assembled system disposable, not merely redundant.
MIMETIKA_TEST(the_matrix_free_apply_is_the_assembled_action) {
  for (const auto& [pname, how] : kProducts) {
    for (const auto& [fname, family] : kFamilies) {
      const exokal::Mesh mesh = mimetika::mesh::box({3, 3, 3}, 3, family);
      FlowModel model(mesh, 3, 1.0, how);
      for (const graphos::Index f : mimetika::boundary_facets(mesh.topology(), 3)) {
        const auto x = exokal::centroid(mesh, 2, f);
        model.flow().emplace<mimetika::PressureBC>(std::vector<graphos::Index>{f}, x[0],
                                                   std::array<double, 3>{1.0, 0.0, 0.0});
      }
      model.build();
      const SparseSystem& A = model.system();

      // a direction with no structure the operator could accidentally respect
      std::vector<double> v(A.n);
      for (std::size_t i = 0; i < A.n; ++i) v[i] = std::sin(1.0 + 2.7 * static_cast<double>(i));

      std::vector<double> y_free;
      model.simulation().apply(v, y_free);

      std::vector<double> y_asm(A.n, 0.0);
      for (std::size_t k = 0; k < A.nnz(); ++k) {
        y_asm[static_cast<std::size_t>(A.row[k])] +=
            A.value[k] * v[static_cast<std::size_t>(A.col[k])];
      }

      double worst = 0.0, scale = 0.0;
      for (std::size_t i = 0; i < A.n; ++i) {
        worst = std::max(worst, std::fabs(y_free[i] - y_asm[i]));
        scale = std::max(scale, std::fabs(y_asm[i]));
      }
      std::printf("  %-15s %-10s  n = %5zu   ||A_free v - A v||_inf / ||A v||_inf = %.3e\n",
                  pname.c_str(), fname.c_str(), A.n, worst / scale);
      CHECK(worst <= 1e-11 * scale);
    }
  }
}

MIMETIKA_TEST_MAIN()
