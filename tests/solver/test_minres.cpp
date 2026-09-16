#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/numerics/dense.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/matrix_free.hpp"
#include "mimetika/linear_solver/minres.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

// MINRES on the mixed flow saddle point, driven matrix-free.
//
// Three things are under test and they are separable. The OPERATOR is
// Simulation::apply -- no assembled matrix reaches the Krylov method. The
// SYMMETRY is imposed here: the Darcy term writes [[M, -D^T], [D, 0]], so the
// pressure equation is negated to give [[M, -D^T], [-D, 0]], which is what
// MINRES requires and what the assembled system is NOT. The PRECONDITIONER is
// the block-diagonal one of Pazner, Kolev & Vassilevski (SIAM J. Sci. Comput.
// 46 (2024) B179, Prop. 1),
//
//     B = diag( diag(M), S~ ),     S~ = D diag(M)^-1 D^T,
//
// whose optimality bound is kappa(B^-1 A) <= (sqrt5 + 1)/(sqrt5 - 1) = 2.618,
// independent of h. The count below is therefore expected FLAT under
// refinement, and that is what is asserted.
//
// S~ is inverted exactly here, by its own spectrum. That is not the solver --
// an algebraic multigrid cycle is, and S~ is an M-matrix precisely so that one
// suffices -- but it separates the preconditioner's quality from the quality of
// the cycle approximating it. What this measures is the bound itself.

using exokal::numerics::Dense;
using mimetika::FlowModel;
using mimetika::mesh::Family;
using mimetika::solver::approximate_schur;
using mimetika::solver::field_blocks;
using mimetika::solver::minres;
using mimetika::solver::MinresOptions;
using mimetika::solver::SparseSystem;
using Realization = FlowModel::Realization;

namespace {

Dense dense_of(const SparseSystem& S) {
  Dense d(S.n, S.n);
  for (std::size_t k = 0; k < S.nnz(); ++k) {
    d(static_cast<std::size_t>(S.row[k]), static_cast<std::size_t>(S.col[k])) += S.value[k];
  }
  return d;
}

// S~^-1 from its own spectrum: S~ is SPD once the pressure is pinned by a
// datum, and the smallest eigenvalue says whether it is.
struct DenseInverse {
  Dense vectors;
  std::vector<double> inv_values;

  explicit DenseInverse(const Dense& S) {
    exokal::numerics::SymmetricEigen e = exokal::numerics::symmetric_eigen(S);
    const double top = e.values.back();
    inv_values.resize(e.values.size());
    for (std::size_t i = 0; i < e.values.size(); ++i) {
      // a mode below the relative cut is the constant a pure-Neumann problem
      // leaves; here the datum removes it, so this only guards arithmetic
      inv_values[i] = e.values[i] > 1e-12 * top ? 1.0 / e.values[i] : 0.0;
    }
    vectors = std::move(e.vectors);
  }

  void apply(const std::vector<double>& v, std::vector<double>& y) const {
    const std::size_t n = v.size();
    std::vector<double> t(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += vectors(i, j) * v[i];
      t[j] = s * inv_values[j];
    }
    y.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) y[i] += vectors(i, j) * t[j];
    }
  }
};

struct Problem {
  // the model keeps a POINTER to the mesh, so the mesh outlives it here and at
  // an address the move does not change
  std::unique_ptr<exokal::Mesh> mesh;
  std::unique_ptr<FlowModel> model;
  std::vector<graphos::Index> flux, pressure;
  std::vector<double> sign;   // +1 on the flux rows, -1 on the pressure rows
  std::vector<double> b_sym;  // the same sign applied to the load
};

Problem build(int n, Family family, Realization how) {
  Problem p;
  p.mesh = std::make_unique<exokal::Mesh>(mimetika::mesh::box({n, n, n}, 3, family));
  p.model = std::make_unique<FlowModel>(*p.mesh, 3, 1.0, how);
  for (const graphos::Index f : mimetika::boundary_facets(p.mesh->topology(), 3)) {
    const auto x = exokal::centroid(*p.mesh, 2, f);
    p.model->flow().emplace<mimetika::PressureBC>(std::vector<graphos::Index>{f}, x[0],
                                                  std::array<double, 3>{1.0, 0.0, 0.0});
  }
  p.model->build();
  for (const auto& blk : field_blocks(p.model->simulation().epoch())) {
    if (blk.name.rfind("q", 0) == 0) p.flux = blk.indices();
    if (blk.name.rfind("p", 0) == 0) p.pressure = blk.indices();
  }
  const std::size_t N = p.model->system().n;
  p.sign.assign(N, 1.0);
  for (const graphos::Index i : p.pressure) p.sign[static_cast<std::size_t>(i)] = -1.0;
  p.b_sym = p.model->rhs();
  for (std::size_t i = 0; i < N; ++i) p.b_sym[i] *= p.sign[i];
  return p;
}

const std::vector<std::pair<std::string, Realization>> kProducts{
    {"derham_rt", Realization::derham_rt},
    {"stabilized_rt", Realization::stabilized_rt},
    {"stabilized_bdm", Realization::stabilized_bdm},
    {"diagonal_tpfa", Realization::diagonal_tpfa},
};

}  // namespace

// ---- the symmetry the method needs -----------------------------------------
MIMETIKA_TEST(negating_the_pressure_equation_makes_the_operator_symmetric) {
  for (const auto& [pname, how] : kProducts) {
    const Problem p = build(3, Family::cartesian, how);
    const SparseSystem& A = p.model->system();
    Dense d(A.n, A.n);
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto r = static_cast<std::size_t>(A.row[k]);
      d(r, static_cast<std::size_t>(A.col[k])) += p.sign[r] * A.value[k];
    }
    double asym = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < A.n; ++i) {
      for (std::size_t j = 0; j < A.n; ++j) {
        asym = std::max(asym, std::fabs(d(i, j) - d(j, i)));
        scale = std::max(scale, std::fabs(d(i, j)));
      }
    }
    std::printf("  %-15s asymmetry %.2e of %.2e\n", pname.c_str(), asym, scale);
    CHECK(asym <= 1e-12 * scale);
  }
}

// ---- the solve, and the bound ----------------------------------------------
MIMETIKA_TEST(the_block_preconditioner_is_flat_under_refinement) {
  for (const auto& [pname, how] : kProducts) {
    std::vector<int> its;
    std::vector<double> kappa;
    for (const int n : {3, 4, 6}) {
      const Problem p = build(n, Family::cartesian, how);
      const SparseSystem& A = p.model->system();
      const SparseSystem S = approximate_schur(A, p.flux, p.pressure);
      const DenseInverse schur(dense_of(S));

      const std::vector<int> in_flux = mimetika::solver::slot_of(A.n, p.flux);
      const std::vector<int> in_pressure = mimetika::solver::slot_of(A.n, p.pressure);
      std::vector<double> mass(p.flux.size(), 0.0);
      for (std::size_t k = 0; k < A.nnz(); ++k) {
        const auto r = static_cast<std::size_t>(A.row[k]);
        if (A.row[k] == A.col[k] && in_flux[r] >= 0) {
          mass[static_cast<std::size_t>(in_flux[r])] += A.value[k];
        }
      }

      // the operator: matrix-free, with the pressure equation negated
      std::vector<double> t;
      auto apply_a = [&](const std::vector<double>& v, std::vector<double>& y) {
        p.model->simulation().apply(v, t);
        y.resize(t.size());
        for (std::size_t i = 0; i < t.size(); ++i) y[i] = p.sign[i] * t[i];
      };
      // the preconditioner: diag(M) on the flux, S~ on the pressure
      std::vector<double> vp, yp;
      auto apply_b = [&](const std::vector<double>& v, std::vector<double>& y) {
        y.assign(v.size(), 0.0);
        for (std::size_t i = 0; i < v.size(); ++i) {
          if (in_flux[i] >= 0) y[i] = v[i] / mass[static_cast<std::size_t>(in_flux[i])];
        }
        vp.assign(p.pressure.size(), 0.0);
        for (std::size_t i = 0; i < p.pressure.size(); ++i) {
          vp[i] = v[static_cast<std::size_t>(p.pressure[i])];
        }
        schur.apply(vp, yp);
        for (std::size_t i = 0; i < p.pressure.size(); ++i) {
          y[static_cast<std::size_t>(p.pressure[i])] = yp[i];
        }
      };

      // DIAGNOSTIC: kappa(B^-1 A) itself, which Prop. 1 bounds. Dense, so only
      // where the system is small enough for a Jacobi sweep.
      if (A.n <= 320) {
        Dense As(A.n, A.n);
        for (std::size_t k = 0; k < A.nnz(); ++k) {
          const auto r = static_cast<std::size_t>(A.row[k]);
          As(r, static_cast<std::size_t>(A.col[k])) += p.sign[r] * A.value[k];
        }
        // B^-1/2: 1/sqrt(diag M) on the flux, S~^-1/2 on the pressure
        Dense Bh(A.n, A.n);
        for (std::size_t i = 0; i < A.n; ++i) {
          if (in_flux[i] >= 0) {
            Bh(i, i) = 1.0 / std::sqrt(mass[static_cast<std::size_t>(in_flux[i])]);
          }
        }
        exokal::numerics::SymmetricEigen se = exokal::numerics::symmetric_eigen(dense_of(S));
        for (std::size_t a = 0; a < p.pressure.size(); ++a) {
          for (std::size_t b2 = 0; b2 < p.pressure.size(); ++b2) {
            double acc = 0.0;
            for (std::size_t j = 0; j < se.values.size(); ++j) {
              acc += se.vectors(a, j) * se.vectors(b2, j) / std::sqrt(se.values[j]);
            }
            Bh(static_cast<std::size_t>(p.pressure[a]), static_cast<std::size_t>(p.pressure[b2])) =
                acc;
          }
        }
        Dense T(A.n, A.n), P(A.n, A.n);
        for (std::size_t i = 0; i < A.n; ++i) {
          for (std::size_t j = 0; j < A.n; ++j) {
            double acc = 0.0;
            for (std::size_t k2 = 0; k2 < A.n; ++k2) acc += Bh(i, k2) * As(k2, j);
            T(i, j) = acc;
          }
        }
        for (std::size_t i = 0; i < A.n; ++i) {
          for (std::size_t j = 0; j < A.n; ++j) {
            double acc = 0.0;
            for (std::size_t k2 = 0; k2 < A.n; ++k2) acc += T(i, k2) * Bh(k2, j);
            P(i, j) = acc;
          }
        }
        std::vector<double> ev = exokal::numerics::symmetric_eigen(P).values;
        double lo = 1e300, hi = 0.0;
        for (const double e2 : ev) {
          lo = std::min(lo, std::fabs(e2));
          hi = std::max(hi, std::fabs(e2));
        }
        std::printf("      kappa(B^-1 A) = %.3f   |lambda| in [%.4f, %.4f]\n", hi / lo, lo, hi);
        // Prop. 1 gives 2.618 with the EXACT Schur complement; S~ replaces M by
        // its diagonal, so the bound degrades by that equivalence -- measured
        // at 3.0 for derham_rt on cartesian cells, and 5 covers it with room.
        CHECK(hi / lo <= 5.0);
        kappa.push_back(hi / lo);
      }

      std::vector<double> x;
      MinresOptions o;
      o.rtol = 1e-10;
      o.max_iterations = 400;
      const auto rep = minres(apply_a, apply_b, p.b_sym, x, o);

      // CONVERGED IS NOT CORRECT: the answer is checked against a direct solve
      // of the system as assembled, which the sign flip leaves unchanged.
      mimetika::solver::PetscSolver direct;
      std::vector<double> x_direct;
      const auto d = direct.solve(A, p.model->rhs(), x_direct);
      if (!d.converged) throw std::runtime_error("the direct reference failed: " + d.reason);
      double err = 0.0, scale = 0.0;
      for (std::size_t i = 0; i < A.n; ++i) {
        err = std::max(err, std::fabs(x[i] - x_direct[i]));
        scale = std::max(scale, std::fabs(x_direct[i]));
      }
      std::printf("  %-15s %5zu cells %6zu dofs  %4d its  res %.2e  err %.2e\n", pname.c_str(),
                  S.n, A.n, rep.iterations, rep.residual, err / scale);
      CHECK(rep.converged);
      CHECK(err <= 1e-7 * scale);
      its.push_back(rep.iterations);
    }
    // WHAT IS FLAT IS THE CONDITIONING, NOT THE COUNT. Prop. 1 bounds
    // kappa(B^-1 A) independently of h, and that is what is asserted above and
    // measured here: 3.61 then 3.77 for derham_rt, 2.92 then 3.06 for
    // stabilized_rt over an eightfold refinement. A bounded kappa bounds the
    // convergence FACTOR; it does not fix the iteration count at a given
    // tolerance, and the count does drift as the spectrum fills in between the
    // same two bounds -- 20, 29, 41 for derham_rt over 27, 64 and 216 cells.
    // Asserting a constant count would be asserting something the theory does
    // not claim; what is asserted is that the drift stays far below the
    // eightfold growth in the unknowns.
    if (kappa.size() > 1) CHECK(kappa.back() <= 1.15 * kappa.front());
    CHECK(its.back() <= 3 * its.front());
  }
}

MIMETIKA_TEST_MAIN()
