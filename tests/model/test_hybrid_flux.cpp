#include <cmath>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/hodge/hybrid_flux.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/hybrid_interface.hpp"
#include "mimetika/model/single_phase_model.hpp"

// THE HYBRIDIZED FLOW, HELD TO THE DENSE ORACLE AND TO THE MIXED ROUTE.
//
// The flux twin of test_hybrid_interface. Three things are asserted, in the
// order they depend on one another:
//
//   * the SPARSE interface system is exokal's DENSE one, entry for entry, on
//     every flux realization -- the sparsity clause (two multiplier blocks
//     couple iff their facets share a cell) is the whole content of the
//     assembler, and a wrong clause is invisible in any norm of an answer;
//   * it is symmetric positive definite once the boundary is pinned, which is
//     the reason to prefer this route: the condensed mixed system exists for
//     the diagonal star alone and the Riesz map pays for an H(div) block;
//   * the hybridized MODEL reproduces the mixed model's column -- the linear
//     pressure exactly -- with the roles swapped: the pressure datum pins a
//     multiplier, the sealed sides are free rows with nothing on them, and an
//     INFLOW datum loads a free row, which is what pins the sign of that load.

using graphos::Index;
using mimetika::SinglePhaseModel;
using mimetika::mesh::Family;
using Realization = exokal::hodge::FluxOperators::Realization;

namespace {

std::vector<double> dense_of(const mimetika::solver::SparseSystem& a) {
  std::vector<double> out(a.n * a.n, 0.0);
  for (std::size_t k = 0; k < a.nnz(); ++k) {
    out[static_cast<std::size_t>(a.row[k]) * a.n + static_cast<std::size_t>(a.col[k])] +=
        a.value[k];
  }
  return out;
}

exokal::hodge::HybridFluxOperators hybridize(const exokal::Mesh& mesh, int dim, Realization how) {
  const exokal::hodge::FluxOperators ops = exokal::hodge::FluxOperators::build(
      mesh, dim, exokal::hodge::Coefficient::uniform(1.0), how);
  return exokal::hodge::HybridFluxOperators::build(mesh, dim, ops);
}

MIMETIKA_TEST(the_sparse_interface_system_is_the_dense_one) {
  for (const int dim : {2, 3}) {
    for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
      const exokal::Mesh m = mimetika::mesh::box({2, 2, dim == 3 ? 2 : 1}, dim, family);
      for (const Realization how : {Realization::derham_bdm, Realization::derham_rt,
                                    Realization::stabilized_rt, Realization::diagonal_tpfa,
                                    Realization::adaptive_rt}) {
        const auto hops = hybridize(m, dim, how);
        const std::vector<char> free = mimetika::hybrid_free_facets(m, dim, hops);
        const exokal::numerics::Dense oracle =
            exokal::hodge::hybrid_interface_system(m, dim, hops, &free);
        const mimetika::solver::SparseSystem sparse =
            mimetika::hybrid_interface_sparse(m, dim, hops, free);
        CHECK(sparse.n == oracle.rows());
        const std::vector<double> d = dense_of(sparse);
        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < oracle.rows(); ++i) {
          for (std::size_t j = 0; j < oracle.cols(); ++j) {
            worst = std::max(worst, std::abs(d[i * oracle.cols() + j] - oracle(i, j)));
            scale = std::max(scale, std::abs(oracle(i, j)));
          }
        }
        CHECK(worst <= 1e-12 * scale);
      }
    }
  }
}

MIMETIKA_TEST(the_interface_system_is_symmetric_positive_definite) {
  const exokal::Mesh m = mimetika::mesh::box({3, 3, 3}, 3, Family::prism);
  for (const Realization how : {Realization::stabilized_rt, Realization::adaptive_rt,
                                Realization::diagonal_tpfa}) {
    const auto hops = hybridize(m, 3, how);
    const std::vector<char> free = mimetika::hybrid_free_facets(m, 3, hops);
    const mimetika::solver::SparseSystem S = mimetika::hybrid_interface_sparse(m, 3, hops, free);
    const std::vector<double> d = dense_of(S);
    double asym = 0.0;
    for (std::size_t i = 0; i < S.n; ++i) {
      for (std::size_t j = 0; j < S.n; ++j) asym = std::max(asym, std::abs(d[i * S.n + j] - d[j * S.n + i]));
    }
    CHECK(asym < 1e-12);
    // positive definite: a conjugate gradient converges on it, and to the
    // right answer -- the residual check inside the solver is the proof
    mimetika::solver::SolverOptions o;
    o.method = "cg";
    o.preconditioner = "hypre";
    o.rtol = 1e-12;
    o.condense = false;
    mimetika::solver::PetscSolver petsc(o);
    std::vector<double> b(S.n, 1.0), x;
    const auto rep = petsc.solve(S, b, x);
    CHECK(rep.converged);
    CHECK(rep.residual < 1e-9);
  }
}

// the column: p prescribed at both ends, the sides sealed, the linear answer
struct Column {
  double max_err{0.0};
  double jump{0.0};
  std::size_t multipliers{0};
};

Column column(int dim, Family family, Realization how, bool hybrid, bool inflow) {
  const double h = 1.0, p_hi = 2.0, p_lo = 1.0;
  const exokal::Mesh m = mimetika::mesh::column(4, dim, family, h);
  const int axis = dim - 1;
  std::vector<Index> top, base, side;
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    const double z = exokal::centroid(m, dim - 1, f)[static_cast<std::size_t>(axis)];
    if (std::abs(z - h) < 1e-9) {
      top.push_back(f);
    } else if (std::abs(z) < 1e-9) {
      base.push_back(f);
    } else {
      side.push_back(f);
    }
  }
  SinglePhaseModel prob(m, dim, 1.0, how);
  prob.flow().emplace<mimetika::NormalFluxBC>(side);
  prob.flow().emplace<mimetika::PressureBC>(base, p_lo);
  // the exact gradient of p = p_lo + (p_hi - p_lo) z / h, and the flux it
  // drives: q = -grad p, so the CANONICAL normal flux on the top, whose
  // canonical normal points +z on these meshes, is -(p_hi - p_lo)/h
  const double grad = (p_hi - p_lo) / h;
  if (inflow) {
    // the datum is stated against each facet's CANONICAL normal, which the
    // complex orients as it stores it -- +z on some top facets, -z on others
    // -- so the exact flux -grad e_z is projected facet by facet
    for (const Index f : top) {
      const mimetika::FacetFrame fr =
          mimetika::FacetFrame::of(m, dim, mimetika::cofacet_of(m, dim, f), f);
      prob.flow().emplace<mimetika::NormalFluxBC>(
          std::vector<Index>{f}, -grad * fr.normal[static_cast<std::size_t>(axis)]);
    }
  } else {
    prob.flow().emplace<mimetika::PressureBC>(top, p_hi);
  }
  prob.build();

  Column out;
  if (hybrid) {
    mimetika::solver::SolverOptions o;
    o.method = "cg";
    o.preconditioner = "hypre";
    o.rtol = 1e-12;
    o.condense = false;
    mimetika::solver::PetscSolver petsc(o);
    const auto rep = prob.hybridized(petsc);
    if (!rep.solve.converged) throw std::runtime_error("hybrid column: " + rep.solve.reason);
    out.jump = rep.jump;
    out.multipliers = rep.multipliers;
  } else {
    mimetika::solver::PetscSolver petsc;
    std::vector<double> x;
    const auto rep = petsc.solve(prob.system(), prob.rhs(), x);
    if (!rep.converged) throw std::runtime_error("mixed column: " + rep.reason);
    prob.accept(x);
  }
  for (Index e = 0; e < static_cast<Index>(prob.n_cells()); ++e) {
    const double z = exokal::centroid(m, dim, e)[static_cast<std::size_t>(axis)];
    out.max_err = std::max(out.max_err, std::abs(prob.cell_pressure(e) - (p_lo + grad * z)));
    const auto q = prob.cell_flux(e);
    out.max_err = std::max(out.max_err, std::abs(q[static_cast<std::size_t>(axis)] + grad));
  }
  return out;
}

MIMETIKA_TEST(the_hybridized_column_is_the_mixed_column) {
  for (const int dim : {2, 3}) {
    for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
      for (const Realization how : {Realization::derham_rt, Realization::stabilized_rt,
                                    Realization::adaptive_rt}) {
        const Column mixed = column(dim, family, how, false, false);
        const Column hyb = column(dim, family, how, true, false);
        CHECK(mixed.max_err < 1e-9);
        CHECK(hyb.max_err < 1e-9);
        CHECK(hyb.jump < 1e-9);
        CHECK(hyb.multipliers > 0);
      }
    }
    // and the two-point star where it claims exactness
    CHECK(column(dim, Family::cartesian, Realization::diagonal_tpfa, true, false).max_err < 1e-9);
  }
}

MIMETIKA_TEST(an_inflow_datum_loads_the_free_row_with_the_right_sign) {
  // the top is given the exact canonical flux instead of its pressure: the
  // natural datum of the hybridized form, on a free row. A wrong sign gives
  // the mirror-image column and fails here by 2(p_hi - p_lo).
  for (const int dim : {2, 3}) {
    for (const Family family : {Family::cartesian, Family::simplex}) {
      const Column hyb = column(dim, family, Realization::stabilized_rt, true, true);
      CHECK(hyb.max_err < 1e-9);
      const Column mixed = column(dim, family, Realization::stabilized_rt, false, true);
      CHECK(mixed.max_err < 1e-9);
    }
  }
}

}  // namespace

MIMETIKA_TEST_MAIN()
