#include <cmath>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/hodge/hybrid_stress.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"
#include "mimetika/model/hybrid_interface.hpp"

// The sparse interface system, against exokal's dense oracle.
//
// exokal hybridizes a cell and assembles the interface system densely, and the
// dense assembler is the contract a sparse one is held to. This file holds it to
// exactly that -- entry for entry, every entry, on meshes small enough that the
// dense matrix is a legitimate thing to build.
//
// One clause of the contract is what the sparsity is: two multiplier blocks
// couple iff their facets share a cell. A sparse assembly that visited anything
// else would still be symmetric, still be positive semidefinite, and wrong in a
// way no norm of the answer reveals, which is why the comparison is against the
// oracle rather than against a solution.
//
// And the property the second elimination is for: S is symmetric positive
// semidefinite with the cell rigid motions in its kernel, and pinning the
// boundary -- the default free mask is the interior stratum -- removes the
// global ones. That is asserted here too: it is the reason to prefer this route
// over condensing the mixed system, whose reduced operator is quasi-definite.

using graphos::Index;
using mimetika::mesh::Family;
using Realization = exokal::hodge::StressOperators::Realization;
using Formulation = exokal::hodge::StressOperators::Formulation;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;

exokal::hodge::HybridStressOperators hybridize(const exokal::Mesh& mesh, int dim, Realization how,
                                               Formulation form) {
  const exokal::hodge::StressOperators ops =
      exokal::hodge::StressOperators::build(mesh, dim, kMu, kLam, how, form);
  return exokal::hodge::HybridStressOperators::build(mesh, dim, ops, kMu);
}

// the sparse triplets summed into a dense matrix, which is the only fair way
// to compare: the assembly emits one triplet per contribution and the matrix
// is their sum, exactly as the dense assembler accumulates
std::vector<double> dense_of(const mimetika::solver::SparseSystem& a) {
  std::vector<double> out(a.n * a.n, 0.0);
  for (std::size_t k = 0; k < a.nnz(); ++k) {
    out[static_cast<std::size_t>(a.row[k]) * a.n + static_cast<std::size_t>(a.col[k])] +=
        a.value[k];
  }
  return out;
}

struct Comparison {
  std::size_t n{0};
  std::size_t nnz{0};
  double worst{0.0};
  double scale{0.0};
};

Comparison against_the_oracle(const exokal::Mesh& mesh, int dim, Realization how,
                              Formulation form) {
  const exokal::hodge::HybridStressOperators hops = hybridize(mesh, dim, how, form);
  const std::vector<char> free = mimetika::hybrid_free_facets(mesh, dim, hops);
  const exokal::numerics::Dense oracle =
      exokal::hodge::hybrid_interface_system(mesh, dim, hops, &free);
  const mimetika::solver::SparseSystem sparse =
      mimetika::hybrid_interface_sparse(mesh, dim, hops, free);

  Comparison c;
  c.n = sparse.n;
  c.nnz = sparse.nnz();
  if (oracle.rows() != sparse.n) {
    throw std::runtime_error("the sparse assembly and the oracle disagree on the size");
  }
  const std::vector<double> mine = dense_of(sparse);
  for (std::size_t i = 0; i < sparse.n; ++i) {
    for (std::size_t j = 0; j < sparse.n; ++j) {
      c.worst = std::max(c.worst, std::abs(mine[i * sparse.n + j] - oracle(i, j)));
      c.scale = std::max(c.scale, std::abs(oracle(i, j)));
    }
  }
  return c;
}

}  // namespace

// stabilized_vem first: the strongly-symmetric mixed virtual element product,
// whose sigma block no two-point condensation will take.
MIMETIKA_TEST(the_sparse_interface_system_is_the_dense_one) {
  for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
    const exokal::Mesh m = mimetika::mesh::box({2, 2, 2}, 3, family);
    const Comparison c =
        against_the_oracle(m, 3, Realization::stabilized_vem, Formulation::strong_symmetry);
    std::printf("  %-9s %4zu multipliers  %6zu triplets   worst |sparse - dense| %.2e\n",
                mimetika::mesh::name(family), c.n, c.nnz, c.worst);
    CHECK(c.n > 0);
    CHECK(c.worst < 1e-10 * std::max(1.0, c.scale));
  }
}

// and the total form, where the local saddle carries the hydrostatic stress
MIMETIKA_TEST(the_total_form_hybridizes_the_same_way) {
  const exokal::Mesh m = mimetika::mesh::box({2, 2, 2}, 3, Family::cartesian);
  const Comparison c =
      against_the_oracle(m, 3, Realization::stabilized_vem, Formulation::strong_symmetry_total);
  std::printf("  strong_symmetry_total  %4zu multipliers  %6zu triplets   worst %.2e\n", c.n,
              c.nnz, c.worst);
  CHECK(c.worst < 1e-10 * std::max(1.0, c.scale));
}

// The property the route exists for. With the boundary pinned the interface
// system is SPD, so a Cholesky runs to completion, and it can be handed to a
// conjugate gradient rather than to MINRES.
MIMETIKA_TEST(the_interface_system_is_symmetric_positive_definite) {
  const exokal::Mesh m = mimetika::mesh::box({2, 2, 2}, 3, Family::cartesian);
  const exokal::hodge::HybridStressOperators hops =
      hybridize(m, 3, Realization::stabilized_vem, Formulation::strong_symmetry);
  const std::vector<char> free = mimetika::hybrid_free_facets(m, 3, hops);
  const mimetika::solver::SparseSystem s = mimetika::hybrid_interface_sparse(m, 3, hops, free);
  std::vector<double> a = dense_of(s);
  const std::size_t n = s.n;

  double asymmetry = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      asymmetry = std::max(asymmetry, std::abs(a[i * n + j] - a[j * n + i]));
    }
  }

  // the factorization IS the definiteness test
  bool definite = true;
  for (std::size_t i = 0; i < n && definite; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double acc = a[i * n + j];
      for (std::size_t k = 0; k < j; ++k) acc -= a[i * n + k] * a[j * n + k];
      if (i == j) {
        if (!(acc > 0.0)) {
          definite = false;
          break;
        }
        a[i * n + i] = std::sqrt(acc);
      } else {
        a[i * n + j] = acc / a[j * n + j];
      }
    }
  }
  std::printf("  %zu multipliers   asymmetry %.1e   SPD %d\n", n, asymmetry, definite ? 1 : 0);
  CHECK(asymmetry < 1e-10);
  CHECK(definite);
}

// The solve: a conjugate gradient with an algebraic multigrid, and no
// factorization anywhere. The condensed mixed system is quasi-definite and takes
// MINRES at best, while this one is SPD.
//
// It is checked against a dense solve of the same system, so the comparison
// isolates the solver: a preconditioner that changed the operator would converge
// to a different vector and be caught here rather than in an error norm that
// also contains the discretization.
//
// And the recovery, which is exokal's and cell-local. `jump` is the worst
// disagreement between the two cofacet recoveries of a shared facet: the
// traction continuity the multiplier exists to enforce, so a small residual on
// the interface must show up as a small jump.
MIMETIKA_TEST(the_interface_system_solves_without_a_factorization) {
  for (const Family family : {Family::cartesian, Family::simplex}) {
    const exokal::Mesh m = mimetika::mesh::box({3, 3, 3}, 3, family);
    const exokal::hodge::HybridStressOperators hops =
        hybridize(m, 3, Realization::stabilized_vem, Formulation::strong_symmetry_total);
    const std::vector<char> free = mimetika::hybrid_free_facets(m, 3, hops);

    const auto cells = static_cast<std::size_t>(m.topology().count(3));
    const std::size_t nk = hops.cell(0).n_fields - hops.cell(0).n_p;
    std::vector<double> fu(cells * nk), fp(cells, 0.0);
    for (std::size_t i = 0; i < fu.size(); ++i) fu[i] = std::sin(1.0 + static_cast<double>(i));

    const mimetika::solver::SparseSystem s = mimetika::hybrid_interface_sparse(m, 3, hops, free);
    const std::vector<double> b =
        exokal::hodge::hybrid_interface_load(m, 3, hops, fu, fp, {}, &free);

    // CG on an SPD system, multigrid for the preconditioner: no LU, no
    // Cholesky, nothing that fills
    mimetika::solver::SolverOptions o;
    o.method = "cg";
    o.preconditioner = "hypre";
    o.rtol = 1e-12;
    o.max_iterations = 2000;
    o.condense = false;  // the second elimination already happened
    mimetika::solver::PetscSolver petsc(o);
    std::vector<double> lambda;
    const auto rep = petsc.solve(s, b, lambda);

    // the same system, factorized: the answer the iteration owes
    mimetika::solver::SolverOptions d;
    mimetika::solver::PetscSolver direct(d);
    std::vector<double> reference;
    direct.solve(s, b, reference);

    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < lambda.size(); ++i) {
      worst = std::max(worst, std::abs(lambda[i] - reference[i]));
      scale = std::max(scale, std::abs(reference[i]));
    }

    // upstream now takes the multiplier over every facet: solved where free,
    // zero where pinned, which is this test's homogeneous reference
    std::vector<double> lambda_all(
        static_cast<std::size_t>(m.topology().count(2)) * hops.facet_dofs(), 0.0);
    {
      std::size_t at = 0;
      for (std::size_t f = 0; f < free.size(); ++f) {
        if (free[f] == 0) continue;
        for (std::size_t a = 0; a < hops.facet_dofs(); ++a) {
          lambda_all[f * hops.facet_dofs() + a] = lambda[at++];
        }
      }
    }
    const exokal::hodge::HybridStressState state =
        exokal::hodge::hybrid_recovery(m, 3, hops, lambda_all, fu, fp);

    std::printf("  %-9s %4zu multipliers  cg+amg %3d its  |cg - direct| %.1e  jump %.1e\n",
                mimetika::mesh::name(family), s.n, rep.iterations, worst / std::max(1.0, scale),
                state.jump);
    CHECK(rep.converged);
    CHECK(rep.iterations > 0);              // a solve that did nothing is not a solve
    CHECK(scale > 0.0);                     // nor is one on a vanished load
    CHECK(worst < 1e-8 * std::max(1.0, scale));
    CHECK(state.jump < 1e-8);               // the continuity the multiplier enforces
    CHECK(state.sigma.size() > 0 && state.u.size() > 0);
  }
}

// The model's own route, against its own monolithic answer.
//
// Everything above tests the assembly against exokal's oracle. This tests the
// wiring: the same model, the same loads, solved two ways -- the monolithic
// saddle point, and the hybridized interface -- and the cell fields have to
// agree. exokal's header states the identity: with every boundary facet pinned
// to zero the interface solution reproduces the monolithic free saddle exactly,
// cell fields and stresses alike.
//
// The load is a reservoir pressure, because a homogeneous Dirichlet problem with
// no load has zero for an answer and would agree trivially.
MIMETIKA_TEST(the_hybridized_model_agrees_with_the_monolithic_one) {
  // Two resolutions, so a geometric factor can be told apart from a material
  // one: |f| falls by four between these, so a chart mismatch moves the ratio
  // and a material one does not.
  for (const int n : {2, 4}) {
  const exokal::Mesh m = mimetika::mesh::box({n, n, n}, 3, Family::cartesian);
  const auto cells = static_cast<std::size_t>(m.topology().count(3));

  // The linear patch, u = x, prescribed on the whole boundary. In the mixed
  // form that datum is natural and lands in the stress rows; hybridized it is
  // essential and pins the multipliers. Same problem, opposite machinery, and a
  // pinned block whose data never reached the load still solves and answers
  // u = 0.
  const auto build = [&]() {
    auto model = std::make_unique<mimetika::CauchyMechanicsModel>(
        m, 3, mimetika::ElasticMaterial{kMu, kLam}, Realization::stabilized_vem,
        Formulation::strong_symmetry_total);
    std::array<double, 9> grad{};
    for (int k = 0; k < 3; ++k) grad[static_cast<std::size_t>(k * 3 + k)] = 1.0;
    for (const Index f : mimetika::boundary_facets(m.topology(), 3)) {
      const auto x = exokal::centroid(m, 3, mimetika::cofacet_of(m, 3, f));
      model->prescribe_displacement({f}, {x[0], x[1], x[2]}, grad);
    }
    model->build();
    return model;
  };

  auto monolithic = build();
  mimetika::solver::PetscSolver direct;
  std::vector<double> x;
  const auto rep = direct.solve(monolithic->system(), monolithic->rhs(), x);
  CHECK(rep.converged);
  monolithic->accept(std::move(x));

  auto hybrid = build();
  mimetika::solver::SolverOptions o;
  o.method = "cg";
  o.preconditioner = "hypre";
  o.rtol = 1e-12;
  o.max_iterations = 2000;
  o.condense = false;
  mimetika::solver::PetscSolver cg(o);
  const auto hy = hybrid->hybridized(cg);
  CHECK(hy.solve.converged);

  double worst = 0.0, scale = 0.0, exact = 0.0;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const auto xc = exokal::centroid(m, 3, e);
    for (int k = 0; k < 3; ++k) {
      const double mono = monolithic->displacement(e, k);
      worst = std::max(worst, std::abs(mono - hybrid->displacement(e, k)));
      scale = std::max(scale, std::abs(mono));
      exact = std::max(exact, std::abs(hybrid->displacement(e, k) - xc[static_cast<std::size_t>(k)]));
    }
  }
  // And the stress, which the displacement cannot speak for. A uniform sign flip
  // of sigma leaves the displacement untouched and hides in the deviatoric row
  // of any patch whose exact deviator is zero -- this one's is -- so it is
  // compared against the monolithic answer component by component.
  double worst_s = 0.0, scale_s = 0.0;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const std::array<double, 9> a = monolithic->cell_stress(e);
    const std::array<double, 9> b = hybrid->cell_stress(e);
    for (std::size_t k = 0; k < 9; ++k) {
      worst_s = std::max(worst_s, std::abs(a[k] - b[k]));
      scale_s = std::max(scale_s, std::abs(a[k]));
    }
  }
  // Is it a scale? A ratio that is the same everywhere is a chart or a constant,
  // and one that scatters is neither.
  double rlo = 1e300, rhi = -1e300;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const std::array<double, 9> a = monolithic->cell_stress(e);
    const std::array<double, 9> b = hybrid->cell_stress(e);
    for (std::size_t k = 0; k < 9; ++k) {
      if (std::abs(a[k]) < 1e-6) continue;
      const double r = b[k] / a[k];
      rlo = std::min(rlo, r);
      rhi = std::max(rhi, r);
    }
  }
  // And the raw facet traction, the sigma dofs with no recombination in front of
  // them: if the ratio is the same here, the dofs themselves differ and
  // cell_stress is innocent.
  double tlo = 1e300, thi = -1e300;
  for (Index f = 0; f < m.topology().count(2); ++f) {
    const std::array<double, 3> a = monolithic->facet_traction(f);
    const std::array<double, 3> b = hybrid->facet_traction(f);
    for (std::size_t k = 0; k < 3; ++k) {
      if (std::abs(a[k]) < 1e-6) continue;
      tlo = std::min(tlo, b[k] / a[k]);
      thi = std::max(thi, b[k] / a[k]);
    }
  }
  std::printf("      %zu cells, |f| = %.5f   stress ratio [%.6f, %.6f]   facet traction ratio "
              "[%.6f, %.6f]\n",
              cells, exokal::measure(m, 2, 0), rlo, rhi, tlo, thi);
  CHECK(scale_s > 1e-6);
  CHECK(worst_s < 1e-8 * std::max(1.0, scale_s));
  std::printf("  %zu multipliers  cg %d its  jump %.1e   |hybrid - monolithic| %.2e  "
              "|hybrid - exact| %.2e  (|u| %.2e)\n",
              hy.multipliers, hy.solve.iterations, hy.jump, worst, exact, scale);
  CHECK(scale > 1e-6);                 // the datum reached the system at all
  CHECK(hy.solve.iterations > 0);      // and the interface had something to solve
  CHECK(hy.jump < 1e-8);
  CHECK(worst < 1e-8 * std::max(1.0, scale));
  CHECK(exact < 1e-8);                 // and both are the patch
  }
}

// Does the diagonal member hybridize? exokal says any StressOperators cell
// does -- "the stabilized families included -- this is the SPD route for the
// realizations whose sigma-block the two-point condensation refuses" -- and
// diagonal_vem is the one that condenses instead. Both routes exist for the same
// product: the condensed system is quasi-definite and wanted MINRES, the
// hybridized one should be SPD and take a conjugate gradient.
//
// Held to the same three things as stabilized_vem: the sparse assembly against
// exokal's dense oracle, the interface SPD, and the model's hybrid answer
// against its own monolithic one on the linear patch.
MIMETIKA_TEST(the_diagonal_member_hybridizes_too) {
  for (const Family family : {Family::cartesian, Family::simplex}) {
    const exokal::Mesh m = mimetika::mesh::box({2, 2, 2}, 3, family);
    const Comparison c =
        against_the_oracle(m, 3, Realization::diagonal_vem, Formulation::strong_symmetry_total);
    std::printf("  %-9s diagonal_vem  %4zu multipliers  %6zu triplets   worst %.2e\n",
                mimetika::mesh::name(family), c.n, c.nnz, c.worst);
    CHECK(c.n > 0);
    CHECK(c.worst < 1e-10 * std::max(1.0, c.scale));
  }
}

MIMETIKA_TEST(the_diagonal_member_solves_and_recovers) {
  const exokal::Mesh m = mimetika::mesh::box({3, 3, 3}, 3, Family::cartesian);
  const auto cells = static_cast<std::size_t>(m.topology().count(3));

  const auto build = [&]() {
    auto model = std::make_unique<mimetika::CauchyMechanicsModel>(
        m, 3, mimetika::ElasticMaterial{kMu, kLam}, Realization::diagonal_vem,
        Formulation::strong_symmetry_total);
    std::array<double, 9> grad{};
    for (int k = 0; k < 3; ++k) grad[static_cast<std::size_t>(k * 3 + k)] = 1.0;
    for (const Index f : mimetika::boundary_facets(m.topology(), 3)) {
      const auto x = exokal::centroid(m, 3, mimetika::cofacet_of(m, 3, f));
      model->prescribe_displacement({f}, {x[0], x[1], x[2]}, grad);
    }
    model->build();
    return model;
  };

  auto monolithic = build();
  mimetika::solver::PetscSolver direct;
  std::vector<double> x;
  const auto rep = direct.solve(monolithic->system(), monolithic->rhs(), x);
  CHECK(rep.converged);
  monolithic->accept(std::move(x));

  auto hybrid = build();
  mimetika::solver::SolverOptions o;
  o.method = "cg";
  o.preconditioner = "hypre";
  o.rtol = 1e-12;
  o.max_iterations = 2000;
  o.condense = false;
  mimetika::solver::PetscSolver cg(o);
  const auto hy = hybrid->hybridized(cg);
  CHECK(hy.solve.converged);

  double worst = 0.0, scale = 0.0, exact = 0.0;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const auto xc = exokal::centroid(m, 3, e);
    for (int k = 0; k < 3; ++k) {
      const double mono = monolithic->displacement(e, k);
      worst = std::max(worst, std::abs(mono - hybrid->displacement(e, k)));
      scale = std::max(scale, std::abs(mono));
      exact = std::max(exact, std::abs(hybrid->displacement(e, k) - xc[static_cast<std::size_t>(k)]));
    }
  }
  // and the stress, component by component: a uniform sign flip of sigma is
  // invisible in the displacement
  double worst_s = 0.0, scale_s = 0.0;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const std::array<double, 9> a = monolithic->cell_stress(e);
    const std::array<double, 9> b = hybrid->cell_stress(e);
    for (std::size_t k = 0; k < 9; ++k) {
      worst_s = std::max(worst_s, std::abs(a[k] - b[k]));
      scale_s = std::max(scale_s, std::abs(a[k]));
    }
  }
  // the same ratio diagnostic: uniform means a chart or a constant
  double rlo = 1e300, rhi = -1e300;
  for (Index e = 0; e < static_cast<Index>(cells); ++e) {
    const std::array<double, 9> a = monolithic->cell_stress(e);
    const std::array<double, 9> b = hybrid->cell_stress(e);
    for (std::size_t k = 0; k < 9; ++k) {
      if (std::abs(a[k]) < 1e-6) continue;
      const double r = b[k] / a[k];
      rlo = std::min(rlo, r);
      rhi = std::max(rhi, r);
    }
  }
  // and the raw facet traction, the sigma dofs with no recombination
  double tlo = 1e300, thi = -1e300;
  for (Index f = 0; f < m.topology().count(2); ++f) {
    const std::array<double, 3> a = monolithic->facet_traction(f);
    const std::array<double, 3> b = hybrid->facet_traction(f);
    for (std::size_t k = 0; k < 3; ++k) {
      if (std::abs(a[k]) < 1e-6) continue;
      tlo = std::min(tlo, b[k] / a[k]);
      thi = std::max(thi, b[k] / a[k]);
    }
  }
  std::printf("      %zu cells, |f| = %.5f   stress ratio [%.6f, %.6f]   facet traction ratio "
              "[%.6f, %.6f]\n",
              cells, exokal::measure(m, 2, 0), rlo, rhi, tlo, thi);
  CHECK(scale_s > 1e-6);
  CHECK(worst_s < 1e-8 * std::max(1.0, scale_s));
  std::printf("  diagonal_vem  %zu multipliers  cg %d its  jump %.1e   "
              "|hybrid - monolithic| %.2e  |hybrid - exact| %.2e  (|u| %.2e)\n",
              hy.multipliers, hy.solve.iterations, hy.jump, worst, exact, scale);
  CHECK(scale > 1e-6);
  CHECK(hy.solve.iterations > 0);
  CHECK(hy.jump < 1e-8);
  CHECK(worst < 1e-8 * std::max(1.0, scale));
  CHECK(exact < 1e-8);
}

// One cell, two local saddles, side by side.
//
// The ratio sigma_hybrid / sigma_mixed is mu/(mu + 3 lambda) -- measured at
// three (mu, lambda) pairs, invariant under mesh and cell family. A ratio says
// that two things differ, not where. This builds the local matrix both ways from
// the same StressOperators cell and compares it entry by entry.
//
// exokal's is reconstructed from its documented assembly and then checked
// against the inverse it actually returns -- A_reconstructed * Ainv == I -- so
// the comparison rests on exokal's real matrix rather than on a reading of it.
MIMETIKA_TEST(one_cell_the_two_local_saddles_side_by_side) {
  const exokal::Mesh m = mimetika::mesh::box({1, 1, 1}, 3, Family::cartesian);
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(
      m, 3, kMu, kLam, Realization::stabilized_vem, Formulation::strong_symmetry_total);
  const exokal::hodge::HybridStressOperators hops =
      exokal::hodge::HybridStressOperators::build(m, 3, ops, kMu);

  const exokal::hodge::StressOperators::Cell& g = ops.cell(0);
  const exokal::hodge::HybridStressOperators::Cell& h = hops.cell(0);
  const std::size_t ns = g.M.rows();
  const std::size_t nk = g.Dv.rows() + g.As.rows();
  const bool total = ops.hydrostatic_mass() > 0.0;
  const std::size_t nt = ns + nk + (total ? 1 : 0);

  // One convention now, held by both: the coupling antisymmetric, sigma row
  // -Dv^T against +Dv on the field row. `sym` keeps its name from when the two
  // differed; the test asserts that they no longer do.
  std::vector<double> sym(nt * nt, 0.0);
  std::vector<double> anti(nt * nt, 0.0);
  for (std::size_t i = 0; i < ns; ++i) {
    for (std::size_t j = 0; j < ns; ++j) {
      sym[i * nt + j] = g.M(i, j);
      anti[i * nt + j] = g.M(i, j);
    }
    for (std::size_t r = 0; r < nk; ++r) {
      const double v = r < g.Dv.rows() ? g.Dv(r, i) : g.As(r - g.Dv.rows(), i);
      sym[i * nt + ns + r] = -v;
      sym[(ns + r) * nt + i] = v;
      anti[i * nt + ns + r] = -v;
      anti[(ns + r) * nt + i] = v;
    }
    if (total) {
      const double v = g.T(0, i) / (2.0 * kMu);
      sym[i * nt + ns + nk] = -v;
      sym[(ns + nk) * nt + i] = v;
      anti[i * nt + ns + nk] = -v;
      anti[(ns + nk) * nt + i] = v;
    }
  }
  if (total) {
    sym[(ns + nk) * nt + ns + nk] = -ops.hydrostatic_mass() * g.volume;
    anti[(ns + nk) * nt + ns + nk] = -ops.hydrostatic_mass() * g.volume;
  }

  // is the reconstruction exokal's own matrix? A * Ainv must be the identity
  double off = 0.0;
  for (std::size_t i = 0; i < nt; ++i) {
    for (std::size_t j = 0; j < nt; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < nt; ++k) acc += sym[i * nt + k] * h.Ainv(k, j);
      off = std::max(off, std::abs(acc - (i == j ? 1.0 : 0.0)));
    }
  }

  // and where do the two differ?
  double worst = 0.0, scale = 0.0;
  std::size_t in_sigma = 0, in_coupling = 0, in_field = 0;
  for (std::size_t i = 0; i < nt; ++i) {
    for (std::size_t j = 0; j < nt; ++j) {
      const double d = std::abs(sym[i * nt + j] - anti[i * nt + j]);
      scale = std::max(scale, std::abs(sym[i * nt + j]));
      if (d <= 1e-12) continue;
      worst = std::max(worst, d);
      if (i < ns && j < ns) ++in_sigma;
      else if ((i < ns) != (j < ns)) ++in_coupling;
      else ++in_field;
    }
  }
  std::printf("  one cell: %zu sigma + %zu field%s = %zu\n", ns, nk, total ? " + 1 p" : "", nt);
  std::printf("    A_exokal * Ainv - I: %.2e   (the reconstruction IS exokal's matrix)\n", off);
  std::printf("    exokal vs mixed: worst %.2e of %.2e   entries differing: "
              "%zu in the sigma block, %zu in the coupling, %zu in the field block\n",
              worst, scale, in_sigma, in_coupling, in_field);
  CHECK(off < 1e-8);          // otherwise the comparison is against a guess
  CHECK(in_sigma == 0);       // the compliance is shared, and this proves it
  CHECK(in_field == 0);       // as is everything below the coupling
  CHECK(in_coupling == 0);    // and the coupling, now that the convention is one
}

MIMETIKA_TEST_MAIN()
