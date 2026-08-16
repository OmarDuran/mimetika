#include <cmath>
#include <vector>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/linear_solver/petsc.hpp"

using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;
using mimetika::solver::PetscSolver;
using mimetika::solver::SparseSystem;

namespace {
bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }
}  // namespace

// A SOLVER IS TESTED ON A MATRIX NOBODY ASSEMBLED FROM PHYSICS. If the first
// thing it sees is a saddle point from a mixed method, a wrong answer has two
// possible causes and the test cannot separate them.
MIMETIKA_TEST(the_direct_solver_solves_a_known_system) {
  SparseSystem A;
  A.n = 3;
  // [ 4 1 0 ; 1 3 1 ; 0 1 2 ], symmetric positive definite
  const int r[] = {0, 0, 1, 1, 1, 2, 2};
  const int c[] = {0, 1, 0, 1, 2, 1, 2};
  const double v[] = {4, 1, 1, 3, 1, 1, 2};
  for (int k = 0; k < 7; ++k) {
    A.row.push_back(r[k]);
    A.col.push_back(c[k]);
    A.value.push_back(v[k]);
  }
  const std::vector<double> x_exact = {1.0, -2.0, 3.0};
  std::vector<double> b(3, 0.0);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    b[static_cast<std::size_t>(A.row[k])] +=
        A.value[k] * x_exact[static_cast<std::size_t>(A.col[k])];
  }

  PetscSolver solver;
  std::vector<double> x;
  const auto rep = solver.solve(A, b, x);
  CHECK(rep.converged);
  CHECK(x.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) CHECK(near(x[i], x_exact[i], 1e-12));
  CHECK(rep.residual < 1e-13);
}

// AN INDEFINITE SYSTEM IS THE POINT. A saddle point has zero diagonal
// entries, which a factorization without symmetric pivoting will divide by.
// This one is small enough to check by hand and has exactly that structure.
MIMETIKA_TEST(the_direct_solver_handles_a_saddle_point) {
  SparseSystem A;
  A.n = 3;
  // [ 2 0 1 ; 0 2 -1 ; 1 -1 0 ] — the last diagonal is ZERO
  const int r[] = {0, 0, 1, 1, 2, 2};
  const int c[] = {0, 2, 1, 2, 0, 1};
  const double v[] = {2, 1, 2, -1, 1, -1};
  for (int k = 0; k < 6; ++k) {
    A.row.push_back(r[k]);
    A.col.push_back(c[k]);
    A.value.push_back(v[k]);
  }
  const std::vector<double> b = {1.0, 1.0, 0.0};
  PetscSolver solver;
  std::vector<double> x;
  const auto rep = solver.solve(A, b, x);
  CHECK(rep.converged);
  CHECK(rep.residual < 1e-12);
}

// AND THEN THE REAL THING: the assembled poroelastic system, solved directly.
// A direct factorization answers "is the operator right" with no
// preconditioner standing between the question and the answer — which is
// exactly what is wanted while a discretization is being validated.
MIMETIKA_TEST(the_assembled_poroelastic_system_solves) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, 3, 1.0, 1.0);
  const exokal::hodge::FluxOperators hodge =
      exokal::hodge::FluxOperators::build(m, exokal::hodge::Coefficient::uniform(1.0),
                                          exokal::hodge::FluxOperators::Realization::derham_bdm);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("flux_operators", hodge);

  Simulation sim(Catalogue::instance().build("poroelasticity", {}),
                 {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();
  const auto bottom = mimetika::FacetSelector::where(m, 3, mimetika::FacetSelector::at(2, 0.0));
  const auto top = mimetika::FacetSelector::where(m, 3, mimetika::FacetSelector::at(2, 2.0));
  mimetika::impose_traction(sim.constraints(), sp, "s_0", 3, m, bottom, std::array<double, 9>{});
  mimetika::impose_traction(sim.constraints(), sp, "s_0", 3, m, top,
                            {0, 0, 0, 0, 0, 0, 0, 0, -1.0});
  mimetika::impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, bottom);
  mimetika::impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, top);
  sim.freeze_constraints();

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const SparseSystem A = SparseSystem::from(jac);
  std::vector<double> b(sim.n_dofs());
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = -jac.residual[i];

  PetscSolver solver;
  std::vector<double> dx;
  const auto rep = solver.solve(A, b, dx);
  CHECK(rep.converged);
  CHECK(rep.residual < 1e-9);

  // the load really was carried: the stress reaches the applied traction
  double biggest = 0.0;
  const std::size_t s_off = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
  const std::size_t s_end = s_off + static_cast<std::size_t>(sp.map(sp.index_of("s_0")).size());
  for (std::size_t i = s_off; i < s_end; ++i) {
    biggest = std::max(biggest, std::abs(sim.state()[i] + dx[i]));
  }
  CHECK(near(biggest, 1.0, 1e-9));

  // and the displacement responded, finitely
  const std::size_t u = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
  double moved = 0.0;
  for (std::size_t i = u; i < u + 24; ++i) moved = std::max(moved, std::abs(dx[i]));
  CHECK(moved > 1e-9 && moved < 1.0);
}

MIMETIKA_TEST_MAIN()
