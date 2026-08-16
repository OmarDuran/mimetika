#include <cmath>
#include <string>
#include <vector>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"

using exokal::hodge::Coefficient;
using exokal::hodge::FluxOperators;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;
using mimetika::physics::Composition;

namespace {

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

}  // namespace

// ONE OBJECT, THREE OPERATORS. A solver asks a discretized problem for a
// residual, a tangent and the tangent's action; today those took six objects
// wired in a particular order. Simulation is that wiring done once, and the
// test is that all three come out of it consistently.
MIMETIKA_TEST(the_simulation_produces_residual_jacobian_and_action) {
  const auto m = mimetika_test::hex_grid(3);
  const graphos::Complex& c = m.topology();
  const FluxOperators hodge =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_bdm);
  exokal::forms::TermContext ctx;
  ctx.provide("flux_operators", hodge);

  const Composition comp = Catalogue::instance().build("single_phase_flow", {});
  Simulation sim(comp, {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  CHECK(sim.n_dofs() > 0);
  CHECK(sim.epoch().n_strata() == 1);

  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    sim.state()[i] = 0.4 + 0.03 * static_cast<double>(i % 9);
  }
  sim.freeze_constraints();  // none yet: the unconstrained case first

  std::vector<double> r;
  sim.residual(r);
  CHECK(r.size() == sim.n_dofs());

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  CHECK(jac.nnz() > 0);

  // the residual assembled through the two paths must agree
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) CHECK(near(r[i], jac.residual[i], 1e-12));

  // and the action must equal the assembled tangent times the direction
  std::vector<double> v(sim.n_dofs()), y;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = -1.1 + 0.07 * static_cast<double>(i % 13);
  sim.apply(v, y);

  std::vector<double> expect(sim.n_dofs(), 0.0);
  for (std::size_t k = 0; k < jac.nnz(); ++k) {
    expect[static_cast<std::size_t>(jac.row[k])] +=
        jac.value[k] * v[static_cast<std::size_t>(jac.col[k])];
  }
  bool nonzero = false;
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    nonzero = nonzero || std::abs(y[i]) > 1e-12;
    CHECK(near(y[i], expect[i], 1e-10));
  }
  CHECK(nonzero);
}

// THE CONSTRAINTS MUST HOLD ON ALL THREE PATHS. A consumer that wires this
// by hand typically remembers the Jacobian and forgets the action, and the
// symptom is a Krylov method that quietly solves a different problem. Here
// the same Constraints object serves every path, and the test checks each.
MIMETIKA_TEST(essential_constraints_hold_on_every_path) {
  const auto m = mimetika_test::hex_grid(3);
  const graphos::Complex& c = m.topology();
  const FluxOperators hodge =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_bdm);
  exokal::forms::TermContext ctx;
  ctx.provide("flux_operators", hodge);

  const Composition comp = Catalogue::instance().build("single_phase_flow", {});
  Simulation sim(comp, {StratumSpec{"ambient", &c, 3, 0}}, ctx);

  // pin a handful of degrees of freedom, as a roller or a sealed face would
  const std::vector<exokal::forms::Index> pinned = {0, 5, 17, 42};
  for (const auto d : pinned) sim.constraints().pin(d, 0.25 * static_cast<double>(d));
  CHECK(sim.constraints().size() == 4);

  for (std::size_t i = 0; i < sim.n_dofs(); ++i) sim.state()[i] = 0.9;
  sim.freeze_constraints();

  // freezing puts the constrained values into the state, so the very first
  // residual is already consistent with them
  for (const auto d : pinned) {
    CHECK(sim.state()[static_cast<std::size_t>(d)] == 0.25 * static_cast<double>(d));
  }

  std::vector<double> r;
  sim.residual(r);
  for (const auto d : pinned) CHECK(near(r[static_cast<std::size_t>(d)], 0.0, 1e-14));

  // perturb a constrained entry: the residual is exactly the discrepancy times
  // the scale the constrained equation is written with, not a penalty times it.
  // The scale is free -- it is the same equation for any nonzero multiple --
  // and it is chosen to match the row the constraint replaced so that a direct
  // factorization is not asked to pivot across twenty orders of magnitude.
  const double s0 = sim.constraints().scale_at(0);
  CHECK(s0 > 0.0);
  sim.state()[0] += 0.375;
  sim.residual(r);
  CHECK(near(r[0], s0 * 0.375, 1e-14 * std::max(1.0, s0)));
  // and what the scale must NOT change: the step lands on the datum exactly,
  // since s dx = -s (x - g) gives x + dx = g whatever s is
  CHECK(near(sim.state()[0] - r[0] / s0, 0.25 * 0.0, 1e-14));
  sim.state()[0] -= 0.375;

  // the Jacobian row of a constrained unknown is a single diagonal entry, and
  // nothing the terms wrote on it survives
  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  for (const auto d : pinned) {
    std::size_t entries = 0;
    double diag = 0.0;
    for (std::size_t k = 0; k < jac.nnz(); ++k) {
      if (jac.row[k] != d) continue;
      ++entries;
      if (jac.col[k] == d) diag = jac.value[k];
    }
    CHECK(entries == 1);
    CHECK(diag == sim.constraints().scale_at(static_cast<std::size_t>(d)));
    CHECK(diag > 0.0);
  }

  // and the action agrees with that Jacobian, constrained rows included —
  // the path most easily left inconsistent
  std::vector<double> v(sim.n_dofs()), y;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5 + 0.11 * static_cast<double>(i % 7);
  sim.apply(v, y);
  std::vector<double> expect(sim.n_dofs(), 0.0);
  for (std::size_t k = 0; k < jac.nnz(); ++k) {
    expect[static_cast<std::size_t>(jac.row[k])] +=
        jac.value[k] * v[static_cast<std::size_t>(jac.col[k])];
  }
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) CHECK(near(y[i], expect[i], 1e-10));
  for (const auto d : pinned) {
    const auto i = static_cast<std::size_t>(d);
    CHECK(near(y[i], sim.constraints().scale_at(i) * v[i],
               1e-14 * std::max(1.0, sim.constraints().scale_at(i))));
  }
}

MIMETIKA_TEST(a_dof_pinned_to_two_values_is_refused) {
  mimetika::Constraints c;
  c.pin(3, 1.0);
  c.pin(3, 2.0);
  bool threw = false;
  try {
    c.finalize(10);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  mimetika::Constraints ok;
  ok.pin(3, 1.0);
  ok.pin(3, 1.0);  // the same value twice is not a conflict
  ok.finalize(10);
  CHECK(ok.pinned(3) && ok.value_at(3) == 1.0);
}

MIMETIKA_TEST_MAIN()
