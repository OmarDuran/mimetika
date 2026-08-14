#include <cmath>
#include <vector>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/physics/boundary_terms.hpp"

using exokal::forms::Index;
using mimetika::BoundaryData;
using mimetika::FacetSelector;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;

namespace {
bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }
}  // namespace

// THE BOUNDARY IS A THIN SET, and it is identified once. A 3x3x3 box of
// hexahedra has 6 * 9 = 54 boundary facets out of 108 — and only those are
// sites for a boundary coupling.
MIMETIKA_TEST(the_boundary_facets_are_the_ones_with_a_single_cofacet) {
  const auto m = mimetika_test::hex_grid(3);
  const auto b = mimetika::boundary_facets(m.topology(), 3);
  CHECK(b.size() == 54);
  CHECK(static_cast<Index>(b.size()) < m.topology().count(2));

  // and a face of the box is a ninth of them
  const auto bottom = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, 3.0));
  CHECK(bottom.size() == 9 && top.size() == 9);
  for (const Index f : bottom) {
    CHECK(std::find(b.begin(), b.end(), f) != b.end());
  }
}

// ESSENTIAL CONDITIONS PIN DEGREES OF FREEDOM, and in a mixed form it is the
// FLUX that is pinned — a sealed face, not a prescribed pressure.
MIMETIKA_TEST(a_sealed_face_pins_the_flux_and_nothing_else) {
  const auto m = mimetika_test::hex_grid(3);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::FluxHodge h = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(1.0),
      exokal::hodge::FluxHodge::Realization::derham);
  exokal::forms::TermContext ctx;
  ctx.provide("flux_hodge", h);

  const auto comp = Catalogue::instance().build("single_phase_flow", {});
  Simulation sim(comp, {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();

  const auto sides = FacetSelector::where(m, 3, FacetSelector::at(0, 0.0));
  mimetika::impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, sides);
  // d moments per facet in the de Rham flow space: sealing a face pins the
  // net flow AND the way it is distributed across the face, which is what
  // "no flow through it" means when the flux is not constant on a facet
  CHECK(sim.constraints().size() == 3 * sides.size());
  sim.freeze_constraints();

  // the pinned dofs are flux dofs, and no pressure dof was touched
  const std::size_t pblock = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
  for (std::size_t d = 0; d < sim.n_dofs(); ++d) {
    if (sim.constraints().pinned(d)) CHECK(d < pblock);
  }

  std::vector<double> r;
  sim.residual(r);
  for (std::size_t d = 0; d < sim.n_dofs(); ++d) {
    if (sim.constraints().pinned(d)) CHECK(near(r[d], 0.0, 1e-14));
  }
}

// THE NATURAL CONDITION IS FREE WHEN IT IS HOMOGENEOUS. A drained face at
// zero pressure needs no term, and attaching one changes nothing — so a model
// cannot be wrong by failing to mention its free boundaries.
MIMETIKA_TEST(a_homogeneous_natural_condition_costs_nothing) {
  const auto m = mimetika_test::hex_grid(3);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::FluxHodge h = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(1.0),
      exokal::hodge::FluxHodge::Realization::derham);

  const auto comp = Catalogue::instance().build("single_phase_flow", {});
  const auto n_facets = static_cast<std::size_t>(c.count(2));

  BoundaryData zero(n_facets), loaded(n_facets);
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, 3.0));
  zero.set(top, 0.0);
  loaded.set(top, 2.5);

  const auto run = [&](const BoundaryData& bd, bool with_term) {
    exokal::forms::TermContext ctx;
    ctx.provide("flux_hodge", h);
    ctx.provide("boundary_pressure", bd);
    Simulation sim(comp, {StratumSpec{"ambient", &c, 3, 0}}, ctx);
    for (std::size_t i = 0; i < sim.n_dofs(); ++i) sim.state()[i] = 0.3;
    sim.freeze_constraints();
    if (with_term) {
      // attached directly, since the composition does not know the geometry
      const_cast<exokal::forms::Model&>(sim.model())
          .add("prescribed_pressure", exokal::forms::On::all());
    }
    std::vector<double> r;
    sim.residual(r);
    return r;
  };

  const auto plain = run(zero, false);
  const auto with_zero = run(zero, true);
  const auto with_load = run(loaded, true);

  for (std::size_t i = 0; i < plain.size(); ++i) CHECK(near(plain[i], with_zero[i], 1e-14));

  // a NONZERO datum does move the residual, and only on the loaded face
  std::size_t moved = 0;
  for (std::size_t i = 0; i < plain.size(); ++i) {
    if (std::abs(with_load[i] - plain[i]) > 1e-12) ++moved;
  }
  // the datum pairs with the CONSTANT moment of each loaded facet
  CHECK(moved == top.size());

  // and it scales with the datum, as a linear term must
  const auto run_scaled = [&](double v) {
    BoundaryData bd(n_facets);
    bd.set(top, v);
    return run(bd, true);
  };
  const auto one = run_scaled(1.0);
  const auto two = run_scaled(2.0);
  for (std::size_t i = 0; i < plain.size(); ++i) {
    CHECK(near(two[i] - plain[i], 2.0 * (one[i] - plain[i]), 1e-12));
  }
}

// THE MIRROR, FOR MECHANICS. A prescribed displacement is natural in the
// Hellinger-Reissner form exactly as a prescribed pressure is in the mixed
// flow form — and both are homogeneous-for-free, so the two halves of a
// poroelastic model behave the same way at their boundaries.
MIMETIKA_TEST(a_prescribed_displacement_is_natural_and_free_when_homogeneous) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, 3, 1.0, 1.0);
  const auto n_facets = static_cast<std::size_t>(c.count(2));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, 2.0));
  CHECK(!top.empty());

  const auto run = [&](const mimetika::BoundaryVectorData& bd, bool with_term) {
    exokal::forms::TermContext ctx;
    ctx.provide("stress_operators", ops);
    ctx.provide("boundary_displacement", bd);
    Simulation sim(Catalogue::instance().build("linear_elasticity", {}),
                   {StratumSpec{"ambient", &c, 3, 0}}, ctx);
    for (std::size_t i = 0; i < sim.n_dofs(); ++i) sim.state()[i] = 0.2;
    sim.freeze_constraints();
    if (with_term) {
      const_cast<exokal::forms::Model&>(sim.model())
          .add("prescribed_displacement", exokal::forms::On::all());
    }
    std::vector<double> r;
    sim.residual(r);
    return r;
  };

  mimetika::BoundaryVectorData zero(n_facets), pushed(n_facets);
  zero.set(top, {0.0, 0.0, 0.0});
  pushed.set(top, {0.0, 0.0, -0.5});

  const auto plain = run(zero, false);
  const auto with_zero = run(zero, true);
  const auto with_push = run(pushed, true);

  // homogeneous costs nothing, as for the pressure
  for (std::size_t i = 0; i < plain.size(); ++i) CHECK(near(plain[i], with_zero[i], 1e-13));

  // a real datum moves the stress rows of the loaded facets, and only those:
  // one moment per facet, since a CONSTANT datum pairs only with the constant
  // facet basis function
  std::size_t moved = 0;
  for (std::size_t i = 0; i < plain.size(); ++i) {
    if (std::abs(with_push[i] - plain[i]) > 1e-12) ++moved;
  }
  CHECK(moved == top.size());

  // and it is linear in the datum
  mimetika::BoundaryVectorData twice(n_facets);
  twice.set(top, {0.0, 0.0, -1.0});
  const auto with_twice = run(twice, true);
  for (std::size_t i = 0; i < plain.size(); ++i) {
    CHECK(near(with_twice[i] - plain[i], 2.0 * (with_push[i] - plain[i]), 1e-12));
  }
}

// AN AFFINE DATUM IS EXACT, which is what a patch test needs. A linear
// displacement pairs with the higher facet basis functions too, so it must
// move MORE degrees of freedom than a constant one — a term that quietly
// integrated only the constant part would pass every test above and fail
// here.
MIMETIKA_TEST(an_affine_displacement_datum_reaches_the_higher_moments) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, 3, 1.0, 1.0);
  const auto n_facets = static_cast<std::size_t>(c.count(2));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, 2.0));

  const auto run = [&](const mimetika::BoundaryVectorData& bd) {
    exokal::forms::TermContext ctx;
    ctx.provide("stress_operators", ops);
    ctx.provide("boundary_displacement", bd);
    Simulation sim(Catalogue::instance().build("linear_elasticity", {}),
                   {StratumSpec{"ambient", &c, 3, 0}}, ctx);
    sim.freeze_constraints();
    const_cast<exokal::forms::Model&>(sim.model())
        .add("prescribed_displacement", exokal::forms::On::all());
    std::vector<double> r;
    sim.residual(r);
    return r;
  };

  mimetika::BoundaryVectorData constant(n_facets), linear(n_facets);
  constant.set(top, {0.1, 0.0, 0.0});
  // u_x = 0.1 + 0.3 (x - x_E)_x : a genuine shear-free stretch
  linear.set_affine(top, {0.1, 0.0, 0.0}, {0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  const auto rc = run(constant), rl = run(linear);
  std::size_t nc = 0, nl = 0;
  for (std::size_t i = 0; i < rc.size(); ++i) {
    if (std::abs(rc[i]) > 1e-12) ++nc;
    if (std::abs(rl[i]) > 1e-12) ++nl;
  }
  CHECK(nc == top.size());  // the constant moment of one component
  CHECK(nl > nc);           // and the linear part reaches further
}

MIMETIKA_TEST_MAIN()
