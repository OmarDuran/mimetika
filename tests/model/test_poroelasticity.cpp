#include <cmath>
#include <vector>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"

using exokal::hodge::StressOperators;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;
using mimetika::physics::Composition;

namespace {
bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }
}  // namespace

// THE CATALOGUE CLAIM, tested rather than asserted: poroelasticity adds no
// implementation. It is flow plus mechanics plus a coupling that contributes
// no field of its own, so its space is exactly the union of the other two.
MIMETIKA_TEST(poroelasticity_adds_no_field_of_its_own) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();

  const Composition mech = Catalogue::instance().build("linear_elasticity", {});
  const Composition flow = Catalogue::instance().build("single_phase_flow", {});
  const Composition poro = Catalogue::instance().build("poroelasticity", {});
  CHECK(mech.size() == 1 && flow.size() == 1 && poro.size() == 3);

  const auto sm = mech.space(c, 3), sf = flow.space(c, 3), sp = poro.space(c, 3);
  CHECK(sm.n_fields() == 3);  // s, u, g
  CHECK(sf.n_fields() == 2);  // q, p
  CHECK(sp.n_fields() == 5);  // and nothing more
  CHECK(sp.size() == sm.size() + sf.size());
  for (const char* f : {"s_0", "u_0", "g_0", "q_0", "p_0"}) CHECK(sp.has(f));

  // the coupling needs both physics, and says so when one is absent
  Composition bad;
  bad.emplace<mimetika::physics::Flow>();
  bad.emplace<mimetika::physics::PoroCoupling>();
  bool threw = false;
  try {
    bad.validate(3);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find("'displacement'") != std::string::npos;
  }
  CHECK(threw);
}

// THE STRESS DEGREES OF FREEDOM ARE d^2 PER FACET, and the rotation carries
// the d(d-1)/2 components weak symmetry needs. Getting either count wrong
// changes the method rather than breaking it, so it is pinned.
MIMETIKA_TEST(the_mixed_elasticity_space_has_the_afw_counts) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();
  const auto s = Catalogue::instance().build("linear_elasticity", {}).space(c, 3);

  const auto n_facets = static_cast<std::size_t>(c.count(2));
  const auto n_cells = static_cast<std::size_t>(c.count(3));
  CHECK(s.map(s.index_of("s_0")).size() == 9 * static_cast<exokal::forms::Index>(n_facets));
  CHECK(s.map(s.index_of("u_0")).size() == 3 * static_cast<exokal::forms::Index>(n_cells));
  CHECK(s.map(s.index_of("g_0")).size() == 3 * static_cast<exokal::forms::Index>(n_cells));
  CHECK(s.map(s.index_of("s_0")).layout().degree() == 2);  // facet cochain
  CHECK(s.map(s.index_of("u_0")).layout().degree() == 3);  // cell-wise
}

// THE SYSTEM IS A SADDLE POINT WITH ADJOINT COUPLINGS, in the convention the
// flux term set: [M, -B^T; +B, 0]. So a constitutive block is symmetric and
// every off-diagonal pair is the NEGATIVE transpose of its partner.
//
// That every physics uses the same convention is the load-bearing part. Flow,
// mechanics and the Biot coupling all land in one system, and two conventions
// meeting there would give a matrix neither symmetric nor antisymmetric —
// structure no solver could exploit, and nothing would visibly break.
//
// The displacement and rotation blocks must also be EMPTY, which is what
// makes this a saddle point rather than a positive-definite system in
// disguise.
MIMETIKA_TEST(the_poroelastic_system_is_a_saddle_point_with_adjoint_couplings) {
  const auto m = mimetika_test::hex_grid(2);
  const graphos::Complex& c = m.topology();

  const StressOperators ops = StressOperators::build(m, 3, 1.0, 1.0);
  CHECK(ops.size() == static_cast<std::size_t>(c.count(3)));
  // WHETHER A CELL STABILIZES IS A PROPERTY OF THE SPACE, not of the cell.
  // The default de Rham realization reconstructs each stress row on the
  // enriched scalar space, so its N is square and unisolvent and there is
  // nothing left for a stabilization to see -- on hexahedra as on simplices.
  // The AFW realization reconstructs on the full linear tensor space, whose
  // moments a hexahedron does not determine, so there every cell stabilizes.
  CHECK(ops.n_stabilized() == 0);
  const StressOperators afw =
      StressOperators::build(m, 3, 1.0, 1.0, StressOperators::Realization::stabilized_bdm);
  CHECK(afw.n_stabilized() == afw.size());

  const exokal::hodge::FluxOperators hodge =
      exokal::hodge::FluxOperators::build(m, exokal::hodge::Coefficient::uniform(1.0),
                                          exokal::hodge::FluxOperators::Realization::derham_bdm);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("flux_operators", hodge);

  const Composition poro = Catalogue::instance().build("poroelasticity", {});
  Simulation sim(poro, {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    sim.state()[i] = 0.2 + 0.01 * static_cast<double>(i % 11);
  }
  sim.freeze_constraints();

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  CHECK(jac.nnz() > 0);

  // gather the assembled matrix densely enough to test its structure
  const auto n = sim.n_dofs();
  std::vector<double> A(n * n, 0.0);
  for (std::size_t k = 0; k < jac.nnz(); ++k) {
    A[static_cast<std::size_t>(jac.row[k]) * n + static_cast<std::size_t>(jac.col[k])] +=
        jac.value[k];
  }

  const auto& sp = sim.epoch().stratum(0).space();
  const auto blk = [&](const char* name) {
    const std::size_t f = sp.index_of(name);
    return std::pair<std::size_t, std::size_t>{
        static_cast<std::size_t>(sp.offset(f)),
        static_cast<std::size_t>(sp.offset(f)) + static_cast<std::size_t>(sp.map(f).size())};
  };
  const auto [s0, s1] = blk("s_0");
  const auto [u0, u1] = blk("u_0");
  const auto [g0, g1] = blk("g_0");
  const auto [p0, p1] = blk("p_0");

  // ADJOINTNESS everywhere: |A_ij| = |A_ji|, because every coupling was
  // written from one array of coefficients
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      worst = std::max(worst, std::abs(std::abs(A[i * n + j]) - std::abs(A[j * n + i])));
    }
  }
  CHECK(worst < 1e-10);

  // AND THE SIGNS, which adjointness-in-magnitude cannot see. The two
  // couplings of a poroelastic system are not the same kind of object:
  //
  //   (s,p)  the BIOT coupling, a constitutive symmetry. Both blocks are
  //          second derivatives of one free energy, so A_ij = +A_ji.
  //   (q,p)  the DARCY pair, adjoint differential operators, which exokal
  //          writes antisymmetrically as [M, -B^T; +B, 0]: A_ij = -A_ji.
  //
  // Giving the Biot block the Darcy convention leaves every structural
  // property here intact and every magnitude unchanged, and makes the
  // undrained response wrong -- p = 5 sigma_0/13 instead of sigma_0/alpha on
  // a confined column at mu = lam = alpha = 1. Nothing short of checking the
  // sign, or solving a problem with a known answer, detects it.
  const auto [q0, q1] = blk("q_0");
  const auto pairing = [&](std::size_t a0, std::size_t a1, std::size_t b0, std::size_t b1,
                           double want) {
    double err = 0.0, mag = 0.0;
    for (std::size_t i = a0; i < a1; ++i) {
      for (std::size_t j = b0; j < b1; ++j) {
        err = std::max(err, std::abs(A[i * n + j] - want * A[j * n + i]));
        mag = std::max(mag, std::abs(A[i * n + j]));
      }
    }
    return std::pair<double, double>{err, mag};
  };
  const auto [biot_err, biot_mag] = pairing(s0, s1, p0, p1, +1.0);
  CHECK(biot_mag > 1e-12);  // the block must exist before its sign means anything
  CHECK(biot_err < 1e-10);
  const auto [darcy_err, darcy_mag] = pairing(q0, q1, p0, p1, -1.0);
  CHECK(darcy_mag > 1e-12);
  CHECK(darcy_err < 1e-10);

  // and the multiplier blocks are empty: no (u,u), no (g,g), no (u,g)
  const auto empty = [&](std::size_t a0, std::size_t a1, std::size_t b0, std::size_t b1) {
    double w = 0.0;
    for (std::size_t i = a0; i < a1; ++i) {
      for (std::size_t j = b0; j < b1; ++j) w = std::max(w, std::abs(A[i * n + j]));
    }
    return w;
  };
  CHECK(empty(u0, u1, u0, u1) == 0.0);
  CHECK(empty(g0, g1, g0, g1) == 0.0);
  CHECK(empty(u0, u1, g0, g1) == 0.0);
  CHECK(empty(u0, u1, p0, p1) == 0.0);  // the Biot coupling goes through the STRESS

  // the Biot coupling is present, and it is the PLAIN transpose: the pore
  // pressure enters the constitutive relation exactly as the volumetric
  // response enters the mass balance, from one coefficient and with one sign.
  // Its symmetry is checked above, with the Darcy pair's antisymmetry beside
  // it so the two conventions cannot be confused for each other again.
  CHECK(empty(s0, s1, p0, p1) > 1e-9);
  CHECK(empty(p0, p1, s0, s1) > 1e-9);

  // and so are the mechanics couplings, in the same convention
  double anti_u = 0.0;
  for (std::size_t i = s0; i < s1; ++i) {
    for (std::size_t j = u0; j < u1; ++j) {
      anti_u = std::max(anti_u, std::abs(A[i * n + j] + A[j * n + i]));
    }
  }
  CHECK(anti_u < 1e-12);

  // the constitutive block itself IS symmetric — it is an inner product
  double sym = 0.0;
  for (std::size_t i = s0; i < s1; ++i) {
    for (std::size_t j = s0; j < s1; ++j) {
      sym = std::max(sym, std::abs(A[i * n + j] - A[j * n + i]));
    }
  }
  CHECK(sym < 1e-10);
}

MIMETIKA_TEST_MAIN()
