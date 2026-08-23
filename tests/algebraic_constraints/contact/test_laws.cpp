#include <cmath>
#include <memory>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/laws.hpp"

// Contact laws: the constitutive layer, tested without any mesh or solver.
//
// A law is a pure function of (traction, jump, state) in the facet frame, so it
// is checked directly against the conditions it is supposed to encode. Every
// test here is a port of tests/contact/test_laws.py, which is the authority on
// the behaviour; exokal is the authority on the operators, and the two meet only
// in the driver.

using mimetika::contact::ContactLaw;
using mimetika::contact::FrictionlessBilateral;
using mimetika::contact::LinearContact;
using mimetika::contact::RateAndStateFriction;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::SlipWeakening;
using mimetika::contact::State;
using mimetika::contact::Status;
using mimetika::contact::Vec3;

namespace {

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

Vec3 vec(double n, double t1 = 0.0, double t2 = 0.0) {
  Vec3 v;
  v[0] = n;
  v[1] = t1;
  v[2] = t2;
  return v;
}

}  // namespace

// -- the taxonomy itself -----------------------------------------------------

// Each law advertises what it needs, and the driver keys off these: a linear
// compliance means one solve and no iteration; path dependence means the
// previous jump must be kept; an asymmetric tangent forbids a symmetric solver.
MIMETIKA_TEST(the_taxonomy_flags_are_declared) {
  const LinearContact lin;
  const FrictionlessBilateral fb;
  const SignoriniCoulomb mc;
  const SlipWeakening sw(0.52, 0.20, 0.02);

  CHECK(lin.n_state() == 0 && !lin.path_dependent() && !lin.rate_dependent());
  CHECK(lin.symmetric_tangent() && lin.has_linear_compliance());

  CHECK(fb.n_state() == 0 && !fb.path_dependent());
  CHECK(fb.symmetric_tangent() && !fb.has_linear_compliance());

  CHECK(mc.n_state() == 1 && mc.path_dependent() && !mc.rate_dependent());
  CHECK(!mc.symmetric_tangent() && !mc.has_linear_compliance());

  // the weakening law inherits every flag: it differs in the coefficient alone
  CHECK(sw.n_state() == 1 && sw.path_dependent() && !sw.symmetric_tangent());
  CHECK(!sw.has_linear_compliance());

  // and rate-and-state is the only one that needs a time increment: it carries
  // a second internal variable and reads the slip rate rather than the slip
  const RateAndStateFriction rs;
  CHECK(rs.n_state() == 2 && rs.path_dependent() && rs.rate_dependent());
  CHECK(!rs.symmetric_tangent() && !rs.has_linear_compliance());
}

// Every law implements the contract, through the base pointer the driver holds
// it by.
MIMETIKA_TEST(every_law_implements_the_contract) {
  std::vector<std::unique_ptr<ContactLaw>> laws;
  laws.push_back(std::make_unique<LinearContact>());
  laws.push_back(std::make_unique<FrictionlessBilateral>());
  laws.push_back(std::make_unique<SignoriniCoulomb>());
  laws.push_back(std::make_unique<SlipWeakening>(0.52, 0.20, 0.02));
  laws.push_back(std::make_unique<RateAndStateFriction>());

  for (const auto& law : laws) {
    for (const int dim : {2, 3}) {
      // the law's own initial state, which is what the driver hands it: zero for
      // most, theta0 for rate-and-state, and starting the latter at zero would
      // put log(0) in its coefficient
      State s = law->initial_state();
      const Vec3 zero;
      const Vec3 t = law->project(zero, s, dim, &zero, &zero, 1.0);
      CHECK(std::isfinite(t[0]) && std::isfinite(t[1]) && std::isfinite(t[2]));
      law->advance(t, &zero, s, dim, &zero, 1.0);
      CHECK(std::isfinite(s[0]) && std::isfinite(s[1]));
      // the unused shear component of a 2D law is never written
      if (dim == 2) CHECK(t[2] == 0.0);
    }
    CHECK(!law->name().empty());
  }
}

// -- linear ------------------------------------------------------------------

MIMETIKA_TEST(the_linear_compliance_is_diagonal_in_the_facet_frame) {
  const LinearContact law(4.0, 8.0);
  const Vec3 c = law.linear_compliance(3);
  CHECK(near(c[0], 0.25) && near(c[1], 0.125) && near(c[2], 0.125));
  const Vec3 c2 = law.linear_compliance(2);
  CHECK(near(c2[0], 0.25) && near(c2[1], 0.125));
}

// Bonded: tension and interpenetration are both admissible, so the projection
// is the identity and the fixed point is reached in one evaluation.
MIMETIKA_TEST(the_linear_projection_is_the_identity) {
  const LinearContact law;
  State s;
  const Vec3 trial = vec(1.7, -0.4, 2.3);
  const Vec3 out = law.project(trial, s, 3);
  CHECK(near(out[0], trial[0]) && near(out[1], trial[1]) && near(out[2], trial[2]));
}

// -- frictionless bilateral --------------------------------------------------

// The fault is held shut and may carry tension. This is the law of the
// incremental problem: a fault under tens of MPa of in-situ compression stays
// closed, so an incremental tensile normal traction must not be read as opening.
MIMETIKA_TEST(the_bilateral_law_keeps_the_normal_traction_and_zeroes_the_shear) {
  const FrictionlessBilateral law;
  State s;
  for (const double tn : {-3.0, 0.0, 2.0}) {
    const Vec3 t = law.project(vec(tn, 5.0, -7.0), s, 3);
    CHECK(near(t[0], tn));  // whatever holds it shut, either sign
    CHECK(near(t[1], 0.0) && near(t[2], 0.0));
  }
}

// -- Signorini + Coulomb -----------------------------------------------------

MIMETIKA_TEST(the_normal_projection_removes_tension) {
  const SignoriniCoulomb law(0.5);
  State s;
  CHECK(near(law.project(vec(2.0), s, 3)[0], 0.0));    // tension clipped: it opens
  CHECK(near(law.project(vec(-3.0), s, 3)[0], -3.0));  // compression preserved
}

// The friction radius follows the projected normal traction, so an open point
// carries no shear automatically -- no separate branch for it.
MIMETIKA_TEST(an_open_point_carries_no_shear) {
  const SignoriniCoulomb law(0.6);
  State s;
  const Vec3 t = law.project(vec(5.0, 1.0, 2.0), s, 3);
  CHECK(near(t[0], 0.0) && near(t[1], 0.0) && near(t[2], 0.0));
}

MIMETIKA_TEST(the_tangential_projection_lands_on_the_friction_disk) {
  for (const double mu : {0.2, 0.6, 1.0}) {
    const SignoriniCoulomb law(mu);
    State s;
    const double tn = -2.0, radius = mu * std::abs(tn);

    // inside the cone: sticking, so the trial is returned untouched
    const Vec3 inside = vec(tn, 0.3 * radius, 0.4 * radius);
    const Vec3 in = law.project(inside, s, 3);
    CHECK(near(in[0], inside[0]) && near(in[1], inside[1]) && near(in[2], inside[2]));

    // outside: sliding, so it lands on the cone, direction preserved
    const Vec3 out = law.project(vec(tn, 10.0 * radius, 0.0), s, 3);
    CHECK(near(out.shear_norm(3), radius, 1e-12 * std::max(1.0, radius)));
    CHECK(out[1] > 0.0 && near(out[2], 0.0));
  }
}

MIMETIKA_TEST(cohesion_shifts_the_friction_bound) {
  const SignoriniCoulomb law(0.0, 1.5);
  State s;
  const Vec3 t = law.project(vec(-1.0, 10.0, 0.0), s, 3);
  CHECK(near(t.shear_norm(3), 1.5));
}

MIMETIKA_TEST(the_status_labels_open_stick_and_slip) {
  const SignoriniCoulomb law(0.5);
  CHECK(law.status(vec(0.0, 0.0, 0.0), 3) == Status::open);
  CHECK(law.status(vec(-2.0, 0.1, 0.0), 3) == Status::stick);
  CHECK(law.status(vec(-2.0, 1.0, 0.0), 3) == Status::slip);
}

// Slip accumulates over steps, as the magnitude of the tangential increment --
// the state a weakening law then reads.
MIMETIKA_TEST(slip_accumulates_over_steps) {
  const SignoriniCoulomb law;
  State s;
  const Vec3 g0 = vec(0.0, 0.0, 0.0);
  const Vec3 g1 = vec(0.0, 0.3, 0.4);
  law.advance(Vec3{}, &g1, s, 3, &g0);
  CHECK(near(s[0], 0.5));  // sqrt(0.3^2 + 0.4^2)
  const Vec3 g2 = vec(0.0, 0.3, 1.6);
  law.advance(Vec3{}, &g2, s, 3, &g1);
  CHECK(near(s[0], 1.7));  // + 1.2
}

// -- slip weakening ----------------------------------------------------------

// The coefficient falls linearly from mu_s to mu_d over d_c and then stays
// there (Novikov et al. 2024, Eq. 23). Benchmark 3's law.
MIMETIKA_TEST(the_friction_coefficient_weakens_linearly_then_saturates) {
  const SlipWeakening law(0.52, 0.20, 0.02);
  CHECK(near(law.mu_static(), 0.52) && near(law.mu_dynamic(), 0.20));

  State s;
  CHECK(near(law.friction_at(s, nullptr, nullptr, 0.0, 3), 0.52));  // unslipped: static

  s[0] = 0.01;  // half of d_c: half way down
  CHECK(near(law.friction_at(s, nullptr, nullptr, 0.0, 3), 0.36));
  s[0] = 0.02;  // exactly d_c: fully weakened
  CHECK(near(law.friction_at(s, nullptr, nullptr, 0.0, 3), 0.20));
  s[0] = 1.0;  // far beyond: saturated, never below mu_d
  CHECK(near(law.friction_at(s, nullptr, nullptr, 0.0, 3), 0.20));
}

// And the current jump weakens it too, not only the committed state: within an
// outer iteration the fault has already slipped by the current iterate's jump,
// and a law that ignored it would lag one step behind.
MIMETIKA_TEST(the_current_jump_weakens_the_fault_within_a_step) {
  const SlipWeakening law(0.52, 0.20, 0.02);
  State s;
  const Vec3 g = vec(0.0, 0.01, 0.0);  // half of d_c, uncommitted
  CHECK(near(law.friction_at(s, &g, nullptr, 0.0, 3), 0.36));
  CHECK(near(law.friction_at(s, nullptr, nullptr, 0.0, 3), 0.52));  // and without it, static
}

// It is a SignoriniCoulomb, so the unilateral normal condition, the projection
// and the slip accumulation are inherited rather than restated: a new law is
// one overridden function.
MIMETIKA_TEST(the_weakening_law_inherits_the_unilateral_normal_condition) {
  const SlipWeakening law(0.52, 0.20, 0.02);
  State s;
  const Vec3 t = law.project(vec(3.0, 1.0, 0.0), s, 3);
  CHECK(near(t[0], 0.0) && near(t[1], 0.0));  // opens, and carries no shear

  // and a weakened point slides at the lower radius
  s[0] = 1.0;  // fully weakened
  const Vec3 slid = law.project(vec(-2.0, 10.0, 0.0), s, 3);
  CHECK(near(slid.shear_norm(3), 0.20 * 2.0));
}

MIMETIKA_TEST(a_weakening_law_that_strengthens_is_refused) {
  bool refused = false;
  try {
    SlipWeakening bad(0.20, 0.52, 0.02);  // mu_d > mu_s
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
  refused = false;
  try {
    SlipWeakening bad(0.52, 0.20, 0.0);  // d_c = 0
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

// -- rate and state ------------------------------------------------------------

// Theta starts at theta0, not at zero, and the driver asks the law rather than
// assuming. The coefficient carries log(theta), so a zero start is minus
// infinity.
MIMETIKA_TEST(rate_and_state_initialises_theta) {
  const RateAndStateFriction law(0.6, 0.01, 0.015, 1e-4, 1e-6, 2.5);
  const State s = law.initial_state();
  CHECK(near(s[1], 2.5));
  CHECK(near(s[0], 0.0));
  CHECK(law.n_state() == 2);
  CHECK(law.rate_dependent());
  CHECK(law.path_dependent());      // inherited: the slip history is still kept
  CHECK(!law.symmetric_tangent());  // and it is still non-associated
}

// The direct effect a ln(V/V0): friction rises with the slip rate.
MIMETIKA_TEST(friction_increases_with_the_slip_rate) {
  const RateAndStateFriction law(0.6, 0.01, 0.0, 1e-4, 1e-6);
  const double fast = law.friction_coefficient(1e-5, 1.0);
  const double slow = law.friction_coefficient(1e-7, 1.0);
  std::printf("  V 1e-5 -> mu %.6f   V 1e-7 -> mu %.6f\n", fast, slow);
  CHECK(fast > slow);
}

// The evolution effect b ln(V0 theta / Dc): friction rises as contact matures.
MIMETIKA_TEST(friction_increases_with_the_state_variable) {
  const RateAndStateFriction law(0.6, 0.0, 0.015, 1e-4, 1e-6);
  const double mature = law.friction_coefficient(1e-6, 10.0);
  const double fresh = law.friction_coefficient(1e-6, 1.0);
  std::printf("  theta 10 -> mu %.6f   theta 1 -> mu %.6f\n", mature, fresh);
  CHECK(mature > fresh);
}

// a and b are separate because the sign of a - b decides whether the fault is
// velocity-weakening and so capable of unstable slip. At steady state
// theta = Dc/V, so mu(V) = mu0 + (a - b) ln(V/V0) and the two effects compete.
MIMETIKA_TEST(the_steady_state_coefficient_is_governed_by_a_minus_b) {
  for (const bool weakening : {true, false}) {
    const double a = 0.010, b = weakening ? 0.015 : 0.005;
    const RateAndStateFriction law(0.6, a, b, 1e-4, 1e-6);
    const double slow = law.friction_coefficient(1e-7, 1e-4 / 1e-7);  // theta = Dc/V
    const double fast = law.friction_coefficient(1e-5, 1e-4 / 1e-5);
    std::printf("  a - b = %+.3f   mu(V=1e-7) %.6f   mu(V=1e-5) %.6f\n", a - b, slow, fast);
    CHECK((fast < slow) == weakening);
  }
}

// The aging law is stable and reaches steady state. Integrated implicitly,
// theta -> Dc/V under sustained slip and is never driven negative; the dt here
// is deliberately large, which is where the explicit form fails.
MIMETIKA_TEST(the_aging_law_is_stable_and_reaches_steady_state) {
  const double Dc = 1e-4, V = 1e-3, dt = 1e-2;
  const RateAndStateFriction law(0.6, 0.01, 0.015, Dc, 1e-6);
  State s = law.initial_state();
  Vec3 g_prev = vec(0.0, 0.0, 0.0);
  for (int k = 0; k < 2000; ++k) {
    const Vec3 g = vec(0.0, V * dt * (k + 1), 0.0);
    law.advance(Vec3{}, &g, s, 3, &g_prev, dt);
    g_prev = g;
    CHECK(s[1] > 0.0);  // never negative, at any step
  }
  std::printf("  theta %.6e   steady state Dc/V %.6e\n", s[1], Dc / V);
  CHECK(near(s[1], Dc / V, 1e-3 * Dc / V));
}

// And it inherits the unilateral normal condition unchanged: a tensile trial is
// projected to zero, whatever the coefficient came out to be.
MIMETIKA_TEST(rate_and_state_inherits_the_unilateral_normal_condition) {
  const RateAndStateFriction law;
  State s = law.initial_state();
  const Vec3 t = law.project(vec(3.0, 1.0, 0.0), s, 3, nullptr, nullptr, 1.0);
  for (std::size_t k = 0; k < 3; ++k) CHECK(near(t[k], 0.0));
}

// The slip-rate floor avoids log(0). A stuck point has V = 0 exactly, which
// would otherwise take the coefficient to minus infinity.
MIMETIKA_TEST(the_slip_rate_floor_avoids_the_logarithm_of_zero) {
  const RateAndStateFriction law;
  const Vec3 g = vec(0.0, 0.0, 0.0);
  CHECK(law.slip_rate(&g, &g, 1.0, 3) >= law.minimum_rate());
  CHECK(law.slip_rate(nullptr, nullptr, 0.0, 3) >= law.minimum_rate());
  CHECK(std::isfinite(law.friction_coefficient(law.slip_rate(&g, &g, 1.0, 3), 1.0)));
  // and the coefficient is clipped at zero rather than going negative
  CHECK(law.friction_coefficient(law.minimum_rate(), 1e-300) >= 0.0);
}

// The rate enters the projection: two identical trial tractions differing only
// in how fast the fault is moving must be projected onto friction disks of
// different radius.
MIMETIKA_TEST(the_slip_rate_changes_the_friction_disk) {
  const RateAndStateFriction law(0.6, 0.05, 0.0, 1e-4, 1e-6);  // a large direct effect
  const Vec3 trial = vec(-1.0, 10.0, 0.0);                     // far outside the cone
  const Vec3 zero = vec(0.0, 0.0, 0.0);
  const Vec3 slow = vec(0.0, 1e-9, 0.0), fast = vec(0.0, 1e-4, 0.0);
  State a = law.initial_state(), b = law.initial_state();
  const Vec3 ts = law.project(trial, a, 3, &slow, &zero, 1.0);
  const Vec3 tf = law.project(trial, b, 3, &fast, &zero, 1.0);
  std::printf("  slow |t_t| %.6f   fast |t_t| %.6f\n", ts.shear_norm(3), tf.shear_norm(3));
  CHECK(tf.shear_norm(3) > ts.shear_norm(3));  // faster slip, stronger fault
}

// -- the dimension contract --------------------------------------------------

// A law is written once for both dimensions: `dim` says how many shear
// components are live, and the second one is not touched in the plane.
MIMETIKA_TEST(a_two_dimensional_law_has_one_shear_direction) {
  const SignoriniCoulomb law(0.5);
  State s;
  const Vec3 t = law.project(vec(-2.0, 10.0, 999.0), s, 2);
  CHECK(near(t.shear_norm(2), 1.0));  // mu |t_n| = 0.5 * 2
  CHECK(near(t[1], 1.0));
  CHECK(t[2] == 0.0);  // the third component is never written in 2D
}

MIMETIKA_TEST_MAIN()
