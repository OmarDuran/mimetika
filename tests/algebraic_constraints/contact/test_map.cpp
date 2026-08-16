#include <cmath>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/map.hpp"

// y = CD(x) AS A NONLINEAR ALGEBRAIC FUNCTION, tested without a mesh.
//
// That the contact problem is ONLY algebra is the claim the design makes, so it
// is the claim these tests check: the mechanics below is a hand-written 2x2
// system. No mesh, no material, no boundary condition and no model object
// appears anywhere in this file -- which is the same thing as saying the driver
// will plug into CauchyElasticityModel and PoroelasticModel without either
// being mentioned here.
//
// The stub is small enough to have a closed form. With one enforcement point,
// to_moments = [1] and the system
//
//     [[a, c], [c, d]] z = [b0, b1],   dofs = [0],   J = [0, 1]
//
// pinning z_0 = x leaves z_1 = (b1 - c x) / d, so
//
//     g(x) = (b1 - c x) / d    and    CD(x) = P(x + r g(x)).
//
// With the identity projection the fixed point is x* = b1 / c, where g(x*) = 0
// -- the bilateral contact condition -- and every statement below is checked
// against those two formulas rather than against a previous run.

using mimetika::contact::ContactLaw;
using mimetika::contact::ContactMap;
using mimetika::contact::ContactMechanics;
using mimetika::contact::fixed_point;
using mimetika::contact::FixedPointOptions;
using mimetika::contact::State;
using mimetika::contact::Vec3;

namespace {

bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol; }

// THE COUPLING SIGN IS NOT FREE. g(x) = (b1 - c x)/d must DECREASE as the
// traction grows -- pushing harder closes the gap -- so c/d > 0. With the
// opposite sign the map has multiplier 1 + r|c|/d > 1 and no r converges, which
// is a statement about contact being unstable, not about the solver.
constexpr double kA11 = 4.0, kC12 = 1.0, kD22 = 2.0;
constexpr double kB0 = 0.0, kB1 = 3.0;

double exact_gap(double x) { return (kB1 - kC12 * x) / kD22; }
constexpr double kExactFixedPoint = kB1 / kC12;

// The 2x2 stub, standing where a mixed elasticity or poromechanics system will:
// pin z_0, solve, report z_1 as the gap.
class StubMechanics final : public ContactMechanics {
 public:
  std::size_t n_points() const override { return 1; }
  int dim() const override { return 1; }  // one component: the normal alone
  std::size_t n_dofs() const override { return 2; }

  void to_moments(const std::vector<Vec3>& x, std::vector<double>& moments) const override {
    moments.assign(1, x[0][0]);
  }

  void solution_operator(const std::vector<double>& moments,
                         std::vector<double>& z) const override {
    ++solves;
    const double x = moments[0];
    z.assign(2, 0.0);
    z[0] = x;                        // pinned
    z[1] = (kB1 - kC12 * x) / kD22;  // the remaining row
  }

  void gap(const std::vector<double>& z, std::vector<Vec3>& g) const override {
    g.assign(1, Vec3{});
    g[0][0] = z[1];  // J = [0, 1]
  }

  mutable int solves{0};
};

// No admissible set: the projection is the identity.
class Identity final : public ContactLaw {
 public:
  std::string name() const override { return "identity"; }
  Vec3 project(const Vec3& trial, State&, int, const Vec3*, const Vec3*, double) const override {
    return trial;
  }
};

// Signorini's normal condition alone: t_n <= 0.
class ClampNormal final : public ContactLaw {
 public:
  std::string name() const override { return "clamp_normal"; }
  Vec3 project(const Vec3& trial, State&, int dim, const Vec3*, const Vec3*,
               double) const override {
    Vec3 t;
    for (int k = 0; k < dim; ++k) {
      t[static_cast<std::size_t>(k)] = std::min(trial[static_cast<std::size_t>(k)], 0.0);
    }
    return t;
  }
};

std::vector<Vec3> at(double x) {
  std::vector<Vec3> v(1);
  v[0][0] = x;
  return v;
}

}  // namespace

// -- the map is a function ---------------------------------------------------

MIMETIKA_TEST(the_gap_is_the_closed_form) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  for (const double x : {-2.0, 0.0, 1.0, 3.0, 7.5}) {
    CHECK(near(map.evaluate(at(x)).gap[0][0], exact_gap(x)));
  }
}

MIMETIKA_TEST(the_map_is_the_closed_form) {
  const StubMechanics m;
  const Identity law;
  for (const double r : {0.1, 0.5, 1.0}) {
    const ContactMap map(m, law, {r});
    for (const double x : {-2.0, 0.0, 3.0}) {
      CHECK(near(map.evaluate(at(x)).value[0][0], x + r * exact_gap(x)));
    }
  }
}

// THE PINNED UNKNOWN REALLY IS PINNED -- an essential condition, not a
// compliance. It matters: with an augmented relation inside the operator the
// solved traction would be the TRIAL value, so an open fracture would come out
// carrying tension.
MIMETIKA_TEST(the_pinned_unknown_really_is_pinned) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  for (const double x : {-1.0, 2.0, 4.0}) {
    CHECK(near(map.evaluate(at(x)).solution[0], x));
  }
}

MIMETIKA_TEST(the_map_is_affine_under_an_affine_projection) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  // CD(a x1 + (1-a) x2) = a CD(x1) + (1-a) CD(x2)
  const double x1 = -1.0, x2 = 5.0, a = 0.3;
  const double lhs = map.evaluate(at(a * x1 + (1.0 - a) * x2)).value[0][0];
  const double rhs =
      a * map.evaluate(at(x1)).value[0][0] + (1.0 - a) * map.evaluate(at(x2)).value[0][0];
  CHECK(near(lhs, rhs));
}

MIMETIKA_TEST(the_residual_vanishes_only_at_the_fixed_point) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  CHECK(map.residual(at(kExactFixedPoint)) < 1e-12);
  CHECK(map.residual(at(kExactFixedPoint + 1.0)) > 1e-6);
}

// AND THE GAP CLOSES THERE: the bilateral contact condition, which is what the
// fixed point means physically.
MIMETIKA_TEST(the_gap_closes_at_the_fixed_point) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  CHECK(near(map.evaluate(at(kExactFixedPoint)).gap[0][0], 0.0));
}

// -- the iteration -----------------------------------------------------------

// A SLOW CONTRACTION IS STILL A CONTRACTION. The multiplier is |1 - r c/d|,
// which is 0.9 at r = 0.2, so reaching 1e-14 takes some 700 iterations -- the
// budget has to be sized to the contraction rate, not to a habit.
MIMETIKA_TEST(fixed_point_finds_it) {
  const StubMechanics m;
  const Identity law;
  for (const double r : {0.2, 0.5, 1.0}) {
    const ContactMap map(m, law, {r});
    FixedPointOptions opt;
    opt.relaxation = 1.0;
    opt.tolerance = 1e-14;
    opt.max_iterations = 5000;
    const auto res = fixed_point(map, opt);
    CHECK(res.converged);
    CHECK(near(res.x[0][0], kExactFixedPoint, 1e-8 * kExactFixedPoint));
  }
}

MIMETIKA_TEST(fixed_point_starts_from_the_supplied_guess) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  const std::vector<Vec3> x0 = at(kExactFixedPoint);
  FixedPointOptions opt;
  opt.tolerance = 1e-12;
  const auto res = fixed_point(map, opt, &x0);
  CHECK(res.converged && res.iterations == 1);  // already there
}

// THE CONTRACTION CONDITION IS THE EXPECTED ONE. CD has multiplier
// |1 - r c/d|, so the iteration contracts exactly for r < 2 d / c = 4 here --
// which is the statement "r < 2 / compliance" the driver derives its
// augmentation from.
MIMETIKA_TEST(the_contraction_condition_is_the_expected_one) {
  const StubMechanics m;
  const Identity law;
  const double limit = 2.0 * kD22 / kC12;
  FixedPointOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-12;
  opt.max_iterations = 500;

  const ContactMap inside(m, law, {0.9 * limit});
  CHECK(fixed_point(inside, opt).converged);

  const ContactMap outside(m, law, {1.5 * limit});
  CHECK(!fixed_point(outside, opt).converged);
}

// A DIVERGING ITERATION MUST REPORT FAILURE, not succeed by overflowing: once
// x is non-finite, tolerance * max(|x|, 1) is inf and an unguarded test would
// pass.
MIMETIKA_TEST(too_large_an_augmentation_diverges_and_says_so) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {50.0});
  FixedPointOptions opt;
  opt.relaxation = 1.0;
  opt.max_iterations = 400;
  const auto res = fixed_point(map, opt);
  CHECK(!res.converged);
}

// AND RELAXATION RESCUES IT: damping is what restores convergence when the
// plain iteration is not a contraction -- the sliding case in practice.
MIMETIKA_TEST(relaxation_rescues_a_divergent_augmentation) {
  const StubMechanics m;
  const Identity law;
  const double limit = 2.0 * kD22 / kC12;
  const ContactMap map(m, law, {1.5 * limit});
  FixedPointOptions opt;
  opt.relaxation = 0.2;
  opt.tolerance = 1e-12;
  opt.max_iterations = 2000;
  const auto res = fixed_point(map, opt);
  CHECK(res.converged);
  CHECK(near(res.x[0][0], kExactFixedPoint, 1e-6));
}

MIMETIKA_TEST(the_iteration_count_is_reported_honestly) {
  const StubMechanics m;
  const Identity law;
  const ContactMap map(m, law, {0.5});
  FixedPointOptions opt;
  opt.max_iterations = 3;
  opt.tolerance = 1e-14;
  const auto res = fixed_point(map, opt);
  CHECK(res.iterations == 3);
  CHECK(!res.converged);
  CHECK(m.solves >= 3);  // one solve per evaluation, no caching behind the back
}

// -- the projection is what makes it contact ---------------------------------

MIMETIKA_TEST(the_projection_is_applied) {
  const StubMechanics m;
  const ClampNormal law;
  const ContactMap map(m, law, {0.5});
  // at a large positive x the trial is positive, so the clamp bites
  CHECK(near(map.evaluate(at(10.0)).value[0][0], 0.0));
}

// A CLAMPING LAW MOVES THE FIXED POINT OFF THE BILATERAL ONE, which is the
// whole point of a unilateral condition: the bilateral answer x* = 3 is
// inadmissible under t_n <= 0, so the constrained solution sits at the bound.
MIMETIKA_TEST(a_clamping_law_moves_the_fixed_point_off_the_bilateral_one) {
  const StubMechanics m;
  const ClampNormal law;
  const ContactMap map(m, law, {0.5});
  FixedPointOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-12;
  const auto res = fixed_point(map, opt);
  CHECK(res.converged);
  CHECK(res.x[0][0] <= 1e-12);                            // admissible
  CHECK(std::abs(res.x[0][0] - kExactFixedPoint) > 1.0);  // and not the bilateral one
}

// -- the driving gap ---------------------------------------------------------

// THE NORMAL IS DRIVEN BY THE TOTAL GAP, THE TANGENTIAL BY THE INCREMENT, and
// the asymmetry is physical: g_n >= 0 is a statement about the absolute gap,
// while Coulomb friction opposes the slip RATE.
MIMETIKA_TEST(the_driving_gap_is_total_in_normal_and_incremental_in_shear) {
  Vec3 g, prev;
  g[0] = 2.0;
  g[1] = 5.0;
  g[2] = 7.0;
  prev[0] = 1.0;
  prev[1] = 3.0;
  prev[2] = 4.0;

  const Vec3 with = mimetika::contact::driving_gap(g, &prev, 3);
  CHECK(near(with[0], 2.0));                        // normal: TOTAL, the previous value ignored
  CHECK(near(with[1], 2.0) && near(with[2], 3.0));  // shear: the increment

  const Vec3 without = mimetika::contact::driving_gap(g, nullptr, 3);
  CHECK(near(without[0], 2.0) && near(without[1], 5.0) && near(without[2], 7.0));
}

// -- prestress ---------------------------------------------------------------

// THE LAW SEES THE TOTAL TRACTION WHILE x STAYS INCREMENTAL. A fault under
// in-situ compression must not be opened by a tensile INCREMENT, so the
// prestress is added before the projection and removed after.
MIMETIKA_TEST(prestress_shows_the_law_the_total_and_returns_the_increment) {
  const StubMechanics m;
  const ClampNormal law;
  ContactMap map(m, law, {0.5});

  // without prestress a positive trial is clamped to zero
  const double x = 10.0;
  CHECK(near(map.evaluate(at(x)).value[0][0], 0.0));

  // with a firmly compressive in-situ state the SAME increment stays free: the
  // total is still compressive, so the clamp does not bite, and what comes back
  // is the increment rather than the total
  std::vector<Vec3> pre(1);
  pre[0][0] = -1000.0;
  map.set_prestress(pre);
  const double y = map.evaluate(at(x)).value[0][0];
  const double trial = x + (-1000.0) + 0.5 * exact_gap(x);
  CHECK(near(y, std::min(trial, 0.0) - (-1000.0)));
  CHECK(y > 0.0);  // a tensile increment, admissible because the total is not
}

MIMETIKA_TEST(an_augmentation_that_is_not_positive_is_refused) {
  const StubMechanics m;
  const Identity law;
  bool refused = false;
  try {
    const ContactMap bad(m, law, {0.0});
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

MIMETIKA_TEST_MAIN()
