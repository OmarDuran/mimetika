#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/laws.hpp"

// THE ASSOCIATIVE MOHR-COULOMB RETURN MAPPING, and the consistent tangent that
// exokal's AD produces from it.
//
// Three things are checked, each against something independent of the code that
// computes it:
//
//   the projection really is the CLOSEST POINT -- against a direct search over
//   the reduced two-dimensional feasible set, which is the definition rather
//   than the implementation;
//
//   the consistent tangent really is the DERIVATIVE -- against central
//   differences of the projection, over every branch the return map has;
//
//   it really is DIFFERENT from the partial return of SignoriniCoulomb, so that
//   the two are not quietly the same code path under another name.
//
// The second is the one the AD exists for. A hand-derived tangent on this law
// carries four closed forms -- one per active set -- each of which must agree
// with the branch the projection took; here the branch is chosen once, by the
// values, and the derivative is whatever re-running that branch produces. The
// test below sweeps 300 random trials precisely so that all four are hit.

using mimetika::contact::AssociativeMohrCoulomb;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::State;
using mimetika::contact::Tangent;
using mimetika::contact::Vec3;

namespace {

constexpr double kFriction = 0.6;

struct Metric {
  double eps_n, eps_t, cohesion;
};
const Metric kMetrics[3] = {{1.0, 1.0, 0.0}, {1.0, 2.5, 0.4}, {3.0, 0.5, 1.2}};

AssociativeMohrCoulomb law(const Metric& m) {
  return AssociativeMohrCoulomb(kFriction, m.cohesion, m.eps_n, m.eps_t);
}

Vec3 make(double a, double b, double c) {
  Vec3 v;
  v[0] = a;
  v[1] = b;
  v[2] = c;
  return v;
}

Vec3 project(const SignoriniCoulomb& l, const Vec3& t) {
  State s;
  return l.project(t, s, 3);
}

// THE PROJECTION BY DEFINITION, not by the code under test: the nearest
// admissible point, found by minimising the distance directly.
//
// Rotational symmetry about the normal axis collapses the problem to the two
// unknowns (t_n, rho >= 0) on a convex planar set. For a FIXED t_n = a the
// feasible rho is the interval [0, c - mu a], on which the nearest point to the
// trial's rho is simply the clamp -- so the inner minimisation is exact and
// what remains is one convex function of a alone,
//
//     phi(a) = (a - t_n)^2/eps_n + (clamp(rho, 0, c - mu a) - rho)^2/eps_t,
//
// minimised over a <= 0 by a scan that re-brackets about the incumbent. Being
// convex, the optimum always lies within one step of the incumbent, so the
// bracket is sound. No optimiser, no candidate enumeration, and in particular
// no closed form shared with what it is checking.
std::array<double, 2> closest_point(const Vec3& trial, const Metric& m) {
  const double tn = trial[0];
  const double rho = std::sqrt(trial[1] * trial[1] + trial[2] * trial[2]);

  const auto best_rho = [&](double a) {
    return std::min(std::max(rho, 0.0), m.cohesion - kFriction * a);
  };
  const auto phi = [&](double a) {
    const double b = best_rho(a);
    return (a - tn) * (a - tn) / m.eps_n + (b - rho) * (b - rho) / m.eps_t;
  };

  // a bracket certain to contain the optimum: moving t_n below this costs more
  // in the normal term than the whole shear term could ever save
  double lo = std::min(tn, 0.0) - m.eps_n * (rho + m.cohesion + 1.0) / m.eps_t - 1.0;
  double hi = 0.0;
  const int n = 400;
  for (int level = 0; level < 60; ++level) {
    const double step = (hi - lo) / n;
    double a_best = lo, v_best = phi(lo);
    for (int i = 1; i <= n; ++i) {
      const double a = lo + i * step;
      const double v = phi(a);
      if (v < v_best) {
        v_best = v;
        a_best = a;
      }
    }
    lo = std::max(lo, a_best - step);  // convexity: the optimum is within one step
    hi = std::min(hi, a_best + step);
    if (hi - lo < 1e-15 * (1.0 + std::abs(a_best))) break;
  }
  const double a = 0.5 * (lo + hi);
  return {a, best_rho(a)};
}

std::mt19937 rng(1u);
Vec3 sample(double scale) {
  std::normal_distribution<double> normal(0.0, scale);
  return make(normal(rng), normal(rng), normal(rng));
}

}  // namespace

// -- it is the closest point --------------------------------------------------

// AGAINST A DIRECT SEARCH over the feasible set: the tolerance is the search's,
// not the law's. What this rules out is a candidate list that is complete for
// the cases someone thought of and misses one -- the failure mode that closed
// forms on a cone actually have.
MIMETIKA_TEST(the_projection_is_the_metric_closest_point) {
  for (const Metric& m : kMetrics) {
    const AssociativeMohrCoulomb l = law(m);
    double worst = 0.0;
    for (int trial = 0; trial < 40; ++trial) {
      const Vec3 t = sample(4.0);
      const std::array<double, 2> ref = closest_point(t, m);
      const Vec3 got = project(l, t);
      const double rho = std::sqrt(got[1] * got[1] + got[2] * got[2]);
      worst = std::max(worst, std::abs(got[0] - ref[0]));
      worst = std::max(worst, std::abs(rho - ref[1]));
    }
    std::printf("  eps (%.2f, %.2f) c %.1f   max deviation from the optimum %.2e\n", m.eps_n,
                m.eps_t, m.cohesion, worst);
    CHECK(worst < 1e-7);
  }
}

MIMETIKA_TEST(the_result_is_always_admissible) {
  for (const Metric& m : kMetrics) {
    const AssociativeMohrCoulomb l = law(m);
    for (int trial = 0; trial < 2000; ++trial) {
      const Vec3 got = project(l, sample(5.0));
      CHECK(got[0] <= 1e-12);
      CHECK(got.shear_norm(3) <= m.cohesion - kFriction * got[0] + 1e-10);
    }
  }
}

// INSIDE THE CONE THE PROJECTION IS THE IDENTITY, not a near-identity. A return
// map that perturbs an already-admissible state would inject spurious work at
// every sticking point of every step.
MIMETIKA_TEST(an_admissible_trial_is_left_alone) {
  const AssociativeMohrCoulomb l = law({1.0, 1.0, 0.0});
  for (const Vec3& t : {make(-4.0, 1.0, 0.5), make(-1.0, 0.0, 0.0), make(0.0, 0.0, 0.0)}) {
    const Vec3 got = project(l, t);
    for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(got[k] - t[k]) < 1e-14);
  }
}

MIMETIKA_TEST(the_projection_is_idempotent) {
  const AssociativeMohrCoulomb l = law({1.0, 2.5, 0.4});
  for (int trial = 0; trial < 500; ++trial) {
    const Vec3 once = project(l, sample(4.0));
    const Vec3 twice = project(l, once);
    for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(twice[k] - once[k]) < 1e-12);
  }
}

// ROTATIONAL SYMMETRY ABOUT THE NORMAL AXIS: the projection may shorten the
// shear but cannot rotate it. This is what licenses collapsing the problem to
// (t_n, rho) in the first place.
MIMETIKA_TEST(the_shear_stays_collinear_with_the_trial) {
  const AssociativeMohrCoulomb l = law({1.0, 2.5, 0.4});
  for (int trial = 0; trial < 200; ++trial) {
    const Vec3 t = sample(4.0);
    const Vec3 got = project(l, t);
    const double a = t.shear_norm(3), b = got.shear_norm(3);
    if (a > 1e-9 && b > 1e-9) {
      CHECK(std::abs(t[1] * got[2] - t[2] * got[1]) < 1e-10 * a * b);
    }
  }
}

// -- the consistent tangent ---------------------------------------------------

// THE AD TANGENT IS THE DERIVATIVE, over every branch. 300 random trials reach
// all four active sets, and the exact tangent is compared against central
// differences of the projection at each. Nothing in this test knows how the
// tangent is obtained; what it certifies is that differentiating the return map
// by re-running it agrees with differencing it.
MIMETIKA_TEST(the_tangent_is_the_derivative_over_every_branch) {
  for (const Metric& m : kMetrics) {
    const AssociativeMohrCoulomb l = law(m);
    double worst = 0.0;
    const double h = 1e-6;
    for (int trial = 0; trial < 300; ++trial) {
      const Vec3 t = sample(3.0);
      const State s;
      const Tangent exact = l.tangent(t, s, 3);
      for (int j = 0; j < 3; ++j) {
        Vec3 lo = t, hi = t;
        lo[static_cast<std::size_t>(j)] -= h;
        hi[static_cast<std::size_t>(j)] += h;
        const Vec3 pl = project(l, lo), ph = project(l, hi);
        for (int i = 0; i < 3; ++i) {
          const double fd =
              (ph[static_cast<std::size_t>(i)] - pl[static_cast<std::size_t>(i)]) / (2.0 * h);
          worst = std::max(worst, std::abs(exact(i, j) - fd));
        }
      }
    }
    std::printf("  eps (%.2f, %.2f) c %.1f   |AD - central difference| %.2e\n", m.eps_n, m.eps_t,
                m.cohesion, worst);
    CHECK(worst < 1e-6);
  }
}

// WHERE THE MAP IS THE IDENTITY SO IS ITS DERIVATIVE, and exactly: the shear
// row is computed as (t_k / rho) * rho, whose derivative is the identity only
// because the two dependences on rho cancel. That they cancel to round-off is
// the statement that the AD carries the whole chain and not part of it.
MIMETIKA_TEST(the_tangent_is_the_identity_where_the_map_is) {
  const AssociativeMohrCoulomb l = law({1.0, 1.0, 0.0});
  const State s;
  const Tangent J = l.tangent(make(-4.0, 1.0, 0.5), s, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) CHECK(std::abs(J(i, j) - (i == j ? 1.0 : 0.0)) < 1e-13);
  }
}

// SLIDING SOFTENS THE TANGENT ALONG THE SLIP DIRECTION -- the rank-deficient
// term I - m (x) m. No stiffness along the slip is the statement that further
// slip costs no further traction.
MIMETIKA_TEST(sliding_softens_the_tangent_along_the_slip_direction) {
  const AssociativeMohrCoulomb l = law({1.0, 1.0, 0.0});
  const State s;
  const Tangent J = l.tangent(make(-1.0, 10.0, 0.0), s, 3);  // shear far outside the cone, along +y
  std::printf("  along the slip %.6f   across it %.6f\n", J(1, 1), J(2, 2));
  CHECK(J(1, 1) < J(2, 2));
}

// THE COUPLING BLOCK IS PRESENT, and it is what distinguishes the associative
// law from the partial return: correcting the shear moves the normal traction
// and the other way about.
MIMETIKA_TEST(the_associative_coupling_block_is_present) {
  const AssociativeMohrCoulomb l = law({1.0, 1.0, 0.0});
  const State s;
  const Tangent J = l.tangent(make(-1.0, 10.0, 0.0), s, 3);
  CHECK(std::max(std::abs(J(1, 0)), std::abs(J(2, 0))) > 1e-6);  // shear responds to normal
  CHECK(std::max(std::abs(J(0, 1)), std::abs(J(0, 2))) > 1e-6);  // and normal to shear
}

// AND IT IS NOT SYMMETRIC. Coulomb friction is not associated in the classical
// sense, so the tangent a Newton step must solve with is unsymmetric -- which
// is a fact about the linear algebra the driver needs, not a curiosity.
MIMETIKA_TEST(the_tangent_is_not_symmetric) {
  const AssociativeMohrCoulomb l = law({3.0, 0.5, 0.0});
  const State s;
  const Tangent J = l.tangent(make(-1.0, 10.0, 0.0), s, 3);
  double worst = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) worst = std::max(worst, std::abs(J(i, j) - J(j, i)));
  }
  std::printf("  worst asymmetry %.4f\n", worst);
  CHECK(worst > 1e-8);
}

// -- every law's tangent is its own projection's derivative --------------------

// THE SAME CERTIFICATE FOR THE WHOLE CATALOGUE. Each law states its projection
// once and the tangent is that body re-run on the AD scalar, so the check that
// matters is the same for all of them: the AD tangent agrees with differencing
// the projection. SignoriniCoulomb is the one the benchmarks use, and its
// nonsmooth branches -- clipping to compression, the friction disk -- are hit
// by the sweep.
MIMETIKA_TEST(every_law_agrees_with_the_difference_quotient_of_its_projection) {
  const mimetika::contact::LinearContact linear(2.0, 3.0);
  const mimetika::contact::FrictionlessBilateral bilateral;
  const SignoriniCoulomb coulomb(0.6, 0.2);
  const mimetika::contact::SlipWeakening weakening(0.6, 0.2, 1e-3);
  const mimetika::contact::RateAndStateFriction rate(0.6, 0.01, 0.015, 1e-4, 1e-6);
  const AssociativeMohrCoulomb associative(0.6, 0.2, 1.0, 2.5);
  const mimetika::contact::ContactLaw* laws[6] = {&linear,    &bilateral, &coulomb,
                                                  &weakening, &rate,      &associative};

  // A JUMP, ITS PREDECESSOR AND A TIME STEP, so the two path- and rate-dependent
  // laws are differentiated AWAY from their trivial branch: SlipWeakening at a
  // partially weakened point, rate-and-state at a finite slip rate rather than
  // at the floor. The tangent is with respect to the TRIAL TRACTION, so these
  // are the step's data and are held fixed on both sides of the difference --
  // which is exactly what `differentiate` does with them.
  const Vec3 g_prev, g = [] {
    Vec3 v;
    v[1] = 5e-4;
    return v;
  }();

  for (const mimetika::contact::ContactLaw* l : laws) {
    double worst = 0.0;
    const double h = 1e-6;
    for (int trial = 0; trial < 200; ++trial) {
      const Vec3 t = sample(2.0);
      State s = l->initial_state();
      s[0] = 4e-4;  // a partially weakened point, so SlipWeakening is not Coulomb
      const Tangent exact = l->tangent(t, s, 3, &g, &g_prev, 1.0);
      for (int j = 0; j < 3; ++j) {
        Vec3 lo = t, hi = t;
        lo[static_cast<std::size_t>(j)] -= h;
        hi[static_cast<std::size_t>(j)] += h;
        State a = s, b = s;
        const Vec3 pl = l->project(lo, a, 3, &g, &g_prev, 1.0);
        const Vec3 ph = l->project(hi, b, 3, &g, &g_prev, 1.0);
        for (int i = 0; i < 3; ++i) {
          const double fd =
              (ph[static_cast<std::size_t>(i)] - pl[static_cast<std::size_t>(i)]) / (2.0 * h);
          worst = std::max(worst, std::abs(exact(i, j) - fd));
        }
      }
    }
    std::printf("  %-26s |AD - central difference| %.2e\n", l->name().c_str(), worst);
    CHECK(worst < 1e-6);
  }
}

// AND IN TWO DIMENSIONS, where the shear is a single component and the friction
// disk degenerates to an interval. The laws are written over `dim` rather than
// over 3, so this is the check that they mean it.
MIMETIKA_TEST(the_tangent_is_the_derivative_in_two_dimensions) {
  const SignoriniCoulomb coulomb(0.6, 0.1);
  const AssociativeMohrCoulomb associative(0.6, 0.1, 1.0, 2.5);
  const mimetika::contact::ContactLaw* laws[2] = {&coulomb, &associative};
  for (const mimetika::contact::ContactLaw* l : laws) {
    double worst = 0.0;
    const double h = 1e-6;
    for (int trial = 0; trial < 200; ++trial) {
      Vec3 t = sample(2.0);
      t[2] = 0.0;  // 2D: only the normal and one shear are live
      State s;
      const Tangent exact = l->tangent(t, s, 2);
      for (int j = 0; j < 2; ++j) {
        Vec3 lo = t, hi = t;
        lo[static_cast<std::size_t>(j)] -= h;
        hi[static_cast<std::size_t>(j)] += h;
        State a = s, b = s;
        const Vec3 pl = l->project(lo, a, 2), ph = l->project(hi, b, 2);
        for (int i = 0; i < 2; ++i) {
          const double fd =
              (ph[static_cast<std::size_t>(i)] - pl[static_cast<std::size_t>(i)]) / (2.0 * h);
          worst = std::max(worst, std::abs(exact(i, j) - fd));
        }
      }
    }
    std::printf("  2D %-23s |AD - central difference| %.2e\n", l->name().c_str(), worst);
    CHECK(worst < 1e-6);
  }
}

// -- it differs from the partial return ----------------------------------------

// IF THESE AGREED EVERYWHERE the associative law would be dead code.
MIMETIKA_TEST(it_is_not_the_same_as_the_partial_return) {
  const AssociativeMohrCoulomb associative(kFriction, 0.4, 1.0, 2.5);
  const SignoriniCoulomb partial(kFriction, 0.4);
  int differing = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const Vec3 t = sample(4.0);
    const Vec3 a = project(associative, t), b = project(partial, t);
    double d = 0.0;
    for (std::size_t k = 0; k < 3; ++k) d = std::max(d, std::abs(a[k] - b[k]));
    if (d > 1e-8) ++differing;
  }
  std::printf("  %d of 400 trials project differently\n", differing);
  CHECK(differing > 50);
}

// THE PHYSICAL SIGNATURE. The partial return updates t_n first and projects the
// shear at that fixed t_n, so sliding never alters the normal traction. The
// closest-point correction moves along the cone normal, which has a component
// along the axis -- the traction-space image of dilatancy.
MIMETIKA_TEST(sliding_changes_the_normal_traction_only_in_the_associative_law) {
  const Vec3 t = make(-1.0, 5.0, 0.0);  // compressive, shear well outside the cone
  const Vec3 a = project(AssociativeMohrCoulomb(kFriction), t);
  const Vec3 b = project(SignoriniCoulomb(kFriction), t);
  std::printf("  associative t_n %+.6f   partial t_n %+.6f\n", a[0], b[0]);
  CHECK(std::abs(b[0] + 1.0) < 1e-12);  // unchanged by the shear correction
  CHECK(a[0] < -1.0 - 1e-6);            // driven further into compression
}

MIMETIKA_TEST(the_two_agree_when_the_trial_is_admissible) {
  for (const Vec3& t : {make(-4.0, 1.0, 0.5), make(-2.0, 0.0, 0.0)}) {
    const Vec3 a = project(AssociativeMohrCoulomb(kFriction), t);
    const Vec3 b = project(SignoriniCoulomb(kFriction), t);
    for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(a[k] - b[k]) < 1e-14);
  }
}

// THE METRIC SELECTS THE PROJECTION: eps_n and eps_t are not free knobs, they
// choose which point of the cone is nearest. A very stiff normal metric resists
// moving t_n and so approaches the partial return.
MIMETIKA_TEST(the_metric_selects_the_projection) {
  const Vec3 t = make(-1.0, 5.0, 0.0);
  const Vec3 isotropic = project(AssociativeMohrCoulomb(kFriction, 0.0, 1.0, 1.0), t);
  const Vec3 stiff = project(AssociativeMohrCoulomb(kFriction, 0.0, 0.01, 1.0), t);
  const Vec3 partial = project(SignoriniCoulomb(kFriction), t);
  std::printf("  isotropic t_n %+.6f   stiff-normal t_n %+.6f   partial t_n %+.6f\n", isotropic[0],
              stiff[0], partial[0]);
  CHECK(std::abs(isotropic[0] - stiff[0]) > 1e-6);
  CHECK(std::abs(stiff[0] - partial[0]) < std::abs(isotropic[0] - partial[0]));
}

MIMETIKA_TEST_MAIN()
