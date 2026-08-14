#include "../mimetika_test.hpp"
#include "novikov_fault.hpp"

// BENCHMARK 3 of Novikov et al. (2024): the inclined displaced fault with
// SLIP-WEAKENING friction (paper 4.2).
//
// The configuration of benchmark 2, with the coefficient falling linearly from
// mu_s = 0.52 to mu_d = 0.20 over a critical slip distance delta_c = 0.02 m
// (paper Eq. 23). Slip now REDUCES the fault's carrying capacity, so the
// problem stops being monotone: below a NUCLEATION PRESSURE p* no quasi-static
// equilibrium exists at all and the fault runs away -- a seismic event.
//
// THE ANSWER IS A PRESSURE, not a profile. The paper's semi-analytical estimate
// (Uenishi & Rice 2003 as modified by Jansen & Meulenbroek 2022) is
// p* = -17.41 MPa and its own DARTS simulation reaches -17.27 MPa. The
// benchmark walks the depletion down, each level WARM-STARTED from the previous
// equilibrium, until the stable branch is lost; the level that fails brackets
// p*.
//
// AND IT IS A RESOLUTION-CRITICAL ANSWER. The Uenishi--Rice critical nucleation
// length here is
//
//     h* ~ 1.16 G delta_c / ((1 - nu)(mu_s - mu_d) |sigma_n'|)  ~  16 m,
//
// so a mesh that does not resolve h* makes every resolvable slipping patch
// supercritical the moment it appears, and the computed p* collapses onto the
// mesh-dependent constant-friction slip ONSET instead of the nucleation
// pressure. The Python reference records exactly this, and the paper's own grid
// is 2 m on the fault. What is asserted below is therefore split: the
// law-and-branch statements hold at any resolution, and p* is REPORTED against
// the paper with the mesh it was computed on named, rather than asserted at a
// resolution that cannot support it.

using namespace novikov;  // the shared inclined-fault setup
using mimetika::contact::SlipWeakening;

namespace {

// paper Eq. 23 and Table 2
constexpr double kMuStatic = 0.52, kMuDynamic = 0.20, kCritical = 0.02;

// the Uenishi--Rice critical nucleation length, from the parameters
double nucleation_length(const Parameters& p, double effective_normal) {
  return 1.16 * p.shear_modulus * kCritical /
         ((1.0 - p.poisson) * (kMuStatic - kMuDynamic) * std::abs(effective_normal));
}

}  // namespace

// -- the law ---------------------------------------------------------------------

// THE COEFFICIENT WEAKENS AND THEN SATURATES, which is the whole of Eq. 23 and
// is what makes the branch fold. Checked here on the benchmark's own numbers so
// that a failure downstream cannot be blamed on the law.
MIMETIKA_TEST(the_benchmark_three_law_is_the_published_one) {
  const SlipWeakening law(kMuStatic, kMuDynamic, kCritical);
  CHECK(law.mu_static() == kMuStatic && law.mu_dynamic() == kMuDynamic);
  CHECK(law.critical_slip() == kCritical);

  const mimetika::contact::State fresh;
  CHECK(std::abs(law.friction_at(fresh, nullptr, nullptr, 0.0, 2) - kMuStatic) < 1e-14);
  mimetika::contact::State half;
  half[0] = 0.5 * kCritical;
  CHECK(std::abs(law.friction_at(half, nullptr, nullptr, 0.0, 2) -
                 0.5 * (kMuStatic + kMuDynamic)) < 1e-14);
  mimetika::contact::State past;
  past[0] = 10.0 * kCritical;
  CHECK(std::abs(law.friction_at(past, nullptr, nullptr, 0.0, 2) - kMuDynamic) < 1e-14);
}

// AND h* IS A LENGTH THE MESH MUST RESOLVE. Reported rather than assumed,
// because it is what decides whether a computed p* means anything.
MIMETIKA_TEST(the_critical_nucleation_length_is_of_the_order_of_ten_metres) {
  const Parameters p = wide();
  // the effective normal traction the fault sits on at the reservoir edge
  const auto insitu = p.resolved(p.fault_a, kDip);
  const double effective = insitu[0] + p.biot * (p.pressure(p.fault_a) + p.depletion);
  const double h = nucleation_length(p, effective);
  std::printf("  sigma_n' %+.2f MPa   h* %.1f m   (the paper meshes the fault at 2 m)\n",
              effective / 1e6, h);
  CHECK(h > 5.0 && h < 60.0);
}

// -- the depletion walk ------------------------------------------------------------

// WALKING DOWN TO NUCLEATION. Each level is warm-started from the previous
// equilibrium, which is what makes this a continuation along the stable branch
// rather than a sequence of unrelated solves: past the fold there is no
// equilibrium to find from any start, and the failure to find one IS the
// physics rather than a solver defect.
//
// Two things must hold at any resolution, and they are what is asserted:
//
//   AN EQUILIBRIUM EXISTS WELL ABOVE p*. At shallow depletion the fault is
//   barely loaded and the weakening law behaves like Coulomb at mu_s.
//
//   THE BRANCH IS EVENTUALLY LOST. Continue far enough and the solve stops
//   converging, or the slip runs past every scale the law has -- 50 delta_c is
//   the Python's own runaway test. A law that never lost the branch would not
//   be weakening at all.
// PLAIN COULOMB AT A FROZEN, PER-POINT FRICTION COEFFICIENT.
//
// The outer iteration below feeds each enforcement point its own mu, taken from
// the previous iterate's slip. The law itself must therefore be Coulomb -- not
// slip-weakening -- with the coefficient READ rather than computed, which is
// exactly what `friction_at` being virtual is for. The frozen value is carried
// in the point's own state, so nothing outside the law has to know about it.
class FrozenCoulomb final : public SignoriniCoulomb {
 public:
  explicit FrozenCoulomb(double fallback) : SignoriniCoulomb(fallback) {}
  std::size_t n_state() const override { return 2; }
  double friction_at(const mimetika::contact::State& state, const Vec3*, const Vec3*, double,
                     int) const override {
    return state[1] > 0.0 ? state[1] : friction();
  }
};

// THE SLIP-WEAKENING BRANCH, TRACKED BY THE MU-UPDATE MAP.
//
// Handing the coupled law to Newton does not work near the fold: the stable and
// the fully-weakened equilibria draw close and Newton hops between their basins,
// reporting a runaway at a pressure where the branch is still alive. That is
// what produced a spurious p* here before.
//
// The physical tracker is an OUTER FIXED POINT ON A FROZEN COEFFICIENT: solve
// plain Coulomb at mu, read the slip, recompute mu from it, repeat. The
// linearisation of that map IS the slip-weakening stability operator, so it
// contracts exactly while the quasi-static branch is stable and diverges at the
// Uenishi--Rice fold -- the iteration's own behaviour is the diagnostic, and
// p* is where it stops contracting.
//
// NEAR THE FOLD THE CONTRACTION FACTOR APPROACHES ONE and the iteration creeps,
// so an iteration cap cannot tell slow convergence from divergence. The
// MU-UPDATE MAGNITUDE can: it shrinks on the stable side and grows past the
// fold, which is what the `growing` counter watches.
//
// ONLY MU IS CONTINUED ACROSS LEVELS. The inner solves are deliberately cold:
// with a warm previous jump the tangential driving becomes increment-based and
// near-threshold facets flip slip direction on noise-scale increments, which
// loses the branch earlier than it is really lost.
MIMETIKA_TEST(the_stable_branch_is_followed_and_then_lost) {
  const Parameters p = wide();
  const Setup s = build(p, 2.0, 1500.0, 300.0, 20.0, 1.0);
  const FrozenCoulomb inner(kMuStatic);
  const std::size_t n = s.fault.size();

  std::vector<double> mu;  // empty = not yet started
  double last_stable = 0.0, nucleated = 0.0;
  Slipped best;
  std::printf("      dp [MPa]  outer  peak slip [mm]   status\n");

  for (int step = 0; step <= 24; ++step) {
    const double level = -14e6 - 0.25e6 * step;
    std::vector<double> mu_pts = mu;
    bool converged = false, runaway = false;
    double last_change = std::numeric_limits<double>::infinity();
    int growing = 0, outer = 0;
    Slipped r;

    for (outer = 1; outer <= 600; ++outer) {
      ContactState warm;
      warm.traction.assign(n, Vec3{});
      warm.jump.assign(n, Vec3{});
      warm.internal.assign(n, mimetika::contact::State{});
      for (std::size_t i = 0; i < n; ++i) {
        warm.internal[i][1] = mu_pts.empty() ? kMuStatic : mu_pts[i];
      }
      r = simulate(s, p, level, inner, &warm);

      std::vector<double> mu_new(n);
      for (std::size_t i = 0; i < n; ++i) {
        mu_new[i] = std::max(kMuDynamic,
                             kMuStatic - (kMuStatic - kMuDynamic) * r.slip[i] / kCritical);
      }
      double change = std::numeric_limits<double>::infinity();
      if (!mu_pts.empty()) {
        change = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          change = std::max(change, std::abs(mu_new[i] - mu_pts[i]));
        }
      }
      if (change < 1e-4) {
        converged = r.converged;
        mu_pts = mu_new;
        break;
      }
      if (r.peak > 50.0 * kCritical) {
        runaway = true;
        break;
      }
      if (std::isfinite(last_change)) {
        growing = change > last_change ? growing + 1 : 0;
        if (growing >= 10) break;  // persistently growing update: past the fold
      }
      last_change = change;
      mu_pts = mu_new;
    }

    std::printf("   %9.2f   %4d   %10.4f   %s\n", level / 1e6, outer, 1e3 * r.peak,
                converged ? "stable" : (runaway ? "RUNAWAY" : "lost"));
    if (!converged) {
      nucleated = level;
      break;
    }
    last_stable = level;
    best = r;
    mu = mu_pts;
  }

  std::printf("  last equilibrium %.4f MPa   bracket (%.2f, %.2f]\n", last_stable / 1e6,
              nucleated / 1e6, last_stable / 1e6);
  std::printf("  paper: p* = -17.41 MPa (semi-analytical), -17.27 (DARTS);"
              " Python port -17.11\n");
  CHECK(last_stable < 0.0);
  CHECK(nucleated < last_stable);
  CHECK(std::abs(last_stable / 1e6 + 17.4) < 1.5);

  // FIG. 14: the pre-nucleation Sigma_C, pointwise against the dataset
  const auto ref = novikov::read_reference("14_left.csv");
  const std::vector<double>& ry = ref["y"];
  const std::vector<double>& rc = ref["Sigma_C_post"];
  double worst = 0.0;
  int compared = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double y = s.y[i];
    if (y < ry.front() || y > ry.back()) continue;
    bool corner = false;
    for (const double e : {-p.fault_b, -p.fault_a, p.fault_a, p.fault_b}) {
      corner = corner || std::abs(y - e) < 4.0;
    }
    if (corner) continue;
    ++compared;
    worst = std::max(worst, std::abs(best.excess[i] - novikov::Reference::at(ry, rc, y)));
  }
  std::printf("  Fig. 14: %d points, worst |dSigma_C| %.3f MPa   peak slip %.3f mm\n", compared,
              worst / 1e6, 1e3 * best.peak);
  CHECK(compared > 50);
  CHECK(worst < 3.0e6);
}

MIMETIKA_TEST_MAIN()
