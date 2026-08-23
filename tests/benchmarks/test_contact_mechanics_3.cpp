#include <cstdio>
#include <cstdlib>

#include "../mimetika_test.hpp"
#include "novikov_fault.hpp"

// Benchmark 3 of Novikov et al. (2024): the inclined displaced fault with
// slip-weakening friction (paper 4.2).
//
// The configuration of benchmark 2, with the coefficient falling linearly from
// mu_s = 0.52 to mu_d = 0.20 over a critical slip distance delta_c = 0.02 m
// (paper Eq. 23). Slip now reduces the fault's carrying capacity, so the
// problem stops being monotone: below a nucleation pressure p* no quasi-static
// equilibrium exists and the fault runs away -- a seismic event.
//
// The answer is a pressure, not a profile. The paper's semi-analytical estimate
// (Uenishi & Rice 2003 as modified by Jansen & Meulenbroek 2022) is
// p* = -17.41 MPa and its own DARTS simulation reaches -17.27 MPa. The
// benchmark walks the depletion down, each level warm-started from the previous
// equilibrium, until the stable branch is lost; the level that fails brackets
// p*.
//
// The answer is resolution-critical. The Uenishi--Rice critical nucleation
// length here is
//
//     h* ~ 1.16 G delta_c / ((1 - nu)(mu_s - mu_d) |sigma_n'|)  ~  16 m,
//
// so a mesh that does not resolve h* makes every resolvable slipping patch
// supercritical the moment it appears, and the computed p* collapses onto the
// mesh-dependent constant-friction slip onset instead of the nucleation
// pressure. The Python reference records this, and the paper's own grid is 2 m
// on the fault. What is asserted below is therefore split: the law-and-branch
// statements hold at any resolution, and p* is reported against the paper with
// the mesh it was computed on named, rather than asserted at a resolution that
// cannot support it.

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

// The coefficient weakens and then saturates, which is the whole of Eq. 23 and
// what makes the branch fold. Checked on the benchmark's own numbers so that a
// failure downstream cannot be blamed on the law.
MIMETIKA_TEST(the_benchmark_three_law_is_the_published_one) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // survive a crash mid-sweep
  const SlipWeakening law(kMuStatic, kMuDynamic, kCritical);
  CHECK(law.mu_static() == kMuStatic && law.mu_dynamic() == kMuDynamic);
  CHECK(law.critical_slip() == kCritical);

  const mimetika::contact::State fresh;
  CHECK(std::abs(law.friction_at(fresh, nullptr, nullptr, 0.0, 2) - kMuStatic) < 1e-14);
  mimetika::contact::State half;
  half[0] = 0.5 * kCritical;
  CHECK(std::abs(law.friction_at(half, nullptr, nullptr, 0.0, 2) - 0.5 * (kMuStatic + kMuDynamic)) <
        1e-14);
  mimetika::contact::State past;
  past[0] = 10.0 * kCritical;
  CHECK(std::abs(law.friction_at(past, nullptr, nullptr, 0.0, 2) - kMuDynamic) < 1e-14);
}

// And h* is a length the mesh must resolve. Reported rather than assumed,
// because it decides whether a computed p* means anything.
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

// Walking down to nucleation. Each level is warm-started from the previous
// equilibrium, which makes this a continuation along the stable branch rather
// than a sequence of unrelated solves: past the fold there is no equilibrium to
// find from any start, and the failure to find one is the physics rather than a
// solver defect.
//
// Two things must hold at any resolution, and they are what is asserted:
//
//   An equilibrium exists well above p*. At shallow depletion the fault is
//   barely loaded and the weakening law behaves like Coulomb at mu_s.
//
//   The branch is eventually lost. Continue far enough and the solve stops
//   converging, or the slip runs past every scale the law has -- 50 delta_c is
//   the Python's own runaway test.
// Plain Coulomb at a frozen, per-point friction coefficient.
//
// The outer iteration below feeds each enforcement point its own mu, taken from
// the previous iterate's slip. The law itself must therefore be Coulomb, not
// slip-weakening, with the coefficient read rather than computed, which is what
// `friction_at` being virtual is for. The frozen value is carried in the point's
// own state, so nothing outside the law has to know about it.
class FrozenCoulomb final : public SignoriniCoulomb {
 public:
  explicit FrozenCoulomb(double fallback) : SignoriniCoulomb(fallback) {}
  std::size_t n_state() const override { return 2; }
  double friction_at(const mimetika::contact::State& state, const Vec3*, const Vec3*, double,
                     int) const override {
    return state[1] > 0.0 ? state[1] : friction();
  }
};

// The slip-weakening branch, tracked by the mu-update map.
//
// Handing the coupled law to Newton does not work near the fold: the stable and
// the fully-weakened equilibria draw close and Newton hops between their basins,
// reporting a runaway at a pressure where the branch is still alive.
//
// The physical tracker is an outer fixed point on a frozen coefficient: solve
// plain Coulomb at mu, read the slip, recompute mu from it, repeat. The
// linearisation of that map is the slip-weakening stability operator, so it
// contracts while the quasi-static branch is stable and diverges at the
// Uenishi--Rice fold; p* is where it stops contracting.
//
// Near the fold the contraction factor approaches one and the iteration creeps,
// so an iteration cap cannot tell slow convergence from divergence. The
// mu-update magnitude can: it shrinks on the stable side and grows past the
// fold, which is what the `growing` counter watches.
//
// Only mu is continued across levels. The inner solves are cold: with a warm
// previous jump the tangential driving becomes increment-based and
// near-threshold facets flip slip direction on noise-scale increments, which
// loses the branch earlier than it is really lost.
MIMETIKA_TEST(the_stable_branch_is_followed_and_then_lost) {
  const Parameters p = wide();
  const Setup s = build(p, 2.0, 1500.0, 300.0, 20.0, 1.0);
  const FrozenCoulomb inner(kMuStatic);
  const std::size_t n = s.fault.size();
  // one construction, one factorization, for the whole sweep
  novikov::Prepared prep = novikov::prepare(s, p);

  // one level: the frozen-mu outer fixed point. Returns whether the branch held.
  std::vector<double> mu;  // continued across levels; only mu is
  Slipped best;
  const auto attempt = [&](double level, std::vector<double>& mu_io, Slipped& got) {
    std::vector<double> mu_pts = mu_io;
    double last_change = std::numeric_limits<double>::infinity();
    int growing = 0, outer = 0;
    bool converged = false, runaway = false;
    Slipped r;
    for (outer = 1; outer <= 120; ++outer) {
      // Preconditioned, not cold. Zero traction with a zero jump is not a state
      // the mechanics could be in -- the fault carries nothing while the rock
      // is fully loaded -- and the first projection then absorbs the whole
      // in-situ imbalance and leaves the physical basin. The locked solve is an
      // equilibrated alternative one direct solve away. mu rides in the state.
      ContactState warm = novikov::locked_start(s, p, level, inner);
      for (std::size_t i = 0; i < n; ++i) {
        warm.internal[i][1] = mu_pts.empty() ? kMuStatic : mu_pts[i];
      }
      r = novikov::solve_on(prep, s, p, level, inner, &warm);

      std::vector<double> mu_new(n);
      for (std::size_t i = 0; i < n; ++i) {
        mu_new[i] =
            std::max(kMuDynamic, kMuStatic - (kMuStatic - kMuDynamic) * r.slip[i] / kCritical);
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
        if (growing >= 10) break;
      }
      last_change = change;
      mu_pts = mu_new;
    }
    std::printf("   %9.3f   %4d   %10.4f   %s\n", level / 1e6, outer, 1e3 * r.peak,
                converged ? "converged" : (runaway ? "RUNAWAY" : "NO EQUILIBRIUM"));
    if (converged) {
      mu_io = mu_pts;
      got = r;
    }
    return converged;
  };

  double last_stable = 0.0, nucleated = 0.0;
  std::printf("      dp [MPa]  outer  peak slip [mm]   status\n");
  for (int step = 0; step <= 8; ++step) {
    const double level = -16.5e6 - 0.25e6 * step;
    if (!attempt(level, mu, best)) {
      nucleated = level;
      break;
    }
    last_stable = level;
  }

  // Bisect the bracket. The paper's p* sits in a window far narrower than any
  // fixed step: 0.016 MPa. The fold is where the mu-update map stops
  // contracting, a property of the level, so it is found by halving the interval
  // between the deepest equilibrium and the first level without one.
  if (nucleated < 0.0 && last_stable < 0.0) {
    for (int k = 0; k < 4; ++k) {
      const double mid = 0.5 * (last_stable + nucleated);
      std::vector<double> mu_try = mu;
      Slipped got;
      if (attempt(mid, mu_try, got)) {
        last_stable = mid;
        mu = mu_try;
        best = got;
      } else {
        nucleated = mid;
      }
    }
  }

  std::printf("  last equilibrium %.4f MPa   bracket (%.2f, %.2f]\n", last_stable / 1e6,
              nucleated / 1e6, last_stable / 1e6);
  std::printf(
      "  paper: p* = -17.41 MPa (semi-analytical), -17.27 (DARTS);"
      " Python port -17.11\n");
  CHECK(last_stable < 0.0);
  CHECK(nucleated < last_stable);
  CHECK(std::abs(last_stable / 1e6 + 17.4) < 1.5);

  // NOTHING TO COMPARE IF NO LEVEL HELD. CHECK records and continues, so the
  // comparison below would index an unassigned state -- guard it rather than
  // segfault on the way to reporting the real failure.
  if (best.slip.size() != n) {
    std::printf("  Fig. 14: skipped -- no equilibrium was found on the sweep\n");
    return;
  }

  // Fig. 14 is a comparison of the pre-nucleation state, not a pointwise one.
  //
  // The paper's curve is its own last equilibrium, at p* = -17.41 MPa; ours is
  // at whatever level our fold lands on. Those are different pressures, so
  // demanding pointwise equality would be demanding that two solutions of two
  // different problems coincide. What is comparable is the state: how far the
  // fault has slipped and the shape of the profile -- the peak and the rms.
  const auto ref = novikov::read_reference("14_right.csv");
  const std::vector<double>& ry = ref["y"];
  const std::vector<double>& rd = ref["delta"];
  double ref_peak = 0.0;
  for (const double v : rd) ref_peak = std::max(ref_peak, std::abs(v));

  double sum = 0.0;
  int compared = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double y = s.y[i];
    if (y < ry.front() || y > ry.back()) continue;
    ++compared;
    const double want = std::abs(novikov::Reference::at(ry, rd, y));
    sum += (best.slip[i] - want) * (best.slip[i] - want);
  }
  const double rms = std::sqrt(sum / std::max(1, compared));
  std::printf("  Fig. 14: peaks %.3f / %.3f mm   rms %.3f mm   (%d points)\n", 1e3 * best.peak,
              1e3 * ref_peak, 1e3 * rms, compared);
  std::printf("  (different pressures -- the pre-nucleation STATE is what is comparable)\n");
  CHECK(compared > 50);
  CHECK(std::abs(best.peak - ref_peak) < 1.5e-3);  // peaks within 1.5 mm
  CHECK(rms < 1.0e-3);                             // Python reports 0.20 mm
}

MIMETIKA_TEST_MAIN()
