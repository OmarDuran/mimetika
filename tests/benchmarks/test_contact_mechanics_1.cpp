#include <algorithm>
#include <memory>
#include <utility>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/cauchy_mechanics.hpp"
#include "mimetika/algebraic_constraints/contact/driver.hpp"
#include "mimetika/benchmarks/novikov_2024.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/solver/petsc.hpp"

// BENCHMARK 1 of Novikov et al. (2024): the VERTICAL DISPLACED FAULT, frictionless.
//
// A reservoir offset across a vertical fault is depleted by -25 MPa. The throw
// puts reservoir against seal on both sides of the fault, which loads it in
// shear; with no friction the fault slips until it carries no shear stress at
// all. This is the first benchmark with a fault, so it is the first that
// exercises the contact driver end to end on a real problem.
//
// GEOMETRY (Fig. 5). The reservoir spans [-b, a] on one side of the fault and
// [-a, b] on the other, with a = 75 m and b = 150 m -- a thickness of a + b =
// 225 m and a throw of b - a = 75 m.
//
// TWO COMPUTATIONS, and they are different problems:
//
//   THE LOCKED FAULT gives the driving Coulomb stress Sigma_C of eq. (18). No
//   contact at all -- the fault is not allowed to slip, so this is the plain
//   continuous medium under the depletion load, solved once.
//
//   THE SLIPPING FAULT gives the tent of eq. (20). The total shear then
//   vanishes, which is the frictionless condition and the thing to check.
//
// THE LAW IS SignoriniCoulomb(friction = 0) -- unilateral and frictionless,
// which is the physical model. A benchmark exists to TEST laws, so the one that
// represents the situation is the one that should be run.
//
// AND IT ONLY WORKS IF THE LAW IS GIVEN THE TOTAL TRACTION. Signorini
// constrains t_n <= 0 on the TOTAL stress, and this is an incremental problem --
// only the depletion response is solved for. The incremental normal traction
// reaches +8 MPa in tension, but the fault sits on -57 MPa of in-situ
// compression and is shut by a wide margin, so the law must be told what it is
// sitting on. That is what the prestress carries. With it, Signorini correctly
// finds the fault closed and agrees with FrictionlessBilateral to round-off;
// without it, it reads the tensile increment as opening. The deficiency is in
// the incremental formulation, not in the law -- and the test below is able to
// tell the two apart, which is the point of running both.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::benchmarks::Parameters;
using mimetika::contact::CauchyContactMechanics;
using mimetika::contact::ContactDriver;
using mimetika::contact::ContactState;
using mimetika::contact::DriverOptions;
using mimetika::contact::Fracture;
using mimetika::contact::FrictionlessBilateral;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::Vec3;
using mimetika::mesh::Family;

namespace {

bool close(double got, double want, double rel) {
  return std::abs(got - want) <= rel * std::abs(want);
}

// BENCHMARK 1 RUNS ON A MUCH WIDER DOMAIN than Table 2's 4500 m box, because
// eqs. (18)-(20) are posed for an UNBOUNDED medium and a box that small does not
// approximate one. The paper says so itself in Sect. 4.1: its own results
// deviate from the semi-analytical ones and the discrepancy disappears once W is
// increased, after which it reruns at W = 18,000 m.
//
// The HEIGHT stays at Table 2's 4500 m, and that is a physical ceiling rather
// than an economy: the in-situ state is defined by depth = D0 - y with D0 = 3500
// m, so a taller domain would put its top above the ground surface, where
// sigma_xx extrapolates to TENSION and a unilateral law opens the fault for
// reasons that have nothing to do with the reservoir.
Parameters wide() {
  Parameters p;
  p.width = 18000.0;
  return p;
}

}  // namespace

// -- the analytic solution ------------------------------------------------------

// THE PUBLISHED CONSTANTS, derived from Table 2 rather than pasted in -- the
// same discipline as Benchmark 0's in-situ state, and for the same reason.
MIMETIKA_TEST(the_published_constants) {
  const Parameters p = wide();
  const double C = p.slip_stress_scale(), A = p.slip_stiffness();
  std::printf("  C %.4e Pa (-2.95e6)   A %.4e Pa (1.2171e9)   C/A %.6f (-0.0024)\n", C, A,
              C / A);
  CHECK(close(C, -2.95e6, 2e-3));
  CHECK(close(A, 1.2171e9, 1e-4));
  CHECK(close(C / A, -0.0024, 2e-2));
}

MIMETIKA_TEST(the_geometry_is_the_published_one) {
  const Parameters p;
  CHECK(p.fault_a == 75.0 && p.fault_b == 150.0);
  CHECK(p.throw_() == 75.0);
  CHECK(p.reservoir_height() == 225.0);
}

MIMETIKA_TEST(the_peak_slip_is_the_published_value) {
  const Parameters p = wide();
  std::printf("  peak |delta| %.5f m (0.18173)\n", p.peak_slip());
  CHECK(close(p.peak_slip(), 0.18173, 1e-3));
}

// THE SLIP PROFILE IS A TENT: zero outside |y| = b, rising linearly over the
// throw, and FLAT at (C/A)(a-b) over the overlap |y| < a where reservoir faces
// reservoir across the fault.
MIMETIKA_TEST(the_slip_profile_has_the_published_shape) {
  const Parameters p = wide();
  for (const double y : {-400.0, -150.0, 150.0, 400.0}) {
    CHECK(std::abs(p.analytic_slip(y)) < 1e-12);
  }
  const double flat = p.analytic_slip(0.0);
  for (const double y : {-74.0, -40.0, 0.0, 40.0, 74.0}) {
    CHECK(close(p.analytic_slip(y), flat, 1e-12));
  }
  CHECK(close(std::abs(flat), p.peak_slip(), 1e-12));
}

// AND IT IS CONTINUOUS across all four breakpoints -- there is no jump in slip
// at the reservoir edges, only a kink.
MIMETIKA_TEST(the_slip_profile_is_continuous) {
  const Parameters p = wide();
  double worst = 0.0, prev = p.analytic_slip(-400.0);
  for (int i = 1; i <= 8000; ++i) {
    const double y = -400.0 + 800.0 * i / 8000.0;
    const double now = p.analytic_slip(y);
    worst = std::max(worst, std::abs(now - prev));
    prev = now;
  }
  std::printf("  largest step over 8000 samples %.3e m\n", worst);
  CHECK(worst < 1e-3);
}

MIMETIKA_TEST(the_slip_profile_is_symmetric_about_the_reservoir) {
  const Parameters p = wide();
  for (int i = 0; i <= 200; ++i) {
    const double y = 400.0 * i / 200.0;
    CHECK(close(p.analytic_slip(y), p.analytic_slip(-y), 1e-12) ||
          std::abs(p.analytic_slip(y)) < 1e-14);
  }
}

// THE COULOMB STRESS IS EVEN AND SINGULAR AT THE FOUR RESERVOIR EDGES, with
// OPPOSITE signs at y = a and y = b: the loading jumps one way at the reservoir
// top and the other at the seal contact, which is what drives slip in a single
// direction over the whole throw.
MIMETIKA_TEST(the_coulomb_stress_is_even_and_singular_at_the_reservoir_edges) {
  const Parameters p = wide();
  for (const double y : {10.0, 60.0, 120.0, 300.0}) {
    CHECK(close(p.analytic_coulomb_stress(y), p.analytic_coulomb_stress(-y), 1e-10));
  }
  const double near_a = p.analytic_coulomb_stress(p.fault_a * (1.0 + 1e-9));
  const double near_b = p.analytic_coulomb_stress(p.fault_b * (1.0 - 1e-9));
  std::printf("  near a %+.3e Pa   near b %+.3e Pa\n", near_a, near_b);
  CHECK(std::abs(near_a) > 1e7 && std::abs(near_b) > 1e7);
  CHECK((near_a > 0.0) != (near_b > 0.0));  // opposite singularities
}

// AND IT DECAYS FAR FROM THE RESERVOIR, which is what makes a truncated domain
// usable at all: by 4 km the driving stress is under 2% of C.
MIMETIKA_TEST(the_coulomb_stress_decays_far_from_the_reservoir) {
  const Parameters p = wide();
  const double far = p.analytic_coulomb_stress(4000.0);
  std::printf("  Sigma_C(4000 m) %.3e Pa   2%% of C %.3e Pa\n", far,
              0.02 * std::abs(p.slip_stress_scale()));
  CHECK(std::abs(far) < 0.02 * std::abs(p.slip_stress_scale()));
}

// -- the graded mesh ------------------------------------------------------------

// THE INTERFACES BECOME NODES, exactly. The reservoir edges y = +-a, +-b and the
// fault x = 0 must land on cell FACES: the depletion is assigned per cell, so an
// edge that bisects a cell shifts the loading by half a cell -- the same failure
// Benchmark 0 refuses outright, here prevented by construction instead.
MIMETIKA_TEST(the_graded_mesh_puts_a_node_on_every_interface) {
  const Parameters p = wide();
  const double a = p.fault_a, b = p.fault_b;
  const std::vector<double> ys = mimetika::mesh::graded_coordinates(
      {-b, -a, a, b}, {-p.height / 2, p.height / 2}, 12.5, 1.35, 500.0);
  const std::vector<double> xs = mimetika::mesh::graded_coordinates(
      {0.0}, {-p.width / 2, p.width / 2}, 12.5, 1.35, 500.0);

  for (const double want : {-b, -a, a, b}) {
    double best = 1e300;
    for (const double y : ys) best = std::min(best, std::abs(y - want));
    CHECK(best < 1e-9);
  }
  double on_fault = 1e300;
  for (const double x : xs) on_fault = std::min(on_fault, std::abs(x));
  CHECK(on_fault < 1e-9);

  // and the grading is monotone outwards: fine at the interfaces, coarse at the
  // boundary, never the other way about
  CHECK(std::abs(ys.front() + p.height / 2) < 1e-9);
  CHECK(std::abs(ys.back() - p.height / 2) < 1e-9);
  std::printf("  %zu x %zu nodes   finest dy %.3f m   coarsest %.1f m\n", xs.size(), ys.size(),
              [&] {
                double m = 1e300;
                for (std::size_t i = 1; i < ys.size(); ++i) m = std::min(m, ys[i] - ys[i - 1]);
                return m;
              }(),
              [&] {
                double m = 0.0;
                for (std::size_t i = 1; i < ys.size(); ++i) m = std::max(m, ys[i] - ys[i - 1]);
                return m;
              }());
}

// -- the simulation --------------------------------------------------------------

namespace {

// THE OFFSET RESERVOIR AND THE FAULT IT LOADS.
//
// Everything is solved NONDIMENSIONALLY -- stress in units of G, length in units
// of the domain height -- for the reason Benchmark 0 records: in SI units the
// compliance and divergence blocks of the mixed saddle point sit ten orders
// apart and the direct factorization breaks down. Slip comes back in metres and
// tractions in pascals.
struct Setup {
  exokal::Mesh mesh;
  std::vector<Index> fault;   // the x = 0 facets, ordered by y
  std::vector<double> y;      // their centroids, in METRES
  std::vector<Index> depleted;
  double unit{1.0}, length{1.0};
};

Setup build(const Parameters& p, double spacing, double boundary_spacing = 500.0) {
  const double a = p.fault_a, b = p.fault_b, L = p.height;
  // the interfaces the mesh must honour: the four reservoir edges in y, and the
  // fault plane in x. Nondimensionalized with everything else.
  const std::vector<double> ys = mimetika::mesh::graded_coordinates(
      {-b / L, -a / L, a / L, b / L}, {-0.5, 0.5}, spacing / L, 1.35, boundary_spacing / L);
  const std::vector<double> xs = mimetika::mesh::graded_coordinates(
      {0.0}, {-p.width / (2 * L), p.width / (2 * L)}, spacing / L, 1.35, boundary_spacing / L);

  Setup s{mimetika::mesh::tensor_product(xs, ys), {}, {}, {}, p.shear_modulus, L};
  const graphos::Complex& c = s.mesh.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, 1);

  // the fault: the INTERIOR facets on x = 0, running the full height
  std::vector<std::pair<double, Index>> ordered;
  for (Index f = 0; f < c.count(1); ++f) {
    const exokal::Point x = exokal::centroid(s.mesh, 1, f);
    if (std::abs(x[0]) > 1e-12) continue;
    const auto lo = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto hi = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    if (hi - lo != 2) continue;  // a jump needs two sides
    ordered.emplace_back(x[1], f);
  }
  std::sort(ordered.begin(), ordered.end());
  for (const auto& [yi, f] : ordered) {
    s.fault.push_back(f);
    s.y.push_back(yi * L);
  }

  // THE RESERVOIR IS DISPLACED ACROSS THE FAULT: [-b, a] on the left and
  // [-a, b] on the right, so each side faces seal over part of the throw. That
  // offset IS the loading -- with no throw there is no shear on the fault and
  // nothing to slip.
  for (Index e = 0; e < c.count(2); ++e) {
    const exokal::Point x = exokal::centroid(s.mesh, 2, e);
    const bool left = x[0] < 0.0;
    const double lower = (left ? -b : -a) / L, upper = (left ? a : b) / L;
    if (x[1] > lower && x[1] < upper) s.depleted.push_back(e);
  }
  return s;
}

// the model, in units of G, with the depletion load and the roller frame
std::unique_ptr<CauchyElasticityModel> make_model(const Setup& s, const Parameters& p,
                                                  bool prescribe_fault) {
  const int dim = 2;
  Parameters q = p;  // in units of G
  q.shear_modulus = p.shear_modulus / s.unit;
  q.depletion = p.depletion / s.unit;

  const graphos::Complex& c = s.mesh.topology();
  std::vector<Index> top, rollers;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    (std::abs(exokal::centroid(s.mesh, 1, f)[1] - 0.5) < 1e-9 ? top : rollers).push_back(f);
  }

  auto model = std::make_unique<CauchyElasticityModel>(
      s.mesh, dim, ElasticMaterial{q.shear_modulus, q.lame()});
  model->mechanics().emplace<mimetika::TractionBC>(top, std::array<double, 9>{});
  model->mechanics().emplace<mimetika::FreeSlipBC>(rollers);
  model->pressurize(s.depleted, q.depletion, q.biot, q.volumetric_compliance(dim));
  // BEFORE build(): prescribing changes which equations the system has
  if (prescribe_fault) model->prescribe_traction(s.fault);
  model->build();
  return model;
}

// THE LOCKED FAULT: no contact at all, so the shear it carries is the driving
// Coulomb stress Sigma_C of eq. (18). On a vertical fault with no in-situ shear
// the Coulomb stress IS sigma_xy.
std::vector<double> pre_slip_coulomb_stress(const Setup& s, const Parameters& p) {
  const std::unique_ptr<CauchyElasticityModel> model = make_model(s, p, false);
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto report = petsc.solve(model->system(), model->rhs(), x);
  if (!report.converged) throw std::runtime_error("locked fault: " + report.reason);
  model->accept(std::move(x));

  std::vector<double> shear;
  for (const Index f : s.fault) {
    // the facet normal is +-e_x, so the tangential component is the y one
    shear.push_back(s.unit * model->facet_traction(f)[1]);
  }
  return shear;
}

struct Slipped {
  std::vector<double> slip;      // metres, tangential
  std::vector<double> normal;    // Pa, the INCREMENTAL normal traction
  std::vector<double> shear;     // Pa, the total shear on the fault
  int iterations{0};
  bool converged{false};
  double peak{0.0};
};

// THE SLIPPING FAULT. `prestress` is the whole question of this benchmark: with
// it the unilateral law sees the total traction and finds the fault shut;
// without it, it reads the tensile increment as opening.
Slipped simulate(const Setup& s, const Parameters& p, const mimetika::contact::ContactLaw& law,
                 bool prestress = true) {
  const int dim = 2;
  const std::unique_ptr<CauchyElasticityModel> model = make_model(s, p, true);
  const Fracture fr(s.mesh, dim, s.fault, model->stress_operators().moments_per_facet());
  const CauchyContactMechanics mech(*model, fr);

  DriverOptions opt;
  opt.relaxation = 1.0;  // the projection is affine here, so no damping is needed
  opt.tolerance = 1e-11;
  opt.max_iterations = 50;
  // NEWTON, NOT PICARD. A fault that cuts the domain has a dense Ghat -- every
  // facet feels every other -- so no scalar augmentation makes I + r Ghat a
  // contraction, and the Picard sweep diverges in the normal component while its
  // slip profile is already right. Newton is indifferent to that, and for this
  // affine law it converges in a single step.
  opt.solver = DriverOptions::Solver::newton;
  const double mu = p.shear_modulus / s.unit;
  const double lam = 2.0 * mu * p.poisson / (1.0 - 2.0 * p.poisson);
  ContactDriver driver(mech, law,
                       mimetika::contact::default_augmentation(s.mesh, dim, s.fault, mu, lam),
                       opt);

  if (prestress) {
    // A CONTACT LAW CONSTRAINS THE TOTAL TRACTION, so an incremental solve has
    // to tell it what it is sitting on. On a vertical fault the in-situ normal
    // traction is sigma_xx(y) ~ -57 MPa and the shear vanishes.
    std::vector<Vec3> pre(s.fault.size());
    for (std::size_t i = 0; i < s.fault.size(); ++i) {
      pre[i][0] = p.horizontal_stress(s.y[i]) / s.unit;
    }
    driver.set_prestress(std::move(pre));
  }

  const ContactState state = driver.solve_step();
  Slipped out;
  out.iterations = state.iterations;
  out.converged = state.converged;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    out.slip.push_back(std::abs(state.jump[i][1]) * s.length);
    out.normal.push_back(state.traction[i][0] * s.unit);
    out.shear.push_back(state.traction[i][1] * s.unit);
    out.peak = std::max(out.peak, out.slip.back());
  }
  return out;
}

}  // namespace

// THE OFFSET IS WHAT LOADS THE FAULT, so the pressure field must be displaced:
// [-b, a] on the left and [-a, b] on the right. With no throw the two sides
// would face reservoir everywhere, there would be no shear, and nothing to slip.
MIMETIKA_TEST(the_offset_reservoir_is_built_correctly) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  CHECK(!s.depleted.empty());
  CHECK(s.depleted.size() < static_cast<std::size_t>(s.mesh.topology().count(2)));

  double left_lo = 1e300, left_hi = -1e300, right_lo = 1e300, right_hi = -1e300;
  for (const Index e : s.depleted) {
    const exokal::Point x = exokal::centroid(s.mesh, 2, e);
    if (x[0] < 0.0) {
      left_lo = std::min(left_lo, x[1]);
      left_hi = std::max(left_hi, x[1]);
    } else {
      right_lo = std::min(right_lo, x[1]);
      right_hi = std::max(right_hi, x[1]);
    }
  }
  std::printf("  left  y in [%+.1f, %+.1f] m   right y in [%+.1f, %+.1f] m\n",
              left_lo * s.length, left_hi * s.length, right_lo * s.length,
              right_hi * s.length);
  CHECK(left_lo < right_lo);  // the two are mirror images in y
  CHECK(left_hi < right_hi);

  // one fault facet per row of cells -- derived, not hardcoded
  std::vector<double> rows;
  for (Index e = 0; e < s.mesh.topology().count(2); ++e) {
    rows.push_back(std::round(exokal::centroid(s.mesh, 2, e)[1] * 1e9));
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  std::printf("  %zu fault facets, %zu cell rows, %lld cells\n", s.fault.size(), rows.size(),
              (long long)s.mesh.topology().count(2));
  CHECK(s.fault.size() == rows.size());
}

// THE LOCKED FAULT CARRIES THE DRIVING STRESS of eq. (18): the plain continuous
// medium under the depletion load, with no contact anywhere. This is Fig. 6's
// left panel, which the paper compares graphically; the same comparison is made
// here as an assertion.
//
// THE FOUR SINGULARITIES ARE SKIPPED, and that is the figure's own convention
// rather than a convenience: eq. (18) is logarithmically singular at y = +-a and
// y = +-b, so the analytic markers in Fig. 6 deliberately avoid them and no
// cell-centred value could follow it there.
//
// THE TRACTION IS READ ON THE FAULT FACETS, not from the cells beside them. In
// Hellinger-Reissner the facet traction moments are primary unknowns, so this is
// the value on the plane itself; a cell-centred sigma_xy is sampled half a cell
// away, where the stress has already decayed, and no amount of refinement along
// the plane fixes an error across it.
MIMETIKA_TEST(the_locked_fault_carries_the_analytic_coulomb_stress) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const std::vector<double> got = pre_slip_coulomb_stress(s, p);

  std::printf("      y [m]    Sigma_C [MPa]   analytic [MPa]\n");
  int compared = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < s.y.size(); ++i) {
    const double y = s.y[i];
    // clear of the four singularities and inside the near field
    const bool usable = std::abs(y) < 400.0 &&
                        std::abs(std::abs(y) - p.fault_a) > 25.0 &&
                        std::abs(std::abs(y) - p.fault_b) > 25.0 && std::abs(y) > 10.0;
    if (!usable) continue;
    const double want = p.analytic_coulomb_stress(y);
    ++compared;
    worst = std::max(worst, std::abs(got[i] - want) / std::abs(p.slip_stress_scale()));
    if (compared % 4 == 1) {
      std::printf("   %9.1f   %+12.4f   %+12.4f\n", y, got[i] / 1e6, want / 1e6);
    }
  }
  std::printf("  %d points compared, worst deviation %.1f%% of C\n", compared, 100 * worst);
  CHECK(compared >= 8);
  CHECK(worst < 0.05);  // 5% of C, away from the singular edges
}

// -- the slipping fault ----------------------------------------------------------

// A FRICTIONLESS FAULT CARRIES NO SHEAR, which is the whole of the constitutive
// statement and the first thing to check. It is not approximately zero: the law
// projects the shear to zero at every enforcement point, so what remains is the
// residual of the fixed point and nothing else.
MIMETIKA_TEST(the_frictionless_fault_carries_no_shear_traction) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
  double shear = 0.0, normal = 0.0;
  for (std::size_t i = 0; i < r.shear.size(); ++i) {
    shear = std::max(shear, std::abs(r.shear[i]));
    normal = std::max(normal, std::abs(r.normal[i]));
  }
  std::printf("  %d iterations, converged=%d   max |shear| %.3e Pa   max |normal| %.3e Pa\n",
              r.iterations, static_cast<int>(r.converged), shear, normal);
  CHECK(r.converged);
  CHECK(shear <= 1e-8 * std::max(normal, 1.0));
}

// AND THE SLIP IS THE TENT OF EQ. (20). The peak is the number the paper
// reports; the profile is compared over the support of the analytic solution.
MIMETIKA_TEST(the_peak_slip_is_close_to_the_analytic_value) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
  CHECK(r.converged);
  std::printf("  peak slip %.5f m   analytic %.5f m   (%+.2f%%)\n", r.peak, p.peak_slip(),
              100.0 * (r.peak / p.peak_slip() - 1.0));

  std::printf("      y [m]      slip [m]    analytic [m]\n");
  for (std::size_t i = 0; i < s.y.size(); ++i) {
    if (std::abs(s.y[i]) <= 1.2 * p.fault_b) {
      std::printf("   %9.1f   %11.5f   %11.5f\n", s.y[i], r.slip[i],
                  std::abs(p.analytic_slip(s.y[i])));
    }
  }
  CHECK(close(r.peak, p.peak_slip(), 0.15));
}

// THE SLIP IS LOCALISED AT THE RESERVOIR: the analytic tent is supported on
// |y| <= b, and while the numerical solution carries a far-field tail the
// reservoir must dominate it by a wide margin.
MIMETIKA_TEST(the_slip_is_localised_at_the_reservoir) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
  double near = 0.0, far = 0.0;
  for (std::size_t i = 0; i < s.y.size(); ++i) {
    double& into = std::abs(s.y[i]) <= p.fault_b ? near : far;
    into = std::max(into, r.slip[i]);
  }
  std::printf("  peak inside |y| <= b %.5f m   outside %.5f m\n", near, far);
  CHECK(near > 0.0 && far < 0.25 * near);
}

// -- what the prestress is for ---------------------------------------------------

// WITHOUT THE PRESTRESS THE UNILATERAL LAW OPENS THE FAULT. The incremental
// normal traction is TENSILE -- depletion unloads the horizontal stress -- so a
// law shown only the increment clips it to zero and reports an open fault. Shown
// the total, the same law finds it shut under 57 MPa of in-situ compression.
//
// This is the failure the benchmark is able to detect, and running both is the
// only way it is detected: the slip profiles differ, but not so obviously that
// an eye would catch it.
MIMETIKA_TEST(without_the_prestress_the_unilateral_law_opens_the_fault) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped with = simulate(s, p, SignoriniCoulomb(0.0), true);
  const Slipped without = simulate(s, p, SignoriniCoulomb(0.0), false);
  CHECK(with.converged && without.converged);

  double kept = -1e300, clipped = -1e300;
  for (const double t : with.normal) kept = std::max(kept, t);
  for (const double t : without.normal) clipped = std::max(clipped, t);
  std::printf("  max incremental t_n:  with prestress %+.3e Pa   without %+.3e Pa\n", kept,
              clipped);
  CHECK(kept > 1e6);        // the tensile increment is KEPT: the fault is shut
  CHECK(clipped < 1e-6);    // and clipped without it: the fault is read as open
}

// AND THE TWO LAWS AGREE ONCE BOTH SEE THE TOTAL TRACTION. FrictionlessBilateral
// holds the fault shut by construction; SignoriniCoulomb(0) decides that it is
// shut. Since it really is, the two must give the same answer -- and to
// round-off, not approximately, because the unilateral branch is simply never
// taken.
MIMETIKA_TEST(the_two_laws_agree_once_both_see_the_total_traction) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped unilateral = simulate(s, p, SignoriniCoulomb(0.0), true);
  const Slipped bonded = simulate(s, p, FrictionlessBilateral(), true);
  CHECK(unilateral.converged && bonded.converged);
  double worst = 0.0;
  for (std::size_t i = 0; i < unilateral.slip.size(); ++i) {
    worst = std::max(worst, std::abs(unilateral.slip[i] - bonded.slip[i]));
  }
  std::printf("  worst |slip difference| %.3e m on a %.4f m peak\n", worst, unilateral.peak);
  CHECK(worst < 1e-12);
}

// -- the domain truncation -------------------------------------------------------

// W = 4500 m IS TOO NARROW, AS THE PAPER SAYS. Section 4.1 reports the reference
// code deviating from the semi-analytical solution on Table 2's box, cured by
// widening to 18 km. Pinning both ends here records WHY the domain is 18 km and
// keeps it from being changed back by accident.
MIMETIKA_TEST(the_narrow_domain_reproduces_the_published_discrepancy) {
  const Parameters narrow_p;  // Table 2's 4500 m box
  const Parameters wide_p = wide();
  const Slipped narrow = simulate(build(narrow_p, 12.5), narrow_p, SignoriniCoulomb(0.0));
  const Slipped broad = simulate(build(wide_p, 12.5), wide_p, SignoriniCoulomb(0.0));
  const double target = narrow_p.peak_slip();  // C/A (a-b): domain independent
  std::printf("  W = 4500 m  peak %.5f m (%+.2f%%)   W = 18000 m  peak %.5f m (%+.2f%%)\n",
              narrow.peak, 100.0 * (narrow.peak / target - 1.0), broad.peak,
              100.0 * (broad.peak / target - 1.0));
  CHECK(narrow.converged && broad.converged);
  CHECK(narrow.peak / target - 1.0 < -0.02);            // short by more than 2%
  CHECK(std::abs(broad.peak / target - 1.0) < 0.03);    // and on the analytic value
}


// THE SOLVE CONVERGES, AND IN A HANDFUL OF ITERATIONS -- Newton, not Picard.
// The count is the diagnostic: an affine law makes the condensed residual linear,
// so Newton is exact in one step and anything beyond a few iterations would mean
// the tangent is not the derivative.
MIMETIKA_TEST(the_solve_converges) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
  std::printf("  converged=%d in %d iterations\n", static_cast<int>(r.converged), r.iterations);
  CHECK(r.converged);
  CHECK(r.iterations <= 10);
}

// THE SLIP IS SYMMETRIC ABOUT THE RESERVOIR, as the analytic profile is.
//
// Compared against the PEAK rather than against the local value: the offset
// reservoir is symmetric only under the point reflection (x, y) -> (-x, -y), and
// the boundary conditions -- free top, roller base -- are not y-symmetric at
// all, so the far field carries a small absolute asymmetry that is large in
// relative terms precisely where the slip is near zero. Checked tightly on
// |y| <= b, the support of the analytic tent, and loosely outside it.
MIMETIKA_TEST(the_slip_is_symmetric_about_the_reservoir) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
  CHECK(r.converged);

  const auto worst_pair = [&](double limit) {
    double worst = 0.0;
    int pairs = 0;
    for (std::size_t i = 0; i < s.y.size(); ++i) {
      if (std::abs(s.y[i]) > limit || s.y[i] <= 0.0) continue;
      for (std::size_t j = 0; j < s.y.size(); ++j) {
        if (std::abs(s.y[j] + s.y[i]) < 1e-6) {
          worst = std::max(worst, std::abs(r.slip[i] - r.slip[j]));
          ++pairs;
        }
      }
    }
    return std::pair<double, int>{worst, pairs};
  };

  const auto tent = worst_pair(p.fault_b);
  const auto outer = worst_pair(3.0 * p.fault_b);
  std::printf("  |y| <= b: %d pairs, worst %.3e (%.2f%% of peak)   |y| <= 3b: %d pairs, %.3e\n",
              tent.second, tent.first, 100 * tent.first / r.peak, outer.second, outer.first);
  CHECK(tent.second >= 3);
  CHECK(tent.first < 5e-3 * r.peak);
  CHECK(outer.first < 2e-2 * r.peak);
}

// REFINEMENT CONVERGES -- but not necessarily ONTO the analytic peak, and the
// distinction is the point of this test.
//
// On the wide domain the peak overshoots by around 1% and SETTLES there. That
// residual is the fault compliance's own discretization error, not a failure to
// converge, so demanding the peak error shrink would be demanding the wrong
// thing. What must hold is that the whole PROFILE converges and the peak stays
// bounded.
MIMETIKA_TEST(refinement_converges_and_stays_close_to_the_analytic_peak) {
  const Parameters p = wide();
  std::vector<double> profile, peak;
  for (const double spacing : {25.0, 12.5}) {
    const Setup s = build(p, spacing);
    const Slipped r = simulate(s, p, SignoriniCoulomb(0.0));
    CHECK(r.converged);
    double sum = 0.0;
    for (std::size_t i = 0; i < s.y.size(); ++i) {
      const double e = r.slip[i] - std::abs(p.analytic_slip(s.y[i]));
      sum += e * e;
    }
    profile.push_back(std::sqrt(sum / static_cast<double>(s.y.size())));
    peak.push_back(std::abs(r.peak / p.peak_slip() - 1.0));
    std::printf("  spacing %5.2f m   profile RMS %.5e m   peak error %+.2f%%\n", spacing,
                profile.back(), 100 * peak.back());
  }
  CHECK(profile[1] < profile[0]);                       // the profile converges
  CHECK(std::max(peak[0], peak[1]) < 0.03);             // and the peak stays put
}

// THE DEFAULT LAW IS THE UNILATERAL ONE. Benchmark 1 runs SignoriniCoulomb at
// zero friction -- the physical model -- rather than the bonded law that happens
// to give the same answer here. A benchmark exists to test laws, so the one that
// represents the situation is the one that should be run, and the fact that it
// KEEPS the tensile increment is the evidence it decided the fault was shut
// rather than being told so.
MIMETIKA_TEST(the_default_law_is_the_unilateral_one) {
  const Parameters p = wide();
  const Setup s = build(p, 25.0);
  const SignoriniCoulomb law(0.0);
  CHECK(law.friction() == 0.0);
  CHECK(!law.has_linear_compliance());  // so it genuinely iterates
  const Slipped r = simulate(s, p, law);
  CHECK(r.converged);
  double kept = -1e300;
  for (const double t : r.normal) kept = std::max(kept, t);
  CHECK(kept > 1e6);  // tension kept: the unilateral branch was NOT taken
}

MIMETIKA_TEST_MAIN()
