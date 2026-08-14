#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <cstdio>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/benchmarks/novikov_2024.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/solver/petsc.hpp"

// BENCHMARK 0 of Novikov et al. (2024): the in-situ state and the depletion
// response, with NO FAULT. The case that has to be right before contact can be
// asked about anything.
//
// Two independent claims, checked separately and against different things:
//
//   THE IN-SITU PROFILES the paper prints follow from the Table 2 parameters.
//   That is a statement about the SETUP and needs no solver, so it is checked
//   first; a benchmark whose initial state is wrong will disagree with the
//   reference for reasons that have nothing to do with the discretization.
//
//   THE DEPLETION RESPONSE is reproduced by the mixed solver. A uniformly
//   depleted, laterally confined domain compacts uniaxially, and the closed
//   forms
//
//       eps_yy = alpha Dp / Kv ,  D sigma'_xx = nu/(1-nu) alpha Dp ,  sigma_yy = 0
//
//   are EXACT for this discretization -- so they are checked to round-off, not
//   to a few digits. Lateral confinement is imposed with ROLLERS: prescribed
//   normal displacement and free slip, which in Hellinger-Reissner means pinning
//   the SHEAR traction dofs, the displacement side being natural there.
//
// THE MECHANICS ALONE IS SOLVED, which is how the benchmark is posed: the
// pressure is data, not an unknown, and it enters as the load alpha T^T p. That
// term carries the same coefficient and the same operator as the coupled
// solver's Biot block, so the two agree exactly wherever both are posed.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::benchmarks::linear_fit;
using mimetika::benchmarks::Parameters;
using mimetika::mesh::Family;
using Realization = exokal::hodge::StressOperators::Realization;

namespace {

constexpr double kDip = 70.0;

bool close(double got, double want, double rel) {
  return std::abs(got - want) <= rel * std::abs(want);
}

std::vector<double> samples(double lo, double hi, int n) {
  std::vector<double> out;
  for (int i = 0; i < n; ++i) out.push_back(lo + (hi - lo) * i / (n - 1));
  return out;
}

// -- the depletion response ----------------------------------------------------

struct Response {
  double vertical_strain{0.0};
  double compaction{0.0};
  double sigma_xx_total{0.0};
  double sigma_xx_effective{0.0};
  double sigma_yy_total{0.0};
};

// UNIFORM DEPLETION OF A CONFINED BLOCK. The response is a STRAIN, so the domain
// size enters only through Dh = h eps_yy, applied afterwards -- the mesh is the
// unit square. Rollers all round but the top, which is a free surface.
//
// `clamp_sides` is the premise guard: with the sides fully clamped instead of
// free to slide the problem is no longer uniaxial and the closed form fails,
// which is what makes the rollers load bearing rather than decorative.
//
// STRESS IS MEASURED IN UNITS OF THE SHEAR MODULUS, and that is not a
// convenience. The mixed system is the saddle point [M, -D^T; D, 0] with
// M ~ h^d/G and D ~ h^{d-1}; stated in pascals with G = 6.5 GPa the two blocks
// sit 10^10 apart at h = 1/6 and the gap widens as h^{-1}, so the direct
// factorization breaks down -- at n = 6 in pascals, at n = 24 in megapascals.
// Dividing every stress-dimensioned quantity by G makes M ~ h^d against
// D ~ h^{d-1}, a ratio of h, and the same solve is exact to eleven digits at
// n = 96. Strain is dimensionless and comes back unchanged; the stresses are
// multiplied by G on the way out, so the caller sees pascals throughout.
Response depletion_response(const Parameters& p, int n, Realization how,
                            bool clamp_sides = false) {
  const int dim = 2;
  const double unit = p.shear_modulus;
  Parameters s = p;  // the same problem, stated in units of G
  s.shear_modulus = p.shear_modulus / unit;
  s.depletion = p.depletion / unit;

  const exokal::Mesh mesh = mimetika::mesh::box({n, n, 1}, dim, Family::cartesian,
                                                {1.0, 1.0, 1.0});
  const graphos::Complex& c = mesh.topology();

  std::vector<Index> top, sides, cells;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    (std::abs(exokal::centroid(mesh, dim - 1, f)[1] - 1.0) < 1e-12 ? top : sides).push_back(f);
  }
  for (Index e = 0; e < c.count(dim); ++e) cells.push_back(e);

  CauchyElasticityModel model(
      mesh, dim, ElasticMaterial{s.shear_modulus, s.lame()}, how);
  model.mechanics().emplace<mimetika::TractionBC>(top, std::array<double, 9>{});
  if (clamp_sides) {
    model.prescribe_displacement(sides, {0.0, 0.0, 0.0});
  } else {
    model.mechanics().emplace<mimetika::FreeSlipBC>(sides);
  }
  model.pressurize(cells, s.depletion, s.biot, s.volumetric_compliance(dim));
  model.build();

  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto report = petsc.solve(model.system(), model.rhs(), x);
  if (!report.converged) throw std::runtime_error("benchmark 0: " + report.reason);
  model.accept(std::move(x));

  Response out;
  double sxx = 0.0, syy = 0.0;
  for (Index e = 0; e < c.count(dim); ++e) {
    const std::array<double, 9> t = model.cell_stress(e);
    sxx += t[0];
    syy = std::max(syy, std::abs(t[4]));
  }
  out.sigma_xx_total = unit * sxx / static_cast<double>(c.count(dim));
  out.sigma_xx_effective = out.sigma_xx_total + p.biot * p.depletion;
  out.sigma_yy_total = unit * syy;

  // THE VERTICAL STRAIN FROM THE DISPLACEMENT FIELD, by fitting u_y against y
  // over the cell centroids. Under uniaxial strain u_y is exactly linear, so the
  // fit is a reading of the gradient and not a smoothing of it.
  std::vector<double> y, uy;
  for (Index e = 0; e < c.count(dim); ++e) {
    y.push_back(exokal::centroid(mesh, dim, e)[1]);
    uy.push_back(model.displacement(e, 1));
  }
  out.vertical_strain = linear_fit(y, uy)[1];
  out.compaction = p.reservoir_height() * out.vertical_strain;
  return out;
}

// -- Fig. 4: the finite reservoir ----------------------------------------------

// DEPLETE A RESERVOIR OF FINITE THICKNESS inside the full domain: different from
// the uniform case above, and harder. That one depletes the WHOLE domain, which
// is why it reproduces the uniaxial closed form to round-off -- there is nothing
// for the rock to arch over. Fig. 4 needs a 225 m reservoir inside a 4500 m
// domain, so the stress steps sharply at the reservoir top and bottom and the
// uniaxial formulae survive only in the interior.
struct Field {
  exokal::Mesh mesh;
  std::vector<std::array<double, 9>> stress;  // the INCREMENT, in Pa
  std::vector<double> pressure;
  std::vector<std::array<double, 3>> centroid;
};

Field finite_reservoir(const Parameters& p, int nx, int ny,
                       Realization how = Realization::derham_afw) {
  const int dim = 2;
  const double half = 0.5 * p.reservoir_height();
  const double spacing = p.height / ny;
  // THE DEPLETION IS ASSIGNED PER CELL, so a reservoir boundary that bisects a
  // cell shifts the step by half a cell. With ny = 180 the boundary at 112.5 m
  // lands on a cell CENTRE: that cell is half inside, the centroid test excludes
  // it, and the figure comes out plausible and wrong by 16 MPa over one cell. It
  // is refused rather than silently approximated.
  if (std::abs(half / spacing - std::round(half / spacing)) > 1e-9) {
    throw std::invalid_argument("finite_reservoir: the reservoir boundary falls inside a cell");
  }

  // AND LENGTH IN UNITS OF THE DOMAIN HEIGHT, for the same reason stress is in
  // units of G: on a 4500 m domain the divergence block carries the cell size
  // and the compliance block its square, so stating the geometry in metres puts
  // three more orders between them on top of the material's. Nondimensionalizing
  // both leaves every block O(1) and the answer -- a stress ratio -- untouched.
  const double unit = p.shear_modulus, L = p.height;
  Parameters s = p;
  s.shear_modulus = p.shear_modulus / unit;
  s.depletion = p.depletion / unit;

  Field out{mimetika::mesh::box({nx, ny, 1}, dim, Family::cartesian,
                                {p.width / L, p.height / L, 1.0},
                                {-p.width / (2 * L), -p.height / (2 * L), 0.0}),
            {}, {}, {}};
  const graphos::Complex& c = out.mesh.topology();

  std::vector<Index> top, rest, depleted;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    (std::abs(exokal::centroid(out.mesh, dim - 1, f)[1] - 0.5) < 1e-9 ? top : rest).push_back(f);
  }
  for (Index e = 0; e < c.count(dim); ++e) {
    const exokal::Point x = exokal::centroid(out.mesh, dim, e);
    out.centroid.push_back({x[0] * L, x[1] * L, x[2] * L});  // reported in metres
    const bool inside = std::abs(x[1] * L) < half;
    out.pressure.push_back(inside ? p.depletion : 0.0);
    if (inside) depleted.push_back(e);
  }

  CauchyElasticityModel model(out.mesh, dim, ElasticMaterial{s.shear_modulus, s.lame()}, how);
  model.mechanics().emplace<mimetika::TractionBC>(top, std::array<double, 9>{});
  model.mechanics().emplace<mimetika::FreeSlipBC>(rest);
  model.pressurize(depleted, s.depletion, s.biot, s.volumetric_compliance(dim));
  model.build();

  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto report = petsc.solve(model.system(), model.rhs(), x);
  if (!report.converged) throw std::runtime_error("finite_reservoir: " + report.reason);
  model.accept(std::move(x));

  for (Index e = 0; e < c.count(dim); ++e) {
    std::array<double, 9> t = model.cell_stress(e);
    for (double& v : t) v *= unit;
    out.stress.push_back(t);
  }
  return out;
}

struct Profile {
  std::vector<double> y, normal, shear;
};

// SAMPLED ALONG THE LINE THE FAULT WOULD OCCUPY -- at `dip` to the horizontal
// through the reservoir centre -- and resolved onto that plane. "Combined" means
// the in-situ state PLUS the depletion increment, which is what the figure plots.
Profile combined_stress_profile(const Parameters& p, int nx, int ny, double dip,
                                double extent = 250.0) {
  const Field f = finite_reservoir(p, nx, ny);
  const auto n = Parameters::fault_normal(dip), t = Parameters::fault_tangent(dip);
  const double theta = dip * M_PI / 180.0;

  Profile out;
  const int steps = 2 * static_cast<int>(extent) + 1;
  for (int i = 0; i < steps; ++i) {
    const double yi = -extent + 2.0 * extent * i / (steps - 1);
    const double xi = yi * std::cos(theta) / std::sin(theta);
    std::size_t best = 0;
    double d2 = std::numeric_limits<double>::infinity();
    for (std::size_t e = 0; e < f.centroid.size(); ++e) {
      const double dx = f.centroid[e][0] - xi, dy = f.centroid[e][1] - yi;
      if (dx * dx + dy * dy < d2) {
        d2 = dx * dx + dy * dy;
        best = e;
      }
    }
    // the increment plus the in-situ state, as a row-major 2x2
    const std::array<double, 4> insitu = p.stress_tensor(yi);
    const std::array<double, 4> total = {f.stress[best][0] + insitu[0],
                                         f.stress[best][1] + insitu[1],
                                         f.stress[best][3] + insitu[2],
                                         f.stress[best][4] + insitu[3]};
    out.y.push_back(yi);
    out.normal.push_back(Parameters::contract(n, total, n));
    out.shear.push_back(Parameters::contract(t, total, n));
  }
  return out;
}

}  // namespace

// -- the in-situ state derives from Table 2 ------------------------------------

// DERIVED FROM (rho_s, rho_fl, phi, g, D0, K0, alpha, p0) -- not tabulated. The
// tolerance is set by the paper's own precision: it prints three significant
// figures, so 2.35 against a computed 2.340 is agreement, not disagreement.
//
// The shear is compared in MAGNITUDE: its sign depends on which way the fault
// tangent is taken, and the paper's choice is opposite to this one.
MIMETIKA_TEST(the_in_situ_profiles_match_the_paper) {
  const Parameters p;
  const std::vector<double> y = samples(-p.height / 2, p.height / 2, 11);

  struct Row {
    const char* name;
    double intercept, gradient;
    bool magnitude;
  };
  // eqs. (17)-(19): (intercept [Pa], gradient [Pa/m]), y measured upwards
  const Row published[5] = {
      {"sigma_yy", -82.60e6, 23.60e3, false},  {"sigma_xx", -57.05e6, 16.30e3, false},
      {"pressure", 35.00e6, -10.06e3, false},  {"sigma_normal_70", -60.04e6, 17.15e3, false},
      {"sigma_shear_70", 8.21e6, -2.35e3, true}};

  std::vector<std::vector<double>> fields(5);
  for (const double yi : y) {
    fields[0].push_back(p.vertical_stress(yi));
    fields[1].push_back(p.horizontal_stress(yi));
    fields[2].push_back(p.pressure(yi));
    fields[3].push_back(p.resolved(yi, kDip)[0]);
    fields[4].push_back(p.resolved(yi, kDip)[1]);
  }

  std::printf("  in-situ profile      intercept [MPa]        gradient [kPa/m]\n");
  for (std::size_t k = 0; k < 5; ++k) {
    const std::array<double, 2> fit = linear_fit(y, fields[k]);
    double intercept = fit[0], gradient = fit[1];
    double want_i = published[k].intercept, want_g = published[k].gradient;
    if (published[k].magnitude) {
      intercept = std::abs(intercept);
      gradient = std::abs(gradient);
      want_i = std::abs(want_i);
      want_g = std::abs(want_g);
    }
    std::printf("    %-16s %+9.2f (%+7.2f)   %+8.2f (%+7.2f)\n", published[k].name,
                intercept / 1e6, want_i / 1e6, gradient / 1e3, want_g / 1e3);
    CHECK(close(intercept, want_i, 5e-3));
    CHECK(close(gradient, want_g, 5e-3));
  }
}

MIMETIKA_TEST(the_bulk_density_and_uniaxial_modulus_are_the_published_ones) {
  const Parameters p;
  std::printf("  rho_b %.1f kg/m^3 (2406.2)   Kv %.4e Pa (15.79e9)\n", p.bulk_density(),
              p.uniaxial_modulus());
  CHECK(close(p.bulk_density(), 2406.2, 1e-4));
  CHECK(close(p.uniaxial_modulus(), 15.79e9, 1e-3));
}

// THE PAPER STATES THESE AS LINEAR PROFILES, so a fit must be exact rather than
// close -- if any of them curved, the coefficients it prints would not describe it.
MIMETIKA_TEST(the_in_situ_state_is_genuinely_linear_in_depth) {
  const Parameters p;
  const std::vector<double> y = samples(-2000.0, 2000.0, 17);
  for (int which = 0; which < 3; ++which) {
    std::vector<double> v;
    for (const double yi : y) {
      v.push_back(which == 0   ? p.vertical_stress(yi)
                  : which == 1 ? p.horizontal_stress(yi)
                               : p.pressure(yi));
    }
    const std::array<double, 2> fit = linear_fit(y, v);
    for (std::size_t i = 0; i < y.size(); ++i) {
      CHECK(std::abs(v[i] - (fit[0] + fit[1] * y[i])) < 1e-12 * std::abs(fit[0]));
    }
  }
}

// K0 ACTS ON EFFECTIVE STRESS. Using the total stress instead is the classic
// slip in this setup, and it is invisible in the profile shape -- both are
// linear in y -- so it is worth an assertion of its own.
MIMETIKA_TEST(the_earth_pressure_coefficient_relates_the_effective_stresses) {
  const Parameters p;
  for (const double y : samples(-1000.0, 1000.0, 9)) {
    const double ev = p.vertical_stress(y) + p.biot * p.pressure(y);
    const double eh = p.horizontal_stress(y) + p.biot * p.pressure(y);
    CHECK(std::abs(eh - p.earth_pressure * ev) < 1e-9 * std::abs(ev));
  }
}

MIMETIKA_TEST(the_state_is_compressive_everywhere) {
  const Parameters p;
  for (const double y : samples(-p.height / 2, p.height / 2, 21)) {
    CHECK(p.vertical_stress(y) < 0.0);
    CHECK(p.horizontal_stress(y) < 0.0);
    CHECK(p.pressure(y) > 0.0);
  }
}

// THE RESOLUTION ONTO THE FAULT IS A GENUINE ROTATION: normal and shear must be
// the tensor contractions, and the trace must be invariant. An error here would
// move every reported fault stress by a fixed factor and look like a material
// discrepancy.
MIMETIKA_TEST(the_resolved_traction_is_a_genuine_rotation) {
  const Parameters p;
  for (const double dip : {90.0, 70.0, 45.0}) {
    const auto n = Parameters::fault_normal(dip), t = Parameters::fault_tangent(dip);
    CHECK(std::abs(n[0] * t[0] + n[1] * t[1]) < 1e-14);
    CHECK(std::abs(std::sqrt(n[0] * n[0] + n[1] * n[1]) - 1.0) < 1e-14);
    for (const double y : samples(-500.0, 500.0, 5)) {
      const auto s = p.stress_tensor(y);
      const auto r = p.resolved(y, dip);
      const double other = Parameters::contract(t, s, t);
      CHECK(std::abs(r[0] + other - (s[0] + s[3])) < 1e-6);
      CHECK(std::abs(r[1] - Parameters::contract(n, s, t)) < 1e-6);
    }
  }
}

// AT DIP 90 THE FAULT NORMAL IS -e_x: a sanity anchor for the rotation, where
// the answer is the horizontal stress itself and no shear at all.
MIMETIKA_TEST(a_vertical_fault_sees_the_horizontal_stress_and_no_shear) {
  const Parameters p;
  for (const double y : samples(-500.0, 500.0, 5)) {
    const auto r = p.resolved(y, 90.0);
    CHECK(std::abs(r[0] - p.horizontal_stress(y)) < 1e-6);
    CHECK(std::abs(r[1]) < 1e-6);
  }
}

// -- the depletion response is reproduced by the solver -------------------------

// THE UNIAXIAL CLOSED FORMS, TO ROUND-OFF, on both stress products. The state is
// uniform, so this is not a convergence statement: either the discretization
// reproduces it exactly or it does not reproduce it at all.
MIMETIKA_TEST(the_depletion_response_is_the_uniaxial_closed_form) {
  const Parameters p;
  for (const Realization how : {Realization::derham_afw, Realization::stabilized_afw}) {
    const Response r = depletion_response(p, 6, how);
    std::printf("  %-16s eps_yy %+.8e (%+.8e)   Dh %+.4f m (%.2f)\n",
                exokal::hodge::StressOperators::name(how), r.vertical_strain,
                p.vertical_strain(), r.compaction, p.compaction());
    std::printf("  %-16s Dsigma'_xx %+.6e (%+.6e)   Dsigma_xx %+.6e (%+.6e)\n", "",
                r.sigma_xx_effective, p.horizontal_effective_increment(), r.sigma_xx_total,
                p.horizontal_total_increment());

    CHECK(close(r.vertical_strain, p.vertical_strain(), 1e-9));
    CHECK(std::abs(r.compaction - (-0.32)) < 5e-3);  // the paper's reported value
    CHECK(close(r.sigma_xx_effective, p.horizontal_effective_increment(), 1e-9));
    CHECK(close(r.sigma_xx_total, p.horizontal_total_increment(), 1e-9));
    CHECK(close(r.sigma_xx_effective, -3.97e6, 1e-3));  // the paper's printed values
    CHECK(close(r.sigma_xx_total, 18.53e6, 1e-3));
    // the top is traction free, so sigma_yy must VANISH rather than be small
    CHECK(r.sigma_yy_total < 1e-6 * std::abs(p.depletion));
  }
}

// A UNIFORM STATE: refinement must change nothing at all. Any drift here would
// be the discretization failing to represent a constant, which no amount of
// resolution repairs.
MIMETIKA_TEST(the_response_is_mesh_independent) {
  const Parameters p;
  for (const int n : {3, 6, 12}) {
    const Response r = depletion_response(p, n, Realization::derham_afw);
    std::printf("  %2d x %-2d cells   eps_yy %+.10e\n", n, n, r.vertical_strain);
    CHECK(close(r.vertical_strain, p.vertical_strain(), 1e-9));
  }
}

// THE ROLLERS ARE WHAT MAKE IT UNIAXIAL, and that premise is worth guarding.
// Compared against the same problem with the sides FULLY CLAMPED rather than
// free to slide: a genuinely unconfined block would be the more obvious
// contrast, but prescribing traction all round leaves the rigid-body modes
// undetermined and the system singular, so clamping is the well-posed
// alternative -- and it gives a visibly different answer.
MIMETIKA_TEST(rollers_are_what_make_the_response_uniaxial) {
  const Parameters p;
  const Response rolling = depletion_response(p, 4, Realization::derham_afw, false);
  const Response clamped = depletion_response(p, 4, Realization::derham_afw, true);
  std::printf("  rolling sigma_xx %+.6e   clamped %+.6e\n", rolling.sigma_xx_total,
              clamped.sigma_xx_total);
  CHECK(std::abs(rolling.sigma_xx_total - clamped.sigma_xx_total) >
        1e-2 * std::abs(rolling.sigma_xx_total));
  // and only the rolling one is the closed form
  CHECK(close(rolling.sigma_xx_total, p.horizontal_total_increment(), 1e-9));
}

// -- Fig. 4: combined stresses across the depleted reservoir ---------------------
//
// A different computation from the uniform-depletion check above: the reservoir
// is 225 m thick inside a 4500 m domain. Because it spans the FULL WIDTH and the
// sides are rollers, the problem is one-dimensional -- so the increment is
// exactly uniaxial inside the reservoir and exactly zero outside it, which is
// the two-plateau step the figure shows. Both plateaus are therefore checked
// exactly rather than approximately.

// THE PUBLISHED PLATEAU VALUES: Sigma_perp and Sigma_par inside and outside,
// which is what Fig. 4's two levels are. Analytic, so this is a check on the
// setup before the profile is computed at all.
MIMETIKA_TEST(the_published_plateau_values) {
  const Parameters p;
  const auto out = p.plateau_outside(kDip), in = p.plateau_inside(kDip);
  std::printf("  outside  sigma_n %+7.2f MPa (-60.04)   |tau| %5.2f MPa (8.21)\n", out[0] / 1e6,
              std::abs(out[1]) / 1e6);
  std::printf("  inside   sigma_n %+7.2f MPa (-43.70)   |tau| %5.2f MPa (14.20)\n", in[0] / 1e6,
              std::abs(in[1]) / 1e6);
  CHECK(close(out[0] / 1e6, -60.04, 5e-3));
  CHECK(close(std::abs(out[1]) / 1e6, 8.21, 5e-3));
  CHECK(close(in[0] / 1e6, -43.7, 1e-2));
  CHECK(close(std::abs(in[1]) / 1e6, 14.2, 1e-2));
}

namespace {
Profile fig4() { return combined_stress_profile(Parameters(), 8, 120, kDip); }
}  // namespace

// INSIDE, THE INCREMENT IS EXACTLY Delta sigma_xx with Delta sigma_yy = 0. The
// reservoir is full width and the sides are rollers, so there is nothing to arch
// over and the uniaxial form is not an approximation.
MIMETIKA_TEST(the_reservoir_interior_matches_the_uniaxial_closed_form) {
  const Parameters p;
  const Profile f = fig4();
  const auto in = p.plateau_inside(kDip);
  std::size_t centre = 0;
  for (std::size_t i = 0; i < f.y.size(); ++i) {
    if (std::abs(f.y[i]) < std::abs(f.y[centre])) centre = i;
  }
  std::printf("  at y = %+.0f m   sigma_n %+.6e (%+.6e)   tau %+.6e (%+.6e)\n", f.y[centre],
              f.normal[centre], in[0], f.shear[centre], in[1]);
  CHECK(close(f.normal[centre], in[0], 1e-6));
  CHECK(close(f.shear[centre], in[1], 1e-6));
}

// AND OUTSIDE IT VANISHES. A full-width reservoir with roller sides leaves the
// seal unstressed -- not approximately: with eps_xx = 0 and sigma_yy = 0 the
// problem is one-dimensional, so outside the depleted band the increment is
// identically zero and the combined stress is the in-situ state.
MIMETIKA_TEST(outside_the_reservoir_the_increment_vanishes) {
  const Parameters p;
  const Profile f = fig4();
  const double half = 0.5 * p.reservoir_height();
  std::size_t checked = 0;
  for (std::size_t i = 0; i < f.y.size(); ++i) {
    if (std::abs(f.y[i]) <= half + 20.0) continue;
    ++checked;
    const auto want = p.resolved(f.y[i], kDip);
    CHECK(close(f.normal[i], want[0], 1e-6));
    CHECK(close(f.shear[i], want[1], 1e-6));
  }
  CHECK(checked > 100);
}

// THE STEP IS THE SIGNATURE OF FIG. 4, and it sits at +-h/2 -- which is what the
// per-cell depletion and the aligned grid together guarantee.
MIMETIKA_TEST(the_profile_steps_at_the_reservoir_boundary) {
  const Parameters p;
  const Profile f = fig4();
  const double half = 0.5 * p.reservoir_height();
  std::vector<std::pair<double, double>> jumps;  // (size, |y|)
  for (std::size_t i = 1; i < f.y.size(); ++i) {
    jumps.emplace_back(std::abs(f.normal[i] - f.normal[i - 1]), std::abs(f.y[i]));
  }
  std::sort(jumps.begin(), jumps.end());
  const auto& a = jumps[jumps.size() - 1];
  const auto& b = jumps[jumps.size() - 2];
  std::printf("  the two largest steps: %.2f MPa at |y| = %.1f m, %.2f MPa at %.1f m (h/2 = %.1f)\n",
              a.first / 1e6, a.second, b.first / 1e6, b.second, half);
  CHECK(std::abs(a.second - half) < 1.5 * p.height / 120);
  CHECK(std::abs(b.second - half) < 1.5 * p.height / 120);
  CHECK(a.first > 10e6);  // a real step, over 10 MPa
}

// DEPLETION UNLOADS THE HORIZONTAL STRESS, so the reservoir interior moves
// toward zero relative to the seal that confines it.
MIMETIKA_TEST(the_reservoir_is_less_compressive_than_the_seal) {
  const Parameters p;
  const Profile f = fig4();
  const double half = 0.5 * p.reservoir_height();
  double in_sum = 0.0, out_sum = 0.0, in_max = -1e300, out_max = -1e300;
  std::size_t ni = 0, no = 0;
  for (std::size_t i = 0; i < f.y.size(); ++i) {
    if (std::abs(f.y[i]) < half - 20.0) {
      in_sum += f.normal[i];
      in_max = std::max(in_max, f.normal[i]);
      ++ni;
    } else if (std::abs(f.y[i]) > half + 20.0) {
      out_sum += f.normal[i];
      out_max = std::max(out_max, f.normal[i]);
      ++no;
    }
  }
  std::printf("  inside mean %+.2f MPa   outside mean %+.2f MPa\n", in_sum / ni / 1e6,
              out_sum / no / 1e6);
  CHECK(in_max < 0.0 && out_max < 0.0);              // both compressive
  CHECK(in_sum / ni > out_sum / no + 10e6);          // by more than 10 MPa
}

// |Sigma_par| RISES FROM ~8 TO ~14 MPa, which is what drives fault reactivation
// and the whole reason the later benchmarks have a fault at all.
MIMETIKA_TEST(the_shear_grows_inside_the_reservoir) {
  const Parameters p;
  const Profile f = fig4();
  const double half = 0.5 * p.reservoir_height();
  double in_min = 1e300, out_max = 0.0, in_sum = 0.0;
  std::size_t ni = 0;
  for (std::size_t i = 0; i < f.y.size(); ++i) {
    const double s = std::abs(f.shear[i]);
    if (std::abs(f.y[i]) < half - 20.0) {
      in_min = std::min(in_min, s);
      in_sum += s;
      ++ni;
    } else if (std::abs(f.y[i]) > half + 20.0) {
      out_max = std::max(out_max, s);
    }
  }
  std::printf("  |tau| inside mean %.2f MPa (14.2)   inside min %.2f > outside max %.2f\n",
              in_sum / ni / 1e6, in_min / 1e6, out_max / 1e6);
  CHECK(in_min > out_max);
  CHECK(close(in_sum / ni / 1e6, 14.2, 2e-2));
}

// AND THE WHOLE PROFILE IS THE ANALYTIC CURVE, TO ROUND-OFF. No arching for an
// infinitely wide reservoir, so this is exact rather than close: one pascal on a
// quantity of order 10^8.
MIMETIKA_TEST(the_profile_matches_the_analytic_curve_to_round_off) {
  const Parameters p;
  const Profile f = fig4();
  double worst_n = 0.0, worst_s = 0.0;
  for (std::size_t i = 0; i < f.y.size(); ++i) {
    const auto want = p.combined_analytic(f.y[i], kDip);
    worst_n = std::max(worst_n, std::abs(f.normal[i] - want[0]));
    worst_s = std::max(worst_s, std::abs(f.shear[i] - want[1]));
  }
  std::printf("  worst deviation from the analytic curve: %.3e Pa normal, %.3e Pa shear\n",
              worst_n, worst_s);
  CHECK(worst_n < 1.0);
  CHECK(worst_s < 1.0);
}

// A GRID THAT BISECTS THE RESERVOIR BOUNDARY IS REFUSED. The depletion is
// assigned per cell, so the boundary must land on a cell face; ny = 180 puts it
// on a cell CENTRE and the step lands half a cell away from where it belongs.
MIMETIKA_TEST(a_grid_that_bisects_the_reservoir_boundary_is_rejected) {
  const Parameters p;
  for (const int ny : {180, 100, 30}) {
    bool refused = false;
    try {
      finite_reservoir(p, 4, ny);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    CHECK(refused);
  }
}

MIMETIKA_TEST(an_aligned_grid_is_accepted_and_depletes_exactly_the_band) {
  const Parameters p;
  for (const int ny : {40, 120, 200}) {
    const Field f = finite_reservoir(p, 4, ny);
    std::size_t depleted = 0;
    for (const double v : f.pressure) depleted += (v != 0.0) ? 1 : 0;
    // exactly the cells inside the band, no half-counted row at either edge
    const auto want = static_cast<std::size_t>(
        4 * std::llround(p.reservoir_height() / (p.height / ny)));
    std::printf("  ny %3d   %zu depleted cells (%zu)\n", ny, depleted, want);
    CHECK(depleted == want);
  }
}

MIMETIKA_TEST_MAIN()
