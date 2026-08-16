#include <cmath>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/cauchy_mechanics.hpp"
#include "mimetika/algebraic_constraints/contact/driver.hpp"
#include "mimetika/mesh/structured.hpp"

// CONTACT OVER CAUCHY ELASTICITY, end to end: the adapter, and the driver on top
// of it.
//
// Everything below the adapter has been checked in isolation -- the laws
// against their conditions, the map against a closed-form stub, the trace
// against the bonded identity. What is checked here is the composition: that
// the three operations of ContactMechanics agree with each other, and that the
// Uzawa iteration built on them reproduces the conditions a contact law encodes
// on a real mesh.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::contact::CauchyContactMechanics;
using mimetika::contact::ContactDriver;
using mimetika::contact::ContactState;
using mimetika::contact::default_augmentation;
using mimetika::contact::DriverOptions;
using mimetika::contact::Fracture;
using mimetika::contact::FrictionlessBilateral;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::Status;
using mimetika::contact::Vec3;
using mimetika::mesh::Family;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;

bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol; }

std::vector<Index> facets_at(const exokal::Mesh& m, int dim, double z, double tol = 1e-9) {
  std::vector<Index> out;
  const graphos::Complex& c = m.topology();
  for (Index f = 0; f < c.count(dim - 1); ++f) {
    const auto x = exokal::centroid(m, dim - 1, f);
    if (std::abs(x[static_cast<std::size_t>(dim - 1)] - z) < tol) out.push_back(f);
  }
  return out;
}

// A COLUMN CUT AT MID-HEIGHT: rollers all round, a traction on top, and the
// fracture across the middle. Everything is uniaxial, so what the fracture does
// is the whole of the answer.
struct Problem {
  exokal::Mesh mesh;
  std::vector<Index> fault;
  std::unique_ptr<CauchyElasticityModel> model;
  double load{0.0};
};

// DISPLACEMENT ON THE WHOLE BOUNDARY, which is how the Python poses every
// driver test and is not a detail. Under pure traction data a fracture spanning
// the domain leaves each side with a rigid-body null mode, and equilibrium then
// FIXES the fault traction -- there is nothing for a law to decide, and the
// constrained operator is singular. With the displacement prescribed the
// traction is genuinely an unknown.
//
// The datum is the confined strain u = (0, 0, e z): the exact solution of the
// bonded problem, so whatever the fracture does is the whole of the difference.
Problem build(int n, int dim, Family family, double strain) {
  Problem p{mimetika::mesh::column(n, dim, family, 1.0), {}, nullptr, strain};
  const graphos::Complex& c = p.mesh.topology();
  const int axis = dim - 1;
  p.fault = facets_at(p.mesh, dim, 0.5);

  std::vector<Index> all;
  for (const Index f : mimetika::boundary_facets(c, dim)) all.push_back(f);

  std::array<double, 3> constant{};
  std::array<double, 9> gradient{};
  gradient[static_cast<std::size_t>(axis * 3 + axis)] = strain;  // u_axis = e x_axis

  p.model = std::make_unique<CauchyElasticityModel>(p.mesh, dim, ElasticMaterial{kMu, kLam});
  // THE AFFINE DATUM EXPANDS ABOUT THE ADJACENT CELL'S CENTROID -- that is
  // where the stress operators take their facet moments -- so the constant is
  // the datum's value THERE, not at the facet. Using the facet's own centroid
  // leaves a residual displacement of order eps * (x_f - x_E), which is exactly
  // the size of the gap being measured.
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, dim - 1);
  for (const Index f : all) {
    const Index cell =
        cob.indices[static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)])];
    const auto xE = exokal::centroid(p.mesh, dim, cell);
    std::array<double, 3> a{};
    a[static_cast<std::size_t>(axis)] = strain * xE[static_cast<std::size_t>(axis)];
    p.model->prescribe_displacement({f}, a, gradient);
  }
  // BEFORE build(): prescribing the traction changes which equations exist
  p.model->prescribe_traction(p.fault);
  p.model->build();
  return p;
}

}  // namespace

// A UNIFORM TRACTION LANDS ENTIRELY ON THE LEADING MOMENT, because the facet
// chart has chi_0 = 1 and int_f chi_b = 0 for b >= 1. The higher moments being
// exactly zero is what makes one enforcement point per facet a consistent
// statement rather than an approximation.
MIMETIKA_TEST(a_uniform_traction_lands_on_the_constant_moment_alone) {
  Problem p = build(4, 3, Family::cartesian, -0.5);
  const int nb = p.model->stress_operators().moments_per_facet();
  const Fracture fr(p.mesh, 3, p.fault, nb);
  const CauchyContactMechanics mech(*p.model, fr);

  std::vector<Vec3> x(mech.n_points());
  x[0][0] = -0.7;
  std::vector<double> moments;
  mech.to_moments(x, moments);

  const std::size_t ndf = 3 * static_cast<std::size_t>(nb);
  CHECK(moments.size() == fr.size() * ndf);
  // moment 0 carries t_k |f|; every higher moment is zero
  CHECK(near(moments[2], -0.7 * fr.frame(0).measure, 1e-12));
  for (std::size_t k = 3; k < ndf; ++k) CHECK(near(moments[k], 0.0, 1e-14));
}

namespace {

// solve one step with the given law and return the converged state
template <class Law>
ContactState run(Problem& p, const Law& law, int dim) {
  const Fracture fr(p.mesh, dim, p.fault, p.model->stress_operators().moments_per_facet());
  const CauchyContactMechanics mech(*p.model, fr);
  DriverOptions opt;
  opt.relaxation = 0.5;
  opt.tolerance = 1e-12;
  opt.max_iterations = 2000;
  const ContactDriver d(mech, law, default_augmentation(p.mesh, dim, p.fault, kMu, kLam), opt);
  return d.solve_step();
}

}  // namespace

// -- Signorini ----------------------------------------------------------------

// TENSION OPENS THE FRACTURE WITH ZERO TRACTION. Pulling the column apart, the
// unilateral condition must let it separate and carry nothing -- the one thing
// a bonded solve cannot do.
MIMETIKA_TEST(tension_opens_the_fracture_with_zero_traction) {
  Problem p = build(4, 3, Family::cartesian, +0.01);
  const SignoriniCoulomb law(0.6);
  const ContactState s = run(p, law, 3);
  std::printf("  tension    %3d iterations   t_n %+.3e   g_n %+.6e\n", s.iterations,
              s.traction[0][0], s.jump[0][0]);
  CHECK(s.converged);
  for (std::size_t i = 0; i < s.traction.size(); ++i) {
    CHECK(std::abs(s.traction[i][0]) < 1e-10);  // traction free
    CHECK(s.jump[i][0] > 0.0);                  // genuinely open
  }
}

// COMPRESSION CLOSES WITHOUT INTERPENETRATION, and the traction it settles on
// is the EXACT confined stress -K_oed * eps. That number is the test: a scheme
// that merely closed the gap could still carry the wrong load.
MIMETIKA_TEST(compression_closes_at_the_exact_confined_stress) {
  const double strain = -0.01;
  Problem p = build(4, 3, Family::cartesian, strain);
  const SignoriniCoulomb law(0.6);
  const ContactState s = run(p, law, 3);
  const double exact = (kLam + 2.0 * kMu) * strain;  // K_oed * eps
  std::printf("  compression %3d iterations   t_n %+.8e   exact %+.8e   g_n %+.2e\n", s.iterations,
              s.traction[0][0], exact, s.jump[0][0]);
  CHECK(s.converged);
  for (std::size_t i = 0; i < s.traction.size(); ++i) {
    CHECK(s.traction[i][0] < 0.0);         // in compression
    CHECK(std::abs(s.jump[i][0]) < 1e-8);  // no interpenetration
    CHECK(std::abs(s.traction[i][0] - exact) < 1e-6 * std::abs(exact));
  }
}

// COMPLEMENTARITY: g_n >= 0, t_n <= 0 and g_n t_n = 0 at every enforcement
// point, under tension, compression and no load alike. The three conditions
// together ARE Signorini, so checking them separately is checking the law.
MIMETIKA_TEST(complementarity_holds_under_tension_compression_and_neither) {
  for (const double strain : {+0.01, -0.01, 0.0}) {
    Problem p = build(4, 3, Family::cartesian, strain);
    const SignoriniCoulomb law(0.6);
    const ContactState s = run(p, law, 3);
    CHECK(s.converged);
    double worst = 0.0;
    for (std::size_t i = 0; i < s.traction.size(); ++i) {
      CHECK(s.jump[i][0] > -1e-8);      // no interpenetration
      CHECK(s.traction[i][0] < 1e-10);  // no tension
      worst = std::max(worst, std::abs(s.jump[i][0] * s.traction[i][0]));
    }
    std::printf("  eps %+.3f   worst |g_n t_n| %.2e\n", strain, worst);
    CHECK(worst < 1e-9);
  }
}

// -- the bilateral law, for contrast ------------------------------------------

// A BILATERAL FAULT IS HELD SHUT AND MAY CARRY TENSION, which is the modelling
// choice an INCREMENTAL problem needs: a fault under in-situ compression must
// not be opened by a tensile increment. The same load that opens it under
// Signorini leaves it closed here.
MIMETIKA_TEST(the_bilateral_law_holds_a_tensioned_fault_shut) {
  Problem p = build(4, 3, Family::cartesian, +0.01);
  const FrictionlessBilateral law;
  const ContactState s = run(p, law, 3);
  std::printf("  bilateral  %3d iterations   t_n %+.6e   g_n %+.2e\n", s.iterations,
              s.traction[0][0], s.jump[0][0]);
  CHECK(s.converged);
  for (std::size_t i = 0; i < s.traction.size(); ++i) {
    CHECK(s.traction[i][0] > 0.0);                         // carrying TENSION, and holding
    CHECK(std::abs(s.jump[i][0]) < 1e-8);                  // shut
    CHECK(std::abs(s.traction[i].shear_norm(3)) < 1e-10);  // and free to slide
  }
}

// A FRACTURE MUST BE PRESCRIBED BEFORE THE MODEL IS BUILT, because prescribing
// changes which equations the system has. Catching the mismatch at construction
// is what keeps a silently bonded solve from being reported as a contact one.
MIMETIKA_TEST(a_fracture_the_model_did_not_prescribe_is_refused) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::cartesian, 1.0);
  const graphos::Complex& c = m.topology();
  std::vector<Index> loaded, confined;
  for (const Index f : mimetika::boundary_facets(c, 3)) {
    const double z = exokal::centroid(m, 2, f)[2];
    (std::abs(z - 1.0) < 1e-9 ? loaded : confined).push_back(f);
  }
  std::array<double, 9> applied{};
  applied[8] = -0.5;
  CauchyElasticityModel model(m, 3, ElasticMaterial{kMu, kLam});
  model.mechanics().emplace<mimetika::TractionBC>(loaded, applied);
  model.mechanics().emplace<mimetika::FreeSlipBC>(confined);
  model.build();  // NO prescribe_traction

  const Fracture fr(m, 3, facets_at(m, 3, 0.5), 3);
  bool refused = false;
  try {
    const CauchyContactMechanics bad(model, fr);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

MIMETIKA_TEST_MAIN()
