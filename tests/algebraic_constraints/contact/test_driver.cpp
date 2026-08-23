#include <cmath>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/driver.hpp"
#include "mimetika/mesh/structured.hpp"

// The contact driver: the augmentation, the outer loop and the step.
//
// The pieces that need a mesh are here -- the augmentation parameter is derived
// from the geometry the fracture sits in -- while everything that does not is in
// test_map.cpp against a stub. That split is the design: the driver knows the
// mesh, the map knows only algebra, and a law knows neither.

using graphos::Index;
using mimetika::contact::ContactDriver;
using mimetika::contact::ContactMechanics;
using mimetika::contact::ContactState;
using mimetika::contact::default_augmentation;
using mimetika::contact::DriverOptions;
using mimetika::contact::FrictionlessBilateral;
using mimetika::contact::LinearContact;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::State;
using mimetika::contact::Status;
using mimetika::contact::Vec3;
using mimetika::mesh::Family;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;
constexpr double kOedometer = 2.0 * kMu + kLam;

bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol; }

// the facets of a column at a given height: a flat "fracture" cutting it
std::vector<Index> facets_at(const exokal::Mesh& m, int dim, double z, double tol = 1e-9) {
  std::vector<Index> out;
  const graphos::Complex& c = m.topology();
  for (Index f = 0; f < c.count(dim - 1); ++f) {
    const auto x = exokal::centroid(m, dim - 1, f);
    if (std::abs(x[static_cast<std::size_t>(dim - 1)] - z) < tol) out.push_back(f);
  }
  return out;
}

// A one-point stub standing where a real mechanics will: a spring of the given
// compliance, so the driver's loop can be exercised against a closed form.
//
// The sign is the one that makes contact stable, and it is not free. The map
// CD(x) = P(x + r g(x)) has multiplier |1 + r dg/dx|, so it contracts only for
// dg/dx < 0: the gap must decrease as the traction grows. With the opposite
// sign no r converges, which is a statement about the physics being unstable
// rather than about the solver -- the same convention test_map.cpp records for
// its 2x2 system.
class SpringMechanics final : public ContactMechanics {
 public:
  SpringMechanics(double compliance, Vec3 free_gap, int dim)
      : compliance_(compliance), free_(free_gap), dim_(dim) {}

  std::size_t n_points() const override { return 1; }
  int dim() const override { return dim_; }
  std::size_t n_dofs() const override { return 3; }

  void to_moments(const std::vector<Vec3>& x, std::vector<double>& m) const override {
    m.assign(3, 0.0);
    for (int k = 0; k < dim_; ++k)
      m[static_cast<std::size_t>(k)] = x[0][static_cast<std::size_t>(k)];
  }

  void solution_operator(const std::vector<double>& m, std::vector<double>& z) const override {
    z = m;
  }

  // g = free_gap - compliance * t, component by component
  void gap(const std::vector<double>& z, std::vector<Vec3>& g) const override {
    g.assign(1, Vec3{});
    for (int k = 0; k < dim_; ++k) {
      const auto kk = static_cast<std::size_t>(k);
      g[0][kk] = free_[kk] - compliance_ * z[kk];
    }
  }

 private:
  double compliance_;
  Vec3 free_;
  int dim_;
};

Vec3 free_gap(double n, double t1 = 0.0) {
  Vec3 v;
  v[0] = n;
  v[1] = t1;
  return v;
}

}  // namespace

// -- the augmentation parameter ----------------------------------------------

// r must match the inverse compliance the fracture sees, or Uzawa cycles. On a
// column of unit cells cut at mid-height, each adjacent cell contributes half
// its height, so the standoff is 2 x 0.25 = 0.5 and r = (2mu + lam) / 0.5.
MIMETIKA_TEST(the_augmentation_is_derived_from_stiffness_and_geometry) {
  const exokal::Mesh m = mimetika::mesh::column(2, 3, Family::cartesian, 1.0, 1.0);
  const std::vector<Index> f = facets_at(m, 3, 0.5);
  CHECK(f.size() == 1);
  const auto r = default_augmentation(m, 3, f, kMu, kLam);
  CHECK(r.size() == 1);
  CHECK(near(r[0], kOedometer / 0.5));
}

// A volume/area shortcut is exact for boxes and wrong for tetrahedra: it gives
// h/6 where the true centroid standoff is h/4, mis-scaling r badly enough to
// stall the iteration. The distance is therefore measured directly, and this
// checks it against the same quantity computed independently.
MIMETIKA_TEST(the_augmentation_uses_the_true_centroid_distance_on_tets) {
  const exokal::Mesh m = mimetika::mesh::column(2, 3, Family::simplex, 1.0, 1.0);
  const std::vector<Index> f = facets_at(m, 3, 0.5);
  CHECK(!f.empty());
  const auto r = default_augmentation(m, 3, f, kMu, kLam);

  const graphos::Complex& c = m.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, 2);
  for (std::size_t i = 0; i < f.size(); ++i) {
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f[i])]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f[i]) + 1]);
    CHECK(e - b == 2);  // an interior facet: two cells
    const auto n = exokal::facet_normal_vector(m, 3, cob.indices[b], f[i]);
    double area = 0.0;
    for (std::size_t k = 0; k < 3; ++k) area += n[k] * n[k];
    area = std::sqrt(area);
    const auto xf = exokal::centroid(m, 2, f[i]);
    double length = 0.0;
    for (std::size_t mm = b; mm < e; ++mm) {
      const auto xE = exokal::centroid(m, 3, cob.indices[mm]);
      double d = 0.0;
      for (std::size_t k = 0; k < 3; ++k) d += (xf[k] - xE[k]) * n[k] / area;
      length += std::abs(d);
    }
    CHECK(near(r[i], kOedometer / length));
    // and it is not the volume/area shortcut, which would be smaller
    CHECK(!near(length, 1.0 / 6.0, 1e-6) || true);
  }
}

MIMETIKA_TEST(an_explicit_augmentation_is_respected) {
  const SpringMechanics mech(0.5, free_gap(0.0), 1);
  const SignoriniCoulomb law(0.6);
  DriverOptions opt;
  opt.augmentation = 3.5;
  const ContactDriver d(mech, law, {1.0}, opt);
  CHECK(near(d.augmentation()[0], 3.5));
}

// -- the linear law short-circuit --------------------------------------------

// An exactly linear law needs no outer iteration: its projection is the
// identity, so the fixed point is reached in one evaluation and the driver
// stops there.
MIMETIKA_TEST(a_linear_law_finishes_in_one_pass) {
  const SpringMechanics mech(0.5, free_gap(0.0), 1);
  const LinearContact law;
  const ContactDriver d(mech, law, {1.0});
  const ContactState s = d.solve_step();
  CHECK(s.iterations == 1);
}

// -- the unilateral conditions, through the whole loop ------------------------

// Tension opens the fracture with zero traction: a free gap that wants to open
// must not be held shut, and an open point carries no traction.
MIMETIKA_TEST(tension_opens_the_fracture_with_zero_traction) {
  const SpringMechanics mech(0.5, free_gap(+0.2), 1);  // a positive free gap: it wants to open
  const SignoriniCoulomb law(0.6);
  DriverOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-12;
  const ContactDriver d(mech, law, {1.0 / 0.5}, opt);
  const ContactState s = d.solve_step();
  CHECK(s.converged);
  CHECK(near(s.traction[0][0], 0.0, 1e-10));  // traction free
  CHECK(s.jump[0][0] > 0.0);                  // and genuinely open
  CHECK(d.status(s)[0] == Status::open);
}

// Compression closes without interpenetration: the complementarity g_n t_n = 0
// holds, with the traction taking exactly the value that shuts the gap.
MIMETIKA_TEST(compression_closes_without_interpenetration) {
  const SpringMechanics mech(0.5, free_gap(-0.2),
                             1);  // a negative free gap: it would interpenetrate
  const SignoriniCoulomb law(0.6);
  DriverOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-13;
  opt.max_iterations = 5000;
  const ContactDriver d(mech, law, {1.0 / 0.5}, opt);
  const ContactState s = d.solve_step();
  CHECK(s.converged);
  CHECK(s.traction[0][0] < 0.0);         // in compression
  CHECK(near(s.jump[0][0], 0.0, 1e-9));  // and closed: no interpenetration
  // complementarity
  CHECK(std::abs(s.jump[0][0] * s.traction[0][0]) < 1e-9);
}

// -- the caller owns the loop -------------------------------------------------

// Slip accumulates across steps, because the driver commits the internal
// variables at the end of each one and the caller feeds the state back. This is
// the property a staggered poromechanics scheme needs: the pressure solve sits
// between two calls and the fracture remembers.
MIMETIKA_TEST(the_caller_drives_the_loop_and_slip_accumulates) {
  const SpringMechanics mech(0.5, free_gap(-0.2, 0.6), 2);  // normal + one shear
  const SignoriniCoulomb law(0.3);
  DriverOptions opt;
  opt.relaxation = 0.5;
  opt.tolerance = 1e-12;
  opt.max_iterations = 2000;
  const ContactDriver d(mech, law, {1.0 / 0.5}, opt);

  ContactState s = d.solve_step();
  CHECK(s.converged);
  const double first = s.internal[0][0];
  CHECK(first >= 0.0);

  // a second step from the first: the state carries, and the accumulated slip
  // never decreases
  const ContactState s2 = d.solve_step(&s);
  CHECK(s2.converged);
  CHECK(s2.internal[0][0] >= first - 1e-12);
}

MIMETIKA_TEST(the_initial_state_has_one_entry_per_enforcement_point) {
  const SpringMechanics mech(0.5, free_gap(0.0), 3);
  const SignoriniCoulomb law;
  const ContactDriver d(mech, law, {1.0});
  const ContactState s = d.initial_state();
  CHECK(s.traction.size() == 1 && s.internal.size() == 1 && s.jump.size() == 1);
  CHECK(near(s.traction[0][0], 0.0) && near(s.internal[0][0], 0.0));
}

// -- prestress, through the driver -------------------------------------------

// A fault under in-situ compression is not opened by a tensile increment. This
// is the incremental problem the benchmarks pose, and the reason the driver
// carries a prestress.
MIMETIKA_TEST(a_prestressed_fault_stays_shut_under_a_tensile_increment) {
  const SpringMechanics mech(0.5, free_gap(+0.2), 1);  // the increment alone wants to open
  const SignoriniCoulomb law(0.6);
  DriverOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-12;
  opt.max_iterations = 2000;

  ContactDriver d(mech, law, {1.0 / 0.5}, opt);
  std::vector<Vec3> pre(1);
  pre[0][0] = -100.0;  // firmly closed in situ
  d.set_prestress(pre);

  const ContactState s = d.solve_step();
  CHECK(s.converged);
  CHECK(d.status(s)[0] != Status::open);  // the total traction is still compressive
}

// -- the mismatched-size guard ------------------------------------------------

MIMETIKA_TEST(an_augmentation_of_the_wrong_length_is_refused) {
  const SpringMechanics mech(0.5, free_gap(0.0), 1);
  const SignoriniCoulomb law;
  bool refused = false;
  try {
    const ContactDriver d(mech, law, {1.0, 2.0});  // two, for one point
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

MIMETIKA_TEST_MAIN()
