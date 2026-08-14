#include <cmath>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/algebraic_constraints/contact/trace.hpp"
#include "mimetika/mesh/structured.hpp"

// THE FRACTURE AND ITS FRAMES: the addressing the trace operator stands on.
//
// The trace itself -- the jump as the adjoint of D and A -- is exercised
// against a solved problem in test_contact_elasticity.cpp, where a known
// displacement field can be imposed and the recovered jump compared to it.
// What is checked here is everything that must hold BEFORE a solve: that a
// fracture is interior, that both cofaces read one frame, and that the rotation
// into and out of that frame is an isometry.
//
// That last point is not decoration. The traction and the jump are rotated by
// the same frame in opposite directions, so an error there would cancel in any
// round trip and show up only as a wrong friction cone -- the kind of defect
// that survives every test that does not name it.

using graphos::Index;
using mimetika::contact::Fracture;
using mimetika::contact::Vec3;
using mimetika::mesh::Family;

namespace {

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

// the facets of a column at a given height: a flat fracture cutting it
std::vector<Index> facets_at(const exokal::Mesh& m, int dim, double z, double tol = 1e-9) {
  std::vector<Index> out;
  const graphos::Complex& c = m.topology();
  for (Index f = 0; f < c.count(dim - 1); ++f) {
    const auto x = exokal::centroid(m, dim - 1, f);
    if (std::abs(x[static_cast<std::size_t>(dim - 1)] - z) < tol) out.push_back(f);
  }
  return out;
}

}  // namespace

// A FRACTURE IS INTERIOR: a jump needs two sides, and a boundary facet has one.
// Refusing at construction is what keeps the driver from producing a plausible
// answer on a fault that is really a boundary.
MIMETIKA_TEST(a_fracture_facet_must_have_two_cofaces) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::cartesian, 1.0, 1.0);
  // the base of the column: a boundary facet
  const std::vector<Index> boundary = facets_at(m, 3, 0.0);
  CHECK(!boundary.empty());
  bool refused = false;
  try {
    const Fracture bad(m, 3, boundary, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);

  // and an interior one is accepted
  const std::vector<Index> interior = facets_at(m, 3, 0.5);
  CHECK(!interior.empty());
  const Fracture ok(m, 3, interior, 3);
  CHECK(ok.size() == interior.size());
  CHECK(ok.n_points() == interior.size());  // one enforcement point per facet
}

MIMETIKA_TEST(an_empty_fracture_and_a_bad_moment_count_are_refused) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::cartesian, 1.0, 1.0);
  const std::vector<Index> f = facets_at(m, 3, 0.5);
  int refused = 0;
  try { const Fracture bad(m, 3, {}, 3); } catch (const std::invalid_argument&) { ++refused; }
  try { const Fracture bad(m, 3, f, 0); } catch (const std::invalid_argument&) { ++refused; }
  try { const Fracture bad(m, 3, f, 4); } catch (const std::invalid_argument&) { ++refused; }
  CHECK(refused == 3);
}

// THE FRAME IS THE FACET'S, so both cofaces read one convention. The normal is
// the CANONICAL one -- the direction the traction dofs are numbered in -- and
// the tangents follow from it alone, which is what makes the two sides of a
// fracture agree on what a shear component means.
MIMETIKA_TEST(the_frame_is_orthonormal_and_facet_intrinsic) {
  for (const int dim : {2, 3}) {
    const exokal::Mesh m = mimetika::mesh::column(4, dim, Family::cartesian, 1.0, 1.0);
    const std::vector<Index> f = facets_at(m, dim, 0.5);
    const Fracture fr(m, dim, f, dim);
    for (std::size_t i = 0; i < fr.size(); ++i) {
      const auto& F = fr.frame(i);
      CHECK(F.n_tangents == dim - 1);
      double nn = 0.0;
      for (std::size_t k = 0; k < 3; ++k) nn += F.normal[k] * F.normal[k];
      CHECK(near(nn, 1.0, 1e-12));
      for (int a = 0; a < F.n_tangents; ++a) {
        double tt = 0.0, tn = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
          tt += F.tangent[static_cast<std::size_t>(a)][k] * F.tangent[static_cast<std::size_t>(a)][k];
          tn += F.tangent[static_cast<std::size_t>(a)][k] * F.normal[k];
        }
        CHECK(near(tt, 1.0, 1e-12));   // unit
        CHECK(near(tn, 0.0, 1e-12));   // and orthogonal to the normal
      }
      if (F.n_tangents == 2) {
        double t12 = 0.0;
        for (std::size_t k = 0; k < 3; ++k) t12 += F.tangent[0][k] * F.tangent[1][k];
        CHECK(near(t12, 0.0, 1e-12));  // and to each other
      }
    }
  }
}

// THE ROTATION IS AN ISOMETRY, in both directions. The traction goes one way
// and the jump the other, so an error here would cancel in a round trip and
// surface only as a wrong friction cone.
MIMETIKA_TEST(the_frame_rotation_preserves_the_norm_and_round_trips) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::cartesian, 1.0, 1.0);
  const std::vector<Index> f = facets_at(m, 3, 0.5);
  const Fracture fr(m, 3, f, 3);

  Vec3 v;
  v[0] = -2.0;
  v[1] = 0.7;
  v[2] = -1.3;
  const double norm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

  for (std::size_t i = 0; i < fr.size(); ++i) {
    const std::array<double, 3> a = fr.to_ambient(v, i);
    double an2 = 0.0;
    for (std::size_t k = 0; k < 3; ++k) an2 += a[k] * a[k];
    CHECK(near(an2, norm2, 1e-12));  // an isometry

    const Vec3 back = fr.to_frame(a, i);
    CHECK(near(back[0], v[0], 1e-12));
    CHECK(near(back[1], v[1], 1e-12));
    CHECK(near(back[2], v[2], 1e-12));
  }
}

// THE NORMAL COMPONENT IS THE NORMAL COMPONENT: a facet-frame vector that is
// purely normal rotates to a multiple of the facet normal, and nothing else.
// This is what makes "t_n < 0 is compression" a statement about the fracture
// rather than about a coordinate axis.
MIMETIKA_TEST(a_purely_normal_traction_rotates_onto_the_facet_normal) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::simplex, 1.0, 1.0);
  const std::vector<Index> f = facets_at(m, 3, 0.5);
  CHECK(!f.empty());
  const Fracture fr(m, 3, f, 3);

  Vec3 n;
  n[0] = -1.0;  // unit compression
  for (std::size_t i = 0; i < fr.size(); ++i) {
    const std::array<double, 3> a = fr.to_ambient(n, i);
    const auto& F = fr.frame(i);
    for (std::size_t k = 0; k < 3; ++k) CHECK(near(a[k], -F.normal[k], 1e-12));
  }
}

// SEVERAL FRACTURES COEXIST, which is the reason a driver owns a facet set
// rather than the mesh owning one fracture: two laws on two disjoint sets are
// two drivers, and neither knows about the other.
MIMETIKA_TEST(disjoint_fractures_are_independent) {
  const exokal::Mesh m = mimetika::mesh::column(6, 3, Family::cartesian, 1.0, 1.0);
  const std::vector<Index> lower = facets_at(m, 3, 1.0 / 3.0);
  const std::vector<Index> upper = facets_at(m, 3, 2.0 / 3.0);
  CHECK(!lower.empty() && !upper.empty());
  CHECK(lower[0] != upper[0]);

  const Fracture a(m, 3, lower, 3);
  const Fracture b(m, 3, upper, 3);
  CHECK(a.size() == lower.size() && b.size() == upper.size());
  CHECK(a.facets()[0] != b.facets()[0]);
}

MIMETIKA_TEST_MAIN()
