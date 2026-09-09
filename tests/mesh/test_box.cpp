#include <array>
#include <cmath>

#include "../mimetika_test.hpp"
#include "exokal/geometry/embedding.hpp"
#include "exokal/preprocess/diagnostics.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"

using graphos::Index;
using mimetika::mesh::box;
using mimetika::mesh::Family;

namespace {

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// sum of the measures of every dim-cell
double total_measure(const exokal::Mesh& m, int dim) {
  double v = 0.0;
  for (Index e = 0; e < m.topology().count(dim); ++e) v += exokal::measure(m, dim, e);
  return v;
}

}  // namespace

// Cells per grid cell of the subdivision:
//
//   cartesian   one hexahedron
//   simplex     six tetrahedra, the Freudenthal cut of the cube
//   prism       two, the triangulated square extruded
MIMETIKA_TEST(the_subdivision_is_the_one_each_family_names) {
  const int n = 4;
  const Index cells = n * n * n;
  CHECK(box({n, n, n}, 3, Family::cartesian).topology().count(3) == cells);
  CHECK(box({n, n, n}, 3, Family::simplex).topology().count(3) == 6 * cells);
  CHECK(box({n, n, n}, 3, Family::prism).topology().count(3) == 2 * cells);

  // in the plane: a quadrilateral, or the two triangles it splits into
  CHECK(box({n, n, 1}, 2, Family::cartesian).topology().count(2) == n * n);
  CHECK(box({n, n, 1}, 2, Family::simplex).topology().count(2) == 2 * n * n);
}

// Every family shares the same (n+1)^3 grid of vertices: the subdivision cuts
// cells and adds no points. A duplicated vertex keeps the cell count right and
// disconnects the mesh.
MIMETIKA_TEST(the_families_share_one_grid_of_vertices) {
  const int n = 4;
  const Index nodes = (n + 1) * (n + 1) * (n + 1);
  for (const Family f : {Family::cartesian, Family::simplex, Family::prism}) {
    CHECK(box({n, n, n}, 3, f).topology().count(0) == nodes);
  }
}

// The cells tile the box: the measures sum to the product of the sides. A cell
// count alone is also satisfied by a subdivision that leaves a gap.
MIMETIKA_TEST(the_cells_fill_the_box_they_were_asked_for) {
  const std::array<double, 3> sides{2.0, 3.0, 0.5};
  const double volume = sides[0] * sides[1] * sides[2];
  for (const Family f : {Family::cartesian, Family::simplex, Family::prism}) {
    CHECK(near(total_measure(box({3, 4, 2}, 3, f, sides), 3), volume, 1e-12));
  }
  // and in the plane, the area
  CHECK(near(total_measure(box({3, 4, 1}, 2, Family::simplex, sides), 2), sides[0] * sides[1],
             1e-12));
}

// The vertex bounding box is [at, at + sides] componentwise.
MIMETIKA_TEST(the_box_sits_at_its_origin) {
  const std::array<double, 3> sides{2.0, 3.0, 0.5};
  const std::array<double, 3> at{-1.0, 5.0, 0.25};
  const exokal::Mesh m = box({2, 2, 2}, 3, Family::simplex, sides, at);
  std::array<double, 3> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
  for (Index v = 0; v < m.topology().count(0); ++v) {
    const auto& p = m.point(v);
    for (int k = 0; k < 3; ++k) {
      lo[static_cast<std::size_t>(k)] = std::min(lo[static_cast<std::size_t>(k)], p[k]);
      hi[static_cast<std::size_t>(k)] = std::max(hi[static_cast<std::size_t>(k)], p[k]);
    }
  }
  for (int k = 0; k < 3; ++k) {
    const auto j = static_cast<std::size_t>(k);
    CHECK(near(lo[j], at[j], 1e-12));
    CHECK(near(hi[j], at[j] + sides[j], 1e-12));
  }
}

// exokal::diagnose checks the complex: dd = 0, coherent orientations, no
// degenerate cell.
MIMETIKA_TEST(the_complex_it_builds_has_no_violations) {
  for (const Family f : {Family::cartesian, Family::simplex, Family::prism}) {
    const exokal::Mesh m = box({3, 3, 3}, 3, f);
    CHECK(exokal::diagnose(m).clean());
  }
}

// Boundary facets of an n^3 box: 6n^2 quadrilaterals, 12n^2 triangles once each
// square is cut in two. The count comes out only if interior facets are shared.
MIMETIKA_TEST(the_boundary_is_the_surface_of_the_box) {
  const int n = 3;
  CHECK(mimetika::boundary_facets(box({n, n, n}, 3, Family::cartesian).topology(), 3).size() ==
        static_cast<std::size_t>(6 * n * n));
  CHECK(mimetika::boundary_facets(box({n, n, n}, 3, Family::simplex).topology(), 3).size() ==
        static_cast<std::size_t>(12 * n * n));
}

MIMETIKA_TEST_MAIN()
