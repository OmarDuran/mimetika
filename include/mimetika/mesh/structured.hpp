#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "exokal/core/mesh.hpp"

// STRUCTURED DOMAINS, in either dimension and every cell family.
//
// A benchmark is a domain plus a material plus conditions, and the domain is
// the part that has to stop being a special case. The same column and the same
// annulus are generated here as quadrilaterals or triangles in the plane, and
// as hexahedra, tetrahedra or PRISMS in space -- so a solver is exercised
// against the same closed form on every cell type it claims to support.
//
// The prism is the interesting one. It is neither a simplex nor a
// tensor-product cell: five faces, two triangles and three quadrilaterals, and
// a mimetic construction that works on it is genuinely polytopal rather than
// hex-and-tet with extra steps. In two dimensions a prism over an interval IS
// a quadrilateral, so the family coincides with cartesian there and says so
// rather than pretending to be different.

namespace mimetika::mesh {

using graphos::Index;
using Point = exokal::Mesh::Point;

enum class Family { cartesian, simplex, prism };

inline const char* name(Family f) {
  switch (f) {
    case Family::cartesian: return "cartesian";
    case Family::simplex: return "simplex";
    case Family::prism: return "prism";
  }
  return "?";
}

namespace structured_detail {

// A LAYER OF PLANAR CELLS, EXTRUDED. Quadrilaterals become hexahedra;
// triangles become prisms, kept whole or split into three tetrahedra.
//
// THE SPLIT MUST AGREE ACROSS A SHARED FACE. A prism's three quadrilateral
// faces each need a diagonal, and two prisms meeting on one must cut it the
// same way -- otherwise the tetrahedra do not share faces, the complex is still
// valid, and it describes a domain riddled with internal boundary. Ordering the
// triangle's vertices by GLOBAL INDEX before applying a fixed rule is what
// makes two neighbours agree: they see the same two indices on the shared face.
inline exokal::Mesh extrude(std::vector<Point> plane,
                            const std::vector<std::vector<Index>>& cells, double h, int layers,
                            Family family) {
  const auto n = static_cast<Index>(plane.size());
  std::vector<Point> pts;
  for (int L = 0; L <= layers; ++L) {
    for (const Point& q : plane) pts.push_back({q[0], q[1], L * h / layers});
  }
  const auto at = [n](Index v, int L) { return v + static_cast<Index>(L) * n; };

  if (family == Family::simplex) {
    std::vector<std::vector<Index>> out;
    for (int L = 0; L < layers; ++L) {
      for (const auto& f : cells) {
        std::array<Index, 3> v{f[0], f[1], f[2]};
        std::sort(v.begin(), v.end());
        const Index a = at(v[0], L), b = at(v[1], L), c = at(v[2], L);
        const Index A = at(v[0], L + 1), B = at(v[1], L + 1), C = at(v[2], L + 1);
        // NO WINDING FIXUP HERE. A subdivision that tiles may still hand back
        // cells wound the other way -- the Freudenthal cut of a cube
        // alternates -- and it does not matter: exokal::Mesh orients what it is
        // given, coherently and then globally, so a generator states the
        // subdivision and nothing else.
        for (std::vector<Index> q : {std::vector<Index>{a, b, c, C},
                                     std::vector<Index>{a, b, B, C},
                                     std::vector<Index>{a, A, B, C}}) {
          out.push_back(std::move(q));
        }
      }
    }
    return exokal::Mesh::from_simplices(3, std::move(pts), out);
  }

  std::vector<std::vector<std::vector<Index>>> out;
  for (int L = 0; L < layers; ++L) {
    for (const auto& f : cells) {
      const std::size_t k = f.size();
      std::vector<std::vector<Index>> cell;
      std::vector<Index> bottom, top;
      for (std::size_t i = 0; i < k; ++i) {
        bottom.push_back(at(f[k - 1 - i], L));   // reversed: the cycle faces down
        top.push_back(at(f[i], L + 1));
      }
      cell.push_back(std::move(bottom));
      cell.push_back(std::move(top));
      for (std::size_t i = 0; i < k; ++i) {
        const Index a = f[i], b = f[(i + 1) % k];
        cell.push_back({at(a, L), at(b, L), at(b, L + 1), at(a, L + 1)});
      }
      out.push_back(std::move(cell));
    }
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), out);
}

}  // namespace structured_detail

// A COLUMN of `n` cells along the last coordinate, unit cross-section.
inline exokal::Mesh column(int n, int dim, Family family, double height = 1.0,
                           double width = 1.0) {
  if (dim == 2) {
    // in the plane a prism over an interval is a quadrilateral
    const bool tri = family == Family::simplex;
    std::vector<Point> p;
    const auto vid = [](int i, int j) { return static_cast<Index>(j * 2 + i); };
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= 1; ++i) p.push_back({i * width, j * height / n, 0.0});
    }
    std::vector<std::vector<Index>> cells;
    for (int j = 0; j < n; ++j) {
      const Index a = vid(0, j), b = vid(1, j), c = vid(1, j + 1), d = vid(0, j + 1);
      if (tri) {
        cells.push_back({a, b, c});
        cells.push_back({a, c, d});
      } else {
        cells.push_back({a, b, c, d});
      }
    }
    return tri ? exokal::Mesh::from_simplices(2, std::move(p), cells)
               : exokal::Mesh::from_polygons(std::move(p), cells);
  }
  if (dim != 3) throw std::invalid_argument("column: dimension");

  // the unit square cross-section, then extruded n times along z
  std::vector<Point> plane{{0, 0, 0}, {width, 0, 0}, {width, width, 0}, {0, width, 0}};
  std::vector<std::vector<Index>> cells;
  if (family == Family::cartesian) {
    cells.push_back({0, 1, 2, 3});
  } else {
    cells.push_back({0, 1, 2});
    cells.push_back({0, 2, 3});
  }
  return structured_detail::extrude(std::move(plane), cells, height, n, family);
}

// A QUARTER ANNULUS from `a` to `b`, graded geometrically in the radius so the
// cells are fine where the gradient is; in three dimensions, one layer of it.
inline exokal::Mesh annulus(int nr, int nt, int dim, Family family, double a = 1.0,
                            double b = 10.0, double height = 1.0, int layers = 1) {
  constexpr double pi = 3.14159265358979323846;
  std::vector<Point> plane;
  const auto vid = [nt](int i, int j) { return static_cast<Index>(i * (nt + 1) + j); };
  const double growth = std::pow(b / a, 1.0 / nr);
  for (int i = 0; i <= nr; ++i) {
    const double r = a * std::pow(growth, i);
    for (int j = 0; j <= nt; ++j) {
      const double th = 0.5 * pi * static_cast<double>(j) / nt;
      plane.push_back({r * std::cos(th), r * std::sin(th), 0.0});
    }
  }
  const bool tri = family == Family::simplex || (dim == 3 && family == Family::prism);
  std::vector<std::vector<Index>> cells;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nt; ++j) {
      const Index p0 = vid(i, j), p1 = vid(i + 1, j), p2 = vid(i + 1, j + 1), p3 = vid(i, j + 1);
      if (tri) {
        cells.push_back({p0, p1, p2});
        cells.push_back({p0, p2, p3});
      } else {
        cells.push_back({p0, p1, p2, p3});
      }
    }
  }
  if (dim == 2) {
    return tri ? exokal::Mesh::from_simplices(2, std::move(plane), cells)
               : exokal::Mesh::from_polygons(std::move(plane), cells);
  }
  if (dim != 3) throw std::invalid_argument("annulus: dimension");
  return structured_detail::extrude(std::move(plane), cells, height, layers, family);
}

}  // namespace mimetika::mesh
