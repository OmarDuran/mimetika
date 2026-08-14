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

// A BOX of `n[k]` cells along each axis, spanning `lengths` from `origin`.
//
// What a column cannot express and a benchmark stated in metres needs: an
// origin away from zero, a different size and a different resolution per axis.
// Fig. 4 of Novikov et al. is the case in point -- a 4500 m square, centred on
// the reservoir, resolved coarsely across and finely down, because the answer
// varies only with depth and steps sharply at the reservoir edges.
inline exokal::Mesh box(const std::array<int, 3>& n, int dim, Family family,
                        const std::array<double, 3>& lengths,
                        const std::array<double, 3>& origin = {0.0, 0.0, 0.0}) {
  for (int k = 0; k < dim; ++k) {
    if (n[static_cast<std::size_t>(k)] < 1) throw std::invalid_argument("box: resolution");
    if (!(lengths[static_cast<std::size_t>(k)] > 0.0)) {
      throw std::invalid_argument("box: lengths must be positive");
    }
  }
  const int nx = n[0], ny = n[1];
  const double lx = lengths[0], ly = lengths[1];
  const bool tri = family == Family::simplex;

  std::vector<Point> plane;
  const auto vid = [nx](int i, int j) { return static_cast<Index>(j * (nx + 1) + i); };
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      plane.push_back({origin[0] + i * lx / nx, origin[1] + j * ly / ny, origin[2]});
    }
  }
  std::vector<std::vector<Index>> cells;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const Index a = vid(i, j), b = vid(i + 1, j), c = vid(i + 1, j + 1), d = vid(i, j + 1);
      if (tri) {
        cells.push_back({a, b, c});
        cells.push_back({a, c, d});
      } else {
        cells.push_back({a, b, c, d});
      }
    }
  }
  if (dim == 2) {
    return tri ? exokal::Mesh::from_simplices(2, std::move(plane), cells)
               : exokal::Mesh::from_polygons(std::move(plane), cells);
  }
  if (dim != 3) throw std::invalid_argument("box: dimension");
  return structured_detail::extrude(std::move(plane), cells, lengths[2], n[2], family);
}

// -- graded coordinate lines ---------------------------------------------------
//
// NODE COORDINATES THAT HONOUR A SET OF INTERFACES and cluster cells at them.
//
// Every interface becomes a node, so a discontinuity in material or loading
// lands on a cell FACE rather than bisecting a cell -- where a cell-centred test
// would put it on the wrong side and shift the answer by half a cell. That is
// the same requirement Benchmark 0's reservoir boundary has, stated once and
// satisfied by construction instead of checked after the fact.
//
// RESOLUTION IS CONCENTRATED AT THE INTERFACES, not spread evenly between them,
// because that is where the error lives: the fields are non-smooth across a
// loading jump -- in the displaced-fault benchmark the analytic Coulomb stress
// is logarithmically singular at the reservoir edges -- so the discretization
// error is dominated by the few cells nearest each interface and is negligible a
// handful of cells away. `spacing` is therefore the size AT an interface; cells
// grow by `growth` away from it, towards mid-span between two interfaces and
// outwards to the extent beyond the outermost.

namespace grading_detail {

// Cell sizes across [left, right]: smallest at BOTH ends, largest mid-span.
//
// A one-sided geometric fan would refine one interface and starve the other.
// The weights are growth^min(i, n-1-i), which is symmetric, so both bounding
// interfaces get the fine cells.
inline std::vector<double> two_sided(double left, double right, double spacing,
                                     double growth) {
  const double length = right - left;
  for (int count = 1; count < 4096; ++count) {
    std::vector<double> w(static_cast<std::size_t>(count));
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
      w[static_cast<std::size_t>(i)] = std::pow(growth, std::min(i, count - 1 - i));
      sum += w[static_cast<std::size_t>(i)];
    }
    double smallest = length;
    for (const double v : w) smallest = std::min(smallest, length * v / sum);
    if (smallest <= spacing) {
      std::vector<double> nodes{left};
      double at = left;
      for (const double v : w) {
        at += length * v / sum;
        nodes.push_back(at);
      }
      return nodes;
    }
  }
  throw std::invalid_argument("graded_coordinates: could not reach the requested spacing");
}

}  // namespace grading_detail

inline std::vector<double> graded_coordinates(std::vector<double> interfaces,
                                              std::array<double, 2> extent, double spacing,
                                              double growth = 1.35, double max_spacing = 0.0) {
  if (interfaces.empty()) throw std::invalid_argument("graded_coordinates: no interfaces");
  std::sort(interfaces.begin(), interfaces.end());
  interfaces.erase(std::unique(interfaces.begin(), interfaces.end()), interfaces.end());
  const double lo = interfaces.front(), hi = interfaces.back();
  if (!(extent[0] <= lo && hi <= extent[1])) {
    throw std::invalid_argument("graded_coordinates: interfaces must lie inside the extent");
  }

  std::vector<double> nodes{lo};
  for (std::size_t i = 0; i + 1 < interfaces.size(); ++i) {
    const std::vector<double> span =
        grading_detail::two_sided(interfaces[i], interfaces[i + 1], spacing, growth);
    nodes.insert(nodes.end(), span.begin() + 1, span.end());
  }

  // outwards to each extent, coarsening geometrically and capped at max_spacing
  for (int side = 0; side < 2; ++side) {
    const double start = side == 0 ? lo : hi;
    const double stop = extent[static_cast<std::size_t>(side)];
    const double step = side == 0 ? -1.0 : 1.0;
    double at = start, size = spacing;
    while ((stop - at) * step > 1e-9) {
      size *= growth;
      if (max_spacing > 0.0) size = std::min(size, max_spacing);
      at += step * size;
      nodes.push_back((stop - at) * step > 0.0 ? at : stop);
    }
    nodes.push_back(stop);
  }

  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

// A TENSOR-PRODUCT MESH FROM EXPLICIT COORDINATE LINES, which is what a graded
// mesh is: `box` with the spacing chosen per node rather than uniformly.
inline exokal::Mesh tensor_product(const std::vector<double>& xs, const std::vector<double>& ys,
                                   Family family = Family::cartesian) {
  if (xs.size() < 2 || ys.size() < 2) {
    throw std::invalid_argument("tensor_product: at least one cell per axis");
  }
  const auto nx = xs.size() - 1, ny = ys.size() - 1;
  const bool tri = family == Family::simplex;
  std::vector<Point> p;
  const auto vid = [nx](std::size_t i, std::size_t j) {
    return static_cast<Index>(j * (nx + 1) + i);
  };
  for (std::size_t j = 0; j <= ny; ++j) {
    for (std::size_t i = 0; i <= nx; ++i) p.push_back({xs[i], ys[j], 0.0});
  }
  std::vector<std::vector<Index>> cells;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const Index a = vid(i, j), b = vid(i + 1, j), c = vid(i + 1, j + 1), d = vid(i, j + 1);
      if (tri) {
        cells.push_back({a, b, c});
        cells.push_back({a, c, d});
      } else {
        cells.push_back({a, b, c, d});
      }
    }
  }
  return tri ? exokal::Mesh::from_simplices(2, std::move(p), cells)
             : exokal::Mesh::from_polygons(std::move(p), cells);
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
