#pragma once

#include <utility>
#include <vector>

#include "exokal/core/mesh.hpp"

// Meshes the tests build models on. Deliberately trivial: what is under
// test here is composition and declaration, not geometry.

namespace mimetika_test {

using exokal::Mesh;
using graphos::Index;

// An n x n x n grid of hexahedra on the unit-spaced lattice.
inline Mesh hex_grid(int n) {
  const auto vid = [n](int i, int j, int k) {
    return static_cast<Index>((k * (n + 1) + j) * (n + 1) + i);
  };
  static constexpr int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                      {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) pts.push_back({double(i), double(j), double(k)});
    }
  }
  std::vector<std::vector<std::vector<Index>>> cells;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        std::vector<std::vector<Index>> cell;
        for (const auto& f : faces) {
          std::vector<Index> cyc;
          for (const int l : f) {
            cyc.push_back(vid(i + (l & 1), j + ((l >> 1) & 1), k + ((l >> 2) & 1)));
          }
          cell.push_back(std::move(cyc));
        }
        cells.push_back(std::move(cell));
      }
    }
  }
  return Mesh::from_polyhedra(std::move(pts), cells);
}

inline Mesh unit_cube() { return hex_grid(1); }

}  // namespace mimetika_test
