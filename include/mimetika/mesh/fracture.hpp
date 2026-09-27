#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "exokal/io/vtu.hpp"

// HOW A FRACTURE IS IDENTIFIED IN A CONFORMAL MESH.
//
// By a NODE SET: the vertices lying on it. A facet belongs to the fracture when
// all of its vertices do, which is exokal's rule and is stated there once --
// `fracture_network` in exokal/preprocess. Nothing here repeats it.
//
// The node set is the right key and not merely a convenient one:
//
//   * Facet indices do not survive a file. The complex derives its facets from
//     the cells, so a facet's number is a property of the mesh object and not
//     of the mesh; a tag written as an index means something else after a round
//     trip. A vertex set is what the complex itself merges shared faces by.
//   * It is what the cut takes. exokal::mixed_dimensional_mesh consumes node
//     sets, so anything else is a conversion with a failure mode -- a facet set
//     whose vertices induce MORE facets than were tagged, which is silent.
//   * It is what the preprocessor already writes.
//
// THE FILE CONVENTION is one PointData array per fracture, `f<i>` followed by
// `_nodes`, `_node_set` or `_node_sets`, carrying a membership flag per point.
// That is what the exokal preprocessor emits, and it has emitted more than one
// spelling: `f0_nodes` .. `f51_nodes` on the 52-fracture benchmark, and
// `f1_node_set` .. `f5_node_set` on the DFN meshes. All are read, because the
// distinction carries no meaning and a mesh should not need rewriting to be
// used. A mesh from anywhere else supplies the same thing: one mask per
// fracture, or a node set built geometrically, as the structured generators do.
//
// NUMBERING IS THE WRITER'S. Some files count from 0 and some from 1, so `i` is
// kept as a label rather than turned into a position; the fractures come back
// ordered by it, not by where the arrays sat in the file.

namespace mimetika::mesh {

struct FractureNodeSets {
  std::vector<std::string> names;              // "f0", "f1", ... in that order
  std::vector<std::vector<graphos::Index>> sets;  // the vertices of each, ascending

  std::size_t size() const { return sets.size(); }
};

namespace fracture_detail {

// "f12_nodes", "f12_node_set", "f12_node_sets" -> 12; anything else -> -1.
// Anchored on both ends, so "vtkOriginalPointIds" and a field called "flux" are
// not fractures.
inline long long fracture_index(const std::string& name) {
  if (name.empty() || name[0] != 'f') return -1;
  std::size_t end = 0;
  for (const char* suffix : {"_node_sets", "_node_set", "_nodes"}) {
    const std::size_t n = std::string(suffix).size();
    if (name.size() > n + 1 && name.compare(name.size() - n, n, suffix) == 0) {
      end = name.size() - n;
      break;
    }
  }
  if (end == 0) return -1;
  const std::string digits = name.substr(1, end - 1);
  if (digits.empty()) return -1;
  for (const char ch : digits) {
    if (ch < '0' || ch > '9') return -1;
  }
  return std::stoll(digits);
}

}  // namespace fracture_detail

// The node sets a read .vtu carries, by the convention above.
//
// A point belongs to the fracture when its flag is nonzero: the preprocessor
// writes UInt8 zero and one, and a mesh annotated by hand may write anything
// truthy, so the test is against zero rather than for one.
inline FractureNodeSets fracture_node_sets(const exokal::VtuFile& file) {
  std::map<long long, const std::vector<long long>*> by_index;
  for (const auto& [name, values] : file.point_data) {
    const long long i = fracture_detail::fracture_index(name);
    if (i >= 0) by_index.emplace(i, &values);
  }
  const auto n_points = static_cast<std::size_t>(file.mesh.count(0));
  FractureNodeSets out;
  for (const auto& [i, values] : by_index) {
    if (values->size() != n_points) {
      throw std::invalid_argument("fracture_node_sets: the mask of f" + std::to_string(i) +
                                  " has " + std::to_string(values->size()) +
                                  " entries against " + std::to_string(n_points) +
                                  " points -- it is a flag per point, not a list of them");
    }
    std::vector<graphos::Index> set;
    for (std::size_t v = 0; v < n_points; ++v) {
      if ((*values)[v] != 0) set.push_back(static_cast<graphos::Index>(v));
    }
    out.names.push_back("f" + std::to_string(i));
    out.sets.push_back(std::move(set));
  }
  return out;
}

// One fracture of a set, by name: the single-fracture formulation is what
// mimetika::strata_of admits today, so a mesh carrying a network is used by
// choosing from it rather than by being refused outright.
inline std::vector<std::vector<graphos::Index>> one_fracture(const FractureNodeSets& all,
                                                             const std::string& name) {
  for (std::size_t i = 0; i < all.names.size(); ++i) {
    if (all.names[i] == name) return {all.sets[i]};
  }
  std::string known;
  for (const std::string& n : all.names) known += (known.empty() ? "" : ", ") + n;
  throw std::invalid_argument("one_fracture: no fracture named '" + name + "'; the mesh carries " +
                              (known.empty() ? std::string("none") : known));
}

}  // namespace mimetika::mesh
