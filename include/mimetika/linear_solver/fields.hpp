#pragma once

#include <string>
#include <vector>

#include "exokal/forms/model.hpp"

// Field index sets: which global unknowns belong to which factor of the space.
//
// A block preconditioner is a statement about the factors of a product space,
// and it needs exactly one thing from the discretization: the index set of each
// factor. Nothing about the operator, the mesh or the physics enters here.
//
// The layout already determines them. Stratum s begins at epoch.offset(s) and
// field f begins space.offset(f) into it, so one (stratum, field) pair owns a
// contiguous run of the global vector. A field carried by several strata owns
// one run per stratum and its block is their union — disjoint and ascending,
// because the strata are laid out in order.
//
// PETSc takes a strided IS per run, so a block costs two integers per stratum
// and nothing is materialized unless a caller asks for the indices themselves.
//
// The runs of stratum_field_blocks partition [0, epoch.size()) exactly. That is
// the invariant a fieldsplit depends on — an unassigned unknown is dropped from
// the preconditioner and a doubly-assigned one is applied twice.

namespace mimetika::solver {

using graphos::Index;

// a half-open run of global indices
struct DofRange {
  std::size_t begin{0};
  std::size_t end{0};

  std::size_t size() const { return end - begin; }
};

// The unknowns of one field, named as the solver will report it.
struct FieldBlock {
  std::string name;
  std::vector<DofRange> ranges;  // disjoint, ascending

  std::size_t size() const {
    std::size_t n = 0;
    for (const DofRange& r : ranges) n += r.size();
    return n;
  }

  // the indices themselves, for an IS that cannot be expressed as one stride
  std::vector<Index> indices() const {
    std::vector<Index> out;
    out.reserve(size());
    for (const DofRange& r : ranges) {
      for (std::size_t i = r.begin; i < r.end; ++i) out.push_back(static_cast<Index>(i));
    }
    return out;
  }
};

// One block per (stratum, field), named "<stratum>.<field>": the atoms of the
// layout, in global order.
inline std::vector<FieldBlock> stratum_field_blocks(const exokal::forms::StratifiedEpoch& e) {
  std::vector<FieldBlock> out;
  for (std::size_t s = 0; s < e.n_strata(); ++s) {
    const exokal::spaces::ProductSpace& sp = e.stratum(s).space();
    const auto base = static_cast<std::size_t>(e.offset(s));
    for (std::size_t f = 0; f < sp.n_fields(); ++f) {
      const std::size_t b = base + static_cast<std::size_t>(sp.offset(f));
      out.push_back(
          FieldBlock{e.entry(s).name + "." + sp.name(f), {DofRange{b, b + sp.map(f).size()}}});
    }
  }
  return out;
}

// One block per distinct field name, gathered over every stratum carrying it,
// in order of first appearance.
//
// This is the split a Riesz map is written against: the factors are H(div) and
// L², not "the ambient stratum's H(div) and the fracture stratum's H(div)". A
// mixed-dimensional problem carries the same field on several strata and they
// belong to the same block.
inline std::vector<FieldBlock> field_blocks(const exokal::forms::StratifiedEpoch& e) {
  std::vector<FieldBlock> out;
  for (std::size_t s = 0; s < e.n_strata(); ++s) {
    const exokal::spaces::ProductSpace& sp = e.stratum(s).space();
    const auto base = static_cast<std::size_t>(e.offset(s));
    for (std::size_t f = 0; f < sp.n_fields(); ++f) {
      const std::size_t b = base + static_cast<std::size_t>(sp.offset(f));
      const DofRange r{b, b + sp.map(f).size()};
      auto it = out.begin();
      for (; it != out.end(); ++it) {
        if (it->name == sp.name(f)) break;
      }
      if (it == out.end()) {
        out.push_back(FieldBlock{sp.name(f), {r}});
      } else {
        it->ranges.push_back(r);
      }
    }
  }
  return out;
}

// The unknowns a diagonal star makes eliminable: the first factor, the flux or
// the stress. Naming them is a permission -- see PetscSolver::set_condensable
// -- and the matrix decides whether the block really is diagonal, so this asks
// nothing about which product built it.
inline std::vector<int> first_field_dofs(const exokal::forms::StratifiedEpoch& e) {
  const std::vector<FieldBlock> blocks = field_blocks(e);
  std::vector<int> out;
  if (blocks.empty()) return out;
  out.reserve(blocks[0].size());
  for (const Index g : blocks[0].indices()) out.push_back(static_cast<int>(g));
  return out;
}

// Do these blocks partition [0, n) exactly? A fieldsplit is only a
// preconditioner for the whole operator if they do: an unknown in no block is
// left out of the preconditioner, and one in two blocks is corrected twice.
inline bool blocks_partition(const std::vector<FieldBlock>& blocks, std::size_t n) {
  std::vector<char> hit(n, 0);
  for (const FieldBlock& b : blocks) {
    for (const DofRange& r : b.ranges) {
      if (r.end > n || r.begin > r.end) return false;
      for (std::size_t i = r.begin; i < r.end; ++i) {
        if (hit[i] != 0) return false;  // claimed twice
        hit[i] = 1;
      }
    }
  }
  for (const char c : hit) {
    if (c == 0) return false;  // claimed by nobody
  }
  return true;
}

}  // namespace mimetika::solver
