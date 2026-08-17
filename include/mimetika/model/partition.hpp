#pragma once

#include <vector>

#include "exokal/forms/model.hpp"
#include "exokal/spaces/partition.hpp"
#include "exokal/spaces/rcb.hpp"
#include "graphos/core/incidence.hpp"
#include "graphos/core/types.hpp"

// THE MODEL'S UNKNOWNS, SHARED OUT AMONG THE PROCESSES.
//
// Nothing here decides anything about the partition: exokal's geometric
// bisection cuts the mesh, and exokal's ownership rule -- an entity belongs to
// the LOWEST-numbered rank in its star -- says who owns what. This translates
// that answer into the three things the rest of mimetika asks for, which are
// three different questions and are often confused:
//
//   owner_of_dof   for the SOLVER: the layout of the algebra, one rank per
//                  global unknown, which is what a renumbering is built from
//   owned_cells    for the ASSEMBLY: which sites a process evaluates terms on.
//                  A term's contribution is a sum, so a row may be assembled
//                  in pieces on several processes and added back together
//   owned_dofs     for the CONSTRAINTS: which rows a process WRITES. A
//                  constraint is a replacement, not a contribution, so exactly
//                  one process may write each of them
//
// Deterministic and communication-free: every process partitions the same mesh
// with the same algorithm and reads the same answer, so no two of them can
// disagree about where an unknown lives.

namespace mimetika {

using graphos::Index;

struct Distribution {
  std::vector<int> owner_of_dof;
  std::vector<char> owned_cells;
  // A TERM IS NOT ALWAYS EVALUATED ON A CELL. A prescribed pressure is a form
  // on the boundary facets, an interface term on the interior ones; those
  // sites are facets, and a mask indexed by cell number would select the wrong
  // ones -- silently, since both are just indices. exokal names the
  // distinction (`evaluated_on_facets`), so both masks are carried.
  std::vector<char> owned_facets;
  std::vector<char> owned_dofs;
  // OWNED PLUS HALO: the cells whose terms this process evaluates.
  //
  // exokal names two conventions. Own the CELLS -- assemble what you own and
  // let the matrix carry your contributions to rows owned elsewhere -- costs
  // one exchange per assembly, and PETSc's off-process stash made that
  // exchange cost more than the assembly it saved (310 s against 2 s on a
  // 22k-cell mesh). Own the ROWS -- assemble everything that contributes to a
  // row you own, including cells you do not -- communicates nothing at all,
  // and pays for it by evaluating the halo twice. The halo is a surface and
  // the subdomain is a volume, so that is the cheaper of the two.
  std::vector<char> assembled_cells;
  // and the facets of those cells: a facet term contributes to the rows of its
  // adjacent cells, so a facet is evaluated wherever one of them is assembled
  std::vector<char> assembled_facets;

  bool empty() const { return owner_of_dof.empty(); }
};

// THE CELL PARTITION ALONE, which is all the mesh can answer before the space
// exists -- and it is needed there: the per-cell products are built before any
// space is, and they are the bulk of the work to divide.
template <class MeshT>
Distribution partition_cells(const MeshT& mesh, int dim, int n_ranks, int rank) {
  Distribution out;
  if (n_ranks < 2) return out;
  const exokal::spaces::CellPartition cells =
      exokal::spaces::rcb(exokal::spaces::cell_centroids(mesh, dim), n_ranks);
  out.owned_cells.assign(static_cast<std::size_t>(mesh.count(dim)), 0);
  for (Index e = 0; e < mesh.count(dim); ++e) {
    out.owned_cells[static_cast<std::size_t>(e)] = cells.owner(e) == rank ? 1 : 0;
  }
  // the facets, by the same rule: the lowest rank in the star, so a facet
  // between two subdomains is assembled once and by one of them
  const exokal::spaces::DofMap one(
      mesh.topology(), exokal::spaces::DofLayout::moments(dim, dim - 1, 1, 1), dim);
  out.owned_facets.assign(static_cast<std::size_t>(mesh.count(dim - 1)), 0);
  for (const auto& part : exokal::spaces::distribute(mesh.topology(), one, cells)) {
    if (part.rank != rank) continue;
    for (const Index f : part.owned) out.owned_facets[static_cast<std::size_t>(f)] = 1;
  }

  // THE HALO: the cells across a facet from an owned one.
  //
  // That is the stencil of a space whose unknowns live on facets and cells,
  // which is every space here -- two cells interact exactly when they share a
  // facet. A space that also carried unknowns on edges or vertices would
  // interact more widely and would need a wider halo; the colouring's
  // `sharing_dims` is where that question is already answered.
  out.assembled_cells = out.owned_cells;
  {
    const graphos::Adjacency down = graphos::incidence(mesh.topology(), dim, dim - 1);
    const graphos::Adjacency up = graphos::incidence(mesh.topology(), dim - 1, dim);
    for (Index e = 0; e < mesh.count(dim); ++e) {
      if (out.owned_cells[static_cast<std::size_t>(e)] == 0) continue;
      for (Index k = down.offsets[static_cast<std::size_t>(e)];
           k < down.offsets[static_cast<std::size_t>(e) + 1]; ++k) {
        const Index f = down.indices[static_cast<std::size_t>(k)];
        for (Index m = up.offsets[static_cast<std::size_t>(f)];
             m < up.offsets[static_cast<std::size_t>(f) + 1]; ++m) {
          out.assembled_cells[static_cast<std::size_t>(up.indices[static_cast<std::size_t>(m)])] = 1;
        }
      }
    }
  }
  {
    const graphos::Adjacency down = graphos::incidence(mesh.topology(), dim, dim - 1);
    out.assembled_facets.assign(static_cast<std::size_t>(mesh.count(dim - 1)), 0);
    for (Index e = 0; e < mesh.count(dim); ++e) {
      if (out.owned_cells[static_cast<std::size_t>(e)] == 0) continue;
      for (Index k = down.offsets[static_cast<std::size_t>(e)];
           k < down.offsets[static_cast<std::size_t>(e) + 1]; ++k) {
        out.assembled_facets[static_cast<std::size_t>(down.indices[static_cast<std::size_t>(k)])] =
            1;
      }
    }
  }
  return out;
}

// and the unknowns, once there is a space to number them
template <class MeshT>
void add_dof_ownership(Distribution& out, const MeshT& mesh, int dim,
                       const exokal::forms::StratifiedEpoch& epoch, int n_ranks, int rank) {
  if (n_ranks < 2) return;
  const exokal::spaces::CellPartition cells =
      exokal::spaces::rcb(exokal::spaces::cell_centroids(mesh, dim), n_ranks);
  out.owner_of_dof.assign(static_cast<std::size_t>(epoch.size()), 0);
  for (std::size_t s = 0; s < epoch.n_strata(); ++s) {
    const auto& space = epoch.stratum(s).space();
    const auto base = static_cast<Index>(epoch.offset(s));
    for (std::size_t f = 0; f < space.n_fields(); ++f) {
      const auto field_base = base + static_cast<Index>(space.offset(f));
      for (const auto& part : exokal::spaces::distribute(mesh.topology(), space.map(f), cells)) {
        for (const Index g : part.owned) {
          out.owner_of_dof[static_cast<std::size_t>(field_base + g)] = part.rank;
        }
      }
    }
  }
  out.owned_dofs.assign(out.owner_of_dof.size(), 0);
  for (std::size_t i = 0; i < out.owner_of_dof.size(); ++i) {
    out.owned_dofs[i] = out.owner_of_dof[i] == rank ? 1 : 0;
  }
}

template <class MeshT>
Distribution distribute_model(const MeshT& mesh, int dim,
                              const exokal::forms::StratifiedEpoch& epoch, int n_ranks, int rank) {
  Distribution out;
  if (n_ranks < 2) return out;

  const exokal::spaces::CellPartition cells =
      exokal::spaces::rcb(exokal::spaces::cell_centroids(mesh, dim), n_ranks);

  out.owner_of_dof.assign(static_cast<std::size_t>(epoch.size()), 0);
  for (std::size_t s = 0; s < epoch.n_strata(); ++s) {
    const auto& space = epoch.stratum(s).space();
    const auto base = static_cast<Index>(epoch.offset(s));
    for (std::size_t f = 0; f < space.n_fields(); ++f) {
      // exokal's own distribution of this field: `owned` is what each rank
      // holds, in the field's numbering, so the field's offset is all that has
      // to be added to reach the global one. The ghost, halo and exchange
      // lists it also computes are what a matrix-free path would need; a
      // matrix sink communicates through the matrix instead.
      const auto field_base = base + static_cast<Index>(space.offset(f));
      for (const auto& part : exokal::spaces::distribute(mesh.topology(), space.map(f),
                                                         cells)) {
        for (const Index g : part.owned) {
          out.owner_of_dof[static_cast<std::size_t>(field_base + g)] = part.rank;
        }
      }
    }
  }

  out.owned_cells.assign(static_cast<std::size_t>(mesh.count(dim)), 0);
  for (Index e = 0; e < mesh.count(dim); ++e) {
    out.owned_cells[static_cast<std::size_t>(e)] = cells.owner(e) == rank ? 1 : 0;
  }
  // the facets, by the same rule: the lowest rank in the star, so a facet
  // between two subdomains is assembled once and by one of them
  {
    const exokal::spaces::DofMap one(
        mesh.topology(), exokal::spaces::DofLayout::moments(dim, dim - 1, 1, 1), dim);
    out.owned_facets.assign(static_cast<std::size_t>(mesh.count(dim - 1)), 0);
    for (const auto& part : exokal::spaces::distribute(mesh.topology(), one, cells)) {
      if (part.rank != rank) continue;
      for (const Index f : part.owned) out.owned_facets[static_cast<std::size_t>(f)] = 1;
    }
  }
  out.owned_dofs.assign(out.owner_of_dof.size(), 0);
  for (std::size_t i = 0; i < out.owner_of_dof.size(); ++i) {
    out.owned_dofs[i] = out.owner_of_dof[i] == rank ? 1 : 0;
  }
  return out;
}

// The owner of every vertex, edge and face, from the same partition: what an
// auxiliary-space solver needs to lay out the maps it is given. One unknown per
// entity of the dimension in question, so an entity's owner is that unknown's.
template <class MeshT>
std::vector<std::vector<int>> entity_owners(const MeshT& mesh, int dim, int n_ranks) {
  std::vector<std::vector<int>> out;
  if (n_ranks < 2) return out;
  const auto& topo = mesh.topology();
  const exokal::spaces::CellPartition cells =
      exokal::spaces::rcb(exokal::spaces::cell_centroids(mesh, dim), n_ranks);
  out.resize(3);
  for (int k = 0; k < 3; ++k) {
    const exokal::spaces::DofMap one(topo, exokal::spaces::DofLayout::moments(dim, k, 1, 1), dim);
    auto& owner = out[static_cast<std::size_t>(k)];
    owner.assign(static_cast<std::size_t>(topo.count(k)), 0);
    for (const auto& part : exokal::spaces::distribute(topo, one, cells)) {
      for (const Index g : part.owned) owner[static_cast<std::size_t>(g)] = part.rank;
    }
  }
  return out;
}

}  // namespace mimetika
