#pragma once

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/preprocess/mixed_dimensional_mesh.hpp"
#include "graphos/core/coboundary.hpp"
#include "graphos/core/incidence.hpp"
#include "mimetika/model/simulation.hpp"

// A mixed-dimensional mesh, said in the terms Simulation takes.
//
// NOTHING TOPOLOGICAL HAPPENS HERE. exokal::mixed_dimensional_mesh has already
// done all of it: the ambient complex cut along the fractures, each fracture
// extracted as a stratum of its own with its own mesh, and the chain maps that
// carry every stratum's cells back into the ambient complex. This only names
// the strata and states which carries which.
//
// WHAT THE CUT BUYS. Across a fracture facet the ambient complex holds two
// copies, one per side, so the traction moments of the two sides are distinct
// unknowns and the normal trace of the stress is free to jump: the discrete
// counterpart of posing the stress in H(div; Omega \ Gamma) rather than
// H(div; Omega). The original facet survives alongside them, detached, and is
// the fracture stratum's own cell -- so the jump across it is a sum over the
// two carrying facets, which is what CarrierMap means by frontier duplication.
//
// The two sides are told apart by topology and not by geometry: after the cut
// each copy has exactly one cofacet, so its incidence sign in that cell is +1
// on one side and -1 on the other.
//
// LIFETIME. The specifications returned borrow the complexes and the chain maps
// from `md`, which must outlive the Simulation built from them.

namespace mimetika {

// The ambient stratum, and the fracture immersed in it.
//
// `ambient` is the mesh the reconstruction was built from: the chain maps carry
// into ITS numbering, which is the index space the two sides are matched
// through, so its facet count is what sizes that match.
inline std::vector<StratumSpec> strata_of(const exokal::MixedDimensionalMesh& md,
                                          const exokal::Mesh& ambient) {
  const int dim = md.dim;
  std::vector<StratumSpec> out;
  out.push_back(StratumSpec{"ambient", &md.top.mesh.topology(), dim, 0});

  const std::vector<const exokal::Stratum*> fractures = md.strata_of_codim(1);
  if (fractures.empty()) return out;
  if (fractures.size() != 1) {
    throw std::invalid_argument(
        "strata_of: " + std::to_string(fractures.size()) +
        " fractures, and an Epoch carries one substratum -- its CarrierMap names a single child "
        "space. Several fractures need either one stratum holding them all, as several connected "
        "components of one complex, or a carrier per child, which exokal does not offer yet");
  }
  for (const auto& generation : md.strata) {
    for (const exokal::Stratum& s : generation) {
      if (dim - s.dim > 1) {
        throw std::invalid_argument(
            "strata_of: the network has fracture intersections, which are carried by the "
            "fractures rather than by the ambient stratum; that is a chain of carriers and is not "
            "wired yet");
      }
    }
  }

  out.push_back(StratumSpec{"fracture", &fractures[0]->mesh.topology(), dim - 1, 1});
  out[0].carries = 1;
  out[0].facet_carrier = &md.top.to_ambient;
  out[0].child_carrier = &fractures[0]->to_ambient;
  out[0].n_ambient_facets = ambient.topology().count(dim - 1);
  return out;
}

// The facets of the cut complex that lie on a fracture.
//
// THE CUT LEAVES THREE COPIES of each fracture facet, and a caller has to tell
// them apart because `boundary_facets` cannot:
//
//   carrying   one cofacet each, the two sides. They LOOK like boundary facets
//              -- that is what having one cofacet means -- so a traction or a
//              roller imposed on the boundary of the cut complex would land on
//              the fracture and overwrite the contact rows. They must be
//              removed from the boundary set.
//   detached   the original, with no cofacet at all. It is the fracture
//              stratum's own cell and carries no bulk equation, so whatever
//              the ambient space numbers on it is unconstrained and the system
//              is singular until those rows are pinned.
//
// Both are consequences of the cut, not of the physics, so both are named here
// rather than rediscovered geometrically wherever a fracture is loaded.
struct FractureFacets {
  std::vector<graphos::Index> carrying;
  std::vector<graphos::Index> detached;
};

inline FractureFacets fracture_facets(const exokal::MixedDimensionalMesh& md) {
  const auto k = static_cast<std::size_t>(md.dim - 1);
  FractureFacets out;
  const std::vector<const exokal::Stratum*> fractures = md.strata_of_codim(1);
  if (fractures.empty()) return out;

  const auto& to_ambient = md.top.to_ambient.index[k];
  const auto& cells = fractures[0]->to_ambient.index[k];
  const std::set<graphos::Index> on_fracture(cells.begin(), cells.end());

  const graphos::Complex& tc = md.top.mesh.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(tc, md.dim - 1);
  for (graphos::Index f = 0; f < tc.count(md.dim - 1); ++f) {
    if (on_fracture.count(to_ambient[static_cast<std::size_t>(f)]) == 0) continue;
    const auto b = cob.offsets[static_cast<std::size_t>(f)];
    (cob.offsets[static_cast<std::size_t>(f) + 1] == b ? out.detached : out.carrying).push_back(f);
  }
  return out;
}

// EVERY FRACTURE AT ONCE.
//
// exokal's node-set path builds one codimension-1 stratum PER fracture, plus
// their mutual intersections one dimension down. A StratifiedEpoch carries a
// single substratum -- its CarrierMap names one child space -- so a network of
// several comes back refused by strata_of. This is the other representation
// that refusal names: ONE stratum holding them all, as several connected
// components of one complex.
//
// NOT BY MERGING THE NODE SETS. A facet belongs to a fracture when all of its
// vertices do, and that test applied to a UNION admits facets belonging to no
// fracture at all: on a tetrahedron with one vertex on one fracture and two on
// another, the union contains all three. Measured on the 5-fracture DFN meshes,
// the union induces 592 facets where the fractures have 392, and 104 where they
// have 48 -- half again as many, and every extra one is a facet that would be
// cut open for no reason. A hexahedral mesh happens to escape it, which is
// worse than failing.
//
// So each node set induces its own facets, by exokal's rule, and the UNION OF
// THOSE is the stratum's cells. Exact by construction, and every face still
// knows which fracture it came from, because that is recorded before the union.
//
// WHAT IT DOES NOT DO. The intersections are not strata of their own, so there
// are no unknowns on the lines where two fractures cross: each face carries its
// own traction and jump, and a crossing is where several faces meet, not an
// object with a state. A contact law is face-local, so nothing in the physics
// asks for more; a formulation that couples along intersection lines would.
struct InducedFractures {
  std::vector<std::vector<graphos::Index>> per_fracture;  // facets of each, ascending
  std::vector<graphos::Index> all;                        // their union, ascending
};

// The facets each node set induces: every vertex of the facet lies in the set.
// exokal states this rule once, in fracture_network; this applies it one
// fracture at a time so the union can be taken afterwards rather than before.
inline InducedFractures induced_fractures(
    const exokal::Mesh& mesh, const std::vector<std::vector<graphos::Index>>& node_sets) {
  const graphos::Complex& c = mesh.topology();
  const int k = mesh.dim() - 1;
  const graphos::Adjacency facet_verts = graphos::incidence(c, k, 0);
  std::vector<char> in_set(static_cast<std::size_t>(c.count(0)), 0);

  InducedFractures out;
  out.per_fracture.resize(node_sets.size());
  for (std::size_t f = 0; f < node_sets.size(); ++f) {
    for (const graphos::Index v : node_sets[f]) in_set[static_cast<std::size_t>(v)] = 1;
    for (graphos::Index e = 0; e < c.count(k); ++e) {
      bool all = true;
      for (auto m = facet_verts.offsets[static_cast<std::size_t>(e)];
           m < facet_verts.offsets[static_cast<std::size_t>(e) + 1] && all; ++m) {
        all = in_set[static_cast<std::size_t>(facet_verts.indices[static_cast<std::size_t>(m)])] != 0;
      }
      if (all) out.per_fracture[f].push_back(e);
    }
    for (const graphos::Index v : node_sets[f]) in_set[static_cast<std::size_t>(v)] = 0;
    out.all.insert(out.all.end(), out.per_fracture[f].begin(), out.per_fracture[f].end());
  }
  std::sort(out.all.begin(), out.all.end());
  out.all.erase(std::unique(out.all.begin(), out.all.end()), out.all.end());
  return out;
}

// That union, as the single codimension-1 stratum of a network. One entity, so
// exokal's descent finds no second generation and cuts no intersection lines.
inline exokal::FractureNetwork one_stratum_network(const exokal::Mesh& mesh,
                                                   const std::vector<graphos::Index>& facets) {
  const graphos::Complex& c = mesh.topology();
  const int k = mesh.dim() - 1;
  graphos::SubcomplexWorkspace ws(c);
  exokal::FractureNetwork net;
  net.levels.emplace_back();
  net.levels[0].push_back(
      exokal::FractureEntity{k, {0}, exokal::fracture_detail::closed_cells(c, facets, k, ws)});
  return net;
}

}  // namespace mimetika
