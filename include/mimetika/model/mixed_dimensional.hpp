#pragma once

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/preprocess/mixed_dimensional_mesh.hpp"
#include "graphos/core/coboundary.hpp"
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

}  // namespace mimetika
