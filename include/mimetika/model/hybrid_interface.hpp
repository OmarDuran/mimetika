#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exokal/hodge/hybrid_flux.hpp"
#include "exokal/hodge/hybrid_stress.hpp"
#include "mimetika/linear_solver/linear.hpp"

// THE INTERFACE SYSTEM, ASSEMBLED SPARSELY.
//
// exokal hybridizes a cell: it inverts the local saddle once and hands back
// the Steklov block S_E = C^T A_E^-1 C, the cell's displacement-to-traction
// map on its own boundary. What is left is a system in the FACET MULTIPLIERS
// alone -- the facet displacement in the moment chart the tractions carry --
// and assembling it is this file's whole job.
//
// exokal's own assemblers are DENSE, and say so: they are the contract, and
// the oracle a sparse one is held to. The contract is one sentence -- two
// multiplier blocks couple iff their facets share a cell -- so the sparsity is
// the facet adjacency through cells, and nothing here has to be discovered:
//
//     for each cell E, for each pair of its facets (f, g),
//         S[lambda(f), lambda(g)] += S_E[f-block, g-block]
//
// A facet's multiplier block is nf = facet_dofs unknowns, contiguous, and a
// PINNED facet has no block at all: the essential condition of a hybridized
// method is the facet displacement itself, which is why the boundary roles
// swap relative to the mixed form.
//
// WHY IT IS WORTH THE SECOND ELIMINATION. S is SPD once any facet is pinned --
// each S_E is symmetric positive semidefinite with the cell's rigid motions
// for a kernel, and pinning removes the global ones. That is the property the
// two-point condensation cannot offer for these realizations: the condensed
// mixed system is quasi-definite and wants MINRES, this one takes a conjugate
// gradient and an algebraic multigrid.

namespace mimetika {

// Which facets carry a multiplier. Defaults to the interior stratum, which is
// the homogeneous Dirichlet reference exokal's oracle uses.
inline std::vector<char> hybrid_free_facets(const exokal::Mesh& mesh, int cell_dim,
                                            const exokal::hodge::HybridStressOperators& hops) {
  return exokal::hodge::hybrid_detail::free_mask(mesh, cell_dim, hops, nullptr);
}
inline std::vector<char> hybrid_free_facets(const exokal::Mesh& mesh, int cell_dim,
                                            const exokal::hodge::HybridFluxOperators& hops) {
  return exokal::hodge::hybrid_flux_detail::free_mask(mesh, cell_dim, hops, nullptr);
}

// The interface system as triplets, in the multiplier numbering the free mask
// induces: facet f's block begins at the f-th free facet's offset, nf wide.
// ONE ASSEMBLER FOR BOTH PHYSICS. The flux and the stress hybridize to the
// same shape -- a Steklov block per cell over its facets' multiplier blocks --
// and the sparsity clause is the same sentence, so the walk is written once
// and instantiated per operator type. Only the offset helper lives in a
// per-physics namespace upstream, which is what the trait below names.
template <class Hops>
inline solver::SparseSystem hybrid_interface_sparse_of(const exokal::Mesh& mesh, int cell_dim,
                                                       const Hops& hops,
                                                       const std::vector<char>& free_facets) {
  const std::size_t nf = hops.facet_dofs();
  // the same offsets either helper computes: a running count of free facets
  // times nf, -1 on a pinned one
  std::vector<std::ptrdiff_t> at(free_facets.size(), -1);
  {
    std::ptrdiff_t next = 0;
    for (std::size_t f = 0; f < free_facets.size(); ++f) {
      if (free_facets[f] != 0) {
        at[f] = next;
        next += static_cast<std::ptrdiff_t>(nf);
      }
    }
  }
  std::size_t n = 0;
  for (const char m : free_facets) n += m != 0 ? nf : 0;

  solver::SparseSystem out;
  out.n = n;
  const graphos::Index cells = mesh.topology().count(cell_dim);
  for (graphos::Index e = 0; e < cells; ++e) {
    const auto& c = hops.cell(e);
    const exokal::numerics::Dense se = exokal::hodge::steklov(c);
    for (std::size_t fi = 0; fi < c.faces.size(); ++fi) {
      const std::ptrdiff_t gi = at[static_cast<std::size_t>(c.faces[fi])];
      if (gi < 0) continue;  // pinned: the facet displacement is data
      for (std::size_t fj = 0; fj < c.faces.size(); ++fj) {
        const std::ptrdiff_t gj = at[static_cast<std::size_t>(c.faces[fj])];
        if (gj < 0) continue;
        // the cell's own block: nf x nf, and a facet pair reached from two
        // cells contributes twice, which is the sum the assembly is
        for (std::size_t a = 0; a < nf; ++a) {
          for (std::size_t b = 0; b < nf; ++b) {
            const double v = se(fi * nf + a, fj * nf + b);
            if (v == 0.0) continue;
            out.row.push_back(static_cast<graphos::Index>(gi + static_cast<std::ptrdiff_t>(a)));
            out.col.push_back(static_cast<graphos::Index>(gj + static_cast<std::ptrdiff_t>(b)));
            out.value.push_back(v);
          }
        }
      }
    }
  }
  return out;
}

inline solver::SparseSystem hybrid_interface_sparse(
    const exokal::Mesh& mesh, int cell_dim, const exokal::hodge::HybridStressOperators& hops,
    const std::vector<char>& free_facets) {
  return hybrid_interface_sparse_of(mesh, cell_dim, hops, free_facets);
}
inline solver::SparseSystem hybrid_interface_sparse(
    const exokal::Mesh& mesh, int cell_dim, const exokal::hodge::HybridFluxOperators& hops,
    const std::vector<char>& free_facets) {
  return hybrid_interface_sparse_of(mesh, cell_dim, hops, free_facets);
}

// THE LOAD, THE PINNED DATUM AND THE RECOVERY ARE EXOKAL'S.
//
// They were mimetika's for exactly as long as exokal lacked them: the load
// carries the pinned multiplier's contribution to the free rows, and the
// recovery reads the multiplier over every facet rather than skipping the
// pinned ones. Both now live upstream -- hybrid_interface_load takes
// `lambda_data`, hybrid_recovery takes lambda over the whole stratum -- so
// they are called rather than copied.
//
// Copying them was what dated: upstream also changed the SIGN of the coupling
// so that the multiplier is the displacement rather than its negative, and a
// local copy went on answering the old convention while the system it fed
// answered the new one. Only the system assembly is mimetika's, because only
// that one has a sparsity question to answer.

}  // namespace mimetika
